"""OpenAI-compatible server for Gemma 4 E2B with the model in Rust on TorchTPU.

The model, the ring cache and the decode loop are the `torchtpu_rs` extension module built from
`rust/` (see `rust/PYO3.md`). This file is the Python half: it hosts the process, imports
`torch_tpu` so the TPU kernels are registered before Rust calls into libtorch, tokenizes with
the checkpoint's chat template, and serves HTTP.

One engine, one thread. `torchtpu_rs.Engine` is a PyO3 `unsendable` class, so it is built on a
dedicated worker thread and every request is sent to that thread through a queue. Requests run
one at a time in arrival order. The engine releases the GIL while the TPU works, so the asyncio
event loop keeps serving `/health` and streaming tokens; each batch of tokens comes back to the
loop through `loop.call_soon_threadsafe`.

Decoding is greedy. `temperature`, `top_p` and similar fields are accepted and ignored, and each
response says so in `system_fingerprint`.

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions,
GET /v1/engine (what is loaded and how it is bound), GET /metrics (Prometheus text).
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("rust_openai_server")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", required=True, help="the W4A16 checkpoint: config, tokenizer, linears")
    p.add_argument("--emb4-dir", required=True, help="the directory holding the int4 tables (model-rest-001)")
    p.add_argument("--module-dir", required=True, help="the directory holding torchtpu_rs.so")
    p.add_argument("--served-name", required=True, help="the model id /v1/models reports")
    p.add_argument("--context", type=int, default=8192, help="cache length in tokens, a multiple of 256")
    p.add_argument("--read-every", type=int, default=1, help="tokens per host read; 1 streams token by token")
    p.add_argument("--max-tokens-default", type=int, default=512)
    p.add_argument("--device", default="tpu", choices=["tpu", "cpu"], help="cpu is for local testing")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


@dataclass
class Job:
    ids: list[int]
    max_new: int
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue
    cancel: threading.Event = field(default_factory=threading.Event)


@dataclass
class Stats:
    requests: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    read_seconds: float = 0.0
    decode_seconds: float = 0.0
    finishes: dict = field(default_factory=lambda: {"stop": 0, "length": 0, "cancelled": 0})
    last_decode_tok_s: float = 0.0
    load_seconds: float = 0.0
    warmup_seconds: float = 0.0


class EngineWorker(threading.Thread):
    """Owns the one `torchtpu_rs.Engine`. Nothing outside this thread may touch it."""

    def __init__(self, args, stop_ids: list[int], warmup_ids: list[int]):
        super().__init__(name="torchtpu-rs-engine", daemon=True)
        self.args, self.stop_ids, self.warmup_ids = args, stop_ids, warmup_ids
        self.jobs: "queue.Queue[Job]" = queue.Queue()
        self.ready = threading.Event()
        self.error: Optional[str] = None
        self.stats = Stats()
        self.layout: Optional[tuple] = None
        self.busy = False

    def run(self):
        import torch
        import torchtpu_rs

        try:
            start = time.perf_counter()
            like = torch.zeros(1, device=self.args.device)
            engine = torchtpu_rs.Engine(self.args.model_dir, self.args.emb4_dir, like, self.args.context)
            self.stats.load_seconds = time.perf_counter() - start
            self.layout = engine.layout()
            log.info("engine loaded in %.1f s: %r", self.stats.load_seconds, engine)
            # The first decode builds the TPU programs; do it before reporting ready.
            start = time.perf_counter()
            engine.decode(self.warmup_ids, 16, self.stop_ids, self.args.read_every)
            self.stats.warmup_seconds = time.perf_counter() - start
            log.info("warmup decode in %.1f s", self.stats.warmup_seconds)
        except BaseException as e:  # PanicException derives from BaseException
            self.error = f"{type(e).__name__}: {e}"
            log.exception("engine failed to start")
            self.ready.set()
            return
        self.ready.set()

        while True:
            job = self.jobs.get()
            self.busy = True
            try:
                self._run(engine, job)
            finally:
                self.busy = False

    def _run(self, engine, job: Job):
        def on_tokens(batch):
            job.loop.call_soon_threadsafe(job.out.put_nowait, ("tokens", list(batch)))
            return not job.cancel.is_set()

        try:
            r = engine.decode(job.ids, job.max_new, self.stop_ids, self.args.read_every, on_tokens)
        except BaseException as e:
            job.loop.call_soon_threadsafe(job.out.put_nowait, ("error", f"{type(e).__name__}: {e}"))
            return
        s = self.stats
        s.requests += 1
        s.prompt_tokens += len(job.ids)
        s.generated_tokens += len(r.tokens)
        s.read_seconds += r.read_s
        s.decode_seconds += r.decode_s
        s.finishes[r.finish] = s.finishes.get(r.finish, 0) + 1
        if len(r.tokens) > 1 and r.decode_s > 0:
            s.last_decode_tok_s = (len(r.tokens) - 1) / r.decode_s
        job.loop.call_soon_threadsafe(
            job.out.put_nowait,
            ("done", {"finish": r.finish, "tokens": len(r.tokens), "read_s": r.read_s, "decode_s": r.decode_s}),
        )


def build_app(args):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    with open(os.path.join(args.model_dir, "generation_config.json")) as f:
        eos = json.load(f)["eos_token_id"]
    stop_ids = eos if isinstance(eos, list) else [eos]

    def chat_ids(messages: list[dict]) -> list[int]:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return list(enc["input_ids"] if hasattr(enc, "keys") else enc)

    warmup_ids = chat_ids([{"role": "user", "content": "Hello"}])
    worker = EngineWorker(args, stop_ids, warmup_ids)
    worker.start()
    app = FastAPI(title="torchtpu_rs OpenAI server")
    fingerprint = "torchtpu_rs-greedy"

    def error(status: int, message: str, kind: str = "invalid_request_error"):
        return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)

    def not_ready():
        if worker.error:
            return error(500, f"engine failed to start: {worker.error}", "server_error")
        if not worker.ready.is_set():
            return error(503, "engine is loading", "server_error")
        return None

    def budget(ids: list[int], requested: Optional[int]):
        room = args.context - len(ids)
        if room < 1:
            return None, error(400, f"prompt is {len(ids)} tokens; the context is {args.context}")
        want = requested if requested and requested > 0 else args.max_tokens_default
        return min(want, room), None

    async def run_job(ids: list[int], max_new: int):
        loop = asyncio.get_running_loop()
        job = Job(ids=ids, max_new=max_new, loop=loop, out=asyncio.Queue())
        worker.jobs.put(job)
        return job

    class Detok:
        """Turns a growing token list into text deltas, holding back a partial UTF-8 character."""

        def __init__(self):
            self.tokens: list[int] = []
            self.sent = ""

        def push(self, batch: list[int]) -> str:
            self.tokens.extend(batch)
            text = tok.decode(self.tokens, skip_special_tokens=True)
            if text.endswith("�"):
                return ""
            delta, self.sent = text[len(self.sent):], text
            return delta

        def flush(self) -> str:
            text = tok.decode(self.tokens, skip_special_tokens=True)
            delta, self.sent = text[len(self.sent):], text
            return delta

    def usage(prompt: int, completion: int) -> dict:
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}

    def finish_reason(finish: str) -> str:
        return "length" if finish == "length" else "stop"

    async def complete(body: dict, ids: list[int], chat: bool, request: Request):
        max_new, err = budget(ids, body.get("max_tokens") or body.get("max_completion_tokens"))
        if err:
            return err
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        job = await run_job(ids, max_new)

        if not body.get("stream"):
            d = Detok()
            while True:
                kind, payload = await job.out.get()
                if kind == "tokens":
                    d.push(payload)
                elif kind == "error":
                    return error(500, payload, "server_error")
                else:
                    break
            d.flush()
            text = d.sent
            choice: dict[str, Any] = {"index": 0, "finish_reason": finish_reason(payload["finish"])}
            if chat:
                choice["message"] = {"role": "assistant", "content": text}
            else:
                choice["text"] = text
            return {
                "id": rid,
                "object": obj,
                "created": created,
                "model": args.served_name,
                "system_fingerprint": fingerprint,
                "choices": [choice],
                "usage": usage(len(ids), payload["tokens"]),
                "timings": {"read_s": payload["read_s"], "decode_s": payload["decode_s"]},
            }

        async def events():
            d = Detok()
            chunk_obj = "chat.completion.chunk" if chat else "text_completion"

            def chunk(delta: Optional[str], finish: Optional[str], first: bool = False):
                c: dict[str, Any] = {"index": 0, "finish_reason": finish}
                if chat:
                    c["delta"] = ({"role": "assistant"} if first else {}) | ({"content": delta} if delta else {})
                else:
                    c["text"] = delta or ""
                return {
                    "id": rid,
                    "object": chunk_obj,
                    "created": created,
                    "model": args.served_name,
                    "system_fingerprint": fingerprint,
                    "choices": [c],
                }

            try:
                if chat:
                    yield f"data: {json.dumps(chunk(None, None, first=True))}\n\n"
                while True:
                    kind, payload = await job.out.get()
                    if kind == "tokens":
                        delta = d.push(payload)
                        if delta:
                            yield f"data: {json.dumps(chunk(delta, None))}\n\n"
                    elif kind == "error":
                        yield f"data: {json.dumps({'error': {'message': payload, 'type': 'server_error'}})}\n\n"
                        break
                    else:
                        tail = d.flush()
                        if tail:
                            yield f"data: {json.dumps(chunk(tail, None))}\n\n"
                        last = chunk(None, finish_reason(payload["finish"]))
                        opts = body.get("stream_options") or {}
                        if opts.get("include_usage"):
                            last["usage"] = usage(len(ids), payload["tokens"])
                        yield f"data: {json.dumps(last)}\n\n"
                        break
                yield "data: [DONE]\n\n"
            finally:
                # Client went away mid-stream: the next on_tokens call returns False.
                job.cancel.set()

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        bad = not_ready()
        return bad if bad else {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": args.served_name, "object": "model", "owned_by": "torchtpu_rs", "max_model_len": args.context}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        bad = not_ready()
        if bad:
            return bad
        body = await request.json()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return error(400, "messages must be a non-empty list")
        for m in messages:
            if isinstance(m.get("content"), list):  # text parts only; this engine is text-only
                m["content"] = "".join(p.get("text", "") for p in m["content"] if p.get("type") == "text")
        return await complete(body, chat_ids(messages), True, request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        bad = not_ready()
        if bad:
            return bad
        body = await request.json()
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            return error(400, "prompt must be a non-empty string")
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        return await complete(body, list(ids), False, request)

    @app.get("/v1/engine")
    async def engine_info():
        import importlib.metadata as md

        import torchtpu_rs

        def version(pkg):
            try:
                return md.version(pkg)
            except md.PackageNotFoundError:
                return None

        layout = worker.layout
        return {
            "served_name": args.served_name,
            "ready": worker.ready.is_set() and not worker.error,
            "error": worker.error,
            "busy": worker.busy,
            "queued": worker.jobs.qsize(),
            "device": args.device,
            "context": args.context,
            "read_every": args.read_every,
            "stop_ids": stop_ids,
            "layout": None if layout is None else dict(zip(["full_rings", "sliding_rings", "shared_layers", "context"], layout)),
            "torch_tpu_defer_and_fuse": os.environ.get("TORCH_TPU_DEFER_AND_FUSE", ""),
            "load_seconds": round(worker.stats.load_seconds, 2),
            "warmup_seconds": round(worker.stats.warmup_seconds, 2),
            "versions": {
                "torchtpu_rs": torchtpu_rs.__version__,
                "built_with": dict(torchtpu_rs.BUILD),
                "torch": version("torch"),
                "torch_tpu": version("torch_tpu"),
                "libtpu": version("libtpu"),
                "transformers": version("transformers"),
                "python": sys.version.split()[0],
            },
            "bindings": {
                "module": "torchtpu_rs: PyO3 0.30 extension-module over an ATen shim (rust/src/aten.cpp) compiled against torch " + torchtpu_rs.BUILD["torch"],
                "engine_class": "Engine — #[pyclass(unsendable)], owned by thread " + worker.name,
                "calls": [
                    "Engine(model_dir, emb4_dir, like, context)  GIL released",
                    "Engine.decode(ids, max_new, stop, read_every, on_tokens) -> DecodeResult  GIL released; on_tokens re-attaches",
                    "Engine.generate(ids, max_new, read_every) -> (tokens, read_s, decode_s)  GIL released",
                    "Engine.logits(ids) -> torch.Tensor  shares storage with the Rust tensor",
                    "Engine.layout() -> (full_rings, sliding_rings, shared_layers, context)",
                ],
            },
        }

    @app.get("/metrics")
    async def metrics():
        s = worker.stats
        lines = [
            "# TYPE torchtpu_rs_requests_total counter",
            f"torchtpu_rs_requests_total {s.requests}",
            "# TYPE torchtpu_rs_prompt_tokens_total counter",
            f"torchtpu_rs_prompt_tokens_total {s.prompt_tokens}",
            "# TYPE torchtpu_rs_generation_tokens_total counter",
            f"torchtpu_rs_generation_tokens_total {s.generated_tokens}",
            "# TYPE torchtpu_rs_prompt_read_seconds_total counter",
            f"torchtpu_rs_prompt_read_seconds_total {s.read_seconds:.6f}",
            "# TYPE torchtpu_rs_decode_seconds_total counter",
            f"torchtpu_rs_decode_seconds_total {s.decode_seconds:.6f}",
            "# TYPE torchtpu_rs_last_decode_tokens_per_second gauge",
            f"torchtpu_rs_last_decode_tokens_per_second {s.last_decode_tok_s:.3f}",
            "# TYPE torchtpu_rs_queue_depth gauge",
            f"torchtpu_rs_queue_depth {worker.jobs.qsize()}",
            "# TYPE torchtpu_rs_busy gauge",
            f"torchtpu_rs_busy {int(worker.busy)}",
            "# TYPE torchtpu_rs_finish_total counter",
        ] + [f'torchtpu_rs_finish_total{{reason="{k}"}} {v}' for k, v in s.finishes.items()]
        return PlainTextResponse("\n".join(lines) + "\n")

    return app


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Import order is load-bearing: torch loads libtorch, torch_tpu registers the TPU kernels on
    # its dispatcher, and only then does the Rust module (linked to that libtorch) import.
    import torch  # noqa: F401

    if args.device == "tpu":
        import torch_tpu  # noqa: F401
    sys.path.insert(0, args.module_dir)
    import torchtpu_rs  # noqa: F401

    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
