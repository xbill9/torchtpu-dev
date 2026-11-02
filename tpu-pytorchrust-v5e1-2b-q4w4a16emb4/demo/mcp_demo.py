"""MCP server for demoing Gemma 4 E2B in Rust on TorchTPU.

Every tool talks to the local shim (demo/shim.py, http://127.0.0.1:8015 by default), which forwards
to the TPU VM. No gcloud, no project id: start the shim, register this server, ask.

Tools: ask_gemma_rust, stream_gemma_rust, engine_info, endpoint_status, serving_metrics.
"""

import json
import logging
import os
import time
from typing import Annotated, Optional

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

NAME = "torchtpu-rs-demo"
SHIM = os.getenv("TORCHTPU_RS_SHIM_URL", "http://127.0.0.1:8015").rstrip("/")
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)

# One line per HTTP request on stderr is noise in an MCP client's log.
logging.getLogger("httpx").setLevel(logging.WARNING)

mcp = MCPServer(os.getenv("MCP_SERVER_NAME", NAME))


async def _model(client: httpx.AsyncClient) -> str:
    r = await client.get(f"{SHIM}/v1/models")
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def _messages(prompt: str, system: Optional[str]) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


def _shim_down(e: Exception) -> str:
    return (
        f"❌ The local shim at {SHIM} did not answer ({type(e).__name__}: {e}). "
        "Start it with `systemctl --user start torchtpu-rs-shim` (installed by demo/install.sh), "
        "and check the TPU with `endpoint_status`."
    )


@mcp.tool(title="Ask Gemma (Rust on TorchTPU)", annotations=READ_ONLY)
async def ask_gemma_rust(
    prompt: str,
    max_tokens: Annotated[int, Field(ge=1, le=4096)] = 512,
    system: Optional[str] = None,
) -> str:
    """Asks Gemma 4 E2B on the TPU v5e-1, where the model is Rust (a PyO3 extension) and every op
    a TorchTPU kernel. Greedy decoding. Returns the answer, the token counts, and the engine's
    own seconds for reading the prompt and for decoding, with the decode rate computed from them."""
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            model = await _model(client)
            start = time.monotonic()
            r = await client.post(
                f"{SHIM}/v1/chat/completions",
                json={"model": model, "messages": _messages(prompt, system), "max_tokens": max_tokens},
            )
            wall = time.monotonic() - start
    except httpx.HTTPError as e:
        return _shim_down(e)
    if r.status_code != 200:
        return f"❌ HTTP {r.status_code}: {r.text[:500]}"
    d = r.json()
    usage, t = d.get("usage", {}), d.get("timings", {})
    n = usage.get("completion_tokens", 0)
    rate = (n - 1) / t["decode_s"] if n > 1 and t.get("decode_s") else 0.0
    return (
        f"{d['choices'][0]['message']['content']}\n\n"
        f"---\n{model} · finish `{d['choices'][0]['finish_reason']}` · "
        f"{usage.get('prompt_tokens')} prompt + {n} completion tokens · "
        f"prompt read {t.get('read_s', 0):.3f} s · decode {rate:.1f} tok/s on the TPU · "
        f"{wall:.2f} s round trip from here"
    )


@mcp.tool(title="Stream from Gemma and time it", annotations=READ_ONLY)
async def stream_gemma_rust(
    prompt: str,
    max_tokens: Annotated[int, Field(ge=1, le=4096)] = 256,
) -> str:
    """Streams an answer token by token through the shim and reports what a client sees: time to
    first token and decode tokens per second, timed here (network included). The completion
    token count is the server's `usage`, not a count of chunks."""
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            model = await _model(client)
            body = {
                "model": model,
                "messages": _messages(prompt, None),
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            start = time.monotonic()
            ttft, text, usage, finish = None, [], {}, None
            async with client.stream("POST", f"{SHIM}/v1/chat/completions", json=body) as r:
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    choice = (chunk.get("choices") or [{}])[0]
                    piece = choice.get("delta", {}).get("content")
                    if piece:
                        ttft = ttft if ttft is not None else time.monotonic() - start
                        text.append(piece)
                    finish = choice.get("finish_reason") or finish
                    usage = chunk.get("usage") or usage
            total = time.monotonic() - start
    except httpx.HTTPError as e:
        return _shim_down(e)
    n = usage.get("completion_tokens", 0)
    rate = (n - 1) / (total - ttft) if ttft is not None and n > 1 and total > ttft else 0.0
    return (
        f"{''.join(text)}\n\n---\nstreamed {n} tokens · time to first token {ttft or 0:.3f} s · "
        f"{rate:.1f} tok/s after the first · {total:.2f} s total · finish `{finish}` (timed here)"
    )


@mcp.tool(title="What is running on the TPU", annotations=READ_ONLY)
async def engine_info() -> str:
    """The server's GET /v1/engine: the torchtpu_rs PyO3 module and what built it (torch, rustc,
    Python), the PyO3 calls and which release the GIL, the thread that owns the unsendable
    Engine, the cache layout, TORCH_TPU_DEFER_AND_FUSE, load and warmup seconds, and the torch,
    torch_tpu and libtpu versions on the VM."""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{SHIM}/v1/engine")
    except httpx.HTTPError as e:
        return _shim_down(e)
    return f"```json\n{json.dumps(r.json(), indent=2)}\n```"


@mcp.tool(title="Endpoint status", annotations=READ_ONLY)
async def endpoint_status() -> str:
    """The shim's view: which TPU VM address it forwards to, whether that server's /health
    answers, and the requests forwarded since the shim started."""
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.get(f"{SHIM}/shim/status")
    except httpx.HTTPError as e:
        return _shim_down(e)
    s = r.json()
    ok = s.get("upstream_health") == 200
    return (
        f"{'🟢' if ok else '🔴'} local {SHIM}/v1 → {s.get('upstream') or 'no TPU endpoint found'}\n"
        f"- TPU /health: {s.get('upstream_health')}\n"
        f"- queued resource `{s.get('queued_resource')}` in {s.get('zone')}, project {s.get('project')}\n"
        f"- upstream found at {s.get('found_at')}; forwarded {s.get('forwarded')}, errors {s.get('errors')}"
    )


@mcp.tool(title="Serving metrics", annotations=READ_ONLY)
async def serving_metrics() -> str:
    """The server's Prometheus counters: requests, prompt and generated tokens, seconds spent
    reading prompts and decoding, finish reasons, queue depth, and the last decode rate."""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{SHIM}/metrics")
    except httpx.HTTPError as e:
        return _shim_down(e)
    return "\n".join(line for line in r.text.splitlines() if line and not line.startswith("#"))


if __name__ == "__main__":
    mcp.run()
