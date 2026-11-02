"""Local OpenAI-compatible shim for the Rust TorchTPU rig.

Runs on this machine at http://127.0.0.1:8015 and forwards every request to the server on the TPU
VM (rust_openai_server.py), streaming included. Point any OpenAI client, the demo MCP server or a
`curl` at the local address; the shim finds the VM itself.

Finding the VM: TORCHTPU_RS_UPSTREAM, when set, is used as is (e.g. http://1.2.3.4:8000).
Otherwise the shim asks gcloud for the external IP of the queued resource's node
(TORCHTPU_RS_PROJECT, TORCHTPU_RS_ZONE, TORCHTPU_RS_QR, read from tpu.env beside the repo when
unset), and asks again whenever the upstream stops answering, since a new flex-start VM has a
new address.

Endpoints: everything under /v1/ is forwarded, plus /health and /metrics. /shim/status reports the
upstream in use, when it was found, and the requests forwarded since start. / (and /ui) is a chat
page (ui.html) that streams from /v1/chat/completions; /v1 is a JSON index of the endpoints.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

log = logging.getLogger("torchtpu-rs-shim")
HERE = os.path.dirname(os.path.abspath(__file__))
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "host"}


def _rig_env() -> dict:
    """tpu.env beside the repo, as a fallback for anything not in the environment."""
    out = {}
    try:
        with open(os.path.join(HERE, "..", "tpu.env")) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


class Upstream:
    def __init__(self):
        env = _rig_env()
        self.pinned = os.getenv("TORCHTPU_RS_UPSTREAM", "").rstrip("/")
        self.project = os.getenv("TORCHTPU_RS_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.zone = os.getenv("TORCHTPU_RS_ZONE") or env.get("GOOGLE_CLOUD_ZONE", "us-west4-a")
        self.qr = os.getenv("TORCHTPU_RS_QR") or env.get("TPU_QR_NAME", "gemma4-rust-qr")
        self.port = int(os.getenv("TORCHTPU_RS_PORT") or env.get("SERVE_PORT", "8000"))
        self.url: Optional[str] = self.pinned or None
        self.found_at: Optional[float] = time.time() if self.pinned else None
        self.lock = asyncio.Lock()
        self.forwarded = 0
        self.errors = 0

    def _gcloud(self, *args: str) -> str:
        cmd = ["gcloud", *args, f"--zone={self.zone}"] + ([f"--project={self.project}"] if self.project else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else ""

    def _discover(self) -> Optional[str]:
        node = self._gcloud(
            "alpha", "compute", "tpus", "queued-resources", "describe", self.qr,
            "--format=value(tpu.nodeSpec[0].nodeId)",
        )
        if not node:
            return None
        ip = self._gcloud(
            "compute", "tpus", "tpu-vm", "describe", node,
            "--format=value(networkEndpoints[0].accessConfig.externalIp)",
        )
        return f"http://{ip}:{self.port}" if ip else None

    async def get(self, refresh: bool = False) -> Optional[str]:
        if self.pinned:
            return self.pinned
        async with self.lock:
            if refresh or not self.url:
                url = await asyncio.to_thread(self._discover)
                if url and url != self.url:
                    log.info("upstream %s (queued resource %s, zone %s)", url, self.qr, self.zone)
                self.url, self.found_at = url, (time.time() if url else None)
            return self.url


def build_app() -> FastAPI:
    app = FastAPI(title="torchtpu-rs local shim")
    up = Upstream()
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    async def forward(request: Request, path: str):
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        streaming = False
        if body and "json" in request.headers.get("content-type", ""):
            try:
                streaming = bool(json.loads(body).get("stream"))
            except (ValueError, AttributeError):
                pass
        for attempt in (0, 1):
            base = await up.get(refresh=attempt == 1)
            if not base:
                up.errors += 1
                return JSONResponse(
                    {"error": {"message": f"no TPU endpoint: queued resource {up.qr} in {up.zone} has no node with an external IP", "type": "shim_error"}},
                    status_code=503,
                )
            req = client.build_request(request.method, f"{base}/{path}", params=request.query_params, headers=headers, content=body)
            try:
                res = await client.send(req, stream=True)
            except httpx.TransportError as e:
                log.warning("upstream %s failed (%s); looking it up again", base, e)
                continue
            up.forwarded += 1
            out_headers = {k: v for k, v in res.headers.items() if k.lower() not in HOP_HEADERS}
            if streaming or res.headers.get("content-type", "").startswith("text/event-stream"):
                async def relay():
                    try:
                        async for chunk in res.aiter_raw():
                            yield chunk
                    finally:
                        await res.aclose()

                return StreamingResponse(relay(), status_code=res.status_code, headers=out_headers,
                                         media_type=res.headers.get("content-type"))
            data = await res.aread()
            await res.aclose()
            return Response(data, status_code=res.status_code, headers=out_headers,
                            media_type=res.headers.get("content-type"))
        up.errors += 1
        return JSONResponse({"error": {"message": "TPU endpoint unreachable", "type": "shim_error"}}, status_code=502)

    # The chat page: plain HTML and JavaScript calling this shim's /v1/chat/completions.
    @app.get("/")
    @app.get("/ui")
    async def ui():
        return FileResponse(os.path.join(HERE, "ui.html"), media_type="text/html")

    # The bare base URL is not an OpenAI endpoint; opening it gets this index instead of the TPU
    # server's 404. Registered before the /v1/{path} catch-all, which would match it.
    @app.get("/v1")
    @app.get("/v1/")
    async def index(request: Request):
        base = str(request.base_url).rstrip("/")
        info = await status()
        model = None
        if info["upstream_health"] == 200:
            try:
                r = await client.get(f"{info['upstream']}/v1/models", timeout=10)
                model = r.json()["data"][0]["id"]
            except (httpx.HTTPError, KeyError, IndexError, ValueError):
                pass
        return {
            "service": "torchtpu-rs local shim: Gemma 4 E2B in Rust on TorchTPU (TPU v5e-1)",
            "openai_base_url": f"{base}/v1",
            "model": model,
            "tpu": info,
            "endpoints": {
                "GET /v1/models": "the served model id",
                "POST /v1/chat/completions": "chat, streaming or not (greedy decoding)",
                "POST /v1/completions": "text completion, streaming or not",
                "GET /v1/engine": "the PyO3 module, what built it, versions on the VM, cache layout",
                "GET /health": "200 once the engine is loaded and warmed up",
                "GET /metrics": "Prometheus counters",
                "GET /shim/status": "which TPU VM address the shim forwards to",
            },
        }

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def v1(path: str, request: Request):
        return await forward(request, f"v1/{path}")

    @app.get("/health")
    async def health(request: Request):
        return await forward(request, "health")

    @app.get("/metrics")
    async def metrics(request: Request):
        return await forward(request, "metrics")

    @app.get("/shim/status")
    async def status():
        url = await up.get()
        upstream_health = None
        if url:
            try:
                r = await client.get(f"{url}/health", timeout=5)
                upstream_health = r.status_code
            except httpx.HTTPError as e:
                upstream_health = f"unreachable: {type(e).__name__}"
        return {
            "upstream": url,
            "upstream_health": upstream_health,
            "pinned": bool(up.pinned),
            "queued_resource": up.qr,
            "zone": up.zone,
            "project": up.project or "(gcloud default)",
            "found_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(up.found_at)) if up.found_at else None,
            "forwarded": up.forwarded,
            "errors": up.errors,
        }

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default=os.getenv("TORCHTPU_RS_SHIM_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("TORCHTPU_RS_SHIM_PORT", "8015")))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
