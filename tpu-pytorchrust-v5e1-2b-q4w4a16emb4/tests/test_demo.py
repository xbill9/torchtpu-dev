"""The local demo pieces, offline: the shim's routes and errors, and the demo MCP server's tools."""

import asyncio
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))


class ShimTests(unittest.TestCase):
    def test_unreachable_pinned_upstream_is_a_502(self):
        os.environ["TORCHTPU_RS_UPSTREAM"] = "http://127.0.0.1:9"
        try:
            import httpx
            import shim

            app = shim.build_app()

            async def go():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://shim") as c:
                    models = await c.get("/v1/models")
                    status = await c.get("/shim/status")
                    page = await c.get("/")
                return models, status, page

            models, status, page = asyncio.run(go())
        finally:
            del os.environ["TORCHTPU_RS_UPSTREAM"]
        self.assertEqual(models.status_code, 502)
        self.assertEqual(models.json()["error"]["type"], "shim_error")
        self.assertEqual(status.json()["upstream"], "http://127.0.0.1:9")
        self.assertTrue(status.json()["pinned"])
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn("/v1/chat/completions", page.text)


class DemoMcpTests(unittest.TestCase):
    def test_tools(self):
        import mcp_demo

        tools = {t.name: t for t in asyncio.run(mcp_demo.mcp.list_tools())}
        self.assertEqual(
            set(tools),
            {"ask_gemma_rust", "stream_gemma_rust", "engine_info", "endpoint_status", "serving_metrics"},
        )
        for t in tools.values():
            self.assertTrue(t.annotations.read_only_hint, t.name)

    def test_shim_down_message_names_the_service(self):
        import mcp_demo

        mcp_demo.SHIM = "http://127.0.0.1:9"
        out = asyncio.run(mcp_demo.endpoint_status())
        self.assertIn("systemctl --user start torchtpu-rs-shim", out)


if __name__ == "__main__":
    unittest.main()
