"""Smoke and regression tests for the tpu-pytorchrust-v5e1-2b-q4w4a16emb4 MCP server and repo invariants.

Standard library + the server's own dependencies only; run via `make test` or
`python3 -m unittest discover -s tests`. No GCP calls are made — tests cover
pure logic, tool registration, template rendering, and repo hygiene.
"""

import asyncio
import filecmp
import subprocess
import sys
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

EXPECTED_DESTRUCTIVE = {
    "destroy_queued_resource",
    "destroy_tpu_vm_instance",
    "manage_queued_resource",
    "manage_rust_server",
    "find_tpu",
}


def run(coro):
    return asyncio.run(coro)


class ToolCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {t.name: t for t in run(server.mcp.list_tools())}

    def test_every_tool_has_title_description_and_annotations(self):
        for name, tool in self.tools.items():
            self.assertTrue(tool.title, f"{name} has no title")
            self.assertTrue(tool.description, f"{name} has no description")
            self.assertIsNotNone(tool.annotations, f"{name} has no annotations")

    def test_destructive_hints_match_expected_set(self):
        destructive = {
            name for name, t in self.tools.items() if t.annotations.destructive_hint
        }
        self.assertEqual(destructive, EXPECTED_DESTRUCTIVE)

    def test_read_only_tools_never_marked_destructive(self):
        for name, t in self.tools.items():
            if t.annotations.read_only_hint:
                self.assertFalse(
                    t.annotations.destructive_hint,
                    f"{name} is both readOnly and destructive",
                )

    def test_action_and_type_enums_in_schema(self):
        props = self.tools["manage_rust_server"].input_schema["properties"]
        self.assertEqual(props["action"]["enum"], ["status", "start", "stop", "restart", "log"])
        self.assertEqual(
            self.tools["estimate_deployment_cost"].input_schema["properties"]["tpu_type"]["enum"],
            ["v6e", "v5e", "v5p"],
        )
        for name in ("create_tpu_vm_instance", "find_tpu_vm"):
            serve = self.tools[name].input_schema["properties"]["serve"]
            self.assertEqual((serve["type"], serve["default"]), ("boolean", True), name)

    def test_help_names_only_registered_tools(self):
        help_text = run(server.get_help())
        named = set(re.findall(r"\*\*`([a-z][a-z0-9_]+)`\*\*", help_text))
        self.assertEqual(named - set(self.tools), set())

    def test_log_tails_are_bounded(self):
        for name in ("manage_rust_server", "get_tpu_system_logs", "get_tpu_vm_serial_log"):
            tail = self.tools[name].input_schema["properties"]["tail"]
            self.assertIn("maximum", tail, f"{name}.tail has no upper bound")


class HelperTests(unittest.TestCase):
    def test_zone_defaults_to_current_global(self):
        self.assertEqual(server._zone(None), server.ZONE)
        self.assertEqual(server._zone("us-east5-b"), "us-east5-b")

    def test_filter_key_metrics_drops_comments_and_noise(self):
        text = (
            "# TYPE torchtpu_rs_requests_total counter\n"
            "torchtpu_rs_requests_total 3\n"
            "torchtpu_rs_decode_seconds_total 1.5\n"
            'torchtpu_rs_finish_total{reason="stop"} 2\n'
        )
        self.assertEqual(
            server._filter_key_metrics(text),
            ["torchtpu_rs_requests_total 3", 'torchtpu_rs_finish_total{reason="stop"} 2'],
        )

    def test_estimate_cost_math_and_rejects_bad_topology(self):
        result = run(server.estimate_deployment_cost(hours=2.0, tpu_type="v6e", topology="2x4"))
        self.assertIn("$21.60", result)  # 8 chips * 1.35 * 2h
        self.assertTrue(run(server.estimate_deployment_cost(topology="0x4")).startswith("❌"))

    def test_min_chips_for_model(self):
        # v5e: 16GB per chip
        self.assertEqual(server._min_chips_for_model("google/gemma-4-31B-it", "v5e-8"), 8)
        self.assertEqual(server._min_chips_for_model("google/gemma-4-26B-it", "v5litepod-8"), 8)
        self.assertEqual(server._min_chips_for_model("google/gemma-4-12B-it", "v5e-1"), 2)
        # v6e: 32GB per chip
        self.assertEqual(server._min_chips_for_model("google/gemma-4-31B-it", "v6e-8"), 4)
        self.assertEqual(server._min_chips_for_model("google/gemma-4-12B-it", "v6e-1"), 1)
        self.assertEqual(server._min_chips_for_model("google/gemma-4-E2B-it", "v5e-1"), 1)
        # Quantized checkpoints fit on a single chip regardless of size
        self.assertEqual(server._min_chips_for_model("google/gemma-4-26B-it-qat-q4_0"), 1)
        self.assertEqual(server._min_chips_for_model("google/gemma-4-31B-it-int4"), 1)

    def test_create_blocks_model_too_big_for_accelerator(self):
        result = run(
            server.create_tpu_vm_instance(
                accelerator="v6e-1", model_name="google/gemma-4-31B-it"
            )
        )
        self.assertTrue(result.startswith("❌"))
        self.assertIn("chips", result)

    def test_served_model_id_falls_back_to_none_when_unreachable(self):
        self.assertIsNone(run(server._get_served_model_id("http://127.0.0.1:9")))

    def test_payload_carries_the_crate_and_server(self):
        import base64
        import io
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(server._payload_b64()))) as tar:
            names = set(tar.getnames())
        self.assertEqual(names, set(server._PAYLOAD_FILES))
        self.assertTrue(all("target/" not in n for n in names))

    def test_run_command_success_and_timeout(self):
        rc, out, _ = run(server.run_command(["echo", "hi"]))
        self.assertEqual((rc, out), (0, "hi"))
        rc, _, err = run(server.run_command(["sleep", "5"], timeout=1))
        self.assertEqual(rc, -1)
        self.assertIn("Timeout", err)


class RustStartupTemplateTests(unittest.TestCase):
    def render(self, serve=True):
        return server._get_rust_startup_script("us-west4-a", serve=serve)

    def test_renders_without_leftover_placeholders(self):
        rendered = self.render()
        for marker in ("{project_id}", "{payload_b64}", "{pip_spec}", "{serve_port}", "{base_repo}"):
            self.assertNotIn(marker, rendered)
        self.assertIn(server.TORCH_PIP_SPEC.split()[0], rendered)
        self.assertIn(f"{server.PYTHON_BIN}-dev", rendered)
        self.assertIn(f"rustup toolchain install {server._rust_toolchain()}", rendered)
        self.assertIn("torch_tpu==0.2.0", rendered)
        self.assertIn(server.BASE_REPO, rendered)
        self.assertIn(server.MODEL_NAME, rendered)
        self.assertIn(f"TORCH_TPU_DEFER_AND_FUSE={server.SERVE_DEFER_AND_FUSE}", rendered)
        self.assertIn("TPU environment ready.", rendered)
        self.assertIn("TPU serving started.", rendered)

    def test_every_stage_is_reported(self):
        rendered = self.render()
        for stage in server._STAGES:
            self.assertIn(f"stage {stage}", rendered)

    def test_smoke_imports_torch_tpu_before_the_rust_module(self):
        smoke = self.render().split("smoke.py <<'PYEOF'")[1].split("PYEOF")[0]
        order = [line.split()[1] for line in smoke.splitlines() if line.startswith("import ")]
        self.assertLess(order.index("torch_tpu"), order.index("torchtpu_rs"))
        self.assertNotIn("torch_xla", smoke)

    def test_rendered_script_passes_bash_syntax_check(self):
        for serve in (True, False):
            proc = subprocess.run(
                ["bash", "-n", "/dev/stdin"], input=self.render(serve), text=True, capture_output=True
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_fits_in_instance_metadata(self):
        # A single metadata value is limited to 256 KB.
        self.assertLess(len(self.render().encode()), 256 * 1024)

    def test_rejects_a_context_that_is_not_a_multiple_of_256(self):
        with self.assertRaises(RuntimeError):
            server._get_rust_startup_script("us-west4-a", context=1000)


class RepoHygieneTests(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for script in ("project-setup.sh", "init.sh", "set_env.sh", "set_adc.sh"):
            proc = subprocess.run(
                ["bash", "-n", str(ROOT / script)], capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, f"{script}: {proc.stderr}")

    def test_skill_snapshots_in_sync_with_sources(self):
        """Sources at the repo root are authoritative; `make skill` regenerates the
        copies. A mismatch means someone edited one side without resyncing."""
        for src, snap in (
            ("server.py", ".claude/skills/tpu-pytorchrust-v5e1-2b-q4w4a16emb4-management/mcp/server.py"),
            ("project-setup.sh", ".claude/skills/tpu-pytorchrust-v5e1-2b-q4w4a16emb4-management/mcp/project-setup.sh"),
            ("requirements.txt", ".claude/skills/tpu-pytorchrust-v5e1-2b-q4w4a16emb4-management/mcp/requirements.txt"),
            ("rust/src/bindings.rs", ".claude/skills/tpu-pytorchrust-v5e1-2b-q4w4a16emb4-management/mcp/payload/rust/src/bindings.rs"),
            ("rust_openai_server.py", ".claude/skills/tpu-pytorchrust-v5e1-2b-q4w4a16emb4-management/mcp/payload/rust_openai_server.py"),
        ):
            self.assertTrue(
                filecmp.cmp(ROOT / src, ROOT / snap, shallow=False),
                f"{snap} is stale — run `make skill`",
            )

    def test_plugin_copy_matches_skill(self):
        for rel in (
            "mcp/server.py",
            "SKILL.md",
            "mcp/startup_script_rust_template.sh",
            "mcp/payload/rust/src/bindings.rs",
            "mcp/payload/rust_openai_server.py",
        ):
            self.assertTrue(
                filecmp.cmp(
                    ROOT / ".claude/skills/tpu-pytorchrust-v5e1-2b-q4w4a16emb4-management" / rel,
                    ROOT / "skills/tpu-pytorchrust-v5e1-2b-q4w4a16emb4-management" / rel,
                    shallow=False,
                ),
                f"skills/tpu-pytorchrust-v5e1-2b-q4w4a16emb4-management/{rel} is stale — run `make skill`",
            )


if __name__ == "__main__":
    unittest.main()
