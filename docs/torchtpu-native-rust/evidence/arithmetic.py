"""Every derived figure in the article, computed from the evidence files beside this script.

    python3 arithmetic.py > arithmetic.txt
"""

import json
import os
import re
import statistics
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def read(name):
    with open(os.path.join(HERE, name)) as f:
        return f.read()


# Decode rate on the ocean prompt, five requests (ocean_prompt_5x.txt).
rates = [float(x) for x in re.findall(r", ([\d.]+) tok/s, client", read("ocean_prompt_5x.txt"))]
rust_median = statistics.median(rates)
print(f"ocean prompt: {len(rates)} requests, decode tok/s median {rust_median:.2f}, min {min(rates):.2f}, max {max(rates):.2f}")

# vLLM on the same chip type, same prompt (gcp_inference_vllm_tpu.txt).
vllm = float(re.search(r"^tpu\s+([\d.]+)\s+[\d.]+", read("gcp_inference_vllm_tpu.txt"), re.M).group(1))
print(f"vLLM TPU server decode {vllm} tok/s; vLLM / Rust = {vllm / rust_median:.1f}x")

# Colab notebook 06, final run (colab_06_output.txt): Rust fused and Python compile, streamed and batched.
colab = read("colab_06_output.txt")
fused = re.search(r"Rust fused, torch 2\.13\s+([\d.]+)\s+([\d.]+)\s+-\s+-\s+([\d.]+)\s+([\d.]+)", colab)
comp = re.search(r"Python compile, 2\.14\.1\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)", colab)
print(f"Colab Rust fused (tch, torch 2.13): streamed {fused.group(3)}, batched {fused.group(4)}")
print(f"Colab Python compile (torch 2.14.1): streamed {comp.group(3)}, batched {comp.group(4)}")
print(f"Rust here / Colab Rust fused streamed = {rust_median / float(fused.group(3)):.2f}")
print(f"Colab compiled streamed / Rust here = {float(comp.group(3)) / rust_median:.2f}x")

# Cost per million output tokens at the measured rate, one request at a time.
usd_hr = float(re.search(r"^tpu\s+[\d.]+\s+(\d+\.\d{4})\s+\d+", read("gcp_inference_vllm_tpu.txt"), re.M).group(1))
per_m_rust = usd_hr / (rust_median * 3600) * 1e6
per_m_vllm = usd_hr / (vllm * 3600) * 1e6
print(f"v5e-1 flex-start USD {usd_hr:.2f}/hr: Rust USD {per_m_rust:.2f} per M output tokens at {rust_median:.2f} tok/s; "
      f"vLLM USD {per_m_vllm:.2f} at {vllm} tok/s (server decode rate, one request at a time)")

# Streaming benchmark through the shim (bench_rust_stream_183042.json).
b = json.loads(read("bench_rust_stream_183042.json"))
ttft = [r["ttft_s"] for r in b["requests"]]
client = [r["client_decode_tok_s"] for r in b["requests"]]
print(f"streaming bench: {len(ttft)} requests, first token {min(ttft):.3f} to {max(ttft):.3f} s, "
      f"client decode {min(client):.1f} to {max(client):.1f} tok/s")

# Boot timeline (tpu_boot_2026-11-02.txt): guest-attribute stage times.
boot = read("tpu_boot_2026-11-02.txt")
t = dict(re.findall(r"(\w+) (\d\d:\d\d:\d\d)", boot.splitlines()[2]))
fmt = "%H:%M:%S"
for a, z in (("python", "unit"), ("torch", "rust"), ("weights", "unit")):
    print(f"boot {a} -> {z}: {datetime.strptime(t[z], fmt) - datetime.strptime(t[a], fmt)}")

# The rendered startup script, from the rig's own renderer.
sys.path.insert(0, os.path.join(REPO, "tpu-pytorchrust-v5e1-2b-q4w4a16emb4"))
import server  # noqa: E402

size = len(server._get_rust_startup_script("us-west4-a").encode())
print(f"rendered startup script {size} bytes = {size / 1024:.0f} KiB (metadata limit 256 KiB)")

# The ATen shim: wrapped ops and lines.
cpp = open(os.path.join(REPO, "tpu-pytorchrust-v5e1-2b-q4w4a16emb4", "rust", "src", "aten.cpp")).read()
ops = re.findall(r"^(?:Tensor\*|int|int64_t|void|PyObject\*|const char\*) (aten_\w+)\(", cpp, re.M)
print(f"aten.cpp: {len(ops)} extern \"C\" functions, {len(cpp.splitlines())} lines")
for f in ("lib.rs", "aten.rs", "bindings.rs", "gemma.rs", "weights.rs", "aten.cpp"):
    n = len(open(os.path.join(REPO, "tpu-pytorchrust-v5e1-2b-q4w4a16emb4", "rust", "src", f)).read().splitlines())
    print(f"  {f}: {n} lines")
