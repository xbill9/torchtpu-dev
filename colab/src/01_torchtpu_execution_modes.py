# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # TorchTPU on a Colab v5e-1: eager, defer-and-fuse, and torch.compile
#
# **Author:** William McLean, Google Developer Expert
#
# **Based on:** the TorchTPU
# [execution model](https://torchtpu.readthedocs.io/en/latest/torchtpu_execution_model/) guide, and
# the static-shape serving code in [`xbill9/torchtpu-dev`](https://github.com/xbill9/torchtpu-dev).
#
# ### Who this notebook is for
#
# PyTorch developers meeting a TPU for the first time, and anyone deciding how to run an existing
# PyTorch model on one. Terms are explained where they first appear. No JAX is involved.
#
# ### The study in one paragraph
#
# TorchTPU is the open-source PyTorch backend for Cloud TPUs, released 2026-10-20. You move tensors to
# `torch.device("tpu")` and your code runs on the chip, with no rewrite. Every operation you call is
# compiled by XLA before it runs, and the backend offers three ways to decide how much code each
# compile covers: one operation at a time (strict eager, the default), whatever has queued up when
# you next need a result (defer-and-fuse, one environment variable), or a whole function you hand to
# `torch.compile`. The choice trades time to the first result against time per step, and two habits
# decide whether the fast modes stay fast: keeping tensor shapes fixed, and not reading values back
# to Python in the middle of a step.
#
# ### What this notebook does
#
# It installs TorchTPU from PyPI and asks, on your chip:
#
# 1. **What does each mode cost?** First-call time and steady step time for a stack of matmuls, in
#    all three modes, and a check that they compute the same numbers.
# 2. **What do tiny operations cost?** The same comparison for a chain of small elementwise ops,
#    where strict eager is at its worst.
# 3. **Why does `torch.compile` recompile?** Four prompt lengths, compiled raw and then padded to a
#    bucket, with the compile count and a check that padding changed nothing.
# 4. **What does one `.item()` cost?** A loop with and without a value read back to Python each
#    step.
#
# ### The three rules used throughout
#
# 1. **Every result is measured here.** Numbers from the docs and the source repo appear beside yours
#    as context, never in place of them.
# 2. **One workload per comparison.** Within a section only the execution mode changes: same shapes,
#    same seed, same step count.
# 3. **Every mode starts cold.** Each measurement runs in a fresh Python process, so no mode inherits
#    another's compiled programs.
#
# ### How each section is organized
#
# * **What the source says:** the claim from the docs or the earlier work, and where it was made.
# * **What we are testing:** the claim, and the evidence that checks it.
# * A code cell to run.
# * **What you should see:** the expected result, and what a different one would mean.
#
# ### Before you start
#
# * **Runtime.** Runtime, then Change runtime type, then TPU. The target is a single v5e chip
#   (16 GB of HBM). Everything also runs on v6e. On 2026-10-21 Colab's v5e-1 runtime had Python
#   3.13.16, and the install below brought `torch` 2.14.1+cpu and `libtpu` 0.0.50.
# * **No token.** Every tensor is synthetic. Nothing is downloaded except the TorchTPU wheel and its
#   dependencies, and nothing asks for a Hugging Face token.
# * **Time.** About 15 minutes, most of it installing and compiling.
# * **Safety.** Nothing is uploaded and nothing outside the Colab VM changes.

# %% [markdown]
# ## 1. Setup and version check
#
# **What we are testing.** Nothing yet. This section installs TorchTPU and records the Python and
# PyTorch versions, because every number below depends on them.
#
# TorchTPU 0.2.0 needs Python 3.12 or newer and PyTorch 2.11 to 2.15. The extra index serves the
# CPU build of PyTorch: the TPU work happens in TorchTPU, and the default PyPI build would pull in
# several GB of CUDA libraries this machine has no use for.

# %%
# Cell 1.1: install TorchTPU
import importlib.metadata as md
import os
import subprocess
import sys

print("Python", sys.version.split()[0])
assert sys.version_info >= (3, 12), "TorchTPU 0.2.0 needs Python 3.12 or newer"

install = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch_tpu==0.2.0",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
    ],
    capture_output=True,
    text=True,
)
if install.returncode != 0:
    print(install.stdout[-2000:])
    print(install.stderr[-4000:])
    raise RuntimeError("pip install torch_tpu failed")

for package in ("torch_tpu", "torch", "libtpu"):
    print(f"{package:10s} {md.version(package)}")

# %% [markdown]
# A TPU chip belongs to one process at a time. If this notebook's own Python session opened it,
# every measurement below would have to share that one session and its compile cache. So the session
# never touches the chip. Cell 1.2 writes a small benchmark script, and each measurement runs it as a
# fresh process with the execution mode set in its environment.

# %%
# Cell 1.2: the benchmark script every section runs
import json
import pathlib

BENCH = pathlib.Path("torchtpu_bench.py")
BENCH.write_text(r'''
import json
import sys
import time

import torch

DEV = torch.device("tpu")
BF16 = torch.bfloat16


def sync():
    """Wait for the TPU. Work is queued asynchronously, so a timer without this measures nothing."""
    torch.accelerator.synchronize()


def first_and_steady(fn, steps):
    """Seconds for the first call (compiles included), then ms per call once warm."""
    start = time.perf_counter()
    out = fn()
    sync()
    first = time.perf_counter() - start
    fn()
    sync()
    start = time.perf_counter()
    for _ in range(steps):
        out = fn()
    sync()
    return first, (time.perf_counter() - start) / steps * 1e3, out


def task_smoke(mode):
    x = torch.ones((5, 5), device=DEV)
    y = torch.randn((5, 5), device=DEV)
    return {"device": str((x + y).device), "sum_of_ones": float(x.sum().cpu())}


def task_matmul(mode):
    torch.manual_seed(0)
    tokens, width, layers = 4096, 2048, 8
    stack = []
    for _ in range(layers):
        stack += [torch.nn.Linear(width, width, bias=False), torch.nn.SiLU()]
    model = torch.nn.Sequential(*stack).to(DEV, BF16).eval()
    x = torch.randn(tokens, width, dtype=BF16).to(DEV)
    fn = torch.compile(model, backend="tpu", dynamic=False) if mode == "compile" else model
    with torch.no_grad():
        first, step_ms, out = first_and_steady(lambda: fn(x), 50)
    flops = 2 * tokens * width * width * layers
    return {
        "first_s": first,
        "step_ms": step_ms,
        "gflop_per_step": flops / 1e9,
        "tflops": flops / (step_ms / 1e3) / 1e12,
        "checksum": float(out.float().abs().sum().cpu()),
    }


def task_tiny(mode):
    x = torch.randn(4096, device=DEV)

    def chain(y):
        for _ in range(64):
            y = torch.sin(y) * 0.5 + 0.25
        return y

    fn = torch.compile(chain, backend="tpu", dynamic=False) if mode == "compile" else chain
    first, step_ms, out = first_and_steady(lambda: fn(x), 20)
    return {"first_s": first, "step_ms": step_ms, "ops_per_step": 64 * 3, "checksum": float(out.sum().cpu())}


def task_shapes(mode):
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch.manual_seed(0)
    table = torch.randn(1024, 64, dtype=BF16).to(DEV)

    def masked_mean(tokens, mask):
        x = table[tokens] * mask[..., None]
        return x.sum(1) / mask.sum(1, keepdim=True)

    def call(fn, length, bucket):
        tokens = torch.randint(1, 1024, (1, length), generator=torch.Generator().manual_seed(length))
        width = (length + bucket - 1) // bucket * bucket
        padded = torch.zeros((1, width), dtype=torch.long)
        padded[:, :length] = tokens
        mask = torch.zeros((1, width), dtype=BF16)
        mask[:, :length] = 1
        start = time.perf_counter()
        out = fn(padded.to(DEV), mask.to(DEV)).float().cpu()
        return (time.perf_counter() - start) * 1e3, out

    lengths = [100, 101, 102, 250]
    result = {}
    outputs = {}
    for name, bucket in (("raw", 1), ("bucketed", 128)):
        torch._dynamo.reset()
        counters.clear()
        fn = torch.compile(masked_mean, backend="tpu", dynamic=False)
        passes = []
        for _ in range(2):
            row = []
            for n in lengths:
                ms, out = call(fn, n, bucket)
                row.append(round(ms, 1))
                outputs[(name, n)] = out
            passes.append(row)
        result[name] = {"ms": passes, "graphs": int(counters["stats"]["unique_graphs"])}
    result["lengths"] = lengths
    result["max_abs_diff"] = max(
        float((outputs[("raw", n)] - outputs[("bucketed", n)]).abs().max()) for n in lengths
    )
    return result


def task_readback(mode):
    torch.manual_seed(0)
    w = (torch.randn(1024, 1024) * 0.03).to(DEV, BF16)
    x0 = torch.randn(256, 1024).to(DEV, BF16)
    result = {}
    for name, check in (("no_read", False), ("item_each_step", True)):
        def run():
            x = x0
            for _ in range(20):
                for _ in range(8):
                    x = torch.tanh(x @ w)
                if check and x.abs().max().item() > 1e4:
                    raise RuntimeError("diverged")
            return x

        first, step_ms, out = first_and_steady(run, 5)
        result[name] = {
            "first_s": first,
            "ms_per_20_steps": step_ms,
            "checksum": float(out.float().abs().sum().cpu()),
        }
    return result


if __name__ == "__main__":
    task, mode = sys.argv[1], sys.argv[2]
    start = time.perf_counter()
    torch.ones(1, device=DEV).cpu()
    result = {"task": task, "mode": mode, "init_s": time.perf_counter() - start, "torch": torch.__version__}
    result.update(globals()["task_" + task](mode))
    print("RESULT " + json.dumps(result))
''')

# Each mode is an environment for a fresh process. Strict eager is the default; compile mode runs
# with default eager settings and wraps the work in torch.compile inside the script.
MODES = {
    "eager": {},
    "blocking": {"TORCH_TPU_LAUNCH_BLOCKING": "1"},
    "fuse": {"TORCH_TPU_DEFER_AND_FUSE": "1"},
    "compile": {},
}


def measure(task, mode, timeout=1200):
    """Run one task in one mode in a fresh process and return its result dict."""
    env = {**os.environ, **MODES[mode]}
    proc = subprocess.run(
        [sys.executable, str(BENCH), task, mode], env=env, capture_output=True, text=True, timeout=timeout
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if proc.returncode != 0 or not lines:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:])
        raise RuntimeError(f"{task} in {mode} mode failed (exit code {proc.returncode})")
    return json.loads(lines[-1][len("RESULT ") :])


smoke = measure("smoke", "eager")
print(f"device {smoke['device']}   torch {smoke['torch']}   first touch of the chip {smoke['init_s']:.1f} s")
assert smoke["device"].startswith("tpu"), "the benchmark did not land on the TPU"
assert smoke["sum_of_ones"] == 25.0

# %% [markdown]
# **What you should see.** A `tpu` device and a sum of 25. The first touch of the chip includes
# loading the TPU runtime, so it takes seconds; every result below is timed after it.

# %% [markdown]
# ## 2. Three modes, one stack of matmuls
#
# **What the source says.** The execution-model guide: strict eager compiles each operation on its
# own at a low optimization level, which gives "the fastest possible time-to-first-result".
# `torch.compile` hands XLA the whole function at a higher level, for "maximum warmed-up throughput".
# Defer-and-fuse (`TORCH_TPU_DEFER_AND_FUSE=1`) lets operations queue until a result is needed and
# compiles the queue as one program, and "shows up to 50% performance improvement over eager
# default, with no change to model code".
#
# **What we are testing.** Eight `Linear` layers of width 2048 with SiLU between them, on 4,096
# tokens in bfloat16: enough work per operation that dispatch overhead should matter little. Each mode reports its first call, its warm step, and a checksum of the output. The
# checksums should agree to bfloat16 precision, because every mode runs the same arithmetic.

# %%
# Cell 2.1: matmul stack in each mode
matmul = {mode: measure("matmul", mode) for mode in ("eager", "fuse", "compile")}
print(f"{matmul['eager']['gflop_per_step']:.0f} GFLOP per step\n")

print(f"{'mode':8s} {'first call s':>13s} {'step ms':>9s} {'TFLOP/s':>8s} {'checksum':>14s}")
for mode, r in matmul.items():
    print(f"{mode:8s} {r['first_s']:13.2f} {r['step_ms']:9.2f} {r['tflops']:8.1f} {r['checksum']:14.1f}")

base = matmul["eager"]
for mode in ("fuse", "compile"):
    r = matmul[mode]
    rel = abs(r["checksum"] - base["checksum"]) / max(abs(base["checksum"]), 1.0)
    print(f"{mode} vs eager: step {base['step_ms'] / r['step_ms']:.2f}x faster, checksum differs by {rel:.1e}")

# %% [markdown]
# **What you should see.** Matching checksums: all three modes run the same arithmetic. The first-call
# column follows the guide. Strict eager answers first, because each operation compiles on its own at
# the low optimization level; the other two compile the whole stack at the higher level before
# anything runs. In two runs on a Colab v5e-1 on 2026-10-21 that was 0.15 s for strict eager against
# 2.5 to 2.7 s for defer-and-fuse and 3.0 to 3.2 s for `torch.compile`.
#
# The warm step favored both fused modes over strict eager, by 1.72x to 1.73x for defer-and-fuse and
# 1.52x for `torch.compile` in those runs, with defer-and-fuse the faster of the two at 1.68 to 1.69 ms
# against 1.91 to 1.92 ms.
# Run it on your chip before deciding between them: the guide predicts `torch.compile` ahead, and on
# this stack of large matmuls the two land close enough that the order can depend on the chip and
# the TorchTPU version.
#
# Google publishes a bfloat16 peak of 197 TFLOP/s for a v5e chip. The TFLOP/s column shows how much of that each mode
# reaches on this workload.

# %% [markdown]
# ## 3. Tiny operations
#
# **What the source says.** "Enqueueing an executable with PjRt takes a small amount of time; usually,
# 50 microseconds or less." When each operation finishes faster than the next can be queued, the chip
# sits idle between them. The guide calls these gaps microbubbles and recommends defer-and-fuse or
# `torch.compile` to merge the operations.
#
# **What we are testing.** A chain of 64 `sin`, multiply and add steps on 4,096 floats: 192 operations
# per call, each one microseconds of work. A fourth mode joins here. Launch-blocking
# (`TORCH_TPU_LAUNCH_BLOCKING=1`) waits for the chip after every operation, which the guide reserves
# for debugging. It shows the cost of every round trip with nothing overlapped.

# %%
# Cell 3.1: a chain of small elementwise ops in each mode
tiny = {mode: measure("tiny", mode) for mode in ("eager", "blocking", "fuse", "compile")}

print(f"{'mode':9s} {'first call s':>13s} {'step ms':>9s} {'us per op':>10s} {'checksum':>12s}")
for mode, r in tiny.items():
    per_op = r["step_ms"] * 1e3 / r["ops_per_step"]
    print(f"{mode:9s} {r['first_s']:13.2f} {r['step_ms']:9.3f} {per_op:10.1f} {r['checksum']:12.4f}")
for mode in ("blocking", "fuse", "compile"):
    print(f"{mode} / eager step time: {tiny[mode]['step_ms'] / tiny['eager']['step_ms']:.2f}x")

# %% [markdown]
# **What you should see.** Here the modes separate in step time. Strict eager pays a dispatch for
# each of the 192 operations; launch-blocking pays that and a full round trip on top. The
# microseconds-per-operation column for strict eager is the dispatch cost the guide describes,
# measured on your machine. In two runs on a Colab v5e-1 on 2026-10-21 it was 51.5 and 71.5
# microseconds, at or above the guide's "50 microseconds or less", so the host was the bottleneck
# for the whole chain. This section moves more between runs than the matmul stack does, because it
# times the host.
#
# Defer-and-fuse cut the step to 0.55x to 0.62x of strict eager in those runs, and `torch.compile`,
# which turns the whole chain into one program before the first call, ran it 99x to 128x faster.
# Launch-blocking was 3.5x to 3.7x slower than strict eager, which is why the guide keeps it for
# debugging.
#
# Real models mix both kinds of work. Normalization, activation and residual adds are tiny
# operations between large matmuls, which is why the guide recommends defer-and-fuse for code you
# cannot wrap in `torch.compile`.

# %% [markdown]
# ## 4. Why `torch.compile` recompiles
#
# **What the source says.** `torch.compile` with `dynamic=False` builds one program per input shape.
# The serving code in this repo's rigs (`tpu_openai_server.py`) is built around that: it pads every
# prompt into one fixed-size buffer and compiles the decode step once, which measured 5.2 ms per
# token for Gemma 4 E2B on a v6e-1 (2026-08-02).
#
# **What we are testing.** A masked mean over embedded tokens, compiled once, then called with
# prompts of 100, 101, 102 and 250 tokens, twice over. First raw, then padded up to a multiple of
# 128 with a mask marking the real tokens. The count of compiled graphs comes from PyTorch's own
# counter. The last line compares each padded result with the raw one, because padding is only safe
# if the mask keeps the padded positions out of the answer.

# %%
# Cell 4.1: raw lengths against 128-token buckets
shapes = measure("shapes", "compile")

print("lengths:", shapes["lengths"])
for name in ("raw", "bucketed"):
    r = shapes[name]
    print(f"\n{name}: {r['graphs']} compiled graph(s)")
    print(f"  first pass ms:  {r['ms'][0]}")
    print(f"  second pass ms: {r['ms'][1]}")
print(f"\nlargest difference, padded vs raw: {shapes['max_abs_diff']:.2e}")

# %% [markdown]
# **What you should see.** Four graphs for the raw lengths, one per length, each first call paying
# a compile and every second-pass call fast. Two graphs once padded: 100, 101 and 102 all round up to
# 128, and 250 to 256. The difference between padded and raw results should be zero or at bfloat16
# rounding, because the mask zeroes the padding before the sum and the divisor counts only real
# tokens.
#
# A server sees far more than four lengths. Without buckets, each new length costs a compile while a
# user waits; with them, the number of compiles is fixed in advance by the bucket list.

# %% [markdown]
# ## 5. The cost of reading a value back
#
# **What the source says.** "An operation like `if my_tensor.all()` will silently construct a
# boolean-valued tensor, synchronize the stream, and copy that boolean to the CPU before
# continuing." In strict eager mode that drains the queue. In defer-and-fuse or compiled mode it also
# splits the fused program in two.
#
# **What we are testing.** Twenty steps of eight `tanh(x @ w)` layers, run twice: once with no reads,
# and once checking `x.abs().max().item()` after every step, the kind of divergence check training
# loops often carry. Both run in strict eager and in defer-and-fuse. The checksums should match
# within each mode, because the check never fires.

# %%
# Cell 5.1: twenty steps with and without a read-back each step
readback = {mode: measure("readback", mode) for mode in ("eager", "fuse")}

print(f"{'mode':6s} {'variant':16s} {'ms per 20 steps':>16s} {'checksum':>12s}")
for mode, r in readback.items():
    for variant in ("no_read", "item_each_step"):
        v = r[variant]
        print(f"{mode:6s} {variant:16s} {v['ms_per_20_steps']:16.2f} {v['checksum']:12.2f}")
    cost = r["item_each_step"]["ms_per_20_steps"] / r["no_read"]["ms_per_20_steps"]
    print(f"{mode:6s} read-back costs {cost:.2f}x\n")

# %% [markdown]
# **What you should see.** The read-back variant slower in both modes. Each `.item()` stops the
# host until the chip has finished everything queued, so the host can no longer stay ahead of the
# chip; in defer-and-fuse it also ends the fused program at that point. In two runs on a Colab v5e-1
# on 2026-10-21 the read-back cost 1.34x to 1.47x in strict eager and 1.44x to 1.62x in
# defer-and-fuse, with defer-and-fuse faster in both variants every time. If you need a check like this, run it every N steps, or
# keep the flag on the device and read it once at the end.

# %% [markdown]
# ## 6. Scorecard
#
# **What we are testing.** Each claim from the docs and the source repo next to the result you just
# measured. The claims are about direction, which should hold on any chip. The absolute times belong
# to your chip and this TorchTPU version.

# %%
# Cell 6.1: the scorecard
fuse_gain = tiny["eager"]["step_ms"] / tiny["fuse"]["step_ms"]
rows = [
    (
        "strict eager has the fastest first call (matmul stack)",
        "guide",
        min(matmul, key=lambda m: matmul[m]["first_s"]),
    ),
    (
        "torch.compile has the fastest warm step (matmul stack)",
        "guide",
        min(matmul, key=lambda m: matmul[m]["step_ms"]),
    ),
    ("defer-and-fuse speedup over strict eager (tiny ops)", "up to 1.5x", f"{fuse_gain:.2f}x"),
    (
        "torch.compile speedup over strict eager (tiny ops)",
        "",
        f"{tiny['eager']['step_ms'] / tiny['compile']['step_ms']:.2f}x",
    ),
    (
        "launch-blocking slowdown (tiny ops)",
        "debug only",
        f"{tiny['blocking']['step_ms'] / tiny['eager']['step_ms']:.2f}x",
    ),
    ("compiled graphs, 4 raw lengths", "one per shape", str(shapes["raw"]["graphs"])),
    ("compiled graphs, same lengths in 128 buckets", "one per bucket", str(shapes["bucketed"]["graphs"])),
    ("padded result vs raw, max difference", "0", f"{shapes['max_abs_diff']:.1e}"),
    (
        "cost of one .item() per step, defer-and-fuse",
        "graph break",
        f"{readback['fuse']['item_each_step']['ms_per_20_steps'] / readback['fuse']['no_read']['ms_per_20_steps']:.2f}x",
    ),
]
print(f"{'claim':58s} {'source':>16s} {'today':>10s}")
for claim, source, today in rows:
    print(f"{claim:58s} {source:>16s} {today:>10s}")

# %% [markdown]
# **What you should see.** Most rows agreeing with their source. Two are worth reading closely. The
# defer-and-fuse row may exceed the guide's "up to 1.5x": a chain of pure elementwise operations is
# close to the best case for fusion (1.62x and 1.80x in two runs on a Colab v5e-1 on 2026-10-21). The
# warm-step row may name defer-and-fuse rather than `torch.compile` for the matmul stack, as it did in
# both runs; the tiny-op rows show where `torch.compile` pulls far ahead.
#
# **So, which one?** Start in strict eager while the code is changing: it gives the fastest first
# result and the clearest errors. Turn on `TORCH_TPU_DEFER_AND_FUSE=1` once it runs, for a speedup
# that needs no code change. Wrap the hot loop in `torch.compile(..., dynamic=False)` when it has
# settled, with inputs padded to a few fixed shapes and no value read back inside the loop.

# %% [markdown]
# ## 7. Your notes
#
# Record where your numbers differ from the ones above, with the evidence. Suggested questions:
#
# * **Chip.** Which chip, Python, PyTorch and TorchTPU versions (Cells 1.1 and 1.2)?
# * **Modes.** Which mode had the fastest first call and which the fastest warm step (Section 2)?
# * **Tiny ops.** What did one dispatch cost in strict eager (Section 3)?
# * **Shapes.** Did bucketing cut the compile count from four to two (Section 4)?
# * **Read-back.** How much did one `.item()` per step cost in each mode (Section 5)?

# %%
# Cell 7.1: your notes (edit this cell)
notes = """
Chip and versions:
Modes (Section 2):
Tiny ops (Section 3):
Shapes (Section 4):
Read-back (Section 5):
Other findings:
"""
print(notes)
