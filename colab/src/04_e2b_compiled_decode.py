# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B on TorchTPU: one compiled decode step, and the speed-up
#
# **Author:** William McLean, Google Developer Expert
#
# **Based on:** the repack
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)
# and its 4-bit-tables twin
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4),
# notebooks 02 and 03 of this series, and the plain-PyTorch decode loop in
# [`xbill9/gemma4-dev`](https://github.com/xbill9/gemma4-dev) (`local-pytorch-1650ti-2b-q4w4a16`).
#
# ### What this notebook is for
#
# Making PyTorch on TorchTPU decode Gemma 4 E2B fast, with Hugging Face Transformers' own model code
# and no custom kernels. Notebook 02 found every way of running the model on a v5e-1 in single digits
# of tokens per second, and notebook 03 found a JAX engine on the same chip 10.6x to 14.8x faster.
# This notebook applies a recipe that took the same checkpoint from 18 to 90 tokens per second in
# plain PyTorch on a 4 GB laptop GPU, and measures each step of it on the TPU.
#
# ### The recipe
#
# 1. **A fixed-size KV cache.** Hugging Face's `StaticCache` holds keys and values in tensors that never
#    change shape, so one decode step is the same program every token.
# 2. **A one-line fix to the sliding-window cache layer**, which tracks its length as a Python
#    integer. A compiled step treats that integer as a constant and recompiles whenever it changes,
#    which is every token.
# 3. **One compiled decode step:** the model's forward pass for one token, plus the choice of the next
#    token, under `torch.compile`.
# 4. **The 4-bit per-layer embedding table**, the largest tensor in the model, kept 4-bit on the
#    chip and unpacked one row at a time.
#
# ### What each section shows
#
# * **Section 2, where the 4-bit weights go.** Why loading the repack 4-bit saved no memory in
#   notebook 02.
# * **Section 3, why the compiled step split.** Notebook 02's compiled step became 24 graphs with 60
#   graph breaks. This section finds the line of code that causes them.
# * **Section 4, the speed-up.** Seven ways, from Hugging Face `generate` to the full recipe, with the
#   JAX engine from notebook 03 as the reference, all on one stopwatch.
# * **Section 5, agreement.** Every way should write the same answer.
# * **Section 6, memory.** What each way holds on the chip.
#
# ### The three rules used throughout
#
# 1. **Every result is measured here.** Numbers from the earlier work appear beside yours as context,
#    never in place of them.
# 2. **One checkpoint, one prompt, one stopwatch.** Every way continues the same chat prompt, greedily,
#    and is timed the same way from outside.
# 3. **Every way starts cold.** Each one runs in a fresh Python process, because only one process can
#    hold the chip and no way should inherit another's compiled programs.
#
# ### How each section is organized
#
# * **What the source says:** the earlier result, and where it came from.
# * **What we are testing:** the claim, and the evidence that checks it.
# * A code cell to run.
# * **What you should see:** the expected result, and what a different one would mean.
#
# ### Before you start
#
# * **Runtime.** Runtime, then Change runtime type, then TPU. The target is a single v5e chip
#   (16 GB of HBM).
# * **No token.** Both repacks and the JAX engine are public. Nothing asks for a Hugging Face token.
# * **Disk and time.** About 8.5 GB of download. The whole notebook takes about 25 minutes, most of it
#   loading the model once per way.
# * **Safety.** Nothing is uploaded and nothing outside the Colab VM changes.

# %% [markdown]
# ## 1. Setup
#
# **What we are testing.** Nothing yet. This section installs TorchTPU, downloads the checkpoint and
# the one 4-bit table file, fetches the JAX engine at a fixed commit, and records the versions every
# number below depends on.

# %%
# Cell 1.1: install, download, versions
import importlib.metadata as md
import json
import os
import pathlib
import subprocess
import sys
import time

install = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch_tpu==0.2.0",
        "compressed-tensors",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
    ],
    capture_output=True,
    text=True,
)
if install.returncode != 0:
    print(install.stdout[-2000:])
    print(install.stderr[-4000:])
    raise RuntimeError("pip install failed")

print("Python", sys.version.split()[0])
for package in ("torch_tpu", "torch", "libtpu", "transformers", "compressed-tensors", "jax"):
    print(f"{package:20s} {md.version(package)}")

from huggingface_hub import snapshot_download

MODEL = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
start = time.perf_counter()
MODEL_DIR = pathlib.Path(snapshot_download(MODEL))
EMB4_DIR = pathlib.Path(snapshot_download(EMB4, allow_patterns=["model-rest-001.safetensors"]))
for name, d in ((MODEL, MODEL_DIR), (EMB4, EMB4_DIR)):
    size = sum(f.stat().st_size for f in d.rglob("*.safetensors"))
    print(f"{name}: {size / 1e9:.2f} GB")
print(f"downloaded in {time.perf_counter() - start:.0f} s")

# The JAX engine from notebook 03, at the same commit.
TPU_JAX_COMMIT = "4b9f8e94619f1995206617e07e3c4ac62685d47b"
ENGINE_DIR = pathlib.Path("/content/tpu-jax" if os.path.isdir("/content") else os.path.expanduser("~/tpu-jax-src"))
if not (ENGINE_DIR / ".git").exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/xbill9/tpu-jax", str(ENGINE_DIR)], check=True)
subprocess.run(["git", "-C", str(ENGINE_DIR), "checkout", "-q", TPU_JAX_COMMIT], check=True)
print("tpu-jax at", TPU_JAX_COMMIT[:7])

# %% [markdown]
# Cell 1.2 tokenizes the prompt once, so every way starts from the same token ids, and writes the
# script every later section runs. Each task in it loads the model in a fresh process, does one thing,
# and prints its result as one line of JSON. The notebook's own Python session never imports PyTorch
# or JAX, so the chip stays free for those processes.

# %%
# Cell 1.2: the prompt and the task script
TOKENIZE = """
import json, sys
from transformers import AutoTokenizer
ids = AutoTokenizer.from_pretrained(sys.argv[1]).apply_chat_template(
    [{"role": "user", "content": "Explain in two sentences why the sky is blue."}], add_generation_prompt=True, tokenize=True)
if hasattr(ids, "keys"):
    ids = ids["input_ids"]
if ids and isinstance(ids[0], list):
    ids = ids[0]
json.dump(list(ids), open("prompt_ids.json", "w"))
"""
subprocess.run([sys.executable, "-c", TOKENIZE, MODEL], check=True, capture_output=True)
PROMPT_IDS = json.loads(pathlib.Path("prompt_ids.json").read_text())
print(f"prompt: {len(PROMPT_IDS)} tokens")
NEW_TOKENS = 64

TASKS = pathlib.Path("decode_tasks.py")
TASKS.write_text(r'''
import collections
import gc
import json
import sys
import time

TASK, MODEL, EMB4_DIR, ENGINE_DIR, NEW_TOKENS = sys.argv[1:6]
NEW_TOKENS = int(NEW_TOKENS)
PROMPT_IDS = json.load(open("prompt_ids.json"))
MAX_CACHE = 256
KEY_MAPPING = {r"^model\.language_model\.": "model."}
if TASK != "jax_bf16":
    import torch


def stopwatch(run):
    """Time run(1) and run(NEW_TOKENS) twice each; return first and warm timings and the tokens."""
    t = {}
    for label in ("first", "warm"):
        for n in (1, NEW_TOKENS):
            start = time.perf_counter()
            tokens = run(n)
            t[f"{label}_{n}"] = time.perf_counter() - start
    return t, tokens


# --- PyTorch on TorchTPU ------------------------------------------------------------------------------------------
def sync():
    torch.accelerator.synchronize()


def hbm():
    return torch.accelerator.memory_allocated() / 1e9


def load(compressed=False):
    import transformers
    from transformers import AutoModelForCausalLM

    kw = {"dtype": torch.bfloat16, "key_mapping": KEY_MAPPING}
    if not compressed:
        # Unpack the 4-bit weights to bf16 once, at load.
        kw["quantization_config"] = transformers.CompressedTensorsConfig(run_compressed=False)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="tpu", **kw).eval()
    sync()
    return model, {"load_s": time.perf_counter() - start, "hbm_load_gb": hbm()}


def task_hbm_compressed():
    """Load with the weights left 4-bit, and watch the chip's memory across the first forward passes."""
    model, info = load(compressed=True)
    gate = model.model.layers[0].mlp.gate_proj
    info["packed_before"] = hasattr(gate, "weight_packed")
    ids = torch.tensor([PROMPT_IDS], device="tpu")
    for i in (1, 2):
        with torch.no_grad():
            model(input_ids=ids)
        sync()
        info[f"hbm_after_forward_{i}_gb"] = hbm()
    info["packed_after"] = hasattr(gate, "weight_packed")
    info["weight_after"] = str(gate.weight.dtype) if hasattr(gate, "weight") else None
    return info


def packed_embedding(packed, scale, embed_scale):
    """The per-layer table, kept 4-bit: look up packed rows, then unpack only those.

    compressed-tensors' pack-quantized layout: eight 4-bit values per int32, lowest first, stored with an
    offset of 8, one scale per 32 values. Unpacking in fp32 and rounding to bf16 once reproduces the bf16
    table, and the result is scaled the way Gemma4TextScaledWordEmbedding scales it.
    """

    class Packed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.packed, self.scale = torch.nn.Buffer(packed), torch.nn.Buffer(scale)
            self.shifts = torch.nn.Buffer(torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device))
            self.embed_scale = float(embed_scale)

        def forward(self, ids):
            p, s = self.packed[ids], self.scale[ids]
            q = ((p.unsqueeze(-1) >> self.shifts) & 15).to(torch.float32) - 8
            w = (q.view(*ids.shape, s.shape[-1], -1) * s.unsqueeze(-1).to(torch.float32)).to(torch.bfloat16)
            return w.view(*ids.shape, -1) * torch.tensor(self.embed_scale, dtype=torch.bfloat16)

    return Packed()


def build(unwrap, ple4):
    """The model with a static KV cache, optionally without the offload wrappers and with the 4-bit table."""
    import transformers.cache_utils as cu
    from transformers import StaticCache

    model, info = load()
    info["wrapped_modules"] = sum(hasattr(m, "_original_forward_func") for m in model.modules())
    if unwrap:
        from compressed_tensors.offload.dispatch import remove_dispatch

        # compressed-tensors wraps the forward of every module to move its inputs to the module's device.
        # Every tensor is already on the TPU, so the wrappers do nothing here but split a compiled step.
        remove_dispatch(model)
        info["wrapped_modules_after"] = sum(hasattr(m, "_original_forward_func") for m in model.modules())
    if ple4:
        from safetensors import safe_open

        name = "model.language_model.embed_tokens_per_layer"
        with safe_open(f"{EMB4_DIR}/model-rest-001.safetensors", "pt") as f:
            packed, scale = f.get_tensor(name + ".weight_packed"), f.get_tensor(name + ".weight_scale")
        old = model.model.embed_tokens_per_layer
        model.model.embed_tokens_per_layer = packed_embedding(packed.to("tpu"), scale.to("tpu"), old.scalar_embed_scale)
        old.weight.data = torch.empty(0, dtype=torch.bfloat16, device="tpu")
        del old
        gc.collect()
        sync()
        info["hbm_ple4_gb"] = hbm()
    # StaticSlidingWindowLayer counts its length in a Python int, which a compiled step would treat as a
    # constant and recompile on every token. While the cache is no longer than the sliding window it never
    # slides, so StaticLayer's methods, which count in a tensor on the chip, give the same results.
    assert model.config.sliding_window >= MAX_CACHE
    for method in ("update", "get_seq_length", "get_mask_sizes"):
        setattr(cu.StaticSlidingWindowLayer, method, getattr(cu.StaticLayer, method))
    return model, StaticCache(config=model.config, max_cache_len=MAX_CACHE), info


def task_diagnose():
    """Compile the step with fullgraph=True, which raises at the first graph break, and report where."""
    model, cache, info = build(unwrap=False, ple4=False)
    ids = torch.tensor([PROMPT_IDS], device="tpu")
    tok = torch.zeros((1, 1), dtype=torch.long, device="tpu")

    def step():
        return model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1)

    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        try:
            torch.compile(step, backend="tpu", dynamic=False, fullgraph=True)()
            info["break"] = None
        except Exception as exc:
            text = str(exc)
            info["break"] = text.splitlines()[0]
            info["frames"] = [line.strip() for line in text.splitlines() if line.strip().startswith("File ")][-3:]
            info["line"] = [line.strip() for line in text.splitlines() if "data_ptr" in line][:1]
    return info


def task_generate():
    model, info = load()
    prompt = torch.tensor([PROMPT_IDS], device="tpu")

    def run(n):
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=n, min_new_tokens=n, do_sample=False)
        sync()
        return out[0, len(PROMPT_IDS) :].tolist()

    timings, tokens = stopwatch(run)
    return {**info, **timings, "tokens": tokens, "hbm_end_gb": hbm()}


def task_step(compiled, unwrap=True, ple4=False):
    from torch._dynamo.utils import counters

    model, cache, info = build(unwrap, ple4)
    ids = torch.tensor([PROMPT_IDS], device="tpu")
    tok = torch.zeros((1, 1), dtype=torch.long, device="tpu")

    def step():
        # One token in, the next token out: the forward pass reads and extends the static cache in place.
        return model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1)

    if compiled:
        step = torch.compile(step, backend="tpu", dynamic=False)

    def decode(n, stream=False):
        with torch.no_grad():
            cache.reset()
            first = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
            first = first.argmax(-1)
            tok.copy_(first.view(1, 1))
            out = [first]
            for _ in range(n - 1):
                nxt = step()
                tok.copy_(nxt.view(1, 1))
                out.append(nxt)
                if stream:
                    int(nxt)  # read each token back as it is made, as a streaming server must
            return torch.cat(out).tolist()

    counters.clear()
    timings, tokens = stopwatch(decode)
    for n in (1, NEW_TOKENS):
        start = time.perf_counter()
        decode(n, stream=True)
        timings[f"stream_{n}"] = time.perf_counter() - start
    try:
        peak = torch.accelerator.max_memory_allocated() / 1e9
    except Exception:
        peak = None
    return {
        **info,
        **timings,
        "tokens": tokens,
        "graphs": int(counters["stats"]["unique_graphs"]),
        "graph_breaks": int(sum(counters["graph_break"].values())),
        "hbm_end_gb": hbm(),
        "hbm_peak_gb": peak,
    }


# --- JAX, the engine from notebook 03 -----------------------------------------------------------------------------
def task_jax_bf16():
    sys.path.insert(0, ENGINE_DIR)
    import jax
    from huggingface_hub import snapshot_download
    from jax_engine import JaxGemmaEngine

    engine = JaxGemmaEngine(MODEL, quant_mode="w4a16", dequant_at_load=True)
    start = time.perf_counter()
    engine.load(local_dir=snapshot_download(MODEL))
    info = {"load_s": time.perf_counter() - start}

    def run(n):
        return engine.generate(PROMPT_IDS, max_new_tokens=n)[0]

    timings, tokens = stopwatch(run)
    return {**info, **timings, "tokens": tokens, "hbm_end_gb": jax.devices()[0].memory_stats()["bytes_in_use"] / 1e9}


if __name__ == "__main__":
    tasks = {
        "hbm_compressed": task_hbm_compressed,
        "diagnose": task_diagnose,
        "generate": task_generate,
        "static_eager": lambda: task_step(compiled=False),
        "static_fuse": lambda: task_step(compiled=False),
        "step_wrapped": lambda: task_step(compiled=True, unwrap=False),
        "step": lambda: task_step(compiled=True),
        "step_ple4": lambda: task_step(compiled=True, ple4=True),
        "jax_bf16": task_jax_bf16,
    }
    try:
        result = tasks[TASK]()
    except Exception as exc:  # a way that fails is a result too; report it instead of crashing the cell
        result = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    print("RESULT " + json.dumps(result))
''')

ENVIRONMENTS = {"static_fuse": {"TORCH_TPU_DEFER_AND_FUSE": "1"}}


def run(task, timeout=1800):
    """Run one task in a fresh process and return its result dict."""
    env = {**os.environ, **ENVIRONMENTS.get(task, {})}
    args = [sys.executable, str(TASKS), task, MODEL, str(EMB4_DIR), str(ENGINE_DIR), str(NEW_TOKENS)]
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:])
        raise RuntimeError(f"task {task} exited with code {proc.returncode} and no result")
    return json.loads(lines[-1][len("RESULT ") :])


print("task script written:", TASKS.resolve())

# %% [markdown]
# ## 2. Where the 4-bit weights go
#
# **What the source says.** Notebook 02 loaded the repack with its weights left 4-bit and found the TPU
# holding the same 9.85 GB as with the weights unpacked to bf16. Transformers' compressed-tensors
# integration (`quantizer_compressed_tensors.py` in 5.20.1) says it leaves the weights compressed and
# has a hook "decompress them on the first forward pass".
#
# **What we are testing.** The chip's memory right after loading the repack 4-bit, after the first
# forward pass and after the second, and whether the first decoder layer's projection still holds
# packed weights afterwards.

# %%
# Cell 2.1: memory across the first forward passes, weights loaded 4-bit
h = run("hbm_compressed")
print(f"after load:            {h['hbm_load_gb']:5.2f} GB   packed weights present: {h['packed_before']}")
print(f"after forward pass 1:  {h['hbm_after_forward_1_gb']:5.2f} GB")
print(f"after forward pass 2:  {h['hbm_after_forward_2_gb']:5.2f} GB   packed weights present: {h['packed_after']}")
print(f"weight dtype afterwards: {h['weight_after']}")

# %% [markdown]
# **What you should see.** About 6.6 GB after loading, 9.8 GB after the first forward pass, no change
# after the second, and the packed weights replaced by bf16 ones. In a run on a Colab v5e-1 on
# 2026-10-23: 6.56, 9.83 and 9.83 GB. The first forward pass unpacks every layer to bf16 once and keeps
# the result, so after that the model is the bf16 model. Keeping a 4-bit format on the chip takes code
# that reads the packed values itself, which is what Section 4's last PyTorch way does for the largest
# table.

# %% [markdown]
# ## 3. Why the compiled step split
#
# **What the source says.** Notebook 02's compiled step became 24 graphs with 60 graph breaks, every
# reason "Data pointer comparison". PyTorch's compiler cannot trace a comparison of two tensors'
# memory addresses, so it cuts the program at each one and runs the code between the pieces
# operation by operation.
#
# **What we are testing.** Where the comparison is. `fullgraph=True` makes `torch.compile` raise at the
# first graph break instead of cutting there, and the error names the line it stopped on.

# %%
# Cell 3.1: the first graph break, by file and line
d = run("diagnose")
print("modules with a wrapped forward:", d["wrapped_modules"])
print("break:", d["break"])
for frame in d.get("frames", []):
    print("  ", frame[:150])
for line in d.get("line", []):
    print("   ->", line)

# %% [markdown]
# **What you should see.** Hundreds of wrapped modules, and a break inside `compressed_tensors/offload`,
# on the line `if tensor.data_ptr() == value.data_ptr():`. When compressed-tensors loads a model it
# wraps the forward of every module so that the module's inputs move to its device first, and that
# check is how it avoids copying a tensor already in place. On a single chip every tensor is already
# on the TPU, so the wrappers move nothing, and `remove_dispatch` from the same package takes them off.
# Section 4 runs the compiled step both with the wrappers and without.

# %% [markdown]
# ## 4. The speed-up
#
# **What the source says.** Notebooks 02 and 03 on a Colab v5e-1: Hugging Face `generate` at 4.7 to
# 5.6 tokens per second in strict eager and 8.8 to 9.7 in defer-and-fuse, and the JAX engine at 130.8
# to 133.9 with bf16 weights. On a GTX 1650 Ti laptop GPU, the recipe at the top of this notebook took
# plain PyTorch from 18.1 to 57 tokens per second before any custom kernel, and to 90 with them.
#
# **What we are testing.** Seven ways, each building on the one before:
#
# 1. **`generate`:** Hugging Face `generate` in strict eager, the baseline.
# 2. **Static cache, strict eager:** the static KV cache and this notebook's own decode loop, with
#    nothing compiled.
# 3. **Static cache, defer-and-fuse:** the same loop with `TORCH_TPU_DEFER_AND_FUSE=1`.
# 4. **Compiled step, wrappers on:** the decode step under `torch.compile`, with compressed-tensors'
#    wrappers left on, the situation notebook 02 measured.
# 5. **Compiled step:** the same with the wrappers removed.
# 6. **Compiled step, 4-bit table:** the same with the per-layer embedding table kept 4-bit.
# 7. **JAX bf16:** the JAX engine from notebook 03 with its weights unpacked to bf16 at load, as the
#    reference.
#
# Every way writes a 1-token and a 64-token greedy continuation twice. Decode speed is 63 tokens divided
# by the warm 64-token time minus the warm 1-token time, which removes the prompt. The compiled ways
# also run once more reading each token back to the host as it is made, which a server that streams
# its reply has to do. This cell takes about 15 minutes.

# %%
# Cell 4.1: run every way (each loads the model fresh; this is the slow cell)
WAYS = ("generate", "static_eager", "static_fuse", "step_wrapped", "step", "step_ple4", "jax_bf16")
LABELS = {
    "generate": "generate",
    "static_eager": "static cache, strict eager",
    "static_fuse": "static cache, defer-and-fuse",
    "step_wrapped": "compiled step, wrappers on",
    "step": "compiled step",
    "step_ple4": "compiled step, 4-bit table",
    "jax_bf16": "JAX bf16",
}
results = {}
for way in WAYS:
    start = time.perf_counter()
    results[way] = run(way)
    print(f"{LABELS[way]:30s} finished in {time.perf_counter() - start:4.0f} s: {results[way].get('error', 'ok')}")


def rate(r, prefix="warm"):
    return (NEW_TOKENS - 1) / (r[f"{prefix}_{NEW_TOKENS}"] - r[f"{prefix}_1"])


# %%
# Cell 4.2: the table
base = rate(results["generate"])
print(
    f"{'way':30s} {'first 64 s':>10s} {'tok/s':>7s} {'x generate':>10s} {'streamed':>9s} {'graphs':>7s} {'breaks':>7s}"
)
for way in WAYS:
    r = results[way]
    if "error" in r:
        print(f"{LABELS[way]:30s} failed: {r['error'][:90]}")
        continue
    streamed = f"{rate(r, 'stream'):9.1f}" if "stream_1" in r else f"{'':9s}"
    graphs = f"{r['graphs']:7d} {r['graph_breaks']:7d}" if "graphs" in r else ""
    print(
        f"{LABELS[way]:30s} {r[f'first_{NEW_TOKENS}']:10.1f} {rate(r):7.1f} {rate(r) / base:10.1f} {streamed} {graphs}"
    )

# %% [markdown]
# **What you should see.** The compiled step as one graph with no breaks, several times faster than
# `generate`, and the 4-bit table faster again. In two runs on a Colab v5e-1 on 2026-10-23: `generate`
# at 5.0 to 5.3 tokens per second; the static cache at 4.2 to 4.5 in strict eager and 10.6 to 11.0 in
# defer-and-fuse; the compiled step at 6.4 to 6.6 as 24 graphs with the wrappers on, and at 31.3 to
# 33.4 as one graph with them off, 5.9 to 6.7 times `generate`; 50.2 to 50.4 with the 4-bit table, 9.4
# to 10.1 times `generate`; and the JAX engine at 131.2 to 132.9.
#
# * **The wrappers were the whole difference between 24 graphs and one.** With them removed, the
#   same step compiled once and every later token reused it. Its first 64-token call, which includes
#   compiling, took 36.5 to 38.3 s.
# * **The static cache helps defer-and-fuse as well.** Under `generate` its first call took 341 to 359 s
#   in notebook 02, because every token is a new shape to compile. With the static cache every step
#   has the same shape, and the whole way, loading and compiling included, finished in 102 to 103 s.
# * **Streaming costs the 4-bit table its lead.** Read back token by token, the two compiled ways ran
#   at 31.8 to 36.5 and 32.4 to 37.2 tokens per second, so on this path each token's round trip to the
#   host, rather than the chip's work, set the pace.
# * **The gap to JAX shrank from 13.8x to 2.6x.** Notebook 03 measured JAX bf16 at 13.8x to 14.8x the
#   fastest PyTorch way; here it is 2.6x the 4-bit-table step in both runs, on the same chip, prompt
#   and stopwatch. The longer the reply, the more the compiled step gains: in 128-token probe runs
#   `generate` fell to 3.4 tokens per second while the compiled step held at 34.0.

# %% [markdown]
# ## 5. Do they agree?
#
# **What we are testing.** Every way should write the same answer. `generate` is told to write all 64
# tokens (`min_new_tokens`), so it never ends its turn, while the other ways end the answer with
# `<turn|>` (token 106) and carry on after it. The comparison therefore runs up to each way's first
# `<turn|>`.

# %%
# Cell 5.1: tokens compared with generate, up to the end of the answer
from tokenizers import Tokenizer  # the Rust tokenizer, for decoding; it does not import PyTorch

reference = results["generate"]["tokens"]
END_OF_TURN = 106
for way in WAYS:
    r = results[way]
    if "tokens" not in r:
        continue
    t = r["tokens"]
    answer = t.index(END_OF_TURN) if END_OF_TURN in t else len(t)
    same = next((i for i, (a, b) in enumerate(zip(t[:answer], reference)) if a != b), answer)
    print(f"{LABELS[way]:30s} {same} of {answer} answer tokens identical to generate")

print("\n" + Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json")).decode(reference))

# %% [markdown]
# **What you should see.** Every way identical through the end of a 56-token, two-sentence answer about
# Rayleigh scattering, which every way wrote on a Colab v5e-1 on 2026-10-23, JAX included. The
# 4-bit table changes nothing in the answer because it holds the same values as the bf16 table: Google's
# QAT checkpoint already stores that table on a 4-bit grid, and the unpacking rounds to bf16 the same
# way. A way that diverged early would point at a numerical difference and is worth a look before
# trusting its speed.

# %% [markdown]
# ## 6. Memory on the chip
#
# **What we are testing.** What each way holds on the chip once it has finished, and, for the compiled
# ways, the most it held at any moment, as TorchTPU reports them.

# %%
# Cell 6.1: memory
print(f"{'way':30s} {'after load GB':>13s} {'at end GB':>10s} {'peak GB':>8s}")
for way in WAYS:
    r = results[way]
    if "error" in r:
        continue
    load_gb = r.get("hbm_ple4_gb", r.get("hbm_load_gb"))
    peak = f"{r['hbm_peak_gb']:8.2f}" if r.get("hbm_peak_gb") else f"{'':8s}"
    load_text = f"{load_gb:13.2f}" if load_gb else f"{'':13s}"
    print(f"{LABELS[way]:30s} {load_text} {r['hbm_end_gb']:10.2f} {peak}")

# %% [markdown]
# **What you should see.** About 9.8 GB for the bf16 PyTorch ways, and 3.4 GB less, about 6.4 GB, once
# the per-layer table is 4-bit: that table is 262,144 rows of 8,960 values, 4.7 GB in bf16 and 1.3 GB
# packed with its scales. On 2026-10-23 the compiled ways peaked at 16.60 to 16.71 GB while compiling,
# against 10 GB for the uncompiled ones, close to the chip's 16 GiB. The 4-bit table did not lower that
# peak. A larger model, or a longer cache, would meet that limit at compile time first.

# %% [markdown]
# ## 7. Scorecard
#
# **What we are testing.** Each claim next to what you measured.

# %%
# Cell 7.1: the scorecard
ok = [w for w in WAYS if "error" not in results[w]]
step, ple4 = results["step"], results["step_ple4"]
rows = [
    (
        "4-bit load unpacks to bf16 at the first forward",
        "yes",
        "yes" if h["packed_before"] and not h["packed_after"] else "no",
    ),
    (
        "first graph break is in compressed_tensors/offload",
        "yes",
        "yes" if "offload" in " ".join(d.get("frames", [])) else "no",
    ),
    (
        "compiled step graphs, wrappers on / off",
        "24 / 1",
        f"{results['step_wrapped'].get('graphs')} / {step.get('graphs')}",
    ),
    ("compiled step over generate", "6x to 7x", f"{rate(step) / base:.1f}x" if "error" not in step else "failed"),
    (
        "4-bit table over compiled step",
        "1.5x to 1.6x",
        f"{rate(ple4) / rate(step):.2f}x" if "error" not in ple4 else "failed",
    ),
    (
        "JAX bf16 over the fastest PyTorch way",
        "about 2.6x",
        f"{rate(results['jax_bf16']) / max(rate(results[w]) for w in ok if w != 'jax_bf16'):.1f}x"
        if "jax_bf16" in ok
        else "failed",
    ),
    ("ways that failed", "none", ", ".join(w for w in WAYS if w not in ok) or "none"),
]
print(f"{'claim':52s} {'expected':>12s} {'today':>12s}")
for claim, expected, today in rows:
    print(f"{claim:52s} {expected:>12s} {today:>12s}")

# %% [markdown]
# **So, which one?** For a process that decodes more than a handful of tokens, the compiled step: a
# static cache, the sliding-window fix, the compressed-tensors wrappers removed, and one
# `torch.compile` call. It pays under 40 s once and then runs 6 to 7 times faster than `generate` on
# a 64-token reply, and 10 times on a 128-token one, writing the same tokens. Keep the per-layer table
# 4-bit as well: it frees 3.4 GB and is faster whenever tokens are collected in a batch rather than
# streamed one at a time, 9.4 to 10.1 times `generate` in all. Use `generate` for a single short
# answer, where a few seconds of strict eager beat half a minute of compiling.
#
# **Scope.** One v5e chip on Colab, one checkpoint, one prompt, greedy decoding, a single sequence at a
# time and a 256-token cache, which stays inside the model's 512-token sliding window, the condition the
# sliding-window fix depends on. Batching, longer contexts and a serving endpoint were not run here.

# %% [markdown]
# ## 8. Your notes
#
# * **Versions.** Which TorchTPU, PyTorch and Transformers versions (Cell 1.1)?
# * **Memory.** Did the first forward pass unpack the weights (Cell 2.1)?
# * **Graph break.** Where was the first one (Cell 3.1)?
# * **Speed.** How fast was the compiled step, and the 4-bit table (Cell 4.2)?
# * **Agreement.** Did every way write the same answer (Cell 5.1)?

# %%
# Cell 8.1: your notes (edit this cell)
notes = """
Versions:
Memory:
Graph break:
Speed:
Agreement:
Other findings:
"""
print(notes)
