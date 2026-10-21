# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B on TorchTPU: loading a QAT repack, and the ways to run it
#
# **Author:** William McLean, Google Developer Expert
#
# **Based on:** the repack
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text),
# notebook 01 of this series, and the serving code in
# [`xbill9/torchtpu-dev`](https://github.com/xbill9/torchtpu-dev).
#
# ### What this notebook is for
#
# Running a real model, Gemma 4 E2B, on a TPU through TorchTPU and Hugging Face Transformers, and
# choosing how to run it. Notebook 01 measured the execution modes on synthetic tensors. This one
# loads a 4-bit checkpoint, checks that it actually reached the chip, and times each way of
# generating text from it.
#
# ### The checkpoint
#
# Google trained Gemma 4 E2B with quantization-aware training (QAT), holding every weight on a 4-bit
# grid. The repack used here stores those trained 4-bit values in the compressed-tensors W4A16 format
# (4-bit weights, 16-bit activations), text-only, at 4.6 billion stored parameters. In an earlier JAX
# study on this chip, its multimodal twin `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct` matched Google's
# QAT model on 128 of 128 greedy tokens.
#
# ### What each section shows
#
# * **Section 2, loading.** Two problems stand between `from_pretrained` and the chip. The repack's
#   tensor names need one remapping before Transformers will match them, and the familiar
#   `model.to("tpu")` leaves this quantized model on the CPU without an error. The section counts
#   where every tensor lands under each loader.
# * **Section 3, five ways to generate.** Hugging Face `generate` in strict eager and in
#   defer-and-fuse; the weights kept 4-bit in memory; a compiled fixed-size buffer, which is how this
#   repo's server runs; and Hugging Face's static KV cache under `torch.compile`. Each way reports its
#   first call, its warm speed, its memory, and the tokens it wrote.
# * **Section 4, agreement.** Every way should write the same greedy tokens, because they run the same
#   weights through the same arithmetic.
# * **Section 5, inside the compiled step.** How many separate graphs `torch.compile` produced, which
#   decides how much of the step runs compiled.
#
# ### The three rules used throughout
#
# 1. **Every result is measured here.** Numbers from the earlier work appear beside yours as context,
#    never in place of them.
# 2. **One checkpoint, one prompt.** Every way loads the same files and continues the same chat prompt
#    for the same number of tokens, greedily.
# 3. **Every way starts cold.** Each one runs in a fresh Python process, because only one process can
#    hold the chip and no way should inherit another's compiled programs.
#
# ### How each section is organized
#
# * **What the source says:** the earlier result or documentation, and where it came from.
# * **What we are testing:** the claim, and the evidence that checks it.
# * A code cell to run.
# * **What you should see:** the expected result, and what a different one would mean.
#
# ### Before you start
#
# * **Runtime.** Runtime, then Change runtime type, then TPU. The target is a single v5e chip
#   (16 GB of HBM). The bf16 weights take about 10 GB of it.
# * **No token.** The repack is public and ungated. Nothing asks for a Hugging Face token.
# * **Disk and time.** About 7 GB of download. The whole notebook takes 20 to 30 minutes, most of it
#   loading the model once per way and compiling.
# * **Safety.** Nothing is uploaded and nothing outside the Colab VM changes.

# %% [markdown]
# ## 1. Setup
#
# **What we are testing.** Nothing yet. This section installs TorchTPU and the compressed-tensors
# reader, downloads the repack, and records the versions every number below depends on.

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
for package in ("torch_tpu", "torch", "libtpu", "transformers", "compressed-tensors"):
    print(f"{package:20s} {md.version(package)}")

from huggingface_hub import snapshot_download

MODEL = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
start = time.perf_counter()
MODEL_DIR = pathlib.Path(snapshot_download(MODEL))
size = sum(f.stat().st_size for f in MODEL_DIR.rglob("*.safetensors"))
print(f"\n{MODEL}: {size / 1e9:.2f} GB of weights, downloaded in {time.perf_counter() - start:.0f} s")

# %% [markdown]
# Cell 1.2 writes the script every later section runs. Each task in it loads the model in a fresh
# process, does one thing, and prints its result as one line of JSON. The notebook's own Python
# session never imports PyTorch, so the chip stays free for those processes.

# %%
# Cell 1.2: the task script
TASKS = pathlib.Path("gemma_tasks.py")
TASKS.write_text(r'''
import collections
import json
import re
import sys
import time

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
TASK = sys.argv[2]
NEW_TOKENS = 32
PROMPT = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]

# The repack's tensors are named for the multimodal model (model.language_model.layers...), while
# its config asks for the text-only class, whose tensors are model.layers... This maps one to the
# other at load time.
KEY_MAPPING = {r"^model\.language_model\.": "model."}


def sync():
    torch.accelerator.synchronize()


def prompt_ids():
    ids = AutoTokenizer.from_pretrained(MODEL).apply_chat_template(PROMPT, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def load(loader="device_map", compressed=False):
    kw = {"dtype": torch.bfloat16, "key_mapping": KEY_MAPPING}
    if not compressed:
        # Unpack the 4-bit weights to bf16 once, at load.
        kw["quantization_config"] = transformers.CompressedTensorsConfig(run_compressed=False)
    start = time.perf_counter()
    if loader == "device_map":
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="tpu", **kw).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval().to("tpu")
    sync()
    return model, time.perf_counter() - start


def placement(model):
    count, gb = collections.Counter(), collections.Counter()
    for _, t in list(model.named_parameters()) + list(model.named_buffers()):
        count[t.device.type] += 1
        gb[t.device.type] += t.numel() * t.element_size() / 1e9
    return {"tensors": dict(count), "gb": dict(gb), "hbm_gb": torch.accelerator.memory_allocated() / 1e9}


def task_names():
    """Checkpoint tensor names against the names the model expects, before and after the mapping."""
    from huggingface_hub import hf_hub_download

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL))
    wanted = set(model.state_dict())
    index = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json")))["weight_map"]
    stored = {re.sub(r"\.weight_(packed|scale|shape)$", ".weight", k) for k in index}
    mapped = {re.sub(r"^model\.language_model\.", "model.", k) for k in stored}
    return {
        "model_tensors": len(wanted),
        "matched_raw": len(stored & wanted),
        "matched_mapped": len(mapped & wanted),
        "unmatched_model_tensors": sorted(wanted - mapped),
        "tied": sorted(getattr(model, "_tied_weights_keys", {}) or {}),
    }


def task_placement(loader):
    model, load_s = load(loader)
    return {"loader": loader, "load_s": load_s, **placement(model)}


def run_generate(model, ids, **kw):
    x = torch.tensor([ids], dtype=torch.long).to("tpu")
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False, **kw)
    sync()
    return out[0, len(ids):].tolist()


def task_generate(compressed=False, static=False):
    model, load_s = load(compressed=compressed)
    ids = prompt_ids()
    kw = {}
    if static:
        model.forward = torch.compile(model.forward, backend="tpu", dynamic=False)
        kw["cache_implementation"] = "static"
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        tokens = run_generate(model, ids, **kw)
        timings.append(time.perf_counter() - start)
    text = AutoTokenizer.from_pretrained(MODEL).decode(tokens, skip_special_tokens=True)
    return {
        "load_s": load_s,
        "first_s": timings[0],
        "warm_s": timings[1],
        "tokens": tokens,
        "text": text,
        **placement(model),
    }


def task_buffer():
    """The serving design in tpu_openai_server.py: one fixed [1, SEQ] buffer, one compiled step."""
    from torch._dynamo.utils import counters

    model, load_s = load()
    ids = prompt_ids()
    n, seq = len(ids), 256

    def step(tokens, mask, last):
        return model(input_ids=tokens, attention_mask=mask, use_cache=False).logits.index_select(1, last)

    compiled = torch.compile(step, backend="tpu", dynamic=False)
    one = torch.ones((1, 1), dtype=torch.long).to("tpu")

    def generate():
        tokens = torch.zeros((1, seq), dtype=torch.long)
        tokens[0, :n] = torch.tensor(ids)
        mask = torch.zeros((1, seq), dtype=torch.long)
        mask[0, :n] = 1
        tokens, mask, pos = tokens.to("tpu"), mask.to("tpu"), torch.tensor([n]).to("tpu")
        out = []
        with torch.no_grad():
            for _ in range(NEW_TOKENS):
                nxt = compiled(tokens, mask, pos - 1).argmax(-1)[:, 0:1]
                out.append(int(nxt.cpu().item()))
                tokens = tokens.index_copy(1, pos, nxt.view(1, 1))
                mask = mask.index_copy(1, pos, one)
                pos = pos + 1
        return out

    counters.clear()
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        tokens = generate()
        timings.append(time.perf_counter() - start)
    return {
        "load_s": load_s,
        "first_s": timings[0],
        "warm_s": timings[1],
        "tokens": tokens,
        "graphs": int(counters["stats"]["unique_graphs"]),
        "graph_breaks": int(sum(counters["graph_break"].values())),
        "break_reasons": sorted(counters["graph_break"])[:5],
        **placement(model),
    }


if __name__ == "__main__":
    task = {
        "names": task_names,
        "place_to": lambda: task_placement("to"),
        "place_device_map": lambda: task_placement("device_map"),
        "eager": task_generate,
        "fuse": task_generate,
        "compressed": lambda: task_generate(compressed=True),
        "buffer": task_buffer,
        "static_cache": lambda: task_generate(static=True),
    }[TASK]
    try:
        result = task()
    except Exception as exc:  # a way that fails is a result too; report it instead of crashing the cell
        result = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    print("RESULT " + json.dumps(result))
''')

ENVIRONMENTS = {"fuse": {"TORCH_TPU_DEFER_AND_FUSE": "1"}}


def run(task, timeout=1800):
    """Run one task in a fresh process and return its result dict."""
    env = {**os.environ, **ENVIRONMENTS.get(task, {})}
    proc = subprocess.run(
        [sys.executable, str(TASKS), MODEL, task], env=env, capture_output=True, text=True, timeout=timeout
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:])
        raise RuntimeError(f"task {task} exited with code {proc.returncode} and no result")
    return json.loads(lines[-1][len("RESULT ") :])


print("task script written:", TASKS.resolve())

# %% [markdown]
# ## 2. Getting the model onto the chip
#
# ### 2.1 Tensor names
#
# **What the source says.** The repack's `config.json` asks for `Gemma4ForCausalLM`, the text-only
# model. Its tensors keep the names of the multimodal model they came from,
# `model.language_model.layers.…`, where the text-only class expects `model.layers.…`. vLLM and the
# JAX engine used in notebook 03 both try several prefixes and load it as it is.
#
# **What we are testing.** How many of the model's tensors find a match in the checkpoint by name,
# before and after the one-line mapping the task script passes as `key_mapping`. This builds the model
# on PyTorch's `meta` device, which allocates nothing, so it takes seconds.

# %%
# Cell 2.1: checkpoint names against model names
names = run("names")
print(f"tensors the model expects:          {names['model_tensors']}")
print(f"matched by name as stored:          {names['matched_raw']}")
print(f"matched after model.language_model. -> model.: {names['matched_mapped']}")
print(f"model tensors with no match:        {names['unmatched_model_tensors']}")
print(f"tied to another tensor:             {names['tied']}")

# %% [markdown]
# **What you should see.** None matched as stored, and every tensor but one after the mapping. The
# one left over is `lm_head.weight`, which Gemma 4 ties to the token embedding, so it is filled from
# `embed_tokens` and nothing is missing. Loaded without the mapping, Transformers would build every
# layer from scratch and the 4-bit unpacking would fail on empty tensors.
#
# ### 2.2 Which loader puts it on the chip
#
# **What the source says.** Google's TorchTPU announcement: with Transformers v5, using a TPU "is as
# simple as passing `device_map="tpu"`". The familiar PyTorch idiom is `model.to(device)` after
# loading.
#
# **What we are testing.** Both loaders, each in its own process, with every parameter and buffer
# counted by the device it ended up on, and the TPU's own count of the memory it holds.

# %%
# Cell 2.2: where every tensor lands
placements = {loader: run(f"place_{loader}") for loader in ("to", "device_map")}
print(
    f"{'loader':12s} {'load s':>7s} {'on tpu':>8s} {'on cpu':>8s} {'GB tpu':>8s} {'GB cpu':>8s} {'TPU memory GB':>14s}"
)
for loader, p in placements.items():
    t, g = p["tensors"], p["gb"]
    print(
        f"{loader:12s} {p['load_s']:7.1f} {t.get('tpu', 0):8d} {t.get('cpu', 0):8d} "
        f"{g.get('tpu', 0):8.2f} {g.get('cpu', 0):8.2f} {p['hbm_gb']:14.2f}"
    )

# %% [markdown]
# **What you should see.** `device_map="tpu"` puts every tensor on the chip, about 9.4 GB once the
# 4-bit weights are unpacked to bf16, and the TPU reports holding that much. `model.to("tpu")` leaves
# them all on the CPU and raises nothing: on a Colab v5e-1 on 2026-10-21 it moved 0 of 1,098 tensors.
# A model left there still runs, on the host CPU, and produces correct text, so the only symptom is
# speed. Every way below loads with `device_map="tpu"`.

# %% [markdown]
# ## 3. Five ways to generate
#
# **What the source says.** Notebook 01: strict eager answers first, defer-and-fuse and
# `torch.compile` are faster once warm, and a compiled function wants fixed shapes. TorchTPU's
# getting-started guide warns that a Hugging Face generate loop under `torch.compile` "will involve
# recompilations at each step unless bounded dynamism is enabled". This repo's own notes from July,
# on a v6e-1 with an earlier TorchTPU build and the bf16 E2B release, recorded three things: the
# compiled buffer at 227.3 tokens per second; defer-and-fuse on `generate` at 0.5 tokens per second
# against 3.4 for strict eager; and that `generate` "cannot be torch.compiled on TPU today — it feeds a
# scalar CPU tensor into the graph".
#
# **What we are testing.** Each way writes 32 tokens twice from the same prompt. The first call pays
# for compiling; the second is the warm speed. The token rate below includes the prompt, because each
# call starts from it.
#
# 1. **Strict eager:** Hugging Face `generate`, unchanged.
# 2. **Defer-and-fuse:** the same, with `TORCH_TPU_DEFER_AND_FUSE=1`.
# 3. **Weights kept 4-bit:** `generate` with the weights left packed in memory and unpacked inside
#    each forward pass.
# 4. **Compiled buffer:** one fixed 256-token buffer and one compiled step that rereads it every
#    token, the design of `tpu_openai_server.py`.
# 5. **Static cache:** `generate` with Hugging Face's fixed-size KV cache and the forward pass under
#    `torch.compile`.

# %%
# Cell 3.1: run every way (each loads the model fresh; this is the slow cell)
WAYS = ("eager", "fuse", "compressed", "buffer", "static_cache")
results = {}
for way in WAYS:
    start = time.perf_counter()
    results[way] = run(way)
    status = results[way].get("error", "ok")
    print(f"{way:13s} finished in {time.perf_counter() - start:5.0f} s: {status}")

# %%
# Cell 3.2: the table
print(f"{'way':13s} {'first call s':>13s} {'warm call s':>12s} {'warm tok/s':>11s} {'TPU memory GB':>14s}")
for way in WAYS:
    r = results[way]
    if "error" in r:
        print(f"{way:13s} failed: {r['error'][:90]}")
        continue
    print(f"{way:13s} {r['first_s']:13.2f} {r['warm_s']:12.2f} {32 / r['warm_s']:11.1f} {r['hbm_gb']:14.2f}")

# %% [markdown]
# **What you should see.** Four working ways, all in single digits of tokens per second. In two runs
# on a Colab v5e-1 on 2026-10-21, warm, that was 4.6 to 5.6 for strict eager, 8.9 to 9.1 for
# defer-and-fuse, 4.7 to 5.2 with the weights kept 4-bit and 6.6 to 6.8 for the compiled buffer. Notebook 03 runs a JAX engine on the same chip
# and checkpoint for comparison.
#
# * **Defer-and-fuse was the fastest warm way, and by far the slowest to start.** Its first call took
#   341 and 359 s in the two runs, against 16 to 28 s for the others: `generate` grows the sequence by one token every step, so
#   every fused group is a new shape to compile. Once compiled, it ran ahead. The July note of 0.5
#   tokens per second did not reproduce on this TorchTPU release.
# * **Keeping the weights 4-bit saved no memory.** The TPU held the same 9.85 GB as with the weights
#   unpacked, so on this path the packed format brings only a smaller download.
# * **The static-cache way failed** with a scalar tensor on the CPU, the same failure the July notes
#   recorded for `generate` under `torch.compile`.
#
# A failed row is a result: it records what this TorchTPU and Transformers pair cannot do yet.

# %% [markdown]
# ## 4. Do they agree?
#
# **What we are testing.** Every way that ran should write the same 32 greedy tokens. They run the same
# weights through the same operations; only how the work is grouped and compiled changes.

# %%
# Cell 4.1: tokens compared with strict eager, and the text itself
reference = results["eager"]["tokens"]
for way in WAYS:
    r = results[way]
    if "tokens" not in r:
        continue
    same = next((i for i, (a, b) in enumerate(zip(r["tokens"], reference)) if a != b), len(reference))
    print(f"{way:13s} first {same} of {len(reference)} tokens identical to strict eager")

print("\n" + results["eager"]["text"])

# %% [markdown]
# **What you should see.** 32 of 32 for every way that ran, and a fluent two-sentence answer. A way
# that diverges early would point at a numerical difference in that path, such as a different
# attention kernel or accumulation order, and is worth a closer look before trusting its speed.

# %% [markdown]
# ## 5. Inside the compiled step
#
# **What the source says.** The execution-model guide: a forced sync "will force a graph break", and
# each break leaves the code around it to run outside the compiled program.
#
# **What we are testing.** The compiled buffer's counts from PyTorch's own compiler statistics: how
# many separate graphs one step became, and how many graph breaks it hit, with the first few reasons.

# %%
# Cell 5.1: graphs and graph breaks in the compiled step
b = results["buffer"]
if "error" in b:
    print("the compiled buffer failed:", b["error"])
else:
    print(f"compiled graphs: {b['graphs']}   graph breaks: {b['graph_breaks']}")
    for reason in b["break_reasons"]:
        print("  break:", reason[:160])

# %% [markdown]
# **What you should see.** One graph and no breaks would mean the whole step is a single compiled
# program. In both runs on a Colab v5e-1 on 2026-10-21 the step became 24 graphs with 60 graph
# breaks, every reason shown being "Data pointer comparison": somewhere in the Gemma 4 forward pass,
# Transformers compares two tensors' memory addresses, which the compiler cannot trace, so it cuts
# the step there and runs the code between the pieces eagerly. That is one reason the compiled
# buffer runs far below the 227.3 tokens per second recorded in July, alongside the different chip
# and software versions. Removing those breaks is the route to a compiled step that runs as one
# program.

# %% [markdown]
# ## 6. Scorecard
#
# **What we are testing.** Each claim next to what you measured.

# %%
# Cell 6.1: the scorecard
ok = [w for w in WAYS if "error" not in results[w]]
fastest = min(ok, key=lambda w: results[w]["warm_s"]) if ok else "none"
agree = all(results[w]["tokens"] == reference for w in ok)
rows = [
    (
        "tensor names matched as stored / mapped",
        "0 / all but tied",
        f"{names['matched_raw']} / {names['matched_mapped']}",
    ),
    ("tensors on TPU after model.to('tpu')", "all", str(placements["to"]["tensors"].get("tpu", 0))),
    ("tensors on TPU with device_map='tpu'", "all", str(placements["device_map"]["tensors"].get("tpu", 0))),
    ("fastest warm way", "compiled (01)", fastest),
    (
        "defer-and-fuse faster than strict eager on generate",
        "no (July)",
        "yes" if results["fuse"].get("warm_s", 1e9) < results["eager"]["warm_s"] else "no",
    ),
    ("all working ways write identical tokens", "yes", "yes" if agree else "no"),
    ("ways that failed", "", ", ".join(w for w in WAYS if w not in ok) or "none"),
]
print(f"{'claim':52s} {'expected':>18s} {'today':>16s}")
for claim, expected, today in rows:
    print(f"{claim:52s} {expected:>18s} {today:>16s}")

# %% [markdown]
# **What you should see.** The two loading rows and the agreement row as expected. The two speed rows
# are where this release differs from the earlier guidance: defer-and-fuse won warm, and the compiled
# buffer came second, held back by its graph breaks.
#
# **So, which one?** Load with `device_map="tpu"`, and check where the tensors went before timing
# anything. For one-off answers, strict eager starts in seconds and writes the same tokens as every
# other way. For a long-running process that can pay a six-minute first call once, defer-and-fuse is
# the fastest way here with no code changes. The compiled buffer is the design with the most headroom,
# once the graph breaks in Section 5 are removed.

# %% [markdown]
# ## 7. Your notes
#
# * **Versions.** Which TorchTPU, PyTorch and Transformers versions (Cell 1.1)?
# * **Loading.** Did `model.to("tpu")` move anything (Cell 2.2)?
# * **Speed.** Which way was fastest warm, and how fast (Cell 3.2)?
# * **Agreement.** Did every way write the same tokens (Cell 4.1)?
# * **Compiled step.** How many graphs and breaks (Cell 5.1)?

# %%
# Cell 7.1: your notes (edit this cell)
notes = """
Versions:
Loading:
Speed:
Agreement:
Compiled step:
Other findings:
"""
print(notes)
