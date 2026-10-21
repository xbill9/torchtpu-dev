# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B on one v5e chip: JAX against TorchTPU
#
# **Author:** William McLean, Google Developer Expert
#
# **Based on:** notebook 02 of this series, the pure-JAX Gemma 4 engine
# [`xbill9/tpu-jax`](https://github.com/xbill9/tpu-jax) used in
# [colab-jax](https://github.com/xbill9/colab-jax), and the repack
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text).
#
# ### What this notebook is for
#
# A head-to-head on one chip: the same checkpoint, the same prompt and the same greedy decode, run by
# a JAX engine and by PyTorch on TorchTPU. Published comparisons between the two stacks use large
# models on newer chips and come from the vendor. This one uses a small Gemma 4 model on the TPU
# Colab gives you, and measures everything in front of you.
#
# ### The two engines
#
# * **JAX:** `tpu-jax`, a Gemma 4 E2B decoder written directly in JAX. Its decode loop is one
#   compiled program with a KV cache that persists across steps. It runs in two configurations here:
#   the 4-bit weights kept packed in memory and unpacked inside each step (its default), and the
#   weights unpacked to bf16 once at load (`dequant_at_load=True`).
# * **PyTorch on TorchTPU:** Hugging Face Transformers' own Gemma 4 model, loaded with
#   `device_map="tpu"` and the weights unpacked to bf16 at load. It runs the three ways notebook 02
#   found to work best: `generate` in strict eager, `generate` in defer-and-fuse, and the compiled
#   fixed buffer from this repo's server.
#
# The JAX bf16 configuration and the PyTorch ways hold the same bf16 weights on the chip, so they are
# the like-for-like pair. The JAX 4-bit configuration shows what a packed format buys when the engine
# is written for it.
#
# ### What each section shows
#
# * **Section 2, same tokens.** Each engine continues the prompt for 64 greedy tokens.
# * **Section 3, same predictions.** Each engine scores one fixed 83-token sequence, and the two
#   stacks' full next-token distributions are compared position by position.
# * **Section 4, speed and memory.** Load time, first-call time, decode speed measured with one
#   stopwatch for every engine, and memory on the chip.
#
# ### The three rules used throughout
#
# 1. **Every result is measured here.** Earlier numbers appear beside yours as context, never in
#    place of them.
# 2. **One stopwatch.** Every engine is timed the same way, from outside, by this notebook. Each
#    engine's own internal timer appears only as context.
# 3. **Every engine starts cold.** Each one runs in a fresh Python process, because only one process
#    can hold the chip.
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
# * **No token.** The repack and the JAX engine are public. Nothing asks for a Hugging Face token.
# * **Disk and time.** About 7 GB of download. The whole notebook takes about 30 minutes, most of it
#   the defer-and-fuse warm-up.
# * **Safety.** Nothing is uploaded and nothing outside the Colab VM changes.

# %% [markdown]
# ## 1. Setup
#
# **What we are testing.** That both stacks can share one Colab runtime. TorchTPU needs `libtpu`
# 0.0.50 or newer, and JAX ships its own `libtpu`. If installing TorchTPU replaced the one JAX
# depends on, the JAX half of this notebook would fail, so Cell 1.1 runs a JAX matmul before and
# after the install.

# %%
# Cell 1.1: versions, the shared libtpu, and the install
import importlib.metadata as md
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

JAX_CHECK = (
    "import jax, jax.numpy as jnp; d = jax.devices()[0]; "
    "print(d.device_kind, float((jnp.ones((256, 256)) @ jnp.ones((256, 256))).sum()))"
)


def jax_check():
    return subprocess.run([sys.executable, "-c", JAX_CHECK], capture_output=True, text=True).stdout.strip()


print("Python", sys.version.split()[0])
print("before:", {p: md.version(p) for p in ("jax", "libtpu", "torch", "transformers")})
print("JAX matmul before:", jax_check())

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
    print(install.stderr[-4000:])
    raise RuntimeError("pip install failed")
print("after: ", {p: md.version(p) for p in ("jax", "libtpu", "torch", "torch_tpu", "compressed-tensors")})
print("JAX matmul after: ", jax_check())

# %% [markdown]
# **What you should see.** The same `libtpu` version before and after, and the JAX matmul printing
# 16777216.0 both times. On 2026-10-21 Colab's image shipped JAX 0.11.2 with `libtpu` 0.0.50, the
# version TorchTPU 0.2.0 asks for, so the install left it alone.

# %%
# Cell 1.2: the checkpoint, the JAX engine, and the prompt
from huggingface_hub import snapshot_download

MODEL = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
MODEL_DIR = pathlib.Path(snapshot_download(MODEL))
size = sum(f.stat().st_size for f in MODEL_DIR.rglob("*.safetensors"))
print(f"{MODEL}: {size / 1e9:.2f} GB of weights")

# The JAX engine at a fixed commit, so the code you run is the code this notebook was written against.
TPU_JAX_COMMIT = "4b9f8e94619f1995206617e07e3c4ac62685d47b"
ENGINE_DIR = pathlib.Path("/content/tpu-jax" if os.path.isdir("/content") else os.path.expanduser("~/tpu-jax-src"))
if not (ENGINE_DIR / ".git").exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/xbill9/tpu-jax", str(ENGINE_DIR)], check=True)
subprocess.run(["git", "-C", str(ENGINE_DIR), "checkout", "-q", TPU_JAX_COMMIT], check=True)
print("tpu-jax at", TPU_JAX_COMMIT[:7])

# One prompt, tokenized once by the model's own chat template, handed to both stacks as the same
# token ids. It runs in a subprocess because importing Transformers imports PyTorch, which would load
# TorchTPU into this notebook's own process.
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

from tokenizers import Tokenizer  # the Rust tokenizer, for decoding; it does not import PyTorch

tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
print(f"prompt: {len(PROMPT_IDS)} tokens")
NEW_TOKENS = 64

# %% [markdown]
# Cell 1.3 writes one script with a task for each engine. Every task loads its engine in a fresh
# process, times a 1-token and a 64-token greedy continuation twice each, and prints one line of JSON.
# The second timing of each is the warm one. Decode speed is then the same formula for everyone:
# 63 tokens divided by the warm 64-token time minus the warm 1-token time, which removes the prompt.

# %%
# Cell 1.3: the task script
TASKS = pathlib.Path("engine_tasks.py")
TASKS.write_text(r'''
import json
import sys
import time

import numpy as np

TASK, MODEL, MODEL_DIR, ENGINE_DIR, NEW_TOKENS = sys.argv[1:6]
NEW_TOKENS = int(NEW_TOKENS)
PROMPT_IDS = json.load(open("prompt_ids.json"))


def stopwatch(run):
    """Time run(1) and run(NEW_TOKENS) twice each; return first and warm timings and the tokens."""
    t = {}
    for label in ("first", "warm"):
        for n in (1, NEW_TOKENS):
            start = time.perf_counter()
            tokens = run(n)
            t[f"{label}_{n}"] = time.perf_counter() - start
    return t, tokens


def jax_task(dequant):
    sys.path.insert(0, ENGINE_DIR)
    import jax
    import jax.numpy as jnp
    from jax_engine import JaxGemmaEngine
    from ports.gemma4.jax_e_model import make_prefill_causal_mask

    engine = JaxGemmaEngine(MODEL, quant_mode="w4a16", dequant_at_load=dequant)
    start = time.perf_counter()
    engine.load(local_dir=MODEL_DIR)
    load_s = time.perf_counter() - start
    stats = []

    def run(n):
        tokens, s = engine.generate(PROMPT_IDS, max_new_tokens=n)
        stats.append(s)
        return tokens

    timings, tokens = stopwatch(run)
    result = {
        "load_s": load_s,
        **timings,
        "tokens": tokens,
        "weight_gb": engine.weight_bytes / 1e9,
        "hbm_gb": jax.devices()[0].memory_stats()["bytes_in_use"] / 1e9,
        "engine_decode_tok_s": stats[-1].decode_tok_per_s,
    }

    # Log-probabilities over the shared sequence, for Section 3. The first engine to run writes it:
    # the prompt plus its own continuation.
    try:
        seq = json.load(open("sequence.json"))
    except FileNotFoundError:
        seq = PROMPT_IDS + tokens
        json.dump(seq, open("sequence.json", "w"))
    if seq is not None:
        n = len(seq)
        valid = jnp.ones((1, n), dtype=bool)
        full = make_prefill_causal_mask(valid)
        sliding = make_prefill_causal_mask(valid, window=engine.config.sliding_window)
        pos = jnp.arange(n, dtype=jnp.int32)[None, :]
        logits = engine.model(
            jnp.asarray([seq], dtype=jnp.int32),
            engine.params,
            pos,
            attention_mask=full,
            sliding_attention_mask=sliding,
            quant_mode=engine.quant_mode,
        )
        lp = jax.nn.log_softmax(logits[0].astype(jnp.float32), axis=-1)
        np.save(f"logprobs_{TASK}.npy", np.asarray(lp))
    return result


def torch_task(way):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    def sync():
        torch.accelerator.synchronize()

    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="tpu",
        key_mapping={r"^model\.language_model\.": "model."},
        quantization_config=transformers.CompressedTensorsConfig(run_compressed=False),
    ).eval()
    sync()
    load_s = time.perf_counter() - start
    prompt = torch.tensor([PROMPT_IDS], dtype=torch.long).to("tpu")

    if way == "buffer":
        seq_len, n_prompt = 256, len(PROMPT_IDS)

        def step(tokens, mask, last):
            return model(input_ids=tokens, attention_mask=mask, use_cache=False).logits.index_select(1, last)

        compiled = torch.compile(step, backend="tpu", dynamic=False)
        one = torch.ones((1, 1), dtype=torch.long).to("tpu")

        def run(n):
            tokens = torch.zeros((1, seq_len), dtype=torch.long)
            tokens[0, :n_prompt] = torch.tensor(PROMPT_IDS)
            mask = torch.zeros((1, seq_len), dtype=torch.long)
            mask[0, :n_prompt] = 1
            tokens, mask, pos = tokens.to("tpu"), mask.to("tpu"), torch.tensor([n_prompt]).to("tpu")
            out = []
            with torch.no_grad():
                for _ in range(n):
                    nxt = compiled(tokens, mask, pos - 1).argmax(-1)[:, 0:1]
                    out.append(int(nxt.cpu().item()))
                    tokens = tokens.index_copy(1, pos, nxt.view(1, 1))
                    mask = mask.index_copy(1, pos, one)
                    pos = pos + 1
            return out
    else:

        def run(n):
            with torch.no_grad():
                out = model.generate(prompt, max_new_tokens=n, min_new_tokens=n, do_sample=False)
            sync()
            return out[0, len(PROMPT_IDS) :].tolist()

    timings, tokens = stopwatch(run)
    result = {"load_s": load_s, **timings, "tokens": tokens, "hbm_gb": torch.accelerator.memory_allocated() / 1e9}

    if way == "eager":
        seq = json.load(open("sequence.json"))
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([seq], dtype=torch.long).to("tpu")).logits
            lp = torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()
        np.save(f"logprobs_{TASK}.npy", lp)
    return result


if __name__ == "__main__":
    tasks = {
        "jax_w4a16": lambda: jax_task(dequant=False),
        "jax_bf16": lambda: jax_task(dequant=True),
        "torch_eager": lambda: torch_task("eager"),
        "torch_fuse": lambda: torch_task("fuse"),
        "torch_buffer": lambda: torch_task("buffer"),
    }
    try:
        result = tasks[TASK]()
    except Exception as exc:  # report a failure as a result instead of crashing the cell
        result = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    print("RESULT " + json.dumps(result))
''')

ENVIRONMENTS = {"torch_fuse": {"TORCH_TPU_DEFER_AND_FUSE": "1"}}
results = {}


def run(task, timeout=1700):
    env = {**os.environ, **ENVIRONMENTS.get(task, {})}
    args = [sys.executable, str(TASKS), task, MODEL, str(MODEL_DIR), str(ENGINE_DIR), str(NEW_TOKENS)]
    start = time.perf_counter()
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:])
        raise RuntimeError(f"{task} exited with code {proc.returncode} and no result")
    results[task] = json.loads(lines[-1][len("RESULT ") :])
    print(f"{task:13s} finished in {time.perf_counter() - start:5.0f} s: {results[task].get('error', 'ok')}")


print("task script written:", TASKS.resolve())

# %% [markdown]
# ## 2. Same tokens
#
# **What the source says.** In the probes behind notebooks 02 and 03 (Colab v5e-1, 2026-10-21), the JAX
# engine and PyTorch on TorchTPU wrote the same 32 greedy tokens from this checkpoint and prompt.
#
# **What we are testing.** 64 greedy tokens from each engine. The JAX 4-bit engine runs first; its
# continuation, appended to the prompt, becomes the fixed sequence every engine scores in Section 3.

# %%
# Cell 2.1: the two JAX configurations
pathlib.Path("sequence.json").unlink(missing_ok=True)
run("jax_w4a16")  # writes sequence.json: the prompt plus its own 64 tokens
run("jax_bf16")

# %%
# Cell 2.2: PyTorch on TorchTPU, strict eager and the compiled buffer
run("torch_eager")
run("torch_buffer")

# %%
# Cell 2.3: PyTorch on TorchTPU, defer-and-fuse (its first call compiles for several minutes)
run("torch_fuse")

# %%
# Cell 2.4: tokens compared with the JAX 4-bit engine
ENGINES = ("jax_w4a16", "jax_bf16", "torch_eager", "torch_fuse", "torch_buffer")
reference = results["jax_w4a16"]["tokens"]
for name in ENGINES:
    r = results[name]
    if "tokens" not in r:
        print(f"{name:13s} failed: {r.get('error', '')[:100]}")
        continue
    same = next((i for i, (a, b) in enumerate(zip(r["tokens"], reference)) if a != b), len(reference))
    print(f"{name:13s} first {same} of {len(reference)} tokens identical to jax_w4a16")
print("\n" + tokenizer.decode(reference))

# %% [markdown]
# **What you should see.** A fluent answer, and long runs of identical tokens across both stacks. In two
# runs on a Colab v5e-1 on 2026-10-21, both JAX configurations and PyTorch's compiled buffer wrote the same 64
# tokens. PyTorch's `generate`, in strict eager and in defer-and-fuse, matched for 56 and then took a
# different token at position 57. Two independently written implementations of Gemma 4, in two
# frameworks, agreeing for 56 to 64 greedy tokens is strong evidence that both are correct.
#
# The split follows how each engine carries the sequence forward. The compiled buffer recomputes the
# whole sequence every step; `generate` reuses a KV cache, and adds each new token's keys and values to
# it. Small rounding differences between those paths can tip a near-tie between two candidate tokens,
# and once one token differs, everything after it does too. Section 3 checks the distributions
# themselves.

# %% [markdown]
# ## 3. Same predictions
#
# **What the source says.** colab-jax notebook 02 compared checkpoints by the KL divergence of their
# next-token distributions: 0.00021 between this repack and Google's QAT model, against 0.07141 for
# Google's own W4A16 export, both in the JAX engine.
#
# **What we are testing.** The same measure between frameworks: the JAX engine's distributions
# against PyTorch's, over every position of the shared 83-token sequence. Same weights, so any
# difference is the arithmetic of the two implementations: operation order, accumulation precision,
# attention kernels.


# %%
# Cell 3.1: KL divergence and top-token agreement between the stacks
def compare(ref_name, other_name):
    p = np.load(f"logprobs_{ref_name}.npy")
    q = np.load(f"logprobs_{other_name}.npy")
    kl = (np.exp(p) * (p - q)).sum(axis=-1)
    same_top = (p.argmax(-1) == q.argmax(-1)).mean()
    return kl, same_top, np.abs(p - q).max()


pairs = [("jax_bf16", "torch_eager"), ("jax_w4a16", "torch_eager"), ("jax_w4a16", "jax_bf16")]
print(
    f"{'reference':10s} {'compared':12s} {'mean KL':>10s} {'max KL':>10s} {'same top token':>15s} {'max |logprob diff|':>19s}"
)
for ref_name, other_name in pairs:
    kl, same_top, max_diff = compare(ref_name, other_name)
    print(f"{ref_name:10s} {other_name:12s} {kl.mean():10.6f} {kl.max():10.6f} {same_top:15.2%} {max_diff:19.4f}")
print(f"\npositions compared: {len(np.load('logprobs_jax_bf16.npy'))}")

# %% [markdown]
# **What you should see.** The same top token at every position, and small but nonzero divergences.
# In both runs on a Colab v5e-1 on 2026-10-21, JAX bf16 against PyTorch eager gave a mean KL of 0.001073 and a
# maximum of 0.008882 over 83 positions, with the top token identical at all 83. That is about five
# times the 0.00021 that separated two different builds of the weights in colab-jax 02: the two
# frameworks' arithmetic differs by more than those two weight files did, and still never changes
# which token comes first. The largest single log-probability difference, 0.876, sits on a token the
# model considers unlikely.
#
# The last row compares the JAX engine with itself, 4-bit against bf16, and comes out exactly zero:
# unpacking at every step and unpacking once at load produce the same bf16 weights and the same
# arithmetic.

# %% [markdown]
# ## 4. Speed and memory
#
# **What the source says.** Notebook 02 measured PyTorch on TorchTPU at single-digit tokens per second
# for this model, counting the prompt. The probes measured the JAX engine at about 103 tokens per
# second of decode by its own timer. The `tpu-jax` source notes that unpacking at load "is 1.60x
# faster at B=1" on a v6e-1.
#
# **What we are testing.** Every engine on the same stopwatch, as described above Cell 1.3. Memory
# is what each framework reports holding on the chip, after the run.

# %%
# Cell 4.1: the table
print(
    f"{'engine':13s} {'load s':>7s} {'first 64 s':>11s} {'warm 1 s':>9s} {'warm 64 s':>10s} "
    f"{'decode tok/s':>13s} {'memory GB':>10s}"
)
decode = {}
for name in ENGINES:
    r = results[name]
    if "error" in r:
        print(f"{name:13s} failed")
        continue
    decode[name] = (NEW_TOKENS - 1) / (r[f"warm_{NEW_TOKENS}"] - r["warm_1"])
    print(
        f"{name:13s} {r['load_s']:7.1f} {r[f'first_{NEW_TOKENS}']:11.2f} {r['warm_1']:9.3f} "
        f"{r[f'warm_{NEW_TOKENS}']:10.3f} {decode[name]:13.1f} {r['hbm_gb']:10.2f}"
    )
for name in ("jax_w4a16", "jax_bf16"):
    if "engine_decode_tok_s" in results[name]:
        print(f"{name}: the engine's own timer says {results[name]['engine_decode_tok_s']:.1f} tok/s")

best_torch = max((n for n in decode if n.startswith("torch")), key=decode.get)
for name in ("jax_w4a16", "jax_bf16"):
    print(f"{name} / {best_torch}: {decode[name] / decode[best_torch]:.1f}x")

# %% [markdown]
# **What you should see.** The JAX engine well ahead on decode speed. In two runs on a Colab v5e-1 on
# 2026-10-21: 130.8 to 133.9 tokens per second for JAX bf16 and 103.1 to 103.3 for JAX 4-bit, against
# 8.8 to 9.7 for PyTorch in defer-and-fuse, 6.3 to 6.4 for the compiled buffer and 4.7 to 4.9 for
# strict eager. That is 10.6x to 11.7x between JAX 4-bit and the best PyTorch way, and 13.8x to 14.8x
# for JAX bf16. The engines' own timers agreed with the shared stopwatch (104.6 to 104.8 and 132.6 to
# 135.7). Unpacking at load made the JAX engine 1.27x to 1.30x faster on this v5e-1, where the source
# measured 1.60x on a v6e-1. JAX also loaded fastest, 7.3 to 7.4 s for the 4-bit configuration against
# 20 to 43 s for PyTorch, and defer-and-fuse paid 650 to 659 s for its first 64-token call.
#
# The gap comes from how each engine is built: the JAX engine keeps a KV cache and runs each decode step as one
# compiled program, while the PyTorch ways run Transformers' general-purpose Gemma 4 code, either
# operation by operation or as a step split by graph breaks (notebook 02, Section 5). Memory follows
# the weight format: 6.62 GB for JAX holding packed 4-bit weights, 9.30 GB for JAX bf16, and 9.83 to
# 11.91 GB for PyTorch's bf16 ways.

# %% [markdown]
# ## 5. Scorecard
#
# **What we are testing.** Every claim next to what you measured.

# %%
# Cell 5.1: the scorecard
ok_tokens = [n for n in ENGINES if "tokens" in results[n]]
kl_like, top_like, _ = compare("jax_bf16", "torch_eager")
rows = [
    ("libtpu shared by both stacks", "yes", "yes"),
    (
        "engines writing 64 of 64 tokens identical to JAX",
        "most",
        f"{sum(results[n]['tokens'] == reference for n in ok_tokens)} of {len(ENGINES)}",
    ),
    ("mean KL, JAX bf16 vs PyTorch eager", "small", f"{kl_like.mean():.6f}"),
    ("  for scale: KL between two weight builds (colab-jax 02)", "0.00021", ""),
    ("same top token, JAX bf16 vs PyTorch eager", "100%", f"{top_like:.2%}"),
    ("fastest decode", "JAX", max(decode, key=decode.get)),
    ("JAX 4-bit / best PyTorch decode", "", f"{decode['jax_w4a16'] / decode[best_torch]:.1f}x"),
]
print(f"{'claim':50s} {'expected':>12s} {'today':>12s}")
for claim, expected, today in rows:
    print(f"{claim:50s} {expected:>12s} {today:>12s}")

# %% [markdown]
# **What you should see.** Every row in line with the sections above.
#
# **So, which one?** For serving this model on this chip today, the JAX engine: it picks the same top
# token as PyTorch at every position and runs more than ten times faster, because it was written for TPU decode from the start. PyTorch on
# TorchTPU earns its place elsewhere: it runs Hugging Face's own model code with no port at all, gives
# the same answers, and is the stack to reach for when the model or the team is PyTorch first. The
# gap measured here is the gap between a general-purpose model implementation and a decode engine;
# closing it on the PyTorch side means a cached decode step compiled as one graph, the job of a serving
# engine such as vLLM-TorchTPU, which keeps vLLM's own KV cache manager.

# %% [markdown]
# ## 6. Your notes
#
# * **Versions.** JAX, `libtpu`, PyTorch and TorchTPU (Cell 1.1)?
# * **Tokens.** Did every engine write the same 64 tokens (Cell 2.4)?
# * **Predictions.** What KL separated the two stacks (Cell 3.1)?
# * **Speed.** Which engine decoded fastest, and by how much (Cell 4.1)?

# %%
# Cell 6.1: your notes (edit this cell)
notes = """
Versions:
Tokens:
Predictions:
Speed:
Other findings:
"""
print(notes)
