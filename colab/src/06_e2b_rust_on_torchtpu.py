# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B in Rust on TorchTPU
#
# **Author:** William McLean, Google Developer Expert
#
# **Based on:** the repacks
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)
# and its 4-bit-tables twin
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4),
# notebook 05 of this series, and [`tch`](https://github.com/LaurentMazare/tch-rs), the Rust bindings for libtorch.
#
# ### What this notebook is for
#
# Running Gemma 4 E2B on one v5e chip with the model written in Rust and every operation executed by
# TorchTPU. The crate in [`colab/rust`](https://github.com/xbill9/torchtpu-dev/tree/main/colab/rust) reads
# the checkpoint, unpacks the 4-bit weights, runs the 35 layers and the ring cache, and drives the decode
# loop. Python hosts the backend and starts the engine.
#
# ### How Rust reaches TorchTPU
#
# TorchTPU registers its kernels on libtorch's dispatcher when Python imports it. `tch` calls that same
# libtorch, loaded once in the same process, so a Rust call such as `x.matmul(&y)` on a `tpu` tensor lands on
# the TorchTPU kernel. Three facts shape the rest:
#
# 1. **`tch` 0.26 is generated for PyTorch 2.13.0.** Its bindings fail to compile against 2.14.1, the version
#    the earlier notebooks ran. TorchTPU supports 2.11 to 2.15, so the Rust ways run on 2.13.0.
# 2. **`tch::Device` has no `tpu`.** Rust moves tensors with `type_as`, which follows a tensor that is already
#    on the chip, so the crate never names the device.
# 3. **`torch.compile` traces Python, so it cannot see Rust.** The fused mode here is defer-and-fuse, which
#    works below ATen and needs only an environment variable.
#
# ### The three rules used throughout
#
# 1. **Every result is measured here.** Numbers from the earlier work appear beside yours as context.
# 2. **One checkpoint, the same prompts, one stopwatch.** Every way reads the same token ids, greedily.
# 3. **Every way starts cold,** in a fresh Python process.
#
# ### How each section is organized
#
# * **What we are testing:** the claim, and the evidence that checks it.
# * A code cell to run.
# * **What you should see:** the expected result, and what a different one would mean.
#
# ### Before you start
#
# * **Runtime.** Runtime, then Change runtime type, then TPU. The target is a single v5e chip.
# * **No token.** Both repacks are public.
# * **Disk and time.** About 8.5 GB of download, a Rust toolchain, and a 2-minute crate build.
# * **Safety.** Nothing is uploaded and nothing outside the Colab VM changes.

# %% [markdown]
# ## 1. Setup
#
# **What we are testing.** Nothing yet. This section pins PyTorch 2.13.0 with TorchTPU, installs Rust,
# builds the crate against that PyTorch, and downloads the checkpoint.

# %%
# Cell 1.1: install, build, download
import importlib.metadata as md
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

start = time.perf_counter()
install = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch==2.13.0",
        "torch_tpu==0.2.0",
        "transformers==5.20.2",
        "compressed-tensors==0.19.0",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
    ],
    capture_output=True,
    text=True,
)
if install.returncode != 0:
    print(install.stderr[-4000:])
    raise RuntimeError("pip install failed")

print("Python", sys.version.split()[0])
for package in ("torch_tpu", "torch", "libtpu", "transformers", "compressed-tensors"):
    print(f"{package:20s} {md.version(package)}")
print(f"installed in {time.perf_counter() - start:.0f} s")

start = time.perf_counter()
subprocess.run(
    "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
    shell=True, check=True, capture_output=True,
)
if not pathlib.Path("torchtpu-dev").exists():
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "https://github.com/xbill9/torchtpu-dev"], check=True
    )
CRATE = pathlib.Path("torchtpu-dev/colab/rust").resolve()
env = {
    **os.environ,
    "LIBTORCH_USE_PYTORCH": "1",
    "PATH": os.path.expanduser("~/.cargo/bin:") + os.environ["PATH"],
}
build = subprocess.run(["cargo", "build", "--release"], cwd=CRATE, env=env, capture_output=True, text=True)
if build.returncode != 0:
    print(build.stderr[-4000:])
    raise RuntimeError("cargo build failed")
print(build.stderr.strip().splitlines()[-1].strip())
print(subprocess.run([env["PATH"].split(":")[0] + "/rustc", "--version"], capture_output=True, text=True).stdout.strip())
MOD = pathlib.Path("rust_module").resolve()
MOD.mkdir(exist_ok=True)
shutil.copy(CRATE / "target/release/libtorchtpu_rs.so", MOD / "torchtpu_rs.so")
print(f"Rust installed and crate built in {time.perf_counter() - start:.0f} s")

from huggingface_hub import snapshot_download

MODEL = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
start = time.perf_counter()
MODEL_DIR = pathlib.Path(snapshot_download(MODEL))
EMB4_DIR = pathlib.Path(snapshot_download(EMB4, allow_patterns=["model-rest-001.safetensors"]))
print(f"downloaded in {time.perf_counter() - start:.0f} s")

# %% [markdown]
# Cell 1.2 builds the prompts with notebook 05's generator, so the short prompt and the record prompts are
# the same token ids. Cell 1.3 writes the task script. Each task runs in a fresh process and prints one line
# of JSON; the notebook's own session never imports PyTorch, so the chip stays free.

# %%
# Cell 1.2: the prompts
PROMPTS = r'''
import json, random, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
QUESTION = "\n\nQuestion: what is the code for record {r:06d}? Answer with the code only."


def ids_of(content):
    ids = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def records(depths):
    """For each depth, about that many tokens of numbered records and a question about one of them."""
    rng, out = random.Random(1234), []
    head = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
    overhead = len(tok(head + QUESTION.format(r=0), add_special_tokens=False)["input_ids"])
    for depth in depths:
        lines, used, codes, i = [], overhead, {}, 0
        while True:
            i += 1
            code = rng.randrange(100000, 999999)
            line = f"Record {i:06d}: the code for record {i:06d} is {code}.\n"
            cost = len(tok(line, add_special_tokens=False)["input_ids"])
            if used + cost > depth and lines:
                break
            lines.append(line)
            codes[i] = code
            used += cost
            if depth == 0:
                break
        target = max(1, len(lines) // 4)
        out.append({"records": len(lines), "target": target, "code": codes[target],
                    "owner": {str(c): r for r, c in codes.items()},
                    "ids": ids_of("".join(lines) + QUESTION.format(r=target))})
    return out


json.dump({"sky": ids_of("Explain in two sentences why the sky is blue."),
           "short": records([0, 4096, 32768])}, open("prompts.json", "w"))
'''
subprocess.run([sys.executable, "-c", PROMPTS, MODEL], check=True, capture_output=True)
PROMPT_SETS = json.loads(pathlib.Path("prompts.json").read_text())
print(f"short prompt: {len(PROMPT_SETS['sky'])} tokens")
for p in PROMPT_SETS["short"]:
    print(f"{len(p['ids']):6d} tokens, {p['records']:5d} records, target {p['target']:5d}")

# %%
# Cell 1.3: the task script
TASKS = pathlib.Path("rust_tasks.py")
TASKS.write_text(r'''
import gc
import json
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers.models.gemma4 import modeling_gemma4 as g4

TASK, MODEL, EMB4_DIR = sys.argv[1:4]
PROMPT_SETS = json.load(open("prompts.json"))
KEY_MAPPING = {r"^model\.language_model\.": "model."}
NEW_TOKENS = 64
CHUNK = 256
END_OF_TURN = 106


def sync():
    torch.accelerator.synchronize()


def hbm():
    return torch.accelerator.memory_allocated() / 1e9


def peak():
    return torch.accelerator.max_memory_allocated() / 1e9


def packed_embedding(packed, scale, embed_scale):
    """Notebook 04's per-layer table, kept 4-bit: look up packed rows, then unpack only those."""

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


def load():
    """Notebook 04's fastest model: bf16 weights, no offload wrappers, the per-layer table 4-bit."""
    from compressed_tensors.offload.dispatch import remove_dispatch
    from safetensors import safe_open

    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="tpu",
        dtype=torch.bfloat16,
        key_mapping=KEY_MAPPING,
        quantization_config=transformers.CompressedTensorsConfig(run_compressed=False),
    ).eval()
    remove_dispatch(model)
    name = "model.language_model.embed_tokens_per_layer"
    with safe_open(f"{EMB4_DIR}/model-rest-001.safetensors", "pt") as f:
        packed, scale = f.get_tensor(name + ".weight_packed"), f.get_tensor(name + ".weight_scale")
    old = model.model.embed_tokens_per_layer
    model.model.embed_tokens_per_layer = packed_embedding(packed.to("tpu"), scale.to("tpu"), old.scalar_embed_scale)
    old.weight.data = torch.empty(0, dtype=torch.bfloat16, device="tpu")
    del old
    gc.collect()
    sync()
    return model, {"load_s": time.perf_counter() - start, "hbm_load_gb": hbm()}


def answer(tokens, prompt):
    """The reply up to the end of the turn, and whose code it is."""
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(snapshot_download(MODEL, allow_patterns=["tokenizer.json"]) + "/tokenizer.json")
    reply = tokenizer.decode(tokens[: tokens.index(END_OF_TURN)] if END_OF_TURN in tokens else tokens).strip()
    return {"answer": reply[:40], "found": reply == str(prompt["code"]), "code_of_record": prompt["owner"].get(reply)}
def graphs():
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"]), int(sum(counters["graph_break"].values()))
# --- notebook 05's ring cache, in Python and compiled -----------------------------------------------------------
class Ring(torch.nn.Module):
    """Keys and values of one layer that computes them (Gemma 4 E2B has one K/V head): rows [slots, D], and the
    position each slot holds, -1 when empty. A full-attention layer has a slot per position; a sliding layer
    reuses its slots in turn, slot = position % slots."""

    def __init__(self, slots, head_dim, window):
        super().__init__()
        self.slots, self.window = slots, window
        self.k = torch.nn.Buffer(torch.zeros(slots, head_dim, dtype=torch.bfloat16, device="tpu"))
        self.v = torch.nn.Buffer(torch.zeros(slots, head_dim, dtype=torch.bfloat16, device="tpu"))
        self.kpos = torch.nn.Buffer(torch.full((slots,), -1, dtype=torch.long, device="tpu"))

    def write(self, pos, k, v):
        slot = pos % self.slots
        self.k.index_copy_(0, slot, k)
        self.v.index_copy_(0, slot, v)
        self.kpos.index_copy_(0, slot, pos)

    def attend(self, q, pos):
        """q [n, heads, D] for the tokens at positions pos [n]: attention over every slot, masked."""
        scores = torch.einsum("nhd,kd->hnk", q, self.k).float()
        kpos, qpos = self.kpos[None, :], pos[:, None]
        keep = (kpos >= 0) & (kpos <= qpos)
        if self.window:
            keep = keep & (kpos > qpos - self.window)
        weights = torch.softmax(scores.masked_fill(~keep[None], float("-inf")), dim=-1).to(q.dtype)
        return torch.einsum("hnk,kd->nhd", weights, self.v)


def ring_attention(
    self, hidden_states, position_embeddings, attention_mask, shared_kv_states, position_ids=None, **kwargs
):
    """Gemma4TextAttention.forward with the ring in place of Transformers' cache and attention."""
    b, n, _ = hidden_states.shape
    cos, sin = position_embeddings
    pos = position_ids[0]
    q = self.q_proj(hidden_states).view(b, n, -1, self.head_dim)
    q = g4.apply_rotary_pos_emb(self.q_norm(q), cos, sin, unsqueeze_dim=2)
    if self.is_kv_shared_layer:
        ring = shared_kv_states[self.layer_type]
    else:
        k = self.k_proj(hidden_states).view(b, n, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(b, n, -1, self.head_dim) if self.v_proj is not None else k
        k = g4.apply_rotary_pos_emb(self.k_norm(k), cos, sin, unsqueeze_dim=2)
        ring = self.ring
        ring.write(pos, k[0, :, 0], self.v_norm(v)[0, :, 0])
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = ring
    return self.o_proj(ring.attend(q[0], pos).reshape(b, n, -1)), None


class RingEngine:
    # A dict of masks tells Gemma4TextModel they are already built; the ring builds its own.
    NO_MASKS = {"full_attention": None, "sliding_attention": None}

    def __init__(self, context):
        assert context % CHUNK == 0
        self.context = context
        self.model, self.info = load()
        model, cfg = self.model, self.model.config
        self.rings, layout = [], {}
        for mod in model.modules():
            if isinstance(mod, g4.Gemma4TextAttention):
                kind = "shared" if mod.is_kv_shared_layer else ("sliding" if mod.is_sliding else "full")
                if kind != "shared":
                    assert cfg.per_layer_config[mod.layer_idx].num_key_value_heads == 1
                    slots = cfg.sliding_window + CHUNK if kind == "sliding" else context
                    mod.ring = Ring(slots, mod.head_dim, cfg.sliding_window if kind == "sliding" else None)
                    self.rings.append(mod.ring)
                    layout.setdefault(kind, {}).update(slots=slots, head_dim=mod.head_dim)
                entry = layout.setdefault(kind, {})
                entry["layers"] = entry.get("layers", 0) + 1
                mod.forward = ring_attention.__get__(mod)
        sync()
        self.info.update(layout=layout, sliding_window=cfg.sliding_window, hbm_rings_gb=hbm(),
                         cache_gb=sum(2 * r.k.numel() * r.k.element_size() for r in self.rings) / 1e9)
        self.tok = torch.zeros((1, 1), dtype=torch.long, device="tpu")
        self.pos = torch.zeros(1, dtype=torch.long, device="tpu")

        def piece(ids, positions, last):
            # One 256-token piece of the prompt; logits only for its last real token.
            logits = model(input_ids=ids, position_ids=positions.view(1, -1), attention_mask=self.NO_MASKS,
                           use_cache=False, logits_to_keep=last).logits
            return logits[:, -1].argmax(-1)

        def step():
            # One token in, the next token out; the ring is written in place and the position moves on.
            logits = model(input_ids=self.tok, position_ids=self.pos.view(1, 1), attention_mask=self.NO_MASKS,
                           use_cache=False, logits_to_keep=1).logits
            nxt = logits[:, -1].argmax(-1)
            self.tok.copy_(nxt.view(1, 1))
            self.pos.add_(1)
            return nxt

        self.piece = torch.compile(piece, backend="tpu", dynamic=False)
        self.step = torch.compile(step, backend="tpu", dynamic=False)

    def read_prompt(self, ids):
        for r in self.rings:
            r.kpos.fill_(-1)
        for p0 in range(0, len(ids), CHUNK):
            part = ids[p0 : p0 + CHUNK]
            first = self.piece(
                torch.tensor([part + [0] * (CHUNK - len(part))], device="tpu"),
                torch.arange(p0, p0 + CHUNK, device="tpu"),
                torch.tensor([len(part) - 1], device="tpu"),
            )
        self.tok.copy_(first.view(1, 1))
        self.pos.fill_(len(ids))
        return first

    def generate(self, ids, stream=False):
        assert len(ids) + NEW_TOKENS <= self.context
        with torch.no_grad():
            start = time.perf_counter()
            tokens = [self.read_prompt(ids)]
            sync()
            read = time.perf_counter()
            for _ in range(NEW_TOKENS - 1):
                tokens.append(self.step())
                if stream:
                    int(tokens[-1])  # read each token back as it is made, as a streaming server must
            tokens = torch.cat(tokens).tolist()
            return tokens, read - start, time.perf_counter() - read

def task_python(context):
    engine = RingEngine(context)
    info = engine.info
    start = time.perf_counter()
    engine.generate(PROMPT_SETS["sky"])
    info["first_call_s"] = time.perf_counter() - start
    for label, stream in (("batched", False), ("batched", False), ("stream", True), ("stream", True)):
        tokens, read_s, decode_s = engine.generate(PROMPT_SETS["sky"], stream)
        info.setdefault(label, []).append({"read_s": read_s, "decode_s": decode_s})
    info.update(tokens=tokens, graphs=graphs())
    info["long"] = []
    for prompt in PROMPT_SETS["short"]:
        tokens, read_s, decode_s = engine.generate(prompt["ids"])
        info["long"].append({"prompt_tokens": len(prompt["ids"]), "read_s": read_s, "decode_s": decode_s,
                             **answer(tokens, prompt)})
    return {**info, "hbm_end_gb": hbm(), "hbm_peak_gb": peak()}


# --- the same engine in Rust ------------------------------------------------------------------------------------
def task_rust(context):
    import torch_tpu  # noqa: F401

    sys.path.insert(0, MODULE_DIR)
    import torchtpu_rs

    start = time.perf_counter()
    engine = torchtpu_rs.Engine(MODEL_DIR, EMB4_DIR, torch.zeros(1, device="tpu"), context)
    sync()
    info = {"load_s": time.perf_counter() - start, "hbm_load_gb": hbm(), "layout": engine.layout()}
    sky = PROMPT_SETS["sky"]
    start = time.perf_counter()
    engine.generate(sky, NEW_TOKENS, 1)
    info["first_call_s"] = time.perf_counter() - start
    for label, read_every in (("stream", 1), ("stream", 1), ("every_8", 8), ("every_8", 8), ("every_8", 8)):
        tokens, read_s, decode_s = engine.generate(sky, NEW_TOKENS, read_every)
        info.setdefault(label, []).append({"read_s": read_s, "decode_s": decode_s})
    info["tokens"] = tokens
    info["long"] = []
    for prompt in PROMPT_SETS["short"]:
        tokens, read_s, decode_s = engine.generate(prompt["ids"], NEW_TOKENS, 1)
        info["long"].append({"prompt_tokens": len(prompt["ids"]), "read_s": read_s, "decode_s": decode_s,
                             **answer(tokens, prompt)})
    return {**info, "hbm_end_gb": hbm(), "hbm_peak_gb": peak()}


if __name__ == "__main__":
    MODEL_DIR, MODULE_DIR = sys.argv[4:6]
    tasks = {"python": lambda: task_python(33024), "rust": lambda: task_rust(33024)}
    try:
        result = tasks[TASK]()
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    print("RESULT " + json.dumps(result))
''')


def run(task, extra_env=None, timeout=1800):
    """Run one task in a fresh process and return its result dict."""
    args = [sys.executable, str(TASKS), task, MODEL, str(EMB4_DIR), str(MODEL_DIR), str(MOD)]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(extra_env or {})})
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines:
        return {"error": f"exit {proc.returncode}: {proc.stderr[-900:]}"}
    return json.loads(lines[-1][len("RESULT ") :])


def rate(timing):
    """Decode tokens per second: the 63 tokens after the first, over the time from the first to the last."""
    return 63 / timing["decode_s"]


print("task script written:", TASKS.resolve())

# %% [markdown]
# ## 2. A Rust call lands on the chip
#
# **What we are testing.** The crate's smallest functions on a `tpu` tensor: an ATen call from Rust whose
# result stays on the chip, and a tensor built on the host in Rust and moved there without naming the device.

# %%
# Cell 2.1: the bridge
BRIDGE = r"""
import sys
sys.path.insert(0, sys.argv[1])
import torch
import torch_tpu
import torchtpu_rs
x = torch.ones(4, 4, device="tpu")
y = torchtpu_rs.matmul_sum(x)
print("Rust matmul then sum:", y.item(), "on", y.device)
z = torchtpu_rs.arange_like(5, x)
print("built in Rust, moved:", z.tolist(), "on", z.device)
print("bf16 kept:", torchtpu_rs.arange_like(3, torch.ones(1, dtype=torch.bfloat16, device="tpu")).dtype)
"""
print(subprocess.run([sys.executable, "-c", BRIDGE, str(MOD)], capture_output=True, text=True, check=True).stdout)

# %% [markdown]
# **What you should see.** 64.0 on `tpu:0`, then `[0.0, 1.0, 2.0, 3.0, 4.0]` on `tpu:0`, then
# `torch.bfloat16`. A result on `cpu`, or an error naming a missing kernel, would mean the two libtorch
# copies did not join up.

# %% [markdown]
# ## 3. The three ways
#
# **What we are testing.** The Rust engine under defer-and-fuse and under strict eager, against notebook 05's
# Python ring compiled with `torch.compile`, all at a 33,024-token context: compile time, tokens, and decode
# speed on the short prompt. The Python ring runs twice, on the 2.13.0 the crate needs and on the 2.14.1
# notebook 05 used, and the cell swaps PyTorch between them. It takes about 12 minutes.

# %%
# Cell 3.1: run every way
WAYS = {
    "Rust fused, torch 2.13": ("rust", {"TORCH_TPU_DEFER_AND_FUSE": "1"}),
    "Rust eager, torch 2.13": ("rust", {}),
    "Python compile, 2.13": ("python", {}),
    "Python compile, 2.14.1": ("python", {}),
}
results = {}
for way, (task, extra) in WAYS.items():
    if way.endswith("2.14.1"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.14.1",
                        "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
                       check=True, capture_output=True)
    start = time.perf_counter()
    results[way] = run(task, extra)
    print(f"{way:24s} finished in {time.perf_counter() - start:4.0f} s: {results[way].get('error', 'ok')}")

print()
print(f"{'way':24s} {'load s':>7s} {'first s':>8s} {'graphs':>7s} {'breaks':>7s} {'streamed':>9s} {'in batches':>11s} {'peak GB':>8s}")
for way, r in results.items():
    if "error" in r:
        continue
    streamed = max(rate(t) for t in r["stream"])
    batch = r.get("batched") or r.get("every_8")
    programs, breaks = r.get("graphs", ("-", "-"))
    print(f"{way:24s} {r['load_s']:7.1f} {r['first_call_s']:8.1f} {programs!s:>7s} {breaks!s:>7s} {streamed:9.1f} "
          f"{max(rate(t) for t in batch):11.1f} {r['hbm_peak_gb']:8.2f}")

reference = results["Python compile, 2.14.1"].get("tokens", [])
for way, r in results.items():
    if "error" not in r and way != "Python compile, 2.14.1":
        same = next((i for i, (a, b) in enumerate(zip(r["tokens"], reference)) if a != b), len(reference))
        print(f"{way}: first {same} of {len(reference)} tokens identical to Python compile on 2.14.1")

# %% [markdown]
# @@SEE3@@

# %% [markdown]
# ## 4. Long prompts
#
# **What we are testing.** The Rust engine on notebook 05's record prompts: reading speed, decode speed, and
# whether it gives the same answers as the Python ring.

# %%
# Cell 4.1: long prompts
print(f"{'way':24s} {'prompt':>7s} {'read s':>7s} {'read tok/s':>10s} {'decode':>7s} {'answer':>8s} {'found':>6s}")
for way, r in results.items():
    if "error" in r:
        continue
    for row in r["long"]:
        print(f"{way:24s} {row['prompt_tokens']:7d} {row['read_s']:7.1f} {row['prompt_tokens'] / row['read_s']:10.0f} "
              f"{rate(row):7.1f} {row['answer']:>8s} {'yes' if row['found'] else 'no':>6s}")

# %% [markdown]
# @@SEE4@@

# %% [markdown]
# ## 5. Scorecard
#
# **What we are testing.** Each claim next to what you measured.

# %%
# Cell 5.1: the scorecard
fuse, eager, py213, py = (results[w] for w in WAYS)
ok = all("error" not in r for r in (fuse, eager, py213, py))
tokens_same = sum(a == b for a, b in zip(fuse.get("tokens", []), py.get("tokens", [])))
answers_same = sum(a["answer"] == b["answer"] for a, b in zip(fuse.get("long", []), py.get("long", [])))
rows = [
    ("Rust layout: full / sliding / shared", "3 / 12 / 20", " / ".join(str(v) for v in fuse.get("layout", [])[:3])),
    ("short-prompt tokens, Rust to compiled 2.14.1", "64 of 64", f"{tokens_same} of 64"),
    ("long-prompt answers, Rust to compiled 2.14.1", "3 of 3", f"{answers_same} of 3"),
    ("compiled programs on torch 2.13 / 2.14.1", "65 / 2", f"{py213.get('graphs', ['?'])[0]} / {py.get('graphs', ['?'])[0]}"),
]
if ok:
    rows.append(("streamed, Rust fused over compiled 2.14.1", "@@RATIO@@",
                 f"{max(rate(t) for t in fuse['stream']) / max(rate(t) for t in py['stream']):.2f}x"))
    rows.append(("streamed, Rust fused over Rust eager", "about 3.2x",
                 f"{max(rate(t) for t in fuse['stream']) / max(rate(t) for t in eager['stream']):.2f}x"))
print(f"{'claim':48s} {'expected':>12s} {'today':>12s}")
for claim, expected, today in rows:
    print(f"{claim:48s} {expected:>12s} {today:>12s}")

# %% [markdown]
# @@WHICH@@
