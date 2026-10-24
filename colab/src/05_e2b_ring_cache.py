# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B on TorchTPU: a ring cache for the full 131k context
#
# **Author:** William McLean, Google Developer Expert
#
# **Based on:** the repack
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)
# and its 4-bit-tables twin
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4),
# notebook 04 of this series, and the K/V layout of the plain-PyTorch engine in
# [`xbill9/gemma4-dev`](https://github.com/xbill9/gemma4-dev) (`local-pytorch-1650ti-2b-q4w4a16`).
#
# ### What this notebook is for
#
# Taking notebook 04's compiled decode step from a 256-token cache to the model's full 131,072-token
# context, on one v5e chip, with Hugging Face Transformers' own model code and no custom kernels.
# Notebook 04's cache depends on a fix that holds only while the cache is no longer than Gemma 4's
# 512-token sliding window. This notebook replaces Transformers' cache with the layout a 4 GB laptop
# GPU uses to hold the full context, and measures it on the TPU.
#
# ### The ring cache
#
# Gemma 4 E2B has 35 layers, and only 15 of them compute keys and values:
#
# 1. **3 full-attention layers** keep one slot per position, up to the whole context.
# 2. **12 sliding-window layers** only ever look back 512 positions, so each keeps a ring of slots that
#    is reused as the text grows, and records which position each slot holds.
# 3. **The other 20 layers** reuse the keys and values of the last full and the last sliding layer
#    before them.
#
# Prompts are read in fixed 256-token pieces and decoding goes one token at a time, so the whole model
# compiles to exactly two programs, whatever the prompt's length. Each sliding ring holds 512 + 256
# slots: room for a full window plus one piece, so writing a piece never overwrites a key that a query in
# the same piece still needs.
#
# ### What each section shows
#
# * **Section 2, the layout.** How many slots each layer gets, and what the cache costs in memory.
# * **Section 3, short prompts.** The ring cache against notebook 04's step: same tokens, and speed.
# * **Section 4, long prompts.** Prompts of 61 to 130,981 tokens: reading speed, decode speed, and
#   whether the model finds a record buried in the text, on the same prompts a laptop GPU and llama.cpp
#   answered.
# * **Section 5, notebook 04's cache past its window.** Why it cannot simply be made longer.
# * **Section 6, memory.** What the chip holds.
#
# ### The three rules used throughout
#
# 1. **Every result is measured here.** Numbers from the earlier work appear beside yours as context,
#    never in place of them.
# 2. **One checkpoint, the same prompts, one stopwatch.** Every way reads the same token ids, greedily,
#    and is timed the same way.
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
# * **No token.** Both repacks are public. Nothing asks for a Hugging Face token.
# * **Disk and time.** About 8.5 GB of download. The whole notebook takes about 15 minutes.
# * **Safety.** Nothing is uploaded and nothing outside the Colab VM changes.

# %% [markdown]
# ## 1. Setup
#
# **What we are testing.** Nothing yet. This section installs TorchTPU, downloads the checkpoint and
# the one 4-bit table file, builds the prompts, and records the versions every number below depends on.

# %%
# Cell 1.1: install, download, versions
import importlib.metadata as md
import json
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
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
start = time.perf_counter()
MODEL_DIR = pathlib.Path(snapshot_download(MODEL))
EMB4_DIR = pathlib.Path(snapshot_download(EMB4, allow_patterns=["model-rest-001.safetensors"]))
for name, d in ((MODEL, MODEL_DIR), (EMB4, EMB4_DIR)):
    size = sum(f.stat().st_size for f in d.rglob("*.safetensors"))
    print(f"{name}: {size / 1e9:.2f} GB")
print(f"downloaded in {time.perf_counter() - start:.0f} s")

# %% [markdown]
# Cell 1.2 builds every prompt once, so every way reads the same token ids. The short prompt is
# notebook 04's. The long ones fill the context with numbered records, each with a random six-digit
# code, and ask for the code of the record a quarter of the way in. They come from the same generator,
# seed and order as the laptop GPU's long-context run, so its answers and llama.cpp's apply to them
# unchanged; the cell checks the target codes against that run's.

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
           "short": records([0, 4096, 32768]), "deep": records([32768, 131000])}, open("prompts.json", "w"))
'''
subprocess.run([sys.executable, "-c", PROMPTS, MODEL], check=True, capture_output=True)
PROMPT_SETS = json.loads(pathlib.Path("prompts.json").read_text())

# Target codes of the laptop GPU's run (gemma4-dev, local-pytorch-1650ti-2b-q4w4a16,
# benchmarks/runs/2026-10-23-long-context-1650ti, prompts-short.json and prompts-deep.json).
LAPTOP_CODES = {"short": [915965, 610395, 713468], "deep": [390185, 492097]}
print(f"short prompt: {len(PROMPT_SETS['sky'])} tokens")
for name in ("short", "deep"):
    for p, code in zip(PROMPT_SETS[name], LAPTOP_CODES[name]):
        same = "same as the laptop run" if p["code"] == code else f"DIFFERENT from the laptop run ({code})"
        print(f"{name:5s} {len(p['ids']):7d} tokens, {p['records']:5d} records, target {p['target']:5d}: {same}")

# %% [markdown]
# Cell 1.3 writes the script every later section runs. Each task in it loads the model in a fresh
# process, does one thing, and prints its result as one line of JSON. The notebook's own Python
# session never imports PyTorch, so the chip stays free for those processes.
#
# Both ways load the model the way notebook 04's fastest one did: weights unpacked to bf16 at load,
# compressed-tensors' module wrappers removed, and the per-layer embedding table kept 4-bit.

# %%
# Cell 1.3: the task script
TASKS = pathlib.Path("ring_tasks.py")
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


# --- the ring cache ---------------------------------------------------------------------------------------------
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


def task_ring(context, prompt_set):
    engine = RingEngine(context)
    info = engine.info
    start = time.perf_counter()
    engine.generate(PROMPT_SETS["sky"])
    info["first_call_s"] = time.perf_counter() - start
    info["graphs_after_short"] = graphs()
    for label, stream in (("warm", False), ("warm", False), ("stream", True)):
        tokens, read_s, decode_s = engine.generate(PROMPT_SETS["sky"], stream)
        info.setdefault(label, []).append({"read_s": read_s, "decode_s": decode_s})
    info["tokens"] = tokens
    info["long"] = []
    for prompt in PROMPT_SETS[prompt_set]:
        tokens, read_s, decode_s = engine.generate(prompt["ids"])
        info["long"].append({"prompt_tokens": len(prompt["ids"]), "read_s": read_s, "decode_s": decode_s,
                             **answer(tokens, prompt)})
    info["graphs_after_long"] = graphs()
    return {**info, "hbm_end_gb": hbm(), "hbm_peak_gb": peak()}


# --- notebook 04's cache ----------------------------------------------------------------------------------------
def task_static(cache_len, prompt):
    """Notebook 04's compiled step: StaticCache, StaticLayer's methods on the sliding layers."""
    import transformers.cache_utils as cu
    from transformers import StaticCache

    model, info = load()
    for method in ("update", "get_seq_length", "get_mask_sizes"):
        setattr(cu.StaticSlidingWindowLayer, method, getattr(cu.StaticLayer, method))
    cache = StaticCache(config=model.config, max_cache_len=cache_len)
    ids = torch.tensor([prompt], device="tpu")
    tok = torch.zeros((1, 1), dtype=torch.long, device="tpu")

    def step():
        return model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1)

    step = torch.compile(step, backend="tpu", dynamic=False)

    def decode(n):
        with torch.no_grad():
            cache.reset()
            start = time.perf_counter()
            first = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
            tok.copy_(first.argmax(-1).view(1, 1))
            tokens = [first.argmax(-1)]
            sync()
            read = time.perf_counter()
            for _ in range(n - 1):
                nxt = step()
                tok.copy_(nxt.view(1, 1))
                tokens.append(nxt)
            tokens = torch.cat(tokens).tolist()
            return tokens, read - start, time.perf_counter() - read

    decode(8)
    for _ in range(2):
        tokens, read_s, decode_s = decode(NEW_TOKENS)
        info.setdefault("warm", []).append({"read_s": read_s, "decode_s": decode_s})
    info.update(tokens=tokens, graphs_after_short=graphs(), hbm_peak_gb=peak())
    return info


def task_static_slots(cache_len):
    """The slots Transformers' StaticCache gives each kind of layer; built on the host, nothing on the chip."""
    import collections

    from transformers import AutoConfig, StaticCache

    cache = StaticCache(config=AutoConfig.from_pretrained(MODEL), max_cache_len=cache_len)
    kinds = collections.Counter((type(layer).__name__, layer.max_cache_len) for layer in cache.layers)
    return {"cache_len": cache_len, "kinds": [[name, slots, n] for (name, slots), n in sorted(kinds.items())]}


if __name__ == "__main__":
    tasks = {
        "static_short": lambda: task_static(256, PROMPT_SETS["sky"]),
        "ring_33k": lambda: task_ring(33024, "short"),
        "ring_131k": lambda: task_ring(131328, "deep"),
        "static_slots": lambda: task_static_slots(4352),
    }
    try:
        result = tasks[TASK]()
    except Exception as exc:  # a way that fails is a result too; report it instead of crashing the cell
        result = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    print("RESULT " + json.dumps(result))
''')


def run(task, timeout=1800):
    """Run one task in a fresh process and return its result dict."""
    args = [sys.executable, str(TASKS), task, MODEL, str(EMB4_DIR)]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:])
        raise RuntimeError(f"task {task} exited with code {proc.returncode} and no result")
    return json.loads(lines[-1][len("RESULT ") :])


def rate(timing):
    """Decode tokens per second: the 63 tokens after the first, over the time from the first to the last."""
    return 63 / timing["decode_s"]


print("task script written:", TASKS.resolve())

# %% [markdown]
# ## 2. The layout
#
# **What the source says.** The laptop GPU's engine (`e2b_engine.py` in gemma4-dev) holds the full
# 131,072-token context in 774 MiB of keys and values: 3 layers with a slot per position, 12 sliding
# layers with 512 slots each, and 20 layers that own none.
#
# **What we are testing.** The same count read from the model's own attention modules on the TPU, and
# the cache's size for a 33,024-token and a 131,328-token context (multiples of the 256-token piece).
# This cell also runs Section 3's and Section 4's ways, which take about 7 minutes.

# %%
# Cell 2.1: run the ring-cache ways, and the layout they built
results = {}
for way in ("static_short", "ring_33k", "ring_131k"):
    start = time.perf_counter()
    results[way] = run(way)
    print(f"{way:13s} finished in {time.perf_counter() - start:4.0f} s: {results[way].get('error', 'ok')}")

ring, big = results["ring_33k"], results["ring_131k"]
print()
for kind in ("full", "sliding", "shared"):
    entry = ring["layout"][kind]
    slots = f"{entry['slots']:7d} slots" if "slots" in entry else "  reuses an earlier layer's"
    print(f"{kind:8s} layers: {entry['layers']:2d}  {slots}  head size {entry.get('head_dim', '')}")
print(f"sliding window: {ring['sliding_window']} positions")
for name, r in (("33,024-token", ring), ("131,328-token", big)):
    print(f"cache for a {name} context: {r['cache_gb']:.3f} GB")

# %% [markdown]
# **What you should see.** 3 full layers, 12 sliding layers with 768 slots (the 512-position window and
# one 256-token piece) and 20 shared layers; full layers with a head size of 512 and sliding layers 256.
# In four runs on a Colab v5e-1 on 2026-10-23 the cache took 0.212 GB for a 33,024-token context and 0.816
# GB for a 131,328-token one. Every full-length slot is two rows of 512 bf16 values in each of 3 layers,
# 6 KB per token, so the full context costs less than a gigabyte on a 16 GB chip.

# %% [markdown]
# ## 3. Short prompts: same tokens, and speed
#
# **What the source says.** Notebook 04 measured its compiled step with the 4-bit table at 50.2 to 50.4
# tokens per second on a 64-token reply to the short prompt, timed from outside, and found every way
# writing the same 56-token answer.
#
# **What we are testing.** Notebook 04's step and the ring cache on the same short prompt, both timed
# the same way here: the 63 tokens after the first, from the first token to the last, collected at the
# end. Whether the two write the same tokens, and how many compiled programs each became. The ring cache
# also runs once more reading each token back to the host as it is made.

# %%
# Cell 3.1: notebook 04's step against the ring cache
static = results["static_short"]
print(f"{'way':30s} {'first call s':>12s} {'tok/s':>7s} {'streamed':>9s} {'graphs':>7s} {'breaks':>7s}")
for label, r in (("notebook 04 step, 256 slots", static), ("ring, 33,024", ring), ("ring, 131,328", big)):
    if "error" in r:
        print(f"{label:30s} failed: {r['error'][:90]}")
        continue
    best = max(rate(t) for t in r["warm"])
    streamed = f"{rate(r['stream'][0]):9.1f}" if "stream" in r else f"{'':9s}"
    first = f"{r['first_call_s']:12.1f}" if "first_call_s" in r else f"{'':12s}"
    print(f"{label:30s} {first} {best:7.1f} {streamed} {r['graphs_after_short'][0]:7d} {r['graphs_after_short'][1]:7d}")

reference = static["tokens"]
answer_end = reference.index(106)
for label, r in (("ring, 33,024", ring), ("ring, 131,328", big)):
    same = next((i for i, (a, b) in enumerate(zip(r["tokens"], reference)) if a != b), len(reference))
    print(f"{label}: first {same} of {len(reference)} tokens identical to notebook 04's step "
          f"(the answer ends at token {answer_end})")

# %% [markdown]
# **What you should see.** All 64 tokens identical, and the ring cache a few percent slower than
# notebook 04's step at a 33,024-token context and more at 131,328. In four runs on a Colab v5e-1 on
# 2026-10-23: notebook 04's step at 47.8 to 47.9 tokens per second on this stopwatch, the ring cache at
# 45.1 to 45.2 with a 33,024-token context and 38.6 to 38.7 with 131,328, and 32.8 to 34.4 and 28.7 to
# 30.7 when streamed. The ring cache compiled to 2 programs with no graph breaks; its first call,
# compiling both, took 91 to 98 s.
#
# * **Decode speed follows the cache's size, not the prompt's.** Every step attends over every slot and
#   masks the empty ones, so the program is the same at every position. A 131,328-slot cache reads 4 times
#   the keys of a 33,024-slot one on every token, whatever the prompt.
# * **Same tokens.** The ring writes and reads the same keys and values as Transformers' cache; only
#   where they live changes.

# %% [markdown]
# ## 4. Long prompts
#
# **What the source says.** On a GTX 1650 Ti laptop GPU (gemma4-dev,
# `benchmarks/runs/2026-10-23-long-context-1650ti`), the plain-PyTorch engine read these prompts at 79.4
# to 147.2 tokens per second up to 32,761 tokens, and the 130,981-token one at 64.7 (2,024 s); decode ran
# at 51.0 to 51.5 tokens per second at 32,761 tokens and 21.8 at 130,981. Its answers: the right code at
# 61 and 4,081 tokens, the code of record 723 instead of 272 on the first 32,761-token prompt, the right
# code on the second, and the code of record 91 instead of 1,091 at 130,981. llama.cpp answered the
# first three prompts the same, wrong code included.
#
# **What we are testing.** Reading speed, decode speed, and the answer for each long prompt: the first
# three with a 33,024-token context, the last two with 131,328.

# %%
# Cell 4.1: long prompts
LAPTOP = {  # the laptop GPU's answers and speeds on the same prompts, from the run named above
    ("short", 0): ("915965", 79.4, 87.86), ("short", 1): ("610395", 97.7, 80.51),
    ("short", 2): ("319550", 79.4, 51.54), ("deep", 0): ("390185", 147.2, 51.04),
    ("deep", 1): ("567239", 64.7, 21.81),
}
print(f"{'context':>8s} {'prompt':>8s} {'read s':>7s} {'read tok/s':>10s} {'decode':>7s} {'answer':>8s} "
      f"{'found':>6s} {'record':>7s}  {'laptop answer':>13s} {'same':>5s}")
same_answers = []
for name, r in (("short", ring), ("deep", big)):
    for i, row in enumerate(r["long"]):
        laptop = LAPTOP[(name, i)]
        same_answers.append(row["answer"] == laptop[0])
        print(f"{r['layout']['full']['slots']:8d} {row['prompt_tokens']:8d} {row['read_s']:7.1f} "
              f"{row['prompt_tokens'] / row['read_s']:10.0f} {rate(row):7.1f} {row['answer']:>8s} "
              f"{'yes' if row['found'] else 'no':>6s} {row['code_of_record'] or '-':>7}  {laptop[0]:>13s} "
              f"{'yes' if same_answers[-1] else 'no':>5s}")
print(f"graphs after every prompt: {ring['graphs_after_long']} and {big['graphs_after_long']} (programs, breaks)")

# %% [markdown]
# **What you should see.** The same five answers as the laptop GPU, two of them wrong in the same way,
# and no new compiled programs however long the prompt. In four runs on a Colab v5e-1 on 2026-10-23: the
# 32,761-token prompt read in 5.0 s (6,601 to 6,618 tokens per second) with a 33,024-token context, and
# the 130,981-token one in 35.5 to 39.8 s (3,288 to 3,693) with 131,328, against 2,024 s on the laptop;
# decode at 45.1 to 45.3 tokens per second at every depth with a 33,024-token context and 38.1 to 38.6
# with 131,328.
#
# * **The wrong answers belong to the model.** Two engines on different hardware, and llama.cpp where it
#   was run, return the same wrong code: finding one record among thousands of near-identical ones is at
#   the edge of what E2B does at 32k tokens and beyond. Five prompts are too few to put a rate on it.
# * **Reading slows with the context, not with the prompt.** Each 256-token piece attends over every slot
#   of the full-attention layers, so the same 32,761-token prompt reads at 50% to 56% of the speed with a
#   131,328-token context as with a 33,024-token one.

# %% [markdown]
# ## 5. Notebook 04's cache past its window
#
# **What the source says.** Notebook 04's fix gives Transformers' sliding-window cache layer the methods
# of the plain static layer, which count in a tensor on the chip. It holds while the cache is no longer
# than the 512-token window, the condition notebook 04 asserts.
#
# **What we are testing.** How many slots Transformers' `StaticCache` gives each kind of layer when it
# is asked for 4,352, enough for the 4,081-token prompt and a reply. The cache is only described here,
# never filled: on the chip, this recipe at that length is a fatal error (below).

# %%
# Cell 5.1: notebook 04's cache asked for 4,352 slots
results["static_slots"] = run("static_slots")
r = results["static_slots"]
for name, slots, layers in r["kinds"]:
    print(f"{layers:2d} layers of {name:26s} {slots:5d} slots")

# %% [markdown]
# **What you should see.** 3 `StaticLayer`s with 4,352 slots and 12 `StaticSlidingWindowLayer`s with 512:
# the sliding layer allocates `min(sliding_window, max_cache_len)` slots (`cache_utils.py` in
# Transformers 5.20.1). Notebook 04's fix gives those layers the plain static layer's `update`, which
# writes each token at its position, so a 4,081-token prompt writes past the end of their 512 slots.
# Run on the chip, twice on a Colab v5e-1 on 2026-10-23, that recipe with the 4,081-token prompt halted
# the chip with a fatal error the first time a result was read back (`RuntimeUnexpectedCoreHalt`, XLA
# error E0200; `runs/2026-10-23/probe_ring_cache_output.ipynb` and `05_e2b_ring_cache_output_run1.ipynb`
# in this repository); in a third run the cell never returned and the Colab session ended there
# (`05_e2b_ring_cache_output_run2.ipynb`). Beyond the window the cache needs its own layout, which is the
# ring.

# %% [markdown]
# ## 6. Memory on the chip
#
# **What we are testing.** What each way holds once loaded, with its cache, and the most it held at any
# moment, as TorchTPU reports them.

# %%
# Cell 6.1: memory
print(f"{'way':30s} {'after load GB':>13s} {'with cache GB':>13s} {'peak GB':>8s}")
for label, r in (("notebook 04 step, 256 slots", static), ("ring, 33,024", ring), ("ring, 131,328", big)):
    if "error" in r:
        continue
    cache = f"{r['hbm_rings_gb']:13.2f}" if "hbm_rings_gb" in r else f"{'':13s}"
    print(f"{label:30s} {r['hbm_load_gb']:13.2f} {cache} {r['hbm_peak_gb']:8.2f}")

# %% [markdown]
# **What you should see.** 6.44 GB after loading for every way, 6.65 GB with the 33,024-token ring and
# 7.26 GB with the 131,328-token one, and a peak of 16.7 to 16.9 GB for every way, close to the chip's
# 16 GiB (17.2 GB). The peak comes from compiling, not from the cache: it is about the same with 256
# slots as with 131,328.

# %% [markdown]
# ## 7. Scorecard
#
# **What we are testing.** Each claim next to what you measured.

# %%
# Cell 7.1: the scorecard
ok = all("error" not in results[w] for w in ("static_short", "ring_33k", "ring_131k"))
rows = [
    ("layers with full slots / sliding rings / shared", "3 / 12 / 20",
     " / ".join(str(ring["layout"][k]["layers"]) for k in ("full", "sliding", "shared")) if ok else "failed"),
    ("cache for a 131,328-token context, GB", "under 1", f"{big['cache_gb']:.2f}" if ok else "failed"),
    ("compiled programs, ring cache", "2", str(ring["graphs_after_long"][0]) if ok else "failed"),
    ("graph breaks, ring cache", "0", str(ring["graphs_after_long"][1] + big["graphs_after_long"][1]) if ok else "failed"),
    ("short-prompt tokens identical to notebook 04", "64 of 64",
     f"{next((i for i, (a, b) in enumerate(zip(ring['tokens'], reference)) if a != b), 64)} of 64" if ok else "failed"),
    ("ring decode, 33,024 over notebook 04 step", "about 0.94x",
     f"{max(rate(t) for t in ring['warm']) / max(rate(t) for t in static['warm']):.2f}x" if ok else "failed"),
    ("130,981-token prompt read, s", "35 to 40", f"{big['long'][1]['read_s']:.0f}" if ok else "failed"),
    ("answers the same as the laptop GPU", "5 of 5", f"{sum(same_answers)} of {len(same_answers)}"),
    ("notebook 04's cache at 4,352: sliding-layer slots", "512",
     " ".join(str(slots) for name, slots, _ in results["static_slots"]["kinds"] if "Sliding" in name)),
]
print(f"{'claim':52s} {'expected':>12s} {'today':>12s}")
for claim, expected, today in rows:
    print(f"{claim:52s} {expected:>12s} {today:>12s}")

# %% [markdown]
# **So, which one?** For anything past 512 tokens, the ring cache: notebook 04's model, Gemma 4's
# attention forward replaced by about 40 lines that keep a slot per position in 3 layers and a reused
# ring in 12, and prompts read in 256-token pieces. It compiles to two programs once, in 91 to 98 s, and
# then reads a 130,981-token prompt in 35 to 40 s and decodes at 38 to 39 tokens per second with the full
# context, writing the same tokens as notebook 04 and the same answers as the laptop GPU. Size the context
# to the work: a 33,024-token context decodes at 45 tokens per second, within a few percent of notebook
# 04's 256-slot step. Notebook 04's step remains the fastest way for short chats.
#
# **Scope.** One v5e chip on Colab, one checkpoint, greedy decoding, a single sequence at a time, six
# prompts, and contexts of 33,024 and 131,328 tokens. Batching, other piece sizes, a context sized to each
# prompt and a serving endpoint were not run here.

# %% [markdown]
# ## 8. Your notes
#
# * **Versions.** Which TorchTPU, PyTorch and Transformers versions (Cell 1.1)?
# * **Prompts.** Did the target codes match the laptop run (Cell 1.2)?
# * **Speed.** How fast did the ring cache decode at each context (Cells 3.1 and 4.1)?
# * **Answers.** Which codes did the model find (Cell 4.1)?
# * **Memory.** What was the peak (Cell 6.1)?

# %%
# Cell 8.1: your notes (edit this cell)
notes = """
Versions:
Prompts:
Speed:
Answers:
Memory:
Other findings:
"""
print(notes)
