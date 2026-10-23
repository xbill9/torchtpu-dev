# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# Probe for notebook 05: the 1650 Ti engine's K/V layout on TorchTPU. Full-attention layers own MAX_CTX slots
# (slot = position); sliding layers a ring of sliding_window + CHUNK slots that records each slot's position, so a
# padded prompt chunk never overwrites a key a query still needs and no hold-back is required. Prompts are read in
# fixed CHUNK-token pieces (one compiled graph), decode is one compiled step (a second graph). Attention is plain
# PyTorch with a mask. Recall prompts are the 1650 Ti rig's (seed 1234, same depth order), so its answers and
# llama.cpp's apply.

# %%
import json, os, pathlib, subprocess, sys, time
import importlib.metadata as md

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_tpu==0.2.0", "compressed-tensors",
                    "--extra-index-url", "https://download.pytorch.org/whl/cpu"], capture_output=True, text=True)
print("pip rc", r.returncode, r.stderr[-800:])
for p in ("torch_tpu", "torch", "libtpu", "transformers", "compressed-tensors"):
    print(f"{p:20s} {md.version(p)}")
from huggingface_hub import snapshot_download
TEXT = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
snapshot_download(TEXT)
EMB4_DIR = snapshot_download(EMB4, allow_patterns=["model-rest-001.safetensors"])

PROMPTS = r'''
import json, random, sys
from transformers import AutoTokenizer
TEXT = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
tok = AutoTokenizer.from_pretrained(TEXT)
QUESTION = "\n\nQuestion: what is the code for record {r:06d}? Answer with the code only."

def ids_of(content):
    ids = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)

def recall(depths):
    rng, out = random.Random(1234), []
    head = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
    overhead = len(tok(head + QUESTION.format(r=0), add_special_tokens=False)["input_ids"])
    for depth in depths:
        records, used, codes, i = [], overhead, {}, 0
        while True:
            i += 1
            code = rng.randrange(100000, 999999)
            line = f"Record {i:06d}: the code for record {i:06d} is {code}.\n"
            cost = len(tok(line, add_special_tokens=False)["input_ids"])
            if used + cost > depth and records:
                break
            records.append(line); codes[i] = code; used += cost
            if depth == 0:
                break
        target = max(1, len(records) // 4)
        out.append({"depth": depth, "records": len(records), "target": target, "code": codes[target],
                    "owners": {str(c): r for r, c in codes.items()},
                    "ids": ids_of("".join(records) + QUESTION.format(r=target))})
    return out

json.dump({"sky": ids_of("Explain in two sentences why the sky is blue."),
           "short": recall([0, 4096, 32768]), "deep": recall([32768, 131000])}, open("prompts.json", "w"))
'''
subprocess.run([sys.executable, "-c", PROMPTS], check=True, capture_output=True)
P = json.load(open("prompts.json"))
print("sky", len(P["sky"]), [(p["depth"], len(p["ids"]), p["target"], p["code"]) for p in P["short"] + P["deep"]])

# %%
TASKS = pathlib.Path("ring_tasks.py")
TASKS.write_text(r'''
import gc, json, sys, time, traceback
import torch, transformers
from transformers import AutoModelForCausalLM
from transformers.models.gemma4 import modeling_gemma4 as g4

TASK, TEXT, EMB4_DIR = sys.argv[1:4]
P = json.load(open("prompts.json"))
KEY_MAPPING = {r"^model\.language_model\.": "model."}
CHUNK = 256
NO_MASKS = {"full_attention": None, "sliding_attention": None}  # a dict: Gemma4TextModel skips building masks
out = {"task": TASK}


def sync():
    torch.accelerator.synchronize()


def hbm():
    return round(torch.accelerator.memory_allocated() / 1e9, 3)


def peak():
    try:
        return round(torch.accelerator.max_memory_allocated() / 1e9, 3)
    except Exception as exc:
        return repr(exc)[:80]


def load():
    from compressed_tensors.offload.dispatch import remove_dispatch
    from safetensors import safe_open

    t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        TEXT, device_map="tpu", dtype=torch.bfloat16, key_mapping=KEY_MAPPING,
        quantization_config=transformers.CompressedTensorsConfig(run_compressed=False)).eval()
    remove_dispatch(model)
    name = "model.language_model.embed_tokens_per_layer"
    with safe_open(f"{EMB4_DIR}/model-rest-001.safetensors", "pt") as f:
        packed, scale = f.get_tensor(name + ".weight_packed").to("tpu"), f.get_tensor(name + ".weight_scale").to("tpu")
    old = model.model.embed_tokens_per_layer
    embed_scale = float(old.scalar_embed_scale)

    class Packed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.packed, self.scale = torch.nn.Buffer(packed), torch.nn.Buffer(scale)
            self.shifts = torch.nn.Buffer(torch.arange(0, 32, 4, dtype=torch.int32, device="tpu"))

        def forward(self, ids):
            p, s = self.packed[ids], self.scale[ids]
            q = ((p.unsqueeze(-1) >> self.shifts) & 15).to(torch.float32) - 8
            w = (q.view(*ids.shape, s.shape[-1], -1) * s.unsqueeze(-1).to(torch.float32)).to(torch.bfloat16)
            return w.view(*ids.shape, -1) * torch.tensor(embed_scale, dtype=torch.bfloat16)

    model.model.embed_tokens_per_layer = Packed()
    old.weight.data = torch.empty(0, dtype=torch.bfloat16, device="tpu")
    del old
    gc.collect()
    sync()
    out["load_s"] = round(time.perf_counter() - t, 1)
    out["hbm_load_gb"] = hbm()
    return model


class Ring(torch.nn.Module):
    """K/V of one layer that owns them (one KV head): rows [slots, D], and the position each slot holds (-1: empty)."""

    def __init__(self, cap, D, window):
        super().__init__()
        self.cap, self.window = cap, window
        self.k = torch.nn.Buffer(torch.zeros(cap, D, dtype=torch.bfloat16, device="tpu"))
        self.v = torch.nn.Buffer(torch.zeros(cap, D, dtype=torch.bfloat16, device="tpu"))
        self.kpos = torch.nn.Buffer(torch.full((cap,), -1, dtype=torch.long, device="tpu"))

    def write(self, pos, k, v):
        slot = pos % self.cap
        self.k.index_copy_(0, slot, k)
        self.v.index_copy_(0, slot, v)
        self.kpos.index_copy_(0, slot, pos)

    def attend(self, q, pos):
        """q [n, H, D] at positions pos [n] -> [n, H, D]."""
        s = torch.einsum("nhd,kd->hnk", q, self.k).float()
        kp, qp = self.kpos[None, :], pos[:, None]
        keep = (kp >= 0) & (kp <= qp)
        if self.window:
            keep = keep & (kp > qp - self.window)
        w = torch.softmax(s.masked_fill(~keep[None], float("-inf")), dim=-1).to(q.dtype)
        return torch.einsum("hnk,kd->nhd", w, self.v)


def ring_forward(self, hidden_states, position_embeddings, attention_mask, shared_kv_states, past_key_values=None,
                 position_ids=None, **kwargs):
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
        v = self.v_norm(v)
        ring = self.ring
        ring.write(pos, k[0, :, 0], v[0, :, 0])
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = ring
    o = ring.attend(q[0], pos)
    return self.o_proj(o.reshape(b, n, -1)), None


class Engine:
    def __init__(self, max_ctx):
        assert max_ctx % CHUNK == 0
        self.max_ctx = max_ctx
        self.model = model = load()
        cfg = model.config
        self.rings = []
        for mod in model.modules():
            if isinstance(mod, g4.Gemma4TextAttention):
                assert cfg.per_layer_config[mod.layer_idx].num_key_value_heads == 1
                if not mod.is_kv_shared_layer:
                    cap = cfg.sliding_window + CHUNK if mod.is_sliding else max_ctx
                    mod.ring = Ring(cap, mod.head_dim, cfg.sliding_window if mod.is_sliding else None)
                    self.rings.append(mod.ring)
                mod.forward = mod.ring_forward = ring_forward.__get__(mod)
        out["rings"] = {"full": sum(r.window is None for r in self.rings), "sliding": sum(r.window is not None for r in self.rings),
                        "head_dims": sorted({r.k.shape[1] for r in self.rings})}
        out["kv_gb"] = round(sum(2 * r.k.numel() * r.k.element_size() for r in self.rings) / 1e9, 3)
        sync()
        out["hbm_rings_gb"] = hbm()
        self.tok = torch.zeros((1, 1), dtype=torch.long, device="tpu")
        self.pos = torch.zeros(1, dtype=torch.long, device="tpu")

        def chunk(ids, positions, last):
            logits = model(input_ids=ids, position_ids=positions.view(1, -1), attention_mask=NO_MASKS,
                           use_cache=False, logits_to_keep=last).logits
            return logits[:, -1].argmax(-1)

        def step():
            logits = model(input_ids=self.tok, position_ids=self.pos.view(1, 1), attention_mask=NO_MASKS,
                           use_cache=False, logits_to_keep=1).logits
            nxt = logits[:, -1].argmax(-1)
            self.tok.copy_(nxt.view(1, 1))
            self.pos.add_(1)
            return nxt

        self.chunk = torch.compile(chunk, backend="tpu", dynamic=False)
        self.step = torch.compile(step, backend="tpu", dynamic=False)

    def read_prompt(self, ids):
        n = len(ids)
        assert n < self.max_ctx
        for r in self.rings:
            r.kpos.fill_(-1)
        for p0 in range(0, n, CHUNK):
            piece = ids[p0 : p0 + CHUNK]
            last = torch.tensor([len(piece) - 1], device="tpu")
            first = self.chunk(torch.tensor([piece + [0] * (CHUNK - len(piece))], device="tpu"),
                               torch.arange(p0, p0 + CHUNK, device="tpu"), last)
        self.tok.copy_(first.view(1, 1))
        self.pos.fill_(n)
        return first

    def generate(self, ids, new, stream=False):
        """Greedy; returns tokens, prompt seconds, decode seconds (new - 1 steps)."""
        with torch.no_grad():
            t0 = time.perf_counter()
            toks = [self.read_prompt(ids)]
            sync()
            t1 = time.perf_counter()
            for _ in range(min(new, self.max_ctx - len(ids)) - 1):
                toks.append(self.step())
                if stream:
                    int(toks[-1])
            toks = torch.cat(toks).tolist()
            t2 = time.perf_counter()
        return toks, t1 - t0, t2 - t1


def counted():
    from torch._dynamo.utils import counters
    return {"graphs": int(counters["stats"]["unique_graphs"]), "breaks": int(sum(counters["graph_break"].values())),
            "break_reasons": sorted(counters["graph_break"])[:5]}


def answer(toks, p):
    from tokenizers import Tokenizer
    from huggingface_hub import snapshot_download
    tk = Tokenizer.from_file(snapshot_download(TEXT, allow_patterns=["tokenizer.json"]) + "/tokenizer.json")
    t = toks[: toks.index(106)] if 106 in toks else toks
    text = tk.decode(t).strip()
    return {"answer": text[:40], "recalled": text == str(p["code"]), "answer_is_code_of": p["owners"].get(text)}


def task_ring(max_ctx, sets):
    eng = Engine(max_ctx)
    t = time.perf_counter()
    toks, _, _ = eng.generate(P["sky"], 8)
    out["first_call_s"] = round(time.perf_counter() - t, 1)
    out["after_compile"] = counted()
    out["hbm_peak_compile_gb"] = peak()
    runs = []
    for label, stream in (("sky", False), ("sky", False), ("sky", True)):
        toks, pre, dec = eng.generate(P["sky"], 64, stream)
        runs.append({"stream": stream, "prompt_s": round(pre, 3), "decode_tok_s": round(63 / dec, 2)})
    out["sky_runs"], out["sky_tokens"] = runs, toks
    depth_runs = []
    for s in sets:
        for p in P[s]:
            toks, pre, dec = eng.generate(p["ids"], 64)
            depth_runs.append({"set": s, "depth": p["depth"], "prompt_tokens": len(p["ids"]), "prompt_s": round(pre, 2),
                               "prompt_tok_s": round(len(p["ids"]) / pre, 1), "decode_tok_s": round(63 / dec, 2),
                               **answer(toks, p), "hbm_peak_gb": peak()})
            print(json.dumps(depth_runs[-1]), flush=True)
    out["depths"] = depth_runs
    out["after_all"] = counted()
    out["hbm_end_gb"] = hbm()


def task_nb04(max_cache, prompt):
    """Notebook 04's recipe: StaticCache, StaticLayer's methods on the sliding layers, compiled step."""
    import transformers.cache_utils as cu
    from transformers import StaticCache

    model = load()
    for m in ("update", "get_seq_length", "get_mask_sizes"):
        setattr(cu.StaticSlidingWindowLayer, m, getattr(cu.StaticLayer, m))
    cache = StaticCache(config=model.config, max_cache_len=max_cache)
    out["sliding_layer_slots"] = sorted({tuple(l.keys.shape) for l in cache.layers if hasattr(l, "keys") and l.keys is not None})[:4] if hasattr(cache, "layers") else None
    ids_list = P["sky"] if prompt == "sky" else P["short"][1]["ids"]
    ids = torch.tensor([ids_list], device="tpu")
    tok = torch.zeros((1, 1), dtype=torch.long, device="tpu")

    def step():
        return model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1)

    step = torch.compile(step, backend="tpu", dynamic=False)

    def decode(n):
        with torch.no_grad():
            cache.reset()
            t0 = time.perf_counter()
            first = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1)
            tok.copy_(first.view(1, 1))
            sync()
            t1 = time.perf_counter()
            toks = [first]
            for _ in range(n - 1):
                nxt = step()
                tok.copy_(nxt.view(1, 1))
                toks.append(nxt)
            toks = torch.cat(toks).tolist()
            return toks, t1 - t0, time.perf_counter() - t1

    decode(8)
    toks, pre, dec = decode(64)
    toks, pre, dec = decode(64)
    out.update({"tokens": toks, "prompt_s": round(pre, 2), "decode_tok_s": round(63 / dec, 2), **counted(), "hbm_peak_gb": peak()})
    if prompt != "sky":
        out.update(answer(toks, P["short"][1]))


try:
    {
        "ring": lambda: task_ring(33024, ["short"]),
        "ring_131k": lambda: task_ring(131328, ["deep"]),
        "nb04_sky": lambda: task_nb04(256, "sky"),
        "nb04_4k": lambda: task_nb04(4352, "4k"),
    }[TASK]()
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"[:3000]
    out["traceback"] = traceback.format_exc()[-3000:]
print("RESULT " + json.dumps(out))
''')


def run(task, timeout=3600):
    t = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, str(TASKS), task, TEXT, EMB4_DIR], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"task": task, "error": f"timeout after {timeout} s", "stdout": (exc.stdout or b"")[-3000:].decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")[-3000:]}
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not lines:
        return {"task": task, "error": f"rc {proc.returncode}", "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-4000:]}
    res = json.loads(lines[-1][7:])
    res["wall_s"] = round(time.perf_counter() - t, 1)
    return res


RESULTS = {}
for task in ("nb04_sky", "ring", "nb04_4k", "ring_131k"):
    RESULTS[task] = run(task)
    print(json.dumps(RESULTS[task])[:5000], flush=True)
json.dump(RESULTS, open("probe_ring_results.json", "w"))

# %%
ref = RESULTS["nb04_sky"].get("tokens")
got = RESULTS["ring"].get("sky_tokens")
if ref and got:
    end = ref.index(106) if 106 in ref else len(ref)
    same = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
    print(f"ring vs notebook 04 on the sky prompt: first {same} tokens identical; answer ends at {end}")
