# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# Probe for notebook 04: Gemma 4 E2B decode on TorchTPU with a static KV cache and one compiled step,
# the recipe of gemma4-dev/local-pytorch-1650ti-2b-q4w4a16 ported to a v5e-1. Each task runs in its
# own process and prints one RESULT line.

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
snapshot_download(EMB4, allow_patterns=["model-rest-001.safetensors", "config.json"])
print("downloaded")

# %%
TASKS = pathlib.Path("decode_tasks.py")
TASKS.write_text(r'''
import collections, json, os, sys, time, traceback
import torch, transformers
import transformers.cache_utils as cu
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

TEXT = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
TASK = sys.argv[1]
MAX_CACHE = 256
PROMPT = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
KEY_MAPPING = {r"^model\.language_model\.": "model."}
out = {"task": TASK}


def sync():
    torch.accelerator.synchronize()


def hbm():
    return round(torch.accelerator.memory_allocated() / 1e9, 3)


def prompt_ids():
    ids = AutoTokenizer.from_pretrained(TEXT).apply_chat_template(PROMPT, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def load(compressed=False):
    kw = {"dtype": torch.bfloat16, "key_mapping": KEY_MAPPING}
    if not compressed:
        kw["quantization_config"] = transformers.CompressedTensorsConfig(run_compressed=False)
    t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(TEXT, device_map="tpu", **kw).eval()
    sync()
    out["load_s"] = round(time.perf_counter() - t, 1)
    out["hbm_after_load_gb"] = hbm()
    return model


def task_hbm_compressed():
    model = load(compressed=True)
    lin = model.model.layers[0].mlp.gate_proj
    out["before"] = {"has_weight_packed": hasattr(lin, "weight_packed"), "weight": str(getattr(getattr(lin, "weight", None), "dtype", None))}
    out["hook"] = hasattr(model, "ct_decompress_hook")
    ids = torch.tensor([prompt_ids()], device="tpu")
    with torch.no_grad():
        model(input_ids=ids)
    sync()
    out["hbm_after_first_forward_gb"] = hbm()
    out["after"] = {"has_weight_packed": hasattr(lin, "weight_packed"), "weight": str(getattr(getattr(lin, "weight", None), "dtype", None))}
    with torch.no_grad():
        model(input_ids=ids)
    sync()
    out["hbm_after_second_forward_gb"] = hbm()


def task_generate():
    model = load()
    ids = prompt_ids()
    x = torch.tensor([ids], device="tpu")
    for n in (32, 32, 128):
        t = time.perf_counter()
        with torch.no_grad():
            g = model.generate(x, max_new_tokens=n, min_new_tokens=n, do_sample=False)
        sync()
        out[f"gen{n}_s"] = round(time.perf_counter() - t, 2)
        out[f"tokens{n}"] = g[0, len(ids):].tolist()
    out["tok_s_128"] = round(128 / out["gen128_s"], 2)


class PackedEmbedding(torch.nn.Module):
    """Scaled embedding over int4 rows (compressed-tensors pack-quantized: 8 nibbles per int32, low first, offset 8)."""

    def __init__(self, packed, scale, embed_scale):
        super().__init__()
        self.packed, self.scale = torch.nn.Buffer(packed), torch.nn.Buffer(scale)
        self.embed_scale = float(embed_scale)
        self.shifts = torch.nn.Buffer(torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device))

    def forward(self, ids):
        p, s = self.packed[ids], self.scale[ids]
        # Unpack in fp32 and round to bf16 once, as the bf16 table was; then scale as Gemma4TextScaledWordEmbedding does.
        q = ((p.unsqueeze(-1) >> self.shifts) & 15).to(torch.float32) - 8
        w = (q.view(*ids.shape, s.shape[-1], -1) * s.unsqueeze(-1).to(torch.float32)).to(torch.bfloat16)
        return w.view(*ids.shape, -1) * torch.tensor(self.embed_scale, dtype=torch.bfloat16)


def build(ple4=False):
    from accelerate.hooks import remove_hook_from_module

    from compressed_tensors.offload.dispatch import remove_dispatch

    model = load()
    remove_hook_from_module(model, recurse=True)
    # compressed-tensors wraps every module's forward to move its inputs to the module's device; the wrapper compares
    # data pointers, which Dynamo cannot trace, so the step splits at every module. Every tensor is already on the TPU.
    remove_dispatch(model)
    out["wrapped_after_remove"] = sum(hasattr(m, "_original_forward_func") for m in model.modules())
    tm = model.model
    if ple4:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        d = snapshot_download(EMB4, allow_patterns=["model-rest-001.safetensors"])
        name = "model.language_model.embed_tokens_per_layer"
        with safe_open(f"{d}/model-rest-001.safetensors", "pt") as f:
            packed = f.get_tensor(name + ".weight_packed")
            scale = f.get_tensor(name + ".weight_scale")
        out["ple4_shapes"] = [list(packed.shape), list(scale.shape), str(scale.dtype)]
        old = tm.embed_tokens_per_layer
        embed_scale = old.scalar_embed_scale
        tm.embed_tokens_per_layer = PackedEmbedding(packed.to("tpu"), scale.to("tpu"), embed_scale)
        old.weight.data = torch.empty(0, dtype=torch.bfloat16, device="tpu")
        del old
        import gc

        gc.collect()
        sync()
        out["hbm_after_ple4_gb"] = hbm()
    # StaticSlidingWindowLayer tracks its length as a Python int: a compiled step would guard on it and recompile
    # every token. While the cache is no longer than the window it never rolls, so StaticLayer's tensor counter is exact.
    out["sliding_window"] = model.config.sliding_window
    assert model.config.sliding_window >= MAX_CACHE
    for name in ("update", "get_seq_length", "get_mask_sizes"):
        setattr(cu.StaticSlidingWindowLayer, name, getattr(cu.StaticLayer, name))
    cache = StaticCache(config=model.config, max_cache_len=MAX_CACHE)
    return model, cache


def task_step(compiled, ple4=False, fullgraph=False):
    from torch._dynamo.utils import counters

    model, cache = build(ple4)
    ids = torch.tensor([prompt_ids()], device="tpu")
    tok = torch.zeros((1, 1), dtype=torch.long, device="tpu")

    def prefill():
        cache.reset()
        return model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1)

    def step():
        return model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1)

    if compiled:
        step = torch.compile(step, backend="tpu", dynamic=False, fullgraph=fullgraph)

    def decode(n, sync_each=False):
        with torch.no_grad():
            t0 = time.perf_counter()
            first = prefill()
            tok.copy_(first.view(1, 1))
            sync()
            t1 = time.perf_counter()
            toks = [first]
            for _ in range(n - 1):
                nxt = step()
                tok.copy_(nxt.view(1, 1))
                toks.append(nxt)
                if sync_each:
                    int(nxt)
            ids_out = torch.cat(toks).tolist()
            t2 = time.perf_counter()
        return ids_out, t1 - t0, t2 - t1

    counters.clear()
    runs = []
    for n, sync_each in ((32, False), (32, False), (128, False), (128, True)):
        toks, pre, dec = decode(n, sync_each)
        runs.append({
            "n": n, "sync_each": sync_each, "prefill_s": round(pre, 2), "decode_s": round(dec, 2),
            "decode_tok_s": round((n - 1) / dec, 2), "graphs_so_far": int(counters["stats"]["unique_graphs"]),
        })
        out[f"tokens{n}"] = toks
    out["runs"] = runs
    out["graph_breaks"] = int(sum(counters["graph_break"].values()))
    out["break_reasons"] = sorted(counters["graph_break"])[:8]
    out["recompiles"] = {k: v for k, v in counters["recompiles"].items()} if "recompiles" in counters else {}
    out["hbm_end_gb"] = hbm()
    try:
        out["hbm_peak_gb"] = round(torch.accelerator.max_memory_allocated() / 1e9, 3)
    except Exception as exc:
        out["hbm_peak_gb"] = repr(exc)[:100]


try:
    {
        "hbm_compressed": task_hbm_compressed,
        "generate": task_generate,
        "step_eager": lambda: task_step(False),
        "step_fuse": lambda: task_step(False),
        "step_fullgraph": lambda: task_step(True, fullgraph=True),
        "step_compiled": lambda: task_step(True),
        "step_compiled_ple4": lambda: task_step(True, ple4=True),
    }[TASK]()
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"[:6000]
    out["traceback"] = traceback.format_exc()[-4000:]
print("RESULT " + json.dumps(out))
''')

ENV = {"step_fuse": {"TORCH_TPU_DEFER_AND_FUSE": "1"}}


def run(task, timeout=2400):
    t = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, str(TASKS), task], env={**os.environ, **ENV.get(task, {})},
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"task": task, "error": f"timeout after {timeout} s"}
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not lines:
        return {"task": task, "error": f"rc {proc.returncode}", "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-4000:]}
    res = json.loads(lines[-1][7:])
    res["wall_s"] = round(time.perf_counter() - t, 1)
    return res


RESULTS = {}
for task in ("generate", "step_fullgraph", "step_compiled", "step_compiled_ple4", "step_eager", "step_fuse"):
    RESULTS[task] = run(task)
    r = {k: v for k, v in RESULTS[task].items() if not k.startswith("tokens")}
    print(json.dumps(r)[:6000], flush=True)
json.dump(RESULTS, open("probe_results.json", "w"))

# %%
ref = RESULTS["generate"].get("tokens128")
for task, r in RESULTS.items():
    for n in (32, 128):
        t, g = r.get(f"tokens{n}"), RESULTS["generate"].get(f"tokens{n}")
        if t and g:
            same = next((i for i, (a, b) in enumerate(zip(t, g)) if a != b), len(g))
            eot = t.index(106) if 106 in t else None
            print(f"{task:20s} n={n}: first {same} of {len(g)} identical to generate; first end-of-turn at {eot}")
print(json.dumps({k: {"tokens128": v.get("tokens128")} for k, v in RESULTS.items()}))
