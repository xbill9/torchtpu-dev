# %%
import subprocess, sys, os, json, time, importlib.metadata as md
def v(p):
    try: return md.version(p)
    except md.PackageNotFoundError: return "-"
print("system:", {p: v(p) for p in ("jax", "jaxlib", "libtpu", "libtpu-nightly", "torch", "transformers", "numpy")})
MODEL = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
from huggingface_hub import snapshot_download
t = time.time(); PATH = snapshot_download(MODEL); print("download s", round(time.time() - t))

def run(code, env_extra=None, timeout=1500):
    env = {**os.environ, **(env_extra or {})}
    try:
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout}s"}
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    return json.loads(lines[-1][7:]) if lines else {"rc": p.returncode, "stderr": p.stderr[-2500:], "stdout": p.stdout[-800:]}

JAX_CHECK = r'''
import json, jax, jax.numpy as jnp
d = jax.devices()[0]
x = (jnp.ones((256, 256)) @ jnp.ones((256, 256))).sum()
print("RESULT " + json.dumps({"jax": jax.__version__, "device": d.device_kind, "sum": float(x)}))
'''
print("jax before:", run(JAX_CHECK))

# %%
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_tpu==0.2.0", "compressed-tensors",
                    "--extra-index-url", "https://download.pytorch.org/whl/cpu"], capture_output=True, text=True)
print("pip rc", r.returncode, r.stderr[-1500:])
print("after:", {p: v(p) for p in ("jax", "libtpu", "torch", "torch_tpu", "compressed-tensors", "transformers")})
TENV = {}
print("jax after torch_tpu install:", run(JAX_CHECK))
print("torch side:", run("import json, torch, torch_tpu; print('RESULT '+json.dumps({'sum': float((torch.ones(4,4,device='tpu')@torch.ones(4,4,device='tpu')).sum().cpu())}))"))
from transformers import AutoTokenizer
PROMPT_IDS = AutoTokenizer.from_pretrained(MODEL).apply_chat_template(
    [{"role": "user", "content": "Explain in two sentences why the sky is blue."}], add_generation_prompt=True, tokenize=True)
if hasattr(PROMPT_IDS, "keys"): PROMPT_IDS = PROMPT_IDS["input_ids"]
if PROMPT_IDS and isinstance(PROMPT_IDS[0], list): PROMPT_IDS = PROMPT_IDS[0]
json.dump(list(PROMPT_IDS), open("prompt_ids.json", "w")); print("prompt tokens", len(PROMPT_IDS))

# %%
TORCH = r'''
import sys, time, json, torch, traceback, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, way = sys.argv[1], sys.argv[2]
out = {"way": way}
def sync(): torch.accelerator.synchronize()
try:
    tok = AutoTokenizer.from_pretrained(MODEL)
    kw = {"dtype": torch.bfloat16, "key_mapping": {r"^model\.language_model\.": "model."}}
    if way != "compressed":
        kw["quantization_config"] = transformers.CompressedTensorsConfig(run_compressed=False)
    t = time.time(); model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval(); out["load_cpu_s"] = time.time() - t
    out["model_class"] = type(model).__name__
    out["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    t = time.time(); model = model.to("tpu"); sync(); out["to_tpu_s"] = time.time() - t
    ids = torch.tensor([json.load(open("prompt_ids.json"))], dtype=torch.long)
    n = ids.shape[1]; out["prompt_ids"] = ids[0].tolist()
    x = ids.to("tpu")
    with torch.no_grad():
        if way in ("eager", "compressed"):
            for k in (8, 32):
                t = time.time(); g = model.generate(x, max_new_tokens=k, min_new_tokens=k, do_sample=False); sync(); out[f"gen{k}_s"] = time.time() - t
            out["tokens"] = g[0, n:].tolist()
        elif way == "hfstatic":
            model.forward = torch.compile(model.forward, backend="tpu", dynamic=False)
            for k in (8, 32):
                t = time.time(); g = model.generate(x, max_new_tokens=k, min_new_tokens=k, do_sample=False, cache_implementation="static"); sync(); out[f"gen{k}_s"] = time.time() - t
            out["tokens"] = g[0, n:].tolist()
        elif way == "buffer":
            SEQ = 256
            def step(tokens, mask, last):
                return model(input_ids=tokens, attention_mask=mask, use_cache=False).logits.index_select(1, last)
            cstep = torch.compile(step, backend="tpu", dynamic=False)
            one = torch.ones((1, 1), dtype=torch.long).to("tpu")
            def gen(k):
                tokens = torch.zeros((1, SEQ), dtype=torch.long); tokens[0, :n] = ids[0]
                mask = torch.zeros((1, SEQ), dtype=torch.long); mask[0, :n] = 1
                tokens, mask = tokens.to("tpu"), mask.to("tpu"); pos = torch.tensor([n]).to("tpu"); got = []
                for _ in range(k):
                    nxt = cstep(tokens, mask, pos - 1).argmax(-1)[:, 0:1]
                    got.append(int(nxt.cpu().item()))
                    tokens = tokens.index_copy(1, pos, nxt.view(1, 1)); mask = mask.index_copy(1, pos, one); pos = pos + 1
                return got
            for k in (8, 32):
                t = time.time(); got = gen(k); out[f"gen{k}_s"] = time.time() - t
            out["tokens"] = got
    out["text"] = tok.decode(out.get("tokens", []), skip_special_tokens=True)
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"[:1200]; out["tb"] = traceback.format_exc()[-1500:]
print("RESULT " + json.dumps(out))
'''
open("torch_way.py", "w").write(TORCH)
results = {}
for way in ("eager", "buffer", "hfstatic", "compressed"):
    t = time.time()
    p = subprocess.run([sys.executable, "torch_way.py", MODEL, way], env={**os.environ, **TENV}, capture_output=True, text=True, timeout=1800)
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    results[way] = json.loads(lines[-1][7:]) if lines else {"rc": p.returncode, "stderr": p.stderr[-2500:]}
    r = dict(results[way]); r.pop("prompt_ids", None)
    print(way, "wall", round(time.time() - t), "s", json.dumps(r)[:2500])

# %%
ENG = "/content/tpu-jax" if os.path.isdir("/content") else os.path.abspath("tpu-jax")
if not os.path.isdir(ENG):
    subprocess.run(["git", "clone", "-q", "https://github.com/xbill9/tpu-jax", ENG], check=True)
subprocess.run(["git", "-C", ENG, "checkout", "-q", "4b9f8e94619f1995206617e07e3c4ac62685d47b"], check=True)
JAXRUN = r'''
import sys, json, time, traceback
sys.path.insert(0, sys.argv[1])
out = {}
try:
    import jax
    from jax_engine import JaxGemmaEngine
    ids = json.load(open("prompt_ids.json"))
    eng = JaxGemmaEngine(sys.argv[2], quant_mode="w4a16")
    t = time.time(); eng.load(local_dir=sys.argv[3]); out["load_s"] = time.time() - t
    out["weight_gb"] = eng.weight_bytes / 1e9
    t = time.time(); toks, st = eng.generate(ids, max_new_tokens=32); out["first_gen32_s"] = time.time() - t
    runs = [eng.generate(ids, max_new_tokens=32) for _ in range(3)]
    out["decode_tok_s"] = [s.decode_tok_per_s for _, s in runs]
    out["prefill_ms"] = [s.prefill_ms for _, s in runs]
    out["tokens"] = runs[0][0]
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"[:1200]; out["tb"] = traceback.format_exc()[-1500:]
print("RESULT " + json.dumps(out))
'''
p = subprocess.run([sys.executable, "-c", JAXRUN, ENG, MODEL, PATH], capture_output=True, text=True, timeout=1800)
lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
jr = json.loads(lines[-1][7:]) if lines else {"rc": p.returncode, "stderr": p.stderr[-3000:]}
print("jax engine:", json.dumps(jr)[:2500])
for way, r in results.items():
    if "tokens" in r and "tokens" in jr:
        a, b = r["tokens"], jr["tokens"]
        same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"{way}: first {same} tokens identical to the JAX engine's")
