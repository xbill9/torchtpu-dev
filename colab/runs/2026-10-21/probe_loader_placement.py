# %%
import subprocess, sys, os, json, time
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_tpu==0.2.0", "compressed-tensors",
                    "--extra-index-url", "https://download.pytorch.org/whl/cpu"], capture_output=True, text=True)
print("pip rc", r.returncode, r.stderr[-800:])
MODEL = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
from huggingface_hub import snapshot_download
snapshot_download(MODEL)
from transformers import AutoTokenizer
ids = AutoTokenizer.from_pretrained(MODEL).apply_chat_template(
    [{"role": "user", "content": "Explain in two sentences why the sky is blue."}], add_generation_prompt=True, tokenize=True)
if hasattr(ids, "keys"): ids = ids["input_ids"]
if ids and isinstance(ids[0], list): ids = ids[0]
json.dump(list(ids), open("prompt_ids.json", "w"))

# %%
CODE = r"""
import json, sys, time, traceback, collections, torch, transformers
from transformers import AutoModelForCausalLM
MODEL = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
how = sys.argv[1]
out = {"how": how}
def sync(): torch.accelerator.synchronize()
try:
    kw = dict(dtype=torch.bfloat16, key_mapping={r"^model\.language_model\.": "model."},
              quantization_config=transformers.CompressedTensorsConfig(run_compressed=False))
    t = time.time()
    if how == "device_map":
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="tpu", **kw).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval().to("tpu")
    sync(); out["load_s"] = time.time() - t
    by_dev = collections.Counter(); by_dev_bytes = collections.Counter(); cpu_names = []
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        by_dev[str(p.device)] += 1; by_dev_bytes[str(p.device)] += p.numel() * p.element_size()
        if p.device.type == "cpu": cpu_names.append(name)
    out["tensors_by_device"] = dict(by_dev); out["gb_by_device"] = {k: v / 1e9 for k, v in by_dev_bytes.items()}
    out["cpu_tensors_sample"] = cpu_names[:8]
    out["hbm_allocated_gb"] = torch.accelerator.memory_allocated() / 1e9 if hasattr(torch.accelerator, "memory_allocated") else None
    ids = json.load(open("prompt_ids.json")); n = len(ids)
    x = torch.tensor([ids], dtype=torch.long).to("tpu")
    with torch.no_grad():
        for k in (8, 32):
            t = time.time(); g = model.generate(x, max_new_tokens=k, min_new_tokens=k, do_sample=False); sync(); out[f"eager_gen{k}_s"] = time.time() - t
        out["eager_tokens"] = g[0, n:].tolist()
        SEQ = 256
        def step(tokens, mask, last):
            return model(input_ids=tokens, attention_mask=mask, use_cache=False).logits.index_select(1, last)
        cstep = torch.compile(step, backend="tpu", dynamic=False)
        one = torch.ones((1, 1), dtype=torch.long).to("tpu")
        def gen(k):
            tokens = torch.zeros((1, SEQ), dtype=torch.long); tokens[0, :n] = torch.tensor(ids)
            mask = torch.zeros((1, SEQ), dtype=torch.long); mask[0, :n] = 1
            tokens, mask = tokens.to("tpu"), mask.to("tpu"); pos = torch.tensor([n]).to("tpu"); got = []
            for _ in range(k):
                nxt = cstep(tokens, mask, pos - 1).argmax(-1)[:, 0:1]
                got.append(int(nxt.cpu().item()))
                tokens = tokens.index_copy(1, pos, nxt.view(1, 1)); mask = mask.index_copy(1, pos, one); pos = pos + 1
            return got
        try:
            for k in (8, 32):
                t = time.time(); got = gen(k); out[f"buffer_gen{k}_s"] = time.time() - t
            out["buffer_tokens"] = got
        except Exception as e:
            out["buffer_error"] = f"{type(e).__name__}: {e}"[:600]
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"[:1500]; out["tb"] = traceback.format_exc()[-1500:]
print("RESULT " + json.dumps(out))
"""
JAX_TOKENS = [818, 7217, 7412, 3730, 1547, 529, 496, 20284, 2760, 121707, 19389, 236764, 1298, 506, 10824, 236789, 236751, 11661, 141891, 1826, 20532, 236764, 3730, 57583, 529, 26808, 919, 11974, 1082, 4890, 236764, 2604]
for how in ("to_tpu", "device_map"):
    p = subprocess.run([sys.executable, "-c", CODE, how], capture_output=True, text=True, timeout=2400)
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    res = json.loads(lines[-1][7:]) if lines else {"rc": p.returncode, "stderr": p.stderr[-3000:]}
    print(how, json.dumps({k: v for k, v in res.items() if not k.endswith("tokens")})[:2500])
    for key in ("eager_tokens", "buffer_tokens"):
        if key in res:
            a = res[key]; same = next((i for i, (x, y) in enumerate(zip(a, JAX_TOKENS)) if x != y), min(len(a), 32))
            print(f"  {key}: first {same} of 32 identical to the JAX engine's")
