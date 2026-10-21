# %%
import subprocess, sys, importlib.metadata as md, json, os, time
def v(p):
    try: return md.version(p)
    except md.PackageNotFoundError: return "-"
print("before:", {p: v(p) for p in ("torch", "transformers", "compressed-tensors", "accelerate", "safetensors", "huggingface_hub")})
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_tpu==0.2.0", "transformers>=5.5", "compressed-tensors",
                    "--extra-index-url", "https://download.pytorch.org/whl/cpu"], capture_output=True, text=True)
print("pip rc", r.returncode, r.stderr[-1500:])
print("after:", {p: v(p) for p in ("torch", "torch_tpu", "libtpu", "transformers", "compressed-tensors")})

# %%
from huggingface_hub import snapshot_download
t = time.time()
path = snapshot_download("xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct")
print("download s", round(time.time() - t), path)
print(subprocess.run(f"du -sh {path}/; ls {path}", shell=True, capture_output=True, text=True).stdout)

# %%
SCRIPT = r'''
import sys, time, json, os, torch, traceback
MODEL = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct"
way = sys.argv[1]
out = {"way": way}
def sync(): torch.accelerator.synchronize()
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    kw = {"dtype": torch.bfloat16}
    if way != "compressed":
        kw["quantization_config"] = transformers.CompressedTensorsConfig(run_compressed=False)
    t = time.time(); model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval(); out["load_cpu_s"] = time.time() - t
    out["model_class"] = type(model).__name__
    out["linear_types"] = sorted({type(m).__name__ for m in model.modules() if "Linear" in type(m).__name__})
    out["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    t = time.time(); model = model.to("tpu"); sync(); out["to_tpu_s"] = time.time() - t
    msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = {k: v.to("tpu") for k, v in ids.items()}
    n = ids["input_ids"].shape[1]; out["prompt_tokens"] = n
    with torch.no_grad():
        t = time.time(); logits = model(**ids).logits; sync(); out["forward1_s"] = time.time() - t
        out["top_next"] = tok.decode([int(logits[0, -1].argmax().cpu())])
        out["finite"] = bool(torch.isfinite(logits.float()).all().cpu())
        if way in ("eager", "fuse", "compressed"):
            t = time.time(); g = model.generate(**ids, max_new_tokens=8, do_sample=False); sync(); out["gen8_s"] = time.time() - t
            t = time.time(); g = model.generate(**ids, max_new_tokens=32, do_sample=False); sync(); out["gen32_s"] = time.time() - t
            out["text"] = tok.decode(g[0, n:], skip_special_tokens=True)
        elif way == "hfstatic":
            model.forward = torch.compile(model.forward, backend="tpu", dynamic=False)
            t = time.time(); g = model.generate(**ids, max_new_tokens=8, do_sample=False, cache_implementation="static"); sync(); out["gen8_s"] = time.time() - t
            t = time.time(); g = model.generate(**ids, max_new_tokens=32, do_sample=False, cache_implementation="static"); sync(); out["gen32_s"] = time.time() - t
            out["text"] = tok.decode(g[0, n:], skip_special_tokens=True)
        elif way == "buffer":
            SEQ = 256
            def step(tokens, mask, last):
                return model(input_ids=tokens, attention_mask=mask, use_cache=False).logits.index_select(1, last)
            cstep = torch.compile(step, backend="tpu", dynamic=False)
            one = torch.ones((1, 1), dtype=torch.long, device="tpu")
            def run(steps):
                tokens = torch.zeros((1, SEQ), dtype=torch.long); tokens[0, :n] = ids["input_ids"][0].cpu()
                mask = torch.zeros((1, SEQ), dtype=torch.long); mask[0, :n] = 1
                tokens, mask = tokens.to("tpu"), mask.to("tpu"); pos = torch.tensor([n], device="tpu")
                got = []
                for _ in range(steps):
                    nxt = cstep(tokens, mask, pos - 1).argmax(-1)[:, 0:1]
                    got.append(int(nxt.cpu().item()))
                    tokens = tokens.index_copy(1, pos, nxt.view(1, 1)); mask = mask.index_copy(1, pos, one); pos = pos + 1
                return got
            t = time.time(); run(8); out["gen8_s"] = time.time() - t
            t = time.time(); got = run(32); out["gen32_s"] = time.time() - t
            out["text"] = tok.decode(got, skip_special_tokens=True)
    out["hbm"] = {k: getattr(torch.accelerator, k)() for k in ("memory_allocated", "max_memory_allocated") if hasattr(torch.accelerator, k)} if hasattr(torch, "accelerator") else None
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"[:1500]; out["tb"] = traceback.format_exc()[-2500:]
print("RESULT " + json.dumps(out, default=str))
'''
open("probe_way.py", "w").write(SCRIPT)
ENV = {"eager": {}, "fuse": {"TORCH_TPU_DEFER_AND_FUSE": "1"}, "buffer": {}, "hfstatic": {}, "compressed": {}}
for way in ("eager", "fuse", "buffer", "hfstatic", "compressed"):
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "probe_way.py", way], env={**os.environ, **ENV[way]}, capture_output=True, text=True, timeout=1500)
        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        print(way, "rc", p.returncode, "wall", round(time.time() - t), "s")
        print(lines[-1][7:] if lines else p.stderr[-3000:])
    except subprocess.TimeoutExpired:
        print(way, "TIMEOUT after 1500 s")
