import glob, sys, torch
side, out = sys.argv[1], sys.argv[2]
M = glob.glob("/home/xbill/.cache/huggingface/hub/models--xbill9--gemma-4-E2B-it-qat-q4_0-w4a16-ct-text/snapshots/*/")[0]
E = glob.glob("/home/xbill/.cache/huggingface/hub/models--xbill9--gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4/snapshots/*/")[0]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(M)
enc = tok.apply_chat_template([{"role": "user", "content": "Why is the sky blue? Explain in one paragraph."}], add_generation_prompt=True, tokenize=True)
ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
if side == "rust":
    sys.path.insert(0, sys.argv[3]); import torchtpu_rs as rs
    eng = rs.Engine(M, E, torch.zeros(1), 1024)
    logits = eng.logits(ids).float(); tokens, _, _ = eng.generate(ids, 64, 1); meta = dict(rs.BUILD)
else:
    from transformers import AutoModelForCausalLM
    import transformers
    hf = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, key_mapping={r"^model\.language_model\.": "model."}, quantization_config=transformers.CompressedTensorsConfig(run_compressed=False)).eval()
    with torch.no_grad():
        x = torch.tensor([ids]); logits = hf(x).logits[0, -1].float()
        tokens = hf.generate(x, max_new_tokens=64, do_sample=False)[0, len(ids):].tolist()
    import transformers; meta = {"torch": torch.__version__, "transformers": transformers.__version__}
torch.save({"logits": logits, "tokens": tokens, "meta": meta}, out)
print(side, meta, len(tokens))
