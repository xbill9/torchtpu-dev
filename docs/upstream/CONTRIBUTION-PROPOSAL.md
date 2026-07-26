# Proposal: `examples/gemma4` for torch_tpu

**From:** xbill (TorchTPU GDE program) · 2026-07-26
**Stack used:** torch_tpu `0.1.1.dev20260725090141`, torch 2.11.0+cpu, transformers 5.14.1,
compressed-tensors 0.17.1, Python 3.12, single v6e-1 (ct6e-standard-1t, flex-start).

## Why this fills a gap

- The repo has `examples/gemma3` but no Gemma 4 example, and Gemma 4's text
  architecture differs in non-obvious ways (quirk list below) that cost real
  debugging time to rediscover.
- Quantization on TorchTPU is currently DIY (custom ops / Pallas kernels).
  Google's own QAT `w4a16-ct` (compressed-tensors int4, group-32) Gemma 4
  checkpoints have no TPU execution path elsewhere today: vLLM-TPU fails to
  load them (a weight-mapping problem — `k_norm.weight` reported uninitialized
  for layers 15–34; the checkpoints themselves load cleanly in transformers),
  and transformers' own `run_compressed=True` path graph-breaks on
  `Tensor.item()` and hits dynamic shapes the TPU backend rejects. This
  example is, to our knowledge, the first end-to-end w4a16-ct execution path
  on TPU — up to the 31B checkpoint on one v6e-1 chip.

## What exists (working today, in our private repo)

1. **Single-file dense Gemma 4 port** (`model.py`, ~1,200 lines, plain torch
   only) in the style of `examples/gemma3`: hand-rolled eager attention,
   explicit causal/sliding masks, HF-identical parameter names so checkpoints
   load via `load_state_dict` after a trivial key remap. It captures the
   Gemma4-vs-Gemma3 quirks: RMSNorm scales by `weight` (not `1 + weight`);
   two attention geometries (sliding: head_dim 256 / 16 KV heads; full:
   global_head_dim 512 / 4 KV heads); `attention_k_eq_v` — full-attention
   layers have no v_proj, V is the raw k_proj output through a scale-less
   v_norm; q/k per-head-dim RMSNorm before RoPE; attention scale = 1.0;
   per-layer-type RoPE (default θ=10k on sliding vs "proportional" θ=1M with
   partial_rotary_factor 0.25 zero-padded on full); RoPE applied in
   (B, S, H, D) layout; a persistent `layer_scalar` ones(1) buffer per layer;
   exclusive sliding-window mask (0 ≤ q−kv < window); √hidden embedding
   scale, tied lm_head, final-logit softcap 30.0.
2. **Static KV cache with prefill/decode split**: preallocated
   `[B, n_kv, MAX_SEQ, D]` buffers, `index_copy_` writes, position-tensor
   masks, no data-dependent control flow — `torch.compile(backend="tpu",
   dynamic=False)` compiles exactly one prefill graph and one decode graph.
   K cached post-k_norm/post-RoPE, V post-v_norm (documented choice).
3. **Quantized-Linear swap hooks**: every projection is a named `nn.Linear`
   subclass (QProjLinear, …) so quantized replacements swap in by class.
4. **w4a16 compressed-tensors execution path**, three interchangeable
   backends validated bit-exact against compressed-tensors' own decompression:
   `W4A16Linear` (in-graph shift/mask/scale dequant — no Pallas needed),
   `FusedW4A16Linear` (Pallas fused dequant-matmul via `pallas.jax_op`,
   nibble-plane layout), `Int8W4A16Linear` (load-time int4→int8, w8a16
   kernel). The 12B checkpoint ran through the cached port with **zero port
   changes** (328/328 Linears swapped).
5. **MoE experts module** (`W4A16Experts`): a TPU-compile-friendly
   replacement for `Gemma4TextExperts` (gemma-4-26B-A4B) — the HF
   data-dependent expert loop becomes a static `index_select` of packed int4
   expert slices + in-graph dequant + two `torch.bmm`s. Verified
   `fullgraph=True`, parity vs HF ~1e-8 (CPU mini-setting). Fit math: 26B-A4B
   experts at int4 g32 ≈ 11.4 GB + 1.4 GB scales → ~19–21 GB total on a
   32 GB v6e-1 (bf16 experts alone are 45.7 GB — impossible).

## Proposed example contents

`examples/gemma4/`: `model.py` (port + static KV cache + quant Linears),
`model_test.py` (see below), `quant_experts.py` + test (MoE), and a bench
script (`tpu_bench.py`) with prompts/batch/steps-per-call knobs. We would
adapt naming/style to whatever `examples/gemma3` conventions require.

## Test methodology (mirrors the `examples/gemma3` `model_test.py` pattern)

- **Layer-by-layer HF parity** (`parity_test.py`): mini config covering both
  attention geometries and the 5:1 sliding/full pattern, identical fuzzed
  weights into HF `Gemma4ForCausalLM` (eager) and the port, fp32 CPU; final
  logits within 1e-4 with per-layer localization on failure.
- **Cached-decode parity** (`decode_test.py`): greedy generation via
  prefill + decode_step must match the no-cache forward token-for-token and
  per-step logits within 1e-4; also batch=2, MAX_SEQ-padding invariance, and
  a `fullgraph=True` compile of `decode_step`.
- **Kernel numerics**: `torch_tpu._internal.utils.utils.assert_close` STRICT
  on all four extreme 12B/31B shapes × both production kernels — 8/8 PASS.
  Dequant bit-exact (max abs diff 0.0) vs compressed-tensors, CPU and TPU.
- **Quality**: 12B int8-swapped vs dequantized-bf16 target, nll/tok 6.86799
  vs 6.86737 over 10,200 scored tokens — a 0.009% delta.

## Headline results (all on one v6e-1; details in RESULTS.md)

| Model (QAT w4a16-ct) | Best config | b1 decode | b4 aggregate | Quant HBM vs bf16 |
|---|---|---:|---:|---|
| E2B | int4 in-graph | 157.5 tok/s | — | 1.06 / 3.75 GB |
| E4B | int4 in-graph | 72.4 tok/s | — | 2.23 / 7.95 GB |
| 12B (no cache) | int8 + Pallas kernel | 44.3 tok/s | 66.5 tok/s | 13.62 / 26 GB |
| **12B (cached port)** | int8 kernel + static KV cache | 36.8 tok/s | **217.4 tok/s** (18.4 ms/step; prefill 128 tok in 46 ms) | 13.62 / 26 GB |
| 31B | hybrid int8/int4, compact scales | **11.4 tok/s** | **24.0 tok/s** (all-fused int4) | 16.47 GB packed (bf16 58.57 GB — does not fit) |

The 31B row is the headline: the largest dense Gemma 4 QAT checkpoint is
single-chip-servable on TorchTPU. The 12B cached row shows the KV cache
turning a weight-bandwidth-bound decode into 3.3× the previous aggregate
record, with per-stream (54.4 tok/s) above even bf16 no-cache b1 (45.2).

**Measured vs. unmeasured.** All numbers above are measured (2026-07-25/26
sessions; 12B cached-port numbers are from session C). Not yet measured:
cached-port 31B (and E2B/E4B), cached long-context scaling (all cached runs
used MAX_SEQ=256 / 128-token prefill), in-graph batch-4 E4B, and the MoE
26B-A4B end-to-end on TPU (the experts module is CPU-mini-validated only).

## Ask

Feedback on fit and naming for `examples/`; whether the MoE module belongs in
the same example; and a pointer to the preferred contribution flow for the
private repo. We can also file the issues in FIELD-FEEDBACK.md separately.
