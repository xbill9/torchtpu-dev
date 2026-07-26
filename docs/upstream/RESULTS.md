# Gemma 4 QAT w4a16-ct on TorchTPU — benchmark book (condensed)

**Setup:** single v6e-1 (ct6e-standard-1t, 32 GB HBM), torch_tpu
`0.1.1.dev20260725090141`, torch 2.11.0+cpu, transformers 5.14.1,
compressed-tensors 0.17.1. Checkpoints: Google's ungated
`gemma-4-{E2B,E4B,12B,31B}-it-qat-w4a16-ct` (int4 symmetric, group-32).
All decode numbers: compiled (`backend="tpu", dynamic=False`), SEQ=256 loop,
greedy. "b4 aggregate" = 4 lockstep streams, summed tok/s.

## Final per-model best configs

| Model | Best config | b1 tok/s (ms/step) | b4 aggregate | Quant-Linear HBM | bf16 baseline (b1) |
|---|---|---:|---:|---:|---:|
| E2B | int4, in-graph dequant | 157.5 (6.3) | — | 1.06 GB (bf16 3.75) | 227.6 |
| E4B | int4, in-graph dequant | 72.4 | 118.9 (fused int4) | 2.23 GB (bf16 7.95) | 114.1 |
| 12B | int8-stored + w8a16 Pallas kernel | 44.3 (22.6) | 66.5 (60.2 ms) | 13.62 GB (bf16 26) | 45.2 (22.1) |
| **12B + static KV cache** | int8 kernel, cached port | 36.8 (27.1) | **217.4 (18.4 ms)** | 13.62 GB | — |
| 31B b1 | hybrid int8/int4, compact scales, 17 GB int8 budget (227/410 layers int8) | **11.4 (87.7)** | — | ~24.5 GB | bf16 does not fit (58.57 GB) |
| 31B b4 | all-fused packed int4 (rep4 scales), 64 MB scoped VMEM | 9.2 (109.0) | **24.0 (166.9 ms)** | 16.47 GB packed (+5.5 GB expanded scales) | — |

12B cached notes: compiled 128-token prefill = 46 ms; cached b1 sits below
no-cache b1 because MAX_SEQ=256 decode is weight-bandwidth-bound either way
and 1-row matmuls waste tiling; cached b4 reads weights once per step for 4
streams — 3.3× the no-cache aggregate, per-stream 54.4 > bf16 b1's 45.2.

## Cost model: bandwidth-bound vs VPU-bound

Kernel component shootout (12B extreme shapes, identical grid/tiling, 50-iter
means, ms):

| | gate 15360×3840 | down 3840×15360 |
|---|---:|---:|
| A fused int4 (production) | 0.415 | 0.352 |
| B int8-stored, plane-permuted | 0.306 | 0.272 |
| C bf16-in-Pallas control | 0.257 | 0.149 |
| D dense XLA matmul | 0.180 | 0.125 |
| nibble unpack+scale (A−C) | 0.158 (38%) | 0.202 (57%) |
| int8 convert+scale (B−C) | 0.049 | 0.122 |
| Pallas/tiling overhead (C−D) | 0.077 | 0.025 |

Four rules fall out, each confirmed at model level:

1. **In-graph dequant path → store int4.** XLA materializes the bf16 weight
   temp regardless, so the path is bandwidth-bound; int8-stored in-graph only
   adds read traffic (E2B: 116.5 vs int4's 157.5 tok/s; E4B: 61.8 vs 72.4).
2. **Pallas kernel path, batch 1 → store int8.** The fused-int4 kernel is
   VPU-bound on nibble unpack (38–57% above); load-time int4→int8 removes it.
   12B model level: 44.3 vs 28.3 tok/s (−2% vs bf16 at half the memory).
3. **Batch ≥ 4 → packed int4 wins.** Lockstep batch amortizes the per-step
   unpack (E2B bf16 b4: 1.98× aggregate at 2.0× step cost), so the path turns
   bandwidth-bound and fewest-bytes wins: 31B b4 all-fused int4 24.0 vs
   hybrid-v2 17.9 aggregate.
4. **Compact scales are a memory unlock, not a speed play.** Natural
   [O, K/32] scales (0.0625 B/elem vs rep4's 0.25) cost extra VPU (32 narrow
   lane-concats vs 8 wide): all-fused-compact loses at both batches on 31B
   (b1 116.3 vs 109.0 ms; b4 226.5 vs 166.9). But the freed 5.5 GB bought the
   int8 coverage behind the 11.4 tok/s 31B record.

Crossover between in-graph and fused kernel (single stream): small models
in-graph (E2B fused 112.9 < 157.5; E4B fused 54.6 < 72.4 — per-call custom-op
overhead × ~300 layers dominates), large models fused (12B 28.3 > 19.3;
31B 9.2 > 4.6 — bandwidth savings dominate).

Secondary findings: XLA's own matmul tiling beats our fixed blk=128/ck=960 by
20–40% on the tall shape (autotuning target); scoped-VMEM control experiment
and the batch-6/8 regression are in FIELD-FEEDBACK.md item 3; `steps_per_call`
chaining (8 decode steps per compiled call) changed nothing — decode is not
launch-bound; `xla_optimization_level=O3` changed nothing (227.6 → 226.9).

## Validation receipts

| Check | Result |
|---|---|
| Dequant vs compressed-tensors decompression | bit-exact, max abs diff 0.0 (CPU and TPU; E2B→31B) |
| Kernel numerics, `assert_close` STRICT (official harness) | 8/8 PASS — 4 extreme 12B/31B shapes × {fused-int4, int8-stored} |
| Single-layer fused kernel vs reference | ~2e-03 max rel (reduction order); model output text identical |
| HF parity (port, fp32 CPU, both attention geometries) | final logits within 1e-4, per-layer localization |
| Cached vs no-cache decode (port) | identical token sequences, per-step logits within 1e-4, incl. batch=2 and MAX_SEQ-padding invariance |
| Perplexity, 12B int8 vs dequantized-bf16 target | nll/tok 6.86799 vs 6.86737 (PPL 961.0 vs 960.4) — **0.009% delta**; 40 BOS-led 256-token windows, 10,200 scored tokens (Gemma windows must start with BOS: without it both score nll ~10.2) |
| Greedy outputs (all quantized paths) | text identical to the dequantized reference |
| xProf | 50-step trace of int8 12B b4 decode archived (110.7 ms/step under ~1.8× tracing overhead) |

Warmup incl. compile: 19.3 s (E2B bf16) → 57.1 s (E2B packed) → 77.4 s (31B
in-graph) / 71.5 s (31B fused); tier-3 GCS compile cache halves repeats
(84.3 → 36.6 s).

**Unmeasured cells:** cached-port 31B / E2B / E4B; cached long-context
(all cached runs at MAX_SEQ=256, 128-token prefill); in-graph b4 E4B;
MoE 26B-A4B end-to-end on TPU.
