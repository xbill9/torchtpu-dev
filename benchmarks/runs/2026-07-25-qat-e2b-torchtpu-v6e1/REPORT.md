# Gemma 4 E2B QAT checkpoints on TorchTPU — steps 1, 2 & 3

**Run:** 2026-07-25 (evening session) · `torchtpu-vm` (ct6e-standard-1t, flex-start, europe-west4-a, project `comglitn`)
**Goal:** validate Google's QAT checkpoints on TorchTPU (vLLM-on-TPU cannot load them — see SKILL.md known-broken note). Steps 1–2 of the plan in project memory; step 3 (packed int4 Pallas kernel) is next.
**Stack:** torch_tpu `0.1.1.dev20260725090141`, torch 2.11.0+cpu, transformers 5.14.1, compressed-tensors 0.17.1, Python 3.12.

## Results

| Checkpoint | Load path | Eager | Compiled (SEQ=256 loop) | Output quality |
| :--- | :--- | ---: | ---: | :--- |
| `gemma-4-E2B-it-qat-q4_0-unquantized` | plain transformers | 3.1 tok/s | **227.6 tok/s** (4.4 ms/step) | coherent |
| `gemma-4-E2B-it-qat-w4a16-ct` | `CompressedTensorsConfig(run_compressed=False)` | 3.3 tok/s | — (see finding 2) | coherent |
| stock `gemma-4-E2B-it` (baseline, earlier run) | plain transformers | 3.4 tok/s | 227.3 tok/s | — |

Both QAT checkpoints are **ungated** on HF. Compiled throughput with QAT weights is
identical to stock — QAT costs nothing at bf16 execution, as expected.

## Findings

1. **The unquantized QAT export loads cleanly in transformers** — no missing-weight
   warnings. vLLM's "k_norm.weight not initialized (layers 15–34)" failure is a
   vLLM weight-mapping problem, not a checkpoint defect.
2. **`w4a16-ct` with default `run_compressed=True` cannot be `torch.compile`d on TPU:**
   the on-the-fly decompression graph-breaks on `Tensor.item()` and then the TPU
   backend rejects the resulting dynamic shapes (`does not support dynamic shape`).
   Compressed execution on TPU therefore requires the step-3 custom kernel.
3. **Dequantized load works end to end:** `run_compressed=False` decompresses the
   int4/group-32 weights to bf16 at load (~2 s for 276 modules), `.to("tpu")` is
   NOT blocked by the quantizer, and generation matches the unquantized QAT model.
   Since the checkpoint is w4a16 (activations stay 16-bit), this dequantized
   forward is numerically the target quantized model.
4. Same infra papercuts as the morning run: startup pip 403s on the wheel registry
   (fixed over IAP SSH with a user token); MCP SSH tools can't reach the VM
   (no IAP) — `verify_pytorch_tpu` and friends need manual `--tunnel-through-iap`
   equivalents on this network.

## Step 3: true packed int4 execution (`w4a16_packed_bench.py`)

No Pallas needed for v1: `W4A16Linear` keeps `weight_packed` (int32) + group-32
scales in HBM and dequantizes inside the compiled forward with traceable ops only
— `(packed >> 4*i) & 0xF`, `- 8`, `* scale` (arithmetic shift is safe: the mask
kills sign-extension bits). All 276 CompressedLinears swapped; static shapes, no
graph breaks, single compile.

| Metric | Packed w4a16 | bf16 baseline |
| :--- | ---: | ---: |
| Quantized-Linear weight bytes in HBM | **1.06 GB** | 3.75 GB (3.56×) |
| Compiled decode (SEQ=256 loop) | **157.5 tok/s** (6.3 ms/step) | 227.6 tok/s (4.4 ms/step) |
| Warmup incl. compile | 57.1 s | 19.3 s |
| Dequant vs compressed-tensors decompress | **bit-exact (0.0)** CPU and TPU | — |
| Greedy output | identical text to dequantized reference | — |

The ~1.9 ms/step overhead is the per-forward dequant (XLA materializes the bf16
weight per layer; unpack runs on the VPU). On E2B this is a memory win, not a
speed win.

### Same script across the QAT family (all ungated, all bit-exact)

| Model | Quantized Linears packed | vs bf16 | Packed decode | bf16 decode |
| :--- | ---: | ---: | ---: | ---: |
| E2B (276 modules) | 1.06 GB | 3.75 GB | 157.5 tok/s | 227.6 |
| E4B (343 modules) | 2.23 GB | 7.95 GB | 72.4 tok/s | 114.1 |
| 12B (328 modules) | 6.13 GB | 21.80 GB | 19.3 tok/s | 45.2 |

All 3.56× smaller; 12B drops from ~26 GB total HBM (barely fit) to ~10 GB. The +29.7 ms/step overhead is consistent with
the dequantized bf16 weights (21.8 GB) round-tripping through HBM each step
(~2 × 21.8 GB ÷ ~1.6 TB/s ≈ 27 ms) — i.e. the cost is exactly the
materialization traffic a **fused Pallas dequant-matmul** (weights never leave
HBM as bf16) would eliminate. That kernel is the clear next optimization.
Meanwhile the packed path frees ~16 GB of HBM on 12B — room for longer buffers,
lockstep batch, or a KV cache. Roadmap: E4B next, maybe the ~27B MoE (which
bf16 cannot fit on one v6e-1 at all — packed int4 is the only way it fits).

Logs: `logs/w4a16_run.log` (E2B), `logs/w4a16_12b.log` (12B).

## Step 3b: fused Pallas dequant-matmul kernel (`w4a16_fused_model_bench.py`)

`pallas.jax_op` (torch_tpu's JAX/Pallas bridge; auto-registers a
torch.library custom op with fake impls, composes with
`torch.compile(backend="tpu")`; requires `pip install jax`, type-annotated
JAX fn). Kernel: grid over 256-row weight blocks, full-K blocks, in-kernel
K chunking (~1024), fp32 accumulator. The make-or-break detail is
**nibble-plane layout**: naive `[K/8, 8] → [K]` unpack is a cross-lane
interleave that Mosaic lowers slowly (2.8 ms/layer) or rejects ("Unsupported
reshape"). Instead each of the 8 shift/mask planes stays elementwise and
planes are concatenated; the *activation* columns are permuted to match
(gather on [256, K] — negligible), and group-32 scales become
plane-independent (`scale[m//4]`, host-side `repeat_interleave(4)` +
in-kernel tile). Also learned: Pallas TPU block last-dims must be ×128 or
full-size; scoped VMEM limit is 32 MB (raise via
`LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536`).

| 12B decode (SEQ=256 loop) | ms/step | tok/s |
| :--- | ---: | ---: |
| bf16 dense | 22.1 | 45.2 |
| packed, in-graph dequant | 51.8 | 19.3 |
| **packed, fused Pallas kernel** | **35.4** | **28.3** (328/328 layers fused) |

E2B fused: 112.9 tok/s (8.9 ms/step) — **slower than in-graph** (157.5).
The fused custom call carries a fixed per-invocation cost (~276–328 calls
per step); on E2B's small layers that overhead exceeds the bandwidth
savings, on 12B's big layers the savings win. **Crossover rule: in-graph
dequant for small models, fused kernel for big ones.** Single-layer
numerics vs reference: ~2e-03 max rel (reduction order); model output text
identical to the in-graph packed run for both models.

**Remaining gap analysis:** ~11 G weights × ~5 VPU ops (shift/and/sub/
convert/mul) ≈ 14 ms/step of unpack on 12B — matches the 13 ms gap to bf16.
The kernel is VPU-bound on nibble extraction, not bandwidth-bound.

**Escapes, with the first already probed** (`int4_bitcast_probe.py`,
`logs/int4_probe.log`):

- ❌ **int4 bitcast — DEAD END on current Mosaic.**
  `lax.bitcast_convert_type(packed ^ 0x88888888, int4)` (XOR flips the
  offset-8 nibble to two's-complement int4) fails to compile:
  *"Changing bitwidths not supported."* Re-probe on future torch_tpu
  nightlies; do not re-derive.
- **Lockstep batch (next best, guaranteed):** unpack cost is per-step, not
  per-stream — batch 4 amortizes it 4×. Needs an S-dimension grid tile in
  the kernel (x [1024, 15360] bf16 = 31 MB no longer fits VMEM whole).
- **int8 middle point:** store weights as int8 (dequant int4→int8 at load):
  no unpack at all, just native convert+mul (~2 VPU ops); 2× memory of
  packed int4 but still 2× under bf16 (~13 GB for 12B) — likely near-parity
  speed. Worth benching against batch amortization.

Even as-is: 28.3 tok/s at ~10 GB HBM beats 19.3, and the freed ~16 GB
enables the KV-cache/batch work.

## Step 4 — zero-code-change sweep (2026-07-26, same VM + fresh vm2)

All three doc-derived free knobs measured on E2B qat-unquantized. **None
helps; the wins are structural.**

| Experiment | Result | Baseline | Verdict |
|---|---|---|---|
| `TPU_DEFER_AND_FUSE=1` on eager HF generate | 0.5 tok/s (10,398 compile reqs / 55 tok, 84% hits) | 3.4 tok/s plain eager | ❌ 7× WORSE — generate's shapes change every step; each fused group is a fresh fingerprint. Static-shape eager chains only. |
| `--seq 512` compiled decode (fused-SDPA attempt) | 194.2 tok/s (5.1 ms/step) | 227.6 @ SEQ=256 | ❌ Fused FLASH_ATTENTION still skipped — failing condition is **`attn_mask is None (current: present)`**, not seq length. HF forward always passes a mask ⇒ fused SDPA is unreachable from stock HF models. Mask-free `is_causal` attention goes on the gemma4-port feature list. |
| `TORCH_TPU_INTERNAL_XLA_OPTIONS=xla_optimization_level=O3` | 226.9 tok/s (4.4 ms/step) | 227.6 | ➖ No change; O2 default already saturates this graph. |

**Startup template first live boot (torchtpu-vm2):** whole startup **1 m 35 s**
(vs ~10 min AR-path), mirror install + extras + smoke test green, and the
tier-3 compile cache confirmed writing `.bin` entries to
`gs://comglitn-torchtpu-wheels/compile-cache/8a5b7822779ee13a/` from an SSH
workload. (Startup smoke test itself runs cache-cold — env vars are written
to /etc/environment after it.)

## Step 5 — lockstep scaling + the 31B single-chip fit (2026-07-26)

**Lockstep batch=4, E2B bf16 (SEQ=256):** 450.5 tok/s aggregate at
8.9 ms/step vs 227.6 tok/s / 4.4 ms at batch 1 — 4 streams for 2.0× the
step cost (1.98× aggregate). Confirms the amortization thesis: per-step,
batch-independent costs (the fused kernel's ~14 ms/step unpack on 12B)
divide by the stream count.

**gemma-4-31B-it-qat-w4a16-ct FITS AND RUNS on one v6e-1** (fresh vm2,
in-graph dequant W4A16Linear, first try):

| metric | value |
|---|---|
| Linears swapped | 410 |
| packed+scale HBM | **16.47 GB** (bf16 would be 58.57 GB — impossible on a 32 GB chip) |
| dequant parity | max abs diff **0.0** vs compressed-tensors reference (CPU and TPU) |
| warmup (incl. compile) | 77.4 s |
| compiled decode | 4.6 tok/s (217.6 ms/step) |

In-graph dequant is the wrong side of the crossover at this size (it
re-materializes ~29 G weights in bf16 every step); the fused Pallas kernel
run is the follow-up. The headline stands regardless: the biggest dense
Gemma 4 QAT checkpoint is single-chip-servable on TorchTPU.

**Fused Pallas kernel on 31B (same night, vm2):** 410/410 layers fused,
warmup 71.5 s, **9.2 tok/s (109.0 ms/step)** — 2.0× over in-graph (217.6 ms).
Output text coherent. Consistent with the VPU-bound model: ~29 G weights /
11 G × the 12B kernel's ~35 ms ≈ 93 ms predicted, 109 observed. With
lockstep batch 4 (unpack amortizes, per the 1.98×-at-2×-cost E2B data)
a ~25-35 tok/s aggregate 31B single-chip server is the projected next stop.

**Fused 31B + lockstep batch 4 (host-loop patch, kernel unchanged):
24.0 tok/s aggregate** (6.0/stream, 166.9 ms/step, 64 MB scoped VMEM).
4 streams for 1.53× the step cost = 2.6× aggregate over single-stream
fused (9.2). Cost model from the two points: ~90 ms fixed (unpack +
per-call overhead) + ~19 ms/stream → batch 8 ≈ 33 tok/s if the down_proj
activation block (~88 MB at batch 8) clears the 96 MB scoped-VMEM notch.
The bench script now takes batch as argv[2].
