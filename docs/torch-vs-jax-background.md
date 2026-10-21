# PyTorch vs JAX on TPU: background for the article

Research notes, gathered 2026-10-21. Every fact carries its source: a URL for published material, a file path and line for measurements on this machine. Numbers are quoted as written; where a source states a ratio, the ratio is quoted, and none is computed here. Marketing language from vendor sources is marked **[vendor]**.

Short names for published sources:

| Name | URL |
| --- | --- |
| Intro | https://torchtpu.readthedocs.io/en/latest/intro_to_torchtpu/ |
| How | https://torchtpu.readthedocs.io/en/latest/how_torchtpu_works/ |
| Exec | https://torchtpu.readthedocs.io/en/latest/torchtpu_execution_model/ |
| Dyn | https://torchtpu.readthedocs.io/en/latest/dynamic_shapes/ |
| Kern | https://torchtpu.readthedocs.io/en/latest/custom_kernels/ |
| Feat | https://torchtpu.readthedocs.io/en/latest/feature_set/ |
| FAQ | https://torchtpu.readthedocs.io/en/latest/faq/ |
| PTB | https://pytorch.org/blog/introducing-torchtpu/ |
| ARX | https://arxiv.org/abs/2610.24161 ("TorchTPU - A Hybrid Eager/Compiled Architecture for Native PyTorch Execution on TPU", v2, 2026-10-20) |
| TTB | https://developers.googleblog.com/integrating-torchtpu-into-torchtitan-and-optimizing-models-for-tpu/ |
| VLB | https://developers.googleblog.com/announcing-new-vllm-and-sglang-inference-engines-on-tpu/ |
| GCB | https://cloud.google.com/blog/topics/ai-infrastructure/torchtpu-is-now-open-source-run-native-pytorch-on-tpus |

Local roots: `G` = `~/gemma4-dev`, `CJ` = `~/colab-jax`, `T` = `~/torchtpu-dev`.

---

#### 1. Where TorchTPU sits

- **It replaces PyTorch/XLA.** FAQ: "PyTorch/XLA is deprecated and replaced by TorchTPU." PyTorch/XLA "historically served as a bridging framework, heavily dependent on JAX runtime components"; TorchTPU offers "zero JAX runtime dependencies".
- **The vendor's account of what was wrong with PyTorch/XLA.** PTB/ARX: it "required intrusive source-code modifications to explicitly define compilable regions, suffered from severe compile-time latency, lacked support for PyTorch's native imperative ('eager') execution mode". TTB: before TorchTPU, developers had to "undertake a complete codebase rewrite into JAX, or grapple with legacy torch_xla". **[vendor]**
- **Positioned beside JAX.** FAQ, "Is TorchTPU replacing JAX?": "No… complement the TPU ecosystem". VLB: "vLLM-Jax, and SGLang-JAX remain important projects. Both will continue to be maintained, supported". VLB on why the PyTorch engines exist: developers "wanted to customize and debug their serving stack in PyTorch".
- **Mechanism.** How: an "out of tree backend… registered as PrivateUse1", autoloaded and renamed `tpu`. It "traces eager ops at the ATen level… constructed as StableHLO programs, and passed to XLA". `torch.compile` "generates and compiles a single StableHLO program for the entire FX graph… TorchTPU does not use TorchInductor, instead reusing the mature XLA compiler stack".
- **Both stacks lower to the same compiler.** JAX docs (https://docs.jax.dev/en/latest/export/shape_poly.html): "a function will be traced, lowered to StableHLO, and compiled for each combination of input types and shapes."

#### 2. Programming model

**PyTorch on TorchTPU is eager first.** Intro: "Your unmodified PyTorch model runs immediately in eager mode." Three eager modes (Exec):

- **Strict:** the default, "one ATen operator at a time asynchronously", compiled at "-O1… to maximize compile speed".
- **Debug:** `TORCH_TPU_LAUNCH_BLOCKING=1`.
- **Defer-and-fuse:** `TORCH_TPU_DEFER_AND_FUSE=1`, compiled at "-O2". It "shows up to 50% performance improvement over eager default, with no change to model code". PTB: "can be 50% or more faster than Strict Eager while reducing peak HBM memory".

`torch.compile` is the fourth option.

**JAX is traced first.** JAX docs (https://docs.jax.dev/en/latest/control-flow.html): "Python control flow and logical operators are evaluated at JIT compile time… the function is re-compiled every time it is called on an input with a new shape or dtype."

**Both pay for a host read inside the hot path.**

- TorchTPU (Exec): "if my_tensor.all()" will "synchronize the stream, and copy that boolean to the CPU", and in fused or compiled mode "this forced sync point will force a graph break".
- JAX (https://docs.jax.dev/en/latest/jit-compilation.html): "Traced values within JIT… can only affect control flow via their static attributes".

**Dynamic shapes are where the two differ most for an LLM decode loop.**

- TorchTPU (Dyn) offers "Bounded Dynamism, where the compile-time upper bounds of dynamic dimensions are fixed", with 128-element buckets and "num_dynamic_buckets (default 1, maximum 8)". Its limitations: "Unbounded dynamism is not supported yet"; data-dependent dynamism "(such as torch.nonzero) are not supported"; "dynamic=True mode is not supported"; bounded executables "are recompiled on each new process start".
- The TorchTPU sources disagree on what is ready.
  - Intro: "torch.compile sets dynamic=False by default on TorchTPU. dynamic=None or dynamic=True are not yet supported for production… decode/generate step on Hugging Face LLMs will involve recompilations at each step unless bounded dynamism is enabled."
  - PTB: "bounded dynamism… is not enabled by default yet."
  - Dyn uses `dynamic=None` as the bounded-dynamism path, with a decode example that recompiles every step under `dynamic=False` (about 38 to 49 ms per step). Under `dynamic=None`, "from step 4 onward execution drops to about 0.4 ms per step (~100x faster)".
  - The article should say the docs disagree.
- PTB/ARX checked bounded dynamism on "839 models… correct numerics on 834 (99%)".

#### 3. Published performance against JAX

All are vendor-reported. All are on chips other than v5e, and none uses a Gemma model.

| Workload | Hardware | PyTorch / TorchTPU | JAX | Source |
| --- | --- | --- | --- | --- |
| Qwen3 MoE 235B | 64 TPU v7 | "1828 TPS for PyTorch/TorchTitan" | "1937 TPS for JAX/MaxText" | PTB/ARX, "an initial smoke test" |
| FLUX.1 | TPU v6e-8 | "2.78 s… 4.1x faster than the starting point" | "about 99% of the throughput of the JAX-based MaxDiffusion on the same hardware" | TTB |
| DLRM v2 | 32-chip TPU v7x | "identical 18.3ms step latency" | "100% performance parity with Google's highly optimized JAX baseline" | ARX |

Also published:

- PTB lists "with similar performance to JAX compile" as a design principle, which makes it a goal rather than a measurement.
- PTB, pure PyTorch with no custom kernels: Muse Glimmer 30B prefill on one TPUv7 chip, "~25.7k TPS with 59% MFU". PTB adds: "for production workloads seeking peak performance we would absolutely recommend… Pallas kernels".

#### 4. Kernels, quantization, distribution

- **Custom kernels go through JAX either way.** Kern: Pallas via `torch.tpu.pallas.jax_op` is "Production Ready… Writing bespoke, maximum-performance TPU kernels in JAX/Pallas and calling them from PyTorch"; the example imports `jax.experimental.pallas`. Tokamax is "Production Ready"; Helion is "Experimental". JAX docs (https://docs.jax.dev/en/latest/pallas/index.html): "Pallas is under active development."
- **Quantization on TorchTPU itself.** Feat, validated: "Full tensor FP8/FP4" and "Per-channel FP8". No INT8 or INT4 entry for TorchTPU itself. VLB lists "FP8, FP4 (MXFP4, NVFP4), INT8 and INT4 checkpoints" for the vLLM and SGLang TorchTPU engines specifically.
- **Process model.** PTB/ARX: "we adopt a one-process-per-device model in contrast with JAX's one-process-per-host model… RL colocation can't be supported out of the box". Intro: "Only one process can access a TPU device."
- **Distribution style.** PTB/ARX: "distributed JAX programs are written for a single host, with no communication code included… the XLA compiler is fully responsible for inserting collective ops… the PyTorch ecosystem doesn't adopt a clean SPMD approach". TorchTPU supports "DDP, FSDP2 and SimpleFSDP, and DTensor" via `backend="tpu_dist"` (Intro).

#### 5. Ecosystem and maturity

- **Model coverage.** Intro: "450 core ATen operators… continuously validated across 5,000 popular HuggingFace models". PTB: no-crash "98% – 4,821 out of 4,917"; numerical parity "97% – 638 out of 657 (w.r.t. GPU)".
- **TPU generations.** FAQ: "TorchTPU actively supports v5e, v5p, Trillium (TPU v6e) and Ironwood (TPU v7x)".
- **Release.** `torch_tpu` 0.2.0 on PyPI (the releases listed are 0.0.0 and 0.2.0).
  - The wheel's own metadata says `Requires-Python: >=3.12` and `torch>=2.11.0,<=2.15.0` (read from the cp312 wheel, 2026-10-21). The README says "Python 3.11+".
  - Feat splits features into "Validated" and "Experimental"; "Colab" is in the validated row.
- **Known gaps (Intro).** "does not currently have native FlexAttention support".

#### 6. Measured on this machine

**Same chip, both stacks, same day: none exists yet.** That is the gap notebook 03 fills.

**A. TorchTPU on a Colab v5e-1, 2026-10-21.**

Execution modes, synthetic tensors, two runs (`T/colab/runs/2026-10-21/01_torchtpu_execution_modes_output.ipynb`; second run kept):

- Matmul stack, first call / warm step:
  - eager 0.15 s / 2.91 ms
  - defer-and-fuse 2.69 s / 1.68 ms
  - `torch.compile` 3.00 s / 1.91 ms
- Tiny operations: 51.5 µs per operation in strict eager.

Gemma 4 E2B, the repack `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct`, 32 greedy tokens (`T/colab/runs/2026-10-21/probe_e2b_repack_ways_output.ipynb`). Versions: `torch` 2.14.1+cpu and `transformers` 5.20.1 preinstalled; `torch_tpu` 0.2.0, `libtpu` 0.0.50 and `compressed-tensors` 0.19.0 added.

| Way | 32 tokens | Notes |
| --- | ---: | --- |
| Weights expanded to bf16, strict eager `generate` | 16.98 s | the 8-token call took 6.32 s |
| Same, defer-and-fuse | 39.94 s | slower than strict eager |
| Kept compressed (`run_compressed=True`), strict eager | 9.21 s | first forward 15.62 s |
| Fixed buffer + `torch.compile` | failed | `expected tensor to be on tpu, got cpu`, shape `[1, 256, 8960]` |
| HF static cache + `torch.compile` | failed | `where(): expected tensor to be on tpu, got cpu`, in Gemma 4's multimodal `forward` |

All three working ways produced the same text. The two failures come from the multimodal class (`Gemma4ForConditionalGeneration`). The `-text` repack loads as `Gemma4ForCausalLM` and is the next probe.

**B. TorchTPU on v6e-1, July 2026** (`G/tpu-pytorch-inf2-2b/benchmarks/runs/2026-07-25-torchtpu-e2b-v6e1/REPORT.md`, L11–18). Despite the directory name, this is v6e-1. Gemma 4 E2B, bf16.

| Way | Speed | Source note |
| --- | ---: | --- |
| TorchTPU eager (HF `generate()`) | 3.4 tok/s | |
| TorchTPU compiled static loop | 227.3 tok/s, 4.4 ms/step | SEQ=256, no KV cache |
| vLLM, concurrency 1 | 209 tok/s | |

- The report states "67×" compiled over eager (L18).
- Caveats from the report (L74–79): in-process timing against HTTP; vLLM baseline from 2026-07-21; one run per configuration.
- The same tree recorded traps that still matter (`T@d660a09:.claude/skills/tpu-management/SKILL.md`):
  - "HF `generate()` cannot be torch.compiled on TPU today — it feeds a scalar CPU tensor into the graph" (L384–389).
  - Defer-and-fuse on `generate()`: "0.5 tok/s vs the 3.4 tok/s plain-eager baseline, 10k compile requests for 55 tokens" (L279–288).
  - Today's probe shows the same direction on v5e with the current release.

**C. Pure JAX on a Colab v5e-1, 2026-10-09** (`CJ/runs/2026-10-09/02_e2b_repack_vs_stock_on_jax_output.ipynb`).

- Decode speed: QAT bf16 135.0, Google w4a16 93.7, repack 93.3 tok/s.
- Repack fidelity against the QAT model: mean KL 0.00021, same top token 99.29%, 128 of 128 greedy tokens equal.
- This is the JAX side notebook 03 re-measures.

**D. Same GPU, both stacks** (`G/gpu-pytorch-g5g-2b/benchmarks/runs/2026-08-31-crossrig-torch-g5g/REPORT.md`). NVIDIA T4G, 2026-08-31, E2B, concurrency 1.

| Measure | JAX | PyTorch + transformers |
| --- | ---: | ---: |
| Decode, streamed (L23–24) | 12.69 | 10.24 |
| Cold boot (L180–181) | 242.2 s | 195.2 s |
| Warm boot | 74.1 s | 24.5 s |
| First completion, cold / warm | 22.9 s / 9.2 s | 1.0 s / 0.7 s |

- **Not a clean A/B:** JAX ran a quantized `ple4+int8head` configuration and PyTorch ran dense weights.
- JAX's first-completion time is per-shape compile on the first request.

**E. vLLM on TPU, JAX model path vs PyTorch fallback, 12B, v6e-1** (`G/QUANTIZATION.md` L296–309, 2026-09-26).

- The JAX path holds bf16 12B in 22.18 GiB; the PyTorch fallback "uses 24.56 GiB for bf16 12B".
- Accuracy: "−0.3 points against the PyTorch path".
- Throughput on the PyTorch path was not recorded in a comparable form.

#### 7. What the article can and cannot claim today

- **Can, from published sources:** TorchTPU replaces PyTorch/XLA; it is eager first with three modes plus `torch.compile`; it lowers through StableHLO and XLA like JAX; Google reports parity or near-parity with JAX on three large workloads on v6e/v7. (All of this is vendor-reported.)
- **Can, from this machine:** on a v5e-1, the mode choice moves a matmul stack 1.5x to 1.7x and a tiny-op chain about 100x. On TorchTPU 0.2.0, Hugging Face `generate` with Gemma 4 runs in strict eager and slows down under defer-and-fuse.
- **Cannot yet:** any same-chip, same-checkpoint, same-day JAX vs TorchTPU number for Gemma 4. Notebook 03 is designed to produce exactly that. Until then, the v6e-1 July numbers (B) and the colab-jax numbers (C) come from different chips or dates, and must not be put side by side as a ratio.
