# Gemma 4 MTP drafters

Google publishes a small drafter ("assistant") model for every Gemma 4 size, in two versions: one trained against the bf16 release and one trained against the QAT release. Both versions plug into speculative decoding as multi-token prediction (MTP) drafters. This page records what exists, how the drafters attach to a target, and where each serving path in this repo stands. Gathered 2026-10-21 from the Hugging Face API, `transformers` 5.19.0, and the `main` branches of `vllm-project/vllm` and `vllm-project/vllm-torchtpu`.

#### The checkpoints

All ten are public, BF16, and ship one `model.safetensors`.

| Checkpoint | Parameters | Architecture | Layers | Hidden size | Created |
| --- | ---: | --- | ---: | ---: | --- |
| `google/gemma-4-E2B-it-assistant` | 78.0M | `Gemma4AssistantForCausalLM` | 4 | 256 | 2026-04-23 |
| `google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant` | 78.0M | `Gemma4AssistantForCausalLM` | 4 | 256 | 2026-05-29 |
| `google/gemma-4-E4B-it-assistant` | 78.8M | `Gemma4AssistantForCausalLM` | 4 | 256 | 2026-04-23 |
| `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant` | 78.8M | `Gemma4AssistantForCausalLM` | 4 | 256 | 2026-05-29 |
| `google/gemma-4-12B-it-assistant` | 422.9M | `Gemma4UnifiedAssistantForCausalLM` | 4 | 1024 | 2026-05-23 |
| `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant` | 422.9M | `Gemma4UnifiedAssistantForCausalLM` | 4 | 1024 | 2026-06-04 |
| `google/gemma-4-26B-A4B-it-assistant` | 419.7M | `Gemma4AssistantForCausalLM` | 4 | 1024 | 2026-04-23 |
| `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` | 419.7M | `Gemma4AssistantForCausalLM` | 4 | 1024 | 2026-05-29 |
| `google/gemma-4-31B-it-assistant` | 469.5M | `Gemma4AssistantForCausalLM` | 4 | 1024 | 2026-04-23 |
| `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` | 469.5M | `Gemma4AssistantForCausalLM` | 4 | 1024 | 2026-05-29 |

- **The QAT drafter is its own set of weights.** At every size the two `model.safetensors` files have different hashes. The configs differ only in `transformers_version` and an added `eos_token_id: [1, 106]` on the QAT side.
- **The 12B drafters use a different architecture**, `Gemma4UnifiedAssistantForCausalLM`, matching the 12B's `gemma4_unified` model type. Every other size uses `Gemma4AssistantForCausalLM`.
- **There is no compressed-tensors drafter.** The QAT drafters are named for the `-qat-q4_0-unquantized` target. The rigs here serve `-qat-w4a16-ct`. Which drafter suits a w4a16 target is unmeasured on this stack.

#### How a drafter attaches to its target

The drafter runs on the target's state rather than on its own:

- `pre_projection` is `Linear(2 * backbone_hidden_size, hidden_size)`: the drafter reads the target's hidden state alongside the token embedding.
- It reuses the target's KV cache instead of keeping one of its own (upstream `gemma4_mtp.py`: "shares KV cache with the target (backbone) model").
- Logits go through a centroid-masked embedder that takes the target's `lm_head` weight, scoring only the tokens in the top centroids.

So a drafter only pairs with the size it was trained for, and the serving engine has to hand it the target's hidden states and KV cache every step.

#### Where each path in this repo stands

| Path | Drafter support |
| --- | --- |
| `tpu_openai_server.py` (every rig, and `colab/`) | None. The compiled static-shape decode loop emits one token per step and exposes neither hidden states nor KV cache to a second model. |
| `transformers` `generate(..., assistant_model=...)` | Supported in 5.19.0 (`gemma4_assistant` and `gemma4_unified_assistant` both ship). Runs eagerly; untested on TorchTPU. The usable entry point here is `tpu_generate.py`. |
| Upstream vLLM | `Gemma4MTP` (`vllm/model_executor/models/gemma4_mtp.py`), selected with `--speculative-config '{"method":"mtp","model":"<drafter>",...}'`. `config/speculative.py` maps both `gemma4_assistant` and `gemma4_unified_assistant` to it, so the 12B drafter is covered. CI pairs `google/gemma-4-E4B-it` with `google/gemma-4-E4B-it-assistant`. |
| vLLM-TorchTPU | Lists five methods: `eagle3`, `mtp`, `dflash`, `ngram`, `suffix`. Its `mtp` documentation and tests cover checkpoints with built-in heads (Qwen3.5, DeepSeek, GLM-5); its source has a Qwen3.5 MTP model and no Gemma 4 drafter code. Its validated Gemma 4 targets are the 31B and 26B-A4B (FP8-Dynamic). Gemma 4 + drafter on this stack is unvalidated. |

#### Measured elsewhere

`~/gemma4-dev/QUANTIZATION.md` ("Speculative decoding with Gemma 4's MTP drafters") has the only measurements, all on vLLM + `tpu_inference` (JAX) on a v5e-1 with E2B, 2026-09-29:

- With `google/gemma-4-E2B-it-assistant`, every target ran faster at 1 request (1.80x to 2.40x) and slower at 16 (0.56x to 0.69x).
- With the QAT drafter, every target ran slower at every concurrency (0.45x to 0.58x).
- On an RTX 3050 under llama.cpp the two drafters were level, so the gap belongs to the vLLM or TPU side.

None of that was measured on TorchTPU, and none of it covered the 12B.

#### Other drafter families

Upstream vLLM also has `Gemma4DSparkModel`, and vLLM-TorchTPU supports DSpark drafting for other models. Third parties publish Gemma 4 DSpark drafters on Hugging Face, for example `deepseek-ai/dspark_gemma4_12b_block7`, `RedHatAI/gemma-4-31B-it-speculator.dspark` and `rasyosef/gemma-4-E2B-it-dspark`. Google publishes none.
