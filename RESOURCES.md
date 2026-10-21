# TorchTPU Resources

Links for native PyTorch on TPU (TorchTPU), collected from the TorchTPU open-source launch on 2026-10-20.

## Announcements

- [TorchTPU is Now Open Source: Run Native PyTorch on TPUs](https://cloud.google.com/blog/topics/ai-infrastructure/torchtpu-is-now-open-source-run-native-pytorch-on-tpus) — Google Cloud blog
- [llm-d 1.0: An Open Source Stack for Distributed Inference](https://cloud.google.com/blog/topics/ai-infrastructure/llm-d-1-0-an-open-source-stack-for-distributed-inference) — Google Cloud blog
- [Announcing New vLLM and SGLang Inference Engines on TPU](https://developers.googleblog.com/announcing-new-vllm-and-sglang-inference-engines-on-tpu/) — inference deep dive
- [Integrating TorchTPU into TorchTitan & Optimizing Models for TPU](https://developers.googleblog.com/integrating-torchtpu-into-torchtitan-and-optimizing-models-for-tpu/) — training deep dive
- [TorchTPU technical paper (arXiv 2610.24161)](https://arxiv.org/abs/2610.24161)

## Ecosystem

- [Introducing TorchTPU](https://pytorch.org/blog/introducing-torchtpu/) — PyTorch Foundation
- [Hugging Face + TorchTPU](https://huggingface.co/blog/hf-torchtpu) — Hugging Face
- [vLLM TorchTPU launch](https://vllm.ai/blog/2026-10-20-vllm-torchtpu) — vLLM
- [SGLang TorchTPU launch](https://lmsys.org/blog/2026-10-20-sglang-torchtpu) — LMSYS
- [Ray support for native PyTorch on Google TPU](https://www.anyscale.com/blog/announcing-ray-support-native-pytorch-google-tpu) — Anyscale

## Docs

- [PyTorch on TPU](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu) — Cloud TPU docs
- [vLLM-TorchTPU](https://vllm-torchtpu.readthedocs.io/en/latest/)
- [llm-d on GKE TPU](https://llm-d.ai/docs/dev/infrastructure/providers/gke/tpu)

## Recipes

- [Ironwood (v7) vLLM-TorchTPU inference](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/vLLM-TorchTPU)
- [Trillium (v6e) vLLM-TorchTPU inference, Gemma 4 26B](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/vLLM-TorchTPU/Gemma4-26B)
- [Ironwood (v7) TorchTitan pre-training](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/torchtitan)
- [llm-d TorchTPU baseline](https://github.com/llm-d/llm-d/tree/main/guides/optimized-baseline/modelserver/tpu)

## Containers

- [vllm/vllm-torchtpu](https://hub.docker.com/r/vllm/vllm-torchtpu) — Docker Hub

## Source Repositories

- Core and training: [torch_tpu](https://github.com/google-pytorch/torch_tpu) · [torchtitan](https://github.com/google-pytorch/torchtitan) · [ray](https://github.com/ray-project/ray)
- Serving: [vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) · [sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu/) · [llm-d](https://github.com/llm-d/llm-d)
- Agents and edge: [tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite) · [LiteRT](https://github.com/google-ai-edge/litert)
