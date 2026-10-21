# colab

TorchTPU serving code for a Colab TPU runtime. This directory sits outside the rig naming scheme: Colab is a runtime you attach to, not capacity you provision with `gcloud`, so none of the MCP server, skill, or provisioning tooling from the rig directories applies here.

Forked from [`tpu-pytorch-v5e1-12b-w4a16`](../tpu-pytorch-v5e1-12b-w4a16/) on 2026-10-21. Nothing in this directory has been run in Colab yet.

#### What is here

| File | Role |
| --- | --- |
| `tpu_compiler.py` | `compile_for_tpu`, input validation, the `LLMEngine` protocol |
| `tpu_engine.py` | `TPUEngine`, built on `tpu_compiler.py` |
| `tpu_server.py` | OpenAI-compatible FastAPI server over `TPUEngine` |
| `tpu_openai_server.py` | Static-shape decode server (the one the TPU VM rigs run) |
| `tpu_generate.py` | One-shot generation from a Hugging Face causal LM |

The rig directories keep the authoritative copies. When one of these files changes there, copy it here.

#### Model

Same checkpoint as the parent rig: `google/gemma-4-12B-it-qat-w4a16-ct` (~7 GB of weights).
