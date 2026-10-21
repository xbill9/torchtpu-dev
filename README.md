# torchtpu-dev

Gemma 4 serving rigs for TorchTPU (native PyTorch on Cloud TPU), one directory per rig. Each rig is self-contained: its own `tpu.env`, MCP server, Claude Code skill and plugin manifest, all named after the directory.

| Directory | Hardware | Model | Status |
| --- | --- | --- | --- |
| [`tpu-pytorch-v5e1-12b-w4a16`](tpu-pytorch-v5e1-12b-w4a16/) | TPU v5e-1 | `google/gemma-4-12B-it-qat-w4a16-ct` | The original rig; benchmark runs and articles live here |
| [`tpu-pytorch-v5e1-2b`](tpu-pytorch-v5e1-2b/) | TPU v5e-1 | `google/gemma-4-E2B-it` | Fork, not yet provisioned |
| [`tpu-pytorch-v5e1-2b-w4a16`](tpu-pytorch-v5e1-2b-w4a16/) | TPU v5e-1 | `google/gemma-4-E2B-it-qat-w4a16-ct` | Fork, not yet provisioned |
| [`tpu-pytorch-v6e1-12b-w4a16`](tpu-pytorch-v6e1-12b-w4a16/) | TPU v6e-1 | `google/gemma-4-12B-it-qat-w4a16-ct` | Fork, not yet provisioned |
| [`tpu-pytorch-v5e4-12b-w4a16`](tpu-pytorch-v5e4-12b-w4a16/) | TPU v5e-4 | `google/gemma-4-12B-it-qat-w4a16-ct` | Fork, not yet provisioned; serves from one chip |
| [`colab`](colab/) | Colab v5e-1 runtime | synthetic tensors (01); Gemma 4 E2B QAT repack (02, 03) | Notebooks 01 (execution modes), 02 (Gemma 4 E2B repack, ways to run it) and 03 (JAX vs TorchTPU head-to-head), verified on a Colab v5e-1; plus serving code |

Rig names follow `<platform>-<runtime>-<hardware>-<model>-<encoding>`. The name describes the rig; the values that matter (`MODEL_NAME`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`, zone) live in each rig's `tpu.env`.

#### Install a rig as a Claude Code plugin

```
/plugin marketplace add xbill9/torchtpu-dev
/plugin install tpu-pytorch-v5e1-12b-w4a16@torchtpu-dev
```

#### Work on a rig

Open Claude Code inside the rig directory. Its `CLAUDE.md`, skill and `make` targets all assume that directory is the working directory.

#### Links

[DRAFTERS.md](DRAFTERS.md) covers Google's Gemma 4 MTP drafters, bf16 and QAT, and which serving paths here can use them.

[RESOURCES.md](RESOURCES.md) collects the TorchTPU announcements, docs, recipes, containers and source repositories.
