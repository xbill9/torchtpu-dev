# colab

Colab notebooks for TorchTPU on a v5e-1, plus the serving code they draw on. This directory sits outside the rig naming scheme: Colab is a runtime you attach to, not capacity you provision with `gcloud`, so none of the MCP server, skill, or provisioning tooling from the rig directories applies here.

Open a notebook in Colab, pick **Runtime → Change runtime type → v5e-1 TPU**, then **Runtime → Run all**.

| Notebook | What it teaches | Token? | Status |
| --- | --- | --- | --- |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xbill9/torchtpu-dev/blob/main/colab/notebooks/01_torchtpu_execution_modes.ipynb) [`01_torchtpu_execution_modes`](notebooks/01_torchtpu_execution_modes.ipynb) | TorchTPU's three execution modes (strict eager, defer-and-fuse, `torch.compile`), tiny-op dispatch cost, static shapes and buckets, and the cost of `.item()`, measured on synthetic tensors | No | Verified on a Colab v5e-1, 2026-10-21 (`runs/2026-10-21/`) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xbill9/torchtpu-dev/blob/main/colab/notebooks/02_e2b_repack_on_torchtpu.ipynb) [`02_e2b_repack_on_torchtpu`](notebooks/02_e2b_repack_on_torchtpu.ipynb) | Gemma 4 E2B QAT repack on TorchTPU: getting it onto the chip (`key_mapping`, `device_map="tpu"`), five ways to generate with speed, memory and token agreement, and the compiled step's graph breaks | No | Verified on a Colab v5e-1, 2026-10-21 (`runs/2026-10-21/`) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xbill9/torchtpu-dev/blob/main/colab/notebooks/03_e2b_jax_vs_torchtpu.ipynb) [`03_e2b_jax_vs_torchtpu`](notebooks/03_e2b_jax_vs_torchtpu.ipynb) | Gemma 4 E2B head-to-head on one v5e chip: the `tpu-jax` engine (4-bit and bf16) against PyTorch on TorchTPU (eager, defer-and-fuse, compiled buffer), on the same checkpoint and prompt: token agreement, KL between the stacks, decode speed on one stopwatch, memory | No | Verified on a Colab v5e-1, 2026-10-21 (`runs/2026-10-21/`) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xbill9/torchtpu-dev/blob/main/colab/notebooks/04_e2b_compiled_decode.ipynb) [`04_e2b_compiled_decode`](notebooks/04_e2b_compiled_decode.ipynb) | Gemma 4 E2B decode made fast on TorchTPU with Transformers' own model code: where the 4-bit weights go, the compressed-tensors wrapper that splits the compiled step into 24 graphs, and seven ways from `generate` to a static-cache compiled step with a 4-bit per-layer table, against the JAX engine on one stopwatch | No | Verified on a Colab v5e-1, 2026-10-23 (`runs/2026-10-23/`) |

#### Authoring model

The layout and tooling follow [`xbill9/colab-jax`](https://github.com/xbill9/colab-jax). **Notebooks are build artifacts: edit `src/*.py`, never `notebooks/*.ipynb`.** Sources are [jupytext](https://jupytext.readthedocs.io) `py:percent` files. The build prepends the Apache-2.0 cell and the Colab badge and sets `metadata.accelerator = "TPU"`. The `.ipynb` files are committed because Colab serves them from GitHub, and `make check` fails if they have drifted from their sources.

```bash
make setup     # pip install -r requirements-dev.txt: jupytext, nbformat, google-colab-cli
make build     # src/*.py -> notebooks/*.ipynb
make check     # fail if committed notebooks are stale
make verify    # run every notebook on a real Colab v5e-1; fail on any error
make sessions  # list live Colab runtimes (these cost compute units)
```

`make verify` provisions real hardware through the [Colab CLI](https://github.com/googlecolab/google-colab-cli) and files each executed copy under `runs/<date>/` as evidence. It needs a user credential with the `colaboratory` scope; see the colab-jax README for both auth routes and for why a timed-out `colab new` must never be retried.

#### Serving code

The `tpu_*.py` files are copies of the rigs' serving code, forked from [`tpu-pytorch-v5e1-12b-w4a16`](../tpu-pytorch-v5e1-12b-w4a16/) on 2026-10-21. The rig directories keep the authoritative copies; when one of these files changes there, copy it here.

| File | Role |
| --- | --- |
| `tpu_compiler.py` | `compile_for_tpu`, input validation, the `LLMEngine` protocol |
| `tpu_engine.py` | `TPUEngine`, built on `tpu_compiler.py` |
| `tpu_server.py` | OpenAI-compatible FastAPI server over `TPUEngine` |
| `tpu_openai_server.py` | Static-shape decode server (the one the TPU VM rigs run) |
| `tpu_generate.py` | One-shot generation from a Hugging Face causal LM |
