# tpu-pytorchrust-v5e1-2b-q4w4a16emb4

Gemma 4 E2B on one TPU v5e chip with the model written in Rust and every operation a TorchTPU kernel, served OpenAI-compatible on port 8000. A Claude Code skill and an MCP server provision it and operate it.

> **Sister rig of [`tpu-pytorch-v5e1-2b-w4a16`](../tpu-pytorch-v5e1-2b-w4a16/)**: the same chip and the same MCP tooling, with the model moved out of Python into Rust. The Rust engine is the one from [Colab notebook 06](../colab/notebooks/06_e2b_rust_on_torchtpu.ipynb), extended here with stop tokens, streaming and the bindings a server needs. First served on a v5e-1 on 2026-11-02: `evidence/tpu_boot_2026-11-02.txt` (boot) and `benchmarks/runs/2026-11-02-rust-stream/` (streaming benchmark).

**GitHub:** https://github.com/xbill9/torchtpu-dev/tree/main/tpu-pytorchrust-v5e1-2b-q4w4a16emb4

---

#### What runs on the VM

| Layer | What | File |
| --- | --- | --- |
| HTTP | FastAPI, OpenAI-compatible: chat and text completions, streaming, `/health`, `/v1/engine`, `/metrics` | `rust_openai_server.py` |
| Tokenizer | The checkpoint's chat template, via Transformers | `rust_openai_server.py` |
| Bindings | PyO3 0.30: the `torchtpu_rs` module, the `Engine` class, the `decode` callback | `rust/src/bindings.rs`, `rust/PYO3.md` |
| Model | 35 layers, 4-bit weights unpacked to bf16, the ring cache, greedy decode with stop tokens | `rust/src/gemma.rs`, `rust/src/weights.rs` |
| ATen | A C++ shim around the ~40 ATen ops the model uses, compiled against the installed torch; no `tch` | `rust/src/aten.cpp`, `rust/src/aten.rs`, `rust/build.rs` |
| Kernels | The libtorch that `torch_tpu` 0.2.0 registers its TPU kernels with | PyPI: `torch==2.14.1`, `torch_tpu==0.2.0` |

Python hosts the process because TorchTPU 0.2.0 links `libtorch_python` and ships no supported C++ API. `TORCH_TPU_DEFER_AND_FUSE=1` is set on the service: `torch.compile` traces Python and cannot see a Rust model, and defer-and-fuse works below ATen. Notebook 06 measured it at about 15 tokens per second streamed against about 5 for strict eager, on a Colab v5e-1.

The checkpoint is `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` for the linears, config and tokenizer, plus the int4 vocabulary tables from `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`. That pairing is the `q4w4a16emb4` encoding in the directory name.

---

#### The PyO3 bindings

`rust/PYO3.md` describes every item Python can call and each place a value crosses the boundary; `rust/python/torchtpu_rs.pyi` is the type stub. In short:

```python
import torch
import torch_tpu          # registers the TPU kernels on libtorch's dispatcher
import torchtpu_rs        # the Rust module, linked to that same libtorch

engine = torchtpu_rs.Engine(model_dir, emb4_dir, torch.zeros(1, device="tpu"), 8192)
r = engine.decode(ids, 256, stop=[1, 106, 50], read_every=1, on_tokens=print)
r.tokens, r.finish, r.read_s, r.decode_s
```

| Crossing | How |
| --- | --- |
| Token ids | `list[int]` in and out; never tensors |
| `like`, `logits()` | `torch.Tensor` and the Rust `Tensor` share the same `at::Tensor`, no copy |
| GIL | Released for load, `decode`, `generate` and `logits`; taken back to call `on_tokens` |
| Threads | `Engine` is `unsendable`: built and used on one thread, which the server dedicates to it |

`tests/test_rust_bindings.py` checks each of these on CPU against the real checkpoint (`make rust fetch test-rust`; 8 tests, 249 s on a laptop, 2026-11-02). Against Transformers on the same torch, 64 greedy tokens matched 64 of 64 with a first-token KL of 1.08e-5 (`evidence/`).

---

#### Deploy

```bash
./project-setup.sh /path/to/project --project <gcp-project-id>   # skill + MCP server
```

Then, from Claude Code with the MCP server connected:

1. `create_tpu_queued_resource`: a flex-start v5e-1 in `us-west4-a`, 4 h maximum. v5e can only be provisioned as a queued resource.
2. `wait_for_rig_ready`, or `get_rig_progress` to read the stage: network, python, torch, rust, crate, weights, smoke, unit, serving.
3. `get_rust_bindings`, `query_queued_gemma4`, `run_rust_benchmark`.
4. `destroy_queued_resource`.

The startup script carries the crate and the server inside the instance metadata (48 KB rendered), installs Python 3.14 (no venv), PyTorch 2.14.1 and TorchTPU from public PyPI, installs the newest Rust nightly, runs `cargo build --release` on the VM, downloads the two checkpoints, runs a Rust call on a tpu tensor as a smoke test, and starts the `torchtpu-rs` systemd unit. `tpu.env` holds every setting.

---

#### Demo from this machine

`demo/install.sh --project <id>` installs a local OpenAI-compatible shim with a chat UI (`http://127.0.0.1:8015/` in a browser, `/v1` for clients; systemd user service `torchtpu-rs-shim`) that finds the TPU VM and forwards to it, and registers a demo MCP server (`torchtpu-rs-demo`: ask, stream, engine info, endpoint status, metrics) with Claude Code. `demo/README.md` has the run order.

---

#### Develop

```bash
make rust        # build rust/build/torchtpu_rs.so against the torch python3 imports
make fetch       # download both checkpoints into the Hugging Face cache
make test-rust   # the PyO3 binding tests, on CPU
make test        # server and template tests (no GCP calls)
make skill       # refresh the skill snapshot, including mcp/payload/
```

The crate has no `tch` dependency, so it builds against whichever torch `python3` imports (checked here with 2.14.1 on Python 3.14.7 and rustc nightly 2026-11-01); the module refuses to import under a different torch. `python3 rust_openai_server.py --device cpu ...` runs the whole server on CPU for testing.

---

#### Layout

| Path | Role |
| --- | --- |
| `server.py` | The MCP server (MCP Python SDK 2.x) |
| `rust/` | The crate: `lib.rs`, `bindings.rs` (all PyO3), `aten.cpp` + `aten.rs` (the ATen shim), `gemma.rs`, `weights.rs`, `build.rs`, `rust-toolchain.toml`, `PYO3.md`, `python/torchtpu_rs.pyi` |
| `evidence/` | CPU parity against Transformers, and the script that produced it |
| `rust_openai_server.py` | The server the VM runs |
| `.claude/skills/<rig>-management/` | The skill; `mcp/startup_script_rust_template.sh` and `SKILL.md` are hand-maintained, the rest is generated by `make skill` |
| `tpu.env` | Every deployment setting |
| `demo/` | Local shim, demo MCP server, installer, run order |
| `tests/` | `test_server.py` (tools, templates, payload), `test_rust_bindings.py` (PyO3 on CPU), `test_demo.py` (shim and demo MCP, offline) |
