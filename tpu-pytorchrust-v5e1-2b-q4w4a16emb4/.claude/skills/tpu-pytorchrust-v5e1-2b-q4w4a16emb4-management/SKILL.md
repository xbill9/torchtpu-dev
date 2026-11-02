---
name: tpu-pytorchrust-v5e1-2b-q4w4a16emb4-management
description: Provision a TPU v5e-1 and serve Gemma 4 E2B with the model written in Rust on TorchTPU (a PyO3 extension over its own ATen shim, no tch), OpenAI-compatible on port 8000, and operate it through the tpu-pytorchrust-v5e1-2b-q4w4a16emb4 MCP server. Use when the user asks to deploy, check, benchmark or tear down the Rust TorchTPU rig, about its PyO3 bindings or startup stages, or about queued resources, quotas and costs for it. Triggers include "Rust on TPU", "torchtpu_rs", "PyO3", "pytorchrust", "TorchTPU rig", "v5e-1 queued resource".
---

# Gemma 4 E2B in Rust on TorchTPU, on a v5e-1

The VM runs `rust_openai_server.py`, an OpenAI-compatible FastAPI server. The model, the ring
cache and the decode loop are the `torchtpu_rs` Python extension, built on the VM from the
crate in `mcp/payload/rust/` against the torch installed there. Every operation runs as a
TorchTPU kernel, reached through a small C++ shim around ATen (`aten.cpp`). Python hosts the
process, tokenizes and serves HTTP. No vLLM, no Docker, no JAX.

Two ways to act:

1. **MCP tools** from the `tpu-pytorchrust-v5e1-2b-q4w4a16emb4` server, when it is connected
   (catalog below).
2. **Direct `gcloud`** otherwise, with `references/tpu-guide.md`.

## Bundled files

- `mcp/server.py`: the MCP server (a snapshot of the repo-root `server.py`).
- `mcp/startup_script_rust_template.sh`: the VM startup script the server renders.
- `mcp/payload/`: what the startup script carries to the VM: `rust/` (the crate,
  `PYO3.md`, the `.pyi` stub) and `rust_openai_server.py`.
- `mcp/project-setup.sh`: installs this skill into a project and registers the server.
- `references/tpu-guide.md`: capacity, quotas, troubleshooting.

## The PyO3 bindings, in short

`mcp/payload/rust/PYO3.md` has the full description. The four crossings:

1. **Token ids are Python lists.** `decode(ids: list[int], ...)` in, `DecodeResult.tokens:
   list[int]` out. Tokens never cross as tensors.
2. **Tensors share storage.** `like` (which names the device) and `logits()` cross as
   `torch.Tensor` without a copy, through `THPVariable_Unpack` / `THPVariable_Wrap` in the shim.
3. **Long calls release the GIL.** `Engine(...)`, `decode`, `generate` and `logits` run in
   `py.detach`, so the asyncio loop keeps answering `/health` during a decode.
4. **The callback takes the GIL back.** `decode(..., on_tokens=f)` calls `f(list[int])` for
   each batch read from the TPU. A false return cancels; an exception is re-raised.

`Engine` is `#[pyclass(unsendable)]`. The server builds it on one dedicated thread and queues
every request to that thread, so requests run one at a time. A call from any other thread
raises `pyo3_runtime.PanicException`, a BaseException subclass.

## Standard lifecycle

1. `estimate_deployment_cost` with `tpu_type="v5e"`, `topology="1x1"`.
2. `create_tpu_queued_resource` (flex-start, 4 h max run). v5e-1 can only be provisioned
   as a queued resource: `gcloud compute instances create` rejects `ct5lp-hightpu-1t`.
   flex-start v5e-1 has only been granted in us-west4-a (the zone in `tpu.env`). `find_tpu`
   sweeps other zones with quota.
3. `get_rig_progress` or `wait_for_rig_ready`. The startup script writes its stage to the VM's
   guest attribute `torchtpu-rs/stage`: network, python, torch, rust, crate (`cargo build`),
   weights, smoke (a Rust call on a tpu tensor), unit, serving. A TPU VM has no serial-port
   command, and reading the stage needs no SSH. A failure reads `ERROR: <stage>: <reason>`.
4. `verify_model_health`, `get_rust_bindings` (GET /v1/engine), `query_queued_gemma4`.
5. `run_rust_benchmark` for TTFT and decode tok/s per request. Results go to
   `benchmarks/runs/<date>-rust-stream/`.
6. `destroy_queued_resource` when done.

## MCP tool catalog

| Task | Tools |
| --- | --- |
| Capacity | `create_tpu_queued_resource`, `find_tpu`, `manage_queued_resource` (deletes others), `list_queued_resources`, `describe_queued_resource`, `destroy_queued_resource`, `get_zones_with_available_quota`, `estimate_deployment_cost` |
| Startup | `get_rig_progress`, `wait_for_rig_ready`, `verify_rust_on_tpu` (SSH) |
| Serving | `manage_rust_server` (systemctl and journal for the `torchtpu-rs` unit), `get_serving_endpoint`, `get_rust_bindings`, `get_deployment_config`, `save_hf_token` |
| Monitoring | `get_system_status`, `verify_model_health`, `get_model_details`, `get_metrics`, `get_tpu_system_logs`, `get_cloud_logging_logs`, `analyze_cloud_logging` |
| Inference | `query_queued_gemma4`, `run_rust_benchmark` |
| GCE flex-start path (v6e/v5p only) | `create_tpu_vm_instance`, `find_tpu_vm`, `list_tpu_vm_instances`, `destroy_tpu_vm_instance`, `get_tpu_vm_serial_log`, `get_tpu_vm_endpoint`, `wait_for_serving_ready`, `wait_for_pytorch_ready` |
| Help | `get_help`, `find_gpu` |

## The server's API

- `POST /v1/chat/completions` and `POST /v1/completions`, streaming or not. Decoding is greedy:
  `temperature` and `top_p` are accepted and ignored, and `system_fingerprint` reads
  `torchtpu_rs-greedy`. Generation stops at the checkpoint's `eos_token_id` list `[1, 106, 50]`.
  Non-streamed responses carry `timings.read_s` and `timings.decode_s` from the engine.
- `GET /health`: 503 while the engine loads and warms up, 200 after.
- `GET /v1/models`, `GET /v1/engine`, `GET /metrics` (`torchtpu_rs_*` Prometheus counters).

## Configuration (`tpu.env`)

`MODEL_NAME` (served id, and the int4 tables), `BASE_REPO` (W4A16 linears, config,
tokenizer), `ACCELERATOR_TYPE=v5e-1`, `PYTHON_BIN` (python3.14), `TORCH_PIP_SPEC` (PyTorch 2.14.1, torch_tpu 0.2.0,
from public PyPI), `SERVE_CONTEXT` (a multiple of 256), `SERVE_READ_EVERY` (1 streams),
`SERVE_DEFER_AND_FUSE=1`, `SERVE_TAGS` (firewall tags that open port 8000 and SSH),
`TPU_QR_NAME`. The crate compiles its ATen shim against whichever torch is installed, so the
torch version is a choice; the module refuses to import under a different torch than it was
built with. The Rust channel (newest nightly) is in `mcp/payload/rust/rust-toolchain.toml`.

## Registering the MCP server

```bash
mcp/project-setup.sh /path/to/project --project <gcp-project-id>   # one project
mcp/project-setup.sh --global                                      # all projects
```

Requires `pip install -r mcp/requirements.txt`, an authenticated `gcloud` with the alpha
components, and the TPU API enabled. No Hugging Face token is needed: both checkpoints are
public.

## Cautions

- `destroy_queued_resource` and `manage_queued_resource` delete infrastructure, and
  `manage_queued_resource` deletes every other queued resource in the zone.
- A flex-start VM self-deletes at its max run duration (4 h).
- The engine serves one sequence at a time; concurrent requests queue.
- `SERVE_TAGS` must name tags your firewall rules open for port 8000 and SSH, or the
  endpoint and the SSH tools are unreachable while the VM is healthy.
