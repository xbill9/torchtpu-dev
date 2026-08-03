# Next session — static-shape TPU serving

Picking up from 2026-08-02. Read this first; it records state that is not
derivable from the tree.

## Where things stand

`tpu_openai_server.py` serves Gemma 4 E2B from a compiled static-shape decode
loop and is **validated on real hardware** — TPU v6e-1, 24/24 live checks plus a
real `openai` SDK client (chat, chat streaming, completions streaming,
list-form content, `finish_reason`). Measured **192 tok/s / 5.2 ms per token**,
one compile, zero errors.

Six OpenAI-contract defects were fixed in `a1a0a06` on branch
`fix/openai-endpoint-contract` (pushed, no PR opened):

| Was | Now |
|---|---|
| `finish_reason` hardcoded `"stop"` | `"length"` vs `"stop"`, both routes, streaming and not |
| `/v1/completions` ignored `stream: true` | real SSE, `[DONE]`-terminated |
| streaming blocked the event loop | `/health` 30 ms → 0.9 ms during a stream |
| `torch.compile` lacked the `dynamic=False` the docstring promised | passed for real |
| over-long prompt streamed a broken 200 | `400` on both paths |
| list-form `content` → 422 even for pure text | accepted; non-text rejected explicitly |

**The TPU VM is deleted.** It was flex-start with
`instanceTerminationAction: DELETE`, so there is no stopped instance to
restart — a new one must be provisioned.

## Read before you start

1. **The endpoint does not come back on its own.** `startup_script_pytorch_template.sh`
   installs the backend and smoke-tests `torch.compile(backend="tpu")`, but it
   has no knowledge of `tpu_openai_server.py`. A fresh VM boots with a working
   TPU and no server. Getting back to a serving endpoint is manual today — see
   the recreate notes below.

2. **`.mcp.json` does not exist and no `TPU_BACKEND_*` vars are set in the
   shell.** `create_tpu_vm_instance(workload="pytorch")` fails fast with an
   explicit message until they are. The backend package name and registry are
   deliberately not in the tracked tree; they live in your environment.

3. **Concrete values are in the gitignored run directory**, not here:
   `benchmarks/runs/2026-08-02-torchtpu-e2b-v6e1-openai/RECREATE.md` lists the
   exact wheel URIs, env-var names observed at runtime, and the launch sequence.
   That directory also holds the rescued logs from the deleted VM — three
   benchmark logs that exist nowhere else.

4. **A 2.59 GiB compile cache survives in GCS** and is worth 19–70 s of warmup
   per run. It is keyed by a backend binary fingerprint, so a backend version
   bump silently invalidates it — a sudden slow warmup means the fingerprint
   moved, not that something broke.

## Work items, highest value first

### 1. Make the deployment reproducible (small, unblocks everything else)

Fold the serving steps into the pytorch startup template so a fresh v6e-1 comes
up already serving: stage `tpu_openai_server.py`, install `transformers` and
`openai` via `TPU_BACKEND_PIP_EXTRAS`, launch under `setsid nohup`, and
`chown` the driver log directory (see #3). Today this is five manual steps and
they are not written down anywhere except the recreate notes.

Worth adding a `wait_for_openai_ready` MCP tool alongside the existing
`wait_for_pytorch_ready`, polling `/health` for `warm: true`.

### 2. Static-shape ring KV cache (the real performance work)

`use_cache=False` in the decode step means **every token pays a full
`SEQ`-wide forward pass**. Confirmed empirically: per-token latency is flat at
~5.2 ms whether the prompt is 12 tokens or 182. Cost is set by buffer size, not
actual sequence length, so raising `--seq` makes every request slower.

Two coupled problems:

- **No KV cache.** Needs a ring buffer written with `index_copy_` at a tensor
  position, same trick the token buffer already uses, so shapes stay static and
  the single-compile property holds.
- **Flash attention never engages.** The backend logs its rejection at startup:
  sequence length must be divisible by 512 (256 is not) and `attn_mask` must be
  absent (it is present). A 512-multiple buffer fixes the first; the mask is the
  harder half and may need a causal-implicit path instead of an explicit mask.

Benchmark against the 192 tok/s / 5.2 ms baseline above. `decode.log` and
`gen_compiled.log` in the run directory are the prior eager-vs-compiled
comparison and are the right shape to extend.

### 3. Driver log directory permissions

The TPU driver creates `/tmp/tpu_logs` as root during boot, so a server running
as a normal user cannot write there and floods its own log with
`Permission denied` — 78 KB of it in one session, and driver-level diagnostics
are lost. Fixed manually last time with `chown`. Belongs in the startup script,
before any workload starts.

### 4. Test coverage

Nothing in `tests/` covers `tpu_openai_server.py`, and the two existing test
files cannot even collect without `torch` installed. A 15-check suite against a
fake engine — no TPU, no torch — validates every fix above and would run in CI.
It exists but was never committed; see the recreate notes for its location.

### 5. Multimodal

The server is text-only. `ChatMessage.content` now accepts OpenAI's
list-of-parts form but rejects any non-text part. The model is multimodal and
`CLAUDE.md` specifies image/audio limits for the vLLM path, so this is a real
gap if the pytorch path is meant to reach parity.

## Things that will bite you

- **`SEQ=256` is the hard ceiling on prompt + output combined.** A 402-token
  prompt is a `400`, not a truncation. Raising it costs latency on every
  request until the KV cache lands.
- **Flex-start VMs self-delete.** Anything on local disk goes with them. The
  server file is committed; nothing else on that VM was, which is why the logs
  had to be rescued before teardown.
- **`temperature: 0` is the default here**, unlike the OpenAI default of 1.0.
  Greedy decoding is what the streaming-vs-non-streaming equivalence check
  relies on.
