# Demo: Gemma 4 E2B in Rust on TorchTPU, from this machine

Three pieces between you and the TPU:

| Piece | Where | What |
| --- | --- | --- |
| OpenAI endpoint | The TPU VM, port 8000 | `rust_openai_server.py` over the `torchtpu_rs` PyO3 module: `/v1/chat/completions` and `/v1/completions` (streaming or not), `/v1/models`, `/v1/engine`, `/health`, `/metrics` |
| Local shim | This machine, `http://127.0.0.1:8015` | `shim.py`, a systemd user service (`torchtpu-rs-shim`) that forwards everything to the TPU, streaming included. It finds the VM by asking gcloud for the queued resource's node, and asks again when the VM stops answering, so a new flex-start VM needs no reconfiguration |
| Chat UI | `http://127.0.0.1:8015/` in a browser | `ui.html`, served by the shim: streaming chat with the conversation kept, Stop (which cancels the decode on the TPU), max tokens and an optional system prompt, per-reply time to first token, decode rate and token count, a TPU status badge, and the engine's build and versions |
| Demo MCP server | Claude Code, user scope | `mcp_demo.py` (`torchtpu-rs-demo`): five read-only tools over the shim |

#### Install

```bash
demo/install.sh --project <gcp-project-with-the-tpu>   # shim service + MCP registration; idempotent
demo/install.sh --uninstall
```

The shim's settings land in `~/.config/torchtpu-rs/shim.env`; the queued resource name, zone and port come from `tpu.env`. Set `TORCHTPU_RS_UPSTREAM=http://<ip>:8000` there to pin the VM instead of looking it up. Restart Claude Code once after installing so it loads `torchtpu-rs-demo`.

#### The demo, in order

1. **Bring up the TPU**, if it is not up: from Claude Code in this repo, `create_tpu_queued_resource`, then `wait_for_rig_ready` (the ops MCP server). On 2026-11-02 the startup script took 4 min 29 s from installing Python to starting the service.
2. **Open the chat UI** at `http://127.0.0.1:8015/` and ask something; the side panel shows what is running on the TPU.
3. **Show the endpoint is live:** `endpoint_status`, or `curl -s 127.0.0.1:8015/shim/status`.
4. **Show what is running on the TPU:** `engine_info`. It names the PyO3 module, the torch it was built against, the nightly rustc, the thread that owns the unsendable `Engine`, and TorchTPU's defer-and-fuse mode.
5. **Ask it something:** `ask_gemma_rust` returns the answer with the engine's own prompt-read seconds and decode rate.
6. **Stream it:** `stream_gemma_rust` times the first token and the decode rate as a client sees them.
7. **Any OpenAI client works** against the shim:

   ```python
   from openai import OpenAI
   c = OpenAI(base_url="http://127.0.0.1:8015/v1", api_key="none")
   model = c.models.list().data[0].id
   for ch in c.chat.completions.create(model=model, stream=True,
           messages=[{"role": "user", "content": "Say hello from a TPU in five words."}]):
       print(ch.choices[0].delta.content or "", end="", flush=True)
   ```

   ```bash
   curl -sN 127.0.0.1:8015/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model":"x","stream":true,"messages":[{"role":"user","content":"Count from 1 to 5."}]}'
   ```

8. **Close with the counters:** `serving_metrics`.
9. **Stop billing:** `destroy_queued_resource` (ops MCP server). A flex-start VM also stops itself 4 hours after it was granted.

#### Checked, 2026-11-02

In Chrome at `http://127.0.0.1:8015/`: a streamed reply (first token 0.28 s, 12.2 tok/s, 123 tokens), a follow-up that used the conversation (21 tokens), and Stop mid-reply, after which the server's `torchtpu_rs_finish_total{reason="cancelled"}` went from 0 to 1 and `torchtpu_rs_busy` to 0.

Through the shim against the v5e-1 in us-west4-a: `/v1/models`; a non-streamed chat (7 tokens, read 0.095 s, decode 0.600 s); a streamed chat (16 SSE lines); the OpenAI Python client streaming "Hello from the TPU."; and all five MCP tools, with `ask_gemma_rust` reporting 11.5 tok/s decode on the TPU and `stream_gemma_rust` a 0.187 s first token. `claude mcp list` shows `torchtpu-rs-demo` connected. `tests/test_demo.py` covers the shim's error path and the tool list offline.

Decoding is greedy and the engine serves one request at a time; concurrent requests queue. It does not parse tool calls, so it answers questions but cannot drive an agent's tools.
