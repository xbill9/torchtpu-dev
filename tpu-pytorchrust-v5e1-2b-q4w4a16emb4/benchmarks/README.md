# Benchmarks

`run_rust_benchmark` (MCP tool) streams four fixed prompts to the live server, one request at a time, and writes each run to `runs/<date>-rust-stream/<time>.json`: per request the time to first token, the completion token count from the server's `usage`, and the client-side decode rate, plus one non-streamed request with the engine's own `timings.read_s` and `timings.decode_s`.

First run, 2026-11-02 (`runs/2026-11-02-rust-stream/183042.json`), v5e-1 in us-west4-a, context 8,192, defer-and-fuse, measured from a laptop over the internet: 8 streamed requests, time to first token 0.175 to 0.256 s, client-side decode 11.5 to 12.6 tokens per second; the engine's own timing for one 48-token request was 11.9 tokens per second.

`serving-report.schema.json` is the shared report format the sibling rigs use for a full deployment report (stack identity, throughput sweep, memory, startup timeline, cost). Its sweep fields map to `vllm bench serve` output; a report for this rig would fill `engine` with `torchtpu_rs` and leave the vLLM-specific `raw` field out.

For comparison on the same chip, Colab notebook 06 measured the `tch` build of this engine on torch 2.13.0 at 14.7 to 15.7 tokens per second streamed with defer-and-fuse and 4.8 to 5.0 in strict eager, at a 33,024-token context (`../../colab/runs/2026-11-01/`). This rig differs in torch version (2.14.1), ATen layer (its own shim), context (8,192), a Python callback per streamed token, and where the stopwatch sits; which of those accounts for the gap is not measured.
