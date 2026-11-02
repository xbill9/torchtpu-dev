---
title: "Gemma 4 in Rust on TorchTPU: Hands On With Colab"
published: false
description: "A Rust crate reads the checkpoint, runs all 35 layers of Gemma 4 E2B and drives the decode loop on one TPU v5e chip in Colab, with every operation a TorchTPU kernel and the same tokens as compiled Python."
tags: rust, pytorch, tpu, gemma
cover_image: https://raw.githubusercontent.com/xbill9/torchtpu-dev/main/docs/torchtpu-rust/devto-cover.415eb5a0.jpg
---

This article provides a step by step guide to running Gemma 4 E2B with the model written in Rust and every operation executed by TorchTPU, on a Colab v5e-1 runtime: one TPU v5e chip. A Rust crate reads the checkpoint, unpacks the 4-bit weights, runs the 35 layers and the ring cache, and drives the decode loop. Python hosts the backend and starts the engine. No JAX is involved.

https://github.com/xbill9/torchtpu-dev/tree/main/colab/rust

**Rust writes the same tokens as compiled Python at about half the streaming speed and less than half the peak memory.** Same chip, same checkpoint, same 19-token prompt, 64 greedy tokens, 33,024-token context:

| One Colab v5e-1 chip | Streamed tok/s | Batched tok/s | First call | Peak memory |
| :--- | ---: | ---: | ---: | ---: |
| Rust, defer-and-fuse | **14.7 to 15.7** | 15.2 to 16.0 | 82 to 84 s | **7.75 GB** |
| Rust, strict eager | 4.8 to 5.0 | 4.8 | 23 to 25 s | 8.62 GB |
| Python `torch.compile` | 33.4 to 33.5 | 45.1 to 45.2 | 98 to 101 s | 16.91 GB |

Both Rust ways match the compiled Python model on all 64 tokens and on all three long-prompt answers.

---

#### What Is Being Measured?

TorchTPU (`torch_tpu`) is a PyTorch backend for TPUs. The question here is how far a model can move out of Python and still run on it. The Rust ways run on PyTorch 2.13.0 and the Python baseline on 2.14.1. The notebook swaps versions between them, for a reason Step 3 covers.

The model is Gemma 4 E2B in the W4A16 repack `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`, with the per-layer table from its `-text-emb4` twin. The Python baseline comes from notebook 05, the ring cache in [PyTorch on TPU: Hands On With Colab](https://dev.to/gde/pytorch-on-tpu-hands-on-with-colab-5b5g).

Versions on the Colab image: Python 3.13.18, `torch_tpu` 0.2.0, `libtpu` 0.0.50, Transformers 5.20.2, `compressed-tensors` 0.19.0, `rustc` 1.99.1. Each way runs in a fresh Python process, so none inherits another's compiled programs.

---

#### Step 1 — Run the Notebook

Open [notebook 06](https://colab.research.google.com/github/xbill9/torchtpu-dev/blob/main/colab/notebooks/06_e2b_rust_on_torchtpu.ipynb) in Colab, pick **Runtime → Change runtime type → v5e-1 TPU**, then **Runtime → Run all**. No Hugging Face token is needed. The notebook installs Rust and builds the crate on the VM:

```text
Finished `release` profile [optimized] target(s) in 1m 44s
rustc 1.99.1 (58cf8f9c3 2026-10-13)
Rust installed and crate built in 122 s
```

From a terminal, the Colab CLI runs it on a real v5e-1 and keeps the executed copy:

```shell
git clone https://github.com/xbill9/torchtpu-dev
cd torchtpu-dev/colab
make setup
python3 tools/verify_on_tpu.py 06_e2b_rust_on_torchtpu
```

```text
OK: 06_e2b_rust_on_torchtpu ran clean on v5e1 -> runs/2026-11-01/06_e2b_rust_on_torchtpu_output.ipynb

1/1 notebooks verified
```

---

#### Step 2 — How Rust Reaches the Chip

TorchTPU 0.2.0 ships no supported C++ API, and it links `libtorch_python`, so Python has to host the process. Importing `torch_tpu` registers TorchTPU's kernels on libtorch's dispatcher. The [`tch`](https://github.com/LaurentMazare/tch-rs) crate calls that same libtorch, loaded once in the same process, so a Rust call on a `tpu` tensor lands on the TorchTPU kernel.

The bridge is a `pyo3` extension. It unpacks a Python tensor, runs ATen calls and hands the result back:

```rust
#[pyfunction]
fn matmul_sum(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let x = take(x)?;
    give(py, x.matmul(&x).sum(Kind::Float))
}
```

Run on the chip:

```text
Rust matmul then sum: 64.0 on tpu:0
built in Rust, moved: [0.0, 1.0, 2.0, 3.0, 4.0] on tpu:0
bf16 kept: torch.bfloat16
```

The result stays on `tpu:0`, and bf16 survives the trip.

---

#### 🔎 Tip: `tch` Has No TPU Device

`tch::Device` has no `tpu` variant. `type_as` follows any tensor already on the chip, so the crate builds on the host and moves by example:

```rust
/// Host to device without naming the device: tch has no `tpu`, but `type_as` follows the tensor.
fn on(t: &Tensor, like: &Tensor) -> Tensor {
    t.type_as(&like.to_kind(t.kind()))
}
```

Python passes in `torch.zeros(1, device="tpu")` and every tensor follows it. Avoid `Tensor::device()` on these tensors for the same reason.

---

#### Step 3 — Pin PyTorch to What `tch` Generates

`tch` 0.26 is generated for PyTorch 2.13.0, and its bindings fail to compile on 2.14.1:

```text
'qr' is not a member of 'torch'
```

So the crate builds against 2.13.0, which TorchTPU 0.2.0 supports, with `LIBTORCH_USE_PYTORCH=1` pointing the build at the installed PyTorch:

```toml
[dependencies]
tch = { version = "0.26.0", features = ["python-extension"] }
torch-sys = { version = "0.26.0", features = ["python-extension"] }
pyo3 = { version = "0.30", features = ["extension-module"] }
safetensors = "0.8.0"
memmap2 = "0.9.12"
```

---

#### Step 4 — The Model in Rust

The crate is about 580 lines. `weights.rs` memory-maps the safetensors files and unpacks the compressed-tensors 4-bit format to bf16. `gemma.rs` holds the model: 35 layers, of which 15 own keys and values and 20 reuse them, the same layout as the Python ring.

The ring's attention masks every slot that is empty or out of the window, so a step is the same program at every position:

```rust
fn attend(&self, q: &Tensor, pos: &Tensor) -> Tensor {
    let scores = q.matmul(&self.k.tr()).to_kind(Kind::Float);
    let (kpos, qpos) = (self.kpos.unsqueeze(0), pos.unsqueeze(1));
    let mut keep = kpos.ge(0).logical_and(&kpos.less_equal_tensor(&qpos));
    if let Some(w) = self.window {
        keep = keep.logical_and(&kpos.greater_tensor(&(&qpos - w)));
    }
    let weights = scores.masked_fill(&keep.logical_not().unsqueeze(1), f64::NEG_INFINITY);
    // softmax, then the values
```

Python starts it with a directory and a tensor on the chip:

```python
import torch_tpu
import torchtpu_rs

engine = torchtpu_rs.Engine(MODEL_DIR, EMB4_DIR, torch.zeros(1, device="tpu"), 33024)
tokens, read_s, decode_s = engine.generate(ids, 64, 1)
```

The last argument sets how often tokens come back to the host: 1 streams, 8 reads in batches of eight.

---

#### Step 5 — Check It Against Transformers

Before any TPU time, the Rust engine runs on CPU beside Hugging Face Transformers, on PyTorch 2.13.0 with the real checkpoint:

```text
load 6.8 (3, 12, 20, 1024)
top5 rust [818, 102500, 4420, 16520, 18318] [25.75, 18.875, 18.75, 18.5, 18.125]
top5 hf   [818, 102500, 4420, 16520, 18318] [25.625, 18.75, 18.625, 18.25, 18.0]
max abs diff 0.375 KL(hf||rs) 1.8714923498919234e-05
same 64 of 64
first diff None
```

The top five candidates are the same, the KL divergence is 1.9e-5, and all 64 tokens match. The layout line reads 3 full-attention rings, 12 sliding rings and 20 shared layers.

---

#### Step 6 — The Three Ways on the Chip

Final run of notebook 06:

```text
way                       load s  first s  graphs  breaks  streamed  in batches  peak GB
Rust fused, torch 2.13      23.1     83.5       -       -      15.1        15.2     7.75
Rust eager, torch 2.13      23.8     22.8       -       -       4.8         4.8     8.62
Python compile, 2.14.1      25.9    101.1       2       0      33.5        45.1    16.91
Rust fused, torch 2.13: first 64 of 64 tokens identical to Python compile on 2.14.1
Rust eager, torch 2.13: first 64 of 64 tokens identical to Python compile on 2.14.1
```

**Defer-and-fuse is the fast mode for Rust.** `torch.compile` traces Python, so it cannot see Rust. `TORCH_TPU_DEFER_AND_FUSE=1` works below ATen and needs no change to the crate. It takes Rust from 4.8 to 15.1 tokens per second streamed, about 3.2x, at the same tokens. In this run Rust fused streams at 0.45x the compiled Python step and batched Python reaches 2.97x Rust (arithmetic, `evidence/arithmetic.txt`).

Compiling costs Python more memory: its 16.91 GB peak against 7.75 GB for Rust fused is 2.18x, close to the chip's 16 GiB.

---

#### 🔎 Tip: Don't Run `torch.compile` on 2.13 For This Model

The compiled Python ring needs 2.14.1. On 2.13.0 the same source behaves differently:

```text
Python compile, 2.13        25.6     34.6      65      64      15.9        16.2    11.26
Python compile, 2.13: first 21 of 64 tokens identical to Python compile on 2.14.1
```

It splits into 65 programs with 64 graph breaks and writes a different reply from token 21. Changing Transformers and `compressed-tensors` versions made no difference, and changing the PyTorch version did. Its cause is untested. The Rust ways on 2.13.0 are unaffected and match 2.14.1.

---

#### Step 7 — Long Prompts

Notebook 05's prompts fill the context with numbered records, each with a random code, and ask about one record:

```text
way                       prompt  read s read tok/s  decode   answer  found
Rust fused, torch 2.13        61     0.1        776    14.6   915965    yes
Rust fused, torch 2.13      4081     1.0       4172    14.8   610395    yes
Rust fused, torch 2.13     32761     8.0       4116    15.6   319550     no
Rust eager, torch 2.13     32761    28.5       1148     4.8   319550     no
Python compile, 2.14.1     32761     5.0       6555    45.1   319550     no
```

Rust fused reads the 32,761-token prompt in 7.6 to 8.0 s, the compiled ring in 5.0 s and Rust eager in about 29 s. All three ways give the same three answers. The third is wrong in every way, the same code the Python ring gives, so the limit belongs to the model at this depth. Decode speed holds steady at every depth, because each step attends over every slot of a cache of the same size.

---

#### Compare and Contrast

| One v5e chip, Gemma 4 E2B | Streamed tok/s | Model code | PyTorch | Peak memory |
| :--- | ---: | :--- | :---: | ---: |
| Python `torch.compile` 🥇 | 33.4 to 33.5 | Python | 2.14.1 | 16.91 GB |
| Rust, defer-and-fuse 🥈 | 14.7 to 15.7 | Rust | 2.13.0 | 7.75 GB |
| Rust, strict eager 🥉 | 4.8 to 5.0 | Rust | 2.13.0 | 8.62 GB |

---

#### So, Which One?

For speed on this chip, stay with the compiled Python ring on PyTorch 2.14.1: 45 tokens per second in batches and 33 streamed.

For a model in Rust, set `TORCH_TPU_DEFER_AND_FUSE=1`: about 15 tokens per second streamed, the same tokens and answers, and less than half the peak memory. TorchTPU marks that variable experimental, so it may change.

Keep strict eager for debugging, at about 5 tokens per second.

---

#### Summary

The goal of this article was to run Gemma 4 E2B on a TPU with the model written in Rust, through TorchTPU and without JAX. The key to the solution was `tch` sharing the one libtorch that TorchTPU registers its kernels with, so a Rust ATen call on a `tpu` tensor reaches a TorchTPU kernel. The Colab v5e-1 results were:

- 🟢 Rust matches compiled Python on 64 of 64 tokens and 3 of 3 long-prompt answers
- 🟢 Rust matches Transformers on CPU on 64 of 64 tokens, with a KL divergence of 1.9e-5
- 🟢 One environment variable takes Rust from about 5 to about 15 tokens per second streamed
- 🟢 Rust peaks at 7.75 GB against 16.91 GB for compile
- ⚠️ Rust fused streams at 0.45x compiled Python, because `torch.compile` cannot trace Rust
- ⚠️ `tch` 0.26 builds against PyTorch 2.13.0 only
- ⚠️ Python still hosts the process, builds the prompts and tokenizes
- ❌ `torch.compile` on PyTorch 2.13.0 splits this model into 65 programs and changes the reply

Scope: Gemma 4 E2B-it QAT only, text only, one Colab v5e-1 runtime per run on 2026-11-01, greedy decoding, one sequence at a time, four prompts, a 33,024-token context. Rust ranges cover three runs and compiled Python ranges two. The earliest Rust run used Transformers 5.21.0 and `compressed-tensors` 0.20.1, and the rest 5.20.2 and 0.19.0. The Rust ways ran on PyTorch 2.13.0 and the Python baseline on 2.14.1, one version apart. The 131,328-token context, a serving endpoint, a Rust Colab kernel, and where the time goes between Rust fused and compile were not run. The final crate formatting commit was built and checked on CPU, and the TPU runs used the source before it.

The strategy for running a Rust model on TorchTPU was validated with an incremental step by step approach.

---

#### References

- Rust crate: https://github.com/xbill9/torchtpu-dev/tree/main/colab/rust
- Notebook 06: https://github.com/xbill9/torchtpu-dev/blob/main/colab/notebooks/06_e2b_rust_on_torchtpu.ipynb
- Executed run: https://github.com/xbill9/torchtpu-dev/tree/main/colab/runs/2026-11-01
- PyTorch on TPU: Hands On With Colab: https://dev.to/gde/pytorch-on-tpu-hands-on-with-colab-5b5g
- tch: https://github.com/LaurentMazare/tch-rs
- W4A16 repack: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text
- 4-bit-tables repack: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4
- Colab CLI: https://github.com/googlecolab/google-colab-cli
