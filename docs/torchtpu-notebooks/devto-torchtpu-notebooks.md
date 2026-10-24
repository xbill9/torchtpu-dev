---
title: "PyTorch on TPU: Hands On With Colab"
published: false
description: "Five Colab notebooks take Gemma 4 E2B on one v5e chip from 5 tokens per second under Hugging Face generate to 50 with one compiled decode step, and to the full 131,072-token context with a ring cache, against a JAX engine on the same chip."
tags: pytorch, tpu, gemma, googlecloud
cover_image: https://raw.githubusercontent.com/xbill9/torchtpu-dev/main/docs/torchtpu-notebooks/devto-cover.36e8e15c.jpg
---

This article provides a step by step guide to running Gemma 4 E2B, Google's quantization-aware trained (QAT) model, in PyTorch on a Google Cloud TPU with TorchTPU. Every step is a Colab notebook that runs on a Colab v5e-1 runtime, one TPU v5e chip with 16 GB of high-bandwidth memory (HBM). The measures are what TorchTPU does with PyTorch code, how far Hugging Face Transformers on TorchTPU trails a JAX engine on the same chip, what closes the gap, and how much context fits.

https://github.com/xbill9/torchtpu-dev/tree/main/colab

**One compiled decode step takes PyTorch from 5 to 50 tokens per second, and a ring cache takes it to the full context.** Same chip, same checkpoint, same 19-token prompt, 64 greedy tokens (32 in notebook 02):

| One Colab v5e-1 chip | Decode tok/s | Context | Notebook |
| :--- | ---: | ---: | :---: |
| Hugging Face `generate` | 5.0 to 5.3 | short prompt | 04 |
| Best way before the fix (defer-and-fuse) | 8.8 to 9.1 | short prompt | 02, 03 |
| Compiled decode step | 31.3 to 33.4 | 256 tokens | 04 |
| Compiled step, 4-bit per-layer table | **50.2 to 50.4** | 256 tokens | 04 |
| Ring cache | **38.6 to 38.7** | **131,328 tokens** | 05 |
| JAX engine, bf16 weights | 130.8 to 132.9 | short prompt | 03, 04 |

JAX was 14.8x the best PyTorch way in notebook 03 and is 2.6x the compiled step with the 4-bit table in notebook 04. On that prompt every way in notebooks 03 to 05 writes the same 56-token answer.

---

#### What Is Being Measured?

TorchTPU (`torch_tpu`) is a PyTorch backend for TPUs: tensors live on a `tpu` device, and the same Python code runs eagerly, fused, or under `torch.compile`. The question for a PyTorch user is how much of the TPU a plain Transformers model gets, measured against a JAX engine that serves the same checkpoint on the same chip.

The model is Gemma 4 E2B in the W4A16 repack `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`: Google's QAT 4-bit weights in the compressed-tensors format. Its twin `-text-emb4` stores the vocabulary tables 4-bit as well. The JAX engine is `xbill9/tpu-jax` at commit `4b9f8e9`.

Every number below comes from an executed notebook in `colab/runs/`. Versions on the Colab image: `torch_tpu` 0.2.0, `torch` 2.14.1+cpu, `libtpu` 0.0.50, `jax` 0.11.2, `compressed-tensors` 0.19.0, Transformers 5.20.1 (notebooks 02 and 03) and 5.20.2 (04 and 05). Each way runs in a fresh Python process, so none inherits another's compiled programs.

---

#### Step 1 — Run the Notebooks

Each notebook opens in Colab from its badge in the repository's `colab/README.md`. Pick **Runtime → Change runtime type → v5e-1 TPU**, then **Runtime → Run all**. No Hugging Face token is needed.

From a terminal, the Colab CLI runs the same notebooks on a real v5e-1 and keeps the executed copy:

```shell
git clone https://github.com/xbill9/torchtpu-dev
cd torchtpu-dev/colab
make setup
python3 tools/verify_on_tpu.py 05_e2b_ring_cache
```

```text
OK: 05_e2b_ring_cache ran clean on v5e1 -> runs/2026-10-23/05_e2b_ring_cache_output.ipynb

1/1 notebooks verified
```

---

#### Step 2 — TorchTPU's Three Modes

Notebook 01 runs one stack of matrix multiplies, 275 GFLOP per step, in each mode:

```text
mode      first call s   step ms  TFLOP/s       checksum
eager             0.15      2.91     94.6          359.0
fuse              2.69      1.68    163.2          359.0
compile           3.00      1.91    143.6          359.0
fuse vs eager: step 1.72x faster, checksum differs by 0.0e+00
compile vs eager: step 1.52x faster, checksum differs by 0.0e+00
```

**Strict eager** runs each operation as it is called. **Defer-and-fuse** (`TORCH_TPU_DEFER_AND_FUSE=1`) collects operations and compiles them in groups. **`torch.compile(backend="tpu")`** compiles a whole function. All three give the same checksum. On a stack of tiny operations, compile is 98.69x strict eager and defer-and-fuse 1.62x.

---

#### 🔎 Tip: Shapes Decide the Compile Count

A compiled function compiles once per input shape. Notebook 01 feeds four lengths:

```text
raw: 4 compiled graph(s)
  first pass ms:  [341.2, 115.7, 116.7, 143.8]
  second pass ms: [1.1, 1.0, 0.8, 0.8]

bucketed: 2 compiled graph(s)
  first pass ms:  [120.5, 1.0, 1.0, 143.7]
  second pass ms: [0.9, 0.9, 0.8, 0.9]
```

Padding to 128-token buckets gives 2 programs instead of 4, with a largest difference of 0 against the unpadded results. A first call costs 115 to 341 ms and a warm one about 1 ms, so every new shape in a decode loop is a new compile. Reading one value back to the host with `.item()` each step costs 1.62x under defer-and-fuse.

---

#### Step 3 — Load the Repack Onto the Chip

Notebook 02 loads the repack. Two settings decide whether the model reaches the TPU at all:

```text
claim                                                          expected            today
tensor names matched as stored / mapped                0 / all but tied          0 / 540
tensors on TPU after model.to('tpu')                                all                0
tensors on TPU with device_map='tpu'                                all             1098
```

The repack's tensors are named for the multimodal model (`model.language_model.*`) and the text model looks for `model.*`, so none of the 540 match until `key_mapping={r"^model\.language_model\.": "model."}` maps them. `model.to('tpu')` leaves every tensor on the host; `device_map="tpu"` in `from_pretrained` puts all 1,098 on the chip.

---

#### Step 4 — Every Way to Generate, Out of the Box

Notebook 02 then runs `generate` four ways, 32 greedy tokens:

```text
way            first call s  warm call s  warm tok/s  TPU memory GB
eager                 16.21         6.99         4.6           9.85
fuse                 359.49         3.50         9.1          10.83
compressed            27.80         6.85         4.7           9.85
buffer                20.66         4.83         6.6           9.83
```

All four write identical tokens. Defer-and-fuse is the fastest warm way at 9.1 tokens per second, and its first call takes 359 s, because `generate` grows the sequence by one token per step and every length is a new shape. A compiled step over a fixed buffer runs at 6.6.

---

#### Step 5 — Against JAX on the Same Chip

Notebook 03 runs the `tpu-jax` engine and PyTorch side by side, on the same checkpoint, prompt and `libtpu`:

```text
engine         load s  first 64 s  warm 1 s  warm 64 s  decode tok/s  memory GB
jax_w4a16         7.3       39.50     0.014      0.625         103.1       6.62
jax_bf16         16.1       23.39     0.010      0.492         130.8       9.30
torch_eager      21.2       20.98     0.203     13.531           4.7       9.87
torch_fuse       42.8      658.66     0.127      7.268           8.8      11.91
torch_buffer     20.0       11.57     0.167     10.113           6.3       9.83
```

Decode speed is 63 tokens over the warm 64-token time minus the warm 1-token time, so the prompt drops out. JAX with 4-bit weights is 11.7x the fastest PyTorch way and JAX with bf16 weights 14.8x.

The two stacks compute the same model. Over 83 positions the mean KL divergence between JAX bf16 and PyTorch eager is 0.001073, with the same top token at every position. All five engines write the same 56-token answer.

---

#### Step 6 — Where the 4-Bit Weights Go

Notebook 04 starts with memory. Loading the repack with its weights left 4-bit, and watching the chip:

```text
after load:             6.56 GB   packed weights present: True
after forward pass 1:   9.83 GB
after forward pass 2:   9.83 GB   packed weights present: False
weight dtype afterwards: torch.bfloat16
```

Transformers' compressed-tensors integration unpacks every layer to bf16 at the first forward pass and keeps the result. After that the model is the bf16 model, so every way in this article computes in bf16, JAX bf16 included. Keeping a table 4-bit on the chip takes code that reads the packed values itself (Step 9).

---

#### Step 7 — Why the Compiled Step Split Into 24 Graphs

A compiled step with `fullgraph=True` raises at its first graph break and names the line:

```text
modules with a wrapped forward: 542
break: Data pointer comparison
   File "/usr/local/lib/python3.13/dist-packages/compressed_tensors/offload/utils.py", line 45, in send_tensors
   -> if tensor.data_ptr() == value.data_ptr():
```

compressed-tensors wraps the forward of all 542 modules so each module's inputs move to its device first, and `torch.compile` cannot trace a comparison of memory addresses, so it cuts the program at every one. On one chip every tensor is already on the TPU, and the package's own `remove_dispatch` takes the wrappers off:

```python
from compressed_tensors.offload.dispatch import remove_dispatch

remove_dispatch(model)  # 542 wrapped modules before, 0 after
```

---

#### Step 8 — One Compiled Decode Step

The step is one token in, the next token out, over Hugging Face's `StaticCache`, whose tensors never change shape. Transformers' sliding-window cache layer counts its length in a Python integer, which a compiled step treats as a constant and recompiles on. While the cache is no longer than the 512-token window, the plain static layer's methods give the same result with a counter on the chip:

```python
for method in ("update", "get_seq_length", "get_mask_sizes"):
    setattr(cu.StaticSlidingWindowLayer, method, getattr(cu.StaticLayer, method))
cache = StaticCache(config=model.config, max_cache_len=256)

def step():
    return model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1)

step = torch.compile(step, backend="tpu", dynamic=False)
```

The seven ways of notebook 04, two runs, 64 tokens:

| Way | tok/s | x `generate` | Graphs / breaks |
| :--- | ---: | ---: | ---: |
| `generate` | 5.0 to 5.3 | 1.0 | |
| Static cache, strict eager | 4.2 to 4.5 | 0.8 to 0.9 | 0 / 0 |
| Static cache, defer-and-fuse | 10.6 to 11.0 | 2.0 to 2.2 | 0 / 0 |
| Compiled step, wrappers on | 6.4 to 6.6 | 1.2 to 1.3 | 24 / 60 |
| Compiled step 🥉 | 31.3 to 33.4 | 5.9 to 6.7 | 1 / 0 |
| Compiled step, 4-bit table 🥈 | 50.2 to 50.4 | 9.4 to 10.1 | 1 / 0 |
| JAX bf16 🥇 | 131.2 to 132.9 | 25.0 to 26.2 | |

With the wrappers removed the step compiles once and every later token reuses it. Its first 64-token call, compiling included, takes 36.5 to 38.3 s.

---

#### Step 9 — Keep the Largest Table 4-Bit

Gemma 4's per-layer embedding table is the largest tensor in E2B. `-text-emb4` stores it 4-bit, and a small module looks up packed rows and unpacks only those:

```python
def forward(self, ids):
    p, s = self.packed[ids], self.scale[ids]
    q = ((p.unsqueeze(-1) >> self.shifts) & 15).to(torch.float32) - 8
    w = (q.view(*ids.shape, s.shape[-1], -1) * s.unsqueeze(-1).to(torch.float32)).to(torch.bfloat16)
    return w.view(*ids.shape, -1) * torch.tensor(self.embed_scale, dtype=torch.bfloat16)
```

It frees 3.4 GB (9.82 GB to 6.44 GB after load) and makes the compiled step 1.51x to 1.60x faster (arithmetic over the two runs, `evidence/arithmetic.txt`), writing the same tokens: Google's QAT checkpoint already stores that table on a 4-bit grid.

---

#### 🔎 Tip: Streaming Sets Its Own Pace

A server that streams its reply reads each token back to the host as it is made. Read back that way, the compiled step runs at 31.8 to 36.5 tokens per second and the 4-bit-table step at 32.4 to 37.2, so each token's round trip to the host sets the pace and the 4-bit table's lead disappears. Collect tokens on the chip and read them at the end where the client allows it.

---

#### Step 10 — A Ring Cache for the Full Context

Notebook 05 replaces Transformers' cache with the layout a 4 GB laptop GPU engine uses to hold E2B's whole context ([gemma4-dev](https://github.com/xbill9/gemma4-dev), `local-pytorch-1650ti-2b-q4w4a16`). Only 15 of the 35 layers compute keys and values:

```text
full     layers:  3    33024 slots  head size 512
sliding  layers: 12      768 slots  head size 256
shared   layers: 20    reuses an earlier layer's  head size 
sliding window: 512 positions
cache for a 33,024-token context: 0.212 GB
cache for a 131,328-token context: 0.816 GB
```

Full-attention layers keep one slot per position. Sliding layers look back 512 positions, so each keeps a ring of 512 + 256 slots that records which position each slot holds: room for the window plus one 256-token piece of the prompt, so writing a piece never overwrites a key the same piece still needs. The other 20 layers reuse the last full and last sliding layer's keys and values. Prompts are read in fixed 256-token pieces, so the model compiles to two programs, one for a piece and one for a decode step, whatever the prompt's length.

The ring replaces `Gemma4TextAttention.forward`; its attention is plain PyTorch with a mask:

```python
def attend(self, q, pos):
    scores = torch.einsum("nhd,kd->hnk", q, self.k).float()
    kpos, qpos = self.kpos[None, :], pos[:, None]
    keep = (kpos >= 0) & (kpos <= qpos)
    if self.window:
        keep = keep & (kpos > qpos - self.window)
    weights = torch.softmax(scores.masked_fill(~keep[None], float("-inf")), dim=-1).to(q.dtype)
    return torch.einsum("hnk,kd->nhd", weights, self.v)
```

On the short prompt the ring writes the same 64 tokens as notebook 04's step:

```text
way                            first call s   tok/s  streamed  graphs  breaks
notebook 04 step, 256 slots                    47.8                 1       0
ring, 33,024                           92.1    45.2      33.3       2       0
ring, 131,328                          90.7    38.7      30.7       2       0
ring, 33,024: first 64 of 64 tokens identical to notebook 04's step (the answer ends at token 56)
ring, 131,328: first 64 of 64 tokens identical to notebook 04's step (the answer ends at token 56)
```

Notebook 05 times decode from the first token to the 64th, collected at the end, which puts notebook 04's step at 47.8 to 47.9.

---

#### Step 11 — Long Prompts and Recall

The long prompts fill the context with numbered records, each with a random six-digit code, and ask for the code of the record a quarter of the way in. They come from the same generator, seed and order as the laptop engine's long-context run, so its answers apply unchanged:

```text
 context   prompt  read s read tok/s  decode   answer  found  record  laptop answer  same
   33024       61     0.1       1218    45.2   915965    yes       1         915965   yes
   33024     4081     0.7       5978    45.2   610395    yes      33         610395   yes
   33024    32761     5.0       6601    45.2   319550     no     723         319550   yes
  131328    32761     9.9       3305    38.5   390185    yes     272         390185   yes
  131328   130981    39.8       3288    38.1   567239     no      91         567239   yes
graphs after every prompt: [2, 0] and [2, 0] (programs, breaks)
```

Across the runs of notebook 05 the 130,981-token prompt reads in 35.5 to 39.8 s. The laptop engine, on a GTX 1650 Ti, read it in 2,024 s and decoded at 21.8 tokens per second there. All five answers match the laptop engine's, the two wrong codes included, and llama.cpp gave the same three answers on the three prompts it was given: finding one record among thousands of near-identical ones is at the edge of what E2B does at 32k tokens and beyond.

---

#### 🔎 Tip: Decode Speed Follows the Cache, Not the Prompt

Every step attends over every slot and masks the empty ones, so a step is the same program at every position. Decode runs at 45.1 to 45.3 tokens per second at every depth with a 33,024-token context and 38.1 to 38.7 with 131,328. Reading follows the same rule: the same 32,761-token prompt reads at 50% to 56% of the speed with the larger context. Size the context to the work.

---

#### Step 12 — Why Notebook 04's Cache Cannot Stretch

Asked for 4,352 slots, Transformers' `StaticCache` gives the sliding layers 512 whatever the length:

```text
 3 layers of StaticLayer                 4352 slots
12 layers of StaticSlidingWindowLayer     512 slots
```

The sliding layer allocates `min(sliding_window, max_cache_len)` slots, and Step 8's fix writes each token at its position, so a 4,081-token prompt writes past the end of those 512 slots. Run on the chip, that recipe halted the TPU with a fatal error (`RuntimeUnexpectedCoreHalt`, XLA error E0200) the first time a result was read back. Beyond the window the cache needs its own layout, which is the ring.

---

#### What Does the Chip Hold?

```text
way                            after load GB with cache GB  peak GB
notebook 04 step, 256 slots             6.44                  16.67
ring, 33,024                            6.44          6.65    16.91
ring, 131,328                           6.44          7.26    16.91
```

The full 131,328-token context costs 0.82 GB beside 6.44 GB of model. The peak, 16.6 to 16.9 GB in every compiled way, comes from compiling: read right after the first call, it was 16.67 GB with a 33,024-token ring and 16.64 GB with a 131,328-token one. That is close to the chip's 16 GiB (17.2 GB), so a larger model meets the limit at compile time first.

---

#### Compare and Contrast

| One v5e chip, Gemma 4 E2B | Decode tok/s | First call | Longest context run | HBM after load |
| :--- | ---: | ---: | ---: | ---: |
| `generate`, defer-and-fuse | 8.8 to 9.1 | 359 s | 19-token prompt | 9.82 GB |
| Compiled step + 4-bit table 🥈 | 50.2 to 50.4 | 36.5 to 37.3 s | 256-token cache | 6.44 GB |
| Ring cache, 131,328 🥉 | 38.6 to 38.7 | 91 to 95 s | 130,981-token prompt | 6.44 + 0.82 GB |
| JAX bf16 🥇 | 130.8 to 132.9 | 23.4 to 23.5 s | 19-token prompt | 9.30 GB in use |

---

#### So, Which One?

For short chats, notebook 04's compiled step with the 4-bit per-layer table: a static cache, the sliding-window fix, the compressed-tensors wrappers removed and one `torch.compile` call. It pays under 40 s once and then decodes at 50 tokens per second, about 10x `generate`, writing the same tokens.

For anything past 512 tokens, notebook 05's ring cache: two programs, compiled once in 91 to 98 s, then the full 131,072-token context at 38 to 39 tokens per second, or 45 with a 33,024-token context. Both are plain PyTorch over Transformers' own Gemma 4 code, with no custom kernels.

For the most tokens per second on this chip, the JAX engine, 2.6x the fastest PyTorch way.

---

#### Summary

The goal of this article was to find how much of a TPU v5e chip PyTorch gets on Gemma 4 E2B through TorchTPU and Hugging Face Transformers, measured against a JAX engine on the same chip. The key to the solution was one compiled decode step: a static cache, compressed-tensors' module wrappers removed, and a ring cache for long context. The results were:

- 🟢 One compiled decode step decodes at 31.3 to 33.4 tokens per second against 5.0 to 5.3 for `generate`, and at 50.2 to 50.4 with the per-layer table kept 4-bit
- 🟢 The gap to JAX bf16 shrinks from 14.8x to 2.6x on the same chip, prompt and stopwatch
- 🟢 Every way writes the same 56-token answer; mean KL between JAX bf16 and PyTorch eager is 0.001073
- 🟢 The ring cache holds the full context in 0.82 GB, reads a 130,981-token prompt in 35.5 to 39.8 s, and decodes at 38.6 to 38.7 tokens per second
- 🟢 Five recall answers identical to a GTX 1650 Ti engine's on the same prompts
- ⚠️ Streaming each token to the host caps both compiled steps at 31.8 to 37.2 tokens per second
- ⚠️ Compiling peaks at 16.6 to 16.9 GB, close to the chip's 16 GiB
- ❌ A 4-bit load saves no memory: Transformers unpacks it to bf16 at the first forward pass
- ❌ Notebook 04's static cache past the 512-token window halts the chip

Scope: Gemma 4 E2B-it QAT only, as the W4A16 repack with its 4-bit-tables twin, text only, one Colab v5e-1 runtime per run on 2026-10-21 (notebooks 01 to 03) and 2026-10-23 (04 and 05), greedy decoding, one sequence at a time, one short prompt and five long ones; notebook 04 ran twice and notebook 05's figures cover three runs and one earlier run of the same code. JAX speeds are the `tpu-jax` engine at `4b9f8e9`. The laptop figures are from gemma4-dev's GTX 1650 Ti run on the same prompts, a different engine on different hardware. Batching, vLLM or SGLang on TorchTPU, and a serving endpoint were not run.

The strategy for running PyTorch on a TPU was validated with an incremental step by step approach.

---

#### References

- Notebooks and executed runs: https://github.com/xbill9/torchtpu-dev/tree/main/colab
- Notebook 01, execution modes: https://github.com/xbill9/torchtpu-dev/blob/main/colab/notebooks/01_torchtpu_execution_modes.ipynb
- Notebook 02, loading the repack: https://github.com/xbill9/torchtpu-dev/blob/main/colab/notebooks/02_e2b_repack_on_torchtpu.ipynb
- Notebook 03, JAX against TorchTPU: https://github.com/xbill9/torchtpu-dev/blob/main/colab/notebooks/03_e2b_jax_vs_torchtpu.ipynb
- Notebook 04, one compiled decode step: https://github.com/xbill9/torchtpu-dev/blob/main/colab/notebooks/04_e2b_compiled_decode.ipynb
- Notebook 05, the ring cache: https://github.com/xbill9/torchtpu-dev/blob/main/colab/notebooks/05_e2b_ring_cache.ipynb
- W4A16 repack: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text
- 4-bit-tables repack: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4
- JAX engine: https://github.com/xbill9/tpu-jax
- Laptop GPU engine: https://github.com/xbill9/gemma4-dev/tree/main/local-pytorch-1650ti-2b-q4w4a16
- Colab CLI: https://github.com/googlecolab/google-colab-cli
