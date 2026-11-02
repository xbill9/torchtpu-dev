# The PyO3 bindings

The model is Rust. Python hosts the process, because TorchTPU 0.2.0 links `libtorch_python` and ships no supported C++ API, and it runs the HTTP server and the tokenizer. [PyO3](https://github.com/PyO3/pyo3) 0.30 builds this crate as a native Python module, `torchtpu_rs`, and turns Rust functions and structs into Python functions and classes.

TorchTPU's kernels are registered on libtorch's dispatcher, so the model reaches them by calling ATen. It does that through a shim of its own rather than through the `tch` crate. `tch` 0.26 wraps every ATen op in generated C++ and compiles only against PyTorch 2.13.0. The shim wraps the 40-odd ops this model uses and compiles against the torch that is installed.

| File | Job | Python objects |
| --- | --- | --- |
| `src/aten.cpp` | `extern "C"` wrappers around the ATen ops the model uses, plus `torch.Tensor` conversion through libtorch_python | `PyObject*` in the two conversion functions only |
| `src/aten.rs` | A Rust `Tensor` over those wrappers: ops, operators, `no_grad`, host reads | none |
| `src/weights.rs` | safetensors and the 4-bit format | none |
| `src/gemma.rs` | The model, the ring cache, the decode loop | none |
| `src/bindings.rs` | Every PyO3 item | all of them |
| `python/torchtpu_rs.pyi` | The type stub | |

#### How the module is built and loaded

| Piece | Setting | Why |
| --- | --- | --- |
| `[lib] crate-type` | `cdylib` | A shared library Python can `import` |
| `pyo3` features | `extension-module` | Does not link libpython; the interpreter that imports the module provides it |
| `build.rs` | Runs `PYO3_PYTHON` (else `python3`) to ask its `torch` for headers, libraries, C++ ABI and version | The shim is compiled for that torch, whatever its version |
| C++ standard | `-std=c++20` | torch 2.14 headers require it |
| Links | `torch_python`, `torch_cpu`, `torch`, `c10`, with an rpath to `torch/lib` | The shim's ATen and THPVariable symbols |
| Toolchain | `rust-toolchain.toml`: the newest nightly; edition 2024 | |
| `#[pymodule] fn torchtpu_rs` | in `src/lib.rs` | The function name is the module name, and must match the file name: the build copies `libtorchtpu_rs.so` to `torchtpu_rs.so` |

The module only loads into the torch it was built against. At import, `bindings.rs` compares `torch.__version__` with the version the shim was compiled for and raises ImportError if they differ: a different torch has a different C++ ABI. `torchtpu_rs.BUILD` records the torch, rustc, Python and crate versions that built the module.

Import order at run time: `import torch` loads libtorch, `import torch_tpu` registers the TPU kernels on its dispatcher, and only then `import torchtpu_rs`, whose ATen calls go through that same dispatcher.

#### What Python can call

| Python | Rust | Arguments in | Result out |
| --- | --- | --- | --- |
| `torchtpu_rs.__version__` | `env!("CARGO_PKG_VERSION")` | | `str` |
| `torchtpu_rs.BUILD` | set by `build.rs` | | `dict`: `torch`, `rustc`, `python`, `crate` |
| `torchtpu_rs.CHUNK` | `gemma::CHUNK` | | `int`, 256 |
| `matmul_sum(x)` | `#[pyfunction]` | `torch.Tensor` | `torch.Tensor`, same device |
| `arange_like(n, like)` | `#[pyfunction]` | `int`, `torch.Tensor` | `torch.Tensor` on `like`'s device |
| `Engine(model_dir, emb4_dir, like, context)` | `#[pyclass(unsendable)]`, `#[new]` | `str` or path, `str` or path, `torch.Tensor`, `int` | an `Engine` |
| `Engine.context` | `#[getter]` | | `int` |
| `Engine.layout()` | `#[pymethods]` | | `tuple[int, int, int, int]` |
| `Engine.logits(ids)` | `#[pymethods]` | `list[int]` | `torch.Tensor`, 1-D, on the device |
| `Engine.generate(ids, max_new, read_every=1)` | `#[pymethods]`, `#[pyo3(signature)]` | `list[int]`, `int`, `int` | `tuple[list[int], float, float]` |
| `Engine.decode(ids, max_new, stop, read_every=1, on_tokens=None)` | `#[pymethods]`, `#[pyo3(signature)]` | `list[int]`, `int`, `list[int]`, `int`, callable or `None` | `DecodeResult` |
| `DecodeResult` | `#[pyclass(frozen, get_all)]` | | `.tokens`, `.finish`, `.read_s`, `.decode_s` |

#### The four crossings

**1. Token ids are plain Python lists.** `Vec<i64>` arguments accept any sequence of ints, and `Vec<i64>` results come back as `list[int]`. Generated tokens never cross as tensors: the decode loop reads them from the TPU into a Rust `Vec` (`Tensor::to_vec_i64`, one copy to the host), and PyO3 turns that into a list. A str or a float where an int belongs raises TypeError before any Rust runs.

**2. Tensors share storage.** A `torch.Tensor` and the shim's Rust `Tensor` both wrap a C++ `at::Tensor`. `take` in `bindings.rs` unpacks the Python object (`THPVariable_Unpack`, in `aten.cpp`) and `give` wraps one back (`THPVariable_Wrap`) into a `torch.Tensor` that Python owns. Neither copies, so a tensor on the TPU stays there. Both run with the GIL held. Only three calls take or return a tensor: `like` (which names the device for every weight: nothing in Rust names `tpu`, and `type_as` follows a tensor already there), `logits`, and the two test functions.

**3. Long calls release the GIL.** `Engine(...)`, `logits`, `generate` and `decode` run inside `py.detach(...)`, so other Python threads (the server's event loop) keep running while the TPU works. A closure that runs without the GIL must be `Send`. The Rust `Tensor` is `Send` (at::Tensor's reference count is atomic) but not `Sync`, which is why `Engine::new` moves `like` into its closure rather than borrowing it.

**4. The callback takes the GIL back.** `decode` calls `on_tokens(list[int])` each time a batch of tokens reaches the host, inside `Python::attach(...)`. A truthy return continues; a false one ends the decode with `finish="cancelled"`. If the callback raises, the decode stops and `decode` raises that same exception. `read_every` sets the batch size: 1 streams token by token, a larger value runs longer fused programs.

#### Errors

| Situation | Python sees |
| --- | --- |
| A torch other than the one the module was built against | `ImportError` at `import torchtpu_rs` |
| Empty `ids`, `max_new` or `read_every` below 1, prompt plus `max_new` over the context | `ValueError`, checked in `bindings.rs` before `gemma.rs` asserts |
| `context` not a positive multiple of 256, or a directory that does not exist | `ValueError` |
| `like` that is not a `torch.Tensor` | `RuntimeError` |
| An `Engine` used from a thread other than the one that built it | `pyo3_runtime.PanicException`, a `BaseException` subclass: `except Exception` does not catch it |
| An exception inside `on_tokens` | That exception, raised from `decode` |
| An ATen error, a missing tensor or a bad checkpoint | `PanicException` carrying the C++ exception text or the failed `unwrap` |

#### Why `unsendable`

The engine owns TPU tensors and one ring cache that a single sequence at a time writes into. `#[pyclass(unsendable)]` makes PyO3 check the calling thread on every access. `rust_openai_server.py` therefore builds the engine on one dedicated thread, sends each request to that thread through a queue, and hands tokens back to the asyncio event loop with `loop.call_soon_threadsafe`. Requests run one at a time, in arrival order.

#### Checked

On a laptop CPU on 2026-11-02, built with rustc 1.101.0-nightly (2026-11-01) against torch 2.14.1+cpu on Python 3.14.7:

- `tests/test_rust_bindings.py` passed all 8 tests (249 s): the module surface and `BUILD`; stop tokens; streamed batches joining to the returned tokens; cancel and length; an exception in the callback; four bad inputs giving `ValueError`; `logits` agreeing with the first decoded token; `PanicException` from a second thread; and another Python thread running during a decode.
- Against Hugging Face Transformers 5.20.2 on the same torch, 64 greedy tokens matched 64 of 64 and the KL divergence of the first-token logits was 1.08e-5 (`../evidence/cpu_parity_shim_torch2.14.1.txt`).
- With `tch` on torch 2.13.0 the earlier build measured a second thread waking 747 times during one 37.4 s decode.
