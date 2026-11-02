//! Gemma 4 E2B in Rust on TorchTPU, as a Python extension module named `torchtpu_rs`.
//!
//! Python imports `torch` and `torch_tpu`; `torch_tpu` registers the TPU kernels on libtorch's
//! dispatcher. This module links that same libtorch through a small C++ shim (`aten.cpp`,
//! wrapped by `aten.rs`), so an ATen call made here on a `tpu` tensor runs as a TorchTPU kernel.
//!
//! Four files, four jobs:
//! - `aten.cpp` / `aten.rs` are the ATen operations the model uses, built against the installed
//!   torch (build.rs). No pinned PyTorch version.
//! - `weights.rs` reads safetensors and unpacks the 4-bit format. No Python.
//! - `gemma.rs` is the model, the ring cache and the decode loop. No Python.
//! - `bindings.rs` is every PyO3 item: the only place Python objects appear. `PYO3.md` beside
//!   `Cargo.toml` describes each crossing, and `python/torchtpu_rs.pyi` is the type stub.
mod aten;
mod bindings;
mod gemma;
mod weights;

use pyo3::prelude::*;

/// The module Python imports as `torchtpu_rs`. Its name has to match the shared library's file
/// name (`torchtpu_rs.so`), which is why the build copies `libtorchtpu_rs.so` to that name.
#[pymodule]
fn torchtpu_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bindings::register(m)
}
