//! Every PyO3 item in the crate. `gemma.rs` and `weights.rs` never see a Python object; this
//! file turns Python arguments into Rust values on the way in and Rust values into Python
//! objects on the way out. `PYO3.md` describes each crossing.
use crate::aten::{self, Device, Kind, Tensor};
use crate::gemma::{CHUNK, Decoded, Gemma};
use pyo3::exceptions::{PyImportError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// --- Tensors across the boundary -------------------------------------------------------------
//
// A `torch.Tensor` wraps a C++ `at::Tensor`, and so does `aten::Tensor`. The shim unpacks one
// from the other (THPVariable_Unpack) and wraps one back (THPVariable_Wrap) inside
// libtorch_python, so the storage on the TPU is shared, not copied. Both run with the GIL held,
// which every caller of `take` and `give` here has.

/// Python `torch.Tensor` -> Rust `Tensor`, sharing storage. TypeError-free: anything else is a
/// RuntimeError, as before.
fn take(ob: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    unsafe { Tensor::from_python(ob.as_ptr().cast()) }
        .ok_or_else(|| PyRuntimeError::new_err("expected a torch.Tensor"))
}

/// Rust `Tensor` -> Python `torch.Tensor`, sharing storage. Python owns the new reference.
fn give(py: Python<'_>, t: Tensor) -> PyResult<Py<PyAny>> {
    let ptr = unsafe { t.to_python() };
    Ok(unsafe { Bound::from_owned_ptr(py, ptr.cast()) }.unbind())
}

// --- Two functions that test the boundary ----------------------------------------------------

/// `matmul_sum(x: torch.Tensor) -> torch.Tensor`: x @ x, then a sum, all from Rust. The result
/// stays on x's device, so on a `tpu` tensor it proves a Rust ATen call reached TorchTPU.
#[pyfunction]
fn matmul_sum(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let x = take(x)?;
    give(py, x.matmul(&x).sum(Kind::Float))
}

/// `arange_like(n: int, like: torch.Tensor) -> torch.Tensor`: built on the host in Rust and
/// moved to `like`'s device. Nothing in Rust names the `tpu` device; the move follows a tensor.
#[pyfunction]
fn arange_like(py: Python<'_>, n: i64, like: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let like = take(like)?;
    give(
        py,
        Tensor::arange(n, (Kind::Float, Device::Cpu)).type_as(&like),
    )
}

// --- The decode result -----------------------------------------------------------------------

/// What `Engine.decode` returns. Frozen: Python reads the fields and cannot change them.
#[pyclass(frozen, get_all, name = "DecodeResult")]
struct PyDecoded {
    /// Generated token ids, without the stop token.
    tokens: Vec<i64>,
    /// "stop", "length" or "cancelled".
    finish: &'static str,
    /// Seconds to read the prompt and produce the first token.
    read_s: f64,
    /// Seconds for the tokens after the first.
    decode_s: f64,
}

#[pymethods]
impl PyDecoded {
    fn __repr__(&self) -> String {
        format!(
            "DecodeResult(tokens=<{}>, finish={:?}, read_s={:.3}, decode_s={:.3})",
            self.tokens.len(),
            self.finish,
            self.read_s,
            self.decode_s
        )
    }
}

impl From<Decoded> for PyDecoded {
    fn from(d: Decoded) -> Self {
        PyDecoded {
            tokens: d.tokens,
            finish: d.finish.as_str(),
            read_s: d.read_s,
            decode_s: d.decode_s,
        }
    }
}

// --- The engine ------------------------------------------------------------------------------

/// `Engine(model_dir, emb4_dir, like, context)`: Gemma 4 E2B loaded onto `like`'s device.
///
/// `unsendable`: the engine holds TPU tensors and a cache that one sequence at a time writes
/// into. If any thread other than the one that built it touches it, PyO3 panics, which Python
/// sees as `pyo3_runtime.PanicException`. That class derives from BaseException, so an
/// `except Exception` does not catch it. A server therefore builds the engine on one dedicated
/// thread and sends every request to that thread.
#[pyclass(unsendable)]
struct Engine(Gemma);

impl Engine {
    /// The checks `gemma.rs` asserts, done here so Python gets a ValueError, not a panic.
    fn check(&self, ids: &[i64], max_new: i64, read_every: i64) -> PyResult<()> {
        if ids.is_empty() {
            return Err(PyValueError::new_err("ids is empty"));
        }
        if max_new < 1 || read_every < 1 {
            return Err(PyValueError::new_err(
                "max_new and read_every must be at least 1",
            ));
        }
        let need = ids.len() as i64 + max_new;
        if need > self.0.context() {
            return Err(PyValueError::new_err(format!(
                "{} prompt tokens + {max_new} new tokens = {need}, over the context of {}",
                ids.len(),
                self.0.context()
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl Engine {
    /// Reads the checkpoint and puts every weight on `like`'s device. `context` is the cache
    /// length in tokens and must be a multiple of 256. Runs with the GIL released.
    #[new]
    fn new(
        py: Python<'_>,
        model_dir: std::path::PathBuf,
        emb4_dir: std::path::PathBuf,
        like: &Bound<'_, PyAny>,
        context: i64,
    ) -> PyResult<Self> {
        if context <= 0 || context % CHUNK != 0 {
            return Err(PyValueError::new_err(format!(
                "context must be a positive multiple of {CHUNK}"
            )));
        }
        for dir in [&model_dir, &emb4_dir] {
            if !dir.is_dir() {
                return Err(PyValueError::new_err(format!(
                    "{} is not a directory",
                    dir.display()
                )));
            }
        }
        let like = take(like)?;
        // `move`: a Tensor may cross threads (Send) but not be shared between them (not
        // Sync), so the closure that runs without the GIL owns it rather than borrowing it.
        let gemma =
            py.detach(move || aten::no_grad(|| Gemma::load(&model_dir, &emb4_dir, &like, context)));
        Ok(Engine(gemma))
    }

    /// The cache length in tokens.
    #[getter]
    fn context(&self) -> i64 {
        self.0.context()
    }

    /// `(full-attention rings, sliding rings, layers that share a ring, context)`.
    fn layout(&self) -> (usize, usize, usize, i64) {
        self.0.layout()
    }

    /// Reads `ids` and returns the logits after the last one, as a torch.Tensor on the TPU.
    fn logits(&mut self, py: Python<'_>, ids: Vec<i64>) -> PyResult<Py<PyAny>> {
        self.check(&ids, 1, 1)?;
        let gemma = &mut self.0;
        let t = py.detach(|| aten::no_grad(|| gemma.read_prompt(&ids)));
        give(py, t)
    }

    /// Fixed-length greedy decode: `(tokens, read_s, decode_s)`. No stop tokens.
    #[pyo3(signature = (ids, max_new, read_every=1))]
    fn generate(
        &mut self,
        py: Python<'_>,
        ids: Vec<i64>,
        max_new: i64,
        read_every: i64,
    ) -> PyResult<(Vec<i64>, f64, f64)> {
        self.check(&ids, max_new, read_every)?;
        let gemma = &mut self.0;
        Ok(py.detach(|| aten::no_grad(|| gemma.generate(&ids, max_new, read_every))))
    }

    /// Greedy decode that ends at a stop token, at `max_new`, or when `on_tokens` returns a
    /// false value.
    ///
    /// The TPU work runs with the GIL released, so other Python threads keep running. Each
    /// batch of tokens that reaches the host is passed to `on_tokens(list[int])` with the GIL
    /// taken back for the call. If `on_tokens` raises, the decode stops and the exception is
    /// raised from `decode`.
    #[pyo3(signature = (ids, max_new, stop, read_every=1, on_tokens=None))]
    fn decode(
        &mut self,
        py: Python<'_>,
        ids: Vec<i64>,
        max_new: i64,
        stop: Vec<i64>,
        read_every: i64,
        on_tokens: Option<Py<PyAny>>,
    ) -> PyResult<PyDecoded> {
        self.check(&ids, max_new, read_every)?;
        let gemma = &mut self.0;
        let mut failed: Option<PyErr> = None;
        let done = py.detach(|| {
            aten::no_grad(|| {
                gemma.decode(&ids, max_new, &stop, read_every, |batch| {
                    let Some(cb) = &on_tokens else { return true };
                    Python::attach(|py| {
                        let call = PyList::new(py, batch).and_then(|l| cb.call1(py, (l,)));
                        match call.and_then(|r| r.is_truthy(py)) {
                            Ok(go_on) => go_on,
                            Err(e) => {
                                failed = Some(e);
                                false
                            }
                        }
                    })
                })
            })
        });
        match failed {
            Some(e) => Err(e),
            None => Ok(done.into()),
        }
    }

    fn __repr__(&self) -> String {
        let (full, sliding, shared, context) = self.0.layout();
        format!(
            "Engine(context={context}, full_rings={full}, sliding_rings={sliding}, shared_layers={shared})"
        )
    }
}

/// Adds every item above to the module. One list, so `PYO3.md` and the stub can be checked
/// against it.
///
/// The ATen shim was compiled against one torch's C++ headers and links its libraries. A
/// different torch in the importing process would have a different C++ ABI, so the import
/// fails with ImportError rather than crashing on the first call.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let built = aten::torch_version();
    let running: String = py.import("torch")?.getattr("__version__")?.extract()?;
    if running != built {
        return Err(PyImportError::new_err(format!(
            "torchtpu_rs was built against torch {built} but torch {running} is imported; rebuild it"
        )));
    }
    let build = PyDict::new(py);
    build.set_item("torch", built)?;
    build.set_item("rustc", env!("TORCHTPU_RS_RUSTC"))?;
    build.set_item("python", env!("TORCHTPU_RS_PYTHON"))?;
    build.set_item("crate", env!("CARGO_PKG_VERSION"))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("BUILD", build)?;
    m.add("CHUNK", CHUNK)?;
    m.add_function(wrap_pyfunction!(matmul_sum, m)?)?;
    m.add_function(wrap_pyfunction!(arange_like, m)?)?;
    m.add_class::<Engine>()?;
    m.add_class::<PyDecoded>()?;
    Ok(())
}
