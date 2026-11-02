//! Rust on TorchTPU. Python imports `torch` and `torch_tpu`, which registers the TPU kernels on
//! libtorch's dispatcher; this extension links that same libtorch through `tch`, so an ATen call
//! made here on a `tpu` tensor reaches the TorchTPU kernel.
mod gemma;
mod weights;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tch::{Kind, Tensor};

fn take(ob: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
    unsafe { Tensor::pyobject_unpack(ptr) }
        .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?
        .ok_or_else(|| PyRuntimeError::new_err("expected a torch.Tensor"))
}

fn give(py: Python<'_>, t: Tensor) -> PyResult<Py<PyAny>> {
    let ptr = t
        .pyobject_wrap()
        .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
    Ok(unsafe { Bound::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) }.unbind())
}

/// x @ x, then a sum, all from Rust; the result stays on x's device.
#[pyfunction]
fn matmul_sum(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let x = take(x)?;
    give(py, x.matmul(&x).sum(Kind::Float))
}

/// A tensor built on the host in Rust, moved to `like`'s device without naming that device.
#[pyfunction]
fn arange_like(py: Python<'_>, n: i64, like: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let like = take(like)?;
    give(
        py,
        Tensor::arange(n, (Kind::Float, tch::Device::Cpu)).type_as(&like),
    )
}

/// Gemma 4 E2B, all in Rust. `like` is any tensor on the target device.
#[pyclass(unsendable)]
struct Engine(gemma::Gemma);

#[pymethods]
impl Engine {
    #[new]
    fn new(
        model_dir: &str,
        emb4_dir: &str,
        like: &Bound<'_, PyAny>,
        context: i64,
    ) -> PyResult<Self> {
        let like = take(like)?;
        Ok(Engine(tch::no_grad(|| {
            gemma::Gemma::load(model_dir.as_ref(), emb4_dir.as_ref(), &like, context)
        })))
    }

    /// (full-attention rings, sliding rings, shared layers, context)
    fn layout(&self) -> (usize, usize, usize, i64) {
        self.0.layout()
    }

    /// The logits after reading `ids`.
    fn logits(&mut self, py: Python<'_>, ids: Vec<i64>) -> PyResult<Py<PyAny>> {
        give(py, tch::no_grad(|| self.0.read_prompt(&ids)))
    }

    /// (tokens, seconds to read the prompt, seconds for the tokens after the first)
    #[pyo3(signature = (ids, max_new, read_every=1))]
    fn generate(&mut self, ids: Vec<i64>, max_new: i64, read_every: i64) -> (Vec<i64>, f64, f64) {
        tch::no_grad(|| self.0.generate(&ids, max_new, read_every))
    }
}

#[pymodule]
fn torchtpu_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul_sum, m)?)?;
    m.add_function(wrap_pyfunction!(arange_like, m)?)?;
    m.add_class::<Engine>()?;
    Ok(())
}
