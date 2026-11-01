# %%
# probe: does a Rust (tch) ATen call on a tpu tensor reach TorchTPU's kernels?
import importlib.metadata as md, os, pathlib, subprocess, sys, time
t0 = time.perf_counter()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.13.0", "torch_tpu==0.2.0",
                    "--extra-index-url", "https://download.pytorch.org/whl/cpu"], capture_output=True, text=True)
print("pip", r.returncode, r.stderr[-1500:] if r.returncode else "")
print("Python", sys.version.split()[0], "cpus", os.cpu_count())
for p in ("torch_tpu", "torch", "libtpu"):
    print(f"{p:12s} {md.version(p)}")
print(f"install {time.perf_counter() - t0:.0f} s")

# %%
t0 = time.perf_counter()
r = subprocess.run("which cargo rustc; curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal 2>&1 | tail -2; $HOME/.cargo/bin/rustc --version",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-400:]); print(f"rustup {time.perf_counter() - t0:.0f} s")

# %%
crate = pathlib.Path("/content/rust"); (crate / "src").mkdir(parents=True, exist_ok=True)
(crate / "Cargo.toml").write_text(r"""[package]
name = "torchtpu_rs"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
description = "Rust ops on TorchTPU tensors: tch over the libtorch that torch_tpu registers its kernels with."

[lib]
name = "torchtpu_rs"
crate-type = ["cdylib"]

[dependencies]
tch = { version = "0.26.0", features = ["python-extension"] }
torch-sys = { version = "0.26.0", features = ["python-extension"] }
pyo3 = { version = "0.30", features = ["extension-module"] }

[profile.release]
opt-level = 3
""")
(crate / "src/lib.rs").write_text(r"""//! Rust on TorchTPU. Python imports `torch` and `torch_tpu`, which registers the TPU kernels on
//! libtorch's dispatcher; this extension links that same libtorch through `tch`, so an ATen call
//! made here on a `tpu` tensor reaches the TorchTPU kernel.
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
    let ptr = t.pyobject_wrap().map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
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
    give(py, Tensor::arange(n, (Kind::Float, tch::Device::Cpu)).type_as(&like))
}

#[pymodule]
fn torchtpu_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul_sum, m)?)?;
    m.add_function(wrap_pyfunction!(arange_like, m)?)?;
    Ok(())
}
""")
t0 = time.perf_counter()
env = {**os.environ, "LIBTORCH_USE_PYTORCH": "1", "PATH": os.path.expanduser("~/.cargo/bin:") + os.environ["PATH"]}
r = subprocess.run(["cargo", "build", "--release"], cwd=crate, env=env, capture_output=True, text=True)
print("cargo", r.returncode, r.stderr[-1800:]); print(f"build {time.perf_counter() - t0:.0f} s")
pathlib.Path("/content/mod").mkdir(exist_ok=True)
subprocess.run(["cp", str(crate / "target/release/libtorchtpu_rs.so"), "/content/mod/torchtpu_rs.so"], check=True)

# %%
TEST = r"""
import sys, time
sys.path.insert(0, "/content/mod")
import torch, torch_tpu
import torchtpu_rs as rs
print("torch", torch.__version__)
x = torch.ones(4, 4, device="tpu")
y = rs.matmul_sum(x)
print("rust result", y, y.device, "value", y.item())
z = rs.arange_like(5, x)
print("rust host-built tensor moved", z.device, z.tolist())
print("bf16", rs.arange_like(3, torch.ones(1, dtype=torch.bfloat16, device="tpu")).dtype)
print("hbm bytes", torch.accelerator.memory_allocated())
def bench(name, device, f, n=2048, reps=20):
    a = torch.rand(n, n, device=device)
    f(a); torch.accelerator.synchronize() if device == "tpu" else None
    t = time.perf_counter()
    for _ in range(reps):
        v = f(a)
    v.item()
    ms = (time.perf_counter() - t) / reps * 1000
    print(f"{name:18s} {device} {ms:9.2f} ms/step", v.device)
py = lambda a: (a @ a).sum()
bench("rust", "tpu", rs.matmul_sum); bench("python", "tpu", py); bench("rust", "tpu", rs.matmul_sum); bench("python", "tpu", py)
bench("rust", "cpu", rs.matmul_sum, reps=3)
"""
pathlib.Path("/content/t.py").write_text(TEST)
for label, extra in (("strict eager", {}), ("defer-and-fuse", {"TORCH_TPU_DEFER_AND_FUSE": "1"})):
    r = subprocess.run([sys.executable, "/content/t.py"], env={**os.environ, **extra}, capture_output=True, text=True, timeout=900)
    print("=====", label, "exit", r.returncode); print(r.stdout[-2500:]); print("STDERR", r.stderr[-1500:])
