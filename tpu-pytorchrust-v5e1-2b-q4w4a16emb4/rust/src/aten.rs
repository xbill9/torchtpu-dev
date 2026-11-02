//! A Rust `Tensor` over the C++ shim in `aten.cpp`: exactly the ATen operations the model uses,
//! compiled against whichever torch is installed, so there is no pinned PyTorch version.
//!
//! The method names follow libtorch (and `tch`, which this replaces). Operations panic with the
//! C++ exception text on failure, the way `unwrap` would; the bindings check inputs first so
//! Python sees ValueError for anything a caller can get wrong.
use std::borrow::Borrow;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::ptr::NonNull;

type Raw = *mut c_void;

unsafe extern "C" {
    fn aten_last_error() -> *const c_char;
    fn aten_torch_version() -> *const c_char;
    fn aten_free(t: Raw);
    fn aten_set_grad_enabled(on: c_int) -> c_int;
    fn aten_dim(t: Raw) -> c_int;
    fn aten_sizes(t: Raw, out: *mut i64);
    fn aten_kind(t: Raw) -> c_int;
    fn aten_numel(t: Raw) -> i64;
    fn aten_shallow_clone(t: Raw) -> Raw;
    fn aten_zeros(s: *const i64, n: c_int, k: c_int) -> Raw;
    fn aten_full(s: *const i64, n: c_int, d: f64, i: i64, is_int: c_int, k: c_int) -> Raw;
    fn aten_arange(start: i64, end: i64, step: i64, k: c_int) -> Raw;
    fn aten_from_data(data: *const c_void, s: *const i64, n: c_int, k: c_int) -> Raw;
    fn aten_cat(ts: *const Raw, n: c_int, dim: i64) -> Raw;
    fn aten_to_kind(t: Raw, k: c_int) -> Raw;
    fn aten_type_as(t: Raw, like: Raw) -> Raw;
    fn aten_to_cpu(t: Raw) -> Raw;
    fn aten_view(t: Raw, s: *const i64, n: c_int) -> Raw;
    fn aten_unsqueeze(t: Raw, d: i64) -> Raw;
    fn aten_select(t: Raw, d: i64, i: i64) -> Raw;
    fn aten_narrow(t: Raw, d: i64, start: i64, len: i64) -> Raw;
    fn aten_t(t: Raw) -> Raw;
    fn aten_index_select(t: Raw, d: i64, idx: Raw) -> Raw;
    fn aten_index_copy_(t: Raw, d: i64, idx: Raw, src: Raw) -> Raw;
    fn aten_fill_(t: Raw, d: f64, i: i64, is_int: c_int) -> c_int;
    fn aten_add(a: Raw, b: Raw) -> Raw;
    fn aten_sub(a: Raw, b: Raw) -> Raw;
    fn aten_mul(a: Raw, b: Raw) -> Raw;
    fn aten_div(a: Raw, b: Raw) -> Raw;
    fn aten_add_s(a: Raw, d: f64, i: i64, is_int: c_int) -> Raw;
    fn aten_sub_s(a: Raw, d: f64, i: i64, is_int: c_int) -> Raw;
    fn aten_mul_s(a: Raw, d: f64, i: i64, is_int: c_int) -> Raw;
    fn aten_div_s(a: Raw, d: f64, i: i64, is_int: c_int) -> Raw;
    fn aten_neg(a: Raw) -> Raw;
    fn aten_pow_s(a: Raw, d: f64, i: i64, is_int: c_int) -> Raw;
    fn aten_remainder_s(a: Raw, d: f64, i: i64, is_int: c_int) -> Raw;
    fn aten_bitwise_and_s(a: Raw, d: f64, i: i64, is_int: c_int) -> Raw;
    fn aten_bitwise_right_shift(a: Raw, b: Raw) -> Raw;
    fn aten_matmul(a: Raw, b: Raw) -> Raw;
    fn aten_linear(x: Raw, w: Raw) -> Raw;
    fn aten_sum(a: Raw, k: c_int) -> Raw;
    fn aten_mean_dim(a: Raw, d: i64, keep: c_int, k: c_int) -> Raw;
    fn aten_softmax(a: Raw, d: i64, k: c_int) -> Raw;
    fn aten_argmax(a: Raw, d: i64, keep: c_int) -> Raw;
    fn aten_gelu_tanh(a: Raw) -> Raw;
    fn aten_tanh(a: Raw) -> Raw;
    fn aten_cos(a: Raw) -> Raw;
    fn aten_sin(a: Raw) -> Raw;
    fn aten_ge_s(a: Raw, d: f64, i: i64, is_int: c_int) -> Raw;
    fn aten_le(a: Raw, b: Raw) -> Raw;
    fn aten_gt(a: Raw, b: Raw) -> Raw;
    fn aten_logical_and(a: Raw, b: Raw) -> Raw;
    fn aten_logical_not(a: Raw) -> Raw;
    fn aten_masked_fill(a: Raw, mask: Raw, d: f64, i: i64, is_int: c_int) -> Raw;
    fn aten_copy_i64(t: Raw, out: *mut i64, n: i64) -> c_int;
    fn aten_is_python_tensor(o: *mut c_void) -> c_int;
    fn aten_from_python(o: *mut c_void) -> Raw;
    fn aten_to_python(t: Raw) -> *mut c_void;
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(aten_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// The torch version whose headers this module was compiled against.
pub fn torch_version() -> String {
    unsafe { CStr::from_ptr(aten_torch_version()) }
        .to_string_lossy()
        .into_owned()
}

/// Runs `f` with autograd off on this thread, then restores the previous mode.
pub fn no_grad<T>(f: impl FnOnce() -> T) -> T {
    let prev = unsafe { aten_set_grad_enabled(0) };
    let out = f();
    unsafe { aten_set_grad_enabled(prev) };
    out
}

/// at::ScalarType values, which are stable across releases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Kind {
    Int = 3,
    Int64 = 4,
    Half = 5,
    Float = 6,
    Double = 7,
    Bool = 11,
    BFloat16 = 15,
}

impl Kind {
    fn from_raw(k: c_int) -> Kind {
        match k {
            3 => Kind::Int,
            4 => Kind::Int64,
            5 => Kind::Half,
            6 => Kind::Float,
            7 => Kind::Double,
            11 => Kind::Bool,
            15 => Kind::BFloat16,
            other => panic!("unsupported at::ScalarType {other}"),
        }
    }
}

/// Only the host is named. Nothing names the TPU: a tensor moves there by `type_as` a tensor
/// that is already on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
}

/// A number passed to an ATen op. Integers stay integers, so an int64 tensor keeps int64
/// arithmetic (a float scalar would promote it).
#[derive(Clone, Copy)]
pub struct Scalar(f64, i64, c_int);

impl From<f64> for Scalar {
    fn from(d: f64) -> Self {
        Scalar(d, 0, 0)
    }
}
impl From<i64> for Scalar {
    fn from(i: i64) -> Self {
        Scalar(0.0, i, 1)
    }
}
impl From<i32> for Scalar {
    fn from(i: i32) -> Self {
        Scalar(0.0, i as i64, 1)
    }
}

/// An owned `at::Tensor`. Cloning the handle (`shallow_clone`) shares storage, as in C++.
pub struct Tensor(NonNull<c_void>);

// at::Tensor's reference count is atomic, so a handle may move between threads. It is not
// Sync: two threads must not use one handle at once.
unsafe impl Send for Tensor {}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { aten_free(self.0.as_ptr()) }
    }
}

fn wrap(raw: Raw, what: &str) -> Tensor {
    match NonNull::new(raw) {
        Some(p) => Tensor(p),
        None => panic!("{what}: {}", last_error()),
    }
}

macro_rules! unary {
    ($($name:ident => $ffi:ident),* $(,)?) => {$(
        pub fn $name(&self) -> Tensor {
            wrap(unsafe { $ffi(self.raw()) }, stringify!($name))
        }
    )*};
}

macro_rules! with_scalar {
    ($($name:ident => $ffi:ident),* $(,)?) => {$(
        pub fn $name(&self, s: impl Into<Scalar>) -> Tensor {
            let Scalar(d, i, is_int) = s.into();
            wrap(unsafe { $ffi(self.raw(), d, i, is_int) }, stringify!($name))
        }
    )*};
}

macro_rules! with_tensor {
    ($($name:ident => $ffi:ident),* $(,)?) => {$(
        pub fn $name(&self, other: &Tensor) -> Tensor {
            wrap(unsafe { $ffi(self.raw(), other.raw()) }, stringify!($name))
        }
    )*};
}

impl Tensor {
    fn raw(&self) -> Raw {
        self.0.as_ptr()
    }

    // --- Construction, on the host ---
    pub fn zeros(shape: impl AsRef<[i64]>, opts: (Kind, Device)) -> Tensor {
        let s = shape.as_ref();
        wrap(
            unsafe { aten_zeros(s.as_ptr(), s.len() as c_int, opts.0 as c_int) },
            "zeros",
        )
    }
    pub fn full(shape: impl AsRef<[i64]>, v: impl Into<Scalar>, opts: (Kind, Device)) -> Tensor {
        let (s, Scalar(d, i, is_int)) = (shape.as_ref(), v.into());
        let raw = unsafe { aten_full(s.as_ptr(), s.len() as c_int, d, i, is_int, opts.0 as c_int) };
        wrap(raw, "full")
    }
    pub fn arange(end: i64, opts: (Kind, Device)) -> Tensor {
        Tensor::arange_start_step(0, end, 1, opts)
    }
    pub fn arange_start_step(start: i64, end: i64, step: i64, opts: (Kind, Device)) -> Tensor {
        wrap(
            unsafe { aten_arange(start, end, step, opts.0 as c_int) },
            "arange",
        )
    }
    /// Copies `data` (raw little-endian bytes of `kind`) into a new host tensor.
    pub fn from_data_size(data: &[u8], shape: &[i64], kind: Kind) -> Tensor {
        let raw = unsafe {
            aten_from_data(
                data.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as c_int,
                kind as c_int,
            )
        };
        wrap(raw, "from_data_size")
    }
    pub fn from_slice<T: Element>(v: &[T]) -> Tensor {
        let bytes = unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        };
        Tensor::from_data_size(bytes, &[v.len() as i64], T::KIND)
    }
    pub fn cat<T: Borrow<Tensor>>(ts: &[T], dim: i64) -> Tensor {
        let raws: Vec<Raw> = ts.iter().map(|t| t.borrow().raw()).collect();
        wrap(
            unsafe { aten_cat(raws.as_ptr(), raws.len() as c_int, dim) },
            "cat",
        )
    }
    pub fn shallow_clone(&self) -> Tensor {
        wrap(unsafe { aten_shallow_clone(self.raw()) }, "shallow_clone")
    }

    // --- Properties ---
    pub fn size(&self) -> Vec<i64> {
        let mut out = vec![0i64; unsafe { aten_dim(self.raw()) } as usize];
        unsafe { aten_sizes(self.raw(), out.as_mut_ptr()) };
        out
    }
    pub fn kind(&self) -> Kind {
        Kind::from_raw(unsafe { aten_kind(self.raw()) })
    }
    pub fn numel(&self) -> i64 {
        unsafe { aten_numel(self.raw()) }
    }

    // --- Shape, dtype, device ---
    pub fn to_kind(&self, k: Kind) -> Tensor {
        wrap(unsafe { aten_to_kind(self.raw(), k as c_int) }, "to_kind")
    }
    pub fn type_as(&self, like: &Tensor) -> Tensor {
        wrap(unsafe { aten_type_as(self.raw(), like.raw()) }, "type_as")
    }
    pub fn to(&self, _d: Device) -> Tensor {
        wrap(unsafe { aten_to_cpu(self.raw()) }, "to")
    }
    pub fn view(&self, shape: impl AsRef<[i64]>) -> Tensor {
        let s = shape.as_ref();
        wrap(
            unsafe { aten_view(self.raw(), s.as_ptr(), s.len() as c_int) },
            "view",
        )
    }
    pub fn unsqueeze(&self, d: i64) -> Tensor {
        wrap(unsafe { aten_unsqueeze(self.raw(), d) }, "unsqueeze")
    }
    pub fn select(&self, d: i64, i: i64) -> Tensor {
        wrap(unsafe { aten_select(self.raw(), d, i) }, "select")
    }
    pub fn narrow(&self, d: i64, start: i64, len: i64) -> Tensor {
        wrap(unsafe { aten_narrow(self.raw(), d, start, len) }, "narrow")
    }
    /// Transpose of a 2-D tensor.
    pub fn tr(&self) -> Tensor {
        wrap(unsafe { aten_t(self.raw()) }, "tr")
    }
    pub fn index_select(&self, d: i64, idx: &Tensor) -> Tensor {
        wrap(
            unsafe { aten_index_select(self.raw(), d, idx.raw()) },
            "index_select",
        )
    }
    pub fn index_copy_(&mut self, d: i64, idx: &Tensor, src: &Tensor) -> Tensor {
        wrap(
            unsafe { aten_index_copy_(self.raw(), d, idx.raw(), src.raw()) },
            "index_copy_",
        )
    }
    pub fn fill_(&mut self, v: impl Into<Scalar>) {
        let Scalar(d, i, is_int) = v.into();
        if unsafe { aten_fill_(self.raw(), d, i, is_int) } != 0 {
            panic!("fill_: {}", last_error());
        }
    }

    // --- Arithmetic and reductions ---
    with_scalar!(
        pow_tensor_scalar => aten_pow_s,
        remainder => aten_remainder_s,
        bitwise_and => aten_bitwise_and_s,
        ge => aten_ge_s,
    );
    with_tensor!(
        matmul => aten_matmul,
        bitwise_right_shift => aten_bitwise_right_shift,
        less_equal_tensor => aten_le,
        greater_tensor => aten_gt,
        logical_and => aten_logical_and,
    );
    unary!(
        tanh => aten_tanh,
        cos => aten_cos,
        sin => aten_sin,
        logical_not => aten_logical_not,
    );
    /// x @ w.T, no bias.
    pub fn linear(&self, w: &Tensor) -> Tensor {
        wrap(unsafe { aten_linear(self.raw(), w.raw()) }, "linear")
    }
    /// GELU with the tanh approximation.
    pub fn gelu_tanh(&self) -> Tensor {
        wrap(unsafe { aten_gelu_tanh(self.raw()) }, "gelu_tanh")
    }
    pub fn sum(&self, k: Kind) -> Tensor {
        wrap(unsafe { aten_sum(self.raw(), k as c_int) }, "sum")
    }
    pub fn mean_dim(&self, d: i64, keepdim: bool, k: Kind) -> Tensor {
        wrap(
            unsafe { aten_mean_dim(self.raw(), d, keepdim as c_int, k as c_int) },
            "mean_dim",
        )
    }
    pub fn softmax(&self, d: i64, k: Kind) -> Tensor {
        wrap(
            unsafe { aten_softmax(self.raw(), d, k as c_int) },
            "softmax",
        )
    }
    pub fn argmax(&self, d: i64, keepdim: bool) -> Tensor {
        wrap(
            unsafe { aten_argmax(self.raw(), d, keepdim as c_int) },
            "argmax",
        )
    }
    pub fn masked_fill(&self, mask: &Tensor, v: impl Into<Scalar>) -> Tensor {
        let Scalar(d, i, is_int) = v.into();
        wrap(
            unsafe { aten_masked_fill(self.raw(), mask.raw(), d, i, is_int) },
            "masked_fill",
        )
    }

    // --- Back to the host ---
    /// The values of an integer tensor, read to the host (a device sync on the TPU).
    pub fn to_vec_i64(&self) -> Vec<i64> {
        let mut out = vec![0i64; self.numel() as usize];
        if unsafe { aten_copy_i64(self.raw(), out.as_mut_ptr(), out.len() as i64) } != 0 {
            panic!("to_vec_i64: {}", last_error());
        }
        out
    }
    /// The first value of an integer tensor, read to the host.
    pub fn int64_value(&self) -> i64 {
        self.to_vec_i64()[0]
    }

    // --- Python ---
    /// torch.Tensor -> Tensor, sharing storage. None if `obj` is not a torch.Tensor.
    ///
    /// # Safety
    /// `obj` must be a live PyObject pointer and the GIL must be held.
    pub unsafe fn from_python(obj: *mut c_void) -> Option<Tensor> {
        if unsafe { aten_is_python_tensor(obj) } == 0 {
            return None;
        }
        Some(wrap(unsafe { aten_from_python(obj) }, "from_python"))
    }
    /// Tensor -> a new reference to a torch.Tensor sharing storage.
    ///
    /// # Safety
    /// The GIL must be held.
    pub unsafe fn to_python(&self) -> *mut c_void {
        let o = unsafe { aten_to_python(self.raw()) };
        if o.is_null() {
            panic!("to_python: {}", last_error());
        }
        o
    }
}

/// Element types `from_slice` accepts.
pub trait Element: Copy {
    const KIND: Kind;
}
impl Element for f32 {
    const KIND: Kind = Kind::Float;
}
impl Element for i64 {
    const KIND: Kind = Kind::Int64;
}

impl From<f64> for Tensor {
    /// A 0-d float64 tensor on the host.
    fn from(v: f64) -> Tensor {
        Tensor::full([], v, (Kind::Double, Device::Cpu))
    }
}

// --- Operators: tensor (op) tensor and tensor (op) scalar, owned or borrowed ---------------
macro_rules! binop {
    ($tr:ident, $m:ident, $tt:ident, $ts:ident) => {
        impl $tr<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $m(self, o: &Tensor) -> Tensor {
                wrap(unsafe { $tt(self.raw(), o.raw()) }, stringify!($m))
            }
        }
        impl $tr<Tensor> for &Tensor {
            type Output = Tensor;
            fn $m(self, o: Tensor) -> Tensor {
                self.$m(&o)
            }
        }
        impl $tr<&Tensor> for Tensor {
            type Output = Tensor;
            fn $m(self, o: &Tensor) -> Tensor {
                (&self).$m(o)
            }
        }
        impl $tr<Tensor> for Tensor {
            type Output = Tensor;
            fn $m(self, o: Tensor) -> Tensor {
                (&self).$m(&o)
            }
        }
        binop!(@scalar $tr, $m, $ts, f64);
        binop!(@scalar $tr, $m, $ts, i64);
    };
    (@scalar $tr:ident, $m:ident, $ts:ident, $t:ty) => {
        impl $tr<$t> for &Tensor {
            type Output = Tensor;
            fn $m(self, o: $t) -> Tensor {
                let Scalar(d, i, is_int) = o.into();
                wrap(unsafe { $ts(self.raw(), d, i, is_int) }, stringify!($m))
            }
        }
        impl $tr<$t> for Tensor {
            type Output = Tensor;
            fn $m(self, o: $t) -> Tensor {
                (&self).$m(o)
            }
        }
    };
}
binop!(Add, add, aten_add, aten_add_s);
binop!(Sub, sub, aten_sub, aten_sub_s);
binop!(Mul, mul, aten_mul, aten_mul_s);
binop!(Div, div, aten_div, aten_div_s);

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        wrap(unsafe { aten_neg(self.raw()) }, "neg")
    }
}
