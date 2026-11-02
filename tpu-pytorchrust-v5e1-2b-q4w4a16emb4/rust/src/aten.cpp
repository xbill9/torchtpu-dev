// The crate's only C++: `extern "C"` wrappers around exactly the ATen operations the model uses,
// compiled by build.rs against the headers of the torch that is installed. Every call goes
// through libtorch's dispatcher, so on a `tpu` tensor it runs as a TorchTPU kernel.
//
// Conventions: a tensor crosses as an owned `at::Tensor*` (freed with aten_free). A call that
// can fail returns nullptr (or -1) and leaves the C++ exception text in aten_last_error(). A
// scalar crosses as (double, int64, is_int) so integer tensors keep integer arithmetic.
#include <ATen/ATen.h>
#include <torch/csrc/autograd/python_variable.h>

#include <cstring>
#include <string>

using at::Tensor;

namespace {
thread_local std::string last_error;

Tensor* box(Tensor t) { return new Tensor(std::move(t)); }

at::Scalar scalar(double d, int64_t i, int is_int) {
  return is_int ? at::Scalar(i) : at::Scalar(d);
}

at::ScalarType kind(int k) { return static_cast<at::ScalarType>(k); }

at::IntArrayRef dims(const int64_t* d, int n) { return at::IntArrayRef(d, static_cast<size_t>(n)); }
}  // namespace

#define TRY(body)                         \
  try {                                   \
    body                                  \
  } catch (const std::exception& e) {     \
    last_error = e.what();                \
  } catch (...) {                         \
    last_error = "unknown C++ exception"; \
  }

#define RET(expr) TRY(return box(expr);) return nullptr;

extern "C" {

const char* aten_last_error() { return last_error.c_str(); }
const char* aten_torch_version() { return TORCH_VERSION_STRING; }
void aten_free(Tensor* t) { delete t; }

// GradMode is thread-local; the engine thread turns it off around every call.
int aten_set_grad_enabled(int on) {
  int prev = at::GradMode::is_enabled();
  at::GradMode::set_enabled(on != 0);
  return prev;
}

// --- Properties --------------------------------------------------------------------------
int aten_dim(Tensor* t) { return static_cast<int>(t->dim()); }
void aten_sizes(Tensor* t, int64_t* out) {
  auto s = t->sizes();
  std::memcpy(out, s.data(), s.size() * sizeof(int64_t));
}
int aten_kind(Tensor* t) { return static_cast<int>(t->scalar_type()); }
int64_t aten_numel(Tensor* t) { return t->numel(); }

// --- Construction (on the host; `type_as` moves to the TPU) ------------------------------
Tensor* aten_shallow_clone(Tensor* t) { RET(*t) }
Tensor* aten_zeros(const int64_t* s, int n, int k) { RET(at::zeros(dims(s, n), at::TensorOptions().dtype(kind(k)))) }
Tensor* aten_full(const int64_t* s, int n, double d, int64_t i, int is_int, int k) {
  RET(at::full(dims(s, n), scalar(d, i, is_int), at::TensorOptions().dtype(kind(k))))
}
Tensor* aten_arange(int64_t start, int64_t end, int64_t step, int k) {
  RET(at::arange(start, end, step, at::TensorOptions().dtype(kind(k))))
}
Tensor* aten_from_data(const void* data, const int64_t* s, int n, int k) {
  RET(at::from_blob(const_cast<void*>(data), dims(s, n), at::TensorOptions().dtype(kind(k))).clone())
}
Tensor* aten_cat(Tensor** ts, int n, int64_t dim) {
  TRY(std::vector<Tensor> v; v.reserve(n); for (int i = 0; i < n; i++) v.push_back(*ts[i]);
      return box(at::cat(v, dim));)
  return nullptr;
}

// --- Shape and dtype ---------------------------------------------------------------------
Tensor* aten_to_kind(Tensor* t, int k) { RET(t->to(kind(k))) }
Tensor* aten_type_as(Tensor* t, Tensor* like) { RET(t->type_as(*like)) }
Tensor* aten_to_cpu(Tensor* t) { RET(t->to(at::kCPU)) }
Tensor* aten_view(Tensor* t, const int64_t* s, int n) { RET(t->view(dims(s, n))) }
Tensor* aten_unsqueeze(Tensor* t, int64_t d) { RET(t->unsqueeze(d)) }
Tensor* aten_select(Tensor* t, int64_t d, int64_t i) { RET(t->select(d, i)) }
Tensor* aten_narrow(Tensor* t, int64_t d, int64_t start, int64_t len) { RET(t->narrow(d, start, len)) }
Tensor* aten_t(Tensor* t) { RET(t->t()) }
Tensor* aten_index_select(Tensor* t, int64_t d, Tensor* idx) { RET(t->index_select(d, *idx)) }
Tensor* aten_index_copy_(Tensor* t, int64_t d, Tensor* idx, Tensor* src) { RET(t->index_copy_(d, *idx, *src)) }
int aten_fill_(Tensor* t, double d, int64_t i, int is_int) {
  TRY(t->fill_(scalar(d, i, is_int)); return 0;)
  return -1;
}

// --- Arithmetic --------------------------------------------------------------------------
Tensor* aten_add(Tensor* a, Tensor* b) { RET(*a + *b) }
Tensor* aten_sub(Tensor* a, Tensor* b) { RET(*a - *b) }
Tensor* aten_mul(Tensor* a, Tensor* b) { RET(*a * *b) }
Tensor* aten_div(Tensor* a, Tensor* b) { RET(*a / *b) }
Tensor* aten_add_s(Tensor* a, double d, int64_t i, int is_int) { RET(at::add(*a, scalar(d, i, is_int))) }
Tensor* aten_sub_s(Tensor* a, double d, int64_t i, int is_int) { RET(at::sub(*a, scalar(d, i, is_int))) }
Tensor* aten_mul_s(Tensor* a, double d, int64_t i, int is_int) { RET(at::mul(*a, scalar(d, i, is_int))) }
Tensor* aten_div_s(Tensor* a, double d, int64_t i, int is_int) { RET(at::div(*a, scalar(d, i, is_int))) }
Tensor* aten_neg(Tensor* a) { RET(at::neg(*a)) }
Tensor* aten_pow_s(Tensor* a, double d, int64_t i, int is_int) { RET(at::pow(*a, scalar(d, i, is_int))) }
Tensor* aten_remainder_s(Tensor* a, double d, int64_t i, int is_int) { RET(at::remainder(*a, scalar(d, i, is_int))) }
Tensor* aten_bitwise_and_s(Tensor* a, double d, int64_t i, int is_int) { RET(at::bitwise_and(*a, scalar(d, i, is_int))) }
Tensor* aten_bitwise_right_shift(Tensor* a, Tensor* b) { RET(at::bitwise_right_shift(*a, *b)) }
Tensor* aten_matmul(Tensor* a, Tensor* b) { RET(at::matmul(*a, *b)) }
Tensor* aten_linear(Tensor* x, Tensor* w) { RET(at::linear(*x, *w)) }
Tensor* aten_sum(Tensor* a, int k) { RET(at::sum(*a, kind(k))) }
Tensor* aten_mean_dim(Tensor* a, int64_t d, int keep, int k) {
  RET(at::mean(*a, at::IntArrayRef{d}, keep != 0, kind(k)))
}
Tensor* aten_softmax(Tensor* a, int64_t d, int k) { RET(at::softmax(*a, d, kind(k))) }
Tensor* aten_argmax(Tensor* a, int64_t d, int keep) { RET(at::argmax(*a, d, keep != 0)) }
Tensor* aten_gelu_tanh(Tensor* a) { RET(at::gelu(*a, "tanh")) }
Tensor* aten_tanh(Tensor* a) { RET(at::tanh(*a)) }
Tensor* aten_cos(Tensor* a) { RET(at::cos(*a)) }
Tensor* aten_sin(Tensor* a) { RET(at::sin(*a)) }

// --- Masks -------------------------------------------------------------------------------
Tensor* aten_ge_s(Tensor* a, double d, int64_t i, int is_int) { RET(at::ge(*a, scalar(d, i, is_int))) }
Tensor* aten_le(Tensor* a, Tensor* b) { RET(at::le(*a, *b)) }
Tensor* aten_gt(Tensor* a, Tensor* b) { RET(at::gt(*a, *b)) }
Tensor* aten_logical_and(Tensor* a, Tensor* b) { RET(at::logical_and(*a, *b)) }
Tensor* aten_logical_not(Tensor* a) { RET(at::logical_not(*a)) }
Tensor* aten_masked_fill(Tensor* a, Tensor* mask, double d, int64_t i, int is_int) {
  RET(at::masked_fill(*a, *mask, scalar(d, i, is_int)))
}

// --- Reading values back to the host -----------------------------------------------------
// Copies an int64 tensor (any device) into `out`, which holds `n` values.
int aten_copy_i64(Tensor* t, int64_t* out, int64_t n) {
  TRY(auto c = t->to(at::kCPU, at::kLong).contiguous(); if (c.numel() != n) {
    last_error = "size mismatch";
    return -1;
  } std::memcpy(out, c.data_ptr<int64_t>(), n * sizeof(int64_t));
      return 0;)
  return -1;
}

// --- Python ------------------------------------------------------------------------------
// torch.Tensor <-> at::Tensor, through libtorch_python. Both directions share storage.
int aten_is_python_tensor(PyObject* o) { return THPVariable_Check(o) ? 1 : 0; }
Tensor* aten_from_python(PyObject* o) { RET(THPVariable_Unpack(o)) }
PyObject* aten_to_python(Tensor* t) {
  TRY(return THPVariable_Wrap(*t);)
  return nullptr;
}

}  // extern "C"
