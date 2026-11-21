// Strict native entry points for the fused CG3D PCG body.
//
// "Strict" means what it means in the sibling packages: this layer converts
// nothing.  It does not make a contiguous copy, does not promote a dtype, does
// not fall back.  Every precondition it checks is one the Python gate
// (``_fast.py``'s decline predicates) checks first, so a failure here means the
// gate and the kernel have drifted apart -- which is exactly the failure this
// duplication is meant to make loud instead of silent.
//
// Nothing below names a grid family or a rank count.  ``nf`` is a plain outer
// extent; the only thing asserted about it is that the padded block, the mask
// and the state agree on it.
#include <torch/extension.h>

#include <limits>
#include <string>
#include <vector>

#include "../_cuda_launch/pybind.h"

void cg3d_spmv_halo_launch(
    const torch::Tensor& phi_halo,
    const torch::Tensor& aW,
    const torch::Tensor& aS,
    const torch::Tensor& aV,
    const torch::Tensor& aC,
    const torch::Tensor& mask,
    torch::Tensor& out,
    int nf,
    int nz,
    int ny,
    int nx,
    int64_t face_stride_c,
    int64_t face_stride_w,
    int64_t face_stride_s,
    const torch::Tensor& partials = torch::Tensor());

void cg3d_axpy_s_launch(
    const torch::Tensor& q,
    const torch::Tensor& s,
    const torch::Tensor& mask,
    const torch::Tensor& beta,
    torch::Tensor& out);

void cg3d_axpy_s_pad_launch(
    const torch::Tensor& q,
    const torch::Tensor& mask,
    const torch::Tensor& beta,
    torch::Tensor& padded,
    int nz,
    int ny,
    int nx,
    int halo);

void cg3d_update_xr_launch(
    const torch::Tensor& x,
    const torch::Tensor& s,
    const torch::Tensor& r,
    const torch::Tensor& qmat,
    const torch::Tensor& mask,
    const torch::Tensor& alpha,
    torch::Tensor& x_out,
    torch::Tensor& r_out,
    const torch::Tensor& valid = torch::Tensor(),
    const torch::Tensor& partials = torch::Tensor(),
    int s_halo = 0);

void cg3d_thomas_mask_launch(
    const torch::Tensor& zMC,
    const torch::Tensor& zML,
    const torch::Tensor& zMU,
    const torch::Tensor& r,
    const torch::Tensor& mask,
    torch::Tensor& q,
    int nf,
    int nz,
    int64_t plane,
    int64_t face_stride_c,
    const torch::Tensor& partials = torch::Tensor());

// How many blocks the DOT-FOLDING instantiation of each kernel launches, which
// is the length of its partial-sum buffer.  The launcher re-derives the same
// number from the same cache entry and refuses a buffer that disagrees, so the
// allocation and the launch cannot drift apart silently.
int64_t cg3d_spmv_halo_block_count(
    const torch::Tensor& phi_halo, int nf, int ny, int nx);
int64_t cg3d_update_xr_block_count(const torch::Tensor& x);
int64_t cg3d_thomas_mask_block_count(
    const torch::Tensor& r, int nf, int64_t plane);

const locomgcm::cuda_launch::LaunchConfig& cg3d_fused_launch_config(
    int device, const std::string& kernel, at::ScalarType dtype);

bool cg3d_thomas_prefers_shared(int device, int nz, at::ScalarType dtype,
                                bool fold);
bool cg3d_thomas_shared_launchable(int device, int nz, at::ScalarType dtype,
                                   bool fold);
const char* cg3d_thomas_shared_decline_reason();

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
// The kernels are instantiated for exactly these two scalar types
// (AT_DISPATCH_FLOATING_TYPES in kernel.cu); this predicate and that
// instantiation list are one decision.
#define CHECK_FLOATING(x)                                                  \
  TORCH_CHECK(                                                             \
      (x).scalar_type() == at::kDouble || (x).scalar_type() == at::kFloat, \
      #x " must be float64 or float32")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_FLOATING(x)

void check_same_kind(
    const torch::Tensor& reference,
    const torch::Tensor& value,
    const char* name) {
  TORCH_CHECK(
      value.scalar_type() == reference.scalar_type(),
      name,
      " dtype does not match the state dtype");
  TORCH_CHECK(
      value.device() == reference.device(),
      name,
      " device does not match the state device");
}

void check_same_shape(
    const torch::Tensor& reference,
    const torch::Tensor& value,
    const char* name) {
  check_same_kind(reference, value, name);
  TORCH_CHECK(
      value.sizes() == reference.sizes(), name, " shape does not match");
}

// The mask is uint8 (0/1), not a floating tensor, so ``CHECK_INPUT`` would
// refuse it and ``check_same_kind`` would demand the state's dtype.  One
// checker, used by every entry point, so they cannot disagree about what a
// mask is.  The dispatcher converts once per solve, after asserting the
// values really are binary -- ``0`` and ``1`` convert back exactly, which is
// what lets a byte stand in for a floating-point multiplicative mask.
void check_mask(const torch::Tensor& reference, const torch::Tensor& mask) {
  CHECK_CUDA(mask);
  CHECK_CONTIGUOUS(mask);
  TORCH_CHECK(mask.scalar_type() == at::kByte,
              "mask must be uint8 (0/1)");
  TORCH_CHECK(mask.device() == reference.device(),
              "mask device does not match the state device");
  TORCH_CHECK(mask.sizes() == reference.sizes(), "mask shape does not match");
}

void check_scalar(
    const torch::Tensor& reference,
    const torch::Tensor& value,
    const char* name) {
  CHECK_INPUT(value);
  check_same_kind(reference, value, name);
  TORCH_CHECK(value.numel() == 1, name, " must hold exactly one element");
}

// A coefficient either broadcasts over the leading extent (rank 3) or is
// resolved per leading index (rank 4).  Returns the per-face stride, 0 for the
// broadcast form.  No branch on WHY it is one or the other.
int64_t face_stride_for(
    const torch::Tensor& coefficient,
    int64_t nf,
    at::IntArrayRef per_face,
    const char* name) {
  if (coefficient.dim() == static_cast<int64_t>(per_face.size())) {
    TORCH_CHECK(coefficient.sizes() == per_face, name, " has the wrong shape");
    return 0;
  }
  TORCH_CHECK(
      coefficient.dim() == static_cast<int64_t>(per_face.size()) + 1,
      name,
      " must be per-face or broadcast over the leading extent");
  TORCH_CHECK(coefficient.size(0) == nf, name, " leading extent must match");
  TORCH_CHECK(
      coefficient.sizes().slice(1) == per_face, name, " has the wrong shape");
  return coefficient.numel() / nf;
}

}  // namespace

namespace {

// out = (seven-point A applied to the padded block) * mask, plus -- when
// ``fold`` -- the per-block partial sums of ``<p, out>`` with ``p`` the centre
// of the padded block.  One body, so the folding and non-folding entry points
// cannot disagree about what they accept.
std::vector<torch::Tensor> cg3d_spmv_halo_impl(
    const torch::Tensor& phi_halo,
    const torch::Tensor& aW,
    const torch::Tensor& aS,
    const torch::Tensor& aV,
    const torch::Tensor& aC,
    const torch::Tensor& mask,
    bool fold) {
  CHECK_INPUT(phi_halo);
  CHECK_INPUT(aW);
  CHECK_INPUT(aS);
  CHECK_INPUT(aV);
  CHECK_INPUT(aC);
  check_same_kind(phi_halo, aW, "aW");
  check_same_kind(phi_halo, aS, "aS");
  check_same_kind(phi_halo, aV, "aV");
  check_same_kind(phi_halo, aC, "aC");
  // Here the mask carries the INTERIOR extents while ``phi_halo`` is the
  // padded block, so their shapes differ by design; the relation between them
  // is checked below.  ``check_mask`` would compare the two directly.
  CHECK_CUDA(mask);
  CHECK_CONTIGUOUS(mask);
  TORCH_CHECK(mask.scalar_type() == at::kByte, "mask must be uint8 (0/1)");
  TORCH_CHECK(mask.device() == phi_halo.device(),
              "mask device does not match the state device");
  TORCH_CHECK(
      mask.dim() == 4, "mask must be [nf, nz, ny, nx] (the state's shape)");
  const int64_t nf = mask.size(0);
  const int64_t nz = mask.size(1);
  const int64_t ny = mask.size(2);
  const int64_t nx = mask.size(3);
  TORCH_CHECK(nf >= 1 && nz >= 1 && ny >= 1 && nx >= 1,
              "CG3D fused SPMV needs a non-degenerate extent");
  TORCH_CHECK(
      phi_halo.dim() == 4 && phi_halo.size(0) == nf && phi_halo.size(1) == nz &&
          phi_halo.size(2) == ny + 2 && phi_halo.size(3) == nx + 2,
      "phi_halo must be [nf, nz, ny+2, nx+2]");
  const int64_t face_stride_c =
      face_stride_for(aC, nf, at::IntArrayRef{nz, ny, nx}, "aC");
  TORCH_CHECK(
      face_stride_for(aV, nf, at::IntArrayRef{nz, ny, nx}, "aV") ==
          face_stride_c,
      "aV and aC must share the same broadcast form");
  const int64_t face_stride_w =
      face_stride_for(aW, nf, at::IntArrayRef{nz, ny, nx + 1}, "aW");
  const int64_t face_stride_s =
      face_stride_for(aS, nf, at::IntArrayRef{nz, ny + 1, nx}, "aS");
  TORCH_CHECK(
      nx <= std::numeric_limits<int>::max() &&
          ny <= std::numeric_limits<int>::max() &&
          nz <= std::numeric_limits<int>::max() &&
          nf <= std::numeric_limits<int>::max(),
      "CG3D fused SPMV extents exceed int range");

  // Shaped like the mask, typed like the STATE: the mask is uint8 now, so
  // ``empty_like(mask)`` would hand the kernel a byte output to write doubles
  // into.  Same for the partial sums.
  auto out = torch::empty(mask.sizes(), phi_halo.options());
  torch::Tensor partials;
  if (fold) {
    partials = torch::empty(
        {cg3d_spmv_halo_block_count(phi_halo, static_cast<int>(nf),
                                    static_cast<int>(ny), static_cast<int>(nx))},
        phi_halo.options());
  }
  cg3d_spmv_halo_launch(
      phi_halo, aW, aS, aV, aC, mask, out,
      static_cast<int>(nf), static_cast<int>(nz), static_cast<int>(ny),
      static_cast<int>(nx), face_stride_c, face_stride_w, face_stride_s,
      partials);
  if (fold) {
    return {out, partials};
  }
  return {out};
}

}  // namespace

torch::Tensor cg3d_spmv_halo_fwd(
    torch::Tensor phi_halo,
    torch::Tensor aW,
    torch::Tensor aS,
    torch::Tensor aV,
    torch::Tensor aC,
    torch::Tensor mask) {
  return cg3d_spmv_halo_impl(phi_halo, aW, aS, aV, aC, mask, false)[0];
}

// {out, per-block partial sums of <p, out>}
std::vector<torch::Tensor> cg3d_spmv_halo_dot_fwd(
    torch::Tensor phi_halo,
    torch::Tensor aW,
    torch::Tensor aS,
    torch::Tensor aV,
    torch::Tensor aC,
    torch::Tensor mask) {
  return cg3d_spmv_halo_impl(phi_halo, aW, aS, aV, aC, mask, true);
}

// out = (q + beta*s) * mask
torch::Tensor cg3d_axpy_s_fwd(
    torch::Tensor q, torch::Tensor s, torch::Tensor beta, torch::Tensor mask) {
  CHECK_INPUT(q);
  CHECK_INPUT(s);
  check_same_shape(q, s, "s");
  check_mask(q, mask);
  check_scalar(q, beta, "beta");
  TORCH_CHECK(q.numel() > 0, "CG3D fused axpy needs a non-empty state");
  auto out = torch::empty_like(q);
  cg3d_axpy_s_launch(q, s, mask, beta, out);
  return out;
}

// The same expression, written into the INTERIOR of a caller-owned padded
// buffer.  ``out_padded`` is mutated in place: it is the graph's resident
// search direction, whose border cells the caller refreshes and whose address
// a captured graph has baked in.
void cg3d_axpy_s_pad(
    torch::Tensor q, torch::Tensor beta, torch::Tensor mask,
    torch::Tensor padded, int64_t halo) {
  CHECK_INPUT(q);
  CHECK_INPUT(padded);
  check_mask(q, mask);
  check_scalar(q, beta, "beta");
  TORCH_CHECK(q.numel() > 0, "CG3D fused axpy needs a non-empty state");
  TORCH_CHECK(q.dim() == 4, "state must be [nf, nz, ny, nx]");
  TORCH_CHECK(halo >= 0 && halo <= std::numeric_limits<int>::max(),
              "halo width out of range");
  const int64_t nf = q.size(0);
  const int64_t nz = q.size(1);
  const int64_t ny = q.size(2);
  const int64_t nx = q.size(3);
  TORCH_CHECK(
      padded.dim() == 4 && padded.size(0) == nf && padded.size(1) == nz &&
          padded.size(2) == ny + 2 * halo && padded.size(3) == nx + 2 * halo,
      "the padded buffer must be [nf, nz, ny+2h, nx+2h]");
  check_same_kind(q, padded, "padded");
  cg3d_axpy_s_pad_launch(
      q, mask, beta, padded, static_cast<int>(nz),
      static_cast<int>(ny), static_cast<int>(nx), static_cast<int>(halo));
}

// x_out = (x + alpha*s)*mask ; r_out = (r - alpha*qmat)*mask
std::vector<torch::Tensor> cg3d_update_xr_fwd(
    torch::Tensor x,
    torch::Tensor s,
    torch::Tensor r,
    torch::Tensor qmat,
    torch::Tensor alpha,
    torch::Tensor mask) {
  CHECK_INPUT(x);
  CHECK_INPUT(s);
  CHECK_INPUT(r);
  CHECK_INPUT(qmat);
  check_same_shape(x, s, "s");
  check_same_shape(x, r, "r");
  check_same_shape(x, qmat, "qmat");
  check_mask(x, mask);
  check_scalar(x, alpha, "alpha");
  TORCH_CHECK(x.numel() > 0, "CG3D fused update needs a non-empty state");
  auto x_out = torch::empty_like(x);
  auto r_out = torch::empty_like(r);
  cg3d_update_xr_launch(x, s, r, qmat, mask, alpha, x_out, r_out);
  return {x_out, r_out};
}

namespace {

// ``s_halo`` describes what ``s`` is: 0 for a dense field of the state's
// shape, h>0 for the whole padded buffer whose interior is the field.  Both
// are contiguous allocations -- the kernel indexes the second with an offset
// rather than being handed a strided view, which it could not index at all.
void check_update_operands(
    const torch::Tensor& x, const torch::Tensor& s, const torch::Tensor& r,
    const torch::Tensor& qmat, const torch::Tensor& alpha,
    const torch::Tensor& mask, const torch::Tensor& valid, int64_t s_halo = 0) {
  CHECK_INPUT(x);
  CHECK_INPUT(s);
  CHECK_INPUT(r);
  CHECK_INPUT(qmat);
  if (s_halo == 0) {
    check_same_shape(x, s, "s");
  } else {
    TORCH_CHECK(x.dim() == 4 && s.dim() == 4, "state must be [nf, nz, ny, nx]");
    TORCH_CHECK(
        s.size(0) == x.size(0) && s.size(1) == x.size(1) &&
            s.size(2) == x.size(2) + 2 * s_halo &&
            s.size(3) == x.size(3) + 2 * s_halo,
        "the padded search direction must be [nf, nz, ny+2h, nx+2h]");
    check_same_kind(x, s, "s");
  }
  check_same_shape(x, r, "r");
  check_same_shape(x, qmat, "qmat");
  check_mask(x, mask);
  check_scalar(x, alpha, "alpha");
  TORCH_CHECK(x.numel() > 0, "CG3D fused update needs a non-empty state");
  TORCH_CHECK(valid.device() == x.device() && valid.scalar_type() == at::kBool &&
             valid.numel() == 1 && valid.is_contiguous(),
             "valid must be a contiguous boolean scalar on the state device");
}

}  // namespace

// Graph-owned x/r are disjoint buffers, updated only on an accepted iteration.
void cg3d_update_xr_inplace(
    torch::Tensor x, torch::Tensor s, torch::Tensor r, torch::Tensor qmat,
    torch::Tensor alpha, torch::Tensor mask, torch::Tensor valid) {
  check_update_operands(x, s, r, qmat, alpha, mask, valid);
  cg3d_update_xr_launch(x, s, r, qmat, mask, alpha, x, r, valid);
}

// As above, and returns the per-block partial sums of ``<r_new, r_new>``.  A
// REJECTED iteration writes neither x/r nor a non-zero contribution, so the
// partials sum to exactly 0 -- which the caller's validity ``where`` discards
// anyway, and which is what the eager reference's untouched ``err_sq`` means.
torch::Tensor cg3d_update_xr_inplace_dot(
    torch::Tensor x, torch::Tensor s, torch::Tensor r, torch::Tensor qmat,
    torch::Tensor alpha, torch::Tensor mask, torch::Tensor valid,
    int64_t s_halo) {
  check_update_operands(x, s, r, qmat, alpha, mask, valid, s_halo);
  // ``empty`` is enough: the reduction is outside the validity guard, so every
  // block writes its slot on every launch, rejected iteration included.
  auto partials =
      torch::empty({cg3d_update_xr_block_count(x)}, x.options());
  cg3d_update_xr_launch(x, s, r, qmat, mask, alpha, x, r, valid, partials,
                        static_cast<int>(s_halo));
  return partials;
}

namespace {

// q = (vertical-line Thomas applied to r) * mask, plus -- when ``fold`` -- the
// per-block partial sums of ``<q, r>``.
std::vector<torch::Tensor> cg3d_thomas_mask_impl(
    const torch::Tensor& zMC,
    const torch::Tensor& zML,
    const torch::Tensor& zMU,
    const torch::Tensor& r,
    const torch::Tensor& mask,
    bool fold) {
  CHECK_INPUT(zMC);
  CHECK_INPUT(zML);
  CHECK_INPUT(zMU);
  CHECK_INPUT(r);
  check_same_kind(r, zMC, "zMC");
  check_same_kind(r, zML, "zML");
  check_same_kind(r, zMU, "zMU");
  check_mask(r, mask);
  TORCH_CHECK(r.dim() == 4, "residual must be [nf, nz, ny, nx]");
  const int64_t nf = r.size(0);
  const int64_t nz = r.size(1);
  const int64_t ny = r.size(2);
  const int64_t nx = r.size(3);
  TORCH_CHECK(nf >= 1 && nz >= 1 && ny >= 1 && nx >= 1,
              "CG3D fused Thomas needs a non-degenerate extent");
  const int64_t face_stride_c =
      face_stride_for(zMC, nf, at::IntArrayRef{nz, ny, nx}, "zMC");
  TORCH_CHECK(
      face_stride_for(zML, nf, at::IntArrayRef{nz, ny, nx}, "zML") ==
              face_stride_c &&
          face_stride_for(zMU, nf, at::IntArrayRef{nz, ny, nx}, "zMU") ==
              face_stride_c,
      "the three Thomas factors must share the same broadcast form");
  auto q = torch::empty_like(r);
  torch::Tensor partials;
  if (fold) {
    partials = torch::empty(
        {cg3d_thomas_mask_block_count(r, static_cast<int>(nf), ny * nx)},
        r.options());
  }
  cg3d_thomas_mask_launch(
      zMC, zML, zMU, r, mask, q, static_cast<int>(nf), static_cast<int>(nz),
      ny * nx, face_stride_c, partials);
  if (fold) {
    return {q, partials};
  }
  return {q};
}

}  // namespace

torch::Tensor cg3d_thomas_mask_fwd(
    torch::Tensor zMC,
    torch::Tensor zML,
    torch::Tensor zMU,
    torch::Tensor r,
    torch::Tensor mask) {
  return cg3d_thomas_mask_impl(zMC, zML, zMU, r, mask, false)[0];
}

// {q, per-block partial sums of <q, r>}
std::vector<torch::Tensor> cg3d_thomas_mask_dot_fwd(
    torch::Tensor zMC,
    torch::Tensor zML,
    torch::Tensor zMU,
    torch::Tensor r,
    torch::Tensor mask) {
  return cg3d_thomas_mask_impl(zMC, zML, zMU, r, mask, true);
}

pybind11::dict launch_config(
    int device, const std::string& kernel, const std::string& dtype) {
  at::ScalarType scalar_type = at::kDouble;
  if (dtype == "float32") {
    scalar_type = at::kFloat;
  } else {
    TORCH_CHECK(dtype == "float64", "unsupported CG3D fused dtype name: ", dtype);
  }
  pybind11::dict item = locomgcm::cuda_launch::to_python(
      cg3d_fused_launch_config(device, kernel, scalar_type));
  item["kernel"] = kernel;
  item["dtype"] = dtype;
  return item;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cg3d_update_xr_inplace", &cg3d_update_xr_inplace,
        "Guarded CG3D update into graph-owned state");
  m.def("cg3d_update_xr_inplace_dot", &cg3d_update_xr_inplace_dot,
        "Guarded CG3D update plus per-block partial sums of <r_new, r_new>");
  m.def("cg3d_spmv_halo_dot_fwd", &cg3d_spmv_halo_dot_fwd,
        "Fused CG3D seven-point apply plus per-block partial sums of <p, Ap>");
  m.def("cg3d_thomas_mask_dot_fwd", &cg3d_thomas_mask_dot_fwd,
        "Fused CG3D Thomas apply plus per-block partial sums of <q, r>");
  m.def("thomas_prefers_shared",
        [](int device, int nz, const std::string& dtype, bool fold) {
          TORCH_CHECK(dtype == "float32" || dtype == "float64",
                      "unsupported CG3D fused dtype name: ", dtype);
          return cg3d_thomas_prefers_shared(
              device, nz, dtype == "float32" ? at::kFloat : at::kDouble, fold);
        },
        "Whether the column-resident Thomas kernel will be launched here");
  m.def("thomas_shared_launchable",
        [](int device, int nz, const std::string& dtype, bool fold) {
          TORCH_CHECK(dtype == "float32" || dtype == "float64",
                      "unsupported CG3D fused dtype name: ", dtype);
          return cg3d_thomas_shared_launchable(
              device, nz, dtype == "float32" ? at::kFloat : at::kDouble, fold);
        },
        "Whether this device could host the column-resident Thomas kernel");
  m.def("thomas_shared_decline_reason", &cg3d_thomas_shared_decline_reason,
        "Why the column-resident Thomas kernel is not used");
  m.def("launch_config", &launch_config,
        "Cached CG3D fused launch configuration (per kernel and dtype)");
  m.def("cg3d_spmv_halo_fwd", &cg3d_spmv_halo_fwd,
        "Fused CG3D seven-point apply from a padded block, mask on the store");
  m.def("cg3d_axpy_s_fwd", &cg3d_axpy_s_fwd,
        "Fused CG3D s=(q+beta*s)*mask");
  m.def("cg3d_axpy_s_pad", &cg3d_axpy_s_pad,
        "Fused CG3D s=(q+beta*s)*mask into a padded buffer's interior");
  m.def("cg3d_update_xr_fwd", &cg3d_update_xr_fwd,
        "Fused CG3D x/r update with the mask folded in");
  m.def("cg3d_thomas_mask_fwd", &cg3d_thomas_mask_fwd,
        "Fused CG3D vertical-line Thomas apply with the mask on the store");
}
