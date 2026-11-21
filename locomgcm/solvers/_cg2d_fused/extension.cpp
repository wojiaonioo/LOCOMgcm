// Strict native entry points for the fused CG2D PCG body.
//
// "Strict" means what it means in the sibling packages: this layer converts
// nothing.  It does not make a contiguous copy, does not promote a dtype, does
// not fall back.  Every precondition it checks is one the Python gate
// (``_fast.py::direct_decline_reason``) checks first, so a failure here means
// the gate and the kernel have drifted apart -- which is exactly the failure
// this duplication is meant to make loud instead of silent.
#include <torch/extension.h>

#include <limits>
#include <string>
#include <vector>

#include "../_cuda_launch/pybind.h"

void cg2d_spmv_launch(
    const torch::Tensor& phi_halo,
    const torch::Tensor& aW,
    const torch::Tensor& aS,
    const torch::Tensor& aC,
    torch::Tensor& out,
    int ny,
    int nx);

void cg2d_update_xr_launch(
    const torch::Tensor& x,
    const torch::Tensor& p,
    const torch::Tensor& r,
    const torch::Tensor& ap,
    const torch::Tensor& alpha,
    torch::Tensor& x_out,
    torch::Tensor& r_out);

void cg2d_update_xr_ring_launch(
    const torch::Tensor& x,
    const torch::Tensor& p_ring,
    const torch::Tensor& r,
    const torch::Tensor& ap,
    const torch::Tensor& alpha,
    torch::Tensor& x_out,
    torch::Tensor& r_out,
    int ny,
    int nx);

void cg2d_update_xrz_launch(
    const torch::Tensor& x,
    const torch::Tensor& p,
    const torch::Tensor& r,
    const torch::Tensor& ap,
    const torch::Tensor& alpha,
    const torch::Tensor& precond_inv,
    torch::Tensor& x_out,
    torch::Tensor& r_out,
    torch::Tensor& z_out);

void cg2d_ssor_precond_launch(
    const torch::Tensor& r,
    const torch::Tensor& pC,
    const torch::Tensor& pW,
    const torch::Tensor& pS,
    torch::Tensor& out,
    int ny,
    int nx,
    bool wrap_x,
    bool wrap_y);

void cg2d_axpy_p_launch(
    const torch::Tensor& z,
    const torch::Tensor& p,
    const torch::Tensor& beta,
    torch::Tensor& out);

const locomgcm::cuda_launch::LaunchConfig& cg2d_fused_launch_config(
    int device, const std::string& kernel, at::ScalarType dtype);

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
// The kernels are instantiated for exactly these two scalar types
// (AT_DISPATCH_FLOATING_TYPES in kernel.cu); this predicate and that
// instantiation list are one decision.
#define CHECK_FLOATING(x)                                                    \
  TORCH_CHECK(                                                               \
      (x).scalar_type() == at::kDouble || (x).scalar_type() == at::kFloat,   \
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

void check_scalar(
    const torch::Tensor& reference,
    const torch::Tensor& value,
    const char* name) {
  CHECK_INPUT(value);
  check_same_kind(reference, value, name);
  TORCH_CHECK(value.numel() == 1, name, " must hold exactly one element");
}

at::ScalarType parse_scalar_type(const std::string& name) {
  if (name == "float64") {
    return at::kDouble;
  }
  if (name == "float32") {
    return at::kFloat;
  }
  TORCH_CHECK(false, "unsupported CG2D fused dtype name: ", name);
}

}  // namespace

// out = five-point Helmholtz(phi_halo); phi_halo carries a width-1 centre halo.
torch::Tensor cg2d_spmv_fwd(
    torch::Tensor phi_halo,
    torch::Tensor aW_edge,
    torch::Tensor aS_edge,
    torch::Tensor aC) {
  CHECK_INPUT(phi_halo);
  CHECK_INPUT(aW_edge);
  CHECK_INPUT(aS_edge);
  CHECK_INPUT(aC);
  check_same_kind(phi_halo, aW_edge, "aW_edge");
  check_same_kind(phi_halo, aS_edge, "aS_edge");
  check_same_kind(phi_halo, aC, "aC");
  TORCH_CHECK(aC.dim() == 2, "aC must be two-dimensional [ny, nx]");
  TORCH_CHECK(phi_halo.dim() >= 2, "phi_halo must have at least two dimensions");
  const int64_t ny = aC.size(0);
  const int64_t nx = aC.size(1);
  TORCH_CHECK(ny >= 2 && nx >= 2, "CG2D fused SPMV needs ny >= 2 and nx >= 2");
  TORCH_CHECK(
      aW_edge.sizes() == (at::IntArrayRef{ny, nx + 1}),
      "aW_edge must be [ny, nx+1]");
  TORCH_CHECK(
      aS_edge.sizes() == (at::IntArrayRef{ny + 1, nx}),
      "aS_edge must be [ny+1, nx]");
  TORCH_CHECK(
      phi_halo.size(-2) == ny + 2 && phi_halo.size(-1) == nx + 2,
      "phi_halo must be [..., ny+2, nx+2]");
  TORCH_CHECK(
      phi_halo.numel() == (ny + 2) * (nx + 2),
      "CG2D fused SPMV serves one field at a time (leading extent 1)");
  TORCH_CHECK(
      nx <= std::numeric_limits<int>::max() &&
          ny <= std::numeric_limits<int>::max(),
      "CG2D fused SPMV extents exceed int range");

  std::vector<int64_t> out_shape(phi_halo.sizes().begin(), phi_halo.sizes().end());
  out_shape[out_shape.size() - 2] = ny;
  out_shape[out_shape.size() - 1] = nx;
  auto out = torch::empty(out_shape, phi_halo.options());
  cg2d_spmv_launch(
      phi_halo, aW_edge, aS_edge, aC, out,
      static_cast<int>(ny), static_cast<int>(nx));
  return out;
}

// x_out = x + alpha*p ; r_out = r - alpha*Ap
std::vector<torch::Tensor> cg2d_update_xr_fwd(
    torch::Tensor x,
    torch::Tensor p,
    torch::Tensor r,
    torch::Tensor ap,
    torch::Tensor alpha) {
  CHECK_INPUT(x);
  CHECK_INPUT(p);
  CHECK_INPUT(r);
  CHECK_INPUT(ap);
  check_same_shape(x, p, "p");
  check_same_shape(x, r, "r");
  check_same_shape(x, ap, "Ap");
  check_scalar(x, alpha, "alpha");
  TORCH_CHECK(x.numel() > 0, "CG2D fused update needs a non-empty state");
  auto x_out = torch::empty_like(x);
  auto r_out = torch::empty_like(r);
  cg2d_update_xr_launch(x, p, r, ap, alpha, x_out, r_out);
  return {x_out, r_out};
}

// x_out = x + alpha*owned(p_ring) ; r_out = r - alpha*Ap, where p_ring carries
// a one-cell ring: [..., ny+2, nx+2] around the [..., ny, nx] state.
std::vector<torch::Tensor> cg2d_update_xr_ring_fwd(
    torch::Tensor x,
    torch::Tensor p_ring,
    torch::Tensor r,
    torch::Tensor ap,
    torch::Tensor alpha) {
  CHECK_INPUT(x);
  CHECK_INPUT(p_ring);
  CHECK_INPUT(r);
  CHECK_INPUT(ap);
  check_same_kind(x, p_ring, "p_ring");
  check_same_shape(x, r, "r");
  check_same_shape(x, ap, "Ap");
  check_scalar(x, alpha, "alpha");
  TORCH_CHECK(x.dim() >= 2, "x must have at least two dimensions");
  TORCH_CHECK(p_ring.dim() >= 2, "p_ring must have at least two dimensions");
  const int64_t ny = x.size(-2);
  const int64_t nx = x.size(-1);
  TORCH_CHECK(ny >= 1 && nx >= 1, "CG2D fused update needs a non-empty state");
  TORCH_CHECK(
      x.numel() == ny * nx,
      "CG2D fused ring update serves one field at a time (leading extent 1)");
  TORCH_CHECK(
      p_ring.size(-2) == ny + 2 && p_ring.size(-1) == nx + 2 &&
          p_ring.numel() == (ny + 2) * (nx + 2),
      "p_ring must be one [..., ny+2, nx+2] field around the state");
  TORCH_CHECK(
      nx + 2 <= std::numeric_limits<int>::max() &&
          ny + 2 <= std::numeric_limits<int>::max(),
      "CG2D fused ring update extents exceed int range");
  auto x_out = torch::empty_like(x);
  auto r_out = torch::empty_like(r);
  cg2d_update_xr_ring_launch(
      x, p_ring, r, ap, alpha, x_out, r_out,
      static_cast<int>(ny), static_cast<int>(nx));
  return {x_out, r_out};
}

// x_out = x + alpha*p ; r_out = r - alpha*Ap ; z_out = precond_inv * r_out
std::vector<torch::Tensor> cg2d_update_xrz_fwd(
    torch::Tensor x,
    torch::Tensor p,
    torch::Tensor r,
    torch::Tensor ap,
    torch::Tensor alpha,
    torch::Tensor precond_inv) {
  CHECK_INPUT(x);
  CHECK_INPUT(p);
  CHECK_INPUT(r);
  CHECK_INPUT(ap);
  CHECK_INPUT(precond_inv);
  check_same_shape(x, p, "p");
  check_same_shape(x, r, "r");
  check_same_shape(x, ap, "Ap");
  check_scalar(x, alpha, "alpha");
  check_same_kind(x, precond_inv, "precond_inv");
  TORCH_CHECK(
      precond_inv.numel() == x.numel(),
      "precond_inv must hold one entry per state cell");
  TORCH_CHECK(x.numel() > 0, "CG2D fused update needs a non-empty state");
  auto x_out = torch::empty_like(x);
  auto r_out = torch::empty_like(r);
  auto z_out = torch::empty_like(r);
  cg2d_update_xrz_launch(x, p, r, ap, alpha, precond_inv, x_out, r_out, z_out);
  return {x_out, r_out, z_out};
}

namespace {

// z = SSOR(r) with the five-point apply on one process's array; wrap_x/wrap_y
// add the periodic-edge terms of an axis the process owns whole.
torch::Tensor ssor_precond(
    const torch::Tensor& r,
    const torch::Tensor& pC,
    const torch::Tensor& pW,
    const torch::Tensor& pS,
    bool wrap_x,
    bool wrap_y) {
  CHECK_INPUT(r);
  CHECK_INPUT(pC);
  CHECK_INPUT(pW);
  CHECK_INPUT(pS);
  check_same_kind(r, pC, "pC");
  check_same_kind(r, pW, "pW");
  check_same_kind(r, pS, "pS");
  TORCH_CHECK(pC.dim() == 2, "pC must be two-dimensional [ny, nx]");
  const int64_t ny = pC.size(0);
  const int64_t nx = pC.size(1);
  TORCH_CHECK(ny >= 2 && nx >= 2, "CG2D fused SSOR needs ny >= 2 and nx >= 2");
  TORCH_CHECK(pW.sizes() == pC.sizes(), "pW must match pC");
  TORCH_CHECK(pS.sizes() == pC.sizes(), "pS must match pC");
  TORCH_CHECK(
      r.size(-2) == ny && r.size(-1) == nx && r.numel() == ny * nx,
      "residual must be one [.., ny, nx] field");
  // The kernel indexes rows and columns as int and the wrap reads as
  // j*nx + (nx-1) and i + (ny-1)*nx in int64; both stay inside the arrays only
  // if the extents are what the kernel was told.
  TORCH_CHECK(
      nx <= std::numeric_limits<int>::max() &&
          ny <= std::numeric_limits<int>::max(),
      "CG2D fused SSOR extents exceed int range");
  auto out = torch::empty_like(r);
  cg2d_ssor_precond_launch(
      r, pC, pW, pS, out, static_cast<int>(ny), static_cast<int>(nx),
      wrap_x, wrap_y);
  return out;
}

}  // namespace

// z = SSOR(r) with the regional tile-local (halo-free) five-point apply.
torch::Tensor cg2d_ssor_precond_fwd(
    torch::Tensor r,
    torch::Tensor pC,
    torch::Tensor pW,
    torch::Tensor pS) {
  return ssor_precond(r, pC, pW, pS, false, false);
}

// The in-array periodic instantiations, one Python op per wrap (``_fast.py``)
// so the tile-local op above keeps its four-tensor signature.
torch::Tensor cg2d_ssor_precond_wrapped_fwd(
    torch::Tensor r,
    torch::Tensor pC,
    torch::Tensor pW,
    torch::Tensor pS,
    bool wrap_x,
    bool wrap_y) {
  TORCH_CHECK(wrap_x || wrap_y, "the wrapped SSOR entry needs a wrapped axis");
  return ssor_precond(r, pC, pW, pS, wrap_x, wrap_y);
}

// out = z + beta * p
torch::Tensor cg2d_axpy_p_fwd(
    torch::Tensor z,
    torch::Tensor p,
    torch::Tensor beta) {
  CHECK_INPUT(z);
  CHECK_INPUT(p);
  check_same_shape(z, p, "p");
  check_scalar(z, beta, "beta");
  TORCH_CHECK(z.numel() > 0, "CG2D fused axpy needs a non-empty state");
  auto out = torch::empty_like(z);
  cg2d_axpy_p_launch(z, p, beta, out);
  return out;
}

pybind11::dict launch_config(
    int device, const std::string& kernel, const std::string& dtype) {
  pybind11::dict item = locomgcm::cuda_launch::to_python(
      cg2d_fused_launch_config(device, kernel, parse_scalar_type(dtype)));
  item["kernel"] = kernel;
  item["dtype"] = dtype;
  return item;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_config", &launch_config,
        "Cached CG2D fused launch configuration (per kernel and dtype)");
  m.def("cg2d_spmv_fwd", &cg2d_spmv_fwd,
        "Fused CG2D five-point Helmholtz matvec from a haloed centre field");
  m.def("cg2d_update_xr_fwd", &cg2d_update_xr_fwd,
        "Fused CG2D x=x+alpha*p, r=r-alpha*Ap");
  m.def("cg2d_update_xr_ring_fwd", &cg2d_update_xr_ring_fwd,
        "Fused CG2D x=x+alpha*p, r=r-alpha*Ap with p carried on a one-cell ring");
  m.def("cg2d_update_xrz_fwd", &cg2d_update_xrz_fwd,
        "Fused CG2D x/r update with the Jacobi preconditioner folded in");
  m.def("cg2d_ssor_precond_fwd", &cg2d_ssor_precond_fwd,
        "Fused CG2D regional SSOR preconditioner apply");
  m.def("cg2d_ssor_precond_wrapped_fwd", &cg2d_ssor_precond_wrapped_fwd,
        "Fused CG2D SSOR preconditioner apply on one process's array "
        "(wrap_x/wrap_y: in-array periodic edges)");
  m.def("cg2d_axpy_p_fwd", &cg2d_axpy_p_fwd,
        "Fused CG2D p=z+beta*p");
}
