#include <torch/extension.h>

void cg3d_thomas_apply_launch(
    const torch::Tensor& zMC,
    const torch::Tensor& zML,
    const torch::Tensor& zMU,
    const torch::Tensor& r,
    torch::Tensor& q);

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DOUBLE(x) \
  TORCH_CHECK((x).scalar_type() == at::kDouble, #x " must be float64")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_DOUBLE(x)

void check_factors(
    const torch::Tensor& zMC,
    const torch::Tensor& zML,
    const torch::Tensor& zMU,
    const torch::Tensor& r) {
  CHECK_INPUT(zMC);
  CHECK_INPUT(zML);
  CHECK_INPUT(zMU);
  CHECK_INPUT(r);
  TORCH_CHECK(r.dim() == 4, "r must be [batch,nz,ny,nx]");
  TORCH_CHECK(zMC.dim() == 3, "zMC must be [nz,ny,nx]");
  TORCH_CHECK(zMC.sizes() == zML.sizes(), "zMC/zML shape mismatch");
  TORCH_CHECK(zMC.sizes() == zMU.sizes(), "zMC/zMU shape mismatch");
  TORCH_CHECK(zMC.size(0) == r.size(1), "zMC nz must match r");
  TORCH_CHECK(zMC.size(1) == r.size(2), "zMC ny must match r");
  TORCH_CHECK(zMC.size(2) == r.size(3), "zMC nx must match r");
  TORCH_CHECK(r.size(0) > 0, "batch must be positive");
  TORCH_CHECK(r.size(1) > 0, "nz must be positive");
  TORCH_CHECK(r.size(2) > 0, "ny must be positive");
  TORCH_CHECK(r.size(3) > 0, "nx must be positive");
  TORCH_CHECK(zMC.device() == r.device(), "zMC/r device mismatch");
  TORCH_CHECK(zML.device() == r.device(), "zML/r device mismatch");
  TORCH_CHECK(zMU.device() == r.device(), "zMU/r device mismatch");
}

}  // namespace

torch::Tensor cg3d_thomas_apply_fwd(
    torch::Tensor zMC,
    torch::Tensor zML,
    torch::Tensor zMU,
    torch::Tensor r) {
  check_factors(zMC, zML, zMU, r);
  auto q = torch::empty_like(r);
  cg3d_thomas_apply_launch(zMC, zML, zMU, r, q);
  return q;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "cg3d_thomas_apply_fwd",
      &cg3d_thomas_apply_fwd,
      "CG3D fused Thomas preconditioner forward");
}
