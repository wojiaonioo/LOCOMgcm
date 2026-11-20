#include <torch/extension.h>

#include <string>
#include <vector>

void statekpp_fused_launch(
    const torch::Tensor &temp, const torch::Tensor &salt,
    const torch::Tensor &pressure, torch::Tensor &dbloc,
    torch::Tensor &dbsfc, torch::Tensor &rho1, torch::Tensor &alpha,
    torch::Tensor &beta, double gravity, double reference_density,
    const std::string &eos);

namespace {

void check_tensor(const torch::Tensor &value, const char *name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.dim() == 4, name, " must be four-dimensional");
  TORCH_CHECK(value.scalar_type() == at::kFloat ||
                  value.scalar_type() == at::kDouble,
              name, " must be float32 or float64");
  for (int64_t dim = 0; dim < value.dim(); ++dim) {
    TORCH_CHECK(value.stride(dim) >= 0, name,
                " must not have negative strides");
  }
}

} // namespace

std::vector<torch::Tensor> statekpp_fused_fwd(
    torch::Tensor temp, torch::Tensor salt, torch::Tensor pressure,
    double gravity, double reference_density, const std::string &eos) {
  check_tensor(temp, "theta");
  check_tensor(salt, "salt");
  check_tensor(pressure, "pressure");
  TORCH_CHECK(temp.sizes() == salt.sizes(),
              "theta and salt must have identical shapes");
  TORCH_CHECK(temp.scalar_type() == salt.scalar_type() &&
                  temp.scalar_type() == pressure.scalar_type(),
              "theta, salt, and pressure must share one dtype");
  TORCH_CHECK(temp.device() == salt.device() &&
                  temp.device() == pressure.device(),
              "theta, salt, and pressure must be on one CUDA device");
  TORCH_CHECK(temp.numel() > 0, "STATEKPP extents must be positive");
  TORCH_CHECK(eos == "jmd95" || eos == "mdjwf",
              "eos must be exactly 'jmd95' or 'mdjwf', got '", eos, "'");
  for (int64_t dim = 0; dim < 4; ++dim) {
    TORCH_CHECK(pressure.size(dim) == 1 ||
                    pressure.size(dim) == temp.size(dim),
                "pressure must broadcast to theta");
  }

  const auto batch = temp.size(0);
  const auto levels = temp.size(1);
  const auto ny = temp.size(2);
  const auto nx = temp.size(3);
  auto dbloc = torch::empty_like(temp);
  auto dbsfc = torch::empty_like(temp);
  auto rho1 = torch::empty({batch, ny, nx}, temp.options());
  auto alpha = torch::empty({batch, levels + 1, ny, nx}, temp.options());
  auto beta = torch::empty({batch, levels + 1, ny, nx}, temp.options());
  statekpp_fused_launch(temp, salt, pressure, dbloc, dbsfc, rho1, alpha, beta,
                        gravity, reference_density, eos);
  return {dbloc, dbsfc, rho1, alpha, beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("statekpp_fused_fwd", &statekpp_fused_fwd,
             "Column-fused KPP STATEKPP (CUDA)");
}
