#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "../../solvers/_cuda_launch/pybind.h"

void kpp_statekpp_launch(const torch::Tensor &temp, const torch::Tensor &salt,
                         const torch::Tensor &pressure, torch::Tensor &dbloc,
                         torch::Tensor &dbsfc, torch::Tensor &alpha,
                         torch::Tensor &beta, torch::Tensor &rho1,
                         double gravity, double reference_density);

void kpp_statekpp_backward_launch(
    const torch::Tensor &, const torch::Tensor &, const torch::Tensor &,
    const std::vector<c10::optional<torch::Tensor>> &,
    const std::vector<c10::optional<torch::Tensor>> &, double, double);

const defgcm::cuda_launch::LaunchConfig &
kpp_statekpp_cuda_launch_config(int device);

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
// The kernel is instantiated for float64 only; this predicate names exactly the
// set the AT_DISPATCH_SWITCH in kernel.cu covers.  A float32 run is widened by
// the Python dispatcher, so reaching here with float32 means the widening was
// skipped -- fail rather than silently give the model a second STATEKPP.
#define CHECK_FLOATING(x)                                                      \
  TORCH_CHECK((x).scalar_type() == at::kDouble,                                \
              #x " must be float64")
#define CHECK_4D(x) TORCH_CHECK((x).dim() == 4, #x " must be four-dimensional")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_FLOATING(x);                                                             \
  CHECK_4D(x)

void check_non_negative_strides(const torch::Tensor &value, const char *name) {
  for (int dimension = 0; dimension < value.dim(); ++dimension) {
    TORCH_CHECK(value.stride(dimension) >= 0, name,
                " must not have negative strides");
  }
}

void check_broadcastable_to(const torch::Tensor &value,
                            const torch::Tensor &target, const char *name) {
  for (int dimension = 0; dimension < target.dim(); ++dimension) {
    TORCH_CHECK(value.size(dimension) == 1 ||
                    value.size(dimension) == target.size(dimension),
                name, " is not broadcastable to temperature shape");
  }
}

} // namespace

// Returns the frozen ``kpp_statekpp`` tuple in order, except that alpha/beta
// still carry Nr (not Nrp1) levels: the virtual bottom layer is appended by the
// Python caller with the same ``torch.cat`` the eager reference uses.
std::vector<torch::Tensor> kpp_statekpp_fwd(torch::Tensor temp,
                                            torch::Tensor salt,
                                            torch::Tensor pressure,
                                            double gravity,
                                            double reference_density) {
  CHECK_INPUT(temp);
  CHECK_INPUT(salt);
  CHECK_INPUT(pressure);
  TORCH_CHECK(temp.sizes() == salt.sizes(),
              "theta and salt must have identical shapes");
  TORCH_CHECK(temp.scalar_type() == salt.scalar_type() &&
                  temp.scalar_type() == pressure.scalar_type(),
              "theta, salt, and pressure must share one floating dtype");
  TORCH_CHECK(temp.numel() > 0, "theta and salt extents must be positive");
  TORCH_CHECK(temp.device() == salt.device() &&
                  temp.device() == pressure.device(),
              "theta, salt, and pressure must be on one CUDA device");
  check_non_negative_strides(temp, "theta");
  check_non_negative_strides(salt, "salt");
  check_non_negative_strides(pressure, "pressure");
  check_broadcastable_to(pressure, temp, "pressure");

  auto dbloc = torch::empty(temp.sizes(), temp.options());
  auto dbsfc = torch::empty(temp.sizes(), temp.options());
  auto alpha = torch::empty(temp.sizes(), temp.options());
  auto beta = torch::empty(temp.sizes(), temp.options());
  auto rho1 = torch::empty({temp.size(0), temp.size(2), temp.size(3)},
                           temp.options());
  kpp_statekpp_launch(temp, salt, pressure, dbloc, dbsfc, alpha, beta, rho1,
                      gravity, reference_density);
  return {dbloc, dbsfc, rho1, alpha, beta};
}

std::vector<c10::optional<torch::Tensor>> kpp_statekpp_bwd(
    torch::Tensor temp, torch::Tensor salt, torch::Tensor pressure,
    std::vector<c10::optional<torch::Tensor>> seeds,
    double gravity, double reference_density, bool need_temp,
    bool need_salt, bool need_pressure) {
  CHECK_INPUT(temp);
  TORCH_CHECK(temp.numel() > 0, "temperature extents must be positive");
  TORCH_CHECK(temp.sizes() == salt.sizes(), "temperature and salt shapes must match");
  auto check = [&](const torch::Tensor &value) {
    CHECK_INPUT(value);
    TORCH_CHECK(value.device() == temp.device(), "STATEKPP backward device mismatch");
    check_non_negative_strides(value, "STATEKPP backward input");
    check_broadcastable_to(value, temp, "STATEKPP backward input");
  };
  check(temp); check(salt); check(pressure);
  TORCH_CHECK(seeds.size() == 5, "STATEKPP requires five optional output seeds");
  for (const auto &seed : seeds) if (seed) check(*seed);
  const bool needs[] = {need_temp, need_salt, need_pressure};
  std::vector<c10::optional<torch::Tensor>> gradients(3);
  for (int index = 0; index < 3; ++index) {
    if (needs[index]) gradients[index] = torch::empty(temp.sizes(), temp.options());
  }
  if (need_temp || need_salt || need_pressure) {
    kpp_statekpp_backward_launch(temp, salt, pressure, seeds, gradients,
                                gravity, reference_density);
  }
  return gradients;
}

pybind11::dict launch_config(int device) {
  return defgcm::cuda_launch::to_python(
      kpp_statekpp_cuda_launch_config(device));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_config", &launch_config,
        "Cached KPP STATEKPP CUDA launch configuration");
  m.def("kpp_statekpp_bwd", &kpp_statekpp_bwd, "STATEKPP T/S/Pa VJP (CUDA)");
  m.def("kpp_statekpp_fwd", &kpp_statekpp_fwd,
        "Fused KPP STATEKPP buoyancy gradients and expansion coefficients "
        "(CUDA)");
}
