#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "../../solvers/_cuda_launch/pybind.h"

void eos_diagnostic_launch(const torch::Tensor &temp, const torch::Tensor &salt,
                           const torch::Tensor &pressure,
                           torch::Tensor &density_anomaly,
                           torch::Tensor &density, int64_t variant,
                           double reference_density);

void eos_backward_launch(
    const torch::Tensor &, const torch::Tensor &, const torch::Tensor &,
    const torch::Tensor &, const c10::optional<torch::Tensor> &,
    const c10::optional<torch::Tensor> &,
    const c10::optional<torch::Tensor> &, const c10::optional<torch::Tensor> &,
    const c10::optional<torch::Tensor> &, int64_t, double);

const locomgcm::cuda_launch::LaunchConfig&
eos_cuda_launch_config(int device);

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
// The kernel is instantiated for float64 only; this predicate names exactly the
// set the AT_DISPATCH_SWITCH in kernel.cu covers.  A float32 run is widened by
// the Python dispatcher, so reaching here with float32 means the widening was
// skipped -- fail rather than silently give the model a second EOS semantics.
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

std::vector<torch::Tensor> eos_fwd(torch::Tensor temp, torch::Tensor salt,
                                   torch::Tensor pressure, int64_t variant,
                                   double reference_density) {
  CHECK_INPUT(temp);
  CHECK_INPUT(salt);
  CHECK_INPUT(pressure);
  TORCH_CHECK(temp.sizes() == salt.sizes(),
              "temp and salt must have identical shapes");
  TORCH_CHECK(temp.scalar_type() == salt.scalar_type() &&
                  temp.scalar_type() == pressure.scalar_type(),
              "temp, salt, and pressure must share one floating dtype");
  TORCH_CHECK(temp.numel() > 0, "temp and salt extents must be positive");
  TORCH_CHECK(temp.device() == salt.device() &&
                  temp.device() == pressure.device(),
              "temp, salt, and pressure must be on one CUDA device");
  check_non_negative_strides(temp, "temp");
  check_non_negative_strides(salt, "salt");
  check_non_negative_strides(pressure, "pressure");
  check_broadcastable_to(pressure, temp, "pressure");
  TORCH_CHECK(variant == 0 || variant == 1,
              "EOS variant must be 0 (JMD95) or 1 (MDJWF)");

  auto density_anomaly = torch::empty(temp.sizes(), temp.options());
  auto density = torch::empty(temp.sizes(), temp.options());
  eos_diagnostic_launch(temp, salt, pressure, density_anomaly, density, variant,
                        reference_density);
  return {density_anomaly, density};
}

std::vector<c10::optional<torch::Tensor>> eos_bwd(
    torch::Tensor temp, torch::Tensor salt, torch::Tensor pressure,
    torch::Tensor density_seed, c10::optional<torch::Tensor> alpha_seed,
    c10::optional<torch::Tensor> beta_seed, int64_t variant,
    double reference_density, bool need_temp, bool need_salt,
    bool need_pressure) {
  CHECK_INPUT(temp);
  TORCH_CHECK(temp.numel() > 0, "temp extents must be positive");
  TORCH_CHECK(temp.sizes() == salt.sizes(), "temp and salt shapes must match");
  auto check_value = [&](const torch::Tensor &value) {
    CHECK_INPUT(value);
    TORCH_CHECK(value.device() == temp.device(), "EOS backward device mismatch");
    check_non_negative_strides(value, "EOS backward input");
    check_broadcastable_to(value, temp, "EOS backward input");
  };
  check_value(temp);
  check_value(salt);
  check_value(pressure);
  check_value(density_seed);
  if (alpha_seed) check_value(*alpha_seed);
  if (beta_seed) check_value(*beta_seed);
  TORCH_CHECK(variant == 0 || variant == 1, "invalid EOS variant");
  c10::optional<torch::Tensor> gt, gs, gp;
  if (need_temp) gt = torch::empty(temp.sizes(), temp.options());
  if (need_salt) gs = torch::empty(temp.sizes(), temp.options());
  if (need_pressure) gp = torch::empty(temp.sizes(), temp.options());
  if (!need_temp && !need_salt && !need_pressure) return {gt, gs, gp};
  eos_backward_launch(temp, salt, pressure, density_seed, alpha_seed, beta_seed,
                      gt, gs, gp, variant, reference_density);
  return {gt, gs, gp};
}

pybind11::dict launch_config(int device) {
  return locomgcm::cuda_launch::to_python(eos_cuda_launch_config(device));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_config", &launch_config,
        "Cached nonlinear-EOS CUDA launch configuration");
  m.def("eos_bwd", &eos_bwd, "EOS local T/S/Pa VJP (CUDA)");
  m.def("eos_fwd", &eos_fwd,
        "Fused JMD95/MDJWF diagnostic-state population (CUDA)");
}
