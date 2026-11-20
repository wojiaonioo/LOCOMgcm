#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../_cuda_launch/pybind.h"

void tridiag_build_cuda_launch(
    const torch::Tensor& tracer,
    const torch::Tensor& layer_thickness,
    const torch::Tensor& interface_spacing,
    const torch::Tensor& diffusivity,
    const torch::Tensor& recip_hfac,
    double dt,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& c,
    torch::Tensor& d);

const defgcm::cuda_launch::LaunchConfig&
tridiag_build_cuda_launch_config(int device, int nz, at::ScalarType dtype);

namespace {

constexpr int64_t kSupportedLevels = 90;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
// The kernel is instantiated for exactly these two scalar types; this predicate
// and the instantiation list in kernel.cu must stay one decision.
#define CHECK_FLOATING(x)                                                    \
  TORCH_CHECK(                                                               \
      (x).scalar_type() == at::kDouble || (x).scalar_type() == at::kFloat,   \
      #x " must be float64 or float32")
#define CHECK_4D(x) TORCH_CHECK((x).dim() == 4, #x " must be four-dimensional")
#define CHECK_INPUT_4D(x) \
  CHECK_CUDA(x);           \
  CHECK_FLOATING(x);       \
  CHECK_4D(x)

at::ScalarType parse_scalar_type(const std::string& name) {
  if (name == "float64") {
    return at::kDouble;
  }
  if (name == "float32") {
    return at::kFloat;
  }
  TORCH_CHECK(false, "tridiag-build CUDA dtype must be float64 or float32");
  return at::kDouble;
}

void check_non_negative_strides(const torch::Tensor& value, const char* name) {
  for (int64_t dimension = 0; dimension < value.dim(); ++dimension) {
    TORCH_CHECK(value.stride(dimension) >= 0, name, " must not have negative strides");
  }
}

void check_field_shape(
    const torch::Tensor& value,
    int64_t batch,
    int64_t levels,
    int64_t ny,
    int64_t nx,
    const char* name) {
  TORCH_CHECK(
      (value.size(0) == 1 || value.size(0) == batch) &&
          value.size(1) == levels && value.size(2) == ny &&
          value.size(3) == nx,
      name,
      " must have shape [1|B, 90, ny, nx]");
}

}  // namespace

std::vector<torch::Tensor> build_fwd(
    torch::Tensor tracer,
    torch::Tensor layer_thickness,
    torch::Tensor interface_spacing,
    torch::Tensor diffusivity,
    double dt,
    torch::Tensor recip_hfac) {
  CHECK_INPUT_4D(tracer);
  CHECK_INPUT_4D(diffusivity);
  CHECK_INPUT_4D(recip_hfac);
  CHECK_CUDA(layer_thickness);
  CHECK_FLOATING(layer_thickness);
  TORCH_CHECK(layer_thickness.dim() == 1, "layer_thickness must be one-dimensional");
  CHECK_CUDA(interface_spacing);
  CHECK_FLOATING(interface_spacing);
  TORCH_CHECK(interface_spacing.dim() == 1, "interface_spacing must be one-dimensional");
  TORCH_CHECK(
      diffusivity.scalar_type() == tracer.scalar_type() &&
          recip_hfac.scalar_type() == tracer.scalar_type() &&
          layer_thickness.scalar_type() == tracer.scalar_type() &&
          interface_spacing.scalar_type() == tracer.scalar_type(),
      "all inputs must share the tracer dtype");
  TORCH_CHECK(
      tracer.size(0) > 0 && tracer.size(1) == kSupportedLevels &&
          tracer.size(2) > 0 && tracer.size(3) > 0,
      "tracer must have non-empty shape [B, 90, ny, nx]");
  TORCH_CHECK(
      layer_thickness.size(0) == kSupportedLevels,
      "layer_thickness must contain 90 levels");
  TORCH_CHECK(
      interface_spacing.size(0) == kSupportedLevels - 1,
      "interface_spacing must contain 89 interfaces");
  TORCH_CHECK(
      diffusivity.device() == tracer.device() &&
          recip_hfac.device() == tracer.device() &&
          layer_thickness.device() == tracer.device() &&
          interface_spacing.device() == tracer.device(),
      "all inputs must be on the tracer device");
  check_non_negative_strides(tracer, "tracer");
  check_non_negative_strides(layer_thickness, "layer_thickness");
  check_non_negative_strides(interface_spacing, "interface_spacing");
  check_non_negative_strides(diffusivity, "diffusivity");
  check_non_negative_strides(recip_hfac, "recip_hfac");
  check_field_shape(
      diffusivity,
      tracer.size(0),
      kSupportedLevels,
      tracer.size(2),
      tracer.size(3),
      "diffusivity");
  check_field_shape(
      recip_hfac,
      tracer.size(0),
      kSupportedLevels,
      tracer.size(2),
      tracer.size(3),
      "recip_hfac");
  auto a = torch::empty_like(tracer);
  auto b = torch::empty_like(tracer);
  auto c = torch::empty_like(tracer);
  auto d = torch::empty_like(tracer);
  tridiag_build_cuda_launch(
      tracer,
      layer_thickness,
      interface_spacing,
      diffusivity,
      recip_hfac,
      dt,
      a,
      b,
      c,
      d);
  return {a, b, c, d};
}

pybind11::dict launch_config(int device, int nz, const std::string& dtype) {
  pybind11::dict result = defgcm::cuda_launch::to_python(
      tridiag_build_cuda_launch_config(device, nz, parse_scalar_type(dtype)));
  result["dtype"] = dtype;
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "launch_config",
      &launch_config,
      "Cached tridiag-build CUDA launch configuration",
      pybind11::arg("device"),
      pybind11::arg("nz"),
      pybind11::arg("dtype") = "float64");
  m.def(
      "build_fwd",
      &build_fwd,
      "Single-launch implicit-diffusion system assembly (CUDA)");
}
