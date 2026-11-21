#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "../_cuda_launch/pybind.h"

void implicit_vertical_tracer_launch(
    const torch::Tensor& rhs,
    const torch::Tensor& dt,
    const torch::Tensor& diffusivity,
    const torch::Tensor& recip_hfac,
    const torch::Tensor& mask,
    const torch::Tensor& area,
    const torch::Tensor& reciprocal_area,
    const torch::Tensor& recip_dr_f,
    const torch::Tensor& recip_dr_c,
    const torch::Tensor& w,
    torch::Tensor& output,
    torch::Tensor& error_code,
    int64_t advection_scheme,
    bool diagnostic_contract);

void implicit_vertical_band_launch(
    const torch::Tensor& lower, const torch::Tensor& main,
    const torch::Tensor& upper, const torch::Tensor& rhs,
    torch::Tensor& output, torch::Tensor& error_code, bool transpose);
const locomgcm::cuda_launch::LaunchConfig& implicit_vertical_band_launch_config(
    int device, int nz, at::ScalarType dtype, bool transpose);

void implicit_vertical_tracer_backward_launch(
    const std::vector<torch::Tensor>& values, const torch::Tensor& solution,
    const torch::Tensor& grad_solution, std::vector<c10::optional<torch::Tensor>>& bars,
    torch::Tensor& error_code, int64_t advection_scheme, bool diagnostic_contract);

void implicit_vertical_momentum_launch(
    const torch::Tensor& rhs,
    const torch::Tensor& dt,
    const torch::Tensor& viscosity,
    const torch::Tensor& recip_hfac,
    const torch::Tensor& mask,
    const torch::Tensor& w_left,
    const torch::Tensor& w_right,
    const torch::Tensor& area_left,
    const torch::Tensor& area_right,
    const torch::Tensor& mask_left,
    const torch::Tensor& mask_right,
    const torch::Tensor& reciprocal_face_area,
    const torch::Tensor& recip_dr_f,
    const torch::Tensor& recip_dr_c,
    torch::Tensor& output,
    torch::Tensor& error_code,
    bool upwind,
    bool diagnostic_contract);

// ``dtype`` selects the kernel instantiation whose occupancy is measured; the
// float64 and float32 templates carry different register and shared-memory
// footprints, so a run must read the configuration for the dtype it launches.
const locomgcm::cuda_launch::LaunchConfig&
implicit_vertical_tracer_launch_config(int device, int nz, at::ScalarType dtype);
const locomgcm::cuda_launch::LaunchConfig&
implicit_vertical_momentum_upwind_launch_config(
    int device, int nz, at::ScalarType dtype);
const locomgcm::cuda_launch::LaunchConfig&
implicit_vertical_momentum_centered_launch_config(
    int device, int nz, at::ScalarType dtype);

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
// The kernel is instantiated for exactly these two scalar types; this predicate
// and the instantiation list in kernel.cu must stay one decision.
#define CHECK_FLOATING(x)                                                    \
  TORCH_CHECK(                                                               \
      (x).scalar_type() == at::kDouble || (x).scalar_type() == at::kFloat,   \
      #x " must be float64 or float32")
#define CHECK_4D(x) TORCH_CHECK((x).dim() == 4, #x " must be four-dimensional")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);        \
  CHECK_FLOATING(x);    \
  CHECK_4D(x)

at::ScalarType parse_scalar_type(const std::string& name) {
  if (name == "float64") {
    return at::kDouble;
  }
  if (name == "float32") {
    return at::kFloat;
  }
  TORCH_CHECK(false, "fused vertical CUDA dtype must be float64 or float32");
  return at::kDouble;
}

void check_direct_inputs(
    const torch::Tensor& rhs,
    const std::vector<torch::Tensor>& values) {
  CHECK_INPUT(rhs);
  TORCH_CHECK(
      rhs.size(0) > 0 && rhs.size(1) > 0 && rhs.size(2) > 0 && rhs.size(3) > 0,
      "rhs extents must be positive");
  for (const auto& value : values) {
    CHECK_INPUT(value);
    TORCH_CHECK(value.device() == rhs.device(), "all inputs must be on the rhs device");
    TORCH_CHECK(
        value.scalar_type() == rhs.scalar_type(),
        "all inputs must share the rhs dtype");
    for (int dimension = 0; dimension < value.dim(); ++dimension) {
      TORCH_CHECK(value.stride(dimension) >= 0, "inputs must not have negative strides");
    }
  }
}

void check_shape(
    const torch::Tensor& value,
    int64_t batch,
    int64_t levels,
    int64_t ny,
    int64_t nx,
    const char* name) {
  TORCH_CHECK(
      value.size(0) == batch && value.size(1) == levels &&
          value.size(2) == ny && value.size(3) == nx,
      name,
      " has an invalid shape");
}

void check_tracer_inputs(
    const torch::Tensor& rhs, const torch::Tensor& dt, const torch::Tensor& diffusivity,
    const torch::Tensor& recip_hfac, const torch::Tensor& mask, const torch::Tensor& area,
    const torch::Tensor& reciprocal_area, const torch::Tensor& recip_dr_f,
    const torch::Tensor& recip_dr_c, const torch::Tensor& w) {
  check_direct_inputs(
      rhs,
      {dt,
       diffusivity,
       recip_hfac,
       mask,
       area,
       reciprocal_area,
       recip_dr_f,
       recip_dr_c,
       w});
  const auto batch = rhs.size(0);
  const auto nz = rhs.size(1);
  const auto ny = rhs.size(2);
  const auto nx = rhs.size(3);
  for (const auto* value : {&dt, &diffusivity, &recip_hfac, &mask, &recip_dr_f}) {
    check_shape(*value, batch, nz, ny, nx, "tracer row input");
  }
  for (const auto* value : {&area, &reciprocal_area}) {
    check_shape(*value, batch, 1, ny, nx, "tracer horizontal input");
  }
  check_shape(recip_dr_c, batch, nz - 1, ny, nx, "recip_dr_c");
  check_shape(w, batch, nz + 1, ny, nx, "tracer w");
}

}  // namespace

std::vector<torch::Tensor> band_fwd(
    torch::Tensor lower, torch::Tensor main, torch::Tensor upper,
    torch::Tensor rhs, bool transpose) {
  check_direct_inputs(rhs, {lower, main, upper, rhs});
  TORCH_CHECK(rhs.size(1) <= 90, "band CUDA supports 1 through 90 levels");
  for (const auto* value : {&lower, &main, &upper}) {
    TORCH_CHECK(value->sizes() == rhs.sizes(), "band CUDA inputs must have the same shape");
  }
  auto output = torch::empty(rhs.sizes(), rhs.options());
  auto error = torch::full({}, std::numeric_limits<int64_t>::max(),
                           rhs.options().dtype(torch::kInt64));
  implicit_vertical_band_launch(lower, main, upper, rhs, output, error, transpose);
  return {output, error};
}

std::vector<torch::Tensor> tracer_fwd(
    torch::Tensor rhs,
    torch::Tensor dt,
    torch::Tensor diffusivity,
    torch::Tensor recip_hfac,
    torch::Tensor mask,
    torch::Tensor area,
    torch::Tensor reciprocal_area,
    torch::Tensor recip_dr_f,
    torch::Tensor recip_dr_c,
    torch::Tensor w,
    int64_t advection_scheme,
    bool diagnostic_contract) {
  check_tracer_inputs(rhs, dt, diffusivity, recip_hfac, mask, area,
                      reciprocal_area, recip_dr_f, recip_dr_c, w);
  auto output = torch::empty(rhs.sizes(), rhs.options());
  auto error_code = torch::full({}, std::numeric_limits<int64_t>::max(),
                                rhs.options().dtype(torch::kInt64));
  implicit_vertical_tracer_launch(
      rhs,
      dt,
      diffusivity,
      recip_hfac,
      mask,
      area,
      reciprocal_area,
      recip_dr_f,
      recip_dr_c,
      w,
      output,
      error_code,
      advection_scheme,
      diagnostic_contract);
  return {output, error_code};
}

std::tuple<std::vector<c10::optional<torch::Tensor>>, torch::Tensor> tracer_bwd(
    std::vector<torch::Tensor> values, torch::Tensor solution,
    torch::Tensor grad_solution, std::vector<bool> needs,
    int64_t advection_scheme, bool diagnostic_contract) {
  TORCH_CHECK(values.size() == 10 && needs.size() == 10, "tracer backward expects ten inputs/needs");
  check_tracer_inputs(values[0], values[1], values[2], values[3], values[4],
                      values[5], values[6], values[7], values[8], values[9]);
  check_direct_inputs(values[0], {solution, grad_solution});
  TORCH_CHECK(solution.sizes() == values[0].sizes() && grad_solution.sizes() == solution.sizes(),
              "tracer backward solution/seed shape mismatch");
  // Public physical-input order: rhs,w,dt,kappa,rm,mask,area,ra,drf,drc.
  const int slots[] = {0, 9, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<c10::optional<torch::Tensor>> bars(10);
  bool active = false;
  for (int i = 0; i < 10; ++i) {
    if (needs[i]) {
      bars[i] = torch::empty(values[slots[i]].sizes(), values[0].options());
      active = true;
    }
  }
  auto error_code = torch::full({}, std::numeric_limits<int64_t>::max(),
                                values[0].options().dtype(torch::kInt64));
  if (active) implicit_vertical_tracer_backward_launch(
      values, solution, grad_solution, bars, error_code, advection_scheme, diagnostic_contract);
  return {bars, error_code};
}

std::vector<torch::Tensor> momentum_fwd(
    torch::Tensor rhs,
    torch::Tensor dt,
    torch::Tensor viscosity,
    torch::Tensor recip_hfac,
    torch::Tensor mask,
    torch::Tensor w_left,
    torch::Tensor w_right,
    torch::Tensor area_left,
    torch::Tensor area_right,
    torch::Tensor mask_left,
    torch::Tensor mask_right,
    torch::Tensor reciprocal_face_area,
    torch::Tensor recip_dr_f,
    torch::Tensor recip_dr_c,
    bool upwind,
    bool diagnostic_contract) {
  check_direct_inputs(
      rhs,
      {dt,
       viscosity,
       recip_hfac,
       mask,
       w_left,
       w_right,
       area_left,
       area_right,
       mask_left,
       mask_right,
       reciprocal_face_area,
       recip_dr_f,
       recip_dr_c});
  const auto batch = rhs.size(0);
  const auto nz = rhs.size(1);
  const auto ny = rhs.size(2);
  const auto nx = rhs.size(3);
  for (const auto* value : {
           &dt, &viscosity, &recip_hfac, &mask, &mask_left, &mask_right,
           &recip_dr_f}) {
    check_shape(*value, batch, nz, ny, nx, "momentum row input");
  }
  for (const auto* value : {
           &area_left, &area_right, &reciprocal_face_area}) {
    check_shape(*value, batch, 1, ny, nx, "momentum horizontal input");
  }
  for (const auto* value : {&w_left, &w_right}) {
    check_shape(*value, batch, nz + 1, ny, nx, "momentum w input");
  }
  check_shape(recip_dr_c, batch, nz - 1, ny, nx, "recip_dr_c");
  auto output = torch::empty(rhs.sizes(), rhs.options());
  auto error_code = torch::full({}, std::numeric_limits<int64_t>::max(),
                                rhs.options().dtype(torch::kInt64));
  implicit_vertical_momentum_launch(
      rhs,
      dt,
      viscosity,
      recip_hfac,
      mask,
      w_left,
      w_right,
      area_left,
      area_right,
      mask_left,
      mask_right,
      reciprocal_face_area,
      recip_dr_f,
      recip_dr_c,
      output,
      error_code,
      upwind,
      diagnostic_contract);
  return {output, error_code};
}

pybind11::dict launch_configs(int device, int nz, const std::string& dtype) {
  const at::ScalarType scalar_type = parse_scalar_type(dtype);
  pybind11::dict result;
  for (const auto& entry : {
           std::pair<const char*, const locomgcm::cuda_launch::LaunchConfig*>{
               "tracer",
               &implicit_vertical_tracer_launch_config(device, nz, scalar_type)},
           std::pair<const char*, const locomgcm::cuda_launch::LaunchConfig*>{
               "momentum_upwind",
               &implicit_vertical_momentum_upwind_launch_config(
                   device, nz, scalar_type)},
           std::pair<const char*, const locomgcm::cuda_launch::LaunchConfig*>{
               "momentum_centered",
               &implicit_vertical_momentum_centered_launch_config(
                   device, nz, scalar_type)}}) {
    pybind11::dict item = locomgcm::cuda_launch::to_python(*entry.second);
    item["dtype"] = dtype;
    result[entry.first] = item;
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "launch_configs",
      &launch_configs,
      "Cached implicit-vertical CUDA launch configurations",
      pybind11::arg("device"),
      pybind11::arg("nz"),
      pybind11::arg("dtype") = "float64");
  m.def(
      "tracer_fwd",
      &tracer_fwd,
      "Direct fused tracer band assembly and Thomas solve (CUDA)");
  m.def("tracer_bwd", &tracer_bwd, "Tracer transpose solve and physical VJP (CUDA)");
  m.def(
      "momentum_fwd",
      &momentum_fwd,
      "Direct fused momentum band assembly and Thomas solve (CUDA)");
  m.def("band_fwd", &band_fwd, "MIT-order four-band solve or transpose (CUDA)");
  m.def("band_launch_config", [](int device, int nz, const std::string& dtype, bool transpose) {
    auto item = locomgcm::cuda_launch::to_python(
        implicit_vertical_band_launch_config(device, nz, parse_scalar_type(dtype), transpose));
    item["dtype"] = dtype;
    return item;
  });
}
