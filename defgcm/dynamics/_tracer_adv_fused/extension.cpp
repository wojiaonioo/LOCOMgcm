#include <torch/extension.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "../../solvers/_cuda_launch/pybind.h"

void tracer_adv_pass_launch(
    const torch::Tensor &t_wide, const torch::Tensor &t_left,
    const torch::Tensor &t_right, const torch::Tensor &mask_wide,
    const torch::Tensor &vel_face, const torch::Tensor &vel_trans,
    const torch::Tensor &recip_dc, const torch::Tensor &recip_vol,
    const torch::Tensor &mask_in_c, const torch::Tensor &tracer_orig,
    const torch::Tensor &local_t, torch::Tensor &out, double dt,
    int64_t dst_start, int64_t dst_stop, bool with_limiter, bool has_mask_in_c,
    bool emit_tendency, bool axis_is_y);

const defgcm::cuda_launch::LaunchConfig &
tracer_adv_fused_launch_config(int device, at::ScalarType dtype,
                               bool axis_is_y);

namespace {

at::ScalarType parse_scalar_type(const std::string &name) {
  if (name == "float64") {
    return at::kDouble;
  }
  if (name == "float32") {
    return at::kFloat;
  }
  TORCH_CHECK(false, "tracer advection fused dtype must be float64 or float32");
  return at::kDouble;
}

std::atomic<int64_t> native_launch_count{0};

void check_input(const torch::Tensor &value, const torch::Tensor &reference,
                 const char *name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.layout() == c10::kStrided, name,
              " must use strided layout");
  TORCH_CHECK(value.dim() == 4, name, " must be four-dimensional");
  TORCH_CHECK(value.device() == reference.device(), name,
              " must share the reference CUDA device");
  TORCH_CHECK(value.scalar_type() == reference.scalar_type(), name,
              " must share the reference dtype");
  for (int64_t dimension = 0; dimension < value.dim(); ++dimension) {
    TORCH_CHECK(value.stride(dimension) >= 0, name,
                " must not have negative strides");
  }
}

// Each dimension either matches the iteration extent or is 1 (broadcast).
void check_shape(const torch::Tensor &value, const char *name,
                 const std::array<int64_t, 4> &extents) {
  for (int64_t dimension = 0; dimension < 4; ++dimension) {
    const int64_t size = value.size(dimension);
    TORCH_CHECK(size == extents[dimension] || size == 1, name, " dimension ",
                dimension, " is ", size, ", expected ", extents[dimension],
                " or 1");
  }
}

} // namespace

torch::Tensor tracer_adv_pass_fwd(
    torch::Tensor t_wide, torch::Tensor t_left, torch::Tensor t_right,
    torch::Tensor mask_wide, torch::Tensor vel_face, torch::Tensor vel_trans,
    torch::Tensor recip_dc, torch::Tensor recip_vol, torch::Tensor mask_in_c,
    torch::Tensor tracer_orig, torch::Tensor local_t, double dt,
    int64_t dst_start, int64_t dst_stop, bool with_limiter,
    bool has_mask_in_c, bool emit_tendency, const std::string &axis) {
  TORCH_CHECK(axis == "x" || axis == "y",
              "tracer advection fused axis must be \"x\" or \"y\"");
  const bool axis_is_y = axis == "y";

  TORCH_CHECK(tracer_orig.is_cuda() && tracer_orig.dim() == 4,
              "tracer_orig must be a rank-4 CUDA tensor");
  TORCH_CHECK(tracer_orig.scalar_type() == at::kDouble ||
                  tracer_orig.scalar_type() == at::kFloat,
              "tracer advection fused supports float64 or float32");
  TORCH_CHECK(tracer_orig.numel() > 0,
              "tracer advection fused inputs must have positive extents");

  const int64_t batch = tracer_orig.size(0);
  const int64_t levels = tracer_orig.size(1);
  const int64_t ny = tracer_orig.size(2);
  const int64_t nx = tracer_orig.size(3);
  // The DST3 stencil reads T(i-2..i+1); a tile narrower than four cells cannot
  // host it and is rejected upstream (tracer_flux.py:_require_stencil_width).
  TORCH_CHECK(ny >= 4 && nx >= 4,
              "tracer advection fused needs at least four cells per axis");

  const int64_t face_ny = axis_is_y ? ny + 1 : ny;
  const int64_t face_nx = axis_is_y ? nx : nx + 1;
  const int64_t wide_ny = axis_is_y ? ny + 4 : ny;
  const int64_t wide_nx = axis_is_y ? nx : nx + 4;
  const int64_t mask_ny = axis_is_y ? ny + 3 : ny;
  const int64_t mask_nx = axis_is_y ? nx : nx + 3;

  const std::array<const torch::Tensor *, 10> inputs = {
      &t_wide,    &t_left,    &t_right,   &mask_wide, &vel_face,
      &vel_trans, &recip_dc,  &recip_vol, &mask_in_c, &local_t,
  };
  const std::array<const char *, 10> names = {
      "t_wide",    "t_left",    "t_right",   "mask_wide", "vel_face",
      "vel_trans", "recip_dc",  "recip_vol", "mask_in_c", "local_t",
  };
  for (size_t index = 0; index < inputs.size(); ++index) {
    check_input(*inputs[index], tracer_orig, names[index]);
  }
  check_input(tracer_orig, tracer_orig, "tracer_orig");

  check_shape(t_wide, "t_wide", {batch, levels, wide_ny, wide_nx});
  check_shape(t_left, "t_left", {batch, levels, face_ny, face_nx});
  check_shape(t_right, "t_right", {batch, levels, face_ny, face_nx});
  check_shape(mask_wide, "mask_wide", {batch, levels, mask_ny, mask_nx});
  check_shape(vel_face, "vel_face", {batch, levels, face_ny, face_nx});
  check_shape(vel_trans, "vel_trans", {batch, levels, face_ny, face_nx});
  check_shape(recip_dc, "recip_dc", {batch, levels, face_ny, face_nx});
  check_shape(recip_vol, "recip_vol", {batch, levels, ny, nx});
  check_shape(local_t, "local_t", {batch, levels, ny, nx});

  // ``has_mask_in_c`` is the caller's decision (the eager branch it mirrors is
  // ``maskInC is not None``), never re-derived here.  With it false the caller
  // passes a [1,1,1,1] placeholder the kernel never dereferences; with it true
  // the shape check below refuses that placeholder.
  if (has_mask_in_c) {
    check_shape(mask_in_c, "mask_in_c", {batch, levels, ny, nx});
  }

  TORCH_CHECK(dst_start >= 0 && dst_stop >= 0,
              "tracer advection fused splice bounds must be non-negative");
  const int64_t face_count = axis_is_y ? ny + 1 : nx + 1;
  TORCH_CHECK(dst_start <= face_count && dst_stop <= face_count,
              "tracer advection fused splice bounds exceed the face count");

  auto out = torch::empty({batch, levels, ny, nx}, tracer_orig.options());
  tracer_adv_pass_launch(t_wide, t_left, t_right, mask_wide, vel_face,
                         vel_trans, recip_dc, recip_vol, mask_in_c, tracer_orig,
                         local_t, out, dt, dst_start, dst_stop, with_limiter,
                         has_mask_in_c, emit_tendency, axis_is_y);
  native_launch_count.fetch_add(1, std::memory_order_relaxed);
  return out;
}

pybind11::dict launch_config(int device, const std::string &dtype,
                             const std::string &axis) {
  TORCH_CHECK(axis == "x" || axis == "y",
              "tracer advection fused axis must be \"x\" or \"y\"");
  pybind11::dict result = defgcm::cuda_launch::to_python(
      tracer_adv_fused_launch_config(device, parse_scalar_type(dtype),
                                     axis == "y"));
  result["dtype"] = dtype;
  result["axis"] = axis;
  return result;
}

int64_t get_native_launch_count() {
  return native_launch_count.load(std::memory_order_relaxed);
}

void reset_native_launch_count() {
  native_launch_count.store(0, std::memory_order_relaxed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_config", &launch_config,
        "Cached fused tracer-advection pass launch configuration",
        pybind11::arg("device"), pybind11::arg("dtype") = "float64",
        pybind11::arg("axis") = "x");
  m.def("tracer_adv_pass_fwd", &tracer_adv_pass_fwd,
        "One fused directional pass of multidimensional tracer advection");
  m.def("native_launch_count", &get_native_launch_count,
        "Number of native fused tracer-advection launches requested");
  m.def("reset_native_launch_count", &reset_native_launch_count,
        "Reset the native fused tracer-advection launch counter");
}
