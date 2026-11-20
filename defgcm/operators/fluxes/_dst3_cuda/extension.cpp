#include <torch/extension.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "../../../solvers/_cuda_launch/pybind.h"

void dst3_transport_flux_launch(
    const torch::Tensor &t_left, const torch::Tensor &t_right,
    const torch::Tensor &tm2, const torch::Tensor &tm1, const torch::Tensor &t0,
    const torch::Tensor &tp1, const torch::Tensor &mask_im1,
    const torch::Tensor &mask_i, const torch::Tensor &mask_ip1,
    const torch::Tensor &u_trans, const torch::Tensor &u_cfl,
    torch::Tensor &upwind_flux, torch::Tensor &flux_dst, bool with_limiter);

const defgcm::cuda_launch::LaunchConfig &
dst3_cuda_launch_config(int device, at::ScalarType dtype);

namespace {

at::ScalarType parse_scalar_type(const std::string& name) {
  if (name == "float64") {
    return at::kDouble;
  }
  if (name == "float32") {
    return at::kFloat;
  }
  TORCH_CHECK(false, "DST3 CUDA dtype must be float64 or float32");
  return at::kDouble;
}

std::atomic<int64_t> native_launch_count{0};

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
// The kernel is instantiated for both floating dtypes; this predicate names
// exactly the set AT_DISPATCH_FLOATING_TYPES in kernel.cu covers.
#define CHECK_FLOATING(x)                                                      \
  TORCH_CHECK((x).scalar_type() == at::kDouble ||                              \
                  (x).scalar_type() == at::kFloat,                             \
              #x " must be float64 or float32")
#define CHECK_RANK(x)                                                          \
  TORCH_CHECK((x).dim() == 3 || (x).dim() == 4,                                \
              #x " must be three- or four-dimensional")
#define CHECK_STRIDED(x)                                                       \
  TORCH_CHECK((x).layout() == c10::kStrided, #x " must use strided layout")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_FLOATING(x);                                                             \
  CHECK_RANK(x);                                                               \
  CHECK_STRIDED(x)

int64_t batch_extent(const torch::Tensor &value) {
  return value.dim() == 4 ? value.size(0) : 1;
}

int64_t trailing_extent(const torch::Tensor &value, int64_t dimension) {
  return value.size(value.dim() - 3 + dimension);
}

void check_non_negative_strides(const torch::Tensor &value, const char *name) {
  for (int dimension = 0; dimension < value.dim(); ++dimension) {
    TORCH_CHECK(value.stride(dimension) >= 0, name,
                " must not have negative strides");
  }
}

void check_matching_input(const torch::Tensor &value,
                          const torch::Tensor &reference,
                          const int64_t output_batch, const char *name) {
  for (int64_t dimension = 0; dimension < 3; ++dimension) {
    TORCH_CHECK(trailing_extent(value, dimension) ==
                    trailing_extent(reference, dimension),
                name, " trailing dimensions must match T_left");
  }
  TORCH_CHECK(batch_extent(value) == 1 || batch_extent(value) == output_batch,
              name, " leading batch dimension is not broadcast-compatible");
  TORCH_CHECK(value.device() == reference.device(), name,
              " must be on the same CUDA device as T_left");
  TORCH_CHECK(value.scalar_type() == reference.scalar_type(), name,
              " must have the same dtype as T_left");
  check_non_negative_strides(value, name);
}

} // namespace

std::vector<torch::Tensor> dst3_transport_flux_fwd(
    torch::Tensor t_left, torch::Tensor t_right, torch::Tensor tm2,
    torch::Tensor tm1, torch::Tensor t0, torch::Tensor tp1,
    torch::Tensor mask_im1, torch::Tensor mask_i, torch::Tensor mask_ip1,
    torch::Tensor u_trans, torch::Tensor u_cfl, bool with_limiter) {
  CHECK_INPUT(t_left);
  CHECK_INPUT(t_right);
  CHECK_INPUT(tm2);
  CHECK_INPUT(tm1);
  CHECK_INPUT(t0);
  CHECK_INPUT(tp1);
  CHECK_INPUT(mask_im1);
  CHECK_INPUT(mask_i);
  CHECK_INPUT(mask_ip1);
  CHECK_INPUT(u_trans);
  CHECK_INPUT(u_cfl);
  const std::array<const torch::Tensor *, 11> inputs = {
      &t_left,   &t_right, &tm2,      &tm1,     &t0,    &tp1,
      &mask_im1, &mask_i,  &mask_ip1, &u_trans, &u_cfl,
  };
  int64_t output_batch = 1;
  bool output_is_four_dimensional = false;
  for (const torch::Tensor *value : inputs) {
    TORCH_CHECK(value->numel() > 0,
                "DST3 CUDA inputs must have positive extents");
    if (value->dim() == 4) {
      output_is_four_dimensional = true;
      if (value->size(0) > output_batch) {
        output_batch = value->size(0);
      }
    }
  }

  check_matching_input(t_left, t_left, output_batch, "T_left");
  check_matching_input(t_right, t_left, output_batch, "T_right");
  check_matching_input(tm2, t_left, output_batch, "Tm2");
  check_matching_input(tm1, t_left, output_batch, "Tm1");
  check_matching_input(t0, t_left, output_batch, "T0");
  check_matching_input(tp1, t_left, output_batch, "Tp1");
  check_matching_input(mask_im1, t_left, output_batch, "mask_im1");
  check_matching_input(mask_i, t_left, output_batch, "mask_i");
  check_matching_input(mask_ip1, t_left, output_batch, "mask_ip1");
  check_matching_input(u_trans, t_left, output_batch, "u_trans");
  check_matching_input(u_cfl, t_left, output_batch, "u_cfl");

  std::vector<int64_t> output_sizes;
  output_sizes.reserve(output_is_four_dimensional ? 4 : 3);
  if (output_is_four_dimensional) {
    output_sizes.push_back(output_batch);
  }
  for (int64_t dimension = 0; dimension < 3; ++dimension) {
    output_sizes.push_back(trailing_extent(t_left, dimension));
  }
  auto upwind_flux = torch::empty(output_sizes, t_left.options());
  auto flux_dst = torch::empty(output_sizes, t_left.options());
  dst3_transport_flux_launch(t_left, t_right, tm2, tm1, t0, tp1, mask_im1,
                             mask_i, mask_ip1, u_trans, u_cfl, upwind_flux,
                             flux_dst, with_limiter);
  native_launch_count.fetch_add(1, std::memory_order_relaxed);
  return {upwind_flux, flux_dst};
}

pybind11::dict launch_config(int device, const std::string &dtype) {
  pybind11::dict result = defgcm::cuda_launch::to_python(
      dst3_cuda_launch_config(device, parse_scalar_type(dtype)));
  result["dtype"] = dtype;
  return result;
}

int64_t get_native_launch_count() {
  return native_launch_count.load(std::memory_order_relaxed);
}

void reset_native_launch_count() {
  native_launch_count.store(0, std::memory_order_relaxed);
}


void dst3_transport_flux_bwd_launch(
    const std::vector<torch::Tensor> &values,
    const std::vector<c10::optional<torch::Tensor>> &seeds,
    std::vector<c10::optional<torch::Tensor>> &gradients,
    int64_t batch, bool limited);
const defgcm::cuda_launch::LaunchConfig &
dst3_cuda_backward_launch_config(int device, at::ScalarType dtype);

std::atomic<int64_t> native_backward_count{0};

std::vector<c10::optional<torch::Tensor>> dst3_transport_flux_bwd(
    const std::vector<torch::Tensor> &values,
    const std::vector<c10::optional<torch::Tensor>> &seeds,
    bool limited, const std::vector<bool> &needs) {
  TORCH_CHECK(values.size() == 11 && seeds.size() == 2 && needs.size() == 11,
              "DST3 backward arity mismatch");
  int64_t batch = 1;
  bool four = false;
  for (const auto &v : values) {
    CHECK_INPUT(v);
    TORCH_CHECK(v.numel() > 0, "DST3 backward requires nonempty inputs");
    batch = std::max(batch, batch_extent(v));
    four |= v.dim() == 4;
  }
  std::vector<int64_t> shape;
  if (four) shape.push_back(batch);
  for (int d = 0; d < 3; ++d) shape.push_back(trailing_extent(values[0], d));
  for (const auto &v : values) check_matching_input(v, values[0], batch, "backward input");
  for (const auto &seed : seeds) {
    if (!seed.has_value()) continue;
    CHECK_INPUT(*seed);
    check_matching_input(*seed, values[0], batch, "cotangent");
    TORCH_CHECK(seed->sizes().vec() == shape, "DST3 cotangent shape mismatch");
  }
  std::vector<c10::optional<torch::Tensor>> gradients(11);
  for (int q = 0; q < 11; ++q)
    if (needs[q]) gradients[q] = torch::empty(values[q].sizes(), values[q].options());
  dst3_transport_flux_bwd_launch(values, seeds, gradients, batch, limited);
  native_backward_count.fetch_add(1, std::memory_order_relaxed);
  return gradients;
}

pybind11::dict backward_launch_config(int device, const std::string &dtype) {
  auto result = defgcm::cuda_launch::to_python(
      dst3_cuda_backward_launch_config(device, parse_scalar_type(dtype)));
  result["dtype"] = dtype;
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dst3_transport_flux_bwd", &dst3_transport_flux_bwd);
  m.def("backward_launch_config", &backward_launch_config);
  m.def("native_backward_count", [] { return native_backward_count.load(); });
  m.def("reset_native_backward_count", [] { native_backward_count.store(0); });

  m.def("launch_config", &launch_config,
        "Cached DST3 transport-flux CUDA launch configuration",
        pybind11::arg("device"), pybind11::arg("dtype") = "float64");
  m.def("dst3_transport_flux_fwd", &dst3_transport_flux_fwd,
        "Fused DST3/DST3-FL transport-flux numerical core (CUDA)");
  m.def("native_launch_count", &get_native_launch_count,
        "Number of native DST3 CUDA launches requested");
  m.def("reset_native_launch_count", &reset_native_launch_count,
        "Reset the native DST3 CUDA launch counter");
}
