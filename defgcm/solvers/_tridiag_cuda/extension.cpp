#include <torch/extension.h>

#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "launch_config.h"

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
// The kernel is instantiated for exactly these two scalar types; this predicate
// and the instantiation list in kernel.cu must stay one decision.
#define CHECK_FLOATING(x)                                                    \
  TORCH_CHECK(                                                               \
      (x).scalar_type() == at::kDouble || (x).scalar_type() == at::kFloat,   \
      #x " must be float64 or float32")
#define CHECK_4D(x) \
  TORCH_CHECK((x).dim() == 4, #x " must be four-dimensional")
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
  TORCH_CHECK(false, "tridiagonal CUDA dtype must be float64 or float32");
  return at::kDouble;
}

void check_inputs(const std::vector<torch::Tensor>& values) {
  const torch::Tensor& reference = values.front();
  for (const auto& value : values) {
    CHECK_INPUT(value);
    TORCH_CHECK(
        value.device() == reference.device(),
        "all tridiagonal inputs must be on the same device");
    TORCH_CHECK(
        value.sizes() == reference.sizes(),
        "all tridiagonal inputs must have the same shape");
    TORCH_CHECK(
        value.scalar_type() == reference.scalar_type(),
        "all tridiagonal inputs must have the same dtype");
    for (int dimension = 0; dimension < value.dim(); ++dimension) {
      TORCH_CHECK(
          value.stride(dimension) >= 0,
          "tridiagonal CUDA inputs must not have negative strides");
    }
  }
  TORCH_CHECK(
      reference.size(0) > 0 && reference.size(1) > 0 &&
          reference.size(2) > 0 && reference.size(3) > 0,
      "tridiagonal CUDA input extents must be positive");
  TORCH_CHECK(
      reference.size(1) <= 90,
      "tridiagonal CUDA kernel supports 1 through 90 vertical levels");
}

}  // namespace

pybind11::dict launch_config_to_dict(const TridiagLaunchConfig& config) {
  pybind11::dict result;
  result["device"] = config.device;
  result["nz"] = config.nz;
  result["device_name"] = config.device_name;
  result["compute_capability_major"] = config.compute_capability_major;
  result["compute_capability_minor"] = config.compute_capability_minor;
  result["multiprocessor_count"] = config.multiprocessor_count;
  result["warp_size"] = config.warp_size;
  result["max_threads_per_sm"] = config.max_threads_per_sm;
  result["shared_bytes_per_sm"] = config.shared_bytes_per_sm;
  result["registers_per_sm"] = config.registers_per_sm;
  result["optin_shared_bytes_per_block"] =
      config.optin_shared_bytes_per_block;
  result["selected_storage"] =
      config.selected_storage == CPrimeStorage::kShared ? "shared" : "global";
  result["selected_storage_code"] = static_cast<int>(config.selected_storage);
  result["selected_block_size"] = config.selected_block_size;
  result["selection_reason"] = config.selection_reason;
  result["storage_override"] = config.storage_override;
  result["block_size_override"] = config.block_size_override;

  pybind11::list candidates;
  for (const auto& candidate : config.candidates) {
    pybind11::dict item;
    const char* storage_name = "recompute";
    if (candidate.storage == CPrimeStorage::kShared) {
      storage_name = "shared";
    } else if (candidate.storage == CPrimeStorage::kGlobal) {
      storage_name = "global";
    }
    item["storage"] = storage_name;
    item["selectable"] = candidate.selectable;
    item["occupancy_block_size"] = candidate.occupancy_block_size;
    item["block_size"] = candidate.block_size;
    item["minimum_grid_size"] = candidate.minimum_grid_size;
    item["registers_per_thread"] = candidate.registers_per_thread;
    item["static_shared_bytes"] = candidate.static_shared_bytes;
    item["dynamic_shared_bytes"] = candidate.dynamic_shared_bytes;
    item["max_threads_per_block"] = candidate.max_threads_per_block;
    item["shared_block_limit"] = candidate.shared_block_limit;
    item["thread_block_limit"] = candidate.thread_block_limit;
    item["register_block_limit"] = candidate.register_block_limit;
    item["active_blocks_per_sm"] = candidate.active_blocks_per_sm;
    item["resident_warps_per_sm"] = candidate.resident_warps_per_sm;
    item["occupancy_percent"] = candidate.occupancy_percent;
    item["limiting_resources"] = candidate.limiting_resources;
    candidates.append(item);
  }
  result["candidates"] = candidates;
  return result;
}

pybind11::dict tridiag_launch_config(
    int device,
    int nz,
    const std::string& storage_override,
    int block_size_override,
    const std::string& dtype) {
  pybind11::dict result = launch_config_to_dict(tridiag_cuda_derive_launch_config(
      device, nz, storage_override, block_size_override,
      parse_scalar_type(dtype)));
  result["dtype"] = dtype;
  return result;
}

torch::Tensor tridiag_fwd(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    int storage_code,
    int block_size,
    bool mitgcm_order) {
  check_inputs({a, b, c, d});
  TORCH_CHECK(
      storage_code == static_cast<int>(CPrimeStorage::kShared) ||
          storage_code == static_cast<int>(CPrimeStorage::kGlobal),
      "tridiagonal CUDA storage must be shared or global");
  TORCH_CHECK(block_size > 0, "tridiagonal CUDA block size must be positive");
  auto output = torch::empty(d.sizes(), d.options());
  tridiag_cuda_launch(
      a,
      b,
      c,
      d,
      output,
      static_cast<CPrimeStorage>(storage_code),
      block_size,
      mitgcm_order);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "launch_config",
      &tridiag_launch_config,
      "Derive the cached tridiagonal CUDA launch configuration",
      pybind11::arg("device"),
      pybind11::arg("nz"),
      pybind11::arg("storage_override"),
      pybind11::arg("block_size_override"),
      pybind11::arg("dtype") = "float64");
  m.def(
      "tridiag_fwd",
      &tridiag_fwd,
      "Occupancy-configured tridiagonal Thomas solve (CUDA)",
      pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"), pybind11::arg("d"),
      pybind11::arg("storage_code"), pybind11::arg("block_size"),
      pybind11::arg("mitgcm_order") = false);
}
