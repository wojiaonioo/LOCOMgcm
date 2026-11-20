#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

#include "launch_config.h"

namespace {

// The CUDA runtime API defines no such callback typedef; only the driver API
// spells this type. See defgcm/solvers/_cuda_launch/launch_config.h.
using OccupancyB2DSize = size_t (*)(int);

// The recurrence is runtime-sized; the validated resource envelope is nz<=90.
constexpr int kMaxSupportedNz = 90;

// A translation unit may declare `extern __shared__` exactly once, so the two
// dtype instantiations alias one 8-byte-aligned byte array instead of each
// declaring their own typed view. The base address is the start of the dynamic
// shared window either way, so the float64 instantiation indexes exactly the
// bytes it indexed before this kernel was templated.
template <typename scalar_t>
__device__ __forceinline__ scalar_t* dynamic_shared() {
  extern __shared__ __align__(8) unsigned char shared_bytes[];
  return reinterpret_cast<scalar_t*>(shared_bytes);
}

struct Strides4 {
  int64_t batch;
  int64_t level;
  int64_t y;
  int64_t x;
};

Strides4 strides4(const torch::Tensor& value) {
  return {
      value.stride(0),
      value.stride(1),
      value.stride(2),
      value.stride(3)};
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t load4(
    const scalar_t* data,
    const Strides4 strides,
    const int64_t batch,
    const int level,
    const int64_t y,
    const int64_t x) {
  const int64_t index =
      batch * strides.batch + static_cast<int64_t>(level) * strides.level +
      y * strides.y + x * strides.x;
  return __ldg(data + index);
}

template <typename scalar_t, CPrimeStorage kStorage>
__device__ __forceinline__ scalar_t staged_c_prime(
    const scalar_t* staging,
    const int k,
    const int lane,
    const int64_t output_index) {
  if constexpr (kStorage == CPrimeStorage::kShared) {
    return staging[k * blockDim.x + lane];
  } else {
    return staging[output_index];
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t recompute_c_prime(
    const scalar_t* __restrict__ a,
    const Strides4 a_strides,
    const scalar_t* __restrict__ b,
    const Strides4 b_strides,
    const scalar_t* __restrict__ c,
    const Strides4 c_strides,
    const int64_t batch,
    const bool mitgcm_order,
    const int target_level,
    const int64_t y,
    const int64_t x) {
  const scalar_t b0 = load4(b, b_strides, batch, 0, y, x);
  const scalar_t reciprocal0 = static_cast<scalar_t>(1.0) / b0;
  const scalar_t c0 = load4(c, c_strides, batch, 0, y, x);
  scalar_t previous_c = mitgcm_order ? c0 * reciprocal0 : c0 / b0;
  for (int k = 1; k <= target_level; ++k) {
    const scalar_t a_value = load4(a, a_strides, batch, k, y, x);
    const scalar_t b_value = load4(b, b_strides, batch, k, y, x);
    const scalar_t c_value = load4(c, c_strides, batch, k, y, x);
    const scalar_t denominator_product = a_value * previous_c;
    const scalar_t denominator = b_value - denominator_product;
    previous_c = mitgcm_order ? c_value * (static_cast<scalar_t>(1.0) / denominator) : c_value / denominator;
  }
  return previous_c;
}

template <typename scalar_t, CPrimeStorage kStorage>
__device__ __forceinline__ void tridiag_thomas_body(
    const scalar_t* __restrict__ a,
    const Strides4 a_strides,
    const scalar_t* __restrict__ b,
    const Strides4 b_strides,
    const scalar_t* __restrict__ c,
    const Strides4 c_strides,
    const scalar_t* __restrict__ d,
    const Strides4 d_strides,
    scalar_t* __restrict__ staging,
    scalar_t* __restrict__ output,
    const int64_t batch_size,
    const int nz,
    const int64_t ny,
    const int64_t nx,
    const bool mitgcm_order) {
  const int lane = threadIdx.x;
  const int64_t nplane = ny * nx;
  const int64_t total_columns = batch_size * nplane;
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + lane;
  if (column >= total_columns) {
    return;
  }

  const int64_t batch = column / nplane;
  const int64_t plane_column = column - batch * nplane;
  const int64_t y = plane_column / nx;
  const int64_t x = plane_column - y * nx;

  const scalar_t b0 = load4(b, b_strides, batch, 0, y, x);
  const scalar_t reciprocal0 = static_cast<scalar_t>(1.0) / b0;
  const scalar_t c0 = load4(c, c_strides, batch, 0, y, x);
  scalar_t previous_c = mitgcm_order ? c0 * reciprocal0 : c0 / b0;
  const scalar_t d0 = load4(d, d_strides, batch, 0, y, x);
  scalar_t previous_d = mitgcm_order ? d0 * reciprocal0 : d0 / b0;
  const int64_t first_output_index =
      batch * static_cast<int64_t>(nz) * nplane + plane_column;
  if constexpr (kStorage == CPrimeStorage::kShared) {
    staging[lane] = previous_c;
  } else if constexpr (kStorage == CPrimeStorage::kGlobal) {
    staging[first_output_index] = previous_c;
  }
  output[first_output_index] = previous_d;

  for (int k = 1; k < nz; ++k) {
    const scalar_t a_value = load4(a, a_strides, batch, k, y, x);
    const scalar_t b_value = load4(b, b_strides, batch, k, y, x);
    const scalar_t c_value = load4(c, c_strides, batch, k, y, x);
    const scalar_t d_value = load4(d, d_strides, batch, k, y, x);

    const scalar_t pivot = b_value - a_value * previous_c;
    const scalar_t reciprocal = static_cast<scalar_t>(1.0) / pivot;
    const scalar_t current_c = mitgcm_order ? c_value * reciprocal : c_value / pivot;
    const scalar_t numerator = d_value - a_value * previous_d;
    const scalar_t current_d = mitgcm_order ? numerator * reciprocal : numerator / pivot;

    const int64_t output_index =
        (batch * static_cast<int64_t>(nz) + k) * nplane + plane_column;
    if constexpr (kStorage == CPrimeStorage::kShared) {
      staging[k * blockDim.x + lane] = current_c;
    } else if constexpr (kStorage == CPrimeStorage::kGlobal) {
      staging[output_index] = current_c;
    }
    output[output_index] = current_d;
    previous_c = current_c;
    previous_d = current_d;
  }

  scalar_t next_result = previous_d;
  for (int k = nz - 2; k >= 0; --k) {
    const int64_t output_index =
        (batch * static_cast<int64_t>(nz) + k) * nplane + plane_column;
    scalar_t current_c;
    if constexpr (kStorage == CPrimeStorage::kRecomputeProbe) {
      current_c = recompute_c_prime(
          a,
          a_strides,
          b,
          b_strides,
          c,
          c_strides,
          batch,
          mitgcm_order,
          k,
          y,
          x);
    } else {
      current_c =
          staged_c_prime<scalar_t, kStorage>(staging, k, lane, output_index);
    }
    const scalar_t backward_product = current_c * next_result;
    next_result = output[output_index] - backward_product;
    output[output_index] = next_result;
  }
}

template <typename scalar_t>
__global__ void tridiag_shared_kernel(
    const scalar_t* __restrict__ a,
    const Strides4 a_strides,
    const scalar_t* __restrict__ b,
    const Strides4 b_strides,
    const scalar_t* __restrict__ c,
    const Strides4 c_strides,
    const scalar_t* __restrict__ d,
    const Strides4 d_strides,
    scalar_t* __restrict__ output,
    const int64_t batch_size,
    const int nz,
    const int64_t ny,
    const int64_t nx,
    const bool mitgcm_order) {
  scalar_t* shared = dynamic_shared<scalar_t>();
  tridiag_thomas_body<scalar_t, CPrimeStorage::kShared>(
      a,
      a_strides,
      b,
      b_strides,
      c,
      c_strides,
      d,
      d_strides,
      shared,
      output,
      batch_size,
      nz,
      ny,
      nx, mitgcm_order);
}

template <typename scalar_t>
__global__ void tridiag_global_kernel(
    const scalar_t* __restrict__ a,
    const Strides4 a_strides,
    const scalar_t* __restrict__ b,
    const Strides4 b_strides,
    const scalar_t* __restrict__ c,
    const Strides4 c_strides,
    const scalar_t* __restrict__ d,
    const Strides4 d_strides,
    scalar_t* __restrict__ c_prime,
    scalar_t* __restrict__ output,
    const int64_t batch_size,
    const int nz,
    const int64_t ny,
    const int64_t nx,
    const bool mitgcm_order) {
  tridiag_thomas_body<scalar_t, CPrimeStorage::kGlobal>(
      a,
      a_strides,
      b,
      b_strides,
      c,
      c_strides,
      d,
      d_strides,
      c_prime,
      output,
      batch_size,
      nz,
      ny,
      nx, mitgcm_order);
}

// Occupancy-only probe. Dispatch never selects this arithmetic variant because
// recomputing every c_prime prefix changes the operation count from O(nz) to
// O(nz^2), outside the launch-only optimization contract for this backend.
template <typename scalar_t>
__global__ void tridiag_recompute_probe_kernel(
    const scalar_t* __restrict__ a,
    const Strides4 a_strides,
    const scalar_t* __restrict__ b,
    const Strides4 b_strides,
    const scalar_t* __restrict__ c,
    const Strides4 c_strides,
    const scalar_t* __restrict__ d,
    const Strides4 d_strides,
    scalar_t* __restrict__ output,
    const int64_t batch_size,
    const int nz,
    const int64_t ny,
    const int64_t nx,
    const bool mitgcm_order) {
  tridiag_thomas_body<scalar_t, CPrimeStorage::kRecomputeProbe>(
      a,
      a_strides,
      b,
      b_strides,
      c,
      c_strides,
      d,
      d_strides,
      nullptr,
      output,
      batch_size,
      nz,
      ny,
      nx, mitgcm_order);
}

thread_local int occupancy_nz = 0;

template <typename scalar_t>
size_t shared_bytes_for_block(int block_size) {
  return static_cast<size_t>(occupancy_nz) * block_size * sizeof(scalar_t);
}

size_t no_dynamic_shared_for_block(int) {
  return 0;
}

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(
      status == cudaSuccess,
      operation,
      " failed: ",
      cudaGetErrorString(status));
}

template <typename Kernel>
TridiagCandidateConfig inspect_candidate(
    CPrimeStorage storage,
    bool selectable,
    Kernel kernel,
    OccupancyB2DSize dynamic_shared_callback,
    const cudaDeviceProp& properties,
    int nz,
    int block_size_override) {
  cudaFuncAttributes attributes{};
  check_cuda(
      cudaFuncGetAttributes(&attributes, kernel), "cudaFuncGetAttributes");

  occupancy_nz = nz;
  int minimum_grid_size = 0;
  int occupancy_block_size = 0;
  check_cuda(
      cudaOccupancyMaxPotentialBlockSizeVariableSMem(
          &minimum_grid_size,
          &occupancy_block_size,
          kernel,
          dynamic_shared_callback,
          attributes.maxThreadsPerBlock),
      "cudaOccupancyMaxPotentialBlockSizeVariableSMem");
  const int block_size =
      block_size_override > 0 ? block_size_override : occupancy_block_size;
  TORCH_CHECK(
      block_size <= attributes.maxThreadsPerBlock,
      "requested tridiagonal CUDA block size ",
      block_size,
      " exceeds kernel limit ",
      attributes.maxThreadsPerBlock);

  const size_t dynamic_shared_bytes = dynamic_shared_callback(block_size);
  TORCH_CHECK(
      dynamic_shared_bytes <=
          static_cast<size_t>(attributes.maxDynamicSharedSizeBytes),
      "tridiagonal CUDA candidate requires ",
      dynamic_shared_bytes,
      " dynamic shared-memory bytes, but the configured kernel permits ",
      attributes.maxDynamicSharedSizeBytes);

  int active_blocks = 0;
  check_cuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, kernel, block_size, dynamic_shared_bytes),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  TORCH_CHECK(
      active_blocks > 0,
      "occupancy API found no active tridiagonal CUDA blocks for block size ",
      block_size);

  const int64_t shared_bytes_per_block =
      static_cast<int64_t>(attributes.sharedSizeBytes) +
      static_cast<int64_t>(dynamic_shared_bytes);
  const int shared_block_limit =
      shared_bytes_per_block == 0
      ? -1
      : static_cast<int>(
            properties.sharedMemPerMultiprocessor / shared_bytes_per_block);
  const int thread_block_limit =
      properties.maxThreadsPerMultiProcessor / block_size;
  const int register_block_limit =
      attributes.numRegs == 0
      ? -1
      : static_cast<int>(
            static_cast<int64_t>(properties.regsPerMultiprocessor) /
            (static_cast<int64_t>(attributes.numRegs) * block_size));
  const int warps_per_block =
      (block_size + properties.warpSize - 1) / properties.warpSize;
  const int resident_warps = active_blocks * warps_per_block;
  const int maximum_warps =
      properties.maxThreadsPerMultiProcessor / properties.warpSize;

  std::vector<std::string> limiting_resources;
  if (shared_block_limit >= 0 && active_blocks == shared_block_limit) {
    limiting_resources.emplace_back("shared");
  }
  if (active_blocks == thread_block_limit) {
    limiting_resources.emplace_back("threads");
  }
  if (register_block_limit >= 0 && active_blocks == register_block_limit) {
    limiting_resources.emplace_back("regs");
  }
  if (limiting_resources.empty()) {
    limiting_resources.emplace_back("occupancy_api");
  }

  return {
      storage,
      selectable,
      occupancy_block_size,
      block_size,
      minimum_grid_size,
      attributes.numRegs,
      static_cast<int64_t>(attributes.sharedSizeBytes),
      static_cast<int64_t>(dynamic_shared_bytes),
      attributes.maxThreadsPerBlock,
      shared_block_limit,
      thread_block_limit,
      register_block_limit,
      active_blocks,
      resident_warps,
      100.0 * resident_warps / maximum_warps,
      limiting_resources};
}

const TridiagCandidateConfig& candidate_for_storage(
    const TridiagLaunchConfig& config,
    CPrimeStorage storage) {
  for (const auto& candidate : config.candidates) {
    if (candidate.storage == storage) {
      return candidate;
    }
  }
  TORCH_CHECK(false, "missing tridiagonal CUDA storage candidate");
  return config.candidates.front();
}

template <typename scalar_t>
TridiagLaunchConfig derive_launch_config_typed(
    int device,
    int nz,
    const std::string& storage_override,
    int block_size_override,
    const cudaDeviceProp& properties) {
  cudaFuncAttributes shared_attributes{};
  check_cuda(
      cudaFuncGetAttributes(
          &shared_attributes, tridiag_shared_kernel<scalar_t>),
      "cudaFuncGetAttributes(shared)");
  const size_t optin_shared_bytes = std::max(
      properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin);
  TORCH_CHECK(
      optin_shared_bytes >= shared_attributes.sharedSizeBytes,
      "static shared memory exceeds the device per-block capacity");
  const size_t maximum_dynamic_shared_bytes =
      optin_shared_bytes - shared_attributes.sharedSizeBytes;
  TORCH_CHECK(
      maximum_dynamic_shared_bytes <=
          static_cast<size_t>(std::numeric_limits<int>::max()),
      "dynamic shared-memory opt-in capacity exceeds CUDA attribute range");
  check_cuda(
      cudaFuncSetAttribute(
          tridiag_shared_kernel<scalar_t>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(maximum_dynamic_shared_bytes)),
      "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");

  TridiagLaunchConfig config{};
  config.device = device;
  config.nz = nz;
  config.device_name = properties.name;
  config.compute_capability_major = properties.major;
  config.compute_capability_minor = properties.minor;
  config.multiprocessor_count = properties.multiProcessorCount;
  config.warp_size = properties.warpSize;
  config.max_threads_per_sm = properties.maxThreadsPerMultiProcessor;
  config.shared_bytes_per_sm = properties.sharedMemPerMultiprocessor;
  config.registers_per_sm = properties.regsPerMultiprocessor;
  config.optin_shared_bytes_per_block = optin_shared_bytes;
  config.storage_override = storage_override;
  config.block_size_override = block_size_override;
  config.candidates.push_back(inspect_candidate(
      CPrimeStorage::kShared,
      true,
      tridiag_shared_kernel<scalar_t>,
      shared_bytes_for_block<scalar_t>,
      properties,
      nz,
      block_size_override));
  config.candidates.push_back(inspect_candidate(
      CPrimeStorage::kGlobal,
      true,
      tridiag_global_kernel<scalar_t>,
      no_dynamic_shared_for_block,
      properties,
      nz,
      block_size_override));
  config.candidates.push_back(inspect_candidate(
      CPrimeStorage::kRecomputeProbe,
      false,
      tridiag_recompute_probe_kernel<scalar_t>,
      no_dynamic_shared_for_block,
      properties,
      nz,
      block_size_override));

  const auto& shared = candidate_for_storage(config, CPrimeStorage::kShared);
  const auto& global = candidate_for_storage(config, CPrimeStorage::kGlobal);
  if (!storage_override.empty()) {
    TORCH_CHECK(
        storage_override == "shared" || storage_override == "global",
        "DEFGCM_TRIDIAG_CUDA_CPRIME_STORAGE must be shared or global");
    config.selected_storage = storage_override == "shared"
        ? CPrimeStorage::kShared
        : CPrimeStorage::kGlobal;
    config.selection_reason = "diagnostic storage override";
  } else if (global.resident_warps_per_sm > shared.resident_warps_per_sm) {
    config.selected_storage = CPrimeStorage::kGlobal;
    std::ostringstream reason;
    reason << "global exposes " << global.resident_warps_per_sm
           << " resident warps/SM versus " << shared.resident_warps_per_sm
           << " for shared";
    config.selection_reason = reason.str();
  } else {
    config.selected_storage = CPrimeStorage::kShared;
    config.selection_reason =
        "shared has no lower resident-warp count and avoids global staging traffic";
  }
  config.selected_block_size =
      candidate_for_storage(config, config.selected_storage).block_size;
  return config;
}

template <typename scalar_t>
void launch_typed(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& d,
    torch::Tensor& output,
    CPrimeStorage storage,
    int block_size,
    bool mitgcm_order) {
  const auto batch = static_cast<int64_t>(d.size(0));
  const auto nz = static_cast<int>(d.size(1));
  const auto ny = static_cast<int64_t>(d.size(2));
  const auto nx = static_cast<int64_t>(d.size(3));
  const int64_t total_columns = batch * ny * nx;
  const int64_t block_count =
      (total_columns + block_size - 1) / block_size;
  TORCH_CHECK(
      block_count <= std::numeric_limits<unsigned int>::max(),
      "tridiagonal CUDA grid exceeds grid.x capacity");

  const dim3 block(block_size);
  const dim3 grid(static_cast<unsigned int>(block_count));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (storage == CPrimeStorage::kShared) {
    const size_t shared_bytes =
        static_cast<size_t>(nz) * block_size * sizeof(scalar_t);
    tridiag_shared_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
        a.data_ptr<scalar_t>(),
        strides4(a),
        b.data_ptr<scalar_t>(),
        strides4(b),
        c.data_ptr<scalar_t>(),
        strides4(c),
        d.data_ptr<scalar_t>(),
        strides4(d),
        output.data_ptr<scalar_t>(),
        batch,
        nz,
        ny,
        nx, mitgcm_order);
  } else {
    TORCH_CHECK(
        storage == CPrimeStorage::kGlobal,
        "recompute occupancy probe must never be dispatched");
    auto c_prime = torch::empty(d.sizes(), d.options());
    tridiag_global_kernel<scalar_t><<<grid, block, 0, stream>>>(
        a.data_ptr<scalar_t>(),
        strides4(a),
        b.data_ptr<scalar_t>(),
        strides4(b),
        c.data_ptr<scalar_t>(),
        strides4(c),
        d.data_ptr<scalar_t>(),
        strides4(d),
        c_prime.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch,
        nz,
        ny,
        nx, mitgcm_order);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TridiagLaunchConfig tridiag_cuda_derive_launch_config(
    int device,
    int nz,
    const std::string& storage_override,
    int block_size_override,
    at::ScalarType dtype) {
  TORCH_CHECK(nz > 0 && nz <= kMaxSupportedNz,
              "launch configuration supports 1 through 90 vertical levels");
  TORCH_CHECK(
      block_size_override >= 0,
      "tridiagonal CUDA block-size override must be non-negative");
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));

  cudaDeviceProp properties{};
  check_cuda(
      cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

  if (dtype == at::kFloat) {
    return derive_launch_config_typed<float>(
        device, nz, storage_override, block_size_override, properties);
  }
  TORCH_CHECK(
      dtype == at::kDouble,
      "tridiagonal CUDA launch configuration supports float64 or float32 only");
  return derive_launch_config_typed<double>(
      device, nz, storage_override, block_size_override, properties);
}

void tridiag_cuda_launch(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& d,
    torch::Tensor& output,
    CPrimeStorage storage,
    int block_size,
    bool mitgcm_order) {
  c10::cuda::CUDAGuard device_guard(d.device());
  if (d.scalar_type() == at::kFloat) {
    launch_typed<float>(a, b, c, d, output, storage, block_size, mitgcm_order);
    return;
  }
  TORCH_CHECK(
      d.scalar_type() == at::kDouble,
      "tridiagonal CUDA kernel supports float64 or float32 only");
  launch_typed<double>(a, b, c, d, output, storage, block_size, mitgcm_order);
}
