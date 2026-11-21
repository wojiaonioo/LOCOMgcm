#pragma once

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace locomgcm::cuda_launch {

// Signature of the per-block dynamic-shared-memory callback handed to
// cudaOccupancyMaxPotentialBlockSizeVariableSMem.
//
// The CUDA *runtime* API has no matching typedef: its occupancy entry point is
// a C++ template over the callable, and only the *driver* API spells the type
// (CUoccupancyB2DSize, cuda.h:3241). Naming a runtime variant by analogy does
// not compile -- verified on CUDA 12.8/13.0, where a grep of the runtime
// headers for such a name returns nothing.
using OccupancyB2DSize = std::size_t (*)(int);

inline constexpr const char* kThreadsPerBlockOverride =
    "LOCOMGCM_CUDA_THREADS_PER_BLOCK";

struct DeviceLimits {
  int multiprocessor_count;
  int warp_size;
  int max_threads_per_sm;
  int64_t shared_bytes_per_sm;
  int registers_per_sm;
};

struct KernelLimits {
  int registers_per_thread;
  int64_t static_shared_bytes;
  int max_threads_per_block;
};

struct CandidateConfig {
  int block_size;
  bool default_candidate;
  int active_blocks_per_sm;
  int resident_warps_per_sm;
  double occupancy_percent;
  int64_t dynamic_shared_bytes;
  int shared_block_limit;
  int thread_block_limit;
  int register_block_limit;
  std::vector<std::string> limiting_resources;
  int active_blocks_without_dynamic_shared;
  int resident_warps_without_dynamic_shared;
  double occupancy_without_dynamic_shared_percent;
};

struct LaunchConfig {
  int device;
  int64_t shape_key;
  std::string device_name;
  int compute_capability_major;
  int compute_capability_minor;
  DeviceLimits device_limits;
  KernelLimits kernel_limits;
  int64_t shared_bytes_per_block_cap;
  int64_t maximum_dynamic_shared_bytes;
  int occupancy_api_block_size;
  int occupancy_api_minimum_grid_size;
  int reference_block_size;
  int override_block_size;
  int selected_block_size;
  std::vector<CandidateConfig> candidates;
};

inline CandidateConfig make_candidate(
    const DeviceLimits& device,
    const KernelLimits& kernel,
    int block_size,
    int64_t dynamic_shared_bytes,
    int active_blocks,
    int active_blocks_without_dynamic_shared,
    bool default_candidate = true) {
  if (device.warp_size <= 0 || device.max_threads_per_sm <= 0 ||
      block_size <= 0 || active_blocks <= 0 ||
      active_blocks_without_dynamic_shared <= 0) {
    throw std::runtime_error("invalid synthetic CUDA occupancy observation");
  }

  const int64_t shared_bytes_per_block =
      kernel.static_shared_bytes + dynamic_shared_bytes;
  const int shared_limit = shared_bytes_per_block == 0
      ? -1
      : static_cast<int>(
            device.shared_bytes_per_sm / shared_bytes_per_block);
  const int thread_limit = device.max_threads_per_sm / block_size;
  const int register_limit = kernel.registers_per_thread == 0
      ? -1
      : static_cast<int>(
            static_cast<int64_t>(device.registers_per_sm) /
            (static_cast<int64_t>(kernel.registers_per_thread) * block_size));

  std::vector<std::string> limiting_resources;
  if (shared_limit >= 0 && active_blocks == shared_limit) {
    limiting_resources.emplace_back("shared");
  }
  if (active_blocks == thread_limit) {
    limiting_resources.emplace_back("threads");
  }
  if (register_limit >= 0 && active_blocks == register_limit) {
    limiting_resources.emplace_back("registers");
  }
  if (limiting_resources.empty()) {
    int smallest_resource_limit = thread_limit;
    if (shared_limit >= 0) {
      smallest_resource_limit = std::min(smallest_resource_limit, shared_limit);
    }
    if (register_limit >= 0) {
      smallest_resource_limit = std::min(smallest_resource_limit, register_limit);
    }
    if (shared_limit == smallest_resource_limit) {
      limiting_resources.emplace_back("shared");
    }
    if (thread_limit == smallest_resource_limit) {
      limiting_resources.emplace_back("threads");
    }
    if (register_limit == smallest_resource_limit) {
      limiting_resources.emplace_back("registers");
    }
  }

  const int warps_per_block =
      (block_size + device.warp_size - 1) / device.warp_size;
  const int maximum_warps =
      device.max_threads_per_sm / device.warp_size;
  const int resident_warps = active_blocks * warps_per_block;
  const int resident_warps_without_dynamic_shared =
      active_blocks_without_dynamic_shared * warps_per_block;
  return {
      block_size,
      default_candidate,
      active_blocks,
      resident_warps,
      100.0 * resident_warps / maximum_warps,
      dynamic_shared_bytes,
      shared_limit,
      thread_limit,
      register_limit,
      std::move(limiting_resources),
      active_blocks_without_dynamic_shared,
      resident_warps_without_dynamic_shared,
      100.0 * resident_warps_without_dynamic_shared / maximum_warps};
}

inline std::size_t select_candidate(
    const std::vector<CandidateConfig>& candidates,
    int occupancy_api_block_size,
    int override_block_size) {
  if (candidates.empty()) {
    throw std::runtime_error("CUDA occupancy API produced no launch candidate");
  }
  if (override_block_size > 0) {
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      if (candidates[index].block_size == override_block_size) {
        return index;
      }
    }
    throw std::runtime_error(
        "LOCOMGCM_CUDA_THREADS_PER_BLOCK does not name a feasible launch candidate");
  }

  std::size_t best = candidates.size();
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    if (!candidates[index].default_candidate) {
      continue;
    }
    if (best == candidates.size()) {
      best = index;
      continue;
    }
    const auto& candidate = candidates[index];
    const auto& incumbent = candidates[best];
    if (candidate.resident_warps_per_sm != incumbent.resident_warps_per_sm) {
      if (candidate.resident_warps_per_sm > incumbent.resident_warps_per_sm) {
        best = index;
      }
      continue;
    }
    const bool candidate_is_api =
        candidate.block_size == occupancy_api_block_size;
    const bool incumbent_is_api =
        incumbent.block_size == occupancy_api_block_size;
    if (candidate_is_api != incumbent_is_api) {
      if (candidate_is_api) {
        best = index;
      }
      continue;
    }
    const int candidate_threads =
        candidate.active_blocks_per_sm * candidate.block_size;
    const int incumbent_threads =
        incumbent.active_blocks_per_sm * incumbent.block_size;
    if (candidate_threads > incumbent_threads ||
        (candidate_threads == incumbent_threads &&
         candidate.block_size > incumbent.block_size)) {
      best = index;
    }
  }
  if (best == candidates.size()) {
    throw std::runtime_error("CUDA occupancy API produced no default candidate");
  }
  return best;
}

#ifdef __CUDACC__

inline void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

inline int diagnostic_block_size_override() {
  const char* text = std::getenv(kThreadsPerBlockOverride);
  if (text == nullptr || *text == '\0') {
    return 0;
  }
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed <= 0 ||
      parsed > INT_MAX) {
    throw std::runtime_error(
        std::string(kThreadsPerBlockOverride) +
        " must be a positive base-10 integer");
  }
  return static_cast<int>(parsed);
}

inline void append_unique(std::vector<int>& values, int value) {
  if (value > 0 && std::find(values.begin(), values.end(), value) == values.end()) {
    values.push_back(value);
  }
}

template <typename Kernel>
LaunchConfig derive_launch_config(
    Kernel kernel,
    int device,
    int64_t shape_key,
    int reference_block_size,
    OccupancyB2DSize dynamic_shared_for_block) {
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
  if (current_device != device) {
    throw std::runtime_error(
        "CUDA launch configuration queried without the requested device guard");
  }

  cudaDeviceProp properties{};
  check_cuda(
      cudaGetDeviceProperties(&properties, device),
      "cudaGetDeviceProperties");
  cudaFuncAttributes attributes{};
  check_cuda(
      cudaFuncGetAttributes(&attributes, kernel),
      "cudaFuncGetAttributes");

  const int64_t shared_bytes_per_block_cap = std::max<int64_t>(
      properties.sharedMemPerBlock,
      properties.sharedMemPerBlockOptin);
  if (static_cast<int64_t>(attributes.sharedSizeBytes) >
      shared_bytes_per_block_cap) {
    throw std::runtime_error(
        "CUDA kernel static shared memory exceeds the device block cap");
  }
  const int64_t maximum_dynamic_shared_bytes =
      shared_bytes_per_block_cap - attributes.sharedSizeBytes;

  int maximum_feasible_block = 0;
  int64_t largest_feasible_dynamic_shared = 0;
  for (int block_size = 1;
       block_size <= attributes.maxThreadsPerBlock;
       ++block_size) {
    const std::size_t requested = dynamic_shared_for_block(block_size);
    if (requested <= static_cast<std::size_t>(maximum_dynamic_shared_bytes)) {
      maximum_feasible_block = block_size;
      largest_feasible_dynamic_shared = std::max<int64_t>(
          largest_feasible_dynamic_shared,
          static_cast<int64_t>(requested));
    }
  }
  if (maximum_feasible_block == 0) {
    throw std::runtime_error(
        "CUDA kernel has no block size within the dynamic shared-memory cap");
  }
  if (largest_feasible_dynamic_shared > attributes.maxDynamicSharedSizeBytes) {
    if (largest_feasible_dynamic_shared > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          "CUDA dynamic shared-memory opt-in exceeds the attribute range");
    }
    check_cuda(
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(largest_feasible_dynamic_shared)),
        "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
    check_cuda(
        cudaFuncGetAttributes(&attributes, kernel),
        "cudaFuncGetAttributes(after shared-memory opt-in)");
  }

  int occupancy_api_minimum_grid_size = 0;
  int occupancy_api_block_size = 0;
  check_cuda(
      cudaOccupancyMaxPotentialBlockSizeVariableSMem(
          &occupancy_api_minimum_grid_size,
          &occupancy_api_block_size,
          kernel,
          dynamic_shared_for_block,
          maximum_feasible_block),
      "cudaOccupancyMaxPotentialBlockSizeVariableSMem");

  const int override_block_size = diagnostic_block_size_override();
  if (override_block_size > attributes.maxThreadsPerBlock) {
    throw std::runtime_error(
        std::string(kThreadsPerBlockOverride) +
        " exceeds cudaFuncGetAttributes.maxThreadsPerBlock");
  }

  std::vector<int> block_sizes;
  for (int block_size = properties.warpSize;
       block_size <= maximum_feasible_block;
       block_size += properties.warpSize) {
    append_unique(block_sizes, block_size);
  }
  append_unique(block_sizes, occupancy_api_block_size);
  append_unique(block_sizes, reference_block_size);
  append_unique(block_sizes, override_block_size);
  if (block_sizes.empty()) {
    append_unique(block_sizes, maximum_feasible_block);
  }
  std::sort(block_sizes.begin(), block_sizes.end());

  const DeviceLimits device_limits{
      properties.multiProcessorCount,
      properties.warpSize,
      properties.maxThreadsPerMultiProcessor,
      static_cast<int64_t>(properties.sharedMemPerMultiprocessor),
      properties.regsPerMultiprocessor};
  const KernelLimits kernel_limits{
      attributes.numRegs,
      static_cast<int64_t>(attributes.sharedSizeBytes),
      attributes.maxThreadsPerBlock};

  std::vector<CandidateConfig> candidates;
  candidates.reserve(block_sizes.size());
  for (const int block_size : block_sizes) {
    if (block_size > maximum_feasible_block) {
      continue;
    }
    const std::size_t dynamic_shared_bytes =
        dynamic_shared_for_block(block_size);
    if (dynamic_shared_bytes >
        static_cast<std::size_t>(maximum_dynamic_shared_bytes)) {
      continue;
    }
    int active_blocks = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            kernel,
            block_size,
            dynamic_shared_bytes),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    int active_blocks_without_dynamic_shared = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_without_dynamic_shared,
            kernel,
            block_size,
            0),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor(no dynamic shared)");
    if (active_blocks > 0 && active_blocks_without_dynamic_shared > 0) {
      const bool default_candidate =
          block_size == occupancy_api_block_size ||
          (block_size >= properties.warpSize &&
           block_size % properties.warpSize == 0);
      candidates.push_back(make_candidate(
          device_limits,
          kernel_limits,
          block_size,
          static_cast<int64_t>(dynamic_shared_bytes),
          active_blocks,
          active_blocks_without_dynamic_shared,
          default_candidate));
    }
  }

  const std::size_t selected = select_candidate(
      candidates,
      occupancy_api_block_size,
      override_block_size);
  return {
      device,
      shape_key,
      properties.name,
      properties.major,
      properties.minor,
      device_limits,
      kernel_limits,
      shared_bytes_per_block_cap,
      maximum_dynamic_shared_bytes,
      occupancy_api_block_size,
      occupancy_api_minimum_grid_size,
      reference_block_size,
      override_block_size,
      candidates[selected].block_size,
      std::move(candidates)};
}

struct CacheKey {
  std::uintptr_t kernel;
  int device;
  int64_t shape_key;

  bool operator<(const CacheKey& other) const {
    return std::tie(kernel, device, shape_key) <
        std::tie(other.kernel, other.device, other.shape_key);
  }

  bool operator==(const CacheKey& other) const {
    return kernel == other.kernel && device == other.device &&
        shape_key == other.shape_key;
  }
};

class LaunchConfigCache {
 public:
  template <typename Kernel>
  const LaunchConfig& get(
      Kernel kernel,
      int device,
      int64_t shape_key,
      int reference_block_size,
      OccupancyB2DSize dynamic_shared_for_block) {
    const CacheKey key{
        reinterpret_cast<std::uintptr_t>(kernel), device, shape_key};
    static thread_local const LaunchConfigCache* last_owner = nullptr;
    static thread_local CacheKey last_key{};
    static thread_local const LaunchConfig* last_config = nullptr;
    if (last_owner == this && last_config != nullptr && last_key == key) {
      return *last_config;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto found = configs_.find(key);
    if (found == configs_.end()) {
      found = configs_.emplace(
          key,
          derive_launch_config(
              kernel,
              device,
              shape_key,
              reference_block_size,
              dynamic_shared_for_block)).first;
    }
    last_owner = this;
    last_key = key;
    last_config = &found->second;
    return *last_config;
  }

 private:
  std::mutex mutex_;
  std::map<CacheKey, LaunchConfig> configs_;
};

#endif  // __CUDACC__

}  // namespace locomgcm::cuda_launch
