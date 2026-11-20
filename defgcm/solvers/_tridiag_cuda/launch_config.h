#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

enum class CPrimeStorage : int {
  kShared = 0,
  kGlobal = 1,
  kRecomputeProbe = 2,
};

struct TridiagCandidateConfig {
  CPrimeStorage storage;
  bool selectable;
  int occupancy_block_size;
  int block_size;
  int minimum_grid_size;
  int registers_per_thread;
  int64_t static_shared_bytes;
  int64_t dynamic_shared_bytes;
  int max_threads_per_block;
  int shared_block_limit;
  int thread_block_limit;
  int register_block_limit;
  int active_blocks_per_sm;
  int resident_warps_per_sm;
  double occupancy_percent;
  std::vector<std::string> limiting_resources;
};

struct TridiagLaunchConfig {
  int device;
  int nz;
  std::string device_name;
  int compute_capability_major;
  int compute_capability_minor;
  int multiprocessor_count;
  int warp_size;
  int max_threads_per_sm;
  int64_t shared_bytes_per_sm;
  int registers_per_sm;
  int64_t optin_shared_bytes_per_block;
  CPrimeStorage selected_storage;
  int selected_block_size;
  std::string selection_reason;
  std::string storage_override;
  int block_size_override;
  std::vector<TridiagCandidateConfig> candidates;
};

// ``dtype`` selects the kernel instantiation whose occupancy is measured; the
// float64 and float32 templates carry different register and shared-memory
// footprints, so a run must read the configuration for the dtype it launches.
TridiagLaunchConfig tridiag_cuda_derive_launch_config(
    int device,
    int nz,
    const std::string& storage_override,
    int block_size_override,
    at::ScalarType dtype);

void tridiag_cuda_launch(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& d,
    torch::Tensor& output,
    CPrimeStorage storage,
    int block_size,
    bool mitgcm_order);
