#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "launch_config.h"

namespace defgcm::cuda_launch {

inline pybind11::dict to_python(const LaunchConfig& config) {
  pybind11::dict result;
  result["device"] = config.device;
  result["shape_key"] = config.shape_key;
  result["device_name"] = config.device_name;
  result["compute_capability_major"] = config.compute_capability_major;
  result["compute_capability_minor"] = config.compute_capability_minor;
  result["multiprocessor_count"] =
      config.device_limits.multiprocessor_count;
  result["warp_size"] = config.device_limits.warp_size;
  result["max_threads_per_sm"] = config.device_limits.max_threads_per_sm;
  result["shared_bytes_per_sm"] =
      config.device_limits.shared_bytes_per_sm;
  result["registers_per_sm"] = config.device_limits.registers_per_sm;
  result["registers_per_thread"] =
      config.kernel_limits.registers_per_thread;
  result["static_shared_bytes"] = config.kernel_limits.static_shared_bytes;
  result["max_threads_per_block"] =
      config.kernel_limits.max_threads_per_block;
  result["shared_bytes_per_block_cap"] =
      config.shared_bytes_per_block_cap;
  result["maximum_dynamic_shared_bytes"] =
      config.maximum_dynamic_shared_bytes;
  result["occupancy_api_block_size"] = config.occupancy_api_block_size;
  result["occupancy_api_minimum_grid_size"] =
      config.occupancy_api_minimum_grid_size;
  result["reference_block_size"] = config.reference_block_size;
  result["override_block_size"] = config.override_block_size;
  result["selected_block_size"] = config.selected_block_size;

  pybind11::list candidates;
  for (const auto& candidate : config.candidates) {
    pybind11::dict item;
    item["block_size"] = candidate.block_size;
    item["default_candidate"] = candidate.default_candidate;
    item["active_blocks_per_sm"] = candidate.active_blocks_per_sm;
    item["resident_warps_per_sm"] = candidate.resident_warps_per_sm;
    item["occupancy_percent"] = candidate.occupancy_percent;
    item["dynamic_shared_bytes"] = candidate.dynamic_shared_bytes;
    item["shared_block_limit"] = candidate.shared_block_limit;
    item["thread_block_limit"] = candidate.thread_block_limit;
    item["register_block_limit"] = candidate.register_block_limit;
    item["limiting_resources"] = candidate.limiting_resources;
    item["active_blocks_without_dynamic_shared"] =
        candidate.active_blocks_without_dynamic_shared;
    item["resident_warps_without_dynamic_shared"] =
        candidate.resident_warps_without_dynamic_shared;
    item["occupancy_without_dynamic_shared_percent"] =
        candidate.occupancy_without_dynamic_shared_percent;
    candidates.append(item);
  }
  result["candidates"] = candidates;
  return result;
}

}  // namespace defgcm::cuda_launch
