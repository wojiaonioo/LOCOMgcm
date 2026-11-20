// One-launch CUDA translation of _vim_coriolis_u_cuda.dispatcher._reference.
//
// Numerical contract: FP_FLOOR against vector_invariant_momentum.py:1006-1028
// on CUDA.  This translation unit is compiled with -fmad=false (set in
// _fast.py).  The named temporaries preserve the eager operation order and must
// not be reassociated or replaced by algebraic equivalents.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/Dispatch.h>

#include <cstdint>
#include <limits>

#include "../../solvers/_cuda_launch/launch_config.h"

namespace {

defgcm::cuda_launch::LaunchConfigCache launch_config_cache;

size_t dynamic_shared_bytes_for_block(int) { return 0; }

struct Strides4 {
  int64_t batch;
  int64_t level;
  int64_t y;
  int64_t x;
};

Strides4 broadcast_strides4(const torch::Tensor &value) {
  switch (value.dim()) {
  case 0:
    return {0, 0, 0, 0};
  case 1:
    return {0, 0, 0, value.size(0) == 1 ? 0 : value.stride(0)};
  case 2:
    return {0, 0, value.size(0) == 1 ? 0 : value.stride(0),
            value.size(1) == 1 ? 0 : value.stride(1)};
  case 3:
    return {0, value.size(0) == 1 ? 0 : value.stride(0),
            value.size(1) == 1 ? 0 : value.stride(1),
            value.size(2) == 1 ? 0 : value.stride(2)};
  default:
    return {value.size(0) == 1 ? 0 : value.stride(0),
            value.size(1) == 1 ? 0 : value.stride(1),
            value.size(2) == 1 ? 0 : value.stride(2),
            value.size(3) == 1 ? 0 : value.stride(3)};
  }
}

template <typename scalar_t> struct Input4 {
  const scalar_t *data;
  Strides4 strides;

  __device__ __forceinline__ scalar_t load(const int64_t batch,
                                           const int64_t level, const int64_t y,
                                           const int64_t x) const {
    const int64_t offset = batch * strides.batch + level * strides.level +
                           y * strides.y + x * strides.x;
    return __ldg(data + offset);
  }
};

template <typename scalar_t>
Input4<scalar_t> input4(const torch::Tensor &value) {
  return {value.data_ptr<scalar_t>(), broadcast_strides4(value)};
}

template <typename scalar_t>
Input4<scalar_t> optional_input4(const torch::Tensor &value) {
  if (!value.defined()) {
    return {nullptr, {0, 0, 0, 0}};
  }
  return input4<scalar_t>(value);
}

struct OptionalBoolInput1 {
  const bool *data;
  int64_t x_stride;

  __device__ __forceinline__ bool load(const int64_t x) const {
    return data[x * x_stride];
  }
};

OptionalBoolInput1 optional_bool_input1(const torch::Tensor &value) {
  if (!value.defined()) {
    return {nullptr, 0};
  }
  return {value.data_ptr<bool>(), value.stride(0)};
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
clamp_min_preserve_nan(const scalar_t value, const scalar_t lower) {
  return value < lower ? lower : value;
}

template <typename scalar_t>
__global__ void vim_coriolis_u_kernel(
    const Input4<scalar_t> v_ext, const Input4<scalar_t> corner_value,
    const Input4<scalar_t> dxg_south, const Input4<scalar_t> dxg_north,
    const Input4<scalar_t> recip_dxc_u, const Input4<scalar_t> hfacz,
    const Input4<scalar_t> face_mask, const OptionalBoolInput1 edge_pair_mask,
    scalar_t *__restrict__ output, const int64_t element_count,
    const int64_t levels, const int64_t ny, const int64_t nx_with_faces,
    const bool has_hfacz, const bool has_face_mask,
    const bool has_edge_pair_mask, const bool edge_face_couples) {
  const int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= element_count) {
    return;
  }

  const int64_t plane = ny * nx_with_faces;
  const int64_t volume = levels * plane;
  const int64_t batch = output_index / volume;
  const int64_t volume_index = output_index - batch * volume;
  const int64_t level = volume_index / plane;
  const int64_t plane_index = volume_index - level * plane;
  const int64_t y = plane_index / nx_with_faces;
  const int64_t x = plane_index - y * nx_with_faces;

  // Four offset windows from the haloed V field.
  const scalar_t v_south_right = v_ext.load(batch, level, y, x + 1);
  const scalar_t v_south_left = v_ext.load(batch, level, y, x);
  const scalar_t v_north_right = v_ext.load(batch, level, y + 1, x + 1);
  const scalar_t v_north_left = v_ext.load(batch, level, y + 1, x);

  const scalar_t south_pair = v_south_right + v_south_left;
  const scalar_t north_pair = v_north_right + v_north_left;
  const scalar_t dxg_south_value = dxg_south.load(batch, level, y, x);
  const scalar_t dxg_north_value = dxg_north.load(batch, level, y, x);

  scalar_t v_bar;
  if (!has_hfacz) {
    // Planetary quarter-sum branch.
    const scalar_t south_scaled = dxg_south_value * south_pair;
    const scalar_t north_scaled = dxg_north_value * north_pair;
    const scalar_t scaled_sum = south_scaled + north_scaled;
    v_bar = static_cast<scalar_t>(0.25) * scaled_sum;
  } else {
    // hFacZ-weighted vorticity branch.
    const scalar_t hfacz_south = hfacz.load(batch, level, y, x);
    const scalar_t hfacz_north = hfacz.load(batch, level, y + 1, x);
    const scalar_t south_hfac_metric = hfacz_south * dxg_south_value;
    const scalar_t south_weighted = south_hfac_metric * south_pair;
    const scalar_t north_hfac_metric = hfacz_north * dxg_north_value;
    const scalar_t north_weighted = north_hfac_metric * north_pair;
    const scalar_t weighted_sum = south_weighted + north_weighted;
    const scalar_t weighted_half = static_cast<scalar_t>(0.5) * weighted_sum;
    const scalar_t hfac_sum = hfacz_south + hfacz_north;
    const scalar_t denominator =
        clamp_min_preserve_nan(hfac_sum, static_cast<scalar_t>(1.0e-9));
    v_bar = weighted_half / denominator;
  }

  // Corner average and row-metric scaling.
  const scalar_t corner_south = corner_value.load(batch, level, y, x);
  const scalar_t corner_north = corner_value.load(batch, level, y + 1, x);
  const scalar_t corner_sum = corner_south + corner_north;
  const scalar_t corner_avg = static_cast<scalar_t>(0.5) * corner_sum;
  const scalar_t corner_velocity = corner_avg * v_bar;
  scalar_t result = corner_velocity * recip_dxc_u.load(batch, level, y, x);

  // An explicit edge-pair mask takes precedence over the
  // topology flag, matching the eager if/elif branch.
  if (has_edge_pair_mask) {
    result = edge_pair_mask.load(x) ? result : static_cast<scalar_t>(0.0);
  } else if (!edge_face_couples && (x == 0 || x + 1 == nx_with_faces)) {
    result = static_cast<scalar_t>(0.0);
  }

  // binary=False means a plain multiply after edge masking.
  if (has_face_mask) {
    result = result * face_mask.load(batch, level, y, x);
  }
  output[output_index] = result;
}

} // namespace

void vim_coriolis_u_launch(
    const torch::Tensor &v_ext, const torch::Tensor &corner_value,
    const torch::Tensor &dxg_south, const torch::Tensor &dxg_north,
    const torch::Tensor &recip_dxc_u, const torch::Tensor &hfacz,
    const torch::Tensor &face_mask, const torch::Tensor &edge_pair_mask,
    torch::Tensor &u_face_coriolis, const bool edge_face_couples) {
  c10::cuda::CUDAGuard device_guard(v_ext.device());
  const int64_t element_count = u_face_coriolis.numel();
  const int64_t output_trailing_start = u_face_coriolis.dim() - 3;
  AT_DISPATCH_FLOATING_TYPES(v_ext.scalar_type(), "vim_coriolis_u_launch", [&] {
    const auto &launch_config = launch_config_cache.get(
        vim_coriolis_u_kernel<scalar_t>, v_ext.get_device(), 0, 256,
        dynamic_shared_bytes_for_block);
    const int threads_per_block = launch_config.selected_block_size;
    const int64_t block_count =
        (element_count + threads_per_block - 1) / threads_per_block;
    TORCH_CHECK(block_count <= std::numeric_limits<unsigned int>::max(),
                "VIM Coriolis U CUDA grid exceeds grid.x capacity");

    const dim3 block(static_cast<unsigned int>(threads_per_block));
    const dim3 grid(static_cast<unsigned int>(block_count));
    vim_coriolis_u_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input4<scalar_t>(v_ext), input4<scalar_t>(corner_value),
            input4<scalar_t>(dxg_south), input4<scalar_t>(dxg_north),
            input4<scalar_t>(recip_dxc_u), optional_input4<scalar_t>(hfacz),
            optional_input4<scalar_t>(face_mask),
            optional_bool_input1(edge_pair_mask),
            u_face_coriolis.data_ptr<scalar_t>(), element_count,
            u_face_coriolis.size(output_trailing_start),
            u_face_coriolis.size(output_trailing_start + 1),
            u_face_coriolis.size(output_trailing_start + 2), hfacz.defined(),
            face_mask.defined(), edge_pair_mask.defined(), edge_face_couples);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

const defgcm::cuda_launch::LaunchConfig &
vim_coriolis_u_launch_config(int device) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  return launch_config_cache.get(vim_coriolis_u_kernel<double>, device, 0, 256,
                                 dynamic_shared_bytes_for_block);
}
