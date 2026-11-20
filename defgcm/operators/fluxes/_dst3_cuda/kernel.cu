// One-launch CUDA translation of _dst3_transport_flux_values.
//
// Numerical contract: BITWISE against tracer_flux.py:106-185 on CUDA.  This
// translation unit must be compiled with -fmad=false (set in _fast.py) so each
// eager multiply and following add/subtract remains two rounded operations.
// The named temporaries below intentionally preserve every eager rounding
// boundary and must not be reassociated or replaced by algebraic equivalents.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/Dispatch.h>

#include <cstdint>
#include <limits>

#include "../../../solvers/_cuda_launch/launch_config.h"

namespace {

defgcm::cuda_launch::LaunchConfigCache launch_config_cache;

size_t dynamic_shared_bytes_for_block(int) { return 0; }

constexpr double kThetaMax = 1.0e20;
constexpr double kCflFloor = 1.0e-20;
constexpr double kOneSixth = 1.0 / 6.0;

struct Strides4 {
  int64_t batch;
  int64_t level;
  int64_t y;
  int64_t x;
};

Strides4 strides4(const torch::Tensor &value) {
  if (value.dim() == 3) {
    return {0, value.stride(0), value.stride(1), value.stride(2)};
  }
  const int64_t batch_stride = value.size(0) == 1 ? 0 : value.stride(0);
  return {batch_stride, value.stride(1), value.stride(2), value.stride(3)};
}

template <typename scalar_t>
struct Input4 {
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
  return {value.data_ptr<scalar_t>(), strides4(value)};
}

// torch.minimum propagates NaN and returns its first operand on an ordered tie.
// CUDA fmin does not propagate a single NaN, so it is not valid here.
template <typename scalar_t>
__device__ __forceinline__ scalar_t minimum_propagate_nan(const scalar_t first,
                                                        const scalar_t second) {
  if (isnan(first)) {
    return first;
  }
  if (isnan(second)) {
    return second;
  }
  return first <= second ? first : second;
}

// torch.clamp(value, min=0) preserves NaN and -0.0.  A strict comparison plus
// selection has both properties; CUDA fmax has neither required selection
// behavior in the exceptional cases.
template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp_min_zero(const scalar_t value) {
  return value < static_cast<scalar_t>(0.0) ? static_cast<scalar_t>(0.0) : value;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t ratio_safe(const scalar_t num,
                                             const scalar_t denom) {
  // tracer_flux.py:110-120.  Keep each materialized eager operation explicit.
  const scalar_t abs_num = fabs(num);
  const scalar_t abs_denom = fabs(denom);
  const scalar_t sign_product = num * denom;
  const scalar_t sign = sign_product >= static_cast<scalar_t>(0.0) ? static_cast<scalar_t>(1.0) : -static_cast<scalar_t>(1.0);
  const scalar_t capped = sign * static_cast<scalar_t>(kThetaMax);
  const scalar_t cap_lhs = abs_denom * static_cast<scalar_t>(kThetaMax);
  const bool use_capped = cap_lhs <= abs_num;
  const scalar_t safe_denom = use_capped ? static_cast<scalar_t>(1.0) : denom;
  const scalar_t raw = num / safe_denom;
  return use_capped ? capped : raw;
}

// _dst3_coefficients: d0=(2-CFL)(1-CFL)/6, d1=(1-CFL^2)/6.  Shared by the
// scheme-33 psi below and the scheme-30 direct form in the kernel body, exactly
// as tracer_flux.py routes both through one helper.
template <typename scalar_t>
__device__ __forceinline__ void
dst3_coefficients(const scalar_t u_cfl, scalar_t &d0, scalar_t &d1) {
  const scalar_t d0_left = static_cast<scalar_t>(2.0) - u_cfl;
  const scalar_t d0_right = static_cast<scalar_t>(1.0) - u_cfl;
  const scalar_t d0_product = d0_left * d0_right;
  d0 = d0_product * static_cast<scalar_t>(kOneSixth);

  const scalar_t cfl_square = u_cfl * u_cfl;
  const scalar_t d1_difference = static_cast<scalar_t>(1.0) - cfl_square;
  d1 = d1_difference * static_cast<scalar_t>(kOneSixth);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
dst3_psi(const scalar_t theta, const scalar_t u_cfl, const bool with_limiter) {
  scalar_t d0;
  scalar_t d1;
  dst3_coefficients(u_cfl, d0, d1);
  const scalar_t psi_product = d1 * theta;
  scalar_t psi = d0 + psi_product;
  if (!with_limiter) {
    return psi;
  }

  // Preserve the two sequential minima and clamp.
  const scalar_t bound_difference = static_cast<scalar_t>(1.0) - u_cfl;
  const scalar_t bound_numerator = theta * bound_difference;
  const scalar_t bound_denominator = u_cfl + static_cast<scalar_t>(kCflFloor);
  const scalar_t sweby_bound = bound_numerator / bound_denominator;
  psi = minimum_propagate_nan(static_cast<scalar_t>(1.0), psi);
  psi = minimum_propagate_nan(psi, sweby_bound);
  psi = clamp_min_zero(psi);
  return psi;
}

template <typename scalar_t>
__global__ void dst3_transport_flux_kernel(
    const Input4<scalar_t> t_left, const Input4<scalar_t> t_right, const Input4<scalar_t> tm2,
    const Input4<scalar_t> tm1, const Input4<scalar_t> t0, const Input4<scalar_t> tp1, const Input4<scalar_t> mask_im1,
    const Input4<scalar_t> mask_i, const Input4<scalar_t> mask_ip1, const Input4<scalar_t> u_trans,
    const Input4<scalar_t> u_cfl, scalar_t *__restrict__ upwind_flux,
    scalar_t *__restrict__ flux_dst, const int64_t element_count,
    const int64_t levels, const int64_t ny, const int64_t nx,
    const bool with_limiter) {
  const int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= element_count) {
    return;
  }

  const int64_t plane = ny * nx;
  const int64_t volume = levels * plane;
  const int64_t batch = output_index / volume;
  const int64_t volume_index = output_index - batch * volume;
  const int64_t level = volume_index / plane;
  const int64_t plane_index = volume_index - level * plane;
  const int64_t y = plane_index / nx;
  const int64_t x = plane_index - y * nx;

  const scalar_t t_left_value = t_left.load(batch, level, y, x);
  const scalar_t t_right_value = t_right.load(batch, level, y, x);
  const scalar_t tm2_value = tm2.load(batch, level, y, x);
  const scalar_t tm1_value = tm1.load(batch, level, y, x);
  const scalar_t t0_value = t0.load(batch, level, y, x);
  const scalar_t tp1_value = tp1.load(batch, level, y, x);
  const scalar_t mask_im1_value = mask_im1.load(batch, level, y, x);
  const scalar_t mask_i_value = mask_i.load(batch, level, y, x);
  const scalar_t mask_ip1_value = mask_ip1.load(batch, level, y, x);
  const scalar_t u_trans_value = u_trans.load(batch, level, y, x);
  const scalar_t u_cfl_value = u_cfl.load(batch, level, y, x);

  // tracer_flux.py:171-173.
  const scalar_t u_trans_abs_positive = fabs(u_trans_value);
  const scalar_t pos_sum = u_trans_value + u_trans_abs_positive;
  const scalar_t pos = static_cast<scalar_t>(0.5) * pos_sum;
  const scalar_t u_trans_abs_negative = fabs(u_trans_value);
  const scalar_t neg_difference = u_trans_value - u_trans_abs_negative;
  const scalar_t neg = static_cast<scalar_t>(0.5) * neg_difference;
  const scalar_t upwind_left = pos * t_left_value;
  const scalar_t upwind_right = neg * t_right_value;
  const scalar_t upwind_value = upwind_left + upwind_right;

  // tracer_flux.py:175-177.
  const scalar_t rjp_difference = tp1_value - t0_value;
  const scalar_t rjp = rjp_difference * mask_ip1_value;
  const scalar_t rj_difference = t0_value - tm1_value;
  const scalar_t rj = rj_difference * mask_i_value;
  const scalar_t rjm_difference = tm1_value - tm2_value;
  const scalar_t rjm = rjm_difference * mask_im1_value;

  // Scheme 33 (with_limiter) takes the ratio form through the Sweby limiter;
  // scheme 30 (!with_limiter) takes MITgcm's DIRECT form and never forms the
  // ratio (gad_dst3_adv_x.F:109-115).  Both branches keep every eager rounding
  // boundary distinct.
  scalar_t dst_value;
  if (with_limiter) {
    const scalar_t theta_p = ratio_safe(rjm, rj);
    const scalar_t theta_m = ratio_safe(rjp, rj);
    const scalar_t psi_p = dst3_psi(theta_p, u_cfl_value, true);
    const scalar_t psi_m = dst3_psi(theta_m, u_cfl_value, true);

    const scalar_t positive_reconstruction_product = psi_p * rj;
    const scalar_t positive_reconstruction =
        tm1_value + positive_reconstruction_product;
    const scalar_t positive_flux = pos * positive_reconstruction;
    const scalar_t negative_reconstruction_product = psi_m * rj;
    const scalar_t negative_reconstruction =
        t0_value - negative_reconstruction_product;
    const scalar_t negative_flux = neg * negative_reconstruction;
    dst_value = positive_flux + negative_flux;
  } else {
    scalar_t d0;
    scalar_t d1;
    dst3_coefficients(u_cfl_value, d0, d1);
    const scalar_t positive_d0_term = d0 * rj;
    const scalar_t positive_d1_term = d1 * rjm;
    const scalar_t positive_correction = positive_d0_term + positive_d1_term;
    const scalar_t positive_reconstruction = tm1_value + positive_correction;
    const scalar_t positive_flux = pos * positive_reconstruction;
    const scalar_t negative_d0_term = d0 * rj;
    const scalar_t negative_d1_term = d1 * rjp;
    const scalar_t negative_correction = negative_d0_term + negative_d1_term;
    const scalar_t negative_reconstruction = t0_value - negative_correction;
    const scalar_t negative_flux = neg * negative_reconstruction;
    dst_value = positive_flux + negative_flux;
  }

  upwind_flux[output_index] = upwind_value;
  flux_dst[output_index] = dst_value;
}

} // namespace

void dst3_transport_flux_launch(
    const torch::Tensor &t_left, const torch::Tensor &t_right,
    const torch::Tensor &tm2, const torch::Tensor &tm1, const torch::Tensor &t0,
    const torch::Tensor &tp1, const torch::Tensor &mask_im1,
    const torch::Tensor &mask_i, const torch::Tensor &mask_ip1,
    const torch::Tensor &u_trans, const torch::Tensor &u_cfl,
    torch::Tensor &upwind_flux, torch::Tensor &flux_dst,
    const bool with_limiter) {
  c10::cuda::CUDAGuard device_guard(t_left.device());
  const int64_t element_count = upwind_flux.numel();
  const int64_t output_trailing_start = upwind_flux.dim() - 3;
  // One instantiation per input dtype.  float64 is the bitwise production
  // path; float32 exists only to measure the cost of the narrower type.
  AT_DISPATCH_FLOATING_TYPES(
      t_left.scalar_type(), "dst3_transport_flux_launch", [&] {
        const auto &launch_config = launch_config_cache.get(
            dst3_transport_flux_kernel<scalar_t>, t_left.get_device(), 0, 256,
            dynamic_shared_bytes_for_block);
        const int threads_per_block = launch_config.selected_block_size;
        const int64_t block_count =
            (element_count + threads_per_block - 1) / threads_per_block;
        TORCH_CHECK(block_count <= std::numeric_limits<unsigned int>::max(),
                    "DST3 CUDA grid exceeds grid.x capacity");

        const dim3 block(threads_per_block);
        const dim3 grid(static_cast<unsigned int>(block_count));
        dst3_transport_flux_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input4<scalar_t>(t_left), input4<scalar_t>(t_right),
                input4<scalar_t>(tm2), input4<scalar_t>(tm1),
                input4<scalar_t>(t0), input4<scalar_t>(tp1),
                input4<scalar_t>(mask_im1), input4<scalar_t>(mask_i),
                input4<scalar_t>(mask_ip1), input4<scalar_t>(u_trans),
                input4<scalar_t>(u_cfl), upwind_flux.data_ptr<scalar_t>(),
                flux_dst.data_ptr<scalar_t>(), element_count,
                upwind_flux.size(output_trailing_start),
                upwind_flux.size(output_trailing_start + 1),
                upwind_flux.size(output_trailing_start + 2), with_limiter);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

const defgcm::cuda_launch::LaunchConfig &
dst3_cuda_launch_config(int device, at::ScalarType dtype) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  // Mirrors the AT_DISPATCH_FLOATING_TYPES in the launcher above: the two
  // instantiations carry different register footprints, so the exported
  // configuration must name the one this dtype actually launches.
  if (dtype == at::kFloat) {
    return launch_config_cache.get(dst3_transport_flux_kernel<float>, device,
                                   0, 256, dynamic_shared_bytes_for_block);
  }
  TORCH_CHECK(
      dtype == at::kDouble,
      "DST3 CUDA launch configuration supports float64 or float32 only");
  return launch_config_cache.get(dst3_transport_flux_kernel<double>, device,
                                 0, 256, dynamic_shared_bytes_for_block);
}
