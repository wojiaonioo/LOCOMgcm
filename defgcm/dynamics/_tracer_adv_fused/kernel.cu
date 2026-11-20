// One-launch CUDA translation of one directional pass of
// HorizontalTracerAdvection._multidim_advective_tendency (DST3 / DST3-FL).
//
// Numerical contract: BITWISE against the eager chain
//   tracer_flux.py:199-272   (_dst3_transport_flux_values)
//   tracer_flux.py:417,437-441 (X adapter CFL + splice) / 527,540-544 (Y)
//   c_grid.py:252-259        (zonal_/meridional_face_difference)
//   tracers.py:107-120       (_multidim_advective_pass_update)
//   tracers.py:774 / 843     (localT -= dt * update)
//   tracers.py:855           (horizontal-only tendency)
//
// One asymmetry is deliberate and is NOT a transcription slip: the final
// ``/ dt`` is spelled here as a multiply by a host-computed reciprocal.  ATen
// lowers a tensor-divided-by-a-CPU-scalar to exactly that (the scalar is
// inverted once on the host and broadcast as a multiply), so writing a true
// device division here differs from eager by up to one ULP on ~0.5% of cells
// on CUDA.  ``dt * update`` above stays a multiply on both sides and is
// unaffected.  This translation unit must be compiled with -fmad=false (set in
// _fast.py) so each eager multiply and following add/subtract remains two
// rounded operations.  The named temporaries below intentionally preserve
// every eager rounding boundary and must not be reassociated or replaced by
// algebraic equivalents.
//
// ratio_safe / dst3_psi / minimum_propagate_nan / clamp_min_zero originate from
// the already bitwise-verified
// defgcm/operators/fluxes/_dst3_cuda/kernel.cu (that package is a separate
// extension and its helpers live in an anonymous namespace, so a shared header
// would mean editing it).  Any correction there must be mirrored here.
// ratio_safe additionally bypasses only zero / finite-nonzero division below.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/Dispatch.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "../../solvers/_cuda_launch/launch_config.h"

namespace {

defgcm::cuda_launch::LaunchConfigCache launch_config_cache;

template <typename scalar_t>
size_t dynamic_shared_bytes_for_block(int block_size) {
  return static_cast<size_t>(block_size) * sizeof(scalar_t);
}

constexpr double kThetaMax = 1.0e20;
constexpr double kCflFloor = 1.0e-20;
constexpr double kOneSixth = 1.0 / 6.0;

struct Strides4 {
  int64_t batch;
  int64_t level;
  int64_t y;
  int64_t x;
};

// Broadcast a 4-D input against the (batch, level, y, x) iteration space: an
// extent-1 dimension contributes stride 0.  Every input is normalised to four
// dimensions on the Python side, so no rank branch is needed here.
Strides4 broadcast_strides(const torch::Tensor &value) {
  return {value.size(0) == 1 ? 0 : value.stride(0),
          value.size(1) == 1 ? 0 : value.stride(1),
          value.size(2) == 1 ? 0 : value.stride(2),
          value.size(3) == 1 ? 0 : value.stride(3)};
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
  return {value.data_ptr<scalar_t>(), broadcast_strides(value)};
}

// torch.minimum propagates NaN and returns its first operand on an ordered tie.
// CUDA fmin does not propagate a single NaN, so it is not valid here.
template <typename scalar_t>
__device__ __forceinline__ scalar_t
minimum_propagate_nan(const scalar_t first, const scalar_t second) {
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
  // tracer_flux.py:165-176.  Keep each materialized eager operation explicit.
  const scalar_t abs_num = fabs(num);
  const scalar_t abs_denom = fabs(denom);
  const scalar_t sign_product = num * denom;
  const scalar_t sign = sign_product >= static_cast<scalar_t>(0.0)
                            ? static_cast<scalar_t>(1.0)
                            : -static_cast<scalar_t>(1.0);
  const scalar_t capped = sign * static_cast<scalar_t>(kThetaMax);
  const scalar_t safe_denom =
      denom == static_cast<scalar_t>(0.0) ? static_cast<scalar_t>(1.0) : denom;
  scalar_t raw;
  if (num == static_cast<scalar_t>(0.0) && isfinite(safe_denom) &&
      safe_denom != static_cast<scalar_t>(0.0)) {
    // IEEE zero / finite-nonzero: preserve the XOR of the operand sign bits.
    // Only skip the raw division; the cap selection below still handles 0/0.
    raw = signbit(num) != signbit(safe_denom) ? -static_cast<scalar_t>(0.0)
                                           : static_cast<scalar_t>(0.0);
  } else {
    raw = num / safe_denom;
  }
  const scalar_t cap_lhs = abs_denom * static_cast<scalar_t>(kThetaMax);
  const bool use_capped = cap_lhs <= abs_num;
  return use_capped ? capped : raw;
}

// _dst3_coefficients: d0=(2-CFL)(1-CFL)/6, d1=(1-CFL^2)/6.  Shared by the
// scheme-33 psi below and the scheme-30 direct form in face_flux, mirroring the
// one-helper routing in tracer_flux.py.
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

// One face transport flux, reproducing the eager splice: faces outside
// [dst_start, dst_stop) keep the first-order upwind value
// (tracer_flux.py:437-441 / 540-544), faces inside take the DST3(-FL) value.
//
// AXIS==0 indexes the face along x (U faces), AXIS==1 along y (V faces).  The
// wide stencil offsets are the eager slice origins:
//   Tm2 = wide[f+0]  Tm1 = wide[f+1]  T0 = wide[f+2]  Tp1 = wide[f+3]
//   mask_im1 = mask_wide[f+0]  mask_i = mask_wide[f+1]  mask_ip1 = mask_wide[f+2]
template <typename scalar_t, int AXIS>
__device__ __forceinline__ scalar_t face_flux(
    const Input4<scalar_t> &t_wide, const Input4<scalar_t> &t_left,
    const Input4<scalar_t> &t_right, const Input4<scalar_t> &mask_wide,
    const Input4<scalar_t> &vel_face, const Input4<scalar_t> &vel_trans,
    const Input4<scalar_t> &recip_dc, const int64_t batch, const int64_t level,
    const int64_t y, const int64_t x, const int64_t face, const scalar_t dt,
    const int64_t dst_start, const int64_t dst_stop, const bool with_limiter) {
  const int64_t face_y = AXIS == 0 ? y : face;
  const int64_t face_x = AXIS == 0 ? face : x;

  const scalar_t trans_value = vel_trans.load(batch, level, face_y, face_x);

  // tracer_flux.py:259-261.
  const scalar_t trans_abs_positive = fabs(trans_value);
  const scalar_t pos_sum = trans_value + trans_abs_positive;
  const scalar_t pos = static_cast<scalar_t>(0.5) * pos_sum;
  const scalar_t trans_abs_negative = fabs(trans_value);
  const scalar_t neg_difference = trans_value - trans_abs_negative;
  const scalar_t neg = static_cast<scalar_t>(0.5) * neg_difference;

  if (face < dst_start || face >= dst_stop) {
    const scalar_t t_left_value = t_left.load(batch, level, face_y, face_x);
    const scalar_t t_right_value = t_right.load(batch, level, face_y, face_x);
    const scalar_t upwind_left = pos * t_left_value;
    const scalar_t upwind_right = neg * t_right_value;
    return upwind_left + upwind_right;
  }

  const int64_t wide_y0 = AXIS == 0 ? y : face;
  const int64_t wide_x0 = AXIS == 0 ? face : x;
  const int64_t step_y = AXIS == 0 ? 0 : 1;
  const int64_t step_x = AXIS == 0 ? 1 : 0;

  const scalar_t tm2_value = t_wide.load(batch, level, wide_y0, wide_x0);
  const scalar_t tm1_value =
      t_wide.load(batch, level, wide_y0 + step_y, wide_x0 + step_x);
  const scalar_t t0_value =
      t_wide.load(batch, level, wide_y0 + 2 * step_y, wide_x0 + 2 * step_x);
  const scalar_t tp1_value =
      t_wide.load(batch, level, wide_y0 + 3 * step_y, wide_x0 + 3 * step_x);
  const scalar_t mask_im1_value =
      mask_wide.load(batch, level, wide_y0, wide_x0);
  const scalar_t mask_i_value =
      mask_wide.load(batch, level, wide_y0 + step_y, wide_x0 + step_x);
  const scalar_t mask_ip1_value =
      mask_wide.load(batch, level, wide_y0 + 2 * step_y, wide_x0 + 2 * step_x);

  // tracer_flux.py:417 / 527: u_cfl = (u_face * dt * recip_dxC).abs()
  const scalar_t vel_value = vel_face.load(batch, level, face_y, face_x);
  const scalar_t recip_value = recip_dc.load(batch, level, face_y, face_x);
  const scalar_t cfl_scaled = vel_value * dt;
  const scalar_t cfl_signed = cfl_scaled * recip_value;
  const scalar_t cfl = fabs(cfl_signed);

  // tracer_flux.py:263-265.
  const scalar_t rjp_difference = tp1_value - t0_value;
  const scalar_t rjp = rjp_difference * mask_ip1_value;
  const scalar_t rj_difference = t0_value - tm1_value;
  const scalar_t rj = rj_difference * mask_i_value;
  const scalar_t rjm_difference = tm1_value - tm2_value;
  const scalar_t rjm = rjm_difference * mask_im1_value;

  // Scheme 33 (with_limiter) takes the ratio form through the Sweby limiter;
  // scheme 30 (!with_limiter) takes MITgcm's DIRECT form (gad_dst3_adv_x.F:
  // 109-115) and never forms the ratio.  Mirrors _dst3_transport_flux_values.
  if (with_limiter) {
    const scalar_t theta_p = ratio_safe(rjm, rj);
    const scalar_t theta_m = ratio_safe(rjp, rj);
    const scalar_t psi_p = dst3_psi(theta_p, cfl, true);
    const scalar_t psi_m = dst3_psi(theta_m, cfl, true);

    const scalar_t positive_reconstruction_product = psi_p * rj;
    const scalar_t positive_reconstruction =
        tm1_value + positive_reconstruction_product;
    const scalar_t positive_flux = pos * positive_reconstruction;
    const scalar_t negative_reconstruction_product = psi_m * rj;
    const scalar_t negative_reconstruction =
        t0_value - negative_reconstruction_product;
    const scalar_t negative_flux = neg * negative_reconstruction;
    return positive_flux + negative_flux;
  }

  scalar_t d0;
  scalar_t d1;
  dst3_coefficients(cfl, d0, d1);
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
  return positive_flux + negative_flux;
}

template <typename scalar_t, int AXIS>
__global__ void tracer_adv_pass_kernel(
    const Input4<scalar_t> t_wide, const Input4<scalar_t> t_left,
    const Input4<scalar_t> t_right, const Input4<scalar_t> mask_wide,
    const Input4<scalar_t> vel_face, const Input4<scalar_t> vel_trans,
    const Input4<scalar_t> recip_dc, const Input4<scalar_t> recip_vol,
    const Input4<scalar_t> mask_in_c, const Input4<scalar_t> tracer_orig,
    const Input4<scalar_t> local_t, scalar_t *__restrict__ out,
    const int64_t tiles_x, const int64_t tiles_y, const int tile_width,
    const int tile_height, const int64_t levels, const int64_t ny,
    const int64_t nx, const scalar_t dt, const scalar_t inv_dt,
    const int64_t dst_start, const int64_t dst_stop, const bool with_limiter,
    const bool has_mask_in_c, const bool emit_tendency) {
  // A tile spans consecutive x lanes and multiple y rows, so both directional
  // passes can reuse faces while keeping global loads coalesced along x.
  const int64_t tile_x = blockIdx.x % tiles_x;
  const int64_t row_index = blockIdx.x / tiles_x;
  const int64_t tile_y = row_index % tiles_y;
  const int64_t column_index = row_index / tiles_y;
  const int64_t batch = column_index / levels;
  const int64_t level = column_index - batch * levels;
  const int local_x = threadIdx.x % tile_width;
  const int local_y = threadIdx.x / tile_width;
  const int64_t x = tile_x * tile_width + local_x;
  const int64_t y = tile_y * tile_height + local_y;
  const bool valid = local_y < tile_height && x < nx && y < ny;
  const int64_t output_index = ((batch * levels + level) * ny + y) * nx + x;

  const int64_t lower_face = AXIS == 0 ? x : y;
  const int64_t upper_face = lower_face + 1;

  extern __shared__ __align__(16) unsigned char face_storage[];
  scalar_t *const shared_flux = reinterpret_cast<scalar_t *>(face_storage);
  scalar_t flux_lower;
  if (valid) {
    flux_lower =
        face_flux<scalar_t, AXIS>(t_wide, t_left, t_right, mask_wide, vel_face,
                                  vel_trans, recip_dc, batch, level, y, x,
                                  lower_face, dt, dst_start, dst_stop,
                                  with_limiter);
    shared_flux[threadIdx.x] = flux_lower;
  }
  // Even unused lanes in a partial tile must reach the shared-memory barrier.
  __syncthreads();
  if (!valid) {
    return;
  }

  const bool share_upper = AXIS == 0
                               ? local_x + 1 < tile_width && x + 1 < nx
                               : local_y + 1 < tile_height && y + 1 < ny;
  scalar_t flux_upper;
  if (share_upper) {
    // The neighbour's lower face has identical inputs to this upper face.
    flux_upper = shared_flux[threadIdx.x + (AXIS == 0 ? 1 : tile_width)];
  } else {
    flux_upper =
        face_flux<scalar_t, AXIS>(t_wide, t_left, t_right, mask_wide, vel_face,
                                  vel_trans, recip_dc, batch, level, y, x,
                                  upper_face, dt, dst_start, dst_stop,
                                  with_limiter);
  }

  // c_grid.py:252-259: div = field[..., 1:] - field[..., :-1].
  const scalar_t div_flux = flux_upper - flux_lower;
  const scalar_t trans_lower = vel_trans.load(batch, level,
                                              AXIS == 0 ? y : lower_face,
                                              AXIS == 0 ? lower_face : x);
  const scalar_t trans_upper = vel_trans.load(batch, level,
                                              AXIS == 0 ? y : upper_face,
                                              AXIS == 0 ? upper_face : x);
  const scalar_t div_transport = trans_upper - trans_lower;

  // tracers.py:107-120 (_multidim_advective_pass_update).
  const scalar_t orig_value = tracer_orig.load(batch, level, y, x);
  const scalar_t correction = orig_value * div_transport;
  const scalar_t update_difference = div_flux - correction;
  const scalar_t recip_volume_value = recip_vol.load(batch, level, y, x);
  scalar_t update = update_difference * recip_volume_value;
  if (has_mask_in_c) {
    const scalar_t mask_value = mask_in_c.load(batch, level, y, x);
    update = update * mask_value;
  }

  // tracers.py:774 / 843: localT = localT - dt * update.
  const scalar_t local_value = local_t.load(batch, level, y, x);
  const scalar_t scaled_update = dt * update;
  const scalar_t updated = local_value - scaled_update;

  if (!emit_tendency) {
    out[output_index] = updated;
    return;
  }
  // tracers.py:855: tendency = (localT - tracer_orig) / dt.  ``inv_dt`` is the
  // host-side reciprocal ATen itself broadcasts for this division; see the
  // contract note at the top of this file.
  const scalar_t tendency_numerator = updated - orig_value;
  out[output_index] = tendency_numerator * inv_dt;
}

template <typename scalar_t, int AXIS>
void launch_axis(const torch::Tensor &t_wide, const torch::Tensor &t_left,
                 const torch::Tensor &t_right, const torch::Tensor &mask_wide,
                 const torch::Tensor &vel_face, const torch::Tensor &vel_trans,
                 const torch::Tensor &recip_dc, const torch::Tensor &recip_vol,
                 const torch::Tensor &mask_in_c,
                 const torch::Tensor &tracer_orig, const torch::Tensor &local_t,
                 torch::Tensor &out, const double dt, const int64_t dst_start,
                 const int64_t dst_stop, const bool with_limiter,
                 const bool has_mask_in_c, const bool emit_tendency) {
  const auto &config = launch_config_cache.get(
      tracer_adv_pass_kernel<scalar_t, AXIS>, t_wide.get_device(), 0, 256,
      dynamic_shared_bytes_for_block<scalar_t>);
  const int threads_per_block = config.selected_block_size;
  const int tile_width = static_cast<int>(
      std::min<int64_t>(out.size(3), std::min(32, threads_per_block)));
  const int tile_height = threads_per_block / tile_width;
  const int64_t tiles_x = (out.size(3) + tile_width - 1) / tile_width;
  const int64_t tiles_y = (out.size(2) + tile_height - 1) / tile_height;
  const int64_t block_count =
      out.size(0) * out.size(1) * tiles_y * tiles_x;
  TORCH_CHECK(block_count <= std::numeric_limits<unsigned int>::max(),
              "tracer advection fused grid exceeds grid.x capacity");
  const dim3 block(threads_per_block);
  const dim3 grid(static_cast<unsigned int>(block_count));
  tracer_adv_pass_kernel<scalar_t, AXIS>
      <<<grid, block, dynamic_shared_bytes_for_block<scalar_t>(threads_per_block),
         at::cuda::getCurrentCUDAStream()>>>(
          input4<scalar_t>(t_wide), input4<scalar_t>(t_left),
          input4<scalar_t>(t_right), input4<scalar_t>(mask_wide),
          input4<scalar_t>(vel_face), input4<scalar_t>(vel_trans),
          input4<scalar_t>(recip_dc), input4<scalar_t>(recip_vol),
          input4<scalar_t>(mask_in_c), input4<scalar_t>(tracer_orig),
          input4<scalar_t>(local_t), out.data_ptr<scalar_t>(), tiles_x, tiles_y,
          tile_width, tile_height, out.size(1), out.size(2), out.size(3),
          static_cast<scalar_t>(dt),
          static_cast<scalar_t>(1) / static_cast<scalar_t>(dt), dst_start,
          dst_stop, with_limiter, has_mask_in_c, emit_tendency);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

void tracer_adv_pass_launch(
    const torch::Tensor &t_wide, const torch::Tensor &t_left,
    const torch::Tensor &t_right, const torch::Tensor &mask_wide,
    const torch::Tensor &vel_face, const torch::Tensor &vel_trans,
    const torch::Tensor &recip_dc, const torch::Tensor &recip_vol,
    const torch::Tensor &mask_in_c, const torch::Tensor &tracer_orig,
    const torch::Tensor &local_t, torch::Tensor &out, const double dt,
    const int64_t dst_start, const int64_t dst_stop, const bool with_limiter,
    const bool has_mask_in_c, const bool emit_tendency, const bool axis_is_y) {
  c10::cuda::CUDAGuard device_guard(t_wide.device());
  AT_DISPATCH_FLOATING_TYPES(
      t_wide.scalar_type(), "tracer_adv_pass_launch", [&] {
        if (axis_is_y) {
          launch_axis<scalar_t, 1>(t_wide, t_left, t_right, mask_wide, vel_face,
                                   vel_trans, recip_dc, recip_vol, mask_in_c,
                                   tracer_orig, local_t, out, dt, dst_start,
                                   dst_stop, with_limiter, has_mask_in_c,
                                   emit_tendency);
        } else {
          launch_axis<scalar_t, 0>(t_wide, t_left, t_right, mask_wide, vel_face,
                                   vel_trans, recip_dc, recip_vol, mask_in_c,
                                   tracer_orig, local_t, out, dt, dst_start,
                                   dst_stop, with_limiter, has_mask_in_c,
                                   emit_tendency);
        }
      });
}

const defgcm::cuda_launch::LaunchConfig &
tracer_adv_fused_launch_config(int device, at::ScalarType dtype,
                               bool axis_is_y) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  // Mirrors AT_DISPATCH_FLOATING_TYPES plus the axis template above: the four
  // instantiations carry different register footprints, so the exported
  // configuration must name the one this call actually launches.
  if (dtype == at::kFloat) {
    if (axis_is_y) {
      return launch_config_cache.get(tracer_adv_pass_kernel<float, 1>, device,
                                     0, 256, dynamic_shared_bytes_for_block<float>);
    }
    return launch_config_cache.get(tracer_adv_pass_kernel<float, 0>, device, 0,
                                   256, dynamic_shared_bytes_for_block<float>);
  }
  TORCH_CHECK(dtype == at::kDouble,
              "tracer advection fused launch configuration supports float64 or "
              "float32 only");
  if (axis_is_y) {
    return launch_config_cache.get(tracer_adv_pass_kernel<double, 1>, device, 0,
                                   256, dynamic_shared_bytes_for_block<double>);
  }
  return launch_config_cache.get(tracer_adv_pass_kernel<double, 0>, device, 0,
                                 256, dynamic_shared_bytes_for_block<double>);
}
