#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "../_cuda_launch/launch_config.h"

namespace {

locomgcm::cuda_launch::LaunchConfigCache launch_config_cache;

// The per-column scratch `c_prime` lives in a global workspace, not in dynamic
// shared memory, so the occupancy derivation carries no shared-memory term and
// the block size it selects is bounded by registers alone. The callback still
// has to exist: the occupancy helper calls it unconditionally.
size_t no_dynamic_shared_bytes(int block_size) {
  (void)block_size;
  return 0;
}

constexpr int64_t kErrorTypesPerLevel = 5;

// Pivot solvability threshold, per dtype.
//
// The rule is not invented here: the eager reference solver
// (locomgcm/solvers/implicit_vertical.py, solve_tridiagonal_system) computes
// tau = (8 * torch.finfo(rhs.dtype).eps) * scale, and it has always derived
// eps from the run dtype rather than from float64. Both instantiations below
// are that same rule spelled at their own scalar type, so the accelerated and
// eager paths reject the same columns by construction; the static_asserts hold
// the two literals to it, since a hand-typed decimal expansion is exactly the
// kind of constant that drifts silently.
//
// The float64 value is the literal this kernel has always carried, character
// for character, so the fp64 instantiation compares against the same constant
// as before, bit for bit.
//
// What remains open is not the scaling but the threshold's meaning at float32
// conditioning: 8*eps*scale has not been shown to separate solvable from
// unsolvable columns there, no float32 result from this kernel has been
// validated against MITgcm or the eager reference, and a float32 solve of a
// stiff vertical system may return a plausible-looking but wrong answer
// without tripping the diagnostic contract. Only the float64 arm is a
// production path. Closing that needs stiff/ill-conditioned columns run on the
// A800 in both dtypes.
template <typename scalar_t>
struct PivotTau;

template <>
struct PivotTau<double> {
  static constexpr double value = 8.0 * 2.2204460492503130808472633361816e-16;
};

template <>
struct PivotTau<float> {
  static constexpr float value = 8.0f * 1.1920928955078125e-07f;
};

static_assert(
    PivotTau<double>::value == 8.0 * std::numeric_limits<double>::epsilon(),
    "float64 pivot tau must be the eager reference's 8*eps rule");
static_assert(
    PivotTau<float>::value == 8.0f * std::numeric_limits<float>::epsilon(),
    "float32 pivot tau must be the eager reference's 8*eps rule");

// Preserve MIT's rounded reciprocal followed by two multiplications. Resolved
// by overload so the double call site still emits exactly __ddiv_rn(1.0, x).
__device__ __forceinline__ double reciprocal_rn(const double value) {
  return __ddiv_rn(1.0, value);
}

__device__ __forceinline__ float reciprocal_rn(const float value) {
  return __fdiv_rn(1.0f, value);
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

template <typename scalar_t>
struct Tensor4 {
  const scalar_t* data;
  Strides4 strides;

  __device__ __forceinline__ scalar_t load(
      const int64_t batch,
      const int level,
      const int64_t y,
      const int64_t x) const {
    return load4(data, strides, batch, level, y, x);
  }
};

template <typename scalar_t>
Tensor4<scalar_t> tensor4(const torch::Tensor& value) {
  return {value.data_ptr<scalar_t>(), strides4(value)};
}

template <typename scalar_t>
struct BandAccessor {
  using value_type = scalar_t;
  Tensor4<scalar_t> lower, main, upper;
};

__device__ __forceinline__ double multiply_rn(double a, double b) {
  return __dmul_rn(a, b);
}
__device__ __forceinline__ float multiply_rn(float a, float b) {
  return __fmul_rn(a, b);
}
__device__ __forceinline__ double subtract_rn(double a, double b) {
  return __dsub_rn(a, b);
}
__device__ __forceinline__ float subtract_rn(float a, float b) {
  return __fsub_rn(a, b);
}

// Only the four-band entry fixes each multiply/subtract rounding. Physical
// tracer/momentum retain their existing expressions and contraction policy.
template <typename Accessor, typename T>
__device__ __forceinline__ T thomas_product(T a, T b) {
  if constexpr (std::is_same<Accessor, BandAccessor<T>>::value) {
    return multiply_rn(a, b);
  } else {
    return a * b;
  }
}

template <typename Accessor, typename T>
__device__ __forceinline__ T thomas_difference(T a, T b, T c) {
  if constexpr (std::is_same<Accessor, BandAccessor<T>>::value) {
    return subtract_rn(a, multiply_rn(b, c));
  } else {
    return a - b * c;
  }
}

template <typename Accessor, typename T>
__device__ __forceinline__ T thomas_subtract(T a, T b) {
  if constexpr (std::is_same<Accessor, BandAccessor<T>>::value) {
    return subtract_rn(a, b);
  } else {
    return a - b;
  }
}

__device__ __forceinline__ void record_error(
    int64_t* error_code,
    const bool condition,
    const int level,
    const int error_type,
    const int64_t column,
    const int64_t total_columns) {
  if (!condition) {
    return;
  }
  const int64_t code =
      (static_cast<int64_t>(level) * kErrorTypesPerLevel + error_type - 1) *
          total_columns +
      column + 1;
  atomicMin(reinterpret_cast<unsigned long long*>(error_code),
            static_cast<unsigned long long>(code));
}

__device__ __forceinline__ void record_result_error(
    int64_t* error_code,
    const bool condition,
    const int64_t result_index,
    const int nz,
    const int64_t total_columns) {
  if (!condition) {
    return;
  }
  const int64_t code =
      static_cast<int64_t>(nz) * kErrorTypesPerLevel * total_columns +
      result_index + 1;
  atomicMin(reinterpret_cast<unsigned long long*>(error_code),
            static_cast<unsigned long long>(code));
}

template <typename scalar_t>
struct TracerAccessor {
  using value_type = scalar_t;

  Tensor4<scalar_t> dt;
  Tensor4<scalar_t> diffusivity;
  Tensor4<scalar_t> recip_hfac;
  Tensor4<scalar_t> mask;
  Tensor4<scalar_t> area;
  Tensor4<scalar_t> reciprocal_area;
  Tensor4<scalar_t> recip_dr_f;
  Tensor4<scalar_t> recip_dr_c;
  Tensor4<scalar_t> w;

  __device__ __forceinline__ void scales(
      const int64_t batch,
      const int k,
      const int64_t y,
      const int64_t x,
      scalar_t& diffusion_scale,
      scalar_t& advection_scale) const {
    const scalar_t row_scale =
        recip_hfac.load(batch, k, y, x) * recip_dr_f.load(batch, k, y, x);
    diffusion_scale = dt.load(batch, k, y, x) * row_scale;
    advection_scale = diffusion_scale;
  }

  __device__ __forceinline__ scalar_t interface_value(
      const int64_t batch,
      const int k,
      const int64_t y,
      const int64_t x) const {
    return diffusivity.load(batch, k + 1, y, x) *
        recip_dr_c.load(batch, k, y, x);
  }

  __device__ __forceinline__ scalar_t row_mask(
      const int64_t batch,
      const int k,
      const int64_t y,
      const int64_t x) const {
    return mask.load(batch, k, y, x);
  }

  __device__ __forceinline__ scalar_t center_value(
      const int64_t batch,
      const int k,
      const int64_t y,
      const int64_t x) const {
    if (k == 0) {
      return static_cast<scalar_t>(0.0);
    }
    const scalar_t transport =
        (w.load(batch, k, y, x) * area.load(batch, 0, y, x)) *
        mask.load(batch, k - 1, y, x);
    return ((static_cast<scalar_t>(0.5) * transport) *
            reciprocal_area.load(batch, 0, y, x)) *
        static_cast<scalar_t>(-1.0);
  }
};

template <typename scalar_t>
struct MomentumAccessor {
  using value_type = scalar_t;

  Tensor4<scalar_t> dt;
  Tensor4<scalar_t> viscosity;
  Tensor4<scalar_t> recip_hfac;
  Tensor4<scalar_t> mask;
  Tensor4<scalar_t> w_left;
  Tensor4<scalar_t> w_right;
  Tensor4<scalar_t> area_left;
  Tensor4<scalar_t> area_right;
  Tensor4<scalar_t> mask_left;
  Tensor4<scalar_t> mask_right;
  Tensor4<scalar_t> reciprocal_face_area;
  Tensor4<scalar_t> recip_dr_f;
  Tensor4<scalar_t> recip_dr_c;

  __device__ __forceinline__ void scales(
      const int64_t batch,
      const int k,
      const int64_t y,
      const int64_t x,
      scalar_t& diffusion_scale,
      scalar_t& advection_scale) const {
    const scalar_t row_scale =
        recip_hfac.load(batch, k, y, x) * recip_dr_f.load(batch, k, y, x);
    diffusion_scale = dt.load(batch, k, y, x) * row_scale;
    advection_scale = row_scale;
  }

  __device__ __forceinline__ scalar_t interface_value(
      const int64_t batch,
      const int k,
      const int64_t y,
      const int64_t x) const {
    return viscosity.load(batch, k + 1, y, x) *
        recip_dr_c.load(batch, k, y, x);
  }

  __device__ __forceinline__ scalar_t row_mask(
      const int64_t batch,
      const int k,
      const int64_t y,
      const int64_t x) const {
    return mask.load(batch, k, y, x);
  }

  __device__ __forceinline__ scalar_t center_value(
      const int64_t batch,
      const int k,
      const int64_t y,
      const int64_t x) const {
    if (k == 0) {
      return static_cast<scalar_t>(0.0);
    }
    const int mask_level = k - 1;
    const scalar_t weighted_left =
        (w_left.load(batch, k, y, x) * area_left.load(batch, 0, y, x)) *
        mask_left.load(batch, mask_level, y, x);
    const scalar_t weighted_right =
        (w_right.load(batch, k, y, x) * area_right.load(batch, 0, y, x)) *
        mask_right.load(batch, mask_level, y, x);
    const scalar_t transport =
        static_cast<scalar_t>(0.5) * (weighted_left + weighted_right);
    return (((static_cast<scalar_t>(0.5) * dt.load(batch, mask_level, y, x)) *
             transport) *
            reciprocal_face_area.load(batch, 0, y, x)) *
        static_cast<scalar_t>(-1.0);
  }
};

// A single row definition feeds the primal and transposed MIT-order sweeps.
template <bool kTracer, bool kUpwind, typename Accessor>
__device__ __forceinline__ void physical_row(
    const Accessor& accessor, int64_t batch, int k, int64_t y, int64_t x,
    int nz, typename Accessor::value_type& lower,
    typename Accessor::value_type& main, typename Accessor::value_type& upper) {
  using T = typename Accessor::value_type;
  T diff, adv;
  accessor.scales(batch, k, y, x, diff, adv);
  lower = upper = T(0);
  if (k > 0) {
    const T mask = accessor.row_mask(batch, k - 1, y, x);
    const T value = accessor.interface_value(batch, k - 1, y, x);
    if constexpr (kTracer) lower = -(diff * mask * value);
    else lower = -(diff * value * mask);
  }
  if (k + 1 < nz) {
    const T mask = accessor.row_mask(batch, k + 1, y, x);
    const T value = accessor.interface_value(batch, k, y, x);
    if constexpr (kTracer) upper = -(diff * mask * value);
    else upper = -(diff * value * mask);
  }
  main = k == 0 ? T(1) - upper : T(1) - (lower + upper);
  if (k > 0) {
    const T center = accessor.center_value(batch, k, y, x);
    const T upwind = kUpwind ? fabs(center) : T(0);
    lower = lower - adv * (center + upwind);
    if constexpr (kTracer) main = main - adv * (center - upwind);
    else main = main + adv * (center + upwind);
  }
  if (k + 1 < nz) {
    const T center = accessor.center_value(batch, k + 1, y, x);
    const T upwind = kUpwind ? fabs(center) : T(0);
    if constexpr (kTracer) main = main + adv * (center + upwind);
    else main = main - adv * (center - upwind);
    upper = upper + adv * (center - upwind);
  }
}

template <bool kTracer, bool kUpwind, bool kTranspose, typename T>
__device__ __forceinline__ void solve_row(
    const BandAccessor<T>& a, int64_t batch, int k, int64_t y, int64_t x,
    int nz, T& lower, T& main, T& upper) {
  main = a.main.load(batch, k, y, x);
  if constexpr (kTranspose) {
    lower = k > 0 ? a.upper.load(batch, k - 1, y, x) : T(0);
    upper = k + 1 < nz ? a.lower.load(batch, k + 1, y, x) : T(0);
  } else {
    lower = a.lower.load(batch, k, y, x);
    upper = a.upper.load(batch, k, y, x);
  }
}

template <bool kTracer, bool kUpwind, bool kTranspose, typename Accessor>
__device__ __forceinline__ void solve_row(
    const Accessor& accessor, int64_t batch, int k, int64_t y, int64_t x,
    int nz, typename Accessor::value_type& lower,
    typename Accessor::value_type& main, typename Accessor::value_type& upper) {
  physical_row<kTracer, kUpwind>(accessor, batch, k, y, x, nz, lower, main, upper);
  if constexpr (kTranspose) {
    using T = typename Accessor::value_type;
    T a, b, c;
    lower = upper = T(0);
    if (k > 0) {
      physical_row<kTracer, kUpwind>(accessor, batch, k - 1, y, x, nz, a, b, c);
      lower = c;
    }
    if (k + 1 < nz) {
      physical_row<kTracer, kUpwind>(accessor, batch, k + 1, y, x, nz, a, b, c);
      upper = a;
    }
  }
}

template <typename scalar_t, bool kTracer, bool kUpwind, typename Accessor, bool kTranspose = false>
__global__ void __launch_bounds__(256) direct_band_thomas_kernel(
    const scalar_t* __restrict__ rhs,
    const Strides4 rhs_strides,
    const Accessor accessor,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ c_prime,
    int64_t* __restrict__ error_code,
    const int64_t batch_size,
    const int nz,
    const int64_t ny,
    const int64_t nx,
    const bool diagnostic_contract) {
  const int lane = threadIdx.x;
  const int block_size = blockDim.x;
  const int64_t nplane = ny * nx;
  const int64_t total_columns = batch_size * nplane;
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * block_size + lane;
  if (column >= total_columns) {
    return;
  }
  const int64_t batch = column / nplane;
  const int64_t plane_column = column - batch * nplane;
  const int64_t y = plane_column / nx;
  const int64_t x = plane_column - y * nx;

  scalar_t lower, main, upper;
  solve_row<kTracer, kUpwind, kTranspose>(
      accessor, batch, 0, y, x, nz, lower, main, upper);

  const scalar_t rhs0 = load4(rhs, rhs_strides, batch, 0, y, x);
  scalar_t check_c = static_cast<scalar_t>(0.0);
  scalar_t check_d = static_cast<scalar_t>(0.0);
  if (diagnostic_contract) {
    const scalar_t scale = fmax(static_cast<scalar_t>(1.0), fabs(main));
    record_error(error_code, !isfinite(main), 0, 1, column, total_columns);
    record_error(
        error_code,
        fabs(main) <= PivotTau<scalar_t>::value * scale,
        0,
        2,
        column,
        total_columns);
    check_c = upper / main;
    check_d = rhs0 / main;
    record_error(error_code, !isfinite(check_c), 0, 4, column, total_columns);
    record_error(error_code, !isfinite(check_d), 0, 5, column, total_columns);
  }

  // Preserve MIT's rounded reciprocal followed by two multiplications.
  scalar_t reciprocal = reciprocal_rn(main);
  scalar_t previous_c = thomas_product<Accessor>(upper, reciprocal);
  scalar_t previous_d = thomas_product<Accessor>(rhs0, reciprocal);
  c_prime[column] = previous_c;
  output[batch * static_cast<int64_t>(nz) * nplane + plane_column] = previous_d;

  for (int k = 1; k < nz; ++k) {
    solve_row<kTracer, kUpwind, kTranspose>(
        accessor, batch, k, y, x, nz, lower, main, upper);

    const scalar_t pivot = thomas_difference<Accessor>(main, lower, previous_c);
    const scalar_t rhs_value = load4(rhs, rhs_strides, batch, k, y, x);
    if (diagnostic_contract) {
      const scalar_t elimination_product = thomas_product<Accessor>(lower, check_c);
      const scalar_t check_pivot = thomas_subtract<Accessor>(main, elimination_product);
      const scalar_t scale = fmax(
          static_cast<scalar_t>(1.0),
          fabs(main) + fabs(elimination_product));
      record_error(
          error_code,
          !isfinite(check_pivot),
          k,
          1,
          column,
          total_columns);
      record_error(
          error_code,
          fabs(check_pivot) <= PivotTau<scalar_t>::value * scale,
          k,
          2,
          column,
          total_columns);
      const scalar_t multiplier = lower / check_pivot;
      const scalar_t next_check_c = upper / check_pivot;
      const scalar_t next_check_d =
          thomas_difference<Accessor>(rhs_value, lower, check_d) / check_pivot;
      record_error(
          error_code,
          !isfinite(multiplier),
          k,
          3,
          column,
          total_columns);
      record_error(
          error_code,
          !isfinite(next_check_c),
          k,
          4,
          column,
          total_columns);
      record_error(
          error_code,
          !isfinite(next_check_d),
          k,
          5,
          column,
          total_columns);
      check_c = next_check_c;
      check_d = next_check_d;
    }
    reciprocal = reciprocal_rn(pivot);
    const scalar_t current_c = thomas_product<Accessor>(upper, reciprocal);
    const scalar_t rhs_numerator = thomas_difference<Accessor>(rhs_value, lower, previous_d);
    const scalar_t current_d = thomas_product<Accessor>(rhs_numerator, reciprocal);
    const int64_t scratch_index =
        static_cast<int64_t>(k) * total_columns + column;
    c_prime[scratch_index] = current_c;
    output[
        batch * static_cast<int64_t>(nz) * nplane +
        static_cast<int64_t>(k) * nplane + plane_column] = current_d;
    previous_c = current_c;
    previous_d = current_d;
  }

  scalar_t next_result = previous_d;
  output[
      batch * static_cast<int64_t>(nz) * nplane +
      static_cast<int64_t>(nz - 1) * nplane + plane_column] = next_result;
  int64_t result_index =
      (batch * static_cast<int64_t>(nz) + nz - 1) * nplane + plane_column;
  record_result_error(
      error_code,
      !isfinite(next_result),
      result_index,
      nz,
      total_columns);
  if (nz > 1) {
    int64_t scratch_index =
        static_cast<int64_t>(nz - 2) * total_columns + column;
    int64_t output_index =
        batch * static_cast<int64_t>(nz) * nplane +
        static_cast<int64_t>(nz - 2) * nplane + plane_column;
    scalar_t prefetched_c = c_prime[scratch_index];
    scalar_t prefetched_d = output[output_index];
    for (int k = nz - 2; k >= 0; --k) {
      const scalar_t current_c = prefetched_c;
      const scalar_t current_d = prefetched_d;
      // Load the next row before this row's dependent arithmetic and store.
      // These are same-thread scratch writes; keep ordinary (not __ldg) loads.
      if (k > 0) {
        prefetched_c = c_prime[scratch_index - total_columns];
        prefetched_d = output[output_index - nplane];
      }
      next_result = thomas_difference<Accessor>(current_d, current_c, next_result);
      output[output_index] = next_result;
      record_result_error(
          error_code,
          !isfinite(next_result),
          output_index,
          nz,
          total_columns);
      scratch_index -= total_columns;
      output_index -= nplane;
    }
  }
}

template <typename scalar_t, bool kTracer, bool kUpwind, typename Accessor, bool kTranspose = false>
void launch_direct_variant(
    const torch::Tensor& rhs,
    const Accessor& accessor,
    torch::Tensor& output,
    torch::Tensor& error_code,
    bool diagnostic_contract) {
  const auto batch = static_cast<int64_t>(rhs.size(0));
  const auto nz = static_cast<int>(rhs.size(1));
  const auto ny = static_cast<int64_t>(rhs.size(2));
  const auto nx = static_cast<int64_t>(rhs.size(3));
  const int64_t total_columns = batch * ny * nx;
  const auto& launch_config = launch_config_cache.get(
      direct_band_thomas_kernel<scalar_t, kTracer, kUpwind, Accessor, kTranspose>,
      rhs.get_device(),
      nz,
      16,
      no_dynamic_shared_bytes);
  const int columns_per_block = launch_config.selected_block_size;
  const int64_t block_count =
      (total_columns + columns_per_block - 1) / columns_per_block;
  TORCH_CHECK(
      block_count <= std::numeric_limits<unsigned int>::max(),
      "fused vertical CUDA grid exceeds grid.x capacity");
  // Level-major so that consecutive lanes address consecutive scratch slots.
  // Allocated per call from the caching allocator on the launch stream, written
  // and read by this kernel only, and released before the entry returns.
  torch::Tensor c_prime = torch::empty(
      {static_cast<int64_t>(nz), total_columns}, rhs.options());

  const dim3 block(columns_per_block);
  const dim3 grid(static_cast<unsigned int>(block_count));
  direct_band_thomas_kernel<scalar_t, kTracer, kUpwind, Accessor, kTranspose>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          rhs.data_ptr<scalar_t>(),
          strides4(rhs),
          accessor,
          output.data_ptr<scalar_t>(),
          c_prime.data_ptr<scalar_t>(),
          error_code.data_ptr<int64_t>(),
          batch,
          nz,
          ny,
          nx,
          diagnostic_contract);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Each thread owns a column, including its two horizontal metric cotangents.
// Optional output pointers avoid materializing gradients of fixed inputs.
template <typename T>
struct TracerBars {
  T *w, *dt, *kappa, *rm, *mask, *area, *ra, *drf, *drc;
};

template <typename T>
__global__ void tracer_physical_pullback_kernel(
    TracerAccessor<T> a, Tensor4<T> solution, Tensor4<T> adjoint,
    TracerBars<T> out, int64_t batch_size, int nz, int64_t ny, int64_t nx,
    int64_t* error_code) {
  const int64_t plane = ny * nx;
  const int64_t columns = batch_size * plane;
  const int64_t column = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= columns) return;
  const int64_t batch = column / plane, cell = column % plane;
  const int64_t y = cell / nx, x = cell % nx;
  const int64_t base = batch * nz * plane + cell;
  const int64_t wbase = batch * (nz + 1) * plane + cell;
  const int64_t cbase = batch * (nz - 1) * plane + cell;
  for (int k = 0; k < nz; ++k) {
    const int64_t i = base + k * plane;
    if (out.dt) out.dt[i] = T(0);
    if (out.kappa) out.kappa[i] = T(0);
    if (out.rm) out.rm[i] = T(0);
    if (out.mask) out.mask[i] = T(0);
    if (out.drf) out.drf[i] = T(0);
  }
  if (out.w) {
    out.w[wbase] = T(0);
    out.w[wbase + nz * plane] = T(0);
  }
  T area_bar = T(0), ra_bar = T(0);
  const T area = a.area.load(batch, 0, y, x);
  const T ra = a.reciprocal_area.load(batch, 0, y, x);
  for (int k = 1; k < nz; ++k) {
    const int64_t hi = base + (k - 1) * plane, lo = hi + plane;
    const T xu = solution.load(batch, k - 1, y, x);
    const T xl = solution.load(batch, k, y, x);
    const T lu = adjoint.load(batch, k - 1, y, x);
    const T ll = adjoint.load(batch, k, y, x);
    const T al = -ll * xu, bl = -ll * xl;
    const T bu = -lu * xu, cu = -lu * xl;
    const T ml = bl - al, mu = bu - cu;
    T lam_l, unused, lam_u;
    a.scales(batch, k, y, x, lam_l, unused);
    a.scales(batch, k - 1, y, x, lam_u, unused);
    const T mask_u = a.mask.load(batch, k - 1, y, x);
    const T mask_l = a.mask.load(batch, k, y, x);
    const T drc = a.recip_dr_c.load(batch, k - 1, y, x);
    const T kap = a.diffusivity.load(batch, k, y, x);
    const T interface = kap * drc;
    const T wi = a.w.load(batch, k, y, x);
    const T transport = (wi * area) * mask_u;
    const T center = ((T(0.5) * transport) * ra) * T(-1);
    if (out.dt || out.rm || out.drf) {
      const T upwind = fabs(center);
      const T plus = center + upwind, minus = center - upwind;
      const T bar_l = ml * mask_u * interface - al * plus - bl * minus;
      const T bar_u = mu * mask_l * interface + bu * plus + cu * minus;
      const T dt_l = a.dt.load(batch, k, y, x);
      const T dt_u = a.dt.load(batch, k - 1, y, x);
      const T rm_l = a.recip_hfac.load(batch, k, y, x);
      const T rm_u = a.recip_hfac.load(batch, k - 1, y, x);
      const T f_l = a.recip_dr_f.load(batch, k, y, x);
      const T f_u = a.recip_dr_f.load(batch, k - 1, y, x);
      if (out.dt) {out.dt[lo] += bar_l * (rm_l * f_l); out.dt[hi] += bar_u * (rm_u * f_u);}
      if (out.rm) {out.rm[lo] += (bar_l * dt_l) * f_l; out.rm[hi] += (bar_u * dt_u) * f_u;}
      if (out.drf) {out.drf[lo] += (bar_l * dt_l) * rm_l; out.drf[hi] += (bar_u * dt_u) * rm_u;}
    }
    if (out.kappa || out.drc) {
      const T ibar = ml * lam_l * mask_u + mu * lam_u * mask_l;
      if (out.kappa) out.kappa[lo] = ibar * drc;
      if (out.drc) out.drc[cbase + (k - 1) * plane] = ibar * kap;
    }
    if (out.mask) {
      out.mask[hi] += ml * lam_l * interface;
      out.mask[lo] += mu * lam_u * interface;
    }
    if (out.w || out.mask || out.area || out.ra) {
      T cbar = (-al - bl) * lam_l + (bu + cu) * lam_u;
      const T ubar = (-al + bl) * lam_l + (bu - cu) * lam_u;
      const T sign = center > T(0) ? T(1) : (center < T(0) ? T(-1) : T(0));
      cbar = cbar + ubar * sign;
      const T tbar = ((cbar * T(-1)) * ra) * T(0.5);
      if (out.w) out.w[wbase + k * plane] = (tbar * mask_u) * area;
      if (out.mask) out.mask[hi] += (tbar * wi) * area;
      if (out.area) area_bar += (tbar * wi) * mask_u;
      if (out.ra) ra_bar += (cbar * T(-1)) * (T(0.5) * transport);
    }
  }
  if (out.area) out.area[column] = area_bar;
  if (out.ra) out.ra[column] = ra_bar;
  // Reuse the existing result-error encoding; it identifies the failed column.
  bool bad = (out.area && !isfinite(area_bar)) || (out.ra && !isfinite(ra_bar));
  for (int k = 0; k < nz; ++k) {
    const int64_t i = base + k * plane;
    bad = bad || (out.dt && !isfinite(out.dt[i])) || (out.kappa && !isfinite(out.kappa[i]))
        || (out.rm && !isfinite(out.rm[i])) || (out.mask && !isfinite(out.mask[i]))
        || (out.drf && !isfinite(out.drf[i]));
    if (k > 0) bad = bad || (out.w && !isfinite(out.w[wbase + k * plane]))
        || (out.drc && !isfinite(out.drc[cbase + (k - 1) * plane]));
  }
  record_result_error(error_code, bad, base, nz, columns);
}

template <typename T>
T* optional_data(c10::optional<torch::Tensor>& value) {
  return value ? value->data_ptr<T>() : nullptr;
}

template <typename T>
void tracer_backward_typed(
    const std::vector<torch::Tensor>& v, const torch::Tensor& solution,
    const torch::Tensor& grad_solution, std::vector<c10::optional<torch::Tensor>>& bars,
    torch::Tensor& error_code, bool diagnostic_contract) {
  const TracerAccessor<T> a{
      tensor4<T>(v[1]), tensor4<T>(v[2]), tensor4<T>(v[3]), tensor4<T>(v[4]),
      tensor4<T>(v[5]), tensor4<T>(v[6]), tensor4<T>(v[7]), tensor4<T>(v[8]), tensor4<T>(v[9])};
  auto lambda = bars[0] ? *bars[0] : torch::empty(v[0].sizes(), v[0].options());
  launch_direct_variant<T, true, true, TracerAccessor<T>, true>(
      grad_solution, a, lambda, error_code, diagnostic_contract);
  if (bars[0]) bars[0] = lambda;
  bool physical = false;
  for (int i = 1; i < 10; ++i) physical = physical || bars[i].has_value();
  if (!physical) return;
  TracerBars<T> out{optional_data<T>(bars[1]), optional_data<T>(bars[2]),
      optional_data<T>(bars[3]), optional_data<T>(bars[4]), optional_data<T>(bars[5]),
      optional_data<T>(bars[6]), optional_data<T>(bars[7]), optional_data<T>(bars[8]),
      optional_data<T>(bars[9])};
  static locomgcm::cuda_launch::LaunchConfigCache physical_launch_cache;
  const int nz = v[0].size(1);
  const auto& config = physical_launch_cache.get(tracer_physical_pullback_kernel<T>,
      v[0].get_device(), nz, 16, no_dynamic_shared_bytes);
  const int64_t count = v[0].size(0) * v[0].size(2) * v[0].size(3);
  const int64_t blocks = (count + config.selected_block_size - 1) / config.selected_block_size;
  TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(), "tracer backward grid exceeds grid.x capacity");
  tracer_physical_pullback_kernel<T><<<static_cast<unsigned int>(blocks),
      config.selected_block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
      a, tensor4<T>(solution), tensor4<T>(lambda), out, v[0].size(0), nz,
      v[0].size(2), v[0].size(3), error_code.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kTracer, bool kUpwind, typename Accessor>
void launch_direct(
    const torch::Tensor& rhs,
    const Accessor& accessor,
    torch::Tensor& output,
    torch::Tensor& error_code,
    bool diagnostic_contract) {
  launch_direct_variant<typename Accessor::value_type, kTracer, kUpwind>(
      rhs, accessor, output, error_code, diagnostic_contract);
}

template <typename scalar_t>
void tracer_launch_typed(
    const torch::Tensor& rhs,
    const torch::Tensor& dt,
    const torch::Tensor& diffusivity,
    const torch::Tensor& recip_hfac,
    const torch::Tensor& mask,
    const torch::Tensor& area,
    const torch::Tensor& reciprocal_area,
    const torch::Tensor& recip_dr_f,
    const torch::Tensor& recip_dr_c,
    const torch::Tensor& w,
    torch::Tensor& output,
    torch::Tensor& error_code,
    bool diagnostic_contract) {
  const TracerAccessor<scalar_t> accessor{
      tensor4<scalar_t>(dt),
      tensor4<scalar_t>(diffusivity),
      tensor4<scalar_t>(recip_hfac),
      tensor4<scalar_t>(mask),
      tensor4<scalar_t>(area),
      tensor4<scalar_t>(reciprocal_area),
      tensor4<scalar_t>(recip_dr_f),
      tensor4<scalar_t>(recip_dr_c),
      tensor4<scalar_t>(w)};
  launch_direct<true, true>(
      rhs,
      accessor,
      output,
      error_code,
      diagnostic_contract);
}

template <typename scalar_t>
void momentum_launch_typed(
    const torch::Tensor& rhs,
    const torch::Tensor& dt,
    const torch::Tensor& viscosity,
    const torch::Tensor& recip_hfac,
    const torch::Tensor& mask,
    const torch::Tensor& w_left,
    const torch::Tensor& w_right,
    const torch::Tensor& area_left,
    const torch::Tensor& area_right,
    const torch::Tensor& mask_left,
    const torch::Tensor& mask_right,
    const torch::Tensor& reciprocal_face_area,
    const torch::Tensor& recip_dr_f,
    const torch::Tensor& recip_dr_c,
    torch::Tensor& output,
    torch::Tensor& error_code,
    bool upwind,
    bool diagnostic_contract) {
  const MomentumAccessor<scalar_t> accessor{
      tensor4<scalar_t>(dt),
      tensor4<scalar_t>(viscosity),
      tensor4<scalar_t>(recip_hfac),
      tensor4<scalar_t>(mask),
      tensor4<scalar_t>(w_left),
      tensor4<scalar_t>(w_right),
      tensor4<scalar_t>(area_left),
      tensor4<scalar_t>(area_right),
      tensor4<scalar_t>(mask_left),
      tensor4<scalar_t>(mask_right),
      tensor4<scalar_t>(reciprocal_face_area),
      tensor4<scalar_t>(recip_dr_f),
      tensor4<scalar_t>(recip_dr_c)};
  if (upwind) {
    launch_direct<false, true>(
        rhs,
        accessor,
        output,
        error_code,
        diagnostic_contract);
  } else {
    launch_direct<false, false>(
        rhs,
        accessor,
        output,
        error_code,
        diagnostic_contract);
  }
}

}  // namespace

void implicit_vertical_band_launch(
    const torch::Tensor& lower, const torch::Tensor& main,
    const torch::Tensor& upper, const torch::Tensor& rhs,
    torch::Tensor& output, torch::Tensor& error_code, bool transpose) {
  c10::cuda::CUDAGuard guard(rhs.device());
  AT_DISPATCH_FLOATING_TYPES(rhs.scalar_type(), "implicit_vertical_band_launch", [&] {
    const BandAccessor<scalar_t> a{
        tensor4<scalar_t>(lower), tensor4<scalar_t>(main), tensor4<scalar_t>(upper)};
    if (transpose) {
      launch_direct_variant<scalar_t, false, false, BandAccessor<scalar_t>, true>(
          rhs, a, output, error_code, true);
    } else {
      launch_direct_variant<scalar_t, false, false, BandAccessor<scalar_t>>(
          rhs, a, output, error_code, true);
    }
  });
}

void implicit_vertical_tracer_launch(
    const torch::Tensor& rhs,
    const torch::Tensor& dt,
    const torch::Tensor& diffusivity,
    const torch::Tensor& recip_hfac,
    const torch::Tensor& mask,
    const torch::Tensor& area,
    const torch::Tensor& reciprocal_area,
    const torch::Tensor& recip_dr_f,
    const torch::Tensor& recip_dr_c,
    const torch::Tensor& w,
    torch::Tensor& output,
    torch::Tensor& error_code,
    int64_t advection_scheme,
    bool diagnostic_contract) {
  TORCH_CHECK(advection_scheme == 1, "fused tracer CUDA implements advection scheme 1 only");
  c10::cuda::CUDAGuard device_guard(rhs.device());
  if (rhs.scalar_type() == at::kFloat) {
    tracer_launch_typed<float>(
        rhs, dt, diffusivity, recip_hfac, mask, area, reciprocal_area,
        recip_dr_f, recip_dr_c, w, output, error_code, diagnostic_contract);
    return;
  }
  TORCH_CHECK(
      rhs.scalar_type() == at::kDouble,
      "fused vertical CUDA kernel supports float64 or float32 only");
  tracer_launch_typed<double>(
      rhs, dt, diffusivity, recip_hfac, mask, area, reciprocal_area,
      recip_dr_f, recip_dr_c, w, output, error_code, diagnostic_contract);
}

void implicit_vertical_momentum_launch(
    const torch::Tensor& rhs,
    const torch::Tensor& dt,
    const torch::Tensor& viscosity,
    const torch::Tensor& recip_hfac,
    const torch::Tensor& mask,
    const torch::Tensor& w_left,
    const torch::Tensor& w_right,
    const torch::Tensor& area_left,
    const torch::Tensor& area_right,
    const torch::Tensor& mask_left,
    const torch::Tensor& mask_right,
    const torch::Tensor& reciprocal_face_area,
    const torch::Tensor& recip_dr_f,
    const torch::Tensor& recip_dr_c,
    torch::Tensor& output,
    torch::Tensor& error_code,
    bool upwind,
    bool diagnostic_contract) {
  c10::cuda::CUDAGuard device_guard(rhs.device());
  if (rhs.scalar_type() == at::kFloat) {
    momentum_launch_typed<float>(
        rhs, dt, viscosity, recip_hfac, mask, w_left, w_right, area_left,
        area_right, mask_left, mask_right, reciprocal_face_area, recip_dr_f,
        recip_dr_c, output, error_code, upwind, diagnostic_contract);
    return;
  }
  TORCH_CHECK(
      rhs.scalar_type() == at::kDouble,
      "fused vertical CUDA kernel supports float64 or float32 only");
  momentum_launch_typed<double>(
      rhs, dt, viscosity, recip_hfac, mask, w_left, w_right, area_left,
      area_right, mask_left, mask_right, reciprocal_face_area, recip_dr_f,
      recip_dr_c, output, error_code, upwind, diagnostic_contract);
}

namespace {

template <typename scalar_t, bool kTracer, bool kUpwind, typename Accessor, bool kTranspose = false>
const locomgcm::cuda_launch::LaunchConfig& config_for(int device, int nz) {
  return launch_config_cache.get(
      direct_band_thomas_kernel<scalar_t, kTracer, kUpwind, Accessor, kTranspose>,
      device,
      nz,
      16,
      no_dynamic_shared_bytes);
}

}  // namespace

const locomgcm::cuda_launch::LaunchConfig& implicit_vertical_band_launch_config(
    int device, int nz, at::ScalarType dtype, bool transpose) {
  c10::cuda::CUDAGuard guard(c10::Device(c10::kCUDA,
                                     static_cast<c10::DeviceIndex>(device)));
  if (dtype == at::kFloat) {
    return transpose ? config_for<float, false, false, BandAccessor<float>, true>(device, nz)
                     : config_for<float, false, false, BandAccessor<float>>(device, nz);
  }
  TORCH_CHECK(dtype == at::kDouble, "band CUDA dtype must be float64 or float32");
  return transpose ? config_for<double, false, false, BandAccessor<double>, true>(device, nz)
                   : config_for<double, false, false, BandAccessor<double>>(device, nz);
}

const locomgcm::cuda_launch::LaunchConfig&
implicit_vertical_tracer_launch_config(
    int device, int nz, at::ScalarType dtype) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  if (dtype == at::kFloat) {
    return config_for<float, true, true, TracerAccessor<float>>(device, nz);
  }
  TORCH_CHECK(
      dtype == at::kDouble,
      "fused vertical CUDA launch configuration supports float64 or float32 only");
  return config_for<double, true, true, TracerAccessor<double>>(device, nz);
}

const locomgcm::cuda_launch::LaunchConfig&
implicit_vertical_momentum_upwind_launch_config(
    int device, int nz, at::ScalarType dtype) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  if (dtype == at::kFloat) {
    return config_for<float, false, true, MomentumAccessor<float>>(device, nz);
  }
  TORCH_CHECK(
      dtype == at::kDouble,
      "fused vertical CUDA launch configuration supports float64 or float32 only");
  return config_for<double, false, true, MomentumAccessor<double>>(device, nz);
}

const locomgcm::cuda_launch::LaunchConfig&
implicit_vertical_momentum_centered_launch_config(
    int device, int nz, at::ScalarType dtype) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  if (dtype == at::kFloat) {
    return config_for<float, false, false, MomentumAccessor<float>>(device, nz);
  }
  TORCH_CHECK(
      dtype == at::kDouble,
      "fused vertical CUDA launch configuration supports float64 or float32 only");
  return config_for<double, false, false, MomentumAccessor<double>>(device, nz);
}

void implicit_vertical_tracer_backward_launch(
    const std::vector<torch::Tensor>& values, const torch::Tensor& solution,
    const torch::Tensor& grad_solution, std::vector<c10::optional<torch::Tensor>>& bars,
    torch::Tensor& error_code, int64_t advection_scheme, bool diagnostic_contract) {
  TORCH_CHECK(advection_scheme == 1, "tracer CUDA physical VJP implements advection scheme 1 only");
  c10::cuda::CUDAGuard guard(solution.device());
  if (solution.scalar_type() == at::kFloat) {
    tracer_backward_typed<float>(values, solution, grad_solution, bars, error_code, diagnostic_contract);
  } else {
    tracer_backward_typed<double>(values, solution, grad_solution, bars, error_code, diagnostic_contract);
  }
}
