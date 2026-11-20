#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/Dispatch.h>

#include <cstdint>
#include <limits>

#include "../../solvers/_cuda_launch/launch_config.h"
#include "local_bldepth.cuh"

namespace {

thread_local int occupancy_nz = 0;
defgcm::cuda_launch::LaunchConfigCache launch_config_cache;

template <typename scalar_t>
size_t dynamic_shared_bytes_for_block(int) {
  return static_cast<size_t>(2 * (occupancy_nz + 2)) * sizeof(scalar_t);
}

// One extern __shared__ array can only have one type in a translation unit, so
// the two instantiations share a byte array and reinterpret it.
template <typename scalar_t>
__device__ __forceinline__ scalar_t* shared_scalars() {
  extern __shared__ unsigned char kpp_bldepth_shared_raw[];
  return reinterpret_cast<scalar_t*>(kpp_bldepth_shared_raw);
}

struct Strides1 {
  int64_t x;
};

struct Strides2 {
  int64_t y;
  int64_t x;
};

struct Strides3 {
  int64_t level;
  int64_t y;
  int64_t x;
};

Strides1 strides1(const torch::Tensor& value) {
  return {value.stride(0)};
}

Strides2 strides2(const torch::Tensor& value) {
  return {value.stride(0), value.stride(1)};
}

Strides3 strides3(const torch::Tensor& value) {
  return {value.stride(0), value.stride(1), value.stride(2)};
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t load2(
    const scalar_t* data,
    const Strides2 strides,
    const int64_t y,
    const int64_t x) {
  return __ldg(data + y * strides.y + x * strides.x);
}

__device__ __forceinline__ int64_t load2_long(
    const int64_t* data,
    const Strides2 strides,
    const int64_t y,
    const int64_t x) {
  return data[y * strides.y + x * strides.x];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t load3(
    const scalar_t* data,
    const Strides3 strides,
    const int level,
    const int64_t y,
    const int64_t x) {
  return __ldg(
      data + static_cast<int64_t>(level) * strides.level +
      y * strides.y + x * strides.x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t load_table(
    const scalar_t* data,
    const Strides2 strides,
    const int64_t iz,
    const int64_t ju) {
  return __ldg(data + iz * strides.y + ju * strides.x);
}

__device__ __forceinline__ int64_t clamp_index(
    const int64_t value,
    const int64_t lower,
    const int64_t upper) {
  if (value < lower) {
    return lower;
  }
  if (value > upper) {
    return upper;
  }
  return value;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp_min_nan(
    const scalar_t value,
    const scalar_t lower) {
  if (isnan(value)) {
    return value;
  }
  return value < lower ? lower : value;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t minimum_nan(
    const scalar_t first,
    const scalar_t second) {
  if (isnan(first)) {
    return first;
  }
  if (isnan(second)) {
    return second;
  }
  return first < second ? first : second;
}

// Fortran SIGN(magnitude, sign_source).  The two runtime semantics differ only
// at sign_source = -0.0 (and NaN) and mirror kpp.py::_fortran_sign:
// sign_zero_positive counts -0.0 as non-negative (ifx/ifort default); otherwise
// the sign bit decides (gfortran default).
template <typename scalar_t>
__device__ __forceinline__ scalar_t fsign(
    const scalar_t magnitude,
    const scalar_t sign_source,
    const bool sign_zero_positive) {
  const scalar_t absolute = fabs(magnitude);
  if (sign_zero_positive) {
    return sign_source >= static_cast<scalar_t>(0.0) ? absolute : -absolute;
  }
  return copysign(absolute, sign_source);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t swfrac(
    const scalar_t depth,
    const scalar_t rfac,
    const scalar_t one_minus_rfac,
    const scalar_t inv_a1,
    const scalar_t inv_a2) {
  const scalar_t facz = -depth;
  const scalar_t first = rfac * exp(facz * inv_a1);
  const scalar_t second = one_minus_rfac * exp(facz * inv_a2);
  const scalar_t fraction = first + second;
  return facz < -static_cast<scalar_t>(200.0) ? static_cast<scalar_t>(0.0) : fraction;
}

template <typename scalar_t>
__global__ void swfrac_layers_kernel(
    const scalar_t* zgrid, const Strides1 zgrid_strides,
    scalar_t* fractions, const int nz,
    const scalar_t rfac, const scalar_t one_minus_rfac,
    const scalar_t inv_a1, const scalar_t inv_a2) {
  const int kl = blockIdx.x * blockDim.x + threadIdx.x + 2;
  if (kl <= nz) {
    const scalar_t depth = -__ldg(
        zgrid + static_cast<int64_t>(kl) * zgrid_strides.x);
    fractions[kl] = swfrac(depth, rfac, one_minus_rfac, inv_a1, inv_a2);
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t wscale_ws(
    const scalar_t sigma,
    const scalar_t hbl,
    const scalar_t ustar,
    const scalar_t bfsfc,
    const scalar_t* wst,
    const Strides2 wst_strides,
    const scalar_t inv_deltaz,
    const scalar_t inv_deltau) {
  constexpr int64_t nni = 890;
  constexpr int64_t nnj = 480;
  constexpr scalar_t zmin = -static_cast<scalar_t>(4.0e-7);
  constexpr scalar_t zmax = static_cast<scalar_t>(0.0);
  constexpr scalar_t umin = static_cast<scalar_t>(0.0);
  constexpr scalar_t wscale_vonk = static_cast<scalar_t>(0.4);
  constexpr scalar_t conc1 = static_cast<scalar_t>(5.0);

  const scalar_t zehat = wscale_vonk * sigma * hbl * bfsfc;

  const scalar_t zdiff = zehat - zmin;
  const scalar_t iz_coordinate = zdiff * inv_deltaz;
  int64_t iz = static_cast<int64_t>(iz_coordinate);
  iz = clamp_index(iz, 0, nni);
  const int64_t izp1 = iz + 1;
  const scalar_t udiff = ustar - umin;
  const scalar_t ju_coordinate = udiff * inv_deltau;
  int64_t ju = static_cast<int64_t>(ju_coordinate);
  ju = clamp_index(ju, 0, nnj);
  const int64_t jup1 = ju + 1;
  const scalar_t zfrac_coordinate = zdiff * inv_deltaz;
  const scalar_t zfrac = zfrac_coordinate - static_cast<scalar_t>(iz);
  const scalar_t ufrac_coordinate = udiff * inv_deltau;
  const scalar_t ufrac = ufrac_coordinate - static_cast<scalar_t>(ju);
  const scalar_t fzfrac = static_cast<scalar_t>(1.0) - zfrac;

  const scalar_t wa_first = fzfrac * load_table(wst, wst_strides, iz, jup1);
  const scalar_t wa_second = zfrac * load_table(wst, wst_strides, izp1, jup1);
  const scalar_t wa = wa_first + wa_second;
  const scalar_t wb_first = fzfrac * load_table(wst, wst_strides, iz, ju);
  const scalar_t wb_second = zfrac * load_table(wst, wst_strides, izp1, ju);
  const scalar_t wb = wb_first + wb_second;
  const scalar_t table_first = (static_cast<scalar_t>(1.0) - ufrac) * wb;
  const scalar_t table_second = ufrac * wa;
  const scalar_t ws_table = table_first + table_second;

  const scalar_t ustar_squared = ustar * ustar;
  const scalar_t ustar_cubed = ustar_squared * ustar;
  const scalar_t stable_numerator_first = wscale_vonk * ustar;
  const scalar_t stable_numerator = stable_numerator_first * ustar_cubed;
  const scalar_t stable_denominator_second = conc1 * zehat;
  const scalar_t stable_denominator = ustar_cubed + stable_denominator_second;
  const scalar_t ws_stable = stable_numerator / stable_denominator;
  return zehat <= zmax ? ws_table : ws_stable;
}

template <typename scalar_t>
__global__ void kpp_bldepth_kernel(
    const int64_t* kmtj,
    const Strides2 kmtj_strides,
    const scalar_t* dvsq,
    const Strides3 dvsq_strides,
    const scalar_t* dbloc,
    const Strides3 dbloc_strides,
    const scalar_t* ritop,
    const Strides3 ritop_strides,
    const scalar_t* ustar,
    const Strides2 ustar_strides,
    const scalar_t* bo,
    const Strides2 bo_strides,
    const scalar_t* bosol,
    const Strides2 bosol_strides,
    const scalar_t* coriol,
    const Strides2 coriol_strides,
    const scalar_t* zgrid,
    const Strides1 zgrid_strides,
    const scalar_t* swfrac_layers,
    const scalar_t* hwide,
    const Strides1 hwide_strides,
    const scalar_t* wst,
    const Strides2 wst_strides,
    scalar_t* hbl_output,
    int64_t* kbl_output,
    scalar_t* bfsfc_output,
    scalar_t* stable_output,
    scalar_t* casea_output,
    scalar_t* rib_output,
    const int nz,
    const int64_t ny,
    const int64_t nx,
    const scalar_t ricr,
    const scalar_t epsilon,
    const scalar_t vtc,
    const scalar_t cekman,
    const scalar_t cmonob,
    const scalar_t min_kpp_hbl,
    const scalar_t phepsi,
    const bool limit_hbl_stable,
    const bool penetrating_sw,
    const bool sign_zero_positive,
    const scalar_t swfrac_rfac,
    const scalar_t swfrac_one_minus_rfac,
    const scalar_t swfrac_inv_a1,
    const scalar_t swfrac_inv_a2,
    const scalar_t wscale_inv_deltaz,
    const scalar_t wscale_inv_deltau,
    const scalar_t inv_vonk) {
  scalar_t* shared_values = shared_scalars<scalar_t>();
  scalar_t* zgrid_shared = shared_values;
  scalar_t* hwide_shared = shared_values + nz + 2;
  for (int index = threadIdx.x; index < nz + 2; index += blockDim.x) {
    zgrid_shared[index] = __ldg(
        zgrid + static_cast<int64_t>(index) * zgrid_strides.x);
    hwide_shared[index] = __ldg(
        hwide + static_cast<int64_t>(index) * hwide_strides.x);
  }
  __syncthreads();

  const int64_t total_columns = ny * nx;
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= total_columns) {
    return;
  }
  const int64_t y = column / nx;
  const int64_t x = column - y * nx;
  const scalar_t ustar_value = load2(ustar, ustar_strides, y, x);
  const scalar_t bo_value = load2(bo, bo_strides, y, x);
  const scalar_t bosol_value = load2(bosol, bosol_strides, y, x);

  // The previous row's lower quotient is this row's upper quotient.
  scalar_t bvsq_first = load3(dbloc, dbloc_strides, 0, y, x) /
      (zgrid_shared[1] - zgrid_shared[2]);
  for (int kl = 2; kl <= nz; ++kl) {
    const int k = kl - 1;
    const scalar_t depth = -zgrid_shared[kl];
    const scalar_t penetrating = __ldg(swfrac_layers + kl);
    const scalar_t absorbed = static_cast<scalar_t>(1.0) - penetrating;
    const scalar_t bfsfc_product = bosol_value * absorbed;
    const scalar_t bfsfc = penetrating_sw ? bo_value + bfsfc_product : bo_value;
    const scalar_t stable = static_cast<scalar_t>(0.5) +
        fsign(static_cast<scalar_t>(0.5), bfsfc, sign_zero_positive);
    const scalar_t one_minus_stable = static_cast<scalar_t>(1.0) - stable;
    const scalar_t sigma_product = one_minus_stable * epsilon;
    const scalar_t sigma = stable + sigma_product;
    const scalar_t ws = wscale_ws(
        sigma,
        depth,
        ustar_value,
        bfsfc,
        wst,
        wst_strides,
        wscale_inv_deltaz,
        wscale_inv_deltau);

    const scalar_t dz_k = zgrid_shared[kl] - zgrid_shared[kl + 1];
    const scalar_t bvsq_second =
        load3(dbloc, dbloc_strides, k, y, x) / dz_k;
    const scalar_t bvsq_sum = bvsq_first + bvsq_second;
    const scalar_t bvsq = static_cast<scalar_t>(0.5) * bvsq_sum;
    bvsq_first = bvsq_second;

    scalar_t vtsq_branch = depth * ws;
    vtsq_branch = vtsq_branch * sqrt(fabs(bvsq));
    vtsq_branch = vtsq_branch * vtc;
    const scalar_t vtsq = bvsq == static_cast<scalar_t>(0.0) ? static_cast<scalar_t>(0.0) : vtsq_branch;
    const scalar_t temp_sum = load3(dvsq, dvsq_strides, k, y, x) + vtsq;
    const scalar_t temp_var2 = clamp_min_nan(temp_sum, phepsi);
    rib_output[static_cast<int64_t>(k) * total_columns + column] =
        load3(ritop, ritop_strides, k, y, x) / temp_var2;
  }

  const int64_t kmtj_value = load2_long(kmtj, kmtj_strides, y, x);
  int64_t kbl = kmtj_value;
  for (int kl = 2; kl <= nz; ++kl) {
    const scalar_t rib =
        rib_output[static_cast<int64_t>(kl - 1) * total_columns + column];
    if (kbl == kmtj_value && rib > ricr) {
      kbl = kl;
    }
  }

  const int64_t kbl_init = kmtj_value < 1 ? 1 : kmtj_value;
  const int64_t kbl_init_clamped = clamp_index(kbl_init, 0, nz + 1);
  scalar_t hbl = -zgrid_shared[kbl_init_clamped];

  const bool interpolate = kbl > 1 && kbl < kmtj_value;
  const int64_t kbl_grid = clamp_index(kbl, 0, nz + 1);
  const int64_t kbl_grid_previous = clamp_index(kbl - 1, 0, nz + 1);
  const scalar_t zk = zgrid_shared[kbl_grid];
  const scalar_t zkm1 = zgrid_shared[kbl_grid_previous];
  const int64_t rk_level = clamp_index(kbl - 1, 0, nz - 1);
  const int64_t rkm1_level = clamp_index(kbl - 2, 0, nz - 1);
  const scalar_t rk = rib_output[rk_level * total_columns + column];
  const scalar_t rkm1 = rib_output[rkm1_level * total_columns + column];
  const scalar_t denom_raw = rk - rkm1;
  const scalar_t denom = denom_raw == static_cast<scalar_t>(0.0) ? static_cast<scalar_t>(1.0) : denom_raw;
  const scalar_t interp_depth = -zkm1;
  const scalar_t interp_dz = zkm1 - zk;
  const scalar_t interp_ri = ricr - rkm1;
  const scalar_t interp_product = interp_dz * interp_ri;
  const scalar_t interp_quotient = interp_product / denom;
  const scalar_t hbl_interp = interp_depth + interp_quotient;
  if (interpolate) {
    hbl = hbl_interp;
  }

  scalar_t penetrating = swfrac(
      hbl,
      swfrac_rfac,
      swfrac_one_minus_rfac,
      swfrac_inv_a1,
      swfrac_inv_a2);
  scalar_t absorbed = static_cast<scalar_t>(1.0) - penetrating;
  scalar_t bfsfc_product = bosol_value * absorbed;
  scalar_t bfsfc = penetrating_sw ? bo_value + bfsfc_product : bo_value;
  scalar_t stable = static_cast<scalar_t>(0.5) +
      fsign(static_cast<scalar_t>(0.5), bfsfc, sign_zero_positive);
  const scalar_t bfsfc_magnitude = clamp_min_nan(fabs(bfsfc), phepsi);
  bfsfc = fsign(static_cast<scalar_t>(1.0), bfsfc, sign_zero_positive) *
      bfsfc_magnitude;

  if (limit_hbl_stable) {
    const scalar_t abs_coriol = fabs(load2(coriol, coriol_strides, y, x));
    const scalar_t coriol_denom = clamp_min_nan(abs_coriol, phepsi);
    const scalar_t hekman_numerator = cekman * ustar_value;
    const scalar_t hekman = hekman_numerator / coriol_denom;
    scalar_t hmonob = cmonob * ustar_value;
    hmonob = hmonob * ustar_value;
    hmonob = hmonob * ustar_value;
    hmonob = hmonob * inv_vonk;
    hmonob = hmonob / bfsfc;
    const scalar_t minimum_limit = minimum_nan(hekman, hmonob);
    const scalar_t hlimit_first = stable * minimum_limit;
    const scalar_t hlimit_second = (stable - static_cast<scalar_t>(1.0)) * zgrid_shared[nz];
    const scalar_t hlimit = hlimit_first + hlimit_second;
    const scalar_t limited_hbl = minimum_nan(hbl, hlimit);
    if (bfsfc > static_cast<scalar_t>(0.0)) {
      hbl = limited_hbl;
    }
  }

  hbl = clamp_min_nan(hbl, min_kpp_hbl);
  kbl = kmtj_value;
  for (int kl = 2; kl <= nz; ++kl) {
    const scalar_t depth = -zgrid_shared[kl];
    if (kbl == kmtj_value && depth > hbl) {
      kbl = kl;
    }
  }

  penetrating = swfrac(
      hbl,
      swfrac_rfac,
      swfrac_one_minus_rfac,
      swfrac_inv_a1,
      swfrac_inv_a2);
  absorbed = static_cast<scalar_t>(1.0) - penetrating;
  bfsfc_product = bosol_value * absorbed;
  bfsfc = penetrating_sw ? bo_value + bfsfc_product : bo_value;
  stable = static_cast<scalar_t>(0.5) +
      fsign(static_cast<scalar_t>(0.5), bfsfc, sign_zero_positive);
  const scalar_t final_bfsfc_magnitude = clamp_min_nan(fabs(bfsfc), phepsi);
  bfsfc = fsign(static_cast<scalar_t>(1.0), bfsfc, sign_zero_positive) *
      final_bfsfc_magnitude;
  const int64_t final_kbl_grid = clamp_index(kbl, 0, nz + 1);
  const scalar_t casea_depth = -zgrid_shared[final_kbl_grid];
  const scalar_t casea_half_width = static_cast<scalar_t>(0.5) * hwide_shared[final_kbl_grid];
  const scalar_t casea_minus_width = casea_depth - casea_half_width;
  const scalar_t casea_argument = casea_minus_width - hbl;
  const scalar_t casea = static_cast<scalar_t>(0.5) +
      fsign(static_cast<scalar_t>(0.5), casea_argument, sign_zero_positive);

  hbl_output[column] = hbl;
  kbl_output[column] = kbl;
  bfsfc_output[column] = bfsfc;
  stable_output[column] = stable;
  casea_output[column] = casea;
}

void launch_variant(
    const torch::Tensor& kmtj,
    const torch::Tensor& dvsq,
    const torch::Tensor& dbloc,
    const torch::Tensor& ritop,
    const torch::Tensor& ustar,
    const torch::Tensor& bo,
    const torch::Tensor& bosol,
    const torch::Tensor& coriol,
    const torch::Tensor& zgrid,
    const torch::Tensor& hwide,
    const torch::Tensor& wst,
    torch::Tensor& hbl,
    torch::Tensor& kbl,
    torch::Tensor& bfsfc,
    torch::Tensor& stable,
    torch::Tensor& casea,
    torch::Tensor& rib,
    const double ricr,
    const double epsilon,
    const double vtc,
    const double cekman,
    const double cmonob,
    const double min_kpp_hbl,
    const double vonk,
    const double phepsi,
    const bool limit_hbl_stable,
    const int jwtype,
    const bool penetrating_sw,
    const bool sign_zero_positive) {
  constexpr int64_t nni = 890;
  constexpr int64_t nnj = 480;
  // PyTorch CUDA div.Scalar computes these reciprocals on the host and then
  // launches multiplication. Preserve that lowering instead of device divide.
  const double deltaz = (0.0 - (-4.0e-7)) / static_cast<double>(nni + 1);
  const double deltau = (4.0e-2 - 0.0) / static_cast<double>(nnj + 1);
  const double inv_deltaz = 1.0 / deltaz;
  const double inv_deltau = 1.0 / deltau;
  double swfrac_rfac = 0.62;
  double swfrac_a1 = 0.6;
  double swfrac_a2 = 20.0;
  if (jwtype == 1) {
    swfrac_rfac = 0.58;
    swfrac_a1 = 0.35;
    swfrac_a2 = 23.0;
  } else if (jwtype == 3) {
    swfrac_rfac = 0.67;
    swfrac_a1 = 1.0;
    swfrac_a2 = 17.0;
  } else if (jwtype == 4) {
    swfrac_rfac = 0.77;
    swfrac_a1 = 1.5;
    swfrac_a2 = 14.0;
  } else if (jwtype == 5) {
    swfrac_rfac = 0.78;
    swfrac_a1 = 1.4;
    swfrac_a2 = 7.9;
  }
  const double swfrac_one_minus_rfac = 1.0 - swfrac_rfac;
  const double swfrac_inv_a1 = 1.0 / swfrac_a1;
  const double swfrac_inv_a2 = 1.0 / swfrac_a2;
  const double inv_vonk = 1.0 / vonk;
  const int nz = static_cast<int>(dvsq.size(0));
  const int64_t ny = dvsq.size(1);
  const int64_t nx = dvsq.size(2);
  const int64_t total_columns = ny * nx;
  // Evaluate each layer once on this device, with the column kernel's dtype
  // and exact swfrac expression. Per-column hbl evaluations stay in that kernel.
  auto swfrac_layers = torch::empty({nz + 2}, dvsq.options());
  occupancy_nz = nz;
  // One instantiation per input dtype.  float64 is the bitwise production
  // path; float32 exists only to measure the cost of the narrower type.  The
  // scalar parameters above stay double on the host -- that is where the eager
  // reference computes them too -- and narrow once at the launch boundary.
  AT_DISPATCH_FLOATING_TYPES(
      dvsq.scalar_type(), "kpp_bldepth_launch", [&] {
        const auto& launch_config = launch_config_cache.get(
            kpp_bldepth_kernel<scalar_t>,
            dvsq.get_device(),
            nz,
            32,
            dynamic_shared_bytes_for_block<scalar_t>);
        const int columns_per_block = launch_config.selected_block_size;
        const int64_t block_count =
            (total_columns + columns_per_block - 1) / columns_per_block;
        TORCH_CHECK(
            block_count <= std::numeric_limits<unsigned int>::max(),
            "kpp_bldepth CUDA grid exceeds grid.x capacity");
        const size_t shared_bytes =
            static_cast<size_t>(2 * (nz + 2)) * sizeof(scalar_t);
        const dim3 block(columns_per_block);
        const dim3 grid(static_cast<unsigned int>(block_count));
        constexpr int layer_threads = 128;
        swfrac_layers_kernel<scalar_t><<<
            (nz - 1 + layer_threads - 1) / layer_threads, layer_threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(
                zgrid.data_ptr<scalar_t>(), strides1(zgrid),
                swfrac_layers.data_ptr<scalar_t>(), nz,
                static_cast<scalar_t>(swfrac_rfac),
                static_cast<scalar_t>(swfrac_one_minus_rfac),
                static_cast<scalar_t>(swfrac_inv_a1),
                static_cast<scalar_t>(swfrac_inv_a2));
        kpp_bldepth_kernel<scalar_t>
            <<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                kmtj.data_ptr<int64_t>(),
                strides2(kmtj),
                dvsq.data_ptr<scalar_t>(),
                strides3(dvsq),
                dbloc.data_ptr<scalar_t>(),
                strides3(dbloc),
                ritop.data_ptr<scalar_t>(),
                strides3(ritop),
                ustar.data_ptr<scalar_t>(),
                strides2(ustar),
                bo.data_ptr<scalar_t>(),
                strides2(bo),
                bosol.data_ptr<scalar_t>(),
                strides2(bosol),
                coriol.data_ptr<scalar_t>(),
                strides2(coriol),
                zgrid.data_ptr<scalar_t>(),
                strides1(zgrid),
                swfrac_layers.data_ptr<scalar_t>(),
                hwide.data_ptr<scalar_t>(),
                strides1(hwide),
                wst.data_ptr<scalar_t>(),
                strides2(wst),
                hbl.data_ptr<scalar_t>(),
                kbl.data_ptr<int64_t>(),
                bfsfc.data_ptr<scalar_t>(),
                stable.data_ptr<scalar_t>(),
                casea.data_ptr<scalar_t>(),
                rib.data_ptr<scalar_t>(),
                nz,
                ny,
                nx,
                static_cast<scalar_t>(ricr),
                static_cast<scalar_t>(epsilon),
                static_cast<scalar_t>(vtc),
                static_cast<scalar_t>(cekman),
                static_cast<scalar_t>(cmonob),
                static_cast<scalar_t>(min_kpp_hbl),
                static_cast<scalar_t>(phepsi),
                limit_hbl_stable,
                penetrating_sw,
                sign_zero_positive,
                static_cast<scalar_t>(swfrac_rfac),
                static_cast<scalar_t>(swfrac_one_minus_rfac),
                static_cast<scalar_t>(swfrac_inv_a1),
                static_cast<scalar_t>(swfrac_inv_a2),
                static_cast<scalar_t>(inv_deltaz),
                static_cast<scalar_t>(inv_deltau),
                static_cast<scalar_t>(inv_vonk));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

template <typename T> struct BldepthView {
  const T* data;
  int64_t level,y,x;
  __device__ T get(int k,int64_t j,int64_t i) const {
    return data?data[k*level+j*y+i*x]:T(0);
  }
};
template <typename T> BldepthView<T> view(const torch::Tensor& value) {
  if(!value.defined()) return {nullptr,0,0,0};
  return {value.data_ptr<T>(),value.dim()==3?value.stride(0):0,
      value.stride(value.dim()-2),value.stride(value.dim()-1)};
}
template <typename T> struct BldepthBackward {
  BldepthView<T> input[7],seed[3],rib;
  T* output[7];
};
template <typename T>
__global__ void kpp_bldepth_backward_kernel(BldepthBackward<T> a,
    const int64_t* kmt,Strides2 ks,const T* zg,int64_t zs,const T* hw,int64_t hs,
    const T* wst,Strides2 ts,int nz,int64_t ny,int64_t nx,
    defgcm_bldepth::Constants<T> c) {
  const int64_t col=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t n=ny*nx;
  if(col>=n) return;
  const int64_t y=col/nx,x=col-y*nx;
  const auto km=kmt[y*ks.y+x*ks.x];
  int64_t kb=km;
  for(int kl=2;kl<=nz;++kl)
    if(kb==km && a.rib.get(kl-1,y,x)>c.ricr) kb=kl;
  const int rk=clamp_index(kb-1,0,nz-1),rp=clamp_index(kb-2,0,nz-1);
  const T u=a.input[3].get(0,y,x),bo=a.input[4].get(0,y,x);
  const T bosol=a.input[5].get(0,y,x),f=a.input[6].get(0,y,x);
  const T gh=a.seed[0].get(0,y,x),gb=a.seed[1].get(0,y,x);
  const auto tail=defgcm_bldepth::column_tail_vjp(km,kb,
      a.rib.get(rk,y,x),a.rib.get(rp,y,x),u,bo,bosol,f,zg,zs,nz,c,gh,gb);
  const T tail_seed[6]={tail.rk,tail.rprev,tail.ustar,tail.bo,tail.bosol,tail.coriol};
  T gu=tail_seed[2],gbo=tail_seed[3],gsol=tail_seed[4];
  // Initialize the unused surface row; each later dbloc write combines the
  // current row's lower contribution with the next row's upper contribution.
  if(a.output[0]) a.output[0][col]=T(0);
  if(a.output[2]) a.output[2][col]=T(0);
  T previous_dbloc=T(0);
  defgcm_bldepth::UstarTableSeeds<T> shared_ustar;
  for(int kl=2;kl<=nz;++kl) {
    const int k=kl-1;
    T seed=a.seed[2].get(k,y,x);
    if(k==rk) seed=seed+tail_seed[0];
    if(k==rp) seed=seed+tail_seed[1];
    const auto row=defgcm_bldepth::richardson_row_vjp(
        a.input[0].get(k,y,x),a.input[1].get(k-1,y,x),a.input[1].get(k,y,x),
        a.input[2].get(k,y,x),u,bo,bosol,zg[(kl-1)*zs],zg[kl*zs],zg[(kl+1)*zs],
        wst,ts.y,ts.x,c,seed,&shared_ustar);
    if(a.output[0]) a.output[0][k*n+col]=row.dvsq;
    if(a.output[1]) a.output[1][(k-1)*n+col]=previous_dbloc+row.dm;
    previous_dbloc=row.dc;
    if(a.output[2]) a.output[2][k*n+col]=row.ritop;
    gu=gu+row.ustar;gbo=gbo+row.bo;gsol=gsol+row.bosol;
  }
  // Match the broadcast ufrac node: sum each branch before subtracting.
  gu=gu+(shared_ustar.wa-shared_ustar.wb)*c.inv_deltau;
  if(a.output[1]) a.output[1][(nz-1)*n+col]=previous_dbloc;
  if(a.output[3]) a.output[3][col]=gu;
  if(a.output[4]) a.output[4][col]=gbo;
  if(a.output[5]) a.output[5][col]=gsol;
  if(a.output[6]) a.output[6][col]=tail_seed[5];
}

}  // namespace

// The A800 verifier disassembles this retained control kernel. With
// -fmad=false it must contain separate double multiply and add instructions and
// no DFMA instruction, proving the flag affected generated device code.
extern "C" __global__ void kpp_bldepth_fma_control_probe(
    const double* first,
    const double* second,
    const double* third,
    double* output) {
  const double product = first[0] * second[0];
  output[0] = product + third[0];
}

void kpp_bldepth_launch(
    const torch::Tensor& kmtj,
    const torch::Tensor& dvsq,
    const torch::Tensor& dbloc,
    const torch::Tensor& ritop,
    const torch::Tensor& ustar,
    const torch::Tensor& bo,
    const torch::Tensor& bosol,
    const torch::Tensor& coriol,
    const torch::Tensor& zgrid,
    const torch::Tensor& hwide,
    const torch::Tensor& wst,
    torch::Tensor& hbl,
    torch::Tensor& kbl,
    torch::Tensor& bfsfc,
    torch::Tensor& stable,
    torch::Tensor& casea,
    torch::Tensor& rib,
    const double ricr,
    const double epsilon,
    const double vtc,
    const double cekman,
    const double cmonob,
    const double min_kpp_hbl,
    const double vonk,
    const double phepsi,
    const bool limit_hbl_stable,
    const int jwtype,
    const bool penetrating_sw,
    const bool sign_zero_positive) {
  c10::cuda::CUDAGuard device_guard(dvsq.device());
  launch_variant(
      kmtj, dvsq, dbloc, ritop, ustar, bo, bosol, coriol, zgrid,
      hwide, wst, hbl, kbl, bfsfc, stable, casea, rib, ricr,
      epsilon, vtc, cekman, cmonob, min_kpp_hbl, vonk, phepsi,
      limit_hbl_stable, jwtype, penetrating_sw, sign_zero_positive);
}

const defgcm::cuda_launch::LaunchConfig&
kpp_bldepth_cuda_launch_config(int device, int nz, at::ScalarType dtype) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  occupancy_nz = nz;
  // Mirrors the AT_DISPATCH_FLOATING_TYPES in launch_variant above: the two
  // instantiations carry different register and shared-memory footprints, so
  // the exported configuration must name the one this dtype actually launches.
  // Exporting kernel<double> unconditionally made a float32 run's launch
  // ledger report a resource basis it never used.
  if (dtype == at::kFloat) {
    return launch_config_cache.get(
        kpp_bldepth_kernel<float>,
        device,
        nz,
        32,
        dynamic_shared_bytes_for_block<float>);
  }
  TORCH_CHECK(
      dtype == at::kDouble,
      "kpp_bldepth CUDA launch configuration supports float64 or float32 only");
  return launch_config_cache.get(
      kpp_bldepth_kernel<double>,
      device,
      nz,
      32,
      dynamic_shared_bytes_for_block<double>);
}

void kpp_bldepth_backward_launch(const std::vector<torch::Tensor>& values,
    const torch::Tensor& rib,const std::vector<torch::Tensor>& seeds,
    std::vector<torch::Tensor>& outputs,const std::vector<double>& p,
    bool limit,int jwtype,bool penetrating_sw,bool sign_zero_positive) {
  c10::cuda::CUDAGuard guard(values[1].device());
  const auto nz=values[1].size(0),ny=values[1].size(1),nx=values[1].size(2);
  AT_DISPATCH_FLOATING_TYPES(values[1].scalar_type(),"kpp_bldepth_backward",[&] {
    BldepthBackward<scalar_t> args;
    for(int i=0;i<7;++i) {
      args.input[i]=view<scalar_t>(values[i+1]);
      args.output[i]=outputs[i].defined()?outputs[i].data_ptr<scalar_t>():nullptr;
    }
    for(int i=0;i<3;++i) args.seed[i]=view<scalar_t>(seeds[i]);
    args.rib=view<scalar_t>(rib);
    const auto c=defgcm_bldepth::constants<scalar_t>(
        p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],limit,jwtype,
        penetrating_sw,sign_zero_positive);
    kpp_bldepth_backward_kernel<scalar_t><<<(ny*nx+127)/128,128,0,
        at::cuda::getCurrentCUDAStream()>>>(args,values[0].data_ptr<int64_t>(),
        strides2(values[0]),values[8].data_ptr<scalar_t>(),values[8].stride(0),
        values[9].data_ptr<scalar_t>(),values[9].stride(0),
        values[11].data_ptr<scalar_t>(),strides2(values[11]),nz,ny,nx,c);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}
