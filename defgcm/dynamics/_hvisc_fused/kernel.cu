// Fused kernels for the HorizontalViscosity Leith/Smag dissipation phase.
//
// Coefficient assembly and the dissipation stencil are separated by the
// all-faces hDiv_flux halo exchange. Every exchange remains outside the kernels.
//   stage 1: u,v and pre-haloed u_ext/v_dyC_ext/dxC_ext ->
//            viscAh_D, hDiv_flux, hz_v3
//   stage 2: hDiv_flux, its optional halos and hz_v3 -> u_result, v_result
//
// Expression association follows defgcm/dynamics/horizontal_diffusion.py.
// Compilation uses -fmad=false to preserve separate multiply/add rounding.
// Arithmetic stays in the storage type, retaining the float32 or float64
// reference order. Preserve the parenthesisation of every expression.
//
// Divisions here correspond to tensor/tensor divisions in the eager reference
// and use -prec-div=true. Scalar-numerator reciprocals such as 1.0/rAz are
// computed by the adapter at the same point as in the reference implementation.

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace {

// Accessors require unit stride on the innermost axis and store only the
// remaining strides. This removes one index multiplication per access.
// The adapter's unit_row materialises non-contiguous innermost axes, and the
// accessor factories check this precondition before passing pointers to CUDA.
template <typename T>
struct A4 {
  const T *__restrict__ p;
  int64_t sb, sk, sj;
  __device__ __forceinline__ T at(int64_t b, int64_t k, int64_t j,
                                  int64_t i) const {
    return p[b * sb + k * sk + j * sj + i];
  }
};

template <typename T>
struct A3 {
  const T *__restrict__ p;
  int64_t sk, sj;
  __device__ __forceinline__ T at(int64_t k, int64_t j, int64_t i) const {
    return p[k * sk + j * sj + i];
  }
};

template <typename T>
struct A2 {
  const T *__restrict__ p;
  int64_t sj;
  __device__ __forceinline__ T at(int64_t j, int64_t i) const {
    return p[j * sj + i];
  }
};

// A null pointer means "mask absent", which the eager reference spells as
// ``center_mask is None`` / ``grid.masks.maskIn* is None`` -- i.e. the multiply
// is skipped entirely, NOT performed against ones (skipping and multiplying by
// 1.0 differ on a signed zero and on a NaN payload).
template <typename T>
struct M2 {
  const T *__restrict__ p;
  int64_t sj;
  __device__ __forceinline__ bool present() const { return p != nullptr; }
  __device__ __forceinline__ T at(int64_t j, int64_t i) const {
    return p[j * sj + i];
  }
};

// ``torch.maximum`` semantics: NaN propagates from either operand, and on a
// tie the FIRST argument is returned (so ``maximum(-0.0, +0.0) == -0.0``).
// CUDA ``fmax`` has neither property.
template <typename T>
__device__ __forceinline__ T torch_maximum(const T first, const T second) {
  if (isnan(first)) {
    return first;
  }
  if (isnan(second)) {
    return second;
  }
  return first >= second ? first : second;
}

// ---------------------------------------------------------------------------
// coefficient assembly
// ---------------------------------------------------------------------------

template <typename T>
__global__ void hvisc_stage1_kernel(
    A4<T> u, A4<T> v, A4<T> u_ext, A4<T> v_dyC_ext, A2<T> dxC_ext,
    A2<T> dxG_ext, A2<T> dyG, A2<T> dyF, A2<T> recip_rA, A2<T> recip_rAz,
    A3<T> hFacW, A3<T> hFacS, A3<T> recip_hFacC, A3<T> maskW, A3<T> maskS,
    A3<T> hfacz, M2<T> maskInC, M2<T> maskInW, M2<T> maskInS, A2<T> l2_d,
    A2<T> l3_d, A2<T> l2_z, A2<T> l3_z, const T *__restrict__ leith2fac_p,
    const T *__restrict__ smag2fac_p, const T *__restrict__ base_p,
    T side_factor, T *__restrict__ viscAh_D_out,
    T *__restrict__ hDiv_flux_out, T *__restrict__ hz_v3_out, int64_t nz,
    int64_t ny, int64_t nx, int64_t total) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }

  const int64_t corner_x = nx + 1;
  const int64_t corner_y = ny + 1;
  const int64_t i = tid % corner_x;
  int64_t rest = tid / corner_x;
  const int64_t j = rest % corner_y;
  rest /= corner_y;
  const int64_t k = rest % nz;
  const int64_t b = rest / nz;

  const T zero = static_cast<T>(0);
  const T quarter = static_cast<T>(0.25);
  const T tiny = static_cast<T>(1e-30);
  const T leith2fac = *leith2fac_p;
  const T smag2fac = *smag2fac_p;
  const T base = *base_p;

  // -- vort3 / strain at a corner (horizontal_diffusion.py:613-627 fed by the
  //    :2305/:2327 exchange products; u_dxC_ext and v_dyC are folded in).
  auto curl_pair = [&](int64_t jj, int64_t ii, T &vort, T &strn) {
    const T curl_u = u_ext.at(b, k, jj + 1, ii) * dxC_ext.at(jj + 1, ii) -
                     u_ext.at(b, k, jj, ii) * dxC_ext.at(jj, ii);
    const T curl_v =
        v_dyC_ext.at(b, k, jj, ii + 1) - v_dyC_ext.at(b, k, jj, ii);
    const T r = recip_rAz.at(jj, ii);
    vort = r * (curl_v - curl_u);
    strn = r * (curl_v + curl_u);
  };

  // -- vort3BC / strainBC (:1280-1291).  ``side_factor`` is 0.0 in this
  //    kernel's applicability domain, but ``value * 0.0`` is NOT the literal
  //    +0.0 for a negative value, so the multiply is written out.
  auto vort3_bc = [&](int64_t jj, int64_t ii) -> T {
    T vort, strn;
    curl_pair(jj, ii, vort, strn);
    return hfacz.at(k, jj, ii) > zero ? vort : vort * side_factor;
  };
  auto strain_bc = [&](int64_t jj, int64_t ii) -> T {
    T vort, strn;
    curl_pair(jj, ii, vort, strn);
    return hfacz.at(k, jj, ii) > zero ? strn : strn * side_factor;
  };

  // -- tension at a centre (:630-646).
  auto tension_at = [&](int64_t jj, int64_t ii) -> T {
    const T tension_u =
        dyF.at(jj, ii) * (u.at(b, k, jj, ii + 1) - u.at(b, k, jj, ii));
    const T tension_v = v.at(b, k, jj + 1, ii) * dxG_ext.at(jj + 1, ii) -
                        v.at(b, k, jj, ii) * dxG_ext.at(jj, ii);
    T out = (tension_u - tension_v) * recip_rA.at(jj, ii);
    if (maskInC.present()) {
      out = out * maskInC.at(jj, ii);
    }
    return out;
  };
  // Zero-padded reads of ``tension`` outside the centre grid: the eager Z-point
  // Smag magnitude builds t00/tjm/tim/timjm as zeros and slice-assigns
  // (:1004-1016), so an out-of-grid neighbour contributes a literal +0.0.
  auto tension_pad = [&](int64_t jj, int64_t ii) -> T {
    if (jj < 0 || jj >= ny || ii < 0 || ii >= nx) {
      return zero;
    }
    return tension_at(jj, ii);
  };

  // -- vrtDx / vrtDy (:716-744; masks precombined with maskIn* at :2914-2925).
  //    ``vrtDx`` is written only on columns < nx and ``vrtDy`` only on rows
  //    < ny (the slice targets), so the complement stays the zeros the eager
  //    ``torch.zeros_like`` supplied.
  auto vrtDx_at = [&](int64_t jj, int64_t ii) -> T {
    if (ii < 0 || ii >= nx || jj < 0 || jj > ny) {
      return zero;
    }
    T mask = maskS.at(k, jj, ii);
    if (maskInS.present()) {
      mask = mask * maskInS.at(jj, ii);
    }
    return (vort3_bc(jj, ii + 1) - vort3_bc(jj, ii)) / dxG_ext.at(jj, ii) *
           mask;
  };
  auto vrtDy_at = [&](int64_t jj, int64_t ii) -> T {
    if (jj < 0 || jj >= ny || ii < 0 || ii > nx) {
      return zero;
    }
    T mask = maskW.at(k, jj, ii);
    if (maskInW.present()) {
      mask = mask * maskInW.at(jj, ii);
    }
    return (vort3_bc(jj + 1, ii) - vort3_bc(jj, ii)) / dyG.at(jj, ii) * mask;
  };

  auto max_abs4 = [&](T a, T c, T d, T e) -> T {
    // ``torch.maximum`` chain, three calls, in the eager order (:775-778).
    // CUDA ``fmax`` returns the non-NaN operand and does not fix the tie
    // selection; ``torch.maximum`` propagates NaN and returns its FIRST
    // argument on a tie (measured: maximum(-0.0, 0.0) == -0.0), so the rule is
    // spelled out.  Every operand here is an ``abs`` result, so the signed-zero
    // half is unreachable in practice -- it costs nothing to be exact.
    T m = torch_maximum(fabs(a), fabs(c));
    m = torch_maximum(m, fabs(d));
    m = torch_maximum(m, fabs(e));
    return m;
  };

  // ---- Z point (corner), every (j, i) ---------------------------------
  {
    T vort_z, strain_z;
    curl_pair(j, i, vort_z, strain_z);
    const bool wet_z = hfacz.at(k, j, i) > zero;
    const T strain_z_bc = wet_z ? strain_z : strain_z * side_factor;

    // vrt_z_max (:850-856): vrtDx(j,i), vrtDx(j,i-1), vrtDy(j,i), vrtDy(j-1,i)
    // where the shifted copies are zero-padded at i==0 / j==0.
    const T vrt_z_max = max_abs4(vrtDx_at(j, i), vrtDx_at(j, i - 1),
                                 vrtDy_at(j, i), vrtDy_at(j - 1, i));

    // smag_z (:1001-1020): tension 4-average at the surrounding centres.
    const T t00 = tension_pad(j, i);
    const T tjm = tension_pad(j - 1, i);
    const T tim = tension_pad(j, i - 1);
    const T timjm = tension_pad(j - 1, i - 1);
    const T tension_sq_z =
        quarter * (((t00 * t00 + tjm * tjm) + tim * tim) + timjm * timjm);
    const T smag_z = sqrt((strain_z_bc * strain_z_bc + tension_sq_z) + tiny);

    // viscAh_Z (:2708-2724 legacy max-norm branch) + base (:2012-2016).
    T viscAh_Z = (leith2fac * vrt_z_max) * l3_z.at(j, i) +
                 (smag2fac * smag_z) * l2_z.at(j, i);
    viscAh_Z = viscAh_Z + base;

    // hz_v3 = (hFacZ * vort3) * viscAh_Z (:1273) -- RAW vort3, not vort3BC.
    hz_v3_out[((b * nz + k) * (ny + 1) + j) * (nx + 1) + i] =
        hfacz.at(k, j, i) * vort_z * viscAh_Z;
  }

  // ---- D point (centre), j < ny and i < nx ------------------------------
  if (j < ny && i < nx) {
    // vrt_d_max (:790-793).
    const T vrt_d_max = max_abs4(vrtDx_at(j, i), vrtDx_at(j + 1, i),
                                 vrtDy_at(j, i), vrtDy_at(j, i + 1));

    // smag_d (:987-998): strain 4-average at the surrounding corners.
    const T s_je_ie = strain_bc(j, i + 1);
    const T s_jn_iw = strain_bc(j + 1, i);
    const T s_js_iw = strain_bc(j, i);
    const T s_jn_ie = strain_bc(j + 1, i + 1);
    const T strain_sq_d =
        quarter * (((s_je_ie * s_je_ie + s_jn_iw * s_jn_iw) +
                    s_js_iw * s_js_iw) +
                   s_jn_ie * s_jn_ie);
    const T tension_d = tension_at(j, i);
    const T smag_d = sqrt((tension_d * tension_d + strain_sq_d) + tiny);

    T viscAh_D = (leith2fac * vrt_d_max) * l3_d.at(j, i) +
                 (smag2fac * smag_d) * l2_d.at(j, i);
    viscAh_D = viscAh_D + base;

    // hDiv (:585-610) -- computed here so hDiv_flux costs one extra store, not
    // a whole extra field read.
    const T u_flux_e =
        (u.at(b, k, j, i + 1) * dyG.at(j, i + 1)) * hFacW.at(k, j, i + 1);
    const T u_flux_w = (u.at(b, k, j, i) * dyG.at(j, i)) * hFacW.at(k, j, i);
    const T v_flux_n =
        (v.at(b, k, j + 1, i) * dxG_ext.at(j + 1, i)) * hFacS.at(k, j + 1, i);
    const T v_flux_s = (v.at(b, k, j, i) * dxG_ext.at(j, i)) * hFacS.at(k, j, i);
    T hdiv = ((u_flux_e - u_flux_w) + (v_flux_n - v_flux_s)) *
             recip_rA.at(j, i) * recip_hFacC.at(k, j, i);
    if (maskInC.present()) {
      hdiv = hdiv * maskInC.at(j, i);
    }

    const int64_t out = ((b * nz + k) * ny + j) * nx + i;
    viscAh_D_out[out] = viscAh_D;
    hDiv_flux_out[out] = hdiv * viscAh_D;  // :1272
  }
}

// ---------------------------------------------------------------------------
// div-vort dissipation stencil
// ---------------------------------------------------------------------------

template <typename T>
__global__ void hvisc_stage2_kernel(
    A4<T> hDiv_flux, A4<T> hDiv_x, A4<T> hDiv_y, A4<T> hz_v3, A2<T> dxC_row,
    A2<T> dyC, A2<T> dxG_ext, A2<T> dyG, A3<T> recip_hFacW, A3<T> recip_hFacS,
    A3<T> maskW, A3<T> maskS, bool u_all_faces, bool v_all_faces, bool keep_west,
    bool keep_east, bool keep_south, bool keep_north, T *__restrict__ u_out,
    T *__restrict__ v_out, int64_t nz, int64_t ny, int64_t nx, int64_t total) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }

  const int64_t corner_x = nx + 1;
  const int64_t corner_y = ny + 1;
  const int64_t i = tid % corner_x;
  int64_t rest = tid / corner_x;
  const int64_t j = rest % corner_y;
  rest /= corner_y;
  const int64_t k = rest % nz;
  const int64_t b = rest / nz;

  const T zero = static_cast<T>(0);

  // ---- U face (j, i), j in [0, ny-1], i in [0, nx] ---------------------
  if (j < ny) {
    const bool interior = (i >= 1 && i <= nx - 1);
    const bool keep = u_all_faces ? (interior || (i == 0 && keep_west) ||
                                     (i == nx && keep_east))
                                  : interior;
    T value = zero;
    if (keep) {
      // dHdx: interior reads hDiv_flux directly (:1310-1313), the all-faces
      // branch reads its 1-cell x halo (:1373-1375).  Same expression, same
      // dxC_row(j, i) denominator.
      const T hd_e = u_all_faces ? hDiv_x.at(b, k, j, i + 1)
                                 : hDiv_flux.at(b, k, j, i);
      const T hd_w =
          u_all_faces ? hDiv_x.at(b, k, j, i) : hDiv_flux.at(b, k, j, i - 1);
      const T dHdx = (hd_e - hd_w) / dxC_row.at(j, i);
      const T dZdy =
          (hz_v3.at(b, k, j + 1, i) - hz_v3.at(b, k, j, i)) / dyG.at(j, i);
      value = (dHdx - recip_hFacW.at(k, j, i) * dZdy) * maskW.at(k, j, i);
    }
    u_out[((b * nz + k) * ny + j) * (nx + 1) + i] = value;
  }

  // ---- V face (j, i), j in [0, ny], i in [0, nx-1] ---------------------
  if (i < nx) {
    const bool interior = (j >= 1 && j <= ny - 1);
    const bool keep = v_all_faces ? (interior || (j == 0 && keep_south) ||
                                     (j == ny && keep_north))
                                  : interior;
    T value = zero;
    if (keep) {
      const T hd_n = v_all_faces ? hDiv_y.at(b, k, j + 1, i)
                                 : hDiv_flux.at(b, k, j, i);
      const T hd_s =
          v_all_faces ? hDiv_y.at(b, k, j, i) : hDiv_flux.at(b, k, j - 1, i);
      const T dHdy = (hd_n - hd_s) / dyC.at(j, i);
      const T dZdx =
          (hz_v3.at(b, k, j, i + 1) - hz_v3.at(b, k, j, i)) / dxG_ext.at(j, i);
      value = (recip_hFacS.at(k, j, i) * dZdx + dHdy) * maskS.at(k, j, i);
    }
    v_out[((b * nz + k) * (ny + 1) + j) * nx + i] = value;
  }
}

template <typename T>
A4<T> acc4(const torch::Tensor &t) {
  TORCH_CHECK(t.stride(3) == 1, "hvisc fused accessor needs a unit row stride");
  return A4<T>{t.data_ptr<T>(), t.stride(0), t.stride(1), t.stride(2)};
}

template <typename T>
A4<T> acc4_opt(const std::optional<torch::Tensor> &t) {
  if (!t.has_value()) {
    return A4<T>{nullptr, 0, 0, 0};
  }
  return acc4<T>(*t);
}

template <typename T>
A3<T> acc3(const torch::Tensor &t) {
  TORCH_CHECK(t.stride(2) == 1, "hvisc fused accessor needs a unit row stride");
  return A3<T>{t.data_ptr<T>(), t.stride(0), t.stride(1)};
}

template <typename T>
A2<T> acc2(const torch::Tensor &t) {
  const int64_t dim = t.dim();
  TORCH_CHECK(t.stride(dim - 1) == 1,
              "hvisc fused accessor needs a unit row stride");
  return A2<T>{t.data_ptr<T>(), t.stride(dim - 2)};
}

template <typename T>
M2<T> accm(const std::optional<torch::Tensor> &t) {
  if (!t.has_value()) {
    return M2<T>{nullptr, 0};
  }
  const int64_t dim = t->dim();
  TORCH_CHECK(t->stride(dim - 1) == 1,
              "hvisc fused accessor needs a unit row stride");
  return M2<T>{t->data_ptr<T>(), t->stride(dim - 2)};
}

} // namespace

void hvisc_stage1_launch(
    const torch::Tensor &u, const torch::Tensor &v, const torch::Tensor &u_ext,
    const torch::Tensor &v_dyC_ext, const torch::Tensor &dxC_ext,
    const torch::Tensor &dxG_ext, const torch::Tensor &dyG,
    const torch::Tensor &dyF, const torch::Tensor &recip_rA,
    const torch::Tensor &recip_rAz, const torch::Tensor &hFacW,
    const torch::Tensor &hFacS, const torch::Tensor &recip_hFacC,
    const torch::Tensor &maskW, const torch::Tensor &maskS,
    const torch::Tensor &hfacz, const std::optional<torch::Tensor> &maskInC,
    const std::optional<torch::Tensor> &maskInW,
    const std::optional<torch::Tensor> &maskInS, const torch::Tensor &l2_d,
    const torch::Tensor &l3_d, const torch::Tensor &l2_z,
    const torch::Tensor &l3_z, const torch::Tensor &leith2fac,
    const torch::Tensor &smag2fac, const torch::Tensor &base,
    double side_mask_fac, torch::Tensor &viscAh_D, torch::Tensor &hDiv_flux,
    torch::Tensor &hz_v3) {
  const c10::cuda::CUDAGuard device_guard(u.device());

  const int64_t batch = viscAh_D.size(0);
  const int64_t nz = viscAh_D.size(1);
  const int64_t ny = viscAh_D.size(2);
  const int64_t nx = viscAh_D.size(3);
  const int64_t total = batch * nz * (ny + 1) * (nx + 1);

  // 128, not 256: blocks are resident whole, so with ~100 registers per
  // thread 4-warp blocks pack the register file more tightly than 8-warp
  // blocks (at 96 registers, 20 vs 16 warps; at the A800's measured 106,
  // 2 blocks x 4 warps).  Block size alone buys nothing (15.45 -> 15.19 ms);
  // the gain comes with the register drop above.  This kernel is latency-bound
  // -- measured DRAM at 17.7% and SM at 32.7% of peak -- so resident warps,
  // not bandwidth or FLOPs, are what it is short of.  The stage-2 launch
  // keeps 256: at 32 registers its occupancy is already block-limited.
  constexpr int threads = 128;
  const int64_t blocks = (total + threads - 1) / threads;
  TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(),
              "hvisc fused stage 1 grid exceeds grid.x capacity");

  AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "hvisc_stage1_launch", [&] {
    hvisc_stage1_kernel<scalar_t>
        <<<static_cast<unsigned int>(blocks), threads, 0,
           at::cuda::getCurrentCUDAStream()>>>(
            acc4<scalar_t>(u), acc4<scalar_t>(v), acc4<scalar_t>(u_ext),
            acc4<scalar_t>(v_dyC_ext), acc2<scalar_t>(dxC_ext),
            acc2<scalar_t>(dxG_ext), acc2<scalar_t>(dyG), acc2<scalar_t>(dyF),
            acc2<scalar_t>(recip_rA), acc2<scalar_t>(recip_rAz),
            acc3<scalar_t>(hFacW), acc3<scalar_t>(hFacS),
            acc3<scalar_t>(recip_hFacC), acc3<scalar_t>(maskW),
            acc3<scalar_t>(maskS), acc3<scalar_t>(hfacz),
            accm<scalar_t>(maskInC), accm<scalar_t>(maskInW),
            accm<scalar_t>(maskInS), acc2<scalar_t>(l2_d), acc2<scalar_t>(l3_d),
            acc2<scalar_t>(l2_z), acc2<scalar_t>(l3_z),
            leith2fac.data_ptr<scalar_t>(), smag2fac.data_ptr<scalar_t>(),
            base.data_ptr<scalar_t>(), static_cast<scalar_t>(side_mask_fac),
            viscAh_D.data_ptr<scalar_t>(), hDiv_flux.data_ptr<scalar_t>(),
            hz_v3.data_ptr<scalar_t>(), nz, ny, nx, total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hvisc_stage2_launch(
    const torch::Tensor &hDiv_flux, const std::optional<torch::Tensor> &hDiv_x,
    const std::optional<torch::Tensor> &hDiv_y, const torch::Tensor &hz_v3,
    const torch::Tensor &dxC_row, const torch::Tensor &dyC,
    const torch::Tensor &dxG_ext, const torch::Tensor &dyG,
    const torch::Tensor &recip_hFacW, const torch::Tensor &recip_hFacS,
    const torch::Tensor &maskW, const torch::Tensor &maskS, bool u_all_faces,
    bool v_all_faces, bool keep_west, bool keep_east, bool keep_south,
    bool keep_north, torch::Tensor &u_out, torch::Tensor &v_out) {
  const c10::cuda::CUDAGuard device_guard(hDiv_flux.device());

  const int64_t batch = hDiv_flux.size(0);
  const int64_t nz = hDiv_flux.size(1);
  const int64_t ny = hDiv_flux.size(2);
  const int64_t nx = hDiv_flux.size(3);
  const int64_t total = batch * nz * (ny + 1) * (nx + 1);

  constexpr int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(),
              "hvisc fused stage 2 grid exceeds grid.x capacity");

  AT_DISPATCH_FLOATING_TYPES(
      hDiv_flux.scalar_type(), "hvisc_stage2_launch", [&] {
        hvisc_stage2_kernel<scalar_t>
            <<<static_cast<unsigned int>(blocks), threads, 0,
               at::cuda::getCurrentCUDAStream()>>>(
                acc4<scalar_t>(hDiv_flux), acc4_opt<scalar_t>(hDiv_x),
                acc4_opt<scalar_t>(hDiv_y), acc4<scalar_t>(hz_v3),
                acc2<scalar_t>(dxC_row), acc2<scalar_t>(dyC),
                acc2<scalar_t>(dxG_ext), acc2<scalar_t>(dyG),
                acc3<scalar_t>(recip_hFacW), acc3<scalar_t>(recip_hFacS),
                acc3<scalar_t>(maskW), acc3<scalar_t>(maskS), u_all_faces,
                v_all_faces, keep_west, keep_east, keep_south, keep_north,
                u_out.data_ptr<scalar_t>(), v_out.data_ptr<scalar_t>(), nz, ny,
                nx, total);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
