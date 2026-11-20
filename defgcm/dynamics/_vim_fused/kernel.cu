// Horizontal-phase fused kernel for vector-invariant momentum advection.
//
// One launch turns the pre-haloed velocity/KE fields into (tend_u, tend_v):
// corner curl -> hFacZ mask -> KE gradients -> planetary Coriolis -> relative
// vorticity flux -> three-term combine.  Every input is read straight from HBM
// and each output element is written exactly once.
//
// Expression association follows the eager reference verbatim
// (defgcm/dynamics/vector_invariant_momentum.py and
// defgcm/operators/c_grid.py::curl_uv_to_corner); this translation unit is
// built with -fmad=false so the transcription stays bitwise.  Arithmetic is
// carried in the STORAGE type, not promoted to double: this stencil has no
// accumulation depth, so keeping T makes both the f64 and the f32 arm bitwise
// comparable against their own eager reference.  Do not "simplify" any
// expression here -- the parenthesisation is load-bearing.

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

template <typename T>
struct A4 {
  const T *__restrict__ p;
  int64_t sb, sk, sj, si;
  __device__ __forceinline__ T at(int64_t b, int64_t k, int64_t j,
                                  int64_t i) const {
    return p[b * sb + k * sk + j * sj + i * si];
  }
};

template <typename T>
struct A3 {
  const T *__restrict__ p;
  int64_t sk, sj, si;
  __device__ __forceinline__ T at(int64_t k, int64_t j, int64_t i) const {
    return p[k * sk + j * sj + i * si];
  }
};

template <typename T>
struct A2 {
  const T *__restrict__ p;
  int64_t sj, si;
  __device__ __forceinline__ T at(int64_t j, int64_t i) const {
    return p[j * sj + i * si];
  }
};

template <typename T>
struct A1 {
  const T *__restrict__ p;
  int64_t sj;
  __device__ __forceinline__ T at(int64_t j) const { return p[j * sj]; }
};

// A null pointer means "no mask supplied", which the eager reference spells as
// ``edge_pair_mask is None`` -- i.e. every face is kept.
struct B1 {
  const bool *__restrict__ p;
  int64_t sj;
  __device__ __forceinline__ bool at(int64_t j) const {
    return p == nullptr ? true : p[j * sj];
  }
};

template <typename T>
__global__ void vim_fused_kernel(
    A4<T> ke_x, A4<T> ke_y, A4<T> u_corner, A4<T> v_corner, A4<T> v_x_halo,
    A4<T> u_y_halo, A3<T> hfacz, A2<T> fg, A2<T> dxC_ext, A2<T> dyC, A2<T> rAz,
    A1<T> dxG_s, A1<T> dxG_n, A1<T> recip_dxC, const T *__restrict__ dyG_p,
    const T *__restrict__ recip_dyC_p, A3<T> maskW, A3<T> maskS, B1 xcm, B1 ycm,
    B1 edge_x, B1 edge_y, bool px_copy, bool py_copy, T *__restrict__ tend_u,
    T *__restrict__ tend_v, int64_t nz, int64_t ny, int64_t nx,
    int64_t total) {
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
  const T half = static_cast<T>(0.5);
  const T quarter = static_cast<T>(0.25);
  const T one = static_cast<T>(1);
  const T hfac_eps = static_cast<T>(1e-9);

  // --- corner relative vorticity, MIT circulation/area form -------------
  // c_grid.py:816-822 (interior) and the seven edge/corner candidates below
  // it collapse to ONE expression once the halo tensors carry the wrapped
  // neighbours; the dyC column index is the only place they differ.
  auto zeta_raw = [&](int64_t jj, int64_t ii) -> T {
    const T recip_rAz = one / rAz.at(jj, ii);
    const int64_t idy = ii < nx ? ii : (nx - 1);
    const T delta_v = dyC.at(jj, idy) * (v_corner.at(b, k, jj + 1, ii + 1) -
                                         v_corner.at(b, k, jj + 1, ii));
    const T delta_u = u_corner.at(b, k, jj + 1, ii + 1) * dxC_ext.at(jj + 1, ii) -
                      u_corner.at(b, k, jj, ii + 1) * dxC_ext.at(jj, ii);
    return recip_rAz * (delta_v - delta_u);
  };

  // Support masking (c_grid.py:905-935, :953-980, :1010-1030) plus the
  // periodic-seam copy (:1042-1046, x before y) and the hFacZ corner mask
  // (vector_invariant_momentum.py:547-551).  The seam copy is expressed as an
  // index remap of the SOURCE corner; hFacZ is read at the TARGET corner,
  // exactly as the eager elementwise ``torch.where`` does.
  auto zeta_masked = [&](int64_t jj, int64_t ii) -> T {
    const int64_t is = (px_copy && ii == nx) ? 0 : ii;
    const int64_t js = (py_copy && jj == ny) ? 0 : jj;
    const T value = (xcm.at(is) && ycm.at(js)) ? zeta_raw(js, is) : zero;
    return hfacz.at(k, jj, ii) > zero ? value : zero;
  };

  const T zeta_here = zeta_masked(j, i);

  if (j < ny) {
    // ---- U face (j, i), j in [0, ny-1], i in [0, nx] -------------------
    const T mask_w = maskW.at(k, j, i);

    const T v_sr = v_x_halo.at(b, k, j, i + 1);
    const T v_sl = v_x_halo.at(b, k, j, i);
    const T v_nr = v_x_halo.at(b, k, j + 1, i + 1);
    const T v_nl = v_x_halo.at(b, k, j + 1, i);
    const T dgs = dxG_s.at(j);
    const T dgn = dxG_n.at(j);
    const T rdx = recip_dxC.at(j);
    const bool keep_x = edge_x.at(i);

    // Planetary branch (vector_invariant_momentum.py:1087-1099 with
    // hFacZ=None, then :1063-1069).
    const T f_avg = half * (fg.at(j, i) + fg.at(j + 1, i));
    const T vbar_p = quarter * (dgs * (v_sr + v_sl) + dgn * (v_nr + v_nl));
    T plan_u = (f_avg * vbar_p) * rdx;
    plan_u = keep_x ? plan_u : zero;
    plan_u = plan_u * mask_w;

    // Relative-vorticity branch (:1100-1118 hFacZ present, then :1063-1069).
    const T hz_s = hfacz.at(k, j, i);
    const T hz_n = hfacz.at(k, j + 1, i);
    const T w_s = (hz_s * dgs) * (v_sr + v_sl);
    const T w_n = (hz_n * dgn) * (v_nr + v_nl);
    const T den_u = hz_s + hz_n;
    const T vbar_z =
        (half * (w_s + w_n)) / (den_u < hfac_eps ? hfac_eps : den_u);
    const T zeta_north = zeta_masked(j + 1, i);
    const T z_avg = half * (zeta_here + zeta_north);
    T zeta_flux_u = (z_avg * vbar_z) * rdx;
    zeta_flux_u = keep_x ? zeta_flux_u : zero;
    zeta_flux_u = zeta_flux_u * mask_w;

    // KE gradient (:451-458 then the maskW multiply at :440-444).
    T dkedx = (ke_x.at(b, k, j, i) - ke_x.at(b, k, j, i + 1)) /
              dxC_ext.at(j + 1, i);
    dkedx = dkedx * mask_w;

    // Three-term combine (:571, left-to-right).
    tend_u[((b * nz + k) * ny + j) * (nx + 1) + i] =
        (dkedx + plan_u) + zeta_flux_u;
  }

  if (i < nx) {
    // ---- V face (j, i), j in [0, ny], i in [0, nx-1] -------------------
    const T mask_s = maskS.at(k, j, i);

    const T u_la = u_y_halo.at(b, k, j + 1, i);
    const T u_lb = u_y_halo.at(b, k, j, i);
    const T u_ra = u_y_halo.at(b, k, j + 1, i + 1);
    const T u_rb = u_y_halo.at(b, k, j, i + 1);
    const bool keep_y = edge_y.at(j);

    // Planetary branch (:1234-1241 with hFacZ=None, then :1145-1152).
    const T f_avg = half * (fg.at(j, i) + fg.at(j, i + 1));
    const T ubar_p = (quarter * (*dyG_p)) * (((u_la + u_lb) + u_ra) + u_rb);
    T plan_v = ((-f_avg) * ubar_p) * (*recip_dyC_p);
    plan_v = keep_y ? plan_v : zero;
    plan_v = plan_v * mask_s;

    // Relative-vorticity branch (:1242-1258).  ``recip_metric`` is None on
    // this branch (:1146), so there is deliberately no reciprocal multiply.
    const T hz_l = hfacz.at(k, j, i);
    const T hz_r = hfacz.at(k, j, i + 1);
    const T w_l = hz_l * (u_la + u_lb);
    const T w_r = hz_r * (u_ra + u_rb);
    const T den_v = hz_l + hz_r;
    const T ubar_z =
        (half * (w_l + w_r)) / (den_v < hfac_eps ? hfac_eps : den_v);
    const T zeta_east = zeta_masked(j, i + 1);
    const T z_avg = half * (zeta_here + zeta_east);
    T zeta_flux_v = (-z_avg) * ubar_z;
    zeta_flux_v = keep_y ? zeta_flux_v : zero;
    zeta_flux_v = zeta_flux_v * mask_s;

    // KE gradient (:509-516 then the maskS multiply at :500-504).
    T dkedy = (ke_y.at(b, k, j, i) - ke_y.at(b, k, j + 1, i)) / dyC.at(j, i);
    dkedy = dkedy * mask_s;

    tend_v[((b * nz + k) * (ny + 1) + j) * nx + i] =
        (dkedy + plan_v) + zeta_flux_v;
  }
}

template <typename T>
A4<T> acc4(const torch::Tensor &t) {
  return A4<T>{t.data_ptr<T>(), t.stride(0), t.stride(1), t.stride(2),
               t.stride(3)};
}

template <typename T>
A3<T> acc3(const torch::Tensor &t) {
  return A3<T>{t.data_ptr<T>(), t.stride(0), t.stride(1), t.stride(2)};
}

template <typename T>
A2<T> acc2(const torch::Tensor &t) {
  return A2<T>{t.data_ptr<T>(), t.stride(0), t.stride(1)};
}

template <typename T>
A1<T> acc1(const torch::Tensor &t) {
  return A1<T>{t.data_ptr<T>(), t.stride(0)};
}

B1 accb(const std::optional<torch::Tensor> &t) {
  if (!t.has_value()) {
    return B1{nullptr, 0};
  }
  return B1{t->data_ptr<bool>(), t->stride(0)};
}

B1 accb_required(const torch::Tensor &t) {
  return B1{t.data_ptr<bool>(), t.stride(0)};
}

} // namespace

void vim_fused_launch(const torch::Tensor &ke_x_ext,
                      const torch::Tensor &ke_y_ext,
                      const torch::Tensor &u_corner,
                      const torch::Tensor &v_corner,
                      const torch::Tensor &v_x_halo,
                      const torch::Tensor &u_y_halo, const torch::Tensor &hfacz,
                      const torch::Tensor &fg, const torch::Tensor &dxC_ext,
                      const torch::Tensor &dyC, const torch::Tensor &rAz,
                      const torch::Tensor &dxG_south,
                      const torch::Tensor &dxG_north,
                      const torch::Tensor &recip_dxC_u, const torch::Tensor &dyG,
                      const torch::Tensor &recip_dyC, const torch::Tensor &maskW,
                      const torch::Tensor &maskS,
                      const torch::Tensor &x_corner_mask,
                      const torch::Tensor &y_corner_mask,
                      const std::optional<torch::Tensor> &edge_pair_mask_x,
                      const std::optional<torch::Tensor> &edge_pair_mask_y,
                      bool periodic_x_copy, bool periodic_y_copy,
                      torch::Tensor &tend_u, torch::Tensor &tend_v) {
  const c10::cuda::CUDAGuard device_guard(ke_x_ext.device());

  const int64_t batch = tend_u.size(0);
  const int64_t nz = tend_u.size(1);
  const int64_t ny = tend_u.size(2);
  const int64_t nx = tend_u.size(3) - 1;
  const int64_t total = batch * nz * (ny + 1) * (nx + 1);

  constexpr int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(),
              "VIM fused grid exceeds grid.x capacity");

  AT_DISPATCH_FLOATING_TYPES(
      ke_x_ext.scalar_type(), "vim_fused_launch", [&] {
        vim_fused_kernel<scalar_t>
            <<<static_cast<unsigned int>(blocks), threads, 0,
               at::cuda::getCurrentCUDAStream()>>>(
                acc4<scalar_t>(ke_x_ext), acc4<scalar_t>(ke_y_ext),
                acc4<scalar_t>(u_corner), acc4<scalar_t>(v_corner),
                acc4<scalar_t>(v_x_halo), acc4<scalar_t>(u_y_halo),
                acc3<scalar_t>(hfacz), acc2<scalar_t>(fg),
                acc2<scalar_t>(dxC_ext), acc2<scalar_t>(dyC),
                acc2<scalar_t>(rAz), acc1<scalar_t>(dxG_south),
                acc1<scalar_t>(dxG_north), acc1<scalar_t>(recip_dxC_u),
                dyG.data_ptr<scalar_t>(), recip_dyC.data_ptr<scalar_t>(),
                acc3<scalar_t>(maskW), acc3<scalar_t>(maskS),
                accb_required(x_corner_mask), accb_required(y_corner_mask),
                accb(edge_pair_mask_x), accb(edge_pair_mask_y),
                periodic_x_copy, periodic_y_copy,
                tend_u.data_ptr<scalar_t>(), tend_v.data_ptr<scalar_t>(), nz,
                ny, nx, total);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
