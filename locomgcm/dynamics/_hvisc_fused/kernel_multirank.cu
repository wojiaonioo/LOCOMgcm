// Multi-rank Leith/Smag coefficient kernels.  Communication is adapter-side:
// P materialises eager intermediates, and Q consumes their exchanged halos.

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

// Both entry points allocate fresh outputs in extension.cpp. Inputs are
// read-only, so sharing storage between inputs does not violate restrict.
template <typename T> struct A4 {
  const T *__restrict__ p; int64_t sb, sk, sj, si;
  __device__ __forceinline__ T at(int64_t b, int64_t k, int64_t j, int64_t i) const {
    return p[b * sb + k * sk + j * sj + i * si];
  }
};
template <typename T> struct A3 {
  const T *__restrict__ p; int64_t sk, sj, si;
  __device__ __forceinline__ T at(int64_t k, int64_t j, int64_t i) const {
    return p[k * sk + j * sj + i * si];
  }
};
template <typename T> struct A2 {
  const T *__restrict__ p; int64_t sj, si;
  __device__ __forceinline__ T at(int64_t j, int64_t i) const {
    return p[j * sj + i * si];
  }
};
template <typename T> struct M2 {
  const T *__restrict__ p; int64_t sj, si;
  __device__ __forceinline__ bool present() const { return p != nullptr; }
  __device__ __forceinline__ T at(int64_t j, int64_t i) const {
    return p[j * sj + i * si];
  }
};

template <typename T>
__device__ __forceinline__ T torch_maximum(T first, T second) {
  if (isnan(first)) return first;
  if (isnan(second)) return second;
  return first >= second ? first : second;
}

template <typename T>
__global__ void hvisc_mr_pre_kernel(
    A4<T> u, A4<T> v, A4<T> u_ext, A4<T> v_dyC_ext, A2<T> dxC_ext,
    A2<T> dxG, A2<T> dyG, A2<T> dyF, A2<T> recip_rA, A2<T> recip_rAz,
    A3<T> hFacW, A3<T> hFacS, A3<T> recip_hFacC, A3<T> maskW,
    A3<T> maskS, A3<T> hfacz, M2<T> maskInC, M2<T> maskInW,
    M2<T> maskInS, T side_factor, T *__restrict__ hdiv_out,
    T *__restrict__ vort_out, T *__restrict__ strain_out,
    T *__restrict__ tension_out, T *__restrict__ vrtDx_out,
    T *__restrict__ vrtDy_out,
    int64_t nz, int64_t ny, int64_t nx, int64_t total) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total) return;
  const int64_t i = tid % (nx + 1);
  int64_t rest = tid / (nx + 1);
  const int64_t j = rest % (ny + 1);
  rest /= ny + 1;
  const int64_t k = rest % nz;
  const int64_t b = rest / nz;
  const T zero = static_cast<T>(0);

  auto curl_pair = [&](int64_t jj, int64_t ii, T &vort, T &strain) {
    const T curl_u = u_ext.at(b, k, jj + 1, ii) * dxC_ext.at(jj + 1, ii) -
                     u_ext.at(b, k, jj, ii) * dxC_ext.at(jj, ii);
    const T curl_v = v_dyC_ext.at(b, k, jj, ii + 1) -
                     v_dyC_ext.at(b, k, jj, ii);
    const T recip = recip_rAz.at(jj, ii);
    vort = recip * (curl_v - curl_u);
    strain = recip * (curl_v + curl_u);
  };
  auto vort_bc = [&](int64_t jj, int64_t ii) {
    T vort, strain;
    curl_pair(jj, ii, vort, strain);
    return hfacz.at(k, jj, ii) > zero ? vort : vort * side_factor;
  };

  T vort, strain;
  curl_pair(j, i, vort, strain);
  const int64_t corner = ((b * nz + k) * (ny + 1) + j) * (nx + 1) + i;
  vort_out[corner] = vort;
  strain_out[corner] = hfacz.at(k, j, i) > zero ? strain : strain * side_factor;

  T vdx = zero;
  if (i < nx) {
    T mask = maskS.at(k, j, i);
    if (maskInS.present()) mask = mask * maskInS.at(j, i);
    vdx = (vort_bc(j, i + 1) - vort_bc(j, i)) / dxG.at(j, i) * mask;
  }
  vrtDx_out[corner] = vdx;

  T vdy = zero;
  if (j < ny) {
    T mask = maskW.at(k, j, i);
    if (maskInW.present()) mask = mask * maskInW.at(j, i);
    vdy = (vort_bc(j + 1, i) - vort_bc(j, i)) / dyG.at(j, i) * mask;
  }
  vrtDy_out[corner] = vdy;

  if (j < ny && i < nx) {
    const T u_flux_e = (u.at(b,k,j,i+1) * dyG.at(j,i+1)) * hFacW.at(k,j,i+1);
    const T u_flux_w = (u.at(b,k,j,i) * dyG.at(j,i)) * hFacW.at(k,j,i);
    const T v_flux_n = (v.at(b,k,j+1,i) * dxG.at(j+1,i)) * hFacS.at(k,j+1,i);
    const T v_flux_s = (v.at(b,k,j,i) * dxG.at(j,i)) * hFacS.at(k,j,i);
    T hdiv = ((u_flux_e - u_flux_w) + (v_flux_n - v_flux_s)) *
             recip_rA.at(j,i) * recip_hFacC.at(k,j,i);
    if (maskInC.present()) hdiv = hdiv * maskInC.at(j,i);
    const T tension_u = dyF.at(j,i) * (u.at(b,k,j,i+1) - u.at(b,k,j,i));
    const T tension_v = v.at(b,k,j+1,i) * dxG.at(j+1,i) -
                        v.at(b,k,j,i) * dxG.at(j,i);
    T tension = (tension_u - tension_v) * recip_rA.at(j,i);
    if (maskInC.present()) tension = tension * maskInC.at(j,i);
    const int64_t center = ((b * nz + k) * ny + j) * nx + i;
    hdiv_out[center] = hdiv;
    tension_out[center] = tension;
  }
}

template <typename T>
__global__ void hvisc_mr_coeff_kernel(
    A4<T> hdiv, A4<T> vort, A4<T> strain, A4<T> vrtDx_x,
    A4<T> vrtDy_y, A4<T> tension_xy, A3<T> hfacz, A2<T> l2_d,
    A2<T> l3_d, A2<T> l2_z, A2<T> l3_z, const T *leith_p,
    const T *smag_p, const T *base_p, T *visc_d_out, T *flux_out,
    T *hz_out, int64_t nz, int64_t ny, int64_t nx, int64_t total) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total) return;
  const int64_t i = tid % (nx + 1);
  int64_t rest = tid / (nx + 1);
  const int64_t j = rest % (ny + 1);
  rest /= ny + 1;
  const int64_t k = rest % nz;
  const int64_t b = rest / nz;
  const T quarter = static_cast<T>(0.25);
  const T tiny = static_cast<T>(1e-30);
  const T leith = *leith_p, smag = *smag_p, base = *base_p;
  auto max4 = [&](T a, T c, T d, T e) {
    T out = torch_maximum(fabs(a), fabs(c));
    out = torch_maximum(out, fabs(d));
    return torch_maximum(out, fabs(e));
  };

  const T vdx_i = vrtDx_x.at(b,k,j,i+1);
  const T vdx_im = vrtDx_x.at(b,k,j,i);
  const T vdy_j = vrtDy_y.at(b,k,j+1,i);
  const T vdy_jm = vrtDy_y.at(b,k,j,i);
  const T vrt_z_max = max4(vdx_i, vdx_im, vdy_j, vdy_jm);
  const T t00 = tension_xy.at(b,k,j+1,i+1);
  const T tjm = tension_xy.at(b,k,j,i+1);
  const T tim = tension_xy.at(b,k,j+1,i);
  const T timjm = tension_xy.at(b,k,j,i);
  const T tension_sq_z =
      quarter * (((t00*t00 + tjm*tjm) + tim*tim) + timjm*timjm);
  const T strain_z = strain.at(b,k,j,i);
  const T smag_z = sqrt((strain_z*strain_z + tension_sq_z) + tiny);
  T visc_z = (leith * vrt_z_max) * l3_z.at(j,i) +
             (smag * smag_z) * l2_z.at(j,i);
  visc_z = visc_z + base;
  hz_out[((b*nz+k)*(ny+1)+j)*(nx+1)+i] =
      hfacz.at(k,j,i) * vort.at(b,k,j,i) * visc_z;

  if (j < ny && i < nx) {
    const T vdx0 = vrtDx_x.at(b,k,j,i+1);
    const T vdx1 = vrtDx_x.at(b,k,j+1,i+1);
    const T vdy0 = vrtDy_y.at(b,k,j+1,i);
    const T vdy1 = vrtDy_y.at(b,k,j+1,i+1);
    const T vrt_d_max = max4(vdx0, vdx1, vdy0, vdy1);
    const T s0 = strain.at(b,k,j,i+1);
    const T s1 = strain.at(b,k,j+1,i);
    const T s2 = strain.at(b,k,j,i);
    const T s3 = strain.at(b,k,j+1,i+1);
    const T strain_sq_d = quarter * (((s0*s0 + s1*s1) + s2*s2) + s3*s3);
    const T tension = tension_xy.at(b,k,j+1,i+1);
    const T smag_d = sqrt((tension*tension + strain_sq_d) + tiny);
    T visc_d = (leith * vrt_d_max) * l3_d.at(j,i) +
               (smag * smag_d) * l2_d.at(j,i);
    visc_d = visc_d + base;
    const int64_t center = ((b*nz+k)*ny+j)*nx+i;
    visc_d_out[center] = visc_d;
    flux_out[center] = hdiv.at(b,k,j,i) * visc_d;
  }
}

template <typename T> A4<T> a4(const torch::Tensor &t) {
  return {t.data_ptr<T>(),t.stride(0),t.stride(1),t.stride(2),t.stride(3)};
}
template <typename T> A3<T> a3(const torch::Tensor &t) {
  return {t.data_ptr<T>(),t.stride(0),t.stride(1),t.stride(2)};
}
template <typename T> A2<T> a2(const torch::Tensor &t) {
  const auto n=t.dim(); return {t.data_ptr<T>(),t.stride(n-2),t.stride(n-1)};
}
template <typename T> M2<T> m2(const std::optional<torch::Tensor> &t) {
  if (!t) return {nullptr,0,0};
  const auto n=t->dim(); return {t->data_ptr<T>(),t->stride(n-2),t->stride(n-1)};
}
} // namespace

void hvisc_mr_pre_launch(
    const torch::Tensor &u, const torch::Tensor &v, const torch::Tensor &u_ext,
    const torch::Tensor &v_dyC_ext, const torch::Tensor &dxC_ext,
    const torch::Tensor &dxG, const torch::Tensor &dyG, const torch::Tensor &dyF,
    const torch::Tensor &recip_rA, const torch::Tensor &recip_rAz,
    const torch::Tensor &hFacW, const torch::Tensor &hFacS,
    const torch::Tensor &recip_hFacC, const torch::Tensor &maskW,
    const torch::Tensor &maskS, const torch::Tensor &hfacz,
    const std::optional<torch::Tensor> &maskInC,
    const std::optional<torch::Tensor> &maskInW,
    const std::optional<torch::Tensor> &maskInS, double side_factor,
    torch::Tensor &hdiv, torch::Tensor &vort, torch::Tensor &strain,
    torch::Tensor &tension, torch::Tensor &vrtDx, torch::Tensor &vrtDy) {
  const c10::cuda::CUDAGuard guard(u.device());
  const int64_t nz=u.size(1), ny=u.size(2), nx=u.size(3)-1;
  const int64_t total=u.size(0)*nz*(ny+1)*(nx+1);
  constexpr int threads=256; const int64_t blocks=(total+threads-1)/threads;
  TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(), "hvisc mr P grid too large");
  AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "hvisc_mr_pre", [&] {
    hvisc_mr_pre_kernel<scalar_t><<<blocks,threads,0,at::cuda::getCurrentCUDAStream()>>>(
      a4<scalar_t>(u),a4<scalar_t>(v),a4<scalar_t>(u_ext),a4<scalar_t>(v_dyC_ext),
      a2<scalar_t>(dxC_ext),a2<scalar_t>(dxG),a2<scalar_t>(dyG),a2<scalar_t>(dyF),
      a2<scalar_t>(recip_rA),a2<scalar_t>(recip_rAz),a3<scalar_t>(hFacW),
      a3<scalar_t>(hFacS),a3<scalar_t>(recip_hFacC),a3<scalar_t>(maskW),
      a3<scalar_t>(maskS),a3<scalar_t>(hfacz),m2<scalar_t>(maskInC),
      m2<scalar_t>(maskInW),m2<scalar_t>(maskInS),static_cast<scalar_t>(side_factor),
      hdiv.data_ptr<scalar_t>(),vort.data_ptr<scalar_t>(),strain.data_ptr<scalar_t>(),
      tension.data_ptr<scalar_t>(),vrtDx.data_ptr<scalar_t>(),vrtDy.data_ptr<scalar_t>(),
      nz,ny,nx,total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hvisc_mr_coeff_launch(
    const torch::Tensor &hdiv, const torch::Tensor &vort,
    const torch::Tensor &strain, const torch::Tensor &vrtDx_x,
    const torch::Tensor &vrtDy_y, const torch::Tensor &tension_xy,
    const torch::Tensor &hfacz, const torch::Tensor &l2_d,
    const torch::Tensor &l3_d, const torch::Tensor &l2_z,
    const torch::Tensor &l3_z, const torch::Tensor &leith,
    const torch::Tensor &smag, const torch::Tensor &base,
    torch::Tensor &visc_d, torch::Tensor &flux, torch::Tensor &hz) {
  const c10::cuda::CUDAGuard guard(hdiv.device());
  const int64_t nz=hdiv.size(1),ny=hdiv.size(2),nx=hdiv.size(3);
  const int64_t total=hdiv.size(0)*nz*(ny+1)*(nx+1);
  constexpr int threads=256; const int64_t blocks=(total+threads-1)/threads;
  TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(), "hvisc mr Q grid too large");
  AT_DISPATCH_FLOATING_TYPES(hdiv.scalar_type(), "hvisc_mr_coeff", [&] {
    hvisc_mr_coeff_kernel<scalar_t><<<blocks,threads,0,at::cuda::getCurrentCUDAStream()>>>(
      a4<scalar_t>(hdiv),a4<scalar_t>(vort),a4<scalar_t>(strain),a4<scalar_t>(vrtDx_x),
      a4<scalar_t>(vrtDy_y),a4<scalar_t>(tension_xy),a3<scalar_t>(hfacz),
      a2<scalar_t>(l2_d),a2<scalar_t>(l3_d),a2<scalar_t>(l2_z),a2<scalar_t>(l3_z),
      leith.data_ptr<scalar_t>(),smag.data_ptr<scalar_t>(),base.data_ptr<scalar_t>(),
      visc_d.data_ptr<scalar_t>(),flux.data_ptr<scalar_t>(),hz.data_ptr<scalar_t>(),
      nz,ny,nx,total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
