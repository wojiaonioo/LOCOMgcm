#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/Dispatch.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "../../solvers/_cuda_launch/launch_config.h"

namespace {

defgcm::cuda_launch::LaunchConfigCache launch_config_cache;

template <typename scalar_t>
size_t dynamic_shared_bytes_for_block(int block_size) {
  return static_cast<size_t>(6 * block_size) * sizeof(scalar_t);
}

// One extern __shared__ array can only have one type in a translation unit, so
// the two instantiations share a byte array and reinterpret it.
template <typename scalar_t>
__device__ __forceinline__ scalar_t *shared_scalars() {
  extern __shared__ unsigned char kpp_blmix_shared_raw[];
  return reinterpret_cast<scalar_t *>(kpp_blmix_shared_raw);
}

struct Strides1 {
  int64_t level;
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

struct Strides4 {
  int64_t component;
  int64_t level;
  int64_t y;
  int64_t x;
};

template <typename scalar_t>
struct Tensor1 {
  const scalar_t* data;
  Strides1 strides;

  __device__ __forceinline__ scalar_t load(const int level) const {
    return __ldg(data + static_cast<int64_t>(level) * strides.level);
  }
};

template <typename scalar_t>
struct Tensor2 {
  const scalar_t* data;
  Strides2 strides;

  __device__ __forceinline__ scalar_t load(
      const int64_t y,
      const int64_t x) const {
    return __ldg(data + y * strides.y + x * strides.x);
  }
};

struct LongTensor2 {
  const int64_t* data;
  Strides2 strides;

  __device__ __forceinline__ int64_t load(
      const int64_t y,
      const int64_t x) const {
    return data[y * strides.y + x * strides.x];
  }
};

template <typename scalar_t>
struct Tensor3 {
  const scalar_t* data;
  Strides3 strides;

  __device__ __forceinline__ scalar_t load(
      const int level,
      const int64_t y,
      const int64_t x) const {
    const int64_t index =
        static_cast<int64_t>(level) * strides.level +
        y * strides.y + x * strides.x;
    return __ldg(data + index);
  }
};

template <typename scalar_t>
struct Tensor4 {
  const scalar_t* data;
  Strides4 strides;

  __device__ __forceinline__ scalar_t load(
      const int component,
      const int level,
      const int64_t y,
      const int64_t x) const {
    const int64_t index =
        static_cast<int64_t>(component) * strides.component +
        static_cast<int64_t>(level) * strides.level +
        y * strides.y + x * strides.x;
    return __ldg(data + index);
  }
};

template <typename scalar_t>
Tensor1<scalar_t> tensor1(const torch::Tensor& value) {
  return {value.data_ptr<scalar_t>(), {value.stride(0)}};
}

template <typename scalar_t>
Tensor2<scalar_t> tensor2(const torch::Tensor& value) {
  return {value.data_ptr<scalar_t>(), {value.stride(0), value.stride(1)}};
}

LongTensor2 long_tensor2(const torch::Tensor& value) {
  return {value.data_ptr<int64_t>(), {value.stride(0), value.stride(1)}};
}

template <typename scalar_t>
Tensor3<scalar_t> tensor3(const torch::Tensor& value) {
  return {
      value.data_ptr<scalar_t>(),
      {value.stride(0), value.stride(1), value.stride(2)}};
}

template <typename scalar_t>
Tensor4<scalar_t> tensor4(const torch::Tensor& value) {
  return {
      value.data_ptr<scalar_t>(),
      {value.stride(0), value.stride(1), value.stride(2), value.stride(3)}};
}

__device__ __forceinline__ int clamp_index(
    const int64_t value,
    const int upper) {
  if (value < 0) {
    return 0;
  }
  if (value > static_cast<int64_t>(upper)) {
    return upper;
  }
  return static_cast<int>(value);
}

// Ternaries preserve NaN, unlike fmax/fmin.  torch.clamp also preserves NaN.
template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp_min(
    const scalar_t value,
    const scalar_t minimum) {
  return value < minimum ? minimum : value;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp_max(
    const scalar_t value,
    const scalar_t maximum) {
  return value > maximum ? maximum : value;
}

// Fixed local mathematics shared by the profile primal and its analytic VJP.
template <typename T> struct MatchCommon { T delhat, r, denominator, f1; };
template <typename T>
__device__ __forceinline__ MatchCommon<T> match_common(
    T h, T stable, T bf, T u, T hw, T z, T conc1, T eps) {
  MatchCommon<T> c;
  c.delhat = (T(0.5) * hw - z) - h;
  c.r = T(1) - c.delhat / hw;
  c.denominator = clamp_min(((u * u) * u) * u, eps);
  c.f1 = ((stable * conc1) * bf) / c.denominator;
  return c;
}
template <typename T> struct MatchValue { T up, down, p, hvalue, gat, raw_dat, dat; };
template <typename T>
__device__ __forceinline__ MatchValue<T> match_value(
    MatchCommon<T> c, T h, T hw, T hwn, T dm, T d, T dp, T w) {
  MatchValue<T> v;
  v.up = (dm - d) / hw;
  v.down = (d - dp) / hwn;
  v.p = T(0.5) * ((T(1) - c.r) * (v.up + fabs(v.up))
                  + c.r * (v.down + fabs(v.down)));
  v.hvalue = d + v.p * c.delhat;
  v.gat = (v.hvalue / h) / w;
  v.raw_dat = (-v.p) / w + c.f1 * v.hvalue;
  v.dat = clamp_max(v.raw_dat, T(0));
  return v;
}
template <typename T>
__device__ __forceinline__ T profile_value(T h, T w, T sig, T gat, T dat) {
  const T g = ((sig - T(2)) + (T(3) - T(2) * sig) * gat) + (sig - T(1)) * dat;
  return ((h * w) * sig) * (T(1) + sig * g);
}
template <typename T>
__device__ __forceinline__ void profile_pullback(
    T h, T w, T sig, T gat, T dat, T seed,
    T& bh, T& bw, T& bsig, T& bgat, T& bdat) {
  const T a2 = T(3) - T(2) * sig, a3 = sig - T(1);
  const T g = ((sig - T(2)) + a2 * gat) + a3 * dat;
  const T hw = h * w, hws = hw * sig, bracket = T(1) + sig * g;
  const T b_hws = seed * bracket, b_bracket = seed * hws;
  const T b_hw = b_hws * sig, b_g = b_bracket * sig;
  bh = b_hw * w;
  bw = b_hw * h;
  bgat = b_g * a2;
  bdat = b_g * a3;
  bsig = b_hws * hw + b_bracket * g + b_g + (b_g * gat) * T(-2) + b_g * dat;
}
template <typename T>
__device__ __forceinline__ T sign_zero(T value) {
  return value > T(0) ? T(1) : (value < T(0) ? T(-1) : T(0));
}

template <typename scalar_t>
__global__ void kpp_blmix_kernel(
    const Tensor2<scalar_t> ustar,
    const Tensor2<scalar_t> bfsfc,
    const Tensor2<scalar_t> hbl,
    const Tensor2<scalar_t> stable,
    const Tensor2<scalar_t> casea,
    const LongTensor2 kbl,
    const Tensor4<scalar_t> diffus,
    const Tensor1<scalar_t> zgrid,
    const Tensor1<scalar_t> hwide,
    const Tensor2<scalar_t> wm_hbl,
    const Tensor2<scalar_t> ws_hbl,
    const Tensor3<scalar_t> wm_levels,
    const Tensor3<scalar_t> ws_levels,
    const Tensor2<scalar_t> wm_d,
    const Tensor2<scalar_t> ws_d,
    scalar_t* __restrict__ blmc,
    scalar_t* __restrict__ ghat,
    scalar_t* __restrict__ dkm1,
    const int nz,
    const int64_t ny,
    const int64_t nx,
    const scalar_t conc1,
    const scalar_t cg,
    const scalar_t phepsi) {
  const int lane = threadIdx.x;
  const int block_size = blockDim.x;
  const int64_t nplane = ny * nx;
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * block_size + lane;
  if (column >= nplane) {
    return;
  }
  const int64_t y = column / nx;
  const int64_t x = column - y * nx;

  // Each lane owns one column. These six matching values remain in shared
  // storage throughout the level loop, with adjacent lanes stored contiguously.
  scalar_t* shared = shared_scalars<scalar_t>();
  scalar_t* gat1 = shared;
  scalar_t* dat1 = shared + 3 * block_size;

  const scalar_t hbl_value = hbl.load(y, x);
  const scalar_t stable_value = stable.load(y, x);
  const scalar_t bfsfc_value = bfsfc.load(y, x);
  const scalar_t ustar_value = ustar.load(y, x);
  const scalar_t casea_plus_phepsi = casea.load(y, x) + phepsi;
  const int64_t casea_int = static_cast<int64_t>(casea_plus_phepsi);
  const int64_t kbl_value = kbl.load(y, x);
  const int64_t kbl_minus_one = kbl_value - 1;
  const int64_t kn_left = casea_int * kbl_minus_one;
  const int64_t one_minus_casea_int = 1 - casea_int;
  const int64_t kn_right = one_minus_casea_int * kbl_value;
  const int64_t kn = kn_left + kn_right;

  const int kn_index = clamp_index(kn, nz + 1);
  const int kn_plus_one_index = clamp_index(kn + 1, nz + 1);
  const scalar_t hw_kn = hwide.load(kn_index);
  const scalar_t hw_knp1 = hwide.load(kn_plus_one_index);
  const auto common = match_common(hbl_value, stable_value, bfsfc_value,
      ustar_value, hw_kn, zgrid.load(kn_index), conc1, phepsi);

  for (int component = 0; component < 3; ++component) {
    const int kn_minus_one_index = clamp_index(kn - 1, nz + 1);
    const scalar_t d_knm1 =
        diffus.load(component, kn_minus_one_index, y, x);
    const scalar_t d_kn = diffus.load(component, kn_index, y, x);
    const scalar_t d_knp1 =
        diffus.load(component, kn_plus_one_index, y, x);
    const scalar_t w_value = component == 0 ? wm_hbl.load(y, x) : ws_hbl.load(y, x);
    const auto matched = match_value(common, hbl_value, hw_kn, hw_knp1,
        d_knm1, d_kn, d_knp1, w_value);
    const int shared_index = component * block_size + lane;
    gat1[shared_index] = matched.gat;
    dat1[shared_index] = matched.dat;
  }

  for (int ki = 1; ki <= nz; ++ki) {
    const scalar_t negative_z = -zgrid.load(ki);
    const scalar_t half_hwide = static_cast<scalar_t>(0.5) * hwide.load(ki);
    const scalar_t sig_numerator = negative_z + half_hwide;
    const scalar_t sig = sig_numerator / hbl_value;
    const scalar_t wm_i = wm_levels.load(ki - 1, y, x);
    const scalar_t ws_i = ws_levels.load(ki - 1, y, x);

    for (int component = 0; component < 3; ++component) {
      const int shared_index = component * block_size + lane;
      const scalar_t w_value = component == 0 ? wm_i : ws_i;
      const scalar_t output_value = profile_value(hbl_value, w_value, sig,
          gat1[shared_index], dat1[shared_index]);
      const int64_t output_index =
          (static_cast<int64_t>(component) * nz + ki - 1) * nplane +
          column;
      blmc[output_index] = output_value;
    }

    const scalar_t one_minus_stable = static_cast<scalar_t>(1.0) - stable_value;
    const scalar_t ghat_numerator = one_minus_stable * cg;
    const scalar_t ws_hbl_product = ws_i * hbl_value;
    const scalar_t ghat_denominator = clamp_min(ws_hbl_product, phepsi);
    const int64_t ghat_index =
        static_cast<int64_t>(ki - 1) * nplane + column;
    ghat[ghat_index] = ghat_numerator / ghat_denominator;
  }

  const int kbl_minus_one_index = clamp_index(kbl_minus_one, nz + 1);
  const scalar_t negative_z_d = -zgrid.load(kbl_minus_one_index);
  const scalar_t sig_d = negative_z_d / hbl_value;
  const scalar_t wm_d_value = wm_d.load(y, x);
  const scalar_t ws_d_value = ws_d.load(y, x);
  for (int component = 0; component < 3; ++component) {
    const int shared_index = component * block_size + lane;
    const scalar_t w_value = component == 0 ? wm_d_value : ws_d_value;
    const scalar_t output_value = profile_value(hbl_value, w_value, sig_d,
        gat1[shared_index], dat1[shared_index]);
    dkm1[static_cast<int64_t>(component) * nplane + column] = output_value;
  }
}

template <typename T> struct ProfileBars { T* data[15]; };
template <typename T>
__global__ void kpp_blmix_backward_kernel(
    Tensor2<T> ustar, Tensor2<T> bfsfc, Tensor2<T> hbl, Tensor2<T> stable,
    Tensor2<T> casea, LongTensor2 kbl, Tensor4<T> diffus, Tensor1<T> zgrid,
    Tensor1<T> hwide, Tensor2<T> wm_h, Tensor2<T> ws_h,
    Tensor3<T> wm_levels, Tensor3<T> ws_levels, Tensor2<T> wm_d, Tensor2<T> ws_d,
    Tensor4<T> b_blmc, Tensor3<T> b_ghat, Tensor3<T> b_dkm1,
    bool use_blmc, bool use_ghat, bool use_dkm1, ProfileBars<T> bars,
    int nz, int64_t ny, int64_t nx, T conc1, T cg, T phepsi) {
  const int64_t plane = ny * nx;
  const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= plane) return;
  const int64_t y = col / nx, x = col % nx;
  const T h = hbl.load(y,x), st = stable.load(y,x), bf = bfsfc.load(y,x), u = ustar.load(y,x);
  const int64_t kb = kbl.load(y,x);
  const int64_t ca = static_cast<int64_t>(casea.load(y,x) + phepsi);
  const int64_t kn = ca * (kb - 1) + (1 - ca) * kb;
  const int km = clamp_index(kn-1,nz+1), kc = clamp_index(kn,nz+1), kp = clamp_index(kn+1,nz+1);
  const T hw = hwide.load(kc), hwn = hwide.load(kp);
  const auto common = match_common(h, st, bf, u, hw, zgrid.load(kc), conc1, phepsi);
  const T sig_d = -zgrid.load(clamp_index(kb-1,nz+1)) / h;
  T bh = T(0), bst = T(0), bf1 = T(0);
  T bw_h_m = T(0), bw_h_s = T(0), bw_d_m = T(0), bw_d_s = T(0);
  for (int k=0; k<nz; ++k) {
    if (bars.data[11]) bars.data[11][k*plane+col] = T(0);
    if (bars.data[12]) bars.data[12][k*plane+col] = T(0);
  }
  if (bars.data[6]) {
    for (int m=0;m<3;++m) for(int k=0;k<nz+2;++k)
      bars.data[6][(int64_t(m)*(nz+2)+k)*plane+col] = T(0);
  }
  for(int m=0;m<3;++m) {
    const T w_h = m==0 ? wm_h.load(y,x) : ws_h.load(y,x);
    const auto matched = match_value(common, h, hw, hwn,
        diffus.load(m,km,y,x), diffus.load(m,kc,y,x), diffus.load(m,kp,y,x), w_h);
    T bgat=T(0), bdat=T(0);
    for(int k=0;k<nz;++k) {
      const T sig = (-zgrid.load(k+1)+T(0.5)*hwide.load(k+1))/h;
      const T w = m==0 ? wm_levels.load(k,y,x) : ws_levels.load(k,y,x);
      const T seed = use_blmc ? b_blmc.load(m,k,y,x) : T(0);
      T dh,dw,ds,dg,dd;
      profile_pullback(h,w,sig,matched.gat,matched.dat,seed,dh,dw,ds,dg,dd);
      bh += dh - (ds*sig)/h;
      bgat += dg; bdat += dd;
      const int slot = m==0 ? 11 : 12;
      if(bars.data[slot]) bars.data[slot][k*plane+col] += dw;
    }
    T dh,dw,ds,dg,dd;
    const T wd = m==0 ? wm_d.load(y,x) : ws_d.load(y,x);
    const T sd = use_dkm1 ? b_dkm1.load(m,y,x) : T(0);
    profile_pullback(h,wd,sig_d,matched.gat,matched.dat,sd,dh,dw,ds,dg,dd);
    bh += dh - (ds*sig_d)/h; bgat += dg; bdat += dd;
    if(m==0) bw_d_m += dw; else bw_d_s += dw;
    // clamp(max=0) passes the input derivative at the tie, like Torch.
    if(matched.raw_dat > T(0)) bdat=T(0);
    const T bH = (bgat/w_h)/h + bdat*common.f1;
    bh -= ((bgat/w_h)*(matched.hvalue/h))/h;
    const T bwh = -(bgat*matched.gat)/w_h + (bdat*matched.p)/(w_h*w_h);
    if(m==0) bw_h_m += bwh; else bw_h_s += bwh;
    const T bp = -bdat/w_h + bH*common.delhat;
    const T bd = bH*matched.p;
    const T br = T(0.5)*bp*((matched.down+fabs(matched.down))-(matched.up+fabs(matched.up)));
    const T bu = (T(0.5)*bp*(T(1)-common.r))*(T(1)+sign_zero(matched.up));
    const T bn = (T(0.5)*bp*common.r)*(T(1)+sign_zero(matched.down));
    bh += -bd + br/hw;
    bf1 += bdat*matched.hvalue;
    if(bars.data[6]) {
      const int64_t base=int64_t(m)*(nz+2)*plane+col;
      bars.data[6][base+km*plane] += bu/hw;
      bars.data[6][base+kc*plane] += bH-bu/hw+bn/hwn;
      bars.data[6][base+kp*plane] -= bn/hwn;
    }
  }
  for(int k=0;k<nz;++k) {
    const T seed=use_ghat?b_ghat.load(k,y,x):T(0);
    const T ws=ws_levels.load(k,y,x), product=ws*h;
    const T denom=clamp_min(product,phepsi), num=(T(1)-st)*cg;
    bst -= (seed/denom)*cg;
    const T bd = product < phepsi ? T(0) : -(seed*(num/denom))/denom;
    bh += bd*ws;
    if(bars.data[12]) bars.data[12][k*plane+col] += bd*h;
  }
  const T den=common.denominator;
  const T bnum=bf1/den;
  bst += (bnum*bf)*conc1;
  const T bbf = bnum*(st*conc1);
  const T u2=u*u,u3=u2*u,u4=u3*u;
  const T bd = u4 < phepsi ? T(0) : -(bf1*common.f1)/den;
  // Reverse the actual three multiplications, including repeated u operands.
  const T bu3=bd*u,bu2=bu3*u;
  const T bu=bd*u3+bu3*u2+bu2*u+bu2*u;
  if(bars.data[0]) bars.data[0][col]=bu;
  if(bars.data[1]) bars.data[1][col]=bbf;
  if(bars.data[2]) bars.data[2][col]=bh;
  if(bars.data[3]) bars.data[3][col]=bst;
  if(bars.data[4]) bars.data[4][col]=T(0); // casea only feeds integer kn.
  if(bars.data[9]) bars.data[9][col]=bw_h_m;
  if(bars.data[10]) bars.data[10][col]=bw_h_s;
  if(bars.data[13]) bars.data[13][col]=bw_d_m;
  if(bars.data[14]) bars.data[14][col]=bw_d_s;
}


void launch_variant(
    const torch::Tensor& ustar,
    const torch::Tensor& bfsfc,
    const torch::Tensor& hbl,
    const torch::Tensor& stable,
    const torch::Tensor& casea,
    const torch::Tensor& kbl,
    const torch::Tensor& diffus,
    const torch::Tensor& zgrid,
    const torch::Tensor& hwide,
    const torch::Tensor& wm_hbl,
    const torch::Tensor& ws_hbl,
    const torch::Tensor& wm_levels,
    const torch::Tensor& ws_levels,
    const torch::Tensor& wm_d,
    const torch::Tensor& ws_d,
    torch::Tensor& blmc,
    torch::Tensor& ghat,
    torch::Tensor& dkm1,
    const double conc1,
    const double cg,
    const double phepsi) {
  const int nz = static_cast<int>(diffus.size(1) - 2);
  const int64_t ny = ustar.size(0);
  const int64_t nx = ustar.size(1);
  const int64_t total_columns = ny * nx;
  // One instantiation per input dtype.  float64 is the bitwise production
  // path; float32 exists only to measure the cost of the narrower type.
  AT_DISPATCH_FLOATING_TYPES(
      ustar.scalar_type(), "kpp_blmix_launch", [&] {
        const auto& launch_config = launch_config_cache.get(
            kpp_blmix_kernel<scalar_t>,
            ustar.get_device(),
            0,
            16,
            dynamic_shared_bytes_for_block<scalar_t>);
        const int columns_per_block = launch_config.selected_block_size;
        const int64_t block_count =
            (total_columns + columns_per_block - 1) / columns_per_block;
        TORCH_CHECK(
            block_count <= std::numeric_limits<unsigned int>::max(),
            "KPP blmix CUDA grid exceeds grid.x capacity");
        const size_t shared_bytes =
            dynamic_shared_bytes_for_block<scalar_t>(columns_per_block);
        const dim3 block(columns_per_block);
        const dim3 grid(static_cast<unsigned int>(block_count));
        kpp_blmix_kernel<scalar_t>
            <<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                tensor2<scalar_t>(ustar),
                tensor2<scalar_t>(bfsfc),
                tensor2<scalar_t>(hbl),
                tensor2<scalar_t>(stable),
                tensor2<scalar_t>(casea),
                long_tensor2(kbl),
                tensor4<scalar_t>(diffus),
                tensor1<scalar_t>(zgrid),
                tensor1<scalar_t>(hwide),
                tensor2<scalar_t>(wm_hbl),
                tensor2<scalar_t>(ws_hbl),
                tensor3<scalar_t>(wm_levels),
                tensor3<scalar_t>(ws_levels),
                tensor2<scalar_t>(wm_d),
                tensor2<scalar_t>(ws_d),
                blmc.data_ptr<scalar_t>(),
                ghat.data_ptr<scalar_t>(),
                dkm1.data_ptr<scalar_t>(),
                nz,
                ny,
                nx,
                static_cast<scalar_t>(conc1),
                static_cast<scalar_t>(cg),
                static_cast<scalar_t>(phepsi));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

}  // namespace

void kpp_blmix_launch(
    const torch::Tensor& ustar,
    const torch::Tensor& bfsfc,
    const torch::Tensor& hbl,
    const torch::Tensor& stable,
    const torch::Tensor& casea,
    const torch::Tensor& kbl,
    const torch::Tensor& diffus,
    const torch::Tensor& zgrid,
    const torch::Tensor& hwide,
    const torch::Tensor& wm_hbl,
    const torch::Tensor& ws_hbl,
    const torch::Tensor& wm_levels,
    const torch::Tensor& ws_levels,
    const torch::Tensor& wm_d,
    const torch::Tensor& ws_d,
    torch::Tensor& blmc,
    torch::Tensor& ghat,
    torch::Tensor& dkm1,
    const double conc1,
    const double cg,
    const double phepsi) {
  c10::cuda::CUDAGuard device_guard(ustar.device());
  launch_variant(
      ustar, bfsfc, hbl, stable, casea, kbl, diffus, zgrid, hwide,
      wm_hbl, ws_hbl, wm_levels, ws_levels, wm_d, ws_d,
      blmc, ghat, dkm1, conc1, cg, phepsi);
}

const defgcm::cuda_launch::LaunchConfig&
kpp_blmix_cuda_launch_config(int device, at::ScalarType dtype) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  // Mirrors the AT_DISPATCH_FLOATING_TYPES in launch_variant above: the two
  // instantiations carry different register and shared-memory footprints, so
  // the exported configuration must name the one this dtype actually launches.
  if (dtype == at::kFloat) {
    return launch_config_cache.get(
        kpp_blmix_kernel<float>,
        device,
        0,
        16,
        dynamic_shared_bytes_for_block<float>);
  }
  TORCH_CHECK(
      dtype == at::kDouble,
      "kpp_blmix CUDA launch configuration supports float64 or float32 only");
  return launch_config_cache.get(
      kpp_blmix_kernel<double>,
      device,
      0,
      16,
      dynamic_shared_bytes_for_block<double>);
}

void kpp_blmix_backward_launch(
    const std::vector<torch::Tensor>& v, const std::vector<c10::optional<torch::Tensor>>& seeds,
    std::vector<c10::optional<torch::Tensor>>& bars, double conc1, double cg, double phepsi) {
  c10::cuda::CUDAGuard guard(v[0].device());
  AT_DISPATCH_FLOATING_TYPES(v[0].scalar_type(), "kpp_blmix_backward", [&] {
    ProfileBars<scalar_t> out;
    for(int i=0;i<15;++i) out.data[i]=bars[i]?bars[i]->data_ptr<scalar_t>():nullptr;
    const int nz=v[6].size(1)-2;
    const int64_t ny=v[0].size(0),nx=v[0].size(1),count=ny*nx;
    static defgcm::cuda_launch::LaunchConfigCache backward_cache;
    const auto& config=backward_cache.get(kpp_blmix_backward_kernel<scalar_t>,
        v[0].get_device(),0,16,[](int){return size_t(0);});
    const int64_t blocks=(count+config.selected_block_size-1)/config.selected_block_size;
    TORCH_CHECK(blocks<=std::numeric_limits<unsigned int>::max(),"BLMIX backward grid exceeds capacity");
    // Inactive seeds carry unused views of existing inputs, never zero fields.
    kpp_blmix_backward_kernel<scalar_t><<<static_cast<unsigned int>(blocks),config.selected_block_size,
        0,at::cuda::getCurrentCUDAStream()>>>(
      tensor2<scalar_t>(v[0]),tensor2<scalar_t>(v[1]),tensor2<scalar_t>(v[2]),
      tensor2<scalar_t>(v[3]),tensor2<scalar_t>(v[4]),long_tensor2(v[5]),
      tensor4<scalar_t>(v[6]),tensor1<scalar_t>(v[7]),tensor1<scalar_t>(v[8]),
      tensor2<scalar_t>(v[9]),tensor2<scalar_t>(v[10]),tensor3<scalar_t>(v[11]),
      tensor3<scalar_t>(v[12]),tensor2<scalar_t>(v[13]),tensor2<scalar_t>(v[14]),
      tensor4<scalar_t>(seeds[0]?*seeds[0]:v[6]),
      tensor3<scalar_t>(seeds[1]?*seeds[1]:v[11]),
      tensor3<scalar_t>(seeds[2]?*seeds[2]:v[11]),
      seeds[0].has_value(),seeds[1].has_value(),seeds[2].has_value(),out,
      nz,ny,nx,scalar_t(conc1),scalar_t(cg),scalar_t(phepsi));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}
