// Regional KPP stages surrounding the reused STATEKPP, bldepth, and blmix
// kernels. Storage is templated; arithmetic in these glue kernels is double.

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

// Sub-expression tap slots for ri_kernel (debug entry only).
#define KPP_RI_NTAP 9

struct S1 { int64_t x; };
struct S2 { int64_t y, x; };
struct S3 { int64_t k, y, x; };

template <class T> __device__ __forceinline__ double ld1(
    const T* p, S1 s, int64_t i) { return static_cast<double>(__ldg(p + i*s.x)); }
template <class T> __device__ __forceinline__ double ld2(
    const T* p, S2 s, int64_t j, int64_t i) {
  return static_cast<double>(__ldg(p + j*s.y + i*s.x));
}
template <class T> __device__ __forceinline__ double ld3(
    const T* p, S3 s, int64_t k, int64_t j, int64_t i) {
  return static_cast<double>(__ldg(p + k*s.k + j*s.y + i*s.x));
}
template <class T> __device__ __forceinline__ void st(T* p, int64_t i, double v) {
  p[i] = static_cast<T>(v);
}
__device__ __forceinline__ int64_t clampi(int64_t v, int64_t lo, int64_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
__device__ __forceinline__ double clamp_min(double v, double lo) {
  return isnan(v) ? v : (v < lo ? lo : v);
}
__device__ __forceinline__ double clamp_max(double v, double hi) {
  return isnan(v) ? v : (v > hi ? hi : v);
}
// Fortran SIGN(a, b) with the runtime -0.0 semantics of kpp.py::_fortran_sign.
__device__ __forceinline__ double fsign(double a, double b, bool sign_zero_positive) {
  if (sign_zero_positive) return b >= 0.0 ? fabs(a) : -fabs(a);
  return copysign(fabs(a), b);
}

// Contraction-proof arithmetic for the two multiply-then-add/subtract sites in
// ri_kernel.  ``-fmad=false`` is passed at build time (dispatcher.py:139), but
// a build that loses the flag changes results SILENTLY, and ``1 - r*r`` is a
// cancellation site: one contracted rounding there is amplified by orders of
// magnitude (fp64 probe: r=0.99 -> 3 ULP, r=0.999 -> 66 ULP, r=0.9999 -> 1854
// ULP), then cubed, then scaled into diffus.  The eager reference cannot
// contract -- each multiply and add is a separate ATen kernel with its own
// rounding -- so matching it bitwise requires the same two roundings here.
// ``__d*_rn`` are IEEE round-to-nearest and are never contracted, so the result
// no longer depends on a compiler flag reaching this translation unit.  Both
// helpers are bit-identical to the plain form whenever -fmad=false IS in
// effect, which makes this change its own control.
__device__ __forceinline__ double one_minus_square(double r) {
  return __dsub_rn(1.0, __dmul_rn(r, r));
}
__device__ __forceinline__ double compose_bg_con_ri(
    double bg, double fcon, double con_coeff, double fri, double ri_coeff) {
  return __dadd_rn(__dadd_rn(bg, __dmul_rn(fcon, con_coeff)),
                   __dmul_rn(fri, ri_coeff));
}
// a*b + c*d with both roundings kept, matching two ATen multiplies and an add.
__device__ __forceinline__ double blend2(double a, double b, double c, double d) {
  return __dadd_rn(__dmul_rn(a, b), __dmul_rn(c, d));
}

template <class T>
__global__ void preprocess_kernel(
    const T* dbloc, S3 sdbloc, const T* dbsfc, S3 sdbsfc,
    const T* rho, S2 srho, const T* alpha, S3 salpha,
    const T* beta, S3 sbeta, const T* u, S3 su, const T* v, S3 sv,
    const T* sft, S2 ssft, const T* sfs, S2 ssfs,
    const T* sfu, S2 ssfu, const T* sfv, S2 ssfv,
    const T* qsw, S2 sqsw, const T* mask, S3 smask,
    const T* kmtj, S2 skmtj, const T* zgrid, S1 szgrid,
    const T* layer_thickness, S1 slayer,
    T* dbloc_sm, T* dbloc_masked, T* ritop, T* dvsq, T* shsq,
    T* ustar, T* bo, T* bosol, int nz, int64_t ny, int64_t nx,
    double gravity, double rho_const, double phepsi, double heat_capacity) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  const int64_t plane = ny*nx;
  const int64_t volume = static_cast<int64_t>(nz)*plane;
  if (tid < volume) {
    const int k = static_cast<int>(tid/plane);
    const int64_t col = tid - static_cast<int64_t>(k)*plane;
    const int64_t j = col/nx, i = col-j*nx;
    const int kp1 = k+1 < nz ? k+1 : nz-1;
    const double mc = ld3(mask, smask, k, j, i);
    const double mp = ld3(mask, smask, kp1, j, i);
    const double raw = ld3(dbloc, sdbloc, k, j, i);
    double smooth = raw;
    if (k < nz-1 && j > 0 && j+1 < ny && i > 0 && i+1 < nx) {
      double m[9], f[9];
      int n = 0;
      for (int dj=-1; dj<=1; ++dj) for (int di=-1; di<=1; ++di) {
        m[n] = ld3(mask, smask, k+1, j+dj, i+di);
        f[n] = ld3(dbloc, sdbloc, k, j+dj, i+di);
        ++n;
      }
      const double edges = m[3]+m[5]+m[1]+m[7];
      const double corners = m[0]+m[6]+m[2]+m[8];
      const double weight = 0.25*m[4] + 0.125*edges + 0.0625*corners;
      const double edge_num = f[3]*m[3] + f[5]*m[5] + f[1]*m[1] + f[7]*m[7];
      const double corner_num = f[0]*m[0] + f[6]*m[6] + f[2]*m[2] + f[8]*m[8];
      const double num = 0.25*f[4]*m[4] + 0.125*edge_num + 0.0625*corner_num;
      smooth = weight >= 0.25 ? num/weight : raw;
    }
    st(dbloc_sm, tid, smooth*mc*mp);
    st(dbloc_masked, tid, raw*mc*mp);
    const double scale = ld1(zgrid, szgrid, 1) - ld1(zgrid, szgrid, k+1);
    double ri = scale*ld3(dbsfc, sdbsfc, k, j, i)*mc*ld3(mask, smask, 0, j, i);
    if (k+1 >= static_cast<int64_t>(ld2(kmtj, skmtj, j, i))) ri = 0.0;
    st(ritop, tid, ri);

    double dv_surface = 0.0, shear = 0.0;
    if (j > 0 && j+1 < ny && i > 0 && i+1 < nx) {
      const double duw = ld3(u,su,0,j,i)-ld3(u,su,k,j,i);
      const double due = ld3(u,su,0,j,i+1)-ld3(u,su,k,j,i+1);
      const double dvs = ld3(v,sv,0,j,i)-ld3(v,sv,k,j,i);
      const double dvn = ld3(v,sv,0,j+1,i)-ld3(v,sv,k,j+1,i);
      dv_surface = 0.5*(duw*duw + due*due + dvs*dvs + dvn*dvn);
      if (k < nz-1) {
        auto du = [&](int64_t y, int64_t x) {
          return ld3(u,su,k,y,x)-ld3(u,su,k+1,y,x);
        };
        auto dv = [&](int64_t y, int64_t x) {
          return ld3(v,sv,k,y,x)-ld3(v,sv,k+1,y,x);
        };
        const double base = du(j,i)*du(j,i) + du(j,i+1)*du(j,i+1) +
                            dv(j,i)*dv(j,i) + dv(j+1,i)*dv(j+1,i);
        const double extra = du(j-1,i)*du(j-1,i) + du(j-1,i+1)*du(j-1,i+1) +
            du(j+1,i)*du(j+1,i) + du(j+1,i+1)*du(j+1,i+1) +
            dv(j,i-1)*dv(j,i-1) + dv(j+1,i-1)*dv(j+1,i-1) +
            dv(j,i+1)*dv(j,i+1) + dv(j+1,i+1)*dv(j+1,i+1);
        shear = 0.5*(0.5*base) + 0.125*extra;
      }
    }
    st(dvsq, tid, dv_surface);
    st(shsq, tid, shear);
  }
  if (tid < plane) {
    const int64_t j=tid/nx, i=tid-j*nx;
    double us = 0.0;
    if (j > 0 && j+1 < ny && i > 0 && i+1 < nx) {
      const double a = ld2(sfu,ssfu,j,i)+ld2(sfu,ssfu,j,i+1);
      const double b = ld2(sfv,ssfv,j,i)+ld2(sfv,ssfv,j+1,i);
      const double work = a*a+b*b;
      // Read the same layer thickness as eager. Deriving it from zgrid
      // would assume a particular vertical-coordinate construction.
      const double layer = ld1(layer_thickness,slayer,0);
      const double eps_sq = phepsi*phepsi*layer*layer;
      const double eps_loc = sqrt(0.5*phepsi*layer);
      us = work < eps_sq ? eps_loc : sqrt(sqrt(work)*0.5);
    }
    st(ustar,tid,us);
    const double r = ld2(rho,srho,j,i);
    const double al = ld3(alpha,salpha,0,j,i), be = ld3(beta,sbeta,0,j,i);
    const double buoy = -gravity*(al*ld2(sft,ssft,j,i)+be*ld2(sfs,ssfs,j,i))/r;
    const double solar = gravity*al*ld2(qsw,sqsw,j,i)*(1.0/heat_capacity)*(1.0/rho_const)/r;
    st(bo,tid,buoy); st(bosol,tid,solar);
  }
}

template <class T>
__global__ void ri_kernel(
    const int64_t* kmtj, S2 skmtj, const T* shsq, S3 ssh,
    const T* dbloc, S3 sd, const T* dbloc_sm, S3 sds,
    const T* bg_s, S3 sbgs, const T* bg_t, S3 sbgt,
    const T* zgrid, S1 sz, T* diffus, int nz, int64_t ny, int64_t nx,
    double ri_inf, double bvsq_con, double difm0, double difs0, double dift0,
    double difmcon, double difscon, double diftcon, double visc_bg,
    double phepsi,
    // Host-computed reciprocals of the two PYTHON-SCALAR divisors.  ATen's
    // div_true_kernel_cuda takes its is_cpu_scalar(2) branch for tensor/scalar
    // and computes inv_b = opmath_t(1.0)/b on the HOST, then launches a*inv_b
    // (its own comment: "may lose one bit of precision").  A device divide here
    // would round differently: fp64 probe puts x/c != x*(1/c) at 23.95% of
    // samples for Riinfty=0.7 and 42.72% for BVSQcon=-2e-5, 1 ULP each, which
    // then survives the cube and the compose into diffus.  The same preserved
    // lowering is already why the wscale table lookups below (inv_dz/inv_du)
    // and the whole reused bldepth kernel are bitwise equal.
    double inv_ri_inf, double inv_bvsq_con,
    // Nullable sub-expression tap, laid out as (KPP_RI_NTAP, nz+2, ny, nx).
    // Written only when the debug entry supplies a buffer; the production path
    // passes nullptr.  Every store sits AFTER the value it records is final, so
    // the tap cannot re-associate the arithmetic it measures.  Slot order must
    // match ``dispatcher.RI_TAP_NAMES``.
    T* dbg) {
  const int64_t tid=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t plane=ny*nx, volume=static_cast<int64_t>(nz+2)*plane;
  if (tid>=volume) return;
  const int ki=static_cast<int>(tid/plane);
  const int64_t col=tid-static_cast<int64_t>(ki)*plane;
  const int64_t j=col/nx, i=col-j*nx;
  for (int m=0;m<3;++m) diffus[(static_cast<int64_t>(m)*(nz+2)+ki)*plane+col]=T(0);
  const int64_t slot=static_cast<int64_t>(nz+2)*plane;
  const int64_t at=static_cast<int64_t>(ki)*plane+col;
  if (dbg!=nullptr) for (int t=0;t<KPP_RI_NTAP;++t) dbg[t*slot+at]=T(0);
  if (ki<1 || ki>nz) return;
  const int k=ki-1;
  const int64_t bottom=kmtj[j*skmtj.y+i*skmtj.x];
  // Seafloor and deeper entries retain their initial zeros, as in eager.
  // Return before forming bottom-2 indices, which can be negative.
  if (ki>=bottom) return;
  const int src=static_cast<int>(clampi(k,0,bottom-2));
  const double dz=ld1(zgrid,sz,src+1)-ld1(zgrid,sz,src+2);
  const double rig_a=ld3(dbloc_sm,sds,src,j,i);
  const double rig_c=clamp_min(ld3(shsq,ssh,src,j,i),phepsi);
  const double rig=rig_a*dz/rig_c;
  const double nsq_a=ld3(dbloc,sd,src,j,i);
  const double nsq=nsq_a/dz;
  const double nsq_clamped=clamp_min(nsq,bvsq_con);
  double ratio=clamp_max((bvsq_con-nsq_clamped)*inv_bvsq_con,1.0);
  double fcon=one_minus_square(ratio); fcon=fcon*fcon*fcon;
  const double rigs=clamp_min(rig,0.0);
  ratio=clamp_max(rigs*inv_ri_inf,1.0);
  double fri=one_minus_square(ratio); fri=fri*fri*fri;
  const int kp1=k+1<nz?k+1:nz-1;
  if (dbg!=nullptr) {
    dbg[0*slot+at]=static_cast<T>(src);   dbg[1*slot+at]=static_cast<T>(rig_a);
    dbg[2*slot+at]=static_cast<T>(rig_c); dbg[3*slot+at]=static_cast<T>(dz);
    dbg[4*slot+at]=static_cast<T>(nsq_a); dbg[5*slot+at]=static_cast<T>(rig);
    dbg[6*slot+at]=static_cast<T>(nsq);   dbg[7*slot+at]=static_cast<T>(fcon);
    dbg[8*slot+at]=static_cast<T>(fri);
  }
  diffus[(0LL*(nz+2)+ki)*plane+col]=static_cast<T>(
      compose_bg_con_ri(visc_bg,fcon,difmcon,fri,difm0));
  diffus[(1LL*(nz+2)+ki)*plane+col]=static_cast<T>(
      compose_bg_con_ri(ld3(bg_s,sbgs,kp1,j,i),fcon,difscon,fri,difs0));
  diffus[(2LL*(nz+2)+ki)*plane+col]=static_cast<T>(
      compose_bg_con_ri(ld3(bg_t,sbgt,kp1,j,i),fcon,diftcon,fri,dift0));
}

template <class T>
__device__ __forceinline__ void wscale(
    double sigma, double hbl, double ustar, double bfsfc,
    const T* wmt, S2 swmt, const T* wst, S2 swst, double& wm, double& ws) {
  constexpr double zmin=-4.0e-7, zmax=0.0, umin=0.0, umax=4.0e-2;
  constexpr int nni=890, nnj=480;
  constexpr double vonk=0.4, conc1=5.0;
  const double dz=(zmax-zmin)/(nni+1), du=(umax-umin)/(nnj+1);
  const double inv_dz=1.0/dz, inv_du=1.0/du;
  const double zehat=vonk*sigma*hbl*bfsfc;
  const double zdiff=zehat-zmin, udiff=ustar-umin;
  int64_t iz=clampi(static_cast<int64_t>(zdiff*inv_dz),0,nni);
  int64_t ju=clampi(static_cast<int64_t>(udiff*inv_du),0,nnj);
  const double zfrac=zdiff*inv_dz-static_cast<double>(iz);
  const double ufrac=udiff*inv_du-static_cast<double>(ju), fz=1.0-zfrac;
  auto bilin=[&](const T* table,S2 strides) {
    const double wa=fz*ld2(table,strides,iz,ju+1)+zfrac*ld2(table,strides,iz+1,ju+1);
    const double wb=fz*ld2(table,strides,iz,ju)+zfrac*ld2(table,strides,iz+1,ju);
    return (1.0-ufrac)*wb+ufrac*wa;
  };
  const double wm_tab=bilin(wmt,swmt), ws_tab=bilin(wst,swst);
  const double u3=ustar*ustar*ustar;
  const double stable=vonk*ustar*u3/(u3+conc1*zehat);
  wm=zehat<=zmax?wm_tab:stable; ws=zehat<=zmax?ws_tab:stable;
}

template <class T>
__global__ void wscale_kernel(
    const T* ustar,S2 su,const T* bfsfc,S2 sb,const T* hbl,S2 sh,
    const T* stable,S2 ss,const int64_t* kbl,S2 sk,const T* zgrid,S1 sz,
    const T* hwide,S1 shw,const T* wmt,S2 swm,const T* wst,S2 sws,
    T* wm_h,T* ws_h,T* wm_l,T* ws_l,T* wm_d,T* ws_d,
    int nz,int64_t ny,int64_t nx,double epsilon,double phepsi,bool sign_zero_positive) {
  const int64_t tid=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t plane=ny*nx, volume=static_cast<int64_t>(nz)*plane;
  if(tid>=volume)return;
  const int k=static_cast<int>(tid/plane); const int64_t col=tid-static_cast<int64_t>(k)*plane;
  const int64_t j=col/nx,i=col-j*nx;
  const double hb=ld2(hbl,sh,j,i), us=ld2(ustar,su,j,i), bf=ld2(bfsfc,sb,j,i), stv=ld2(stable,ss,j,i);
  const int ki=k+1;
  const double sig=(-ld1(zgrid,sz,ki)+0.5*ld1(hwide,shw,ki))/hb;
  const double sigma=stv*sig+(1.0-stv)*(sig>epsilon?epsilon:sig);
  double wm,ws; wscale(sigma,hb,us,bf,wmt,swm,wst,sws,wm,ws);
  st(wm_l,tid,wm); st(ws_l,tid,ws);
  if(k==0){
    const double sigma_h=stv+(1.0-stv)*epsilon;
    wscale(sigma_h,hb,us,bf,wmt,swm,wst,sws,wm,ws);
    wm=fsign(1.0,wm,sign_zero_positive)*clamp_min(fabs(wm),phepsi);
    ws=fsign(1.0,ws,sign_zero_positive)*clamp_min(fabs(ws),phepsi);
    st(wm_h,col,wm);st(ws_h,col,ws);
    const int64_t kb=kbl[j*sk.y+i*sk.x];
    const double sigd=-ld1(zgrid,sz,clampi(kb-1,0,nz+1))/hb;
    const double sigmad=stv*sigd+(1.0-stv)*(sigd>epsilon?epsilon:sigd);
    wscale(sigmad,hb,us,bf,wmt,swm,wst,sws,wm,ws);
    st(wm_d,col,wm);st(ws_d,col,ws);
  }
}

template <class T>
__global__ void finalize_kernel(
    const T* dkm1,const T* hbl,S2 sh,const int64_t* kbl,S2 sk,
    const T* casea,S2 sc,const T* diffus,const T* blmc,const T* ghat,
    const T* zgrid,S1 sz,const T* mask,S3 sm,const T* bg_s,S3 sbgs,
    const T* bg_t,S3 sbgt,T* visc,T* kzs,T* kzt,T* gout,
    int nz,int64_t ny,int64_t nx,double visc_bg) {
  const int64_t tid=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t plane=ny*nx,volume=static_cast<int64_t>(nz)*plane;if(tid>=volume)return;
  const int k=static_cast<int>(tid/plane);const int64_t col=tid-static_cast<int64_t>(k)*plane;
  const int64_t j=col/nx,i=col-j*nx,kb=kbl[j*sk.y+i*sk.x];
  const double cellmask=ld3(mask,sm,k,j,i)*ld3(mask,sm,k>0?k-1:0,j,i);
  const double hb=ld2(hbl,sh,j,i),ca=ld2(casea,sc,j,i);
  for(int m=0;m<3;++m){
    double value=diffus[(static_cast<int64_t>(m)*(nz+2)+k)*plane+col];
    if(k>=1 && k<kb){
      double bl=blmc[(static_cast<int64_t>(m)*nz+k-1)*plane+col];
      if(k==kb-1 && k<nz){
        const double delta=(hb+ld1(zgrid,sz,k))/(ld1(zgrid,sz,k)-ld1(zgrid,sz,k+1));
        const double dkmp5=blend2(ca,value,1.0-ca,bl);
        const double om=1.0-delta;
        // Explicit squares, not pow(x, 2.0).  The repo's FP-bitwise rule
        // (kpp.py:678-680) requires integer powers to be written as
        // multiplication, and the eager reference's ``** 2`` takes ATen's
        // exponent-2 fast path (x*x), whereas CUDA ``pow`` is only accurate to
        // ~2 ULP for a runtime exponent.  Sole caller passed 2.0, so the
        // exponent parameter is gone with it.
        const double om_squared=__dmul_rn(om,om);
        const double delta_squared=__dmul_rn(delta,delta);
        const double dstar=blend2(om_squared,dkm1[static_cast<int64_t>(m)*plane+col],
                                  delta_squared,dkmp5);
        bl=blend2(om,value,delta,dstar);
      }
      const double bg=m==0?visc_bg:(m==1?ld3(bg_s,sbgs,nz-1,j,i):ld3(bg_t,sbgt,nz-1,j,i));
      value=bl>bg?bl:bg;
    }
    T* out=m==0?visc:(m==1?kzs:kzt);st(out,tid,value*cellmask);
  }
  double gv=0.0;
  const int ki=k+1;
  if(ki<kb){
    gv=ghat[tid];
    if(ki==kb-1 && ki<nz) gv=(1.0-ca)*gv;
  }
  st(gout,tid,gv*cellmask);
}

#define KPP_LAUNCH_1D(kernel, count, ...) do {                              \
  constexpr int block = 256;                                               \
  const int64_t grids = ((count) + block - 1) / block;                     \
  kernel<<<static_cast<unsigned int>(grids), block, 0,                     \
           at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__);               \
  C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
} while (false)

} // namespace

void kpp_preprocess_launch(
    const torch::Tensor& dbloc,const torch::Tensor& dbsfc,const torch::Tensor& rho,
    const torch::Tensor& alpha,const torch::Tensor& beta,const torch::Tensor& u,
    const torch::Tensor& v,const torch::Tensor& sft,const torch::Tensor& sfs,
    const torch::Tensor& sfu,const torch::Tensor& sfv,const torch::Tensor& qsw,
    const torch::Tensor& mask,const torch::Tensor& kmtj,const torch::Tensor& zgrid,
    const torch::Tensor& layer_thickness,
    torch::Tensor& dbloc_sm,
    torch::Tensor& dbloc_masked,torch::Tensor& ritop,torch::Tensor& dvsq,
    torch::Tensor& shsq,torch::Tensor& ustar,torch::Tensor& bo,torch::Tensor& bosol,
    double gravity,double rho_const,double phepsi,double heat_capacity){
  c10::cuda::CUDAGuard guard(dbloc.device());const auto nz=dbloc.size(0),ny=dbloc.size(1),nx=dbloc.size(2);
  AT_DISPATCH_FLOATING_TYPES(dbloc.scalar_type(),"kpp_preprocess",[&]{
    KPP_LAUNCH_1D(preprocess_kernel<scalar_t>,nz*ny*nx,
      dbloc.data_ptr<scalar_t>(),S3{dbloc.stride(0),dbloc.stride(1),dbloc.stride(2)},
      dbsfc.data_ptr<scalar_t>(),S3{dbsfc.stride(0),dbsfc.stride(1),dbsfc.stride(2)},
      rho.data_ptr<scalar_t>(),S2{rho.stride(0),rho.stride(1)},alpha.data_ptr<scalar_t>(),S3{alpha.stride(0),alpha.stride(1),alpha.stride(2)},
      beta.data_ptr<scalar_t>(),S3{beta.stride(0),beta.stride(1),beta.stride(2)},u.data_ptr<scalar_t>(),S3{u.stride(0),u.stride(1),u.stride(2)},
      v.data_ptr<scalar_t>(),S3{v.stride(0),v.stride(1),v.stride(2)},sft.data_ptr<scalar_t>(),S2{sft.stride(0),sft.stride(1)},
      sfs.data_ptr<scalar_t>(),S2{sfs.stride(0),sfs.stride(1)},sfu.data_ptr<scalar_t>(),S2{sfu.stride(0),sfu.stride(1)},
      sfv.data_ptr<scalar_t>(),S2{sfv.stride(0),sfv.stride(1)},qsw.data_ptr<scalar_t>(),S2{qsw.stride(0),qsw.stride(1)},
      mask.data_ptr<scalar_t>(),S3{mask.stride(0),mask.stride(1),mask.stride(2)},kmtj.data_ptr<scalar_t>(),S2{kmtj.stride(0),kmtj.stride(1)},
      zgrid.data_ptr<scalar_t>(),S1{zgrid.stride(0)},
      layer_thickness.data_ptr<scalar_t>(),S1{layer_thickness.stride(0)},
      dbloc_sm.data_ptr<scalar_t>(),dbloc_masked.data_ptr<scalar_t>(),ritop.data_ptr<scalar_t>(),dvsq.data_ptr<scalar_t>(),shsq.data_ptr<scalar_t>(),
      ustar.data_ptr<scalar_t>(),bo.data_ptr<scalar_t>(),bosol.data_ptr<scalar_t>(),static_cast<int>(nz),ny,nx,gravity,rho_const,phepsi,heat_capacity);
  });
}

void kpp_ri_launch(const torch::Tensor& kmtj,const torch::Tensor& shsq,const torch::Tensor& dbloc,
 const torch::Tensor& dbloc_sm,const torch::Tensor& bg_s,const torch::Tensor& bg_t,const torch::Tensor& zgrid,
 torch::Tensor& diffus,double ri_inf,double bvsq,double dm0,double ds0,double dt0,double dmc,double dsc,double dtc,double visc,double eps,
 double inv_ri_inf,double inv_bvsq,torch::Tensor& dbg_tap){
 c10::cuda::CUDAGuard guard(shsq.device());const auto nz=shsq.size(0),ny=shsq.size(1),nx=shsq.size(2);
 const bool dbg=dbg_tap.defined();
 if(dbg) TORCH_CHECK(dbg_tap.size(0)==KPP_RI_NTAP,"ri tap slot count mismatch");
 AT_DISPATCH_FLOATING_TYPES(shsq.scalar_type(),"kpp_ri",[&]{KPP_LAUNCH_1D(ri_kernel<scalar_t>,(nz+2)*ny*nx,
 kmtj.data_ptr<int64_t>(),S2{kmtj.stride(0),kmtj.stride(1)},shsq.data_ptr<scalar_t>(),S3{shsq.stride(0),shsq.stride(1),shsq.stride(2)},
 dbloc.data_ptr<scalar_t>(),S3{dbloc.stride(0),dbloc.stride(1),dbloc.stride(2)},dbloc_sm.data_ptr<scalar_t>(),S3{dbloc_sm.stride(0),dbloc_sm.stride(1),dbloc_sm.stride(2)},
 bg_s.data_ptr<scalar_t>(),S3{bg_s.stride(0),bg_s.stride(1),bg_s.stride(2)},bg_t.data_ptr<scalar_t>(),S3{bg_t.stride(0),bg_t.stride(1),bg_t.stride(2)},
 zgrid.data_ptr<scalar_t>(),S1{zgrid.stride(0)},diffus.data_ptr<scalar_t>(),static_cast<int>(nz),ny,nx,ri_inf,bvsq,dm0,ds0,dt0,dmc,dsc,dtc,visc,eps,inv_ri_inf,inv_bvsq,
 dbg?dbg_tap.data_ptr<scalar_t>():nullptr);});
}

void kpp_wscale_launch(const torch::Tensor& ustar,const torch::Tensor& bfsfc,const torch::Tensor& hbl,const torch::Tensor& stable,
 const torch::Tensor& kbl,const torch::Tensor& zgrid,const torch::Tensor& hwide,const torch::Tensor& wmt,const torch::Tensor& wst,
 torch::Tensor& wm_h,torch::Tensor& ws_h,torch::Tensor& wm_l,torch::Tensor& ws_l,torch::Tensor& wm_d,torch::Tensor& ws_d,double epsilon,double phepsi,
 bool sign_zero_positive){
 c10::cuda::CUDAGuard guard(ustar.device());const auto nz=wm_l.size(0),ny=ustar.size(0),nx=ustar.size(1);
 AT_DISPATCH_FLOATING_TYPES(ustar.scalar_type(),"kpp_wscale",[&]{KPP_LAUNCH_1D(wscale_kernel<scalar_t>,nz*ny*nx,
 ustar.data_ptr<scalar_t>(),S2{ustar.stride(0),ustar.stride(1)},bfsfc.data_ptr<scalar_t>(),S2{bfsfc.stride(0),bfsfc.stride(1)},hbl.data_ptr<scalar_t>(),S2{hbl.stride(0),hbl.stride(1)},
 stable.data_ptr<scalar_t>(),S2{stable.stride(0),stable.stride(1)},kbl.data_ptr<int64_t>(),S2{kbl.stride(0),kbl.stride(1)},zgrid.data_ptr<scalar_t>(),S1{zgrid.stride(0)},
 hwide.data_ptr<scalar_t>(),S1{hwide.stride(0)},wmt.data_ptr<scalar_t>(),S2{wmt.stride(0),wmt.stride(1)},wst.data_ptr<scalar_t>(),S2{wst.stride(0),wst.stride(1)},
 wm_h.data_ptr<scalar_t>(),ws_h.data_ptr<scalar_t>(),wm_l.data_ptr<scalar_t>(),ws_l.data_ptr<scalar_t>(),wm_d.data_ptr<scalar_t>(),ws_d.data_ptr<scalar_t>(),static_cast<int>(nz),ny,nx,epsilon,phepsi,
 sign_zero_positive);});
}

void kpp_finalize_launch(const torch::Tensor& dkm1,const torch::Tensor& hbl,const torch::Tensor& kbl,const torch::Tensor& casea,
 const torch::Tensor& diffus,const torch::Tensor& blmc,const torch::Tensor& ghat,const torch::Tensor& zgrid,const torch::Tensor& mask,
 const torch::Tensor& bg_s,const torch::Tensor& bg_t,torch::Tensor& visc,torch::Tensor& kzs,torch::Tensor& kzt,torch::Tensor& gout,double visc_bg){
 c10::cuda::CUDAGuard guard(hbl.device());const auto nz=ghat.size(0),ny=hbl.size(0),nx=hbl.size(1);
 AT_DISPATCH_FLOATING_TYPES(hbl.scalar_type(),"kpp_finalize",[&]{KPP_LAUNCH_1D(finalize_kernel<scalar_t>,nz*ny*nx,
 dkm1.data_ptr<scalar_t>(),hbl.data_ptr<scalar_t>(),S2{hbl.stride(0),hbl.stride(1)},kbl.data_ptr<int64_t>(),S2{kbl.stride(0),kbl.stride(1)},casea.data_ptr<scalar_t>(),S2{casea.stride(0),casea.stride(1)},
 diffus.data_ptr<scalar_t>(),blmc.data_ptr<scalar_t>(),ghat.data_ptr<scalar_t>(),zgrid.data_ptr<scalar_t>(),S1{zgrid.stride(0)},mask.data_ptr<scalar_t>(),S3{mask.stride(0),mask.stride(1),mask.stride(2)},
 bg_s.data_ptr<scalar_t>(),S3{bg_s.stride(0),bg_s.stride(1),bg_s.stride(2)},bg_t.data_ptr<scalar_t>(),S3{bg_t.stride(0),bg_t.stride(1),bg_t.stride(2)},
 visc.data_ptr<scalar_t>(),kzs.data_ptr<scalar_t>(),kzt.data_ptr<scalar_t>(),gout.data_ptr<scalar_t>(),static_cast<int>(nz),ny,nx,visc_bg);});
}
