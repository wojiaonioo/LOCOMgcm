// Fused CUDA kernels for the CG2D preconditioned-conjugate-gradient body.
//
// Semantic authority = ``defgcm/solvers/cg2d.py::pcg_solve`` (the eager
// loop, lines 794-837) together with the two expressions it calls out to:
//   * the matvec, ``operators/c_grid.py::edge_coefficient_helmholtz_from_haloed_center``
//     (five statements, nine rounded elementwise ops), and
//   * the preconditioner apply, ``solvers/cg2d_preconditioner.py``
//     (``apply_cg2d_jacobi_preconditioner`` = one multiply,
//      ``apply_cg2d_ssor_preconditioner_wrapped`` = a multiply plus four
//      sliced in-place ``+=`` and, on a wrapped axis, two edge ``+=``;
//      ``apply_cg2d_ssor_preconditioner_edge`` = the K1 stencil read from the
//      exchanged residual, which the dispatcher serves with K1 itself -- on
//      the tile (``ssor_halo``), or on the tile plus a one-cell ring from a
//      width-2 residual halo (``ssor_ring``), whose ``p`` update is K2c).
//
// Every kernel below reproduces the eager result BIT FOR BIT.  Three things
// make that true and all three are load-bearing:
//   1. ``-fmad=false`` (see ``_fast.py::_CUDA_FLAGS``).  nvcc contracts
//      ``a + b * c`` into one FMA by default, which rounds ONCE; the eager
//      reference launches a multiply kernel and then an add kernel, which
//      rounds TWICE.  Without the flag every accumulation below differs from
//      eager in the last place.
//   2. The accumulation order is the reference's statement order, not a
//      "mathematically equivalent" regrouping.  Floating-point addition is not
//      associative, so the order IS the result.
//   3. No division appears in any kernel.  ``alpha = rz / pAp`` and
//      ``beta = rz_new / rz`` stay on the host as the eager 0-d tensor
//      divisions and arrive here as device pointers, read as ``alpha[0]``, so
//      the host never synchronises to fetch their value.  (A tensor divided by
//      a *Python scalar* is a reciprocal-multiply in eager CUDA, but both of
//      these are tensor/tensor, i.e. a true division -- reproducing them in a
//      kernel would be correct too, and is simply not needed here.)
//
// There is no mask argument anywhere in this file, and that is not an
// omission: unlike the CG3D body, the CG2D iteration has no ``* mask`` factor.
// The land/wet distinction is baked into the assembled coefficients
// (``aC``/``aW_edge``/``aS_edge`` vanish on closed faces and dry centres), so a
// dry row of the operator is the zero row and stays zero through the whole
// iteration.  Adding a mask multiply here would change the arithmetic.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/Dispatch.h>

#include <cstdint>
#include <limits>

#include "../_cuda_launch/launch_config.h"

namespace {

defgcm::cuda_launch::LaunchConfigCache launch_config_cache;

//: Every kernel here is register/thread bound; none allocates dynamic shared
//: memory.  The occupancy API still wants the callback.
size_t dynamic_shared_bytes_for_block(int) {
  return 0;
}

constexpr int kReferenceBlockSize = 256;

// ---------------------------------------------------------------------------
// K1: the five-point Helmholtz matvec, from an ALREADY-EXCHANGED centre halo.
//
// Reference (c_grid.py:288-296, halo == 1):
//     result = aC * inner
//     result = result + aW_w * west + aW_e * east
//     result = result + aS_s * south + aS_n * north
// which evaluates as five multiplies and four adds in exactly this order:
//     t = aC*inner; t = t + aW_w*west; t = t + aW_e*east;
//                   t = t + aS_s*south; t = t + aS_n*north
// The halo supplies the out-of-range neighbours, so there is no conditional
// here -- the reference's slices read the halo cells too.
//   phi_halo : [(ny+2) * (nx+2)]   aW : [ny * (nx+1)]
//   aS       : [(ny+1) * nx]       aC : [ny * nx]   out : [ny * nx]
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void cg2d_spmv_kernel(
    const scalar_t* __restrict__ phi_halo,
    const scalar_t* __restrict__ aW,
    const scalar_t* __restrict__ aS,
    const scalar_t* __restrict__ aC,
    scalar_t* __restrict__ out,
    int ny,
    int nx) {
  const int64_t count = static_cast<int64_t>(ny) * nx;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int i = static_cast<int>(index % nx);
  const int j = static_cast<int>(index / nx);
  const int64_t halo_width = static_cast<int64_t>(nx) + 2;
  const int64_t centre = static_cast<int64_t>(j + 1) * halo_width + (i + 1);
  const int64_t west_row = static_cast<int64_t>(j) * (static_cast<int64_t>(nx) + 1);
  const int64_t south_row = static_cast<int64_t>(j) * nx;

  scalar_t accumulator = aC[index] * phi_halo[centre];
  accumulator = accumulator + aW[west_row + i] * phi_halo[centre - 1];
  accumulator = accumulator + aW[west_row + i + 1] * phi_halo[centre + 1];
  accumulator = accumulator + aS[south_row + i] * phi_halo[centre - halo_width];
  accumulator =
      accumulator + aS[south_row + nx + i] * phi_halo[centre + halo_width];
  out[index] = accumulator;
}

// ---------------------------------------------------------------------------
// K2a: the state update without a preconditioner (cg2d.py:801-802).
//     x = x + alpha * p
//     r = r - alpha * Ap
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void cg2d_update_xr_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ p,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ ap,
    const scalar_t* __restrict__ alpha,
    scalar_t* __restrict__ x_out,
    scalar_t* __restrict__ r_out,
    int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const scalar_t a = alpha[0];
  x_out[index] = x[index] + a * p[index];
  r_out[index] = r[index] - a * ap[index];
}

// ---------------------------------------------------------------------------
// K2c: K2a for a search direction carried with a one-cell ring
// (``cg2d.py::pcg_solve`` with ``ring_matvec``: ``x = x + alpha * owned(p)``).
// ``p_ring`` is ``[(ny+2) * (nx+2)]`` and cell (j, i) of the tile sits at
// ``(j+1)*(nx+2) + (i+1)``, i.e. ``p`` is read at row stride nx+2; the two
// statements and their rounding are K2a's.  A separate kernel rather than a
// stride parameter on K2a, so the unringed update stays the kernel it was.
//   x, r, ap, x_out, r_out : [ny * nx]
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void cg2d_update_xr_ring_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ p_ring,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ ap,
    const scalar_t* __restrict__ alpha,
    scalar_t* __restrict__ x_out,
    scalar_t* __restrict__ r_out,
    int ny,
    int nx) {
  const int64_t count = static_cast<int64_t>(ny) * nx;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int i = static_cast<int>(index % nx);
  const int j = static_cast<int>(index / nx);
  const int64_t ring =
      static_cast<int64_t>(j + 1) * (static_cast<int64_t>(nx) + 2) + (i + 1);
  const scalar_t a = alpha[0];
  x_out[index] = x[index] + a * p_ring[ring];
  r_out[index] = r[index] - a * ap[index];
}

// ---------------------------------------------------------------------------
// K2b: the state update with the Jacobi preconditioner folded in
// (cg2d.py:801-802 plus cg2d.py:830 through
// ``apply_cg2d_jacobi_preconditioner``, which is ``scale * residual``).
//
// The eager loop applies the preconditioner AFTER the convergence test, so on
// the final iteration this kernel computes a ``z`` the loop then discards.
// That is wasted work, never a different number: ``z`` is a pure function of
// the new residual, and the discarded value never reaches the state.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void cg2d_update_xrz_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ p,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ ap,
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ precond_inv,
    scalar_t* __restrict__ x_out,
    scalar_t* __restrict__ r_out,
    scalar_t* __restrict__ z_out,
    int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const scalar_t a = alpha[0];
  const scalar_t r_new = r[index] - a * ap[index];
  x_out[index] = x[index] + a * p[index];
  r_out[index] = r_new;
  z_out[index] = precond_inv[index] * r_new;
}

// ---------------------------------------------------------------------------
// K3: the regional SSOR preconditioner apply on one process's array
// (``cg2d_preconditioner.py::apply_cg2d_ssor_preconditioner_wrapped``).  The
// reference is ``z = pC * r`` followed by the west, east, south and north adds:
// on an unwrapped axis the sliced adds of the tile-local apply,
//     z[..., :, 1: ] += pW[..., :, 1: ] * r[..., :, :-1]
//     z[..., :, :-1] += pW[..., :, 1: ] * r[..., :, 1: ]
// on a wrapped axis one whole-array add per direction,
//     z += pW * roll(r, +1, x)          z += roll(pW * r, -1, x)
// (and the same along y with pS).
// Read per cell (i along x, j along y) both forms are
//     z(j,i)  = pC(j,i)*r(j,i)
//     i>=1    : += pW(j,i)   * r(j,i-1)     i==0 & wrap x    : += pW(j,0) * r(j,nx-1)
//     i<=nx-2 : += pW(j,i+1) * r(j,i+1)     i==nx-1 & wrap x : += pW(j,0) * r(j,0)
//     j>=1    : += pS(j,i)   * r(j-1,i)     j==0 & wrap y    : += pS(0,i) * r(ny-1,i)
//     j<=ny-2 : += pS(j+1,i) * r(j+1,i)     j==ny-1 & wrap y : += pS(0,i) * r(0,i)
// in that order (MIT cg2d.F:216-222: centre, west, east, south, north), with
// the bounds conditions standing in for the sliced assignments.  A wrapped
// axis is a periodic axis the process owns whole: the edge between cell n-1
// and cell 0 is cell 0's west/south edge, which is MIT's single-tile exchange
// ``pW(sNx+1) = pW(1)``.  An unwrapped edge contributes nothing, as across a
// wall or an open boundary.  Rank seams are not served here: their residual
// arrives through the exchange and the apply is the K1 stencil with
// (pC, pW_edge, pS_edge) in place of (aC, aW_edge, aS_edge) (dispatcher
// ``ssor_halo``).
//
// The wrap flags are template parameters, not arguments: the unwrapped
// instantiation is the pre-wrap kernel with nothing added to its body, and a
// wrapped one tests its flag at compile time, so neither pays a runtime test
// of a flag in every thread.  Every wrap read stays inside the arrays: with
// index = j*nx + i < ny*nx, ``index + nx - 1`` (i == 0) is at most
// j*nx + nx - 1, ``index - (nx - 1)`` (i == nx-1) is j*nx, ``index + (ny-1)*nx``
// (j == 0) is at most ny*nx - 1, and ``i`` (j == ny-1) is below nx.
// ---------------------------------------------------------------------------
template <typename scalar_t, bool kWrapX, bool kWrapY>
__global__ void cg2d_ssor_precond_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ pC,
    const scalar_t* __restrict__ pW,
    const scalar_t* __restrict__ pS,
    scalar_t* __restrict__ out,
    int ny,
    int nx) {
  const int64_t count = static_cast<int64_t>(ny) * nx;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int i = static_cast<int>(index % nx);
  const int j = static_cast<int>(index / nx);

  scalar_t accumulator = pC[index] * r[index];
  if (i >= 1) {
    accumulator = accumulator + pW[index] * r[index - 1];
  } else if (kWrapX) {
    accumulator = accumulator + pW[index] * r[index + nx - 1];
  }
  if (i <= nx - 2) {
    accumulator = accumulator + pW[index + 1] * r[index + 1];
  } else if (kWrapX) {
    const int64_t row_start = index - (nx - 1);
    accumulator = accumulator + pW[row_start] * r[row_start];
  }
  if (j >= 1) {
    accumulator = accumulator + pS[index] * r[index - nx];
  } else if (kWrapY) {
    accumulator =
        accumulator + pS[index] * r[index + static_cast<int64_t>(ny - 1) * nx];
  }
  if (j <= ny - 2) {
    accumulator = accumulator + pS[index + nx] * r[index + nx];
  } else if (kWrapY) {
    accumulator = accumulator + pS[i] * r[i];
  }
  out[index] = accumulator;
}

// ---------------------------------------------------------------------------
// K4: the search-direction update (cg2d.py:837), ``p = z + beta * p``.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void cg2d_axpy_p_kernel(
    const scalar_t* __restrict__ z,
    const scalar_t* __restrict__ p,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ out,
    int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const scalar_t b = beta[0];
  out[index] = z[index] + b * p[index];
}

unsigned int block_count_for(int64_t elements, int threads_per_block) {
  const int64_t blocks = (elements + threads_per_block - 1) / threads_per_block;
  TORCH_CHECK(
      blocks <= std::numeric_limits<unsigned int>::max(),
      "CG2D fused grid exceeds grid.x capacity");
  return static_cast<unsigned int>(blocks);
}

}  // namespace

void cg2d_spmv_launch(
    const torch::Tensor& phi_halo,
    const torch::Tensor& aW,
    const torch::Tensor& aS,
    const torch::Tensor& aC,
    torch::Tensor& out,
    int ny,
    int nx) {
  c10::cuda::CUDAGuard device_guard(phi_halo.device());
  const int64_t count = static_cast<int64_t>(ny) * nx;
  AT_DISPATCH_FLOATING_TYPES(phi_halo.scalar_type(), "cg2d_spmv_launch", [&] {
    const auto& plan = launch_config_cache.get(
        cg2d_spmv_kernel<scalar_t>,
        phi_halo.get_device(),
        0,
        kReferenceBlockSize,
        dynamic_shared_bytes_for_block);
    const int threads = plan.selected_block_size;
    cg2d_spmv_kernel<scalar_t>
        <<<dim3(block_count_for(count, threads)), dim3(threads), 0,
           at::cuda::getCurrentCUDAStream()>>>(
            phi_halo.data_ptr<scalar_t>(),
            aW.data_ptr<scalar_t>(),
            aS.data_ptr<scalar_t>(),
            aC.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            ny,
            nx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void cg2d_update_xr_launch(
    const torch::Tensor& x,
    const torch::Tensor& p,
    const torch::Tensor& r,
    const torch::Tensor& ap,
    const torch::Tensor& alpha,
    torch::Tensor& x_out,
    torch::Tensor& r_out) {
  c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t count = x.numel();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cg2d_update_xr_launch", [&] {
    const auto& plan = launch_config_cache.get(
        cg2d_update_xr_kernel<scalar_t>,
        x.get_device(),
        0,
        kReferenceBlockSize,
        dynamic_shared_bytes_for_block);
    const int threads = plan.selected_block_size;
    cg2d_update_xr_kernel<scalar_t>
        <<<dim3(block_count_for(count, threads)), dim3(threads), 0,
           at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            p.data_ptr<scalar_t>(),
            r.data_ptr<scalar_t>(),
            ap.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            x_out.data_ptr<scalar_t>(),
            r_out.data_ptr<scalar_t>(),
            count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void cg2d_update_xr_ring_launch(
    const torch::Tensor& x,
    const torch::Tensor& p_ring,
    const torch::Tensor& r,
    const torch::Tensor& ap,
    const torch::Tensor& alpha,
    torch::Tensor& x_out,
    torch::Tensor& r_out,
    int ny,
    int nx) {
  c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t count = static_cast<int64_t>(ny) * nx;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cg2d_update_xr_ring_launch", [&] {
    const auto& plan = launch_config_cache.get(
        cg2d_update_xr_ring_kernel<scalar_t>,
        x.get_device(),
        0,
        kReferenceBlockSize,
        dynamic_shared_bytes_for_block);
    const int threads = plan.selected_block_size;
    cg2d_update_xr_ring_kernel<scalar_t>
        <<<dim3(block_count_for(count, threads)), dim3(threads), 0,
           at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            p_ring.data_ptr<scalar_t>(),
            r.data_ptr<scalar_t>(),
            ap.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            x_out.data_ptr<scalar_t>(),
            r_out.data_ptr<scalar_t>(),
            ny,
            nx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void cg2d_update_xrz_launch(
    const torch::Tensor& x,
    const torch::Tensor& p,
    const torch::Tensor& r,
    const torch::Tensor& ap,
    const torch::Tensor& alpha,
    const torch::Tensor& precond_inv,
    torch::Tensor& x_out,
    torch::Tensor& r_out,
    torch::Tensor& z_out) {
  c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t count = x.numel();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cg2d_update_xrz_launch", [&] {
    const auto& plan = launch_config_cache.get(
        cg2d_update_xrz_kernel<scalar_t>,
        x.get_device(),
        0,
        kReferenceBlockSize,
        dynamic_shared_bytes_for_block);
    const int threads = plan.selected_block_size;
    cg2d_update_xrz_kernel<scalar_t>
        <<<dim3(block_count_for(count, threads)), dim3(threads), 0,
           at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            p.data_ptr<scalar_t>(),
            r.data_ptr<scalar_t>(),
            ap.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            precond_inv.data_ptr<scalar_t>(),
            x_out.data_ptr<scalar_t>(),
            r_out.data_ptr<scalar_t>(),
            z_out.data_ptr<scalar_t>(),
            count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

namespace {

template <typename scalar_t, bool kWrapX, bool kWrapY>
void launch_ssor_precond(
    const torch::Tensor& r,
    const torch::Tensor& pC,
    const torch::Tensor& pW,
    const torch::Tensor& pS,
    torch::Tensor& out,
    int ny,
    int nx) {
  const int64_t count = static_cast<int64_t>(ny) * nx;
  const auto& plan = launch_config_cache.get(
      cg2d_ssor_precond_kernel<scalar_t, kWrapX, kWrapY>,
      r.get_device(),
      0,
      kReferenceBlockSize,
      dynamic_shared_bytes_for_block);
  const int threads = plan.selected_block_size;
  cg2d_ssor_precond_kernel<scalar_t, kWrapX, kWrapY>
      <<<dim3(block_count_for(count, threads)), dim3(threads), 0,
         at::cuda::getCurrentCUDAStream()>>>(
          r.data_ptr<scalar_t>(),
          pC.data_ptr<scalar_t>(),
          pW.data_ptr<scalar_t>(),
          pS.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),
          ny,
          nx);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void cg2d_ssor_precond_launch(
    const torch::Tensor& r,
    const torch::Tensor& pC,
    const torch::Tensor& pW,
    const torch::Tensor& pS,
    torch::Tensor& out,
    int ny,
    int nx,
    bool wrap_x,
    bool wrap_y) {
  c10::cuda::CUDAGuard device_guard(r.device());
  AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "cg2d_ssor_precond_launch", [&] {
    if (wrap_x && wrap_y) {
      launch_ssor_precond<scalar_t, true, true>(r, pC, pW, pS, out, ny, nx);
    } else if (wrap_x) {
      launch_ssor_precond<scalar_t, true, false>(r, pC, pW, pS, out, ny, nx);
    } else if (wrap_y) {
      launch_ssor_precond<scalar_t, false, true>(r, pC, pW, pS, out, ny, nx);
    } else {
      launch_ssor_precond<scalar_t, false, false>(r, pC, pW, pS, out, ny, nx);
    }
  });
}

void cg2d_axpy_p_launch(
    const torch::Tensor& z,
    const torch::Tensor& p,
    const torch::Tensor& beta,
    torch::Tensor& out) {
  c10::cuda::CUDAGuard device_guard(z.device());
  const int64_t count = z.numel();
  AT_DISPATCH_FLOATING_TYPES(z.scalar_type(), "cg2d_axpy_p_launch", [&] {
    const auto& plan = launch_config_cache.get(
        cg2d_axpy_p_kernel<scalar_t>,
        z.get_device(),
        0,
        kReferenceBlockSize,
        dynamic_shared_bytes_for_block);
    const int threads = plan.selected_block_size;
    cg2d_axpy_p_kernel<scalar_t>
        <<<dim3(block_count_for(count, threads)), dim3(threads), 0,
           at::cuda::getCurrentCUDAStream()>>>(
            z.data_ptr<scalar_t>(),
            p.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

// The launch decision each kernel actually uses, per device and dtype -- the
// same cache entry the launchers above read, so what is reported IS what runs
// (the configurable launch policy: a hard-coded block size
// is a silent failure class, and a reported config derived separately from the
// one used would be the same defect with a nicer log line).
const defgcm::cuda_launch::LaunchConfig& cg2d_fused_launch_config(
    int device, const std::string& kernel, at::ScalarType dtype) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
#define DEFGCM_CG2D_PLAN(...)                                       \
  AT_DISPATCH_FLOATING_TYPES(dtype, "cg2d_fused_launch_config", [&] { \
    selected = &launch_config_cache.get(                        \
        __VA_ARGS__, device, 0, kReferenceBlockSize,            \
        dynamic_shared_bytes_for_block);                        \
  });
  const defgcm::cuda_launch::LaunchConfig* selected = nullptr;
  if (kernel == "spmv") {
    DEFGCM_CG2D_PLAN(cg2d_spmv_kernel<scalar_t>)
  } else if (kernel == "update_xr") {
    DEFGCM_CG2D_PLAN(cg2d_update_xr_kernel<scalar_t>)
  } else if (kernel == "update_xr_ring") {
    DEFGCM_CG2D_PLAN(cg2d_update_xr_ring_kernel<scalar_t>)
  } else if (kernel == "update_xrz") {
    DEFGCM_CG2D_PLAN(cg2d_update_xrz_kernel<scalar_t>)
  } else if (kernel == "ssor_precond") {
    DEFGCM_CG2D_PLAN(cg2d_ssor_precond_kernel<scalar_t, false, false>)
  } else if (kernel == "ssor_precond_wrap_x") {
    DEFGCM_CG2D_PLAN(cg2d_ssor_precond_kernel<scalar_t, true, false>)
  } else if (kernel == "ssor_precond_wrap_y") {
    DEFGCM_CG2D_PLAN(cg2d_ssor_precond_kernel<scalar_t, false, true>)
  } else if (kernel == "ssor_precond_wrap_xy") {
    DEFGCM_CG2D_PLAN(cg2d_ssor_precond_kernel<scalar_t, true, true>)
  } else if (kernel == "axpy_p") {
    DEFGCM_CG2D_PLAN(cg2d_axpy_p_kernel<scalar_t>)
  } else {
    TORCH_CHECK(false, "unknown CG2D fused kernel name: ", kernel);
  }
#undef DEFGCM_CG2D_PLAN
  TORCH_CHECK(selected != nullptr, "CG2D fused launch configuration missing");
  return *selected;
}
