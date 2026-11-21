#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kColsPerBlock = 32;

template <typename scalar_t>
__global__ void cg3d_thomas_apply_batch1_kernel(
    const scalar_t* __restrict__ zMC,
    const scalar_t* __restrict__ zML,
    const scalar_t* __restrict__ zMU,
    const scalar_t* __restrict__ r,
    scalar_t* __restrict__ q,
    int nz,
    int ncols) {
  const int lane = threadIdx.x;
  const int col = static_cast<int>(blockIdx.x) * kColsPerBlock + lane;
  extern __shared__ scalar_t shared[];
  scalar_t* zmc_s = shared;
  scalar_t* r_s = zmc_s + kColsPerBlock * nz;
  scalar_t* q_s = r_s + kColsPerBlock * nz;

  for (int k = 0; k < nz; ++k) {
    const int sidx = k * kColsPerBlock + lane;
    if (col < ncols) {
      const int gidx = k * ncols + col;
      zmc_s[sidx] = zMC[gidx];
      r_s[sidx] = r[gidx];
    } else {
      zmc_s[sidx] = static_cast<scalar_t>(0);
      r_s[sidx] = static_cast<scalar_t>(0);
    }
  }
  __syncthreads();

  if (col >= ncols) {
    return;
  }

  q_s[lane] = zmc_s[lane] * r_s[lane];
  for (int k = 1; k < nz; ++k) {
    const int sidx = k * kColsPerBlock + lane;
    const int prev = sidx - kColsPerBlock;
    const int gidx = k * ncols + col;
    q_s[sidx] = zmc_s[sidx] * (r_s[sidx] - zML[gidx] * q_s[prev]);
  }

  for (int k = nz - 2; k >= 0; --k) {
    const int sidx = k * kColsPerBlock + lane;
    const int next = sidx + kColsPerBlock;
    const int gidx = k * ncols + col;
    q_s[sidx] = q_s[sidx] - zMU[gidx] * q_s[next];
  }

  for (int k = 0; k < nz; ++k) {
    q[k * ncols + col] = q_s[k * kColsPerBlock + lane];
  }
}

template <typename scalar_t>
__global__ void cg3d_thomas_apply_kernel(
    const scalar_t* __restrict__ zMC,
    const scalar_t* __restrict__ zML,
    const scalar_t* __restrict__ zMU,
    const scalar_t* __restrict__ r,
    scalar_t* __restrict__ q,
    int64_t batch,
    int nz,
    int64_t ny,
    int64_t nx) {
  const int lane = threadIdx.x;
  const int64_t nplane = ny * nx;
  const int64_t total_cols = batch * nplane;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * kColsPerBlock + lane;
  extern __shared__ scalar_t shared[];
  scalar_t* zmc_s = shared;
  scalar_t* r_s = zmc_s + kColsPerBlock * nz;
  scalar_t* q_s = r_s + kColsPerBlock * nz;

  int64_t b = 0;
  int64_t plane_col = 0;
  if (col < total_cols) {
    if (batch == 1) {
      plane_col = col;
    } else {
      b = col / nplane;
      plane_col = col - b * nplane;
    }
  }

  for (int k = 0; k < nz; ++k) {
    const int sidx = k * kColsPerBlock + lane;
    if (col < total_cols) {
      const int64_t factor_idx = static_cast<int64_t>(k) * nplane + plane_col;
      const int64_t r_idx = b * nz * nplane + static_cast<int64_t>(k) * nplane + plane_col;
      zmc_s[sidx] = zMC[factor_idx];
      r_s[sidx] = r[r_idx];
    } else {
      zmc_s[sidx] = static_cast<scalar_t>(0);
      r_s[sidx] = static_cast<scalar_t>(0);
    }
  }
  __syncthreads();

  if (col >= total_cols) {
    return;
  }

  q_s[lane] = zmc_s[lane] * r_s[lane];
  for (int k = 1; k < nz; ++k) {
    const int sidx = k * kColsPerBlock + lane;
    const int prev = sidx - kColsPerBlock;
    const int64_t factor_idx = static_cast<int64_t>(k) * nplane + plane_col;
    q_s[sidx] = zmc_s[sidx] * (r_s[sidx] - zML[factor_idx] * q_s[prev]);
  }

  for (int k = nz - 2; k >= 0; --k) {
    const int sidx = k * kColsPerBlock + lane;
    const int next = sidx + kColsPerBlock;
    const int64_t factor_idx = static_cast<int64_t>(k) * nplane + plane_col;
    q_s[sidx] = q_s[sidx] - zMU[factor_idx] * q_s[next];
  }

  for (int k = 0; k < nz; ++k) {
    const int64_t q_idx = b * nz * nplane + static_cast<int64_t>(k) * nplane + plane_col;
    q[q_idx] = q_s[k * kColsPerBlock + lane];
  }
}

}  // namespace

void cg3d_thomas_apply_launch(
    const torch::Tensor& zMC,
    const torch::Tensor& zML,
    const torch::Tensor& zMU,
    const torch::Tensor& r,
    torch::Tensor& q) {
  const auto batch = static_cast<int64_t>(r.size(0));
  const auto nz = static_cast<int>(r.size(1));
  const auto ny = static_cast<int64_t>(r.size(2));
  const auto nx = static_cast<int64_t>(r.size(3));
  const auto total_cols = batch * ny * nx;

  c10::cuda::CUDAGuard device_guard(r.device());
  const dim3 block(kColsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>((total_cols + kColsPerBlock - 1) / kColsPerBlock));
  const size_t shmem =
      static_cast<size_t>(3) * kColsPerBlock * static_cast<size_t>(nz) * sizeof(double);

  if (batch == 1 && total_cols <= std::numeric_limits<int>::max()) {
    if (shmem > 48 * 1024) {
      cudaError_t attr_status = cudaFuncSetAttribute(
          cg3d_thomas_apply_batch1_kernel<double>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(shmem));
      TORCH_CHECK(
          attr_status == cudaSuccess,
          "cudaFuncSetAttribute failed: ",
          cudaGetErrorString(attr_status));
    }

    cg3d_thomas_apply_batch1_kernel<double>
        <<<grid, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
            zMC.data_ptr<double>(),
            zML.data_ptr<double>(),
            zMU.data_ptr<double>(),
            r.data_ptr<double>(),
            q.data_ptr<double>(),
            nz,
            static_cast<int>(total_cols));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  if (shmem > 48 * 1024) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cg3d_thomas_apply_kernel<double>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    TORCH_CHECK(
        attr_status == cudaSuccess,
        "cudaFuncSetAttribute failed: ",
        cudaGetErrorString(attr_status));
  }

  cg3d_thomas_apply_kernel<double>
      <<<grid, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
          zMC.data_ptr<double>(),
          zML.data_ptr<double>(),
          zMU.data_ptr<double>(),
          r.data_ptr<double>(),
          q.data_ptr<double>(),
          batch,
          nz,
          ny,
          nx);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
