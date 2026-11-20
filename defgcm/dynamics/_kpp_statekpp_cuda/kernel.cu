// Fused KPP STATEKPP (dbloc / dbsfc / rho1 / alpha / beta) in one CUDA kernel.
//
// Numerical contract: BITWISE against the frozen eager reference
// ``defgcm.dynamics.kpp.kpp_statekpp`` evaluated with ``eos_type="jmd95"``.
// Two properties below are load-bearing and must not be "simplified":
//
//  1. This translation unit MUST be compiled with ``-fmad=false`` (declared in
//     ``_fast.py::_CUDA_FLAGS``).  Eager torch evaluates every polynomial term
//     as an independent multiply followed by an independent add, so each term
//     is rounded twice; a contracted FMA rounds once and moves the last bit.
//     dbloc/dbsfc are ill-conditioned buoyancy DIFFERENCES that feed the hbl
//     Rib scan, where a 1-ULP density perturbation amplifies to ~1e-6 m of
//     boundary-layer depth. ``-prec-div=true`` and
//     ``-prec-sqrt=true`` preserve IEEE division and the roots of salt and salt cubed.
//  2. Every sum keeps Python's left-to-right association and every product
//     keeps its operand order. Density and alpha/beta call the same MITgcm
//     surface-density and bulk-modulus expressions.
//
// Compare against kpp_statekpp evaluated on CUDA with the same dtype so that
// the reference and fused paths use the same device math implementations.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/Dispatch.h>

#include <cstdint>
#include <limits>

#include "../../solvers/_cuda_launch/launch_config.h"

#include "local_statekpp.cuh"

namespace {

defgcm::cuda_launch::LaunchConfigCache launch_config_cache;

size_t dynamic_shared_bytes_for_block(int) {
  return 0;
}

using defgcm::local_eos::Jmd95;
using defgcm::local_eos::clamp_and_root;
using defgcm::local_eos::density_anomaly;
using defgcm::local_eos::alpha_beta;

struct Strides4 {
  int64_t batch;
  int64_t level;
  int64_t y;
  int64_t x;
};

Strides4 strides4(const torch::Tensor &value) {
  return {value.stride(0), value.stride(1), value.stride(2), value.stride(3)};
}

template <typename scalar_t>
struct Input4 {
  const scalar_t *data;
  Strides4 strides;
  int64_t batch_size;
  int64_t level_size;
  int64_t y_size;
  int64_t x_size;

  __device__ __forceinline__ scalar_t load(const int64_t batch,
                                         const int64_t level, const int64_t y,
                                         const int64_t x) const {
    const int64_t input_batch = batch_size == 1 ? 0 : batch;
    const int64_t input_level = level_size == 1 ? 0 : level;
    const int64_t input_y = y_size == 1 ? 0 : y;
    const int64_t input_x = x_size == 1 ? 0 : x;
    const int64_t offset = input_batch * strides.batch +
                           input_level * strides.level + input_y * strides.y +
                           input_x * strides.x;
    return __ldg(data + offset);
  }
};

template <typename scalar_t>
Input4<scalar_t> input4(const torch::Tensor &value) {
  return {value.data_ptr<scalar_t>(),
          strides4(value),
          value.size(0),
          value.size(1),
          value.size(2),
          value.size(3)};
}

// One thread per (batch, level, y, x) output cell.  The cell re-evaluates the
// three density points and the one alpha/beta point that kpp.py:197-213 reads,
// so no cross-thread communication and no reduction is involved.
template <typename scalar_t>
__global__ void kpp_statekpp_kernel(
    const Input4<scalar_t> temp, const Input4<scalar_t> salt, const Input4<scalar_t> pressure,
    scalar_t *__restrict__ dbloc, scalar_t *__restrict__ dbsfc,
    scalar_t *__restrict__ alpha_out, scalar_t *__restrict__ beta_out,
    scalar_t *__restrict__ rho1, const int64_t element_count,
    const int64_t levels, const int64_t ny, const int64_t nx,
    const scalar_t gravity, const scalar_t reference_density) {
  using K = Jmd95;
  const int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= element_count) {
    return;
  }

  const int64_t plane = ny * nx;
  const int64_t volume = levels * plane;
  const int64_t batch = output_index / volume;
  const int64_t volume_index = output_index - batch * volume;
  const int64_t level = volume_index / plane;
  const int64_t plane_index = volume_index - level * plane;
  const int64_t y = plane_index / nx;
  const int64_t x = plane_index - y * nx;

  const scalar_t temp_k = temp.load(batch, level, y, x);
  scalar_t salt_k = static_cast<scalar_t>(0.0);
  scalar_t salt_root_k = static_cast<scalar_t>(0.0);
  clamp_and_root(salt.load(batch, level, y, x), salt_k, salt_root_k);
  const scalar_t p_k = pressure.load(batch, level, y, x) * K::si_to_p;

  // kpp.py:197  rho_diag = rho(T_k, S_k, p_k)
  const scalar_t rho_diag =
      density_anomaly<Jmd95>(temp_k, salt_k, salt_root_k, p_k, reference_density);

  // kpp.py:206-207  dbsfc = g*(rho_diag - rho_surf)/(rho_diag + rho_const);
  // dbsfc[0] is overwritten with 0 by the eager code, so the surface level
  // simply skips the (discarded) evaluation.
  scalar_t dbsfc_value = static_cast<scalar_t>(0.0);
  if (level != 0) {
    // kpp.py:199-200  rho_surf = rho(T_0, S_0, p_k)
    const scalar_t temp_surface = temp.load(batch, 0, y, x);
    scalar_t salt_surface = static_cast<scalar_t>(0.0);
    scalar_t salt_root_surface = static_cast<scalar_t>(0.0);
    clamp_and_root(salt.load(batch, 0, y, x), salt_surface, salt_root_surface);
    const scalar_t rho_surf = density_anomaly<Jmd95>(temp_surface, salt_surface,
                                            salt_root_surface, p_k,
                                            reference_density);
    dbsfc_value = gravity * (rho_diag - rho_surf) / (rho_diag + reference_density);
  }
  dbsfc[output_index] = dbsfc_value;

  // kpp.py:202-204  dbloc[k] = g*(rho_diag[k+1] - rho_upper[k])
  //                            / (rho_diag[k+1] + rho_const), dbloc[Nr-1] = 0.
  scalar_t dbloc_value = static_cast<scalar_t>(0.0);
  if (level + 1 < levels) {
    const scalar_t p_next = pressure.load(batch, level + 1, y, x) * K::si_to_p;
    // rho_diag[k+1] = rho(T_{k+1}, S_{k+1}, p_{k+1})
    const scalar_t temp_next = temp.load(batch, level + 1, y, x);
    scalar_t salt_next = static_cast<scalar_t>(0.0);
    scalar_t salt_root_next = static_cast<scalar_t>(0.0);
    clamp_and_root(salt.load(batch, level + 1, y, x), salt_next,
                   salt_root_next);
    const scalar_t rho_below = density_anomaly<Jmd95>(temp_next, salt_next,
                                             salt_root_next, p_next,
                                             reference_density);
    const scalar_t difference = defgcm::local_eos::jmd95_density_difference(
        temp_next, salt_next, salt_root_next, temp_k, salt_k, salt_root_k, p_next);
    dbloc_value = gravity * difference / (rho_below + reference_density);
  }
  dbloc[output_index] = dbloc_value;

  // kpp.py:209  rho1 = rho_diag[0] + rho_const
  if (level == 0) {
    rho1[batch * plane + plane_index] = rho_diag + reference_density;
  }

  // kpp.py:211  alpha_nr, beta_nr = jmd95_alpha_beta(theta, salt, p)
  scalar_t alpha_value = static_cast<scalar_t>(0.0);
  scalar_t beta_value = static_cast<scalar_t>(0.0);
  alpha_beta<Jmd95>(temp_k, salt_k, salt_root_k, p_k, alpha_value, beta_value);
  alpha_out[output_index] = alpha_value;
  beta_out[output_index] = beta_value;
}

struct StatekppColumn {
  Input4<double> temperature, salinity, pressure_pa;
  Input4<double> seeds[5];
  double *gt, *gs, *gp;
  int64_t batch, y, x, levels, ny, nx;
  __device__ double temp(int64_t k) const { return temperature.load(batch, k, y, x); }
  __device__ double salt(int64_t k) const { return salinity.load(batch, k, y, x); }
  __device__ double pressure(int64_t k) const { return pressure_pa.load(batch, k, y, x); }
  __device__ double seed(int output, int64_t k) const {
    return seeds[output].data ? seeds[output].load(batch, k, y, x) : 0.0;
  }
  __device__ int64_t offset(int64_t k) const {
    return ((batch * levels + k) * ny + y) * nx + x;
  }
  __device__ void write(int64_t k, double t, double s, double p) {
    const int64_t index = offset(k);
    if (gt) gt[index] = t;
    if (gs) gs[index] = s;
    if (gp) gp[index] = p;
  }
  __device__ void add_surface(double t, double s) {
    if (gt) gt[offset(0)] += t;
    if (gs) gs[offset(0)] += s;
  }
};

__global__ void statekpp_vjp_kernel(StatekppColumn column, int64_t count,
                                    double gravity, double reference_density) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  column.batch = index / (column.ny * column.nx);
  column.y = (index / column.nx) % column.ny;
  column.x = index % column.nx;
  defgcm::local_statekpp::vjp_column<Jmd95>(
      column, column.levels, gravity, reference_density);
}

defgcm::cuda_launch::LaunchConfigCache backward_launch_config_cache;

} // namespace

void kpp_statekpp_launch(const torch::Tensor &temp, const torch::Tensor &salt,
                         const torch::Tensor &pressure, torch::Tensor &dbloc,
                         torch::Tensor &dbsfc, torch::Tensor &alpha,
                         torch::Tensor &beta, torch::Tensor &rho1,
                         const double gravity, const double reference_density) {
  c10::cuda::CUDAGuard device_guard(temp.device());
  const int64_t element_count = temp.numel();
  // ONE instantiation, deliberately.  dbloc/dbsfc subtract two density
  // anomalies whose difference is three decades smaller than either operand, so
  // a float32 instantiation would not merely be less precise -- it would be a
  // second semantics for the same model, disagreeing with its own eager
  // fallback (which evaluates all of STATEKPP in float64).  A reduced-precision
  // run is served by the dispatcher, which widens before this call.
  AT_DISPATCH_SWITCH(
      temp.scalar_type(), "kpp_statekpp_launch",
      AT_DISPATCH_CASE(at::ScalarType::Double, [&] {
        const auto &launch_config = launch_config_cache.get(
            kpp_statekpp_kernel<scalar_t>,
            temp.get_device(),
            0,
            256,
            dynamic_shared_bytes_for_block);
        const int threads_per_block = launch_config.selected_block_size;
        const int64_t block_count =
            (element_count + threads_per_block - 1) / threads_per_block;
        TORCH_CHECK(block_count <= std::numeric_limits<unsigned int>::max(),
                    "KPP STATEKPP CUDA grid exceeds grid.x capacity");

        const dim3 block(threads_per_block);
        const dim3 grid(static_cast<unsigned int>(block_count));
        kpp_statekpp_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input4<scalar_t>(temp), input4<scalar_t>(salt),
                input4<scalar_t>(pressure), dbloc.data_ptr<scalar_t>(),
                dbsfc.data_ptr<scalar_t>(), alpha.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(), rho1.data_ptr<scalar_t>(),
                element_count, temp.size(1), temp.size(2), temp.size(3),
                static_cast<scalar_t>(gravity),
                static_cast<scalar_t>(reference_density));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}

const defgcm::cuda_launch::LaunchConfig &
kpp_statekpp_cuda_launch_config(int device) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  return launch_config_cache.get(
      kpp_statekpp_kernel<double>,
      device,
      0,
      256,
      dynamic_shared_bytes_for_block);
}

void kpp_statekpp_backward_launch(
    const torch::Tensor &temp, const torch::Tensor &salt,
    const torch::Tensor &pressure,
    const std::vector<c10::optional<torch::Tensor>> &seeds,
    const std::vector<c10::optional<torch::Tensor>> &gradients,
    double gravity, double reference_density) {
  c10::cuda::CUDAGuard guard(temp.device());
  StatekppColumn column{};
  column.temperature = input4<double>(temp);
  column.salinity = input4<double>(salt);
  column.pressure_pa = input4<double>(pressure);
  for (int index = 0; index < 5; ++index) {
    if (seeds[index]) column.seeds[index] = input4<double>(*seeds[index]);
  }
  column.gt = gradients[0] ? gradients[0]->data_ptr<double>() : nullptr;
  column.gs = gradients[1] ? gradients[1]->data_ptr<double>() : nullptr;
  column.gp = gradients[2] ? gradients[2]->data_ptr<double>() : nullptr;
  column.levels = temp.size(1); column.ny = temp.size(2); column.nx = temp.size(3);
  const int64_t count = temp.size(0) * column.ny * column.nx;
  const auto &config = backward_launch_config_cache.get(
      statekpp_vjp_kernel, temp.get_device(), 0, 128, dynamic_shared_bytes_for_block);
  const int block = config.selected_block_size;
  const int64_t blocks = (count + block - 1) / block;
  TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(),
              "STATEKPP backward grid exceeds grid.x capacity");
  statekpp_vjp_kernel<<<static_cast<unsigned int>(blocks), block, 0,
                        at::cuda::getCurrentCUDAStream()>>>(column, count, gravity, reference_density);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
