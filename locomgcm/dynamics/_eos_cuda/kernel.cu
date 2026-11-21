#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/Dispatch.h>

#include <cstdint>
#include <limits>

#include "../../solvers/_cuda_launch/launch_config.h"

#include "local_eos.cuh"

namespace {

locomgcm::cuda_launch::LaunchConfigCache launch_config_cache;

size_t dynamic_shared_bytes_for_block(int) {
  return 0;
}

constexpr int64_t kJmd95 = 0;
constexpr int64_t kMdjwf = 1;

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

// The native ABI is float64. Reduced storage precision is handled by the
// dispatcher, including reconstruction from the already narrowed anomaly.
template <class Eos>
__device__ __forceinline__ void diagnostic_outputs(
    double temp, double salt_input, double pressure_pa, double reference_density,
    double &anomaly, double &density) {
  double salt, root;
  locomgcm::local_eos::clamp_and_root(salt_input, salt, root);
  anomaly = locomgcm::local_eos::density_anomaly<Eos>(
      temp, salt, root, pressure_pa * Eos::si_to_p, reference_density);
  density = anomaly + reference_density;
}

template <typename scalar_t>
__global__ void eos_diagnostic_kernel(
    const Input4<scalar_t> temp, const Input4<scalar_t> salt, const Input4<scalar_t> pressure,
    scalar_t *__restrict__ density_anomaly, scalar_t *__restrict__ density,
    const int64_t element_count, const int64_t levels, const int64_t ny,
    const int64_t nx, const int64_t variant, const scalar_t reference_density) {
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

  const scalar_t temp_value = temp.load(batch, level, y, x);
  const scalar_t salt_value = salt.load(batch, level, y, x);
  const scalar_t pressure_value = pressure.load(batch, level, y, x);
  scalar_t anomaly_value;
  scalar_t density_value;
  if (variant == kJmd95) {
    diagnostic_outputs<locomgcm::local_eos::Jmd95>(temp_value, salt_value, pressure_value, reference_density,
                  anomaly_value, density_value);
  } else {
    diagnostic_outputs<locomgcm::local_eos::Mdjwf>(temp_value, salt_value, pressure_value, reference_density,
                  anomaly_value, density_value);
  }
  density_anomaly[output_index] = anomaly_value;
  density[output_index] = density_value;
}

__global__ void eos_local_vjp_kernel(
    Input4<double> temp, Input4<double> salt, Input4<double> pressure,
    Input4<double> density_seed, Input4<double> alpha_seed,
    Input4<double> beta_seed, bool use_alpha, bool use_beta,
    double *temp_grad, double *salt_grad, double *pressure_grad,
    int64_t count, int64_t levels, int64_t ny, int64_t nx, int64_t variant,
    double reference_density) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int64_t plane = ny * nx;
  const int64_t batch = index / (levels * plane);
  const int64_t level = (index / plane) % levels;
  const int64_t y = (index / nx) % ny;
  const int64_t x = index % nx;
  const double t = temp.load(batch, level, y, x);
  const double s = salt.load(batch, level, y, x);
  const double p = pressure.load(batch, level, y, x);
  const double gr = density_seed.load(batch, level, y, x);
  const double ga = use_alpha ? alpha_seed.load(batch, level, y, x) : 0.0;
  const double gb = use_beta ? beta_seed.load(batch, level, y, x) : 0.0;
  locomgcm::local_eos::InputPartials gradient;
  if (variant == kJmd95) {
    gradient = locomgcm::local_eos::local_vjp<locomgcm::local_eos::Jmd95>(
        t, s, p, reference_density, gr, ga, gb);
  } else {
    gradient = locomgcm::local_eos::local_vjp<locomgcm::local_eos::Mdjwf>(
        t, s, p, reference_density, gr, ga, gb);
  }
  if (temp_grad) temp_grad[index] = gradient.temp;
  if (salt_grad) salt_grad[index] = gradient.salt;
  // The autograd boundary reduces broadcast pressure to its original shape.
  if (pressure_grad) pressure_grad[index] = gradient.pressure;
}

locomgcm::cuda_launch::LaunchConfigCache backward_launch_config_cache;

} // namespace

void eos_diagnostic_launch(const torch::Tensor &temp, const torch::Tensor &salt,
                           const torch::Tensor &pressure,
                           torch::Tensor &density_anomaly,
                           torch::Tensor &density, const int64_t variant,
                           const double reference_density) {
  c10::cuda::CUDAGuard device_guard(temp.device());
  const int64_t element_count = temp.numel();
  // ONE instantiation, deliberately.  The eager reference evaluates the
  // JMD95/MDJWF polynomials in float64 even for a float32 state and narrows the
  // anomaly on the way out, so a float32 instantiation would be a second,
  // ~1e-4-relative-different answer for the same model -- and a dormant one,
  // reachable by anyone who widens the dtype predicate.  A reduced-precision run
  // is served by the dispatcher, which widens before this call.
  AT_DISPATCH_SWITCH(
      temp.scalar_type(), "eos_diagnostic_launch",
      AT_DISPATCH_CASE(at::ScalarType::Double, [&] {
        const auto& launch_config = launch_config_cache.get(
            eos_diagnostic_kernel<scalar_t>,
            temp.get_device(),
            0,
            256,
            dynamic_shared_bytes_for_block);
        const int threads_per_block = launch_config.selected_block_size;
        const int64_t block_count =
            (element_count + threads_per_block - 1) / threads_per_block;
        TORCH_CHECK(block_count <= std::numeric_limits<unsigned int>::max(),
                    "EOS CUDA grid exceeds grid.x capacity");

        const dim3 block(threads_per_block);
        const dim3 grid(static_cast<unsigned int>(block_count));
        eos_diagnostic_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input4<scalar_t>(temp), input4<scalar_t>(salt),
                input4<scalar_t>(pressure),
                density_anomaly.data_ptr<scalar_t>(),
                density.data_ptr<scalar_t>(), element_count, temp.size(1),
                temp.size(2), temp.size(3), variant,
                static_cast<scalar_t>(reference_density));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}

const locomgcm::cuda_launch::LaunchConfig&
eos_cuda_launch_config(int device) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  return launch_config_cache.get(
      eos_diagnostic_kernel<double>,
      device,
      0,
      256,
      dynamic_shared_bytes_for_block);
}

void eos_backward_launch(
    const torch::Tensor &temp, const torch::Tensor &salt,
    const torch::Tensor &pressure, const torch::Tensor &density_seed,
    const c10::optional<torch::Tensor> &alpha_seed,
    const c10::optional<torch::Tensor> &beta_seed,
    const c10::optional<torch::Tensor> &temp_grad,
    const c10::optional<torch::Tensor> &salt_grad,
    const c10::optional<torch::Tensor> &pressure_grad,
    int64_t variant, double reference_density) {
  c10::cuda::CUDAGuard device_guard(temp.device());
  const auto &config = backward_launch_config_cache.get(
      eos_local_vjp_kernel, temp.get_device(), 0, 256,
      dynamic_shared_bytes_for_block);
  const int block = config.selected_block_size;
  const int64_t blocks = (temp.numel() + block - 1) / block;
  TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(),
              "EOS backward grid exceeds grid.x capacity");
  // An inactive coefficient seed never reads its placeholder input.
  eos_local_vjp_kernel<<<static_cast<unsigned int>(blocks), block, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      input4<double>(temp), input4<double>(salt), input4<double>(pressure),
      input4<double>(density_seed),
      input4<double>(alpha_seed ? *alpha_seed : density_seed),
      input4<double>(beta_seed ? *beta_seed : density_seed),
      alpha_seed.has_value(), beta_seed.has_value(),
      temp_grad ? temp_grad->data_ptr<double>() : nullptr,
      salt_grad ? salt_grad->data_ptr<double>() : nullptr,
      pressure_grad ? pressure_grad->data_ptr<double>() : nullptr,
      temp.numel(), temp.size(1),
      temp.size(2), temp.size(3), variant, reference_density);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
