#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

#include "../_cuda_launch/launch_config.h"

namespace {

thread_local int occupancy_nz = 0;
defgcm::cuda_launch::LaunchConfigCache launch_config_cache;

// A translation unit may declare `extern __shared__` exactly once, so the two
// dtype instantiations alias one 8-byte-aligned byte array instead of each
// declaring their own typed view. The base address is the start of the dynamic
// shared window either way, so the float64 instantiation indexes exactly the
// bytes it indexed before this kernel was templated.
template <typename scalar_t>
__device__ __forceinline__ scalar_t* dynamic_shared() {
  extern __shared__ __align__(8) unsigned char shared_bytes[];
  return reinterpret_cast<scalar_t*>(shared_bytes);
}

template <typename scalar_t>
size_t dynamic_shared_bytes_for_block(int) {
  return static_cast<size_t>(occupancy_nz) * sizeof(scalar_t);
}

struct Strides4 {
  int64_t batch;
  int64_t level;
  int64_t y;
  int64_t x;
};

Strides4 strides4(const torch::Tensor& value) {
  return {
      value.stride(0),
      value.stride(1),
      value.stride(2),
      value.stride(3)};
}

template <typename scalar_t>
struct Tensor4 {
  const scalar_t* data;
  Strides4 strides;
  int64_t batches;

  __device__ __forceinline__ scalar_t load(
      const int64_t batch,
      const int level,
      const int64_t y,
      const int64_t x) const {
    const int64_t source_batch = batches == 1 ? 0 : batch;
    const int64_t index =
        source_batch * strides.batch +
        static_cast<int64_t>(level) * strides.level + y * strides.y +
        x * strides.x;
    return __ldg(data + index);
  }
};

template <typename scalar_t>
struct MutableTensor4 {
  scalar_t* data;
  Strides4 strides;

  __device__ __forceinline__ void store(
      const int64_t batch,
      const int level,
      const int64_t y,
      const int64_t x,
      const scalar_t value) const {
    const int64_t index =
        batch * strides.batch + static_cast<int64_t>(level) * strides.level +
        y * strides.y + x * strides.x;
    data[index] = value;
  }
};

template <typename scalar_t>
Tensor4<scalar_t> tensor4(const torch::Tensor& value) {
  return {value.data_ptr<scalar_t>(), strides4(value), value.size(0)};
}

template <typename scalar_t>
MutableTensor4<scalar_t> mutable_tensor4(torch::Tensor& value) {
  return {value.data_ptr<scalar_t>(), strides4(value)};
}

template <typename scalar_t>
__global__ void tridiag_build_kernel(
    const Tensor4<scalar_t> tracer,
    const scalar_t* __restrict__ layer_thickness,
    const int64_t layer_stride,
    const scalar_t* __restrict__ interface_spacing,
    const int64_t interface_stride,
    const Tensor4<scalar_t> diffusivity,
    const Tensor4<scalar_t> recip_hfac,
    const scalar_t dt,
    const MutableTensor4<scalar_t> a,
    const MutableTensor4<scalar_t> b,
    const MutableTensor4<scalar_t> c,
    const MutableTensor4<scalar_t> d,
    const int64_t batch_size,
    const int nz,
    const int64_t ny,
    const int64_t nx) {
  scalar_t* shared_layer_thickness = dynamic_shared<scalar_t>();
  for (int level = threadIdx.x; level < nz; level += blockDim.x) {
    shared_layer_thickness[level] =
        __ldg(layer_thickness + static_cast<int64_t>(level) * layer_stride);
  }
  __syncthreads();

  const int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t nplane = ny * nx;
  const int64_t values_per_batch = static_cast<int64_t>(nz) * nplane;
  const int64_t total_values = batch_size * values_per_batch;
  if (output_index >= total_values) {
    return;
  }

  const int64_t batch = output_index / values_per_batch;
  const int64_t batch_offset = output_index - batch * values_per_batch;
  const int level = static_cast<int>(batch_offset / nplane);
  const int64_t plane_offset = batch_offset - static_cast<int64_t>(level) * nplane;
  const int64_t y = plane_offset / nx;
  const int64_t x = plane_offset - y * nx;

  const scalar_t dz_current = shared_layer_thickness[level];
  const scalar_t recip_hfac_current = recip_hfac.load(batch, level, y, x);
  const scalar_t one = static_cast<scalar_t>(1.0);
  const scalar_t row_scale = (-dt * recip_hfac_current) * (one / dz_current);
  scalar_t lower = static_cast<scalar_t>(0.0);
  scalar_t upper = static_cast<scalar_t>(0.0);

  if (level > 0 && recip_hfac.load(batch, level - 1, y, x) != 0) {
    const scalar_t dz_interface =
        __ldg(interface_spacing + static_cast<int64_t>(level - 1) * interface_stride);
    lower = row_scale * diffusivity.load(batch, level, y, x) * (one / dz_interface);
  }
  if (level + 1 < nz && recip_hfac.load(batch, level + 1, y, x) != 0) {
    const scalar_t dz_interface =
        __ldg(interface_spacing + static_cast<int64_t>(level) * interface_stride);
    upper = row_scale * diffusivity.load(batch, level + 1, y, x) * (one / dz_interface);
  }
  const scalar_t main = one - (lower + upper);

  a.store(batch, level, y, x, lower);
  b.store(batch, level, y, x, main);
  c.store(batch, level, y, x, upper);
  d.store(batch, level, y, x, tracer.load(batch, level, y, x));
}

template <typename scalar_t>
void launch_typed(
    const torch::Tensor& tracer,
    const torch::Tensor& layer_thickness,
    const torch::Tensor& interface_spacing,
    const torch::Tensor& diffusivity,
    const torch::Tensor& recip_hfac,
    double dt,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& c,
    torch::Tensor& d) {
  const auto batch_size = static_cast<int64_t>(tracer.size(0));
  const auto nz = static_cast<int>(tracer.size(1));
  const auto ny = static_cast<int64_t>(tracer.size(2));
  const auto nx = static_cast<int64_t>(tracer.size(3));
  occupancy_nz = nz;
  const auto& launch_config = launch_config_cache.get(
      tridiag_build_kernel<scalar_t>,
      tracer.get_device(),
      nz,
      256,
      dynamic_shared_bytes_for_block<scalar_t>);
  const int threads_per_block = launch_config.selected_block_size;
  const int64_t total_values = batch_size * static_cast<int64_t>(nz) * ny * nx;
  const int64_t block_count =
      (total_values + threads_per_block - 1) / threads_per_block;
  TORCH_CHECK(
      block_count <= std::numeric_limits<unsigned int>::max(),
      "tridiag-build CUDA grid exceeds grid.x capacity");

  const dim3 block(static_cast<unsigned int>(threads_per_block));
  const dim3 grid(static_cast<unsigned int>(block_count));
  const size_t shared_bytes = static_cast<size_t>(nz) * sizeof(scalar_t);
  tridiag_build_kernel<scalar_t><<<
      grid,
      block,
      shared_bytes,
      at::cuda::getCurrentCUDAStream()>>>(
      tensor4<scalar_t>(tracer),
      layer_thickness.data_ptr<scalar_t>(),
      layer_thickness.stride(0),
      interface_spacing.data_ptr<scalar_t>(),
      interface_spacing.stride(0),
      tensor4<scalar_t>(diffusivity),
      tensor4<scalar_t>(recip_hfac),
      static_cast<scalar_t>(dt),
      mutable_tensor4<scalar_t>(a),
      mutable_tensor4<scalar_t>(b),
      mutable_tensor4<scalar_t>(c),
      mutable_tensor4<scalar_t>(d),
      batch_size,
      nz,
      ny,
      nx);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void tridiag_build_cuda_launch(
    const torch::Tensor& tracer,
    const torch::Tensor& layer_thickness,
    const torch::Tensor& interface_spacing,
    const torch::Tensor& diffusivity,
    const torch::Tensor& recip_hfac,
    double dt,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& c,
    torch::Tensor& d) {
  c10::cuda::CUDAGuard device_guard(tracer.device());
  if (tracer.scalar_type() == at::kFloat) {
    launch_typed<float>(
        tracer, layer_thickness, interface_spacing, diffusivity, recip_hfac, dt, a, b, c, d);
    return;
  }
  TORCH_CHECK(
      tracer.scalar_type() == at::kDouble,
      "tridiag-build CUDA kernel supports float64 or float32 only");
  launch_typed<double>(
      tracer, layer_thickness, interface_spacing, diffusivity, recip_hfac, dt, a, b, c, d);
}

const defgcm::cuda_launch::LaunchConfig&
tridiag_build_cuda_launch_config(int device, int nz, at::ScalarType dtype) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  occupancy_nz = nz;
  if (dtype == at::kFloat) {
    return launch_config_cache.get(
        tridiag_build_kernel<float>,
        device,
        nz,
        256,
        dynamic_shared_bytes_for_block<float>);
  }
  TORCH_CHECK(
      dtype == at::kDouble,
      "tridiag-build CUDA launch configuration supports float64 or float32 only");
  return launch_config_cache.get(
      tridiag_build_kernel<double>,
      device,
      nz,
      256,
      dynamic_shared_bytes_for_block<double>);
}
