// One thread gathers all batch contributions for one spatial face.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include "../../../solvers/_cuda_launch/launch_config.h"
#include "vjp.h"

namespace {
struct View {
  int64_t b, k, j, i;
};
template <typename T> struct Inputs {
  const T *p[13];
  View s[13];
};
template <typename T> struct Outputs {
  T *p[11];
  bool broadcast[11];
};
View strides(const torch::Tensor &t) {
  int o = t.dim() - 3;
  return {t.dim() == 3 || t.size(0) == 1 ? 0 : t.stride(0),
          t.stride(o), t.stride(o + 1), t.stride(o + 2)};
}
template <typename T> __global__ void dst3_transport_flux_bwd_kernel(
    Inputs<T> in, Outputs<T> out, int64_t nb, int64_t nk,
    int64_t nj, int64_t ni, bool limited) {
  int64_t cell = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t volume = nk * nj * ni;
  if (cell >= volume) return;
  int64_t k = cell / (nj * ni), j = (cell / ni) % nj, i = cell % ni;
  T sums[11] = {};
  for (int64_t b = 0; b < nb; ++b) {
    T v[13], g[11];
    #pragma unroll
    for (int q = 0; q < 13; ++q) {
      View s = in.s[q];
      v[q] = in.p[q] ? in.p[q][b * s.b + k * s.k + j * s.j + i * s.i] : T(0);
    }
    dst3::vjp(v, v[11], v[12], limited, g, in.p[11] != nullptr, in.p[12] != nullptr);
    #pragma unroll
    for (int q = 0; q < 11; ++q) {
      if (out.p[q]) {
        if (out.broadcast[q]) sums[q] += g[q];
        else out.p[q][b * volume + cell] = g[q];
      }
    }
  }
  #pragma unroll
  for (int q = 0; q < 11; ++q)
    if (out.p[q] && out.broadcast[q]) out.p[q][cell] = sums[q];
}
defgcm::cuda_launch::LaunchConfigCache backward_cache;
size_t shared_bytes(int) { return 0; }
} // namespace

void dst3_transport_flux_bwd_launch(
    const std::vector<torch::Tensor> &values,
    const std::vector<c10::optional<torch::Tensor>> &seeds,
    std::vector<c10::optional<torch::Tensor>> &gradients,
    int64_t batch, bool limited) {
  c10::cuda::CUDAGuard guard(values[0].device());
  int offset = values[0].dim() - 3;
  int64_t nk = values[0].size(offset), nj = values[0].size(offset + 1);
  int64_t ni = values[0].size(offset + 2);
  AT_DISPATCH_FLOATING_TYPES(values[0].scalar_type(), "dst3_transport_flux_bwd", [&] {
    Inputs<scalar_t> in{};
    Outputs<scalar_t> out{};
    for (int q = 0; q < 11; ++q) {
      in.p[q] = values[q].data_ptr<scalar_t>();
      in.s[q] = strides(values[q]);
      if (gradients[q].has_value()) {
        out.p[q] = gradients[q]->data_ptr<scalar_t>();
        out.broadcast[q] = values[q].dim() == 3 || values[q].size(0) == 1;
      }
    }
    for (int q = 0; q < 2; ++q) {
      if (seeds[q].has_value()) {
        in.p[11 + q] = seeds[q]->data_ptr<scalar_t>();
        in.s[11 + q] = strides(*seeds[q]);
      }
    }
    const auto &config = backward_cache.get(dst3_transport_flux_bwd_kernel<scalar_t>,
        values[0].get_device(), 0, 256, shared_bytes);
    int threads = config.selected_block_size;
    int64_t blocks = (nk * nj * ni + threads - 1) / threads;
    TORCH_CHECK(blocks <= 2147483647, "DST3 backward grid exceeds grid.x");
    dst3_transport_flux_bwd_kernel<scalar_t>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            in, out, batch, nk, nj, ni, limited);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

const defgcm::cuda_launch::LaunchConfig &
dst3_cuda_backward_launch_config(int device, at::ScalarType dtype) {
  c10::cuda::CUDAGuard guard(c10::Device(c10::kCUDA, device));
  if (dtype == at::kFloat)
    return backward_cache.get(dst3_transport_flux_bwd_kernel<float>, device, 0, 256, shared_bytes);
  TORCH_CHECK(dtype == at::kDouble, "DST3 backward requires float32 or float64");
  return backward_cache.get(dst3_transport_flux_bwd_kernel<double>, device, 0, 256, shared_bytes);
}
