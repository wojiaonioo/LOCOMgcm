// Column-fused KPP STATEKPP for JMD95/MDJWF storage in float32 or float64.
// All arithmetic is deliberately double and this translation unit is built
// with -fmad=false. Expression association follows kpp.py and
// equation_of_state.py; do not refactor the polynomial expressions.

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <string>

#include "../_eos_cuda/local_eos.cuh"

namespace {

using defgcm::local_eos::Jmd95;
using defgcm::local_eos::Mdjwf;
using defgcm::local_eos::clamp_and_root;
using defgcm::local_eos::density_anomaly;
using defgcm::local_eos::alpha_beta;

struct Strides4 {
  int64_t batch;
  int64_t level;
  int64_t y;
  int64_t x;
};

template <typename T>
struct Input4 {
  const T *__restrict__ data;
  Strides4 strides;
  int64_t batch_size;
  int64_t level_size;
  int64_t y_size;
  int64_t x_size;

  __device__ __forceinline__ double load(int64_t batch, int64_t level,
                                         int64_t y, int64_t x) const {
    const int64_t ib = batch_size == 1 ? 0 : batch;
    const int64_t ik = level_size == 1 ? 0 : level;
    const int64_t ij = y_size == 1 ? 0 : y;
    const int64_t ii = x_size == 1 ? 0 : x;
    const int64_t offset = ib * strides.batch + ik * strides.level +
                           ij * strides.y + ii * strides.x;
    return static_cast<double>(__ldg(data + offset));
  }
};

template <typename T>
Input4<T> input4(const torch::Tensor &value) {
  return {value.data_ptr<T>(),
          {value.stride(0), value.stride(1), value.stride(2), value.stride(3)},
          value.size(0), value.size(1), value.size(2), value.size(3)};
}

namespace jmd95_column {

using defgcm::local_eos::jmd95_divided_difference;
using defgcm::local_eos::jmd95_polynomial;

struct BulkPolynomials {
  double kf, ks, ksr, c, cs, c2, c2s;
};

__device__ __forceinline__ BulkPolynomials bulk_polynomials(double t) {
  const double kf[] = {Jmd95::kfw0, Jmd95::kfw1, Jmd95::kfw2,
                       Jmd95::kfw3, Jmd95::kfw4};
  const double ks[] = {Jmd95::ksw0, Jmd95::ksw1, Jmd95::ksw2, Jmd95::ksw3};
  const double ksr[] = {Jmd95::ksw4, Jmd95::ksw5, Jmd95::ksw6};
  const double c[] = {Jmd95::kp0, Jmd95::kp1, Jmd95::kp2, Jmd95::kp3};
  const double cs[] = {Jmd95::kp4, Jmd95::kp5, Jmd95::kp6};
  const double c2[] = {Jmd95::kp8, Jmd95::kp9, Jmd95::kp10};
  const double c2s[] = {Jmd95::kp11, Jmd95::kp12, Jmd95::kp13};
  return {jmd95_polynomial(kf, t), jmd95_polynomial(ks, t),
          jmd95_polynomial(ksr, t), jmd95_polynomial(c, t),
          jmd95_polynomial(cs, t), jmd95_polynomial(c2, t),
          jmd95_polynomial(c2s, t)};
}

// Column form of local_eos::jmd95_density_difference. Only the seven bulk
// polynomials are supplied by the caller; all associations stay identical.
// The previous layer's ta is this layer's tb, so its rounded values carry over.
__device__ __forceinline__ double density_difference(
    double ta, double sa, double ra, double tb, double sb, double rb, double p,
    const BulkPolynomials& bulk_a, const BulkPolynomials& bulk_b) {
  const double f[] = {Jmd95::fw0, Jmd95::fw1, Jmd95::fw2,
                      Jmd95::fw3, Jmd95::fw4, Jmd95::fw5};
  const double w[] = {Jmd95::sw0, Jmd95::sw1, Jmd95::sw2, Jmd95::sw3, Jmd95::sw4};
  const double wr[] = {Jmd95::sw5, Jmd95::sw6, Jmd95::sw7};
  const double kf[] = {Jmd95::kfw0, Jmd95::kfw1, Jmd95::kfw2,
                       Jmd95::kfw3, Jmd95::kfw4};
  const double ks[] = {Jmd95::ksw0, Jmd95::ksw1, Jmd95::ksw2, Jmd95::ksw3};
  const double ksr[] = {Jmd95::ksw4, Jmd95::ksw5, Jmd95::ksw6};
  const double c[] = {Jmd95::kp0, Jmd95::kp1, Jmd95::kp2, Jmd95::kp3};
  const double cs[] = {Jmd95::kp4, Jmd95::kp5, Jmd95::kp6};
  const double c2[] = {Jmd95::kp8, Jmd95::kp9, Jmd95::kp10};
  const double c2s[] = {Jmd95::kp11, Jmd95::kp12, Jmd95::kp13};
  const double root_sum = ra + rb;
  const bool positive = root_sum > 0.0;
  const double safe_sum = positive ? root_sum : 1.0;
  const double ds15 = positive ? (sa + ra * rb + sb) / safe_sum : 0.0;
  const double dt = ta - tb, ds = sa - sb;
  double delta_r = dt * (
      jmd95_divided_difference(f, ta, tb) + sa * jmd95_divided_difference(w, ta, tb)
      + sa * ra * jmd95_divided_difference(wr, ta, tb));
  delta_r = delta_r + ds * (jmd95_polynomial(w, tb)
      + ds15 * jmd95_polynomial(wr, tb) + Jmd95::sw8 * (sa + sb));
  double delta_k = dt * (
      jmd95_divided_difference(kf, ta, tb) + sa * jmd95_divided_difference(ks, ta, tb)
      + sa * ra * jmd95_divided_difference(ksr, ta, tb));
  delta_k = delta_k + ds * (bulk_b.ks + ds15 * bulk_b.ksr);
  delta_k = delta_k + p * (
      dt * (jmd95_divided_difference(c, ta, tb)
            + sa * jmd95_divided_difference(cs, ta, tb))
      + ds * (bulk_b.cs + ds15 * Jmd95::kp7));
  delta_k = delta_k + p * p * (
      dt * (jmd95_divided_difference(c2, ta, tb)
            + sa * jmd95_divided_difference(c2s, ta, tb))
      + ds * bulk_b.c2s);
  double ka = bulk_a.kf + sa * bulk_a.ks + sa * ra * bulk_a.ksr;
  ka = ka + p * (bulk_a.c + sa * bulk_a.cs + sa * ra * Jmd95::kp7)
      + p * p * (bulk_a.c2 + sa * bulk_a.c2s);
  const double surface_b = jmd95_polynomial(f, tb) + sb * jmd95_polynomial(w, tb)
      + sb * rb * jmd95_polynomial(wr, tb) + Jmd95::sw8 * sb * sb;
  double kb = bulk_b.kf + sb * bulk_b.ks + sb * rb * bulk_b.ksr;
  kb = kb + p * (bulk_b.c + sb * bulk_b.cs + sb * rb * Jmd95::kp7)
      + p * p * (bulk_b.c2 + sb * bulk_b.c2s);
  return delta_r * ka / (ka - p) - surface_b * p * delta_k / ((ka - p) * (kb - p));
}

} // namespace jmd95_column

template <typename T, class Eos>
__global__ __launch_bounds__(128, 2) void statekpp_column_kernel(
    const Input4<T> temp, const Input4<T> salt, const Input4<T> pressure,
    T *__restrict__ dbloc, T *__restrict__ dbsfc, T *__restrict__ rho1,
    T *__restrict__ alpha, T *__restrict__ beta, int64_t columns,
    int64_t levels, int64_t ny, int64_t nx, double gravity,
    double reference_density) {
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= columns) {
    return;
  }
  const int64_t plane = ny * nx;
  const int64_t batch = column / plane;
  const int64_t plane_index = column - batch * plane;
  const int64_t y = plane_index / nx;
  const int64_t x = plane_index - y * nx;
  const int64_t volume_base = batch * levels * plane + plane_index;
  const int64_t coeff_base = batch * (levels + 1) * plane + plane_index;

  double temp_surface = temp.load(batch, 0, y, x);
  double salt_surface;
  double salt_root_surface;
  clamp_and_root(salt.load(batch, 0, y, x), salt_surface, salt_root_surface);
  double temp_k = temp_surface;
  double salt_k = salt_surface;
  double salt_root_k = salt_root_surface;
  double p_k = pressure.load(batch, 0, y, x) * Eos::si_to_p;
  double rho_diag = density_anomaly<Eos>(temp_k, salt_k, salt_root_k, p_k,
                                         reference_density);
  rho1[column] = static_cast<T>(rho_diag + reference_density);
  jmd95_column::BulkPolynomials bulk_k;
  if constexpr (std::is_same_v<Eos, Jmd95>) {
    bulk_k = jmd95_column::bulk_polynomials(temp_k);
  }

  for (int64_t k = 0; k < levels; ++k) {
    const int64_t output = volume_base + k * plane;
    const int64_t coefficient = coeff_base + k * plane;
    double alpha_k;
    double beta_k;
    alpha_beta<Eos>(temp_k, salt_k, salt_root_k, p_k, alpha_k, beta_k);
    alpha[coefficient] = static_cast<T>(alpha_k);
    beta[coefficient] = static_cast<T>(beta_k);

    double dbsfc_k = 0.0;
    if (k != 0) {
      const double rho_surface = density_anomaly<Eos>(
          temp_surface, salt_surface, salt_root_surface, p_k,
          reference_density);
      dbsfc_k =
          gravity * (rho_diag - rho_surface) / (rho_diag + reference_density);
    }
    dbsfc[output] = static_cast<T>(dbsfc_k);

    if (k + 1 < levels) {
      const double p_next =
          pressure.load(batch, k + 1, y, x) * Eos::si_to_p;
      const double temp_next = temp.load(batch, k + 1, y, x);
      double salt_next;
      double salt_root_next;
      clamp_and_root(salt.load(batch, k + 1, y, x), salt_next,
                     salt_root_next);
      const double rho_below = density_anomaly<Eos>(
          temp_next, salt_next, salt_root_next, p_next, reference_density);
      double difference;
      if constexpr (std::is_same_v<Eos, Jmd95>) {
        const auto bulk_next = jmd95_column::bulk_polynomials(temp_next);
        difference = jmd95_column::density_difference(
            temp_next, salt_next, salt_root_next,
            temp_k, salt_k, salt_root_k, p_next, bulk_next, bulk_k);
        bulk_k = bulk_next;
      } else {
        const double rho_upper = density_anomaly<Eos>(
            temp_k, salt_k, salt_root_k, p_next, reference_density);
        difference = rho_below - rho_upper;
      }
      dbloc[output] = static_cast<T>(gravity * difference / (rho_below + reference_density));
      temp_k = temp_next;
      salt_k = salt_next;
      salt_root_k = salt_root_next;
      p_k = p_next;
      rho_diag = rho_below;
    } else {
      dbloc[output] = static_cast<T>(0.0);
      alpha[coeff_base + levels * plane] = static_cast<T>(alpha_k);
      beta[coeff_base + levels * plane] = static_cast<T>(beta_k);
    }
  }
}

} // namespace

void statekpp_fused_launch(
    const torch::Tensor &temp, const torch::Tensor &salt,
    const torch::Tensor &pressure, torch::Tensor &dbloc,
    torch::Tensor &dbsfc, torch::Tensor &rho1, torch::Tensor &alpha,
    torch::Tensor &beta, double gravity, double reference_density,
    const std::string &eos) {
  c10::cuda::CUDAGuard device_guard(temp.device());
  const int64_t columns = temp.size(0) * temp.size(2) * temp.size(3);
  constexpr int threads = 128;
  const int64_t blocks = (columns + threads - 1) / threads;
  TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(),
              "STATEKPP fused grid exceeds grid.x capacity");
  AT_DISPATCH_SWITCH(
      temp.scalar_type(), "statekpp_fused_launch",
      AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
        if (eos == "jmd95") {
          statekpp_column_kernel<scalar_t, Jmd95><<<
              static_cast<unsigned int>(blocks), threads, 0,
              at::cuda::getCurrentCUDAStream()>>>(
              input4<scalar_t>(temp), input4<scalar_t>(salt),
              input4<scalar_t>(pressure), dbloc.data_ptr<scalar_t>(),
              dbsfc.data_ptr<scalar_t>(), rho1.data_ptr<scalar_t>(),
              alpha.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(), columns,
              temp.size(1), temp.size(2), temp.size(3), gravity,
              reference_density);
        } else {
          statekpp_column_kernel<scalar_t, Mdjwf><<<
              static_cast<unsigned int>(blocks), threads, 0,
              at::cuda::getCurrentCUDAStream()>>>(
              input4<scalar_t>(temp), input4<scalar_t>(salt),
              input4<scalar_t>(pressure), dbloc.data_ptr<scalar_t>(),
              dbsfc.data_ptr<scalar_t>(), rho1.data_ptr<scalar_t>(),
              alpha.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(), columns,
              temp.size(1), temp.size(2), temp.size(3), gravity,
              reference_density);
        }
      })
      AT_DISPATCH_CASE(at::ScalarType::Double, [&] {
        if (eos == "jmd95") {
          statekpp_column_kernel<scalar_t, Jmd95><<<
              static_cast<unsigned int>(blocks), threads, 0,
              at::cuda::getCurrentCUDAStream()>>>(
              input4<scalar_t>(temp), input4<scalar_t>(salt),
              input4<scalar_t>(pressure), dbloc.data_ptr<scalar_t>(),
              dbsfc.data_ptr<scalar_t>(), rho1.data_ptr<scalar_t>(),
              alpha.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(), columns,
              temp.size(1), temp.size(2), temp.size(3), gravity,
              reference_density);
        } else {
          statekpp_column_kernel<scalar_t, Mdjwf><<<
              static_cast<unsigned int>(blocks), threads, 0,
              at::cuda::getCurrentCUDAStream()>>>(
              input4<scalar_t>(temp), input4<scalar_t>(salt),
              input4<scalar_t>(pressure), dbloc.data_ptr<scalar_t>(),
              dbsfc.data_ptr<scalar_t>(), rho1.data_ptr<scalar_t>(),
              alpha.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(), columns,
              temp.size(1), temp.size(2), temp.size(3), gravity,
              reference_density);
        }
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
