#pragma once

#include <cstdint>

#include "../_eos_cuda/local_eos.cuh"

namespace defgcm::local_statekpp {

// Pull back g*(a-b)/(a+rho_const); g/rho_const are fixed in this native ABI.
// At fixed pressure the explicit rho_const and anomaly offsets cancel.
// Upstream g/rho_const dependencies through pressure still flow through Pa.
struct BuoyancySeeds { double a, b; };

__device__ __forceinline__ BuoyancySeeds buoyancy_seeds(
    double a, double b, double gravity, double reference_density, double seed) {
  const double density = a + reference_density;
  const double difference_seed = seed / density * gravity;
  const double denominator_seed =
      -seed * (gravity * (a - b)) / (density * density);
  return {difference_seed + denominator_seed, -difference_seed};
}

template <class Eos>
__device__ __forceinline__ double anomaly(
    double t, double s_input, double pressure_pa, double reference_density) {
  double s, root;
  local_eos::clamp_and_root(s_input, s, root);
  return local_eos::density_anomaly<Eos>(
      t, s, root, pressure_pa * Eos::si_to_p, reference_density);
}

// The column adapter owns layout, optional seeds and output activity. This
// stage owns D/U/H sampling and deterministic surface/adjacent-layer gathering;
// the same mathematics can be consumed by a fused KPP column implementation.
// Alpha/beta seeds cover real levels; the public cat folds the virtual layer.
template <class Eos, class Column>
__device__ __forceinline__ void vjp_column(
    Column &column, int64_t levels, double gravity, double reference_density) {
  const double surface_t = column.temp(0);
  const double surface_s = column.salt(0);
  double surface_temp_seed = 0.0, surface_salt_seed = 0.0;
  double next_density_seed = 0.0, next_pressure_seed = 0.0;
  double next_temp_seed = 0.0, next_salt_seed = 0.0;
  for (int64_t k = 0; k < levels; ++k) {
    const double t = column.temp(k), s = column.salt(k), p = column.pressure(k);
    const double diagonal = anomaly<Eos>(t, s, p, reference_density);
    double density_seed = next_density_seed;
    double pressure_seed = next_pressure_seed;
    next_density_seed = next_pressure_seed = 0.0;
    if (k == 0) density_seed += column.seed(2, 0);
    if (k > 0 && column.seed(1, k) != 0.0) {
      const double surface = anomaly<Eos>(surface_t, surface_s, p, reference_density);
      const auto seeds = buoyancy_seeds(diagonal, surface, gravity,
                                       reference_density, column.seed(1, k));
      density_seed += seeds.a;
      const auto h = local_eos::local_vjp<Eos>(
          surface_t, surface_s, p, reference_density, seeds.b, 0.0, 0.0);
      surface_temp_seed += h.temp;
      surface_salt_seed += h.salt;
      pressure_seed += h.pressure;
    }
    auto gradient = local_eos::local_vjp<Eos>(
        t, s, p, reference_density, density_seed,
        column.seed(3, k), column.seed(4, k));
    gradient.temp += next_temp_seed;
    gradient.salt += next_salt_seed;
    next_temp_seed = next_salt_seed = 0.0;
    gradient.pressure += pressure_seed;
    if (k + 1 < levels && column.seed(0, k) != 0.0) {
      const double p_next = column.pressure(k + 1);
      const double below = anomaly<Eos>(column.temp(k + 1), column.salt(k + 1),
                                        p_next, reference_density);
      if constexpr (std::is_same_v<Eos, local_eos::Jmd95>) {
        const auto difference = local_eos::jmd95_difference_partials(
            column.temp(k + 1), column.salt(k + 1), t, s, p_next);
        const double density = below + reference_density;
        const double seed = column.seed(0, k);
        const double difference_seed = seed / density * gravity;
        next_density_seed = -seed * (gravity * difference.value) / (density * density);
        next_temp_seed = difference_seed * difference.temp_a;
        next_salt_seed = difference_seed * difference.salt_a;
        gradient.temp += difference_seed * difference.temp_b;
        gradient.salt += difference_seed * difference.salt_b;
        next_pressure_seed = difference_seed * difference.pressure;
      } else {
        const double upper = anomaly<Eos>(t, s, p_next, reference_density);
        const auto seeds = buoyancy_seeds(below, upper, gravity,
                                         reference_density, column.seed(0, k));
        next_density_seed = seeds.a;
        const auto u = local_eos::local_vjp<Eos>(
            t, s, p_next, reference_density, seeds.b, 0.0, 0.0);
        gradient.temp += u.temp;
        gradient.salt += u.salt;
        next_pressure_seed = u.pressure;
      }
    }
    column.write(k, gradient.temp, gradient.salt, gradient.pressure);
  }
  column.add_surface(surface_temp_seed, surface_salt_seed);
}

} // namespace defgcm::local_statekpp
