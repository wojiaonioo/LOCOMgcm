#pragma once

#include <cmath>
#include <type_traits>

// EOS-owned pointwise mathematics shared by diagnostic and KPP consumers.
// Inputs are double; pressure is already in bar (JMD95) or dbar (MDJWF).
// Storage narrowing, pressure sampling and layout remain with the caller.
// The diagnostic density and alpha/beta share the MITgcm P0 density and bulk modulus.
namespace locomgcm::local_eos {

struct Jmd95 {
  static constexpr double fw0 = 999.842594e0;
  static constexpr double fw1 = 6.793952e-02;
  static constexpr double fw2 = -9.095290e-03;
  static constexpr double fw3 = 1.001685e-04;
  static constexpr double fw4 = -1.120083e-06;
  static constexpr double fw5 = 6.536332e-09;
  static constexpr double sw0 = 8.24493e-01;
  static constexpr double sw1 = -4.0899e-03;
  static constexpr double sw2 = 7.6438e-05;
  static constexpr double sw3 = -8.2467e-07;
  static constexpr double sw4 = 5.3875e-09;
  static constexpr double sw5 = -5.72466e-03;
  static constexpr double sw6 = 1.0227e-04;
  static constexpr double sw7 = -1.6546e-06;
  static constexpr double sw8 = 4.8314e-04;
  static constexpr double kfw0 = 1.965933e+04;
  static constexpr double kfw1 = 1.444304e+02;
  static constexpr double kfw2 = -1.706103e+00;
  static constexpr double kfw3 = 9.648704e-03;
  static constexpr double kfw4 = -4.190253e-05;
  static constexpr double ksw0 = 5.284855e+01;
  static constexpr double ksw1 = -3.101089e-01;
  static constexpr double ksw2 = 6.283263e-03;
  static constexpr double ksw3 = -5.084188e-05;
  static constexpr double ksw4 = 3.886640e-01;
  static constexpr double ksw5 = 9.085835e-03;
  static constexpr double ksw6 = -4.619924e-04;
  static constexpr double kp0 = 3.186519e+00;
  static constexpr double kp1 = 2.212276e-02;
  static constexpr double kp2 = -2.984642e-04;
  static constexpr double kp3 = 1.956415e-06;
  static constexpr double kp4 = 6.704388e-03;
  static constexpr double kp5 = -1.847318e-04;
  static constexpr double kp6 = 2.059331e-07;
  static constexpr double kp7 = 1.480266e-04;
  static constexpr double kp8 = 2.102898e-04;
  static constexpr double kp9 = -1.202016e-05;
  static constexpr double kp10 = 1.394680e-07;
  static constexpr double kp11 = -2.040237e-06;
  static constexpr double kp12 = 6.128773e-08;
  static constexpr double kp13 = 6.207323e-10;
  static constexpr double si_to_p = 1.0e-05;
};

struct Mdjwf {
  static constexpr double n0 = 9.99843699e+02;
  static constexpr double n1 = 7.35212840e+00;
  static constexpr double n2 = -5.45928211e-02;
  static constexpr double n3 = 3.98476704e-04;
  static constexpr double n4 = 2.96938239e+00;
  static constexpr double n5 = -7.23268813e-03;
  static constexpr double n6 = 2.12382341e-03;
  static constexpr double n7 = 1.04004591e-02;
  static constexpr double n8 = 1.03970529e-07;
  static constexpr double n9 = 5.18761880e-06;
  static constexpr double n10 = -3.24041825e-08;
  static constexpr double n11 = -1.23869360e-11;
  static constexpr double d0 = 1.00000000e+00;
  static constexpr double d1 = 7.28606739e-03;
  static constexpr double d2 = -4.60835542e-05;
  static constexpr double d3 = 3.68390573e-07;
  static constexpr double d4 = 1.80809186e-10;
  static constexpr double d5 = 2.14691708e-03;
  static constexpr double d6 = -9.27062484e-06;
  static constexpr double d7 = -1.78343643e-10;
  static constexpr double d8 = 4.76534122e-06;
  static constexpr double d9 = 1.63410736e-09;
  static constexpr double d10 = 5.30848875e-06;
  static constexpr double d11 = -3.03175128e-16;
  static constexpr double d12 = -1.27934137e-17;
  static constexpr double si_to_p = 1.0e-04;
};

// This three-component differential belongs only to the EOS point mapping.
// Branches inspect the primal; the sentinel root's inactive derivative is zero.
struct InputPartials {
  double value, temp, salt, pressure;
  __device__ InputPartials(double v = 0.0, double t = 0.0, double s = 0.0,
                            double p = 0.0)
      : value(v), temp(t), salt(s), pressure(p) {}
};

__device__ __forceinline__ InputPartials operator+(
    InputPartials a, InputPartials b) {
  return {a.value + b.value, a.temp + b.temp, a.salt + b.salt,
          a.pressure + b.pressure};
}
__device__ __forceinline__ InputPartials operator-(
    InputPartials a, InputPartials b) {
  return {a.value - b.value, a.temp - b.temp, a.salt - b.salt,
          a.pressure - b.pressure};
}
__device__ __forceinline__ InputPartials operator*(
    InputPartials a, InputPartials b) {
  return {a.value * b.value,
          a.temp * b.value + a.value * b.temp,
          a.salt * b.value + a.value * b.salt,
          a.pressure * b.value + a.value * b.pressure};
}
__device__ __forceinline__ InputPartials operator/(
    InputPartials a, InputPartials b) {
  const double quotient = a.value / b.value;
  return {quotient, (a.temp - quotient * b.temp) / b.value,
          (a.salt - quotient * b.salt) / b.value,
          (a.pressure - quotient * b.pressure) / b.value};
}
__device__ __forceinline__ bool operator<(InputPartials a, double b) {
  return a.value < b;
}
__device__ __forceinline__ bool operator>(InputPartials a, double b) {
  return a.value > b;
}
__device__ __forceinline__ InputPartials sqrt(InputPartials a) {
  const double root = ::sqrt(a.value);
  const double scale = 0.5 / root;
  return {root, a.temp * scale, a.salt * scale, a.pressure * scale};
}
using ::sqrt;

template <typename Scalar>
__device__ __forceinline__ void clamp_and_root(Scalar salt_input, Scalar &salt,
                                                Scalar &salt_root) {
  salt = salt_input < 0.0 ? 0.0 : salt_input;
  const bool positive = salt > 0.0;
  const Scalar safe = positive ? salt : 1.0;
  salt_root = positive ? sqrt(safe) : 0.0;
}

template <typename Scalar>
__device__ __forceinline__ void jmd95_surface_density_and_bulk_mod(
    const Scalar t, const Scalar s, const Scalar s_root, const Scalar p_bar,
    const Scalar t2, const Scalar t3, Scalar &rho_p0, Scalar &bulk_mod) {
  const Scalar t4 = t3 * t;
  const Scalar s3o2 = s * s_root;
  const Scalar rfresh =
      Jmd95::fw0 + Jmd95::fw1 * t + Jmd95::fw2 * t2 + Jmd95::fw3 * t3 +
      Jmd95::fw4 * t4 + Jmd95::fw5 * t4 * t;
  const Scalar rsalt =
      s * (Jmd95::sw0 + Jmd95::sw1 * t + Jmd95::sw2 * t2 +
           Jmd95::sw3 * t3 + Jmd95::sw4 * t4) +
      s3o2 * (Jmd95::sw5 + Jmd95::sw6 * t + Jmd95::sw7 * t2) +
      Jmd95::sw8 * s * s;
  rho_p0 = rfresh + rsalt;
  const Scalar bulk_fresh =
      Jmd95::kfw0 + Jmd95::kfw1 * t + Jmd95::kfw2 * t2 +
      Jmd95::kfw3 * t3 + Jmd95::kfw4 * t4;
  const Scalar bulk_salt =
      s * (Jmd95::ksw0 + Jmd95::ksw1 * t + Jmd95::ksw2 * t2 +
           Jmd95::ksw3 * t3) +
      s3o2 * (Jmd95::ksw4 + Jmd95::ksw5 * t + Jmd95::ksw6 * t2);
  const Scalar bulk_pres =
      p_bar * (Jmd95::kp0 + Jmd95::kp1 * t + Jmd95::kp2 * t2 +
               Jmd95::kp3 * t3) +
      p_bar * s * (Jmd95::kp4 + Jmd95::kp5 * t + Jmd95::kp6 * t2) +
      p_bar * s3o2 * Jmd95::kp7 +
      p_bar * p_bar * (Jmd95::kp8 + Jmd95::kp9 * t + Jmd95::kp10 * t2) +
      p_bar * p_bar * s *
          (Jmd95::kp11 + Jmd95::kp12 * t + Jmd95::kp13 * t2);
  bulk_mod = bulk_fresh + bulk_salt + bulk_pres;
}

template <typename Scalar>
__device__ __forceinline__ Scalar jmd95_density_anomaly(
    const Scalar t, const Scalar s, const Scalar s_root, const Scalar p_bar,
    const Scalar reference_density) {
  const Scalar t2 = t * t;
  const Scalar t3 = t2 * t;
  Scalar rho_p0, bulk_mod;
  jmd95_surface_density_and_bulk_mod(t, s, s_root, p_bar, t2, t3, rho_p0, bulk_mod);
  return rho_p0 / (1.0 - p_bar / bulk_mod) - reference_density;
}

template <typename Scalar, int N>
__device__ __forceinline__ Scalar jmd95_polynomial(
    const double (&coefficients)[N], Scalar t) {
  Scalar value = coefficients[N - 1];
#pragma unroll
  for (int i = N - 2; i >= 0; --i) value = value * t + coefficients[i];
  return value;
}

template <typename Scalar, int N>
__device__ __forceinline__ Scalar jmd95_divided_difference(
    const double (&coefficients)[N], Scalar a, Scalar b) {
  Scalar value = coefficients[N - 1], difference = 0.0;
#pragma unroll
  for (int i = N - 2; i >= 0; --i) {
    difference = value + b * difference;
    value = value * a + coefficients[i];
  }
  return difference;
}

template <typename Scalar>
__device__ __forceinline__ Scalar jmd95_density_difference(
    Scalar ta, Scalar sa, Scalar ra, Scalar tb, Scalar sb, Scalar rb, Scalar p) {
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
  const Scalar root_sum = ra + rb;
  const bool positive = root_sum > 0.0;
  const Scalar safe_sum = positive ? root_sum : 1.0;
  const Scalar ds15 = positive ? (sa + ra * rb + sb) / safe_sum : 0.0;
  const Scalar dt = ta - tb, ds = sa - sb;
  Scalar delta_r = dt * (
      jmd95_divided_difference(f, ta, tb) + sa * jmd95_divided_difference(w, ta, tb)
      + sa * ra * jmd95_divided_difference(wr, ta, tb));
  delta_r = delta_r + ds * (jmd95_polynomial(w, tb)
      + ds15 * jmd95_polynomial(wr, tb) + Jmd95::sw8 * (sa + sb));
  Scalar delta_k = dt * (
      jmd95_divided_difference(kf, ta, tb) + sa * jmd95_divided_difference(ks, ta, tb)
      + sa * ra * jmd95_divided_difference(ksr, ta, tb));
  delta_k = delta_k + ds * (jmd95_polynomial(ks, tb)
      + ds15 * jmd95_polynomial(ksr, tb));
  delta_k = delta_k + p * (
      dt * (jmd95_divided_difference(c, ta, tb)
            + sa * jmd95_divided_difference(cs, ta, tb))
      + ds * (jmd95_polynomial(cs, tb) + ds15 * Jmd95::kp7));
  delta_k = delta_k + p * p * (
      dt * (jmd95_divided_difference(c2, ta, tb)
            + sa * jmd95_divided_difference(c2s, ta, tb))
      + ds * jmd95_polynomial(c2s, tb));
  Scalar ka = jmd95_polynomial(kf, ta) + sa * jmd95_polynomial(ks, ta)
      + sa * ra * jmd95_polynomial(ksr, ta);
  ka = ka + p * (jmd95_polynomial(c, ta) + sa * jmd95_polynomial(cs, ta)
                  + sa * ra * Jmd95::kp7)
      + p * p * (jmd95_polynomial(c2, ta) + sa * jmd95_polynomial(c2s, ta));
  const Scalar surface_b = jmd95_polynomial(f, tb) + sb * jmd95_polynomial(w, tb)
      + sb * rb * jmd95_polynomial(wr, tb) + Jmd95::sw8 * sb * sb;
  Scalar kb = jmd95_polynomial(kf, tb) + sb * jmd95_polynomial(ks, tb)
      + sb * rb * jmd95_polynomial(ksr, tb);
  kb = kb + p * (jmd95_polynomial(c, tb) + sb * jmd95_polynomial(cs, tb)
                  + sb * rb * Jmd95::kp7)
      + p * p * (jmd95_polynomial(c2, tb) + sb * jmd95_polynomial(c2s, tb));
  return delta_r * ka / (ka - p) - surface_b * p * delta_k / ((ka - p) * (kb - p));
}

struct DifferencePartials {
  double value, temp_a, salt_a, temp_b, salt_b, pressure;
};

__device__ __forceinline__ DifferencePartials jmd95_difference_partials(
    double ta, double sa_input, double tb, double sb_input, double pressure_pa) {
  // Two evaluations of the existing three-direction differential retain all
  // five inputs, including the common pressure's near-cancelling derivative.
  InputPartials sa, ra, sb, rb;
  clamp_and_root(InputPartials(sa_input, 0.0, 1.0), sa, ra);
  clamp_and_root(InputPartials(sb_input), sb, rb);
  const auto a = jmd95_density_difference(
      InputPartials(ta, 1.0), sa, ra, InputPartials(tb), sb, rb,
      InputPartials(pressure_pa, 0.0, 0.0, 1.0) * Jmd95::si_to_p);
  clamp_and_root(InputPartials(sa_input), sa, ra);
  clamp_and_root(InputPartials(sb_input, 0.0, 1.0), sb, rb);
  const auto b = jmd95_density_difference(
      InputPartials(ta), sa, ra, InputPartials(tb, 1.0), sb, rb,
      InputPartials(pressure_pa * Jmd95::si_to_p));
  return {a.value, a.temp, a.salt, b.temp, b.salt, a.pressure};
}

template <typename Scalar>
__device__ __forceinline__ void jmd95_alpha_beta(
    const Scalar t1, const Scalar s1, const Scalar sqrt_s, const Scalar p1,
    Scalar &alpha, Scalar &beta) {
  const Scalar t2 = t1 * t1;
  const Scalar t3 = t2 * t1;
  const Scalar p2 = p1 * p1;
  Scalar rho_p0, bulk_mod;
  jmd95_surface_density_and_bulk_mod(t1, s1, sqrt_s, p1, t2, t3, rho_p0, bulk_mod);
  const Scalar k_minus_p = bulk_mod - p1;
  const Scalar k_minus_p_squared = k_minus_p * k_minus_p;
  const Scalar s3 = s1 * s1 * s1;
  const Scalar s3o2_a = s3 > 0.0 ? sqrt(s3) : Scalar(0.0);
  const Scalar drho_fresh =
      (Jmd95::fw1 + 2.0 * Jmd95::fw2 * t1 + 3.0 * Jmd95::fw3 * t2 +
       4.0 * Jmd95::fw4 * t3 + 5.0 * Jmd95::fw5 * t3 * t1);
  const Scalar drho_salt =
      s1 * (Jmd95::sw1 + 2.0 * Jmd95::sw2 * t1 +
            3.0 * Jmd95::sw3 * t2 + 4.0 * Jmd95::sw4 * t3) +
      s3o2_a * (Jmd95::sw6 + 2.0 * Jmd95::sw7 * t1);
  const Scalar dk_fresh =
      (Jmd95::kfw1 + 2.0 * Jmd95::kfw2 * t1 + 3.0 * Jmd95::kfw3 * t2 +
       4.0 * Jmd95::kfw4 * t3);
  const Scalar dk_salt =
      s1 * (Jmd95::ksw1 + 2.0 * Jmd95::ksw2 * t1 +
            3.0 * Jmd95::ksw3 * t2) +
      s3o2_a * (Jmd95::ksw5 + 2.0 * Jmd95::ksw6 * t1);
  const Scalar dk_pressure =
      p1 * (Jmd95::kp1 + 2.0 * Jmd95::kp2 * t1 +
            3.0 * Jmd95::kp3 * t2) +
      p1 * s1 * (Jmd95::kp5 + 2.0 * Jmd95::kp6 * t1) +
      p2 * (Jmd95::kp9 + 2.0 * Jmd95::kp10 * t1) +
      p2 * s1 * (Jmd95::kp12 + 2.0 * Jmd95::kp13 * t1);
  const Scalar drho_p0_dtheta = drho_fresh + drho_salt;
  const Scalar dk_dtheta = dk_fresh + dk_salt + dk_pressure;
  alpha = (bulk_mod * bulk_mod * drho_p0_dtheta -
           bulk_mod * p1 * drho_p0_dtheta - rho_p0 * p1 * dk_dtheta) /
          k_minus_p_squared;
  const Scalar s3o2_b = 1.5 * sqrt_s;
  const Scalar drho_p0_ds =
      Jmd95::sw0 + Jmd95::sw1 * t1 + Jmd95::sw2 * t2 + Jmd95::sw3 * t3 +
      Jmd95::sw4 * t3 * t1 +
      s3o2_b * (Jmd95::sw5 + Jmd95::sw6 * t1 + Jmd95::sw7 * t2) +
      2.0 * Jmd95::sw8 * s1;
  const Scalar dk_ds_salt =
      (Jmd95::ksw0 + Jmd95::ksw1 * t1 + Jmd95::ksw2 * t2 +
       Jmd95::ksw3 * t3) +
      s3o2_b * (Jmd95::ksw4 + Jmd95::ksw5 * t1 + Jmd95::ksw6 * t2);
  const Scalar dk_ds_pressure =
      p1 * (Jmd95::kp4 + Jmd95::kp5 * t1 + Jmd95::kp6 * t2) +
      s3o2_b * p1 * Jmd95::kp7 +
      p2 * (Jmd95::kp11 + Jmd95::kp12 * t1 + Jmd95::kp13 * t2);
  const Scalar dk_ds = dk_ds_salt + dk_ds_pressure;
  beta = (bulk_mod * bulk_mod * drho_p0_ds -
          bulk_mod * p1 * drho_p0_ds - rho_p0 * p1 * dk_ds) /
         k_minus_p_squared;
}

template <typename Scalar>
__device__ __forceinline__ Scalar mdjwf_density_anomaly(
    const Scalar t1, const Scalar s, const Scalar sp5, const Scalar p1,
    const Scalar reference_density) {
  const Scalar t2 = t1 * t1;
  const Scalar p1t1 = p1 * t1;
  const Scalar rho_num =
      Mdjwf::n0 + t1 * (Mdjwf::n1 + t1 * (Mdjwf::n2 + Mdjwf::n3 * t1)) +
      s * (Mdjwf::n4 + Mdjwf::n5 * t1 + Mdjwf::n6 * s) +
      p1 * (Mdjwf::n7 + Mdjwf::n8 * t2 + Mdjwf::n9 * s +
            p1 * (Mdjwf::n10 + Mdjwf::n11 * t2));
  const Scalar rho_den =
      Mdjwf::d0 +
      t1 * (Mdjwf::d1 +
            t1 * (Mdjwf::d2 + t1 * (Mdjwf::d3 + t1 * Mdjwf::d4))) +
      s * (Mdjwf::d5 + t1 * (Mdjwf::d6 + Mdjwf::d7 * t2) +
           sp5 * (Mdjwf::d8 + Mdjwf::d9 * t2)) +
      p1 * (Mdjwf::d10 +
            p1t1 * (Mdjwf::d11 * t2 + Mdjwf::d12 * p1));
  const Scalar rho = rho_num * (1.0 / rho_den);
  return rho - reference_density;
}

template <typename Scalar>
__device__ __forceinline__ void mdjwf_alpha_beta(
    const Scalar t1, const Scalar s1, const Scalar sp5, const Scalar p1,
    Scalar &alpha, Scalar &beta) {
  const Scalar t2 = t1 * t1;
  const Scalar p1t1 = p1 * t1;
  const Scalar rho_num =
      Mdjwf::n0 + t1 * (Mdjwf::n1 + t1 * (Mdjwf::n2 + Mdjwf::n3 * t1)) +
      s1 * (Mdjwf::n4 + Mdjwf::n5 * t1 + Mdjwf::n6 * s1) +
      p1 * (Mdjwf::n7 + Mdjwf::n8 * t2 + Mdjwf::n9 * s1 +
            p1 * (Mdjwf::n10 + Mdjwf::n11 * t2));
  const Scalar rho_den =
      1.0 /
      (Mdjwf::d0 +
       t1 * (Mdjwf::d1 +
             t1 * (Mdjwf::d2 + t1 * (Mdjwf::d3 + t1 * Mdjwf::d4))) +
       s1 * (Mdjwf::d5 + t1 * (Mdjwf::d6 + Mdjwf::d7 * t2) +
             sp5 * (Mdjwf::d8 + Mdjwf::d9 * t2)) +
       p1 * (Mdjwf::d10 +
             p1t1 * (Mdjwf::d11 * t2 + Mdjwf::d12 * p1)));
  const Scalar dnum_dtheta =
      Mdjwf::n1 + t1 * (2.0 * Mdjwf::n2 + 3.0 * Mdjwf::n3 * t1) +
      Mdjwf::n5 * s1 +
      p1t1 * (2.0 * Mdjwf::n8 + 2.0 * Mdjwf::n11 * p1);
  const Scalar dden_dtheta =
      Mdjwf::d1 +
      t1 * (2.0 * Mdjwf::d2 +
            t1 * (3.0 * Mdjwf::d3 + 4.0 * Mdjwf::d4 * t1)) +
      s1 * (Mdjwf::d6 +
            t1 * (3.0 * Mdjwf::d7 * t1 + 2.0 * Mdjwf::d9 * sp5)) +
      p1 * p1 * (3.0 * Mdjwf::d11 * t2 + Mdjwf::d12 * p1);
  alpha = rho_den *
          (dnum_dtheta - (rho_num * rho_den) * dden_dtheta);

  const Scalar dnum_dsalt = Mdjwf::n4 + Mdjwf::n5 * t1 +
                            2.0 * Mdjwf::n6 * s1 + Mdjwf::n9 * p1;
  const Scalar dden_dsalt =
      Mdjwf::d5 + t1 * (Mdjwf::d6 + Mdjwf::d7 * t2) +
      1.5 * sp5 * (Mdjwf::d8 + Mdjwf::d9 * t2);
  beta =
      rho_den * (dnum_dsalt - (rho_num * rho_den) * dden_dsalt);
}

template <class Eos, typename Scalar>
__device__ __forceinline__ Scalar density_anomaly(
    Scalar t, Scalar s, Scalar root, Scalar p, Scalar reference_density) {
  if constexpr (std::is_same_v<Eos, Jmd95>) {
    return jmd95_density_anomaly(t, s, root, p, reference_density);
  } else {
    return mdjwf_density_anomaly(t, s, root, p, reference_density);
  }
}

template <class Eos, typename Scalar>
__device__ __forceinline__ void alpha_beta(
    Scalar t, Scalar s, Scalar root, Scalar p, Scalar &alpha, Scalar &beta) {
  if constexpr (std::is_same_v<Eos, Jmd95>) {
    jmd95_alpha_beta(t, s, root, p, alpha, beta);
  } else {
    mdjwf_alpha_beta(t, s, root, p, alpha, beta);
  }
}

// Pressure is supplied in Pa here, so the returned pressure component already
// includes the EOS-specific unit conversion. Alpha/beta are independent outputs:
// differentiating their full expressions includes all coefficient feedback.
template <class Eos>
__device__ __forceinline__ InputPartials local_vjp(
    double temp, double salt_input, double pressure_pa, double reference_density,
    double density_seed, double alpha_seed, double beta_seed) {
  const InputPartials t(temp, 1.0, 0.0, 0.0);
  const InputPartials s_input(salt_input, 0.0, 1.0, 0.0);
  const InputPartials p = InputPartials(pressure_pa, 0.0, 0.0, 1.0) * Eos::si_to_p;
  InputPartials s, root;
  clamp_and_root(s_input, s, root);
  InputPartials result;
  if (density_seed != 0.0) {
    result = density_seed * density_anomaly<Eos>(
        t, s, root, p, InputPartials(reference_density));
  }
  if (alpha_seed != 0.0 || beta_seed != 0.0) {
    InputPartials alpha, beta;
    alpha_beta<Eos>(t, s, root, p, alpha, beta);
    result = result + alpha_seed * alpha + beta_seed * beta;
  }
  return result;
}

} // namespace locomgcm::local_eos
