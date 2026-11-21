// Local reverse of the two transport fluxes. Also compiled on CPU by the tests.
#pragma once
#include <cmath>

#ifdef __CUDACC__
#define DST3_INLINE __device__ __forceinline__
#else
#define DST3_INLINE inline
#endif

namespace dst3 {

template <typename T> DST3_INLINE T minimum(T a, T b) {
  if (a != a) return a;
  if (b != b) return b;
  return a <= b ? a : b;
}

// The generated MITgcm adjoint selects the first argument at an ordered tie.
template <typename T> DST3_INLINE void minimum_vjp(T a, T b, T g, T &ga, T &gb) {
  ga = a > b ? T(0) : g;
  gb = b >= a ? T(0) : g;
}

template <typename T> DST3_INLINE T ratio(T n, T d) {
  bool capped = fabs(d) * T(1.e20) <= fabs(n);
  T safe = capped ? T(1) : d;
  return capped ? (n * d >= T(0) ? T(1.e20) : -T(1.e20)) : n / safe;
}

template <typename T> DST3_INLINE void ratio_vjp(T n, T d, T g, T &gn, T &gd) {
  // Double where, as _ratio_safe: the division sees d only where its quotient
  // is selected, so a capped face routes nothing to n or d.
  bool capped = fabs(d) * T(1.e20) <= fabs(n);
  T safe = capped ? T(1) : d;
  T raw_g = capped ? T(0) : g;
  gn = raw_g / safe;
  // ATen div's denominator adjoint forms (n / d) / d before multiplying
  // the cotangent; keep that order on the selected faces.
  gd = capped ? T(0) : -raw_g * ((n / safe) / safe);
}

template <typename T> DST3_INLINE T psi(T theta, T c, T d0, T d1) {
  T p = minimum(T(1), d0 + d1 * theta);
  T q = theta * (T(1) - c) / (c + T(1.e-20));
  T m = minimum(p, q);
  return m < T(0) ? T(0) : m;
}

template <typename T> DST3_INLINE void psi_vjp(
    T theta, T c, T d0, T d1, T g, T &gt, T &gc, T &gd0, T &gd1) {
  T raw = d0 + d1 * theta;
  T p = minimum(T(1), raw);
  T denom = c + T(1.e-20);
  T bound = theta * (T(1) - c) / denom;
  T m = minimum(p, bound);
  // MAX(0, m) keeps the constant branch at either signed zero.
  T gm = m > T(0) ? g : T(0);
  T gp, gb, unused, gr;
  minimum_vjp(p, bound, gm, gp, gb);
  minimum_vjp(T(1), raw, gp, unused, gr);
  // Double where, as _dst3_psi: only a bound in [0, 1] can be selected, and
  // only there does the division's adjoint enter; outside, bound / denom
  // overflows float32 at c = 0 and the reference forms no derivative path.
  bool live = bound >= T(0) && bound <= T(1);
  T gn = live ? gb / denom : T(0);
  gt = gr * d1 + gn * (T(1) - c);
  gc = -gn * theta - (live ? gb * (bound / denom) : T(0));
  gd0 = gr;
  gd1 = gr * theta;
}

// Input order is the public core's order. Seeds correspond to upwind and DST3.
// The caller skips storage for inactive inputs; no dependency is dropped here.
template <typename T> DST3_INLINE void vjp(
    const T *v, T gu, T gf, bool limited, T *g,
    bool use_upwind = true, bool use_dst = true) {
  for (int i = 0; i < 11; ++i) g[i] = T(0);
  T u = v[9], c = v[10];
  T pos = T(0.5) * (u + fabs(u));
  T neg = T(0.5) * (u - fabs(u));
  // A missing output cotangent removes its graph entirely, unlike an explicit
  // zero cotangent that still traverses division and can propagate NaNs.
  if (!use_dst) {
    g[0] = gu * pos;
    g[1] = gu * neg;
    T gp = gu * v[0], gm = gu * v[1];
    T sign = u >= T(0) ? T(1) : T(-1);
    g[9] = T(0.5) * gp + T(0.5) * gm + sign * (T(0.5) * gp - T(0.5) * gm);
    return;
  }
  T rjm = (v[3] - v[2]) * v[6];
  T rj = (v[4] - v[3]) * v[7];
  T rjp = (v[5] - v[4]) * v[8];
  T d0 = (T(2) - c) * (T(1) - c) * T(1.0 / 6.0);
  T d1 = (T(1) - c * c) * T(1.0 / 6.0);
  T bp = gf * pos, bm = gf * neg;
  T grjm, grj, grjp, gd0, gd1, pc, mc;
  if (limited) {
    T tp = ratio(rjm, rj), tm = ratio(rjp, rj);
    T pp = psi(tp, c, d0, d1), pm = psi(tm, c, d0, d1);
    pc = v[3] + pp * rj;
    mc = v[4] - pm * rj;
    T gtp, gtm, gcp, gcm, d0p, d0m, d1p, d1m;
    psi_vjp(tp, c, d0, d1, bp * rj, gtp, gcp, d0p, d1p);
    psi_vjp(tm, c, d0, d1, -bm * rj, gtm, gcm, d0m, d1m);
    T gjp, gjm;
    ratio_vjp(rjm, rj, gtp, grjm, gjp);
    ratio_vjp(rjp, rj, gtm, grjp, gjm);
    grj = bp * pp - bm * pm + gjp + gjm;
    gd0 = d0p + d0m;
    gd1 = d1p + d1m;
    g[10] = gcp + gcm;
  } else {
    pc = v[3] + (d0 * rj + d1 * rjm);
    mc = v[4] - (d0 * rj + d1 * rjp);
    grjm = bp * d1;
    grjp = -bm * d1;
    grj = bp * d0 - bm * d0;
    gd0 = bp * rj - bm * rj;
    gd1 = bp * rjm - bm * rjp;
  }
  T gp = (use_upwind ? gu * v[0] : T(0)) + gf * pc;
  T gm = (use_upwind ? gu * v[1] : T(0)) + gf * mc;
  // abs_with_zero_slope(u, +1): +0 and -0 both take the positive branch.
  T sign = u >= T(0) ? T(1) : T(-1);
  g[9] = T(0.5) * gp + T(0.5) * gm +
         sign * (T(0.5) * gp - T(0.5) * gm);
  g[0] = use_upwind ? gu * pos : T(0);
  g[1] = use_upwind ? gu * neg : T(0);
  g[2] = -grjm * v[6];
  g[3] = bp + grjm * v[6] - grj * v[7];
  g[4] = bm + grj * v[7] - grjp * v[8];
  g[5] = grjp * v[8];
  g[6] = grjm * (v[3] - v[2]);
  g[7] = grj * (v[4] - v[3]);
  g[8] = grjp * (v[5] - v[4]);
  T a = gd0 * T(1.0 / 6.0), b = gd1 * T(1.0 / 6.0);
  g[10] += -a * (T(1) - c) - a * (T(2) - c) - b * c - b * c;
}

} // namespace dst3
#undef DST3_INLINE
