// Sea-ice LSR line recurrence (SEAICE_LSR_TRIDIAGU/V), the ordered residual
// sums of SEAICE_RESIDUAL and the per-iteration advance of a replayed block of
// linear iterations, in the operation order of the eager reference
// (defgcm/seaice/dynamics/sweep.py, lsr.py, defgcm/seaice/space.py). Built with
// -fmad=false and IEEE division: every product and sum rounds separately.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

// Device state of a replayed block (mirrored in graph.py).
namespace lsr_state {
enum Ints { kActiveU, kActiveV, kCountU, kCountV, kM, kMMax, kRan, kSweptU, kSweptV, kInts };
enum Reals { kS1, kS2, kS1A, kS2A, kWU, kWV, kErrorU, kErrorV, kReals };
enum Log { kLogRan, kLogSweptU, kLogSweptV, kLogCheckedU, kLogCheckedV, kLogActiveU, kLogActiveV,
           kLogCountU, kLogCountV, kLogM, kLogInts };
constexpr int kLogReals = 4;
}  // namespace lsr_state

constexpr int kMaxPoints = 64;
constexpr int kSweepThreads = 64;
constexpr int kFoldThreads = 256;
constexpr int kFoldChunk = 1024;
constexpr int kAdvanceThreads = 512;

// One direction of the native LINE rectangle. U: lines are rows (line stride
// one row), points are columns; V: lines are columns, points are rows.
struct Lines {
  long tile;     // elements between tiles (bj-major, bi-minor)
  long line;     // elements between neighbouring lines
  long point;    // elements between neighbouring points of a line
  int first;     // native index of the first line and of the first point
  int nlines;
  int npoints;
  int ntiles;
  int rows;      // native tile rows and columns
  int cols;
};

// bet and CUU of every line, one thread per (tile, line) (TRIDIAGU:2011-2034).
__global__ void factor_kernel(const double* __restrict__ a, const double* __restrict__ b,
                              const double* __restrict__ c, double* __restrict__ factors,
                              Lines g) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= g.ntiles * g.nlines) return;
  const int t = id / g.nlines, k = id % g.nlines;
  const long base = t * g.tile + (g.first + k) * g.line + g.first * g.point;
  double* bet = factors + ((long)t * g.nlines + k) * g.npoints;
  double* cuu = factors + ((long)(g.ntiles + t) * g.nlines + k) * g.npoints;
  double upper = c[base] / b[base];
  bet[0] = b[base];
  cuu[0] = upper;
  for (int p = 1; p < g.npoints; ++p) {
    const long o = base + p * g.point;
    const double d = b[o] - a[o] * upper;
    bet[p] = d;
    upper = c[o] / d;
    cuu[p] = upper;
  }
}

struct SweepSide {
  const double* coef;     // [a, b, c, r1, r2, rhs, mask] native fields
  const double* factors;  // [bet, CUU] per line
  const double* old;      // sweep entry (uTmp/vTmp)
  const double* relax;    // 0-d relaxation factor
  double* out;            // every cell of every tile is written
  Lines g;
};

// One tile of one direction: an ordered Gauss-Seidel sweep. Lines run in
// increasing order and read the previous line's new values; every other
// neighbour is the sweep entry ``old``. The block evaluates each line's loads
// and relaxations in parallel; thread 0 alone runs the forward elimination
// and the back substitution, so every value has the operands and the
// operation order of the single-thread recurrence.
__device__ void sweep_tile(const SweepSide& s, long field, int t, bool swept) {
  const Lines& g = s.g;
  const long tb = t * g.tile;
  const int tid = threadIdx.x, nt = blockDim.x;
  // Cells outside the LINE rectangle keep ``old``; all of them when unswept.
  for (int idx = tid; idx < g.tile; idx += nt) {
    const int r = idx / g.cols, c = idx % g.cols;
    const bool inside = swept && r >= g.first && r < g.rows - g.first && c >= g.first && c < g.cols - g.first;
    if (!inside) s.out[tb + idx] = s.old[tb + idx];
  }
  if (!swept) return;
  const double* a = s.coef;
  const double* b = s.coef + field;
  const double* c = s.coef + 2 * field;
  const double* r1 = s.coef + 3 * field;
  const double* r2 = s.coef + 4 * field;
  const double* rhs = s.coef + 5 * field;
  const double* mask = s.coef + 6 * field;
  const double* old = s.old;
  const double relax = *s.relax;
  const int np = g.npoints;
  __shared__ double previous[kMaxPoints], load[kMaxPoints], lower[kMaxPoints];
  __shared__ double divisor[kMaxPoints], upper[kMaxPoints], base[kMaxPoints], values[kMaxPoints];
  for (int p = tid; p < np; p += nt) {
    previous[p] = old[tb + (g.first - 1) * g.line + (g.first + p) * g.point];
  }
  __syncthreads();
  for (int k = 0; k < g.nlines; ++k) {
    const long lb = tb + (g.first + k) * g.line;
    const double* bet = s.factors + ((long)t * g.nlines + k) * np;
    const double* cuu = s.factors + ((long)(g.ntiles + t) * g.nlines + k) * np;
    for (int p = tid; p < np; p += nt) {
      const long o = lb + (g.first + p) * g.point;
      double aa3 = 0.0;
      if (p == 0) {
        aa3 = 0.0 - a[o] * old[o - g.point];
      } else if (p == np - 1) {
        aa3 = 0.0 - c[o] * old[o + g.point];
      }
      double value = rhs[o] + aa3;
      value = value + r1[o] * previous[p];
      value = value + r2[o] * old[o + g.line];
      load[p] = value * mask[o];
      lower[p] = a[o];
      divisor[p] = p == 0 ? b[o] : bet[p];
      upper[p] = cuu[p];
      base[p] = old[o];
    }
    __syncthreads();
    if (tid == 0) {
      double v = load[0] / divisor[0];
      values[0] = v;
      for (int p = 1; p < np; ++p) {
        v = (load[p] - lower[p] * v) / divisor[p];
        values[p] = v;
      }
      for (int p = np - 2; p >= 0; --p) {
        v = values[p] - upper[p] * v;
        values[p] = v;
      }
    }
    __syncthreads();
    for (int p = tid; p < np; p += nt) {
      const double updated = base[p] + relax * (values[p] - base[p]);
      s.out[lb + (g.first + p) * g.point] = updated;
      previous[p] = updated;
    }
    __syncthreads();
  }
}

// Block b < ntiles sweeps U tile b, the others V. ``state`` (replayed blocks)
// decides the executed iteration and its swept directions on the device;
// without it ``swept`` holds the host's choice.
__global__ void sweep_kernel(SweepSide u, SweepSide v, long field, int swept, int* __restrict__ state) {
  if (state != nullptr) {
    const int ran = (state[lsr_state::kActiveU] || state[lsr_state::kActiveV]) &&
                    state[lsr_state::kM] <= state[lsr_state::kMMax];
    swept = ran ? ((state[lsr_state::kActiveU] ? 1 : 0) | (state[lsr_state::kActiveV] ? 2 : 0)) : 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      state[lsr_state::kRan] = ran;
      state[lsr_state::kSweptU] = swept & 1;
      state[lsr_state::kSweptV] = (swept >> 1) & 1;
    }
  }
  const int tiles = u.g.ntiles;
  if ((int)blockIdx.x < tiles) {
    sweep_tile(u, field, blockIdx.x, swept & 1);
  } else {
    sweep_tile(v, field, blockIdx.x - tiles, (swept & 2) != 0);
  }
}

// A strided 4-D block in logical row-major order, from ``origin``.
struct Block4 {
  long n0, n1, n2, n3;
  long s0, s1, s2, s3;
  long origin;
  __host__ __device__ long count() const { return n0 * n1 * n2 * n3; }
  __device__ long offset(long e) const {
    const long i3 = e % n3;
    e /= n3;
    const long i2 = e % n2;
    e /= n2;
    const long i1 = e % n1;
    const long i0 = e / n1;
    return origin + i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;
  }
};

// Left folds from +0.0 in logical row-major order of each block. All threads
// stage chunks in shared memory; lane f of warp 0 folds block f, so the
// additions of each fold keep their order and the folds share one launch.
__device__ void fold_blocks(const double* const* x, int folds, Block4 shape, double* totals) {
  __shared__ double chunk[2][kFoldChunk + 1];
  const int tid = threadIdx.x, nt = blockDim.x;
  const long n = shape.count();
  double total = 0.0;
  for (long start = 0; start < n; start += kFoldChunk) {
    const int length = (int)min((long)kFoldChunk, n - start);
    for (int f = 0; f < folds; ++f) {
      for (int e = tid; e < length; e += nt) chunk[f][e] = x[f][shape.offset(start + e)];
    }
    __syncthreads();
    if (tid < folds) {
      const double* values = chunk[tid];
#pragma unroll 8
      for (int e = 0; e < length; ++e) total = total + values[e];
    }
    __syncthreads();
  }
  if (tid < folds) totals[tid] = total;
}

__global__ void ordered_sum_kernel(const double* x0, const double* x1, double* out, int folds, Block4 shape) {
  const double* x[2] = {x0, x1};
  fold_blocks(x, folds, shape, out);
}

__device__ bool before_nan(double value, double best) {
  return isnan(value) || (!isnan(best) && value > best);
}

// NaN-propagating maximum of |(pre - old) * mask| over INNER, as
// space.max_abs; the order of a maximum does not change its value.
__device__ double max_abs_change(const double* pre, const double* old, const double* mask,
                                 Block4 inner, double* scratch) {
  const int tid = threadIdx.x, nt = blockDim.x;
  double best = 0.0;
  bool any = false;
  const long n = inner.count();
  for (long e = tid; e < n; e += nt) {
    const long o = inner.offset(e);
    const double value = fabs((pre[o] - old[o]) * mask[o]);
    if (!any || before_nan(value, best)) best = value;
    any = true;
  }
  scratch[tid] = any ? best : -1.0;  // |x| >= 0: -1 marks an empty thread
  __syncthreads();
  for (int width = nt / 2; width > 0; width /= 2) {
    if (tid < width) {
      const double other = scratch[tid + width];
      if (other != -1.0 && (scratch[tid] == -1.0 || before_nan(other, scratch[tid]))) scratch[tid] = other;
    }
    __syncthreads();
  }
  return scratch[0];
}

struct AdvanceArgs {
  int* state;
  double* reals;
  int* log_ints;          // [K, kLogInts]
  double* log_reals;      // [K, kLogReals]
  const double* pre_u;
  const double* pre_v;
  double* u;              // iteration entry, then X6 result
  double* v;
  const int* exchange;    // X6 source of every native cell
  long cells;
  const double* mask_u;   // seaiceMaskU, seaiceMaskV
  const double* mask_v;
  const double* flex_u;   // residual means of a flex check (0-d), or null
  const double* flex_v;
  Block4 inner;
  int position;           // row of this iteration in the block log
  int check;              // m % SOLV_NCHECK == 0 at this position
  int flex;               // SEAICEuseLSRflex
};

// seaice_lsr.F:917-987 for one executed iteration: the check and its stop and
// relaxation changes, the record row, and X6 of both velocities.
__global__ void advance_kernel(AdvanceArgs a) {
  using namespace lsr_state;
  __shared__ double scratch[kAdvanceThreads];
  __shared__ int ran_s;
  const int tid = threadIdx.x;
  int* s = a.state;
  double* r = a.reals;
  const int ran = s[kRan];
  const int m = s[kM];
  int checked_u = 0, checked_v = 0;
  if (ran && a.check) {
    if (a.flex) {
      if (tid == 0) {
        checked_u = checked_v = 1;
        r[kS1] = *a.flex_u;
        r[kS2] = *a.flex_v;
        // These assignments intentionally also apply to a stopped direction.
        if (r[kS1] < r[kErrorU]) { s[kCountU] = m; s[kActiveU] = 0; }
        if (r[kS2] < r[kErrorV]) { s[kCountV] = m; s[kActiveV] = 0; }
      }
    } else {
      checked_u = s[kActiveU];
      checked_v = s[kActiveV];
      const bool safeguard = m > 1;
      double s1 = r[kS1], s2 = r[kS2];
      if (checked_u) s1 = max_abs_change(a.pre_u, a.u, a.mask_u, a.inner, scratch);
      __syncthreads();
      if (checked_v) s2 = max_abs_change(a.pre_v, a.v, a.mask_v, a.inner, scratch);
      if (tid == 0) {
        if (checked_u) {
          if (safeguard && s1 > r[kS1A]) r[kWU] = 0.0;
          r[kS1A] = s1;
          r[kS1] = s1;
        }
        if (checked_v) {
          if (safeguard && s2 > r[kS2A]) r[kWV] = 0.0;
          r[kS2A] = s2;
          r[kS2] = s2;
        }
        if (checked_u && s1 < r[kErrorU]) { s[kCountU] = m; s[kActiveU] = 0; }
        if (checked_v && s2 < r[kErrorU]) { s[kCountV] = m; s[kActiveV] = 0; }
      }
    }
  }
  if (tid == 0) {
    int* row = a.log_ints + a.position * kLogInts;
    row[kLogRan] = ran;
    row[kLogSweptU] = s[kSweptU];
    row[kLogSweptV] = s[kSweptV];
    row[kLogCheckedU] = checked_u;
    row[kLogCheckedV] = checked_v;
    row[kLogActiveU] = s[kActiveU];
    row[kLogActiveV] = s[kActiveV];
    row[kLogCountU] = s[kCountU];
    row[kLogCountV] = s[kCountV];
    row[kLogM] = m;
    double* values = a.log_reals + a.position * kLogReals;
    values[0] = r[kS1];
    values[1] = r[kS2];
    values[2] = r[kWU];
    values[3] = r[kWV];
    if (ran) s[kM] = m + 1;
    ran_s = ran;
  }
  // The maxima above read ``u``/``v``; X6 overwrites them only afterwards.
  __syncthreads();
  if (ran_s) {
    for (long i = tid; i < a.cells; i += blockDim.x) {
      a.u[i] = a.pre_u[a.exchange[i]];
      a.v[i] = a.pre_v[a.exchange[i]];
    }
  }
}

Lines lines_of(const torch::Tensor& field, bool transpose) {
  // field: [bj, bi, j, i] contiguous native storage.
  const long rows = field.size(-2), cols = field.size(-1);
  Lines g;
  g.tile = rows * cols;
  g.first = 2;
  g.ntiles = (int)(field.numel() / g.tile);
  g.rows = (int)rows;
  g.cols = (int)cols;
  if (!transpose) {
    g.line = cols; g.point = 1; g.nlines = (int)rows - 4; g.npoints = (int)cols - 4;
  } else {
    g.line = 1; g.point = cols; g.nlines = (int)cols - 4; g.npoints = (int)rows - 4;
  }
  return g;
}

Block4 block_of(const torch::Tensor& x) {
  return Block4{x.size(0), x.size(1), x.size(2), x.size(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), 0};
}

SweepSide side(const torch::Tensor& coef, const torch::Tensor& factors, const torch::Tensor& old,
               const torch::Tensor& relax, torch::Tensor& out, bool transpose) {
  return SweepSide{coef.data_ptr<double>(), factors.data_ptr<double>(), old.data_ptr<double>(),
                   relax.data_ptr<double>(), out.data_ptr<double>(), lines_of(old, transpose)};
}

void launch_sweep(const SweepSide& u, const SweepSide& v, long field, int swept, int* state) {
  TORCH_CHECK(u.g.npoints <= kMaxPoints && v.g.npoints <= kMaxPoints, "LSR line longer than the kernel buffer");
  sweep_kernel<<<u.g.ntiles + v.g.ntiles, kSweepThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      u, v, field, swept, state);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor seaice_lsr_factor(torch::Tensor a, torch::Tensor b, torch::Tensor c, bool transpose) {
  const at::cuda::OptionalCUDAGuard guard(a.device());
  const Lines g = lines_of(a, transpose);
  auto shape = a.sizes().vec();
  shape.pop_back();
  shape.pop_back();
  shape.insert(shape.begin(), 2);
  shape.push_back(g.nlines);
  shape.push_back(g.npoints);
  auto factors = torch::empty(shape, a.options());
  const int threads = 128;
  const int total = g.ntiles * g.nlines;
  factor_kernel<<<(total + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<double>(), b.data_ptr<double>(), c.data_ptr<double>(), factors.data_ptr<double>(), g);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return factors;
}

std::vector<torch::Tensor> seaice_lsr_sweep(torch::Tensor coef_u, torch::Tensor factors_u,
                                            torch::Tensor old_u, torch::Tensor relax_u,
                                            torch::Tensor coef_v, torch::Tensor factors_v,
                                            torch::Tensor old_v, torch::Tensor relax_v,
                                            bool swept_u, bool swept_v) {
  const at::cuda::OptionalCUDAGuard guard(old_u.device());
  auto out_u = torch::empty_like(old_u);
  auto out_v = torch::empty_like(old_v);
  const int swept = (swept_u ? 1 : 0) | (swept_v ? 2 : 0);
  launch_sweep(side(coef_u, factors_u, old_u, relax_u, out_u, false),
               side(coef_v, factors_v, old_v, relax_v, out_v, true), old_u.numel(), swept, nullptr);
  return {out_u, out_v};
}

void seaice_lsr_sweep_state(torch::Tensor coef_u, torch::Tensor factors_u, torch::Tensor old_u,
                            torch::Tensor coef_v, torch::Tensor factors_v, torch::Tensor old_v,
                            torch::Tensor reals, torch::Tensor state, torch::Tensor out_u, torch::Tensor out_v) {
  using namespace lsr_state;
  const at::cuda::OptionalCUDAGuard guard(old_u.device());
  auto relax_u = reals.select(0, kWU), relax_v = reals.select(0, kWV);
  launch_sweep(side(coef_u, factors_u, old_u, relax_u, out_u, false),
               side(coef_v, factors_v, old_v, relax_v, out_v, true), old_u.numel(), 0,
               state.data_ptr<int>());
}

torch::Tensor seaice_ordered_sum(torch::Tensor block) {
  const at::cuda::OptionalCUDAGuard guard(block.device());
  auto out = torch::empty({}, block.options());
  ordered_sum_kernel<<<1, kFoldThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      block.data_ptr<double>(), nullptr, out.data_ptr<double>(), 1, block_of(block));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor seaice_ordered_sum_pair(torch::Tensor first, torch::Tensor second) {
  const at::cuda::OptionalCUDAGuard guard(first.device());
  auto out = torch::empty({2}, first.options());
  ordered_sum_kernel<<<1, kFoldThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      first.data_ptr<double>(), second.data_ptr<double>(), out.data_ptr<double>(), 2, block_of(first));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void seaice_lsr_advance(torch::Tensor state, torch::Tensor reals, torch::Tensor log_ints,
                        torch::Tensor log_reals, torch::Tensor pre_u, torch::Tensor pre_v,
                        torch::Tensor u, torch::Tensor v, torch::Tensor exchange,
                        torch::Tensor mask_u, torch::Tensor mask_v,
                        c10::optional<torch::Tensor> flex_u, c10::optional<torch::Tensor> flex_v,
                        int64_t position, bool check, bool flex) {
  const at::cuda::OptionalCUDAGuard guard(u.device());
  AdvanceArgs a;
  a.state = state.data_ptr<int>();
  a.reals = reals.data_ptr<double>();
  a.log_ints = log_ints.data_ptr<int>();
  a.log_reals = log_reals.data_ptr<double>();
  a.pre_u = pre_u.data_ptr<double>();
  a.pre_v = pre_v.data_ptr<double>();
  a.u = u.data_ptr<double>();
  a.v = v.data_ptr<double>();
  a.exchange = exchange.data_ptr<int>();
  a.cells = u.numel();
  a.mask_u = mask_u.data_ptr<double>();
  a.mask_v = mask_v.data_ptr<double>();
  a.flex_u = flex_u.has_value() ? flex_u->data_ptr<double>() : nullptr;
  a.flex_v = flex_v.has_value() ? flex_v->data_ptr<double>() : nullptr;
  // INNER = (3, 3, 3, 3) of the contiguous native [bj, bi, j, i] storage.
  const long rows = u.size(-2), cols = u.size(-1);
  a.inner = Block4{u.size(0), u.size(1), rows - 6, cols - 6, u.stride(0), u.stride(1), cols, 1, 3 * cols + 3};
  a.position = (int)position;
  a.check = check ? 1 : 0;
  a.flex = flex ? 1 : 0;
  advance_kernel<<<1, kAdvanceThreads, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
