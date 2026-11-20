// Fused CUDA kernels for the CG3D preconditioned-conjugate-gradient body.
//
// Semantic authority = ``defgcm/solvers/cg3d.py::CG3DSolver.solve`` (the
// eager loop) together with the two expressions it calls out to:
//   * the operator apply, ``cg3d_operator.py::apply_cg3d_operator`` -- its
//     ``grid is not None`` branch, i.e. the PRODUCTION path, which reads an
//     already-exchanged padded block and applies the four horizontal terms
//     UNCONDITIONALLY (lines 341-345), then the two vertical terms (357-359);
//   * the preconditioner apply, ``cg3d_preconditioner.py:120-127`` -- the
//     vertical-line Thomas recurrence.
//
// TOPOLOGY.  There is none here, by construction.  Every
// kernel takes a padded interior block ``[F, nz, ny+2, nx+2]`` plus coefficient
// tensors and a mask, and contains no grid, no boundary type, no face index and
// no edge special-casing.  ``F`` is a plain outer dimension -- 1 for a regional
// lat-lon field, ``n_faces`` for a face-leading cubed-sphere field, and the
// kernel cannot tell which.  Whatever produced the padding (a zero fill, a
// periodic wrap, a peer-to-peer receive from one of N ranks, a neighbouring
// cube face rotated into this face's frame) is the injected exchange's business.
//
// THE FOUR HORIZONTAL TERMS ARE UNCONDITIONAL, and that is the whole
// correctness delta against the kernel this file replaces.  That kernel
// guarded them (``if (ii >= 1)`` ...) because it reproduced the ``grid is None``
// LOCAL-ONLY branch.  Against the production branch a guard is wrong, not
// conservative: where a side does not couple, ``aW[..., 0]`` is exactly 0.0
// (allocated zero at ``cg3d_operator.py:128``, written only under
// ``west_couples``), so the unconditional kernel computes what eager computes;
// where it DOES couple -- a periodic seam, a rank boundary -- the guard silently
// drops a real term.
//
// Bitwise parity with eager rests on three things, all load-bearing:
//   1. ``-fmad=false`` (see ``_fast.py::_CUDA_FLAGS``).  nvcc contracts
//      ``a + b * c`` into one FMA, rounding ONCE; eager launches a multiply
//      kernel and then an add kernel, rounding TWICE.
//   2. The accumulation order is the reference's statement order (aC, west,
//      east, south, north, below, above).  Floating-point addition is not
//      associative, so the order IS the result.
//   3. The mask multiplies the FULLY ACCUMULATED sum at the very end, never
//      folded into phi: ``(aC*phi + ...)*m != aC*(phi*m) + ...`` in fp64.
//   And: no kernel divides.  ``cg_beta`` and ``alpha`` stay on the host as the
//   eager 0-d tensor divisions and arrive as device pointers read as
//   ``beta[0]``, so the host never synchronises to fetch their value.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/Dispatch.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "../_cuda_launch/launch_config.h"

namespace {

defgcm::cuda_launch::LaunchConfigCache launch_config_cache;

//: None of these kernels allocates dynamic shared memory; the occupancy API
//: still wants the callback.
size_t dynamic_shared_bytes_for_block(int) {
  return 0;
}

constexpr int kReferenceBlockSize = 256;
constexpr int kWarpSize = 32;

int block_count_for(int64_t count, int threads) {
  const int64_t blocks = (count + threads - 1) / threads;
  return static_cast<int>(blocks);
}

// ---------------------------------------------------------------------------
// The dot-product fold.
//
// Three of the four kernels below have, in registers, exactly the operand pair
// the eager loop then re-reads from memory to form an inner product:
//   * the Thomas apply holds ``q`` as it stores it, and ``eta = <q, r>``;
//   * the seven-point apply holds ``Ap`` as it stores it and the centre value
//     of the padded block IS ``p``, so ``<p, Ap>`` needs no load at all;
//   * the x/r update holds ``r_new`` as it stores it, and ``<r_new, r_new>``.
// Folding the reduction into those kernels removes, per inner product, the
// product kernel's two loads and one store plus the reduction kernel's load.
//
// THIS CHANGES THE SUMMATION ORDER and is therefore a class-B (algorithmically
// equivalent, not bitwise) change.  Everything else in this file stays class-A.
//
// The partial sums are DETERMINISTIC, and that is a requirement, not a
// nicety: the graph replays the same iteration and the caller compares two
// solves bitwise.  Each warp reduces its 32 lanes through a fixed butterfly,
// the per-warp values go to a fixed shared slot, and warp 0 reduces those
// through the same butterfly -- one double per block, at a block-independent
// index.  Nothing depends on scheduling order, so the same input always
// produces the same partial array and the same host-side ``torch.sum``.
//
// Every thread of the block must reach this function: ``__shfl_down_sync``
// with a full mask and ``__syncthreads`` are both block-uniform operations.
// The kernels below therefore have NO early return -- a thread with no work
// arrives with a contribution of zero.  Full-warp participation also requires
// a block size that is a whole number of warps, which the launchers check.
// ---------------------------------------------------------------------------
constexpr int kMaxWarpsPerBlock = 32;

// The interior-to-padded index map, in ONE place.  ``i`` numbers the interior
// points of ``[nf, nz, ny, nx]`` in row-major order; the result indexes the
// same cell of ``[nf, nz, ny+2h, nx+2h]``.  Padding touches only j and i, and
// the face and level strides are whole multiples of the padded plane, so
// flattening (face, level) into one index is correct for any ``nf`` and
// ``nz``. All kernels share this index map.
__device__ __forceinline__ int64_t padded_index(
    int64_t i, int ny, int nx, int halo) {
  const int ii = static_cast<int>(i % nx);
  int64_t tmp = i / nx;
  const int jj = static_cast<int>(tmp % ny);
  const int64_t plane_index = tmp / ny;  // (face, level) flattened
  const int64_t padded_plane =
      static_cast<int64_t>(ny + 2 * halo) * (nx + 2 * halo);
  return plane_index * padded_plane +
      static_cast<int64_t>(jj + halo) * (nx + 2 * halo) + (ii + halo);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename scalar_t>
__device__ __forceinline__ void emit_partial(
    scalar_t contribution, scalar_t* __restrict__ partials) {
  __shared__ scalar_t staged[kMaxWarpsPerBlock];
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int warps_per_block = static_cast<int>(blockDim.x) / kWarpSize;
  const scalar_t total = warp_reduce_sum(contribution);
  if (lane == 0) {
    staged[warp] = total;
  }
  __syncthreads();
  if (warp == 0) {
    const scalar_t value =
        lane < warps_per_block ? staged[lane] : static_cast<scalar_t>(0);
    const scalar_t block_total = warp_reduce_sum(value);
    if (lane == 0) {
      partials[blockIdx.x] = block_total;
    }
  }
}

// ---------------------------------------------------------------------------
// K1 -- the seven-point apply from a padded block, mask on the store.
//
// Reference (cg3d_operator.py:341-345, :357-359, then cg3d.py's ``* mask4``):
//   out  = aC*inner
//   out += aW[..., :nx]*west ; out += aW[..., 1:nx+1]*east
//   out += aS[:ny, ...]*south ; out += aS[1:ny+1, ...]*north
//   out[:, 1:]  += aV[1:]*phi[:, :-1]        (level k>=1   reads aV[k])
//   out[:, :-1] += aV[1:]*phi[:, 1:]         (level k<=nz-2 reads aV[k+1])
//   out *= mask
//
// The vertical terms read the INTERIOR of the padded block, which is bitwise
// the unhaloed field, so one input tensor serves both directions.
//
// ``coefficient_face_stride`` is 0 when the coefficients broadcast over ``F``
// (today's assembly) and the full per-face extent when they are face-resolved
// (what a cube assembly would produce).  One integer, so the kernel needs no
// change the day such an assembly exists.
//
// ONE THREAD PER HORIZONTAL COLUMN, marching in k.  The previous
// mapping was one thread per point, and every point re-read from memory the
// three centre-column values (its own, the level below, the level above) and
// two ``aV`` entries a plane apart.  Marching a column in a register window
// reads each of those once: ``below``/``centre``/``above`` rotate, and
// ``aV[k]`` is the "above" coefficient of level k-1 and the "below"
// coefficient of level k.  The horizontal reads stay where they were -- ``ii``
// remains the fastest-varying thread index, so a warp still walks a contiguous
// run of ``x`` and every load is coalesced.
//
// This is a class-A (bit-neutral) change: the seven terms are accumulated in
// the same order, from the same values, with the mask still multiplying the
// completed sum.  Only WHERE a value comes from changes -- a register instead
// of a second load of the same address -- which fp64 cannot observe.
// ---------------------------------------------------------------------------
template <typename scalar_t, bool kFoldDot>
__global__ void cg3d_spmv_halo_kernel_impl(
    const scalar_t* __restrict__ phi_halo,
    const scalar_t* __restrict__ aW,
    const scalar_t* __restrict__ aS,
    const scalar_t* __restrict__ aV,
    const scalar_t* __restrict__ aC,
    const uint8_t* __restrict__ mask,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ partials,
    int nf,
    int nz,
    int ny,
    int nx,
    int64_t face_stride_c,
    int64_t face_stride_w,
    int64_t face_stride_s) {
  const int64_t columns = static_cast<int64_t>(nf) * ny * nx;
  const int64_t t =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // No early return: every thread of the block must reach ``emit_partial``,
  // whose warp shuffle needs the full mask.  Out-of-range threads contribute 0.
  scalar_t local = static_cast<scalar_t>(0);
  if (t < columns) {
    const int ii = static_cast<int>(t % nx);
    const int64_t tmp = t / nx;
    const int jj = static_cast<int>(tmp % ny);
    const int ff = static_cast<int>(tmp / ny);

    // Padded block: [nf, nz, ny+2, nx+2].
    const int64_t hplane = static_cast<int64_t>(ny + 2) * (nx + 2);
    // Coefficients: aC/aV [.., nz, ny, nx], aW [.., nz, ny, nx+1],
    //               aS [.., nz, ny+1, nx].
    const int64_t cplane = static_cast<int64_t>(ny) * nx;
    const int64_t wplane = static_cast<int64_t>(ny) * (nx + 1);
    const int64_t splane = static_cast<int64_t>(ny + 1) * nx;

    int64_t hbase = static_cast<int64_t>(ff) * nz * hplane +
        static_cast<int64_t>(jj + 1) * (nx + 2) + (ii + 1);
    int64_t ci = static_cast<int64_t>(ff) * face_stride_c +
        static_cast<int64_t>(jj) * nx + ii;
    int64_t wi = static_cast<int64_t>(ff) * face_stride_w +
        static_cast<int64_t>(jj) * (nx + 1) + ii;
    int64_t si = static_cast<int64_t>(ff) * face_stride_s +
        static_cast<int64_t>(jj) * nx + ii;
    int64_t oi = static_cast<int64_t>(ff) * nz * cplane +
        static_cast<int64_t>(jj) * nx + ii;

    scalar_t below = static_cast<scalar_t>(0);
    scalar_t centre = phi_halo[hbase];
    scalar_t above = static_cast<scalar_t>(0);
    scalar_t below_coefficient = static_cast<scalar_t>(0);
    scalar_t above_coefficient = static_cast<scalar_t>(0);
    if (nz > 1) {
      above = phi_halo[hbase + hplane];
      above_coefficient = aV[ci + cplane];
    }

    for (int kk = 0; kk < nz; ++kk) {
      scalar_t acc = aC[ci] * centre;
      acc = acc + aW[wi] * phi_halo[hbase - 1];
      acc = acc + aW[wi + 1] * phi_halo[hbase + 1];
      acc = acc + aS[si] * phi_halo[hbase - (nx + 2)];
      acc = acc + aS[si + nx] * phi_halo[hbase + (nx + 2)];
      if (kk >= 1) {
        acc = acc + below_coefficient * below;
      }
      if (kk <= nz - 2) {
        acc = acc + above_coefficient * above;
      }
      const scalar_t value = acc * static_cast<scalar_t>(mask[oi]);
      out[oi] = value;
      if constexpr (kFoldDot) {
        // ``p`` is the CENTRE of the padded block, bitwise the unhaloed search
        // direction, so <p, Ap> costs no load at all.
        local += centre * value;
      }

      below = centre;
      centre = above;
      below_coefficient = above_coefficient;
      hbase += hplane;
      ci += cplane;
      wi += wplane;
      si += splane;
      oi += cplane;
      if (kk + 2 <= nz - 1) {
        above = phi_halo[hbase + hplane];
        above_coefficient = aV[ci + cplane];
      }
    }
  }
  if constexpr (kFoldDot) {
    emit_partial(local, partials);
  }
}

// ---------------------------------------------------------------------------
// K2 -- s_out = (q + beta*s) * mask      [cg3d.py:256]
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void cg3d_axpy_s_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ s,
    const uint8_t* __restrict__ mask,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ out,
    int64_t n) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const scalar_t b = beta[0];
  out[i] = (q[i] + b * s[i]) * static_cast<scalar_t>(mask[i]);
}

// ---------------------------------------------------------------------------
// K2' -- the same expression, reading and writing the INTERIOR of a padded
// buffer whose border cells the caller keeps up to date.
//
// The search direction is both the output of this kernel and the input to the
// next iteration's SPMV, which needs it haloed.  Writing it into the middle of
// a resident padded allocation removes, per iteration, the exchange's two
// ``torch.cat`` calls and the ``s.copy_(direction)`` that carried the new
// direction into the graph's own buffer -- the halo cells are then refreshed
// in place (``geometry/padded_field.py``) instead of a new padded tensor being
// built.
//
// Bit-neutral against ``cg3d_axpy_s_kernel``: same operands, same order, mask
// last.  Only the store address changes -- ``i`` indexes the interior, and
// ``padded_offset`` maps it to the same cell of the larger buffer.
// ---------------------------------------------------------------------------
// ``s`` IS the padded buffer: the previous direction lives in the same cells
// this kernel is about to overwrite, so the operand and the destination share
// one index, avoiding a separate full-field transfer.
template <typename scalar_t>
__global__ void cg3d_axpy_s_pad_kernel(
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ mask,
    const scalar_t* __restrict__ beta,
    scalar_t* padded,
    int nz,
    int ny,
    int nx,
    int halo,
    int64_t n) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t index = padded_index(i, ny, nx, halo);
  const scalar_t b = beta[0];
  padded[index] =
      (q[i] + b * padded[index]) * static_cast<scalar_t>(mask[i]);
}

// ---------------------------------------------------------------------------
// K3 -- x_out = (x + alpha*s) * mask ; r_out = (r - alpha*q) * mask
//       [cg3d.py:265-266]
// ---------------------------------------------------------------------------
template <typename scalar_t, bool kFoldDot>
__global__ void cg3d_update_xr_kernel_impl(
    const scalar_t* x,
    const scalar_t* __restrict__ s,
    const scalar_t* r,
    const scalar_t* __restrict__ qmat,
    const uint8_t* __restrict__ mask,
    const scalar_t* __restrict__ alpha,
    scalar_t* x_out,
    scalar_t* r_out,
    scalar_t* __restrict__ partials,
    const bool* valid,
    int64_t n,
    // ``s`` may be the INTERIOR of a padded buffer, which is not contiguous.
    // These describe its stride so the kernel can index it; ``s_halo == 0``
    // means ``s`` is a plain dense field and the mapping is the identity.
    int s_ny,
    int s_nx,
    int s_halo) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // Graph-owned outputs may alias x/r. A breakdown preserves them without
  // evaluating the rejected update or transferring another full field.  A
  // rejected iteration therefore contributes ZERO to the folded ``<r, r>``,
  // which is what the caller's ``where(valid, ...)`` then discards.
  const bool active = i < n && (valid == nullptr || valid[0]);
  scalar_t local = static_cast<scalar_t>(0);
  if (active) {
    const scalar_t a = alpha[0];
    const scalar_t m = static_cast<scalar_t>(mask[i]);
    const int64_t si =
        s_halo == 0 ? i : padded_index(i, s_ny, s_nx, s_halo);
    x_out[i] = (x[i] + a * s[si]) * m;
    const scalar_t residual = (r[i] - a * qmat[i]) * m;
    r_out[i] = residual;
    if constexpr (kFoldDot) {
      local = residual * residual;
    }
  }
  if constexpr (kFoldDot) {
    // Not under ``active``: the shuffle needs the whole warp (see the fold
    // comment above), so the early return the guarded version could take is
    // replaced by a zero contribution.
    emit_partial(local, partials);
  }
}

// ---------------------------------------------------------------------------
// K4 -- the vertical-line Thomas apply with the mask folded onto the store.
//
// Reference (cg3d_preconditioner.py:120-127, then cg3d.py:238's ``* mask4``):
//   q[0]  = zMC[0]*r[0]
//   q[k]  = zMC[k]*(r[k] - zML[k]*q[k-1])          k = 1 .. nz-1
//   q[k]  = q[k] - zMU[k]*q[k+1]                   k = nz-2 .. 0
//   q    *= mask
//
// One thread per horizontal column, sequential sweeps over its own column, no
// shared memory and therefore no ``nz`` ceiling.  (The separate
// ``_cg3d_thomas`` kernel, which serves the EAGER preconditioner under its own
// registry key, stages the column in shared memory and is untouched.)
//
// TWO SWEEPS, NOT THREE.  A mask sweep would be needed because the
// backward pass leaves level nz-1 holding its forward value, so a per-level
// mask applied during the backward pass would multiply the wrong thing at that
// one level.  Seeding the backward carry with the forward value of level nz-1
// and storing ``value * mask`` fixes exactly that: level nz-1 is stored from
// the carry before the loop starts, and every level below is stored once the
// ``- zMU*next`` correction has been applied to it.  The neighbour values the
// two recurrences need (``q[k-1]`` forward, ``q[k+1]`` backward) come from a
// register instead of a reload of the address just written.
//
// Bit-neutral (class A): the mask still multiplies a FULLY FORMED value, the
// last operation before the store, exactly as the deleted third sweep did.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// K4' -- the same recurrence with the column staged in SHARED memory.
//
// The two-sweep kernel below writes each level of ``q`` to global memory in
// the forward pass and reads it back in the backward pass: nz stores plus nz
// loads of a value that was just computed.  Staging the column in shared
// memory keeps it on chip, so global traffic falls to ``r`` in and ``q`` out.
//
// **The arithmetic is identical, statement for statement**, to
// ``cg3d_thomas_mask_kernel_impl``: same three expressions, same order, mask
// multiplying the fully formed value at the same point.  Only where the
// intermediate lives changes, so the STORED FIELD is bit-identical (class A).
// The FOLDED DOT is not: one block now covers 32 columns
// rather than ``blockDim.x`` points, so the partial sums are grouped
// differently and their total differs in the last places.  That is class B
// (algorithmically equivalent, not bitwise), so this kernel
// is not simply a drop-in replacement.
//
// ``kColumnsPerBlock`` columns per block, ``nz`` scalars each -- ONE staged
// array, the forward pass's ``q``, because that is the only value the backward
// pass needs and cannot recompute.  ``zMC``/``zML``/``zMU``/``r``/``mask``
// stream from global memory in both kernels alike.  At nz=90 fp64 that is
// 22.5 KB per block, under the 48 KB default, so no ``cudaFuncSetAttribute``
// opt-in is needed on any device that meets the CUDA minimum; the launcher
// asks anyway, because a deeper column would need it.
//
// (That figure is per BLOCK.  The budget that matters for occupancy is per SM,
// and it is the reason ``cg3d_thomas_use_shared`` below is not simply "yes":
// 720 B per column is a hard cap on how many columns an SM can host at once,
// whatever the block shape.  See its comment for the measurement.)
//
// ONE predicate decides whether this kernel runs (``cg3d_thomas_use_shared``),
// read by the block count, the launcher and the reported launch plan, so what
// is reported is what runs.
// ---------------------------------------------------------------------------
constexpr int kColumnsPerBlock = 32;

template <typename scalar_t, bool kFoldDot>
__global__ void cg3d_thomas_mask_shared_kernel(
    const scalar_t* __restrict__ zMC,
    const scalar_t* __restrict__ zML,
    const scalar_t* __restrict__ zMU,
    const scalar_t* __restrict__ r,
    const uint8_t* __restrict__ mask,
    scalar_t* __restrict__ q,
    scalar_t* __restrict__ partials,
    int nf,
    int nz,
    int64_t plane,
    int64_t face_stride_c) {
  const int64_t columns = static_cast<int64_t>(nf) * plane;
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t t =
      static_cast<int64_t>(blockIdx.x) * kColumnsPerBlock + lane;
  extern __shared__ char shared_raw[];
  scalar_t* column = reinterpret_cast<scalar_t*>(shared_raw);

  scalar_t local = static_cast<scalar_t>(0);
  if (t < columns) {
    const int64_t cell = t % plane;
    const int64_t ff = t / plane;
    const int64_t base = ff * nz * plane + cell;
    const int64_t cbase = ff * face_stride_c + cell;

    // Forward sweep, staged: the value each level needs from the level above
    // is the one just written to shared memory, not a global reload.
    scalar_t previous = zMC[cbase] * r[base];
    column[lane] = previous;
    for (int k = 1; k < nz; ++k) {
      const int64_t idx = base + static_cast<int64_t>(k) * plane;
      const int64_t cidx = cbase + static_cast<int64_t>(k) * plane;
      previous = zMC[cidx] * (r[idx] - zML[cidx] * previous);
      column[k * kColumnsPerBlock + lane] = previous;
    }
    // Backward sweep, storing ``value * mask`` exactly where the two-sweep
    // kernel stores it.  ``previous`` still carries level nz-1's forward
    // value, which is what that level's store needs.
    {
      const int64_t idx = base + static_cast<int64_t>(nz - 1) * plane;
      const scalar_t stored = previous * static_cast<scalar_t>(mask[idx]);
      q[idx] = stored;
      if constexpr (kFoldDot) {
        local += stored * r[idx];
      }
    }
    for (int k = nz - 2; k >= 0; --k) {
      const int64_t idx = base + static_cast<int64_t>(k) * plane;
      const int64_t cidx = cbase + static_cast<int64_t>(k) * plane;
      previous = column[k * kColumnsPerBlock + lane] - zMU[cidx] * previous;
      const scalar_t stored = previous * static_cast<scalar_t>(mask[idx]);
      q[idx] = stored;
      if constexpr (kFoldDot) {
        local += stored * r[idx];
      }
    }
  }
  if constexpr (kFoldDot) {
    emit_partial(local, partials);
  }
}

template <typename scalar_t, bool kFoldDot>
__global__ void cg3d_thomas_mask_kernel_impl(
    const scalar_t* __restrict__ zMC,
    const scalar_t* __restrict__ zML,
    const scalar_t* __restrict__ zMU,
    const scalar_t* __restrict__ r,
    const uint8_t* __restrict__ mask,
    scalar_t* __restrict__ q,
    scalar_t* __restrict__ partials,
    int nf,
    int nz,
    int64_t plane,
    int64_t face_stride_c) {
  const int64_t columns = static_cast<int64_t>(nf) * plane;
  const int64_t t =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // No early return: ``emit_partial`` needs every lane of the warp.
  scalar_t local = static_cast<scalar_t>(0);
  if (t < columns) {
    const int64_t cell = t % plane;
    const int64_t ff = t / plane;
    const int64_t base = ff * nz * plane + cell;
    const int64_t cbase = ff * face_stride_c + cell;

    scalar_t previous = zMC[cbase] * r[base];
    for (int k = 1; k < nz; ++k) {
      const int64_t idx = base + static_cast<int64_t>(k) * plane;
      const int64_t cidx = cbase + static_cast<int64_t>(k) * plane;
      // Store level k-1, whose forward value is now final and which the
      // backward pass will read back.  Level nz-1 is never stored unmasked:
      // it leaves the loop in ``previous`` and is stored below.
      q[idx - plane] = previous;
      previous = zMC[cidx] * (r[idx] - zML[cidx] * previous);
    }
    {
      const int64_t idx = base + static_cast<int64_t>(nz - 1) * plane;
      const scalar_t stored = previous * static_cast<scalar_t>(mask[idx]);
      q[idx] = stored;
      if constexpr (kFoldDot) {
        local += stored * r[idx];
      }
    }
    for (int k = nz - 2; k >= 0; --k) {
      const int64_t idx = base + static_cast<int64_t>(k) * plane;
      const int64_t cidx = cbase + static_cast<int64_t>(k) * plane;
      previous = q[idx] - zMU[cidx] * previous;
      const scalar_t stored = previous * static_cast<scalar_t>(mask[idx]);
      q[idx] = stored;
      if constexpr (kFoldDot) {
        local += stored * r[idx];
      }
    }
  }
  if constexpr (kFoldDot) {
    emit_partial(local, partials);
  }
}

}  // namespace

namespace {

// The partial-sum buffer is sized by the caller from this, so the count the
// launcher will use and the count the allocation assumes are one expression.
int64_t block_count_checked(int64_t count, int threads) {
  TORCH_CHECK(threads % kWarpSize == 0 && threads / kWarpSize <= kMaxWarpsPerBlock,
              "CG3D fused reduction needs a whole number of warps per block, "
              "at most ", kMaxWarpsPerBlock, "; got ", threads);
  return (count + threads - 1) / threads;
}

void check_partials(const torch::Tensor& partials, const torch::Tensor& reference,
                    int64_t blocks) {
  TORCH_CHECK(partials.defined() && partials.is_cuda() &&
                  partials.is_contiguous() &&
                  partials.scalar_type() == reference.scalar_type() &&
                  partials.device() == reference.device() &&
                  partials.numel() == blocks,
              "CG3D fused partial-sum buffer must be a contiguous device "
              "tensor of ", blocks, " elements in the state dtype");
}

}  // namespace

int64_t cg3d_spmv_halo_block_count(
    const torch::Tensor& phi_halo, int nf, int ny, int nx) {
  c10::cuda::CUDAGuard device_guard(phi_halo.device());
  int64_t blocks = 0;
  AT_DISPATCH_FLOATING_TYPES(phi_halo.scalar_type(), "cg3d_spmv_halo_blocks", [&] {
    const auto& plan = launch_config_cache.get(
        cg3d_spmv_halo_kernel_impl<scalar_t, true>,
        phi_halo.get_device(), 0, kReferenceBlockSize,
        dynamic_shared_bytes_for_block);
    blocks = block_count_checked(static_cast<int64_t>(nf) * ny * nx,
                                 plan.selected_block_size);
  });
  return blocks;
}

void cg3d_spmv_halo_launch(
    const torch::Tensor& phi_halo,
    const torch::Tensor& aW,
    const torch::Tensor& aS,
    const torch::Tensor& aV,
    const torch::Tensor& aC,
    const torch::Tensor& mask,
    torch::Tensor& out,
    int nf,
    int nz,
    int ny,
    int nx,
    int64_t face_stride_c,
    int64_t face_stride_w,
    int64_t face_stride_s,
    const torch::Tensor& partials) {
  c10::cuda::CUDAGuard device_guard(phi_halo.device());
  const int64_t columns = static_cast<int64_t>(nf) * ny * nx;
  const bool fold = partials.defined();
  AT_DISPATCH_FLOATING_TYPES(phi_halo.scalar_type(), "cg3d_spmv_halo_launch", [&] {
    auto dispatch = [&](auto fold_tag) {
      constexpr bool kFold = decltype(fold_tag)::value;
      const auto& plan = launch_config_cache.get(
          cg3d_spmv_halo_kernel_impl<scalar_t, kFold>,
          phi_halo.get_device(),
          0,
          kReferenceBlockSize,
          dynamic_shared_bytes_for_block);
      const int threads = plan.selected_block_size;
      const int64_t blocks = block_count_checked(columns, threads);
      if (kFold) {
        check_partials(partials, out, blocks);
      }
      cg3d_spmv_halo_kernel_impl<scalar_t, kFold>
          <<<dim3(static_cast<int>(blocks)), dim3(threads), 0,
             at::cuda::getCurrentCUDAStream()>>>(
              phi_halo.data_ptr<scalar_t>(),
              aW.data_ptr<scalar_t>(),
              aS.data_ptr<scalar_t>(),
              aV.data_ptr<scalar_t>(),
              aC.data_ptr<scalar_t>(),
              mask.data_ptr<uint8_t>(),
              out.data_ptr<scalar_t>(),
              kFold ? partials.data_ptr<scalar_t>() : nullptr,
              nf,
              nz,
              ny,
              nx,
              face_stride_c,
              face_stride_w,
              face_stride_s);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (fold) {
      dispatch(std::true_type{});
    } else {
      dispatch(std::false_type{});
    }
  });
}

void cg3d_axpy_s_launch(
    const torch::Tensor& q,
    const torch::Tensor& s,
    const torch::Tensor& mask,
    const torch::Tensor& beta,
    torch::Tensor& out) {
  c10::cuda::CUDAGuard device_guard(q.device());
  const int64_t count = q.numel();
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "cg3d_axpy_s_launch", [&] {
    const auto& plan = launch_config_cache.get(
        cg3d_axpy_s_kernel<scalar_t>,
        q.get_device(),
        0,
        kReferenceBlockSize,
        dynamic_shared_bytes_for_block);
    const int threads = plan.selected_block_size;
    cg3d_axpy_s_kernel<scalar_t>
        <<<dim3(block_count_for(count, threads)), dim3(threads), 0,
           at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<scalar_t>(),
            s.data_ptr<scalar_t>(),
            mask.data_ptr<uint8_t>(),
            beta.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void cg3d_axpy_s_pad_launch(
    const torch::Tensor& q,
    const torch::Tensor& mask,
    const torch::Tensor& beta,
    torch::Tensor& padded,
    int nz,
    int ny,
    int nx,
    int halo) {
  c10::cuda::CUDAGuard device_guard(q.device());
  const int64_t count = q.numel();
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "cg3d_axpy_s_pad_launch", [&] {
    const auto& plan = launch_config_cache.get(
        cg3d_axpy_s_pad_kernel<scalar_t>,
        q.get_device(),
        0,
        kReferenceBlockSize,
        dynamic_shared_bytes_for_block);
    const int threads = plan.selected_block_size;
    cg3d_axpy_s_pad_kernel<scalar_t>
        <<<dim3(block_count_for(count, threads)), dim3(threads), 0,
           at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<scalar_t>(),
            mask.data_ptr<uint8_t>(),
            beta.data_ptr<scalar_t>(),
            padded.data_ptr<scalar_t>(),
            nz,
            ny,
            nx,
            halo,
            count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

int64_t cg3d_update_xr_block_count(const torch::Tensor& x) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t blocks = 0;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cg3d_update_xr_blocks", [&] {
    const auto& plan = launch_config_cache.get(
        cg3d_update_xr_kernel_impl<scalar_t, true>,
        x.get_device(), 0, kReferenceBlockSize, dynamic_shared_bytes_for_block);
    blocks = block_count_checked(x.numel(), plan.selected_block_size);
  });
  return blocks;
}

void cg3d_update_xr_launch(
    const torch::Tensor& x,
    const torch::Tensor& s,
    const torch::Tensor& r,
    const torch::Tensor& qmat,
    const torch::Tensor& mask,
    const torch::Tensor& alpha,
    torch::Tensor& x_out,
    torch::Tensor& r_out,
    const torch::Tensor& valid,
    const torch::Tensor& partials,
    int s_halo) {
  c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t count = x.numel();
  const bool fold = partials.defined();
  // s may be the interior of a padded buffer; the kernel maps its index
  // when s_halo != 0.  x is [nf, nz, ny, nx], so those are the extents.
  const int s_ny = static_cast<int>(x.size(x.dim() - 2));
  const int s_nx = static_cast<int>(x.size(x.dim() - 1));
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cg3d_update_xr_launch", [&] {
    auto dispatch = [&](auto fold_tag) {
      constexpr bool kFold = decltype(fold_tag)::value;
      const auto& plan = launch_config_cache.get(
          cg3d_update_xr_kernel_impl<scalar_t, kFold>,
          x.get_device(),
          0,
          kReferenceBlockSize,
          dynamic_shared_bytes_for_block);
      const int threads = plan.selected_block_size;
      const int64_t blocks = block_count_checked(count, threads);
      if (kFold) {
        check_partials(partials, x, blocks);
      }
      cg3d_update_xr_kernel_impl<scalar_t, kFold>
          <<<dim3(static_cast<int>(blocks)), dim3(threads), 0,
             at::cuda::getCurrentCUDAStream()>>>(
              x.data_ptr<scalar_t>(),
              s.data_ptr<scalar_t>(),
              r.data_ptr<scalar_t>(),
              qmat.data_ptr<scalar_t>(),
              mask.data_ptr<uint8_t>(),
              alpha.data_ptr<scalar_t>(),
              x_out.data_ptr<scalar_t>(),
              r_out.data_ptr<scalar_t>(),
              kFold ? partials.data_ptr<scalar_t>() : nullptr,
              valid.defined() ? valid.data_ptr<bool>() : nullptr,
              count,
              s_ny,
              s_nx,
              s_halo);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (fold) {
      dispatch(std::true_type{});
    } else {
      dispatch(std::false_type{});
    }
  });
}

namespace {

//: Shared memory one block of the column-resident Thomas kernel needs.
int64_t cg3d_thomas_shared_bytes(int nz, int64_t element_size) {
  return static_cast<int64_t>(kColumnsPerBlock) * nz * element_size;
}

//: Why the column-resident Thomas kernel is not used.  Empty would mean "it
//: is"; the string is what the ledger and the launch report say instead.
constexpr const char* kThomasSharedDeclineReason =
    "occupancy-bound: 9.30% (7 warps/SM, pinned by shared memory) against the "
    "two-sweep kernel's 49.97%, both stalled on dependent global loads "
    "(long_scoreboard 94% vs 97% of stalled cycles); 3.207 vs 2.272 ms at "
    "800x800x90 fp64";

//: Whether the device COULD host the staged column -- the capability question
//: on its own, which the tests that compare the two implementations need in
//: order to tell "declined" apart from "impossible here".
template <typename scalar_t>
bool cg3d_thomas_shared_is_launchable(int device, int nz, bool fold) {
  const int64_t bytes = cg3d_thomas_shared_bytes(nz, sizeof(scalar_t));
  if (bytes > std::numeric_limits<int>::max()) {
    return false;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return false;
  }
  const int64_t cap = std::max<int64_t>(properties.sharedMemPerBlock,
                                        properties.sharedMemPerBlockOptin);
  if (bytes > cap) {
    return false;
  }
  auto opt_in = [&](auto kernel) {
    return cudaFuncSetAttribute(
               kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
               static_cast<int>(bytes)) == cudaSuccess;
  };
  return fold ? opt_in(cg3d_thomas_mask_shared_kernel<scalar_t, true>)
              : opt_in(cg3d_thomas_mask_shared_kernel<scalar_t, false>);
}

// ONE predicate, read by the block count, the launcher and the reported launch
// plan.  Two copies of this decision would let the partial-sum buffer be sized
// for one kernel and filled by the other -- which ``check_partials`` would
// catch, but only after the wrong kernel had already been chosen.
//
// **FALSE, and the MECHANISM was measured, not just the wall clock.**  The
// column-resident kernel does what it was built to do: 2.666 GB per launch
// against the two-sweep kernel's 3.633, a 26.6% cut.  It is nonetheless 41%
// SLOWER -- 3.207 vs 2.272 ms at 800x800x90 fp64 -- and these are the readings
// that explain it:
//
//   * OCCUPANCY IS PINNED BY SHARED MEMORY.  ``launch__occupancy_limit_shared_mem``
//     is 6, the smallest of the four limits (registers 36, warps 64, blocks
//     32), so shared memory is what binds -- and the measured
//     ``sm__warps_active`` is 9.30% against a 9.375% ceiling, i.e. the ceiling
//     IS the achieved value and nothing else is holding the kernel back.  The
//     two-sweep kernel runs at 49.97%, bound by registers instead.
//   * BOTH KERNELS FAIL THE SAME WAY; ONE HAS MORE WARPS TO HIDE IT.  Of the
//     cycles a warp cannot issue, ``long_scoreboard`` (waiting on a global
//     load) is 94% here and 97% there.  The recurrence is latency-bound in
//     both; the two-sweep kernel simply has 5.4x the resident warps to overlap
//     that waiting with.  It is NOT shared-memory synchronisation: barrier is
//     ~1% and membar does not appear.
//   * MORE LOADS PER THREAD DO NOT SUBSTITUTE FOR WARPS.  Prefetching 2 and 4
//     levels ahead cuts registers 56 -> 48 -> 40 and moves occupancy not at
//     all (9.30 / 9.31%), because registers were never the binding resource;
//     3.339 and 2.936 ms, still behind.
//   * THE BUDGET IS PER SM, SO BLOCK SHAPE CANNOT FIX IT.  90 levels x 8 B =
//     720 B per column against 167936 B per SM caps resident columns at ~224
//     however they are grouped: 32/64/128 columns per block give 224/192/128
//     resident columns, 9.30/9.31/6.24% occupancy, 3.207/3.176/4.157 ms.  That
//     negative control is what rules out "wrong block shape" as the cause.
//
// Bytes are the instrument; the target is time, and on time this loses.
//
// The kernel and its tests stay: the arithmetic is verified bit-neutral across
// 80 shape x dtype x variant checks.  Reviving it means changing the MAPPING
// so a column costs less resident shared memory -- a warp per column, or
// staging the column in registers in segments -- not a block-shape knob, which
// is the one thing measured above to be incapable of helping.
template <typename scalar_t>
bool cg3d_thomas_use_shared(int device, int nz, bool fold) {
  (void)device;
  (void)nz;
  (void)fold;
  return false;
}

}  // namespace

int64_t cg3d_thomas_mask_block_count(
    const torch::Tensor& r, int nf, int64_t plane) {
  c10::cuda::CUDAGuard device_guard(r.device());
  const int nz = static_cast<int>(r.size(1));
  int64_t blocks = 0;
  AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "cg3d_thomas_mask_blocks", [&] {
    const int64_t columns = static_cast<int64_t>(nf) * plane;
    if (cg3d_thomas_use_shared<scalar_t>(r.get_device(), nz, true)) {
      blocks = (columns + kColumnsPerBlock - 1) / kColumnsPerBlock;
      return;
    }
    const auto& plan = launch_config_cache.get(
        cg3d_thomas_mask_kernel_impl<scalar_t, true>,
        r.get_device(), 0, kReferenceBlockSize, dynamic_shared_bytes_for_block);
    blocks = block_count_checked(columns, plan.selected_block_size);
  });
  return blocks;
}

void cg3d_thomas_mask_launch(
    const torch::Tensor& zMC,
    const torch::Tensor& zML,
    const torch::Tensor& zMU,
    const torch::Tensor& r,
    const torch::Tensor& mask,
    torch::Tensor& q,
    int nf,
    int nz,
    int64_t plane,
    int64_t face_stride_c,
    const torch::Tensor& partials) {
  c10::cuda::CUDAGuard device_guard(r.device());
  const int64_t columns = static_cast<int64_t>(nf) * plane;
  const bool fold = partials.defined();
  AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "cg3d_thomas_mask_launch", [&] {
    auto dispatch = [&](auto fold_tag) {
      constexpr bool kFold = decltype(fold_tag)::value;
      if (cg3d_thomas_use_shared<scalar_t>(r.get_device(), nz, kFold)) {
        const int64_t blocks =
            (columns + kColumnsPerBlock - 1) / kColumnsPerBlock;
        if (kFold) {
          check_partials(partials, r, blocks);
        }
        const size_t shared = static_cast<size_t>(
            cg3d_thomas_shared_bytes(nz, sizeof(scalar_t)));
        cg3d_thomas_mask_shared_kernel<scalar_t, kFold>
            <<<dim3(static_cast<int>(blocks)), dim3(kColumnsPerBlock), shared,
               at::cuda::getCurrentCUDAStream()>>>(
                zMC.data_ptr<scalar_t>(),
                zML.data_ptr<scalar_t>(),
                zMU.data_ptr<scalar_t>(),
                r.data_ptr<scalar_t>(),
                mask.data_ptr<uint8_t>(),
                q.data_ptr<scalar_t>(),
                kFold ? partials.data_ptr<scalar_t>() : nullptr,
                nf,
                nz,
                plane,
                face_stride_c);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
      }
      const auto& plan = launch_config_cache.get(
          cg3d_thomas_mask_kernel_impl<scalar_t, kFold>,
          r.get_device(),
          0,
          kReferenceBlockSize,
          dynamic_shared_bytes_for_block);
      const int threads = plan.selected_block_size;
      const int64_t blocks = block_count_checked(columns, threads);
      if (kFold) {
        check_partials(partials, r, blocks);
      }
      cg3d_thomas_mask_kernel_impl<scalar_t, kFold>
          <<<dim3(static_cast<int>(blocks)), dim3(threads), 0,
             at::cuda::getCurrentCUDAStream()>>>(
              zMC.data_ptr<scalar_t>(),
              zML.data_ptr<scalar_t>(),
              zMU.data_ptr<scalar_t>(),
              r.data_ptr<scalar_t>(),
              mask.data_ptr<uint8_t>(),
              q.data_ptr<scalar_t>(),
              kFold ? partials.data_ptr<scalar_t>() : nullptr,
              nf,
              nz,
              plane,
              face_stride_c);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (fold) {
      dispatch(std::true_type{});
    } else {
      dispatch(std::false_type{});
    }
  });
}

// The launch decision each kernel actually uses, per device and dtype -- the
// same cache entry the launchers above read, so what is reported IS what runs
// (the configurable launch policy).
// Which Thomas kernel a given (device, nz, dtype) will actually launch.  It
// asks ``cg3d_thomas_use_shared`` -- the launcher's own predicate -- rather
// than restating the condition, so a test that reads this reads the decision
// that will be made, not a second opinion about it.
bool cg3d_thomas_prefers_shared(int device, int nz, at::ScalarType dtype,
                                bool fold) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  bool shared = false;
  AT_DISPATCH_FLOATING_TYPES(dtype, "cg3d_thomas_prefers_shared", [&] {
    shared = cg3d_thomas_use_shared<scalar_t>(device, nz, fold);
  });
  return shared;
}

// Whether this device COULD host the column-resident kernel, as opposed to
// whether it is chosen.  Only the tests that compare the two implementations
// ask; production reads ``cg3d_thomas_prefers_shared`` above.
bool cg3d_thomas_shared_launchable(int device, int nz, at::ScalarType dtype,
                                   bool fold) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  bool launchable = false;
  AT_DISPATCH_FLOATING_TYPES(dtype, "cg3d_thomas_shared_launchable", [&] {
    launchable = cg3d_thomas_shared_is_launchable<scalar_t>(device, nz, fold);
  });
  return launchable;
}

//: Why the column-resident kernel is declined, for the ledger and the report.
const char* cg3d_thomas_shared_decline_reason() {
  return kThomasSharedDeclineReason;
}

const defgcm::cuda_launch::LaunchConfig& cg3d_fused_launch_config(
    int device, const std::string& kernel, at::ScalarType dtype) {
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
  // The dot-folding instantiation is a DIFFERENT kernel image with its own
  // register count, hence its own occupancy answer.  Each variant is reported
  // under its own name so what is reported is still what runs.
#define DEFGCM_CG3D_PLAN(KERNEL)                                          \
  AT_DISPATCH_FLOATING_TYPES(dtype, "cg3d_fused_launch_config", [&] { \
    selected = &launch_config_cache.get(                              \
        KERNEL, device, 0, kReferenceBlockSize,                       \
        dynamic_shared_bytes_for_block);                              \
  });
  const defgcm::cuda_launch::LaunchConfig* selected = nullptr;
  if (kernel == "spmv_halo") {
    DEFGCM_CG3D_PLAN((cg3d_spmv_halo_kernel_impl<scalar_t, false>))
  } else if (kernel == "axpy_s") {
    DEFGCM_CG3D_PLAN(cg3d_axpy_s_kernel<scalar_t>)
  } else if (kernel == "axpy_s_pad") {
    DEFGCM_CG3D_PLAN(cg3d_axpy_s_pad_kernel<scalar_t>)
  } else if (kernel == "update_xr") {
    DEFGCM_CG3D_PLAN((cg3d_update_xr_kernel_impl<scalar_t, false>))
  } else if (kernel == "thomas_mask") {
    DEFGCM_CG3D_PLAN((cg3d_thomas_mask_kernel_impl<scalar_t, false>))
  } else if (kernel == "spmv_halo_dot") {
    DEFGCM_CG3D_PLAN((cg3d_spmv_halo_kernel_impl<scalar_t, true>))
  } else if (kernel == "update_xr_dot") {
    DEFGCM_CG3D_PLAN((cg3d_update_xr_kernel_impl<scalar_t, true>))
  } else if (kernel == "thomas_mask_dot") {
    DEFGCM_CG3D_PLAN((cg3d_thomas_mask_kernel_impl<scalar_t, true>))
  } else {
    TORCH_CHECK(false, "unknown CG3D fused kernel name: ", kernel);
  }
#undef DEFGCM_CG3D_PLAN
  TORCH_CHECK(selected != nullptr, "CG3D fused launch configuration missing");
  return *selected;
}
