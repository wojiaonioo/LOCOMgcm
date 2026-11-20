"""Triton implementation of reusable 2 by 2 block column factors.

Each lane owns one physical column. Coefficients and work arrays keep that
column axis contiguous; the vertical recurrence is runtime-sized. RHS lanes
share each stored matrix factor. Launches explicitly preserve separate
multiply/add rounding, matching the compiled implicit residual's policy.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _inverse2(a, b, c, d):
    scale = tl.maximum(tl.maximum(tl.abs(a), tl.abs(b)), tl.maximum(tl.abs(c), tl.abs(d)))
    finite_scale = (scale < float("inf")) & (scale > 0.)
    safe_scale = tl.where(finite_scale, scale, 1.)
    aa, bb, cc, dd = a / safe_scale, b / safe_scale, c / safe_scale, d / safe_scale
    determinant = aa * dd - bb * cc
    valid = finite_scale & (tl.abs(determinant) < float("inf")) & (determinant != 0.)
    safe_det = tl.where(valid, determinant, 1.)
    i00, i01 = dd / safe_det / safe_scale, -bb / safe_det / safe_scale
    i10, i11 = -cc / safe_det / safe_scale, aa / safe_det / safe_scale
    valid = valid & (tl.abs(i00) < float("inf")) & (tl.abs(i01) < float("inf"))
    valid = valid & (tl.abs(i10) < float("inf")) & (tl.abs(i11) < float("inf"))
    return (tl.where(valid, i00, 1.), tl.where(valid, i01, 0.),
            tl.where(valid, i10, 0.), tl.where(valid, i11, 1.), valid)


@triton.jit
def _factor_kernel(A, B, C, INVERSE, MULTIPLIER, INFO,
                   NC: tl.constexpr, NZ: tl.constexpr, BLOCK: tl.constexpr):
    column = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = column < NC
    b00 = tl.load(B + column, mask, other=1.)
    b01 = tl.load(B + NC + column, mask, other=0.)
    b10 = tl.load(B + 2 * NC + column, mask, other=0.)
    b11 = tl.load(B + 3 * NC + column, mask, other=1.)
    i00, i01, i10, i11, valid = _inverse2(b00, b01, b10, b11)
    status = tl.where(valid, 0, 1).to(tl.int32)
    tl.store(INVERSE + column, i00, mask)
    tl.store(INVERSE + NC + column, i01, mask)
    tl.store(INVERSE + 2 * NC + column, i10, mask)
    tl.store(INVERSE + 3 * NC + column, i11, mask)
    tl.store(MULTIPLIER + column, 0., mask)
    tl.store(MULTIPLIER + NC + column, 0., mask)
    tl.store(MULTIPLIER + 2 * NC + column, 0., mask)
    tl.store(MULTIPLIER + 3 * NC + column, 0., mask)
    for level in range(1, NZ):
        index = level * 4 * NC + column
        prev = (level - 1) * 4 * NC + column
        a00 = tl.load(A + index, mask, other=0.)
        a01 = tl.load(A + index + NC, mask, other=0.)
        a10 = tl.load(A + index + 2 * NC, mask, other=0.)
        a11 = tl.load(A + index + 3 * NC, mask, other=0.)
        m00, m01 = a00 * i00 + a01 * i10, a00 * i01 + a01 * i11
        m10, m11 = a10 * i00 + a11 * i10, a10 * i01 + a11 * i11
        c00 = tl.load(C + prev, mask, other=0.)
        c01 = tl.load(C + prev + NC, mask, other=0.)
        c10 = tl.load(C + prev + 2 * NC, mask, other=0.)
        c11 = tl.load(C + prev + 3 * NC, mask, other=0.)
        b00 = tl.load(B + index, mask, other=1.) - (m00 * c00 + m01 * c10)
        b01 = tl.load(B + index + NC, mask, other=0.) - (m00 * c01 + m01 * c11)
        b10 = tl.load(B + index + 2 * NC, mask, other=0.) - (m10 * c00 + m11 * c10)
        b11 = tl.load(B + index + 3 * NC, mask, other=1.) - (m10 * c01 + m11 * c11)
        i00, i01, i10, i11, valid = _inverse2(b00, b01, b10, b11)
        status = tl.where(valid, status, level + 1)
        tl.store(INVERSE + index, i00, mask)
        tl.store(INVERSE + index + NC, i01, mask)
        tl.store(INVERSE + index + 2 * NC, i10, mask)
        tl.store(INVERSE + index + 3 * NC, i11, mask)
        tl.store(MULTIPLIER + index, m00, mask)
        tl.store(MULTIPLIER + index + NC, m01, mask)
        tl.store(MULTIPLIER + index + 2 * NC, m10, mask)
        tl.store(MULTIPLIER + index + 3 * NC, m11, mask)
    tl.store(INFO + column, status, mask)


@triton.jit
def _apply_kernel(INVERSE, MULTIPLIER, C, RHS, WORK, OUT,
                  NC: tl.constexpr, NZ: tl.constexpr, NRHS: tl.constexpr,
                  TRANSPOSE: tl.constexpr, BLOCK: tl.constexpr, RHS_BLOCK: tl.constexpr):
    column = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    right = tl.program_id(1) * RHS_BLOCK + tl.arange(0, RHS_BLOCK)
    cmask = column < NC
    mask = (right[:, None] < NRHS) & cmask[None, :]
    offset = right[:, None] * NC + column[None, :]
    y0 = tl.load(RHS + offset, mask, other=0.)
    y1 = tl.load(RHS + NRHS * NC + offset, mask, other=0.)
    if TRANSPOSE:
        i00 = tl.load(INVERSE + column, cmask, other=1.)
        i01 = tl.load(INVERSE + NC + column, cmask, other=0.)
        i10 = tl.load(INVERSE + 2 * NC + column, cmask, other=0.)
        i11 = tl.load(INVERSE + 3 * NC + column, cmask, other=1.)
        q0, q1 = i00[None, :] * y0 + i10[None, :] * y1, i01[None, :] * y0 + i11[None, :] * y1
        y0, y1 = q0, q1
    tl.store(WORK + offset, y0, mask)
    tl.store(WORK + NRHS * NC + offset, y1, mask)
    for level in range(1, NZ):
        index = level * 4 * NC + column
        prev = (level - 1) * 4 * NC + column
        roffset = level * 2 * NRHS * NC + offset
        r0 = tl.load(RHS + roffset, mask, other=0.)
        r1 = tl.load(RHS + roffset + NRHS * NC, mask, other=0.)
        if TRANSPOSE:
            c00 = tl.load(C + prev, cmask, other=0.)
            c01 = tl.load(C + prev + NC, cmask, other=0.)
            c10 = tl.load(C + prev + 2 * NC, cmask, other=0.)
            c11 = tl.load(C + prev + 3 * NC, cmask, other=0.)
            q0 = r0 - (c00[None, :] * y0 + c10[None, :] * y1)
            q1 = r1 - (c01[None, :] * y0 + c11[None, :] * y1)
            i00 = tl.load(INVERSE + index, cmask, other=1.)
            i01 = tl.load(INVERSE + index + NC, cmask, other=0.)
            i10 = tl.load(INVERSE + index + 2 * NC, cmask, other=0.)
            i11 = tl.load(INVERSE + index + 3 * NC, cmask, other=1.)
            y0 = i00[None, :] * q0 + i10[None, :] * q1
            y1 = i01[None, :] * q0 + i11[None, :] * q1
        else:
            m00 = tl.load(MULTIPLIER + index, cmask, other=0.)
            m01 = tl.load(MULTIPLIER + index + NC, cmask, other=0.)
            m10 = tl.load(MULTIPLIER + index + 2 * NC, cmask, other=0.)
            m11 = tl.load(MULTIPLIER + index + 3 * NC, cmask, other=0.)
            q0 = r0 - (m00[None, :] * y0 + m01[None, :] * y1)
            q1 = r1 - (m10[None, :] * y0 + m11[None, :] * y1)
            y0, y1 = q0, q1
        tl.store(WORK + roffset, y0, mask)
        tl.store(WORK + roffset + NRHS * NC, y1, mask)
    if not TRANSPOSE:
        index = (NZ - 1) * 4 * NC + column
        i00 = tl.load(INVERSE + index, cmask, other=1.)
        i01 = tl.load(INVERSE + index + NC, cmask, other=0.)
        i10 = tl.load(INVERSE + index + 2 * NC, cmask, other=0.)
        i11 = tl.load(INVERSE + index + 3 * NC, cmask, other=1.)
        q0, q1 = i00[None, :] * y0 + i01[None, :] * y1, i10[None, :] * y0 + i11[None, :] * y1
        y0, y1 = q0, q1
    last = (NZ - 1) * 2 * NRHS * NC + offset
    tl.store(OUT + last, y0, mask)
    tl.store(OUT + last + NRHS * NC, y1, mask)
    for reverse_level in range(1, NZ):
        level = NZ - 1 - reverse_level
        roffset = level * 2 * NRHS * NC + offset
        q0 = tl.load(WORK + roffset, mask, other=0.)
        q1 = tl.load(WORK + roffset + NRHS * NC, mask, other=0.)
        if TRANSPOSE:
            index = (level + 1) * 4 * NC + column
            m00 = tl.load(MULTIPLIER + index, cmask, other=0.)
            m01 = tl.load(MULTIPLIER + index + NC, cmask, other=0.)
            m10 = tl.load(MULTIPLIER + index + 2 * NC, cmask, other=0.)
            m11 = tl.load(MULTIPLIER + index + 3 * NC, cmask, other=0.)
            x0 = q0 - (m00[None, :] * y0 + m10[None, :] * y1)
            x1 = q1 - (m01[None, :] * y0 + m11[None, :] * y1)
        else:
            index = level * 4 * NC + column
            c00 = tl.load(C + index, cmask, other=0.)
            c01 = tl.load(C + index + NC, cmask, other=0.)
            c10 = tl.load(C + index + 2 * NC, cmask, other=0.)
            c11 = tl.load(C + index + 3 * NC, cmask, other=0.)
            r0 = q0 - (c00[None, :] * y0 + c01[None, :] * y1)
            r1 = q1 - (c10[None, :] * y0 + c11[None, :] * y1)
            i00 = tl.load(INVERSE + index, cmask, other=1.)
            i01 = tl.load(INVERSE + index + NC, cmask, other=0.)
            i10 = tl.load(INVERSE + index + 2 * NC, cmask, other=0.)
            i11 = tl.load(INVERSE + index + 3 * NC, cmask, other=1.)
            x0 = i00[None, :] * r0 + i01[None, :] * r1
            x1 = i10[None, :] * r0 + i11[None, :] * r1
        y0, y1 = x0, x1
        tl.store(OUT + roffset, y0, mask)
        tl.store(OUT + roffset + NRHS * NC, y1, mask)


@torch.library.custom_op("defgcm_native::block2_factor", mutates_args=())
def factor_native(lower: torch.Tensor, diagonal: torch.Tensor,
                  upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inverse = torch.empty_like(diagonal)
    multiplier = torch.empty_like(diagonal)
    info = torch.empty(diagonal.shape[-1], dtype=torch.int32, device=diagonal.device)
    nz, _, _, nc = diagonal.shape
    _factor_kernel[(triton.cdiv(nc, 128),)](
        lower, diagonal, upper, inverse, multiplier, info, nc, nz, 128,
        num_warps=4, enable_fp_fusion=False)
    return inverse, multiplier, info


@factor_native.register_fake
def _factor_fake(lower, diagonal, upper):
    return (torch.empty_like(diagonal), torch.empty_like(diagonal),
            torch.empty(diagonal.shape[-1], dtype=torch.int32, device=diagonal.device))


@torch.library.custom_op("defgcm_native::block2_apply", mutates_args=())
def _apply_native(inverse: torch.Tensor, multiplier: torch.Tensor, upper: torch.Tensor,
                  rhs: torch.Tensor, transpose: bool) -> torch.Tensor:
    nz, _, nrhs, nc = rhs.shape
    output, work = torch.empty_like(rhs), torch.empty_like(rhs)
    rhs_block = min(4, triton.next_power_of_2(nrhs))
    _apply_kernel[(triton.cdiv(nc, 128), triton.cdiv(nrhs, rhs_block))](
        inverse, multiplier, upper, rhs, work, output, nc, nz, nrhs, transpose, 128, rhs_block,
        num_warps=4, enable_fp_fusion=False)
    return output


@_apply_native.register_fake
def _apply_fake(inverse, multiplier, upper, rhs, transpose):
    return torch.empty_like(rhs)


def apply_native(inverse: torch.Tensor, multiplier: torch.Tensor, upper: torch.Tensor,
                 rhs: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    return _apply_native(inverse, multiplier, upper, rhs, transpose)
