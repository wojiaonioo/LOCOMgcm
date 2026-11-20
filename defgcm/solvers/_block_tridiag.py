"""Factor and solve batches of 2 by 2 block tridiagonal systems.

Coefficient layout is (level, row, column, physical_column); right-hand sides
are (level, component, right_hand_side, physical_column). The physical column
axis is contiguous in the native implementation. Factorization is separate
from applying the factors, including their transpose. These are linear
algebra primitives for an equation-level derivative, not autograd boundaries.

A failed 2 by 2 pivot sets a per-column status and uses an identity work pivot.
The caller must check that status and the residual of its complete equation.
A singular block pivot does not imply that a bordered full matrix is singular.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlockFactors:
    inverse: torch.Tensor
    multiplier: torch.Tensor
    upper: torch.Tensor
    info: torch.Tensor
    backend: str


def _product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack((
        torch.stack((a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0],
                     a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1])),
        torch.stack((a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0],
                     a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1]))))


def _inverse(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = a.abs().amax(dim=(0, 1))
    valid_scale = torch.isfinite(scale) & (scale > 0)
    safe_scale = torch.where(valid_scale, scale, 1.)
    scaled = a / safe_scale
    determinant = scaled[0, 0] * scaled[1, 1] - scaled[0, 1] * scaled[1, 0]
    valid = valid_scale & torch.isfinite(determinant) & (determinant != 0)
    denominator = torch.where(valid, determinant, 1.)
    value = torch.stack((torch.stack((scaled[1, 1], -scaled[0, 1])),
                         torch.stack((-scaled[1, 0], scaled[0, 0]))))
    value = value / denominator / safe_scale
    valid = valid & torch.isfinite(value).all(dim=(0, 1))
    eye = torch.eye(2, dtype=a.dtype, device=a.device)[..., None]
    return torch.where(valid, value, eye), ~valid


def _apply(a: torch.Tensor, value: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    if transpose:
        return torch.stack((a[0, 0][None] * value[0] + a[1, 0][None] * value[1],
                            a[0, 1][None] * value[0] + a[1, 1][None] * value[1]))
    return torch.stack((a[0, 0][None] * value[0] + a[0, 1][None] * value[1],
                        a[1, 0][None] * value[0] + a[1, 1][None] * value[1]))


def factor(lower: torch.Tensor, diagonal: torch.Tensor, upper: torch.Tensor, *,
           backend: str = "torch") -> BlockFactors:
    """Return B=L*U factors, storing the inverse of each diagonal U block.

    The stored inverse is scaled before its determinant is formed. No inverse
    or factor is ever reused for a different coefficient tensor.
    """
    if (lower.shape != diagonal.shape or upper.shape != diagonal.shape
            or diagonal.ndim != 4 or diagonal.shape[1:3] != (2, 2)
            or diagonal.shape[0] < 1 or diagonal.shape[-1] < 1):
        raise ValueError("Block coefficients require matching (nz, 2, 2, ncolumn) tensors")
    if diagonal.dtype != torch.float64:
        raise ValueError("The block factorization currently requires float64")
    if any(value.dtype != diagonal.dtype or value.device != diagonal.device
           for value in (lower, upper)):
        raise ValueError("Block coefficients must share their dtype and device")
    if backend not in ("torch", "cuda"):
        raise ValueError("Block backend must be 'torch' or 'cuda'")
    lower, diagonal, upper = (value.contiguous() for value in (lower, diagonal, upper))
    with torch.no_grad():
        if backend == "cuda":
            if not diagonal.is_cuda:
                raise ValueError("The native block backend requires CUDA tensors")
            from ._block_tridiag_triton import factor_native
            inverse, multiplier, info = factor_native(lower, diagonal, upper)
        else:
            inverses, multipliers = [], []
            inverse, failed = _inverse(diagonal[0])
            inverses.append(inverse)
            multipliers.append(torch.zeros_like(diagonal[0]))
            info = failed.to(torch.int32)
            for level in range(1, diagonal.shape[0]):
                multiplier = _product(lower[level], inverse)
                pivot = diagonal[level] - _product(multiplier, upper[level - 1])
                inverse, failed = _inverse(pivot)
                inverses.append(inverse)
                multipliers.append(multiplier)
                info = torch.where(failed, torch.full_like(info, level + 1), info)
            inverse = torch.stack(inverses)
            multiplier = torch.stack(multipliers)
        return BlockFactors(inverse, multiplier, upper, info, backend)


def apply_factors(factors: BlockFactors, rhs: torch.Tensor, *,
                  transpose: bool = False) -> torch.Tensor:
    """Solve all RHS with one factorization, in either orientation."""
    nz, _, _, nc = factors.inverse.shape
    if rhs.ndim != 4 or rhs.shape[:2] != (nz, 2) or rhs.shape[-1] != nc:
        raise ValueError("Block right-hand sides require (nz, 2, nrhs, ncolumn)")
    if rhs.dtype != factors.inverse.dtype or rhs.device != factors.inverse.device:
        raise ValueError("Block right-hand sides must match the factors")
    if rhs.shape[2] < 1:
        raise ValueError("At least one right-hand side is required")
    with torch.no_grad():
        if factors.backend == "cuda":
            from ._block_tridiag_triton import apply_native
            return apply_native(factors.inverse, factors.multiplier, factors.upper,
                                rhs.contiguous(), transpose=transpose)
        inverse, multiplier, upper = factors.inverse, factors.multiplier, factors.upper
        work = []
        if transpose:
            work.append(_apply(inverse[0], rhs[0], transpose=True))
            for level in range(1, nz):
                value = rhs[level] - _apply(upper[level - 1], work[-1], transpose=True)
                work.append(_apply(inverse[level], value, transpose=True))
            result = [work[-1]]
            for level in range(nz - 2, -1, -1):
                result.append(work[level] - _apply(multiplier[level + 1], result[-1],
                                                  transpose=True))
        else:
            work.append(rhs[0])
            for level in range(1, nz):
                work.append(rhs[level] - _apply(multiplier[level], work[-1]))
            result = [_apply(inverse[-1], work[-1])]
            for level in range(nz - 2, -1, -1):
                result.append(_apply(inverse[level], work[level] - _apply(upper[level], result[-1])))
        return torch.stack(list(reversed(result)))


def matvec(lower: torch.Tensor, diagonal: torch.Tensor, upper: torch.Tensor,
           value: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    """Apply B or B transpose without materializing a dense column matrix."""
    def multiply(matrix, vector):
        if transpose:
            matrix = matrix.transpose(1, 2)
        return torch.stack((
            matrix[:, 0, 0, None] * vector[:, 0] + matrix[:, 0, 1, None] * vector[:, 1],
            matrix[:, 1, 0, None] * vector[:, 0] + matrix[:, 1, 1, None] * vector[:, 1]), dim=1)
    result = multiply(diagonal, value)
    zero = torch.zeros_like(result[:1])
    if transpose:
        above = multiply(upper[:-1], value[:-1])
        below = multiply(lower[1:], value[1:])
    else:
        above = multiply(lower[1:], value[:-1])
        below = multiply(upper[:-1], value[1:])
    return result + torch.cat((zero, above), dim=0) + torch.cat((below, zero), dim=0)
