"""Ordered LSR line sweeps: SEAICE_LSR_TRIDIAGU/V with the per-pass line factors.

One sweep updates the LINE rectangle (native rows 2..24, columns 2..43, i.e.
jMin..jMax and iMin..iMax with SEAICE_OLx=1, seaice_lsr.F:177-180) of all four
tiles. U lines are rows solved along i; V lines are columns solved along j and
use the transposed storage, so both directions share one implementation. Lines
are swept in increasing order and read the already updated previous line
(Gauss-Seidel; SEAICE_LSR_ZEBRA and SEAICE_VECTORIZE_LSR are undefined in the
offline_exf_seaice build). Tiles are independent; no halo is refreshed here.

The matrix-only part of the Thomas recurrence (bet and CUU, TRIDIAGU:2011-2034,
TRIDIAGV:2164-2189) depends on AU/BU/CU alone. ``prepare_lines`` evaluates it
once per pass with the source's operations; every sweep of the pass reuses the
identical values. The load, forward elimination, back substitution and
relaxation keep the source's per-line operation order.

Without gradients both entries are opaque operators (``defgcm_seaice::``), so a
compiled step holds one node per entry; the gate ``seaice_lsr_cuda`` selects the
CUDA kernel of ``_lsr_cuda`` for them. With gradients the reference runs as
ordinary differentiable tensor code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from defgcm.accel.tensor_context import forward_mode_active

from .types import DynamicsGeometry, LinearSystem

# Native trim of the solve rectangle (types.SOLVE) in both storage orders; the
# scripted functions below spell it as a local, TorchScript has no int globals.
_TRIM = 2


def _factor_lines(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Stacked [bet, CUU] along the last axis of the LINE rectangle.

    ``bet[..., 0]`` is BU itself, the divisor MIT applies to the first point.
    """
    lo = 2
    hi_l, hi_p = a.size(-2) - lo, a.size(-1) - lo
    lower = a[..., lo:hi_l, lo:hi_p]
    diagonal = b[..., lo:hi_l, lo:hi_p]
    upper = c[..., lo:hi_l, lo:hi_p]
    cuu: List[Tensor] = [upper[..., 0] / diagonal[..., 0]]
    bet: List[Tensor] = [diagonal[..., 0]]
    for i in range(1, hi_p - lo):
        d = diagonal[..., i] - lower[..., i] * cuu[-1]
        bet.append(d)
        cuu.append(upper[..., i] / d)
    return torch.stack((torch.stack(bet, -1), torch.stack(cuu, -1)))


def _sweep_lines(coef: Tensor, factors: Tensor, old: Tensor, relax: Tensor) -> Tensor:
    """One ordered sweep in line-major storage.

    ``coef`` stacks [a, b, c, r1, r2, rhs, mask]; ``old`` is the sweep entry
    (uTmp/vTmp). Halo lines and the points beside each line keep ``old``.
    """
    a, b, c, r1, r2, rhs, mask = (coef[0], coef[1], coef[2], coef[3],
                                  coef[4], coef[5], coef[6])
    bet, cuu = factors[0], factors[1]
    lo = 2
    hi_l, hi_p = old.size(-2) - lo, old.size(-1) - lo
    n = hi_p - lo
    previous = old[..., lo - 1, lo:hi_p]
    lines: List[Tensor] = []
    for line in range(lo, hi_l):
        k = line - lo
        lower = a[..., line, lo:hi_p]
        # AA3 at iMin and iMax, zero inside (TRIDIAGU:2001-2005).
        first = 0.0 - lower[..., 0] * old[..., line, lo - 1]
        last = 0.0 - c[..., line, hi_p - 1] * old[..., line, hi_p]
        aa3 = torch.cat((first.unsqueeze(-1), torch.zeros_like(lower[..., 1:n - 1]),
                         last.unsqueeze(-1)), -1)
        load = (rhs[..., line, lo:hi_p] + aa3 + r1[..., line, lo:hi_p] * previous
                + r2[..., line, lo:hi_p] * old[..., line + 1, lo:hi_p]) * mask[..., line, lo:hi_p]
        values: List[Tensor] = [load[..., 0] / b[..., line, lo]]
        for i in range(1, n):
            values.append((load[..., i] - lower[..., i] * values[-1]) / bet[..., k, i])
        answer: List[Tensor] = [values[-1]]
        for i in range(n - 2, -1, -1):
            answer.append(values[i] - cuu[..., k, i] * answer[-1])
        answer.reverse()
        base = old[..., line, lo:hi_p]
        previous = base + relax * (torch.stack(answer, -1) - base)
        lines.append(previous)
    out = old.clone()
    out[..., lo:hi_l, lo:hi_p] = torch.stack(lines, -2)
    return out


# CPU cannot fuse TorchScript pointwise groups, so the scripted copy runs the
# same operations with less dispatch. CUDA runs the Python source: its
# TorchScript fuser would merge and contract the load expression.
_factor_lines_script = torch.jit.script(_factor_lines)
_sweep_lines_script = torch.jit.script(_sweep_lines)


def _lines(field: Tensor, transpose: bool) -> Tensor:
    return field.transpose(-1, -2) if transpose else field


def _scripted(reference: Tensor) -> bool:
    return reference.device.type == "cpu" and not torch.is_grad_enabled()


def reference_factors(a: Tensor, b: Tensor, c: Tensor, transpose: bool) -> Tensor:
    run = _factor_lines_script if _scripted(a) else _factor_lines
    return run(_lines(a, transpose), _lines(b, transpose), _lines(c, transpose))


def reference_sweep(coef: Tensor, factors: Tensor, old: Tensor, relax: Tensor,
                    transpose: bool) -> Tensor:
    run = _sweep_lines_script if _scripted(old) else _sweep_lines
    return _lines(run(_lines(coef, transpose), factors, _lines(old, transpose), relax),
                  transpose).contiguous()


@torch.library.custom_op("defgcm_seaice::lsr_factor", mutates_args=())
def _factor_op(a: Tensor, b: Tensor, c: Tensor, transpose: bool) -> Tensor:
    from ._lsr_cuda.dispatcher import factor

    return factor(a, b, c, transpose, lambda: reference_factors(a, b, c, transpose))


@_factor_op.register_fake
def _factor_fake(a: Tensor, b: Tensor, c: Tensor, transpose: bool) -> Tensor:
    lines, points = (a.size(-1), a.size(-2)) if transpose else (a.size(-2), a.size(-1))
    return a.new_empty((2, *a.shape[:-2], lines - 2 * _TRIM, points - 2 * _TRIM))


@torch.library.custom_op("defgcm_seaice::lsr_sweep", mutates_args=())
def _sweep_op(coef_u: Tensor, factors_u: Tensor, old_u: Tensor, relax_u: Tensor,
              coef_v: Tensor, factors_v: Tensor, old_v: Tensor, relax_v: Tensor,
              swept_u: bool, swept_v: bool) -> tuple[Tensor, Tensor]:
    from ._lsr_cuda.dispatcher import sweep_both

    # Both outputs are contiguous on every path, as the fake below declares.
    def reference() -> tuple[Tensor, Tensor]:
        keep = torch.contiguous_format
        return (reference_sweep(coef_u, factors_u, old_u, relax_u, False) if swept_u else old_u.clone(memory_format=keep),
                reference_sweep(coef_v, factors_v, old_v, relax_v, True) if swept_v else old_v.clone(memory_format=keep))

    return sweep_both(coef_u, factors_u, old_u, relax_u, coef_v, factors_v, old_v, relax_v,
                      swept_u, swept_v, reference)


@_sweep_op.register_fake
def _sweep_fake(coef_u: Tensor, factors_u: Tensor, old_u: Tensor, relax_u: Tensor,
                coef_v: Tensor, factors_v: Tensor, old_v: Tensor, relax_v: Tensor,
                swept_u: bool, swept_v: bool) -> tuple[Tensor, Tensor]:
    keep = torch.contiguous_format
    return torch.empty_like(old_u, memory_format=keep), torch.empty_like(old_v, memory_format=keep)


@dataclass(frozen=True)
class LineSystem:
    """One pass's sweep operands; factors stack [bet, CUU] per line."""

    coef_u: Tensor
    factors_u: Tensor
    coef_v: Tensor
    factors_v: Tensor


def _differentiable(*values: Tensor) -> bool:
    return torch.is_grad_enabled() and any(value.requires_grad for value in values)


def _forward_mode_factors(a: Tensor, b: Tensor, c: Tensor, transpose: bool) -> Tensor:
    # lsr_factor has no autograd and would drop a forward-mode tangent: the
    # reference lines, with the decline recorded as the operator's dispatcher would.
    from ._lsr_cuda.dispatcher import LSR_KEY, decline_for_forward_mode

    decline_for_forward_mode(LSR_KEY, "factor", (a, b, c))
    return reference_factors(a, b, c, transpose)


def prepare_lines(system: LinearSystem, g: DynamicsGeometry) -> LineSystem:
    """Per pass, after SEAICE_LSR_CALC_COEFFS: stacked operands and line factors."""
    coef_u = torch.stack((system.AU, system.BU, system.CU, system.uRt1, system.uRt2,
                          system.rhsU, g.seaiceMaskU))
    coef_v = torch.stack((system.AV, system.BV, system.CV, system.vRt1, system.vRt2,
                          system.rhsV, g.seaiceMaskV))
    # A live forward mode is read first, so its decline is recorded whatever
    # the operands require; without a level the order is the original one.
    factor = (_forward_mode_factors if forward_mode_active()
              else reference_factors if _differentiable(coef_u, coef_v)
              else torch.ops.defgcm_seaice.lsr_factor)
    return LineSystem(coef_u, factor(system.AU, system.BU, system.CU, False),
                      coef_v, factor(system.AV, system.BV, system.CV, True))


def sweep(lines: LineSystem, u: Tensor, v: Tensor, swept_u: bool, swept_v: bool,
          relax_u: Tensor, relax_v: Tensor) -> tuple[Tensor, Tensor]:
    """seaice_lsr.F:875-910: both snapshots precede either direction."""
    if forward_mode_active():
        # lsr_sweep has no autograd and would drop a forward-mode tangent; read
        # before the requires_grad branch so the decline is always recorded.
        from ._lsr_cuda.dispatcher import LSR_KEY, decline_for_forward_mode

        decline_for_forward_mode(LSR_KEY, "sweep", (lines.coef_u, lines.factors_u, u, relax_u,
                                                    lines.coef_v, lines.factors_v, v, relax_v))
        return (reference_sweep(lines.coef_u, lines.factors_u, u, relax_u, False) if swept_u else u.clone(),
                reference_sweep(lines.coef_v, lines.factors_v, v, relax_v, True) if swept_v else v.clone())
    if _differentiable(lines.coef_u, lines.coef_v, lines.factors_u, lines.factors_v,
                       u, v, relax_u, relax_v):
        return (reference_sweep(lines.coef_u, lines.factors_u, u, relax_u, False) if swept_u else u.clone(),
                reference_sweep(lines.coef_v, lines.factors_v, v, relax_v, True) if swept_v else v.clone())
    return torch.ops.defgcm_seaice.lsr_sweep(lines.coef_u, lines.factors_u, u, relax_u,
                                             lines.coef_v, lines.factors_v, v, relax_v,
                                             swept_u, swept_v)
