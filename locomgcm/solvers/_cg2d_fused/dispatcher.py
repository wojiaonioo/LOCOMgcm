"""Unified-runtime dispatcher for the fused CG2D PCG iteration body.

The gate read and the ledger write live in ``_pcg_core/dispatch.py``, shared
with cg3d: one place reads a gate, one place writes a record, for every PCG
client (the shared acceleration dispatcher).  This module supplies only what
is specific to the five-point body -- which kernels exist and which operands
they take.

Admission checks operand shape: face-leading coefficients cannot be indexed
as ``aC[j, i]`` and decline with ``coefficient_rank``. The preconditioner is
selected by the name of the ALGORITHM the caller would otherwise run eagerly
(``ssor``, ``ssor_wrap_x``/``_y``/``_xy``, ``ssor_halo``, ``ssor_ring``,
``jacobi``, or the cube's seam-aware ``ssor_edge``, which is a different
expression and is declined) -- an algorithm name, not a topology.
``ssor_wrap_*`` name the axes on which the apply reads the opposite end of the
array; ``ssor_halo`` names the apply that reads an exchanged residual, and the
caller hands over the exchange (``precond_halo``) the way it hands the SPMV its
halo; ``ssor_ring`` is that apply evaluated on the tile plus a one-cell ring
from a width-2 residual exchange, which is what lets the loop carry the search
direction with its ring and skip the search-direction exchange.

Granularity: the gate is read once per solve, and one ledger row is written PER
KERNEL carrying that kernel's launch count.  Reading the gate per iteration
would put a snapshot lookup on the PCG critical path; writing one row for the
whole solve under-reported ignition, because the pre-loop ``spmv``/``precond``
and the first in-loop ``spmv`` all run before any iteration completes, so a
solve that broke down at ``pAp ~ 0`` launched three kernels and still reported
``requested-zero``. The iteration count is not lost:
the ``update_xr``/``update_xrz`` row's count IS the number of iterations served.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

from locomgcm.solvers._pcg_core.dispatch import FusedIterationLedger, open_iteration

from ._fast import (
    SSOR_PRECOND_OPS,
    SUPPORTED_DTYPES,
    bundle_decline_reason,
    cg2d_axpy_p,
    cg2d_spmv,
    cg2d_update_xr,
    cg2d_update_xr_ring,
    cg2d_update_xrz,
)

_KEY = "cg2d_fused_pcg"
_SITE = "cg2d.pcg_solve_fused"
_BACKEND = "cuda"
#: The preconditioners this package implements.  ``ssor`` is the regional
#: production default on a tile with no coupled edge; ``ssor_wrap_x``/``_y``/
#: ``_xy`` the same five-point apply on a single process that owns those
#: periodic axes (``apply_cg2d_ssor_preconditioner_wrapped``, kernel K3 with the
#: wrap as a template parameter); ``ssor_halo`` the apply from an exchanged
#: residual (``apply_cg2d_ssor_preconditioner_edge`` with the regional exchange),
#: which is the SPMV stencil with ``(pC, pW_edge, pS_edge)`` and is served by the
#: SPMV kernel; ``ssor_ring`` the same SPMV apply on the tile plus a one-cell ring
#: from a width-2 residual exchange, with the ringed search direction's x update
#: served by K2c; ``jacobi`` is the ``LOCOMGCM_CG2D_PRECOND=jacobi`` opt-in.  The cube's
#: ``apply_cg2d_ssor_preconditioner_edge`` reads a face-resolved halo the kernels
#: cannot index, so a caller running it names it ``ssor_edge`` and is declined
#: -- a statement about which apply is being reproduced, not about which grid
#: produced it.
SUPPORTED_PRECONDITIONERS = (
    "ssor",
    "ssor_wrap_x",
    "ssor_wrap_y",
    "ssor_wrap_xy",
    "ssor_halo",
    "ssor_ring",
    "jacobi",
)
#: ``(wrap_x, wrap_y)`` of each apply kernel K3 serves.
_SSOR_WRAP = {
    "ssor": (False, False),
    "ssor_wrap_x": (True, False),
    "ssor_wrap_y": (False, True),
    "ssor_wrap_xy": (True, True),
}
#: The kinds whose preconditioner reads the caller's residual exchange
#: (``precond_halo``), and the operand layout each is checked with.
_EXCHANGED_RESIDUAL_LAYOUT = {"ssor_halo": "edge", "ssor_ring": "ring"}


def _variant(precond_kind: str) -> str:
    return f"pcg_fused_{precond_kind}"


def dtype_accepted(dtype: torch.dtype) -> bool:
    """Whether this dispatcher serves ``dtype``.

    Named and exported so a CPU-only test can pin the predicate itself: on a
    CPU host the device check fires first, so a float16 tensor there declines
    for ``non_cuda`` whatever this says, and a test that only went through
    :func:`decline_reason` would stay green across a flip of this line.
    """
    return dtype in SUPPORTED_DTYPES


def decline_reason(
    *,
    states: Sequence[Any],
    aW_edge: Any,
    aS_edge: Any,
    aC: Any,
    precond_kind: str,
    precond_operands: Sequence[Any],
) -> str | None:
    """Why this solve cannot take the fused path (``None`` = it can).

    Only the preconditioner ALGORITHM and the operand layout.  There is no
    topology argument to pass and none to test.
    """
    if precond_kind not in SUPPORTED_PRECONDITIONERS:
        return "unsupported_preconditioner"
    expected_operands = 1 if precond_kind == "jacobi" else 3
    if len(precond_operands) != expected_operands:
        return "unsupported_preconditioner"
    return bundle_decline_reason(
        states,
        aW_edge,
        aS_edge,
        aC,
        precond_operands,
        precond_layout=_EXCHANGED_RESIDUAL_LAYOUT.get(precond_kind, "cell"),
    )


class Cg2dFusedIteration(FusedIterationLedger):
    """The fused kernels for one solve, plus that solve's ledger entry.

    Satisfies ``_pcg_core.loop.PcgKernelSet``; use it as a context manager
    around :func:`locomgcm.solvers._pcg_core.loop.pcg_iterate`.
    """

    def __init__(
        self,
        *,
        aW_edge: torch.Tensor,
        aS_edge: torch.Tensor,
        aC: torch.Tensor,
        precond_kind: str,
        precond_operands: Sequence[torch.Tensor],
        precond_halo: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> None:
        super().__init__(key=_KEY, site=_SITE, variant=_variant(precond_kind),
                         backend=_BACKEND)
        self._aW_edge = aW_edge
        self._aS_edge = aS_edge
        self._aC = aC
        self._precond_kind = precond_kind
        self._precond_operands = tuple(precond_operands)
        self._precond_halo = precond_halo
        wrap = _SSOR_WRAP.get(precond_kind)
        self._ssor_op = None if wrap is None else SSOR_PRECOND_OPS[wrap]

    @property
    def ringed(self) -> bool:
        """Whether ``precond`` returns ``z`` with a one-cell ring and the loop
        must carry ``p`` with it (``pcg_solve_fused`` reads this)."""
        return self._precond_kind == "ssor_ring"

    # -- kernels ------------------------------------------------------------

    def spmv(self, phi_halo: torch.Tensor) -> torch.Tensor:
        """``A p`` from an already-exchanged centre halo (the caller exchanges).

        The halo stays outside the kernel on purpose: multi-rank needs the P2P
        exchange between the search-direction update and this apply, and a
        kernel that did its own halo would either duplicate that machinery or
        silently read stale ghosts.
        """
        self.count("spmv")
        return cg2d_spmv(phi_halo, self._aW_edge, self._aS_edge, self._aC)

    def update(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        ap: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``(x + alpha p, r - alpha Ap, z)``; ``z`` is ``None`` under SSOR.

        Under Jacobi the preconditioner is pointwise in the NEW residual, so it
        rides along in the same launch.  Under SSOR it reads the new residual's
        neighbours, which needs the residual fully written -- a device-wide
        dependency, hence a second launch through :meth:`precond`.  Under
        ``ssor_ring`` ``p`` carries its ring and K2c reads its tile cells.
        """
        self.iterations += 1
        if self._precond_kind == "jacobi":
            (precond_inv,) = self._precond_operands
            self.count("update_xrz")
            return cg2d_update_xrz(x, p, r, ap, alpha, precond_inv)
        if self._precond_kind == "ssor_ring":
            self.count("update_xr_ring")
            x_out, r_out = cg2d_update_xr_ring(x, p, r, ap, alpha)
            return x_out, r_out, None
        self.count("update_xr")
        x_out, r_out = cg2d_update_xr(x, p, r, ap, alpha)
        return x_out, r_out, None

    def precond(self, r: torch.Tensor) -> torch.Tensor:
        """``M^-1 r``.

        Jacobi is a single elementwise multiply, which is already one launch;
        it stays the eager expression rather than gaining a kernel that would
        do the same work behind a heavier call.  SSOR is the fused five-point
        apply: K3 on one process's array (wrapping the periodic axes it owns),
        or, for ``ssor_halo``, the caller's exchange followed by the SPMV kernel
        on ``(pC, pW_edge, pS_edge)`` -- the exchange stays outside the kernel
        exactly as it does for :meth:`spmv`.  ``ssor_ring`` is the same with a
        width-2 exchange and ring-shaped coefficients, returning ``z`` on the
        tile plus its ring.
        """
        if self._precond_kind == "jacobi":
            (precond_inv,) = self._precond_operands
            scale = (
                precond_inv.unsqueeze(0)
                if r.ndim == precond_inv.ndim + 1
                else precond_inv
            )
            # Eager, and counted anyway: the ledger answers "did this dispatch
            # do the preconditioning", not "did a hand-written kernel run".
            self.count("jacobi_precond")
            return scale * r
        if self._precond_kind in _EXCHANGED_RESIDUAL_LAYOUT:
            pC, pW_edge, pS_edge = self._precond_operands
            self.count(f"{self._precond_kind}_precond")
            return cg2d_spmv(self._precond_halo(r), pW_edge, pS_edge, pC)
        pC, pW, pS = self._precond_operands
        self.count("ssor_precond")
        return self._ssor_op(r, pC, pW, pS)

    def axpy(
        self, z: torch.Tensor, p: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """``z + beta * p``."""
        self.count("axpy_p")
        return cg2d_axpy_p(z, p, beta)


def pcg_iteration_kernels(
    *,
    states: Sequence[torch.Tensor],
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
    precond_kind: str,
    precond_operands: Sequence[torch.Tensor],
    precond_halo: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Cg2dFusedIteration | None:
    """Return the fused iteration kernels for this solve, or ``None``.

    ``None`` means the caller must run the eager :func:`pcg_solve`; the reason
    is already in the ledger when the gate was on, and nothing is recorded when
    it was off (recording a disabled key is an error by construction --
    ``accel/runtime.py`` raises).

    ``precond_halo`` is the residual exchange of the ``ssor_halo`` apply
    (width 1) or the ``ssor_ring`` apply (width 2) and must be absent for
    every other preconditioner: a halo apply without its exchange, or an
    exchange handed to an apply that never reads it, is a wiring error and
    raises here rather than declining.
    """
    if (precond_kind in _EXCHANGED_RESIDUAL_LAYOUT) != (precond_halo is not None):
        raise ValueError(
            "precond_halo must be given exactly for precond_kind in "
            f"{tuple(_EXCHANGED_RESIDUAL_LAYOUT)} "
            f"(got precond_kind={precond_kind!r}, "
            f"precond_halo={'set' if precond_halo is not None else 'None'})"
        )
    variant = (
        _variant(precond_kind)
        if precond_kind in SUPPORTED_PRECONDITIONERS
        else "pcg_fused_unknown"
    )
    return open_iteration(
        key=_KEY,
        site=_SITE,
        variant=variant,
        backend=_BACKEND,
        decline=lambda: decline_reason(
            states=states,
            aW_edge=aW_edge,
            aS_edge=aS_edge,
            aC=aC,
            precond_kind=precond_kind,
            precond_operands=precond_operands,
        ),
        build=lambda: Cg2dFusedIteration(
            aW_edge=aW_edge,
            aS_edge=aS_edge,
            aC=aC,
            precond_kind=precond_kind,
            precond_operands=precond_operands,
            precond_halo=precond_halo,
        ),
    )


__all__ = [
    "Cg2dFusedIteration",
    "SUPPORTED_PRECONDITIONERS",
    "decline_reason",
    "dtype_accepted",
    "pcg_iteration_kernels",
]
