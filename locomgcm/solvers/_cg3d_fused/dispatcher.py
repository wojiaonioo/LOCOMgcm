"""Unified-runtime dispatcher for the fused CG3D PCG iteration body.

The gate read and the ledger write live in ``_pcg_core/dispatch.py``, shared
with cg2d: one place reads a gate, one place writes a record, for every PCG
client.  This module supplies only what is specific to the seven-point body --
which kernels exist and which operands they take -- and the adapter that turns
the caller's exchange handle into the loop's ``halo`` injection.

Padded-block kernels consume the neighbours supplied by the injected exchange
without inspecting grid family or rank count.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

from locomgcm.solvers._pcg_core.dispatch import FusedIterationLedger, open_iteration

from ._fast import (
    SUPPORTED_DTYPES,
    bundle_decline_reason,
    cg3d_axpy_s,
    cg3d_axpy_s_pad,
    cg3d_spmv_halo,
    cg3d_spmv_halo_dot,
    cg3d_thomas_mask,
    cg3d_thomas_mask_dot,
    cg3d_update_xr,
    to_kernel_mask,
)

_KEY = "cg3d_fused_pcg"
_SITE = "cg3d.CG3DSolver.solve"
_BACKEND = "cuda"
_VARIANT = "pcg_fused_thomas"


def dtype_accepted(dtype: torch.dtype) -> bool:
    """Whether this dispatcher serves ``dtype``.

    Named and exported so a CPU-only test can pin the predicate itself: on a
    CPU host the device check fires first, so a float16 tensor there declines
    for ``non_cuda`` whatever this says.
    """
    return dtype in SUPPORTED_DTYPES


def make_center_halo(
    grid: Any,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """The width-1 zero-filled centre exchange, or ``None`` if none was given.

    This is the ``halo`` injection, and it is the SAME call the eager production
    apply makes (``cg3d_operator.py:311-320``), so the fused SPMV consumes the
    same padded block the eager operator consumes -- which is what makes the
    bitwise claim a claim about one stencil rather than two.

    ``None`` when the caller supplied no exchange handle.  That is a statement
    about the INJECTION, not about a grid family or a rank count: without a
    padded block there is nothing to hand a stencil kernel, and the gate
    declines with ``no_halo_injection``.  (The eager reference it mirrors
    selects a genuinely different operator on the same condition -- the
    local-only branch at ``cg3d_operator.py:346-355``, which drops the
    boundary-coupling terms.  Collapsing those two branches is the injection
    layer's follow-up, not the solver's.)
    """
    if grid is None:
        return None
    from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange

    def halo(field: torch.Tensor) -> torch.Tensor:
        return exchange(
            field,
            grid=grid,
            location="T",
            extent=HaloExtentV1.uniform(1),
            fill=FillSpec.ZERO,
        )

    # The in-place counterpart, carried ON the closure so a caller that has the
    # halo injection also has the refresh -- same grid, same location, same
    # extent, same fill, therefore the same border cells.  ``cg3d.py`` passes
    # the injection down without knowing about either, so attaching it here
    # keeps the two from being resolved against different grids.
    halo.refresh = make_center_refresh(grid)
    return halo


def make_center_refresh(
    grid: Any,
) -> Callable[[Any], Any] | None:
    """The in-place counterpart of :func:`make_center_halo`.

    Same grid, same location, same extent, same fill -- so the border cells a
    ``PaddedField`` ends up holding are the ones ``exchange`` would have put in
    the tensor it returned.  What differs is only that the cells are written
    into a buffer that already exists rather than concatenated into a new one.

    ``None`` on the same condition as ``make_center_halo``: no grid, no
    injection.
    """
    if grid is None:
        return None
    from locomgcm.geometry.exchange import FillSpec

    def refresh(padded) -> None:
        padded.refresh(grid=grid, location="T", fill=FillSpec.ZERO)

    return refresh


def decline_reason(
    *,
    states: Sequence[Any],
    aW3d: Any,
    aS3d: Any,
    aV3d: Any,
    aC3d: Any,
    mask: Any,
    thomas: Sequence[Any],
    halo: Callable[[torch.Tensor], torch.Tensor] | None,
) -> str | None:
    """Why this solve cannot take the fused path (``None`` = it can)."""
    if halo is None:
        return "no_halo_injection"
    return bundle_decline_reason(states, aW3d, aS3d, aV3d, aC3d, mask, thomas)


class Cg3dFusedIteration(FusedIterationLedger):
    """The fused kernels for one solve, plus that solve's ledger entry.

    Satisfies ``_pcg_core.loop.PcgKernelSet``.  The mapping onto that protocol
    is the loop rotation described in ``_pcg_core/loop.py``: ``precond`` is the
    Thomas apply with the mask folded in, ``update`` returns ``z=None`` because
    the preconditioner is a vertical line solve rather than a pointwise scale,
    and every kernel multiplies by the mask exactly where eager does.
    """

    def __init__(
        self,
        *,
        aW3d: torch.Tensor,
        aS3d: torch.Tensor,
        aV3d: torch.Tensor,
        aC3d: torch.Tensor,
        mask: torch.Tensor,
        thomas: Sequence[torch.Tensor],
    ) -> None:
        super().__init__(key=_KEY, site=_SITE, variant=_VARIANT, backend=_BACKEND)
        self._aW3d = aW3d
        self._aS3d = aS3d
        self._aV3d = aV3d
        self._aC3d = aC3d
        # ONE conversion point for the whole solve.  The kernels multiply by a
        # 0/1 mask, so a byte carries it exactly at an eighth of a float64
        # mask's bytes -- and the mask is read by every kernel of every
        # iteration, so those bytes are the second-largest term in the
        # iteration's traffic after the state itself.  ``to_kernel_mask``
        # refuses a non-binary mask rather than truncating it.
        self._mask = to_kernel_mask(mask)
        self._thomas = tuple(thomas)

    def spmv(self, phi_halo: torch.Tensor) -> torch.Tensor:
        """``(A s) * mask`` from an already-exchanged padded block."""
        self.count("spmv_halo")
        return cg3d_spmv_halo(
            phi_halo, self._aW3d, self._aS3d, self._aV3d, self._aC3d, self._mask
        )

    def spmv_dot(self, phi_halo: torch.Tensor):
        """``((A s) * mask, partial sums of <s, (A s)*mask>)``.

        Only the captured graph calls this: the host core reduces through the
        caller-injected ``dot``, which may be a face-ordered reducer, and a
        kernel that has already summed its block cannot honour that ordering.
        """
        self.count("spmv_halo")
        return cg3d_spmv_halo_dot(
            phi_halo, self._aW3d, self._aS3d, self._aV3d, self._aC3d, self._mask
        )

    def precond(self, r: torch.Tensor) -> torch.Tensor:
        """``(M^-1 r) * mask`` -- the vertical-line Thomas solve."""
        zMC, zML, zMU = self._thomas
        self.count("thomas_mask")
        return cg3d_thomas_mask(zMC, zML, zMU, r, self._mask)

    def precond_dot(self, r: torch.Tensor):
        """``((M^-1 r) * mask, partial sums of <q, r>)`` -- graph path only."""
        zMC, zML, zMU = self._thomas
        self.count("thomas_mask")
        return cg3d_thomas_mask_dot(zMC, zML, zMU, r, self._mask)

    def update(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        ap: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``((x + alpha s)*mask, (r - alpha q)*mask, None)``."""
        self.iterations += 1
        self.count("update_xr")
        x_out, r_out = cg3d_update_xr(x, p, r, ap, alpha, self._mask)
        return x_out, r_out, None

    def axpy(
        self, z: torch.Tensor, p: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """``(z + beta p) * mask``."""
        self.count("axpy_s")
        return cg3d_axpy_s(z, p, beta, self._mask)

    def axpy_into(self, z: torch.Tensor, p, beta: torch.Tensor) -> None:
        """``(z + beta p) * mask`` written into a padded field's interior.

        ``p`` is a :class:`~locomgcm.geometry.padded_field.PaddedField`; its
        interior is both the ``p`` operand and the destination, avoiding a
        full-field copy. Graph path only -- the host core has no resident
        buffer to write into.
        """
        self.count("axpy_s")
        cg3d_axpy_s_pad(z, beta, self._mask, p.buffer, p.halo)


def pcg_iteration_kernels(
    *,
    states: Sequence[torch.Tensor],
    aW3d: torch.Tensor,
    aS3d: torch.Tensor,
    aV3d: torch.Tensor,
    aC3d: torch.Tensor,
    mask: torch.Tensor,
    thomas: Sequence[torch.Tensor],
    halo: Callable[[torch.Tensor], torch.Tensor] | None,
) -> Cg3dFusedIteration | None:
    """Return the fused iteration kernels for this solve, or ``None``.

    ``None`` means the caller must run the eager loop; the reason is already in
    the ledger when the gate was on, and nothing is recorded when it was off
    (recording a disabled key is an error by construction).
    """
    return open_iteration(
        key=_KEY,
        site=_SITE,
        variant=_VARIANT,
        backend=_BACKEND,
        decline=lambda: decline_reason(
            states=states,
            aW3d=aW3d,
            aS3d=aS3d,
            aV3d=aV3d,
            aC3d=aC3d,
            mask=mask,
            thomas=thomas,
            halo=halo,
        ),
        build=lambda: Cg3dFusedIteration(
            aW3d=aW3d,
            aS3d=aS3d,
            aV3d=aV3d,
            aC3d=aC3d,
            mask=mask,
            thomas=thomas,
        ),
    )


__all__ = [
    "Cg3dFusedIteration",
    "decline_reason",
    "dtype_accepted",
    "make_center_halo",
    "pcg_iteration_kernels",
]
