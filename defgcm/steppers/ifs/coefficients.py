"""Build free-surface and nonhydrostatic pressure coefficients.

CG2D west/south edge coefficients surround cell-centered diagonals; CG3D
coefficients use surface-first [nz, ny, nx] cells and staggered horizontal
faces. Geometry lengths are meters, positive downward in the vertical; hFac
and interior masks are dimensionless. Builders use the requested dtype/device.
"""
from __future__ import annotations

from typing import Any

import torch

from defgcm.dycore_parameters import NonhydrostaticParams
from defgcm.geometry import StructuredCGrid
from defgcm.geometry.grid import runtime_memo_constant
from defgcm.solvers import build_cg3d_operator, build_cg3d_preconditioner
from defgcm.steppers.base import TimeStepBundle
from defgcm.steppers.implicit_free_surface_cg2d import _assemble_helmholtz_coefficients


def memoized_helmholtz_coefficients(
    *,
    grid: StructuredCGrid,
    gravity: float,
    dt_mom: float,
    dt_freesurf: float,
    dtype: torch.dtype,
    device: torch.device,
    alpha_beta: float,
    interior_mask: torch.Tensor | None,
    layout_key: tuple[Any, int] | None,
    ctx: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the fixed-geometry edge coefficients for this ownership layout."""
    # The coefficients derive from the grid, so the grid owns them (its
    # runtime cache, which also holds the graphed PCG that captures their
    # addresses); the key holds the stepper's own inputs as values, so
    # compiled steps of any model can evaluate the guard on it.  A grid
    # shallow-copied for a communication session shares the cache.
    key = ("_helmholtz_coefficients", gravity, dt_mom, dt_freesurf, dtype, device,
           alpha_beta, layout_key)

    def build():
        from defgcm.distributed.comm_adjoint import constant_transport

        # pcg_solve_graphed captures these addresses (cg2d.py:833-842), so
        # the cold-build tensors must be eager-owned before that capture.
        # The memoized coefficients are state-independent constants: their
        # seam exchange runs once and must stay outside any bound
        # differentiation window (a replayed segment never re-issues it).
        with constant_transport():
            return _assemble_helmholtz_coefficients(
                grid, gravity, dt_mom, dt_freesurf, dtype, device, alpha_beta,
                interior_mask=interior_mask,
                ctx=ctx,
            )

    return runtime_memo_constant(grid, key, build)


def cg3d_coefficients(
    *,
    grid: StructuredCGrid,
    params: NonhydrostaticParams,
    time_scales: TimeStepBundle,
    gravity: float,
    dtype: torch.dtype,
    device: torch.device,
    mask_in_c: torch.Tensor | None,
):
    """Get the CG3D operator/preconditioner, rebuilding active geometry."""
    # Owned by the grid it derives from, keyed by the stepper's own inputs
    # as values (the Helmholtz coefficients' rule above).
    key = (
        "_cg3d_coefficients",
        params,
        gravity,
        time_scales.dt_mom,
        time_scales.dt_freesurf,
        dtype,
        device,
    )
    from contextlib import nullcontext
    from defgcm.distributed.comm_adjoint import constant_transport

    fixed = cg3d_geometry_fixed(
        grid=grid, dtype=dtype, device=device, mask_in_c=mask_in_c, gravity=gravity,
    )

    def build():
        # Fixed geometry exchanges are preparation, not window rounds.
        # Active geometry is rebuilt with its original graph on each call.
        with constant_transport() if fixed else nullcontext():
            operator = build_cg3d_operator(
                grid, params, time_scales, dtype=dtype, device=device,
                recip_bo=1.0 / gravity, mask_in_c=mask_in_c,
            )
            return (operator, build_cg3d_preconditioner(operator))

    if not fixed:
        return build()
    return runtime_memo_constant(grid, key, build)


def cg3d_geometry_fixed(*, grid, dtype, device, mask_in_c, gravity) -> bool:
    """Whether the CG3D coefficients are constants of the run (memoised)."""
    geometry = (grid.layer_thickness_for(dtype, device),
                grid.hfac_w_for(dtype, device), grid.hfac_s_for(dtype, device),
                grid.metrics.cell_area_for(dtype, device),
                grid.metrics.dxC_for(dtype, device), grid.metrics.dyG_for(dtype, device),
                grid.metrics.dxG_for(dtype, device), grid.metrics.dyC_for(dtype, device),
                grid.vertical_coordinate.interface_spacing, mask_in_c, gravity)
    return not getattr(grid.masks, "hfac_time_varying", False) and not any(
        isinstance(value, torch.Tensor) and value.requires_grad for value in geometry)
