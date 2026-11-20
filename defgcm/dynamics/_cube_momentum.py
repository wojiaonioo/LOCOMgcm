"""Cubed-sphere momentum utility functions.

Provides cube-specific helpers used by the UNIFIED operator paths in
``vector_invariant_momentum.py`` and ``horizontal_diffusion.py``, and by
the stepper's HPG routines.  The former independent momentum/viscosity
reimplementations (``cube_vector_invariant_momentum``,
``cube_harmonic_viscosity``, ``_Halo``) have been deleted — the unified
operator ``evaluate()`` methods now handle cube grids directly.
"""
from __future__ import annotations

from typing import Any

import torch

from defgcm.dynamics.planetary_coriolis import spherical_coriolis
from defgcm.geometry.exchange import (
    HaloExtentV1,
    _exchange_cubed_sphere_scalar,
)


def exchange_cube_velocities(
    u: torch.Tensor,
    v: torch.Tensor,
    grid,
    ol: int,
    *,
    with_signs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for the shared vector seam topology helper."""
    from defgcm.dynamics._vector_seam_topology import (
        exchange_vector_velocities_for_grid,
    )

    return exchange_vector_velocities_for_grid(
        u,
        v,
        grid,
        ol,
        with_signs=with_signs,
    )


def reconcile_cube_vector_redundant_edges(
    u: torch.Tensor,
    v: torch.Tensor,
    grid,
    *,
    u_mask: torch.Tensor | None = None,
    v_mask: torch.Tensor | None = None,
    with_signs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for the shared vector seam topology helper."""
    from defgcm.dynamics._vector_seam_topology import (
        reconcile_vector_redundant_edges_for_grid,
    )

    return reconcile_vector_redundant_edges_for_grid(
        u,
        v,
        grid,
        u_mask=u_mask,
        v_mask=v_mask,
        with_signs=with_signs,
    )


def build_fcorig_corner(grid, dtype, device, *, omega: float) -> torch.Tensor:
    """Coriolis parameter ``fCoriG = 2*omega*sin(yG)`` at cube corners ``[6,ny+1,nx+1]``."""
    yG = grid.metrics.face_geometry["yG"]
    return spherical_coriolis(yG.to(dtype=dtype, device=device), omega=omega)


def build_cube_hFacZ(grid, dtype, device) -> torch.Tensor:
    """Compatibility wrapper for the shared hFacZ topology helper."""
    from defgcm.dynamics._hfacz_topology import build_hfacz_for_grid

    return build_hfacz_for_grid(grid, dtype, device)


def _cube_mask_tendency(tend_u, tend_v, grid):
    """Compatibility wrapper for the shared vector mask topology helper."""
    from defgcm.dynamics._vector_mask_topology import mask_vector_tendency_for_grid

    return mask_vector_tendency_for_grid(tend_u, tend_v, grid)


def cube_phi_hyd(
    temp: torch.Tensor,
    salt: torch.Tensor,
    grid,
    *,
    gravity: float = 9.81,
    reference_density: float = 1035.0,
    hFacC: "torch.Tensor | None" = None,
    eos: "Any | None" = None,
    prev_total_phi_hyd: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Baroclinic hydrostatic potential ``phiHydC[n_face,nz,ny,nx]`` (m^2 s^-2).

    The density anomaly comes from the shared, config-driven equation of state
    (:meth:`JMD95EquationOfState.density_fields`) -- the SAME
    grid-independent EOS the lat-lon path uses -- so the cube no longer
    re-hardcodes JMD95Z.  ``eos=None`` builds the historical default (JMD95Z,
    ``dynamic_pressure=False``); a configured ``eos`` selects the variant +
    dynamic-pressure handling, with ``prev_total_phi_hyd`` (the lagged
    ``phiHyd + g*eta`` from the previous step, MITgcm ``totPhiHyd``) forwarded
    for the JMD95P path.

    The anomaly is integrated to the cell-centre hydrostatic potential with
    MITgcm ``integr_GeoPot=2``; both layers are LOCAL per column (no horizontal
    exchange) so the regional aligned routine applies to the face-first cube
    state ``[n_face,nz,ny,nx]`` unchanged.

    *hFacC*: when provided, overrides ``grid.masks.hFacC``.  Use a locally-
    scattered mask when ``temp``/``salt`` have ``[n_local, ...]`` shape (the
    multi-rank path).  ctx=None → *hFacC* defaults to ``grid.masks.hFacC``.

    The shared _pressure_field uses rhoConst*(g*z) in both regional and
    cube paths. Preserve that multiply order to avoid reassociation differences.
    """
    from defgcm.dynamics.equation_of_state import JMD95EquationOfState
    from defgcm.operators.vertical import hydrostatic_pressure_potential

    dtype, device = temp.dtype, temp.device
    vertical = grid.vertical_coordinate
    if eos is None:
        eos = JMD95EquationOfState(
            gravity=gravity, reference_density=reference_density, dynamic_pressure=False,
        )
    # The EOS consumer boundary, so the cube takes the ``eos_cuda`` dispatch
    # the lat-lon diagnostics take.
    rho_anom, _ = eos.density_fields(
        temp, salt, grid, prev_total_phi_hyd=prev_total_phi_hyd,
    )
    if hFacC is None:
        hFacC = grid.masks.hFacC
    return hydrostatic_pressure_potential(
        rho_anom,
        vertical.center_spacing.to(dtype=dtype, device=device),
        vertical.bottom_half_thickness.to(dtype=dtype, device=device),
        gravity=gravity, reference_density=eos.reference_density,
        hFacC=hFacC,
    )


def cube_hydrostatic_pressure_gradient(
    phi_hyd: torch.Tensor,
    grid,
    *,
    hFacC: "torch.Tensor | None" = None,
    ctx: "Any | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seam-aware ``-nabla phiHyd`` at u-/v-faces on the cube.

    Returns ``(gU[n_face,nz,ny,nx+1], gV[n_face,nz,ny+1,nx])``.

    When *ctx* is provided (multi-rank), the scalar halo exchange is
    routed to ``exchange_cubed_sphere_scalar_distributed`` and the dxC/dyC
    metrics are scattered to local face layout.  *hFacC* should already be
    the locally-scattered mask.  ctx=None: unchanged single-rank path.
    """
    dtype, device = phi_hyd.dtype, phi_hyd.device
    n = grid.nx
    n_local_or_6 = len(ctx.local_face_ids) if ctx is not None else grid.face_count

    if ctx is not None:
        from defgcm.geometry.exchange import exchange_cubed_sphere_scalar_distributed
        from defgcm.geometry.cube_decomposition import scatter_faces
        ph = exchange_cubed_sphere_scalar_distributed(
            phi_hyd, grid=grid, location="T",
            extent=HaloExtentV1.uniform(1), ctx=ctx,
        )
        dxC = scatter_faces(grid.metrics.dxC_row_for(dtype, device), ctx)
        dyC = scatter_faces(grid.metrics.dyC_for(dtype, device), ctx)
    else:
        ph = _exchange_cubed_sphere_scalar(
            phi_hyd, grid=grid, location="T", extent=HaloExtentV1.uniform(1),
        )
        dxC = grid.metrics.dxC_row_for(dtype, device)
        dyC = grid.metrics.dyC_for(dtype, device)

    dxC_edge = dxC.new_zeros(n_local_or_6, n, n + 1)
    dxC_edge[:, :, 0:n] = dxC
    dxC_edge[:, :, n] = dxC[:, :, n - 1]
    dxC_edge = dxC_edge.unsqueeze(1)
    dyC_edge = dyC.unsqueeze(1)

    ch = None
    if hFacC is not None:
        cwet = (hFacC.to(dtype=dtype, device=device) > 0).to(dtype)
        if ctx is not None:
            from defgcm.geometry.exchange import exchange_cubed_sphere_scalar_distributed
            ch = exchange_cubed_sphere_scalar_distributed(
                cwet, grid=grid, location="T",
                extent=HaloExtentV1.uniform(1), ctx=ctx,
            )
        else:
            ch = _exchange_cubed_sphere_scalar(
                cwet, grid=grid, location="T", extent=HaloExtentV1.uniform(1),
            )

    from defgcm.dynamics.hydrostatic_pressure import (
        pressure_gradient_from_haloed_potential,
    )

    gU, gV = pressure_gradient_from_haloed_potential(
        ph,
        dxC_u=dxC_edge,
        dyC_v=dyC_edge,
        wet_center_halo=ch,
    )
    return gU.contiguous(), gV.contiguous()
