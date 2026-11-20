"""Vector-invariant momentum advection for structured C-grid.

Decomposes momentum advection into:
  - Kinetic energy gradient: -∂KE/∂x at u-face, -∂KE/∂y at v-face
  - Vorticity flux: (ζ+f)·v̄ at u-face, -(ζ+f)·ū at v-face

Follows MITgcm `mom_vecinv.F`, `mom_calc_ke.F` (selectKEscheme=0/1/2/3),
`mom_vi_u/v_coriolis.F` (selectVortScheme=0-4), `mom_vi_u/v_coriolis_c4.F`
(highOrderVorticity, upwindVorticity) and `mom_vi_coriolis.F`
(selectCoriScheme=0/1/2/3).

Coriolis is merged into the (ζ+f) term, so a separate FCoriolis module
should NOT be used when this module is active.

This module is the adapter: it reads geometry and masks from the grid, runs
the halo exchanges, chooses the scheme from the tuples below and calls the
tensor kernels in ``kinetic_energy.py``, ``vorticity_flux.py``,
``planetary_coriolis.py``, ``nh_coriolis.py`` and
``_vector_invariant_vertical_shear.py``.  A new vorticity or KE scheme is a
kernel in one of those files plus an entry in its tuple and a branch here.

References:
    MITgcm/pkg/mom_vecinv/mom_vecinv.F
    MITgcm/pkg/mom_common/mom_calc_ke.F  (selectKEscheme=0/1/2/3)
    MITgcm/pkg/mom_vecinv/mom_vi_u_coriolis.F  (selectVortScheme=0-4)
    MITgcm/pkg/mom_vecinv/mom_vi_u_coriolis_c4.F  (high-order / upwind vorticity)
    MITgcm/pkg/mom_vecinv/mom_vi_coriolis.F  (selectCoriScheme=0/1/2/3)
    MITgcm/pkg/mom_vecinv/mom_vi_u_grad_ke.F
    MITgcm/pkg/mom_common/mom_calc_relvort3.F
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Literal

import torch
import torch.distributed as dist

from defgcm.accel.constant_cache import constant_cache
from defgcm.geometry import StructuredCGrid
from defgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from defgcm.geometry.halo_exchange import HaloExchange
from defgcm.geometry.metric_bundle import (
    MomentumMetricBundle,
    build_momentum_metric_bundle,
    build_vertshear_metric_bundle,
)
from defgcm.state import DiagnosticState, PrognosticState, StateTendency
from defgcm.state.state import resolved_input

from defgcm.geometry.metrics import _cache_device_key
from defgcm.constants import ROTATION_PERIOD_EARTH, omega_from_rotation_period

from .base import TendencyModule, TendencyStepDomain
from . import viscosity_geometry as _viscosity_geometry
from ._horizontal_diffusion_topology import _has_distributed_neighbor
from ._vector_invariant_topology import (
    apply_uv_tendency_masks_for_grid,
    corner_coriolis_field_for_grid,
    relative_vorticity,
    resolve_coriolis_profile_tensor,
    vector_invariant_halo_kwargs_for_grid,
)
from ._vector_invariant_vertical_shear import (
    vertical_shear_all_z,
    vertical_shear_per_layer,
    vertical_shear_wbars_area_weighted,
    vertical_shear_wbars_simple,
)
from .nh_coriolis import u_coriolis_nh_from_haloed_w
from .planetary_coriolis import f_cori_cos_rows, latitude_center_cos_from_metrics
from .vorticity_flux import (
    _absolute_vorticity,
    _apply_face_mask,
    _apply_positive_face_mask,
    _apply_u_edge_pair_mask,
    _apply_v_edge_pair_mask,
    _mask_corner_value_by_hfac,
    _u_face_coriolis_from_vbar,
    _u_face_coriolis_from_vbar_row,
    _u_vbar_hfacz,
    _u_vbar_hfacz_row,
    _u_vbar_no_hfac,
    _u_vbar_no_hfac_row,
    _v_face_coriolis_from_ubar,
    _v_face_coriolis_from_ubar_row,
    _v_face_coriolis_from_ubar_row_unscaled,
    _v_ubar_hfacz,
    _v_ubar_hfacz_row,
    _v_ubar_no_hfac,
    _v_ubar_no_hfac_row,
    coriolis_hfac_weighted_from_haloed_transports,
    coriolis_wet_point_from_haloed_transports,
    corner_vorticity_over_hfac,
    energy_conserving_from_haloed_transports,
    energy_enstrophy_from_haloed_factors,
    energy_fourth_order_from_haloed_transports,
    energy_from_haloed_transports,
    energy_upwind_from_haloed_transports,
    enstrophy_fourth_order_from_haloed_transports,
    enstrophy_from_haloed_transports,
    enstrophy_upwind_from_haloed_transports,
    hfac_face_transports,
)
from .kinetic_energy import (
    ke_center_area_hfac_weighted,
    ke_center_area_weighted,
    ke_center_hfac_weighted,
    ke_center_unweighted,
    ke_gradient_u_from_haloed_ke,
    ke_gradient_v_from_haloed_ke,
)
from defgcm.dynamics._registry import register_dynamics_module


# The schemes this module accepts, as (name, MITgcm code) pairs.  This is the
# capability declaration: each name has exactly one branch below, and the
# name-to-code correspondence is written only here.
#: selectVortScheme (mom_vi_u_coriolis.F): 0 is the enstrophy-conserving
#: average, 1 the same with hFacZ weights, 2 the energy-conserving form,
#: 3 the energy- and enstrophy-conserving form, 4 the energy-conserving form
#: without r_hFacZ.
VORT_SCHEMES = (
    ("enstrophy", 0),
    ("enstrophy_hfacz", 1),
    ("energy", 2),
    ("energy_enstrophy", 3),
    ("energy_no_rhfacz", 4),
)
#: How the corner vorticity reaches the face (MITgcm highOrderVorticity,
#: upwindVorticity; both on is refused, config_check.F:945-951): the centred
#: average of the scheme, the fourth-order interpolation or the upstream corner
#: of mom_vi_u/v_coriolis_c4.F, which has schemes 0 and 2 only.
VORT_RECONSTRUCTIONS = ("centered", "high_order", "upwind")
#: selectKEscheme (mom_calc_ke.F).
KE_SCHEMES = (
    ("unweighted", 0),
    ("area_weighted", 1),
    ("hfac_weighted", 2),
    ("area_hfac_weighted", 3),
)
#: select3dCoriScheme (mom_u_coriolis_nh.F): 0 is no 3-D Coriolis term.
CORI_3D_SCHEMES = (("none", 0), ("original", 1))
#: selectCoriScheme (mom_vi_coriolis.F), read on the split-Coriolis path only:
#: 0 the simple average without hFac, 1 the partial-cell wet-point (Jamart)
#: average, 2 the hFac-weighted average, 3 the energy-conserving form.
CORI_SCHEMES = (
    ("simple", 0),
    ("wet_point", 1),
    ("hfac_weighted", 2),
    ("energy_conserving", 3),
)
#: upwindShear (mom_vi_u_vertshear.F): only the centred average is implemented.
SHEAR_SCHEMES = (("centered", False),)


def _scheme_name(
    schemes: tuple[tuple[str, int], ...],
    code: int,
    parameter: str,
) -> str:
    """The name of MITgcm scheme ``code`` in ``schemes``; unlisted codes are refused."""
    for name, value in schemes:
        if value == code:
            return name
    raise NotImplementedError(
        f"{parameter}={code} is not implemented; supported values are "
        + ", ".join(str(value) for _, value in schemes)
    )


def _vim_trace(message: str) -> None:
    if os.environ.get("DEFGCM_DIST_TRACE") != "1":
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[rank{rank}] vector_invariant: {message}", flush=True)


def _compute_ke_center(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
    *,
    select_ke_scheme: int = 0,
) -> torch.Tensor:
    """Kinetic energy at cell centers (selectKEscheme=0/1/2/3).

    Scheme 0 is MITgcm's unweighted vector-invariant KE:
        0.25 * (u_w^2 + u_e^2 + v_s^2 + v_n^2)

    Scheme 1 weights U/V terms by ``rAw``/``rAs`` and multiplies by
    ``recip_rA``. Scheme 2 weights U/V terms by ``hFacW``/``hFacS`` and
    multiplies by ``recip_hFacC``. Scheme 3 applies both area and hFac
    weights, then multiplies by ``recip_hFacC * recip_rA``.

    Parameters
    ----------
    u : [B, nz, ny, nx+1] or [B, ny, nx+1]
    v : [B, nz, ny+1, nx] or [B, ny+1, nx]

    Returns
    -------
    KE : [B, (nz,) ny, nx]
    """
    scheme = _scheme_name(KE_SCHEMES, select_ke_scheme, "select_ke_scheme")
    dtype = u.dtype
    device = u.device

    if scheme == "unweighted":
        return ke_center_unweighted(u, v)
    elif scheme == "area_weighted":
        rA, rAw, rAs = _ke_area_metrics(grid, dtype, device)
        return ke_center_area_weighted(
            u, v, rA=rA, rAw=rAw, rAs=rAs, ny=grid.ny, nx=grid.nx,
        )
    elif scheme == "hfac_weighted":
        hFacW, hFacS, hFacC = _ke_hfac_fields(grid, dtype, device)
        return ke_center_hfac_weighted(u, v, hFacW=hFacW, hFacS=hFacS, hFacC=hFacC)
    elif scheme == "area_hfac_weighted":
        rA, rAw, rAs = _ke_area_metrics(grid, dtype, device)
        hFacW, hFacS, hFacC = _ke_hfac_fields(grid, dtype, device)
        return ke_center_area_hfac_weighted(
            u,
            v,
            rA=rA,
            rAw=rAw,
            rAs=rAs,
            hFacW=hFacW,
            hFacS=hFacS,
            hFacC=hFacC,
            ny=grid.ny,
            nx=grid.nx,
        )


def _ke_area_metrics(
    grid: StructuredCGrid,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centre, U-face and V-face areas read by the area-weighted KE schemes."""
    rA = grid.metrics.cell_area_for(dtype, device)
    rAw = grid.metrics.rAw_for(dtype, device)
    rAs = grid.metrics.rAs_for(dtype, device)
    return rA, rAw, rAs


def _ke_hfac_fields(
    grid: StructuredCGrid,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``hFacW``/``hFacS``/``hFacC`` read by the hFac-weighted KE schemes."""
    hFacW = grid.hfac_w_for(dtype, device)
    hFacS = grid.hfac_s_for(dtype, device)
    if grid.masks.hFacC is not None:
        hFacC = _viscosity_geometry._hfac_c(
            grid, dtype, device, site="vector_invariant_momentum._hfac_c"
        )
    else:
        hFacC = torch.ones((grid.nz, grid.ny, grid.nx), dtype=dtype, device=device)
    return hFacW, hFacS, hFacC


def _ke_gradient_u(ke: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    """KE gradient at u-faces: dKEdx(i) = -(KE(i) - KE(i-1)) / dx.

    Follows MITgcm mom_vi_u_grad_ke.F:30.
    Output is zero at domain-boundary faces for non-periodic grids.

    Parameters
    ----------
    ke : [*, ny, nx]

    Returns
    -------
    dKEdx : [*, ny, nx+1]
    """
    nx = grid.nx
    ny = grid.ny

    dxC = grid.metrics.dxC_for(ke.dtype, ke.device)                       # [ny,nx+1] or [6,ny,nx+1]
    if dxC.ndim >= 3 and ke.ndim >= 4:
        dxC = dxC.unsqueeze(-3)                                           # [6,1,ny,nx+1]

    ke_ext = exchange(
        ke,
        grid=grid,
        location="U",
        extent=HaloExtentV1(
            west_left_extent=1,
            east_right_extent=1,
            south_bottom_extent=0,
            north_top_extent=0,
        ),
        fill=FillSpec.REPLICATE,
    )

    # The expression already materialises a [*ke.shape[:-1], nx+1] field, which
    # is exactly the shape the discarded zeros() buffer declared — bind it
    # instead of allocating and copying over it.
    result = ke_gradient_u_from_haloed_ke(ke_ext, dxC, nx=nx)

    if grid.masks.has_3d_masks and ke.ndim >= 4:
        result = _apply_face_mask(
            result,
            grid.masks.u_float(dtype=ke.dtype, device=ke.device),
        )
    elif getattr(grid, "face_count", 1) == 6 and grid.masks.hFacW is not None:
        result = _apply_positive_face_mask(result, grid.masks.hFacW)

    return result


def _ke_gradient_v(ke: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    """KE gradient at v-faces: dKEdy(j) = -(KE(j) - KE(j-1)) / dy.

    Follows MITgcm mom_vi_v_grad_ke.F:30.

    Parameters
    ----------
    ke : [*, ny, nx]

    Returns
    -------
    dKEdy : [*, ny+1, nx]
    """
    ny = grid.ny
    # dy = dyC (meridional centre-to-centre distance) at the v-face: the proper
    # denominator of this ∂KE/∂y gradient.  2-D on cubed-sphere / variable-
    # spacing grids, bit-equal to the scalar on uniform grids.
    dy = grid.metrics.dyC_for(ke.dtype, ke.device)
    if dy.ndim >= 3 and ke.ndim >= 4:
        dy = dy.unsqueeze(-3)

    ke_ext = exchange(
        ke,
        grid=grid,
        location="V",
        extent=HaloExtentV1(
            west_left_extent=0,
            east_right_extent=0,
            south_bottom_extent=1,
            north_top_extent=1,
        ),
        fill=FillSpec.REPLICATE,
    )

    # Same as the U analog: the expression's shape is the buffer's shape.
    result = ke_gradient_v_from_haloed_ke(ke_ext, dy, ny=ny)

    if getattr(grid, "face_count", 1) == 6 and grid.masks.hFacS is not None:
        result = _apply_positive_face_mask(result, grid.masks.hFacS)
    elif grid.masks.has_3d_masks and ke.ndim >= 4:
        result = _apply_face_mask(
            result,
            grid.masks.v_float(dtype=ke.dtype, device=ke.device),
        )

    return result


def _combine_split_coriolis_tendencies(
    dKEdx: torch.Tensor,
    dKEdy: torch.Tensor,
    plan_u: torch.Tensor,
    plan_v: torch.Tensor,
    zeta_flux_u: torch.Tensor,
    zeta_flux_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine KE-gradient, planetary, and relative-vorticity tendencies."""
    return dKEdx + plan_u + zeta_flux_u, dKEdy + plan_v + zeta_flux_v


def _combine_vorticity_tendencies(
    dKEdx: torch.Tensor,
    dKEdy: torch.Tensor,
    vort_flux_u: torch.Tensor,
    vort_flux_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine KE-gradient and absolute-vorticity flux tendencies."""
    return dKEdx + vort_flux_u, dKEdy + vort_flux_v


def _add_uv_tendency_pair(
    tend_u: torch.Tensor,
    tend_v: torch.Tensor,
    add_u: torch.Tensor,
    add_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add caller-provided U/V tendency increments."""
    return tend_u + add_u, tend_v + add_v


def _compute_hFacZ(
    grid: StructuredCGrid,
    dtype: torch.dtype,
    device: torch.device,
    *,
    hZoption: int | None = None,
) -> torch.Tensor:
    """Memoize-once wrapper over :func:`_compute_hFacZ_uncached` (Site 1).

    hFacZ is a per-grid geometry constant *while hFac is static*, so a cache
    hit returns the exact tensor the recompute path produced (bitwise-identical
    by construction).  The key disambiguates grid identity (weak), dtype,
    device, the resolved ``hZoption``, and whether a distributed step ctx is
    attached to the grid (which changes the cube exchange path).

    Three things send it back to the per-call recompute: ``DEFGCM_CACHE_CONSTANTS=0``;
    a grid whose hFac is rewritten in place every step under
    ``nonlinFreeSurf>=1``; and a grid whose hFac carries a graph, i.e. a
    geometry (bathymetry) control.  hFacZ is an hFac-DERIVED quantity and the
    key is grid identity, which neither the in-place write nor the presence of
    a graph moves, so the memo would otherwise serve the pre-write geometry for
    the rest of the run, or a detached constant whose bathymetry gradient is
    silently zero -- the same two defects the ``horizontal_diffusion`` sites
    decline on (MITgcm recomputes hFacZ in ``mom_calc_hfacz.F`` on every
    ``MOM_*`` call).  The decision and its declaration are imported from that
    module rather than restated here: one predicate, three call sites.
    """
    # Switch off: recompute in the caller's graph; ``_build`` below runs
    # under no_grad (see horizontal_diffusion._divvort_recip_hfac).
    if not _viscosity_geometry._constant_cache_enabled():
        return _compute_hFacZ_uncached(grid, dtype, device, hZoption=hZoption)
    bypass = _viscosity_geometry._hfac_memo_bypass_reason(grid)
    if bypass is not None:
        _viscosity_geometry._record_hfac_memo_bypass(
            "hfacz_corner", "vector_invariant_momentum._compute_hFacZ", bypass,
        )
        return _compute_hFacZ_uncached(grid, dtype, device, hZoption=hZoption)

    option = _viscosity_geometry._hfacz_option(hZoption)
    ctx_present = getattr(grid, "_cube_step_ctx", None) is not None
    inner_key = (dtype, str(device), option, ctx_present)

    def _build():
        # Constant: keep it out of the autograd graph (requires_grad=False).
        # no_grad changes no arithmetic, so the value stays bitwise-equal.
        with torch.no_grad():
            return _compute_hFacZ_uncached(grid, dtype, device, hZoption=hZoption)

    return constant_cache.get_or_build(
        ("vector_invariant_momentum._compute_hFacZ", inner_key),
        _build,
        variant="hfacz_corner",
        site="vector_invariant_momentum._compute_hFacZ",
        owner=grid,
    )


def _compute_hFacZ_uncached(
    grid: StructuredCGrid,
    dtype: torch.dtype,
    device: torch.device,
    *,
    hZoption: int | None = None,
) -> torch.Tensor:
    """Compute hFac at corner (vorticity) points.

    Follows MITgcm mom_calc_hfacz.F (hZoption=0, default):
      hFacZ(i,j) = min(hFacW(i,j), hFacW(i,j-1), hFacS(i,j), hFacS(i-1,j))

    With hZoption=1 on a cubed-sphere grid, the shared MIN-3
    cube-corner override is applied at the 24 face-local corner positions.
    hZoption=2 is intentionally gated with NotImplementedError, matching
    horizontal_diffusion's behavior.

    This means: a corner point is only open if ALL 4 surrounding faces are open.
    Free-slip at walls: if any adjacent face is dry, hFacZ = 0.

    Shape: [ny+1, nx+1] for 2-D masks, [nz, ny+1, nx+1] for 3-D masks.
    """
    ny = grid.ny
    nx = grid.nx
    option = _viscosity_geometry._hfacz_option(hZoption)

    if getattr(grid, "face_count", 1) == 6:
        if grid.masks.hFacW is None or grid.masks.hFacS is None:
            hFacW = torch.ones(
                grid.face_count, grid.nz, ny, nx + 1,
                dtype=dtype, device=device,
            )
            hFacS = torch.ones(
                grid.face_count, grid.nz, ny + 1, nx,
                dtype=dtype, device=device,
            )
        else:
            hFacW = _viscosity_geometry._hfac_w(
                grid, dtype, device, site="vector_invariant_momentum._hfac_w"
            )
            hFacS = _viscosity_geometry._hfac_s(
                grid, dtype, device, site="vector_invariant_momentum._hfac_s"
            )
        return _viscosity_geometry._compute_hFacZ_4neighbor(
            hFacW,
            hFacS,
            grid,
            hZoption=option,
        )

    # MITgcm ``mom_calc_hfacz.F`` fills the hFacZ corner halos via EXCH
    # (periodic self-wrap on a single tile), so the vector-invariant Coriolis
    # consumer — which reads the FULL [ny+1, nx+1] corner field, not just the
    # interior block — must use ``FillSpec.WRAP`` for the boundary rows /
    # columns.  When hFacW/hFacS are unit masks, MITgcm's wall / open-edge
    # support additionally collapses self-wrapped boundary corners to zero;
    # the shared core reproduces this with ``region_support=True`` (a second
    # ZERO-fill 4-neighbour MIN factor).  Both behaviours, plus the cube
    # hZoption=1 corner override, now live in the single shared core.
    unit_hfac = (
        not grid.masks.has_3d_masks
        or grid.masks.hFacW is None
        or grid.masks.hFacS is None
    )
    if unit_hfac:
        if grid.masks.has_3d_masks:
            hFacW = torch.ones(grid.nz, ny, nx + 1, dtype=dtype, device=device)
            hFacS = torch.ones(grid.nz, ny + 1, nx, dtype=dtype, device=device)
        else:
            hFacW = torch.ones(ny, nx + 1, dtype=dtype, device=device)
            hFacS = torch.ones(ny + 1, nx, dtype=dtype, device=device)
    else:
        hFacW = _viscosity_geometry._hfac_w(
            grid, dtype, device, site="vector_invariant_momentum._hfac_w"
        )
        hFacS = _viscosity_geometry._hfac_s(
            grid, dtype, device, site="vector_invariant_momentum._hfac_s"
        )

    return _viscosity_geometry._compute_hFacZ_4neighbor(
        hFacW,
        hFacS,
        grid,
        hZoption=option,
        region_fill=FillSpec.WRAP,
        region_support=unit_hfac,
    )


def _edge_pair_mask(
    length: int,
    *,
    grid: StructuredCGrid,
    axis: Literal["x", "y"],
    extent: HaloExtentV1,
    device: torch.device,
) -> torch.Tensor | None:
    if grid.edge_face_couples(axis):
        return None
    core = torch.ones(length, dtype=torch.bool, device=device)
    core_ext = exchange(
        core,
        grid=grid,
        location="F",
        extent=extent,
        fill=FillSpec.ZERO,
        constant=True,
    )
    return core_ext[:-1] & core_ext[1:]


def _region_u_row_metrics(
    metrics: MomentumMetricBundle,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Region U-face row metrics ``(dxG_south, dxG_north, recip_dxC_u)``.

    This helper is reached only by the uniform-x regional fast path.  It derives
    the factored columns from the complete ``dxG`` / ``dxC`` fields of the
    regional bundle, preserving the established arithmetic.
    """
    ny = metrics.ny
    dxG_row_ext = metrics.dxG[:, 0]
    dxC_row = metrics.dxC[:, 0]
    dxG_south = dxG_row_ext[:ny].view(ny, 1)
    dxG_north = dxG_row_ext[1 : ny + 1].view(ny, 1)
    recip_dxC_u = (1.0 / dxC_row).view(ny, 1)
    return dxG_south, dxG_north, recip_dxC_u


def _region_v_dyg(
    metrics: MomentumMetricBundle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Region V-face uniform meridional metric ``(dyG, recip_dyC)`` (scalars).

    This helper is reached only by the uniform-y regional fast path and derives
    both scalars from the complete ``dyG`` / ``dyC`` fields of the regional
    bundle.  The planetary V leaf retains its established scalar broadcast
    arithmetic on that path.
    """
    dyG = metrics.dyG[0, 0]
    dyC = metrics.dyC[0, 0]
    return dyG, 1.0 / dyC


# The four region Coriolis helpers below halo the SAME velocity with the SAME
# extent and fill: both u-face helpers pad ``v`` in x, both v-face helpers pad
# ``u`` in y.  Naming the two extents once lets ``evaluate`` run each exchange a
# single time and hand the result to both consumers, instead of every helper
# materialising its own copy of an identical field.
_REGION_X_HALO = HaloExtentV1(
    west_left_extent=1,
    east_right_extent=1,
    south_bottom_extent=0,
    north_top_extent=0,
)
_REGION_Y_HALO = HaloExtentV1(
    west_left_extent=0,
    east_right_extent=0,
    south_bottom_extent=1,
    north_top_extent=1,
)


def _region_v_x_halo(v: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    """ZERO-filled x-halo of ``v`` shared by the two u-face Coriolis helpers."""
    return exchange(
        v,
        grid=grid,
        location="V",
        extent=_REGION_X_HALO,
        fill=FillSpec.ZERO,
    )  # [..., ny+1, nx+2]


def _region_u_y_halo(u: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    """ZERO-filled y-halo of ``u`` shared by the two v-face Coriolis helpers."""
    return exchange(
        u,
        grid=grid,
        location="U",
        extent=_REGION_Y_HALO,
        fill=FillSpec.ZERO,
    )  # [..., ny+2, nx+1]


def _u_face_vorticity_flux(
    v: torch.Tensor,
    corner_value: torch.Tensor,
    grid: StructuredCGrid,
    *,
    average: str,
    hFacZ: torch.Tensor,
    v_ext: torch.Tensor | None = None,
    v_x_halo: torch.Tensor | None = None,
    dxG_2d: torch.Tensor | None = None,
    dxC_2d: torch.Tensor | None = None,
    metrics: MomentumMetricBundle,
) -> torch.Tensor:
    """U-face flux of a corner vorticity with the hFacZ-weighted or the no-hFac average.

    ``average="hfacz"`` is the vorticity flux at u-faces (MITgcm
    selectVortScheme=1) of the relative or absolute vorticity.
    ``average="no_hfac"`` is the planetary Coriolis term at u-faces, MITgcm
    ``selectCoriScheme=0`` (no hFac), of ``fCoriG``; it does not read ``hFacZ``.

    MITgcm ``mom_vi_u_coriolis.F:75-90`` (scheme 1, modified-hFac variant):

        vBarXY = 0.5 * (
            hFacZ(i,j)   * (v(i,j)·dxG(i,j)   + v(i-1,j)·dxG(i-1,j))
          + hFacZ(i,j+1) * (v(i,j+1)·dxG(i,j+1) + v(i-1,j+1)·dxG(i-1,j+1))
        ) / max(eps, hFacZ(i,j) + hFacZ(i,j+1))

        vort3u = 0.5 * (omega3(i,j) + omega3(i,j+1))

        uCoriolisTerm(i,j) = +vort3u · vBarXY · recip_dxC(i,j) · _maskW(i,j,k)

    For a **non-rotated** grid, ``dxG(i,j)`` depends only on ``j`` (the v-face
    row latitude), so the inner parentheses reduce to
    ``dxG_row[j] · (v(i,j) + v(i-1,j))`` — but ``dxG_row[j]`` at the south
    v-face row is **not** the same as ``dxG_row[j+1]`` at the north v-face
    row on a spherical polar grid, and ``recip_dxC(i,j)`` at the u-face row
    (= cell-centre latitude yC[j]) is yet a third per-row value.

    Keep the three metrics separate on spherical polar grids. On a
    rectangular grid they are equal constants, so this reduces algebraically
    to ``vort3u * vBarXY_over_dx``.

    MITgcm ``mom_vi_coriolis.F:62-66`` uses **un-weighted** vBarXY for the
    planetary contribution, even when ``selectVortScheme=1`` is in effect
    for the relative vorticity.  Splitting is essential because at partial
    cells the hFacZ-weighted vBarXY differs from the no-hFac vBarXY, and
    MITgcm's planetary Coriolis is computed via the no-hFac branch::

        vBarXY = 0.25 * (
            v(i,j)·dxG(i,j)   + v(i-1,j)·dxG(i,j)
          + v(i,j+1)·dxG(i,j+1) + v(i-1,j+1)·dxG(i,j+1)
        )
        uCori  = 0.5·(fG(i,j) + fG(i,j+1)) · vBarXY · recip_dxC(i,j) · maskW

    For non-rotated spherical-polar grids ``dxG`` and ``recip_dxC`` vary
    with latitude only, so the per-row treatment is the same as for the
    scheme-1 flux.

    The seam u-faces LOCOMgcm i=0/i=nx (= MIT i=1/sNx+1)
    receive the MIT-equivalent planetary value via periodic-wrapped v
    (`mom_vi_coriolis.F:60-72` selectCoriScheme=0 with `vFld(0,j)` from EXCH
    self-wrap); for wall/open they remain zero via ``edge_face_couples("x")``,
    preserving the baseline bitwise.

    Parameters
    ----------
    v : [B, nz, ny+1, nx] or [B, ny+1, nx]
    corner_value : [*, nz, ny+1, nx+1] or [*, ny+1, nx+1] — absolute or
        relative vorticity (``average="hfacz"``), or ``fCoriG`` (``"no_hfac"``)
    grid : StructuredCGrid
    average : "no_hfac" or "hfacz"
    hFacZ : [nz, ny+1, nx+1] or [ny+1, nx+1]

    Returns
    -------
    tendency_u : [B, (nz,) ny, nx+1]
    """
    ny = grid.ny
    nx = grid.nx

    # ---- Cube path (pre-haloed v_ext + 2-D metrics from caller) ----
    if v_ext is not None:
        if average == "no_hfac":
            v_bar = _u_vbar_no_hfac(v_ext, dxG_2d, ny=ny, nx=nx)
        elif average == "hfacz":
            v_bar = _u_vbar_hfacz(v_ext, dxG_2d, hFacZ, ny=ny, nx=nx)
        return _u_face_coriolis_from_vbar(
            v_bar,
            corner_value,
            _viscosity_geometry._recip(dxC_2d),
            face_mask=grid.masks.hFacW,
        )

    x_halo = _REGION_X_HALO
    v_ext_local = v_x_halo if v_x_halo is not None else _region_v_x_halo(v, grid)

    face_mask = (
        grid.masks.u_float(dtype=v.dtype, device=v.device)
        if grid.masks.has_3d_masks and v.ndim >= 4 else None
    )
    if not grid.metrics.is_uniform_x:
        dxG, dxC = metrics.dxG, metrics.dxC
        dxG_ext_local = exchange(
            dxG,
            grid=grid,
            location="V",
            extent=x_halo,
            fill=FillSpec.ZERO,
        )
        if average == "no_hfac":
            v_bar = _u_vbar_no_hfac(v_ext_local, dxG_ext_local, ny=ny, nx=nx)
        elif average == "hfacz":
            v_bar = _u_vbar_hfacz(v_ext_local, dxG_ext_local, hFacZ, ny=ny, nx=nx)
        result = _u_face_coriolis_from_vbar(
            v_bar,
            corner_value,
            _viscosity_geometry._recip(dxC),
            face_mask=face_mask,
        )
        # A coupled x axis keeps the values supplied by the halo exchange.
        edge_pair_mask = _edge_pair_mask(
            nx, grid=grid, axis="x", extent=x_halo, device=v.device
        )
        if edge_pair_mask is not None:
            result = _apply_u_edge_pair_mask(result, edge_pair_mask)
        return result

    dxG_south, dxG_north, recip_dxC_u = _region_u_row_metrics(metrics)
    edge_face_couples = grid.edge_face_couples("x")
    edge_pair_mask = _edge_pair_mask(
        nx,
        grid=grid,
        axis="x",
        extent=x_halo,
        device=v.device,
    )
    # The single accelerator consumer boundary for the ``vim_coriolis_u``
    # native kernel: one wiring point serves both averages (the vorticity-flux
    # and planetary branches).  The kernel's interface selects its variant by
    # whether hFacZ is passed; its eager fallback is the dispatcher's ``_reference``.
    from defgcm.accel import (
        consumer_snapshot,
        in_accelerated_dispatch,
        requested,
    )

    if not in_accelerated_dispatch():
        with consumer_snapshot():
            if requested(("vim_coriolis_u",)):
                from ._vim_coriolis_u_cuda import dispatcher

                if average == "no_hfac":
                    native_hfacz = None
                elif average == "hfacz":
                    native_hfacz = hFacZ
                return dispatcher.u_face_coriolis_from_haloed_v_row_metrics_cuda(
                    v_ext_local,
                    corner_value,
                    dxG_south=dxG_south,
                    dxG_north=dxG_north,
                    recip_dxC_u=recip_dxC_u,
                    hFacZ=native_hfacz,
                    face_mask=face_mask,
                    edge_pair_mask=edge_pair_mask,
                    edge_face_couples=edge_face_couples,
                )

    if average == "no_hfac":
        v_bar = _u_vbar_no_hfac_row(
            v_ext_local, dxG_south=dxG_south, dxG_north=dxG_north, ny=ny, nx=nx,
        )
    elif average == "hfacz":
        v_bar = _u_vbar_hfacz_row(
            v_ext_local, hFacZ, dxG_south=dxG_south, dxG_north=dxG_north, ny=ny, nx=nx,
        )
    return _u_face_coriolis_from_vbar_row(
        v_bar,
        corner_value,
        recip_dxC_u,
        face_mask=face_mask,
        edge_face_couples=edge_face_couples,
        edge_pair_mask=edge_pair_mask,
    )


def _v_face_vorticity_flux(
    u: torch.Tensor,
    corner_value: torch.Tensor,
    grid: StructuredCGrid,
    *,
    average: str,
    hFacZ: torch.Tensor,
    u_ext: torch.Tensor | None = None,
    u_y_halo: torch.Tensor | None = None,
    dyG_2d: torch.Tensor | None = None,
    dyC_2d: torch.Tensor | None = None,
    metrics: MomentumMetricBundle,
) -> torch.Tensor:
    """V-face flux of a corner vorticity with the hFacZ-weighted or the no-hFac average.

    ``average="hfacz"`` is the vorticity flux at v-faces (selectVortScheme=1).

    MITgcm mom_vi_v_coriolis.F:75-89:

      uBarXY = 0.5 * (
          (u(i,j)*dy*hFacZ(i,j) + u(i,j-1)*dy*hFacZ(i,j))
        + (u(i+1,j)*dy*hFacZ(i+1,j) + u(i+1,j-1)*dy*hFacZ(i+1,j))
      ) / max(eps, hFacZ(i,j) + hFacZ(i+1,j))

      vort3v = 0.5 * (omega3(i,j) + omega3(i+1,j))

      vCoriolisTerm(i,j) = -vort3v * uBarXY * recip_dyC * maskS

    Uniform grid simplification (dy cancels):

      uBarXY_over_dy = 0.5 * (
          hFacZ(i,j) * (u(i,j) + u(i,j-1))
        + hFacZ(i+1,j) * (u(i+1,j) + u(i+1,j-1))
      ) / max(eps, hFacZ(i,j) + hFacZ(i+1,j))

      tendency_v(i,j) = -vort3v * uBarXY_over_dy * maskS

    ``average="no_hfac"`` is the planetary Coriolis at v-face, MITgcm
    ``selectCoriScheme=0`` (no hFac), the mirror of the u-face planetary
    term::

        uBarXY = 0.25 * dyG · (
            u(i,j) + u(i,j-1) + u(i+1,j) + u(i+1,j-1)
        )
        vCori  = -0.5·(fG(i,j) + fG(i+1,j)) · uBarXY · recip_dyC · maskS

    Uniform spherical-polar grid → ``dyG = dyC = const``, so no per-row y-halo
    padding is needed when widening to the full j ∈ [0, ny] range.

    The seam v-faces LOCOMgcm j=0/j=ny (= MIT j=1/sNy+1)
    receive the MIT-equivalent planetary value via periodic-wrapped u
    (`mom_vi_coriolis.F:127-139` selectCoriScheme=0 with `uFld(i,0)` from
    MIT's EXCH self-wrap); for wall/open they remain zero via
    ``edge_face_couples("y")``, preserving the baseline bitwise.

    Parameters
    ----------
    u : [B, nz, ny, nx+1] or [B, ny, nx+1]
    corner_value : [*, ny+1, nx+1] — vorticity (``"hfacz"``) or ``fCoriG`` (``"no_hfac"``)
    grid : StructuredCGrid
    average : "no_hfac" or "hfacz"
    hFacZ : [*, ny+1, nx+1]

    Returns
    -------
    tendency_v : [B, (nz,) ny+1, nx]
    """
    ny = grid.ny
    nx = grid.nx

    # ---- Cube path (pre-haloed u_ext + 2-D metrics from caller) ----
    if u_ext is not None:
        if average == "no_hfac":
            u_bar = _v_ubar_no_hfac(u_ext, dyG_2d, ny=ny, nx=nx)
        elif average == "hfacz":
            u_bar = _v_ubar_hfacz(u_ext, dyG_2d, hFacZ, ny=ny, nx=nx)
        return _v_face_coriolis_from_ubar(
            u_bar,
            corner_value,
            _viscosity_geometry._recip(dyC_2d),
            face_mask=grid.masks.hFacS,
        )

    y_halo = _REGION_Y_HALO
    u_ext_local = u_y_halo if u_y_halo is not None else _region_u_y_halo(u, grid)

    face_mask = (
        grid.masks.v_float(dtype=u.dtype, device=u.device)
        if grid.masks.has_3d_masks and u.ndim >= 4 else None
    )
    if not grid.metrics.is_uniform_y:
        dyG, dyC = metrics.dyG, metrics.dyC
        dyG_ext_local = exchange(
            dyG,
            grid=grid,
            location="U",
            extent=y_halo,
            fill=FillSpec.ZERO,
        )
        if average == "no_hfac":
            u_bar = _v_ubar_no_hfac(u_ext_local, dyG_ext_local, ny=ny, nx=nx)
        elif average == "hfacz":
            u_bar = _v_ubar_hfacz(u_ext_local, dyG_ext_local, hFacZ, ny=ny, nx=nx)
        result = _v_face_coriolis_from_ubar(
            u_bar,
            corner_value,
            _viscosity_geometry._recip(dyC),
            face_mask=face_mask,
        )
        # A coupled y axis keeps the values supplied by the halo exchange.
        edge_pair_mask = _edge_pair_mask(
            ny, grid=grid, axis="y", extent=y_halo, device=u.device
        )
        if edge_pair_mask is not None:
            result = _apply_v_edge_pair_mask(result, edge_pair_mask)
        return result
    if average == "no_hfac":
        dyG, recip_dyC = _region_v_dyg(metrics)
        edge_face_couples = grid.edge_face_couples("y")
        edge_pair_mask = _edge_pair_mask(
            ny,
            grid=grid,
            axis="y",
            extent=y_halo,
            device=u.device,
        )
        u_bar = _v_ubar_no_hfac_row(u_ext_local, dyG=dyG, ny=ny, nx=nx)
        return _v_face_coriolis_from_ubar_row(
            u_bar,
            corner_value,
            recip_dyC,
            face_mask=face_mask,
            edge_face_couples=edge_face_couples,
            edge_pair_mask=edge_pair_mask,
        )
    elif average == "hfacz":
        edge_face_couples = grid.edge_face_couples("y")
        edge_pair_mask = _edge_pair_mask(
            ny,
            grid=grid,
            axis="y",
            extent=y_halo,
            device=u.device,
        )
        u_bar = _v_ubar_hfacz_row(u_ext_local, hFacZ, ny=ny, nx=nx)
        return _v_face_coriolis_from_ubar_row_unscaled(
            u_bar,
            corner_value,
            face_mask=face_mask,
            edge_face_couples=edge_face_couples,
            edge_pair_mask=edge_pair_mask,
        )


def _vorticity_flux_for_scheme(
    vort_scheme: str,
    reconstruction: str,
    u: torch.Tensor,
    v: torch.Tensor,
    corner_value: torch.Tensor,
    hFacZ: torch.Tensor,
    grid: StructuredCGrid,
    *,
    metrics: MomentumMetricBundle,
    halo_kw_u: dict[str, torch.Tensor],
    halo_kw_v: dict[str, torch.Tensor],
    uface_kw: dict[str, torch.Tensor],
    vface_kw: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """U/V flux of the corner vorticity ``corner_value`` for one VORT_SCHEMES name.

    ``reconstruction`` (VORT_RECONSTRUCTIONS) is read by schemes 0 and 2 only;
    the module refuses a non-centred value with the other schemes.
    """
    if vort_scheme == "enstrophy":
        t = _face_transports(
            u, v, grid, metrics=metrics, halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
        )
        vort3r = corner_vorticity_over_hfac(corner_value, hFacZ)
        masks = _face_masks(u, v, grid)
        if reconstruction == "centered":
            flux_u, flux_v = enstrophy_from_haloed_transports(
                t.u_ext, t.v_ext, vort3r, dxc=t.dxc, dyc=t.dyc, **masks)
        elif reconstruction == "high_order":
            vort3r_x, vort3r_y = _corner_ghost_halos(vort3r, grid)
            flux_u, flux_v = enstrophy_fourth_order_from_haloed_transports(
                t.u_ext, t.v_ext, vort3r_x, vort3r_y, dxc=t.dxc, dyc=t.dyc, **masks)
        elif reconstruction == "upwind":
            flux_u, flux_v = enstrophy_upwind_from_haloed_transports(
                t.u_ext, t.v_ext, vort3r, dxc=t.dxc, dyc=t.dyc, **masks)
        return _region_edge_pair_masks(flux_u, flux_v, grid)
    elif vort_scheme == "enstrophy_hfacz":
        flux_u = _u_face_vorticity_flux(
            v, corner_value, grid, average="hfacz", hFacZ=hFacZ,
            metrics=metrics, **uface_kw)
        flux_v = _v_face_vorticity_flux(
            u, corner_value, grid, average="hfacz", hFacZ=hFacZ,
            metrics=metrics, **vface_kw)
        return flux_u, flux_v
    elif vort_scheme == "energy":
        t = _face_transports(
            u, v, grid, metrics=metrics, halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
        )
        masks = _face_masks(u, v, grid)
        if reconstruction == "centered":
            flux_u, flux_v = energy_from_haloed_transports(
                t.u_ext, t.v_ext, corner_value, hFacZ, dxc=t.dxc, dyc=t.dyc, **masks)
        elif reconstruction == "high_order":
            vort3r_x, vort3r_y = _corner_ghost_halos(
                corner_vorticity_over_hfac(corner_value, hFacZ), grid)
            flux_u, flux_v = energy_fourth_order_from_haloed_transports(
                t.u_ext, t.v_ext, vort3r_x, vort3r_y, dxc=t.dxc, dyc=t.dyc, **masks)
        elif reconstruction == "upwind":
            flux_u, flux_v = energy_upwind_from_haloed_transports(
                t.u_ext, t.v_ext, corner_vorticity_over_hfac(corner_value, hFacZ),
                dxc=t.dxc, dyc=t.dyc, **masks)
        return _region_edge_pair_masks(flux_u, flux_v, grid)
    elif vort_scheme == "energy_enstrophy":
        t = _face_transports(
            u, v, grid, metrics=metrics, halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
        )
        vort3r_x, vort3r_y = _corner_ghost_halos(
            corner_vorticity_over_hfac(corner_value, hFacZ), grid)
        # MITgcm multiplies velocity, face length and hFac onto the corner mean
        # one at a time (mom_vi_u_coriolis.F:118-141), so the factors are haloed
        # apart instead of as one transport.
        hfac_s_x = exchange(
            t.hfac_s, grid=grid, location="V", extent=_REGION_X_HALO,
            fill=FillSpec.ZERO, constant=not t.hfac_s.requires_grad,
        )
        hfac_w_y = exchange(
            t.hfac_w, grid=grid, location="U", extent=_REGION_Y_HALO,
            fill=FillSpec.ZERO, constant=not t.hfac_w.requires_grad,
        )
        dxg_x = exchange(
            metrics.dxG, grid=grid, location="V", extent=_REGION_X_HALO,
            fill=FillSpec.ZERO, constant=not metrics.dxG.requires_grad,
        )
        dyg_y = exchange(
            metrics.dyG, grid=grid, location="U", extent=_REGION_Y_HALO,
            fill=FillSpec.ZERO, constant=not metrics.dyG.requires_grad,
        )
        flux_u, flux_v = energy_enstrophy_from_haloed_factors(
            uface_kw["v_x_halo"], dxg_x, hfac_s_x,
            vface_kw["u_y_halo"], dyg_y, hfac_w_y,
            vort3r_x, vort3r_y,
            dxc=t.dxc, dyc=t.dyc, **_face_masks(u, v, grid),
        )
        return _region_edge_pair_masks(flux_u, flux_v, grid)
    elif vort_scheme == "energy_no_rhfacz":
        return _energy_conserving_flux(
            u, v, corner_value, grid, metrics=metrics,
            halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
        )


@dataclass(frozen=True)
class _FaceTransports:
    """Haloed face transports and the face metrics their consumers divide by.

    ``u_ext`` is ``u*dyG*hFacW`` with one y-halo row on each side
    ``[..., ny+2, nx+1]``, ``v_ext`` is ``v*dxG*hFacS`` with one x-halo column
    on each side ``[..., ny+1, nx+2]``; ``dxc``/``hfac_w`` sit on U faces and
    ``dyc``/``hfac_s`` on V faces.  Read by the energy-conserving corner flux
    (selectVortScheme=4, selectCoriScheme=3) and the selectCoriScheme=1/2
    planetary terms.
    """

    u_ext: torch.Tensor
    v_ext: torch.Tensor
    dxc: torch.Tensor
    dyc: torch.Tensor
    hfac_w: torch.Tensor
    hfac_s: torch.Tensor


def _face_transports(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
    *,
    metrics: MomentumMetricBundle,
    halo_kw_u: dict[str, torch.Tensor],
    halo_kw_v: dict[str, torch.Tensor],
) -> _FaceTransports:
    """Face transports ``u*dyG*hFacW``, ``v*dxG*hFacS`` and their one-wide halos."""
    hfac_w = (
        _viscosity_geometry._hfac_w(
            grid, u.dtype, u.device, site="vector_invariant_momentum._hfac_w"
        )
        if grid.masks.hFacW is not None
        else grid.masks.u_float(dtype=u.dtype, device=u.device)
    )
    hfac_s = (
        _viscosity_geometry._hfac_s(
            grid, v.dtype, v.device, site="vector_invariant_momentum._hfac_s"
        )
        if grid.masks.hFacS is not None
        else grid.masks.v_float(dtype=v.dtype, device=v.device)
    )
    if metrics.is_cube:
        # Take owned metrics from the same oriented halo frame as velocity.
        dxg = halo_kw_v["dxG_2d"][..., :, 1:-1]
        dyg = halo_kw_u["dyG_2d"][..., 1:-1, :]
        dxc, dyc = halo_kw_v["dxC_2d"], halo_kw_u["dyC_2d"]
    else:
        dxg, dxc, dyg, dyc = metrics.dxG, metrics.dxC, metrics.dyG, metrics.dyC

    u_transport, v_transport = hfac_face_transports(
        u, v, dyg=dyg, dxg=dxg, hfac_w=hfac_w, hfac_s=hfac_s
    )
    if metrics.is_cube:
        # Exchange the weighted vector so rotations also exchange its face
        # metric and wet fraction; no rank-local metric is used at a seam.
        u_kw, v_kw = vector_invariant_halo_kwargs_for_grid(
            u_transport, v_transport, grid
        )
        u_ext, v_ext = u_kw["u_ext"], v_kw["v_ext"]
    else:
        u_ext = _region_u_y_halo(u_transport, grid)
        v_ext = _region_v_x_halo(v_transport, grid)
    return _FaceTransports(
        u_ext=u_ext, v_ext=v_ext, dxc=dxc, dyc=dyc, hfac_w=hfac_w, hfac_s=hfac_s,
    )


def _region_edge_pair_masks(
    tend_u: torch.Tensor,
    tend_v: torch.Tensor,
    grid: StructuredCGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the edge faces of each non-coupling regional axis.

    Coupled (periodic or distributed) axes keep the values supplied by the
    halo exchange.
    """
    u_edge_pair_mask = _edge_pair_mask(
        grid.nx, grid=grid, axis="x", extent=_REGION_X_HALO, device=tend_u.device
    )
    if u_edge_pair_mask is not None:
        tend_u = _apply_u_edge_pair_mask(tend_u, u_edge_pair_mask)
    v_edge_pair_mask = _edge_pair_mask(
        grid.ny, grid=grid, axis="y", extent=_REGION_Y_HALO, device=tend_v.device
    )
    if v_edge_pair_mask is not None:
        tend_v = _apply_v_edge_pair_mask(tend_v, v_edge_pair_mask)
    return tend_u, tend_v


def _energy_conserving_flux(
    u: torch.Tensor,
    v: torch.Tensor,
    corner_value: torch.Tensor,
    grid: StructuredCGrid,
    *,
    metrics: MomentumMetricBundle,
    halo_kw_u: dict[str, torch.Tensor],
    halo_kw_v: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT energy-conserving corner flux, without an r_hFacZ factor.

    selectVortScheme=4 with the vorticity (mom_vi_u/v_coriolis.F:151-168).
    Each corner's face transport is multiplied by that corner's value before
    averaging to the output face.  This is independent of the
    highOrderVorticity C4 stencil.
    """
    t = _face_transports(
        u, v, grid, metrics=metrics, halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
    )
    tend_u, tend_v = energy_conserving_from_haloed_transports(
        t.u_ext, t.v_ext, corner_value,
        dxc=t.dxc, dyc=t.dyc, hfac_w=t.hfac_w, hfac_s=t.hfac_s,
    )
    if not metrics.is_cube:
        tend_u, tend_v = _region_edge_pair_masks(tend_u, tend_v, grid)
    return tend_u, tend_v


def _face_masks(
    u: torch.Tensor, v: torch.Tensor, grid: StructuredCGrid
) -> dict[str, torch.Tensor]:
    """``maskW``/``maskS`` as the ``mask_w``/``mask_s`` keywords of the corner-flux kernels."""
    return {
        "mask_w": grid.masks.u_float(dtype=u.dtype, device=u.device),
        "mask_s": grid.masks.v_float(dtype=v.dtype, device=v.device),
    }


def _corner_ghost_halos(
    q: torch.Tensor,
    grid: StructuredCGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corner field ``[..., ny+1, nx+1]`` with one more column, and one more row, on each side.

    The owned corners (without the repeated last column or row) are exchanged
    as a Z field and only the outer ghost column (row) is taken; the corners
    computed here are kept.  On a non-coupling axis the ghosts are zero and only
    the edge faces read them, which the caller zeroes.
    """
    nx, ny = grid.nx, grid.ny
    q_x_ghost = exchange(
        q[..., :nx], grid=grid, location="Z",
        extent=HaloExtentV1(
            west_left_extent=1, east_right_extent=2,
            south_bottom_extent=0, north_top_extent=0,
        ),
        fill=FillSpec.ZERO,
    )
    q_y_ghost = exchange(
        q[..., :ny, :], grid=grid, location="Z",
        extent=HaloExtentV1(
            west_left_extent=0, east_right_extent=0,
            south_bottom_extent=1, north_top_extent=2,
        ),
        fill=FillSpec.ZERO,
    )
    q_x = torch.cat((q_x_ghost[..., :1], q, q_x_ghost[..., -1:]), dim=-1)
    q_y = torch.cat((q_y_ghost[..., :1, :], q, q_y_ghost[..., -1:, :]), dim=-2)
    return q_x, q_y


def _wet_point_coriolis(
    u: torch.Tensor,
    v: torch.Tensor,
    f_g: torch.Tensor,
    grid: StructuredCGrid,
    *,
    metrics: MomentumMetricBundle,
    halo_kw_u: dict[str, torch.Tensor],
    halo_kw_v: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """selectCoriScheme=1 on a regional grid (mom_vi_coriolis.F:73-89, :140-156)."""
    if metrics.is_cube:
        # ponytail: the hFac halos below use the regional sliced frame; the cube
        # needs them in the rotated frame of exchange_hfac_pair (SPEC_coriolis 4).
        raise NotImplementedError(
            "select_cori_scheme=1 (wet-point average) is not implemented on the "
            "cubed sphere"
        )
    t = _face_transports(
        u, v, grid, metrics=metrics, halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
    )
    # hFac enters the numerator and the wet-fraction denominator; with a
    # differentiable geometry its halo keeps the gradient (as
    # viscosity_geometry.py does for the same hFac halos).
    hfac_s_ext = exchange(
        t.hfac_s, grid=grid, location="V", extent=_REGION_X_HALO,
        fill=FillSpec.ZERO, constant=not t.hfac_s.requires_grad,
    )
    hfac_w_ext = exchange(
        t.hfac_w, grid=grid, location="U", extent=_REGION_Y_HALO,
        fill=FillSpec.ZERO, constant=not t.hfac_w.requires_grad,
    )
    tend_u, tend_v = coriolis_wet_point_from_haloed_transports(
        t.u_ext, t.v_ext, f_g,
        hfac_w_ext=hfac_w_ext, hfac_s_ext=hfac_s_ext, dxc=t.dxc, dyc=t.dyc,
        mask_w=grid.masks.u_float(dtype=u.dtype, device=u.device),
        mask_s=grid.masks.v_float(dtype=v.dtype, device=v.device),
    )
    return _region_edge_pair_masks(tend_u, tend_v, grid)


def _hfac_weighted_coriolis(
    u: torch.Tensor,
    v: torch.Tensor,
    f_g: torch.Tensor,
    grid: StructuredCGrid,
    *,
    metrics: MomentumMetricBundle,
    halo_kw_u: dict[str, torch.Tensor],
    halo_kw_v: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """selectCoriScheme=2 (mom_vi_coriolis.F:90-104, :157-171)."""
    t = _face_transports(
        u, v, grid, metrics=metrics, halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
    )
    tend_u, tend_v = coriolis_hfac_weighted_from_haloed_transports(
        t.u_ext, t.v_ext, f_g,
        dxc=t.dxc, dyc=t.dyc, hfac_w=t.hfac_w, hfac_s=t.hfac_s,
    )
    if not metrics.is_cube:
        tend_u, tend_v = _region_edge_pair_masks(tend_u, tend_v, grid)
    return tend_u, tend_v


def _planetary_coriolis_for_scheme(
    cori_scheme: str,
    u: torch.Tensor,
    v: torch.Tensor,
    f_g: torch.Tensor,
    hFacZ: torch.Tensor,
    grid: StructuredCGrid,
    *,
    metrics: MomentumMetricBundle,
    halo_kw_u: dict[str, torch.Tensor],
    halo_kw_v: dict[str, torch.Tensor],
    uface_kw: dict[str, torch.Tensor],
    vface_kw: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """U/V planetary term of MOM_VI_CORIOLIS for one CORI_SCHEMES name."""
    if cori_scheme == "simple":
        plan_u = _u_face_vorticity_flux(
            v, f_g, grid, average="no_hfac", hFacZ=hFacZ,
            metrics=metrics, **uface_kw)
        plan_v = _v_face_vorticity_flux(
            u, f_g, grid, average="no_hfac", hFacZ=hFacZ,
            metrics=metrics, **vface_kw)
        return plan_u, plan_v
    elif cori_scheme == "wet_point":
        return _wet_point_coriolis(
            u, v, f_g, grid, metrics=metrics,
            halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
        )
    elif cori_scheme == "hfac_weighted":
        return _hfac_weighted_coriolis(
            u, v, f_g, grid, metrics=metrics,
            halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
        )
    elif cori_scheme == "energy_conserving":
        # mom_vi_coriolis.F:105-120, :172-187: the selectVortScheme=4
        # expression with fCoriG as the corner value.
        return _energy_conserving_flux(
            u, v, f_g, grid, metrics=metrics,
            halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
        )


def _vectorize_vertshear() -> bool:
    """Whole-z-axis vectorization toggle for the vertical-shear k-loops.

    Default ON.  Set ``DEFGCM_VECTORIZE_VERTSHEAR=0`` to fall back to the original
    per-layer Python ``for k in range(nz)`` loops (kept verbatim below so the
    OFF path is bit-identical to the pre-optimization baseline).  Both paths are
    bitwise-equivalent; the toggle exists for ablation and as a safety fallback.  The enable
    decision is owned by the unified control plane (``defgcm.accel``).
    """
    from defgcm.accel import gate

    return gate("vectorize_vertshear").effective_value



_VERTSHEAR_EXT_U = HaloExtentV1(
    west_left_extent=1,
    east_right_extent=1,
    south_bottom_extent=0,
    north_top_extent=0,
)
_VERTSHEAR_EXT_V = HaloExtentV1(
    west_left_extent=0,
    east_right_extent=0,
    south_bottom_extent=1,
    north_top_extent=1,
)


def vertical_shear_tendency_for_grid(
    u: torch.Tensor,
    v: torch.Tensor,
    w_interface: torch.Tensor,
    grid: StructuredCGrid,
    *,
    select_ke_scheme: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vector-invariant vertical shear ``-w·∂u/∂z`` / ``-w·∂v/∂z`` for regional or cube grids.

    MITgcm ``mom_vi_u_vertshear.F`` / ``mom_vi_v_vertshear.F`` with
    ``upwindShear=.FALSE.``.  ``rAdvAreaWeight`` selects the area-weighted
    (``.TRUE.``, selectKEscheme 0/2) vs simple-average (``.FALSE.``, schemes
    1/3) face stencil.

    ``w_interface`` must use MIT upward-positive convention (``[batch, nz+1, ...]``
    region / ``[6, nz+1, ...]`` cube). This is the convention stored by
    ``_cached_w_interface`` in the implicit free-surface stepper and by cube
    harness MIT-aligned injections.  The ``rkSign = -1`` convention lives
    inside the shear kernels (the unary minus on the vertical deltas), NOT
    here — do not negate ``w``.

    Halo-fills ``w`` (and ``rA`` / ``maskC`` in the area-weighted path) via
    ``HaloExchange.center(..., REPLICATE)`` so the U-face (x±1) / V-face (y±1)
    extended frame carries the neighbour cell on EVERY geometry:

    - region periodic axis: REPLICATE on a PERIODIC boundary wraps (exchange.py
      routes PERIODIC -> WRAP), so the extended-frame average == the old explicit
      ``periodic_xm/ym`` wrap byte-for-byte (the explicit branches disappear);
    - region closed axis: the outer faces are produced from a REPLICATE
      edge-ghost but then ZEROED by ``metrics.u_face_keep`` / ``v_face_keep``,
      matching the old interior-only ``wbar[..., 1:nx]`` carve;
    - cube: the underlying primitive does the cross-face scalar exchange, so all
      faces are interior (``u_face_keep`` / ``v_face_keep`` are all ones).

    Geometry- AND decomposition-neutral: single-card region, distributed region
    (multi-rank x/y/2-D split), and cube all route through the same kernels
    (narrow-waist metric bundle + ``HaloExchange``).  The
    metric bundle builds the seam U/V-face area reciprocals from the haloed
    neighbour area for a distributed rank (``build_vertshear_metric_bundle``),
    and ``HaloExchange.center`` self-dispatches the cross-rank P2P seam exchange,
    so the distributed seam tendency is byte-identical to the single-card global
    result (verified 2-rank gloo, x/y/2-D split × periodic/closed, max|d|=0).

    Returns ``(shear_u, shear_v)`` to add to the momentum tendency.
    """
    ke_scheme = _scheme_name(KE_SCHEMES, select_ke_scheme, "select_ke_scheme")
    metrics = build_vertshear_metric_bundle(grid, u.dtype, u.device)
    halo = HaloExchange(grid)
    w = w_interface
    if w.ndim == 3:
        w = w.unsqueeze(0)

    # Halo-fill the centred fields once; REPLICATE self-dispatches per geometry
    # (region per-axis fill incl. PERIODIC->WRAP, cube cross-face exchange).
    maskC = metrics.maskC
    w_ext_u = halo.center(
        w, location="U", extent=_VERTSHEAR_EXT_U, fill=FillSpec.REPLICATE
    )
    w_ext_v = halo.center(
        w, location="V", extent=_VERTSHEAR_EXT_V, fill=FillSpec.REPLICATE
    )
    maskC_ext_u = (
        halo.center(
            maskC, location="U", extent=_VERTSHEAR_EXT_U, fill=FillSpec.REPLICATE
        )
        if maskC is not None
        else None
    )
    maskC_ext_v = (
        halo.center(
            maskC, location="V", extent=_VERTSHEAR_EXT_V, fill=FillSpec.REPLICATE
        )
        if maskC is not None
        else None
    )

    # MIT rAdvAreaWeight: area-weighted w-bars unless the KE scheme is already
    # area weighted (mom_vi_u_vertshear.F:44-47).
    if ke_scheme in ("unweighted", "hfac_weighted"):
        rA = metrics.rA
        rA_ext_u = halo.center(
            rA, location="U", extent=_VERTSHEAR_EXT_U, fill=FillSpec.REPLICATE, constant=True
        )
        rA_ext_v = halo.center(
            rA, location="V", extent=_VERTSHEAR_EXT_V, fill=FillSpec.REPLICATE, constant=True
        )
        wbar_xm, wbar_xp, wbar_ym, wbar_yp = vertical_shear_wbars_area_weighted(
            w_ext_u,
            w_ext_v,
            maskC_ext_u,
            maskC_ext_v,
            rA_ext_u,
            rA_ext_v,
            u,
            v,
            recip_rAw=metrics.recip_rAw,
            recip_rAs=metrics.recip_rAs,
            u_face_keep=metrics.u_face_keep,
            v_face_keep=metrics.v_face_keep,
        )
    elif ke_scheme in ("area_weighted", "area_hfac_weighted"):
        wbar_xm, wbar_xp, wbar_ym, wbar_yp = vertical_shear_wbars_simple(
            w_ext_u,
            w_ext_v,
            maskC_ext_u,
            maskC_ext_v,
            u,
            v,
            u_face_keep=metrics.u_face_keep,
            v_face_keep=metrics.v_face_keep,
        )

    if not _vectorize_vertshear():
        return vertical_shear_per_layer(
            u,
            v,
            wbar_xm,
            wbar_xp,
            wbar_ym,
            wbar_yp,
            metrics.recip_drF,
            recip_hFacW=metrics.recip_hFacW,
            recip_hFacS=metrics.recip_hFacS,
        )
    tend_u, tend_v = vertical_shear_all_z(
        u,
        v,
        wbar_xm,
        wbar_xp,
        wbar_ym,
        wbar_yp,
        metrics.recip_drF,
        recip_hFacW=metrics.recip_hFacW,
        recip_hFacS=metrics.recip_hFacS,
    )
    # The whole-z vectorized kernel ran (the per-layer loop returned
    # earlier), so the hit is recorded here, not at the gate predicate.
    from defgcm.accel import current_rank, record_dispatch

    record_dispatch(
        "vectorize_vertshear",
        outcome="hit",
        variant="whole_z_axis",
        site="vector_invariant_momentum.vertical_shear_tendency_for_grid",
        backend="torch",
        rank=current_rank(),
    )
    return tend_u, tend_v


def _latitude_center_cos(grid: StructuredCGrid, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Infer cos(yC) from spherical-polar ``dxC = rSphere*cos(yC)*dlon`` rows."""
    if grid.metrics.del_x is not None or grid.metrics.del_y is not None:
        return torch.ones(grid.ny, dtype=dtype, device=device)
    dx_c = grid.metrics.dxC_for(dtype, device)[:, 0]
    dy = grid.metrics.dyC_for(dtype, device)[0, 0]
    return latitude_center_cos_from_metrics(dx_c, dy, dtype=dtype)


def _u_coriolis_nh_runtime_f_cori_cos(
    grid: StructuredCGrid,
    configured: float | None,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    ny: int,
    omega: float,
) -> torch.Tensor:
    """Return MIT's runtime ``fCoriCos`` rows for the scheme-1 kernels.

    ``configured`` is the resolved ``fCoriCos`` of a constant-F/Beta map
    (``EvaluationInputs.f_cori_cos``); ``None`` uses ``2*omega*cos(lat)``.
    """
    if configured is None:
        return f_cori_cos_rows(
            _latitude_center_cos(grid, dtype, device),
            ny=ny,
            omega=omega,
        )
    value = torch.as_tensor(configured, dtype=dtype, device=device)
    if value.numel() != 1:
        raise ValueError(
            "f_cori_cos must be a scalar for constant-F/Beta maps, "
            f"got shape {tuple(value.shape)}"
        )
    return value.reshape(1, 1, 1).expand(1, ny, 1)


def _u_coriolis_nh_scheme1(
    w_interface: torch.Tensor,
    grid: StructuredCGrid,
    u_ref: torch.Tensor,
    *,
    f_cori_cos: float | None,
    omega: float,
) -> torch.Tensor:
    """MOM_U_CORIOLIS_NH scheme 1.

    ``w_interface`` is MIT upward-positive with shape ``[batch,nz+1,ny,nx]``.
    The non-rotated SCS path has angleCosC=1, so only U receives the
    z-coordinate ``gravitySign*fCoriCos*wMean = -fCoriCos*wMean`` term; the
    curvilinear/rotated V branch is intentionally outside this helper.
    """
    dtype = u_ref.dtype
    device = u_ref.device
    batch, nz, ny, nx_plus_1 = u_ref.shape
    nx = nx_plus_1 - 1
    w = w_interface.to(dtype=dtype, device=device)
    if w.ndim == 3:
        w = w.unsqueeze(0)
    if w.shape != (batch, nz + 1, ny, nx):
        raise ValueError(
            "w_interface must be MIT upward-positive [batch,nz+1,ny,nx], "
            f"got {tuple(w.shape)}"
        )

    f_cori_cos = _u_coriolis_nh_runtime_f_cori_cos(
        grid,
        f_cori_cos,
        dtype=dtype,
        device=device,
        ny=ny,
        omega=omega,
    )
    # expects MIT upward-positive w_interface; the tensor kernel below preserves
    # the MIT gravitySign=-1 sign in the returned U-face tendency.
    w_ext = exchange(
        w,
        grid=grid,
        location="W",
        extent=HaloExtentV1(
            west_left_extent=1,
            east_right_extent=1,
            south_bottom_extent=0,
            north_top_extent=0,
        ),
        fill=FillSpec.ZERO,
    )
    return u_coriolis_nh_from_haloed_w(
        w_ext,
        u_ref,
        f_cori_cos=f_cori_cos,
        periodic_x=grid.boundary_topology.periodic_x,
        west_distributed=_has_distributed_neighbor(grid, "west"),
        east_distributed=_has_distributed_neighbor(grid, "east"),
    )


@dataclass(frozen=True)
@register_dynamics_module("vector_invariant_momentum_advection")
class VectorInvariantMomentumAdvection(TendencyModule):
    """Vector-invariant momentum advection with merged Coriolis.

    Decomposes the momentum advection into:
      tendency_u = -∂KE/∂x + (ζ+f)·v̄   (at u-faces)
      tendency_v = -∂KE/∂y - (ζ+f)·ū   (at v-faces)

    When this module is active, do NOT add a separate FCoriolis module,
    since Coriolis is already merged into the (ζ+f) vorticity flux.

    Parameters
    ----------
    coriolis_parameter : float
        Constant Coriolis parameter f [s⁻¹] (f-plane approximation).
        Ignored if ``coriolis_profile_g`` is provided.
    coriolis_profile_g : tuple[float, ...] | None
        Optional 1-D Coriolis profile evaluated at vorticity y-points
        (``yG``), length ``ny+1``. When provided, the 2-D fCoriG field is
        constructed by broadcasting this profile along the x-axis, matching
        MITgcm's ``selectCoriMap=2`` (spherical polar) behaviour where
        ``fCoriG(i,j) = 2·Ω·sin(yG(j))``.  The profile is stored as a tuple
        to keep the dataclass hashable.
    select_vort_scheme : int
        MITgcm ``selectVortScheme`` (``VORT_SCHEMES``).  Scheme 1 uses
        modified-hFac velocity averaging; schemes 0, 2, 3 and 4 use face
        transports, 0, 2 and 3 with ``r_hFacZ`` and on regional grids only.
    vort_reconstruction : str
        ``VORT_RECONSTRUCTIONS``: ``"centered"`` (MITgcm default),
        ``"high_order"`` (``highOrderVorticity``) or ``"upwind"``
        (``upwindVorticity``); the last two only with schemes 0 and 2, as
        ``mom_vi_u/v_coriolis_c4.F:205-212`` stops otherwise, and
        ``"high_order"`` not with open boundaries.  The grid conditions are
        checked at model assembly (``validate_vorticity_composition``).
    select_ke_scheme : int
        Kinetic-energy discretization scheme. MITgcm schemes 0, 1, 2, and 3
        are implemented; schemes 1 and 3 also select the simple-average
        vertical-shear branch.
    upwind_shear : bool
        MITgcm ``upwindShear`` for the explicit vertical shear; only the
        centred form (``False``) is implemented.
    rotation_period : float
        MITgcm ``rotationPeriod`` in seconds (default 86164); ``omega`` follows
        ``ini_parms.F:487-490``.  It sets ``fCoriG`` on the cube and the
        latitude form of ``fCoriCos``; regional ``fCoriG`` comes from
        ``coriolis_profile_g`` or ``coriolis_parameter``.
    select_cori_scheme : int
        MITgcm ``selectCoriScheme`` for the planetary term on the split path
        (``split_coriolis=True``, MIT ``useAbsVorticity=.FALSE.``); the
        combined path does not read it, as in ``mom_vecinv.F:680-682``.

    Scheme values outside ``VORT_SCHEMES``, ``KE_SCHEMES``,
    ``CORI_3D_SCHEMES``, ``SHEAR_SCHEMES`` and ``CORI_SCHEMES`` are refused at
    construction.
    """

    coriolis_parameter: float = 0.0
    coriolis_profile_g: tuple[float, ...] | None = None
    select_vort_scheme: int = 1
    vort_reconstruction: str = "centered"
    select_ke_scheme: int = 0
    split_coriolis: bool = True
    enable_vertical_shear: bool = True
    select_3d_cori_scheme: int = 0
    upwind_shear: bool = False
    select_cori_scheme: int = 0
    rotation_period: float = ROTATION_PERIOD_EARTH
    name: str = "vector_invariant_momentum"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    # Topology capability (defgcm/grid_support.py).
    # Cube claim backed by a bitwise registry-injection check.
    grid_support = ("latlon", "cube")
    # Compiled execution constraint: ``coriolis_profile_g`` resolved to a tensor,
    # keyed by ``(dtype, device)``.  The memo lives on the module that OWNS the
    # profile, so two modules with different profiles cannot share an entry --
    # which is what the ``id()`` key in the traced ``constant_cache`` lookup was
    # buying at the cost of a graph break every step.
    #
    # 🔴 Contract: ``coriolis_profile_g`` is treated as immutable after
    # construction (the dataclass is ``frozen=True``, so rebinding it needs
    # ``object.__setattr__``).  Rebinding it without calling
    # :meth:`prepare_host_constants` again leaves this memo serving the previous
    # profile's tensor.  That is a documented contract break, not an observed
    # defect: nothing in the repository rebinds the field.
    _coriolis_profile_memo: dict[tuple[torch.dtype, str], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        vort_scheme = _scheme_name(
            VORT_SCHEMES, self.select_vort_scheme, "select_vort_scheme")
        if self.vort_reconstruction not in VORT_RECONSTRUCTIONS:
            raise ValueError(
                f"vort_reconstruction={self.vort_reconstruction!r} is not one of "
                f"{VORT_RECONSTRUCTIONS}"
            )
        if self.vort_reconstruction != "centered" and vort_scheme not in ("enstrophy", "energy"):
            raise NotImplementedError(
                f"vort_reconstruction={self.vort_reconstruction!r} needs select_vort_scheme "
                f"0 or 2 (MOM_VI_U/V_CORIOLIS_C4), got {self.select_vort_scheme}"
            )
        _scheme_name(KE_SCHEMES, self.select_ke_scheme, "select_ke_scheme")
        _scheme_name(
            CORI_3D_SCHEMES, self.select_3d_cori_scheme, "select_3d_cori_scheme")
        _scheme_name(SHEAR_SCHEMES, self.upwind_shear, "upwind_shear")
        _scheme_name(CORI_SCHEMES, self.select_cori_scheme, "select_cori_scheme")

    def prepare_host_constants(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Eager pre-step: resolve ``coriolis_profile_g`` into the memo.

        Called by ``ImplicitFreeSurfaceStepper.prepare_step_host``.  Idempotent:
        a warm entry is left alone, so the tensor the traced step reads is the
        one the first eager step built.
        """
        self._coriolis_profile_tensor(grid, dtype=dtype, device=device)

    def _coriolis_profile_tensor(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        """The memoised profile tensor, or ``None`` when there is no profile.

        A warm hit is a plain dict lookup on ``(dtype, device)`` -- both of them
        constants for Dynamo -- so it traces.  A miss falls through to
        :func:`resolve_coriolis_profile_tensor`, which is the eager legacy path.
        """
        profile = self.coriolis_profile_g
        if profile is None:
            return None
        key = (dtype, _cache_device_key(device))
        cached = self._coriolis_profile_memo.get(key)
        if cached is None:
            cached = resolve_coriolis_profile_tensor(
                grid, profile, dtype=dtype, device=device
            )
            self._coriolis_profile_memo[key] = cached
        return cached

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        u = state.u.data
        v = state.v.data

        vort_scheme = _scheme_name(
            VORT_SCHEMES, self.select_vort_scheme, "select_vort_scheme")
        ke_scheme = _scheme_name(KE_SCHEMES, self.select_ke_scheme, "select_ke_scheme")
        cori_scheme = _scheme_name(
            CORI_SCHEMES, self.select_cori_scheme, "select_cori_scheme")
        omega = omega_from_rotation_period(self.rotation_period)
        inputs = diagnostics.inputs
        select_3d_cori_scheme = int(resolved_input(
            inputs, "select_3d_cori_scheme", self.select_3d_cori_scheme))
        cori_3d_scheme = _scheme_name(
            CORI_3D_SCHEMES, select_3d_cori_scheme, "select_3d_cori_scheme")

        # ---- Topology pre-exchange: optional paired (u,v) exchange for helpers ----
        halo_kw_u, halo_kw_v = vector_invariant_halo_kwargs_for_grid(u, v, grid)

        # Narrow-waist momentum metrics: region reads the bundle's per-row
        # columns; cube reaches the helpers via halo_kw (v_ext set) so the
        # bundle's region fields are never consulted (wires cube).
        mom_metrics = build_momentum_metric_bundle(grid, u.dtype, u.device)

        # ---- Fused horizontal-phase accelerator boundary (vim_fused_cuda) ----
        # One CUDA launch replaces these stages below (curl,
        # hFacZ corner mask, both KE gradients, both Coriolis faces, both
        # vorticity-flux faces, the three-term combine).  The dispatcher
        # returns None whenever it cannot reproduce this configuration
        # bitwise, and the frozen eager block underneath is then the single
        # fallback implementation.  The tail after this block (vertical shear,
        # NH Coriolis, dry-face masks, StateTendency packing) is shared.
        fused = None
        from defgcm.accel import (
            consumer_snapshot,
            in_accelerated_dispatch,
            requested,
        )

        if not in_accelerated_dispatch():
            with consumer_snapshot():
                if requested(("vim_fused_cuda",)):
                    from ._vim_fused import dispatcher as _vim_fused_dispatcher

                    fused = _vim_fused_dispatcher.vim_fused_dispatch(
                        u,
                        v,
                        grid,
                        ke_scheme=ke_scheme,
                        vort_scheme=vort_scheme,
                        split_coriolis=self.split_coriolis,
                        cori_scheme=cori_scheme,
                        coriolis_parameter=self.coriolis_parameter,
                        coriolis_profile_g=self.coriolis_profile_g,
                        omega=omega,
                        # The fused adapter must read the
                        # SAME prepared tensor the eager block reads, or its
                        # ``corner_coriolis_field`` rebuilds the ``id(profile)``

                        # undeclared break).
                        coriolis_profile_tensor=self._coriolis_profile_tensor(
                            grid, dtype=u.dtype, device=u.device
                        ),
                        metrics=mom_metrics,
                    )

        if fused is not None:
            tend_u, tend_v = fused
        else:
            _vim_trace("compute_ke_center begin")
            ke = _compute_ke_center(u, v, grid, select_ke_scheme=self.select_ke_scheme)
            _vim_trace("compute_ke_center done")

            _vim_trace("ke_gradient_u begin")
            dKEdx = _ke_gradient_u(ke, grid)
            _vim_trace("ke_gradient_u done")
            _vim_trace("ke_gradient_v begin")
            dKEdy = _ke_gradient_v(ke, grid)
            _vim_trace("ke_gradient_v done")

            # Build the corner Coriolis field.
            f_g_2d = corner_coriolis_field_for_grid(
                u,
                grid,
                coriolis_parameter=self.coriolis_parameter,
                coriolis_profile_g=self.coriolis_profile_g,
                omega=omega,
                profile_tensor=self._coriolis_profile_tensor(
                    grid, dtype=u.dtype, device=u.device
                ),
            )

            _vim_trace("compute_hFacZ begin")
            hFacZ = _compute_hFacZ(grid, dtype=u.dtype, device=u.device)
            _vim_trace("compute_hFacZ done")
            _vim_trace("relative_vorticity begin")
            zeta = relative_vorticity(u, v, grid)
            _vim_trace("relative_vorticity done")

            # Region path (halo kwargs empty -> every helper would halo u/v itself):
            # do each exchange once and pass it to both consumers of that face pair.
            # The cube path keeps its own pre-haloed v_ext/u_ext untouched.
            uface_kw = halo_kw_v if halo_kw_v else {"v_x_halo": _region_v_x_halo(v, grid)}
            vface_kw = halo_kw_u if halo_kw_u else {"u_y_halo": _region_u_y_halo(u, grid)}

            if self.split_coriolis:
                _vim_trace("split coriolis begin")
                zeta_masked = _mask_corner_value_by_hfac(zeta, hFacZ)
                plan_u, plan_v = _planetary_coriolis_for_scheme(
                    cori_scheme, u, v, f_g_2d, hFacZ, grid,
                    metrics=mom_metrics, halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
                    uface_kw=uface_kw, vface_kw=vface_kw,
                )
                zeta_flux_u, zeta_flux_v = _vorticity_flux_for_scheme(
                    vort_scheme, self.vort_reconstruction, u, v, zeta_masked, hFacZ, grid,
                    metrics=mom_metrics, halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
                    uface_kw=uface_kw, vface_kw=vface_kw,
                )
                tend_u, tend_v = _combine_split_coriolis_tendencies(
                    dKEdx,
                    dKEdy,
                    plan_u,
                    plan_v,
                    zeta_flux_u,
                    zeta_flux_v,
                )
                _vim_trace("split coriolis done")
            else:
                _vim_trace("combined vorticity begin")
                # MOM_VECINV masks relative vorticity before MOM_CALC_ABSVORT3;
                # the planetary part remains present at dry corners.
                omega3 = _absolute_vorticity(
                    _mask_corner_value_by_hfac(zeta, hFacZ), f_g_2d
                )
                vort_flux_u, vort_flux_v = _vorticity_flux_for_scheme(
                    vort_scheme, self.vort_reconstruction, u, v, omega3, hFacZ, grid,
                    metrics=mom_metrics, halo_kw_u=halo_kw_u, halo_kw_v=halo_kw_v,
                    uface_kw=uface_kw, vface_kw=vface_kw,
                )
                tend_u, tend_v = _combine_vorticity_tendencies(
                    dKEdx,
                    dKEdy,
                    vort_flux_u,
                    vort_flux_v,
                )
                _vim_trace("combined vorticity done")

        if self.enable_vertical_shear and grid.nz > 1:
            w_interface = None if inputs is None else inputs.w_interface
            if w_interface is None:
                # the explicit-input requirement: the term is configured, so a missing w is
                # an unwired input, not an absent term.  The regional and cube
                # steppers set the cache before the tendencies of every step,
                # the regional one on a fresh run's first step from the initial
                # u, v (INITIALISE_VARIA's INTEGR_CONTINUITY).
                raise RuntimeError(
                    "enable_vertical_shear needs the MIT upward-positive w in "
                    "EvaluationInputs.w_interface (MOM_VI_U/V_VERTSHEAR); none was given"
                )
            _vim_trace("vertical shear begin")
            shear_scheme = _scheme_name(SHEAR_SCHEMES, self.upwind_shear, "upwind_shear")
            if shear_scheme == "centered":
                shear_u, shear_v = vertical_shear_tendency_for_grid(
                    u,
                    v,
                    w_interface,
                    grid,
                    select_ke_scheme=self.select_ke_scheme,
                )
            tend_u, tend_v = _add_uv_tendency_pair(
                tend_u,
                tend_v,
                shear_u,
                shear_v,
            )
            _vim_trace("vertical shear done")

        if cori_3d_scheme == "original":
            w_interface = None if inputs is None else inputs.w_interface
            if w_interface is None:
                # the explicit-input requirement: the term is configured, so a missing w is
                # an unwired input, not an absent term.
                raise RuntimeError(
                    f"select_3d_cori_scheme={select_3d_cori_scheme} needs the "
                    "MIT upward-positive w in EvaluationInputs.w_interface "
                    "(MOM_U_CORIOLIS_NH); none was given"
                )
            _vim_trace("nh coriolis scheme1 begin")
            tend_u = tend_u + _u_coriolis_nh_scheme1(
                w_interface, grid, u, f_cori_cos=inputs.f_cori_cos, omega=omega)
            _vim_trace("nh coriolis scheme1 done")

        # Zero tendency on dry faces.
        if getattr(grid, "face_count", 1) == 6 or grid.masks.has_3d_masks:
            _vim_trace("apply uv face masks begin")
            tend_u, tend_v = apply_uv_tendency_masks_for_grid(tend_u, tend_v, grid)
            _vim_trace("apply uv face masks done")

        _vim_trace("evaluate done")
        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=tend_u,
            v=tend_v,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )


def validate_vorticity_composition(*, modules, grid: StructuredCGrid) -> None:
    """Refuse at assembly the vorticity schemes this grid has no port for.

    selectVortScheme 0, 2 and 3 and their C4 forms are regional: the cube needs
    the corner masks of mask_vector_tendency_for_grid, cross-face corner halos
    and the face-corner rule of mom_vi_u_coriolis_c4.F:74-116.  The
    fourth-order form multiplies its differences by ``maskInW``/``maskInS``
    under ALLOW_OBCS (mom_vi_u/v_coriolis_c4.F:141-147, :182-188), zero on and
    outside an open boundary (obcs_init_fixed.F); LOCOMgcm's strip masks are not
    checked against those positions, so an open axis is refused.  Without an
    open boundary the factor is zero only next to a dry column, where both
    corner values are already zero, and the kernel leaves it out.
    """
    topology = grid.boundary_topology
    for module in modules:
        if not isinstance(module, VectorInvariantMomentumAdvection):
            continue
        vort_scheme = _scheme_name(VORT_SCHEMES, module.select_vort_scheme, "select_vort_scheme")
        if getattr(grid, "face_count", 1) == 6 and vort_scheme in (
            "enstrophy", "energy", "energy_enstrophy",
        ):
            raise NotImplementedError(
                f"select_vort_scheme={module.select_vort_scheme} is not implemented "
                "on the cubed sphere"
            )
        if module.vort_reconstruction == "high_order" and (topology.open_x or topology.open_y):
            raise NotImplementedError(
                "vort_reconstruction='high_order' (MITgcm highOrderVorticity) is not "
                "implemented with an open boundary (grid boundary_topology open_x="
                f"{topology.open_x}, open_y={topology.open_y}): the maskInW/maskInS "
                "factor of mom_vi_u/v_coriolis_c4.F is not ported"
            )
