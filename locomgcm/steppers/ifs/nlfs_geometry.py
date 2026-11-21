"""Regional nonlinear surface-geometry installation and next-layer production.

hFac values are dimensionless C/U/V wet thickness fractions. eta_h is
upward-positive in m; CALC_SURF_DR keeps its derivative, as MITgcm's adjoint
does (initialise_varia.F:314-318 adds the surface update to hfac*_ad).
UPDATE_SURF_DR changes only the surface grid layer, retaining tensor dtype.
The r* coordinate (select_rStar > 0) instead rescales every level by the
column factors of ``r_star.py``: this module takes their halos from the grid
topology and installs ``h0Fac*rStarFac`` (UPDATE_R_STAR).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from locomgcm.steppers.ifs import r_star
from locomgcm.steppers.implicit_free_surface_nonlin import (
    SurfaceEtaBoundary, compute_surface_hfac,
    recip_hfac_new_freesurf, rescale_tracer_tendency_freesurf,
)


@dataclass(frozen=True)
class SurfaceGeometry:
    resting: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    surface: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def apply_surf_dr(grid, *, resting, surface) -> None:
    """MITgcm ``UPDATE_SURF_DR(.TRUE.)`` -- ``_hFac* := hFac_surf*``.

    Called at the ``forward_step.F:852`` point (MITgcm 27256b2d): after THERMODYNAMICS and
    DYNAMICS have run on the lagged geometry, before SOLVE_FOR_PRESSURE and
    INTEGR_CONTINUITY, which see ``h(eta^n)``.

    With autograd enabled the surface level is installed by tensor
    replacement (``MaskSet.replace_surface_hfac``): an operation of this
    step may already have saved the current hFac for backward.  The test is
    grad mode itself, as for ``carry._carry``, whatever the controls are.
    The grid's ``hfac_*_for`` entries are then rebuilt from the new tensors
    (``StructuredCGrid.refresh_hfac_runtime``).
    """
    masks = grid.masks
    hc, hw, hs = surface
    # Producer edge for the hFac-derived per-grid memos.  The three
    # writes below leave ``id(grid)`` and ``id(masks)`` untouched, so every
    # cache keyed on grid identity that holds an hFac-derived tensor has no
    # way to notice them.  Declare it here, on the mutation site itself, and
    # BEFORE the writes, so no consumer can memoize in between.  The
    # resting geometry goes with it (MITgcm h0FacC/W/S, SURFACE.h).
    masks.mark_hfac_time_varying(resting)
    if torch.is_grad_enabled():
        masks.replace_surface_hfac(surface)
        grid.refresh_hfac_runtime()
    else:
        masks.hFacC[0].copy_(hc[0].to(device=masks.hFacC.device))
        masks.hFacW[0].copy_(hw[0].to(device=masks.hFacW.device))
        masks.hFacS[0].copy_(hs[0].to(device=masks.hFacS.device))

def refresh_hfac_surf(
    grid: StructuredCGrid, eta_h_new: torch.Tensor, *, resting, hfac_inf,
    obc_eta: torch.Tensor | SurfaceEtaBoundary,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MITgcm ``CALC_SURF_DR(etaH)`` at ``forward_step.F:959`` (MITgcm 27256b2d).

    Stores ``hFac_surf* = h(eta^{n+1})`` for the next step to consume.
    """
    hc, hw, hs = compute_surface_hfac(
        grid,
        eta_h_new,
        *resting,
        hFacInf=hfac_inf,
        # OBCS_APPLY_SURF_DR consumes initial eta edges for myIter=-1,
        # and prescribed side samples after the initial UPDATE_ETAH.
        obc_eta=obc_eta,
    )
    return (
        hc.unsqueeze(0), hw.unsqueeze(0), hs.unsqueeze(0),
    )


@dataclass(frozen=True)
class RStarColumns:
    """Static r* column quantities, read by CALC_R_STAR and by the continuity rate.

    ``resting`` is ``(h0FacC, h0FacW, h0FacS)``; ``r_low_c``/``recip_depth_c``
    are ``R_low``/``recip_Rcol`` ``[ny, nx]``; ``depth_w``/``depth_s`` are
    ``rSurfW-rLowW``/``rSurfS-rLowS`` on the ``[ny, nx+1]``/``[ny+1, nx]``
    faces; ``wet_*`` is ``kSurf <= Nr``; ``area_x``/``area_y`` are ``rA`` with
    one topology halo column/row on each side and ``recip_area_w/s`` are
    ``recip_rAw``/``recip_rAs``; ``area_weight`` is ``rStarAreaWeight``
    (``calc_r_star.F:64-68``).
    """

    resting: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    r_low_c: torch.Tensor
    recip_depth_c: torch.Tensor
    depth_w: torch.Tensor
    depth_s: torch.Tensor
    wet_c: torch.Tensor
    wet_w: torch.Tensor
    wet_s: torch.Tensor
    area_x: torch.Tensor
    area_y: torch.Tensor
    recip_area_w: torch.Tensor
    recip_area_s: torch.Tensor
    area_weight: bool


@dataclass(frozen=True)
class RStarGeometry:
    """The stepper's r* state: static columns and the factors of the latest CALC_R_STAR.

    ``factors is None`` is a fresh run, whose first step runs the
    INITIALISE_VARIA calls (``initialise_varia.F:301-307, :338-341``).
    """

    columns: RStarColumns | None = None
    factors: r_star.RStarFactors | None = None


def _halo_x(grid, field):
    return exchange(
        field, grid=grid, location="T", extent=HaloExtentV1(1, 1, 0, 0),
        fill=FillSpec.WRAP if grid.edge_face_couples("x") else FillSpec.REPLICATE,
    )


def _halo_y(grid, field):
    return exchange(
        field, grid=grid, location="T", extent=HaloExtentV1(0, 0, 1, 1),
        fill=FillSpec.WRAP if grid.edge_face_couples("y") else FillSpec.REPLICATE,
    )


def _positive_reciprocal(value):
    positive = value > 0
    return torch.where(
        positive, 1.0 / torch.where(positive, value, torch.ones_like(value)),
        torch.zeros_like(value),
    )


def r_star_columns(
    grid: StructuredCGrid, resting, *, area_weight: bool, dtype, device,
) -> RStarColumns:
    """``ini_masks_etc.F:136-140, :221-231, :332-372`` on the resting geometry.

    The face column depths read ``R_low`` of the neighbour across the face,
    taken by the grid's topology (numerics rule 11) as ``compute_surface_hfac``
    takes etaH; at a wall the face is dry (``h0Fac = 0``) and keeps factor 1.
    """
    h0c, h0w, h0s = (value.to(dtype=dtype, device=device) for value in resting)
    dr_f = grid.layer_thickness_for(dtype, device)
    r_low_c, recip_depth_c = r_star.column_low_and_recip_depth(h0c, dr_f)
    r_low_x = _halo_x(grid, r_low_c)
    r_low_y = _halo_y(grid, r_low_c)
    area = grid.metrics.cell_area_for(dtype, device)
    return RStarColumns(
        resting=(h0c, h0w, h0s),
        r_low_c=r_low_c,
        recip_depth_c=recip_depth_c,
        depth_w=r_star.face_column_depth(r_low_x[:, :-1], r_low_x[:, 1:], h0w, dr_f),
        depth_s=r_star.face_column_depth(r_low_y[:-1, :], r_low_y[1:, :], h0s, dr_f),
        wet_c=(h0c > 0).any(dim=0),
        wet_w=(h0w > 0).any(dim=0),
        wet_s=(h0s > 0).any(dim=0),
        area_x=_halo_x(grid, area),
        area_y=_halo_y(grid, area),
        recip_area_w=_positive_reciprocal(grid.metrics.rAw_for(dtype, device)),
        recip_area_s=_positive_reciprocal(grid.metrics.rAs_for(dtype, device)),
        area_weight=area_weight,
    )


def calc_r_star_factors(
    grid: StructuredCGrid, columns: RStarColumns, eta_h: torch.Tensor, *,
    previous: r_star.RStarFactors, dt_free_surf: float,
) -> r_star.RStarFactors:
    """MITgcm ``CALC_R_STAR(etaH)``; ``eta_h`` is ``[ny, nx]`` or ``[1, ny, nx]``."""
    eta = eta_h[0] if eta_h.ndim == 3 else eta_h
    eta_x = _halo_x(grid, eta)
    eta_y = _halo_y(grid, eta)
    if columns.area_weight:
        mean_w = r_star.area_weighted_face_mean(
            eta_x[:, :-1], eta_x[:, 1:], columns.area_x[:, :-1], columns.area_x[:, 1:],
            columns.recip_area_w)
        mean_s = r_star.area_weighted_face_mean(
            eta_y[:-1, :], eta_y[1:, :], columns.area_y[:-1, :], columns.area_y[1:, :],
            columns.recip_area_s)
    else:
        mean_w = r_star.simple_face_mean(eta_x[:, :-1], eta_x[:, 1:])
        mean_s = r_star.simple_face_mean(eta_y[:-1, :], eta_y[1:, :])
    return r_star.calc_r_star(
        eta=eta, mean_w=mean_w, mean_s=mean_s,
        r_low_c=columns.r_low_c, recip_depth_c=columns.recip_depth_c,
        depth_w=columns.depth_w, depth_s=columns.depth_s,
        wet_c=columns.wet_c, wet_w=columns.wet_w, wet_s=columns.wet_s,
        previous=previous, dt_free_surf=dt_free_surf,
    )


def initialise_r_star(
    grid: StructuredCGrid, columns: RStarColumns, eta_seed: torch.Tensor, *, dt_free_surf: float,
) -> RStarGeometry:
    """INITIALISE_VARIA's r* calls (``initialise_varia.F:301-307, :338-341``).

    CALC_R_STAR(etaH) over the INI_NLFS_VARS factors 1, UPDATE_R_STAR(.TRUE.),
    then, after the initial INTEGR_CONTINUITY, CALC_R_STAR(etaH) again on
    the unchanged etaH, which leaves ratios 1 and rates 0.
    """
    first = calc_r_star_factors(
        grid, columns, eta_seed, previous=r_star.resting_r_star_factors(
            columns.r_low_c, columns.depth_w, columns.depth_s),
        dt_free_surf=dt_free_surf,
    )
    install_r_star_hfac(grid, columns, first)
    return RStarGeometry(
        columns,
        calc_r_star_factors(
            grid, columns, eta_seed, previous=first, dt_free_surf=dt_free_surf,
        ),
    )


def install_r_star_hfac(grid, columns: RStarColumns, factors: r_star.RStarFactors) -> None:
    """MITgcm ``UPDATE_R_STAR(.TRUE.)``: every level ``hFac = h0Fac*rStarFac``.

    Same installation discipline as :func:`apply_surf_dr`: tensor replacement
    and a rebuild of the grid's ``hfac_*_for`` entries with autograd enabled,
    an in-place write without.
    """
    h0c, h0w, h0s = columns.resting
    grid.masks.install_r_star_hfac(
        (r_star.r_star_hfac(h0c, factors.fac_c), r_star.r_star_hfac(h0w, factors.fac_w),
         r_star.r_star_hfac(h0s, factors.fac_s)),
        resting=columns.resting,
    )
    if torch.is_grad_enabled():
        grid.refresh_hfac_runtime()


def surface_geometry_has_masks(masks) -> bool:
    return (
        masks.hFacC is not None
        and masks.hFacW is not None
        and masks.hFacS is not None
    )


def rescale_surface_tracers(tend_temp, tend_salt, *, hfac_c, surface_hfac_c) -> tuple:
    recip_hfac_new = recip_hfac_new_freesurf(
        hfac_c.to(dtype=tend_temp.dtype, device=tend_temp.device),
        surface_hfac_c,
    )
    # MITgcm FREESURF_RESCALE_G, called TWICE per level from
    # temp_integrate.F:411,433 -- once on the AB2-combined tendency
    # and once on the stored memory gtNm1, so the memory carries the
    # scaling into the next step.  (The momentum side rescales only
    # the combined tendency; the asymmetry is in the source.)
    hfac_c_now = hfac_c[0:1]
    tend_temp = rescale_tracer_tendency_freesurf(
        tend_temp, hFacC_now=hfac_c_now, hFac_surfC=surface_hfac_c,
    )
    tend_salt = rescale_tracer_tendency_freesurf(
        tend_salt, hFacC_now=hfac_c_now, hFac_surfC=surface_hfac_c,
    )
    return tend_temp, tend_salt, recip_hfac_new


def rescale_r_star_tracers(tend_temp, tend_salt, *, hfac_c, exp_c) -> tuple:
    """FREESURF_RESCALE_G and ``recip_hFacNew``, r* forms (thermodynamics.F:206-216)."""
    recip_hfac_c = _positive_reciprocal(hfac_c.to(dtype=tend_temp.dtype, device=tend_temp.device))
    return (
        r_star.r_star_rescale_tendency(tend_temp, exp_c),
        r_star.r_star_rescale_tendency(tend_salt, exp_c),
        r_star.r_star_recip_hfac_new(recip_hfac_c, exp_c),
    )


def surface_hfac(hfac):
    return hfac[0:1]
