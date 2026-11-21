"""Nonlinear free-surface hFac and tendency rescaling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.boundaries.open import BoundarySide
from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from locomgcm.geometry.grid import _side_is_open, top_layer_thickness_value


# ---------------------------------------------------------------------------
# Nonlinear free surface momentum rescaling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceEtaBoundary:
    """Upward-positive ETA samples on each side, retaining distinct corner values."""

    west: torch.Tensor
    east: torch.Tensor
    south: torch.Tensor
    north: torch.Tensor


def apply_surface_eta_boundary(
    eta: torch.Tensor, grid: StructuredCGrid, *, obc_eta: SurfaceEtaBoundary,
) -> torch.Tensor:
    """MIT OBCS_APPLY_ETA for surface-height history at its sampled clock.

    Height is upward-positive. A boundary cell is overwritten only when its
    interior normal face is wet, including distributed ownership. North/south
    then east/west follows MIT's corner assignment order.
    """
    if not grid.boundary_topology.has_open_boundaries:
        return eta
    result = eta.clone()
    wet_w = grid.masks.hFacW[0].to(device=eta.device) > 0
    wet_s = grid.masks.hFacS[0].to(device=eta.device) > 0
    if _side_is_open(BoundarySide.NORTH.value, grid):
        value = obc_eta.north.to(dtype=eta.dtype, device=eta.device)
        result[..., -1, :] = torch.where(wet_s[-2, :], value, result[..., -1, :])
    if _side_is_open(BoundarySide.SOUTH.value, grid):
        value = obc_eta.south.to(dtype=eta.dtype, device=eta.device)
        result[..., 0, :] = torch.where(wet_s[1, :], value, result[..., 0, :])
    if _side_is_open(BoundarySide.EAST.value, grid):
        value = obc_eta.east.to(dtype=eta.dtype, device=eta.device)
        result[..., -1] = torch.where(wet_w[:, -2], value, result[..., -1])
    if _side_is_open(BoundarySide.WEST.value, grid):
        value = obc_eta.west.to(dtype=eta.dtype, device=eta.device)
        result[..., 0] = torch.where(wet_w[:, 1], value, result[..., 0])
    return result


def compute_surface_hfac(
    grid: StructuredCGrid,
    eta: torch.Tensor,
    h0FacC: torch.Tensor,
    h0FacW: torch.Tensor,
    h0FacS: torch.Tensor,
    hFacInf: float = 0.2,
    obc_eta: float | torch.Tensor | SurfaceEtaBoundary = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Surface-level ``hFac_surfC/W/S`` for the given free-surface height.

    Transcribes MITgcm ``CALC_SURF_DR`` (``model/src/calc_surf_dr.F:79-183``)
    for z-level coordinates with ``nonlinFreeSurf >= 1`` and
    ``select_rStar = 0``.  Returns the three surface-level arrays; the caller
    decides *when* to write them into the geometry, because MITgcm's
    ``UPDATE_SURF_DR`` is a separate call at a different point of the step
    (see ``ImplicitFreeSurfaceStepper`` and ``forward_step.F:852``).

    Two MITgcm details that are easy to get wrong, both verified against the
    pinned tree (`27256b2d6ae1`, the MITgcm revision every reference uses;
    at 242228367a50 forward_step.F lies one line higher):

    * **There is no ``hFacSup`` clamp.**  With ``select_rStar = 0`` the whole
      tree references ``hFacSup`` only to print a WARNING
      (``calc_surf_dr.F:126-135``) and in the r* routine
      (``calc_r_star.F:194,249``), which this path does not take.
    * **The lower clamp is on the r-position, not on hFac**, via ``Rmin_surf``
      (``ini_nlfs_vars.F:156-171``).  With ``Ro_surf == 0`` (no ice shelf, no
      p-coordinates -- both structural in LOCOMgcm) and no wet column
      shallower than the top layer, that clamp collapses to ``hFac >= hFacInf``,
      which is what this function applies.  The caller is responsible for
      rejecting a grid that violates the second precondition; see
      ``ImplicitFreeSurfaceStepper._check_nonlin_free_surf_preconditions``.

    Parameters
    ----------
    grid : StructuredCGrid
    eta : [batch, ny, nx] or [ny, nx] — current free surface height (etaH)
    h0FacC/W/S : resting-state hFac tensors (saved at first step)
    hFacInf : minimum hFac for surface cell (MITgcm default 0.2)
    obc_eta : prescribed open-boundary η (MITgcm OBWeta/OBEeta/OBSeta/OBNeta).
        Four side samples, a scalar, or a ``[ny, nx]`` tensor whose edge cells carry the
        prescribed η on each open side (west=``[:, 0]``, east=``[:, -1]``,
        south=``[0, :]``, north=``[-1, :]`` — the layout produced by
        ``Model._sync_stepper_obc_eta``).  Used only at the OB inflow faces
        (see OBCS_APPLY_SURF_DR transcription below).  Default ``0.0`` reduces
        the OB override to the resting hFac, i.e. bitwise-identical to the
        no-prescribed-OB-eta case (e.g. internal_wave).
    """
    if eta.ndim == 3:
        if eta.shape[0] != 1:
            raise ValueError(
                f"nonlinear free-surface hFac update only supports batch=1, got {eta.shape[0]}",
            )
        eta_2d = eta[0]
    elif eta.ndim == 2:
        eta_2d = eta
    else:
        raise ValueError(
            f"expected eta to be [ny,nx] or [1,ny,nx], got shape {tuple(eta.shape)}",
        )

    dtype = eta_2d.dtype
    device = eta_2d.device
    ny, nx = grid.ny, grid.nx
    drF0 = top_layer_thickness_value(grid)

    # Prescribed open-boundary η at each side (MITgcm OBWeta/OBEeta/OBSeta/OBNeta).
    # obc_eta is a scalar or a [ny, nx] tensor whose outermost edge cells hold
    # the prescribed η; pull the per-side edge so the OB-inflow-face override
    # below uses h0Fac + OBeta/drF (obcs_apply_surf_dr.F:132-139).
    if isinstance(obc_eta, SurfaceEtaBoundary):
        obeta_w = obc_eta.west.to(dtype=dtype, device=device)
        obeta_e = obc_eta.east.to(dtype=dtype, device=device)
        obeta_s = obc_eta.south.to(dtype=dtype, device=device)
        obeta_n = obc_eta.north.to(dtype=dtype, device=device)
    elif isinstance(obc_eta, torch.Tensor):
        obe = obc_eta.to(dtype=dtype, device=device)
        if obe.ndim == 3:
            obe = obe[0]
        obeta_w, obeta_e = obe[:, 0], obe[:, -1]   # west / east columns
        obeta_s, obeta_n = obe[0, :], obe[-1, :]   # south / north rows
    else:
        obeta_w = obeta_e = obeta_s = obeta_n = float(obc_eta)

    # Center cells
    h0_surf_c = h0FacC[0].to(dtype=dtype, device=device)
    hFac_surfC = torch.clamp_min(h0_surf_c + eta_2d / drF0, hFacInf)
    hFacC_new = torch.where(h0_surf_c > 0, hFac_surfC, h0_surf_c)

    # U-faces (W in MITgcm naming): [ny, nx+1]
    # MIT calc_surf_dr.F:145-163 takes MIN(rSurf(i-1), rSurf(i)) on EVERY face
    # i=1..sNx+1, NOT the average ("hFacW not larger than the 2 surrounding
    # hFacC"); the two edge faces read the etaH halo, which update_etah.F fills
    # by EXCH.  The halo follows the grid's own topology (numerics rule 11):
    # the wrap cell on a periodic axis, the neighbour rank's cell on a
    # distributed side, so a coupled edge face gets MIN(neighbour, own) from
    # both of its addresses.  On a wall or open edge the REPLICATE ghost is the
    # own cell, so MIN(own, own) keeps that face's historical value bit for bit
    # (masked by h0FacW=0 at a wall, overridden below at an open boundary).
    eta_x = exchange(
        eta_2d,
        grid=grid,
        location="T",
        extent=HaloExtentV1(1, 1, 0, 0),
        fill=FillSpec.WRAP if grid.edge_face_couples("x") else FillSpec.REPLICATE,
    )
    eta_u = torch.minimum(eta_x[:, 0:nx + 1], eta_x[:, 1:nx + 2])

    h0_surf_w = h0FacW[0].to(dtype=dtype, device=device)
    hFac_surfW = torch.clamp_min(h0_surf_w + eta_u / drF0, hFacInf)
    hFacW_new = torch.where(h0_surf_w > 0, hFac_surfW, h0_surf_w)
    # MIT OBCS_APPLY_SURF_DR (calc_surf_dr.F:188 -> obcs_apply_surf_dr.F:125-143):
    # at an OB inflow u-face the surface thickness uses the prescribed OB eta
    # (OBWeta/OBEeta), NOT the interior-eta rescaling:
    #     hFac_surfW = MAX(hFacInf, h0FacW + OBeta*recip_drF)
    # (note: only a lower clamp to hFacInf — no upper hFacSup clamp at the OB
    # face, matching obcs_apply_surf_dr.F:139).  With OBeta=0 (no prescribed OB
    # eta) the OB inflow face stays at the resting hFacW = h0FacW.  Omitting this
    # breaks the mean-zero OB normal-velocity transport, producing a spurious net
    # OB inflow (barotropic mass leak at the first interior cell).
    #
    # Grid-/distributed-agnostic guard: decide each side independently with
    # _side_is_open (west vs east are distinct — boundary_topology.open_x cannot
    # tell them apart), and it returns False at an internal rank seam (a tile
    # whose west/east face is shared with a neighbour rank, not a physical OB).
    # On a tile-split run only the rank owning the physical global OB applies the
    # override at its local [:, 1] / [:, -2] face — the same per-side seam-aware
    # rule used by _compute_interior_mask and the OBCS apply layer.
    if _side_is_open(BoundarySide.WEST.value, grid):
        hFacW_new[:, 1] = torch.where(h0_surf_w[:, 1] > 0, torch.clamp_min(h0_surf_w[:, 1] + obeta_w / drF0, hFacInf), h0_surf_w[:, 1])
    if _side_is_open(BoundarySide.EAST.value, grid):
        hFacW_new[:, -2] = torch.where(h0_surf_w[:, -2] > 0, torch.clamp_min(h0_surf_w[:, -2] + obeta_e / drF0, hFacInf), h0_surf_w[:, -2])

    # V-faces (S in MITgcm naming): [ny+1, nx]; calc_surf_dr.F:165-183, with the
    # same topology-driven halo as the U faces above.
    eta_y = exchange(
        eta_2d,
        grid=grid,
        location="T",
        extent=HaloExtentV1(0, 0, 1, 1),
        fill=FillSpec.WRAP if grid.edge_face_couples("y") else FillSpec.REPLICATE,
    )
    eta_v = torch.minimum(eta_y[0:ny + 1, :], eta_y[1:ny + 2, :])

    h0_surf_s = h0FacS[0].to(dtype=dtype, device=device)
    hFac_surfS = torch.clamp_min(h0_surf_s + eta_v / drF0, hFacInf)
    hFacS_new = torch.where(h0_surf_s > 0, hFac_surfS, h0_surf_s)
    # MIT OBCS_APPLY_SURF_DR at the OB inflow v-face (south/north):
    #     hFac_surfS = MAX(hFacInf, h0FacS + OBeta*recip_drF)   (OBeta=0 -> h0FacS).
    # Per-side, seam-aware guard (see the U-face note above).
    if _side_is_open(BoundarySide.SOUTH.value, grid):
        hFacS_new[1, :] = torch.where(h0_surf_s[1, :] > 0, torch.clamp_min(h0_surf_s[1, :] + obeta_s / drF0, hFacInf), h0_surf_s[1, :])
    if _side_is_open(BoundarySide.NORTH.value, grid):
        hFacS_new[-2, :] = torch.where(h0_surf_s[-2, :] > 0, torch.clamp_min(h0_surf_s[-2, :] + obeta_n / drF0, hFacInf), h0_surf_s[-2, :])
    return hFacC_new, hFacW_new, hFacS_new


def _surface_rescale_ratio(
    hfac_now: torch.Tensor,
    hfac_surf: torch.Tensor,
) -> torch.Tensor:
    """``_hFac / hFac_surf`` at the surface level, zero where the face is dry.

    Both operands are surface-level slices.  ``hfac_now`` is the geometry the
    tendency was evaluated on -- in MITgcm that is ``_hFacW/_hFacS``, which
    ``UPDATE_SURF_DR`` has *not* yet refreshed and therefore still holds
    ``h(eta^{n-1})`` (``forward_step.F:852`` runs after ``DYNAMICS``).
    ``hfac_surf`` is ``hFac_surfW/S`` = ``h(eta^n)``, computed by the previous
    step's ``CALC_SURF_DR`` (``forward_step.F:959``).
    """
    safe = torch.where(hfac_surf > 0, hfac_surf, torch.ones_like(hfac_surf))
    return torch.where(hfac_now > 0, hfac_now / safe, torch.zeros_like(hfac_now))


def _rescale_tendency_nonlin_freesurf(
    tend_u: torch.Tensor,
    tend_v: torch.Tensor,
    *,
    hFacW_now: torch.Tensor,
    hFacS_now: torch.Tensor,
    hFac_surfW: torch.Tensor,
    hFac_surfS: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rescale the surface momentum tendency for a nonlinear free surface.

    MITgcm ``timestep.F:303-317``, taken for ``nonlinFreeSurf > 1`` on standard
    z-coordinates (no r*, no sigma, not vector-invariant)::

        IF ( k.EQ.kSurfW(i,j,bi,bj) ) THEN
          gUtmp(i,j) = gUtmp(i,j) * _hFacW(i,j,k,bi,bj) / hFac_surfW(i,j,bi,bj)
        ENDIF

    Two properties of that line decide what this function may and may not do:

    * The ratio is applied to ``gUtmp`` only.  In synchronous time stepping the
      hydrostatic pressure gradient is folded into ``gU`` *before* the
      Adams-Bashforth combination (``timestep.F:116-125``), so it is inside
      ``gUtmp`` and *is* rescaled, while ``gUdPx`` -- which on this branch holds
      only ``-psFac*phiSurfX`` (``timestep.F:330-333``) -- is not.  Callers must
      therefore rescale the assembled slow tendency and add the surface-pressure
      predictor term afterwards.
    * The AB2 memory ``guNm1`` is **not** rescaled.  The tracer side does rescale
      its memory (``temp_integrate.F:433``); the asymmetry is real.

    ``hFacW_now`` / ``hFacS_now`` and ``hFac_surfW`` / ``hFac_surfS`` are
    surface-level slices shaped ``[1, ny, nx+1]`` and ``[1, ny+1, nx]``; the
    caller owns the time-layer bookkeeping that decides which eta each was
    built from.

    Returns rescaled copies; only ``k = 0`` is modified.
    """
    ratio_u = _surface_rescale_ratio(
        hFacW_now.to(dtype=tend_u.dtype, device=tend_u.device),
        hFac_surfW.to(dtype=tend_u.dtype, device=tend_u.device),
    )
    ratio_v = _surface_rescale_ratio(
        hFacS_now.to(dtype=tend_v.dtype, device=tend_v.device),
        hFac_surfS.to(dtype=tend_v.dtype, device=tend_v.device),
    )
    if ratio_u.ndim == 2:
        ratio_u = ratio_u.unsqueeze(0)
    if ratio_v.ndim == 2:
        ratio_v = ratio_v.unsqueeze(0)

    # Out-of-place assembly so autograd can back-propagate through first-step
    # diagnostics.
    scaled_u = tend_u[:, 0:1, :, :] * ratio_u
    scaled_v = tend_v[:, 0:1, :, :] * ratio_v
    return (
        torch.cat((scaled_u, tend_u[:, 1:, :, :]), dim=1),
        torch.cat((scaled_v, tend_v[:, 1:, :, :]), dim=1),
    )


def recip_hfac_new_freesurf(
    hFacC_now: torch.Tensor,
    hFac_surfC: torch.Tensor,
) -> torch.Tensor:
    """MITgcm ``recip_hFacNew`` of a ``select_rStar=0`` run (``thermodynamics.F:236-244``).

    The implicit vertical stage of the tracer step (IMPLDIFF, and the KPP
    nonlocal source and column solve that ride on it) divides by the thickness
    the explicit update has already moved to: ``1/hFac_surfC`` on the surface
    level ``kSurfC``, ``recip_hFacC`` below it, zero on dry cells.  The explicit
    tendency itself reaches the same thickness through FREESURF_RESCALE_G.

    ``hFacC_now`` is the full ``[nz, ny, nx]`` geometry the tendency read
    (``h(eta^{n-1})`` on the surface level), ``hFac_surfC`` the surface slice
    ``h(eta^n)``.  Levels below the surface are the reciprocal the solves derive
    from the grid themselves.
    """
    wet = hFacC_now > 0
    recip = torch.where(
        wet, 1.0 / torch.where(wet, hFacC_now, torch.ones_like(hFacC_now)),
        torch.zeros_like(hFacC_now),
    )
    surf = hFac_surfC.to(dtype=hFacC_now.dtype, device=hFacC_now.device).reshape(
        (1,) + tuple(hFacC_now.shape[-2:]))
    safe = torch.where(surf > 0, surf, torch.ones_like(surf))
    recip_surface = torch.where(wet[0:1], 1.0 / safe, torch.zeros_like(surf))
    return torch.cat((recip_surface, recip[1:]), dim=0)


def rescale_tracer_tendency_freesurf(
    tendency: torch.Tensor,
    *,
    hFacC_now: torch.Tensor,
    hFac_surfC: torch.Tensor,
) -> torch.Tensor:
    """MITgcm ``FREESURF_RESCALE_G`` (``freesurf_rescale_g.F:61-71``).

    ``gTracer(k=kSurfC) *= _hFacC / hFac_surfC``.  This is the ``select_rStar=0``
    branch, i.e. the one a plain ``nonlinFreeSurf > 0`` run takes. The r*
    branch is not supported by the stepper.

    ``temp_integrate.F:406-440`` calls it **twice** per level: once on the
    Adams-Bashforth-combined tendency and once on the stored memory ``gtNm1``,
    so the memory carries the scaling into the next step.  Callers must do both.
    """
    ratio = _surface_rescale_ratio(
        hFacC_now.to(dtype=tendency.dtype, device=tendency.device),
        hFac_surfC.to(dtype=tendency.dtype, device=tendency.device),
    )
    if ratio.ndim == 2:
        ratio = ratio.unsqueeze(0)
    scaled = tendency[:, 0:1, :, :] * ratio
    return torch.cat((scaled, tendency[:, 1:, :, :]), dim=1)
