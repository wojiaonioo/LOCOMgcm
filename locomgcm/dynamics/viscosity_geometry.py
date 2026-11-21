"""Wet fractions, corner fractions, reciprocals and lengths for horizontal viscosity.

Everything here reads the grid: the current ``hFacC/W/S`` (``[nz, ny, nx]``,
``[nz, ny, nx+1]``, ``[nz, ny+1, nx]``), the resting fractions of a nonlinear free
surface, the horizontal metrics, and the corner ``hFacZ`` ``[..., ny+1, nx+1]``
built through ``exchange``.  Reciprocals are zero where the fraction is zero and
are selected before the division.  The memoized entries (corner hFacZ, reciprocal
hFac and masks, Leith lengths) are per-grid constants only while hFac is static;
:func:`_hfac_memo_bypass_reason` decides when they recompute instead.  All tensors
are returned in the caller's dtype and device.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.accel.constant_cache import constant_cache
from locomgcm.geometry import BoundaryType, StructuredCGrid
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from locomgcm.geometry.grid import (
    device_resident_masks_enabled,
    record_device_resident_use,
    runtime_memo_constant,
)
from locomgcm.geometry.metric_bundle import _positive_reciprocal

from ._hfacz_topology import apply_hfacz_corner_min3_override


def _constant_cache_enabled() -> bool:
    """Toggle for memoizing per-grid static constants (default ON).

    Set ``LOCOMGCM_CACHE_CONSTANTS=0`` to fall back to the per-step recompute path
    for the divvort-budget cache sites in this module.  The enable decision is
    owned by the unified control plane (``locomgcm.accel``), not by this
    module.
    """
    from locomgcm.accel import gate

    return gate("cache_constants").effective_value


def _hfac_c(grid, dtype, device, *, site):
    """``hFacC`` via memoized device cache (toggle ON) or per-step ``.to``.

    ``site`` is the caller's name in the device-resident bookkeeping.
    """
    if device_resident_masks_enabled() and hasattr(grid, "hfac_c_for"):
        record_device_resident_use(site, "hFacC")
        return grid.hfac_c_for(dtype, device)
    return grid.masks.hFacC.to(dtype=dtype, device=device)


def _hfac_w(grid, dtype, device, *, site):
    """``hFacW`` via memoized device cache (toggle ON) or per-step ``.to``.

    ``site`` is the caller's name in the device-resident bookkeeping.
    """
    if device_resident_masks_enabled() and hasattr(grid, "hfac_w_for"):
        record_device_resident_use(site, "hFacW")
        return grid.hfac_w_for(dtype, device)
    return grid.masks.hFacW.to(dtype=dtype, device=device)


def _hfac_s(grid, dtype, device, *, site):
    """``hFacS`` via memoized device cache (toggle ON) or per-step ``.to``.

    ``site`` is the caller's name in the device-resident bookkeeping.
    """
    if device_resident_masks_enabled() and hasattr(grid, "hfac_s_for"):
        record_device_resident_use(site, "hFacS")
        return grid.hfac_s_for(dtype, device)
    return grid.masks.hFacS.to(dtype=dtype, device=device)


def _hfac_memo_bypass_reason(grid: "StructuredCGrid") -> str | None:
    """Why the hFac-DERIVED memos (Sites 1 and 2) must not serve this grid.

    Returns ``None`` when the memo may serve, else the declared decline reason.
    One function so the decision and its label cannot drift apart
    by the shared support predicate; the boolean form
    :func:`_hfac_memo_enabled` is defined from it, never beside it.

    ``_constant_cache_enabled()`` alone is not sufficient for a cache whose
    *content* derives from ``hFac``.  Two distinct things make it unsound:

    ``hfac_time_varying_nonlin_free_surf``
        Under ``nonlinFreeSurf>=1`` the surface level of ``hFacC/hFacW/hFacS``
        is rewritten every step (in place, or by tensor replacement under autograd)
        (``steppers/implicit_free_surface.py`` ``_apply_surf_dr``, MITgcm
        ``UPDATE_SURF_DR`` at ``forward_step.F:852`` of MITgcm 27256b2d) while ``id(grid)`` -- the
        cache key -- does not move, so an id-keyed memo keeps serving the
        pre-write geometry for the rest of the run.  MITgcm recomputes both
        quantities instead: ``recip_hFac*`` in ``update_surf_dr.F`` and the
        corner ``hFacZ`` in ``mom_calc_hfacz.F`` on every ``MOM_*`` call.

    ``grad``
        Under a GEOMETRY control (MITgcm ``ALLOW_DEPTH_CONTROL``, the
        ``verification/bottom_ctrl_5x5`` build) ``hFacC/hFacW/hFacS`` are built
        from the bathymetry leaf and carry its graph
        (``geometry/masks.py`` ``from_bathymetry_3d``). The memo builds its
        entry under ``torch.no_grad()`` and then serves that detached tensor to
        every later call, so ``d(corner hFacZ)/d(bathy)`` and
        ``d(recip_hFac*)/d(bathy)`` come back **zero with no error anywhere** --
        a silent failure.  The
        same defect as ``grid.ensure_grid_device_resident``'s former
        ``no_grad`` migration, and it takes the same direction: when the hFac a
        site reads carries a graph, recompute instead of caching.

        Declining the cache is the whole fix; the cache is not made
        differentiable.  A memo entry that carried ``grad_fn`` would be reused
        across steps and pin one step's graph, which is a second defect, not a
        repair.

    Both send the site back to the per-call recompute -- bit-for-bit the
    ``LOCOMGCM_CACHE_CONSTANTS=0`` path, which is also the behaviour this module had
    before the memos landed. The caller declares the skip through
    :func:`_record_hfac_memo_bypass`, so the run stays distinguishable from one
    where the switch is simply off.

    A grid that cannot answer the time-varying question (no ``masks``, or a
    mask object with no such attribute) is treated as time-varying, i.e. NOT
    cacheable: an unknown answer must not open a cache whose staleness is
    silent.  No production path reaches that today -- Sites 1/2 take their hFac
    arguments from ``grid.masks.hFac*`` and would already have raised -- but the
    conservative direction is the one to fail towards.

   The Leith length-scale memo reads only dxF/dyF/dxV/dyU and keeps the plain
    gate; the bool masks behind ``MaskSet._float_for`` likewise stay cacheable
    because ``UPDATE_SURF_DR`` preserves wet/dry and a bool tensor carries no
    graph in the first place.

    Both causes are constant across the steps of one run -- the latch is
    one-way, and whether the bathymetry is a differentiable leaf is fixed when
    the grid is built -- so this predicate cannot guard-fail per step, which is
    the constraint a Python value read inside the compiled region has to meet.
    ``requires_grad`` is a property of the tensor, not of
    the ambient grad mode, so an ``inference_mode``/``no_grad`` forward over a
    geometry-control grid takes the same branch as its backward-bearing twin.
    """
    if not _constant_cache_enabled():
        return None
    masks = getattr(grid, "masks", None)
    if masks is None:
        return "hfac_time_varying_nonlin_free_surf"
    if getattr(masks, "hfac_time_varying", True):
        return "hfac_time_varying_nonlin_free_surf"
    if _hfac_carries_grad(masks):
        return "grad"
    return None


def _hfac_carries_grad(masks: object) -> bool:
    """Whether any of ``hFacC/hFacW/hFacS`` is a differentiable geometry field.

    ``None`` entries (2-D mask sets with no partial cells) and plain constants
    both answer False, which is every non-geometry-control configuration --
    those keep the memo exactly as before.
    """
    for name in ("hFacC", "hFacW", "hFacS"):
        tensor = getattr(masks, name, None)
        if tensor is not None and getattr(tensor, "requires_grad", False):
            return True
    return False


def _hfac_memo_enabled(grid: "StructuredCGrid") -> bool:
    """Whether the hFac-DERIVED memos may serve this grid (Sites 1 and 2).

    The boolean view of :func:`_hfac_memo_bypass_reason`; see it for the two
    causes that close the cache and why each one is unsound to keep open.
    """
    return _hfac_memo_bypass_reason(grid) is None


def _record_hfac_memo_bypass(variant: str, site: str, reason: str) -> None:
    """Declare one hFac-derived memo skipped, with the cause that closed it.

    Without this the bypass has the same shape as the switch being off: both
    just recompute and record nothing, so a whole ``nonlinFreeSurf>=1`` run
    would report these variants as ``requested-zero`` -- indistinguishable from
    "the memo was deleted" or "the call site is never reached", which is the
    silent failure that hides whether the requested path ran.  The
    repository's existing form for "requested but structurally unreachable" is
    ``record_dispatch(outcome="inapplicable", reason=...)``
    (``geometry/grid.py:110-118``), which is what this emits.

    ``reason`` is the caller's, not a constant re-derived here: the two causes
    (``hfac_time_varying_nonlin_free_surf``, ``grad``) have different owners
    and different fixes, and a single hard-coded token would report a geometry
    control as a nonlinear-free-surface run.  It comes from
    :func:`_hfac_memo_bypass_reason`, the one place the decision is made.

    Called FROM the bypass line, never from :func:`_hfac_memo_enabled` -- a
    record made inside a gate predicate relabels "requested" as "ignited"
    (the cache-observation contract).

    Cross-rank uniformity: unlike cache events, ``record_dispatch`` rows DO
    enter the SPMD dispatch signature (``accel/runtime.py``
    ``dispatch_signature`` excludes only ``_CACHE_EVENTS``).  These rows are
    uniform by construction -- the predicate behind them reads
    ``LOCOMGCM_CACHE_CONSTANTS`` (one environment value, identical on every rank),
    the ``MaskSet`` latch, which ``_apply_surf_dr`` sets on every rank at the
    same step boundary (being a latch, it cannot flip mid-run on one rank
    only), and ``hFac*.requires_grad``, which is decided when the grid is built
    and is therefore the same on every rank that was handed the same grid.  So
    every rank takes this branch on the same calls, the same number of times,
    in the same order.

   The gate-off early return is an API guard, not a second copy of the bypass
    decision: ``record_dispatch`` raises when the switch is disabled, and a
    disabled switch is not a bypass -- it has its own reporting state.
    """
    if not _constant_cache_enabled():
        return
    from locomgcm.accel import current_rank, record_dispatch

    record_dispatch(
        "cache_constants",
        outcome="inapplicable",
        variant=variant,
        site=site,
        backend="torch",
        rank=current_rank(),
        reason=reason,
    )


HZOPTION = 0
"""MIT ``mom_calc_hfacz.F`` hZoption selector for this module.

MIT defines ``PARAMETER ( hZoption = 0 )`` at ``mom_calc_hfacz.F:80``.
The default path at ``mom_calc_hfacz.F:215-222`` uses the normal
four-neighbour minimum.  The cubed-sphere special block begins at
``mom_calc_hfacz.F:229`` and, for ``hZoption=1``, lines 229-300 overwrite
face-local cube corners with the three valid neighbour values.  This module
implements only that hZoption=1 min-of-three branch; hZoption=2's later
SUM/3 face-dependent branch is intentionally not implemented here.
"""


def _hfacz_option(hZoption: int | None) -> int:
    option = HZOPTION if hZoption is None else int(hZoption)
    if option == 2:
        raise NotImplementedError(
            "mom_calc_hfacz hZoption=2 SUM/3 cube-corner branch is not "
            "implemented"
        )
    return option


def _hfacz_4neighbor_min(
    hFacW_here: torch.Tensor,
    hFacW_south: torch.Tensor,
    hFacS_here: torch.Tensor,
    hFacS_west: torch.Tensor,
) -> torch.Tensor:
    """MIT ``mom_calc_hfacz`` hZoption=0 four-neighbour minimum."""
    return torch.minimum(
        torch.minimum(hFacW_here, hFacW_south),
        torch.minimum(hFacS_here, hFacS_west),
    )


def _hfacz_4neighbor_smooth(
    hFacW_here: torch.Tensor,
    hFacW_south: torch.Tensor,
    hFacS_here: torch.Tensor,
    hFacS_west: torch.Tensor,
) -> torch.Tensor:
    """MIT ``mom_calc_hfacz`` corner hFac under ``ALLOW_DEPTH_CONTROL``.

    ``mom_calc_hfacz.F:86`` opens an ``#ifdef ALLOW_DEPTH_CONTROL`` branch
    (:86-157) that replaces the whole ``hZoption`` selector — the selector and
    the cubed-sphere corner block live in the ``#else`` at :158 and are
    unreachable in a depth-control build.  The branch computes, at :107-139::

        hFacZOpenJ = SMOOTHMIN( hFacW(i,j), hFacW(i,j-1) )
                     * maskW(i,j)*maskW(i,j-1)
        hFacZOpenI = SMOOTHMIN( hFacS(i,j), hFacS(i-1,j) )
                     * maskS(i,j)*maskS(i-1,j)
        hFacZ      = SMOOTHMIN( hFacZOpenI, hFacZOpenJ )
                     * maskW(i,j)*maskW(i,j-1)*maskS(i,j)*maskS(i-1,j)

    With ``smoothAbsFuncRange < 0`` the ``SMOOTHMIN_RS`` of
    ``update_masks_etc.F:364`` degenerates to ``.5*(a+b)``, so this is a
    two-level arithmetic mean gated by the four face masks — *not* the flat
    four-neighbour ``MIN`` of the default path.  The nesting is kept literal
    because the two forms differ: the mean of means weights the i-pair and the
    j-pair equally, whatever their counts.

    When ``USE_SMOOTH_MIN`` is undefined this branch would be nested ``MIN``
    with the same mask factors, which for non-negative hFac is the flat
    four-neighbour ``MIN`` — that is why the default path here needs no
    ``ALLOW_DEPTH_CONTROL`` variant of its own.
    """
    open_j_wet = (hFacW_here > 0.0) & (hFacW_south > 0.0)   # maskW * maskW
    open_i_wet = (hFacS_here > 0.0) & (hFacS_west > 0.0)    # maskS * maskS
    zero = torch.zeros_like(hFacW_here)
    open_j = torch.where(open_j_wet, 0.5 * (hFacW_here + hFacW_south), zero)
    open_i = torch.where(open_i_wet, 0.5 * (hFacS_here + hFacS_west), zero)
    return torch.where(open_i_wet & open_j_wet, 0.5 * (open_i + open_j), zero)


def _hfacz_smooth_geometry(grid: "StructuredCGrid") -> bool:
    """True when the grid was built with MITgcm's depth-control geometry.

    Reads ``MaskSet.smooth_abs_func_range``, which ``from_bathymetry_3d``
    stores.  A grid with no mask set (duck-typed test grids) answers False =
    the default ``MIN`` corner, which is what every existing case wants.
    """
    return getattr(getattr(grid, "masks", None), "smooth_abs_func_range", None) is not None


def _hfacz_corner_reduce(
    hFacW_here: torch.Tensor,
    hFacW_south: torch.Tensor,
    hFacS_here: torch.Tensor,
    hFacS_west: torch.Tensor,
    *,
    smooth: bool,
) -> torch.Tensor:
    """Select the corner reduction that matches the grid's geometry mode."""
    if smooth:
        return _hfacz_4neighbor_smooth(hFacW_here, hFacW_south, hFacS_here, hFacS_west)
    return _hfacz_4neighbor_min(hFacW_here, hFacW_south, hFacS_here, hFacS_west)


def _compute_hFacZ_4neighbor(
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    grid: "StructuredCGrid",
    *,
    hZoption: int | None = None,
    region_fill: FillSpec = FillSpec.ZERO,
    region_support: bool = False,
    smooth: bool | None = None,
) -> torch.Tensor:
    """hFac at vorticity (corner) points — MITgcm 4-neighbor formula.

    MITgcm ``mom_calc_hfacz.F`` (hZoption=0):
        hFacZ(i,j,k) = MIN(hFacW(i,j,k), hFacW(i,j-1,k),
                           hFacS(i,j,k), hFacS(i-1,j,k))

    MITgcm ``mom_calc_hfacz.F`` (hZoption=1, cubed sphere only):
        the 24 face-local cube-corner positions are overwritten with the
        MIN of the three valid face-local operands.

    Returns shape ``[..., ny+1, nx+1]``.

    Inputs:
        hFacW: ``[..., ny, nx+1]``
        hFacS: ``[..., ny+1, nx]``

    The j-neighbour row of ``hFacW`` and the i-neighbour column of ``hFacS``
    at the domain boundary are obtained through :func:`exchange` so the
    helper consults the ``Grid`` topology layer instead of hard-constructing
    zero edge rows / columns.  A cubed-sphere ``Grid`` supplies the
    neighbouring face's hFac through the same call.

    Edge (non-cube) ghost-fill semantics are caller-selected:

    * ``region_fill`` controls the halo fill used for the boundary corner
      rows / columns.  The two divvort Laplacian consumers
      (``_u_laplacian`` / ``_v_laplacian``) read only the interior corner
      block ``[..., 1:ny, 1:nx]``, so for them the edge choice is dead and
      the default ``FillSpec.ZERO`` is used.  The vector-invariant Coriolis
      consumer instead reads the full ``[ny+1, nx+1]`` corner field
      (including boundary rows / columns), so it passes ``FillSpec.WRAP`` to
      replicate MITgcm's single-tile EXCH self-wrap of ``mom_calc_hfacz.F``;
      so does the divvort-budget corner of :func:`_divvort_corner_hfacz`.
    * ``region_support`` (vector-invariant unit-hFac fallback only)
      multiplies the wrap-filled 4-neighbour MIN by a second ``ZERO``-fill
      4-neighbour MIN, reproducing the wall / open-edge support that the
      legacy VIM path applied when ``hFacW``/``hFacS`` were unit masks.

    Both knobs default to the byte-identical interior-only behaviour, so the
    divvort callers and the cube branch are unaffected.

    ``smooth`` selects the corner reduction; ``None`` follows the grid's
    geometry mode (:func:`_hfacz_smooth_geometry`), ``False`` is the plain
    four-neighbour ``MIN`` of the resting ``h0FacZ`` (``mom_vecinv.F:318-320``).
    """
    smooth = _hfacz_smooth_geometry(grid) if smooth is None else smooth
    boundary = grid.boundary_topology
    if (
        getattr(grid, "face_count", 1) == 6
        and BoundaryType.SEAM in (boundary.west_east, boundary.south_north)
    ):
        from locomgcm.geometry.exchange import (
            exchange_cubed_sphere_vector,
            exchange_cubed_sphere_vector_distributed,
        )

        option = _hfacz_option(hZoption)
        n = grid.nx
        ol = 1
        # Face-local routing: distributed paired exchange when ctx attached
        # (local grid), else the byte-identical single-rank exchange.
        ctx = getattr(grid, "_cube_step_ctx", None)
        if ctx is not None:
            hWc, hSc = exchange_cubed_sphere_vector_distributed(
                hFacW, hFacS, grid=grid,
                extent=HaloExtentV1.uniform(ol), ctx=ctx, with_signs=False,
            )
        else:
            hWc, hSc = exchange_cubed_sphere_vector(
                hFacW,
                hFacS,
                grid=grid,
                extent=HaloExtentV1.uniform(ol),
                with_signs=False,
            )
        i0 = j0 = ol
        cj = slice(j0, j0 + n + 1)
        ci = slice(i0, i0 + n + 1)
        hW_here = hWc[..., cj, ci]
        hW_below = hWc[..., j0 - 1:j0 + n, ci]
        hS_here = hSc[..., cj, ci]
        hS_left = hSc[..., cj, i0 - 1:i0 + n]
        hFacZ = _hfacz_corner_reduce(
            hW_here, hW_below, hS_here, hS_left,
            smooth=smooth,
        )
        return apply_hfacz_corner_min3_override(
            hFacZ,
            hFacW,
            hFacS,
            grid,
            option=option,
        )

    # A one-row periodic tile still has four wet neighbours at each corner.
    # Use the same topology-aware halos for every nonempty tile size.
    # j-halo of hFacW (one ghost row south + north) and i-halo of hFacS
    # (one ghost column west + east) routed through the Grid/exchange()
    # layer.  hFacW has the U-location column staggering; hFacS the
    # V-location row staggering.
    y_halo = HaloExtentV1(
        west_left_extent=0,
        east_right_extent=0,
        south_bottom_extent=1,
        north_top_extent=1,
    )
    x_halo = HaloExtentV1(
        west_left_extent=1,
        east_right_extent=1,
        south_bottom_extent=0,
        north_top_extent=0,
    )
    hFacW_ext = exchange(
        hFacW,
        grid=grid,
        location="U",
        extent=y_halo,
        fill=region_fill,
        constant=not hFacW.requires_grad,
    )  # [..., ny+2, nx+1]
    hFacS_ext = exchange(
        hFacS,
        grid=grid,
        location="V",
        extent=x_halo,
        fill=region_fill,
        constant=not hFacS.requires_grad,
    )  # [..., ny+1, nx+2]

    # hFacZ(j,i) = MIN(hFacW(j,i), hFacW(j-1,i), hFacS(j,i), hFacS(j,i-1)).
    # With the extended arrays: hFacW_ext[j+1,i]=hFacW(j,i),
    # hFacW_ext[j,i]=hFacW(j-1,i); hFacS_ext[j,i+1]=hFacS(j,i),
    # hFacS_ext[j,i]=hFacS(j,i-1).  At the interior corner block
    # [1:ny, 1:nx] every operand lies inside the domain, so this MIN is
    # bit-identical to a hand-rolled in-domain 4-neighbour MIN regardless
    # of the (dead) edge ghost values.
    hFacZ = _hfacz_corner_reduce(
        hFacW_ext[..., 1:, :],
        hFacW_ext[..., :-1, :],
        hFacS_ext[..., :, 1:],
        hFacS_ext[..., :, :-1],
        smooth=smooth,
    )  # [..., ny+1, nx+1]

    if region_support:
        # Vector-invariant unit-hFac fallback: multiply by the ZERO-fill
        # 4-neighbour MIN so wall / open boundary corners collapse to the
        # in-domain support (the WRAP-filled MIN alone would self-wrap a
        # nonzero value onto a closed edge).  See legacy
        # vector_invariant_momentum._compute_hFacZ_uncached unit-hFac path.
        # Stays on MIN under the depth-control geometry as well: this factor
        # is a 0/1 support INDICATOR over unit hFac, not a partial-cell
        # fraction, and the two agree there (0.5*(1+1) == 1).  Routing it
        # through the smooth form would let a half-open corner scale the
        # result instead of gating it.
        hFacW_support_ext = exchange(
            hFacW,
            grid=grid,
            location="U",
            extent=y_halo,
            fill=FillSpec.ZERO,
        )
        hFacS_support_ext = exchange(
            hFacS,
            grid=grid,
            location="V",
            extent=x_halo,
            fill=FillSpec.ZERO,
        )
        hFacZ = hFacZ * _hfacz_4neighbor_min(
            hFacW_support_ext[..., 1:, :],
            hFacW_support_ext[..., :-1, :],
            hFacS_support_ext[..., :, 1:],
            hFacS_support_ext[..., :, :-1],
        )

    option = _hfacz_option(hZoption)
    return apply_hfacz_corner_min3_override(
        hFacZ,
        hFacW,
        hFacS,
        grid,
        option=option,
    )


def _recip(x: torch.Tensor) -> torch.Tensor:
    """``1/x`` where ``x != 0``, else 0 (MITgcm's recip_hFac*, ini_masks_etc.F:467-479).

    The denominator is selected before the division (double where, as
    geometry/metric_bundle._positive_reciprocal): with a single where, a
    differentiable ``x`` that is 0 somewhere (hFac under a geometry control) gets
    ``0 * (-1/0**2)`` = NaN from the unselected branch in the backward pass.  The
    forward value is the same ``1.0 / x`` on every nonzero entry.
    Differs from ``metric_bundle._positive_reciprocal`` only in the test (a
    negative ``x`` is inverted here, zeroed there) and from
    ``_vector_invariant_topology.nonzero_reciprocal`` only in the double where.
    """
    nonzero = x != 0
    denominator = torch.where(nonzero, x, torch.ones_like(x))
    return torch.where(nonzero, 1.0 / denominator, torch.zeros_like(x))


def _side_drag_closed_fractions(
    grid: "StructuredCGrid",
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    hFacZ: torch.Tensor,
    *,
    region_fill: FillSpec = FillSpec.ZERO,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Face and corner fractions of the no-slip side-drag closed fraction.

    A ``NONLIN_FRSURF`` MITgcm build forms ``h0FacW - hFacZ`` with the resting
    ``h0FacW``/``h0FacS`` (``mom_u_sidedrag.F:107-108``,
    ``mom_v_sidedrag.F:104-105``) and, under ``no_slip_sides`` with
    ``nonlinFreeSurf>0``, passes as ``hFacZ`` the plain four-neighbour ``MIN``
    of the resting faces (``mom_vecinv.F:311-323``, ``mom_fluxform.F:293-305``).
    So the closed fraction does not follow the free surface; only the drag's
    ``recip_hFac`` does.  A grid whose hFac is static (:attr:`MaskSet.resting_hfac`
    is ``None``) has ``h0Fac = hFac``: the caller's current fractions come back
    unchanged.  ``region_fill`` is the caller's corner halo fill.
    """
    resting = grid.masks.resting_hfac
    if resting is None:
        return hFacW, hFacS, hFacZ
    h0FacW = resting[1].to(dtype=hFacW.dtype, device=hFacW.device)
    h0FacS = resting[2].to(dtype=hFacS.dtype, device=hFacS.device)
    h0FacZ = _compute_hFacZ_4neighbor(
        h0FacW, h0FacS, grid, hZoption=0, region_fill=region_fill, smooth=False,
    )
    return h0FacW, h0FacS, h0FacZ


def _harmonic_viscosity_length_scales(
    recip_dx_sq: torch.Tensor,
    recip_dy_sq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT useAreaViscLength=False harmonic length scales."""
    l2 = 2.0 / (recip_dx_sq + recip_dy_sq)
    l3 = l2.pow(1.5)
    return l2, l3


def _grid_clamp_length_scales(
    grid: "StructuredCGrid",
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``L2_D`` ``[ny, nx]`` and ``L2_Z`` ``[ny+1, nx+1]`` of the grid+dt clamps.

    ``mom_init_fixed.F:96-129`` with ``useAreaViscLength=.FALSE.`` (the
    default, ``set_defaults.F:207``): ``2/(recip_dxF**2+recip_dyF**2)`` at D
    points and ``2/(recip_dxV**2+recip_dyU**2)`` at Z points.
    ``mom_calc_visc.F:385-388`` scales the harmonic clamp by ``L2`` and
    :519-531 the biharmonic clamp by ``L2**2``.
    """
    dxF = grid.metrics.dxF_for(dtype, device)
    dyF = grid.metrics.dyF_for(dtype, device)
    dxV = grid.metrics.dxV_for(dtype, device)
    dyU = grid.metrics.dyU_for(dtype, device)
    l2_d, _ = _harmonic_viscosity_length_scales(
        (1.0 / dxF).pow(2),
        (1.0 / dyF).pow(2),
    )
    l2_z, _ = _harmonic_viscosity_length_scales(
        (1.0 / dxV).pow(2),
        (1.0 / dyU).pow(2),
    )
    return l2_d, l2_z


def _strain_metric_flags(grid, *, dtype, device) -> tuple[bool, bool]:
    """Static U/V stencil choices, prepared eagerly in the consuming precision.

    True selects the metric stencil (``dyF/dxF``, ``dxV/dyU``, ``recip_rAw`` /
    ``recip_rAs``, as written in mom_u/v_x/yviscflux.F and mom_fluxform.F);
    False selects the row stencil (per-row ``dxC``, scalar ``dy``), which is
    exact only for uniform Cartesian spacing with ``rAw = dyG dxC`` (``rAs =
    dxG dyC``).  Non-uniform Cartesian spacing keeps those area products, so
    it is routed to the metric stencil by the spacing test.

    The memo belongs to the grid, so sharing a viscosity module between
    grids cannot reuse another grid's verdict. A direct eager operator call
    can populate a missing entry; model steps prepare it before tracing.
    """
    metrics = grid.metrics
    def build():
        uniform = metrics.is_uniform_x and metrics.is_uniform_y
        return (
            not uniform or not torch.equal(
                metrics.rAw_for(dtype, device),
                metrics.dyG_for(dtype, device) * metrics.dxC_for(dtype, device),
            ),
            not uniform or not torch.equal(
                metrics.rAs_for(dtype, device),
                metrics.dxG_for(dtype, device) * metrics.dyC_for(dtype, device),
            ),
        )
    return runtime_memo_constant(
        grid, ("strain_metric_flags", dtype, str(torch.device(device))), build,
    )


def _divvort_recip_hfac_and_masks(
    grid: "StructuredCGrid",
    hFacC: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memoize-once recip_hFacC/W/S + maskW/maskS (Site 2).

    Pure elementwise functions of the grid hFac masks, hence a per-grid
    constant *while hFac is static*.  A cache hit returns the exact tensors
    the recompute path produces (bitwise-identical by construction).  Keyed
    weakly on the grid; the inner key carries dtype/device and the source
    ``nz`` (the unit-hFac fall-back sizes hFac by ``u``'s vertical extent,
    so ``nz`` disambiguates it).  ``LOCOMGCM_CACHE_CONSTANTS=0`` -- and a grid
    whose hFac is rewritten per step, see :func:`_hfac_memo_enabled` --
    recomputes every call.
    """
    def _compute() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        recip_hFacC = _positive_reciprocal(hFacC)
        recip_hFacW = _positive_reciprocal(hFacW)
        recip_hFacS = _positive_reciprocal(hFacS)
        maskW = (hFacW > 0).to(dtype=dtype)
        maskS = (hFacS > 0).to(dtype=dtype)
        return recip_hFacC, recip_hFacW, recip_hFacS, maskW, maskS

    # Switch off: recompute in the caller's graph.  ``get_or_build`` would
    # run ``_build`` below, whose no_grad drops the hFac graph of a
    # nonlinear free surface or a geometry control.
    if not _constant_cache_enabled():
        return _compute()
    bypass = _hfac_memo_bypass_reason(grid)
    if bypass is not None:
        _record_hfac_memo_bypass(
            "divvort_recip_hfac",
            "HorizontalDiffusion._divvort_recip_hfac",
            bypass,
        )
        return _compute()

    inner_key = (dtype, str(device), int(hFacC.shape[-3]))

    def _build():
        # Constants: detach from autograd; no_grad changes no arithmetic.
        with torch.no_grad():
            return _compute()

    return constant_cache.get_or_build(
        ("HorizontalDiffusion._divvort_recip_hfac", inner_key),
        _build,
        variant="divvort_recip_hfac",
        site="HorizontalDiffusion._divvort_recip_hfac",
        owner=grid,
    )


def _divvort_corner_hfacz(
    grid: "StructuredCGrid",
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Memoize-once the divvort-budget corner hFacZ (Site 1, viscosity).

    Geometry constant (periodic-wrap 4-neighbour MIN + cube valence-3 corner
    override) *while hFac is static*.  A cache hit returns the exact
    recompute tensor.  Keyed weakly on the grid; inner key carries
    dtype/device, the source ``nz`` and whether a distributed step ctx is
    attached (the corner override reads it).  ``LOCOMGCM_CACHE_CONSTANTS=0`` --
    and a grid whose hFac is rewritten per step, see
    :func:`_hfac_memo_enabled` -- recomputes every call.
    """
    def _compute() -> torch.Tensor:
        # The periodic-wrap four-neighbour reduction of the vector-invariant
        # consumer, same exchanges and same geometry mode.
        return _compute_hFacZ_4neighbor(hFacW, hFacS, grid, region_fill=FillSpec.WRAP)

    # Switch off: recompute in the caller's graph (see _divvort_recip_hfac).
    if not _constant_cache_enabled():
        return _compute()
    bypass = _hfac_memo_bypass_reason(grid)
    if bypass is not None:
        _record_hfac_memo_bypass(
            "divvort_corner_hfacz",
            "HorizontalDiffusion._divvort_corner_hfacz",
            bypass,
        )
        return _compute()

    ctx_present = getattr(grid, "_cube_step_ctx", None) is not None
    inner_key = (dtype, str(device), int(hFacW.shape[-3]), ctx_present)

    def _build():
        # Constant: detach from autograd; no_grad changes no arithmetic.
        with torch.no_grad():
            return _compute()

    return constant_cache.get_or_build(
        ("HorizontalDiffusion._divvort_corner_hfacz", inner_key),
        _build,
        variant="divvort_corner_hfacz",
        site="HorizontalDiffusion._divvort_corner_hfacz",
        owner=grid,
    )


def _leith_length_scales(
    grid: "StructuredCGrid",
    *,
    dxF: torch.Tensor,
    dyF: torch.Tensor,
    dxV: torch.Tensor,
    dyU: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    ny: int,
    nx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memoize-once the Leith/Smag harmonic length scales (Site 3).

    ``L^2 = 2/(dx^-2 + dy^-2)`` and ``L^3 = (L^2)^1.5`` at D (dxF,dyF) and
    Z (dxV,dyU) points.  Pure function of the horizontal metrics, which
    ``MetricSet._cached_tensor`` already serves as per-(dtype,device) static
    tensors, so every step re-derived the same bits through 4 reciprocals,
    4 squares, 2 sums and 2 ``pow(1.5)``.  A cache hit returns the exact
    recompute tensors.  Keyed weakly on the grid; inner key carries
    dtype/device/ny/nx.  ``LOCOMGCM_CACHE_CONSTANTS=0`` recomputes every call.
    """
    def _compute() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # D-point (centre): useAreaViscLength=F harmonic mean.
        recip_dxF_sq = (1.0 / dxF).pow(2)
        recip_dyF_sq = (1.0 / dyF).pow(2)
        l2_d, l3_d = _harmonic_viscosity_length_scales(
            recip_dxF_sq.view(1, 1, ny, nx),
            recip_dyF_sq.view(1, 1, ny, nx),
        )
        # Z-point (corner).
        recip_dxV_sq = (1.0 / dxV).pow(2)
        recip_dyU_sq = (1.0 / dyU).pow(2)
        l2_z, l3_z = _harmonic_viscosity_length_scales(
            recip_dxV_sq.view(1, 1, ny + 1, nx + 1),
            recip_dyU_sq.view(1, 1, ny + 1, nx + 1),
        )
        return l2_d, l3_d, l2_z, l3_z

    if not _constant_cache_enabled():
        return _compute()

    inner_key = (dtype, str(device), int(ny), int(nx))

    def _build():
        # Constant: detach from autograd; no_grad changes no arithmetic.
        with torch.no_grad():
            return _compute()

    return constant_cache.get_or_build(
        ("HorizontalDiffusion._leith_length_scales", inner_key),
        _build,
        variant="leith_length_scales",
        site="HorizontalDiffusion._leith_length_scales",
        owner=grid,
    )


@dataclass(frozen=True)
class DivVortGeometry:
    """Metrics and wet fractions of one div-vort evaluation, read in one place.

    Metrics are the grid's 2-D ``[ny(+1), nx(+1)]`` tensors; ``hFac*`` and their
    reciprocals are ``[nz, ...]`` (unit fractions when the grid has none);
    ``maskW``/``maskS`` are ``hFac > 0`` in the evaluation dtype; ``maskInC`` is
    the OBCS interior mask or ``None`` when the grid has none.  The corner
    ``hFacZ`` is not here: it is built after the curl halos are exchanged.
    """

    nx: int
    ny: int
    dyG: torch.Tensor
    dxG: torch.Tensor
    dxC: torch.Tensor
    dyC: torch.Tensor
    dxF: torch.Tensor
    dyF: torch.Tensor
    dxV: torch.Tensor
    dyU: torch.Tensor
    recip_rA: torch.Tensor
    recip_rAz: torch.Tensor
    hFacC: torch.Tensor
    hFacW: torch.Tensor
    hFacS: torch.Tensor
    recip_hFacC: torch.Tensor
    recip_hFacW: torch.Tensor
    recip_hFacS: torch.Tensor
    maskW: torch.Tensor
    maskS: torch.Tensor
    maskInC: torch.Tensor | None


def divvort_geometry(
    grid: "StructuredCGrid",
    *,
    nz: int,
    dtype: torch.dtype,
    device: torch.device,
) -> DivVortGeometry:
    """Read the div-vort metrics and wet fractions of ``grid``.

    ``nz`` sizes the unit fractions of a grid without hFac.
    """
    nx = grid.nx
    ny = grid.ny

    # Grid metrics
    dyG = grid.metrics.dyG_for(dtype, device)
    dxG_row = grid.metrics.dxG_for(dtype, device)
    dxC_row = grid.metrics.dxC_for(dtype, device)
    dyC = grid.metrics.dyC_for(dtype, device)
    dxF = grid.metrics.dxF_for(dtype, device)
    dyF = grid.metrics.dyF_for(dtype, device)
    dxV = grid.metrics.dxV_for(dtype, device)
    dyU = grid.metrics.dyU_for(dtype, device)

    cell_area = grid.metrics.cell_area_for(dtype, device)  # [ny, nx]
    recip_rA = 1.0 / cell_area.clamp(min=1e-30)

    # hFac fields
    if grid.masks.hFacC is not None:
        hFacC = _hfac_c(grid, dtype, device, site="horizontal_diffusion._hfac_c")  # [nz, ny, nx]
        hFacW = _hfac_w(grid, dtype, device, site="horizontal_diffusion._hfac_w")  # [nz, ny, nx+1]
        hFacS = _hfac_s(grid, dtype, device, site="horizontal_diffusion._hfac_s")  # [nz, ny+1, nx]
    else:
        hFacC = torch.ones(nz, ny, nx, dtype=dtype, device=device)
        hFacW = torch.ones(nz, ny, nx + 1, dtype=dtype, device=device)
        hFacS = torch.ones(nz, ny + 1, nx, dtype=dtype, device=device)

    recip_hFacC, recip_hFacW, recip_hFacS, maskW, maskS = (
        _divvort_recip_hfac_and_masks(
            grid, hFacC, hFacW, hFacS, dtype, device
        )
    )

    maskInC = None
    if grid.masks.maskInC is not None:
        maskInC = grid.masks.maskInC.to(dtype=dtype, device=device)

    rAz_row = grid.metrics.rAz_for(dtype, device)
    recip_rAz = 1.0 / rAz_row.clamp(min=1e-30)
    return DivVortGeometry(
        nx=nx,
        ny=ny,
        dyG=dyG,
        dxG=dxG_row,
        dxC=dxC_row,
        dyC=dyC,
        dxF=dxF,
        dyF=dyF,
        dxV=dxV,
        dyU=dyU,
        recip_rA=recip_rA,
        recip_rAz=recip_rAz,
        hFacC=hFacC,
        hFacW=hFacW,
        hFacS=hFacS,
        recip_hFacC=recip_hFacC,
        recip_hFacW=recip_hFacW,
        recip_hFacS=recip_hFacS,
        maskW=maskW,
        maskS=maskS,
        maskInC=maskInC,
    )
