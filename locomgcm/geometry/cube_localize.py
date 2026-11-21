"""Localize a global cubed-sphere grid to a rank's owned faces.

Cube analogue of ``locomgcm/distributed/localize.py::localize_structured_grid``.
The regional localizer slices the horizontal ``(y, x)`` plane; the cube localizer
slices the leading **face** dimension (``[6, ...] -> [n_local, ...]``) with
``scatter_faces``.  ``nx``/``ny``/``face_count`` are unchanged — face size is
preserved, only the number of faces a rank holds shrinks.

The returned grid carries the :class:`CubeRankContext` on ``_cube_step_ctx`` so
the cube halo exchanges (``exchange`` / ``exchange_velocity_pair`` /
``exchange_hfac_pair`` and the VIM metric assembly at
``_vector_invariant_topology.py:63``) route to the distributed P2P path.

Producer/consumer convention: ``hFacW``/``hFacS`` are taken as already
cross-seam reconciled (``_reconcile_cube_seam_hfac`` runs once at build inside
``from_csN_grid``).  ``scatter_faces`` only slices — it does NOT re-reconcile —
so the input MUST be a fully built (reconciled) grid.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from locomgcm.geometry.cube_decomposition import CubeRankContext, scatter_faces
from locomgcm.geometry.cubed_sphere_grid import CubedSphereGrid
from locomgcm.geometry.cubed_sphere_metric_builder import CubedSphereMetricSet
from locomgcm.geometry.masks import MaskSet
from locomgcm.geometry.metrics import MetricSet


def _scatter_opt(t: torch.Tensor | None, ctx: CubeRankContext) -> torch.Tensor | None:
    return None if t is None else scatter_faces(t, ctx)


def _localize_cube_metrics(metrics: MetricSet, ctx: CubeRankContext) -> MetricSet:
    """Scatter every face-varying metric tensor to the rank's owned faces.

    The 11 leading-face tensors are scattered; the 3 face-invariant scalars
    (``dx_scalar``/``dy_scalar``/``dyG_scalar_1d``) are carried through by
    ``replace``. ``_runtime_cache`` is an instance attribute, not a dataclass
    field; ``replace`` constructs a fresh empty cache, lazily rebuilt on first
    access.  When ``metrics`` is a
    :class:`CubedSphereMetricSet`, its 18 ``face_geometry`` values (each
    leading-face) are scattered too — they are the source of truth for the cube
    2-D metric accessors, so missing them would silently desync the operators.
    """
    updates: dict[str, object] = dict(
        dx_center=scatter_faces(metrics.dx_center, ctx),
        dy_center=scatter_faces(metrics.dy_center, ctx),
        cell_area=scatter_faces(metrics.cell_area, ctx),
        cell_volume=scatter_faces(metrics.cell_volume, ctx),
        depth_center=scatter_faces(metrics.depth_center, ctx),
        face_area_u=scatter_faces(metrics.face_area_u, ctx),
        face_area_v=scatter_faces(metrics.face_area_v, ctx),
        dxG_row=scatter_faces(metrics.dxG_row, ctx),
        dxC_row=scatter_faces(metrics.dxC_row, ctx),
        dxG_row_ext=scatter_faces(metrics.dxG_row_ext, ctx),
        rAz_row=scatter_faces(metrics.rAz_row, ctx),
    )
    if isinstance(metrics, CubedSphereMetricSet):
        updates["face_geometry"] = {
            name: scatter_faces(t, ctx) for name, t in metrics.face_geometry.items()
        }
    return replace(metrics, **updates)


def _localize_cube_masks(masks: MaskSet, ctx: CubeRankContext) -> MaskSet:
    """Scatter every face-varying mask; ``None`` optional fields stay ``None``."""
    return replace(
        masks,
        center=scatter_faces(masks.center, ctx),
        u=scatter_faces(masks.u, ctx),
        v=scatter_faces(masks.v, ctx),
        hFacC=_scatter_opt(masks.hFacC, ctx),
        hFacW=_scatter_opt(masks.hFacW, ctx),
        hFacS=_scatter_opt(masks.hFacS, ctx),
        maskInC=_scatter_opt(masks.maskInC, ctx),
        maskInW=_scatter_opt(masks.maskInW, ctx),
        maskInS=_scatter_opt(masks.maskInS, ctx),
    )




def _cube_local_grid_cache_enabled() -> bool:
    """Memoize the rank-local cube grid across steps (default ON).

    ``LOCOMGCM_CUBE_LOCAL_GRID_CACHE=0`` rebuilds it every step (A/B + bitwise check),
    the same toggle discipline as device-resident masks / fused kernels.  The
    enable decision is owned by the unified control plane
    (``locomgcm.accel``), not by this module."""
    from locomgcm.accel import gate

    return gate("cube_local_grid_cache").effective_value


def _record_local_grid_cache(event: str) -> None:
    """Cache record for LOCOMGCM_CUBE_LOCAL_GRID_CACHE (warm reuse == ignition)."""
    from locomgcm.accel import current_rank, record_cache_event

    record_cache_event(
        "cube_local_grid_cache",
        event=event,
        variant="rank_local_cube_grid",
        site="cube_localize.localize_cube_grid",
        backend="torch",
        rank=current_rank(),
    )


def _local_grid_key(grid, ctx):
    sample = grid.metrics.dx_center
    return (getattr(ctx, "rank", None), getattr(ctx, "decomposition", None),
            sample.dtype, str(sample.device))


def cached_local_cube_grid(grid, ctx):
    """Read the current context's existing owner without building a grid or a halo."""
    if not _cube_local_grid_cache_enabled():
        return None
    return getattr(grid, "_local_grid_memo", {}).get(_local_grid_key(grid, ctx))


def localize_cube_grid(grid: CubedSphereGrid, ctx: CubeRankContext) -> CubedSphereGrid:
    """Return a rank-local cube grid: masks/metrics scattered to ``ctx``'s faces.

    ``nx``/``ny``/``face_count`` (=6) and the global topology tables
    (``tile_connectivity``, ``cube_corner_indices``, ``vertical_coordinate``,
    ``boundary_topology``) are preserved unchanged.  The :class:`CubeRankContext`
    is attached on ``_cube_step_ctx`` so the front-door cube exchanges route to
    the distributed path.

    The cube stepper, implicit vertical solvers and runtime diagnostics share
    this entry point. At R=1, all six faces remain owned locally and the
    copied metric/mask values equal the input grid. Algorithm declarations
    (advection_topology/mit_tiling) remain unchanged for every rank count.
    ``cube_corner_indices`` retain GLOBAL face IDs; consumers map them through
    ``ctx.global_to_local`` before indexing a rank-local tensor.

    The rank-local grid is a per-(grid, context, communication session) CONSTANT (masks/metrics are
    time-invariant), so it is memoized — built once, reused every step.  Beyond
    skipping the per-step ``scatter_faces`` re-slice (~3.45 GB/step at cs768),
    this keeps the local grid object stable so the VIM static-metric halo memo
    it owns HITS instead of re-haloing the dxC/dyC/dyG/dxG cross-card
    exchange every step.  Bitwise-safe: reuse changes no values, and the momentum
    site re-sets ``_cached_w_interface`` after localize so no stale w is served.
    """
    if not _cube_local_grid_cache_enabled():
        return _build_local_cube_grid(grid, ctx)
    # The global grid owns its rank-local grids, keyed by values only.  A model
    # has one rank context; the cached local grid follows the global grid's
    # current communication session instead of keeping one per session.
    session = getattr(grid, "comm_session", None)
    memo = getattr(grid, "_local_grid_memo", None)
    if memo is None:
        memo = {}
        object.__setattr__(grid, "_local_grid_memo", memo)
    key = _local_grid_key(grid, ctx)
    hit = memo.get(key)
    if hit is not None:
        if getattr(hit, "comm_session", None) is not session:
            object.__setattr__(hit, "comm_session", session)
        # Warm reuse is the ignition event for a cache-class accelerator
        # (the cache-observation contract); recorded on the line that serves the memo,
        # not in the gate predicate above.
        _record_local_grid_cache("warm-reuse")
        return hit
    local = _build_local_cube_grid(grid, ctx)
    memo[key] = local
    _record_local_grid_cache("cold-construction")
    return local


def _build_local_cube_grid(
    grid: CubedSphereGrid, ctx: CubeRankContext
) -> CubedSphereGrid:
    """Build (scatter) the rank-local cube grid — the uncached core of
    :func:`localize_cube_grid`."""
    if grid.face_count != 6:
        raise ValueError("localize_cube_grid requires a 6-face cube grid")
    local = replace(
        grid,
        metrics=_localize_cube_metrics(grid.metrics, ctx),
        masks=_localize_cube_masks(grid.masks, ctx),
        # Preserve the source's advection topology and MIT tiling. The tile
        # schedule cuts each owned face into its ``mit_tiling`` tiles and runs
        # them on distributed halos of the tiling's width.
    )
    # CubedSphereGrid is frozen and has no ctx field; stash via object.__setattr__
    # (the same mechanism the cube stepper uses for _cached_w_interface).  VIM
    # reads this exact name at _vector_invariant_topology.py:63 — it is the ONLY
    # ctx attr in the repo (``_cube_ctx`` is a phantom name; do not introduce it).
    object.__setattr__(local, "_cube_step_ctx", ctx)
    # replace() omits this runtime binding; exchanges must retain its AD schedule.
    object.__setattr__(local, "comm_session", getattr(grid, "comm_session", None))
    return local
