"""Narrow-waist 2-D metric bundle for geometry-neutral operators.

An operator reads metrics ONLY as uniform 2-D tensors; the 1D-row vs 2D-face
distinction (one name, two different objects) is resolved at bundle
construction, never inside the operator.

``TracerMetricBundle`` is a dtype/device-resolved SNAPSHOT of the horizontal
metrics + masks that horizontal/vertical tracer advection consumes.  Every field
is pre-shaped to broadcast against a tracer-like tensor ``[*lead, nz, ny±1, nx±1]``:

- region (``StructuredCGrid``): horizontal metrics have NO leading face dim
  (``[ny±1, nx±1]``) and right-broadcast over ``[batch, nz, ...]``;
- cube (``CubedSphereGrid``): horizontal metrics carry the face dim and an
  inserted singleton vertical axis (``[6, 1, ny±1, nx±1]``) so they broadcast
  over ``[6, nz, ...]``.

Masks (``hFac*``) already carry the vertical axis in ``grid.masks`` and are used
verbatim.  ``recip_volume = 1/(cell_area·drF·hFacC)`` is computed identically for
both geometries (matches BOTH the regional ``_multidim_advective_tendency`` and
the cube ``gad_advection_first_order_cube`` current arithmetic → byte-identical
when each path is rewired to read the bundle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _unsqueeze_face(t: torch.Tensor, *, is_cube: bool) -> torch.Tensor:
    """Insert the singleton vertical axis after the leading face dim on cube.

    Cube horizontal metric ``[6, ny±1, nx±1]`` -> ``[6, 1, ny±1, nx±1]`` so it
    broadcasts over a ``[6, nz, ny±1, nx±1]`` tracer.  Region ``[ny±1, nx±1]``
    is returned unchanged (right-broadcasts over ``[batch, nz, ...]``).
    """
    if is_cube:
        return t.unsqueeze(1)
    return t


@dataclass(frozen=True)
class TracerMetricBundle:
    """dtype/device-resolved 2-D metric + mask snapshot for tracer advection."""

    is_cube: bool
    nz: int
    ny: int
    nx: int

    # Horizontal metrics, pre-shaped to broadcast vs ``[*lead, nz, ny±1, nx±1]``.
    layer_thickness: torch.Tensor   # drF   [nz, 1, 1]
    dyG: torch.Tensor               # u-face zonal edge   [ny, nx+1] | [6,1,ny,nx+1]
    dxG: torch.Tensor               # v-face merid edge    [ny+1, nx] | [6,1,ny+1,nx]
    cell_area: torch.Tensor         # rA center            [ny, nx]   | [6,1,ny,nx]
    # NOTE: recip_dxC/recip_dyC are narrow-waist PLACEHOLDER fields — DST3 consumes
    # neither (region self-computes in tracer_flux.py; cube DST3 needs a tile-layout-aware exchange
    # the bundle does not yet provide). Kept for contract symmetry / future first-order rewiring.
    recip_dxC: torch.Tensor         # 1/dxC u-face (X CFL) [ny, nx+1] | [6,1,ny,nx+1]
    recip_dyC: torch.Tensor         # 1/dyC v-face (Y CFL) [ny+1, nx] | [6,1,ny+1,nx]

    # Masks (carry the vertical axis already): [nz,...] region | [6,nz,...] cube.
    hFacW: torch.Tensor | None      # [nz, ny, nx+1]   | [6, nz, ny, nx+1]
    hFacS: torch.Tensor | None      # [nz, ny+1, nx]   | [6, nz, ny+1, nx]
    hFacC: torch.Tensor | None      # [nz, ny, nx]     | [6, nz, ny, nx]
    recip_hFacC: torch.Tensor       # 1/hFacC where>0  (else 0)
    c_mask: torch.Tensor            # (hFacC>0) as dtype
    recip_volume: torch.Tensor      # 1/(cell_area·drF·hFacC) where>0 (else 0)

    # OBCS interior masks ([ny±1, nx±1] region | per-face cube), None if absent.
    maskInC: torch.Tensor | None
    maskInW: torch.Tensor | None
    maskInS: torch.Tensor | None


def _maybe(value: Any, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor | None:
    if value is None:
        return None
    return value.to(dtype=dtype, device=device)


def build_tracer_metric_bundle(
    grid: Any,
    dtype: torch.dtype,
    device: torch.device | str,
) -> TracerMetricBundle:
    """Build the tracer-advection metric snapshot from ``grid`` metrics + masks.

    Uses only the complete 2-D ``*_for`` accessors.  Uniform regional metrics
    retain the same expanded values as the former broadcast representation,
    while vector Cartesian metrics preserve both horizontal axes.
    """
    is_cube = getattr(grid, "face_count", 1) == 6
    nz, ny, nx = int(grid.nz), int(grid.ny), int(grid.nx)
    metrics = grid.metrics
    masks = grid.masks

    drF = grid.layer_thickness_for(dtype, device).view(nz, 1, 1)

    dyG = _unsqueeze_face(metrics.dyG_for(dtype, device), is_cube=is_cube)
    dxG = _unsqueeze_face(metrics.dxG_for(dtype, device), is_cube=is_cube)
    cell_area = _unsqueeze_face(metrics.cell_area_for(dtype, device), is_cube=is_cube)

    dxC = metrics.dxC_for(dtype, device)
    dyC = metrics.dyC_for(dtype, device)
    recip_dxC = _unsqueeze_face(
        torch.where(dxC != 0, 1.0 / dxC, torch.zeros_like(dxC)), is_cube=is_cube
    )
    recip_dyC = _unsqueeze_face(
        torch.where(dyC != 0, 1.0 / dyC, torch.zeros_like(dyC)), is_cube=is_cube
    )

    hFacW = _maybe(getattr(masks, "hFacW", None), dtype, device)
    hFacS = _maybe(getattr(masks, "hFacS", None), dtype, device)
    hFacC = _maybe(getattr(masks, "hFacC", None), dtype, device)

    if hFacC is not None:
        recip_hFacC = _positive_reciprocal(hFacC)
        c_mask = (hFacC > 0).to(dtype=dtype)
        # cell_area broadcasts to [*, nz, ny, nx]; drF is [nz,1,1]; hFacC carries nz.
        local_volume = cell_area * drF * hFacC
        recip_volume = _positive_reciprocal(local_volume)
    else:
        # Mask-free fallback (synthetic grids): volume = cell_area·drF.
        recip_hFacC = torch.ones((), dtype=dtype, device=device)
        c_mask = torch.ones((), dtype=dtype, device=device)
        vol = cell_area * drF
        recip_volume = _positive_reciprocal(vol)

    maskInC = _maybe(getattr(masks, "maskInC", None), dtype, device)
    maskInW = _maybe(getattr(masks, "maskInW", None), dtype, device)
    maskInS = _maybe(getattr(masks, "maskInS", None), dtype, device)

    return TracerMetricBundle(
        is_cube=is_cube,
        nz=nz,
        ny=ny,
        nx=nx,
        layer_thickness=drF,
        dyG=dyG,
        dxG=dxG,
        cell_area=cell_area,
        recip_dxC=recip_dxC,
        recip_dyC=recip_dyC,
        hFacW=hFacW,
        hFacS=hFacS,
        hFacC=hFacC,
        recip_hFacC=recip_hFacC,
        c_mask=c_mask,
        recip_volume=recip_volume,
        maskInC=maskInC,
        maskInW=maskInW,
        maskInS=maskInS,
    )


# ---------------------------------------------------------------------------
# Vertical-shear metric bundle (vector-invariant -w·∂u/∂z)
# ---------------------------------------------------------------------------


def _positive_reciprocal(metric: torch.Tensor) -> torch.Tensor:
    """Positive-only reciprocal preserving MIT dry/closed metric semantics.

    Differs from ``viscosity_geometry._recip`` only in the test: ``> 0`` maps a
    negative metric to 0 instead of inverting it.
    """
    positive = metric > 0
    # Select the denominator before division: an inactive 1/0 would otherwise
    # produce NaN in the active-geometry pullback despite a zero output seed.
    denominator = torch.where(positive, metric, torch.ones_like(metric))
    return torch.where(positive, 1.0 / denominator, torch.zeros_like(metric))


def _cube_exchanged_area(rA: torch.Tensor, grid: Any, *, location: str) -> torch.Tensor:
    """Cross-face-exchanged cell area for the cube native-absent fallback.

    Mirrors the legacy ``_cube_vertical_shear`` halo of ``rA`` (extent x±1 for U,
    y±1 for V, REPLICATE) so the adjacent-average fallback reproduces the old
    ``_cube_vertical_shear_area_reciprocals`` byte-for-byte.
    """
    from .exchange import FillSpec, HaloExtentV1, exchange

    if location == "U":
        ext = HaloExtentV1(
            west_left_extent=1, east_right_extent=1,
            south_bottom_extent=0, north_top_extent=0,
        )
    else:
        ext = HaloExtentV1(
            west_left_extent=0, east_right_extent=0,
            south_bottom_extent=1, north_top_extent=1,
        )
    return exchange(rA, grid=grid, location=location, extent=ext, fill=FillSpec.REPLICATE)


@dataclass(frozen=True)
class VertShearMetricBundle:
    """dtype/device-resolved metric + mask snapshot for vertical-shear advection.

    Sibling of :class:`TracerMetricBundle` (NOT a subclass).  Pre-shapes every
    field so the geometry-neutral ``vertical_shear`` core reads uniform tensors
    and never branches on geometry.  All vertical-velocity inputs are MIT
    upward-positive at the operator boundary; the ``rkSign = -1`` convention is
    applied later inside ``_vertical_shear_combine_all_z`` (NOT here).

    Face-area reciprocals are built differently for each geometry. Reusing
    one geometry's reciprocal in another changes the operator:

    - ``recip_rAs`` (V-face):  region = ``1 / rAz_row[:ny]`` (MITgcm's
      vorticity-point row area ``rAz`` *is* the correct ``rAs`` at v-points —
      "same-name-different-thing"; it is NOT a ``cell_area`` average and NOT the
      cube native ``rAs``).  cube = native ``face_geometry["rAs"]``.
    - ``recip_rAw`` (U-face):  region = adjacent ``cell_area`` average with the
      first/last column copied from the cell area (periodic-reuse).  cube =
      native ``face_geometry["rAw"]``.

    Cross-feeding either reciprocal produces ~10% face-area error at seams
    (historically the 1.8e-6 vStar residual).  The build branches on
    ``grid.face_count == 6``.

    All face metrics carry the FULL face extent (``recip_rAw``: ``nx+1`` u-faces,
    ``recip_rAs``: ``ny+1`` v-faces) so the core's extended-frame whole-z
    arithmetic indexes them directly.  The outer-face carve (which faces the
    region leaves zero) is a SEPARATE 0/1 ``u_face_keep`` / ``v_face_keep`` mask
    derived from ``grid.boundary_topology`` so the core stays branch-free.

    Field shapes (region | cube):

    - ``rA``           center area    ``[ny, nx]``        | ``[6, ny, nx]``
    - ``maskC``        (hFacC>0)      ``[nz, ny, nx]``    | ``[6, nz, ny, nx]`` | None
    - ``recip_rAw``    1/U-face area  ``[ny, nx+1]``      | ``[6, ny, nx+1]``
    - ``recip_rAs``    1/V-face area  ``[ny+1, nx]``      | ``[6, ny+1, nx]``
    - ``recip_drF``    1/layer thick  ``[nz]``
    - ``recip_hFacW``  1/hFacW        ``[nz, ny, nx+1]``  | ``[6, nz, ny, nx+1]`` | None
    - ``recip_hFacS``  1/hFacS        ``[nz, ny+1, nx]``  | ``[6, nz, ny+1, nx]`` | None
    - ``u_face_keep``  0/1 U-face     ``[ny, nx+1]``      | ``[6, ny, nx+1]``
    - ``v_face_keep``  0/1 V-face     ``[ny+1, nx]``      | ``[6, ny+1, nx]``
    """

    is_cube: bool
    nz: int
    ny: int
    nx: int

    rA: torch.Tensor                 # center area  [ny, nx] | [6, ny, nx]
    maskC: torch.Tensor | None       # (hFacC>0)    [nz, ny, nx] | [6, nz, ny, nx]
    recip_rAw: torch.Tensor          # 1/U-face area  [ny, nx+1] | [6, ny, nx+1]
    recip_rAs: torch.Tensor          # 1/V-face area  [ny+1, nx] | [6, ny+1, nx]
    recip_drF: torch.Tensor          # 1/layer thickness  [nz]
    recip_hFacW: torch.Tensor | None  # 1/hFacW  [nz, ny, nx+1] | [6, nz, ny, nx+1]
    recip_hFacS: torch.Tensor | None  # 1/hFacS  [nz, ny+1, nx] | [6, nz, ny+1, nx]
    u_face_keep: torch.Tensor        # 0/1 U-face keep mask  [ny, nx+1] | [6, ny, nx+1]
    v_face_keep: torch.Tensor        # 0/1 V-face keep mask  [ny+1, nx] | [6, ny+1, nx]


def _region_vertshear_recip_rAw(rA: torch.Tensor) -> torch.Tensor:
    """Region U-face area reciprocal, ``[ny, nx+1]`` (extended-frame faces).

    Reproduces ``_regional_vertical_shear_area_reciprocals`` byte-for-byte at the
    interior + periodic-reuse faces:

    - interior face ``i`` (1..nx-1): ``0.5*(rA[:, i] + rA[:, i-1])``;
    - face 0:  ``rA[:, 0]`` (the old ``rAw[:, 0] = rA[:, 0]`` copy-first-col);
    - face nx: ``rA[:, 0]`` (periodic-x reuses the face-0 metric ``recip_rAw[:, 0]``).

    The face-0 / face-nx values are only USED when ``u_face_keep`` keeps them
    (periodic-x); for closed-x they are masked out, so their exact value is
    immaterial — but we still set them to the periodic-consistent value.
    """
    ny, nx = rA.shape[-2], rA.shape[-1]
    rAw = rA.new_empty((ny, nx + 1))
    rAw[:, 0] = rA[:, 0]
    rAw[:, 1:nx] = 0.5 * (rA[:, 1:nx] + rA[:, 0 : nx - 1])
    rAw[:, nx] = rA[:, 0]
    return _positive_reciprocal(rAw)


def _region_vertshear_recip_rAw_distributed(
    rA: torch.Tensor, grid: Any
) -> torch.Tensor:
    """Distributed region U-face area reciprocal, ``[ny, nx+1]``.

    A distributed rank's west/east seam U-faces need the NEIGHBOUR cell area in
    the adjacent average; the single-card ``_region_vertshear_recip_rAw`` copies
    ``rA[:, 0]`` at the outer faces instead.  Halo-fill ``rA`` (x±1, REPLICATE on
    a real neighbour fetches the neighbour column) and take the adjacent average
    over the FULL extended frame, byte-for-byte matching the legacy distributed
    builder ``_regional_vertical_shear_face_wbars_distributed`` (recip_rAw_all =
    ``0.5*(rA_ext_u[1:nx+2] + rA_ext_u[0:nx+1])``).
    """
    from .exchange import FillSpec, HaloExtentV1, exchange

    nx = rA.shape[-1]
    ext_u = HaloExtentV1(
        west_left_extent=1, east_right_extent=1,
        south_bottom_extent=0, north_top_extent=0,
    )
    rA_ext_u = exchange(rA, grid=grid, location="U", extent=ext_u, fill=FillSpec.REPLICATE,
                        constant=True)
    rAw = 0.5 * (rA_ext_u[..., 1 : nx + 2] + rA_ext_u[..., 0 : nx + 1])
    return _positive_reciprocal(rAw)


def _region_vertshear_recip_rAs(
    rAz_row: torch.Tensor, *, nx: int, ny: int
) -> torch.Tensor:
    """Region V-face area reciprocal, ``[ny+1, nx]`` (extended-frame faces).

    The regional V-face area is ``rAz_row`` (vorticity-point row
    area used AS ``rAs``), NOT a ``cell_area`` average.  Reproduces
    ``_regional_vertical_shear_face_wbars`` byte-for-byte:

    - face ``j`` (0..ny-1): ``1 / rAz_row[j]``;
    - face ny: ``1 / rAz_row[0]`` (periodic-y reuses the face-0 metric, which the
      old code took from ``recip_rAs_all[0] = 1/rAz_row[0]``).

    Broadcast to ``[ny+1, nx]`` (constant along x for a non-rotated grid).
    """
    recip_row = _positive_reciprocal(rAz_row)  # [ny+1]
    recip_faces = rAz_row.new_empty((ny + 1,))
    recip_faces[0:ny] = recip_row[0:ny]
    recip_faces[ny] = recip_row[0]
    return recip_faces.view(ny + 1, 1).expand(ny + 1, nx).contiguous()


def _region_vertshear_recip_rAs_distributed(
    rAz_row: torch.Tensor, *, nx: int, ny: int
) -> torch.Tensor:
    """Distributed region V-face area reciprocal, ``[ny+1, nx]``.

    A distributed rank's north seam V-face (index ``ny``) needs its OWN local
    ``rAz_row[ny]`` (the localized full-length ``rAz_row`` carries the true
    north-face vorticity-point area), NOT the periodic face-0 reuse the
    single-card builder applies.  Reproduces the legacy distributed builder
    ``_regional_vertical_shear_face_wbars_distributed`` (``recip_rAs_faces =
    recip_rAs_all = 1/rAz_row``, full ``ny+1`` length) byte-for-byte.
    """
    recip_faces = _positive_reciprocal(rAz_row)  # [ny+1]; face ny == 1/rAz_row[ny]
    return recip_faces.view(ny + 1, 1).expand(ny + 1, nx).contiguous()


def _region_vertshear_face_keep(
    grid: Any, rA: torch.Tensor, *, nx: int, ny: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Region 0/1 outer-face keep masks from ``grid.boundary_topology`` + neighbors.

    The old region builder writes interior faces (U: 1..nx-1, V: 1..ny-1), keeps
    periodic outer faces, and (for a distributed grid) keeps the real-neighbor
    seam face.  Encode exactly that carve as 0/1 masks:

    - U interior faces 1..nx-1 -> keep; face 0 keep iff periodic_x OR west
      distributed-neighbor; face nx keep iff periodic_x OR east neighbor.
    - V interior faces 1..ny-1 -> keep; face 0 keep iff periodic_y OR south
      neighbor; face ny keep iff periodic_y OR north neighbor.

    (Distributed grids are delegated to the legacy distributed builder by the
    core, so these neighbor branches only matter if a future single-card path
    routes distributed seams here; kept for fidelity to the old carve.)
    """
    from locomgcm.dynamics._horizontal_diffusion_topology import _has_distributed_neighbor

    topo = getattr(grid, "boundary_topology", None)
    periodic_x = bool(topo.periodic_x) if topo is not None else False
    periodic_y = bool(topo.periodic_y) if topo is not None else False

    u_keep = rA.new_zeros((ny, nx + 1))
    u_keep[:, 1:nx] = 1.0
    if periodic_x or _has_distributed_neighbor(grid, "west"):
        u_keep[:, 0] = 1.0
    if periodic_x or _has_distributed_neighbor(grid, "east"):
        u_keep[:, nx] = 1.0

    v_keep = rA.new_zeros((ny + 1, nx))
    v_keep[1:ny, :] = 1.0
    if periodic_y or _has_distributed_neighbor(grid, "south"):
        v_keep[0, :] = 1.0
    if periodic_y or _has_distributed_neighbor(grid, "north"):
        v_keep[ny, :] = 1.0

    return u_keep, v_keep


def build_vertshear_metric_bundle(
    grid: Any,
    dtype: torch.dtype,
    device: torch.device | str,
) -> VertShearMetricBundle:
    """Build the vertical-shear metric snapshot from ``grid`` metrics + masks.

    Branches on ``grid.face_count == 6`` to reproduce the legacy region / cube
    face-area reciprocals byte-for-byte (see
    :class:`VertShearMetricBundle`).  ``recip_rAs`` region == ``1/rAz_row[:ny]``;
    cube == native ``face_geometry["rAs"]``.  NEVER cross-feed.
    """
    is_cube = getattr(grid, "face_count", 1) == 6
    nz, ny, nx = int(grid.nz), int(grid.ny), int(grid.nx)
    metrics = grid.metrics
    masks = grid.masks

    drF = grid.layer_thickness_for(dtype, device)  # [nz]
    recip_drF = 1.0 / drF.clamp(min=1e-30)

    hFacW = _maybe(getattr(masks, "hFacW", None), dtype, device)
    hFacS = _maybe(getattr(masks, "hFacS", None), dtype, device)
    hFacC = _maybe(getattr(masks, "hFacC", None), dtype, device)
    recip_hFacW = _positive_reciprocal(hFacW) if hFacW is not None else None
    recip_hFacS = _positive_reciprocal(hFacS) if hFacS is not None else None
    # maskC carries a leading dim so the z axis is at dim 1 for BOTH geometries
    # (region [1, nz, ny, nx] vs cube [6, nz, ny, nx]) — the core's k-1 center
    # mask shift and halo exchange then operate on the same axes branch-free.
    maskC = (hFacC > 0).to(dtype=dtype) if hFacC is not None else None
    if maskC is not None and not is_cube and maskC.ndim == 3:
        maskC = maskC.unsqueeze(0)

    rA = metrics.cell_area_for(dtype, device)

    if is_cube:
        # Rank-local face count: on a multi-card localized grid (``localize_cube_grid``)
        # the metrics/masks are scattered to the rank's owned faces while
        # ``grid.face_count`` stays 6, so the face-varying tensors here are
        # already shaped ``[n_local, ...]``.  Derive ``nf`` from the (scattered)
        # cell-area tensor when it carries a face dim — exactly the htdiff
        # ``nf = tracer.shape[0]`` pattern — falling back to the rank context's
        # owned-face count, then ``grid.face_count`` (single-rank, ==6 → byte-id).
        ctx = getattr(grid, "_cube_step_ctx", None)
        if rA.ndim == 3:
            nf = int(rA.shape[0])
        elif ctx is not None:
            nf = len(ctx.local_face_ids)
        else:
            nf = grid.face_count
        if rA.ndim != 3:
            rA = rA.unsqueeze(0).expand(nf, ny, nx).contiguous()
        face_geometry = getattr(metrics, "face_geometry", None)
        rAw_native = None
        rAs_native = None
        if face_geometry is not None:
            rAw_native = face_geometry.get("rAw")
            rAs_native = face_geometry.get("rAs")
        # Cube native U/V face areas (cs32 loader: rAw [6,ny,nx+1], rAs [6,ny+1,nx]).
        # At cube seams these are NOT a neighbour cell-area average and vertical
        # shear normalizes by recip_rAw/recip_rAs directly.  When native areas are
        # absent (synthetic grids only — production cs32 / six_face always carry
        # them), fall back to the adjacent-average of the CROSS-FACE-EXCHANGED cell
        # areas (NOT the un-exchanged copy-first-col), byte-for-byte matching the
        # legacy ``_cube_vertical_shear_area_reciprocals`` fallback.
        if rAw_native is not None:
            recip_rAw = _positive_reciprocal(rAw_native.to(dtype=dtype, device=device))
        else:
            rA_ext_u = _cube_exchanged_area(rA, grid, location="U")
            rAw = 0.5 * (rA_ext_u[..., 1 : nx + 2] + rA_ext_u[..., 0 : nx + 1])
            recip_rAw = _positive_reciprocal(rAw)
        if rAs_native is not None:
            recip_rAs = _positive_reciprocal(rAs_native.to(dtype=dtype, device=device))
        else:
            rA_ext_v = _cube_exchanged_area(rA, grid, location="V")
            rAs = 0.5 * (rA_ext_v[..., 1 : ny + 2, :] + rA_ext_v[..., 0 : ny + 1, :])
            recip_rAs = _positive_reciprocal(rAs)
        # Cube: cross-face exchange fills every face -> keep all faces.  Use the
        # rank-local face count ``nf`` (== ``grid.face_count`` single-rank, byte-id;
        # == n_local on a localized multi-card grid) so the keep mask matches the
        # rank-local wbar tensors.
        u_face_keep = rA.new_ones((nf, ny, nx + 1))
        v_face_keep = rA.new_ones((nf, ny + 1, nx))
    else:
        if metrics.del_x is None:
            from locomgcm.dynamics._horizontal_diffusion_topology import (
                _has_any_distributed_neighbor,
            )

            rAz_row = metrics.rAz_row_for(dtype, device)
            if _has_any_distributed_neighbor(grid):
                recip_rAw = _region_vertshear_recip_rAw_distributed(rA, grid)
                recip_rAs = _region_vertshear_recip_rAs_distributed(
                    rAz_row, nx=nx, ny=ny
                )
            else:
                recip_rAw = _region_vertshear_recip_rAw(rA)
                recip_rAs = _region_vertshear_recip_rAs(
                    rAz_row, nx=nx, ny=ny
                )
        else:
            recip_rAw = _positive_reciprocal(metrics.rAw_for(dtype, device))
            recip_rAs = _positive_reciprocal(metrics.rAs_for(dtype, device))
        u_face_keep, v_face_keep = _region_vertshear_face_keep(
            grid, rA, nx=nx, ny=ny
        )

    return VertShearMetricBundle(
        is_cube=is_cube,
        nz=nz,
        ny=ny,
        nx=nx,
        rA=rA,
        maskC=maskC,
        recip_rAw=recip_rAw,
        recip_rAs=recip_rAs,
        recip_drF=recip_drF,
        recip_hFacW=recip_hFacW,
        recip_hFacS=recip_hFacS,
        u_face_keep=u_face_keep,
        v_face_keep=v_face_keep,
    )


# ---------------------------------------------------------------------------
# Vector-invariant momentum (Coriolis / vorticity-flux) metric bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MomentumMetricBundle:
    """dtype/device-resolved metric snapshot for vector-invariant momentum.

    Sibling of :class:`VertShearMetricBundle` (NOT a subclass).  Pre-shapes the
    geometry metrics that the Coriolis / vorticity-flux U/V leaves consume so
    ``evaluate`` reads ONE bundle instead of re-deriving the same metrics
    inside four separate functions.

    THE byte-identity invariant (the one retained region/cube fork):
    the region leaves are **factored-form** (``dxG_row[j]·(v_r+v_l)``) and apply
    the per-row metric as a ``[ny, 1]`` / ``[1, nx]`` broadcast column against a
    full-extent face tensor.  The cube leaves are **per-cell-form**
    (``v_r·dxG_r + v_l·dxG_l``) on genuinely 2-D rotated metrics.  An IEEE
    multiply does NOT distribute over an add, so a single per-cell kernel CANNOT
    reproduce the region factored result bit-for-bit (probed: ~3e-11 abs).  The
    uniform regional fast path retains that factored arithmetic, while variable
    Cartesian axes consume the complete 2-D fields carried by this bundle.
    Cube metrics remain **pre-haloed 2-D** tensors supplied by the halo path;
    the two are NEVER cross-fed.

    Region fields (``is_cube=False``) are sourced directly from the complete
    2-D ``*_for`` accessors:

    - ``dxG``  ``[ny+1, nx]``
    - ``dxC``  ``[ny, nx+1]``
    - ``dyG``  ``[ny, nx+1]``
    - ``dyC``  ``[ny+1, nx]``

    Cube fields (``is_cube=True``) are intentionally ``None`` — NOT a deferred
    TODO.  Unlike the vertical-shear bundle (whose cube metrics are static and
    flow through :func:`build_vertshear_metric_bundle` + a ``HaloExchange``), the
    cube Coriolis/vorticity-flux metrics ``dxG``/``dxC``/``dyG``/``dyC`` MUST be
    cross-face-haloed in lock-step with the per-step velocity halo so the leaf's
    per-cell stencil indexes velocity and metric on the same extended frame.
    That velocity-coupled exchange is built every step in
    :func:`vector_invariant_halo_kwargs_for_grid` (cube metrics memoized once in
    ``_haloed_static_metrics``) and delivered to the unified core leaves via the
    ``v_ext``/``dxG_2d`` ``halo_kw`` kwargs — the ``v_ext is not None`` branch in
    ``_vorticity_flux_u`` / ``_planetary_only_*``.  A static bundle cannot carry
    that per-step-sliced frame, so the cube branch carries NO metric (
    verified: the cube evaluate path is unchanged and byte-identical CPU+GPU on
    cs32x15, max|d|=0; Gate C 380-step also max|d|=0).
    """

    is_cube: bool
    nz: int
    ny: int
    nx: int

    # Complete regional face metrics; None on cube (cube metrics are halo-coupled).
    dxG: torch.Tensor | None
    dxC: torch.Tensor | None
    dyG: torch.Tensor | None
    dyC: torch.Tensor | None


def build_momentum_metric_bundle(
    grid: Any,
    dtype: torch.dtype,
    device: torch.device | str,
) -> MomentumMetricBundle:
    """Build the momentum-advection metric snapshot from ``grid`` metrics.

    Branches on ``grid.face_count == 6``.  The region branch retains complete
    ``dxG_for`` / ``dxC_for`` / ``dyG_for`` / ``dyC_for`` fields so variable
    Cartesian axes cannot collapse to a row or scalar.  The cube branch carries
    NO metric (fields ``None``): cube metrics are velocity-coupled per-step and
    flow through ``vector_invariant_halo_kwargs_for_grid``, not this static
    bundle (see :class:`MomentumMetricBundle`).
    """
    is_cube = getattr(grid, "face_count", 1) == 6
    nz, ny, nx = int(grid.nz), int(grid.ny), int(grid.nx)
    metrics = grid.metrics

    if is_cube:
        return MomentumMetricBundle(
            is_cube=True,
            nz=nz,
            ny=ny,
            nx=nx,
            dxG=None,
            dxC=None,
            dyG=None,
            dyC=None,
        )

    return MomentumMetricBundle(
        is_cube=False,
        nz=nz,
        ny=ny,
        nx=nx,
        dxG=metrics.dxG_for(dtype, device),
        dxC=metrics.dxC_for(dtype, device),
        dyG=metrics.dyG_for(dtype, device),
        dyC=metrics.dyC_for(dtype, device),
    )
