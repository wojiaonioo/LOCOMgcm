"""Topology adapters for vector-invariant momentum assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from locomgcm.accel.constant_cache import constant_cache

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import HaloExtentV1
from locomgcm.operators.c_grid import curl_uv_to_corner

from ._cube_corner_dispatch import (
    _at_clamped,
    _cs_face_first,
    _cube_corner_name,
    _cube_corner_positions,
    _has_cube_corner_override,
)
from ._host_constants import host_profile_to_tensor
from .planetary_coriolis import spherical_coriolis
if TYPE_CHECKING:
    from locomgcm.geometry.cube_decomposition import CubeRankContext




def vector_invariant_halo_kwargs_for_grid(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
    ctx: "CubeRankContext | None" = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return pre-haloed velocity/metric kwargs for vorticity flux helpers.

    When *ctx* is provided (either directly or via ``grid._cube_step_ctx``) the
    three cube vector exchanges are routed to the distributed path
    (``exchange_cubed_sphere_vector_distributed``).  ctx=None and no grid
    attribute retains the single-rank path (byte-identical to the original).
    """
    if getattr(grid, "face_count", 1) != 6:
        return {}, {}

    # Allow ctx to be carried via grid when VIM.evaluate() cannot forward it.
    if ctx is None:
        ctx = getattr(grid, "_cube_step_ctx", None)

    from ._vector_seam_topology import exchange_vector_velocities_for_grid

    overlap = 2
    n = grid.nx
    uc, vc = exchange_vector_velocities_for_grid(
        u,
        v,
        grid,
        overlap,
        with_signs=True,
        ctx=ctx,
    )

    face_geometry = grid.metrics.face_geometry
    dtype, device = u.dtype, u.device

    dxCh, dyCh, dyGh, dxGh = _haloed_static_metrics(
        grid, face_geometry, overlap, ctx, dtype, device
    )

    def with_vertical_axis(metric: torch.Tensor) -> torch.Tensor:
        return metric.unsqueeze(1) if u.ndim == 4 else metric

    v_kwargs = {
        "v_ext": vc[..., overlap : overlap + n + 1, overlap - 1 : overlap + n + 1],
        "dxG_2d": with_vertical_axis(dxGh)[
            ..., overlap : overlap + n + 1, overlap - 1 : overlap + n + 1
        ],
        "dxC_2d": with_vertical_axis(dxCh)[
            ..., overlap : overlap + n, overlap : overlap + n + 1
        ],
    }
    u_kwargs = {
        "u_ext": uc[..., overlap - 1 : overlap + n + 1, overlap : overlap + n + 1],
        "dyG_2d": with_vertical_axis(dyGh)[
            ..., overlap - 1 : overlap + n + 1, overlap : overlap + n + 1
        ],
        "dyC_2d": with_vertical_axis(dyCh)[
            ..., overlap : overlap + n + 1, overlap : overlap + n
        ],
    }
    return u_kwargs, v_kwargs


def _exchange_static_metrics(
    grid: StructuredCGrid,
    face_geometry: dict,
    overlap: int,
    ctx: "CubeRankContext | None",
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the cube vector halo exchange for the four static metrics.

    This is the original (uncached) recompute path, factored out verbatim so
    both the toggle-OFF fall-back and the memoize-once cache call into the same
    code.  Returns ``(dxCh, dyCh, dyGh, dxGh)`` (haloed).
    """
    from locomgcm.geometry.exchange import exchange_cubed_sphere_vector

    if ctx is not None:
        from locomgcm.geometry.exchange import exchange_cubed_sphere_vector_distributed

        # On a localized grid (ctx attached) face_geometry is ALREADY scattered
        # to local faces by localize_cube_grid, so feed it directly — do NOT
        # scatter again (a second scatter_faces would index a global face id into
        # an [n_local,...] tensor and crash).  Unified contract: ctx present ⟺
        # the grid (and its metrics) is already rank-local.
        dxC_local = face_geometry["dxC"].to(dtype=dtype, device=device)
        dyC_local = face_geometry["dyC"].to(dtype=dtype, device=device)
        dyG_local = face_geometry["dyG"].to(dtype=dtype, device=device)
        dxG_local = face_geometry["dxG"].to(dtype=dtype, device=device)

        dxCh, dyCh = exchange_cubed_sphere_vector_distributed(
            dxC_local,
            dyC_local,
            grid=grid,
            extent=HaloExtentV1.uniform(overlap),
            ctx=ctx,
            with_signs=False,
        )
        dyGh, dxGh = exchange_cubed_sphere_vector_distributed(
            dyG_local,
            dxG_local,
            grid=grid,
            extent=HaloExtentV1.uniform(overlap),
            ctx=ctx,
            with_signs=False,
        )
    else:
        dxCh, dyCh = exchange_cubed_sphere_vector(
            face_geometry["dxC"].to(dtype=dtype, device=device),
            face_geometry["dyC"].to(dtype=dtype, device=device),
            grid=grid,
            extent=HaloExtentV1.uniform(overlap),
            with_signs=False,
        )
        dyGh, dxGh = exchange_cubed_sphere_vector(
            face_geometry["dyG"].to(dtype=dtype, device=device),
            face_geometry["dxG"].to(dtype=dtype, device=device),
            grid=grid,
            extent=HaloExtentV1.uniform(overlap),
            with_signs=False,
        )
    return dxCh, dyCh, dyGh, dxGh


def _haloed_static_metrics(
    grid: StructuredCGrid,
    face_geometry: dict,
    overlap: int,
    ctx: "CubeRankContext | None",
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memoize-once the haloed static metrics (Site 3).

    The four haloed metrics depend only on (grid, dtype, device, ctx-presence),
    so a cache hit returns the exact tensors produced by the original recompute
    path — bitwise-identical by construction.  ``LOCOMGCM_CACHE_CONSTANTS=0`` disables
    the cache and recomputes every call.
    """
    inner_key = (dtype, str(device), ctx is not None, overlap)

    def _build() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Constants: keep them out of the autograd graph (requires_grad=False).
        # Forward values are unaffected (no_grad changes no arithmetic), so the
        # cached tensors stay bitwise-equal to the recompute path.
        with torch.no_grad():
            return _exchange_static_metrics(
                grid, face_geometry, overlap, ctx, dtype, device
            )

    # The cold build runs outside
    # traced code inside ConstantCache so the memo never hands a CUDAGraph
    # output to the next step; the gate bypass, weakref eviction and the
    # warm-reuse ignition record live in the shared helper.
    return constant_cache.get_or_build(
        inner_key,
        _build,
        variant="halo_static_metrics",
        site="_vector_invariant_topology._haloed_static_metrics",
        owner=grid,
    )


def resolve_coriolis_profile_tensor(
    grid: StructuredCGrid,
    coriolis_profile_g: tuple[float, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """The 1-D vorticity-row Coriolis profile as a tensor. **Eager entry point.**

    Compiled execution constraint: the ``constant_cache``
    key ``(id(coriolis_profile_g), ...)`` is
    ``id() with unsupported args -- Dynamo doesn't know how to trace id() call
    with args (TupleVariable(length=161),)``, i.e. an unconditional
    graph break every step, warm hit or not.  ``id()`` is fine *here* because
    this function is only reached from the eager pre-step
    (``ImplicitFreeSurfaceStepper.prepare_step_host`` →
    ``VectorInvariantMomentumAdvection.prepare_host_constants``) or from an
    eager caller; the compiled step reads the resolved tensor back through a
    ``(dtype, device)``-keyed memo on the module that owns the profile.

    Same key, same builder, same tensor as the inline form it replaces.
    """
    return constant_cache.get_or_build(
        (tuple(coriolis_profile_g), "coriolis_profile_g", dtype, str(device)),
        lambda: host_profile_to_tensor(
            coriolis_profile_g, dtype=dtype, device=device
        ),
        variant="host_profile", site="vector_invariant_topology.coriolis_profile_g",
        owner=grid,
    )


def corner_coriolis_field_for_grid(
    u: torch.Tensor,
    grid: StructuredCGrid,
    *,
    coriolis_parameter: float,
    coriolis_profile_g: tuple[float, ...] | None,
    omega: float,
    profile_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the vorticity-point Coriolis field for regional or cube grids.

    The cube builds ``2*omega*sin(yG)`` from its corner latitudes; regional
    grids take ``coriolis_profile_g`` or ``coriolis_parameter``.

    ``profile_tensor`` is the pre-step's answer for ``coriolis_profile_g`` at
    this dtype/device (see :func:`resolve_coriolis_profile_tensor`).  Passing it
    keeps the whole body traceable; omitting it resolves the profile here, which
    is the eager legacy path.
    """
    expected_ny_plus_1 = grid.ny + 1
    if getattr(grid, "face_count", 1) == 6:
        yG = grid.metrics.face_geometry["yG"].to(dtype=u.dtype, device=u.device)
        result = spherical_coriolis(yG, omega=omega)
        return result.unsqueeze(1) if u.ndim == 4 else result

    if coriolis_profile_g is not None:
        if len(coriolis_profile_g) != expected_ny_plus_1:
            raise ValueError(
                f"coriolis_profile_g length {len(coriolis_profile_g)} "
                f"does not match ny+1 = {expected_ny_plus_1}"
            )
        # Grid-constant profile, eager cold build + traced warm reuse.
        f_g_1d = (
            profile_tensor
            if profile_tensor is not None
            else resolve_coriolis_profile_tensor(
                grid, coriolis_profile_g, dtype=u.dtype, device=u.device
            )
        )
        return f_g_1d.view(expected_ny_plus_1, 1).expand(
            expected_ny_plus_1,
            grid.nx + 1,
        )

    f_scalar = u.new_tensor(coriolis_parameter)
    return f_scalar.expand(expected_ny_plus_1, grid.nx + 1)


def metric_at(
    metric: torch.Tensor | None,
    *,
    face: int,
    j: int,
    i: int,
    default: torch.Tensor,
) -> torch.Tensor:
    if metric is None:
        return default
    data = metric.to(dtype=default.dtype, device=default.device)
    if data.ndim >= 3 and data.shape[0] > face:
        data = data[face]
    elif data.ndim >= 2 and data.shape[0] > face and data.shape[0] == 6:
        data = data[face]
    if data.ndim >= 2:
        jj = min(max(j, 0), data.shape[-2] - 1)
        ii = min(max(i, 0), data.shape[-1] - 1)
        return data[..., jj, ii]
    if data.ndim == 1:
        jj = min(max(j, 0), data.shape[0] - 1)
        return data[jj]
    return data


def metric_source(
    grid: StructuredCGrid,
    name: str,
    fallback: torch.Tensor,
) -> torch.Tensor:
    face_geometry = getattr(grid.metrics, "face_geometry", None)
    if face_geometry is not None and name in face_geometry:
        return face_geometry[name]
    return fallback


def relvort3_metric_sources(
    grid: StructuredCGrid,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dxC = metric_source(grid, "dxC", grid.metrics.dxC_for(dtype, device))
    dyC = metric_source(grid, "dyC", grid.metrics.dyC_for(dtype, device))
    rAz = metric_source(grid, "rAz", grid.metrics.rAz_for(dtype, device))
    return dxC, dyC, rAz


def nonzero_reciprocal(value: torch.Tensor) -> torch.Tensor:
    """Reciprocal with exact zero mapped to zero.

    Single where, unlike ``viscosity_geometry._recip``: a ``value`` that carries
    a gradient and is 0 somewhere back-propagates NaN (``0 * -1/0**2``).
    """
    return torch.where(value != 0, 1.0 / value, torch.zeros_like(value))


def relvort3_recip_rAz(
    rAz_metric: torch.Tensor,
    *,
    face: int,
    j: int,
    i: int,
    default: torch.Tensor,
) -> torch.Tensor:
    rAz = metric_at(rAz_metric, face=face, j=j, i=i, default=default)
    return nonzero_reciprocal(rAz)


def apply_corner_relvort3_override(
    zeta: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
) -> torch.Tensor:
    if not _has_cube_corner_override(grid):
        return zeta

    result = zeta.clone()
    result_faces = _cs_face_first(result, grid)
    u_faces = _cs_face_first(u, grid)
    v_faces = _cs_face_first(v, grid)
    dxC_metric, dyC_metric, rAz_metric = relvort3_metric_sources(
        grid,
        zeta.dtype,
        zeta.device,
    )
    dx_default = zeta.new_tensor(grid.dx)
    dy_default = zeta.new_tensor(grid.dy)
    area_default = zeta.new_tensor(grid.dx * grid.dy)

    for face, local_face, j, i in _cube_corner_positions(grid):
        out_face = result_faces[local_face]
        u_face = u_faces[local_face]
        v_face = v_faces[local_face]
        corner = _cube_corner_name(grid, j, i)

        if corner == "sw":
            v_right = _at_clamped(v_face, j, i) * metric_at(
                dyC_metric, face=local_face, j=j, i=i, default=dy_default
            )
            u_north = _at_clamped(u_face, j, i) * metric_at(
                dxC_metric, face=local_face, j=j, i=i, default=dx_default
            )
            u_south = _at_clamped(u_face, j - 1, i) * metric_at(
                dxC_metric, face=local_face, j=j - 1, i=i, default=dx_default
            )
            numerator = (v_right - u_north) + u_south
        elif corner == "se":
            v_left = _at_clamped(v_face, j, i - 1) * metric_at(
                dyC_metric, face=local_face, j=j, i=i - 1, default=dy_default
            )
            u_north = _at_clamped(u_face, j, i) * metric_at(
                dxC_metric, face=local_face, j=j, i=i, default=dx_default
            )
            u_south = _at_clamped(u_face, j - 1, i) * metric_at(
                dxC_metric, face=local_face, j=j - 1, i=i, default=dx_default
            )
            if face == 1:
                numerator = (-u_north - v_left) + u_south
            elif face == 3:
                numerator = (-v_left + u_south) - u_north
            else:
                numerator = (u_south - u_north) - v_left
        elif corner == "nw":
            v_right = _at_clamped(v_face, j, i) * metric_at(
                dyC_metric, face=local_face, j=j, i=i, default=dy_default
            )
            u_north = _at_clamped(u_face, j, i) * metric_at(
                dxC_metric, face=local_face, j=j, i=i, default=dx_default
            )
            u_south = _at_clamped(u_face, j - 1, i) * metric_at(
                dxC_metric, face=local_face, j=j - 1, i=i, default=dx_default
            )
            if face == 0:
                numerator = (u_south + v_right) - u_north
            elif face == 2:
                numerator = (-u_north + u_south) + v_right
            else:
                numerator = (v_right - u_north) + u_south
        else:
            v_left = _at_clamped(v_face, j, i - 1) * metric_at(
                dyC_metric, face=local_face, j=j, i=i - 1, default=dy_default
            )
            u_north = _at_clamped(u_face, j, i) * metric_at(
                dxC_metric, face=local_face, j=j, i=i, default=dx_default
            )
            u_south = _at_clamped(u_face, j - 1, i) * metric_at(
                dxC_metric, face=local_face, j=j - 1, i=i, default=dx_default
            )
            if face % 2 == 0:
                numerator = (-u_north - v_left) + u_south
            else:
                numerator = (u_south - u_north) - v_left
        out_face[..., j, i] = relvort3_recip_rAz(
            rAz_metric, face=local_face, j=j, i=i, default=area_default
        ) * numerator

    return result


def relative_vorticity(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
) -> torch.Tensor:
    """Relative vorticity ζ at corner points — the unified vector-invariant entry.

    ONE narrow-waist entry returning the COMPLETE corner vorticity for every
    geometry in a single call.  The irreducible geometry fork lives in an
    ``is_cube`` reduction leaf inside
    :func:`~locomgcm.operators.c_grid.curl_uv_to_corner`, plus a
    corner-override injection here:

    - **Reduction leaf** (``curl_uv_to_corner`` dispatches on
      ``face_count == 6``): the cube leaf (``_curl_cube``) sums the circulation
      **per-cell** (``vdyc_here − vdyc_left``) on 2-D rotated face metrics with a
      cross-face vector halo; the region leaf sums it **factored**
      (``dyC·(v_r − v_l)``) on per-row metrics with a periodic/wall halo.  An
      IEEE multiply does not distribute over an add, so these CANNOT collapse to
      one bitwise leaf — the same FP-distributivity reason ``_vorticity_flux_u``
      keeps a cube sum4 / region sum2 fork.  ``_curl_cube`` is moreover
      **shared infrastructure** (momentum here + horizontal viscosity
      ``_horizontal_diffusion_topology``), so the leaf stays
      single-instance, not merged.

    - **Corner-override injection**: a single-process cube bakes the valence-3
      vertex override into ``_curl_cube``; a *localized* cube face
      (``face_count != 6`` carrying ``cube_corner_indices``) gets it via
      :func:`apply_corner_relvort3_override`.  Both are cube-vertex logic kept as
      two analytic overrides; this entry injects the applicable one.  A true region (planar) grid has no cube vertex, so the override
      is a no-op (``cube_corner_indices`` empty).

    Calls ``curl_uv_to_corner`` and ``apply_corner_relvort3_override``
    in that order through a shared entry point.
    """
    zeta = curl_uv_to_corner(u, v, grid)
    if getattr(grid, "face_count", 1) != 6:
        zeta = apply_corner_relvort3_override(zeta, u, v, grid)
    return zeta


def apply_uv_tendency_masks_for_grid(
    tend_u: torch.Tensor,
    tend_v: torch.Tensor,
    grid: StructuredCGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply topology-appropriate wet-face masks to U/V tendencies."""
    if getattr(grid, "face_count", 1) == 6:
        from ._vector_mask_topology import mask_vector_tendency_for_grid

        return mask_vector_tendency_for_grid(tend_u, tend_v, grid)

    if grid.masks.has_3d_masks:
        return (
            tend_u * grid.masks.u_float(dtype=tend_u.dtype, device=tend_u.device),
            tend_v * grid.masks.v_float(dtype=tend_v.dtype, device=tend_v.device),
        )

    return tend_u, tend_v
