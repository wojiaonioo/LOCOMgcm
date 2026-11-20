"""Localize global regional runtime objects for horizontal decomposition.

Vertical velocity boundary fields are sliced without sign conversion: callers
must preserve the producer/consumer convention at the runtime hand-off.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace

import torch

from defgcm.distributed.layout import ParallelLayout, build_cartesian_neighbors
from defgcm.boundaries.open import SideBoundaryTensorSet
from defgcm.boundaries.tidal import OBCSTidalConfig, TidalComponent
from defgcm.dynamics.surface_flux import PrescribedSurfaceFluxSource
from defgcm.geometry.grid import StructuredCGrid
from defgcm.geometry.masks import MaskSet
from defgcm.geometry.metrics import MetricSet
from defgcm.io.era5 import ERA5BulkSource, ERA5WindSource
from defgcm.io.exf_source import ExfBulkSource, FieldSeries
from defgcm.io.runner.builders.obcs_bridge import BoundaryBundleSource
from defgcm.state import PrognosticState


def localize_structured_grid(
    grid: StructuredCGrid,
    layout: ParallelLayout,
) -> StructuredCGrid:
    """Return a rank-local grid with sliced metrics/masks and neighbours.

    The vertical coordinate is kept intact.  Horizontal center quantities are
    sliced to ``[ny_local, nx_local]``.  U/V staggered quantities include the
    extra east/north face: ``[ny_local, nx_local+1]`` and
    ``[ny_local+1, nx_local]``.
    """

    _check_layout_matches_grid(grid, layout)
    periodic_x = grid.boundary_topology.periodic_x
    periodic_y = grid.boundary_topology.periodic_y
    return replace(
        grid,
        nx=layout.nx,
        ny=layout.ny,
        metrics=_localize_metrics(grid.metrics, layout),
        masks=_localize_masks(grid.masks, layout),
        distributed_neighbors=build_cartesian_neighbors(
            nPx=layout.nPx,
            nPy=layout.nPy,
            rank=layout.rank,
            periodic_x=periodic_x,
            periodic_y=periodic_y,
        ),
        parallel_layout=layout,
    )


def localize_prognostic_state(
    state: PrognosticState,
    layout: ParallelLayout,
) -> PrognosticState:
    """Slice a global regional prognostic state to a rank-local owned tile."""

    y = slice(layout.y_start, layout.y_stop)
    x = slice(layout.x_start, layout.x_stop)
    y_v = slice(layout.y_start, layout.y_stop + 1)
    x_u = slice(layout.x_start, layout.x_stop + 1)
    return state.with_tensors(
        eta=_slice_center(state.eta.data, y, x),
        u=_slice_center(state.u.data, y, x_u),
        v=_slice_center(state.v.data, y_v, x),
        temp=_slice_center(state.temp.data, y, x),
        salt=_slice_center(state.salt.data, y, x),
    )


def localize_ic_tensors(
    tensors: dict[str, torch.Tensor],
    layout: ParallelLayout,
) -> dict[str, torch.Tensor]:
    """Slice the canonical IC tensor map to a rank-local owned C-grid tile."""

    localized: dict[str, torch.Tensor] = {}
    center_names = {"temperature", "salinity", "temp", "salt", "eta", "bathymetry"}
    for name, tensor in tensors.items():
        if name in center_names:
            localized[name] = _slice_center(tensor, *_center_slices(layout))
        elif name == "u":
            localized[name] = _slice_center(tensor, *_u_slices(layout))
        elif name == "v":
            localized[name] = _slice_center(tensor, *_v_slices(layout))
        else:
            localized[name] = tensor
    return localized


def localize_wind_source(
    source: ERA5WindSource | None,
    layout: ParallelLayout,
) -> ERA5WindSource | None:
    """Slice cell-centered ERA5 wind-stress forcing to a rank-local tile."""

    if source is None:
        return None
    y, x = _center_slices(layout)
    return ERA5WindSource(
        times=source.times,
        time_values=source.time_values,
        tau_x=_slice_center(source.tau_x, y, x),
        tau_y=_slice_center(source.tau_y, y, x),
        period_seconds=source.period_seconds,
    )


def localize_surface_flux_source(
    source: PrescribedSurfaceFluxSource | None,
    layout: ParallelLayout,
) -> PrescribedSurfaceFluxSource | None:
    """Slice prescribed Qnet/EmPmR fields to a rank-local tracer tile."""

    if source is None:
        return None
    y, x = _center_slices(layout)
    return PrescribedSurfaceFluxSource(
        times=source.times,
        time_values=source.time_values,
        heat_flux=_slice_center(source.heat_flux, y, x),
        freshwater_flux=_slice_center(source.freshwater_flux, y, x),
        period_seconds=source.period_seconds,
    )


def localize_center_field(tensor: torch.Tensor, layout: ParallelLayout) -> torch.Tensor:
    """Slice a static tracer-centred [..., ny, nx] field to a rank-local tile."""

    return _slice_center(tensor, *_center_slices(layout))


def localize_bulk_source(
    source: ERA5BulkSource | ExfBulkSource | None,
    layout: ParallelLayout,
) -> ERA5BulkSource | ExfBulkSource | None:
    """Slice model-grid bulk-formula forcing to a rank-local tile.

    The 7 ERA5 bulk fields are cell-centered [n_times, ny, nx]; ``lat``/``lon`` are the
    model-grid 1-D axes (length ny / nx). Mirrors :func:`localize_wind_source`
    (center slices), additionally slicing lat along y and lon along x so the
    axes stay consistent with the sliced fields.  An EXF source's records are
    [record, ny, nx] on the tracer grid; a constant field's empty record axis
    slices to the local shape its constant fills.
    """
    if source is None:
        return None
    y, x = _center_slices(layout)
    if isinstance(source, ExfBulkSource):
        return replace(source, **{
            item.name: replace(getattr(source, item.name),
                               records=_slice_center(getattr(source, item.name).records, y, x))
            for item in fields(source) if isinstance(getattr(source, item.name), FieldSeries)})
    sliced = {n: _slice_center(getattr(source, n), y, x) for n in source._FIELDS}
    return ERA5BulkSource(
        times=source.times,
        time_values=source.time_values,
        lat=source.lat[y].clone(memory_format=torch.contiguous_format),
        lon=source.lon[x].clone(memory_format=torch.contiguous_format),
        period_seconds=source.period_seconds,
        **sliced,
    )


def localize_boundary_bundle_source(
    source: BoundaryBundleSource,
    layout: ParallelLayout,
) -> BoundaryBundleSource:
    """Slice a global OBC bundle source to the physical sides owned by a rank.

    Internal rank seams are not open boundaries.  W/E sides are retained only
    by ranks touching the global west/east edge and are sliced along y; S/N
    sides are retained only by ranks touching the global south/north edge and
    are sliced along x.
    """

    specs = _local_boundary_side_specs(source.side_names, layout, source.valid_along)
    if not specs:
        return BoundaryBundleSource(
            times_seconds=source.times_seconds,
            side_names=(),
            fields={name: _empty_boundary_field(tensor) for name, tensor in source.fields.items()},
            valid_along={},
            forcing_period_seconds=source.forcing_period_seconds,
            forcing_cycle_seconds=source.forcing_cycle_seconds,
        )

    fields = {
        name: _localize_boundary_field(tensor, specs)
        for name, tensor in source.fields.items()
    }
    valid_along = {spec.side_name: spec.local_valid for spec in specs}
    return BoundaryBundleSource(
        times_seconds=source.times_seconds,
        side_names=tuple(spec.side_name for spec in specs),
        fields=fields,
        valid_along=valid_along,
        forcing_period_seconds=source.forcing_period_seconds,
        forcing_cycle_seconds=source.forcing_cycle_seconds,
    )


def localize_tidal_config(
    config: OBCSTidalConfig | None,
    layout: ParallelLayout,
) -> OBCSTidalConfig | None:
    """Slice OBCS tidal harmonics to the physical sides owned by a rank.

    W/E side tensors are indexed along global y and only retained by ranks on
    the global west/east edge.  S/N side tensors are indexed along global x and
    retained by ranks on the global south/north edge.  Internal rank seams are
    not open boundaries.
    """

    if config is None:
        return None
    return OBCSTidalConfig(
        components=[
            TidalComponent(
                period_seconds=component.period_seconds,
                u_tide_cs=_localize_tidal_side_set(component.u_tide_cs, layout),
                u_tide_sn=_localize_tidal_side_set(component.u_tide_sn, layout),
                v_tide_cs=_localize_tidal_side_set(component.v_tide_cs, layout),
                v_tide_sn=_localize_tidal_side_set(component.v_tide_sn, layout),
            )
            for component in config.components
        ]
    )


def _localize_metrics(metrics: MetricSet, layout: ParallelLayout) -> MetricSet:
    # Fail closed on native vector/staggered metrics: this constructor rebuilds only
    # the row-wise/legacy fields, so slicing a vector-Cartesian MetricSet here would
    # silently drop del_x/del_y and every native 2-D staggered area (rAw/rAs/...),
    # rerouting rAs_for consumers onto a wrong rAz_row broadcast.
    # ponytail: preserve-and-slice the native fields (+ distributed gather tests)
    # when a distributed vector-spacing case actually exists.
    _native_fields = (
        "del_x", "del_y", "dxF", "dyF", "dxG", "dyG", "dxC", "dyC",
        "dxV", "dyU", "rA", "rAw", "rAs", "rAz",
    )
    _present = [n for n in _native_fields if getattr(metrics, n, None) is not None]
    if _present:
        raise NotImplementedError(
            "distributed localization of native vector/staggered metrics is not "
            f"supported (would silently drop {_present}); use uniform scalar or "
            "spherical row-wise metrics, or add preserve-and-slice support"
        )
    y = slice(layout.y_start, layout.y_stop)
    x = slice(layout.x_start, layout.x_stop)
    y_v = slice(layout.y_start, layout.y_stop + 1)
    x_u = slice(layout.x_start, layout.x_stop + 1)
    return MetricSet(
        dx_scalar=metrics.dx_scalar,
        dy_scalar=metrics.dy_scalar,
        dx_center=_slice_center(metrics.dx_center, y, x),
        dy_center=_slice_center(metrics.dy_center, y, x),
        cell_area=_slice_center(metrics.cell_area, y, x),
        cell_volume=_slice_center(metrics.cell_volume, y, x),
        depth_center=_slice_center(metrics.depth_center, y, x),
        face_area_u=_slice_center(metrics.face_area_u, y, x_u),
        face_area_v=_slice_center(metrics.face_area_v, y_v, x),
        dxG_row=metrics.dxG_row[y].clone(memory_format=torch.contiguous_format),
        dxC_row=metrics.dxC_row[y].clone(memory_format=torch.contiguous_format),
        dyG_scalar_1d=metrics.dyG_scalar_1d,
        dxG_row_ext=metrics.dxG_row_ext[y_v].clone(memory_format=torch.contiguous_format),
        rAz_row=metrics.rAz_row[y_v].clone(memory_format=torch.contiguous_format),
        # Sliced from the global rows, like the $coriolis_profile_c/_g sentinels
        # (_canonical_helpers._slice_y_profile), never recomputed locally.
        yC_row=None if metrics.yC_row is None else metrics.yC_row[y],
        yG_row=None if metrics.yG_row is None else metrics.yG_row[y_v],
    )


def _localize_masks(masks: MaskSet, layout: ParallelLayout) -> MaskSet:
    y = slice(layout.y_start, layout.y_stop)
    x = slice(layout.x_start, layout.x_stop)
    y_v = slice(layout.y_start, layout.y_stop + 1)
    x_u = slice(layout.x_start, layout.x_stop + 1)
    return MaskSet(
        center=_slice_center(masks.center, y, x),
        u=_slice_center(masks.u, y, x_u),
        v=_slice_center(masks.v, y_v, x),
        hFacC=_slice_optional_center(masks.hFacC, y, x),
        hFacW=_slice_optional_center(masks.hFacW, y, x_u),
        hFacS=_slice_optional_center(masks.hFacS, y_v, x),
        maskInC=_slice_optional_center(masks.maskInC, y, x),
        maskInW=_slice_optional_center(masks.maskInW, y, x_u),
        maskInS=_slice_optional_center(masks.maskInS, y_v, x),
        # Geometry mode flag (depth-control corner hFacZ) is per-grid, not
        # per-rank; a localized MaskSet must keep it or the corner reduction
        # silently changes across ranks.
        smooth_abs_func_range=masks.smooth_abs_func_range,
    )


def _center_slices(layout: ParallelLayout) -> tuple[slice, slice]:
    return slice(layout.y_start, layout.y_stop), slice(layout.x_start, layout.x_stop)


def _u_slices(layout: ParallelLayout) -> tuple[slice, slice]:
    return slice(layout.y_start, layout.y_stop), slice(layout.x_start, layout.x_stop + 1)


def _v_slices(layout: ParallelLayout) -> tuple[slice, slice]:
    return slice(layout.y_start, layout.y_stop + 1), slice(layout.x_start, layout.x_stop)


def _slice_center(tensor: torch.Tensor, y: slice, x: slice) -> torch.Tensor:
    # Even a contiguous slice can retain global storage (notably a y-only
    # partition). Clone directly into the local contiguous allocation.
    return tensor[..., y, x].clone(memory_format=torch.contiguous_format)


def _slice_optional_center(
    tensor: torch.Tensor | None,
    y: slice,
    x: slice,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return _slice_center(tensor, y, x)


def _localize_tidal_side_set(
    values: SideBoundaryTensorSet,
    layout: ParallelLayout,
) -> SideBoundaryTensorSet:
    return SideBoundaryTensorSet(
        west=_localize_tidal_side_tensor(values.west, "west", layout),
        east=_localize_tidal_side_tensor(values.east, "east", layout),
        south=_localize_tidal_side_tensor(values.south, "south", layout),
        north=_localize_tidal_side_tensor(values.north, "north", layout),
    )


def _localize_tidal_side_tensor(
    tensor: torch.Tensor | None,
    side: str,
    layout: ParallelLayout,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if side == "west":
        if layout.tile_i != 0:
            return None
        along = slice(layout.y_start, layout.y_stop)
    elif side == "east":
        if layout.tile_i != layout.nPx - 1:
            return None
        along = slice(layout.y_start, layout.y_stop)
    elif side == "south":
        if layout.tile_j != 0:
            return None
        along = slice(layout.x_start, layout.x_stop)
    elif side == "north":
        if layout.tile_j != layout.nPy - 1:
            return None
        along = slice(layout.x_start, layout.x_stop)
    else:
        raise ValueError(f"unknown tidal side {side!r}")
    if tensor.shape[-1] < along.stop:
        raise ValueError(
            f"tidal {side} tensor length {tensor.shape[-1]} is shorter than "
            f"required global slice stop {along.stop}"
        )
    return tensor[..., along].clone(memory_format=torch.contiguous_format)


def _check_layout_matches_grid(grid: StructuredCGrid, layout: ParallelLayout) -> None:
    if layout.global_nx != grid.nx or layout.global_ny != grid.ny:
        raise ValueError(
            f"layout global shape {(layout.global_ny, layout.global_nx)} "
            f"does not match grid {(grid.ny, grid.nx)}"
        )
    if getattr(grid, "face_count", 1) != 1:
        raise NotImplementedError("distributed regional localization supports face_count=1")


@dataclass(frozen=True)
class _BoundarySideSpec:
    source_index: int
    side_name: str
    along_slice: slice
    local_valid: int


_SIDE_ALIASES = {
    "w": "W",
    "west": "W",
    "e": "E",
    "east": "E",
    "s": "S",
    "south": "S",
    "n": "N",
    "north": "N",
}


def _canonical_side_name(side_name: str) -> str | None:
    return _SIDE_ALIASES.get(str(side_name).lower())


def _local_boundary_side_specs(
    side_names: tuple[str, ...],
    layout: ParallelLayout,
    valid_along: Mapping[str, int] | None,
) -> tuple[_BoundarySideSpec, ...]:
    specs: list[_BoundarySideSpec] = []
    for index, side_name in enumerate(side_names):
        canonical = _canonical_side_name(side_name)
        if canonical is None:
            continue
        if canonical == "W":
            if layout.tile_i != 0:
                continue
            start, stop = layout.y_start, layout.y_stop
        elif canonical == "E":
            if layout.tile_i != layout.nPx - 1:
                continue
            start, stop = layout.y_start, layout.y_stop
        elif canonical == "S":
            if layout.tile_j != 0:
                continue
            start, stop = layout.x_start, layout.x_stop
        else:
            if layout.tile_j != layout.nPy - 1:
                continue
            start, stop = layout.x_start, layout.x_stop

        global_valid = _global_valid_along(side_name, canonical, layout, valid_along)
        valid_start = min(start, global_valid)
        valid_stop = min(stop, global_valid)
        local_valid = max(0, valid_stop - valid_start)
        if local_valid <= 0:
            continue
        specs.append(
            _BoundarySideSpec(
                source_index=index,
                side_name=side_name,
                along_slice=slice(start, stop),
                local_valid=local_valid,
            )
        )
    return tuple(specs)


def _global_valid_along(
    side_name: str,
    canonical: str,
    layout: ParallelLayout,
    valid_along: Mapping[str, int] | None,
) -> int:
    default = layout.global_ny if canonical in {"W", "E"} else layout.global_nx
    if valid_along is None:
        return default
    candidates = (
        side_name,
        canonical,
        canonical.lower(),
        {
            "W": "west",
            "E": "east",
            "S": "south",
            "N": "north",
        }[canonical],
    )
    for candidate in candidates:
        if candidate in valid_along:
            return int(valid_along[candidate])
    return default


def _empty_boundary_field(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 3:
        raise ValueError(
            "boundary fields must have shape (time, side, ..., along); "
            f"got {tuple(tensor.shape)}"
        )
    return tensor.new_empty((tensor.shape[0], 0, *tensor.shape[2:-1], 0))


def _localize_boundary_field(
    tensor: torch.Tensor,
    specs: tuple[_BoundarySideSpec, ...],
) -> torch.Tensor:
    if tensor.ndim < 3:
        raise ValueError(
            "boundary fields must have shape (time, side, ..., along); "
            f"got {tuple(tensor.shape)}"
        )
    max_along = max(spec.local_valid for spec in specs)
    out = tensor.new_zeros((tensor.shape[0], len(specs), *tensor.shape[2:-1], max_along))
    for out_index, spec in enumerate(specs):
        values = tensor[:, spec.source_index, ..., spec.along_slice]
        if values.shape[-1] < spec.local_valid:
            raise ValueError(
                f"boundary side {spec.side_name!r} slice {spec.along_slice} "
                f"has length {values.shape[-1]}, expected at least {spec.local_valid}"
            )
        out[:, out_index, ..., : spec.local_valid] = values[..., : spec.local_valid]
    return out.contiguous()
