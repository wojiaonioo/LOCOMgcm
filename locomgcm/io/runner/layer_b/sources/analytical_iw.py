"""Analytical internal-wave sources for the MITgcm verification case.

The ``experiments`` package is imported lazily inside the functions/methods
that use it (local imports at each call site) rather than at module top level,
so that importing this module — and the Layer-B source registry as a whole —
does not depend on the ``experiments`` tree. This keeps the io layering clean
for locomgcm-only installs / production runners (cf. ``tpxo9.py``).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np
import torch
import xarray as xr

from locomgcm.io.runner.layer_a import (
    BundleMeta,
    ConventionTag,
    DataBundle,
    LocationKind,
    ProvenanceStep,
    VariableSpec,
)

from .._registry import register_source
from .._types import DataQuery, SourceCapabilities, SourceContext, TimeCoverage


_IC_VARIABLES = ("temperature", "salinity", "eta", "u", "v", "bathymetry")
_OBC_VARIABLES = ("temperature", "salinity", "u", "v")

_CF_NAMES = {
    "temperature": "sea_water_potential_temperature",
    "salinity": "sea_water_salinity",
    "eta": "sea_surface_height_above_geoid",
    "u": "sea_water_x_velocity",
    "v": "sea_water_y_velocity",
    "bathymetry": "sea_floor_depth_below_geoid",
}

_UNITS = {
    "temperature": "degC",
    "salinity": "1e-3",
    "eta": "m",
    "u": "m s-1",
    "v": "m s-1",
    "bathymetry": "m",
}

_SIDE_ATTRS = {"W": "west", "E": "east", "S": "south", "N": "north"}


def _cfg(config: dict[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return default


def _grid_coords(grid: Any, *, nx: int, ny: int, nz: int, depth: float) -> dict[str, Any]:
    lat_off = np.arange(ny, dtype="float64") - 0.5 * (ny - 1)
    lon_off = np.arange(nx, dtype="float64") - 0.5 * (nx - 1)
    lat = float(grid.center_lat) + lat_off * float(grid.resolution_deg)
    lon = float(grid.center_lon) + lon_off * float(grid.resolution_deg)
    thicknesses = np.asarray(
        getattr(grid, "vertical_layer_thicknesses", ()) or (depth / nz,) * nz,
        dtype="float64",
    )
    if thicknesses.size != nz:
        thicknesses = np.full((nz,), depth / nz, dtype="float64")
    interfaces = np.concatenate([[0.0], np.cumsum(thicknesses)])
    z = 0.5 * (interfaces[:-1] + interfaces[1:])
    return {"latitude": lat, "longitude": lon, "z": z}


def _var_spec(name: str, dims: tuple[str, ...]) -> VariableSpec:
    return VariableSpec(
        name=name,
        cf_standard_name=_CF_NAMES[name],
        units=_UNITS[name],
        dtype="float64",
        dims=dims,
    )


def _as_tref(value: Any, *, nz: int) -> np.ndarray:
    if value is None:
        from experiments.alignment.mitgcm_internal_wave import build_tref

        return np.asarray(build_tref(nz), dtype="float64")
    tref = np.asarray(value, dtype="float64")
    if tref.shape != (nz,):
        raise ValueError(f"T_ref/Tref must have shape ({nz},), got {tref.shape}")
    return tref


def _boundary_source_class() -> type[Any]:
    from experiments.alignment import mitgcm_internal_wave as _iw

    return _iw.AnalyticalIWBoundarySource


def _time_seconds_and_coords(query: DataQuery) -> tuple[np.ndarray, Any]:
    extras = query.extras
    explicit_seconds = extras.get("time_seconds")
    explicit_times = extras.get("times")

    if explicit_seconds is not None:
        seconds = np.asarray(list(explicit_seconds), dtype="float64")
        if query.time_range is not None and isinstance(query.time_range[0], datetime):
            t0 = query.time_range[0]
            coords = np.array(
                [np.datetime64(t0 + timedelta(seconds=float(s)), "ns") for s in seconds]
            )
        else:
            coords = seconds.copy()
        return seconds, coords

    if explicit_times is not None:
        values = list(explicit_times)
        if not values:
            raise ValueError("AnalyticalIWBoundarySource: query.extras['times'] is empty")
        first = values[0]
        if isinstance(first, datetime):
            seconds = np.asarray(
                [(v - first).total_seconds() for v in values],
                dtype="float64",
            )
            coords = np.array([np.datetime64(v, "ns") for v in values])
        else:
            seconds = np.asarray(values, dtype="float64")
            coords = seconds.copy()
        return seconds, coords

    if query.time_range is None:
        return np.asarray([0.0], dtype="float64"), np.asarray([0.0], dtype="float64")

    t0, t1 = query.time_range
    if not isinstance(t0, datetime) or not isinstance(t1, datetime):
        raise TypeError("AnalyticalIWBoundarySource: time_range must contain datetimes")

    span = (t1 - t0).total_seconds()
    step = extras.get("dt_seconds", extras.get("interval_seconds"))
    if step is None:
        seconds = np.asarray([0.0] if span == 0.0 else [0.0, span], dtype="float64")
    else:
        step = float(step)
        if step <= 0.0:
            raise ValueError("AnalyticalIWBoundarySource: dt_seconds must be positive")
        seconds = np.arange(0.0, span + 0.5 * step, step, dtype="float64")
        if seconds[-1] < span:
            seconds = np.append(seconds, span)
    coords = np.array(
        [np.datetime64(t0 + timedelta(seconds=float(s)), "ns") for s in seconds]
    )
    return seconds, coords


@register_source("analytical_iw_ic")
class AnalyticalIWInitialConditionSource:
    """Generate internal-wave initial conditions by wrapping legacy builders."""

    capabilities = SourceCapabilities(
        produces=(LocationKind.GRIDDED_3D_OCEAN,),
        variables=_IC_VARIABLES,
        time_coverage_strategy=TimeCoverage.INSTANT,
        convention=ConventionTag.LOCOMGCM,
        requires_target_grid=True,
    )

    def __init__(self, ctx: SourceContext, **config: Any) -> None:
        self._ctx = ctx
        self._config = dict(config)

    def fetch(self, query: DataQuery) -> DataBundle:
        if query.location_kind != LocationKind.GRIDDED_3D_OCEAN:
            raise ValueError(
                "AnalyticalIWInitialConditionSource only produces GRIDDED_3D_OCEAN"
            )
        if query.target_grid is None:
            raise ValueError("AnalyticalIWInitialConditionSource: target_grid required")

        from experiments.alignment import mitgcm_internal_wave as _iw
        from experiments.alignment.mitgcm_internal_wave import (
            build_bathymetry,
            build_initial_temperature,
        )

        grid = query.target_grid
        config = self._config
        nx = int(_cfg(config, "nx", default=grid.nx))
        ny = int(_cfg(config, "ny", default=grid.ny))
        nz = int(_cfg(config, "nz", default=grid.nz))
        lx = float(_cfg(config, "LX", "lx", default=_iw.LX))
        depth = float(_cfg(config, "H", "depth", "total_depth", default=grid.depth_m))
        salt_value = float(_cfg(config, "S_ref", "salt_value", default=35.0))

        dx = lx / float(nx + 1)
        tref = _as_tref(_cfg(config, "T_ref", "Tref", "tref", default=None), nz=nz)
        temp = np.asarray(build_initial_temperature(nz, nx, tref), dtype="float64")
        if ny != temp.shape[2]:
            temp = np.repeat(temp, ny, axis=2)
        bathy = np.asarray(build_bathymetry(nx, dx), dtype="float64").reshape(1, nx)
        if ny != 1:
            bathy = np.repeat(bathy, ny, axis=0)

        coords = _grid_coords(grid, nx=nx, ny=ny, nz=nz, depth=depth)
        coords["time"] = [np.datetime64("2019-01-01", "ns")]

        requested = [v for v in query.variables if v in _IC_VARIABLES]
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        var_specs: list[VariableSpec] = []
        dims_3d = ("time", "z", "latitude", "longitude")
        dims_2d = ("time", "latitude", "longitude")
        dims_mesh = ("latitude", "longitude")

        for name in requested:
            if name == "temperature":
                data_vars[name] = (dims_3d, temp)
                var_specs.append(_var_spec(name, dims_3d))
            elif name == "salinity":
                data_vars[name] = (
                    dims_3d,
                    np.full((1, nz, ny, nx), salt_value, dtype="float64"),
                )
                var_specs.append(_var_spec(name, dims_3d))
            elif name == "eta":
                data_vars[name] = (dims_2d, np.zeros((1, ny, nx), dtype="float64"))
                var_specs.append(_var_spec(name, dims_2d))
            elif name in ("u", "v"):
                data_vars[name] = (dims_3d, np.zeros((1, nz, ny, nx), dtype="float64"))
                var_specs.append(_var_spec(name, dims_3d))
            elif name == "bathymetry":
                data_vars[name] = (dims_mesh, bathy)
                var_specs.append(_var_spec(name, dims_mesh))

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        return self._wrap(
            ds,
            var_specs,
            location_kind=LocationKind.GRIDDED_3D_OCEAN,
            time_range=query.time_range,
            target_grid_id=grid.grid_id,
            params={
                "wrapped": (
                    "build_bathymetry",
                    "build_tref",
                    "build_initial_temperature",
                ),
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "LX": lx,
                "H": depth,
            },
        )

    def _wrap(
        self,
        ds: xr.Dataset,
        var_specs: list[VariableSpec],
        *,
        location_kind: LocationKind,
        time_range: tuple[datetime, datetime] | None,
        target_grid_id: str | None,
        params: dict[str, Any],
    ) -> DataBundle:
        meta_no_prov = BundleMeta(
            schema_version="0.1.0",
            location_kind=location_kind,
            convention=self.capabilities.convention,
            variables=tuple(var_specs),
            time_range=time_range,
            target_grid_id=target_grid_id,
            provenance=(),
        )
        prov = ProvenanceStep(
            kind="source",
            name="AnalyticalIWInitialConditionSource",
            params=params,
            timestamp=datetime.utcnow(),
            bundle_hash="00000000",
        )
        meta = BundleMeta(
            schema_version=meta_no_prov.schema_version,
            location_kind=meta_no_prov.location_kind,
            convention=meta_no_prov.convention,
            variables=meta_no_prov.variables,
            time_range=meta_no_prov.time_range,
            target_grid_id=meta_no_prov.target_grid_id,
            provenance=(prov,),
            extras=meta_no_prov.extras,
        )
        return DataBundle(data=ds, meta=meta)


@register_source("analytical_iw_obcs")
class AnalyticalIWBoundaryDataSource:
    """Generate internal-wave OBC time series from the alignment boundary source."""

    capabilities = SourceCapabilities(
        produces=(LocationKind.BOUNDARY_TIMESERIES,),
        variables=_OBC_VARIABLES,
        time_coverage_strategy=TimeCoverage.ON_DEMAND,
        convention=ConventionTag.LOCOMGCM,
        requires_target_grid=True,
    )

    def __init__(self, ctx: SourceContext, **config: Any) -> None:
        self._ctx = ctx
        self._config = dict(config)

    def fetch(self, query: DataQuery) -> DataBundle:
        if query.location_kind != LocationKind.BOUNDARY_TIMESERIES:
            raise ValueError(
                "AnalyticalIWBoundaryDataSource only produces BOUNDARY_TIMESERIES"
            )
        if query.target_grid is None:
            raise ValueError("AnalyticalIWBoundaryDataSource: target_grid required")

        grid = query.target_grid
        config = self._config
        nx = int(_cfg(config, "nx", default=grid.nx))
        ny = int(_cfg(config, "ny", default=grid.ny))
        nz = int(_cfg(config, "nz", default=grid.nz))
        depth = float(_cfg(config, "H", "depth", "total_depth", default=grid.depth_m))
        seconds, time_coord = _time_seconds_and_coords(query)
        sides = tuple(query.extras.get("sides", config.get("sides", ("W", "E"))))
        n_along = int(query.extras.get("along", ny))

        source = self._legacy_source(nz=nz, ny=ny, depth=depth)
        requested = [v for v in query.variables if v in _OBC_VARIABLES]
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        var_specs: list[VariableSpec] = []
        dims = ("time", "side", "z", "along")

        for name in requested:
            arr = self._sample_variable(
                source=source,
                variable=name,
                seconds=seconds,
                sides=sides,
                n_along=n_along,
            )
            data_vars[name] = (dims, arr)
            var_specs.append(_var_spec(name, dims))

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": time_coord,
                "side": np.asarray(sides),
                "z": _grid_coords(grid, nx=nx, ny=ny, nz=nz, depth=depth)["z"],
                "along": np.arange(n_along, dtype="int64"),
            },
        )
        return self._wrap(
            ds,
            var_specs,
            time_range=query.time_range,
            target_grid_id=grid.grid_id,
            params={
                "wrapped": "AnalyticalIWBoundarySource",
                "sides": sides,
                "n_times": int(seconds.size),
            },
        )

    def _legacy_source(self, *, nz: int, ny: int, depth: float) -> Any:
        from experiments.alignment import mitgcm_internal_wave as _iw

        config = self._config
        tref = _as_tref(_cfg(config, "T_ref", "Tref", "tref", default=None), nz=nz)
        ob_timescale = _cfg(
            config,
            "ob_timescale",
            "OB_TIMESCALE",
            default=_iw.OB_TIMESCALE,
        )
        omega = _cfg(config, "omega", default=None)
        if omega is not None:
            ob_timescale = 2.0 * math.pi / float(omega)
        u_inflow = float(
            _cfg(config, "u_inflow", "U_INFLOW", "amplitude", default=_iw.U_INFLOW)
        )
        return _boundary_source_class()(
            nz=nz,
            ny=ny,
            Tref=torch.tensor(tref, dtype=torch.float64),
            u_inflow=u_inflow,
            ob_timescale=float(ob_timescale),
            f0=float(_cfg(config, "f0", "F0", default=_iw.F0)),
            N2=float(_cfg(config, "N_squared", "N2", default=_iw.N2)),
            gravity=float(_cfg(config, "gravity", default=_iw.GRAVITY)),
            alpha=float(_cfg(config, "alpha", default=_iw.ALPHA)),
            total_depth=depth,
            salt_value=float(_cfg(config, "S_ref", "salt_value", default=35.0)),
        )

    def _sample_variable(
        self,
        *,
        source: Any,
        variable: str,
        seconds: np.ndarray,
        sides: Iterable[str],
        n_along: int,
    ) -> np.ndarray:
        phase = float(_cfg(self._config, "phase", default=0.0))
        omega = 2.0 * math.pi / float(source.ob_timescale)
        shifted = seconds + (phase / omega if phase else 0.0)
        side_names = tuple(sides)
        out = np.zeros(
            (seconds.size, len(side_names), source.nz, n_along),
            dtype="float64",
        )
        for it, sec in enumerate(shifted):
            sample = source.sample(float(sec))
            value_set = {
                "temperature": sample.temp,
                "salinity": sample.salt,
                "u": sample.u,
                "v": sample.v,
            }[variable]
            for iside, side in enumerate(side_names):
                tensor = getattr(value_set, _SIDE_ATTRS.get(side, ""), None)
                if tensor is None:
                    continue
                values = tensor.detach().cpu().numpy().astype("float64", copy=False)
                width = min(n_along, values.shape[-1])
                out[it, iside, :, :width] = values[:, :width]
        return out

    def _wrap(
        self,
        ds: xr.Dataset,
        var_specs: list[VariableSpec],
        *,
        time_range: tuple[datetime, datetime] | None,
        target_grid_id: str | None,
        params: dict[str, Any],
    ) -> DataBundle:
        meta_no_prov = BundleMeta(
            schema_version="0.1.0",
            location_kind=LocationKind.BOUNDARY_TIMESERIES,
            convention=self.capabilities.convention,
            variables=tuple(var_specs),
            time_range=time_range,
            target_grid_id=target_grid_id,
            provenance=(),
        )
        prov = ProvenanceStep(
            kind="source",
            name="AnalyticalIWBoundaryDataSource",
            params=params,
            timestamp=datetime.utcnow(),
            bundle_hash="00000000",
        )
        meta = BundleMeta(
            schema_version=meta_no_prov.schema_version,
            location_kind=meta_no_prov.location_kind,
            convention=meta_no_prov.convention,
            variables=meta_no_prov.variables,
            time_range=meta_no_prov.time_range,
            target_grid_id=meta_no_prov.target_grid_id,
            provenance=(prov,),
            extras=meta_no_prov.extras,
        )
        return DataBundle(data=ds, meta=meta)
