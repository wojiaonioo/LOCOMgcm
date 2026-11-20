"""Analytical source — idealized field generators for fixtures + testing.

Supports a small library of named analytical functions (`zero`, `constant`,
`linear_z`). Used by micro-case baseline tests to exercise the pipeline
without external file dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr

from defgcm.io.runner.layer_a import (
    BundleMeta,
    ConventionTag,
    DataBundle,
    LocationKind,
    ProvenanceStep,
    VariableSpec,
)

from .._registry import register_source
from .._types import (
    DataQuery,
    SourceCapabilities,
    SourceContext,
    TimeCoverage,
    validate_query_against_capabilities,
)


_CONVENTION_SENSITIVE_VARS = frozenset({"bathymetry", "eta", "u", "v"})


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a finite numeric value")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a finite numeric value") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    return out


def _instant_timestamp(query: DataQuery) -> datetime:
    if query.time_range is None:
        return datetime(2019, 1, 1)
    return query.time_range[0]


def _zero(shape: tuple[int, ...], **_) -> np.ndarray:
    return np.zeros(shape, dtype="float64")


def _constant(shape: tuple[int, ...], value: float = 0.0, **_) -> np.ndarray:
    value = _finite_float(value, "constant value")
    return np.full(shape, value, dtype="float64")


def _linear_z(
    shape: tuple[int, ...],
    z_centers: np.ndarray,
    surface: float = 28.0,
    bottom: float = 4.0,
    **_,
) -> np.ndarray:
    """Linearly varies a 3D field from `surface` (z=0) to `bottom` (z=max)."""
    surface = _finite_float(surface, "linear_z surface")
    bottom = _finite_float(bottom, "linear_z bottom")
    if len(shape) != 4:
        raise ValueError("linear_z requires a 4-D (time, z, y, x) shape")
    if z_centers is None or len(z_centers) != shape[1]:
        raise ValueError("linear_z requires z_centers matching shape[1]")
    z_norm = z_centers / z_centers[-1]
    profile = surface + (bottom - surface) * z_norm
    out = np.zeros(shape, dtype="float64")
    out[...] = profile.reshape(1, -1, 1, 1)
    return out


_FUNCTIONS = {
    "zero": _zero,
    "constant": _constant,
    "linear_z": _linear_z,
}


@register_source("analytical")
class AnalyticalSource:
    """Generate analytical fields per a named function."""

    capabilities = SourceCapabilities(
        produces=(
            LocationKind.GRIDDED_3D_OCEAN,
            LocationKind.GRIDDED_2D_SURFACE,
            LocationKind.MESH,
        ),
        variables=("temperature", "salinity", "u", "v", "eta", "bathymetry"),
        time_coverage_strategy=TimeCoverage.INSTANT,
        convention=ConventionTag.OPAQUE,
        requires_target_grid=True,
    )

    def __init__(
        self,
        ctx: SourceContext,
        *,
        function: str = "zero",
        variables: tuple[str, ...] = ("temperature",),
        location_kind: str = "gridded_3d_ocean",
        function_params: dict[str, Any] | None = None,
        **_unused: Any,
    ) -> None:
        if function not in _FUNCTIONS:
            raise ValueError(
                f"AnalyticalSource: unknown function {function!r}; "
                f"known: {sorted(_FUNCTIONS)}"
            )
        if function_params is not None and not isinstance(function_params, Mapping):
            raise TypeError("AnalyticalSource: function_params must be a mapping")
        params = dict(function_params or {})
        for key in params:
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "AnalyticalSource: function_params keys must be nonempty strings"
                )
        if function == "constant":
            params["value"] = _finite_float(
                params.get("value", 0.0), "function_params.value"
            )
        elif function == "linear_z":
            params["surface"] = _finite_float(
                params.get("surface", 28.0), "function_params.surface"
            )
            params["bottom"] = _finite_float(
                params.get("bottom", 4.0), "function_params.bottom"
            )

        configured_variables = tuple(variables)
        if not configured_variables:
            raise ValueError("AnalyticalSource: variables must be nonempty")
        unknown = tuple(
            v for v in configured_variables if v not in self.capabilities.variables
        )
        if unknown:
            raise ValueError(
                f"AnalyticalSource: variables not in capabilities: {unknown}"
            )
        location = LocationKind(location_kind)
        if location not in self.capabilities.produces:
            raise ValueError(
                f"AnalyticalSource: unsupported location_kind {location.value!r}"
            )
        if function == "linear_z" and location is not LocationKind.GRIDDED_3D_OCEAN:
            raise ValueError(
                "AnalyticalSource: linear_z only supports GRIDDED_3D_OCEAN"
            )
        if function != "zero" and not (
            function == "constant" and params.get("value", 0.0) == 0.0
        ):
            unsafe = tuple(
                v for v in configured_variables if v in _CONVENTION_SENSITIVE_VARS
            )
            if unsafe:
                raise ValueError(
                    "AnalyticalSource: nonzero OPAQUE output is not allowed for "
                    f"convention-sensitive variables {unsafe}"
                )

        self._ctx = ctx
        self._function_name = function
        self._function = _FUNCTIONS[function]
        self._variables = configured_variables
        self._location_kind = location
        self._function_params = params
        self.capabilities = SourceCapabilities(
            produces=(self._location_kind,),
            variables=self._variables,
            time_coverage_strategy=self.capabilities.time_coverage_strategy,
            convention=self.capabilities.convention,
            requires_target_grid=self.capabilities.requires_target_grid,
        )

    def fetch(self, query: DataQuery) -> DataBundle:
        if not query.variables:
            raise ValueError("AnalyticalSource: query.variables must be nonempty")
        validate_query_against_capabilities(
            query, self.capabilities, source_name="AnalyticalSource"
        )
        grid = query.target_grid
        if grid is None:
            raise ValueError("AnalyticalSource: target_grid required")

        _lat_off = np.arange(grid.ny, dtype="float64") - 0.5 * (grid.ny - 1)
        lat = grid.center_lat + _lat_off * grid.resolution_deg
        _lon_off = np.arange(grid.nx, dtype="float64") - 0.5 * (grid.nx - 1)
        lon = grid.center_lon + _lon_off * grid.resolution_deg
        thicknesses = np.array(grid.vertical_layer_thicknesses, dtype="float64")
        z_interfaces = np.concatenate([[0.0], np.cumsum(thicknesses)])
        z_centers = 0.5 * (z_interfaces[:-1] + z_interfaces[1:])

        requested = tuple(query.variables)

        data_vars: dict[str, tuple] = {}
        var_specs: list[VariableSpec] = []

        for vname in requested:
            if query.location_kind == LocationKind.GRIDDED_3D_OCEAN:
                shape = (1, grid.nz, grid.ny, grid.nx)
                dims = ("time", "z", "latitude", "longitude")
                arr = self._function(shape, z_centers=z_centers, **self._function_params)
            elif query.location_kind == LocationKind.GRIDDED_2D_SURFACE:
                shape = (1, grid.ny, grid.nx)
                dims = ("time", "latitude", "longitude")
                arr = self._function(shape, **self._function_params)
            elif query.location_kind == LocationKind.MESH:
                shape = (grid.ny, grid.nx)
                dims = ("latitude", "longitude")
                arr = self._function(shape, **self._function_params)
            else:
                raise ValueError(
                    f"AnalyticalSource: unsupported location_kind {query.location_kind}"
                )

            data_vars[vname] = (dims, arr)
            var_specs.append(
                VariableSpec(
                    name=vname,
                    cf_standard_name=f"sea_water_{vname}",
                    units="1",
                    dtype="float64",
                    dims=dims,
                )
            )

        coords: dict[str, Any] = {"latitude": lat, "longitude": lon}
        if query.location_kind == LocationKind.GRIDDED_3D_OCEAN:
            coords["z"] = z_centers
        if query.location_kind != LocationKind.MESH:
            emitted_time = _instant_timestamp(query)
            coords["time"] = [np.datetime64(emitted_time, "ns")]

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        meta_time_range = None
        if query.location_kind != LocationKind.MESH and query.time_range is not None:
            meta_time_range = (emitted_time, emitted_time)

        meta_no_prov = BundleMeta(
            schema_version="0.1.0",
            location_kind=query.location_kind,
            convention=self.capabilities.convention,
            variables=tuple(var_specs),
            time_range=meta_time_range,
            target_grid_id=grid.grid_id,
            provenance=(),
        )
        prov = ProvenanceStep(
            kind="source",
            name="AnalyticalSource",
            params={
                "function": self._function_name,
                "variables": self._variables,
                "location_kind": query.location_kind.value,
                "function_params": self._function_params,
            },
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
