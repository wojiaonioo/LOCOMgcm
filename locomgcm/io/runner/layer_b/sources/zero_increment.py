"""Zero-increment source — produces a zero-filled bundle matching a target spec.

Useful for testing Merger plugin paths (e.g. `merge: add_increment` with all
zero values is a no-op increment; lets pipelines exercise the merger code
without needing real DA increments).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
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
from .._types import (
    DataQuery,
    SourceCapabilities,
    SourceContext,
    TimeCoverage,
    validate_query_against_capabilities,
)


def _instant_timestamp(query: DataQuery) -> datetime:
    if query.time_range is None:
        return datetime(2019, 1, 1)
    return query.time_range[0]


@register_source("zero_increment")
class ZeroIncrementSource:
    """Zero-filled bundle matching a configured variable spec."""

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
        vars: tuple[str, ...] = ("temperature",),
        **_unused: Any,
    ) -> None:
        self._ctx = ctx
        self._vars = tuple(vars)
        if not self._vars:
            raise ValueError("ZeroIncrementSource: vars must be nonempty")
        unknown = tuple(v for v in self._vars if v not in self.capabilities.variables)
        if unknown:
            raise ValueError(
                f"ZeroIncrementSource: vars not in capabilities: {unknown}"
            )
        self.capabilities = SourceCapabilities(
            produces=self.capabilities.produces,
            variables=self._vars,
            time_coverage_strategy=self.capabilities.time_coverage_strategy,
            convention=self.capabilities.convention,
            requires_target_grid=self.capabilities.requires_target_grid,
        )

    def fetch(self, query: DataQuery) -> DataBundle:
        if not query.variables:
            raise ValueError(
                "ZeroIncrementSource.fetch: query.variables must be nonempty"
            )
        validate_query_against_capabilities(
            query, self.capabilities, source_name="ZeroIncrementSource"
        )

        grid = query.target_grid
        if grid is None:
            raise ValueError(
                "ZeroIncrementSource.fetch: requires target_grid (none provided)"
            )
        _lat_off = np.arange(grid.ny, dtype="float64") - 0.5 * (grid.ny - 1)
        lat = grid.center_lat + _lat_off * grid.resolution_deg
        _lon_off = np.arange(grid.nx, dtype="float64") - 0.5 * (grid.nx - 1)
        lon = grid.center_lon + _lon_off * grid.resolution_deg

        # Build z-axis from layer thicknesses (cumulative centers)
        thicknesses = np.array(grid.vertical_layer_thicknesses, dtype="float64")
        interfaces = np.concatenate([[0.0], np.cumsum(thicknesses)])
        centers = 0.5 * (interfaces[:-1] + interfaces[1:])

        var_specs: list[VariableSpec] = []
        data_vars: dict[str, tuple] = {}

        requested = tuple(query.variables)
        for vname in requested:
            if query.location_kind == LocationKind.GRIDDED_3D_OCEAN:
                shape = (1, grid.nz, grid.ny, grid.nx)
                dims = ("time", "z", "latitude", "longitude")
            elif query.location_kind == LocationKind.GRIDDED_2D_SURFACE:
                shape = (1, grid.ny, grid.nx)
                dims = ("time", "latitude", "longitude")
            elif query.location_kind == LocationKind.MESH:
                shape = (grid.ny, grid.nx)
                dims = ("latitude", "longitude")
            else:
                raise ValueError(
                    f"ZeroIncrementSource: unsupported location_kind {query.location_kind}"
                )
            arr = np.zeros(shape, dtype="float64")
            data_vars[vname] = (dims, arr)
            var_specs.append(
                VariableSpec(
                    name=vname,
                    cf_standard_name=f"sea_water_{vname}",  # naive; sufficient for testing
                    units="1",
                    dtype="float64",
                    dims=dims,
                )
            )

        coords: dict[str, Any] = {"latitude": lat, "longitude": lon}
        if query.location_kind == LocationKind.GRIDDED_3D_OCEAN:
            coords["z"] = centers
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
            name="ZeroIncrementSource",
            params={"vars": self._vars, "location_kind": query.location_kind.value},
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
