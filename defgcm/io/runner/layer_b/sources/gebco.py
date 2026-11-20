"""GEBCO bathymetry netcdf source.

GEBCO `elevation` is positive above sea level, negative below. Source emits
the data as-is (CF_STANDARD convention); ConventionConverter (Layer C) handles
the conversion to MIT or LOCOMgcm convention later.

Time-invariant; `LocationKind.MESH`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from defgcm.external_paths import resolve_input_path
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
)


@register_source("gebco_netcdf")
class GebcoNetcdfSource:
    capabilities = SourceCapabilities(
        produces=(LocationKind.MESH,),
        variables=("bathymetry", "elevation"),
        time_coverage_strategy=TimeCoverage.INSTANT,
        convention=ConventionTag.CF_STANDARD,
        requires_target_grid=False,
    )

    def __init__(self, ctx: SourceContext, *, path: Path | str, **_unused: Any) -> None:
        self._ctx = ctx
        self._path = resolve_input_path(path)

    def fetch(self, query: DataQuery) -> DataBundle:
        if query.location_kind != LocationKind.MESH:
            raise ValueError(
                f"GebcoNetcdfSource: only produces MESH; query asked for {query.location_kind}"
            )

        ds_raw = xr.open_dataset(self._path)
        # Canonicalize coord names: GEBCO sometimes uses 'lat'/'lon'
        coord_map: dict[str, str] = {}
        if "lat" in ds_raw.coords and "latitude" not in ds_raw.coords:
            coord_map["lat"] = "latitude"
        if "lon" in ds_raw.coords and "longitude" not in ds_raw.coords:
            coord_map["lon"] = "longitude"
        ds_canon = ds_raw.rename(coord_map) if coord_map else ds_raw

        # Build bathymetry variable (canonical local name) from elevation
        if "elevation" not in ds_canon.data_vars:
            raise ValueError(
                f"GebcoNetcdfSource: file {self._path} lacks 'elevation' variable; "
                f"got: {sorted(ds_canon.data_vars)}"
            )

        elev = ds_canon["elevation"]
        elev_arr = np.asarray(elev.values, dtype="float64")

        out_ds = xr.Dataset(
            data_vars={
                "bathymetry": (("latitude", "longitude"), elev_arr),
            },
            coords={
                "latitude": np.asarray(ds_canon["latitude"].values, dtype="float64"),
                "longitude": np.asarray(ds_canon["longitude"].values, dtype="float64"),
            },
        )

        var = VariableSpec(
            name="bathymetry",
            cf_standard_name="height_above_reference_ellipsoid",
            units="m",
            dtype="float64",
            dims=("latitude", "longitude"),
        )

        meta_no_prov = BundleMeta(
            schema_version="0.1.0",
            location_kind=LocationKind.MESH,
            convention=ConventionTag.CF_STANDARD,
            variables=(var,),
            time_range=None,
            target_grid_id=None,
            provenance=(),
        )
        prov = ProvenanceStep(
            kind="source",
            name="GebcoNetcdfSource",
            params={"path": str(self._path)},
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
        return DataBundle(data=out_ds, meta=meta)
