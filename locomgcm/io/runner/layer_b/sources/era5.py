"""ERA5 surface-forcing netcdf source.

Reads ERA5 hourly netcdf files (u10/v10/msl/...). Emits CF_STANDARD bundle
with LocationKind.GRIDDED_2D_SURFACE.
"""

from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path

from locomgcm.external_paths import resolve_input_path
from typing import Any

import numpy as np
import pandas as pd
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
)


# Map native ERA5 var names to canonical short names
_ERA5_VAR_MAP = {
    "u10": "wind_u",
    "v10": "wind_v",
    "msl": "sea_level_pressure",
    "t2m": "air_temperature_2m",
    "sst": "sea_surface_temperature",
}

# Reverse map for query-time lookup
_CANONICAL_TO_ERA5 = {v: k for k, v in _ERA5_VAR_MAP.items()}

# CF standard names
_CF_NAMES = {
    "wind_u": "eastward_wind",
    "wind_v": "northward_wind",
    "sea_level_pressure": "air_pressure_at_mean_sea_level",
    "air_temperature_2m": "air_temperature",
    "sea_surface_temperature": "sea_surface_temperature",
}

_UNITS = {
    "wind_u": "m s-1",
    "wind_v": "m s-1",
    "sea_level_pressure": "Pa",
    "air_temperature_2m": "K",
    "sea_surface_temperature": "K",
}


@register_source("era5_netcdf")
class Era5NetcdfSource:
    capabilities = SourceCapabilities(
        produces=(LocationKind.GRIDDED_2D_SURFACE,),
        variables=tuple(_CANONICAL_TO_ERA5),
        time_coverage_strategy=TimeCoverage.FILE_GLOB_SCAN,
        convention=ConventionTag.CF_STANDARD,
        requires_target_grid=False,
    )

    def __init__(self, ctx: SourceContext, *, path: Path | str, **_unused: Any) -> None:
        self._ctx = ctx
        p = resolve_input_path(path)
        self._path = p
        if p.is_dir():
            # Recursive glob — handles legacy `data/era5/<year>/era5_*.nc` layout
            # as well as flat `data/era5/era5_*.nc` layout.
            self._files = sorted(p.rglob("era5_*.nc"))
        elif p.is_file():
            self._files = [p]
        else:
            self._files = sorted(Path(item) for item in glob.glob(str(p)))
        if not self._files:
            raise ValueError(f"Era5NetcdfSource: no files matched {path}")

    def fetch(self, query: DataQuery) -> DataBundle:
        if query.location_kind != LocationKind.GRIDDED_2D_SURFACE:
            raise ValueError(
                f"Era5NetcdfSource: only GRIDDED_2D_SURFACE; got {query.location_kind}"
            )

        # ERA5 files are typically split per-variable (era5_u10_*.nc has only
        # u10; era5_v10_*.nc has only v10; etc.). We need to MERGE across
        # files (not concat) since each file has different vars.
        # Also: ERA5 newer datasets use `valid_time` instead of `time`.
        per_file_ds: list[xr.Dataset] = []
        for f in self._files:
            # chunks={} → dask-backed/lazy; rechunked to time=1 below so the
            # forcing grid_interp + to_zarr stream per time-step instead of
            # materialising the whole (time, lat, lon) wind series at once.
            ds_one = xr.open_dataset(f, chunks={})
            # Rename ERA5-specific time dim → canonical 'time'
            if "valid_time" in ds_one.dims and "time" not in ds_one.dims:
                ds_one = ds_one.rename({"valid_time": "time"})
            # Apply time subset EARLY so we don't materialize full year × global grid
            if query.time_range is not None and "time" in ds_one.dims:
                t0, t1 = query.time_range
                ds_one = ds_one.sel(time=slice(pd.Timestamp(t0), pd.Timestamp(t1)))
            per_file_ds.append(ds_one)
        # xr.merge across per-variable files (compatible coords)
        if len(per_file_ds) == 1:
            ds_raw = per_file_ds[0]
        else:
            ds_raw = xr.merge(per_file_ds, compat="override", join="outer")

        coord_map: dict[str, str] = {}
        if "lat" in ds_raw.coords:
            coord_map["lat"] = "latitude"
        if "lon" in ds_raw.coords:
            coord_map["lon"] = "longitude"
        ds_canon = ds_raw.rename(coord_map) if coord_map else ds_raw

        # Pre-slice to model bbox (ERA5 global = 721 × 1440 = 1M cells × hourly
        # = 8.7 GB for 1 year × 1 var; we want only the SCS patch).
        g = query.target_grid
        if g is not None and hasattr(g, "center_lat"):
            lat_half = g.ny * g.resolution_deg / 2
            lon_half = g.nx * g.resolution_deg / 2
            margin = 1.0
            lat_min = g.center_lat - lat_half - margin
            lat_max = g.center_lat + lat_half + margin
            lon_min = g.center_lon - lon_half - margin
            lon_max = g.center_lon + lon_half + margin
            try:
                ds_test = ds_canon.sel(latitude=slice(lat_min, lat_max))
                if ds_test.sizes.get("latitude", 0) == 0:
                    ds_canon = ds_canon.sel(latitude=slice(lat_max, lat_min))
                else:
                    ds_canon = ds_test
            except Exception:
                ds_canon = ds_canon.sel(latitude=slice(lat_max, lat_min))
            try:
                ds_canon = ds_canon.sel(longitude=slice(lon_min, lon_max))
            except Exception:
                pass

        # Chunk along time only (lat/lon stay single chunks) so every op runs
        # within one time-block exactly as eager → bit-identical, streamed.
        if "time" in ds_canon.dims:
            ds_canon = ds_canon.chunk({"time": 1})

        out_data_vars: dict[str, tuple] = {}
        var_specs: list[VariableSpec] = []
        for canonical in query.variables:
            if canonical not in _CANONICAL_TO_ERA5:
                continue
            native = _CANONICAL_TO_ERA5[canonical]
            if native not in ds_canon.data_vars:
                continue
            # float64: ERA5 u10/v10 are natively float32 in the netcdf, but
            # legacy _extract_wind_source interpolates onto float64 target
            # coords (float32 data x float64 weights -> float64 result) and
            # keeps float64 through the Large&Pond stress. Declaring float32
            # here makes grid_interp downcast the float64 interp result back
            # to float32 -> ~3e-8 vs legacy on ustress/vstress. Widening
            # float32->float64 is lossless so the interpolated values match.
            # `.astype(...).data` widens + keeps the dask array lazy (was
            # np.asarray(.values), which force-materialised the full wind series).
            arr = ds_canon[native].astype("float64").data
            out_data_vars[canonical] = (("time", "latitude", "longitude"), arr)
            var_specs.append(
                VariableSpec(
                    name=canonical,
                    cf_standard_name=_CF_NAMES[canonical],
                    units=_UNITS[canonical],
                    dtype="float64",
                    dims=("time", "latitude", "longitude"),
                )
            )

        if not out_data_vars:
            raise ValueError(
                f"Era5NetcdfSource: no requested variables found in files; "
                f"requested {query.variables}, available natively {list(ds_canon.data_vars)}"
            )

        time_vals = ds_canon["time"].values
        lat_vals = np.asarray(ds_canon["latitude"].values, dtype="float64")
        lon_vals = np.asarray(ds_canon["longitude"].values, dtype="float64")
        out_ds = xr.Dataset(
            data_vars=out_data_vars,
            coords={"time": time_vals, "latitude": lat_vals, "longitude": lon_vals},
        )

        # Compute bundle's time_range from data
        if len(time_vals) > 0:
            t_start = pd.Timestamp(time_vals[0]).to_pydatetime()
            t_end = pd.Timestamp(time_vals[-1]).to_pydatetime()
            time_range: tuple[datetime, datetime] | None = (t_start, t_end)
        else:
            time_range = None

        meta_no_prov = BundleMeta(
            schema_version="0.1.0",
            location_kind=LocationKind.GRIDDED_2D_SURFACE,
            convention=ConventionTag.CF_STANDARD,
            variables=tuple(var_specs),
            time_range=time_range,
            target_grid_id=None,
            provenance=(),
        )
        prov = ProvenanceStep(
            kind="source",
            name="Era5NetcdfSource",
            params={"path": str(self._path), "files": tuple(str(f) for f in self._files)},
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
