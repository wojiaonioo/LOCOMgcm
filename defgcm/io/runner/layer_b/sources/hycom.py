"""HYCOM netcdf source — reads daily T/S/u/v/eta files.

Native HYCOM names: water_temp, salinity, water_u, water_v, surf_el.
Canonical: temperature, salinity, u, v, eta.

Supports two location_kinds:
  - GRIDDED_3D_OCEAN: full 3D ocean state
  - BOUNDARY_TIMESERIES: extract 4-side boundary strips (via query.extras['sides'])
"""

from __future__ import annotations

import math
import glob
import re
from datetime import date, datetime
from pathlib import Path

from defgcm.external_paths import resolve_input_path
from typing import Any

import numpy as np
import pandas as pd
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


_HYCOM_NAME_MAP = {
    "water_temp": "temperature",
    "salinity": "salinity",
    "water_u": "u",
    "water_v": "v",
    "surf_el": "eta",
}
_CANONICAL_TO_HYCOM = {v: k for k, v in _HYCOM_NAME_MAP.items()}

_CF_NAMES = {
    "temperature": "sea_water_potential_temperature",
    "salinity": "sea_water_salinity",
    "u": "sea_water_x_velocity",
    "v": "sea_water_y_velocity",
    "eta": "sea_surface_height_above_geoid",
}

_UNITS = {
    "temperature": "degC",
    "salinity": "1e-3",
    "u": "m s-1",
    "v": "m s-1",
    "eta": "m",
}

_FILE_DATE_PATTERN = re.compile(r"hycom_uvts_(\d{8})\.nc$")


def _parse_file_date(path: Path) -> date | None:
    m = _FILE_DATE_PATTERN.search(path.name)
    if not m:
        return None
    s = m.group(1)
    try:
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


@register_source("hycom_netcdf")
class HycomNetcdfSource:
    capabilities = SourceCapabilities(
        produces=(
            LocationKind.GRIDDED_3D_OCEAN,
            LocationKind.BOUNDARY_TIMESERIES,
        ),
        variables=tuple(_CANONICAL_TO_HYCOM),
        time_coverage_strategy=TimeCoverage.FILE_GLOB_SCAN,
        convention=ConventionTag.CF_STANDARD,
        requires_target_grid=False,
    )

    def __init__(self, ctx: SourceContext, *, path: Path | str,
                 steady: bool = False, **_unused: Any) -> None:
        self._ctx = ctx
        p = resolve_input_path(path)
        self._path = p
        if p.is_dir():
            files = sorted(p.rglob("hycom_uvts_*.nc"))
        elif p.is_file():
            files = [p]
        else:
            files = sorted(Path(item) for item in glob.glob(str(p)))
        if not files:
            raise ValueError(f"HycomNetcdfSource: no files matched {path}")
        self._files = files
        # steady=True: use a SINGLE snapshot (the start_date file) held constant
        # in time, instead of the legacy `ceil(days+1)+1` daily-file time series.
        # For cases whose boundary T/S do not vary in time (only tides do) — e.g.
        # Gong ISWFM, which ships one IC snapshot and runs sponge+tides OBCS.
        # The resulting single-time boundary is handled downstream as a steady
        # prescribe (mit_binary._extract_obcs_timing period 0).
        self._steady = bool(steady)

    def fetch(self, query: DataQuery) -> DataBundle:
        # Filter files by query.time_range.
        #
        # Load source_days = days + 1 and select ceil(source_days) + 1
        # daily HYCOM files starting at start_date.
        # For a 7-day run that is ceil(8) + 1 = 9 files = 72 3-hourly OBC
        # records. A naive `t0_d <= d <= t1_d` window (t1_d = start + days)
        # only picks 8 files / 64 records. This file-count rule applies
        # to HYCOM only; ERA5 keeps its
        # exact [start, start+days] window.
        if query.time_range is not None:
            t0, t1 = query.time_range
            t0_d = t0.date() if isinstance(t0, datetime) else t0
            t1_d = t1.date() if isinstance(t1, datetime) else t1
            window_days = max(0, (t1_d - t0_d).days)
            # steady mode: one snapshot held constant (no daily time series).
            required_count = 1 if self._steady else max(1, math.ceil(window_days + 1) + 1)
            dated = [
                (p, _parse_file_date(p)) for p in self._files
            ]
            dated = [(p, d) for p, d in dated if d is not None and d >= t0_d]
            dated.sort(key=lambda item: item[1])
            selected = [p for p, _ in dated[:required_count]]
            if not selected:
                raise ValueError(
                    f"HycomNetcdfSource: no files with date >= {t0_d} "
                    f"under {self._path}"
                )
            if len(selected) < required_count:
                raise ValueError(
                    f"HycomNetcdfSource: need {required_count} HYCOM files "
                    f"starting at {t0_d} to cover {window_days} days, "
                    f"found {len(selected)} under {self._path}"
                )
        else:
            selected = list(self._files)

        # Pre-slice bbox PER FILE before concat. Concatenating 9 global HYCOM
        # files (each 40×1126×939×4vars) and then slicing materializes the
        # full ~48 GB concat → OOM. Slicing each lazily-opened file first
        # keeps the concat tiny.
        g = query.target_grid
        bbox = None
        if g is not None and hasattr(g, "center_lat"):
            lat_half = g.ny * g.resolution_deg / 2
            lon_half = g.nx * g.resolution_deg / 2
            margin = 1.0   # 1° margin to allow grid_interp boundary cells
            bbox = (
                g.center_lat - lat_half - margin,
                g.center_lat + lat_half + margin,
                g.center_lon - lon_half - margin,
                g.center_lon + lon_half + margin,
            )

        def _open_sliced(path: Path) -> xr.Dataset:
            # Decode HYCOM int16-packed vars (water_temp/salinity/water_u/
            # water_v) at FLOAT64 precision. HYCOM netcdf stores
            # scale_factor/add_offset as float32 attrs, so a plain
            # xr.open_dataset() decodes `packed*scale + offset` in float32 —
            # losing ~1e-6 at T/S magnitudes (~20-35, add_offset=20.0) while
            # u/v (add_offset=0.0) only lose ~1e-8. Legacy's HYCOM zarr cache
            # round-trip (build_hycom_zarr_cache → open_hycom_zarr_cache)
            # stores the data still int16-packed and re-decodes through
            # xarray with the zarr-JSON attrs widened to Python float, i.e.
            # at float64. Promoting scale_factor/add_offset to Python float
            # here before decode_cf reproduces that float64 decode exactly
            # (verified bitwise vs the legacy cache).
            # chunks={"time": 1} → dask-backed, lazy along time only (z/lat/lon
            # stay single chunks). The downstream operator chain (mask / fill /
            # interp / subset_boundary) is all dask-aware and value-preserving
            # per chunk, and the zarr consumer's to_zarr streams chunk-by-chunk —
            # so the full (time, z, lat, lon) interior grid is NEVER materialised
            # at once. decode_cf is element-wise, so the int16→float64 decode is
            # bit-identical under time-chunking.
            ds = xr.open_dataset(path, decode_cf=False, mask_and_scale=False,
                                 chunks={"time": 1})
            for _v in ds.data_vars:
                for _a in ("scale_factor", "add_offset"):
                    if _a in ds[_v].attrs:
                        ds[_v].attrs[_a] = float(ds[_v].attrs[_a])
            ds = xr.decode_cf(ds)
            coord_map: dict[str, str] = {}
            if "lat" in ds.coords:
                coord_map["lat"] = "latitude"
            if "lon" in ds.coords:
                coord_map["lon"] = "longitude"
            if "depth" in ds.coords:
                coord_map["depth"] = "z"
            ds = ds.rename(coord_map) if coord_map else ds
            if bbox is not None and "latitude" in ds.coords:
                lat_min, lat_max, lon_min, lon_max = bbox
                # HYCOM lat may be ascending or descending — try both slice dirs
                sub = ds.sel(latitude=slice(lat_min, lat_max))
                if sub.sizes.get("latitude", 0) == 0:
                    sub = ds.sel(latitude=slice(lat_max, lat_min))
                ds = sub
                try:
                    ds = ds.sel(longitude=slice(lon_min, lon_max))
                except Exception:
                    pass
            return ds

        ds_list = [_open_sliced(f) for f in selected]
        ds_canon = xr.concat(ds_list, dim="time") if len(ds_list) > 1 else ds_list[0]

        # Build output
        if query.location_kind == LocationKind.GRIDDED_3D_OCEAN:
            return self._fetch_3d(ds_canon, query, selected)
        elif query.location_kind == LocationKind.BOUNDARY_TIMESERIES:
            return self._fetch_boundary(ds_canon, query, selected)
        else:
            raise ValueError(
                f"HycomNetcdfSource: unsupported location_kind {query.location_kind}"
            )

    def _fetch_3d(
        self, ds: xr.Dataset, query: DataQuery, files: list[Path]
    ) -> DataBundle:
        # GRIDDED_3D_OCEAN is used by BOTH the state pipeline (IC, only needs
        # t=0) AND the boundary pipeline (full time-series, sliced to 4 sides
        # by the layer_c `subset_boundary` op downstream).  Default keeps the
        # full time-series for correctness.  Set query.extras["first_time_only"]
        # = True for state-pipeline-style fetches to skip materializing T-1
        # discarded frames (~24× memory saving at 720²×50×8day HYCOM; see
        # `_runner.py` which sets this for `pipeline.name == "state"`).
        only_first = bool(query.extras.get("first_time_only", False))
        time_slice = slice(0, 1) if only_first else slice(None)

        out_data_vars: dict[str, tuple] = {}
        var_specs: list[VariableSpec] = []
        for canonical in query.variables:
            if canonical not in _CANONICAL_TO_HYCOM:
                continue
            native = _CANONICAL_TO_HYCOM[canonical]
            if native not in ds.data_vars:
                continue
            da = ds[native].isel(time=time_slice)
            if "z" in da.dims:
                dims = ("time", "z", "latitude", "longitude")
            else:
                dims = ("time", "latitude", "longitude")
            # `.data` keeps the dask array lazy (the source is opened with
            # chunks={"time": 1}); the old np.asarray(da.values) here force-
            # computed the whole HYCOM time-series — the boundary OOM source.
            arr = da.astype("float64").data
            out_data_vars[canonical] = (dims, arr)
            var_specs.append(
                VariableSpec(
                    name=canonical,
                    cf_standard_name=_CF_NAMES[canonical],
                    units=_UNITS[canonical],
                    dtype="float64",
                    dims=dims,
                )
            )

        coords: dict[str, Any] = {
            "time": ds["time"].isel(time=time_slice).values,
            "latitude": np.asarray(ds["latitude"].values, dtype="float64"),
            "longitude": np.asarray(ds["longitude"].values, dtype="float64"),
        }
        if "z" in ds.coords:
            coords["z"] = np.asarray(ds["z"].values, dtype="float64")

        out_ds = xr.Dataset(data_vars=out_data_vars, coords=coords)
        return self._wrap(out_ds, var_specs, files, location_kind=LocationKind.GRIDDED_3D_OCEAN)

    def _fetch_boundary(
        self, ds: xr.Dataset, query: DataQuery, files: list[Path]
    ) -> DataBundle:
        # Extract 4 side strips: N (last lat row), E (last lon col),
        # S (first lat row), W (first lon col).
        #
        # Memory note: previously materialized full (T, Z, Y, X) per variable
        # via `da.values` BEFORE slicing → ~60 GB at d4 720²×50×64-frame×4-vars.
        # Now isel each side first (lazy), THEN .values — boundary strips total
        # ~75 MB (~800× saved).  Critical for d4 30-day chain at PREP_MEM=192G.
        sides = query.extras.get("sides", ("N", "E", "S", "W"))
        width = int(query.extras.get("width", 1))

        # For simplicity, accumulate to (time, side, z, along) shape.
        # 'along' is per-side variable-length; we pad / select to nx (or ny)
        # uniformly. Use min-of-(nx, ny) for the canonical 'along' dim.
        n_along = min(len(ds["latitude"]), len(ds["longitude"]))

        out_data_vars: dict[str, tuple] = {}
        var_specs: list[VariableSpec] = []
        for canonical in query.variables:
            if canonical not in _CANONICAL_TO_HYCOM:
                continue
            native = _CANONICAL_TO_HYCOM[canonical]
            if native not in ds.data_vars:
                continue
            da = ds[native]
            if "z" in da.dims:
                # Cut the 4 sides lazily THEN materialize: (T, Z, n_along) each.
                strips = []
                for side in sides:
                    if side == "N":
                        s_da = da.isel(latitude=-1, longitude=slice(0, n_along))
                    elif side == "S":
                        s_da = da.isel(latitude=0, longitude=slice(0, n_along))
                    elif side == "E":
                        s_da = da.isel(latitude=slice(0, n_along), longitude=-1)
                    elif side == "W":
                        s_da = da.isel(latitude=slice(0, n_along), longitude=0)
                    else:
                        continue
                    strips.append(np.asarray(s_da.values, dtype="float64"))
                stacked = np.stack(strips, axis=1)  # (time, side, z, along)
                out_data_vars[canonical] = (("time", "side", "z", "along"), stacked)
                dims = ("time", "side", "z", "along")
            else:
                # Surface 2D var — same lazy slice pattern.
                strips = []
                for side in sides:
                    if side == "N":
                        s_da = da.isel(latitude=-1, longitude=slice(0, n_along))
                    elif side == "S":
                        s_da = da.isel(latitude=0, longitude=slice(0, n_along))
                    elif side == "E":
                        s_da = da.isel(latitude=slice(0, n_along), longitude=-1)
                    elif side == "W":
                        s_da = da.isel(latitude=slice(0, n_along), longitude=0)
                    else:
                        continue
                    strips.append(np.asarray(s_da.values, dtype="float64"))
                stacked = np.stack(strips, axis=1)  # (time, side, along)
                out_data_vars[canonical] = (("time", "side", "along"), stacked)
                dims = ("time", "side", "along")
            var_specs.append(
                VariableSpec(
                    name=canonical,
                    cf_standard_name=_CF_NAMES[canonical],
                    units=_UNITS[canonical],
                    dtype="float64",
                    dims=dims,
                )
            )

        coords: dict[str, Any] = {
            "time": ds["time"].values,
            "side": np.array(list(sides)),
            "along": np.arange(n_along),
        }
        if any("z" in v.dims for v in var_specs):
            coords["z"] = np.asarray(ds["z"].values, dtype="float64") if "z" in ds.coords else np.arange(ds.sizes.get("z", 1), dtype="float64")
        # boundary still needs CF coords for completeness in canonical CF metadata
        coords["latitude"] = np.asarray(ds["latitude"].values, dtype="float64")
        coords["longitude"] = np.asarray(ds["longitude"].values, dtype="float64")

        out_ds = xr.Dataset(data_vars=out_data_vars, coords=coords)
        return self._wrap(
            out_ds, var_specs, files, location_kind=LocationKind.BOUNDARY_TIMESERIES
        )

    def _wrap(
        self,
        ds: xr.Dataset,
        var_specs: list[VariableSpec],
        files: list[Path],
        *,
        location_kind: LocationKind,
    ) -> DataBundle:
        if "time" in ds.coords and len(ds["time"]) > 0:
            t_vals = ds["time"].values
            t_start = pd.Timestamp(t_vals[0]).to_pydatetime()
            t_end = pd.Timestamp(t_vals[-1]).to_pydatetime()
            time_range: tuple[datetime, datetime] | None = (t_start, t_end)
        else:
            time_range = None

        meta_no_prov = BundleMeta(
            schema_version="0.1.0",
            location_kind=location_kind,
            convention=ConventionTag.CF_STANDARD,
            variables=tuple(var_specs),
            time_range=time_range,
            target_grid_id=None,
            provenance=(),
        )
        prov = ProvenanceStep(
            kind="source",
            name="HycomNetcdfSource",
            params={
                "path": str(self._path),
                "files": tuple(str(f) for f in files),
                "location_kind": location_kind.value,
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
