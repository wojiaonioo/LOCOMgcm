"""HYCOM raw-by-variable source for the global real-data pipeline.

This reads the 0.08-degree HYCOM files under::

    data/global/_raw_by_variable/hycom/

The older ``hycom_netcdf`` source consumes pre-combined ``hycom_uvts_*.nc``
files.  For the p25 global case those combined files are smoke-test products
on a 5-degree grid, so using them with 0.25-degree bathymetry can create severe
coastal/deep-water extrapolation artifacts.  This source keeps the native
0.08-degree HYCOM grid until the pipeline's normal interpolation operators.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from locomgcm.external_paths import resolve_input_path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

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


_VARIABLES = ("temperature", "salinity", "u", "v", "eta")
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


def _month_from_query(query: DataQuery, fallback: str) -> str:
    if query.time_range is None:
        return fallback
    t0 = query.time_range[0]
    return f"{t0.year:04d}-{t0.month:02d}"


def _time_from_query(query: DataQuery, fallback: str) -> np.datetime64:
    if query.time_range is not None:
        return np.datetime64(query.time_range[0])
    return np.datetime64(f"{fallback}-01T00:00:00")


def _coord_slice(values: np.ndarray, lower: float, upper: float) -> slice:
    if values[0] <= values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


@register_source("hycom_raw_by_variable")
class HycomRawByVariableSource:
    capabilities = SourceCapabilities(
        produces=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.BOUNDARY_TIMESERIES),
        variables=_VARIABLES,
        time_coverage_strategy=TimeCoverage.FILE_GLOB_SCAN,
        convention=ConventionTag.CF_STANDARD,
        requires_target_grid=False,
    )

    def __init__(
        self,
        ctx: SourceContext,
        *,
        path: Path | str,
        steady: bool = True,
        month: str = "1993-07",
        pregrid_to_target: bool = False,
        **_: Any,
    ) -> None:
        self._ctx = ctx
        self._path = resolve_input_path(path)
        self._steady = bool(steady)
        self._month = str(month)
        self._pregrid_to_target = bool(pregrid_to_target)
        self._target_cache: dict[tuple[Any, ...], xr.Dataset] = {}
        self._grid_file = self._path / "grid/hycom_reanalysis_grid.mat"
        if not self._grid_file.exists():
            raise ValueError(f"hycom_raw_by_variable: missing grid file {self._grid_file}")
        mat = sio.loadmat(self._grid_file)
        self._lat = np.asarray(mat["lat"].ravel(), dtype="float64")
        lon_native = np.asarray(mat["lon"].ravel(), dtype="float64")
        lon360 = np.where(lon_native < 0.0, lon_native + 360.0, lon_native)
        self._lon_order = np.argsort(lon360)
        self._lon = lon360[self._lon_order].astype("float64")
        self._z = np.asarray(mat["depth"].ravel(), dtype="float64")

        self._files = sorted(self._path.rglob("*.nc")) + [self._grid_file]
        if not self._files:
            raise ValueError(f"hycom_raw_by_variable: no files under {self._path}")

    def _paths_for_month(self, month: str) -> dict[str, Any]:
        paths: dict[str, Any] = {
            "u": self._path / "uvel" / f"uu-{month}.nc",
            "v": self._path / "vvel" / f"vv-{month}.nc",
            "eta": self._path / "ssh" / f"global_ss-{month}.nc",
            "temperature": [
                self._path / "temp" / f"global_tt-{month}-level-{k:02d}.nc"
                for k in range(len(self._z))
            ],
            "salinity": [
                self._path / "salt" / f"global_ss-{month}-level-{k:02d}.nc"
                for k in range(len(self._z))
            ],
        }
        missing: list[Path] = []
        for value in paths.values():
            if isinstance(value, list):
                missing.extend([p for p in value if not p.exists()])
            elif not Path(value).exists():
                missing.append(Path(value))
        if missing:
            raise ValueError(
                "hycom_raw_by_variable: missing files for "
                f"{month}: {', '.join(str(p) for p in missing[:5])}"
            )
        return paths

    # Prep chunks only along time (the preprocessing memory bound): z, lat
    # and lon stay whole through every interpolation, so each opener reads one
    # dask chunk per variable (``chunks=-1``) and the per-level tracer files are
    # joined into one z chunk.  A raw file holds one record, so the block is
    # the whole 3-D field of one time: 40 x 2001 x 4500 float64 = 2.9 GB on the
    # 1/12-degree GLBu0.08 grid.
    def _open_2d(self, path: Path, var: str) -> xr.DataArray:
        ds = xr.open_dataset(path, decode_cf=False, mask_and_scale=False, chunks=-1)
        da = ds[var]
        # Raw files use lev/layer as a singleton first dim for per-level files.
        first_dim = da.dims[0]
        if first_dim not in ("lat", "lon"):
            da = da.isel({first_dim: 0}, drop=True)
        da = da.rename({"lat": "latitude", "lon": "longitude"})
        da = da.isel(longitude=self._lon_order)
        da = da.assign_coords(latitude=self._lat, longitude=self._lon)
        fill = da.attrs.get("_FillValue", da.attrs.get("missing_value", -30000.0))
        da = da.where(da > float(fill) + 1.0)
        return da.astype("float64")

    def _open_tracer(self, paths: list[Path], native_var: str) -> xr.DataArray:
        levels = [self._open_2d(path, native_var) for path in paths]
        da = xr.concat(levels, dim=xr.IndexVariable("z", self._z))
        return da.chunk({"z": -1})

    def _open_velocity(self, path: Path, native_var: str) -> xr.DataArray:
        ds = xr.open_dataset(path, decode_cf=False, mask_and_scale=False, chunks=-1)
        da = ds[native_var].rename({"layer": "z", "lat": "latitude", "lon": "longitude"})
        da = da.isel(longitude=self._lon_order)
        da = da.assign_coords(z=self._z, latitude=self._lat, longitude=self._lon)
        fill = da.attrs.get("_FillValue", da.attrs.get("missing_value", -30000.0))
        da = da.where(da > float(fill) + 1.0)
        return da.astype("float64")

    def _subset_bbox(self, ds: xr.Dataset, query: DataQuery) -> xr.Dataset:
        g = query.target_grid
        if g is None:
            return ds
        lat_half = g.ny * g.resolution_deg / 2.0
        lon_half = g.nx * g.resolution_deg / 2.0
        margin = max(1.0, 2.0 * float(g.resolution_deg))
        lat_min = g.center_lat - lat_half - margin
        lat_max = g.center_lat + lat_half + margin
        lon_min = g.center_lon - lon_half - margin
        lon_max = g.center_lon + lon_half + margin
        ds = ds.sel(latitude=_coord_slice(self._lat, lat_min, lat_max))
        if lon_min <= float(self._lon[0]) and lon_max >= float(self._lon[-1]):
            return ds
        lon_min = max(float(self._lon[0]), lon_min)
        lon_max = min(float(self._lon[-1]), lon_max)
        return ds.sel(longitude=slice(lon_min, lon_max))

    @staticmethod
    def _target_coords(g: Any) -> tuple[np.ndarray, np.ndarray]:
        lat_off = np.arange(g.ny, dtype="float64") - 0.5 * (g.ny - 1)
        lon_off = np.arange(g.nx, dtype="float64") - 0.5 * (g.nx - 1)
        return (
            g.center_lat + lat_off * g.resolution_deg,
            g.center_lon + lon_off * g.resolution_deg,
        )

    @staticmethod
    def _fill_nearest_2d(values: np.ndarray) -> np.ndarray:
        mask = ~np.isfinite(values)
        if not mask.any():
            return values
        if mask.all():
            raise ValueError("hycom_raw_by_variable: cannot fill an all-NaN horizontal level")
        indices = distance_transform_edt(
            mask,
            return_distances=False,
            return_indices=True,
        )
        return values[tuple(indices)]

    def _interp_2d_to_target(
        self,
        values: np.ndarray,
        *,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype="float64")
        values = np.ascontiguousarray(values[:, self._lon_order])
        interp = RegularGridInterpolator(
            (self._lat, self._lon),
            values,
            bounds_error=False,
            fill_value=np.nan,
        )
        out = interp(target_points).reshape(target_lat.size, target_lon.size)
        return self._fill_nearest_2d(out)

    def _read_2d_file(self, path: Path, var: str) -> np.ndarray:
        with xr.open_dataset(path, decode_cf=False, mask_and_scale=False) as ds:
            da = ds[var]
            first_dim = da.dims[0]
            if first_dim not in ("lat", "lon"):
                da = da.isel({first_dim: 0}, drop=True)
            arr = np.asarray(da.values, dtype="float64")
            fill = da.attrs.get("_FillValue", da.attrs.get("missing_value", -30000.0))
        arr[arr <= float(fill) + 1.0] = np.nan
        return arr

    def _pregrid_tracer(
        self,
        paths: list[Path],
        var: str,
        *,
        name: str,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        out = np.empty((len(paths), target_lat.size, target_lon.size), dtype="float64")
        log = getattr(self._ctx, "logger", None)
        for k, path in enumerate(paths):
            if log is not None and (k == 0 or (k + 1) % 10 == 0 or k + 1 == len(paths)):
                log.info(
                    "hycom_raw_by_variable: pregrid %s level %02d/%02d",
                    name,
                    k + 1,
                    len(paths),
                )
            arr = self._read_2d_file(path, var)
            out[k] = self._interp_2d_to_target(
                arr,
                target_lat=target_lat,
                target_lon=target_lon,
                target_points=target_points,
            )
        return out

    def _pregrid_velocity(
        self,
        path: Path,
        var: str,
        *,
        name: str,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        out = np.empty((len(self._z), target_lat.size, target_lon.size), dtype="float64")
        log = getattr(self._ctx, "logger", None)
        with xr.open_dataset(path, decode_cf=False, mask_and_scale=False) as ds:
            da = ds[var]
            fill = da.attrs.get("_FillValue", da.attrs.get("missing_value", -30000.0))
            for k in range(len(self._z)):
                if log is not None and (k == 0 or (k + 1) % 10 == 0 or k + 1 == len(self._z)):
                    log.info(
                        "hycom_raw_by_variable: pregrid %s level %02d/%02d",
                        name,
                        k + 1,
                        len(self._z),
                    )
                arr = np.asarray(da.isel(layer=k).values, dtype="float64")
                arr[arr <= float(fill) + 1.0] = np.nan
                out[k] = self._interp_2d_to_target(
                    arr,
                    target_lat=target_lat,
                    target_lon=target_lon,
                    target_points=target_points,
                )
        return out

    def _pregrid_eta(
        self,
        path: Path,
        *,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        arr = self._read_2d_file(path, "surf_el")
        return self._interp_2d_to_target(
            arr,
            target_lat=target_lat,
            target_lon=target_lon,
            target_points=target_points,
        )

    def _fetch_pregridded(
        self,
        query: DataQuery,
        *,
        month: str,
        paths: dict[str, Any],
        t0: np.datetime64,
        vars_requested: tuple[str, ...],
    ) -> xr.Dataset:
        g = query.target_grid
        if g is None:
            raise ValueError("hycom_raw_by_variable pregrid_to_target requires query.target_grid")
        target_lat, target_lon = self._target_coords(g)
        yy, xx = np.meshgrid(target_lat, target_lon, indexing="ij")
        target_points = np.stack([yy.ravel(), xx.ravel()], axis=-1)

        key = (
            month,
            tuple(vars_requested),
            int(g.nx),
            int(g.ny),
            float(g.center_lat),
            float(g.center_lon),
            float(g.resolution_deg),
            str(t0),
        )
        cached = self._target_cache.get(key)
        if cached is not None:
            return cached

        data_vars: dict[str, Any] = {}
        if "temperature" in vars_requested:
            data_vars["temperature"] = (
                ("z", "latitude", "longitude"),
                self._pregrid_tracer(
                    paths["temperature"],
                    "water_temp",
                    name="temperature",
                    target_lat=target_lat,
                    target_lon=target_lon,
                    target_points=target_points,
                ),
            )
        if "salinity" in vars_requested:
            data_vars["salinity"] = (
                ("z", "latitude", "longitude"),
                self._pregrid_tracer(
                    paths["salinity"],
                    "salinity",
                    name="salinity",
                    target_lat=target_lat,
                    target_lon=target_lon,
                    target_points=target_points,
                ),
            )
        if "u" in vars_requested:
            data_vars["u"] = (
                ("z", "latitude", "longitude"),
                self._pregrid_velocity(
                    paths["u"],
                    "water_u",
                    name="u",
                    target_lat=target_lat,
                    target_lon=target_lon,
                    target_points=target_points,
                ),
            )
        if "v" in vars_requested:
            data_vars["v"] = (
                ("z", "latitude", "longitude"),
                self._pregrid_velocity(
                    paths["v"],
                    "water_v",
                    name="v",
                    target_lat=target_lat,
                    target_lon=target_lon,
                    target_points=target_points,
                ),
            )
        if "eta" in vars_requested:
            data_vars["eta"] = (
                ("latitude", "longitude"),
                self._pregrid_eta(
                    paths["eta"],
                    target_lat=target_lat,
                    target_lon=target_lon,
                    target_points=target_points,
                ),
            )

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"z": self._z, "latitude": target_lat, "longitude": target_lon},
        ).expand_dims(time=[t0])
        self._target_cache[key] = ds
        return ds

    def fetch(self, query: DataQuery) -> DataBundle:
        if query.location_kind not in self.capabilities.produces:
            raise ValueError(f"hycom_raw_by_variable: unsupported kind {query.location_kind}")
        month = _month_from_query(query, self._month) if not self._steady else self._month
        paths = self._paths_for_month(month)
        t0 = _time_from_query(query, month)

        vars_requested = tuple(v for v in query.variables if v in _VARIABLES)
        data_vars: dict[str, Any] = {}
        specs: list[VariableSpec] = []

        if self._pregrid_to_target:
            ds = self._fetch_pregridded(
                query,
                month=month,
                paths=paths,
                t0=t0,
                vars_requested=vars_requested,
            )
        else:
            if "temperature" in vars_requested:
                data_vars["temperature"] = self._open_tracer(paths["temperature"], "water_temp")
            if "salinity" in vars_requested:
                data_vars["salinity"] = self._open_tracer(paths["salinity"], "salinity")
            if "u" in vars_requested:
                data_vars["u"] = self._open_velocity(paths["u"], "water_u")
            if "v" in vars_requested:
                data_vars["v"] = self._open_velocity(paths["v"], "water_v")
            if "eta" in vars_requested:
                data_vars["eta"] = self._open_2d(paths["eta"], "surf_el")

            ds = xr.Dataset(data_vars=data_vars)
            ds = self._subset_bbox(ds, query)
            ds = ds.expand_dims(time=[t0])

        for name in vars_requested:
            if name not in ds.data_vars:
                continue
            specs.append(
                VariableSpec(
                    name=name,
                    cf_standard_name=_CF_NAMES[name],
                    units=_UNITS[name],
                    dtype="float64",
                    dims=tuple(ds[name].dims),
                )
            )

        if "time" in ds.coords and len(ds["time"]) > 0:
            stamp = pd.Timestamp(ds["time"].values[0]).to_pydatetime()
            time_range: tuple[datetime, datetime] | None = (stamp, stamp)
        else:
            time_range = None

        meta_no_prov = BundleMeta(
            schema_version="0.1.0",
            location_kind=query.location_kind,
            convention=ConventionTag.CF_STANDARD,
            variables=tuple(specs),
            time_range=time_range,
            target_grid_id=None,
            provenance=(),
        )
        prov = ProvenanceStep(
            kind="source",
            name="HycomRawByVariableSource",
            params={
                "path": str(self._path),
                "month": month,
                "location_kind": query.location_kind.value,
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
