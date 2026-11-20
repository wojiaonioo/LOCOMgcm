from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import xarray as xr

from .hycom import HYCOMSurfacePatchRequest, _target_coordinate

HYCOM_SURFACE_VARIABLES = ("water_u", "water_v", "water_temp", "salinity")

# Root attr holding the resolved source path list; written after the last
# append, so a store without it is an interrupted build.
_SOURCE_PATHS_ATTR = "source_paths"


@dataclass(frozen=True)
class HYCOMZarrCacheInfo:
    store_path: Path
    created: bool


def _store_name(
    paths: list[Path],
    request: HYCOMSurfacePatchRequest,
    *,
    padding_cells: int,
    max_depth_m: float | None,
) -> str:
    """Cache directory name spelled out from the request.

    Floats go through repr, which round-trips exactly, so two requests that
    differ in one of these fields never share a name. The source list is only
    summarised (count, first and last file); the full list sits in the store
    attrs and is compared before reuse.
    """
    fields = [
        f"lat{request.center_lat!r}",
        f"lon{request.center_lon!r}",
        f"nx{request.nx}",
        f"ny{request.ny}",
        f"res{request.resolution_deg!r}",
        f"t{request.time_index}",
        request.interp_method,
        f"pad{padding_cells}",
        f"depth{max_depth_m!r}",
        f"{len(paths)}files",
    ]
    if paths:
        fields += [paths[0].stem, paths[-1].stem]
    return "hycom_surface_" + "_".join(fields) + ".zarr"


def _coordinate_slice(values: xr.DataArray, lower: float, upper: float) -> slice:
    first = float(values[0].item())
    last = float(values[-1].item())
    if first <= last:
        return slice(lower, upper)
    return slice(upper, lower)


def _subset_dataset(
    dataset: xr.Dataset,
    request: HYCOMSurfacePatchRequest,
    *,
    padding_cells: int,
    max_depth_m: float | None = None,
) -> xr.Dataset:
    target_lat = _target_coordinate(request.center_lat, request.ny, request.resolution_deg)
    target_lon = _target_coordinate(request.center_lon, request.nx, request.resolution_deg)
    lat_pad = padding_cells * request.resolution_deg
    lon_pad = padding_cells * request.resolution_deg
    lat_lower = float(target_lat.min().item()) - lat_pad
    lat_upper = float(target_lat.max().item()) + lat_pad
    lon_lower = float(target_lon.min().item()) - lon_pad
    lon_upper = float(target_lon.max().item()) + lon_pad
    if max_depth_m is not None and max_depth_m > 0:
        depth_mask = dataset.depth.values <= max_depth_m * 1.5
        n_depths = max(1, int(depth_mask.sum()))
        depth_sel = slice(0, n_depths)
    else:
        depth_sel = slice(0, 1)
    subset = dataset[list(HYCOM_SURFACE_VARIABLES)].isel(depth=depth_sel).sel(
        lat=_coordinate_slice(dataset.lat, lat_lower, lat_upper),
        lon=_coordinate_slice(dataset.lon, lon_lower, lon_upper),
    )
    return subset.chunk(
        {
            "time": 1,
            "depth": subset.sizes["depth"],
            "lat": subset.sizes["lat"],
            "lon": subset.sizes["lon"],
        }
    )


def build_hycom_zarr_cache(
    paths: list[Path],
    request: HYCOMSurfacePatchRequest,
    cache_dir: Path,
    *,
    padding_cells: int = 2,
    max_depth_m: float | None = None,
    force_rebuild: bool = False,
) -> HYCOMZarrCacheInfo:
    cache_dir.mkdir(parents=True, exist_ok=True)
    store_path = cache_dir / _store_name(
        paths, request, padding_cells=padding_cells, max_depth_m=max_depth_m
    )
    source_paths = [str(path.resolve()) for path in paths]

    if force_rebuild and store_path.exists():
        print(f"Force rebuilding cache: removing {store_path}", flush=True)
        shutil.rmtree(store_path, ignore_errors=True)

    if store_path.exists():
        try:
            with xr.open_zarr(store_path, consolidated=False, chunks={}) as dataset:
                cached_paths = dataset.attrs.get(_SOURCE_PATHS_ATTR)
        except Exception:
            cached_paths = None
        if cached_paths is not None and list(cached_paths) == source_paths:
            return HYCOMZarrCacheInfo(store_path=store_path, created=False)
        shutil.rmtree(store_path, ignore_errors=True)

    print(f"Building Zarr cache ({len(paths)} files)...", flush=True)
    for index, path in enumerate(paths):
        if index % max(1, len(paths) // 10) == 0 or index == len(paths) - 1:
            print(f"  -> Processing file {index + 1}/{len(paths)}: {path.name}", flush=True)
        with xr.open_dataset(path, engine="netcdf4") as dataset:
            subset = _subset_dataset(dataset, request, padding_cells=padding_cells, max_depth_m=max_depth_m)
            subset.to_zarr(
                store_path,
                mode="w" if index == 0 else "a",
                append_dim=None if index == 0 else "time",
                consolidated=False,
            )
    import zarr

    zarr.open_group(str(store_path), mode="r+").attrs[_SOURCE_PATHS_ATTR] = source_paths

    return HYCOMZarrCacheInfo(store_path=store_path, created=True)


def open_hycom_zarr_cache(store_path: str | Path) -> xr.Dataset:
    return xr.open_zarr(Path(store_path), consolidated=False, chunks={})
