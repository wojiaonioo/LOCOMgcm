"""Load WOA23 annual climatology temperature and salinity.

WOA23 ``decav`` annual climatology files (``woa23_decav_t00_04.nc`` and
``woa23_decav_s00_04.nc``) provide 0.25° global temperature (``t_an``) and
salinity (``s_an``) on 102 standard depth levels (0 m → 5500 m).

This module interpolates a regional subset of WOA23 onto the SCS model grid
defined by ``target_lat`` / ``target_lon`` / ``target_depths``, producing
``[nz, ny, nx]`` torch tensors compatible with
:class:`locomgcm.io.hycom.HYCOMFullDepthPatch`.

The land mask in WOA23 (NaN / _FillValue) is filled along depth → longitude →
latitude with ``ffill+bfill`` so the resulting tensors contain no NaN. Land
cells in the model grid are masked elsewhere by the dycore via topo-derived
hFac, so the values written to dry cells are placeholders.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr


WOA23_TEMP_FILENAME = "woa23_decav_t00_04.nc"
WOA23_SALT_FILENAME = "woa23_decav_s00_04.nc"
WOA23_TEMP_VARIABLE = "t_an"
WOA23_SALT_VARIABLE = "s_an"


def _interp_field(
    da: xr.DataArray,
    *,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_depths: np.ndarray,
) -> torch.Tensor:
    lat_min = float(target_lat.min()) - 0.5
    lat_max = float(target_lat.max()) + 0.5
    lon_min = float(target_lon.min()) - 0.5
    lon_max = float(target_lon.max()) + 0.5

    sub = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    # Fill below-bottom NaNs along depth so each WOA column has a valid value
    # for every depth (deep cells get ffill'd from the deepest valid level).
    sub = sub.ffill(dim="depth").bfill(dim="depth")

    vertically = sub.interp(
        depth=xr.DataArray(target_depths, dims=("depth",)),
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # Fill horizontal NaNs ON THE WOA SOURCE GRID (per depth level) BEFORE
    # interpolating to the model grid. Without this, xarray.interp returns
    # NaN whenever any of the 4 corner cells is land, and the post-interp
    # lon-fill grabs values from a shallow-shelf neighbor whose deep levels
    # were ffill'd from the SURFACE, producing a wrong constant column at
    # deep ocean cells near coasts.
    vertically = (
        vertically
        .ffill(dim="lon").bfill(dim="lon")
        .ffill(dim="lat").bfill(dim="lat")
    )

    horizontally = vertically.interp(
        lat=xr.DataArray(target_lat, dims=("lat",)),
        lon=xr.DataArray(target_lon, dims=("lon",)),
        method="linear",
    )

    values = torch.as_tensor(horizontally.values.copy(), dtype=torch.float64)
    return torch.nan_to_num(values, nan=0.0)


def load_woa23_temp_salt(
    woa23_dir: str | Path,
    *,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_depths: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(temp, salt)`` torch tensors of shape ``[nz, ny, nx]``."""
    woa23_dir = Path(woa23_dir)
    t_path = woa23_dir / WOA23_TEMP_FILENAME
    s_path = woa23_dir / WOA23_SALT_FILENAME
    if not t_path.exists():
        raise FileNotFoundError(f"WOA23 temperature file not found: {t_path}")
    if not s_path.exists():
        raise FileNotFoundError(f"WOA23 salinity file not found: {s_path}")

    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    target_depths = np.asarray(target_depths, dtype=np.float64)

    with xr.open_dataset(t_path, decode_times=False) as t_ds:
        t_an = t_ds[WOA23_TEMP_VARIABLE].isel(time=0)
        temp = _interp_field(
            t_an,
            target_lat=target_lat,
            target_lon=target_lon,
            target_depths=target_depths,
        )
    with xr.open_dataset(s_path, decode_times=False) as s_ds:
        s_an = s_ds[WOA23_SALT_VARIABLE].isel(time=0)
        salt = _interp_field(
            s_an,
            target_lat=target_lat,
            target_lon=target_lon,
            target_depths=target_depths,
        )

    return temp, salt
