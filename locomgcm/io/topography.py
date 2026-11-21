"""Topography processing: load GMRT bathymetry, interpolate to model grid."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import xarray as xr

from locomgcm.io.hycom import _target_coordinate


@dataclass(frozen=True)
class ModelBathymetry:
    """Bathymetry on the model grid.

    Attributes:
        depth: [ny, nx] 2D tensor.  Positive = ocean depth below sea level.
                                     Zero or negative = land.
        lat: 1-D target latitudes.
        lon: 1-D target longitudes.
    """

    depth: torch.Tensor
    lat: torch.Tensor
    lon: torch.Tensor

    @property
    def ocean_mask(self) -> torch.Tensor:
        """Binary mask: True where ocean (depth > 0)."""
        return self.depth > 0.0


def load_bathymetry(
    path: str | Path,
    *,
    center_lat: float = 15.0,
    center_lon: float = 115.0,
    nx: int = 50,
    ny: int = 50,
    resolution_deg: float = 1.0 / 12.0,
    max_depth: float = 200.0,
    variable: str = "z",
) -> ModelBathymetry:
    """Load GMRT-style topography and interpolate to the model grid.

    The GMRT `z` variable is elevation (positive above sea level,
    negative below).  We convert to ocean depth (positive below sea level).

    Parameters
    ----------
    path : str or Path
        Path to the topography NetCDF (e.g. topo_27.1.nc).
    center_lat, center_lon, nx, ny, resolution_deg :
        Model grid specification.
    max_depth : float
        Clip ocean depth to this value (for the model's total depth).
    variable : str
        Name of the elevation variable in the file.
    """
    ds = xr.open_dataset(Path(path))
    try:
        return _extract_bathymetry(
            ds,
            center_lat=center_lat,
            center_lon=center_lon,
            nx=nx,
            ny=ny,
            resolution_deg=resolution_deg,
            max_depth=max_depth,
            variable=variable,
        )
    finally:
        ds.close()


def _extract_bathymetry(
    ds: xr.Dataset,
    *,
    center_lat: float,
    center_lon: float,
    nx: int,
    ny: int,
    resolution_deg: float,
    max_depth: float,
    variable: str,
) -> ModelBathymetry:
    target_lat = _target_coordinate(center_lat, ny, resolution_deg)
    target_lon = _target_coordinate(center_lon, nx, resolution_deg)

    # Interpolate to model grid
    interpolated = ds[variable].interp(
        lat=xr.DataArray(target_lat.numpy(), dims=("lat",)),
        lon=xr.DataArray(target_lon.numpy(), dims=("lon",)),
        method="linear",
    )

    # Convert elevation to depth: depth = -elevation (positive below sea level)
    elevation = torch.as_tensor(interpolated.values, dtype=torch.float64)
    depth = -elevation

    # Land (elevation >= 0) → depth <= 0 → set to 0
    depth = torch.clamp(depth, min=0.0, max=max_depth)

    return ModelBathymetry(
        depth=depth,
        lat=target_lat,
        lon=target_lon,
    )
