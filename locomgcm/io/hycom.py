from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import ceil, cos, pi
from pathlib import Path

import numpy as np

import torch
import xarray as xr

from locomgcm.geometry.grid import _compute_interior_mask
from locomgcm.model import TorchRegionalOceanModel
from locomgcm.operators.reconstructions import center_to_u_centered, center_to_v_centered

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class HYCOMSurfacePatchRequest:
    center_lat: float = 15.0
    center_lon: float = 115.0
    nx: int = 50
    ny: int = 50
    resolution_deg: float = 0.25
    time_index: int = 0
    interp_method: str = "linear"
    cfl_safety_factor: float = 0.5


@dataclass(frozen=True)
class HYCOMSurfacePatch:
    lat: torch.Tensor
    lon: torch.Tensor
    u_center: torch.Tensor
    v_center: torch.Tensor
    temp: torch.Tensor
    salt: torch.Tensor
    source_depth_m: float
    time_value: str
    dx_m: float
    dy_m: float
    max_abs_u: float
    max_abs_v: float
    stable_dt_seconds: float
    filled_point_count: int


def _meters_per_degree_lat() -> float:
    return pi * EARTH_RADIUS_M / 180.0


def _meters_per_degree_lon(lat_deg: float) -> float:
    return _meters_per_degree_lat() * cos(pi * lat_deg / 180.0)


def _target_coordinate(center: float, count: int, spacing_deg: float) -> torch.Tensor:
    offsets = torch.arange(count, dtype=torch.float64) - 0.5 * (count - 1)
    return center + offsets * spacing_deg


def _estimate_stable_dt(max_abs_u: float, max_abs_v: float, dx_m: float, dy_m: float, safety_factor: float) -> float:
    speed_sum = max_abs_u / max(dx_m, 1.0e-12) + max_abs_v / max(dy_m, 1.0e-12)
    if speed_sum == 0.0:
        return float("inf")
    return safety_factor / speed_sum



def choose_stable_timestep(total_seconds: float, stable_dt_seconds: float) -> tuple[int, float]:
    if total_seconds <= 0.0:
        raise ValueError("total_seconds must be positive")
    if stable_dt_seconds <= 0.0:
        raise ValueError("stable_dt_seconds must be positive")
    if stable_dt_seconds == float("inf"):
        return 1, total_seconds
    steps = max(1, ceil(total_seconds / stable_dt_seconds))
    dt_seconds = total_seconds / steps
    return steps, dt_seconds


def _missing_point_count(patch: "xr.Dataset") -> int:
    """Points the horizontal gap fill will supply: the largest NaN count over the fields."""
    return max(int(patch[name].isnull().sum()) for name in patch.data_vars)


def extract_surface_patch(dataset: xr.Dataset, request: HYCOMSurfacePatchRequest) -> HYCOMSurfacePatch:
    target_lat = _target_coordinate(request.center_lat, request.ny, request.resolution_deg)
    target_lon = _target_coordinate(request.center_lon, request.nx, request.resolution_deg)

    lat_min = float(dataset.lat.min())
    lat_max = float(dataset.lat.max())
    lon_min = float(dataset.lon.min())
    lon_max = float(dataset.lon.max())
    if float(target_lat.min()) < lat_min or float(target_lat.max()) > lat_max:
        raise ValueError("target latitude range falls outside the HYCOM domain")
    if float(target_lon.min()) < lon_min or float(target_lon.max()) > lon_max:
        raise ValueError("target longitude range falls outside the HYCOM domain")

    surface = dataset[["water_u", "water_v", "water_temp", "salinity"]].isel(time=request.time_index, depth=0)
    surface = surface.compute()
    # HYCOM encodes land as 0 for all fields; convert to NaN so ffill/bfill
    # can extrapolate valid ocean values across coastal and bottom gaps.
    is_ocean = (surface["water_temp"] != 0.0) | (surface["salinity"] != 0.0)
    surface = surface.where(is_ocean)
    interpolated = surface.interp(
        lat=xr.DataArray(target_lat.numpy(), dims=("lat",)),
        lon=xr.DataArray(target_lon.numpy(), dims=("lon",)),
        method=request.interp_method,
    )
    # HYCOM land (made NaN above) and missing data, filled below.
    filled_point_count = _missing_point_count(interpolated)
    # Native xarray robust extrapolation: fill spatially across missing gaps
    interpolated = interpolated.ffill(dim="lon").bfill(dim="lon").ffill(dim="lat").bfill(dim="lat")

    tensors: dict[str, torch.Tensor] = {}
    for source_name, target_name in (
        ("water_u", "u_center"),
        ("water_v", "v_center"),
        ("water_temp", "temp"),
        ("salinity", "salt"),
    ):
        raw_vals = torch.as_tensor(interpolated[source_name].values.copy(), dtype=torch.float64)
        n_nan = int(torch.isnan(raw_vals).sum().item())
        if n_nan > 0:
            warnings.warn(
                f"{source_name}: {n_nan} NaN values remain after ffill/bfill extrapolation",
                stacklevel=2,
            )
        tensors[target_name] = torch.nan_to_num(raw_vals, nan=0.0)

    dx_m = request.resolution_deg * _meters_per_degree_lon(request.center_lat)
    dy_m = request.resolution_deg * _meters_per_degree_lat()
    max_abs_u = float(tensors["u_center"].abs().max().item())
    max_abs_v = float(tensors["v_center"].abs().max().item())
    stable_dt_seconds = _estimate_stable_dt(
        max_abs_u=max_abs_u,
        max_abs_v=max_abs_v,
        dx_m=dx_m,
        dy_m=dy_m,
        safety_factor=request.cfl_safety_factor,
    )

    time_value = str(dataset.time.isel(time=request.time_index).values)
    return HYCOMSurfacePatch(
        lat=torch.as_tensor(interpolated.lat.values, dtype=torch.float64),
        lon=torch.as_tensor(interpolated.lon.values, dtype=torch.float64),
        u_center=tensors["u_center"],
        v_center=tensors["v_center"],
        temp=tensors["temp"],
        salt=tensors["salt"],
        source_depth_m=float(dataset.depth.isel(depth=0).item()),
        time_value=time_value,
        dx_m=float(dx_m),
        dy_m=float(dy_m),
        max_abs_u=max_abs_u,
        max_abs_v=max_abs_v,
        stable_dt_seconds=float(stable_dt_seconds),
        filled_point_count=filled_point_count,
    )


def load_surface_patch(path: str | Path, request: HYCOMSurfacePatchRequest) -> HYCOMSurfacePatch:
    dataset = xr.open_dataset(Path(path), engine="netcdf4")
    try:
        return extract_surface_patch(dataset, request)
    finally:
        dataset.close()


def _replicate_surface_to_levels(field_2d: torch.Tensor, nz: int) -> torch.Tensor:
    if nz < 1:
        raise ValueError("nz must be positive")
    return field_2d.unsqueeze(0).expand(nz, -1, -1).contiguous()


def patch_to_model_state(model: TorchRegionalOceanModel, patch: HYCOMSurfacePatch):
    center_shape = (1, model.grid.nz, model.grid.ny, model.grid.nx)
    u_center = _replicate_surface_to_levels(patch.u_center, model.grid.nz).to(
        dtype=model.config.dtype,
        device=model.config.device,
    ).reshape(center_shape)
    v_center = _replicate_surface_to_levels(patch.v_center, model.grid.nz).to(
        dtype=model.config.dtype,
        device=model.config.device,
    ).reshape(center_shape)
    temp = _replicate_surface_to_levels(patch.temp, model.grid.nz).to(
        dtype=model.config.dtype,
        device=model.config.device,
    ).reshape(center_shape)
    salt = _replicate_surface_to_levels(patch.salt, model.grid.nz).to(
        dtype=model.config.dtype,
        device=model.config.device,
    ).reshape(center_shape)

    u_face = center_to_u_centered(u_center, model.grid, face_mask=None)
    v_face = center_to_v_centered(v_center, model.grid, face_mask=None)

    # Dry faces are cleared by selection, not by multiplying with a 0/1 mask:
    # 0 * NaN is NaN, so a NaN left on a dry face by the external input would
    # survive a multiplicative mask (numerics rule 10).
    masks = model.grid.masks
    if masks.hFacW is not None:
        hfW = masks.hFacW.to(dtype=u_face.dtype, device=u_face.device)
        if hfW.ndim == 3:
            hfW = hfW.unsqueeze(0)
        u_face = torch.where(hfW > 0, u_face, torch.zeros_like(u_face))
    if masks.hFacS is not None:
        hfS = masks.hFacS.to(dtype=v_face.dtype, device=v_face.device)
        if hfS.ndim == 3:
            hfS = hfS.unsqueeze(0)
        v_face = torch.where(hfS > 0, v_face, torch.zeros_like(v_face))

    return model.create_state(u=u_face, v=v_face, temp=temp, salt=salt)


@dataclass(frozen=True)
class HYCOMFullDepthPatch:
    """Full-depth HYCOM data patch with [nz, ny, nx] fields."""

    lat: torch.Tensor
    lon: torch.Tensor
    u_center: torch.Tensor   # [nz, ny, nx]
    v_center: torch.Tensor   # [nz, ny, nx]
    temp: torch.Tensor        # [nz, ny, nx]
    salt: torch.Tensor        # [nz, ny, nx]
    depth_values: torch.Tensor  # [nz] — model/HYCOM target center depths in meters
    time_value: str
    dx_m: float
    dy_m: float
    max_abs_u: float
    max_abs_v: float
    stable_dt_seconds: float
    filled_point_count: int


def extract_full_depth_patch(
    dataset: xr.Dataset,
    request: HYCOMSurfacePatchRequest,
    *,
    depth_indices: list[int] | None = None,
) -> HYCOMFullDepthPatch:
    """Extract a full-depth HYCOM patch with [nz, ny, nx] shaped fields.

    Parameters
    ----------
    dataset : xr.Dataset — HYCOM dataset with (time, depth, lat, lon) dims
    request : HYCOMSurfacePatchRequest — defines center, extent, resolution
    depth_indices : list[int] | None — which depth indices to read.
        None = all available depths.

    Returns
    -------
    HYCOMFullDepthPatch with 3D fields.
    """
    target_lat = _target_coordinate(request.center_lat, request.ny, request.resolution_deg)
    target_lon = _target_coordinate(request.center_lon, request.nx, request.resolution_deg)

    lat_min = float(dataset.lat.min())
    lat_max = float(dataset.lat.max())
    lon_min = float(dataset.lon.min())
    lon_max = float(dataset.lon.max())
    if float(target_lat.min()) < lat_min or float(target_lat.max()) > lat_max:
        raise ValueError("target latitude range falls outside the HYCOM domain")
    if float(target_lon.min()) < lon_min or float(target_lon.max()) > lon_max:
        raise ValueError("target longitude range falls outside the HYCOM domain")

    if depth_indices is None:
        depth_indices = list(range(dataset.sizes["depth"]))

    subset = dataset[["water_u", "water_v", "water_temp", "salinity"]].isel(
        time=request.time_index,
        depth=depth_indices,
    )
    subset = subset.compute()
    is_ocean = (subset["water_temp"] != 0.0) | (subset["salinity"] != 0.0)
    subset = subset.where(is_ocean)
    interpolated = subset.interp(
        lat=xr.DataArray(target_lat.numpy(), dims=("lat",)),
        lon=xr.DataArray(target_lon.numpy(), dims=("lon",)),
        method=request.interp_method,
    )
    
    # Native xarray robust extrapolation
    filled_point_count = _missing_point_count(interpolated)
    interpolated = interpolated.ffill(dim="lon").bfill(dim="lon").ffill(dim="lat").bfill(dim="lat")
    if "depth" in interpolated.dims:
        interpolated = interpolated.ffill(dim="depth").bfill(dim="depth")

    field_names = [("water_u", "u_center"), ("water_v", "v_center"),
                   ("water_temp", "temp"), ("salinity", "salt")]
    tensors: dict[str, torch.Tensor] = {}
    for source_name, target_name in field_names:
        raw_vals = torch.as_tensor(interpolated[source_name].values.copy(), dtype=torch.float64)
        n_nan = int(torch.isnan(raw_vals).sum().item())
        if n_nan > 0:
            warnings.warn(
                f"{source_name}: {n_nan} NaN values remain after ffill/bfill extrapolation",
                stacklevel=2,
            )
        tensors[target_name] = torch.nan_to_num(raw_vals, nan=0.0)

    depth_values = torch.as_tensor(
        dataset.depth.isel(depth=depth_indices).values, dtype=torch.float64,
    )
    dx_m = request.resolution_deg * _meters_per_degree_lon(request.center_lat)
    dy_m = request.resolution_deg * _meters_per_degree_lat()
    max_abs_u = float(tensors["u_center"].abs().max().item())
    max_abs_v = float(tensors["v_center"].abs().max().item())
    stable_dt_seconds = _estimate_stable_dt(
        max_abs_u=max_abs_u,
        max_abs_v=max_abs_v,
        dx_m=dx_m,
        dy_m=dy_m,
        safety_factor=request.cfl_safety_factor,
    )
    time_value = str(dataset.time.isel(time=request.time_index).values)

    return HYCOMFullDepthPatch(
        lat=torch.as_tensor(interpolated.lat.values, dtype=torch.float64),
        lon=torch.as_tensor(interpolated.lon.values, dtype=torch.float64),
        u_center=tensors["u_center"],
        v_center=tensors["v_center"],
        temp=tensors["temp"],
        salt=tensors["salt"],
        depth_values=depth_values,
        time_value=time_value,
        dx_m=float(dx_m),
        dy_m=float(dy_m),
        max_abs_u=max_abs_u,
        max_abs_v=max_abs_v,
        stable_dt_seconds=float(stable_dt_seconds),
        filled_point_count=filled_point_count,
    )


def extract_interpolated_3d_patch(
    dataset: xr.Dataset,
    request: HYCOMSurfacePatchRequest,
    *,
    target_depths: np.ndarray,
) -> HYCOMFullDepthPatch:
    """Extract a full-depth HYCOM patch, interpolating native depths to target depths.

    Parameters
    ----------
    dataset : xr.Dataset — HYCOM dataset with (time, depth, lat, lon) dims
    request : HYCOMSurfacePatchRequest — defines center, extent, resolution
    target_depths : np.ndarray — vertical target coordinates to interpolate

    Returns
    -------
    HYCOMFullDepthPatch with interpolated 3D fields.
    """
    target_lat = _target_coordinate(request.center_lat, request.ny, request.resolution_deg)
    target_lon = _target_coordinate(request.center_lon, request.nx, request.resolution_deg)

    lat_min = float(dataset.lat.min())
    lat_max = float(dataset.lat.max())
    lon_min = float(dataset.lon.min())
    lon_max = float(dataset.lon.max())
    if float(target_lat.min()) < lat_min or float(target_lat.max()) > lat_max:
        raise ValueError("target latitude range falls outside the HYCOM domain")
    if float(target_lon.min()) < lon_min or float(target_lon.max()) > lon_max:
        raise ValueError("target longitude range falls outside the HYCOM domain")

    pad = 0.5  # Add buffering before interpolation
    subset = dataset[["water_u", "water_v", "water_temp", "salinity"]].isel(
        time=request.time_index,
    ).sel(
        lat=slice(lat_min - 1e-5 if float(target_lat.min()) - pad < lat_min else float(target_lat.min()) - pad, 
                  lat_max + 1e-5 if float(target_lat.max()) + pad > lat_max else float(target_lat.max()) + pad),
        lon=slice(lon_min - 1e-5 if float(target_lon.min()) - pad < lon_min else float(target_lon.min()) - pad,
                  lon_max + 1e-5 if float(target_lon.max()) + pad > lon_max else float(target_lon.max()) + pad),
    )
    subset = subset.compute()
    is_ocean = (subset["water_temp"] != 0.0) | (subset["salinity"] != 0.0)
    subset = subset.where(is_ocean)
    
    # Ensure vertical continuous validity BEFORE interpolating target depths
    if "depth" in subset.dims:
        subset = subset.ffill(dim="depth").bfill(dim="depth")

    # 1. Interpolate vertically first
    vertically_interpolated = subset.interp(
        depth=xr.DataArray(target_depths, dims=("depth",)),
        method="linear",
        kwargs={"fill_value": "extrapolate"}
    )
    
    # 2. Interpolate horizontally
    interpolated = vertically_interpolated.interp(
        lat=xr.DataArray(target_lat.numpy(), dims=("lat",)),
        lon=xr.DataArray(target_lon.numpy(), dims=("lon",)),
        method=request.interp_method,
    )
    
    # 3. Robust extrapolation for any entirely missing chunks
    filled_point_count = _missing_point_count(interpolated)
    interpolated = interpolated.ffill(dim="lon").bfill(dim="lon").ffill(dim="lat").bfill(dim="lat")

    field_names = [("water_u", "u_center"), ("water_v", "v_center"),
                   ("water_temp", "temp"), ("salinity", "salt")]
    tensors: dict[str, torch.Tensor] = {}
    for source_name, target_name in field_names:
        raw_vals = torch.as_tensor(interpolated[source_name].values.copy(), dtype=torch.float64)
        n_nan = int(torch.isnan(raw_vals).sum().item())
        if n_nan > 0:
            warnings.warn(
                f"{source_name}: {n_nan} NaN values remain after ffill/bfill extrapolation",
                stacklevel=2,
            )
        tensors[target_name] = torch.nan_to_num(raw_vals, nan=0.0)

    dx_m = request.resolution_deg * _meters_per_degree_lon(request.center_lat)
    dy_m = request.resolution_deg * _meters_per_degree_lat()
    max_abs_u = float(tensors["u_center"].abs().max().item())
    max_abs_v = float(tensors["v_center"].abs().max().item())
    stable_dt_seconds = _estimate_stable_dt(
        max_abs_u=max_abs_u,
        max_abs_v=max_abs_v,
        dx_m=dx_m,
        dy_m=dy_m,
        safety_factor=request.cfl_safety_factor,
    )
    time_value = str(dataset.time.isel(time=request.time_index).values)

    return HYCOMFullDepthPatch(
        lat=torch.as_tensor(interpolated.lat.values, dtype=torch.float64),
        lon=torch.as_tensor(interpolated.lon.values, dtype=torch.float64),
        u_center=tensors["u_center"],
        v_center=tensors["v_center"],
        temp=tensors["temp"],
        salt=tensors["salt"],
        depth_values=torch.as_tensor(target_depths, dtype=torch.float64),
        time_value=time_value,
        dx_m=float(dx_m),
        dy_m=float(dy_m),
        max_abs_u=max_abs_u,
        max_abs_v=max_abs_v,
        stable_dt_seconds=float(stable_dt_seconds),
        filled_point_count=filled_point_count,
    )


def load_full_depth_patch(
    path: str | Path,
    request: HYCOMSurfacePatchRequest,
    *,
    depth_indices: list[int] | None = None,
) -> HYCOMFullDepthPatch:
    """Load a full-depth HYCOM patch from a NetCDF file."""
    dataset = xr.open_dataset(Path(path), engine="netcdf4")
    try:
        return extract_full_depth_patch(dataset, request, depth_indices=depth_indices)
    finally:
        dataset.close()


def full_depth_patch_to_model_state(
    model: TorchRegionalOceanModel,
    patch: HYCOMFullDepthPatch,
) -> ...:
    """Convert a full-depth HYCOM patch to model state (3D fields, no replication).

    If model.grid.nz differs from patch nz, the patch is vertically
    interpolated or truncated to match.
    """
    nz_patch = patch.u_center.shape[0]
    nz_model = model.grid.nz
    center_shape = (1, nz_model, model.grid.ny, model.grid.nx)

    if nz_patch >= nz_model:
        # Truncate to model depth
        u_center = patch.u_center[:nz_model].to(
            dtype=model.config.dtype, device=model.config.device,
        ).unsqueeze(0)
        v_center = patch.v_center[:nz_model].to(
            dtype=model.config.dtype, device=model.config.device,
        ).unsqueeze(0)
        temp = patch.temp[:nz_model].to(
            dtype=model.config.dtype, device=model.config.device,
        ).unsqueeze(0)
        salt = patch.salt[:nz_model].to(
            dtype=model.config.dtype, device=model.config.device,
        ).unsqueeze(0)
    else:
        # Pad by repeating deepest layer
        def _pad(field: torch.Tensor) -> torch.Tensor:
            extra = nz_model - nz_patch
            bottom = field[-1:].expand(extra, -1, -1)
            return torch.cat([field, bottom], dim=0)

        u_center = _pad(patch.u_center).to(
            dtype=model.config.dtype, device=model.config.device,
        ).unsqueeze(0)
        v_center = _pad(patch.v_center).to(
            dtype=model.config.dtype, device=model.config.device,
        ).unsqueeze(0)
        temp = _pad(patch.temp).to(
            dtype=model.config.dtype, device=model.config.device,
        ).unsqueeze(0)
        salt = _pad(patch.salt).to(
            dtype=model.config.dtype, device=model.config.device,
        ).unsqueeze(0)

    u_face = center_to_u_centered(u_center, model.grid, face_mask=None)
    v_face = center_to_v_centered(v_center, model.grid, face_mask=None)

    # Zero out fields in dry cells (MITgcm: ini_vel.F masks by _maskW/_maskS,
    # ini_theta.F/ini_salt.F mask by maskC when maskIniTemp/maskIniSalt=.TRUE.).
    # This prevents extrapolated HYCOM data below bathymetry from persisting in
    # the model state, ensuring physical consistency even if downstream operators
    # do not all apply their own masks.  The clearing selects zero on dry cells
    # instead of multiplying by a 0/1 mask, because 0 * NaN is NaN (numerics
    # rule 10).
    masks = model.grid.masks
    if masks.hFacC is not None:
        hfC = masks.hFacC.to(dtype=temp.dtype, device=temp.device)
        if hfC.ndim == 3:
            hfC = hfC.unsqueeze(0)  # [1, nz, ny, nx]
        wetC = hfC > 0
        temp = torch.where(wetC, temp, torch.zeros_like(temp))
        salt = torch.where(wetC, salt, torch.zeros_like(salt))

    if masks.hFacW is not None:
        hfW = masks.hFacW.to(dtype=u_face.dtype, device=u_face.device)
        if hfW.ndim == 3:
            hfW = hfW.unsqueeze(0)
        u_face = torch.where(hfW > 0, u_face, torch.zeros_like(u_face))
    if masks.hFacS is not None:
        hfS = masks.hFacS.to(dtype=v_face.dtype, device=v_face.device)
        if hfS.ndim == 3:
            hfS = hfS.unsqueeze(0)
        v_face = torch.where(hfS > 0, v_face, torch.zeros_like(v_face))

    # Also zero velocities at OBC strip faces (MITgcm: init_varia / OBCS_APPLY_UV
    # sets U/V at the first ghost row/column to 0 unless the user prescribes a
    # non-zero boundary flow).  Without this, HYCOM-interpolated velocities in
    # the boundary-strip cells (maskInC=0) linger and leak into interior
    # dynamics through AB2 history and sponge terms, creating a persistent
    # 0.3-1.0 m/s offset between LOCOMgcm and MITgcm at step 0.
    topology = model.grid.boundary_topology
    if topology.open_x or topology.open_y:
        _, maskInW, maskInS = _compute_interior_mask(
            model.grid, u_face.dtype, u_face.device,
        )
        # maskInW: [ny, nx+1], maskInS: [ny+1, nx]
        u_face = u_face * maskInW.unsqueeze(0).unsqueeze(0)
        v_face = v_face * maskInS.unsqueeze(0).unsqueeze(0)

    return model.create_state(u=u_face, v=v_face, temp=temp, salt=salt)
