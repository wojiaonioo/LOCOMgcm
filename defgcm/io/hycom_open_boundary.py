from __future__ import annotations

from defgcm.sampling import SampleClock, SamplingRule

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from enum import StrEnum
import math
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from defgcm.boundaries import BoundarySide, SideBoundaryTensorSet

from .hycom import HYCOMSurfacePatchRequest, _target_coordinate


class BoundaryTimeMode(StrEnum):
    CONSTANT = "constant"
    LINEAR = "linear"


@dataclass(frozen=True)
class BoundaryInterpolationInfo:
    mode: str
    model_time_seconds: float
    sample_time_seconds: float
    update_interval_seconds: float | None
    lower_index: int
    upper_index: int
    lower_time_seconds: float
    upper_time_seconds: float
    lower_time_value: str
    upper_time_value: str
    lower_weight: float
    upper_weight: float

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "model_time_seconds": self.model_time_seconds,
            "sample_time_seconds": self.sample_time_seconds,
            "update_interval_seconds": self.update_interval_seconds,
            "lower_index": self.lower_index,
            "upper_index": self.upper_index,
            "lower_time_seconds": self.lower_time_seconds,
            "upper_time_seconds": self.upper_time_seconds,
            "lower_time_value": self.lower_time_value,
            "upper_time_value": self.upper_time_value,
            "lower_weight": self.lower_weight,
            "upper_weight": self.upper_weight,
        }


@dataclass(frozen=True)
class OpenBoundaryField:
    sampling_rule = SamplingRule(SampleClock.END)

    times: torch.Tensor
    time_values: tuple[str, ...]
    west: torch.Tensor | None = None
    east: torch.Tensor | None = None
    south: torch.Tensor | None = None
    north: torch.Tensor | None = None
    _times_list: list[float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_times_list", [float(value) for value in self.times.tolist()])

    def at(
        self,
        sample_time_seconds: float,
        *,
        mode: str | BoundaryTimeMode = BoundaryTimeMode.LINEAR,
        model_time_seconds: float | None = None,
        update_interval_seconds: float | None = None,
    ) -> tuple[SideBoundaryTensorSet, BoundaryInterpolationInfo]:
        interpolation = self._interpolation_info(
            sample_time_seconds,
            mode=mode,
            model_time_seconds=model_time_seconds,
            update_interval_seconds=update_interval_seconds,
        )
        sampled: dict[str, torch.Tensor | None] = {}
        for side in BoundarySide:
            side_values = getattr(self, side.value)
            if side_values is None:
                sampled[side.value] = None
                continue
            lower = side_values[interpolation.lower_index]
            upper = side_values[interpolation.upper_index]
            sampled[side.value] = interpolation.lower_weight * lower + interpolation.upper_weight * upper
        return (
            SideBoundaryTensorSet(
                west=sampled["west"],
                east=sampled["east"],
                south=sampled["south"],
                north=sampled["north"],
            ),
            interpolation,
        )

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "OpenBoundaryField":
        def _move(values: torch.Tensor | None) -> torch.Tensor | None:
            if values is None:
                return None
            kwargs: dict[str, object] = {"device": device}
            if dtype is not None:
                kwargs["dtype"] = dtype
            return values.to(**kwargs)

        return OpenBoundaryField(
            times=self.times.to(device=device, dtype=torch.float64),
            time_values=self.time_values,
            west=_move(self.west),
            east=_move(self.east),
            south=_move(self.south),
            north=_move(self.north),
        )

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "OpenBoundaryField":
        return self.to_runtime(device, dtype)

    def as_dict(self) -> dict[str, object]:
        return {
            "times": self.times,
            "time_values": list(self.time_values),
            "west": self.west,
            "east": self.east,
            "south": self.south,
            "north": self.north,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OpenBoundaryField":
        return cls(
            times=torch.as_tensor(payload["times"], dtype=torch.float64),
            time_values=tuple(str(value) for value in payload["time_values"]),
            west=payload.get("west"),
            east=payload.get("east"),
            south=payload.get("south"),
            north=payload.get("north"),
        )

    def _interpolation_info(
        self,
        sample_time_seconds: float,
        *,
        mode: str | BoundaryTimeMode,
        model_time_seconds: float | None = None,
        update_interval_seconds: float | None = None,
    ) -> BoundaryInterpolationInfo:
        if self.times.ndim != 1 or self.times.numel() < 1:
            raise ValueError("boundary times must be a non-empty 1D tensor")

        actual_mode = BoundaryTimeMode(mode)
        requested_model_time_seconds = float(sample_time_seconds if model_time_seconds is None else model_time_seconds)
        times_list = self._times_list
        num_times = len(times_list)
        target = min(max(float(sample_time_seconds), times_list[0]), times_list[-1])
        if actual_mode == BoundaryTimeMode.CONSTANT or num_times == 1:
            index = bisect_right(times_list, target) - 1
            index = max(0, min(index, num_times - 1))
            time_seconds = times_list[index]
            time_value = self.time_values[index]
            return BoundaryInterpolationInfo(
                mode=actual_mode.value,
                model_time_seconds=requested_model_time_seconds,
                sample_time_seconds=float(sample_time_seconds),
                update_interval_seconds=update_interval_seconds,
                lower_index=index,
                upper_index=index,
                lower_time_seconds=time_seconds,
                upper_time_seconds=time_seconds,
                lower_time_value=time_value,
                upper_time_value=time_value,
                lower_weight=1.0,
                upper_weight=0.0,
            )

        upper_index = bisect_left(times_list, target)
        if upper_index <= 0:
            lower_index = 0
            upper_index = 0
        elif upper_index >= num_times:
            lower_index = num_times - 1
            upper_index = lower_index
        else:
            lower_index = upper_index - 1
            if times_list[upper_index] == target:
                lower_index = upper_index

        lower_time_seconds = times_list[lower_index]
        upper_time_seconds = times_list[upper_index]
        if lower_index == upper_index or upper_time_seconds == lower_time_seconds:
            lower_weight = 1.0
            upper_weight = 0.0
        else:
            upper_weight = (target - lower_time_seconds) / (upper_time_seconds - lower_time_seconds)
            lower_weight = 1.0 - upper_weight
        return BoundaryInterpolationInfo(
            mode=actual_mode.value,
            model_time_seconds=requested_model_time_seconds,
            sample_time_seconds=float(sample_time_seconds),
            update_interval_seconds=update_interval_seconds,
            lower_index=lower_index,
            upper_index=upper_index,
            lower_time_seconds=lower_time_seconds,
            upper_time_seconds=upper_time_seconds,
            lower_time_value=self.time_values[lower_index],
            upper_time_value=self.time_values[upper_index],
            lower_weight=float(lower_weight),
            upper_weight=float(upper_weight),
        )


@dataclass(frozen=True)
class HYCOMTracerOpenBoundary:
    temp: SideBoundaryTensorSet
    salt: SideBoundaryTensorSet
    source_depth_m: float
    time_value: str
    filled_point_count: int
    u: SideBoundaryTensorSet | None = None
    v: SideBoundaryTensorSet | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "temp": self.temp.as_dict(),
            "salt": self.salt.as_dict(),
            "u": self.u.as_dict() if self.u else {},
            "v": self.v.as_dict() if self.v else {},
            "source_depth_m": self.source_depth_m,
            "time_value": self.time_value,
            "filled_point_count": self.filled_point_count,
        }


@dataclass(frozen=True)
class HYCOMOpenBoundarySample:
    temp: SideBoundaryTensorSet
    salt: SideBoundaryTensorSet
    interpolation: BoundaryInterpolationInfo
    source_depth_m: float
    filled_point_count: int
    u: SideBoundaryTensorSet | None = None
    v: SideBoundaryTensorSet | None = None

    def as_boundary(self) -> HYCOMTracerOpenBoundary:
        return HYCOMTracerOpenBoundary(
            temp=self.temp,
            salt=self.salt,
            u=self.u,
            v=self.v,
            source_depth_m=self.source_depth_m,
            time_value=self.interpolation.lower_time_value
            if self.interpolation.lower_index == self.interpolation.upper_index
            else (
                f"{self.interpolation.lower_time_value}->{self.interpolation.upper_time_value}"
                f" ({self.interpolation.upper_weight:.3f})"
            ),
            filled_point_count=self.filled_point_count,
        )


@dataclass(frozen=True)
class HYCOMOpenBoundarySource:

    sampling_rule = SamplingRule(SampleClock.END, providers=("temp", "salt", "u", "v"))
    times: torch.Tensor
    time_values: tuple[str, ...]
    temp: OpenBoundaryField
    salt: OpenBoundaryField
    source_depth_m: float
    filled_point_count: int
    u: OpenBoundaryField | None = None
    v: OpenBoundaryField | None = None
    _sample_cache: dict[tuple[str, float], HYCOMOpenBoundarySample] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def sample(
        self,
        model_time_seconds: float,
        *,
        mode: str | BoundaryTimeMode = BoundaryTimeMode.LINEAR,
        update_interval_seconds: float | None = None,
    ) -> HYCOMOpenBoundarySample:
        actual_update_interval = _normalize_update_interval_seconds(update_interval_seconds)
        sample_time_seconds = _effective_sample_time_seconds(
            model_time_seconds,
            update_interval_seconds=actual_update_interval,
        )
        actual_mode = BoundaryTimeMode(mode)
        cache_key = (actual_mode.value, sample_time_seconds)
        cached = self._sample_cache.get(cache_key)
        if cached is not None:
            return replace(
                cached,
                interpolation=replace(
                    cached.interpolation,
                    model_time_seconds=float(model_time_seconds),
                    sample_time_seconds=sample_time_seconds,
                    update_interval_seconds=actual_update_interval,
                ),
            )

        temp, interpolation = self.temp.at(
            sample_time_seconds,
            mode=actual_mode,
            model_time_seconds=model_time_seconds,
            update_interval_seconds=actual_update_interval,
        )
        salt, _ = self.salt.at(
            sample_time_seconds,
            mode=actual_mode,
            model_time_seconds=model_time_seconds,
            update_interval_seconds=actual_update_interval,
        )
        u_sample = None
        v_sample = None
        if self.u is not None and self.v is not None:
            u_sample, _ = self.u.at(
                sample_time_seconds,
                mode=actual_mode,
                model_time_seconds=model_time_seconds,
                update_interval_seconds=actual_update_interval,
            )
            v_sample, _ = self.v.at(
                sample_time_seconds,
                mode=actual_mode,
                model_time_seconds=model_time_seconds,
                update_interval_seconds=actual_update_interval,
            )

        sample = HYCOMOpenBoundarySample(
            temp=temp,
            salt=salt,
            u=u_sample,
            v=v_sample,
            interpolation=interpolation,
            source_depth_m=self.source_depth_m,
            filled_point_count=self.filled_point_count,
        )
        self._sample_cache[cache_key] = sample
        while len(self._sample_cache) > 2:
            self._sample_cache.pop(next(iter(self._sample_cache)))
        return sample

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "HYCOMOpenBoundarySource":
        return HYCOMOpenBoundarySource(
            times=self.times.to(device=device, dtype=torch.float64),
            time_values=self.time_values,
            temp=self.temp.to_runtime(device, dtype),
            salt=self.salt.to_runtime(device, dtype),
            source_depth_m=self.source_depth_m,
            filled_point_count=self.filled_point_count,
            u=self.u.to_runtime(device, dtype) if self.u is not None else None,
            v=self.v.to_runtime(device, dtype) if self.v is not None else None,
        )

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "HYCOMOpenBoundarySource":
        return self.to_runtime(device, dtype)

    def as_dict(self) -> dict[str, object]:
        return {
            "times": self.times,
            "time_values": list(self.time_values),
            "temp": self.temp.as_dict(),
            "salt": self.salt.as_dict(),
            "u": self.u.as_dict() if self.u else {},
            "v": self.v.as_dict() if self.v else {},
            "source_depth_m": self.source_depth_m,
            "filled_point_count": self.filled_point_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HYCOMOpenBoundarySource":
        if "times" not in payload:
            raise ValueError("boundary payload missing required 'times' field")
        u_field = OpenBoundaryField.from_dict(payload["u"]) if "u" in payload and payload["u"] else None
        v_field = OpenBoundaryField.from_dict(payload["v"]) if "v" in payload and payload["v"] else None
        return cls(
            times=torch.as_tensor(payload["times"], dtype=torch.float64),
            time_values=tuple(str(value) for value in payload["time_values"]),
            temp=OpenBoundaryField.from_dict(payload["temp"]),
            salt=OpenBoundaryField.from_dict(payload["salt"]),
            u=u_field,
            v=v_field,
            source_depth_m=float(payload["source_depth_m"]),
            filled_point_count=int(payload["filled_point_count"]),
        )


def replicate_open_boundary_field_vertical(
    field: OpenBoundaryField,
    *,
    nz: int,
) -> OpenBoundaryField:
    def _replicate(values: torch.Tensor | None) -> torch.Tensor | None:
        if values is None:
            return None
        if values.ndim != 3:
            raise ValueError(f"expected time-varying side tensor with 3 dims, got {values.ndim}")
        return values.expand(-1, nz, -1).contiguous()

    return OpenBoundaryField(
        times=field.times,
        time_values=field.time_values,
        west=_replicate(field.west),
        east=_replicate(field.east),
        south=_replicate(field.south),
        north=_replicate(field.north),
    )


def replicate_open_boundary_source_vertical(
    source: HYCOMOpenBoundarySource,
    *,
    nz: int,
) -> HYCOMOpenBoundarySource:
    if nz < 1:
        raise ValueError("nz must be positive")
    return HYCOMOpenBoundarySource(
        times=source.times,
        time_values=source.time_values,
        temp=replicate_open_boundary_field_vertical(source.temp, nz=nz),
        salt=replicate_open_boundary_field_vertical(source.salt, nz=nz),
        source_depth_m=source.source_depth_m,
        filled_point_count=source.filled_point_count,
    )


def _normalize_update_interval_seconds(update_interval_seconds: float | None) -> float | None:
    if update_interval_seconds is None:
        return None
    actual_interval = float(update_interval_seconds)
    if actual_interval <= 0.0:
        return None
    return actual_interval


def _effective_sample_time_seconds(
    model_time_seconds: float,
    *,
    update_interval_seconds: float | None,
) -> float:
    actual_time = max(float(model_time_seconds), 0.0)
    if update_interval_seconds is None:
        return actual_time
    # Hold the boundary values fixed within each refresh window and update at window edges.
    scaled_time = math.nextafter(actual_time / update_interval_seconds, math.inf)
    return math.floor(scaled_time) * update_interval_seconds


def _fill_missing_1d(data: torch.Tensor) -> tuple[torch.Tensor, int]:
    filled = data.clone()
    nan_mask = torch.isnan(filled)
    count = int(nan_mask.sum().item())
    if count == 0:
        return filled, 0
    valid_mask = ~nan_mask
    if not valid_mask.any():
        return torch.zeros_like(filled), count
    
    indices = torch.arange(filled.numel(), device=filled.device)
    valid_indices = indices[valid_mask]
    valid_values = filled[valid_mask]
    nan_indices = indices[nan_mask]
    
    # 1D nearest neighbor: find closest valid index for each nan point
    dists = torch.abs(nan_indices.unsqueeze(1) - valid_indices.unsqueeze(0))
    nearest_idx = torch.argmin(dists, dim=1)
    
    filled[nan_mask] = valid_values[nearest_idx]
    return filled, count


def _interpolate_side(
    surface: xr.Dataset,
    *,
    side: str,
    target_lat: torch.Tensor,
    target_lon: torch.Tensor,
    method: str,
) -> xr.Dataset:
    if side == "west":
        interpolated = surface.interp(
            lat=xr.DataArray(target_lat.numpy(), dims=("lat",)),
            lon=float(target_lon[0].item()),
            method=method,
        )
        fill_dim = "lat"
    elif side == "east":
        interpolated = surface.interp(
            lat=xr.DataArray(target_lat.numpy(), dims=("lat",)),
            lon=float(target_lon[-1].item()),
            method=method,
        )
        fill_dim = "lat"
    elif side == "south":
        interpolated = surface.interp(
            lat=float(target_lat[0].item()),
            lon=xr.DataArray(target_lon.numpy(), dims=("lon",)),
            method=method,
        )
        fill_dim = "lon"
    else:
        interpolated = surface.interp(
            lat=float(target_lat[-1].item()),
            lon=xr.DataArray(target_lon.numpy(), dims=("lon",)),
            method=method,
        )
        fill_dim = "lon"
    return interpolated.ffill(dim=fill_dim).bfill(dim=fill_dim)


def _extract_side_tensor(side_dataset: xr.Dataset, name: str) -> tuple[torch.Tensor, int]:
    values = torch.as_tensor(side_dataset[name].values.copy(), dtype=torch.float64)
    if values.ndim == 1:
        filled, count = _fill_missing_1d(values)
        return filled.unsqueeze(0).unsqueeze(0), count

    filled_slices: list[torch.Tensor] = []
    filled_point_count = 0
    for time_index in range(values.shape[0]):
        filled, count = _fill_missing_1d(values[time_index])
        filled_slices.append(filled)
        filled_point_count = max(filled_point_count, count)
    return torch.stack(filled_slices, dim=0).unsqueeze(1), filled_point_count


def _relative_times_seconds(surface: xr.Dataset) -> tuple[torch.Tensor, tuple[str, ...]]:
    time_values = tuple(str(value) for value in surface.time.values)
    base_time = surface.time.values[0]
    relative = (surface.time.values - base_time) / np.timedelta64(1, "s")
    return torch.as_tensor(relative, dtype=torch.float64), time_values


def extract_tracer_open_boundary_source(
    dataset: xr.Dataset,
    request: HYCOMSurfacePatchRequest,
    *,
    depth_index: int = 0,
) -> HYCOMOpenBoundarySource:
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

    surface = dataset[["water_temp", "salinity", "water_u", "water_v"]].isel(depth=depth_index, time=slice(request.time_index, None)).compute()
    is_ocean = (surface["water_temp"] != 0.0) | (surface["salinity"] != 0.0)
    surface = surface.where(is_ocean).ffill("lat").bfill("lat").ffill("lon").bfill("lon")
    if surface.sizes.get("time", 0) < 1:
        raise ValueError("requested HYCOM boundary time index falls outside the dataset time range")

    temp_sides: dict[str, torch.Tensor] = {}
    salt_sides: dict[str, torch.Tensor] = {}
    u_sides: dict[str, torch.Tensor] = {}
    v_sides: dict[str, torch.Tensor] = {}
    filled_point_count = 0
    for side in ("west", "east", "south", "north"):
        side_dataset = _interpolate_side(
            surface,
            side=side,
            target_lat=target_lat,
            target_lon=target_lon,
            method=request.interp_method,
        )
        temp, temp_fill_count = _extract_side_tensor(side_dataset, "water_temp")
        salt, salt_fill_count = _extract_side_tensor(side_dataset, "salinity")
        u_tensor, u_fill_count = _extract_side_tensor(side_dataset, "water_u")
        v_tensor, v_fill_count = _extract_side_tensor(side_dataset, "water_v")
        temp_sides[side] = temp
        salt_sides[side] = salt
        u_sides[side] = u_tensor
        v_sides[side] = v_tensor
        filled_point_count = max(filled_point_count, temp_fill_count, salt_fill_count, u_fill_count, v_fill_count)

    times, time_values = _relative_times_seconds(surface)
    temp = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=temp_sides["west"],
        east=temp_sides["east"],
        south=temp_sides["south"],
        north=temp_sides["north"],
    )
    salt = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=salt_sides["west"],
        east=salt_sides["east"],
        south=salt_sides["south"],
        north=salt_sides["north"],
    )
    u_field = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=u_sides["west"],
        east=u_sides["east"],
        south=u_sides["south"],
        north=u_sides["north"],
    )
    v_field = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=v_sides["west"],
        east=v_sides["east"],
        south=v_sides["south"],
        north=v_sides["north"],
    )
    return HYCOMOpenBoundarySource(
        times=times,
        time_values=time_values,
        temp=temp,
        salt=salt,
        u=u_field,
        v=v_field,
        source_depth_m=float(dataset.depth.isel(depth=depth_index).item()),
        filled_point_count=filled_point_count,
    )


def extract_tracer_open_boundary(dataset: xr.Dataset, request: HYCOMSurfacePatchRequest) -> HYCOMTracerOpenBoundary:
    return extract_tracer_open_boundary_source(dataset, request).sample(
        0.0,
        mode=BoundaryTimeMode.CONSTANT,
    ).as_boundary()


def load_tracer_open_boundary(path: str | Path, request: HYCOMSurfacePatchRequest) -> HYCOMTracerOpenBoundary:
    dataset = xr.open_dataset(Path(path), engine="netcdf4")
    try:
        return extract_tracer_open_boundary(dataset, request)
    finally:
        dataset.close()


def load_tracer_open_boundary_source(path: str | Path, request: HYCOMSurfacePatchRequest) -> HYCOMOpenBoundarySource:
    dataset = xr.open_dataset(Path(path), engine="netcdf4")
    try:
        return extract_tracer_open_boundary_source(dataset, request)
    finally:
        dataset.close()


def extract_full_depth_tracer_open_boundary_source(
    dataset: xr.Dataset,
    request: HYCOMSurfacePatchRequest,
    *,
    depth_indices: list[int] | None = None,
) -> HYCOMOpenBoundarySource:
    """Extract tracer open boundary data at multiple depth layers.

    Unlike ``extract_tracer_open_boundary_source`` which reads a single depth,
    this function reads multiple depths and produces boundary tensors shaped
    [time, nz, side_length].

    Parameters
    ----------
    dataset : xr.Dataset — HYCOM dataset with (time, depth, lat, lon) dims
    request : HYCOMSurfacePatchRequest
    depth_indices : list[int] | None — depth indices to read; None = all.

    Returns
    -------
    HYCOMOpenBoundarySource with [time, nz, side_len] shaped boundary fields.
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

    pad = 0.5
    base_subset = dataset[["water_temp", "salinity", "water_u", "water_v"]].isel(
        depth=depth_indices,
        time=slice(request.time_index, None),
    ).sel(
        lat=slice(lat_min - 1e-5 if float(target_lat.min()) - pad < lat_min else float(target_lat.min()) - pad,
                  lat_max + 1e-5 if float(target_lat.max()) + pad > lat_max else float(target_lat.max()) + pad),
        lon=slice(lon_min - 1e-5 if float(target_lon.min()) - pad < lon_min else float(target_lon.min()) - pad,
                  lon_max + 1e-5 if float(target_lon.max()) + pad > lon_max else float(target_lon.max()) + pad),
    )

    if base_subset.sizes.get("time", 0) < 1:
        raise ValueError("requested HYCOM boundary time index falls outside the dataset time range")

    temp_sides_list: dict[str, list[torch.Tensor]] = {s: [] for s in ("west", "east", "south", "north")}
    salt_sides_list: dict[str, list[torch.Tensor]] = {s: [] for s in ("west", "east", "south", "north")}
    u_sides_list: dict[str, list[torch.Tensor]] = {s: [] for s in ("west", "east", "south", "north")}
    v_sides_list: dict[str, list[torch.Tensor]] = {s: [] for s in ("west", "east", "south", "north")}
    filled_point_count = 0

    num_times = base_subset.sizes["time"]
    for t_idx in range(num_times):
        if t_idx % max(1, num_times // 5) == 0 or t_idx == num_times - 1:
            print(f"  -> Tracing 3D open boundaries ({t_idx+1}/{num_times})...", flush=True)

        snapshot = base_subset.isel(time=t_idx).compute()
        is_ocean = (snapshot["water_temp"] != 0.0) | (snapshot["salinity"] != 0.0)
        snapshot = snapshot.where(is_ocean).ffill("lat").bfill("lat").ffill("lon").bfill("lon").ffill("depth").bfill("depth")

        for side in ("west", "east", "south", "north"):
            side_ds = _interpolate_side(
                snapshot,
                side=side,
                target_lat=target_lat,
                target_lon=target_lon,
                method=request.interp_method,
            )
            t_tensor, t_fill = _extract_side_tensor(side_ds, "water_temp")
            s_tensor, s_fill = _extract_side_tensor(side_ds, "salinity")
            u_tensor, u_fill = _extract_side_tensor(side_ds, "water_u")
            v_tensor, v_fill = _extract_side_tensor(side_ds, "water_v")

            # t_tensor shape: [depth, 1, side_len] -> Reorder to [1, depth, side_len]
            temp_sides_list[side].append(t_tensor.squeeze(1).unsqueeze(0))
            salt_sides_list[side].append(s_tensor.squeeze(1).unsqueeze(0))
            u_sides_list[side].append(u_tensor.squeeze(1).unsqueeze(0))
            v_sides_list[side].append(v_tensor.squeeze(1).unsqueeze(0))
            filled_point_count = max(filled_point_count, t_fill, s_fill, u_fill, v_fill)

    temp_sides: dict[str, torch.Tensor] = {}
    salt_sides: dict[str, torch.Tensor] = {}
    u_sides: dict[str, torch.Tensor] = {}
    v_sides: dict[str, torch.Tensor] = {}
    for side in ("west", "east", "south", "north"):
        temp_sides[side] = torch.cat(temp_sides_list[side], dim=0)
        salt_sides[side] = torch.cat(salt_sides_list[side], dim=0)
        u_sides[side] = torch.cat(u_sides_list[side], dim=0)
        v_sides[side] = torch.cat(v_sides_list[side], dim=0)

    times, time_values = _relative_times_seconds(base_subset)
    temp = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=temp_sides["west"],
        east=temp_sides["east"],
        south=temp_sides["south"],
        north=temp_sides["north"],
    )
    salt = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=salt_sides["west"],
        east=salt_sides["east"],
        south=salt_sides["south"],
        north=salt_sides["north"],
    )
    u_field = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=u_sides["west"],
        east=u_sides["east"],
        south=u_sides["south"],
        north=u_sides["north"],
    )
    v_field = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=v_sides["west"],
        east=v_sides["east"],
        south=v_sides["south"],
        north=v_sides["north"],
    )
    source_depth_m = float(dataset.depth.isel(depth=depth_indices[0]).item())
    return HYCOMOpenBoundarySource(
        times=times,
        time_values=time_values,
        temp=temp,
        salt=salt,
        u=u_field,
        v=v_field,
        source_depth_m=source_depth_m,
        filled_point_count=filled_point_count,
    )


def extract_interpolated_3d_tracer_open_boundary_source(
    dataset: xr.Dataset,
    request: HYCOMSurfacePatchRequest,
    *,
    target_depths: np.ndarray,
) -> HYCOMOpenBoundarySource:
    """Extract tracer open boundary data interpolated to specified depth levels.

    Parameters
    ----------
    dataset : xr.Dataset — HYCOM dataset with (time, depth, lat, lon) dims
    request : HYCOMSurfacePatchRequest
    target_depths : np.ndarray — vertical target coordinate values to interpolate to.

    Returns
    -------
    HYCOMOpenBoundarySource with [time, nz, side_len] shaped boundary fields.
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

    # --- Memory-efficient streaming: process one target depth at a time ---
    pad = 0.5
    base_subset = dataset[["water_temp", "salinity", "water_u", "water_v"]].isel(
        time=slice(request.time_index, None),
    ).sel(
        lat=slice(lat_min - 1e-5 if float(target_lat.min()) - pad < lat_min else float(target_lat.min()) - pad,
                  lat_max + 1e-5 if float(target_lat.max()) + pad > lat_max else float(target_lat.max()) + pad),
        lon=slice(lon_min - 1e-5 if float(target_lon.min()) - pad < lon_min else float(target_lon.min()) - pad,
                  lon_max + 1e-5 if float(target_lon.max()) + pad > lon_max else float(target_lon.max()) + pad),
    )
    if base_subset.sizes.get("time", 0) < 1:
        raise ValueError("requested HYCOM boundary time index falls outside the dataset time range")

    temp_sides_list: dict[str, list[torch.Tensor]] = {s: [] for s in ("west", "east", "south", "north")}
    salt_sides_list: dict[str, list[torch.Tensor]] = {s: [] for s in ("west", "east", "south", "north")}
    u_sides_list: dict[str, list[torch.Tensor]] = {s: [] for s in ("west", "east", "south", "north")}
    v_sides_list: dict[str, list[torch.Tensor]] = {s: [] for s in ("west", "east", "south", "north")}
    filled_point_count = 0

    num_times = base_subset.sizes["time"]
    target_depths_array = xr.DataArray(target_depths, dims=("depth",))

    for t_idx in range(num_times):
        if t_idx % max(1, num_times // 5) == 0 or t_idx == num_times - 1:
            print(f"  -> Tracing 3D open boundaries ({t_idx+1}/{num_times})...", flush=True)

        snapshot = base_subset.isel(time=t_idx).compute()
        is_ocean = (snapshot["water_temp"] != 0.0) | (snapshot["salinity"] != 0.0)
        snapshot = snapshot.where(is_ocean).ffill("lat").bfill("lat").ffill("lon").bfill("lon").ffill("depth").bfill("depth")
        vert_interp = snapshot.interp(
            depth=target_depths_array,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )

        for side in ("west", "east", "south", "north"):
            side_ds = _interpolate_side(
                vert_interp,
                side=side,
                target_lat=target_lat,
                target_lon=target_lon,
                method=request.interp_method,
            )
            t_tensor, t_fill = _extract_side_tensor(side_ds, "water_temp")
            s_tensor, s_fill = _extract_side_tensor(side_ds, "salinity")
            u_tensor, u_fill = _extract_side_tensor(side_ds, "water_u")
            v_tensor, v_fill = _extract_side_tensor(side_ds, "water_v")
            
            temp_sides_list[side].append(t_tensor.squeeze(1).unsqueeze(0))
            salt_sides_list[side].append(s_tensor.squeeze(1).unsqueeze(0))
            u_sides_list[side].append(u_tensor.squeeze(1).unsqueeze(0))
            v_sides_list[side].append(v_tensor.squeeze(1).unsqueeze(0))
            filled_point_count = max(filled_point_count, t_fill, s_fill, u_fill, v_fill)

    temp_sides_t: dict[str, torch.Tensor] = {s: torch.cat(temp_sides_list[s], dim=0) for s in temp_sides_list}
    salt_sides_t: dict[str, torch.Tensor] = {s: torch.cat(salt_sides_list[s], dim=0) for s in salt_sides_list}
    u_sides_t: dict[str, torch.Tensor] = {s: torch.cat(u_sides_list[s], dim=0) for s in u_sides_list}
    v_sides_t: dict[str, torch.Tensor] = {s: torch.cat(v_sides_list[s], dim=0) for s in v_sides_list}

    times, time_values = _relative_times_seconds(base_subset)
    temp = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=temp_sides_t["west"],
        east=temp_sides_t["east"],
        south=temp_sides_t["south"],
        north=temp_sides_t["north"],
    )
    salt = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=salt_sides_t["west"],
        east=salt_sides_t["east"],
        south=salt_sides_t["south"],
        north=salt_sides_t["north"],
    )
    u_field = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=u_sides_t["west"],
        east=u_sides_t["east"],
        south=u_sides_t["south"],
        north=u_sides_t["north"],
    )
    v_field = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=v_sides_t["west"],
        east=v_sides_t["east"],
        south=v_sides_t["south"],
        north=v_sides_t["north"],
    )
    return HYCOMOpenBoundarySource(
        times=times,
        time_values=time_values,
        temp=temp,
        salt=salt,
        u=u_field,
        v=v_field,
        source_depth_m=float(target_depths[0]),
        filled_point_count=filled_point_count,
    )


# ── Barotropic boundary source (Flather) ─────────────────────────────────────


@dataclass(frozen=True)
class BarotropicOpenBoundarySample:
    """Time-sampled barotropic boundary data for Flather condition."""
    eta: SideBoundaryTensorSet
    normal_velocity: SideBoundaryTensorSet
    tangential_velocity: SideBoundaryTensorSet | None
    interpolation: BoundaryInterpolationInfo


@dataclass(frozen=True)
class BarotropicOpenBoundarySource:
    """Time-varying barotropic boundary data source.

    Provides ``eta`` (SSH) and ``normal_velocity`` (depth-averaged)
    for each boundary side, with time interpolation via ``sample()``.
    """

    sampling_rule = SamplingRule(
        SampleClock.END, providers=("eta", "normal_velocity", "tangential_velocity"),
    )
    times: torch.Tensor
    time_values: tuple[str, ...]
    eta: OpenBoundaryField
    normal_velocity: OpenBoundaryField
    tangential_velocity: OpenBoundaryField | None = None

    def sample(
        self,
        model_time_seconds: float,
        *,
        mode: str | BoundaryTimeMode = BoundaryTimeMode.LINEAR,
        update_interval_seconds: float | None = None,
    ) -> BarotropicOpenBoundarySample:
        actual_update_interval = _normalize_update_interval_seconds(update_interval_seconds)
        sample_time_seconds = _effective_sample_time_seconds(
            model_time_seconds,
            update_interval_seconds=actual_update_interval,
        )
        eta, interpolation = self.eta.at(
            sample_time_seconds,
            mode=mode,
            model_time_seconds=model_time_seconds,
            update_interval_seconds=actual_update_interval,
        )
        nvel, _ = self.normal_velocity.at(
            sample_time_seconds,
            mode=mode,
            model_time_seconds=model_time_seconds,
            update_interval_seconds=actual_update_interval,
        )
        tvel = None
        if self.tangential_velocity is not None:
            tvel, _ = self.tangential_velocity.at(
                sample_time_seconds,
                mode=mode,
                model_time_seconds=model_time_seconds,
                update_interval_seconds=actual_update_interval,
            )
        return BarotropicOpenBoundarySample(
            eta=eta,
            normal_velocity=nvel,
            tangential_velocity=tvel,
            interpolation=interpolation,
        )

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "BarotropicOpenBoundarySource":
        return BarotropicOpenBoundarySource(
            times=self.times.to(device=device, dtype=torch.float64),
            time_values=self.time_values,
            eta=self.eta.to_runtime(device, dtype),
            normal_velocity=self.normal_velocity.to_runtime(device, dtype),
            tangential_velocity=self.tangential_velocity.to_runtime(device, dtype) if self.tangential_velocity else None,
        )

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "BarotropicOpenBoundarySource":
        return self.to_runtime(device, dtype)


def _compute_layer_thicknesses(depth_centers: np.ndarray) -> np.ndarray:
    """Compute layer thicknesses from non-uniform HYCOM depth centers.

    Uses midpoint interfaces: z_if[0]=0, z_if[k]=(z[k-1]+z[k])/2, z_if[-1]=2*z[-1]-z_if[-2].
    Layer thickness dz[k] = z_if[k+1] - z_if[k].
    """
    nz = len(depth_centers)
    if nz == 1:
        return np.array([depth_centers[0] * 2.0])
    interfaces = np.zeros(nz + 1)
    interfaces[0] = 0.0
    for k in range(1, nz):
        interfaces[k] = 0.5 * (depth_centers[k - 1] + depth_centers[k])
    interfaces[nz] = 2.0 * depth_centers[-1] - interfaces[nz - 1]
    return np.diff(interfaces)


def _depth_average_velocity(
    u_side: torch.Tensor,
    dz: torch.Tensor,
) -> torch.Tensor:
    """Thickness-weighted depth average.

    Parameters
    ----------
    u_side : [time, nz, side_len] — per-layer normal velocity
    dz : [nz] — layer thicknesses

    Returns
    -------
    [time, side_len] — depth-averaged velocity
    """
    # Mask NaN before averaging
    valid = torch.isfinite(u_side)
    u_clean = torch.where(valid, u_side, torch.zeros_like(u_side))
    dz_exp = dz.view(1, -1, 1).expand_as(u_side)
    dz_masked = torch.where(valid, dz_exp, torch.zeros_like(dz_exp))
    total_thickness = dz_masked.sum(dim=1).clamp_min(1e-8)
    return (u_clean * dz_masked).sum(dim=1) / total_thickness


def extract_barotropic_open_boundary_source(
    uvts_dataset: xr.Dataset,
    request: HYCOMSurfacePatchRequest,
    *,
    ssh_dataset: xr.Dataset | None = None,
    max_depth_m: float = 200.0,
) -> BarotropicOpenBoundarySource:
    """Extract time-varying barotropic boundary data for Flather condition.

    Parameters
    ----------
    uvts_dataset : HYCOM dataset with water_u, water_v (and depth dimension)
    request : patch request defining the target grid
    ssh_dataset : HYCOM SSH dataset with surf_el variable. If None, eta is zeros.
    max_depth_m : maximum depth for depth-averaged velocity computation

    Returns
    -------
    BarotropicOpenBoundarySource with eta and depth-averaged normal velocity
    """
    target_lat = _target_coordinate(request.center_lat, request.ny, request.resolution_deg)
    target_lon = _target_coordinate(request.center_lon, request.nx, request.resolution_deg)

    # ── Depth levels for velocity averaging ──
    depth_values = uvts_dataset.depth.values.astype(float)
    depth_mask = depth_values <= max_depth_m
    depth_indices = np.where(depth_mask)[0].tolist()
    if not depth_indices:
        depth_indices = [0]
    dz = torch.as_tensor(
        _compute_layer_thicknesses(depth_values[depth_indices]),
        dtype=torch.float64,
    )

    # ── Extract velocity on boundary sides ──
    uv_subset = uvts_dataset[["water_u", "water_v"]].isel(
        depth=depth_indices,
        time=slice(request.time_index, None),
    )
    if uv_subset.sizes.get("time", 0) < 1:
        raise ValueError("requested HYCOM boundary time index is outside dataset time range")

    nvel_sides: dict[str, torch.Tensor] = {}
    tvel_sides: dict[str, torch.Tensor] = {}
    filled_point_count = 0
    for side in ("west", "east", "south", "north"):
        # Determine normal component: u for west/east, v for south/north
        # Tangential is the opposite
        nvar_name = "water_u" if side in ("west", "east") else "water_v"
        tvar_name = "water_v" if side in ("west", "east") else "water_u"
        
        ndepth_slices: list[torch.Tensor] = []
        tdepth_slices: list[torch.Tensor] = []
        for di in range(len(depth_indices)):
            side_ds = _interpolate_side(
                uv_subset[[nvar_name, tvar_name]].isel(depth=di),
                side=side,
                target_lat=target_lat,
                target_lon=target_lon,
                method=request.interp_method,
            )
            ntensor, nfill_count = _extract_side_tensor(side_ds, nvar_name)
            ttensor, tfill_count = _extract_side_tensor(side_ds, tvar_name)
            ndepth_slices.append(ntensor)
            tdepth_slices.append(ttensor)
            filled_point_count = max(filled_point_count, nfill_count, tfill_count)
            
        # Stack: [time, nz, side_len]
        n_full = torch.cat(ndepth_slices, dim=1)
        t_full = torch.cat(tdepth_slices, dim=1)
        
        # Depth-average: [time, side_len]
        nvel_sides[side] = _depth_average_velocity(n_full, dz)
        tvel_sides[side] = _depth_average_velocity(t_full, dz)

    times, time_values = _relative_times_seconds(uv_subset)

    normal_velocity = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=nvel_sides["west"],
        east=nvel_sides["east"],
        south=nvel_sides["south"],
        north=nvel_sides["north"],
    )

    tangential_velocity = OpenBoundaryField(
        times=times,
        time_values=time_values,
        west=tvel_sides["west"],
        east=tvel_sides["east"],
        south=tvel_sides["south"],
        north=tvel_sides["north"],
    )

    # ── Extract eta (SSH) on boundary sides ──
    if ssh_dataset is not None:
        ssh_sub = ssh_dataset[["surf_el"]].isel(time=slice(request.time_index, None))
        if ssh_sub.sizes.get("time", 0) < 1:
            raise ValueError("SSH dataset time range is insufficient")
        # Align time axes: use min of both time counts
        n_times = min(uv_subset.sizes["time"], ssh_sub.sizes["time"])
        ssh_sub = ssh_sub.isel(time=slice(0, n_times))
        eta_sides: dict[str, torch.Tensor] = {}
        for side in ("west", "east", "south", "north"):
            side_ds = _interpolate_side(
                ssh_sub,
                side=side,
                target_lat=target_lat,
                target_lon=target_lon,
                method=request.interp_method,
            )
            tensor, fill_count = _extract_side_tensor(side_ds, "surf_el")
            # Shape: [time, 1, side_len] → squeeze depth to [time, side_len]
            eta_sides[side] = tensor.squeeze(1)
            filled_point_count = max(filled_point_count, fill_count)

        ssh_times, ssh_time_values = _relative_times_seconds(ssh_sub)
        eta_field = OpenBoundaryField(
            times=ssh_times,
            time_values=ssh_time_values,
            west=eta_sides["west"],
            east=eta_sides["east"],
            south=eta_sides["south"],
            north=eta_sides["north"],
        )
        # Use SSH times as primary (they may differ from UVTS times)
        times = ssh_times
        time_values = ssh_time_values
    else:
        # Zero eta fallback — still better than bare Flather because it's explicit
        n_times = uv_subset.sizes["time"]
        ny, nx = request.ny, request.nx
        zero_we = torch.zeros((n_times, ny), dtype=torch.float64)
        zero_sn = torch.zeros((n_times, nx), dtype=torch.float64)
        eta_field = OpenBoundaryField(
            times=times,
            time_values=time_values,
            west=zero_we.clone(),
            east=zero_we.clone(),
            south=zero_sn.clone(),
            north=zero_sn.clone(),
        )

    return BarotropicOpenBoundarySource(
        times=times,
        time_values=time_values,
        eta=eta_field,
        normal_velocity=normal_velocity,
        tangential_velocity=tangential_velocity,
    )
