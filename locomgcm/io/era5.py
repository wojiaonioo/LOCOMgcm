"""ERA5 field IO: load, clip, de-accumulate, regrid and time-interpolate.

The bulk physics (Large & Pond drag, ``wind_speed_to_stress``,
``specific_humidity_from_dewpoint``) lives in
``locomgcm.dynamics.bulk_formulae`` (the IO layer holds no formulae — MIT
``EXF_GETFFIELDS`` ↔ ``EXF_BULKFORMULAE`` separation).  The
loaders below still *call* those functions at load time (lazy imports), so the
stored ``ERA5WindSource.tau_*`` / ``ERA5BulkSource.aqh`` contracts are
unchanged; converting stress per-step instead would change results (stress is
nonlinear in wind, lerp∘stress ≠ stress∘lerp).
"""
from __future__ import annotations

from locomgcm.sampling import SampleClock, SamplingRule

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from locomgcm.accel.tensor_context import ordinary_tensor_mode
from locomgcm.io.hycom import (
    EARTH_RADIUS_M,
    _meters_per_degree_lat,
    _meters_per_degree_lon,
    _target_coordinate,
)


class _ModelTimeSlot:
    """Current-stream scalar upload with a reusable pinned staging buffer."""

    @ordinary_tensor_mode()
    def __init__(self, times: torch.Tensor) -> None:
        self.host = torch.empty(
            (), dtype=times.dtype, device="cpu", pin_memory=times.is_cuda,
        )
        self.device = torch.empty((), dtype=times.dtype, device=times.device)
        self.copy_done = torch.cuda.Event() if times.is_cuda else None

    def sample(self, value: float | torch.Tensor) -> torch.Tensor:
        # Tensor clocks keep their autograd/forward-mode dependency.  A traced
        # host clock is a fresh tensor too: the staged upload below writes two
        # persistent buffers in place (CPU fill_, then copy_ to the device), and
        # a Dynamo capture with an AOT backend drops those writes on torch 2.9,
        # so a traced sampler read the slot's previous clock without an error.
        # The slot is an eager mechanism (no host sync per upload); the same
        # split as ``locomgcm.seaice.dynamics.types.scalar``.
        if isinstance(value, torch.Tensor) or torch.compiler.is_compiling():
            return torch.as_tensor(value, dtype=self.device.dtype, device=self.device.device)
        if self.copy_done is not None:
            # Wait only for the preceding scalar upload before reusing its
            # pinned source, never for the model work queued after that upload.
            self.copy_done.synchronize()
        self.host.fill_(value)
        self.device.copy_(self.host, non_blocking=True)
        if self.copy_done is not None:
            self.copy_done.record(torch.cuda.current_stream(self.device.device))
        # Clamp can retain its input when record times are differentiable.
        # That saved scalar must survive the next upload into the slot.
        return self.device.clone()


@dataclass(frozen=True)
class ERA5WindSource:
    """Time-varying wind stress fields on the model grid.

    Attributes:
        times: 1-D tensor of relative times in seconds from the first record.
        time_values: human-readable time labels.
        tau_x: [n_times, ny, nx] x-component of wind stress (Pa) on cell center.
        tau_y: [n_times, ny, nx] y-component of wind stress (Pa) on cell center.
        period_seconds: if positive, the forcing cycles with this period.
    """

    sampling_rule = SamplingRule(SampleClock.START)

    times: torch.Tensor
    time_values: tuple[str, ...]
    tau_x: torch.Tensor
    tau_y: torch.Tensor
    period_seconds: float | None = None
    _time_slot: _ModelTimeSlot = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_time_slot", _ModelTimeSlot(self.times))

    def sample(self, model_time_seconds: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (tau_x_2d, tau_y_2d) interpolated to *model_time_seconds*."""
        t = self._time_slot.sample(model_time_seconds)
        if self.period_seconds is not None and self.period_seconds > 0.0:
            t = t % self.period_seconds

        t = torch.clamp(t, self.times[0], self.times[-1])
        target = t.unsqueeze(0)
        upper_idx = torch.searchsorted(self.times, target, right=False).squeeze()
        upper_idx = torch.clamp(upper_idx, 0, self.times.shape[0] - 1)
        lower_idx = torch.clamp(upper_idx - 1, 0, self.times.shape[0] - 1)

        # 18 host reads + 21 waits + 11.4 device->host copies per step came from
        # here, the whole non-PCG wait budget outside the dt scalars.  ``tensor[idx]`` with a 0-d INTEGER index is basic
        # indexing, and ATen implements it as ``applySelect(..., idx.item())``
        # (``ATen/TensorIndexing.h``) -- one blocking device->host read per
        # gather, six per call.  ``index_select`` takes the index on the device
        # and gathers there; the gathered values, their dtype and every
        # arithmetic operation below are unchanged.
        idx_low = lower_idx.reshape(1)
        idx_high = upper_idx.reshape(1)
        t_low = self.times.index_select(0, idx_low).reshape(())
        t_high = self.times.index_select(0, idx_high).reshape(())

        diff = t_high - t_low
        # Native safe division
        diff = torch.where(diff == 0.0, torch.ones_like(diff), diff)
        w = (t - t_low) / diff
        w = torch.where(t_high == t_low, torch.zeros_like(w), w)

        tau_x_low = self.tau_x.index_select(0, idx_low).squeeze(0)
        tau_x_high = self.tau_x.index_select(0, idx_high).squeeze(0)
        tau_y_low = self.tau_y.index_select(0, idx_low).squeeze(0)
        tau_y_high = self.tau_y.index_select(0, idx_high).squeeze(0)

        tau_x = (1.0 - w) * tau_x_low + w * tau_x_high
        tau_y = (1.0 - w) * tau_y_low + w * tau_y_high
        return tau_x, tau_y

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "ERA5WindSource":
        tensor_dtype = self.tau_x.dtype if dtype is None else dtype
        return ERA5WindSource(
            times=self.times.to(device=device, dtype=torch.float64),
            time_values=self.time_values,
            tau_x=self.tau_x.to(device=device, dtype=tensor_dtype),
            tau_y=self.tau_y.to(device=device, dtype=tensor_dtype),
            period_seconds=self.period_seconds,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "times": self.times,
            "time_values": list(self.time_values),
            "tau_x": self.tau_x,
            "tau_y": self.tau_y,
            "period_seconds": self.period_seconds,
        }


# ---------------------------------------------------------------------------
# Load and interpolate ERA5 to model grid
# ---------------------------------------------------------------------------


def load_era5_wind_source(
    path: str,
    *,
    center_lat: float = 15.0,
    center_lon: float = 115.0,
    nx: int = 50,
    ny: int = 50,
    resolution_deg: float = 1.0 / 12.0,
    cyclic: bool = True,
) -> ERA5WindSource:
    """Load ERA5 u10/v10 NetCDF and produce an :class:`ERA5WindSource`.

    Parameters
    ----------
    path : str
        Path to the ERA5 NetCDF file (with u10, v10, latitude/longitude, time).
    center_lat, center_lon, nx, ny, resolution_deg :
        Model grid specification (same as HYCOMSurfacePatchRequest).
    cyclic : bool
        If True, set ``period_seconds`` to the data span so the forcing cycles.
    """
    ds = xr.open_dataset(path)
    try:
        return _extract_wind_source(
            ds,
            center_lat=center_lat,
            center_lon=center_lon,
            nx=nx,
            ny=ny,
            resolution_deg=resolution_deg,
            cyclic=cyclic,
        )
    finally:
        ds.close()


def era5_wind_paths(
    era5_dir: str | Path, *, start_date: datetime, window_seconds: int
) -> tuple[list[Path], list[Path]]:
    """The yearly u10 and v10 files ``load_era5_wind_source_by_time`` opens for
    ``[start_date, start_date + window_seconds]``; a missing file raises."""
    era5_dir = Path(era5_dir)
    end_date = start_date + timedelta(seconds=int(window_seconds))

    # Only iterate years that actually contain the requested window. ERA5
    # files are full-year, so each year touched is loaded in full and then
    # time-sliced. No buffer year is added — the simulation should not
    # request times beyond [start_date, end_date].
    years = list(range(start_date.year, end_date.year + 1))

    u10_paths: list[Path] = []
    v10_paths: list[Path] = []
    for y in years:
        u_path = era5_dir / str(y) / f"era5_u10_{y}.nc"
        v_path = era5_dir / str(y) / f"era5_v10_{y}.nc"
        if not u_path.exists() or not v_path.exists():
            raise FileNotFoundError(
                f"ERA5 yearly file missing for {y}: expected {u_path} and {v_path}"
            )
        u10_paths.append(u_path)
        v10_paths.append(v_path)
    return u10_paths, v10_paths


def load_era5_wind_source_by_time(
    era5_dir: str | Path,
    *,
    start_date: datetime,
    window_seconds: int,
    center_lat: float,
    center_lon: float,
    nx: int,
    ny: int,
    resolution_deg: float,
    cyclic: bool = False,
) -> ERA5WindSource:
    """Load ERA5 u10/v10 covering ``[start_date, start_date + window_seconds]`` from a
    yearly-organized archive (``era5_dir/<year>/era5_{u10,v10}_<year>.nc``).

    Files are auto-discovered for every year touched by the requested window
    and concatenated along time. The result is spatially subset to a small box
    around the model domain (with a 1° pad) before interpolation, so we never
    pull a global field through xarray's interp engine.

    Parameters
    ----------
    era5_dir : path
        Root directory containing per-year subdirs.
    start_date : datetime
        Simulation start time (UTC).
    window_seconds : int
        Forcing window length in INTEGER seconds. End is
        ``start_date + timedelta(seconds=window_seconds)``. Must span enough of
        the archive that >=2 forcing records bracket the whole run, so that
        ``ERA5WindSource.sample`` linearly interpolates per step (matching MIT
        EXF) instead of clamping to a single constant record. Integer seconds
        avoid the float-days boundary trap that could drop/add a record exactly
        at a forcing cadence boundary.
    cyclic : bool
        If True, set ``period_seconds`` to the data span. Defaults to False
        because a properly time-windowed source should not cycle within the
        simulation.
    """
    end_date = start_date + timedelta(seconds=int(window_seconds))
    u10_paths, v10_paths = era5_wind_paths(
        era5_dir, start_date=start_date, window_seconds=window_seconds
    )

    # Spatial pre-filter: 1° pad around the model SCS box
    target_lat = _target_coordinate(center_lat, ny, resolution_deg)
    target_lon = _target_coordinate(center_lon, nx, resolution_deg)
    lat_pad = 1.0
    lon_pad = 1.0
    lat_lo = float(target_lat.min().item()) - lat_pad
    lat_hi = float(target_lat.max().item()) + lat_pad
    lon_lo = float(target_lon.min().item()) - lon_pad
    lon_hi = float(target_lon.max().item()) + lon_pad

    def _open_and_clip(paths: list[Path], var: str) -> xr.DataArray:
        pieces: list[xr.DataArray] = []
        for p in paths:
            ds = xr.open_dataset(p)
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            time_name = "valid_time" if "valid_time" in ds.coords else "time"
            da = ds[var]
            # ERA5 lat is 90→-90 (descending). Use sortby to ascending then sel.
            da = da.sortby(lat_name).sel(
                {
                    lat_name: slice(lat_lo, lat_hi),
                    lon_name: slice(lon_lo, lon_hi),
                }
            )
            # Time window slice
            da = da.sel(
                {time_name: slice(np.datetime64(start_date), np.datetime64(end_date))}
            )
            da = da.load()
            ds.close()
            pieces.append(da)
        if len(pieces) == 1:
            return pieces[0]
        time_dim = "valid_time" if "valid_time" in pieces[0].dims else "time"
        return xr.concat(pieces, dim=time_dim)

    u10 = _open_and_clip(u10_paths, "u10")
    v10 = _open_and_clip(v10_paths, "v10")

    merged = xr.Dataset({"u10": u10, "v10": v10})
    return _extract_wind_source(
        merged,
        center_lat=center_lat,
        center_lon=center_lon,
        nx=nx,
        ny=ny,
        resolution_deg=resolution_deg,
        cyclic=cyclic,
    )


def _extract_wind_source(
    ds: xr.Dataset,
    *,
    center_lat: float,
    center_lon: float,
    nx: int,
    ny: int,
    resolution_deg: float,
    cyclic: bool,
) -> ERA5WindSource:
    target_lat = _target_coordinate(center_lat, ny, resolution_deg)
    target_lon = _target_coordinate(center_lon, nx, resolution_deg)

    # ERA5 uses 'latitude'/'longitude' (may be descending); handle both
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    time_name = "valid_time" if "valid_time" in ds.dims else "time"

    # Variable names: ERA5 reanalysis uses u10/v10
    u10_name = "u10" if "u10" in ds.data_vars else "10m_u_component_of_wind"
    v10_name = "v10" if "v10" in ds.data_vars else "10m_v_component_of_wind"

    interp_coords = {
        lat_name: xr.DataArray(target_lat.numpy(), dims=("y",)),
        lon_name: xr.DataArray(target_lon.numpy(), dims=("x",)),
    }
    interpolated = ds[[u10_name, v10_name]].interp(interp_coords, method="linear")

    # Build relative times
    time_vals = interpolated[time_name].values
    time_labels = tuple(str(t) for t in time_vals)
    base_time = time_vals[0]
    relative_seconds = (time_vals - base_time) / np.timedelta64(1, "s")
    times = torch.as_tensor(relative_seconds, dtype=torch.float64)

    # Convert to stress
    u10_tensor = torch.as_tensor(interpolated[u10_name].values, dtype=torch.float64)
    v10_tensor = torch.as_tensor(interpolated[v10_name].values, dtype=torch.float64)

    # Handle NaN by filling with 0
    u10_tensor = torch.nan_to_num(u10_tensor, nan=0.0)
    v10_tensor = torch.nan_to_num(v10_tensor, nan=0.0)

    # Lazy: io -> dynamics only inside loader bodies (layering rule).
    from locomgcm.dynamics.bulk_formulae import wind_speed_to_stress

    tau_x_all, tau_y_all = wind_speed_to_stress(u10_tensor, v10_tensor)

    period_seconds = float(times[-1].item()) if cyclic else None

    return ERA5WindSource(
        times=times,
        time_values=time_labels,
        tau_x=tau_x_all,
        tau_y=tau_y_all,
        period_seconds=period_seconds,
    )


# ---------------------------------------------------------------------------
# ERA5 bulk-formula inputs : read + de-accumulate the atmospheric
# state that feeds exf_bulk_fluxes -> surface_forcing_from_bulk.
# Native (lat, lon) grid; the cube-face regrid is a separate downstream step.
# ---------------------------------------------------------------------------

# ERA5 flux and precipitation accumulations cover one hour. Divide J/m^2
# and metres by this interval to obtain W/m^2 and m/s, respectively.
_ERA5_ACCUM_SECONDS = 3600.0

# field -> (ERA5 file stem, native nc var name, is-accumulated)
_ERA5_BULK_SPEC = {
    "atemp": ("atemp", "t2m", False),
    "dewp": ("d2m", "d2m", False),
    "surfp": ("sp", "sp", False),
    "uwind": ("u10", "u10", False),
    "vwind": ("v10", "v10", False),
    "swdown": ("swdown", "ssrd", True),
    "lwdown": ("lwdown", "strd", True),
    "precip": ("tp", "tp", True),
}


@dataclass(frozen=True)
class ERA5BulkInputs:
    """Bulk-formula atmospheric inputs at one model time, native (lat, lon) grid.

    All MITgcm/EXF convention, ready for ``exf_bulk_fluxes``: ``atemp`` 2-m air
    temp [K], ``aqh`` 2-m specific humidity [kg/kg], ``uwind``/``vwind`` 10-m wind
    [m/s], ``swdown``/``lwdown`` downward short/long-wave [W/m²] (de-accumulated),
    ``precip`` [m/s] (de-accumulated), ``runoff`` [m/s] positive into the ocean
    (zero from the ERA5 sources), ``lwflux``/``swflux`` net long/shortwave
    [W/m², upward] that EXF keeps where it does not form them from the
    downward fields (zero from the ERA5 sources, which always form them).
    Each field is ``[nlat, nlon]``.
    """

    atemp: torch.Tensor
    aqh: torch.Tensor
    uwind: torch.Tensor
    vwind: torch.Tensor
    swdown: torch.Tensor
    lwdown: torch.Tensor
    precip: torch.Tensor
    runoff: torch.Tensor
    lwflux: torch.Tensor
    swflux: torch.Tensor


@dataclass(frozen=True)
class ERA5BulkSource:
    """Time-varying ERA5 bulk-formula inputs on the native (lat, lon) grid.

    De-accumulation (/3600) and the dewpoint→specific-humidity conversion are
    applied at load time; :meth:`sample` only linearly interpolates in time
    (matching MIT EXF — guards the constant-hold artifact, memory
    ``short-run-wind-constant-hold-artifact``).  The downstream step regrids the
    sampled fields to the cube faces and runs the bulk + producer.
    """

    sampling_rule = SamplingRule(SampleClock.START)
    # Records stay on the device and are interpolated with a tensor clock
    # inside the consumer's compiled region (``prepare_sample_time``).
    samples_on_host = False
    # ERA5 has downward radiation only; the consumer forms lwflux/swflux from it.
    longwave_from_down = True
    shortwave_from_down = True

    times: torch.Tensor               # [n_times] relative seconds from record 0
    time_values: tuple[str, ...]
    lat: torch.Tensor                 # [nlat]
    lon: torch.Tensor                 # [nlon]
    atemp: torch.Tensor               # [n_times, nlat, nlon] ...
    aqh: torch.Tensor
    uwind: torch.Tensor
    vwind: torch.Tensor
    swdown: torch.Tensor
    lwdown: torch.Tensor
    precip: torch.Tensor
    period_seconds: float | None = None
    _time_slot: _ModelTimeSlot = field(init=False, repr=False, compare=False)

    _FIELDS = ("atemp", "aqh", "uwind", "vwind", "swdown", "lwdown", "precip")

    def __post_init__(self) -> None:
        object.__setattr__(self, "_time_slot", _ModelTimeSlot(self.times))

    def sample(self, model_time_seconds: float | torch.Tensor) -> ERA5BulkInputs:
        """Linearly interpolate every bulk field to *model_time_seconds*."""
        t = self.prepare_sample_time(model_time_seconds)
        if self.period_seconds is not None and self.period_seconds > 0.0:
            t = t % self.period_seconds
        t = torch.clamp(t, self.times[0], self.times[-1])
        upper = torch.clamp(
            torch.searchsorted(self.times, t.unsqueeze(0), right=False).squeeze(),
            0, self.times.shape[0] - 1,
        )
        lower = torch.clamp(upper - 1, 0, self.times.shape[0] - 1)
        # Same 0-d-integer-index host read as ``ERA5WindSource.sample`` above
        #: 2 + 2*len(_FIELDS) blocking reads per call,
        # here at ``ERA5BulkForcing`` evaluate time.  ``index_select`` gathers
        # on the device; values and arithmetic are unchanged.
        idx_low = lower.reshape(1)
        idx_high = upper.reshape(1)
        t_low = self.times.index_select(0, idx_low).reshape(())
        t_high = self.times.index_select(0, idx_high).reshape(())
        diff = torch.where(t_high == t_low, torch.ones_like(t_high), t_high - t_low)
        w = torch.where(t_high == t_low, torch.zeros_like(diff), (t - t_low) / diff)
        out = {}
        for name in self._FIELDS:
            fld = getattr(self, name)
            out[name] = (
                (1.0 - w) * fld.index_select(0, idx_low).squeeze(0)
                + w * fld.index_select(0, idx_high).squeeze(0)
            )
        zero = torch.zeros_like(out["precip"])
        return ERA5BulkInputs(**out, runoff=zero, lwflux=zero, swflux=zero)

    def prepare_sample_time(self, model_time_seconds: float | torch.Tensor) -> torch.Tensor:
        """Stage a host clock once; tensor clocks retain their dependencies.

        A caller can pass the returned scalar to ``sample`` inside its compiled
        step. All atmospheric records stay on the runtime device.
        """
        return self._time_slot.sample(model_time_seconds)

    def to_runtime(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "ERA5BulkSource":
        td = dtype or self.atemp.dtype
        moved = {n: getattr(self, n).to(device=device, dtype=td) for n in self._FIELDS}
        return ERA5BulkSource(
            times=self.times.to(device=device, dtype=torch.float64),
            time_values=self.time_values,
            lat=self.lat.to(device=device, dtype=torch.float64),
            lon=self.lon.to(device=device, dtype=torch.float64),
            period_seconds=self.period_seconds,
            **moved,
        )


def _regrid_bulk_fields(
    raw: dict[str, "xr.DataArray"],
    target_lat: "np.ndarray",
    target_lon: "np.ndarray",
) -> dict[str, "xr.DataArray"]:
    """Bilinearly interpolate native-grid ERA5 fields onto the model grid.

    Same xarray bilinear as the prep ``grid_interp`` op (``layer_c/operators/simple.py``).
    A single wrapped longitude column (``lon[0]+360``) is appended when the source
    spans ~globally so dateline-straddling target columns interpolate periodically
    instead of going NaN; harmless for a regional sub-box (target lon interior to
    the source range).
    """
    tlat = np.asarray(target_lat, dtype="float64")
    tlon = np.asarray(target_lon, dtype="float64")
    out: dict[str, xr.DataArray] = {}
    for field, da in raw.items():
        lat_name = "latitude" if "latitude" in da.coords else "lat"
        lon_name = "longitude" if "longitude" in da.coords else "lon"
        lon_vals = da[lon_name]
        if float(lon_vals.max() - lon_vals.min()) > 180.0:          # ~global → pad wrap
            wrap = da.isel({lon_name: 0}).assign_coords(
                {lon_name: float(lon_vals[0]) + 360.0}
            )
            da = xr.concat([da, wrap], dim=lon_name)
        out[field] = da.interp(
            {lat_name: xr.DataArray(tlat, dims=(lat_name,)),
             lon_name: xr.DataArray(tlon, dims=(lon_name,))},
            method="linear",
        )
    return out


def load_era5_bulk_source_by_time(
    era5_dir: str | Path,
    *,
    start_date: datetime,
    window_seconds: int,
    lat_box: tuple[float, float] | None = None,
    lon_box: tuple[float, float] | None = None,
    target_lat: "np.ndarray | None" = None,
    target_lon: "np.ndarray | None" = None,
    cyclic: bool = False,
) -> ERA5BulkSource:
    """Load the 8 ERA5 bulk-input vars over ``[start, start+window]`` + de-accumulate.

    Reads the yearly archive ``era5_dir/<year>/era5_<stem>_<year>.nc`` for each of
    t2m/d2m/sp/u10/v10/ssrd/strd/tp, de-accumulates ssrd/strd/tp by /3600, and
    derives specific humidity ``aqh`` from d2m + sp.  ``lat_box``/``lon_box`` are
    optional spatial clips (``None`` = global, for the cube); the cube-face regrid
    is downstream.  The window must bracket ≥2 records so :meth:`ERA5BulkSource.sample`
    interpolates (not constant-hold).
    """
    era5_dir = Path(era5_dir)
    end_date = start_date + timedelta(seconds=int(window_seconds))
    years = list(range(start_date.year, end_date.year + 1))

    def _open_and_clip(stem: str, var: str) -> xr.DataArray:
        pieces: list[xr.DataArray] = []
        for y in years:
            p = era5_dir / str(y) / f"era5_{stem}_{y}.nc"
            if not p.exists():
                raise FileNotFoundError(f"ERA5 yearly file missing: {p}")
            ds = xr.open_dataset(p)
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            time_name = "valid_time" if "valid_time" in ds.coords else "time"
            da = ds[var].sortby(lat_name)
            sel: dict[str, slice] = {}
            if lat_box is not None:
                sel[lat_name] = slice(lat_box[0], lat_box[1])
            if lon_box is not None:
                sel[lon_name] = slice(lon_box[0], lon_box[1])
            if sel:
                da = da.sel(sel)
            da = da.sel({time_name: slice(np.datetime64(start_date), np.datetime64(end_date))})
            da = da.load()
            ds.close()
            pieces.append(da)
        if len(pieces) == 1:
            return pieces[0]
        time_dim = "valid_time" if "valid_time" in pieces[0].dims else "time"
        return xr.concat(pieces, dim=time_dim)

    raw = {field: _open_and_clip(stem, var) for field, (stem, var, _) in _ERA5_BULK_SPEC.items()}

    # Regrid the native ERA5 lat-lon onto the model grid at load time (the runtime
    # ERA5BulkForcing then only time-interpolates) — mirrors how the prep bakes the
    # model-grid wind stress, so the bulk path needs no per-step regrid.
    if target_lat is not None and target_lon is not None:
        raw = _regrid_bulk_fields(raw, target_lat, target_lon)

    ref = raw["atemp"]
    lat_name = "latitude" if "latitude" in ref.coords else "lat"
    lon_name = "longitude" if "longitude" in ref.coords else "lon"
    time_name = "valid_time" if "valid_time" in ref.coords else "time"
    lat = torch.as_tensor(ref[lat_name].values, dtype=torch.float64)
    lon = torch.as_tensor(ref[lon_name].values, dtype=torch.float64)
    time_coord = ref[time_name].values
    times = torch.as_tensor(
        (time_coord - time_coord[0]) / np.timedelta64(1, "s"), dtype=torch.float64
    )
    time_labels = tuple(str(np.datetime_as_string(t, unit="h")) for t in time_coord)

    def _tensor(field: str) -> torch.Tensor:
        a = torch.as_tensor(raw[field].values, dtype=torch.float64)
        if _ERA5_BULK_SPEC[field][2]:           # de-accumulate accumulated fields
            a = a / _ERA5_ACCUM_SECONDS
        return a

    # Lazy: io -> dynamics only inside loader bodies (layering rule).
    from locomgcm.dynamics.bulk_formulae import specific_humidity_from_dewpoint

    atemp = _tensor("atemp")
    aqh = specific_humidity_from_dewpoint(_tensor("dewp"), _tensor("surfp"))
    period_seconds = float(times[-1].item()) if cyclic else None
    return ERA5BulkSource(
        times=times, time_values=time_labels, lat=lat, lon=lon,
        atemp=atemp, aqh=aqh,
        uwind=_tensor("uwind"), vwind=_tensor("vwind"),
        swdown=_tensor("swdown"), lwdown=_tensor("lwdown"), precip=_tensor("precip"),
        period_seconds=period_seconds,
    )
