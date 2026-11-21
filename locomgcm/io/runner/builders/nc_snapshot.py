"""Universal NetCDF snapshot writer for yaml-driven runner cases."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

MAX_QUEUE = 2

_CURRENT_FRAME_TIME_SECONDS: float | None = None
_CURRENT_FRAME_ID: int | None = None


@dataclass(frozen=True)
class NCVarSpec:
    """One variable to write into per-day NetCDF snapshots."""

    name: str
    dims: tuple[str, ...]
    extract: Callable[[Any], np.ndarray]
    dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dims or self.dims[0] != "time":
            raise ValueError(f"NCVarSpec({self.name!r}).dims must start with 'time'; got {self.dims}")


def current_frame_time_seconds() -> float:
    """Return the time (seconds) for the frame currently being extracted.

    Part of the ``NCVarSpec.extract`` contract: extractors that need the
    frame timestamp (e.g. OBCS overlays that resample boundary data) call
    this from inside the writer's extract phase. Outside of an active
    extract phase the accessor raises — callers must not cache the value.
    """
    if _CURRENT_FRAME_TIME_SECONDS is None:
        raise RuntimeError("NC snapshot frame time is only available during extraction")
    return float(_CURRENT_FRAME_TIME_SECONDS)


def _coord_depth(grid: Any) -> np.ndarray:
    return grid.center_depth.detach().cpu().numpy().astype(np.float64)


# The grid-derived defaults below are the row index (``lat``) and the
# cumulative cell width in metres along the first row (``lon``). The stage 2
# runtime passes degree axes from ``latlon_centre_axes`` for spherical-polar
# lat-lon grids, so these defaults remain for Cartesian grids.
def _coord_lat(grid: Any) -> np.ndarray:
    return np.arange(grid.ny, dtype=np.float64)


def _coord_lon(grid: Any) -> np.ndarray:
    dx_row = grid.metrics.dx_center.detach().cpu().numpy().astype(np.float64)
    if dx_row.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.cumsum(dx_row[0])


def latlon_centre_axes(
    center_lat: float,
    center_lon: float,
    resolution_deg: float,
    *,
    ny: int,
    nx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre latitudes and longitudes, in degrees, of a regular lat-lon grid.

    ``center + (i - (n - 1) / 2) * resolution_deg`` in float64, the
    convention of the stage 1a regridding target grid (``grid.center_lat``,
    ``grid.center_lon``, ``grid.resolution_deg``). The stage 2 nc snapshots,
    the comparison adapter (``HycomAdapter``) and the model-only stage 4
    (``helper.preprocess.model_only``) all take their degree axes from here.
    """
    res = float(resolution_deg)
    lat = float(center_lat) + (np.arange(int(ny)) - (int(ny) - 1) / 2.0) * res
    lon = float(center_lon) + (np.arange(int(nx)) - (int(nx) - 1) / 2.0) * res
    return lat.astype(np.float64), lon.astype(np.float64)


def _state_to_nc_fields(
    state: Any,
    grid: Any,
    *,
    diagnose_w: bool,
) -> dict[str, np.ndarray]:
    fields = {
        "eta": state.eta.data[0].detach().cpu().numpy(),
        "temp": state.temp.data[0].detach().cpu().numpy(),
        "salt": state.salt.data[0].detach().cpu().numpy(),
        "u": state.u.data[0, :, :, : grid.nx].detach().cpu().numpy(),
        "v": state.v.data[0, :, : grid.ny, :].detach().cpu().numpy(),
    }
    if diagnose_w:
        from locomgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

        _, w_iface, _ = diagnose_vertical_velocity(state, grid)
        # expects MIT upward-positive in the NC snapshots; diagnostic returns
        # LOCOMgcm downward-positive.
        fields["w"] = (-w_iface[0, : grid.nz]).detach().cpu().numpy()
    return fields


_Frame = dict[str, np.ndarray | float | int]


def default_state_variables(grid: Any, *, diagnose_w: bool) -> list[NCVarSpec]:
    """Return the canonical eta/temp/salt/u/v(/w) registry for regional runs."""

    if not diagnose_w:
        # Plain state extraction only needs the staggered-face crop lengths.
        grid = SimpleNamespace(nx=grid.nx, ny=grid.ny)

    names_and_dims = [
        ("eta", ("time", "lat", "lon")),
        ("temp", ("time", "depth", "lat", "lon")),
        ("salt", ("time", "depth", "lat", "lon")),
        ("u", ("time", "depth", "lat", "lon")),
        ("v", ("time", "depth", "lat", "lon")),
    ]
    if diagnose_w:
        names_and_dims.append(("w", ("time", "depth", "lat", "lon")))

    cache_frame_id: int | None = None
    cache_fields: dict[str, np.ndarray] | None = None
    cache_remaining = 0

    def fields_for(state_obj: Any) -> dict[str, np.ndarray]:
        nonlocal cache_frame_id, cache_fields, cache_remaining
        if cache_fields is None or cache_frame_id != _CURRENT_FRAME_ID:
            cache_frame_id = _CURRENT_FRAME_ID
            cache_fields = _state_to_nc_fields(state_obj, grid, diagnose_w=diagnose_w)
            cache_remaining = len(names_and_dims)
        return cache_fields

    def make_extract(name: str) -> Callable[[Any], np.ndarray]:
        def extract(state_obj: Any) -> np.ndarray:
            nonlocal cache_frame_id, cache_fields, cache_remaining
            fields = fields_for(state_obj)
            result = fields[name]
            cache_remaining -= 1
            if cache_remaining <= 0:
                cache_frame_id = None
                cache_fields = None
                cache_remaining = 0
            return result

        return extract

    return [NCVarSpec(name=name, dims=dims, extract=make_extract(name)) for name, dims in names_and_dims]


def replace_variable(variables: Sequence[NCVarSpec], name: str, new_spec: NCVarSpec) -> list[NCVarSpec]:
    replaced = False
    result: list[NCVarSpec] = []
    for spec in variables:
        if spec.name == name and not replaced:
            result.append(new_spec)
            replaced = True
        else:
            result.append(spec)
    if not replaced:
        raise KeyError(name)
    return result


class _AsyncNCWriter:
    """Single-worker streaming NetCDF writer with bounded frame backlog.

    Concurrency model
    -----------------
    * The **producer** (caller of ``write``) is always the simulation
      main thread. ``write`` runs ``_extract_frame`` synchronously —
      i.e. the GPU→CPU copies and OBCS overlays happen on the producer
      thread, so simulation state may be mutated immediately after the
      call returns.
    * A single **consumer** worker thread (``ThreadPoolExecutor`` with
      ``max_workers=1``) serializes ``netCDF4`` append calls. The worker
      is spawned eagerly in ``__init__`` so its startup cost is paid
      once, not on the first frame.
    * Backpressure: at most ``MAX_QUEUE`` frame futures are in flight at
      any time. When that ceiling is hit the producer blocks on the
      oldest future before submitting the next one — bounding peak
      memory regardless of how slow the disk is.

    Lifecycle
    ---------
    ``flush()`` is single-shot and finalizing: it drains the backlog,
    raises any captured worker error, and shuts the executor down. Any
    subsequent ``write`` raises ``RuntimeError``; a later ``flush`` returns
    at once, re-raising a captured worker error if there was one.  The
    runtime relies on this: it drains the writer before writing the station
    file and flushes again once the loop has returned.

    The worker's appends go through netCDF4 without xarray's HDF5 lock, so
    no other thread may call into NetCDF/HDF5 until ``flush`` has returned.
    """

    def __init__(
        self,
        output_dir: Path,
        grid: Any,
        dump_freq_seconds: float,
        *,
        variables: Sequence[NCVarSpec],
        coords: Mapping[str, np.ndarray],
        dataset_attrs: Mapping[str, Any],
        time_units: str,
        records_per_file: int | None,
    ) -> None:
        self._variables = list(variables)
        self._coords = {name: np.asarray(value) for name, value in coords.items()}
        self._dataset_attrs = dict(dataset_attrs)
        self._time_units = str(time_units)
        self._time_scale = 3600.0 if self._time_units == "hours" else 1.0
        self._nc_output_dir = output_dir / "nc_snapshots"
        self._nc_output_dir.mkdir(parents=True, exist_ok=True)
        self._flush_every = (
            max(1, int(records_per_file))
            if records_per_file is not None
            else max(1, int((24.0 * 3600.0) / max(float(dump_freq_seconds), 1.0)))
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nc_writer")
        # Force the lazy executor to create its worker before the first frame.
        self._executor.submit(lambda: None).result()
        self._pending_futures: deque[Future[None]] = deque()
        self._error: BaseException | None = None
        self._closed = False
        self._frame_counter = 0
        # Absolute dump cadence — used to route each frame to its dayNNN file by
        # global physical time (resume-safe), instead of a per-writer counter.
        self._dump_freq_seconds = max(float(dump_freq_seconds), 1.0)
        # dayNNN files this writer has already written a frame to; touched only
        # by the worker thread.
        self._started_days: set[int] = set()

    def write(self, step: int, time_s: float, state_obj: Any) -> None:
        self._raise_if_error()
        if self._closed:
            raise RuntimeError("NC snapshot writer is closed")
        self._drop_completed_futures()
        frame = self._extract_frame(step, time_s, state_obj)
        future = self._executor.submit(self._write_frame_guarded, frame)
        self._pending_futures.append(future)
        self._check_queue_pressure()
        self._raise_if_error()

    def flush(self) -> None:
        try:
            self._raise_if_error()
            while self._pending_futures:
                self._pending_futures.popleft().result()
            self._raise_if_error()
        finally:
            self._executor.shutdown(wait=True)
            self._closed = True

    def _extract_frame(self, step: int, time_s: float, state_obj: Any) -> _Frame:
        # The module-level frame globals form the public extractor
        # protocol exposed via ``current_frame_time_seconds`` — they are
        # only meaningful inside this block and are save/restored so a
        # nested extractor call (none today, but cheap defense) can't
        # observe a stale value.
        global _CURRENT_FRAME_ID, _CURRENT_FRAME_TIME_SECONDS
        previous_time = _CURRENT_FRAME_TIME_SECONDS
        previous_frame_id = _CURRENT_FRAME_ID
        _CURRENT_FRAME_TIME_SECONDS = float(time_s)
        _CURRENT_FRAME_ID = self._frame_counter
        self._frame_counter += 1
        try:
            # np.ascontiguousarray hands the worker a tightly-packed,
            # C-contiguous buffer that no producer code path can mutate
            # — extractors that return torch-backed views are still
            # safe because numpy() already detached, and copies are
            # only made when the input wasn't already contiguous.
            frame: _Frame = {
                spec.name: np.ascontiguousarray(spec.extract(state_obj))
                for spec in self._variables
            }
        finally:
            _CURRENT_FRAME_TIME_SECONDS = previous_time
            _CURRENT_FRAME_ID = previous_frame_id
        frame["__step__"] = int(step)
        frame["__time_s__"] = float(time_s)
        return frame

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _drop_completed_futures(self) -> None:
        while self._pending_futures and self._pending_futures[0].done():
            self._pending_futures.popleft().result()

    def _check_queue_pressure(self) -> None:
        if len(self._pending_futures) >= MAX_QUEUE:
            self._pending_futures.popleft().result()

    def _write_frame_guarded(self, frame: _Frame) -> None:
        try:
            self._write_frame_impl(frame)
        except BaseException as exc:
            self._error = exc
            raise

    def _write_frame_impl(self, frame: _Frame) -> None:
        # Resume-safe file routing: derive the dayNNN index from the frame's
        # ABSOLUTE physical time, not a per-writer counter that restarts at 1.
        # A resumed run builds a fresh writer; restarting the counter made the
        # resume segment write day001 again and overwrite the first segment's
        # day001 — i.e. the IC (day001.nc would hold the final step instead
        # of t=0, and the true IC frames would be lost). Routing by global time keeps dayNNN continuous across
        # resume boundaries; an overlapping resume frame appends to its file.
        #
        # A file is (re)created only by the first frame this writer routes to
        # it, and only when that frame opens the file (position 0) or the file
        # does not exist yet; every later frame appends.  Deciding from the
        # position alone let a second frame that rounds into position 0 --
        # the end frame of a run shorter than half a dump interval, t=60 s
        # against a 21600 s cadence -- recreate day001.nc over the t=0 frame,
        # and a first frame at a non-zero position append to a file that did
        # not exist.
        day_counter, pos_in_day = self._file_routing(frame["__time_s__"])
        ds = self._frame_to_dataset(frame, day_counter)
        nc_path = self._nc_output_dir / f"day{day_counter:03d}.nc"
        if day_counter not in self._started_days and (
            pos_in_day == 0 or not nc_path.exists()
        ):
            ds.to_netcdf(nc_path, mode="w", unlimited_dims=["time"])
        else:
            _append_frame_to_netcdf(nc_path, ds)
        self._started_days.add(day_counter)

    def _file_routing(self, time_s: float) -> tuple[int, int]:
        """Map a frame's absolute time (s) → (1-based dayNNN, position-in-file)."""
        global_idx = int(round(float(time_s) / self._dump_freq_seconds))
        return global_idx // self._flush_every + 1, global_idx % self._flush_every

    def _frame_to_dataset(self, frame: _Frame, day_counter: int) -> Any:
        import xarray as xr

        time_values = np.array([float(frame["__time_s__"]) / self._time_scale], dtype=np.float64)
        data_vars = {spec.name: (spec.dims, np.expand_dims(frame[spec.name], axis=0)) for spec in self._variables}
        ds = xr.Dataset(data_vars, coords={"time": time_values, **self._coords})
        ds.time.attrs["time_unit"] = self._time_units
        ds.attrs["day"] = day_counter
        for key, value in self._dataset_attrs.items():
            if key != "day":
                ds.attrs[key] = value
        for spec in self._variables:
            if spec.attrs:
                ds[spec.name].attrs.update(dict(spec.attrs))
        return ds


def _append_frame_to_netcdf(path: Path, ds: Any) -> None:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Streaming NC snapshot writer requires the 'netCDF4' package "
            "(install with `pip install netcdf4` or `conda install -c "
            "conda-forge netcdf4`)."
        ) from exc

    with Dataset(str(path), "a") as nc:
        index = len(nc.dimensions["time"])
        nc.variables["time"][index : index + 1] = np.asarray(ds["time"].values)
        for name in ds.data_vars:
            nc.variables[name][index : index + 1, ...] = np.asarray(ds[name].values)


def _default_coords(grid: Any, coords: Mapping[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    result = {"depth": _coord_depth(grid), "lat": _coord_lat(grid), "lon": _coord_lon(grid)}
    if coords is None:
        return result
    for name, value in coords.items():
        if value is not None:
            result[name] = np.asarray(value)
    return result


def make_nc_writer(
    output_dir: Path,
    grid: Any,
    dump_freq_seconds: float,
    *,
    variables: Sequence[NCVarSpec],
    coords: Mapping[str, np.ndarray | None] | None = None,
    dataset_attrs: Mapping[str, Any] | None = None,
    time_units: str = "seconds",
    records_per_file: int | None = None,
) -> tuple[Callable[[int, float, Any], None], Callable[[], None]]:
    """Return ``(write, flush)`` callables for streaming per-day NetCDF snapshots.

    Parameters
    ----------
    output_dir : Path
        Run output directory; snapshots land under ``output_dir/nc_snapshots/``.
    grid : Any
        StructuredCGrid-like object; used to derive default lat/lon/depth
        coordinates when the caller does not override them in ``coords``.
    dump_freq_seconds : float
        Snapshot cadence in seconds — only used to derive ``records_per_file``
        when that argument is left ``None`` (frames-per-day = 86400 / cadence).
    variables : Sequence[NCVarSpec]
        Ordered registry of variables to write each frame. Use
        :func:`default_state_variables` for the canonical eta/temp/salt/u/v/w
        set and :func:`replace_variable` to swap individual extractors
        (e.g. an OBCS-overlay W extractor).
    coords : Mapping[str, np.ndarray | None] | None
        Coordinate overrides per dim name. ``None`` values fall back to the
        grid-derived defaults, so callers can pass ``{"lat": custom_lat,
        "lon": None}`` to override only one axis.
    dataset_attrs : Mapping[str, Any] | None
        Extra ``ds.attrs`` to set on every per-day Dataset (``"day"`` is
        managed by the writer and cannot be overridden).
    time_units : str
        Either ``"seconds"`` (the default) or ``"hours"``; only affects the
        ``time_unit`` attribute and the divisor applied to the per-frame
        ``time_s`` argument.
    records_per_file : int | None
        Frames per ``dayNNN.nc`` file. When ``None``, derived from
        ``dump_freq_seconds`` to roughly one calendar day per file.

    Returns
    -------
    (write, flush) : tuple of callables
        ``write(step, time_s, state_obj)`` extracts on the producer thread
        and submits the frame to the background writer.
        ``flush()`` is single-shot and finalizing — see
        :class:`_AsyncNCWriter` for the concurrency / lifecycle contract.
    """

    writer = _AsyncNCWriter(
        output_dir, grid, dump_freq_seconds, variables=list(variables),
        coords=_default_coords(grid, coords),
        dataset_attrs={} if dataset_attrs is None else dict(dataset_attrs),
        time_units=time_units, records_per_file=records_per_file,
    )
    return writer.write, writer.flush
