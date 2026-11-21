"""High-frequency station (point) output.

Reproduces the spirit of Gong et al. 2023's bespoke MITgcm "Fast Station Output"
(by Lequan Chi): instead of dumping the whole 3-D field at high cadence, record
only the full water column (all depth levels) at a handful of stations (nearest
grid point to each ``lat``/``lon``) into a compact NetCDF ``(time, station, depth)``.

Cheap by construction — a few points × ``nz`` × ``n_record_steps`` — so it can run
at 1-min cadence for an entire multi-day run to feed mooring-style diagnostics
(e.g. the isotherm-depth + baroclinic-velocity time series of Gong Fig. 4/5).

Read-only on the model state, write-only output: it never touches the physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from locomgcm.io.runner.builders.nc_snapshot import _coord_depth

# 3-D state fields are stored (batch, depth, lat, lon); a station column is
# ``field.data[0, :, j, i]``.  ``eta`` is 2-D: ``field.data[0, j, i]``.
_DEFAULT_VARS: tuple[str, ...] = ("temp", "salt", "u", "v", "eta")
_2D_VARS = frozenset({"eta"})


@dataclass(frozen=True)
class StationPoint:
    name: str
    global_j: int
    global_i: int
    lat: float
    lon: float


class StationRecorder:
    """Accumulate per-station water columns at a fixed step interval."""

    def __init__(
        self,
        stations: Sequence[Mapping[str, Any]],
        lat: np.ndarray,
        lon: np.ndarray,
        depth: np.ndarray,
        interval_steps: int,
        variables: Sequence[str] = _DEFAULT_VARS,
    ) -> None:
        if not stations:
            raise ValueError("StationRecorder needs at least one station")
        self.lat = np.asarray(lat, dtype=np.float64)
        self.lon = np.asarray(lon, dtype=np.float64)
        self.depth = np.asarray(depth, dtype=np.float64)
        self.interval_steps = max(1, int(interval_steps))
        self.variables = tuple(variables)

        self.points = resolve_station_points(stations, self.lat, self.lon)
        self.names = [p.name for p in self.points]
        self.ji = [(p.global_j, p.global_i) for p in self.points]
        self.snap_lat = [p.lat for p in self.points]
        self.snap_lon = [p.lon for p in self.points]

        self._times: list[float] = []
        self._buf: dict[str, list[np.ndarray]] = {v: [] for v in self.variables}

    def should_record(self, global_step: int) -> bool:
        return int(global_step) % self.interval_steps == 0

    def record(self, abs_time: float, state: Any) -> None:
        """Extract + stash each station column (cheap GPU index -> tiny CPU copy)."""
        self._times.append(float(abs_time))
        for var in self.variables:
            cols = np.empty(
                (len(self.ji),) if var in _2D_VARS else (len(self.ji), self.depth.size),
                dtype=np.float64,
            )
            for k, (j, i) in enumerate(self.ji):
                cols[k] = _extract_column(state, var, j, i)
            self._buf[var].append(cols)

    @property
    def n_records(self) -> int:
        return len(self._times)

    def flush(self, path: str | Path) -> Path | None:
        """Write the accumulated columns to ``path`` (NetCDF).  No-op if empty."""
        if not self._times:
            return None
        data = {var: np.stack(self._buf[var], axis=0) for var in self.variables}
        return _write_station_dataset(
            path,
            points=self.points,
            depth=self.depth,
            times=np.asarray(self._times, dtype=np.float64),
            variables=self.variables,
            data=data,
        )


class DistributedStationRecorder:
    """Rank-local station recorder; rank 0 writes the merged NetCDF.

    The station lookup is global, but each rank records only stations whose
    center cell lies inside its ``ParallelLayout`` tile.  Local state indexing
    therefore subtracts the layout offsets before extracting columns.
    """

    def __init__(
        self,
        station_points: Sequence[StationPoint],
        depth: np.ndarray,
        interval_steps: int,
        variables: Sequence[str],
        layout: Any,
    ) -> None:
        self.points = tuple(station_points)
        self.depth = np.asarray(depth, dtype=np.float64)
        self.interval_steps = max(1, int(interval_steps))
        self.variables = tuple(variables)
        self.layout = layout
        self.names = [p.name for p in self.points]
        self.ji = [(p.global_j, p.global_i) for p in self.points]
        self.snap_lat = [p.lat for p in self.points]
        self.snap_lon = [p.lon for p in self.points]

        owned: list[tuple[int, int, int]] = []
        for station_idx, point in enumerate(self.points):
            if (
                layout.x_start <= point.global_i < layout.x_stop
                and layout.y_start <= point.global_j < layout.y_stop
            ):
                owned.append(
                    (
                        station_idx,
                        point.global_j - layout.y_start,
                        point.global_i - layout.x_start,
                    )
                )
        self.owned_station_indices = [item[0] for item in owned]
        self.local_j_i = [(item[1], item[2]) for item in owned]
        self._owned = owned
        self._times: list[float] = []
        self._buf: dict[str, list[np.ndarray]] = {v: [] for v in self.variables}

    def should_record(self, global_step: int) -> bool:
        return int(global_step) % self.interval_steps == 0

    def record(self, abs_time: float, state: Any) -> None:
        self._times.append(float(abs_time))
        for var in self.variables:
            cols = np.empty(
                (len(self._owned),) if var in _2D_VARS else (len(self._owned), self.depth.size),
                dtype=np.float64,
            )
            for k, (_station_idx, local_j, local_i) in enumerate(self._owned):
                cols[k] = _extract_column(state, var, local_j, local_i)
            self._buf[var].append(cols)

    @property
    def n_records(self) -> int:
        return len(self._times)

    def flush(self, path: str | Path) -> Path | None:
        if not self._times:
            return None
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed station output requires torch.distributed")
        from locomgcm.distributed.gather import require_whole_job_layout

        require_whole_job_layout(self.layout)
        payload = {
            "rank": int(self.layout.rank),
            "times": np.asarray(self._times, dtype=np.float64),
            "station_indices": np.asarray(self.owned_station_indices, dtype=np.int64),
            "data": {
                var: np.stack(self._buf[var], axis=0)
                for var in self.variables
            },
        }
        gathered = [None for _ in range(int(self.layout.world_size))] if self.layout.rank == 0 else None
        dist.gather_object(payload, object_gather_list=gathered, dst=0)
        if self.layout.rank != 0:
            return None
        if gathered is None or any(item is None for item in gathered):
            raise RuntimeError("station gather to rank 0 returned incomplete payloads")
        merged = _merge_station_payloads(
            gathered,
            points=self.points,
            depth=self.depth,
            variables=self.variables,
        )
        return _write_station_dataset(
            path,
            points=self.points,
            depth=self.depth,
            times=np.asarray(self._times, dtype=np.float64),
            variables=self.variables,
            data=merged,
        )


def resolve_station_points(
    stations: Sequence[Mapping[str, Any]],
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[StationPoint, ...]:
    points: list[StationPoint] = []
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    for k, s in enumerate(stations):
        name = str(s.get("name", f"st{k}"))
        j = int(np.argmin(np.abs(lat_arr - float(s["lat"]))))
        i = int(np.argmin(np.abs(lon_arr - float(s["lon"]))))
        points.append(
            StationPoint(
                name=name,
                global_j=j,
                global_i=i,
                lat=float(lat_arr[j]),
                lon=float(lon_arr[i]),
            )
        )
    return tuple(points)


def build_station_recorder(
    run_cfg: Any,
    grid: Any,
    dt: float,
    *,
    distributed_context: Any | None = None,
) -> StationRecorder | DistributedStationRecorder | None:
    """Construct a recorder from ``run_cfg.output.station_output`` (or None)."""
    out = getattr(run_cfg, "output", None)
    cfg = getattr(out, "station_output", None) if out is not None else None
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    stations = list(getattr(cfg, "stations", []) or [])
    if not stations:
        return None
    interval_s = float(getattr(cfg, "interval_seconds", dt) or dt)
    interval_steps = max(1, int(round(interval_s / float(dt))))
    variables = tuple(getattr(cfg, "variables", _DEFAULT_VARS) or _DEFAULT_VARS)
    # Geographic cell-center lat/lon in DEGREES for the nearest-station lookup.
    # NOT the nc_snapshot _coord_lat/_coord_lon helpers -- those return grid
    # INDICES (arange(ny)) / cumulative distance, so matching a station's
    # (lat, lon) against them lands on the wrong column (e.g. DS 20.737N /
    # 117.746E resolved to j=21 instead of j=613).  Derive degrees from the grid
    # origin + resolution (uniform spacing; a variable-spacing grid would need the
    # per-cell lon/lat array).
    g = run_cfg.grid
    res = float(g.resolution_deg)
    lat = (float(g.center_lat) - g.ny / 2.0 * res) + (np.arange(g.ny) + 0.5) * res
    lon = (float(g.center_lon) - g.nx / 2.0 * res) + (np.arange(g.nx) + 0.5) * res
    depth = _coord_depth(grid)
    if distributed_context is not None and getattr(distributed_context, "is_distributed", False):
        layout = getattr(grid, "parallel_layout", None)
        if layout is None:
            raise RuntimeError("distributed station output requires grid.parallel_layout")
        return DistributedStationRecorder(
            station_points=resolve_station_points(stations, lat, lon),
            depth=depth,
            interval_steps=interval_steps,
            variables=variables,
            layout=layout,
        )
    return StationRecorder(
        stations=stations,
        lat=lat,
        lon=lon,
        depth=depth,
        interval_steps=interval_steps,
        variables=variables,
    )


def _extract_column(state: Any, var: str, j: int, i: int) -> np.ndarray | float:
    field = getattr(state, var).data
    col = field[0, j, i] if var in _2D_VARS else field[0, :, j, i]
    arr = col.detach().cpu().numpy().astype(np.float64)
    if var in _2D_VARS:
        return float(arr)
    return arr


def _merge_station_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    points: Sequence[StationPoint],
    depth: np.ndarray,
    variables: Sequence[str],
) -> dict[str, np.ndarray]:
    if not payloads:
        raise RuntimeError("no station payloads gathered")
    times = np.asarray(payloads[0]["times"], dtype=np.float64)
    n_time = times.size
    n_station = len(points)
    owner_counts = np.zeros(n_station, dtype=np.int64)
    merged: dict[str, np.ndarray] = {}
    for var in variables:
        shape = (n_time, n_station) if var in _2D_VARS else (n_time, n_station, depth.size)
        merged[var] = np.full(shape, np.nan, dtype=np.float64)

    for payload in payloads:
        p_times = np.asarray(payload["times"], dtype=np.float64)
        if p_times.shape != times.shape or not np.allclose(p_times, times, rtol=0.0, atol=0.0):
            raise RuntimeError("station payload times differ across ranks")
        station_indices = np.asarray(payload["station_indices"], dtype=np.int64)
        if station_indices.size == 0:
            continue
        if np.any(station_indices < 0) or np.any(station_indices >= n_station):
            raise RuntimeError("station payload contains an out-of-range station index")
        owner_counts[station_indices] += 1
        for var in variables:
            values = np.asarray(payload["data"][var], dtype=np.float64)
            expected = (
                (n_time, station_indices.size)
                if var in _2D_VARS
                else (n_time, station_indices.size, depth.size)
            )
            if values.shape != expected:
                raise RuntimeError(
                    f"station payload for {var!r} has shape {values.shape}, expected {expected}"
                )
            merged[var][:, station_indices] = values

    if not np.all(owner_counts == 1):
        raise RuntimeError(
            "distributed station ownership is not one-to-one: "
            f"{owner_counts.tolist()}"
        )
    return merged


def _write_station_dataset(
    path: str | Path,
    *,
    points: Sequence[StationPoint],
    depth: np.ndarray,
    times: np.ndarray,
    variables: Sequence[str],
    data: Mapping[str, np.ndarray],
) -> Path:
    import xarray as xr

    coords = {
        "time": np.asarray(times, dtype=np.float64),
        "station": np.asarray([p.name for p in points], dtype=object),
        "depth": np.asarray(depth, dtype=np.float64),
        "station_lat": ("station", np.asarray([p.lat for p in points], dtype=np.float64)),
        "station_lon": ("station", np.asarray([p.lon for p in points], dtype=np.float64)),
    }
    data_vars: dict[str, Any] = {}
    for var in variables:
        arr = np.asarray(data[var], dtype=np.float64)
        dims = ("time", "station") if var in _2D_VARS else ("time", "station", "depth")
        data_vars[var] = (dims, arr)
    out = Path(path)
    xr.Dataset(data_vars, coords=coords).to_netcdf(out)
    return out
