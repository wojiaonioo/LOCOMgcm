from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr

from .base import ExperimentAdapter, StandardGrid

_MITGCM_VAR_MAP = {"temp": "T", "salt": "S", "u": "U", "v": "V", "eta": "Eta", "w": "W"}
_NC_VARS = ("temp", "salt", "u", "v", "eta", "w")
_ACCEPTED_TYPES = {"mitgcm_comparison", "mitgcm_only", "hycom_comparison", None}


class HycomAdapter(ExperimentAdapter):
    """Adapter for LOCOMgcm NC + MITgcm binary experiment outputs."""

    @classmethod
    def can_handle(cls, metadata: dict) -> bool:
        exp_type = metadata.get("type")
        return exp_type in _ACCEPTED_TYPES

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def load_grid(self) -> StandardGrid:
        nc_dir = self.experiment_dir / self._model_path()
        nc_files = sorted(nc_dir.glob("day*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No day*.nc files in {nc_dir}")
        with xr.open_dataset(nc_files[0]) as ds:
            lat = ds.lat.values.astype(np.float64)
            lon = ds.lon.values.astype(np.float64)
            depth = (
                ds.depth.values.astype(np.float64)
                if "depth" in ds.dims
                else np.array([0.0])
            )
        ny, nx, nz = len(lat), len(lon), len(depth)
        # Stage 2 writes degree axes for a spherical-polar grid, but runs from
        # before that change (and Cartesian grids) store the row index in `lat`
        # and metres in `lon`.  Recover geographic coords from
        # config_snapshot/source_config.yaml when the stored values are
        # obviously non-geographic (e.g. lon > 360 °).
        needs_lat = lat.size > 1 and (np.nanmax(lat) > 90 or np.nanmin(lat) < -90
                                      or np.allclose(lat, np.arange(ny)))
        needs_lon = lon.size > 1 and (np.nanmax(lon) > 360 or np.nanmin(lon) < -180)
        if needs_lat or needs_lon:
            real = self._geo_axes_from_config(nx, ny)
            if real is not None:
                if needs_lat:
                    lat = real[0]
                if needs_lon:
                    lon = real[1]
            else:
                # Recovery is REQUIRED (NC axes are index/metres) but the
                # snapshot is missing — do NOT silently keep non-geographic
                # coords. They make grid_signature mismatch every degree-based
                # run, so cross-run diffs fail with a misleading "Grids differ".
                # Common cause: a SKIP_PREP/reused-prep run that never wrote
                # config_snapshot/. Re-run stage1a or copy config_snapshot from
                # the source run.
                import warnings
                warnings.warn(
                    "load_grid: NC lat/lon are non-geographic (lat~indices, "
                    "lon~metres) and config_snapshot/source_config.yaml is "
                    f"missing under {self.experiment_dir} — cannot recover "
                    "degree axes. The stored grid will not match geographic "
                    "runs (cross-run diff will report 'Grids differ'). Copy "
                    "config_snapshot/ from the source run or re-run stage1a.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        hfac = self._load_hfac(nx, ny, nz)
        return StandardGrid(
            nx=nx, ny=ny, nz=nz,
            lat=lat, lon=lon, depth=depth,
            cell_area=None, hfac=hfac,
        )

    # ------------------------------------------------------------------
    # Model data: lazy via open_mfdataset
    # ------------------------------------------------------------------

    def load_model_data(self) -> xr.Dataset:
        nc_dir = self.experiment_dir / self._model_path()
        nc_files = sorted(nc_dir.glob("day*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No day*.nc files in {nc_dir}")
        merged = xr.open_mfdataset(
            nc_files,
            combine="nested",
            concat_dim="time",
            parallel=False,
        )
        available = [v for v in _NC_VARS if v in merged]
        result = merged[available]
        # NC time is in hours; normalise to seconds to match MITgcm convention.
        # Heuristic: if max time < 100_000 it's almost certainly hours, not seconds.
        t = result.time.values.astype(np.float64)
        if float(t.max()) < 100_000:
            result = result.assign_coords(time=t * 3600.0)
        return result

    # ------------------------------------------------------------------
    # Reference: lightweight header scan only; actual data via iterator
    # ------------------------------------------------------------------

    def load_reference_data(self) -> xr.Dataset:
        """Coord-only dataset (no field arrays) for pipeline metadata queries.

        Time is in seconds (MITgcm step × dt), matching the normalised NC time.
        Use iter_reference_steps() for actual field I/O.
        """
        steps, time_vals = self._scan_mitgcm_steps()  # already in seconds
        grid = self.load_grid()
        return xr.Dataset(
            coords={
                "time": time_vals,
                "depth": grid.depth,
                "lat": grid.lat,
                "lon": grid.lon,
            }
        )

    def iter_reference_steps(self) -> Iterator[tuple[float, dict[str, np.ndarray]]]:
        """Yield (time_seconds, {var: ndarray}) one MITgcm step at a time.

        Uses locomgcm.io.mds_v2 multi-tile reader so it handles both
        single-tile (`Eta.NNNNNNNNNN.data`) and tile-suffixed
        (`Eta.NNNNNNNNNN.001.001.data`) MDS layouts.

        Shapes:
          eta  → (ny, nx)   float32
          others → (nz, ny, nx)  float32
        """
        from locomgcm.io.mds_v2 import discover_tile_index, read_multi_tile_field

        run_dir, _ = self._ref_dirs()
        steps, time_vals = self._scan_mitgcm_steps()

        for step, t in zip(steps, time_vals):
            arrays: dict[str, np.ndarray] = {}
            for var, mit_name in _MITGCM_VAR_MAP.items():
                try:
                    idx = discover_tile_index(run_dir, field_stem=mit_name, step=step)
                    field = read_multi_tile_field(run_dir, mit_name, step=step, tile_index=idx)
                except (ValueError, FileNotFoundError):
                    continue
                arr = field.array  # (nrecords, nz, ny, nx)
                if var == "eta":
                    # Eta written as (nrecords, 1, ny, nx) or (nrecords, ny, nx)
                    a = arr[0] if arr.ndim == 4 else arr[0]
                    if a.ndim == 3:
                        a = a[0]
                    arrays[var] = a.astype(np.float32)
                else:
                    arrays[var] = arr[0].astype(np.float32)
            yield float(t), arrays

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_mitgcm_steps(self) -> tuple[list[int], np.ndarray]:
        """Scan run dir for Eta.*.data files → sorted steps + time array.

        Accepts both single-tile `Eta.NNNNNNNNNN.data` and tile-suffixed
        `Eta.NNNNNNNNNN.001.001.data` (MPI optfile always writes tile suffix).
        """
        run_dir, inp_dir = self._ref_dirs()
        dt = self._detect_dt(inp_dir / "data")
        # Optional tile suffix .NNN.NNN before .data
        pattern = re.compile(r"Eta\.(\d{10})(?:\.\d{3}\.\d{3})?\.data$")
        steps = sorted(
            {int(m.group(1)) for f in run_dir.iterdir() if (m := pattern.match(f.name))}
        )
        if not steps:
            raise FileNotFoundError(f"No Eta.*.data files in {run_dir}")
        time_vals = np.array([s * dt for s in steps], dtype=np.float64)
        return steps, time_vals

    def _model_path(self) -> str:
        sources = self.metadata.get("sources", {})
        return sources.get("model", {}).get("path", "nc_snapshots/")

    def _geo_axes_from_config(
        self, nx: int, ny: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Recover (lat, lon) geographic axes from config_snapshot/source_config.yaml.

        For nc files whose `lat`/`lon` hold the row index and metres (runs
        written before stage 2 stored degrees), rebuild the degree axes from
        the snapshot's `grid.center_lat`, `grid.center_lon` and
        `grid.resolution_deg` with the formula stage 2 now uses.
        """
        cfg_path = self.experiment_dir / "config_snapshot" / "source_config.yaml"
        if not cfg_path.exists():
            return None
        try:
            import yaml as _yaml
            with cfg_path.open("r", encoding="utf-8") as fh:
                cfg = _yaml.safe_load(fh) or {}
            grid = cfg.get("grid", {})
            center_lat = float(grid["center_lat"])
            center_lon = float(grid["center_lon"])
            res = float(grid["resolution_deg"])
        except Exception:
            return None
        from locomgcm.io.runner.builders.nc_snapshot import latlon_centre_axes

        return latlon_centre_axes(center_lat, center_lon, res, ny=ny, nx=nx)

    def _ref_dirs(self):
        sources = self.metadata.get("sources", {})
        ref_path = sources.get("reference", {}).get("path", "mitgcm/")
        run_dir = self.experiment_dir / ref_path / "run"
        inp_dir = self.experiment_dir / ref_path / "input"
        return run_dir, inp_dir

    def _detect_dt(self, data_file: Path) -> float:
        dt = 3600.0
        if data_file.exists():
            m = re.search(r"deltaTClock\s*=\s*([0-9.]+)", data_file.read_text(), re.I)
            if m:
                dt = float(m.group(1))
        return dt

    def _load_hfac(self, nx: int, ny: int, nz: int) -> np.ndarray:
        run_dir, _ = self._ref_dirs()
        hfac_path = run_dir / "hFacC.data"
        if hfac_path.exists():
            return (
                np.fromfile(hfac_path, dtype=">f8")
                .reshape(nz, ny, nx)
                .astype(np.float64)
            )
        return np.ones((nz, ny, nx), dtype=np.float64)
