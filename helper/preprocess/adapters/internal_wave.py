"""Adapter for verification/internal_wave (IW) comparison.

The preferred path reads LOCOMgcm NetCDF snapshots from
``<experiment_dir>/nc_snapshots/day*.nc``.  MITgcm reference snapshots are read
directly from binary dump files.

Model ``w`` loaded from NC is expected to be MITgcm upward-positive because the
pipeline compares it directly against MITgcm ``W``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from locomgcm.external_paths import resolve
from typing import Iterator

import numpy as np
import xarray as xr

from .base import ExperimentAdapter, StandardGrid

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_NC_VARS = ("eta", "temp", "u", "w")


class InternalWaveAdapter(ExperimentAdapter):
    """ExperimentAdapter for IW comparison."""

    @classmethod
    def can_handle(cls, metadata: dict) -> bool:
        return metadata.get("type") == "internal_wave_comparison"

    def __init__(self, experiment_dir, metadata):
        super().__init__(experiment_dir, metadata)
        self._grid_cache: StandardGrid | None = None
        self._mods = None

    # ------------------------------------------------------------------
    # Lazy imports (heavy torch deps; defer until pipeline asks)
    # ------------------------------------------------------------------

    def _imports(self):
        if self._mods is not None:
            return self._mods
        from experiments.alignment.mitgcm_internal_wave import (
            NX, NY, NZ, H, DY, DZ,
        )
        self._mods = {
            "NX": NX, "NY": NY, "NZ": NZ,
            "H": H, "DY": DY, "DZ": DZ,
            "MITGCM_RUN": self._run_dir(),
            "DT": 500.0,
        }
        return self._mods

    def _default_run_dir(self) -> Path:
        return resolve("mitgcm_root") / "verification" / "internal_wave" / "run"

    def _run_dir(self) -> Path:
        sources = self.metadata.get("sources", {})
        ref_path = sources.get("reference", {}).get("path")
        if ref_path:
            path = Path(ref_path)
            return path if path.is_absolute() else self.experiment_dir / path
        configured = self.metadata.get("model_args", {}).get("mitgcm_run_dir")
        return Path(configured) if configured else self._default_run_dir()

    def _model_path(self) -> str:
        sources = self.metadata.get("sources", {})
        return sources.get("model", {}).get("path", "nc_snapshots/")

    def _nc_files(self) -> list[Path]:
        nc_dir = self.experiment_dir / self._model_path()
        return sorted(nc_dir.glob("day*.nc"))

    def _has_nc_snapshots(self) -> bool:
        return bool(self._nc_files())

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def load_grid(self) -> StandardGrid:
        if self._grid_cache is not None:
            return self._grid_cache
        nc_files = self._nc_files()
        if nc_files:
            with xr.open_dataset(nc_files[0], decode_timedelta=False) as ds:
                lat = ds.lat.values.astype(np.float64)
                lon = ds.lon.values.astype(np.float64)
                depth = (
                    ds.depth.values.astype(np.float64)
                    if "depth" in ds.coords
                    else np.array([0.0], dtype=np.float64)
                )
            nx, ny, nz = len(lon), len(lat), len(depth)
            hfac = self._load_hfac(nx, ny, nz)
            self._grid_cache = StandardGrid(
                nx=nx, ny=ny, nz=nz,
                lat=lat, lon=lon, depth=depth,
                cell_area=None, hfac=hfac,
            )
            return self._grid_cache
        m = self._imports()
        nx, ny, nz = m["NX"], m["NY"], m["NZ"]
        # Cartesian x: cell-centre positions in metres (placed in lon slot to
        # keep downstream lat/lon-aware code happy).
        lx_m = 13.3e3
        dx = lx_m / nx
        lon = (np.arange(nx) + 0.5) * dx
        lat = np.array([0.0], dtype=np.float64)
        depth = (np.arange(nz) + 0.5) * m["DZ"]
        hfac = self._load_hfac(nx, ny, nz)
        self._grid_cache = StandardGrid(
            nx=nx, ny=ny, nz=nz,
            lat=lat, lon=lon, depth=depth,
            cell_area=None, hfac=hfac,
        )
        return self._grid_cache

    def _load_hfac(self, nx: int, ny: int, nz: int) -> np.ndarray:
        run_dir = self._run_dir()
        # IW MITgcm writes hFacC tiled like Eta (NX/2 per tile) when 2 tiles.
        for cand in (run_dir / "hFacC.001.001.data", run_dir / "hFacC.data"):
            if not cand.exists():
                continue
            raw = np.fromfile(cand, dtype=">f8")
            full = nz * ny * nx
            if raw.size == full:
                return raw.reshape(nz, ny, nx).astype(np.float64)
            snx = nx // 2
            tile = nz * ny * snx
            if raw.size == tile:
                d1 = raw.reshape(nz, ny, snx)
                d2_path = run_dir / "hFacC.002.001.data"
                if d2_path.exists():
                    d2 = np.fromfile(d2_path, dtype=">f8").reshape(nz, ny, snx)
                    return np.concatenate([d1, d2], axis=-1).astype(np.float64)
        return np.ones((nz, ny, nx), dtype=np.float64)

    # ------------------------------------------------------------------
    # Snapshot orchestration
    # ------------------------------------------------------------------

    def _discover_reference_steps(self) -> list[int]:
        run_dir = self._run_dir()
        pattern = __import__("re").compile(r"Eta\.(\d{10})\.001\.001\.data")
        return sorted({
            int(m.group(1)) for f in run_dir.iterdir()
            if (m := pattern.match(f.name))
        })

    def _scan_mitgcm_steps(self) -> tuple[list[int], np.ndarray]:
        """Mimic HycomAdapter API so pipeline can use the iter-style path."""
        dump_steps = self._discover_reference_steps()
        m = self._imports()
        time_vals = np.array([s * m["DT"] for s in dump_steps], dtype=np.float64)
        return dump_steps, time_vals

    # ------------------------------------------------------------------
    # Model data
    # ------------------------------------------------------------------

    def load_model_data(self) -> xr.Dataset:
        nc_files = self._nc_files()
        if nc_files:
            merged = xr.open_mfdataset(
                nc_files,
                combine="nested",
                concat_dim="time",
                parallel=False,
                decode_timedelta=False,
            )
            available = [v for v in _NC_VARS if v in merged]
            result = merged[available]
            t = result.time.values.astype(np.float64)
            units = str(
                result.time.attrs.get("time_unit", result.time.attrs.get("units", ""))
            ).lower()
            if "hour" in units:
                t = t * 3600.0
            result = result.assign_coords(time=t)
            return result

        raise FileNotFoundError(
            f"InternalWaveAdapter requires LOCOMgcm NC snapshots under {self.experiment_dir / self._model_path()}"
        )

    # ------------------------------------------------------------------
    # Reference data
    # ------------------------------------------------------------------

    def load_reference_data(self) -> xr.Dataset:
        """Coord-only Dataset; actual data via ``iter_reference_steps``."""
        dump_steps = self._discover_reference_steps()
        m = self._imports()
        grid = self.load_grid()
        time_vals = np.array([s * m["DT"] for s in dump_steps], dtype=np.float64)
        return xr.Dataset(
            coords={
                "time": time_vals,
                "depth": grid.depth,
                "lat": grid.lat,
                "lon": grid.lon,
            }
        )

    def iter_reference_steps(self) -> Iterator[tuple[float, dict[str, np.ndarray]]]:
        """Stream (time_seconds, {var: ndarray}) for each MITgcm dump step.

        Returned arrays follow MITgcm convention:
          - ``eta``: (ny, nx)
          - ``temp``/``u``/``w``: (nz, ny, nx)
          - ``w`` is upward-positive (MITgcm native).  LOCOMgcm NC output is
            already upward-positive.
        """
        m = self._imports()
        dump_steps = self._discover_reference_steps()
        for step in dump_steps:
            t = step * m["DT"]
            ref = self._load_mit_snapshot(step)
            yield float(t), {
                "eta":  ref["ssh"].astype(np.float32),
                "temp": ref["temp"].astype(np.float32),
                "u":    ref["u"].astype(np.float32),
                "w":    ref["w"].astype(np.float32),
            }

    def _load_mit_field(self, prefix: str, shape: tuple[int, ...], step: int) -> np.ndarray:
        m = self._imports()
        run_dir = self._run_dir()
        step_str = f"{step:010d}"
        t1 = run_dir / f"{prefix}.{step_str}.001.001.data"
        g = run_dir / f"{prefix}.{step_str}.data"
        path = t1 if t1.exists() else g
        data = np.fromfile(path, dtype=">f8")
        expected = int(np.prod(shape))
        if data.size == expected:
            return data.reshape(shape).astype(np.float64)
        snx = int(m["NX"]) // 2
        tile_shape = shape[:-1] + (snx,)
        tile_expected = int(np.prod(tile_shape))
        if data.size == tile_expected:
            d1 = data.reshape(tile_shape)
            t2 = run_dir / f"{prefix}.{step_str}.002.001.data"
            d2 = np.fromfile(t2, dtype=">f8").reshape(tile_shape)
            return np.concatenate([d1, d2], axis=-1).astype(np.float64)
        raise ValueError(f"Cannot load {prefix} at step {step}: got {data.size} vals, expected {expected}")

    def _load_mit_snapshot(self, step: int) -> dict[str, np.ndarray]:
        m = self._imports()
        nx, ny, nz = int(m["NX"]), int(m["NY"]), int(m["NZ"])
        return {
            "ssh": self._load_mit_field("Eta", (ny, nx), step),
            "temp": self._load_mit_field("T", (nz, ny, nx), step),
            "u": self._load_mit_field("U", (nz, ny, nx), step),
            "w": self._load_mit_field("W", (nz, ny, nx), step),
        }
