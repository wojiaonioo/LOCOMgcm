"""CLI: build a LOCOMgcm-only dashboard ``preprocessed.zarr`` from a lat-lon run dir.

Runs whose ``metadata.type`` is a global free-forecast (LOCOMgcm-only, no MIT
reference) have no comparison adapter, so ``helper.preprocess.run_pipeline``
cannot ingest them (``detect_adapter`` raises "No adapter for type"). This
driver reads ``nc_snapshots/*.nc`` plus the stage1a grid tensors and writes the
model-only store the dashboard reads, via
:func:`helper.preprocess.zarr_writer.write_zarr_store_model_only`.

Horizontal cell centres come from ``center + (i - (n - 1) / 2) * res``
(:func:`locomgcm.io.runner.builders.nc_snapshot.latlon_centre_axes`), the
axes stage 2 writes into the nc files of a spherical-polar grid; the nc coords
themselves are not read, because older runs and Cartesian grids store a row
index and metres there. ``res`` and
the snapshot cadence default to the values recorded in ``run_params.yaml``;
pass ``--res`` / ``--snap-seconds`` to override (e.g. regional runs). Stage 4
in LOCOMgcm-only mode calls :func:`build_dashboard_zarr` with all four values
taken from the run configuration.

Usage:
  python -m helper.preprocess.model_only outputs/data/<run-id>
  # then point the dashboard at outputs/data/<run-id>/preprocessed.zarr
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from locomgcm.io.runner.builders.nc_snapshot import latlon_centre_axes

from .adapters.base import StandardGrid
from .zarr_writer import write_zarr_store_model_only


def _read_yaml(path: Path) -> dict:
    path = Path(path)
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def build_dashboard_zarr(
    run_dir: Path | str,
    *,
    res: float | None = None,
    snap_seconds: float | None = None,
    center_lat: float = 0.0,
    center_lon: float = 180.0,
    out: Path | str | None = None,
) -> Path:
    """Write ``<run_dir>/preprocessed.zarr`` (model-only) from a LOCOMgcm-only run.

    ``res`` / ``snap_seconds`` default to ``resolution_deg`` /
    ``snapshot_interval_seconds`` in ``_pipeline_meta/model_input/run_params.yaml``.
    """
    run_dir = Path(run_dir)
    rp = _read_yaml(run_dir / "_pipeline_meta" / "model_input" / "run_params.yaml")
    out = Path(out) if out is not None else run_dir / "preprocessed.zarr"

    if res is None:
        res = float(rp.get("resolution_deg") or 0.0)
        if not res:
            raise ValueError(
                "resolution_deg not found in run_params.yaml — pass --res explicitly"
            )
    if snap_seconds is None:
        snap_seconds = float(rp.get("snapshot_interval_seconds") or 0.0)
        if not snap_seconds:
            raise ValueError(
                "snapshot_interval_seconds not found in run_params.yaml — "
                "pass --snap-seconds explicitly"
            )

    files = sorted(glob.glob(str(run_dir / "nc_snapshots" / "*.nc")))
    if not files:
        raise FileNotFoundError(f"no nc_snapshots/*.nc under {run_dir}")
    print(f"[model_only] {len(files)} nc files: "
          f"{Path(files[0]).name} .. {Path(files[-1]).name}", flush=True)
    # nested concat along time by FILE ORDER (does not rely on nc time-coord values)
    model_ds = xr.open_mfdataset(files, combine="nested", concat_dim="time")

    # The nc time-coord is the ABSOLUTE physical time. Filename order is NOT
    # time order for resumed runs — the resume segment's dayNNN can precede
    # earlier files (nc_snapshot routes by global time). Sort by the time coord
    # and drop resume-overlap duplicate frames so the dashboard axis is
    # monotonic. Fall back to arange*cadence only if the coord is degenerate.
    times_raw = np.asarray(model_ds["time"].values, dtype=np.float64)
    if np.unique(times_raw).size > 1:
        tunit = xr.open_dataset(files[0])["time"].attrs.get("time_unit", "hours")
        tscale = 3600.0 if tunit == "hours" else 1.0
        model_ds = model_ds.sortby("time")
        times = np.asarray(model_ds["time"].values, dtype=np.float64)
        _, uniq = np.unique(times, return_index=True)
        if uniq.size != times.size:
            print(f"[model_only] dropping {times.size - uniq.size} duplicate-time "
                  f"frame(s) (resume overlap)", flush=True)
            model_ds = model_ds.isel(time=np.sort(uniq))
        model_time = np.asarray(model_ds["time"].values, dtype=np.float64) * tscale
    else:
        # Degenerate coord (integer index / all-zero) — synthesize from cadence.
        model_time = np.arange(model_ds.sizes["time"], dtype=np.float64) * snap_seconds

    nt = model_ds.sizes["time"]
    nz = model_ds.sizes["depth"]
    ny = model_ds.sizes["lat"]
    nx = model_ds.sizes["lon"]

    lat, lon = latlon_centre_axes(center_lat, center_lon, res, ny=ny, nx=nx)
    depth = np.asarray(model_ds["depth"].values, dtype=np.float64)

    # land mask from the stage1a grid tensors (data_loader reads timeseries/hfac.npy)
    grid_npz = run_dir / "_pipeline_meta" / "model_input" / "grid" / "tensors.npz"
    hfac = None
    if grid_npz.exists():
        hfac = np.asarray(np.load(grid_npz)["hFacC"], dtype=np.float64)
        print(f"[model_only] hfac from {grid_npz.name}: shape={hfac.shape} "
              f"land_frac={(hfac == 0).mean():.3f}", flush=True)
    else:
        print(f"[model_only] WARNING: {grid_npz} missing — dashboard will NOT mask land",
              flush=True)

    grid = StandardGrid(nx=nx, ny=ny, nz=nz, lat=lat, lon=lon, depth=depth,
                        cell_area=None, hfac=hfac)
    print(f"[model_only] grid={nx}x{ny}x{nz} nt={nt} res={res} snap={snap_seconds}s "
          f"lon[{lon[0]:.2f},{lon[-1]:.2f}] lat[{lat[0]:.2f},{lat[-1]:.2f}]", flush=True)
    write_zarr_store_model_only(out, model_ds, model_time, grid)
    print(f"[model_only] wrote {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="LOCOMgcm-only run -> dashboard preprocessed.zarr (lat-lon, model-only)")
    p.add_argument("run_dir", type=Path, help="run output dir (contains nc_snapshots/)")
    p.add_argument("--out", type=Path, default=None,
                   help="output zarr (default <run_dir>/preprocessed.zarr)")
    p.add_argument("--res", type=float, default=None,
                   help="grid resolution in deg (default: run_params resolution_deg)")
    p.add_argument("--snap-seconds", type=float, default=None,
                   help="snapshot cadence in s (default: run_params snapshot_interval_seconds)")
    p.add_argument("--center-lat", type=float, default=0.0)
    p.add_argument("--center-lon", type=float, default=180.0)
    args = p.parse_args(argv)
    build_dashboard_zarr(
        args.run_dir, res=args.res, snap_seconds=args.snap_seconds,
        center_lat=args.center_lat, center_lon=args.center_lon, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
