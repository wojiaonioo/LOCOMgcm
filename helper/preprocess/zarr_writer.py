"""
Streaming Zarr writer for the preprocess pipeline.

Architecture decision on temporal chunks:
  writing chunks=(NT, 1, 1, 1) for shape (NT,NZ,NY,NX) with NZ*NY*NX chunk files
  (e.g. 30*150*150 = 675,000 files) is unacceptably slow on any filesystem, so
  the arrays are chunked one horizontal slab at a time.  Every reader goes
  through ``helper.preprocess.series.open_series`` and indexes frames.

Store layout:
  <store>.zarr/
    model/
      temp_spatial    chunks=(1,1,Ny,Nx)   ← horizontal slicing
      salt_spatial / u_spatial / v_spatial / ssh_spatial
    mitgcm/             same
    diff/               same (spatial only)
    timeseries/hfac.npy partial-cell fraction (NZ, NY, NX)
    <attrs>: lat, lon, depth, time
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr
import zarr

from .adapters.base import StandardGrid

_4D_VARS = ("temp", "salt", "u", "v", "w")


def write_zarr_store(
    store_path: Path | str,
    model_ds: xr.Dataset,
    ref_iter: Iterator[tuple[float, dict[str, np.ndarray]]],
    ref_time_vals: np.ndarray,
    model_time_vals: np.ndarray,
    aligned_model_idx: list[int],
    aligned_ref_idx: list[int],
    grid: StandardGrid,
) -> None:
    store_path = Path(store_path)
    store = zarr.open(str(store_path), mode="w")

    nt = len(aligned_model_idx)
    ny, nx, nz = grid.ny, grid.nx, grid.nz

    model_4d = [v for v in _4D_VARS if v in model_ds]
    has_ssh = "eta" in model_ds

    # Pre-allocate spatial arrays
    _preallocate_spatial(store, "model",  model_4d, has_ssh, nt, nz, ny, nx)
    _preallocate_spatial(store, "mitgcm", model_4d, has_ssh, nt, nz, ny, nx)
    _preallocate_spatial(store, "diff",   model_4d, has_ssh, nt, nz, ny, nx)

    # The spatial arrays are the canonical time series. Keep only per-iteration
    # slabs in memory while producing model, reference, and difference arrays.
    print(f"  [zarr] Streaming {nt} aligned steps ...", flush=True)

    ref_aligned_set = set(aligned_ref_idx)
    ref_step_to_zarr = {r: a for a, r in enumerate(aligned_ref_idx)}

    for ref_seq_pos, (t_ref, ref_arrays) in enumerate(ref_iter):
        if ref_seq_pos not in ref_aligned_set:
            continue

        zarr_t = ref_step_to_zarr[ref_seq_pos]
        model_t = aligned_model_idx[zarr_t]

        # Per-iteration local slabs (freed each step) — replaces the full-series
        # ts_bufs. Used only to compute diff = model - ref for THIS timestep.
        model_slabs: dict[str, np.ndarray] = {}
        ref_slabs: dict[str, np.ndarray] = {}

        # --- ref (mitgcm) ---
        for var in model_4d:
            if var in ref_arrays:
                slab = ref_arrays[var].astype(np.float32)
                store[f"mitgcm/{var}_spatial"][zarr_t] = slab
                ref_slabs[var] = slab
        if has_ssh and "eta" in ref_arrays:
            slab = ref_arrays["eta"].astype(np.float32)
            store["mitgcm/ssh_spatial"][zarr_t] = slab
            ref_slabs["ssh"] = slab

        # --- model (lazy NC slab) ---
        for var in model_4d:
            slab = model_ds[var].isel(time=model_t).values.astype(np.float32)
            store[f"model/{var}_spatial"][zarr_t] = slab
            model_slabs[var] = slab
        if has_ssh:
            slab = model_ds["eta"].isel(time=model_t).values.astype(np.float32)
            store["model/ssh_spatial"][zarr_t] = slab
            model_slabs["ssh"] = slab

        # --- diff (from this step's local slabs; bit-identical float32 subtract) ---
        for var in model_4d:
            if var in ref_slabs:
                store[f"diff/{var}_spatial"][zarr_t] = model_slabs[var] - ref_slabs[var]
        if has_ssh and "ssh" in ref_slabs:
            store["diff/ssh_spatial"][zarr_t] = model_slabs["ssh"] - ref_slabs["ssh"]

        if (zarr_t + 1) % 4 == 0 or zarr_t == nt - 1:
            print(f"  [zarr] step {zarr_t + 1}/{nt}", flush=True)

    ts_dir = store_path / "timeseries"
    ts_dir.mkdir(exist_ok=True)
    if grid.hfac is not None:
        np.save(ts_dir / "hfac.npy", grid.hfac)

    aligned_model_times = model_time_vals[aligned_model_idx]
    store.attrs["lat"]   = grid.lat.tolist()
    store.attrs["lon"]   = grid.lon.tolist()
    store.attrs["depth"] = grid.depth.tolist()
    store.attrs["time"]  = aligned_model_times.tolist()
    print("  [zarr] Store complete.", flush=True)


def write_zarr_store_model_only(
    store_path: Path | str,
    model_ds: xr.Dataset,
    model_time_vals: np.ndarray,
    grid: StandardGrid,
    *,
    write_success: bool = True,
) -> None:
    """Write a LOCOMgcm-only dashboard Zarr store (no MITgcm reference).

    Mirrors :func:`write_zarr_store` but emits only the ``model`` group — for
    runs that have no stage3 MIT reference (e.g. global free-forecast runs). The
    dashboard backend renders spatial fields
    from ``model/<var>_spatial`` and reads the land mask from
    ``timeseries/hfac.npy``; the ``mitgcm``/``diff`` groups are simply absent and
    the dashboard reference panel shows "not found" instead of crashing.

    ``model_time_vals`` (seconds) becomes the ``time`` attr; all ``time`` frames
    in ``model_ds`` are written (no MIT alignment subsetting).
    """
    store_path = Path(store_path)
    store = zarr.open(str(store_path), mode="w")

    nt = model_ds.sizes["time"]
    ny, nx, nz = grid.ny, grid.nx, grid.nz
    model_4d = [v for v in _4D_VARS if v in model_ds]
    has_ssh = "eta" in model_ds

    _preallocate_spatial(store, "model", model_4d, has_ssh, nt, nz, ny, nx)
    print(f"  [zarr] Streaming {nt} model-only steps ...", flush=True)
    for t in range(nt):
        for var in model_4d:
            store[f"model/{var}_spatial"][t] = model_ds[var].isel(time=t).values.astype(np.float32)
        if has_ssh:
            store["model/ssh_spatial"][t] = model_ds["eta"].isel(time=t).values.astype(np.float32)
        if (t + 1) % 4 == 0 or t == nt - 1:
            print(f"  [zarr] step {t + 1}/{nt}", flush=True)

    ts_dir = store_path / "timeseries"
    ts_dir.mkdir(exist_ok=True)
    if grid.hfac is not None:
        np.save(ts_dir / "hfac.npy", np.asarray(grid.hfac))

    store.attrs["lat"]   = grid.lat.tolist()
    store.attrs["lon"]   = grid.lon.tolist()
    store.attrs["depth"] = grid.depth.tolist()
    store.attrs["time"]  = np.asarray(model_time_vals).tolist()
    if write_success:
        (store_path / "_SUCCESS").touch()
    print("  [zarr] Model-only store complete.", flush=True)

def _preallocate_spatial(
    store: zarr.Group,
    group: str,
    model_4d: list[str],
    has_ssh: bool,
    nt: int, nz: int, ny: int, nx: int,
) -> None:
    grp = store.require_group(group)
    for var in model_4d:
        grp.create_array(f"{var}_spatial", shape=(nt, nz, ny, nx),
                         chunks=(1, 1, ny, nx), dtype=np.float32)
    if has_ssh:
        grp.create_array("ssh_spatial", shape=(nt, ny, nx),
                         chunks=(1, ny, nx), dtype=np.float32)
