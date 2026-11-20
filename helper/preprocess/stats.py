"""
Spatial statistics computation for the preprocess pipeline.

Step 4: per-(t, z) metrics written to stats/spatial_stats.json
Step 5: Taylor diagram data written to stats/taylor_data.json

All statistics use finite-volume area weighting: cos(lat) (spherical cell area)
x hFac (partial-cell / land-sea fraction) per (t, z) slice — the same weighting
convention as helper/preprocess/metrics/verification.area_hfac_weights, so the
dashboard's per-slice stats and the comparison-metrics verification group agree.
Land / fully-dry cells (weight 0) are excluded.

spatial_stats.json schema:
{
  "var_name": {
    "time_index": [0, 1, ...],
    "depth_index": [0, 1, ...],
    "bias":      [[t0z0, t0z1, ...], [t1z0, ...]],   # shape (NT, NZ)
    "rmse":      [[...]],
    "corr":      [[...]],
    "std_ratio": [[...]],
    "skill":     [[...]],
    "max_err":   [[...]]
  }
}

taylor_data.json schema:
{
  "var_name": {
    "depth_index": [0, 1, ...],
    "normalized_std": [[d0t0, d0t1, ...], [d1t0, ...]],  # shape (NZ, NT)
    "corr":           [[...]],
  }
}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from helper.preprocess.series import open_series


_VARS_4D = ("temp", "salt", "u", "v", "w")
_VARS_ALL = ("temp", "salt", "u", "v", "w", "ssh")


def compute_stats(store_path: Path, grid_lats: np.ndarray, hfac: np.ndarray | None = None) -> None:
    """Compute spatial_stats.json and taylor_data.json from the store's series.

    Args:
        store_path:  Path to the .zarr store directory.
        grid_lats:   1-D array of latitudes in degrees, shape (NY,).
        hfac:        (NZ, NY, NX) partial-cell fraction for volume weighting.
                     ``None`` (or shape mismatch) falls back to cos-lat only;
                     when absent it is also recovered from the store's
                     ``timeseries/hfac.npy`` auxiliary mask if present.
    """
    ts_dir = store_path / "timeseries"
    stats_dir = store_path / "stats"
    stats_dir.mkdir(exist_ok=True)

    # cos-lat (spherical cell area); |lat|>90 (index/metre axis) -> uniform.
    lat = np.asarray(grid_lats, dtype=np.float64)
    if lat.size and float(np.nanmax(np.abs(lat))) <= 90.0:
        cosw = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)  # (NY,)
    else:
        cosw = np.ones(lat.shape, dtype=np.float64)

    hfac = _resolve_hfac(hfac, ts_dir)  # (NZ, NY, NX) or None
    hfac_surf = hfac.max(axis=0) if hfac is not None else None  # any-wet column

    spatial_stats: dict = {}
    taylor_data: dict = {}

    for var in _VARS_ALL:
        model = open_series(store_path, "model", var)   # (NT, [NZ,] NY, NX)
        ref   = open_series(store_path, "mitgcm", var)
        if model is None or ref is None:
            continue

        print(f"  [stats] Computing {var} ...", flush=True)

        is_3d = model.ndim == 4   # (NT, NZ, NY, NX); False for ssh (NT, NY, NX)
        nt = model.shape[0]
        nz = model.shape[1] if is_3d else 1
        nx = model.shape[-1]
        hfac_ok = hfac is not None and is_3d and hfac.shape == (nz, model.shape[-2], nx)

        bias_arr      = np.full((nt, nz), np.nan)
        rmse_arr      = np.full((nt, nz), np.nan)
        corr_arr      = np.full((nt, nz), np.nan)
        std_ratio_arr = np.full((nt, nz), np.nan)
        skill_arr     = np.full((nt, nz), np.nan)
        max_err_arr   = np.full((nt, nz), np.nan)

        for t in range(nt):
            for k in range(nz):
                if is_3d:
                    m_slice = np.asarray(model[t, k], dtype=np.float64)   # (NY, NX)
                    r_slice = np.asarray(ref[t, k], dtype=np.float64)
                    level_hfac = hfac[k] if hfac_ok else None
                else:
                    m_slice = np.asarray(model[t], dtype=np.float64)      # (NY, NX)
                    r_slice = np.asarray(ref[t], dtype=np.float64)
                    level_hfac = hfac_surf if (hfac_surf is not None and hfac_surf.shape == m_slice.shape) else None

                w2d = cosw[:, np.newaxis] * (level_hfac if level_hfac is not None else 1.0)
                w2d = np.broadcast_to(w2d, m_slice.shape)

                result = _compute_slice_stats(m_slice, r_slice, w2d)
                if result is None:
                    continue
                bias_arr[t, k]      = result["bias"]
                rmse_arr[t, k]      = result["rmse"]
                corr_arr[t, k]      = result["corr"]
                std_ratio_arr[t, k] = result["std_ratio"]
                skill_arr[t, k]     = result["skill"]
                max_err_arr[t, k]   = result["max_err"]

        spatial_stats[var] = {
            "time_index":  list(range(nt)),
            "depth_index": list(range(nz)),
            "bias":      _to_list(bias_arr),
            "rmse":      _to_list(rmse_arr),
            "corr":      _to_list(corr_arr),
            "std_ratio": _to_list(std_ratio_arr),
            "skill":     _to_list(skill_arr),
            "max_err":   _to_list(max_err_arr),
        }

        # Taylor data: per (z, t)
        taylor_data[var] = {
            "depth_index":     list(range(nz)),
            "normalized_std":  _to_list(std_ratio_arr.T),   # (NZ, NT)
            "corr":            _to_list(corr_arr.T),
        }

    _write_json(stats_dir / "spatial_stats.json", spatial_stats)
    _write_json(stats_dir / "taylor_data.json",   taylor_data)
    print("  [stats] spatial_stats.json + taylor_data.json written.", flush=True)


# ---------------------------------------------------------------------------
# Per-slice statistics
# ---------------------------------------------------------------------------

def _compute_slice_stats(
    model: np.ndarray,    # (NY, NX) float64, may contain NaN
    ref: np.ndarray,      # (NY, NX) float64, may contain NaN
    w2d: np.ndarray,      # (NY, NX) cos-lat x hFac finite-volume weight
) -> dict | None:
    """Return dict of stats for one (t, z) slice, or None if no valid data.

    Cells with zero weight (land / fully dry) are excluded along with NaNs.
    """
    diff = model - ref

    valid = (np.isfinite(diff) & np.isfinite(model) & np.isfinite(ref)
             & np.isfinite(w2d) & (w2d > 0.0))
    if valid.sum() < 2:
        return None

    m = model[valid]
    r = ref[valid]
    d = diff[valid]
    w = w2d[valid].astype(np.float64)
    w = w / w.sum()   # normalise

    bias      = float(np.sum(w * d))
    rmse      = float(np.sqrt(np.sum(w * d ** 2)))
    max_err   = float(np.max(np.abs(d)))

    var_m = max(float(np.sum(w * (m - np.sum(w * m)) ** 2)), 0.0)
    var_r = max(float(np.sum(w * (r - np.sum(w * r)) ** 2)), 0.0)
    std_m = float(np.sqrt(var_m))
    std_r = float(np.sqrt(var_r))

    if std_m < 1e-14 or std_r < 1e-14:
        corr      = 0.0
        std_ratio = 0.0 if std_r < 1e-14 else std_m / std_r
    else:
        corr      = float(np.sum(w * (m - np.sum(w * m)) * (r - np.sum(w * r))) / (std_m * std_r))
        std_ratio = std_m / std_r

    # Willmott (1981) skill score
    ref_mean = np.sum(w * r)
    denom = np.sum(w * (np.abs(m - ref_mean) + np.abs(r - ref_mean)) ** 2)
    skill = 1.0 - np.sum(w * d ** 2) / denom if denom > 1e-14 else 1.0

    return {
        "bias":      bias,
        "rmse":      rmse,
        "corr":      float(np.clip(corr, -1, 1)),
        "std_ratio": std_ratio,
        "skill":     float(np.clip(skill, 0, 1)),
        "max_err":   max_err,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_hfac(hfac: np.ndarray | None, ts_dir: Path) -> np.ndarray | None:
    """Use the passed hFac, else recover it from the store's hfac.npy mask."""
    if hfac is not None:
        arr = np.asarray(hfac, dtype=np.float64)
        return arr if arr.ndim == 3 else None
    path = ts_dir / "hfac.npy"
    if path.exists():
        arr = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
        return arr if arr.ndim == 3 else None
    return None


def _to_list(arr: np.ndarray) -> list:
    """Convert float64 ndarray to nested list, replacing nan with None."""
    flat = arr.tolist()
    return _nan_to_none(flat)


def _nan_to_none(obj):
    if isinstance(obj, float):
        return None if (obj != obj) else obj   # nan check
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    return obj


def _write_json(path: Path, obj: dict) -> None:
    with path.open("w") as f:
        json.dump(obj, f, separators=(",", ":"))
