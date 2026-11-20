"""
Conservation diagnostics — physical-consistency axis.

Reads frame-indexable series from a preprocessed zarr store and computes domain-integrated
budgets in TRUE physical units (real finite-volume weighting hFac·area(m²)·drF(m),
not the old normalized proxy), plus the conservation diagnostics that matter:
per-model drift (Q−Q₀), model−reference residual, linear drift rate per day, and
a closure flag.

  - ssh_area_mean : area-weighted mean sea surface height η  (m)
  - temp_mass     : ∫ T dV  (degC·m³)   — extensive, proportional to heat content
  - salt_mass     : ∫ S dV  (psu·m³)    — extensive, proportional to salt content

Output: <store>/diag/conservation.json
Each quantity block carries (via budget_series): model / mitgcm / diff series,
model_drift / mitgcm_drift / residual, drift_rate_per_day{model,mitgcm,residual},
rel_residual, closure_ok.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._budget import budget_series, cell_area_m2, cell_volume_m3, store_time_seconds
from .adapters.base import StandardGrid
from .series import open_series, series_nt


def compute_conservation(store_path: Path, grid: StandardGrid, metadata: dict | None = None) -> None:
    """Compute conservation diagnostics and write diag/conservation.json."""
    diag_dir = store_path / "diag"
    diag_dir.mkdir(exist_ok=True)

    area = cell_area_m2(grid)                       # (NY, NX) m²
    vol = cell_volume_m3(grid, metadata)            # (NZ, NY, NX) m³
    wet2d = np.any(np.asarray(vol) > 0.0, axis=0)   # surface wet mask
    area_wet = area * wet2d                          # zero over land

    nt = series_nt(store_path)
    time_s = store_time_seconds(store_path, nt)

    result: dict = {
        "time_index": list(range(nt)),
        "time_seconds": [float(v) for v in time_s.tolist()],
        "units": {"ssh_area_mean": "m", "temp_mass": "degC m^3", "salt_mass": "psu m^3"},
        "weighting": "finite_volume:hfac*area(m^2)*drF(m)",
    }

    # ---- SSH: area-weighted mean η (m) ----------------------------------
    ssh = {}
    denom = float(np.nansum(area_wet))
    for group in ("model", "mitgcm"):
        arr = open_series(store_path, group, "ssh")  # (NT, NY, NX)
        if arr is None or denom <= 0.0:
            continue
        ssh[group] = [float(np.nansum(area_wet * arr[t]) / denom) for t in range(arr.shape[0])]
    result["ssh_area_mean"] = budget_series(ssh.get("model"), ssh.get("mitgcm"), time_s)

    # ---- Tracer volume integrals ∫field dV (real m³) --------------------
    for var in ("temp", "salt"):
        series = {}
        for group in ("model", "mitgcm"):
            arr = open_series(store_path, group, var)  # (NT, NZ, NY, NX)
            if arr is None:
                continue
            series[group] = [float(np.nansum(vol * arr[t])) for t in range(arr.shape[0])]
        result[f"{var}_mass"] = budget_series(series.get("model"), series.get("mitgcm"), time_s)

    _write_json(diag_dir / "conservation.json", result)
    print("  [diag] conservation.json written.", flush=True)


def _write_json(path: Path, obj: dict) -> None:
    with path.open("w") as f:
        json.dump(obj, f, separators=(",", ":"))
