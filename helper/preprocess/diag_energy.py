"""
Energy diagnostics — physical-consistency axis.

Domain-integrated energy in TRUE physical units (Joules), using real
finite-volume weighting (hFac·area(m²)·drF(m)) instead of the old normalized
proxy, plus conservation-budget diagnostics (drift / residual / rate / closure).

  - ke           : KE(t)   = ½ρ₀ ∫(u²+v²) dV          (J)
  - ke_by_depth  : KE_z(t) = ½ρ₀ ∫(u²+v²) dV per level (J)
  - ape          : APE(t)  = ½ρ₀ g ∫η² dA              (J)

Output: <store>/diag/energy.json   (ke / ape blocks carry budget_series fields;
ke also keeps `ratio` model/mitgcm and `ke_by_depth`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._budget import budget_series, cell_area_m2, cell_volume_m3, store_time_seconds
from .adapters.base import StandardGrid
from .series import open_series, series_nt

RHO0 = 1025.0   # reference density kg/m³
GRAV = 9.81     # m/s²


def compute_energy(store_path: Path, grid: StandardGrid, metadata: dict | None = None) -> None:
    """Compute energy diagnostics and write diag/energy.json."""
    diag_dir = store_path / "diag"
    diag_dir.mkdir(exist_ok=True)

    area = cell_area_m2(grid)                       # (NY, NX) m²
    vol = cell_volume_m3(grid, metadata)            # (NZ, NY, NX) m³
    wet2d = np.any(np.asarray(vol) > 0.0, axis=0)
    area_wet = area * wet2d

    nt = series_nt(store_path)
    time_s = store_time_seconds(store_path, nt)

    result: dict = {
        "time_index": list(range(nt)),
        "time_seconds": [float(v) for v in time_s.tolist()],
        "units": {"ke": "J", "ape": "J"},
        "weighting": "finite_volume:hfac*area(m^2)*drF(m)",
    }

    # ---- Kinetic energy: ½ρ₀ ∫(u²+v²) dV  (J) ---------------------------
    ke = {}
    ke_by_depth = {}
    for group in ("model", "mitgcm"):
        u_arr = open_series(store_path, group, "u")
        v_arr = open_series(store_path, group, "v")
        if u_arr is None or v_arr is None:
            continue
        n = u_arr.shape[0]
        ke_t = np.full(n, np.nan)
        ke_tz = np.full((n, grid.nz), np.nan)
        for t in range(n):
            u = np.asarray(u_arr[t], dtype=np.float64)
            v = np.asarray(v_arr[t], dtype=np.float64)
            dens_ke = 0.5 * RHO0 * (u ** 2 + v ** 2)   # (NZ,NY,NX) J/m³
            cell_ke = vol * dens_ke                                   # J per cell
            ke_t[t] = float(np.nansum(cell_ke))
            ke_tz[t] = np.nansum(cell_ke, axis=(1, 2))
        ke[group] = ke_t.tolist()
        ke_by_depth[group] = _nan_to_none(ke_tz.tolist())

    result["ke"] = budget_series(ke.get("model"), ke.get("mitgcm"), time_s)
    if "model" in ke and "mitgcm" in ke:
        m, r = np.array(ke["model"]), np.array(ke["mitgcm"])
        result["ke"]["ratio"] = _nan_to_none(np.where(np.abs(r) > 1e-30, m / r, np.nan).tolist())
    if ke_by_depth:
        result["ke_by_depth"] = ke_by_depth

    # ---- Available PE from η: ½ρ₀ g ∫η² dA  (J) -------------------------
    ape = {}
    for group in ("model", "mitgcm"):
        ssh = open_series(store_path, group, "ssh")  # (NT, NY, NX)
        if ssh is None:
            continue
        ape[group] = [float(0.5 * RHO0 * GRAV * np.nansum(
                          area_wet * np.asarray(ssh[t], dtype=np.float64) ** 2))
                      for t in range(ssh.shape[0])]
    result["ape"] = budget_series(ape.get("model"), ape.get("mitgcm"), time_s)

    _write_json(diag_dir / "energy.json", result)
    print("  [diag] energy.json written.", flush=True)


def _nan_to_none(obj):
    if isinstance(obj, float):
        return None if (obj != obj) else obj
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    return obj


def _write_json(path: Path, obj: dict) -> None:
    with path.open("w") as f:
        json.dump(obj, f, separators=(",", ":"), default=_nan_to_none)
