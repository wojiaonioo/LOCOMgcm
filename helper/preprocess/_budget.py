"""Finite-volume weighting + conservation-budget helpers for the physical-axis
diagnostics (diag_conservation / diag_energy).

Replaces the old normalized weighting (w/=w.sum(), dz/=dz.sum()) that made the
reported masses/energies dimensionless proxies. Here cells carry their TRUE
physical size — horizontal area in m² (spherical) and layer thickness drF in m —
so domain integrals come out in real units (m³-weighted): ∫T dV in degC·m³,
KE = ½ρ₀∫(u²+v²)dV in J, etc. On top of the raw series, ``budget_series``
attaches the conservation diagnostics that actually matter: per-model drift
(Q−Q₀), the model−reference residual, a linear drift rate per day, and a
closure flag (relative residual below tolerance).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RE_M = 6_371_000.0  # Earth radius (m), matches diag_transport


def cell_area_m2(grid) -> np.ndarray:
    """(NY, NX) horizontal cell area in m².

    Spherical from the lon/lat spacing (dx = R·cos(lat)·Δlon, dy = R·Δlat),
    mirroring diag_transport. Computed from the grid axes rather than
    grid.cell_area to avoid the all-ones placeholder some adapters emit.
    """
    lat = np.asarray(grid.lat, dtype=np.float64)
    lon = np.asarray(grid.lon, dtype=np.float64)
    dlon = float(np.mean(np.abs(np.diff(lon)))) if lon.size > 1 else 0.25
    dlat = float(np.mean(np.abs(np.diff(lat)))) if lat.size > 1 else 0.25
    dx = RE_M * np.deg2rad(dlon) * np.clip(np.cos(np.deg2rad(lat)), 0.0, None)  # (NY,)
    dy = RE_M * np.deg2rad(dlat)
    return (dx * dy)[:, None] * np.ones((grid.ny, grid.nx), dtype=np.float64)


def layer_thickness_m(grid, metadata: dict | None) -> np.ndarray:
    """(NZ,) layer thickness drF in metres (UNnormalized).

    Prefers metadata.grid.vertical.layer_thicknesses_m; else derives from the
    depth array's cell edges (matches metrics._layer_thickness).
    """
    vertical = ((metadata or {}).get("grid", {}) or {}).get("vertical")
    if vertical and "layer_thicknesses_m" in vertical:
        dz = np.asarray(vertical["layer_thicknesses_m"], dtype=np.float64)
        if dz.ndim == 1 and dz.size == int(grid.nz) and np.isfinite(dz).all() and (dz > 0).all():
            return dz
    depth = np.asarray(getattr(grid, "depth", np.arange(grid.nz)), dtype=np.float64)
    if depth.size <= 1:
        return np.ones(int(grid.nz), dtype=np.float64)
    edges = np.empty(depth.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    edges[0] = max(0.0, depth[0] - (edges[1] - depth[0]))
    edges[-1] = depth[-1] + (depth[-1] - edges[-2])
    return np.maximum(np.diff(edges), 1.0e-12)


def cell_volume_m3(grid, metadata: dict | None) -> np.ndarray:
    """(NZ, NY, NX) wet cell volume in m³ = hFac · area(m²) · drF(m)."""
    hfac = (np.asarray(grid.hfac, dtype=np.float64)
            if getattr(grid, "hfac", None) is not None
            else np.ones((grid.nz, grid.ny, grid.nx), dtype=np.float64))
    return hfac * layer_thickness_m(grid, metadata)[:, None, None] * cell_area_m2(grid)[None, :, :]


def store_time_seconds(store_path: Path, nt: int) -> np.ndarray:
    """Snapshot times in seconds from the zarr store attrs; falls back to index."""
    try:
        meta = json.loads((Path(store_path) / "zarr.json").read_text())
        times = (meta.get("attributes", {}) or {}).get("time")
        if isinstance(times, list) and len(times) >= nt and nt > 0:
            return np.asarray(times[:nt], dtype=np.float64)
    except Exception:
        pass
    return np.arange(max(nt, 0), dtype=np.float64)


def budget_series(model, mitgcm, time_s, *, closure_tol: float = 1.0e-2) -> dict:
    """Raw series + conservation-budget diagnostics for one domain-integrated quantity.

    Returns model / mitgcm / diff series, per-model drift (Q−Q₀), the model−ref
    residual, drift_rate_per_day (linear slope ×86400) for each, the relative
    residual magnitude, and closure_ok (rel_residual < ``closure_tol``).
    """
    out: dict = {}
    m = np.asarray(model, dtype=np.float64) if model is not None else None
    r = np.asarray(mitgcm, dtype=np.float64) if mitgcm is not None else None
    if m is not None:
        out["model"] = _nan_to_none(m.tolist())
    if r is not None:
        out["mitgcm"] = _nan_to_none(r.tolist())
    if m is not None and r is not None and m.shape == r.shape:
        out["diff"] = _nan_to_none((m - r).tolist())

    def _drift(a):
        return (a - a[0]) if (a is not None and a.size) else None

    md, rd = _drift(m), _drift(r)
    if md is not None:
        out["model_drift"] = _nan_to_none(md.tolist())
    if rd is not None:
        out["mitgcm_drift"] = _nan_to_none(rd.tolist())
    res = (md - rd) if (md is not None and rd is not None and md.shape == rd.shape) else None
    if res is not None:
        out["residual"] = _nan_to_none(res.tolist())

    t = np.asarray(time_s, dtype=np.float64)

    def _rate(a):
        if a is None or a.size < 2 or t.size != a.size or not np.isfinite(t).all() or np.ptp(t) <= 0:
            return None
        good = np.isfinite(a)
        if good.sum() < 2:
            return None
        slope = float(np.polyfit(t[good], a[good], 1)[0])  # per second
        return slope * 86400.0 if np.isfinite(slope) else None

    out["drift_rate_per_day"] = {"model": _rate(m), "mitgcm": _rate(r), "residual": _rate(res)}

    if res is not None and m is not None:
        scale = float(np.nanmean(np.abs(m)))
        rel = float(np.nanmax(np.abs(res)) / scale) if scale > 1.0e-30 else None
        out["rel_residual"] = rel
        out["closure_ok"] = bool(rel is not None and np.isfinite(rel) and rel < closure_tol)
    return out


def _nan_to_none(obj):
    if isinstance(obj, float):
        return None if (obj != obj) else obj
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    return obj
