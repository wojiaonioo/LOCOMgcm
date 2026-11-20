"""
Transport diagnostics — physical-consistency axis (tab4).

Boundary-normal volume transport through all 4 lateral faces, integrated with the
TRUE finite-volume weighting hFac·edge(m)·drF(m) (the old code used dz_m=1.0, so
the reported Sv were a proxy, not real volume flux):
  - North/South: ∫∫ v · hFac · dx · drF   along the E–W edge row
  - East/West:   ∫∫ u · hFac · dy · drF   along the N–S edge column

On top of the per-face series this adds the diagnostics that make transport a real
budget check rather than four disconnected lines:
  - net   : outward-positive net flux  N − S + E − W   (v>0 leaves at north / enters
            at south; u>0 leaves at east / enters at west). For an incompressible
            interior this should be small vs the gross face throughput — a COARSE
            volume-budget consistency check, NOT exact mass conservation (free
            surface + C-grid half-cell offset both leak into it; see caveats below).
  - closure        : per-group max|net| vs gross throughput → rel + closure_ok flag.
  - face_correlation : Pearson_r(LOCOMgcm, MIT) over time per face (is the time evolution
            of the flux tracked, regardless of a constant bias?).
  - rolling_correlation : windowed LOCOMgcm–MIT correlation per face (is a mismatch
            persistent or event-driven?).
  - drift_rate_per_day  : linear Sv/day for model / mitgcm / diff per face.

Caveats (documented, not silently hidden):
  * Faces are taken at the outermost wet row/column of the stored u/v fields; the
    Arakawa C-grid puts u/v a half-cell off the tracer edge, so the absolute Sv
    carry a half-cell offset. Because the SAME indexing is applied to LOCOMgcm and MIT
    the *comparison* (diff, correlation) is consistent; only the absolute closure
    is approximate.
  * The partial-cell weight uses the stored tracer-point hFac (hFacC), not the
    face fractions hFacW (u-faces) / hFacS (v-faces). For partial-cell boundary
    topography this perturbs the absolute Sv at the bottom layer; again it is
    applied identically to LOCOMgcm and MIT, so the comparison is unaffected.
  * dx uses the boundary-row latitude (cos φ); dy is uniform.
  * closure rel = max_t|net| / mean_t(Σ|face|): a conservative single-number flag
    pairing the worst-case net against the mean gross throughput (NOT a like-for-
    like ratio); a transient net spike is judged against the average throughput.

Output: <store>/diag/transport.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._budget import layer_thickness_m, store_time_seconds
from .adapters.base import StandardGrid
from .series import open_series, series_nt

# 1 Sverdrup = 1e6 m³/s
M3_PER_S_TO_SV = 1.0e-6

# Earth radius for arc-length
RE_M = 6_371_000.0

FACES = ("north", "south", "east", "west")


def compute_transport(store_path: Path, grid: StandardGrid, metadata: dict | None = None) -> None:
    """Compute lateral transport diagnostics and write diag/transport.json."""
    diag_dir = store_path / "diag"
    diag_dir.mkdir(exist_ok=True)

    lat = np.asarray(grid.lat, dtype=np.float64)
    lon = np.asarray(grid.lon, dtype=np.float64)
    dlon_deg = float(np.mean(np.abs(np.diff(lon)))) if lon.size > 1 else 0.25
    dlat_deg = float(np.mean(np.abs(np.diff(lat)))) if lat.size > 1 else 0.25

    # Zonal cell width at the actual boundary latitude (cos φ); meridional uniform.
    lat_north = float(lat[-1]) if lat.size else 0.0
    lat_south = float(lat[0]) if lat.size else 0.0
    dx_north_m = RE_M * np.deg2rad(dlon_deg) * max(np.cos(np.deg2rad(lat_north)), 0.0)
    dx_south_m = RE_M * np.deg2rad(dlon_deg) * max(np.cos(np.deg2rad(lat_south)), 0.0)
    dy_m = RE_M * np.deg2rad(dlat_deg)

    drF = layer_thickness_m(grid, metadata)  # (NZ,) real layer thickness (m)
    hfac = (np.asarray(grid.hfac, dtype=np.float64)
            if getattr(grid, "hfac", None) is not None else None)

    series: dict[str, dict[str, np.ndarray]] = {f: {} for f in FACES}

    for group in ("model", "mitgcm"):
        u_arr = open_series(store_path, group, "u")  # (NT, NZ, NY, NX)
        v_arr = open_series(store_path, group, "v")
        if u_arr is None or v_arr is None:
            continue
        nt = u_arr.shape[0]
        hf = hfac if hfac is not None else np.ones(u_arr.shape[1:], dtype=np.float64)

        face_vals = {f: np.zeros(nt) for f in FACES}
        for t in range(nt):
            u = np.asarray(u_arr[t], dtype=np.float64)   # (NZ, NY, NX)
            v = np.asarray(v_arr[t], dtype=np.float64)
            face_vals["north"][t] = _face_transport(v[:, -1, :], hf[:, -1, :], drF, dx_north_m)
            face_vals["south"][t] = _face_transport(v[:,  0, :], hf[:,  0, :], drF, dx_south_m)
            face_vals["east"][t]  = _face_transport(u[:, :, -1], hf[:, :, -1], drF, dy_m)
            face_vals["west"][t]  = _face_transport(u[:, :,  0], hf[:, :,  0], drF, dy_m)
        for f in FACES:
            series[f][group] = face_vals[f] * M3_PER_S_TO_SV  # → Sv

    nt = series_nt(store_path)
    time_s = store_time_seconds(store_path, nt)

    result: dict = {
        "time_index": list(range(nt)),
        "time_seconds": [float(v) for v in time_s.tolist()],
        "units": "Sv (10^6 m^3/s)",
        "weighting": "finite_volume:hfac*edge(m)*drF(m)",
    }

    # ---- Per-face series + diff + drift rate -----------------------------
    drift: dict[str, dict] = {}
    for f in FACES:
        m = series[f].get("model")
        r = series[f].get("mitgcm")
        block: dict = {}
        if m is not None:
            block["model"] = m.tolist()
        if r is not None:
            block["mitgcm"] = r.tolist()
        if m is not None and r is not None and m.shape == r.shape:
            block["diff"] = (m - r).tolist()
        result[f] = block
        drift[f] = {
            "model":  _rate_per_day(m, time_s),
            "mitgcm": _rate_per_day(r, time_s),
            "diff":   _rate_per_day((m - r) if (m is not None and r is not None
                                               and m.shape == r.shape) else None, time_s),
        }
    result["drift_rate_per_day"] = drift

    # ---- Outward-positive net flux N − S + E − W -------------------------
    net: dict = {}
    closure: dict = {}
    for group in ("model", "mitgcm"):
        faces = {f: series[f].get(group) for f in FACES}
        if any(faces[f] is None for f in FACES):
            continue
        net_series = faces["north"] - faces["south"] + faces["east"] - faces["west"]
        net[group] = net_series.tolist()
        gross = float(np.nanmean(sum(np.abs(faces[f]) for f in FACES)))
        max_net = float(np.nanmax(np.abs(net_series))) if net_series.size else np.nan
        rel = (max_net / gross) if gross > 0.0 else None
        closure[group] = {
            "max_abs_net_sv": max_net if np.isfinite(max_net) else None,
            "gross_throughput_sv": gross if np.isfinite(gross) else None,
            "rel": rel,
            "closure_ok": bool(rel is not None and np.isfinite(rel) and rel < CLOSURE_TOL),
        }
    if "model" in net and "mitgcm" in net and len(net["model"]) == len(net["mitgcm"]):
        net["diff"] = [a - b for a, b in zip(net["model"], net["mitgcm"])]
    result["net"] = net
    result["closure"] = closure
    result["closure_tol"] = CLOSURE_TOL

    # ---- LOCOMgcm–MIT temporal correlation per face ----------------------------
    face_corr: dict = {}
    for f in FACES:
        face_corr[f] = _pearson_time(series[f].get("model"), series[f].get("mitgcm"))
    result["face_correlation"] = face_corr

    # ---- Rolling-window LOCOMgcm–MIT correlation per face ----------------------
    result["rolling_correlation"] = _rolling_correlation(series, time_s)

    _write_json(diag_dir / "transport.json", result)
    print("  [diag] transport.json written.", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLOSURE_TOL = 0.1   # max|net| < 10% of gross throughput ⇒ closure_ok


def _face_transport(
    vel: np.ndarray,      # (NZ, N_face) velocity normal to face
    hfac: np.ndarray,     # (NZ, N_face) partial-cell fraction
    drF: np.ndarray,      # (NZ,) layer thickness (m)
    edge_len_m: float,    # horizontal edge length of one face cell (m)
) -> float:
    """Volume transport through one face in m³/s (real finite volume)."""
    return float(np.nansum(vel * hfac * drF[:, None]) * edge_len_m)


def _rate_per_day(series: np.ndarray | None, time_s: np.ndarray) -> float | None:
    """Linear slope per day (×86400) of a Sv series against time in seconds."""
    if series is None or series.size < 2:
        return None
    t = np.asarray(time_s, dtype=np.float64)
    if t.size != series.size or not np.isfinite(t).all() or np.ptp(t) <= 0:
        return None
    good = np.isfinite(series)
    if good.sum() < 2:
        return None
    slope = float(np.polyfit(t[good], series[good], 1)[0])  # Sv per second
    return slope * 86400.0 if np.isfinite(slope) else None


def _pearson_time(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Pearson_r between two time series; None if degenerate."""
    if a is None or b is None or a.shape != b.shape or a.size < 2:
        return None
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 2:
        return None
    aa, bb = a[good], b[good]
    sa, sb = aa.std(), bb.std()
    if sa <= 0.0 or sb <= 0.0:
        return None
    c = float(np.mean((aa - aa.mean()) * (bb - bb.mean())) / (sa * sb))
    return float(np.clip(c, -1.0, 1.0))


def _rolling_correlation(series: dict, time_s: np.ndarray) -> dict:
    """Windowed LOCOMgcm–MIT correlation per face. Window ~ nt/4 (min 3)."""
    nt = int(time_s.size)
    if nt < 4:
        return {"window": None, "hours": []}
    w = max(3, nt // 4)
    if w >= nt:
        return {"window": None, "hours": []}
    starts = list(range(0, nt - w + 1))
    centres_s = [float(np.mean(time_s[s:s + w])) for s in starts]
    out: dict = {"window": w, "hours": [c / 3600.0 for c in centres_s]}
    for f in FACES:
        m = series[f].get("model")
        r = series[f].get("mitgcm")
        if m is None or r is None or m.shape != r.shape:
            out[f] = None
            continue
        out[f] = [_pearson_time(m[s:s + w], r[s:s + w]) for s in starts]
    return out


def _write_json(path: Path, obj) -> None:
    with path.open("w") as f:
        json.dump(obj, f, separators=(",", ":"))
