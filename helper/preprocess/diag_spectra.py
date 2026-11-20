"""
Spectral diagnostics — .

Computes:
  1. SSH time-series power spectral density (Welch) at a few key grid points
     (domain centre + 4 boundary midpoints) for model and mitgcm.
  2. Horizontal wavenumber spectra of temperature at a chosen depth/time
     (2-D FFT → isotropic average).

Output: <store>/diag/spectra.json
Schema:
{
  "ssh_psd": {
    "points": [{"label": "centre", "lat_idx": 75, "lon_idx": 75}, ...],
    "freqs_per_hour": [...],           # shape (Nf,)
    "model":  [[psd_pt0], [psd_pt1], ...],   # shape (Npoints, Nf)
    "mitgcm": [...]
  },
  "temp_wavenumber": {
    "depth_idx": 0,
    "time_idx":  0,
    "wavenumber_per_km": [...],        # shape (Nk,)
    "model_psd":  [...],
    "mitgcm_psd": [...]
  },
  "spatial_coherence": {               # ISC skill (only when both groups present)
    "depth_idx": 0, "time_idx": 0,
    "wavenumber_per_km": [...],        # shape (Nk,) — shared with temp_wavenumber
    "model_psd": [...], "mitgcm_psd": [...],
    "coherence": [...],                # Pearson_r(P_to, P_mit) per ring
    "rel_error": [...],                # RMSE_k / sigma_mit_k per ring
    "skill":     [...],                # 1 - rel_error per ring
    "nyquist_per_km": <float>,
    "high_k_band_per_km": [lo, hi],    # = [Nyquist/2, Nyquist], band of the ratio
    "high_k_energy_ratio": <float>     # model/mit power in that band (>>1 = grid noise)
  }
}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._spectral_skill import wavenumber_skill
from .adapters.base import StandardGrid
from .series import open_series


def _welch(x: np.ndarray, dt_hours: float = 3.0, nperseg_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Simple Welch PSD; falls back to periodogram for very short series."""
    from scipy.signal import welch as _welch_scipy
    n = len(x)
    nperseg = max(4, int(n * nperseg_frac))
    fs = 1.0 / dt_hours   # cycles / hour
    f, psd = _welch_scipy(x, fs=fs, nperseg=nperseg, detrend="linear")
    return f, psd


def compute_spectra(
    store_path: Path,
    grid: StandardGrid,
    dt_hours: float = 3.0,
    temp_depth_idx: int = 0,
    temp_time_idx: int = -1,
) -> None:
    """Compute spectral diagnostics and write diag/spectra.json.

    Args:
        dt_hours:       Snapshot interval in hours (used as Welch sampling rate).
        temp_depth_idx: Depth level for horizontal wavenumber spectrum.
        temp_time_idx:  Time index for horizontal wavenumber spectrum (-1 = last step).
    """
    diag_dir = store_path / "diag"
    diag_dir.mkdir(exist_ok=True)

    ny, nx = grid.ny, grid.nx

    # ---- Key points for SSH PSD ------------------------------------------
    # Snap cardinal points to nearest wet cell so PSD does not collapse to
    # all-zero on dry land cells (e.g. north→Taiwan, west→Vietnam coast).
    wet_2d = None
    if getattr(grid, "hfac", None) is not None:
        hfac = np.asarray(grid.hfac)
        if hfac.ndim == 3:
            wet_2d = hfac[0] > 0
        elif hfac.ndim == 2:
            wet_2d = hfac > 0
    points = _define_key_points(ny, nx, wet_2d=wet_2d)

    ssh_psd: dict = {"points": points}

    for group in ("model", "mitgcm"):
        ssh = open_series(store_path, group, "ssh")  # (NT, NY, NX)
        if ssh is None:
            continue
        all_psds = []
        freqs = None
        for pt in points:
            ts = np.asarray(ssh[:, pt["lat_idx"], pt["lon_idx"]], dtype=np.float64)
            ts = ts - np.nanmean(ts)   # remove mean
            if np.all(np.isfinite(ts)) and ts.std() > 1e-10:
                f, psd = _welch(ts, dt_hours=dt_hours)
            else:
                # Degenerate series (all-NaN, constant, or dry-land remnant).
                # Match welch output shape so freqs/psd lengths stay consistent.
                nperseg = max(4, int(len(ts) * 0.5))
                f = np.fft.rfftfreq(nperseg, d=dt_hours)
                psd = np.full(f.shape, np.nan, dtype=np.float64)
            if freqs is None:
                freqs = f
            all_psds.append(psd.tolist())
        ssh_psd["freqs_per_hour"] = freqs.tolist() if freqs is not None else []
        ssh_psd[group] = all_psds

    # ---- Horizontal wavenumber spectrum of temperature -------------------
    dx_km = _approx_dx_km(grid)
    t_idx = temp_time_idx

    temp_wn: dict = {
        "depth_idx": temp_depth_idx,
        "time_idx":  t_idx,
    }

    slices: dict[str, np.ndarray] = {}
    for group in ("model", "mitgcm"):
        temp = open_series(store_path, group, "temp")  # (NT, NZ, NY, NX)
        if temp is None:
            continue
        slice_2d = np.asarray(temp[t_idx, temp_depth_idx], dtype=np.float64).copy()
        slice_2d -= np.nanmean(slice_2d)
        slice_2d = np.nan_to_num(slice_2d, nan=0.0)
        slices[group] = slice_2d
        wn, psd = _iso_wavenumber_spectrum(slice_2d, dx_km=dx_km)
        temp_wn["wavenumber_per_km"] = wn.tolist()
        temp_wn[f"{group}_psd"] = psd.tolist()

    # ---- Per-wavenumber ISC skill (needs both model & mitgcm) ------------
    spatial_coherence: dict = {}
    if "model" in slices and "mitgcm" in slices:
        spatial_coherence = wavenumber_skill(slices["model"], slices["mitgcm"], dx_km=dx_km)
        if spatial_coherence:
            spatial_coherence["depth_idx"] = temp_depth_idx
            spatial_coherence["time_idx"] = t_idx

    result = {
        "ssh_psd":          ssh_psd,
        "temp_wavenumber":  temp_wn,
        "spatial_coherence": spatial_coherence,
    }

    _write_json(diag_dir / "spectra.json", result)
    print("  [diag] spectra.json written.", flush=True)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _define_key_points(ny: int, nx: int, wet_2d: np.ndarray | None = None) -> list[dict]:
    """Define 5 SSH PSD points: centre + 4 cardinal directions.

    If ``wet_2d`` (NY×NX bool, True where wet) is provided, snap each cardinal
    point to its nearest wet cell so PSD is not collapsed to all-zero on a
    dry-land grid index.
    """
    cy, cx = ny // 2, nx // 2
    def _clamp(v: int, n: int) -> int:
        return max(0, min(n - 1, v))

    seeds = [
        ("centre",  _clamp(cy, ny),       _clamp(cx, nx)),
        ("north",   _clamp(ny - 3, ny),   _clamp(cx, nx)),
        ("south",   _clamp(2, ny),        _clamp(cx, nx)),
        ("east",    _clamp(cy, ny),       _clamp(nx - 3, nx)),
        ("west",    _clamp(cy, ny),       _clamp(2, nx)),
    ]

    out: list[dict] = []
    for label, j0, i0 in seeds:
        if wet_2d is not None and not bool(wet_2d[j0, i0]):
            j, i = _find_nearest_wet(wet_2d, j0, i0)
        else:
            j, i = j0, i0
        out.append({"label": label, "lat_idx": int(j), "lon_idx": int(i)})
    return out


def _find_nearest_wet(wet_2d: np.ndarray, j0: int, i0: int,
                      max_radius: int = 64) -> tuple[int, int]:
    """Spiral-search outward from (j0, i0) to find the nearest wet cell."""
    ny, nx = wet_2d.shape
    for r in range(1, max_radius + 1):
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if abs(dj) != r and abs(di) != r:
                    continue
                j, i = j0 + dj, i0 + di
                if 0 <= j < ny and 0 <= i < nx and bool(wet_2d[j, i]):
                    return j, i
    return j0, i0


def _approx_dx_km(grid: StandardGrid) -> float:
    """Approximate mean horizontal grid spacing in km."""
    lat_rad = np.deg2rad(np.mean(grid.lat))
    dlon    = float(np.mean(np.diff(grid.lon))) if len(grid.lon) > 1 else 0.25
    dlat    = float(np.mean(np.diff(grid.lat))) if len(grid.lat) > 1 else 0.25
    dx_km   = 111.32 * dlon * np.cos(lat_rad)
    dy_km   = 111.32 * dlat
    return float(0.5 * (dx_km + dy_km))


def _iso_wavenumber_spectrum(field: np.ndarray, dx_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropically-averaged horizontal wavenumber PSD.

    Returns (wavenumber [1/km], power) arrays of length min(NY,NX)//2.
    """
    ny, nx = field.shape
    # 2-D FFT
    f2d = np.fft.fft2(field)
    psd2d = (np.abs(f2d) ** 2) / (ny * nx)

    # Shift so zero-frequency is at centre
    psd2d = np.fft.fftshift(psd2d)

    # Build wavenumber grid
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_km))  # 1/km
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_km))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    # Bin into isotropic rings
    nb = min(ny, nx) // 2
    if nb == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    k_max = float(K.max())
    k_edges = np.linspace(0, k_max, nb + 1)
    k_centres = 0.5 * (k_edges[:-1] + k_edges[1:])
    psd_iso = np.zeros(nb)
    for i in range(nb):
        mask = (K >= k_edges[i]) & (K < k_edges[i + 1])
        if mask.sum() > 0:
            psd_iso[i] = float(np.mean(psd2d[mask]))

    return k_centres, psd_iso


def _write_json(path: Path, obj) -> None:
    with path.open("w") as f:
        json.dump(obj, f, separators=(",", ":"))
