"""Per-wavenumber spectral-skill helpers for the spectra/transport (tab4)
physical+verification diagnostics.

The horizontal wavenumber spectrum on its own only shows LOCOMgcm and MITgcm energy
side by side; it does not say *which length scales the two models agree on*.
This module turns the spectrum into an Inter-comparison Spectral (ISC) skill: it
bins the 2-D FFT power into the SAME isotropic rings the legacy spectrum uses and,
per ring k, reports

  - model_psd[k] / mitgcm_psd[k] : ring-averaged power. Bit-identical to the legacy
    ``diag_spectra._iso_wavenumber_spectrum`` on every ring EXCEPT the final
    (highest-|k|) ring, which is intentionally inclusive of the K==k_max corner
    cell here (the legacy spectrum's strict ``K < edge`` drops it); so the shared
    wavenumber axis matches but the last point of model_psd may differ slightly
    from temp_wavenumber's last point.
  - coherence[k]    : Pearson_r between the per-cell model and MITgcm power inside
                      the ring — does the *spatial pattern* of power at this scale
                      line up between the two models?
  - rel_error[k]    : RMSE_k / σ_mit_k  (ring RMSE of model−mit power, normalised
                      by the MITgcm power std in the ring)
  - skill[k]        : 1 − rel_error[k]  (1 = perfect, 0 = error == reference
                      variability, <0 = worse than reference variability)

plus the axis-Nyquist wavenumber and a high-wavenumber energy ratio (model/mit
over the resolvable upper-k band [Nyquist/2, Nyquist]) that flags grid noise: a LOCOMgcm
spectrum that is much hotter than MITgcm near the grid scale ⇒ ratio ≫ 1 ⇒ noise.
The per-ring detail arrays run out to the 2-D diagonal corner (≈√2·Nyquist); the
scalar summary (skill_summary) is restricted to the resolvable band k ≤ Nyquist so
sparsely-populated/aliased corner rings do not pose as a physical discrepancy.

These are *pure* functions (numpy in, dict/array out) so both
``diag_spectra.compute_spectra`` (writes diag/spectra.json) and
``metrics/spectral.build_spectral`` (surfaces it in comparison_metrics) reuse
them, mirroring how ``_budget.py`` is shared by the conservation/energy diags.
"""
from __future__ import annotations

import numpy as np

# Cells required in a ring before a Pearson coherence is trustworthy. The
# innermost rings hold only a handful of (kx,ky) cells; correlating 2–3 points
# is noise, so those shells report NaN coherence (the figure shows a gap).
_MIN_RING_CELLS = 4


def _psd2d(field: np.ndarray) -> np.ndarray:
    """fftshifted per-cell power spectral density of a mean-removed 2-D field."""
    ny, nx = field.shape
    f2d = np.fft.fft2(field)
    psd = (np.abs(f2d) ** 2) / (ny * nx)
    return np.fft.fftshift(psd)


def _wavenumber_grid(ny: int, nx: int, dx_km: float) -> np.ndarray:
    """fftshifted isotropic wavenumber magnitude |k| (1/km) for an NY×NX field."""
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_km))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_km))
    KX, KY = np.meshgrid(kx, ky)
    return np.sqrt(KX ** 2 + KY ** 2)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two 1-D arrays; NaN if degenerate."""
    if a.size < _MIN_RING_CELLS:
        return float("nan")
    sa, sb = a.std(), b.std()
    if not (np.isfinite(sa) and np.isfinite(sb)) or sa <= 0.0 or sb <= 0.0:
        return float("nan")
    c = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
    return float(np.clip(c, -1.0, 1.0))


def wavenumber_skill(
    field_model: np.ndarray,
    field_mit: np.ndarray,
    dx_km: float,
) -> dict:
    """Ring-binned spectrum + ISC skill for one model/MITgcm 2-D field pair.

    Both fields must already be mean-removed and NaN-free (the caller does this
    so the FFT is well defined). They must share shape (NY, NX).

    Returns a dict with equal-length lists over the ``nb = min(NY,NX)//2`` rings:
        wavenumber_per_km, model_psd, mitgcm_psd, coherence, rel_error, skill
    plus scalars nyquist_per_km, high_k_energy_ratio, and high_k_band_per_km
    ([lo, hi] = [Nyquist/2, Nyquist], the exact band summed for the ratio).

    Empty dict if the fields are too small to bin (nb == 0).
    """
    if field_model.shape != field_mit.shape:
        raise ValueError(f"shape mismatch {field_model.shape} vs {field_mit.shape}")
    ny, nx = field_model.shape

    pm = _psd2d(field_model)
    pr = _psd2d(field_mit)
    K = _wavenumber_grid(ny, nx, dx_km)

    nb = min(ny, nx) // 2
    if nb == 0:
        return {}
    k_max = float(K.max())
    k_edges = np.linspace(0.0, k_max, nb + 1)
    k_centres = 0.5 * (k_edges[:-1] + k_edges[1:])

    model_psd = np.full(nb, np.nan)
    mit_psd = np.full(nb, np.nan)
    coherence = np.full(nb, np.nan)
    rel_error = np.full(nb, np.nan)

    for i in range(nb):
        # Final ring closes on the upper edge so the highest-|k| (grid-scale)
        # cells are not silently dropped — they carry the grid-noise signal.
        if i == nb - 1:
            mask = (K >= k_edges[i]) & (K <= k_edges[i + 1])
        else:
            mask = (K >= k_edges[i]) & (K < k_edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        pmi = pm[mask]
        pri = pr[mask]
        model_psd[i] = float(np.mean(pmi))
        mit_psd[i] = float(np.mean(pri))
        # Ring power means are always reported; the comparison statistics
        # (coherence + skill) need enough samples to be trustworthy, so both are
        # suppressed together below _MIN_RING_CELLS.
        if n < _MIN_RING_CELLS:
            continue
        coherence[i] = _pearson(pmi, pri)
        rmse = float(np.sqrt(np.mean((pmi - pri) ** 2)))
        sigma = float(np.std(pri))
        rel_error[i] = (rmse / sigma) if sigma > 0.0 else np.nan

    skill = 1.0 - rel_error  # NaN propagates where rel_error is NaN

    nyquist = 1.0 / (2.0 * dx_km) if dx_km > 0.0 else float("nan")

    # High-wavenumber energy ratio over the resolvable upper-k band
    # [½·Nyquist, Nyquist]: model power / mit power. ≫1 ⇒ LOCOMgcm is hotter near the
    # grid scale than MITgcm ⇒ grid-noise signature. The band is closed on both
    # ends so the reported high_k_band_per_km is exactly the set summed (the 2-D
    # diagonal-corner rings above the axis-Nyquist are excluded — aliased).
    lo = 0.5 * nyquist if np.isfinite(nyquist) else float("nan")
    hi = nyquist
    high_ratio = float("nan")
    if np.isfinite(lo):
        band = (k_centres >= lo) & (k_centres <= hi) & np.isfinite(model_psd) & np.isfinite(mit_psd)
        m_sum = float(np.nansum(model_psd[band]))
        r_sum = float(np.nansum(mit_psd[band]))
        if r_sum > 0.0:
            high_ratio = m_sum / r_sum

    return {
        "wavenumber_per_km": k_centres.tolist(),
        "model_psd": _nan_to_none(model_psd.tolist()),
        "mitgcm_psd": _nan_to_none(mit_psd.tolist()),
        "coherence": _nan_to_none(coherence.tolist()),
        "rel_error": _nan_to_none(rel_error.tolist()),
        "skill": _nan_to_none(skill.tolist()),
        "nyquist_per_km": float(nyquist) if np.isfinite(nyquist) else None,
        "high_k_band_per_km": ([float(lo), float(hi)]
                               if np.isfinite(lo) else None),
        "high_k_energy_ratio": high_ratio if np.isfinite(high_ratio) else None,
    }


def skill_summary(coh: dict | None) -> dict:
    """Scalar summary of a ``wavenumber_skill`` block for scorecards / metrics.

    Restricted to the physically-resolvable band k ≤ Nyquist: the per-ring detail
    arrays extend to the 2-D diagonal corner (≈√2·Nyquist), but those rings are
    sparsely populated/aliased, so summarising over them would present a
    discrepancy at a scale finer than the grid as if it were physical. Returns
    median skill / coherence, the wavenumber of lowest skill, the count of valid
    resolvable shells, and the grid-noise ratio — all within k ≤ Nyquist.
    """
    if not coh:
        return {}
    k = np.asarray(coh.get("wavenumber_per_km") or [], dtype=np.float64)
    skill = _to_arr(coh.get("skill"), k.size)
    cohr = _to_arr(coh.get("coherence"), k.size)

    nyq = coh.get("nyquist_per_km")
    resolvable = (k <= float(nyq)) if (nyq is not None and np.isfinite(float(nyq))) else np.ones(k.size, bool)

    valid = np.isfinite(skill) & resolvable
    median_skill = float(np.median(skill[valid])) if valid.any() else None
    worst_k = None
    if valid.any():
        idx = int(np.argmin(np.where(valid, skill, np.inf)))
        worst_k = float(k[idx])

    vc = np.isfinite(cohr) & resolvable
    median_coh = float(np.median(cohr[vc])) if vc.any() else None

    return {
        "median_skill": median_skill,
        "median_coherence": median_coh,
        "worst_skill_wavenumber_per_km": worst_k,
        "n_valid_shells": int(valid.sum()),
        "resolvable_band_only": True,
        "nyquist_per_km": coh.get("nyquist_per_km"),
        "high_k_energy_ratio": coh.get("high_k_energy_ratio"),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _to_arr(seq, n: int) -> np.ndarray:
    """List (with None) → float array with NaN for missing, padded/truncated to n."""
    a = np.full(n, np.nan)
    if not seq:
        return a
    for i, v in enumerate(seq[:n]):
        a[i] = np.nan if v is None else float(v)
    return a


def _nan_to_none(obj):
    if isinstance(obj, float):
        return None if (obj != obj) else obj
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    return obj
