"""Spectral verification metric (METplus ISC line-type analogue).

Surfaces the per-wavenumber Inter-comparison Spectral (ISC) skill that
``diag_spectra`` writes
into the ``spatial_coherence`` block — Pearson coherence, relative spectral error
and skill per isotropic wavenumber shell, plus a scalar summary (median skill,
the scale that drops skill the most, and the high-wavenumber grid-noise ratio).

The raw ``legacy_diag`` payload is retained for back-compat with any consumer
that still reads the SSH-PSD / temperature-wavenumber arrays directly.
"""
from __future__ import annotations

from typing import Any

from .._spectral_skill import skill_summary
from .schema import MetricsContext, load_json


def build_spectral(ctx: MetricsContext, sampling: dict[str, Any]) -> dict[str, Any]:
    path = ctx.diag_dir / "spectra.json"
    raw = load_json(path)
    if not raw:
        return {"availability": "skipped", "skip_reason": "spectra_json_missing_or_unreadable"}

    coherence = raw.get("spatial_coherence") or {}
    summary = skill_summary(coherence) if coherence else {}

    return {
        "availability": "computed",
        "line_type": "ISC",
        "definitions": {
            "method": "per-wavenumber isotropic-shell skill on the 2-D temperature FFT",
            "coherence": "Pearson_r(P_to(k), P_mit(k)) over the (kx,ky) cells in ring k",
            "rel_error": "ring RMSE(P_to - P_mit) / std(P_mit)",
            "skill": "1 - rel_error (1=perfect, 0=error~reference variability, <0=worse)",
            "high_k_energy_ratio": "model/mit power for k >= Nyquist/2 (>>1 flags grid noise)",
            "spatial_nodes": sampling.get("spectral_nodes", []),
            "iw_ny_1_policy": "skipped if key-point or 2d spectrum cannot be computed",
        },
        "spatial_coherence": coherence,
        "summary": summary,
        "legacy_diag": raw,
    }
