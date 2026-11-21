"""Categorical verification (CTS line type) — does LOCOMgcm place high-impact
features (strong currents, SSH / tracer extremes) where MITgcm does?

CNT (rmse/corr) answers "how close are the values"; CTS answers "do the two models
agree on WHERE the action is", which is what a forecaster reads off a map. LOCOMgcm is
the forecast, MIT the reference.

Event definition — percentile-adaptive (user-chosen): per snapshot a cell is an
"event" when its transformed field exceeds the q-th percentile (default P90) of
the REFERENCE field over that snapshot's wet cells. The transform isolates the
active signal:
  * velocity (u, v, w): |value|              — strong flow
  * ssh / temp / salt:  |value − wet-mean|   — extremes / anomalies (the large-scale
                                               offset is removed so "event" = local)
The percentile auto-scales across cases, so no per-variable absolute cutoff is
hand-tuned. The threshold itself is an UNWEIGHTED percentile (a distribution
cutoff), but the 2×2 contingency counts (hits / misses / false-alarms /
correct-negatives) are volume-weighted (hFac·cos-lat·drF, the finite-volume weighting) so a
deep partial cell does not count the same as a thin one. ``base_rate`` is
therefore the VOLUME-weighted event fraction (contrast the neighborhood/FSS
module's ``base_rate``, which is the unweighted cell fraction the Roberts-Lean
reference requires).

Note on the anomaly transform: ssh/temp/salt are anomalized against EACH field's
OWN volume-weighted wet-mean (LOCOMgcm minus LOCOMgcm's mean, MIT minus MIT's mean), so a
uniform inter-model offset is by design invisible to CTS/FSS — "where are the
local extremes" is the question here; the absolute mean bias is the CNT (me)
metric's job. This is intentional division of labor, not a blind spot.

Scores: POD, FAR, CSI, HSS, ETS — the numpy 2×2 core is pinned bitwise to
``xskillscore.Contingency`` on small arrays (the library is the oracle; the dense
720²×50×31 reduction runs in-house, matching the CNT-engine philosophy).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._parallel import map_bounded, var_footprint_bytes
from .availability import compared_variables
from .schema import MetricsContext, hfac_mask, load_series, read_store_attrs
from .spatial_decomposition import _environment, _layer_thickness
from .verification import area_hfac_weights

CTS_SCORES = ("pod", "far", "csi", "hss", "ets")

_ANOMALY_VARS = {"ssh", "temp", "salt"}
_DEFAULT_PERCENTILE = 90.0


def build_categorical(
    ctx: MetricsContext,
    variables: dict[str, Any],
    sampling: dict[str, Any],
    *,
    full: bool,
    percentile: float = _DEFAULT_PERCENTILE,
) -> dict[str, Any]:
    """CTS line type: per-variable POD/FAR/CSI/HSS/ETS for the high-percentile event."""
    compared = compared_variables(variables)
    if not compared:
        return {"availability": "skipped", "skip_reason": "no_comparable_variables"}

    env = _environment(ctx)
    drf = _layer_thickness(ctx)
    lat = _grid_lat(ctx)

    # CTS holds no per-cell accumulator (only per-frame transients), so the
    # footprint is small; parallel workers are still memory-budget-bounded.
    footprint = var_footprint_bytes(ctx.grid, full=False)

    def _worker(var):
        return _categorical_variable(ctx, env, var, drf, lat, percentile)

    results = map_bounded(_worker, compared, footprint)
    by_variable = {var: blk for var, blk in zip(compared, results) if blk is not None}

    if not by_variable:
        return {"availability": "skipped", "skip_reason": "categorical_unavailable"}

    return {
        "availability": "computed",
        "line_type": "CTS",
        "engine": "numpy-2x2 (xskillscore.Contingency oracle)",
        "event_definition": "|value| (velocity) or |value - wet-mean| (ssh/tracer) "
                            f"> P{percentile:g} of the reference field per snapshot",
        "percentile": float(percentile),
        "weighting": "finite_volume:hfac*cos(lat)*drF (counts); unweighted percentile (threshold)",
        "diff_convention": "model_is_forecast_mitgcm_is_reference",
        "by_variable": by_variable,
        "aggregate": _aggregate(by_variable),
    }


def _categorical_variable(ctx, env, var, drf, lat, percentile) -> dict[str, Any] | None:
    model = load_series(ctx, "model", var)
    ref = load_series(ctx, "mitgcm", var)
    if model is None or ref is None or model.shape != ref.shape or model.ndim not in (3, 4):
        return None
    is3d = model.ndim == 4
    base_mask = hfac_mask(ctx, var, model.shape[1:])
    weights = area_hfac_weights(env.hfac, lat, is3d=is3d, shape=model.shape, drf=drf)
    w = (np.ones(model.shape[1:], dtype=np.float64) if weights is None
         else np.asarray(weights.values, dtype=np.float64))
    anomaly = var in _ANOMALY_VARS

    H = M = FA = CN = 0.0
    nt = int(model.shape[0])
    for t in range(nt):
        mt = np.asarray(model[t], dtype=np.float64)
        rt = np.asarray(ref[t], dtype=np.float64)
        valid = np.isfinite(mt) & np.isfinite(rt)
        if base_mask is not None:
            valid &= np.asarray(base_mask, dtype=bool)
        if not valid.any():
            continue
        am = _signal(mt, w, valid, anomaly)
        ar = _signal(rt, w, valid, anomaly)
        thr = float(np.percentile(ar[valid], percentile))
        ev_m = (am > thr) & valid
        ev_r = (ar > thr) & valid
        H += float((w * (ev_m & ev_r)).sum())
        M += float((w * (ev_r & ~ev_m)).sum())
        FA += float((w * (ev_m & ~ev_r)).sum())
        CN += float((w * (valid & ~ev_m & ~ev_r)).sum())

    total = H + M + FA + CN
    if total <= 0.0:
        return None
    return {
        "counts": {"hits": H, "misses": M, "false_alarms": FA, "correct_negatives": CN},
        "base_rate": (H + M) / total,
        "scores": scores_2x2(H, M, FA, CN),
    }


def _signal(field: np.ndarray, w: np.ndarray, valid: np.ndarray, anomaly: bool) -> np.ndarray:
    """Transform a field to its event signal: |value| or |value - weighted wet-mean|."""
    if not anomaly:
        return np.abs(field)
    wv = w * valid
    denom = float(wv.sum())
    mean = float((wv * np.where(valid, field, 0.0)).sum() / denom) if denom > 0 else 0.0
    return np.abs(field - mean)


def scores_2x2(hits: float, misses: float, false_alarms: float, correct_negatives: float) -> dict[str, float | None]:
    """POD/FAR/CSI/HSS/ETS from a (weighted) 2×2 contingency table.

    Standard verification formulas (Wilks; matches xskillscore.Contingency):
      POD = H/(H+M), FAR = FA/(H+FA), CSI = H/(H+M+FA),
      HSS = 2(H·CN − M·FA) / [(H+M)(M+CN) + (H+FA)(FA+CN)],
      ETS = (H − H_rand) / (H+M+FA − H_rand),  H_rand = (H+M)(H+FA)/N.
    """
    H, Mi, FAl, CN = float(hits), float(misses), float(false_alarms), float(correct_negatives)
    N = H + Mi + FAl + CN

    def _safe(num, den):
        return float(num / den) if den > 0.0 else None

    pod = _safe(H, H + Mi)
    far = _safe(FAl, H + FAl)
    csi = _safe(H, H + Mi + FAl)
    hss_den = (H + Mi) * (Mi + CN) + (H + FAl) * (FAl + CN)
    hss = _safe(2.0 * (H * CN - Mi * FAl), hss_den)
    h_rand = ((H + Mi) * (H + FAl) / N) if N > 0.0 else 0.0
    ets = _safe(H - h_rand, H + Mi + FAl - h_rand)
    return {"pod": pod, "far": far, "csi": csi, "hss": hss, "ets": ets}


def _aggregate(by_variable: dict[str, Any]) -> dict[str, Any]:
    csis = [b["scores"]["csi"] for b in by_variable.values() if b["scores"].get("csi") is not None]
    pods = [b["scores"]["pod"] for b in by_variable.values() if b["scores"].get("pod") is not None]
    worst = None
    worst_val = np.inf
    for var, b in by_variable.items():
        c = b["scores"].get("csi")
        if c is not None and float(c) < worst_val:
            worst_val = float(c)
            worst = var
    return {
        "median_csi": float(np.median(csis)) if csis else None,
        "median_pod": float(np.median(pods)) if pods else None,
        "worst_csi_var": worst,
        "compared_variable_count": len(by_variable),
    }


def _grid_lat(ctx: MetricsContext) -> np.ndarray | None:
    attrs = read_store_attrs(ctx)
    lat = attrs.get("lat")
    if lat is None:
        lat = getattr(ctx.grid, "lat", None)
    return None if lat is None else np.asarray(lat, dtype=np.float64)
