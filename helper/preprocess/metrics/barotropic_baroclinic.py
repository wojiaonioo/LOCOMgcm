from __future__ import annotations

from typing import Any

import numpy as np

from .schema import MetricsContext, hfac_mask, infer_nt, load_series, metric_block, safe_div


def build_barotropic_baroclinic(ctx: MetricsContext, variables: dict[str, Any]) -> dict[str, Any]:
    available = [
        var for var in ("u", "v")
        if variables.get(var, {}).get("availability_status") == "compared"
    ]
    if not available:
        return {"availability": "skipped", "skip_reason": "u_v_not_comparable"}

    nt = infer_nt(ctx)
    by_variable: dict[str, Any] = {}
    for var in available:
        model = load_series(ctx, "model", var)
        ref = load_series(ctx, "mitgcm", var)
        if model is None or ref is None or model.ndim != 4 or model.shape != ref.shape:
            continue
        final = nt - 1
        mask = hfac_mask(ctx, var, np.asarray(model[final]).shape)
        m_bt = _depth_mean(np.asarray(model[final], dtype=np.float64), mask)
        r_bt = _depth_mean(np.asarray(ref[final], dtype=np.float64), mask)
        m_bc = np.asarray(model[final], dtype=np.float64) - m_bt[np.newaxis, :, :]
        r_bc = np.asarray(ref[final], dtype=np.float64) - r_bt[np.newaxis, :, :]
        bt_metrics = metric_block(m_bt, r_bt)
        bc_metrics = metric_block(m_bc, r_bc, mask=mask)
        by_variable[var] = {
            "final_barotropic_mode": bt_metrics,
            "final_baroclinic_residual": bc_metrics,
            "baroclinic_to_barotropic_rmse": safe_div(bc_metrics.get("rmse"), bt_metrics.get("rmse")),
        }
    return {
        "availability": "computed" if by_variable else "skipped",
        "skip_reason": None if by_variable else "barotropic_metrics_unavailable",
        "by_variable": by_variable,
    }


def _depth_mean(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.nanmean(arr, axis=0)
    weights = mask.astype(np.float64)
    den = weights.sum(axis=0)
    num = np.nansum(np.where(mask, arr, np.nan) * weights, axis=0)
    return np.where(den > 0, num / np.maximum(den, 1.0e-30), np.nan)

