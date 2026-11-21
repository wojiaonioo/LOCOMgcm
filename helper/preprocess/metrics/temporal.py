from __future__ import annotations

from typing import Any

import numpy as np

from ._parallel import map_bounded, var_footprint_bytes
from .availability import compared_variables
from .sampling import node_id_for_index
from .schema import MetricsContext, hfac_mask, infer_nt, infer_times, load_series, metric_block, safe_div


def build_temporal_diagnostics(ctx: MetricsContext, variables: dict[str, Any]) -> dict[str, Any]:
    compared = compared_variables(variables)
    nt = infer_nt(ctx)
    if nt <= 0 or not compared:
        return {"availability": "skipped", "skip_reason": "no_comparable_timeseries"}

    times = infer_times(ctx)

    def _work(var):
        series = _rmse_series(ctx, var)
        if not series:
            return None
        rmse = np.array([row["rmse"] if row["rmse"] is not None else np.nan for row in series], dtype=np.float64)
        return {
            "sampled_rmse_series": series,
            "growth_rates": {
                "initial_to_final": safe_div(float(rmse[-1]) if np.isfinite(rmse[-1]) else None,
                                             float(rmse[0]) if np.isfinite(rmse[0]) else None),
                "max_log_rmse_growth_per_day": _max_log_growth(rmse, times),
                "final_window_slope_per_day": _final_slope(rmse, times),
            },
            "top_error_persistence": _top_error_persistence(ctx, var),
        }

    results = map_bounded(_work, compared, var_footprint_bytes(ctx.grid, full=False))
    by_variable = {var: blk for var, blk in zip(compared, results) if blk is not None}
    return {
        "availability": "computed" if by_variable else "skipped",
        "skip_reason": None if by_variable else "rmse_series_empty",
        "phases": _phases(nt),
        "by_variable": by_variable,
    }


def build_mean_fluctuation(ctx: MetricsContext, variables: dict[str, Any]) -> dict[str, Any]:
    compared = compared_variables(variables)
    if not compared:
        return {"availability": "skipped", "skip_reason": "no_comparable_variables"}

    def _work(var):
        model = load_series(ctx, "model", var)
        ref = load_series(ctx, "mitgcm", var)
        if model is None or ref is None or model.shape != ref.shape:
            return None
        m_mean = _time_mean(model)
        r_mean = _time_mean(ref)
        mask = hfac_mask(ctx, var, m_mean.shape)
        if mask is not None:
            m_mean = np.where(mask, m_mean, np.nan)
            r_mean = np.where(mask, r_mean, np.nan)
        mean_metrics = metric_block(m_mean, r_mean, mask=mask)
        anomaly_rmse, variance_ratio = _anomaly_metrics(model, ref, m_mean, r_mean, mask)
        return {
            "time_mean_error": mean_metrics,
            "anomaly_rmse": anomaly_rmse,
            "variance_ratio_model_over_ref": variance_ratio,
        }

    # Two input frames plus running sums/counts/means and anomaly temporaries.
    # Use a conservative ten float64-sized fields per worker.
    cells = int(ctx.grid.nz) * int(ctx.grid.ny) * int(ctx.grid.nx)
    footprint = 10 * cells * 8
    results = map_bounded(_work, compared, footprint)
    out = {var: blk for var, blk in zip(compared, results) if blk is not None}
    return {
        "availability": "computed" if out else "skipped",
        "skip_reason": None if out else "mean_fluctuation_unavailable",
        "by_variable": out,
    }


def build_temporal_frequency(ctx: MetricsContext, variables: dict[str, Any]) -> dict[str, Any]:
    compared = compared_variables(variables)
    nt = infer_nt(ctx)
    if nt < 4 or not compared:
        return {"availability": "skipped", "skip_reason": "need_at_least_4_snapshots"}
    times = np.asarray(infer_times(ctx), dtype=np.float64)
    dt_seconds = float(np.median(np.diff(times))) if times.size > 1 else 1.0

    def _work(var):
        model = load_series(ctx, "model", var)
        ref = load_series(ctx, "mitgcm", var)
        if model is None or ref is None or model.shape != ref.shape:
            return None
        diff_mean = np.array([_domain_mean(np.asarray(model[t] - ref[t], dtype=np.float64)) for t in range(nt)])
        if not np.all(np.isfinite(diff_mean)):
            return None
        demeaned = diff_mean - diff_mean.mean()
        power = np.abs(np.fft.rfft(demeaned)) ** 2
        freqs = np.fft.rfftfreq(nt, d=max(dt_seconds, 1.0))
        if power.size <= 1:
            return None
        nonzero = power[1:]
        top_idx = int(np.argmax(nonzero) + 1)
        split = max(2, power.size // 3)
        return {
            "domain_mean_error_series": diff_mean.tolist(),
            "dominant_frequency_cycles_per_day": float(freqs[top_idx] * 86400.0),
            "dominant_period_days": safe_div(1.0, float(freqs[top_idx] * 86400.0)),
            "low_frequency_error_fraction": safe_div(float(power[:split].sum()), float(power.sum())),
            "high_frequency_error_fraction": safe_div(float(power[split:].sum()), float(power.sum())),
        }

    results = map_bounded(_work, compared, var_footprint_bytes(ctx.grid, full=False))
    by_variable = {var: blk for var, blk in zip(compared, results) if blk is not None}
    return {
        "availability": "computed" if by_variable else "skipped",
        "skip_reason": None if by_variable else "temporal_frequency_unavailable",
        "by_variable": by_variable,
    }


def _rmse_series(ctx: MetricsContext, var: str) -> list[dict[str, Any]]:
    model = load_series(ctx, "model", var)
    ref = load_series(ctx, "mitgcm", var)
    if model is None or ref is None or model.shape != ref.shape:
        return []
    rows = []
    for t in range(model.shape[0]):
        m = np.asarray(model[t], dtype=np.float64)
        r = np.asarray(ref[t], dtype=np.float64)
        metrics = metric_block(m, r, mask=hfac_mask(ctx, var, m.shape))
        rows.append({
            "sample_id": node_id_for_index(t),
            "rmse": metrics.get("rmse"),
            "rel_rmse": metrics.get("rel_rmse"),
            "corr": metrics.get("corr"),
        })
    return rows


def _max_log_growth(rmse: np.ndarray, times: list[float]) -> float | None:
    valid = np.isfinite(rmse) & (rmse > 0)
    if valid.sum() < 2:
        return None
    t_days = np.asarray(times, dtype=np.float64) / 86400.0
    vals = np.log(rmse[valid])
    ts = t_days[valid]
    dt = np.diff(ts)
    dv = np.diff(vals)
    good = dt > 0
    if not good.any():
        return None
    return float(np.max(dv[good] / dt[good]))


def _final_slope(rmse: np.ndarray, times: list[float]) -> float | None:
    valid = np.isfinite(rmse)
    if valid.sum() < 3:
        return None
    idx = np.where(valid)[0][-min(5, valid.sum()):]
    x = np.asarray(times, dtype=np.float64)[idx] / 86400.0
    y = rmse[idx]
    if np.ptp(x) <= 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _top_error_persistence(ctx: MetricsContext, var: str) -> dict[str, Any]:
    model = load_series(ctx, "model", var)
    ref = load_series(ctx, "mitgcm", var)
    if model is None or ref is None or model.shape != ref.shape or model.shape[0] < 3:
        return {"top5_jaccard_initial_to_mid": None, "top5_jaccard_mid_to_final": None}
    mids = [0, model.shape[0] // 2, model.shape[0] - 1]
    masks = [_top_mask(np.asarray(model[i] - ref[i], dtype=np.float64)) for i in mids]
    return {
        "top5_jaccard_initial_to_mid": _jaccard(masks[0], masks[1]),
        "top5_jaccard_mid_to_final": _jaccard(masks[1], masks[2]),
    }


def _top_mask(diff: np.ndarray) -> np.ndarray:
    if diff.ndim == 3:
        finite3d = np.isfinite(diff)
        count = finite3d.sum(axis=0)
        field = np.where(
            count > 0,
            np.nansum(np.where(finite3d, np.abs(diff), 0.0), axis=0) / np.maximum(count, 1),
            np.nan,
        )
    else:
        field = np.abs(diff)
    finite = np.isfinite(field)
    if not finite.any():
        return np.zeros_like(field, dtype=bool)
    threshold = np.quantile(field[finite], 0.95)
    return finite & (field >= threshold)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float | None:
    union = int((a | b).sum())
    if union == 0:
        return None
    return float((a & b).sum() / union)


def _phases(nt: int) -> dict[str, Any]:
    return {
        "spinup": {"start_sample_id": node_id_for_index(0), "end_sample_id": node_id_for_index(max(0, int(0.15 * (nt - 1)))), "rule": "first 15%"},
        "quasi_steady": {"start_sample_id": node_id_for_index(max(0, int(0.5 * (nt - 1)))), "end_sample_id": node_id_for_index(max(0, int(0.85 * (nt - 1)))), "rule": "middle 50-85%"},
        "drift": {"start_sample_id": node_id_for_index(max(0, int(0.85 * (nt - 1)))), "end_sample_id": node_id_for_index(nt - 1), "rule": "last 15%"},
    }


def _anomaly_metrics(
    model: np.ndarray,
    ref: np.ndarray,
    m_mean: np.ndarray,
    r_mean: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[float | None, float | None]:
    sumsq = 0.0
    n = 0
    m_var = 0.0
    r_var = 0.0
    for t in range(model.shape[0]):
        ma = np.asarray(model[t], dtype=np.float64) - m_mean
        ra = np.asarray(ref[t], dtype=np.float64) - r_mean
        valid = np.isfinite(ma) & np.isfinite(ra)
        if mask is not None:
            valid &= mask
        if not valid.any():
            continue
        diff = ma[valid] - ra[valid]
        sumsq += float(np.sum(diff * diff))
        m_var += float(np.sum(ma[valid] * ma[valid]))
        r_var += float(np.sum(ra[valid] * ra[valid]))
        n += int(valid.sum())
    return (
        float(np.sqrt(sumsq / n)) if n > 0 else None,
        safe_div(m_var, r_var),
    )


def _time_mean(series) -> np.ndarray:
    """NaN-aware time mean without materializing a complete Zarr series."""
    total = np.zeros(series.shape[1:], dtype=np.float64)
    count = np.zeros(series.shape[1:], dtype=np.int64)
    for t in range(series.shape[0]):
        frame = np.asarray(series[t], dtype=np.float64)
        finite = np.isfinite(frame)
        total += np.where(finite, frame, 0.0)
        count += finite
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _domain_mean(arr: np.ndarray) -> float:
    return float(np.nanmean(arr))
