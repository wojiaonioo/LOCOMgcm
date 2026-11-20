from __future__ import annotations

"""Vertical-velocity comparison diagnostics.

The preprocessed comparison series store ``w`` in MITgcm's upward-positive
convention.  LOCOMgcm's internal diagnostic ``w`` is downward-positive, but
the NC snapshot writers negate it at write time (``nc_snapshot.py`` /
``_runtime.py``), so IW/exp4 adapters read NC ``w`` already upward-positive;
MITgcm reference W is native upward-positive.  This module therefore consumes
comparison W directly and does not apply another sign conversion.
"""

from typing import Any

import numpy as np

from ._parallel import map_bounded, var_footprint_bytes
from .sampling import node_id_for_index, selected_indices
from .schema import MetricsContext, hfac_mask, load_series, metric_block


def build_w_diagnostics(
    ctx: MetricsContext,
    variables: dict[str, Any],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    if variables.get("w", {}).get("availability_status") != "compared":
        return {
            "availability": "skipped",
            "skip_reason": variables.get("w", {}).get("skip_reason") or "missing_w",
            "convention": "comparison W is MITgcm upward-positive",
            "summary": {"worst_depth_m": None, "worst_node": None},
        }
    model = load_series(ctx, "model", "w")
    ref = load_series(ctx, "mitgcm", "w")
    if model is None or ref is None or model.shape != ref.shape:
        return {
            "availability": "skipped",
            "skip_reason": "w_series_unreadable",
            "convention": "comparison W is MITgcm upward-positive",
            "summary": {"worst_depth_m": None, "worst_node": None},
        }

    depth = np.asarray(getattr(ctx.grid, "depth", np.arange(model.shape[1])), dtype=np.float64)

    # expects MIT upward-positive comparison W; do not negate here.
    def _work(idx):
        if idx >= model.shape[0]:
            return None
        m = np.asarray(model[idx], dtype=np.float64)
        r = np.asarray(ref[idx], dtype=np.float64)
        mask = hfac_mask(ctx, "w", m.shape)
        block = metric_block(m, r, mask=mask)
        node = {
            "w_metrics": block,
            "vertical_profile": {
                "max_abs_w_by_depth": _max_abs_profile(depth, m, r, mask),
                "rmse_by_depth": _rmse_profile(depth, m, r, mask),
            },
            "continuity": {
                "div_h_transport_rmse": None,
                "dw_dz_plus_div_h_rmse": None,
                "column_integral_residual_rmse": None,
                "corr_w_with_transport_divergence": None,
                "skip_reason": "horizontal transport divergence intermediates unavailable in preprocess store",
            },
        }
        return (node_id_for_index(idx), node, block.get("rmse"))

    results = [x for x in map_bounded(_work, list(selected_indices(sampling)),
                                      var_footprint_bytes(ctx.grid, full=False)) if x is not None]
    by_node = {nid: node for nid, node, _ in results}
    # worst node by rmse — reduced after the parallel map (order-independent max).
    worst_node = {"sample_id": None, "rmse": -np.inf}
    for nid, _node, rmse in results:
        if rmse is not None and float(rmse) > float(worst_node["rmse"]):
            worst_node = {"sample_id": nid, "rmse": float(rmse)}
    summary = _summary(depth, model, ref, hfac_mask(ctx, "w", tuple(model.shape[1:])), worst_node["sample_id"])
    return {
        "availability": "computed",
        "convention": "comparison W is MITgcm upward-positive",
        "by_node": by_node,
        "summary": summary,
    }


def _max_abs_profile(
    depth: np.ndarray,
    model: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows = []
    for k in range(model.shape[0]):
        valid = np.isfinite(model[k]) & np.isfinite(ref[k])
        if mask is not None:
            valid &= mask[k]
        rows.append({
            "depth_m": float(depth[k]) if k < len(depth) else float(k),
            "model": _max_abs(model[k], valid),
            "mitgcm": _max_abs(ref[k], valid),
            "diff": _max_abs(model[k] - ref[k], valid),
        })
    return rows


def _rmse_profile(
    depth: np.ndarray,
    model: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows = []
    for k in range(model.shape[0]):
        metrics = metric_block(model[k], ref[k], mask=mask[k] if mask is not None else None)
        rows.append({
            "depth_m": float(depth[k]) if k < len(depth) else float(k),
            "rmse": metrics.get("rmse"),
        })
    return rows


def _summary(
    depth: np.ndarray,
    model: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray | None,
    worst_node: str | None,
) -> dict[str, Any]:
    nz = int(model.shape[1])
    sumsq = np.zeros(nz, dtype=np.float64)
    count = np.zeros(nz, dtype=np.int64)
    for t in range(model.shape[0]):
        diff = np.asarray(model[t], dtype=np.float64) - np.asarray(ref[t], dtype=np.float64)
        valid = np.isfinite(diff)
        if mask is not None:
            valid &= mask
        sumsq += np.sum(np.where(valid, diff * diff, 0.0), axis=(1, 2))
        count += np.sum(valid, axis=(1, 2))
    arr = np.sqrt(np.divide(sumsq, count, out=np.full(nz, np.nan), where=count > 0))
    if not np.isfinite(arr).any():
        return {"worst_depth_m": None, "worst_node": worst_node}
    k = int(np.nanargmax(arr))
    return {
        "worst_depth_m": float(depth[k]) if k < len(depth) else float(k),
        "worst_node": worst_node,
    }


def _max_abs(arr: np.ndarray, valid: np.ndarray) -> float | None:
    if not valid.any():
        return None
    return float(np.nanmax(np.abs(arr[valid])))
