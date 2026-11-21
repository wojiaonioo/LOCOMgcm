from __future__ import annotations

from typing import Any

import numpy as np

from .availability import compared_variables
from .schema import MetricsContext, infer_nt, infer_times, load_json


MAX_DETAILED_NODES = 30


def build_sampling(ctx: MetricsContext, variables: dict[str, Any]) -> dict[str, Any]:
    nt = infer_nt(ctx)
    times = infer_times(ctx)
    if nt <= 0:
        return {
            "max_nodes": MAX_DETAILED_NODES,
            "available_snapshot_count": 0,
            "selected_nodes": [],
            "spectral_nodes": [],
            "event_score_source": "none",
        }

    scores = _event_scores(ctx, variables, nt)
    reasons: dict[int, set[str]] = {}

    def add(idx: int, reason: str) -> None:
        if 0 <= idx < nt:
            reasons.setdefault(int(idx), set()).add(reason)

    add(0, "first")
    add(nt - 1, "final")
    for idx in _log_indices(nt, count=8):
        add(idx, "log_early")
    for idx in _uniform_indices(nt, count=12):
        add(idx, "uniform")
    for idx in _event_indices(scores, count=8):
        add(idx, "rmse_peak")

    ordered = _cap_indices(reasons, nt)
    nodes = []
    for idx in ordered:
        reason = _primary_reason(reasons[idx])
        nodes.append({
            "sample_id": f"t{idx:04d}",
            "snapshot_index": idx,
            "step": idx,
            "time_seconds": float(times[idx]) if idx < len(times) else float(idx),
            "phase": _phase(idx, nt),
            "selection_reason": reason,
            "all_selection_reasons": sorted(reasons[idx]),
            "event_score": None if scores[idx] is None else float(scores[idx]),
        })

    spectral_nodes = [n["sample_id"] for n in nodes[::max(1, len(nodes) // 6 or 1)]][:6]
    return {
        "max_nodes": MAX_DETAILED_NODES,
        "available_snapshot_count": nt,
        "selected_nodes": nodes,
        "spectral_nodes": spectral_nodes,
        "event_score_source": "stats/spatial_stats.json rmse max over compared variables",
    }


def selected_indices(sampling: dict[str, Any]) -> list[int]:
    return [int(n["snapshot_index"]) for n in sampling.get("selected_nodes", [])]


def node_id_for_index(index: int) -> str:
    return f"t{index:04d}"


def final_or_event_indices(sampling: dict[str, Any]) -> list[int]:
    out = []
    for node in sampling.get("selected_nodes", []):
        reasons = set(node.get("all_selection_reasons") or [node.get("selection_reason")])
        if "final" in reasons or "rmse_peak" in reasons:
            out.append(int(node["snapshot_index"]))
    return out


def _event_scores(ctx: MetricsContext, variables: dict[str, Any], nt: int) -> list[float | None]:
    stats = load_json(ctx.stats_dir / "spatial_stats.json")
    vars_compared = compared_variables(variables)
    scores = np.full(nt, np.nan, dtype=np.float64)
    for var in vars_compared:
        block = stats.get(var)
        if not block or "rmse" not in block:
            continue
        arr = np.asarray(block["rmse"], dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.shape[0] != nt:
            continue
        with np.errstate(all="ignore"):
            var_scores = np.nanmax(arr, axis=1)
        median = np.nanmedian(var_scores)
        scale = median if np.isfinite(median) and median > 0 else 1.0
        scores = np.fmax(scores, var_scores / scale)
    return [None if not np.isfinite(v) else float(v) for v in scores]


def _log_indices(nt: int, count: int) -> list[int]:
    if nt <= 2:
        return []
    upper = max(1, nt - 2)
    vals = np.unique(np.geomspace(1, upper, num=min(count, upper)).astype(int))
    return [int(v) for v in vals if 0 < v < nt - 1]


def _uniform_indices(nt: int, count: int) -> list[int]:
    if nt <= 2:
        return []
    vals = np.linspace(1, nt - 2, num=min(count, max(nt - 2, 1))).round().astype(int)
    return sorted({int(v) for v in vals if 0 < v < nt - 1})


def _event_indices(scores: list[float | None], count: int) -> list[int]:
    finite = [(i, v) for i, v in enumerate(scores) if v is not None]
    finite.sort(key=lambda iv: float(iv[1]), reverse=True)
    return [i for i, _ in finite[:count]]


def _cap_indices(reasons: dict[int, set[str]], nt: int) -> list[int]:
    priority_reason = {"first": 0, "final": 1, "rmse_peak": 2, "uniform": 3, "log_early": 4}
    items = sorted(
        reasons,
        key=lambda idx: (min(priority_reason.get(r, 99) for r in reasons[idx]), idx),
    )
    keep = sorted(items[:MAX_DETAILED_NODES])
    if 0 not in keep and nt > 0:
        keep = [0] + keep[:-1]
    if nt - 1 not in keep and nt > 1:
        keep = sorted(set(keep[:-1] + [nt - 1]))
    return keep[:MAX_DETAILED_NODES]


def _primary_reason(reasons: set[str]) -> str:
    for reason in ("first", "final", "rmse_peak", "log_early", "uniform"):
        if reason in reasons:
            return reason
    return sorted(reasons)[0]


def _phase(idx: int, nt: int) -> str:
    if idx == 0:
        return "initial"
    if idx == nt - 1:
        return "final"
    frac = idx / max(nt - 1, 1)
    if frac < 0.15:
        return "spinup"
    if frac < 0.5:
        return "transition"
    if frac < 0.85:
        return "quasi_steady"
    return "drift"

