"""Per-timestep sampled diagnostics.

(There is no ``build_global_summary``: the verdict
and per-variable aggregate come from the single volume-weighted CNT
``verification`` group — see ``verification_driver`` + ``ai_digest`` — so a
hand-rolled, hfac-masked-but-unweighted global summary would have no consumer.)

What remains is ``build_per_timestep_sampled``: the sampled-node spatial
fingerprints + per-snapshot metric blocks the dashboard / offline tooling reads
for selected snapshots.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._parallel import map_bounded, var_footprint_bytes
from .availability import compared_variables
from .sampling import node_id_for_index, selected_indices
from .schema import MetricsContext, hfac_mask, load_series, metric_block


def build_per_timestep_sampled(
    ctx: MetricsContext,
    variables: dict[str, Any],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    compared = compared_variables(variables)
    indices = selected_indices(sampling)
    if not compared or not indices:
        return {"availability": "skipped", "skip_reason": "no_sampled_comparable_variables"}

    node_meta = {int(n["snapshot_index"]): n for n in sampling.get("selected_nodes", [])}

    def _work(idx):
        node = {
            "time_seconds": node_meta.get(idx, {}).get("time_seconds"),
            "variables": {},
        }
        for var in compared:
            model = load_series(ctx, "model", var)
            ref = load_series(ctx, "mitgcm", var)
            if model is None or ref is None or idx >= model.shape[0] or idx >= ref.shape[0]:
                continue
            m_slice = np.asarray(model[idx], dtype=np.float64)
            r_slice = np.asarray(ref[idx], dtype=np.float64)
            mask = hfac_mask(ctx, var, m_slice.shape)
            entry = {
                "global": metric_block(m_slice, r_slice, mask=mask),
                "spatial_fingerprint": _fingerprint(m_slice - r_slice, mask=mask),
            }
            if m_slice.ndim == 3 and m_slice.shape[0] > 0:
                entry["surface"] = metric_block(m_slice[0], r_slice[0], mask=mask[0] if mask is not None else None)
                entry["column_weighted"] = metric_block(_column_mean(m_slice, mask), _column_mean(r_slice, mask))
            node["variables"][var] = entry
        return (node_id_for_index(idx), node)

    # Parallelized over the sampled snapshots (nodes); each node reads one frame per
    # variable → small footprint, budget-bounded like the per-variable groups.
    results = map_bounded(_work, indices, var_footprint_bytes(ctx.grid, full=False))
    by_node = {node_id: node for node_id, node in results}
    return {"availability": "computed", "by_node": by_node}


def _column_mean(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if arr.ndim != 3:
        return arr
    if mask is None:
        return np.nanmean(arr, axis=0)
    weights = mask.astype(np.float64)
    den = np.sum(weights, axis=0)
    num = np.nansum(np.where(mask, arr, np.nan) * weights, axis=0)
    return np.where(den > 0, num / np.maximum(den, 1.0e-30), np.nan)


def _fingerprint(diff: np.ndarray, mask: np.ndarray | None) -> dict[str, Any]:
    d = np.asarray(diff, dtype=np.float64)
    if d.ndim == 3:
        if mask is not None:
            count = mask.sum(axis=0)
            d2 = np.where(
                count > 0,
                np.nansum(np.where(mask, np.abs(d), 0.0), axis=0) / np.maximum(count, 1),
                np.nan,
            )
            sign_src = d[mask] if np.any(mask) else np.array([])
        else:
            d2 = np.nanmean(np.abs(d), axis=0)
            sign_src = d[np.isfinite(d)]
    else:
        d2 = np.abs(d)
        if mask is not None:
            d2 = np.where(mask, d2, np.nan)
            sign_src = d[mask] if np.any(mask) else np.array([])
        else:
            sign_src = d[np.isfinite(d)]
    finite = np.isfinite(d2)
    if not finite.any():
        return {
            "abs_error_centroid_ij": None,
            "top5_bbox_ij": None,
            "signed_positive_fraction": None,
            "signed_negative_fraction": None,
        }
    jj, ii = np.indices(d2.shape)
    weights = np.where(finite, d2, 0.0)
    wsum = float(weights.sum())
    if wsum <= 0:
        centroid = None
    else:
        centroid = [float((jj * weights).sum() / wsum), float((ii * weights).sum() / wsum)]
    vals = d2[finite]
    threshold = np.quantile(vals, 0.95) if vals.size else np.inf
    top = finite & (d2 >= threshold)
    if top.any():
        rows, cols = np.where(top)
        bbox = [int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())]
    else:
        bbox = None
    sign_src = sign_src[np.isfinite(sign_src)]
    return {
        "abs_error_centroid_ij": centroid,
        "top5_bbox_ij": bbox,
        "signed_positive_fraction": float((sign_src > 0).mean()) if sign_src.size else None,
        "signed_negative_fraction": float((sign_src < 0).mean()) if sign_src.size else None,
    }
