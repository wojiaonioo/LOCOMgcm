from __future__ import annotations

from typing import Any

import numpy as np

from ._parallel import map_bounded, var_footprint_bytes
from .availability import compared_variables
from .sampling import node_id_for_index, selected_indices
from .schema import MetricsContext, hfac_mask, load_series, metric_block


def build_vertical_slices(
    ctx: MetricsContext,
    variables: dict[str, Any],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    vars_3d = [
        var for var in compared_variables(variables)
        if (variables.get(var, {}).get("shape") or [0]).__len__() == 4
    ]
    indices = selected_indices(sampling)
    if not vars_3d:
        return {"availability": "skipped", "skip_reason": "no_comparable_3d_variables"}
    if not indices:
        return {"availability": "skipped", "skip_reason": "no_sampled_nodes"}

    def _work(var):
        model = load_series(ctx, "model", var)
        ref = load_series(ctx, "mitgcm", var)
        if model is None or ref is None or model.ndim != 4 or model.shape != ref.shape:
            return None
        by_node: dict[str, Any] = {}
        for idx in indices:
            if idx >= model.shape[0]:
                continue
            m = np.asarray(model[idx], dtype=np.float64)
            r = np.asarray(ref[idx], dtype=np.float64)
            mask = hfac_mask(ctx, var, m.shape)
            by_node[node_id_for_index(idx)] = {
                "surface_layer": metric_block(m[0], r[0], mask=mask[0] if mask is not None else None),
                "bottom_wet_layer": _bottom_wet_metric(m, r, mask),
                "mid_layer": metric_block(
                    m[m.shape[0] // 2],
                    r[r.shape[0] // 2],
                    mask=mask[m.shape[0] // 2] if mask is not None else None,
                ),
                "vertical_profile": _profile(ctx, var, m, r, mask),
            }
        return {"by_node": by_node}

    results = map_bounded(_work, vars_3d, var_footprint_bytes(ctx.grid, full=False))
    by_variable = {var: blk for var, blk in zip(vars_3d, results) if blk is not None}

    if not by_variable:
        return {"availability": "skipped", "skip_reason": "3d_series_unreadable"}
    return {"availability": "computed", "by_variable": by_variable}


def _profile(
    ctx: MetricsContext,
    var: str,
    model: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    depth = np.asarray(getattr(ctx.grid, "depth", np.arange(model.shape[0])), dtype=np.float64)
    rows = []
    for k in range(model.shape[0]):
        rows.append({
            "k": k,
            "depth_m": float(depth[k]) if k < len(depth) else float(k),
            "metrics": metric_block(model[k], ref[k], mask=mask[k] if mask is not None else None),
        })
    return rows


def _bottom_wet_metric(
    model: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray | None,
) -> dict[str, Any]:
    if mask is None:
        return metric_block(model[-1], ref[-1])
    wet = mask.astype(bool)
    nz, ny, nx = wet.shape
    bottom_m = np.full((ny, nx), np.nan, dtype=np.float64)
    bottom_r = np.full((ny, nx), np.nan, dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            ks = np.where(wet[:, j, i])[0]
            if ks.size == 0:
                continue
            k = int(ks[-1])
            bottom_m[j, i] = model[k, j, i]
            bottom_r[j, i] = ref[k, j, i]
    return metric_block(bottom_m, bottom_r)
