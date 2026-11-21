from __future__ import annotations

from typing import Any

import numpy as np

from .availability import compared_variables
from .sampling import node_id_for_index, selected_indices
from .schema import MetricsContext, hfac_mask, load_series, safe_div


def build_sign_structure(
    ctx: MetricsContext,
    variables: dict[str, Any],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    compared = compared_variables(variables)
    indices = selected_indices(sampling)
    if not compared or not indices:
        return {"availability": "skipped", "skip_reason": "no_sampled_comparable_variables"}

    by_variable = {}
    projected_abs: dict[int, dict[str, np.ndarray]] = {idx: {} for idx in indices}
    for var in compared:
        model = load_series(ctx, "model", var)
        ref = load_series(ctx, "mitgcm", var)
        if model is None or ref is None or model.shape != ref.shape:
            continue
        by_node = {}
        for idx in indices:
            d = np.asarray(model[idx] - ref[idx], dtype=np.float64)
            mask = hfac_mask(ctx, var, d.shape)
            by_node[node_id_for_index(idx)] = _sign_metrics(d, mask)
            projected_abs[idx][var] = _project_abs(d, mask)
        by_variable[var] = {"by_node": by_node}

    cross_variable = {
        "by_node": {
            node_id_for_index(idx): {
                "abs_error_corr_matrix": _corr_matrix(projected_abs[idx])
            }
            for idx in indices
        }
    }
    return {
        "availability": "computed" if by_variable else "skipped",
        "skip_reason": None if by_variable else "sign_metrics_unavailable",
        "by_variable": by_variable,
        "cross_variable": cross_variable,
    }


def _sign_metrics(diff: np.ndarray, mask: np.ndarray | None) -> dict[str, Any]:
    valid = np.isfinite(diff)
    if mask is not None:
        valid &= mask
    values = diff[valid]
    if values.size == 0:
        return {
            "signed_bias_over_mae": None,
            "positive_error_fraction": None,
            "negative_error_fraction": None,
            "neighbor_same_sign_fraction_x": None,
            "neighbor_opposite_sign_fraction_x": None,
            "neighbor_same_sign_fraction_y": None,
            "dipole_index_near_coast": None,
            "red_blue_lobe_balance": None,
        }
    mae = float(np.mean(np.abs(values)))
    surface = diff[0] if diff.ndim == 3 else diff
    surface_mask = mask[0] if mask is not None and diff.ndim == 3 else mask
    sx_same, sx_opp = _neighbor_sign(surface, surface_mask, axis=1)
    sy_same, sy_opp = _neighbor_sign(surface, surface_mask, axis=0)
    pos = int((values > 0).sum())
    neg = int((values < 0).sum())
    return {
        "signed_bias_over_mae": safe_div(float(np.mean(values)), mae),
        "positive_error_fraction": float(pos / values.size),
        "negative_error_fraction": float(neg / values.size),
        "neighbor_same_sign_fraction_x": sx_same,
        "neighbor_opposite_sign_fraction_x": sx_opp,
        "neighbor_same_sign_fraction_y": sy_same,
        "dipole_index_near_coast": safe_div(sx_opp, sx_same) if sx_opp is not None else None,
        "red_blue_lobe_balance": safe_div(float(pos - neg), float(pos + neg)) if (pos + neg) > 0 else None,
    }


def _neighbor_sign(field: np.ndarray, mask: np.ndarray | None, axis: int) -> tuple[float | None, float | None]:
    if field.ndim != 2:
        return None, None
    if axis == 1:
        a, b = field[:, :-1], field[:, 1:]
        valid = np.isfinite(a) & np.isfinite(b)
        if mask is not None:
            valid &= mask[:, :-1] & mask[:, 1:]
    else:
        a, b = field[:-1, :], field[1:, :]
        valid = np.isfinite(a) & np.isfinite(b)
        if mask is not None:
            valid &= mask[:-1, :] & mask[1:, :]
    if not valid.any():
        return None, None
    signs_a = np.sign(a[valid])
    signs_b = np.sign(b[valid])
    nonzero = (signs_a != 0) & (signs_b != 0)
    if not nonzero.any():
        return None, None
    same = signs_a[nonzero] == signs_b[nonzero]
    opp = signs_a[nonzero] == -signs_b[nonzero]
    return float(same.mean()), float(opp.mean())


def _project_abs(diff: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if diff.ndim == 3:
        arr = np.abs(diff)
        if mask is not None:
            count = mask.sum(axis=0)
            return np.where(
                count > 0,
                np.nansum(np.where(mask, arr, 0.0), axis=0) / np.maximum(count, 1),
                np.nan,
            )
        return np.nanmean(arr, axis=0)
    arr = np.abs(diff)
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    return arr


def _corr_matrix(fields: dict[str, np.ndarray]) -> dict[str, float | None]:
    out = {}
    keys = sorted(fields)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            fa = fields[a]
            fb = fields[b]
            valid = np.isfinite(fa) & np.isfinite(fb)
            if valid.sum() < 4:
                out[f"{a}:{b}"] = None
                continue
            va = fa[valid]
            vb = fb[valid]
            if np.std(va) <= 1.0e-14 or np.std(vb) <= 1.0e-14:
                out[f"{a}:{b}"] = None
            else:
                out[f"{a}:{b}"] = float(np.corrcoef(va, vb)[0, 1])
    return out
