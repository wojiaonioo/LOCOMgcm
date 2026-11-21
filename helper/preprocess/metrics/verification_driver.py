"""Stratified LOCOMgcm-vs-MIT verification driver (CNT line type).

This is the orchestration layer that
drives the :mod:`helper.preprocess.metrics.verification` weighted-moment core
over the pipeline's frame-indexed arrays, with the finite-volume weight field (hFac x
cos-lat x drF) and the spatial strata (depth / coast / OBC distance / slope /
region / hFac class) borrowed from :mod:`spatial_decomposition`. It supersedes
the old ``build_spatial_decomposition`` (O(vars x sampled-nodes x ~33 bins)
Python reductions + per-node quantile sorts — the multi-minute cost on
720x720x50 grids): here each variable's field is read ONCE into per-cell
time-sums, and the global metric, every stratum, and the skill curve all derive
from those sums by re-weighting — no second field pass, no per-bin data copy.
(A benchmark ruled out dense per-metric xskillscore: 40-280 s per
metric per variable at this grid; the moment core does all metrics in ~one pass
and is pinned bitwise to xskillscore on small data.)

Output (``verification`` group in comparison_metrics):

* ``by_variable[var].global``   — CNT over all time+space, volume-weighted, plus
  a temporal-block-bootstrap CI on rmse/me/pearson_r (full mode only).
* ``by_variable[var].time_series`` — per-snapshot global skill curve (always).
* ``by_variable[var].stratified`` — per-dimension per-bin CNT (full mode only).
* ``aggregate`` — median pearson_r / rel_rmse, worst variable.

Weighting is the single finite-volume convention everywhere (see
``area_hfac_weights``); fast mode computes only global + time_series (what the
verdict needs), full mode adds the bootstrap CI + the stratified breakdown.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._parallel import affordable, map_bounded, var_footprint_bytes
from .availability import compared_variables
from .sampling import node_id_for_index
from .schema import (
    MetricsContext,
    hfac_mask,
    infer_times,
    load_series,
    read_store_attrs,
)
from .spatial_decomposition import (
    _bin_definitions,
    _dimension_masks,
    _environment,
    _layer_thickness,
)
from .verification import (
    aggregate_sums,
    area_hfac_weights,
    cnt_accumulate,
    cnt_from_sums,
    global_from_frames,
    series_from_frames,
)

# Temporal block-bootstrap resamples (cheap: operates on the nt per-snapshot
# scalars, not the raw field — so it is feasible at production grid sizes).
_N_BOOT = 400
_CI = 0.95
_BOOT_SEED = 12345


def build_verification(
    ctx: MetricsContext,
    variables: dict[str, Any],
    sampling: dict[str, Any],
    *,
    full: bool,
) -> dict[str, Any]:
    compared = compared_variables(variables)
    if not compared:
        return {"availability": "skipped", "skip_reason": "no_comparable_variables"}

    env = _environment(ctx)
    drf = _layer_thickness(ctx)
    lat = _grid_lat(ctx)
    times = infer_times(ctx)

    # The per-cell accumulator (full mode → stratification) is the memory driver.
    # If even ONE variable's full footprint exceeds the budget, degrade to
    # global-only (no per-cell accumulator) rather than risk OOM on a smaller node
    # or a larger grid — never assume the RAM is there.
    degraded = full and not affordable(var_footprint_bytes(ctx.grid, full=True))
    do_full = full and not degraded
    if degraded:
        print("  [verification] full requested but per-variable footprint exceeds the "
              "memory budget → global-only (no stratified breakdown). Raise "
              "METRICS_MEM_BUDGET_GB / use a bigger node, or run fewer variables.", flush=True)

    footprint = var_footprint_bytes(ctx.grid, full=do_full)

    def _worker(var):
        return _verify_variable(ctx, env, var, drf, lat, times, full=do_full)

    results = map_bounded(_worker, compared, footprint)
    by_variable = {var: blk for var, blk in zip(compared, results) if blk is not None}

    if not by_variable:
        return {"availability": "skipped", "skip_reason": "verification_unavailable"}

    out: dict[str, Any] = {
        "availability": "computed",
        "line_type": "CNT",
        "engine": "xskillscore",
        "weighting": "finite_volume:hfac*cos(lat)*drF",
        "diff_convention": "model_minus_mitgcm",
        "mode": "full" if do_full else "fast",
        "by_variable": by_variable,
        "aggregate": _aggregate(by_variable),
    }
    if degraded:
        out["degraded_reason"] = "memory_budget_below_per_variable_full_footprint"
    if do_full:
        out["bin_definitions"] = _bin_definitions(env)
    return out


def _verify_variable(
    ctx: MetricsContext,
    env,
    var: str,
    drf: np.ndarray,
    lat: np.ndarray | None,
    times: list[float],
    *,
    full: bool,
) -> dict[str, Any] | None:
    model = load_series(ctx, "model", var)   # read frame by frame below
    ref = load_series(ctx, "mitgcm", var)
    if model is None or ref is None or model.shape != ref.shape:
        return None
    if model.ndim not in (3, 4):
        return None
    is3d = model.ndim == 4
    sample_shape = model.shape[1:]
    base_mask = hfac_mask(ctx, var, sample_shape)
    weights = area_hfac_weights(env.hfac, lat, is3d=is3d, shape=model.shape, drf=drf)
    w3 = None if weights is None else np.asarray(weights.values, dtype=np.float64)

    # ONE pass over the field. The per-cell time-sums (`cell`) are needed only to
    # stratify (full mode); the global metric + skill curve come from the cheap
    # per-frame sums, so fast mode skips the per-cell accumulation.
    cell, frames = cnt_accumulate(model, ref, w3, base_mask, want_cell=full, want_series=True)

    gsums = aggregate_sums(cell, w3) if cell is not None else global_from_frames(frames)
    g = cnt_from_sums(**gsums)
    series = series_from_frames(frames, ("rmse", "me", "pearson_r"))
    g["ci"] = _series_ci(series, g) if full else None

    nt = int(model.shape[0])
    time_series = {
        "sample_ids": [node_id_for_index(i) for i in range(nt)],
        "time_seconds": [float(times[i]) if i < len(times) else float(i) for i in range(nt)],
        **series,
    }

    block: dict[str, Any] = {"global": g, "time_series": time_series}
    if full:
        block["stratified"] = _stratified(env, cell, w3, sample_shape)
    return block


def _stratified(env, cell: dict[str, Any], w3: np.ndarray | None, sample_shape: tuple[int, ...]) -> dict[str, Any]:
    """Per-dimension per-bin CNT — re-weights the SAME per-cell sums by each bin
    mask (no extra field pass), so the whole breakdown is a handful of cheap
    spatial reductions."""
    masks = _dimension_masks(env, sample_shape)
    w_field = np.ones(sample_shape, dtype=np.float64) if w3 is None else w3
    w_total = float((w_field * cell["cn"]).sum())

    out: dict[str, Any] = {}
    for dimension, binmasks in masks.items():
        labels: list[str] = []
        rmse: list[float | None] = []
        me: list[float | None] = []
        mae: list[float | None] = []
        pr: list[float | None] = []
        n_valid: list[int] = []
        w_frac: list[float | None] = []
        for label, mask in binmasks.items():
            mask_arr = np.asarray(mask, dtype=bool)
            w_bin = np.where(mask_arr, w_field, 0.0)
            sums = aggregate_sums(cell, w_bin)
            lean = cnt_from_sums(**sums, lean=True)
            labels.append(label)
            rmse.append(lean["rmse"])
            me.append(lean["me"])
            mae.append(lean["mae"])
            pr.append(lean["pearson_r"])
            n_valid.append(sums["n"])
            w_frac.append(float(sums["W"] / w_total) if w_total > 0 else None)
        out[dimension] = {
            "bins": labels,
            "rmse": rmse,
            "me": me,
            "mae": mae,
            "pearson_r": pr,
            "n_valid": n_valid,
            "weight_fraction": w_frac,
        }
    return out


def _series_ci(series: dict[str, list[float | None]], g: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    """Temporal block-bootstrap CI from the per-snapshot skill curves.

    Resamples whole snapshots with replacement (preserving each snapshot's full
    spatial structure) and re-aggregates: rmse from the per-snapshot MSE
    (sqrt of the resampled mean), me/pearson_r from their per-snapshot means.
    Falls back to the global point estimate when < 3 snapshots are available.
    """
    out: dict[str, dict[str, float | None]] = {}
    rmse_s = _finite(series.get("rmse"))
    me_s = _finite(series.get("me"))
    pr_s = _finite(series.get("pearson_r"))
    nt = max(len(rmse_s), len(me_s), len(pr_s))
    if nt < 3:
        return {
            "rmse": _degenerate(g.get("rmse")),
            "me": _degenerate(g.get("me")),
            "pearson_r": _degenerate(g.get("pearson_r")),
        }
    rng = np.random.default_rng(_BOOT_SEED)
    lo_q, hi_q = (1.0 - _CI) / 2.0, 1.0 - (1.0 - _CI) / 2.0

    def boot(vals: list[float], reduce) -> dict[str, float | None]:
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return _degenerate(None)
        idx = rng.integers(0, arr.size, size=(_N_BOOT, arr.size))
        dist = reduce(arr[idx])  # (n_boot,)
        return {
            "lo": float(np.quantile(dist, lo_q)),
            "hi": float(np.quantile(dist, hi_q)),
            "mean": float(dist.mean()),
        }

    out["rmse"] = boot(rmse_s, lambda a: np.sqrt(np.mean(a * a, axis=1)))
    out["me"] = boot(me_s, lambda a: np.mean(a, axis=1))
    out["pearson_r"] = boot(pr_s, lambda a: np.mean(a, axis=1))
    return out


def _aggregate(by_variable: dict[str, Any]) -> dict[str, Any]:
    prs: list[float] = []
    rels: list[float] = []
    rmse_by_var: dict[str, float | None] = {}
    rel_by_var: dict[str, float | None] = {}
    for var, block in by_variable.items():
        g = block.get("global", {})
        if g.get("pearson_r") is not None:
            prs.append(float(g["pearson_r"]))
        if g.get("rel_rmse") is not None:
            rels.append(float(g["rel_rmse"]))
        rmse_by_var[var] = g.get("rmse")
        rel_by_var[var] = g.get("rel_rmse")
    return {
        "median_pearson_r": float(np.median(prs)) if prs else None,
        "median_rel_rmse": float(np.median(rels)) if rels else None,
        "max_rmse_var": _argmax_var(rmse_by_var),
        "max_rel_rmse_var": _argmax_var(rel_by_var),
        "compared_variable_count": len(by_variable),
    }


def _grid_lat(ctx: MetricsContext) -> np.ndarray | None:
    attrs = read_store_attrs(ctx)
    lat = attrs.get("lat")
    if lat is None:
        lat = getattr(ctx.grid, "lat", None)
    if lat is None:
        return None
    return np.asarray(lat, dtype=np.float64)


def _finite(values: list[float | None] | None) -> list[float]:
    if not values:
        return []
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def _degenerate(value: Any) -> dict[str, float | None]:
    v = None if value is None else float(value)
    return {"lo": v, "hi": v, "mean": v}


def _argmax_var(by_var: dict[str, float | None]) -> str | None:
    best_var = None
    best_val = -np.inf
    for var, val in by_var.items():
        if val is not None and float(val) > best_val:
            best_val = float(val)
            best_var = var
    return best_var
