"""Neighborhood verification (NBR / FSS line type) — Fractions Skill Score across
spatial scales.

CNT rmse double-penalizes a feature that is right but displaced by a few cells
(wrong at its true location AND at the predicted one). FSS asks instead: at scale
L, do LOCOMgcm and MIT have the same *fraction* of events in each L×L neighborhood? A
displaced feature scores ~0 at the grid scale and recovers as L grows — so the
FSS-vs-scale curve reads off the smallest scale at which LOCOMgcm is skilful.

Per snapshot the binary event uses the same percentile-adaptive cutoff as the
categorical (CTS) module — |value| (velocity) or |value − wet-mean| (ssh/tracer)
above the P-q percentile of the reference field. Each field is anomalized against
ITS OWN wet-mean, so (like CTS) a uniform inter-model offset is deliberately
invisible here and is instead captured by the CNT bias (me) metric. FSS is
evaluated on the SURFACE field (k=0 for 3-D variables) via
``scores.spatial.fss_2d_single_field`` (the library is used directly — a windowed
prefix-sum, not a dense per-metric reduction) and averaged over snapshots.

Land cells are set to a −1 sentinel (non-event in both fields) since the library
takes no mask. This is the standard pragmatic masked-FSS handling: windows
straddling the coast mix guaranteed-non-event land cells into BOTH fractions, a
small CONSERVATIVE bias (verified ~0.016 on a half-land synthetic) that only ever
DEPRESSES FSS near coasts, never inflates skill.

``base_rate`` here is the UNWEIGHTED event cell fraction (NOT volume-weighted like
the CTS base_rate) — the Roberts & Lean (2008) "useful" reference FSS =
0.5 + base_rate/2 is unweighted-cell-count based, matching the scores library, so
the unweighted base rate is the correct pairing. The scale where the mean FSS
first crosses the useful reference is the model's skilful scale.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._parallel import map_bounded, var_footprint_bytes
from .availability import compared_variables
from .categorical import _grid_lat, _signal  # shared event-signal transform
from .schema import MetricsContext, hfac_mask, load_series

_ANOMALY_VARS = {"ssh", "temp", "salt"}
_DEFAULT_PERCENTILE = 90.0
_DEFAULT_SCALES = (1, 3, 5, 11, 21, 41)
_LAND_FILL = -1.0  # signal sentinel below any |·| threshold ⇒ non-event in both fields


def build_neighborhood(
    ctx: MetricsContext,
    variables: dict[str, Any],
    sampling: dict[str, Any],
    *,
    full: bool,
    percentile: float = _DEFAULT_PERCENTILE,
    scales: tuple[int, ...] = _DEFAULT_SCALES,
) -> dict[str, Any]:
    """FSS line type: per-variable FSS-vs-scale curve for the high-percentile event."""
    compared = compared_variables(variables)
    if not compared:
        return {"availability": "skipped", "skip_reason": "no_comparable_variables"}

    try:
        from scores.spatial import fss_2d_single_field
    except Exception as exc:  # scores not installed
        return {"availability": "skipped", "skip_reason": f"scores_unavailable:{exc}"}

    # FSS works on 2-D surface fields → small (surface) footprint per worker.
    footprint = var_footprint_bytes(ctx.grid, full=False, surface=True)

    def _worker(var):
        return _neighborhood_variable(ctx, var, percentile, scales, fss_2d_single_field)

    results = map_bounded(_worker, compared, footprint)
    by_variable = {var: blk for var, blk in zip(compared, results) if blk is not None}

    if not by_variable:
        return {"availability": "skipped", "skip_reason": "neighborhood_unavailable"}

    return {
        "availability": "computed",
        "line_type": "NBR/FSS",
        "engine": "scores.spatial.fss_2d_single_field",
        "event_definition": "|value| (velocity) or |value - wet-mean| (ssh/tracer) "
                            f"> P{percentile:g} of the reference field per snapshot (surface field)",
        "percentile": float(percentile),
        "diff_convention": "model_is_forecast_mitgcm_is_reference",
        "by_variable": by_variable,
        "aggregate": _aggregate(by_variable),
    }


def _neighborhood_variable(ctx, var, percentile, scales, fss_fn) -> dict[str, Any] | None:
    model = load_series(ctx, "model", var)
    ref = load_series(ctx, "mitgcm", var)
    if model is None or ref is None or model.shape != ref.shape or model.ndim not in (3, 4):
        return None
    is3d = model.ndim == 4
    ny, nx = model.shape[-2], model.shape[-1]
    use_scales = [int(s) for s in scales if s <= min(ny, nx)]
    if not use_scales:
        return None

    surf_mask = _surface_wet_mask(ctx, var, model.shape, is3d)
    anomaly = var in _ANOMALY_VARS

    per_scale: dict[int, list[float]] = {s: [] for s in use_scales}
    base_rates: list[float] = []
    nt = int(model.shape[0])
    for t in range(nt):
        mt = np.asarray(model[t, 0] if is3d else model[t], dtype=np.float64)
        rt = np.asarray(ref[t, 0] if is3d else ref[t], dtype=np.float64)
        valid = np.isfinite(mt) & np.isfinite(rt)
        if surf_mask is not None:
            valid &= surf_mask
        if not valid.any():
            continue
        w_unit = valid.astype(np.float64)  # threshold/percentile is unweighted
        am = _signal(mt, w_unit, valid, anomaly)
        ar = _signal(rt, w_unit, valid, anomaly)
        thr = float(np.percentile(ar[valid], percentile))
        # Skip a wet-but-degenerate slice with NO reference event (e.g. a constant
        # field, or >q% of cells tied at the percentile): the FSS library returns
        # 0.0 (not NaN) when both event sums are zero, which would masquerade as
        # zero spatial skill and drag the curve down. Mirror the categorical
        # no-event handling — a "no event" snapshot is undefined, not zero-skill.
        ev_r = (ar > thr) & valid
        if not ev_r.any():
            continue
        base_rates.append(float(ev_r[valid].mean()))   # UNWEIGHTED cell fraction (see below)
        # Land / invalid → below threshold in BOTH fields (the library has no mask).
        am = np.where(valid, am, _LAND_FILL)
        ar = np.where(valid, ar, _LAND_FILL)
        for s in use_scales:
            try:
                fss = float(fss_fn(am, ar, event_threshold=thr, window_size=(s, s)))
            except Exception:
                fss = np.nan
            per_scale[s].append(fss)

    curve = []
    for s in use_scales:
        vals = [v for v in per_scale[s] if np.isfinite(v)]
        curve.append(float(np.mean(vals)) if vals else None)
    base_rate = float(np.mean(base_rates)) if base_rates else None
    useful = (0.5 + base_rate / 2.0) if base_rate is not None else None
    return {
        "scales_cells": use_scales,
        "fss": curve,
        "base_rate": base_rate,
        "fss_useful": useful,
        "skilful_scale_cells": _skilful_scale(use_scales, curve, useful),
    }


def _skilful_scale(scales: list[int], curve: list[float | None], useful: float | None) -> int | None:
    """Smallest scale whose mean FSS reaches the useful reference, else None."""
    if useful is None:
        return None
    for s, v in zip(scales, curve):
        if v is not None and v >= useful:
            return int(s)
    return None


def _surface_wet_mask(ctx, var, shape, is3d) -> np.ndarray | None:
    if is3d:
        mask = hfac_mask(ctx, var, shape[1:])         # 3D wet mask (nz,ny,nx)
        return None if mask is None else np.asarray(mask, dtype=bool)[0]
    return hfac_mask(ctx, var, shape[1:])             # already (ny,nx)


def _aggregate(by_variable: dict[str, Any]) -> dict[str, Any]:
    scales = next((b["scales_cells"] for b in by_variable.values()), [])
    skilful = {v: b.get("skilful_scale_cells") for v, b in by_variable.items()}
    worst = None
    worst_scale = -1
    for var, s in skilful.items():
        eff = s if s is not None else 1 << 30  # never-skilful sorts worst
        if eff > worst_scale:
            worst_scale = eff
            worst = var
    return {
        "scales_cells": scales,
        "skilful_scale_by_var": skilful,
        "worst_var": worst,
        "compared_variable_count": len(by_variable),
    }
