from __future__ import annotations

from typing import Any

import numpy as np

from .schema import (
    VARIABLE_ORDER,
    VARIABLE_SPECS,
    MetricsContext,
    diff_array_path,
    load_series,
)


def build_variables(ctx: MetricsContext) -> dict[str, Any]:
    """Build the mandatory variable availability manifest.

    A variable is comparable only when model, MITgcm reference, and diff arrays
    all exist for the aligned run and the finite model/ref intersection is
    non-empty.  Reference presence prefers the streaming hints collected by
    ``run_pipeline`` because zarr_writer preallocates reference arrays for
    model variables even when a reference variable never arrived.
    """
    hints = ctx.availability_hints or {}
    ref_steps_hint = hints.get("ref_observed_steps")
    ref_valid_hint = hints.get("ref_valid_counts", {})

    out: dict[str, Any] = {
        "availability_contract": (
            "compared iff model/<var>_spatial, mitgcm/<var>_spatial, and "
            "diff/<var>_spatial exist for aligned steps and finite "
            "model/ref valid_fill_count > 0"
        ),
        "summary": {
            "compared": [],
            "skipped": [],
        },
    }

    for var in VARIABLE_ORDER:
        spec = VARIABLE_SPECS[var]
        model = load_series(ctx, "model", var)
        ref = load_series(ctx, "mitgcm", var)
        model_present = model is not None
        ref_array_present = ref is not None
        if isinstance(ref_steps_hint, dict):
            ref_present = ref_array_present and int(ref_steps_hint.get(var, 0)) > 0
        else:
            ref_present = ref_array_present
        diff_present = diff_array_path(ctx, var).exists() and model_present and ref_present

        shape = _shape(model if model_present else ref)
        finite_counts = {
            "model": _finite_count(ctx, "model", var) if model_present else 0,
            "mitgcm": int(ref_valid_hint.get(var, 0)) if ref_present and var in ref_valid_hint else (
                _finite_count(ctx, "mitgcm", var) if ref_present else 0
            ),
            "intersection": _valid_pair_count(ctx, var) if model_present and ref_present else 0,
        }

        status, reason = _status(model_present, ref_present, diff_present, finite_counts["intersection"])
        entry = {
            **spec,
            "shape": shape,
            "model_present": model_present,
            "ref_present": ref_present,
            "diff_present": diff_present,
            "valid_fill_count": finite_counts["intersection"],
            "finite_counts": finite_counts,
            "availability_status": status,
            "compared": status == "compared",
            "skip_reason": reason,
        }
        out[var] = entry
        out["summary"]["compared" if status == "compared" else "skipped"].append(var)

    return out


def compared_variables(variables: dict[str, Any]) -> list[str]:
    return [
        var for var in VARIABLE_ORDER
        if variables.get(var, {}).get("availability_status") == "compared"
    ]


def wet_bathymetry_stats(bathy: np.ndarray | None, wet2d: np.ndarray | None) -> dict[str, float | None]:
    """Return bathymetry stats over wet ocean cells only."""
    if bathy is None or wet2d is None:
        return {"min_m": None, "max_m": None, "mean_m": None}
    wet = np.asarray(wet2d, dtype=bool)
    if not wet.any():
        return {"min_m": None, "max_m": None, "mean_m": None}
    vals = np.asarray(bathy, dtype=np.float64)[wet]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"min_m": None, "max_m": None, "mean_m": None}
    return {
        "min_m": float(np.nanmin(vals)),
        "max_m": float(np.nanmax(vals)),
        "mean_m": float(np.nanmean(vals)),
    }


def _shape(arr) -> list[int] | None:
    if arr is None:
        return None
    return [int(v) for v in arr.shape]


def _finite_count(ctx: MetricsContext, group: str, var: str) -> int:
    arr = load_series(ctx, group, var)
    if arr is None:
        return 0
    total = 0
    for t in range(arr.shape[0]):
        total += int(np.isfinite(arr[t]).sum())
    return total


def _valid_pair_count(ctx: MetricsContext, var: str) -> int:
    model = load_series(ctx, "model", var)
    ref = load_series(ctx, "mitgcm", var)
    if model is None or ref is None or model.shape != ref.shape:
        return 0
    total = 0
    for t in range(model.shape[0]):
        total += int((np.isfinite(model[t]) & np.isfinite(ref[t])).sum())
    return total


def _status(
    model_present: bool,
    ref_present: bool,
    diff_present: bool,
    valid_fill_count: int,
) -> tuple[str, str | None]:
    if not model_present:
        return "missing_model", "model_series_missing"
    if not ref_present:
        return "missing_reference", "reference_series_missing_or_unobserved"
    if not diff_present:
        return "missing_diff", "diff_array_missing"
    if valid_fill_count <= 0:
        return "no_valid_cells", "finite_model_reference_intersection_empty"
    return "compared", None
