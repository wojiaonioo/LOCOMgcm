"""Diagnostic digest — one-line verdict + dominant error modes.

The verdict and per-variable ranking come from the single volume-weighted CNT
``verification`` group (hFac x cos-lat x drF), not an hfac-masked-but-unweighted
global summary. ``verification`` is computed in
both fast (global + skill curve) and full (+ stratified + bootstrap CI) mode, so
the verdict is the same volume-weighted quantity regardless of metrics depth.
"""
from __future__ import annotations

from typing import Any


def build_ai_digest(
    case: dict[str, Any],
    variables: dict[str, Any],
    verification: dict[str, Any],
    provenance_alignment: dict[str, Any],
) -> dict[str, Any]:
    aggregate = (verification.get("aggregate") or {}) if verification.get("availability") == "computed" else {}
    worst_rmse = aggregate.get("max_rmse_var")
    worst_rel = aggregate.get("max_rel_rmse_var")
    median_rel = aggregate.get("median_rel_rmse")
    verdict = _verdict(median_rel, provenance_alignment)
    compared = variables.get("summary", {}).get("compared", [])
    skipped = variables.get("summary", {}).get("skipped", [])
    provenance_clause = " and provenance mismatches present" if _has_alignment_mismatch(provenance_alignment) else ""
    worst_rmse_value = None
    if worst_rmse:
        worst_rmse_value = (
            verification.get("by_variable", {})
            .get(worst_rmse, {})
            .get("global", {})
            .get("rmse")
        )
    one_line = (
        f"{case.get('case_id', 'unknown')}: {verdict}; "
        f"{len(compared)} variables compared; "
        f"worst RMSE variable={worst_rmse or 'none'}"
        f"{'' if worst_rmse_value is None else f' ({worst_rmse_value:.3g})'}; "
        f"volume-weighted median relative RMSE={_fmt(median_rel)}{provenance_clause}."
    )
    return {
        "one_line_summary": one_line,
        "verdict": verdict,
        "verdict_basis": "volume_weighted_median_rel_rmse (verification; hFac*cos-lat*drF)",
        "dominant_error_modes": _dominant_modes(verification),
        "worst_variable_by_rmse": [worst_rmse] if worst_rmse else [],
        "worst_variable_by_rel_rmse": [worst_rel] if worst_rel else [],
        "worst_environment": _worst_environment(verification),
        "availability_note": {
            "compared": compared,
            "skipped": skipped,
        },
    }


def _verdict(median_rel: Any, provenance_alignment: dict[str, Any]) -> str:
    if _has_alignment_mismatch(provenance_alignment):
        return "warn"
    if median_rel is None:
        return "info"
    if float(median_rel) > 1.0:
        return "fail"
    if float(median_rel) > 0.25:
        return "warn"
    return "pass"


def _has_alignment_mismatch(provenance_alignment: dict[str, Any]) -> bool:
    summary = provenance_alignment.get("summary", {})
    return int(summary.get("mismatch", 0) or 0) > 0


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3g}"


def _dominant_modes(verification: dict[str, Any]) -> list[dict[str, Any]]:
    modes = []
    for rank, (var, g) in enumerate(_vars_by_global(verification, "rmse")[:3], start=1):
        modes.append({
            "rank": rank,
            "label": f"{var} global RMSE",
            "affected_vars": [var],
            "evidence": {"metric": "rmse", "value": g.get("rmse")},
        })
    env = _worst_environment(verification)
    if env:
        modes.append({
            "rank": len(modes) + 1,
            "label": f"largest spatial bin: {env[0]}",
            "affected_vars": [],
            "evidence": {"metric": "stratified_rmse_rank", "value": env[0]},
        })
    return modes


def _vars_by_global(verification: dict[str, Any], metric: str) -> list[tuple[str, dict[str, Any]]]:
    if verification.get("availability") != "computed":
        return []
    rows = []
    for var, block in verification.get("by_variable", {}).items():
        g = block.get("global", {})
        val = g.get(metric)
        if val is not None:
            rows.append((var, g, float(val)))
    rows.sort(key=lambda x: -x[2])
    return [(var, g) for var, g, _ in rows]


def _worst_environment(verification: dict[str, Any]) -> list[str]:
    """Top stratified bins by rmse across variables (verification, full mode).

    Empty in fast mode — the stratified breakdown is only computed for full.
    """
    if verification.get("availability") != "computed":
        return []
    rows = []
    for var, block in verification.get("by_variable", {}).items():
        strata = block.get("stratified") or {}
        for dimension, table in strata.items():
            bins = table.get("bins", [])
            rmses = table.get("rmse", [])
            for i, name in enumerate(bins):
                val = rmses[i] if i < len(rmses) else None
                if val is not None:
                    rows.append((float(val), f"{var}:{dimension}:{name}"))
    rows.sort(reverse=True)
    return [label for _, label in rows[:3]]
