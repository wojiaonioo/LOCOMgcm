from __future__ import annotations

from typing import Any

import numpy as np

from .schema import MetricsContext, load_json


FACES = ("north", "south", "east", "west")


def build_boundary_flux(ctx: MetricsContext) -> dict[str, Any]:
    raw = load_json(ctx.diag_dir / "transport.json")
    if not raw:
        return {"availability": "skipped", "skip_reason": "transport_json_missing"}
    by_side = {}
    for face in FACES:
        block = raw.get(face, {})
        if not block:
            continue
        by_side[face] = {
            "units": raw.get("units", "Sv"),
            "final": {
                "defgcm": _last(block.get("model")),
                "mitgcm": _last(block.get("mitgcm")),
                "diff": _last(block.get("diff")),
            },
            "drift_rate_per_snapshot": {
                "defgcm": _slope(block.get("model")),
                "mitgcm": _slope(block.get("mitgcm")),
                "diff": _slope(block.get("diff")),
            },
            "max_abs_diff": _max_abs(block.get("diff")),
        }
    # LOCOMgcm–MIT temporal correlation per face (carried straight from the diag).
    face_corr = raw.get("face_correlation") or {}
    for face in FACES:
        if face in by_side and face in face_corr:
            by_side[face]["correlation"] = face_corr.get(face)

    return {
        "availability": "computed" if by_side else "skipped",
        "skip_reason": None if by_side else "no_boundary_transport_faces",
        "by_side": by_side,
        "closure": raw.get("closure") or {},
        "net": _net_summary(raw.get("net") or {}),
        "summary": {
            "worst_side_by_abs_final_diff": _worst_side(by_side, "final"),
            "worst_side_by_max_abs_diff": _worst_side(by_side, "max"),
            "min_face_correlation": _min_corr(face_corr),
        },
    }


def _net_summary(net: dict[str, Any]) -> dict[str, Any]:
    """Final outward-positive net flux per group (Sv) from the net series."""
    return {group: _last(net.get(group)) for group in ("model", "mitgcm", "diff")
            if net.get(group) is not None}


def _min_corr(face_corr: dict[str, Any]) -> float | None:
    vals = [float(v) for v in face_corr.values()
            if v is not None and np.isfinite(float(v))]
    return min(vals) if vals else None


def _last(series: Any) -> float | None:
    if not isinstance(series, list) or not series:
        return None
    val = series[-1]
    return float(val) if val is not None else None


def _slope(series: Any) -> float | None:
    if not isinstance(series, list) or len(series) < 2:
        return None
    vals = np.asarray([float(v) if v is not None else np.nan for v in series], dtype=np.float64)
    valid = np.isfinite(vals)
    if valid.sum() < 2:
        return None
    x = np.arange(vals.size, dtype=np.float64)[valid]
    y = vals[valid]
    return float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else None


def _max_abs(series: Any) -> float | None:
    if not isinstance(series, list) or not series:
        return None
    vals = np.asarray([float(v) if v is not None else np.nan for v in series], dtype=np.float64)
    if not np.isfinite(vals).any():
        return None
    return float(np.nanmax(np.abs(vals)))


def _worst_side(by_side: dict[str, Any], mode: str) -> str | None:
    best_side = None
    best_val = -np.inf
    for side, block in by_side.items():
        if mode == "final":
            val = block.get("final", {}).get("diff")
            val = abs(float(val)) if val is not None else None
        else:
            val = block.get("max_abs_diff")
        if val is not None and float(val) > best_val:
            best_val = float(val)
            best_side = side
    return best_side

