from __future__ import annotations

from typing import Any

import numpy as np

from .sampling import node_id_for_index, selected_indices
from .schema import MetricsContext, load_json, safe_div


def build_energy_conservation(ctx: MetricsContext, sampling: dict[str, Any]) -> dict[str, Any]:
    conservation = load_json(ctx.diag_dir / "conservation.json")
    energy = load_json(ctx.diag_dir / "energy.json")
    if not conservation and not energy:
        return {"availability": "skipped", "skip_reason": "conservation_energy_json_missing"}

    by_node: dict[str, Any] = {}
    for idx in selected_indices(sampling):
        node = node_id_for_index(idx)
        by_node[node] = {
            "locomgcm": _energy_at(energy, "model", idx),
            "mitgcm": _energy_at(energy, "mitgcm", idx),
            "difference": _energy_diff_at(energy, idx),
            "mass": _mass_at(conservation, idx),
            "tracer_inventory": _tracer_at(conservation, idx),
        }
    return {
        "availability": "computed",
        "by_node": by_node,
        "summary": {
            "total_energy_growth_rate_per_day": {
                "locomgcm": _growth_rate(_total_energy_series(energy, "model"), energy.get("time_index", [])),
                "mitgcm": _growth_rate(_total_energy_series(energy, "mitgcm"), energy.get("time_index", [])),
                "diff": _growth_rate(_total_energy_diff_series(energy), energy.get("time_index", [])),
            },
            "mass_drift_rate_m3_per_day": _growth_rate(
                conservation.get("ssh_area_mean", {}).get("diff", []),
                conservation.get("time_index", []),
            ),
            "tracer_closure_max_abs": {
                "temp": _max_abs(conservation.get("temp_mass", {}).get("diff", [])),
                "salt": _max_abs(conservation.get("salt_mass", {}).get("diff", [])),
            },
        },
    }


def _energy_at(energy: dict[str, Any], group: str, idx: int) -> dict[str, Any]:
    ke = _at(energy.get("ke", {}).get(group), idx)
    ape = _at(energy.get("ape", {}).get(group), idx)
    return {"ke": ke, "ape": ape, "free_surface_pe": ape, "total_energy": _sum_optional(ke, ape)}


def _energy_diff_at(energy: dict[str, Any], idx: int) -> dict[str, Any]:
    model = _energy_at(energy, "model", idx)
    ref = _energy_at(energy, "mitgcm", idx)
    return {
        key: _diff_optional(model.get(key), ref.get(key))
        for key in ("ke", "ape", "free_surface_pe", "total_energy")
    }


def _mass_at(conservation: dict[str, Any], idx: int) -> dict[str, Any]:
    ssh = conservation.get("ssh_area_mean", {})
    return {
        "eta_volume_anomaly_m3": _at(ssh.get("model"), idx),
        "eta_volume_diff_m3": _at(ssh.get("diff"), idx),
    }


def _tracer_at(conservation: dict[str, Any], idx: int) -> dict[str, Any]:
    out = {}
    for var in ("temp", "salt"):
        block = conservation.get(f"{var}_mass", {})
        if block:
            out[var] = {
                "model": _at(block.get("model"), idx),
                "mitgcm": _at(block.get("mitgcm"), idx),
                "diff": _at(block.get("diff"), idx),
            }
    return out


def _total_energy_series(energy: dict[str, Any], group: str) -> list[float | None]:
    ke = energy.get("ke", {}).get(group, [])
    ape = energy.get("ape", {}).get(group, [])
    n = max(len(ke), len(ape))
    return [_sum_optional(_at(ke, i), _at(ape, i)) for i in range(n)]


def _total_energy_diff_series(energy: dict[str, Any]) -> list[float | None]:
    model = _total_energy_series(energy, "model")
    ref = _total_energy_series(energy, "mitgcm")
    return [_diff_optional(_at(model, i), _at(ref, i)) for i in range(max(len(model), len(ref)))]


def _growth_rate(series: list[Any], time_index: list[Any]) -> float | None:
    vals = np.asarray([float(v) if v is not None else np.nan for v in series], dtype=np.float64)
    valid = np.isfinite(vals)
    if valid.sum() < 2:
        return None
    # Existing diag time_index is an index, not seconds. Report per index-day proxy.
    x = np.asarray(time_index if len(time_index) == len(vals) else list(range(len(vals))), dtype=np.float64)
    x = x[valid]
    y = vals[valid]
    if np.ptp(x) <= 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _max_abs(series: list[Any]) -> float | None:
    vals = np.asarray([float(v) if v is not None else np.nan for v in series], dtype=np.float64)
    if not np.isfinite(vals).any():
        return None
    return float(np.nanmax(np.abs(vals)))


def _at(series: Any, idx: int) -> Any:
    if not isinstance(series, list) or idx < 0 or idx >= len(series):
        return None
    return series[idx]


def _sum_optional(a: Any, b: Any) -> float | None:
    vals = [v for v in (a, b) if v is not None]
    if not vals:
        return None
    return float(sum(vals))


def _diff_optional(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)

