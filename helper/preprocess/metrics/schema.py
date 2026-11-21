from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from helper.preprocess.series import open_series, series_nt


SCHEMA_VERSION = "comparison_metrics.v1"

VARIABLE_ORDER = ("ssh", "temp", "salt", "u", "v", "w")

VARIABLE_SPECS: dict[str, dict[str, Any]] = {
    "ssh": {
        "source_name": "eta",
        "location": "center_2d",
        "units": "m",
        "dims": ["time", "y", "x"],
    },
    "temp": {
        "source_name": "temp",
        "location": "center_3d",
        "units": "degC",
        "dims": ["time", "z", "y", "x"],
    },
    "salt": {
        "source_name": "salt",
        "location": "center_3d",
        "units": "psu",
        "dims": ["time", "z", "y", "x"],
    },
    "u": {
        "source_name": "u",
        "location": "u_face",
        "units": "m s-1",
        "dims": ["time", "z", "y", "x"],
    },
    "v": {
        "source_name": "v",
        "location": "v_face",
        "units": "m s-1",
        "dims": ["time", "z", "y", "x"],
    },
    "w": {
        "source_name": "w",
        "location": "w_interface",
        "units": "m s-1",
        "dims": ["time", "z", "y", "x"],
    },
}

MANDATORY_GROUPS = (
    "case",
    "variables",
    "conventions",
    "ai_digest",
    "quality_control",
    "provenance_alignment",
)

OPTIONAL_GROUPS = (
    "sampling",
    "per_timestep_sampled",
    "vertical_slices",
    "verification",
    "categorical_cts",
    "neighborhood_fss",
    "temporal_diagnostics",
    "mean_fluctuation",
    "barotropic_baroclinic",
    "temporal_frequency",
    "spectral",
    "energy_conservation",
    "boundary_transport_flux_budget",
    "w_diagnostics",
    "sign_pair_structure",
    "operator_residuals",
)


@dataclass
class MetricsContext:
    experiment_dir: Path
    store_path: Path
    grid: Any
    metadata: dict[str, Any]
    availability_hints: dict[str, Any] | None = None
    created_utc: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    @property
    def stats_dir(self) -> Path:
        return self.store_path / "stats"

    @property
    def diag_dir(self) -> Path:
        return self.store_path / "diag"


def unavailable(reason: str) -> dict[str, Any]:
    return {"availability": "skipped", "skip_reason": reason}


def diff_array_path(ctx: MetricsContext, var: str) -> Path:
    return ctx.store_path / "diff" / f"{var}_spatial" / "zarr.json"


def load_series(ctx: MetricsContext, group: str, var: str) -> Any | None:
    return open_series(ctx.store_path, group, var)


def infer_nt(ctx: MetricsContext) -> int:
    return series_nt(ctx.store_path)


def read_store_attrs(ctx: MetricsContext) -> dict[str, Any]:
    try:
        import zarr

        store = zarr.open_group(str(ctx.store_path), mode="r")
        return dict(store.attrs)
    except Exception:
        meta_path = ctx.store_path / "zarr.json"
        if not meta_path.exists():
            return {}
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            return {}
        return dict(meta.get("attributes", {}))


def infer_times(ctx: MetricsContext) -> list[float]:
    attrs = read_store_attrs(ctx)
    times = attrs.get("time")
    if isinstance(times, list):
        return [float(t) for t in times]
    return [float(i) for i in range(infer_nt(ctx))]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(json_sanitize(obj), f, separators=(",", ":"), sort_keys=True)


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(den) <= 1.0e-30:
        return None
    val = float(num) / float(den)
    return val if math.isfinite(val) else None


def hfac_mask(ctx: MetricsContext, var: str, sample_shape: tuple[int, ...]) -> np.ndarray | None:
    hfac = getattr(ctx.grid, "hfac", None)
    if hfac is None:
        return None
    hfac_arr = np.asarray(hfac)
    if len(sample_shape) == 2:
        if hfac_arr.ndim == 3:
            return np.any(hfac_arr > 0.0, axis=0)
        return None
    if len(sample_shape) == 3 and hfac_arr.shape == sample_shape:
        return hfac_arr > 0.0
    return None


def metric_block(
    model: np.ndarray,
    ref: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    total_count: int | None = None,
    error_sumsq_total: float | None = None,
) -> dict[str, Any]:
    """Compute the schema metric block for a model/reference slice."""
    m_arr = np.asarray(model, dtype=np.float64)
    r_arr = np.asarray(ref, dtype=np.float64)
    if m_arr.shape != r_arr.shape:
        return empty_metric_block(total_count=int(np.prod(m_arr.shape)))
    valid = np.isfinite(m_arr) & np.isfinite(r_arr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    denom = int(total_count if total_count is not None else valid.size)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return empty_metric_block(total_count=denom)

    m = m_arr[valid]
    r = r_arr[valid]
    d = m - r
    abs_d = np.abs(d)
    sumsq = float(np.sum(d * d))
    rmse = float(np.sqrt(sumsq / n_valid))
    mae = float(np.mean(abs_d))
    bias = float(np.mean(d))
    ref_rms = float(np.sqrt(np.mean(r * r)))
    model_rms = float(np.sqrt(np.mean(m * m)))
    m_std = float(np.std(m))
    r_std = float(np.std(r))
    if n_valid < 2 or m_std <= 1.0e-14 or r_std <= 1.0e-14:
        corr = None
    else:
        corr = float(np.corrcoef(m, r)[0, 1])
        corr = float(np.clip(corr, -1.0, 1.0))

    block = {
        "n_valid": n_valid,
        "valid_fraction": float(n_valid / denom) if denom > 0 else None,
        "rmse": rmse,
        "mae": mae,
        "max_abs": float(np.max(abs_d)),
        "p95_abs": float(np.quantile(abs_d, 0.95)),
        "p99_abs": float(np.quantile(abs_d, 0.99)),
        "bias": bias,
        "bias_over_mae": safe_div(bias, mae),
        "corr": corr,
        "ref_rms": ref_rms,
        "model_rms": model_rms,
        "rel_rmse": safe_div(rmse, ref_rms),
        "rel_rmse_denominator": "ref_rms" if ref_rms > 1.0e-30 else None,
        "std_ratio": safe_div(m_std, r_std),
    }
    if error_sumsq_total is not None:
        block["error_sumsq_share"] = safe_div(sumsq, error_sumsq_total)
    return block


def empty_metric_block(total_count: int = 0) -> dict[str, Any]:
    return {
        "n_valid": 0,
        "valid_fraction": 0.0 if total_count > 0 else None,
        "rmse": None,
        "mae": None,
        "max_abs": None,
        "p95_abs": None,
        "p99_abs": None,
        "bias": None,
        "bias_over_mae": None,
        "corr": None,
        "ref_rms": None,
        "model_rms": None,
        "rel_rmse": None,
        "rel_rmse_denominator": None,
        "std_ratio": None,
    }
