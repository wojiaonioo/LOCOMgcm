"""Stage5 compare — thin wrapper over the ``preprocessed.zarr`` artifact.

Stage4 (``stage4_postproc.py``) drives ``helper.preprocess.run_pipeline`` and
writes ``<manifest.run_dir>/preprocessed.zarr`` (see
``_common.run_helper_preprocess`` docstring), which carries per-variable
``model/<v>_spatial`` and ``mitgcm/<v>_spatial`` groups. This stage simply

  1. locates that zarr;
  2. for each variable in ``run_cfg.compare.variables`` pulls the spatial
     pair and computes correlation / RMSE / max-abs / mean-abs / count;
  3. writes a report to ``<manifest.run_dir>/comparison_metrics.summary.json``
     using the established output filename
     and records it as an artifact;
  4. derives a PASS/FAIL verdict from ``run_cfg.compare.tolerance``.

Variable name mapping — ``CompareConfig.variables`` uses canonical CF names
(``temperature``, ``salinity``, ``eta``), while the zarr groups produced by
``helper.preprocess.zarr_writer`` use short keys (``temp``, ``salt``, ``ssh``,
``u``, ``v``, ``w``). The mapping below converts before lookup.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr

from defgcm.io.runner.config import CompareConfig, load_run_config
from defgcm.io.runner.layer_d import ArtifactKind, ArtifactRef

from ._common import (
    build_stage_cli_argparser,
    open_manifest_for_stage,
    preprocessed_zarr_is_complete,
)


STAGE_NAME = "stage5_compare"


# Canonical compare name → zarr group key produced by helper.preprocess
# (see helper/preprocess/zarr_writer.py:35,67).
_COMPARE_TO_ZARR_VAR = {
    "u": "u",
    "v": "v",
    "w": "w",
    "temperature": "temp",
    "salinity": "salt",
    "eta": "ssh",
}


@dataclass(frozen=True)
class VariableMetrics:
    name: str
    corr: float
    rmse: float
    max_abs_diff: float
    mean_abs_diff: float
    sample_count: int


def compute_variable_metrics(a: Any, b: Any, name: str) -> VariableMetrics:
    """Compute comparison metrics with memory bounded to one first-axis frame."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for {name}: {a.shape} != {b.shape}")

    n = 0
    mean_a = mean_b = 0.0
    m2_a = m2_b = co_moment = 0.0
    diff_sumsq = diff_abs_sum = 0.0
    max_abs = 0.0
    all_close = True

    # Zarr is chunked along time. Indexing the first axis here keeps h960 peak
    # memory at O(one frame), while the batch-moment merge remains stable for
    # large-offset fields such as temperature and salinity.
    n_frames = int(a.shape[0]) if a.ndim else 1
    for frame_idx in range(n_frames):
        a_frame = a[frame_idx] if a.ndim else a
        b_frame = b[frame_idx] if b.ndim else b
        a_flat = np.asarray(a_frame, dtype="float64").ravel()
        b_flat = np.asarray(b_frame, dtype="float64").ravel()
        mask = np.isfinite(a_flat) & np.isfinite(b_flat)
        a_v = a_flat[mask]
        b_v = b_flat[mask]
        batch_n = int(a_v.size)
        if batch_n == 0:
            continue

        batch_mean_a = float(np.mean(a_v))
        batch_mean_b = float(np.mean(b_v))
        centered_a = a_v - batch_mean_a
        centered_b = b_v - batch_mean_b
        batch_m2_a = float(np.dot(centered_a, centered_a))
        batch_m2_b = float(np.dot(centered_b, centered_b))
        batch_co_moment = float(np.dot(centered_a, centered_b))

        combined_n = n + batch_n
        delta_a = batch_mean_a - mean_a
        delta_b = batch_mean_b - mean_b
        cross_weight = n * batch_n / combined_n
        m2_a += batch_m2_a + delta_a * delta_a * cross_weight
        m2_b += batch_m2_b + delta_b * delta_b * cross_weight
        co_moment += batch_co_moment + delta_a * delta_b * cross_weight
        mean_a += delta_a * batch_n / combined_n
        mean_b += delta_b * batch_n / combined_n
        n = combined_n

        diff = b_v - a_v
        abs_diff = np.abs(diff)
        diff_sumsq += float(np.dot(diff, diff))
        diff_abs_sum += float(np.sum(abs_diff))
        max_abs = max(max_abs, float(np.max(abs_diff)))
        all_close = all_close and bool(np.allclose(a_v, b_v))

    if n == 0:
        return VariableMetrics(name=name, corr=0.0, rmse=0.0, max_abs_diff=0.0, mean_abs_diff=0.0, sample_count=0)

    rmse = float(np.sqrt(diff_sumsq / n))
    mean_abs = diff_abs_sum / n
    std_a = float(np.sqrt(max(m2_a, 0.0) / n))
    std_b = float(np.sqrt(max(m2_b, 0.0) / n))
    if std_a > 1e-15 and std_b > 1e-15:
        corr = float(np.clip(co_moment / np.sqrt(m2_a * m2_b), -1.0, 1.0))
    else:
        corr = 1.0 if all_close else 0.0
    return VariableMetrics(
        name=name, corr=corr, rmse=rmse, max_abs_diff=max_abs,
        mean_abs_diff=mean_abs, sample_count=n,
    )


def compare_preprocessed_zarr(preprocessed_zarr: Path, cfg: CompareConfig) -> dict:
    """Compare ``model/<v>_spatial`` vs ``mitgcm/<v>_spatial`` for each
    variable in ``cfg.variables`` and return a report dict.

    Returns a dict shaped like the previous ``compare_runs`` output so
    downstream tooling does not need to change.
    """
    per_variable: dict[str, dict] = {}
    skipped: list[str] = []

    # Read the model and mitgcm groups via zarr directly. helper.preprocess
    # writes arrays without the `_ARRAY_DIMENSIONS` / `dimension_names`
    # metadata that xarray's strict zarr-v3 backend requires, so use the raw
    # zarr API instead of xarray here.
    root = zarr.open_group(str(preprocessed_zarr), mode="r")
    model_g = root["model"]
    mitgcm_g = root["mitgcm"]
    model_arrays = set(model_g.array_keys())
    mitgcm_arrays = set(mitgcm_g.array_keys())

    for var_name in cfg.variables:
        zarr_key = _COMPARE_TO_ZARR_VAR.get(var_name, var_name)
        # helper.preprocess names them `<short>_spatial`.
        spatial = f"{zarr_key}_spatial"
        if spatial not in model_arrays or spatial not in mitgcm_arrays:
            skipped.append(var_name)
            continue
        m = compute_variable_metrics(
            model_g[spatial],
            mitgcm_g[spatial],
            var_name,
        )
        per_variable[var_name] = {
            "corr": m.corr,
            "rmse": m.rmse,
            "max_abs_diff": m.max_abs_diff,
            "mean_abs_diff": m.mean_abs_diff,
            "sample_count": m.sample_count,
        }

    verdict = "PASS"
    reasons: list[str] = []
    corr_min = cfg.tolerance.get("corr_min", 0.0)
    rmse_max = cfg.tolerance.get("rmse_max", {})
    for var, m in per_variable.items():
        if m["sample_count"] == 0:
            verdict = "FAIL"
            reasons.append(f"{var}: no pairwise finite samples")
        if m["corr"] < corr_min:
            verdict = "FAIL"
            reasons.append(f"{var}: corr {m['corr']:.4f} < min {corr_min}")
        rmse_limit = rmse_max.get(var) if isinstance(rmse_max, dict) else None
        if rmse_limit is not None and m["rmse"] > rmse_limit:
            verdict = "FAIL"
            reasons.append(f"{var}: rmse {m['rmse']:.4e} > max {rmse_limit}")

    if not per_variable:
        # No comparable variables at all → degrade verdict but keep the report.
        verdict = "FAIL"
        reasons.append("no variables in compare.variables had a matching spatial pair")
    elif skipped:
        verdict = "FAIL"
        reasons.append(f"requested variables missing spatial pairs: {', '.join(sorted(skipped))}")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "tolerance": dict(cfg.tolerance),
        "per_variable": per_variable,
        "variables_compared": sorted(per_variable),
        "variables_skipped": sorted(skipped),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_stage_cli_argparser(STAGE_NAME)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger(STAGE_NAME)
    if args.config is None:
        parser.error("--config required")
    run_cfg = load_run_config(args.config, args.overlay, args.sets)
    output_root = args.output or run_cfg.output.root
    manifest = open_manifest_for_stage(args, run_cfg)

    if args.dry_run:
        logger.info(f"[dry-run] {STAGE_NAME}")
        return 0

    preprocessed_zarr = manifest.run_dir / "preprocessed.zarr"
    if not preprocessed_zarr_is_complete(preprocessed_zarr):
        logger.error(f"preprocessed.zarr missing or incomplete: {preprocessed_zarr}")
        manifest.mark_failed(STAGE_NAME, error="preprocessed.zarr missing or incomplete (run stage4 first)")
        return 1

    manifest.mark_running(STAGE_NAME)
    try:
        report = compare_preprocessed_zarr(preprocessed_zarr, run_cfg.compare)
    # Preserve the comparison output filename.
        # Atomic write: a killed write must not leave a truncated JSON that
        # later fails to parse (or, worse, parses partially).
        report_path = manifest.run_dir / "comparison_metrics.summary.json"
        _tmp = report_path.with_suffix(".json.tmp")
        _tmp.write_text(json.dumps(report, indent=2))
        import os as _os
        _os.replace(_tmp, report_path)
        verdict = report["verdict"]
        logger.info(f"verdict: {verdict}")
        if report["variables_skipped"]:
            logger.info(f"variables skipped (no spatial pair in zarr): {report['variables_skipped']}")
        if report["reasons"]:
            for r in report["reasons"]:
                logger.info(f"  - {r}")
        artifact = ArtifactRef(
            name="comparison_metrics_summary",
            path=report_path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash="00000000",
            schema_version="0.1.0",
            extras={"verdict": verdict, "n_vars": len(report["per_variable"])},
        )
        manifest.record_artifact(STAGE_NAME, artifact)
        manifest.mark_completed(STAGE_NAME)
        return 0 if verdict == "PASS" else 5
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed")
        manifest.mark_failed(STAGE_NAME, error=str(e))
        return 5


if __name__ == "__main__":
    sys.exit(main())
