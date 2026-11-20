from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .ai_digest import build_ai_digest
from .availability import build_variables, wet_bathymetry_stats
from .barotropic_baroclinic import build_barotropic_baroclinic
from .boundary_flux import build_boundary_flux
from .categorical import build_categorical
from .energy_conservation import build_energy_conservation
from .neighborhood import build_neighborhood
from .global_summary import build_per_timestep_sampled
from .sampling import build_sampling
from .schema import (
    OPTIONAL_GROUPS,
    SCHEMA_VERSION,
    MetricsContext,
    infer_nt,
    read_store_attrs,
    unavailable,
    write_json,
)
from .sign_structure import build_sign_structure
from .spectral import build_spectral
from .temporal import build_mean_fluctuation, build_temporal_diagnostics, build_temporal_frequency
from .verification_driver import build_verification
from .vertical_slices import build_vertical_slices
from .w_diagnostics import build_w_diagnostics


SUMMARY_LIMIT_BYTES = 50_000

_FAST_SKIP_REASON = (
    "fast metrics mode — only variables + the slim CNT verification (global "
    "volume-weighted metrics + per-snapshot skill curve → the verdict) computed; "
    "set METRICS_FULL=1 (run_case.sh) or DEFGCM_METRICS_FULL=1 (env) for the full "
    "diagnostics: the stratified per-bin (depth/coast/obc/slope/region/hfac) CNT "
    "breakdown + bootstrap CI, plus the spectral/temporal/energy/w/boundary groups "
    "(minutes on large grids)"
)


def write_comparison_metrics(
    experiment_dir: Path | str,
    store_path: Path | str,
    grid: Any,
    metadata: dict[str, Any],
    *,
    availability_hints: dict[str, Any] | None = None,
    full: bool | None = None,
) -> tuple[Path, Path]:
    """Write schema-driven summary/details metrics next to an experiment.

    ``full`` selects how many diagnostic groups are computed:

    * ``False`` (default — resolved from ``DEFGCM_METRICS_FULL`` env when ``None``):
      compute only ``variables`` + ``global_summary`` (verdict + per-variable
      rmse/corr). The heavy groups (the systematic CNT ``verification``,
      spectral, temporal, energy, w/boundary diagnostics) are marked skipped.
    * ``True``: compute every group (needed for offline alignment investigation
      — e.g. the per-bin hfac/region/depth CNT breakdown + bootstrap CI).
    """
    ctx = MetricsContext(
        experiment_dir=Path(experiment_dir),
        store_path=Path(store_path),
        grid=grid,
        metadata=metadata,
        availability_hints=availability_hints,
    )

    if full is None:
        full = os.environ.get("DEFGCM_METRICS_FULL", "").strip().lower() in {"1", "true", "yes", "on"}

    variables = build_variables(ctx)
    sampling = build_sampling(ctx, variables)
    provenance_alignment = _provenance_alignment(ctx)
    case = _case(ctx)

    # Systematic CNT verification (weighted-moment engine, single finite-volume
    # weighting hFac*cos-lat*drF) — the SINGLE source of the verdict + per-variable
    # aggregate (not an hfac-masked-but-unweighted global summary).
    # Always computed: fast mode runs the slim variant (global metrics + per-
    # snapshot skill curve — one field read, no stratified/CI) so the verdict is
    # the same volume-weighted quantity in fast and full; full adds the stratified
    # per-bin (depth/coast/obc/slope/region/hfac) breakdown + bootstrap CI.
    verification = build_verification(ctx, variables, sampling, full=full)
    ai_digest = build_ai_digest(case, variables, verification, provenance_alignment)

    def _heavy(build):
        return build() if full else unavailable(_FAST_SKIP_REASON)

    details: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": ctx.created_utc,
        "metrics_mode": "full" if full else "fast",
        "case": case,
        "variables": variables,
        "conventions": _conventions(variables),
        "ai_digest": ai_digest,
        "quality_control": {},
        "provenance_alignment": provenance_alignment,
        "sampling": sampling,
        "per_timestep_sampled": _heavy(lambda: build_per_timestep_sampled(ctx, variables, sampling)),
        "vertical_slices": _heavy(lambda: build_vertical_slices(ctx, variables, sampling)),
        "verification": verification,
        "categorical_cts": _heavy(lambda: build_categorical(ctx, variables, sampling, full=full)),
        "neighborhood_fss": _heavy(lambda: build_neighborhood(ctx, variables, sampling, full=full)),
        "temporal_diagnostics": _heavy(lambda: build_temporal_diagnostics(ctx, variables)),
        "mean_fluctuation": _heavy(lambda: build_mean_fluctuation(ctx, variables)),
        "barotropic_baroclinic": _heavy(lambda: build_barotropic_baroclinic(ctx, variables)),
        "temporal_frequency": _heavy(lambda: build_temporal_frequency(ctx, variables)),
        "spectral": _heavy(lambda: build_spectral(ctx, sampling)),
        "energy_conservation": _heavy(lambda: build_energy_conservation(ctx, sampling)),
        "boundary_transport_flux_budget": _heavy(lambda: build_boundary_flux(ctx)),
        "w_diagnostics": _heavy(lambda: build_w_diagnostics(ctx, variables, sampling)),
        "sign_pair_structure": _heavy(lambda: build_sign_structure(ctx, variables, sampling)),
        "operator_residuals": unavailable("operator/intermediate residual sidecars unavailable"),
    }
    details["quality_control"] = _quality_control(ctx, details)

    summary = _summary(details)
    summary_path = ctx.experiment_dir / "comparison_metrics.summary.json"
    details_path = ctx.experiment_dir / "comparison_metrics.details.json"
    write_json(details_path, details)
    write_json(summary_path, summary)

    summary_size = summary_path.stat().st_size
    if summary_size > SUMMARY_LIMIT_BYTES:
        summary = _summary(details, emergency=True)
        write_json(summary_path, summary)
    return summary_path, details_path


def _case(ctx: MetricsContext) -> dict[str, Any]:
    attrs = read_store_attrs(ctx)
    hfac = getattr(ctx.grid, "hfac", None)
    hfac_arr = np.asarray(hfac, dtype=np.float64) if hfac is not None else None
    wet3d = hfac_arr > 0 if hfac_arr is not None else None
    wet2d = np.any(wet3d, axis=0) if wet3d is not None else None
    dz = _layer_thickness(ctx)
    bathy = np.sum(hfac_arr * dz[:, None, None], axis=0) if hfac_arr is not None else None
    case_type = ctx.metadata.get("type", "unknown")
    return {
        "case_id": ctx.experiment_dir.name,
        "case_family": _case_family(case_type),
        "comparison_target": "DEFgcm_vs_MITgcm",
        "diff_convention": "model_minus_mitgcm",
        "metadata_type": case_type,
        "grid": {
            "nx": int(ctx.grid.nx),
            "ny": int(ctx.grid.ny),
            "nz": int(ctx.grid.nz),
            "horizontal": _horizontal(ctx, attrs),
            "vertical": {
                "type": "z_level",
                "center_depth_m": [float(v) for v in np.asarray(ctx.grid.depth).tolist()],
                "layer_thickness_m": [float(v) for v in dz.tolist()],
                "center_depths_m": _metadata_center_depths(ctx),
                "layer_thicknesses_m": [float(v) for v in dz.tolist()],
            },
            "bathymetry": {
                **wet_bathymetry_stats(bathy, wet2d),
                "wet_fraction_2d": float(wet2d.mean()) if wet2d is not None else None,
                "wet_fraction_3d": float(wet3d.mean()) if wet3d is not None else None,
            },
            "masks": {
                "has_land": bool(wet2d is not None and not wet2d.all()),
                "has_obstacle": bool(wet2d is not None and not wet2d.all()),
                "has_partial_cells": bool(hfac_arr is not None and np.any((hfac_arr > 0.0) & (hfac_arr < 0.999))),
                "hfac_min": float(np.nanmin(hfac_arr)) if hfac_arr is not None else None,
                "hfac_max": float(np.nanmax(hfac_arr)) if hfac_arr is not None else None,
            },
            "boundaries": {
                "open_sides": _open_sides(ctx),
                "sponge_width_cells": ctx.metadata.get("physics", {}).get("sponge_width"),
            },
        },
        "source_artifacts": {
            "zarr_path": str(ctx.store_path),
            "metadata_path": str(ctx.experiment_dir / "metadata.yaml"),
            "provenance_path": str(ctx.experiment_dir / "provenance.yaml"),
        },
    }


def _horizontal(ctx: MetricsContext, attrs: dict[str, Any]) -> dict[str, Any]:
    lon = np.asarray(attrs.get("lon", getattr(ctx.grid, "lon", [])), dtype=np.float64)
    lat = np.asarray(attrs.get("lat", getattr(ctx.grid, "lat", [])), dtype=np.float64)
    if lat.size == 1 or float(np.nanmax(np.abs(lat))) > 90.0 or float(np.nanmax(np.abs(lon))) > 360.0:
        return {
            "type": "cartesian",
            "dx_m": _mean_spacing(lon),
            "dy_m": _mean_spacing(lat),
        }
    return {
        "type": "spherical",
        "lon_range": [float(np.nanmin(lon)), float(np.nanmax(lon))] if lon.size else None,
        "lat_range": [float(np.nanmin(lat)), float(np.nanmax(lat))] if lat.size else None,
    }


def _conventions(variables: dict[str, Any]) -> dict[str, Any]:
    return {
        "diff": "model_minus_mitgcm",
        "w": {
            "defgcm_internal": "downward_positive",
            "mitgcm_reference": "upward_positive",
            "comparison_value": "MITgcm upward-positive",
            "conversion_applied": variables.get("w", {}).get("availability_status") == "compared",
            "conversion_location": "LOCOMgcm NC snapshot writers negate diagnostic w at write time (nc_snapshot._state_to_nc_fields with diagnose_w=True; OBCS overlay via _runtime._obcs_w_extract_factory); IW/exp4 adapters and metrics read NC w as-is",
        },
        "vertical_coordinate": {
            "defgcm_center_depth": "positive_down",
            "mitgcm_rC": "positive_up_negative_in_ocean",
        },
    }


def _provenance_alignment(ctx: MetricsContext) -> dict[str, Any]:
    path = ctx.experiment_dir / "provenance.yaml"
    if not path.exists():
        return {"availability": "skipped", "skip_reason": "provenance_yaml_missing"}
    try:
        import yaml

        prov = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        return {"availability": "skipped", "skip_reason": f"provenance_yaml_unreadable:{exc}"}
    alignment = prov.get("alignment")
    if not alignment:
        return {"availability": "skipped", "skip_reason": "alignment_section_missing"}
    checks = alignment.get("checks", [])
    critical = [
        c for c in checks
        if c.get("status") in {"MISMATCH", "UNVERIFIED"} and c.get("severity") == "CRITICAL"
    ][:12]
    return {
        "availability": "computed",
        "summary": alignment.get("summary", {}),
        "critical_mismatches_or_unverified": critical,
    }


def _quality_control(ctx: MetricsContext, details: dict[str, Any]) -> dict[str, Any]:
    groups = {}
    skipped = {}
    for name in OPTIONAL_GROUPS:
        block = details.get(name, {})
        status = block.get("availability", "computed") if isinstance(block, dict) else "computed"
        groups[name] = status
        if status == "skipped":
            skipped[name] = block.get("skip_reason")
    variables = details.get("variables", {})
    return {
        "availability": "computed",
        "summary_limit_bytes": SUMMARY_LIMIT_BYTES,
        "snapshot_count": infer_nt(ctx),
        "group_availability": groups,
        "skipped_groups": skipped,
        "variable_status_counts": _variable_status_counts(variables),
        "near_zero_reference_policy": "rel_rmse is null when ref_rms <= 1e-30; use absolute metrics",
        "precision": "metrics computed in float64 from frame-indexed store arrays",
        "derived_diagnostics": {
            "spatial_stats_json": (ctx.stats_dir / "spatial_stats.json").exists(),
            "taylor_data_json": (ctx.stats_dir / "taylor_data.json").exists(),
            "diag_json_dir": ctx.diag_dir.exists(),
        },
    }


def _summary(details: dict[str, Any], *, emergency: bool = False) -> dict[str, Any]:
    variables = {
        var: {
            "availability_status": block.get("availability_status"),
            "compared": block.get("compared"),
            "shape": block.get("shape"),
            "valid_fill_count": block.get("valid_fill_count"),
            "skip_reason": block.get("skip_reason"),
        }
        for var, block in details.get("variables", {}).items()
        if isinstance(block, dict) and "availability_status" in block
    }
    optional = _optional_summaries(details)
    summary = {
        "schema_version": details.get("schema_version"),
        "created_utc": details.get("created_utc"),
        "details_artifact": "comparison_metrics.details.json",
        "case": details.get("case"),
        "variables": {
            "availability_contract": details.get("variables", {}).get("availability_contract"),
            "summary": details.get("variables", {}).get("summary"),
            **variables,
        },
        "conventions": details.get("conventions"),
        "ai_digest": details.get("ai_digest"),
        "quality_control": details.get("quality_control"),
        "provenance_alignment": details.get("provenance_alignment"),
        "sampling": {
            "max_nodes": details.get("sampling", {}).get("max_nodes"),
            "available_snapshot_count": details.get("sampling", {}).get("available_snapshot_count"),
            "selected_nodes": details.get("sampling", {}).get("selected_nodes", []),
        },
        **optional,
    }
    if emergency:
        summary["case"] = {
            "case_id": details.get("case", {}).get("case_id"),
            "case_family": details.get("case", {}).get("case_family"),
            "grid": details.get("case", {}).get("grid", {}),
        }
        summary["sampling"]["selected_nodes"] = summary["sampling"]["selected_nodes"][:10]
        keep = {"verification", "temporal_diagnostics", "energy_conservation", "boundary_transport_flux_budget", "w_diagnostics"}
        for name in OPTIONAL_GROUPS:
            if name != "sampling" and name not in keep:
                summary.pop(name, None)
    return summary


def _optional_summaries(details: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name in OPTIONAL_GROUPS:
        if name == "sampling":
            continue
        block = details.get(name, {})
        if not isinstance(block, dict):
            continue
        compact = {"availability": block.get("availability", "computed")}
        if block.get("skip_reason"):
            compact["skip_reason"] = block.get("skip_reason")
        if name == "verification" and block.get("availability") == "computed":
            compact["mode"] = block.get("mode")
            compact["weighting"] = block.get("weighting")
            compact["aggregate"] = block.get("aggregate")
            compact["global_by_variable"] = {
                var: _compact_verification_global(v.get("global", {}))
                for var, v in block.get("by_variable", {}).items()
            }
            compact["worst_strata"] = _worst_strata(block)
        elif name in {"categorical_cts", "neighborhood_fss"} and block.get("availability") == "computed":
            compact["aggregate"] = block.get("aggregate")
        elif name == "temporal_diagnostics" and block.get("availability") == "computed":
            compact["growth_rates"] = {
                var: v.get("growth_rates", {})
                for var, v in block.get("by_variable", {}).items()
            }
        elif name in {"energy_conservation", "boundary_transport_flux_budget", "w_diagnostics"}:
            compact["summary"] = block.get("summary")
        out[name] = compact
    return out


def _compact_verification_global(g: dict[str, Any]) -> dict[str, Any]:
    keys = ("rmse", "rel_rmse", "me", "mae", "pearson_r", "r2", "std_ratio", "n_valid")
    out = {key: g.get(key) for key in keys}
    out["ci"] = g.get("ci")
    return out


def _worst_strata(block: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    """Top stratified bins by rmse across variables (full mode only)."""
    rows = []
    for var, vblock in block.get("by_variable", {}).items():
        strata = vblock.get("stratified") or {}
        for dimension, table in strata.items():
            bins = table.get("bins", [])
            rmses = table.get("rmse", [])
            wfrac = table.get("weight_fraction", [])
            for i, name in enumerate(bins):
                val = rmses[i] if i < len(rmses) else None
                if val is None:
                    continue
                rows.append({
                    "variable": var,
                    "dimension": dimension,
                    "bin": name,
                    "rmse": float(val),
                    "weight_fraction": wfrac[i] if i < len(wfrac) else None,
                })
    rows.sort(key=lambda r: -r["rmse"])
    return rows[:limit]


def _case_family(case_type: str) -> str:
    if case_type == "internal_wave_comparison":
        return "internal_wave"
    if case_type == "exp4_comparison":
        return "exp4"
    if case_type in {"mitgcm_comparison", "hycom_comparison", "mitgcm_only", None}:
        return "regional"
    return str(case_type or "unknown")


def _open_sides(ctx: MetricsContext) -> list[str]:
    family = _case_family(ctx.metadata.get("type", "unknown"))
    if family == "internal_wave":
        return ["west", "east"]
    if family in {"exp4", "regional"}:
        return ["west", "east", "south", "north"]
    return []


def _mean_spacing(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    return float(np.nanmean(np.diff(values)))


def _layer_thickness(ctx: MetricsContext) -> np.ndarray:
    vertical = (ctx.metadata.get("grid", {}) or {}).get("vertical")
    if vertical and "layer_thicknesses_m" in vertical:
        dz = np.asarray(vertical["layer_thicknesses_m"], dtype=np.float64)
        if dz.ndim != 1 or dz.size != int(ctx.grid.nz):
            raise ValueError("metadata grid.vertical.layer_thicknesses_m must be 1-D with length nz")
        if not np.isfinite(dz).all() or not (dz > 0.0).all():
            raise ValueError("metadata grid.vertical.layer_thicknesses_m must be finite and positive")
        return dz
    depth = np.asarray(getattr(ctx.grid, "depth", np.arange(ctx.grid.nz)), dtype=np.float64)
    if depth.size <= 1:
        return np.ones(int(ctx.grid.nz), dtype=np.float64)
    edges = np.empty(depth.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    edges[0] = max(0.0, depth[0] - (edges[1] - depth[0]))
    edges[-1] = depth[-1] + (depth[-1] - edges[-2])
    return np.maximum(np.diff(edges), 1.0e-12)


def _metadata_center_depths(ctx: MetricsContext) -> list[float]:
    vertical = (ctx.metadata.get("grid", {}) or {}).get("vertical")
    if vertical and "center_depths_m" in vertical:
        centers = np.asarray(vertical["center_depths_m"], dtype=np.float64)
    else:
        centers = np.asarray(getattr(ctx.grid, "depth", np.arange(ctx.grid.nz)), dtype=np.float64)
    return [float(v) for v in centers.tolist()]


def _variable_status_counts(variables: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in variables.values():
        if not isinstance(block, dict):
            continue
        status = block.get("availability_status")
        if status:
            counts[status] = counts.get(status, 0) + 1
    return counts
