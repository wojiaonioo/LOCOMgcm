"""Shared stage-CLI scaffold used by stage1a / stage1b / stage1c / stage5.

Provides:
  - build_stage_cli_argparser: uniform argparse setup for every stage CLI
  - build_sources: instantiate Layer B sources declared in RunConfig
  - build_execution_context: per-stage exec context with stage-specific output dir
  - build_run_metadata: the run's metadata.yaml contents (stage 1a and stage 2)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from defgcm.io.runner._canonical_helpers import (
    ASSEMBLY_SOURCE_TYPES,
    implicit_vertical_advection_identity,
)
from defgcm.io.runner._time_steps import compute_n_steps
from defgcm.io.runner.config import RunConfig
from defgcm.io.runner.execution import ExecutionContext
from defgcm.io.runner.layer_b import (
    DATA_SOURCES,
    DataSource,
    Grid,
    SourceContext,
)
from defgcm.io.runner.layer_d import ConsumerContext
from defgcm.io.runner.manifest import ManifestWriter


_PROVENANCE_COMMIT_ENV = "DEFGCM_PROVENANCE_COMMIT"
_PROVENANCE_DIRTY_ENV = "DEFGCM_PROVENANCE_DIRTY"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _collect_git_state(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Collect a fail-closed commit/dirty pair for artifact provenance."""
    env = os.environ if environ is None else environ
    has_commit_injection = _PROVENANCE_COMMIT_ENV in env
    has_dirty_injection = _PROVENANCE_DIRTY_ENV in env
    source = (
        "injected"
        if has_commit_injection and has_dirty_injection
        else "mixed"
        if has_commit_injection or has_dirty_injection
        else "git"
    )

    commit = "unknown"
    dirty = True
    branch = "unknown"
    commit_verified = False
    dirty_verified = False
    branch_verified = False

    if has_commit_injection:
        injected_commit = env[_PROVENANCE_COMMIT_ENV]
        if _COMMIT_RE.fullmatch(injected_commit) is not None:
            commit = injected_commit
            commit_verified = True
    if has_dirty_injection:
        injected_dirty = env[_PROVENANCE_DIRTY_ENV]
        if injected_dirty in {"0", "1"}:
            dirty = injected_dirty == "1"
            dirty_verified = True

    complete_injection = has_commit_injection and has_dirty_injection
    injection_invalid = (
        (has_commit_injection and not commit_verified)
        or (has_dirty_injection and not dirty_verified)
    )
    if not complete_injection and not injection_invalid:
        try:
            module_dir = Path(__file__).resolve().parent
            root_probe = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=module_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            root_text = root_probe.stdout.strip()
            if root_probe.returncode != 0 or not root_text:
                raise RuntimeError("git repository root probe failed")
            repo_root = Path(root_text)

            if not has_commit_injection:
                git_head = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                candidate = git_head.stdout.strip()
                if git_head.returncode == 0 and _COMMIT_RE.fullmatch(candidate):
                    commit = candidate
                    commit_verified = True
            if not has_dirty_injection:
                git_dirty = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if git_dirty.returncode == 0:
                    dirty = bool(git_dirty.stdout.strip())
                    dirty_verified = True
            git_branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if git_branch.returncode == 0 and git_branch.stdout.strip():
                branch = git_branch.stdout.strip()
                branch_verified = True
        except Exception:
            pass
    elif complete_injection and not injection_invalid:
        branch_verified = True

    verified = commit_verified and dirty_verified and branch_verified
    if not verified:
        commit = "unknown"
        dirty = True
    return {
        "commit": commit,
        "commit_short": commit[:8] if commit != "unknown" else "unknown",
        "branch": branch,
        "dirty": dirty,
        "probe_status": "verified" if verified else "unverified",
        "source": source,
    }


def build_stage_cli_argparser(stage_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=stage_name)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--set", action="append", dest="sets", default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--persist-intermediate", action="store_true")
    # : orchestrator-propagated `--mitgcm-only` (no-op for stages other
    # than stage1a, which threads it into write_legacy_metadata to flip the
    # metadata.yaml description string).
    parser.add_argument(
        "--mitgcm-only",
        action="store_true",
        help="MITgcm-only mode (orchestrator skips stage1b+stage2+stage5; "
             "stage1a tweaks metadata.yaml description)",
    )
    # : `--skip-prereq` is multi-valued (append). The orchestrator passes
    # `--skip-prereq stage2_torch_run` to stage4 in mitgcm-only mode so the
    # missing LOCOMgcm-side prereq is ignored. `validate_prereqs` honors the skip
    # set by filtering the prereq list before any manifest lookup.
    parser.add_argument(
        "--skip-prereq",
        action="append",
        dest="skip_prereqs",
        default=[],
        help="Prerequisite stage names to skip during validate_prereqs "
             "(orchestrator-propagated; multi-valued).",
    )
    return parser


def build_grid_from_config(run_cfg: RunConfig) -> Grid:
    """Construct a Grid from RunConfig for source instantiation."""
    return Grid(
        nx=run_cfg.grid.nx,
        ny=run_cfg.grid.ny,
        nz=run_cfg.grid.nz,
        center_lat=run_cfg.grid.center_lat,
        center_lon=run_cfg.grid.center_lon,
        resolution_deg=run_cfg.grid.resolution_deg,
        depth_m=run_cfg.grid.depth_m,
        vertical_layer_thicknesses=run_cfg.grid.vertical_layer_thicknesses,
        grid_id=f"{run_cfg.run.experiment_name}_{run_cfg.grid.nx}x{run_cfg.grid.ny}x{run_cfg.grid.nz}",
    )


def build_sources(
    run_cfg: RunConfig,
    *,
    cache_root: Path,
    logger: logging.Logger,
) -> dict[str, DataSource]:
    """Instantiate the Layer B sources declared in run_cfg.sources.

    The assembly reads the ``ASSEMBLY_SOURCE_TYPES`` entries itself.
    """
    model_grid = build_grid_from_config(run_cfg)

    def grid_resolver(grid_id: str) -> Grid:
        if grid_id == model_grid.grid_id or grid_id == "model_grid":
            return model_grid
        raise ValueError(f"unknown grid_id {grid_id!r}")

    ctx = SourceContext(
        cache_root=cache_root,
        target_grid_resolver=grid_resolver,
        logger=logger,
    )
    sources: dict[str, DataSource] = {}
    for spec in run_cfg.sources:
        if spec.type in ASSEMBLY_SOURCE_TYPES:
            continue
        try:
            sources[spec.name] = DATA_SOURCES.instantiate(spec.type, ctx, **dict(spec.config))
        except Exception as e:
            if spec.on_error == "skip":
                logger.warning(f"source {spec.name!r} ({spec.type}) failed to instantiate; skipping: {e}")
            else:
                raise
    return sources


def attach_prereq_zarr_sources(
    sources: dict[str, DataSource],
    manifest: "ManifestWriter",
    prereq_stages: list[str],
    *,
    cache_root: Path,
    logger: logging.Logger,
    run_cfg: RunConfig,
) -> None:
    """Register prereq stages' zarr artifacts as named ZarrBundleSource sources.

    For each artifact recorded by a prereq stage (e.g. stage1a_prepare_common
    writing `prepared_state_zarr`), register a source under the conventional
    name `<stage_short>_<pipeline_name>`. Pipeline names in yaml like
    `primary_source: stage1a_state` resolve to the `state` pipeline's
    `prepared_state_zarr` artifact.
    """
    model_grid = build_grid_from_config(run_cfg)

    def grid_resolver(grid_id: str) -> Grid:
        if grid_id == model_grid.grid_id or grid_id == "model_grid":
            return model_grid
        raise ValueError(f"unknown grid_id {grid_id!r}")

    src_ctx = SourceContext(
        cache_root=cache_root,
        target_grid_resolver=grid_resolver,
        logger=logger,
    )

    # Map stage name → short alias used in yaml `primary_source` refs
    stage_alias = {
        "stage1a_prepare_common": "stage1a",
        "stage1b_export_to": "stage1b",
        "stage1c_export_mit": "stage1c",
    }

    from defgcm.io.runner.manifest import ManifestReader
    reader = ManifestReader(manifest.run_dir)
    for stage_name in prereq_stages:
        alias = stage_alias.get(stage_name, stage_name)
        stage_entry = reader.get_stage(stage_name) or {}
        for artifact in stage_entry.get("artifacts", []):
            art_name = artifact.get("name", "")
            art_path = artifact.get("path", "")
            if not art_name or not art_path:
                continue
            art_path_obj = Path(art_path)
            if not art_path_obj.is_absolute():
                art_path_obj = manifest.run_dir / art_path_obj
            # Convention: prepared_<pipe>_zarr → <stage_alias>_<pipe>
            # Strip "prepared_" prefix and "_zarr" suffix.  The stage-1b consumer
            # names keep their torchocean_ spelling: prepared runs record them in
            # their manifests.
            short = art_name
            if short == "model_input_initial_state_zarr":
                short = "initial_state"
            else:
                for prefix in ("prepared_", "torchocean_"):
                    if short.startswith(prefix):
                        short = short[len(prefix):]
                        break
                for suffix in ("_zarr",):
                    if short.endswith(suffix):
                        short = short[: -len(suffix)]
                        break
            ref_name = f"{alias}_{short}"
            try:
                sources[ref_name] = DATA_SOURCES.instantiate(
                    "zarr_bundle", src_ctx, path=art_path_obj
                )
                logger.info(f"registered prereq source {ref_name!r} → {art_path_obj}")
            except Exception as e:
                logger.warning(f"failed to attach {ref_name!r} from {art_path_obj}: {e}")


def _stepper_kwarg_or_extra(run_cfg: Any, key: str, default: Any) -> Any:
    # The stepper kwargs are what the model runs with; case.extras is the
    # legacy channel and only consulted when the stepper is silent.
    from defgcm.io.runner.model_input import _extra

    stepper = getattr(getattr(run_cfg, "model", None), "stepper", None)
    kwargs = dict(getattr(stepper, "kwargs", None) or {})
    if key in kwargs:
        return kwargs[key]
    return _extra(run_cfg, key, default)


def _steps_from_config(run_cfg: Any) -> int:
    # Routes through the SSOT helper so LOCOMgcm and MIT-export paths cannot
    # disagree on step count — see defgcm/io/runner/_time_steps.py.
    context = getattr(run_cfg.run, "experiment_name", "") or "(unspecified)"
    return compute_n_steps(
        days=run_cfg.time.days,
        dt_seconds=run_cfg.time.dt,
        steps_override=run_cfg.time.steps,
        seconds_override=getattr(run_cfg.time, "seconds", None),
        context=context,
    )


def _extras_spacing(run_cfg: Any, spacing_key: str, extent_key: str, n: int) -> float | None:
    """``case.extras`` grid spacing (``dx``, else ``lx / nx``); None when absent or zero."""
    from defgcm.io.runner.model_input import _extra

    value = _extra(run_cfg, spacing_key)
    if value is None:
        extent = _extra(run_cfg, extent_key)
        value = None if extent is None else float(extent) / max(int(n), 1)
    if value is None or float(value) == 0.0:
        return None
    return float(value)


def build_run_metadata(
    run_cfg: RunConfig,
    *,
    description: str,
    mitgcm_only: bool = False,
    mitgcm_run_dir: Path | None = None,
) -> dict:
    """Contents of a run directory's ``metadata.yaml``.

    Stage 1a writes the file (``write_legacy_metadata``) and stage 2 rewrites
    it (``experiments.cases._runtime._write_metadata``), so that a stage-2-only
    rerun with ``--set`` records the configuration that ran. Both build it
    here; they pass different ``description`` strings, and only stage 2 knows
    ``mitgcm_run_dir``. ``helper.preprocess.adapters`` picks the adapter by
    ``type``, the exp4 and internal-wave adapters read ``model_args``, and the
    metrics writer reads ``type``, ``grid.vertical`` and ``physics.sponge_width``.

    ``type`` is ``case.extras.case_type`` (the adapter key) when the MITgcm
    comparison is on and the case names one, ``mitgcm_comparison`` for any
    other run with a MITgcm reference, and ``defgcm_only`` otherwise.
    ``grid.dx`` / ``grid.dy`` appear only when ``case.extras`` gives a
    non-zero spacing or extent.
    """
    from datetime import datetime

    from defgcm.io.runner.manifest._writer import _to_plain_python
    from defgcm.io.runner.model_input import _extra

    grid_cfg = run_cfg.grid
    # Vertical center depths (cumulative midpoints)
    vlt = list(grid_cfg.vertical_layer_thicknesses)
    center_depths: list[float] = []
    z_cum = 0.0
    for t in vlt:
        center_depths.append(z_cum + t / 2)
        z_cum += t
    interface_depths = [0.0]
    for t in vlt:
        interface_depths.append(interface_depths[-1] + t)
    implicit_vertical_advection = implicit_vertical_advection_identity(run_cfg)

    comparison = bool(run_cfg.mit_comparison.enabled)
    has_reference = comparison or mitgcm_only
    case_type = _extra(run_cfg, "case_type")
    if comparison and case_type is not None:
        run_type = str(case_type)
    elif has_reference:
        run_type = "mitgcm_comparison"
    else:
        run_type = "defgcm_only"
    sources: dict[str, Any] = {"model": {"type": "defgcm_nc", "path": "nc_snapshots/"}}
    if has_reference:
        sources["reference"] = {"type": "mitgcm_binary", "path": "mitgcm/"}
        if mitgcm_run_dir is not None:
            if str(run_cfg.case.kind) == "mit_verification":
                sources["reference"]["path"] = str(mitgcm_run_dir)
            else:
                sources["reference"]["path"] = str(mitgcm_run_dir.parent)

    grid: dict[str, Any] = {
        "center_lat": grid_cfg.center_lat,
        "center_lon": grid_cfg.center_lon,
        "nx": int(grid_cfg.nx),
        "ny": int(grid_cfg.ny),
        "nz": int(grid_cfg.nz),
        "resolution_deg": grid_cfg.resolution_deg,
        "depth_m": grid_cfg.depth_m,
    }
    for key, extent_key, n in (("dx", "lx", grid_cfg.nx), ("dy", "ly", grid_cfg.ny)):
        spacing = _extras_spacing(run_cfg, key, extent_key, n)
        if spacing is not None:
            grid[key] = spacing
    grid["vertical"] = {
        "center_depths_m": center_depths,
        "interface_depths_m": interface_depths,
        "layer_thicknesses_m": vlt,
    }

    metadata = {
        "type": run_type,
        "created_at": datetime.utcnow().isoformat(),
        "description": description,
        "case": str(_extra(run_cfg, "case_name", run_cfg.run.experiment_name)),
        "grid": grid,
        "time": {
            "dt": run_cfg.time.dt,
            "days": run_cfg.time.days,
            "start_date": str(run_cfg.run.start_date),
            # Undeclared substeps default to the parent ``dt``
            # (``TimeConfig.__post_init__``).
            "dt_seconds": float(run_cfg.time.dt),
            "dt_mom_seconds": float(run_cfg.time.dt_mom),
            "dt_tracer_seconds": float(run_cfg.time.dt_tracer),
            "dt_freesurf_seconds": float(run_cfg.time.dt_freesurf),
            "dt_clock_seconds": float(run_cfg.time.dt_clock),
            "total_seconds": float(run_cfg.time.days) * 86400.0,
            "total_steps": _steps_from_config(run_cfg),
            # The resolved cadence (a ``*_seconds`` override wins over
            # ``*_hours``), so SNAPSHOT_SECONDS=30 is not recorded as 3600.
            "snapshot_interval_seconds": run_cfg.output.resolved_snapshot_seconds(),
        },
        "physics": {
            "viscosity": float(run_cfg.physics.viscosity),
            "diffusivity": float(run_cfg.physics.diffusivity),
            "sponge_width": int(run_cfg.boundary.sponge_width),
            "sponge_timescale_hours": float(run_cfg.boundary.sponge_timescale_hours),
            # `boundary.sponge_mode == "linear"` is the modeled equivalent of
            # the legacy `use_linear_sponge` arg; the explicit boolean keeps
            # the metrics writer's `physics.use_linear_sponge` key present.
            "use_linear_sponge": (str(run_cfg.boundary.sponge_mode) == "linear"),
            "vertical_advection_scheme": str(run_cfg.advection.vertical),
            "implicit_vertical_advection": implicit_vertical_advection,
        },
        "sources": sources,
        "model_args": {
            "device": run_cfg.run.device,
            # physics.reference_density is the canonical face;
            # the extras channel stays as the legacy fallback.  Conflicts raise
            # in from_manifest._resolve_reference_density before we get here.
            "reference_density": (
                float(run_cfg.physics.reference_density)
                if getattr(run_cfg.physics, "reference_density", None) is not None
                else float(_extra(run_cfg, "reference_density", 1025.0))
            ),
            "thermal_expansion": float(_extra(run_cfg, "thermal_expansion", 2.0e-4)),
            "haline_contraction": float(_extra(run_cfg, "haline_contraction", 0.0)),
            "diff_kz_t": run_cfg.vertical_diffusion.diffusivity_kr,
            "visc_az": float(run_cfg.physics.implicit_viscosity or 0.0),
            "visc_ah": float(run_cfg.physics.viscosity),
            "diff_kh_t": float(run_cfg.physics.diffusivity),
            "implicit_surf_press": float(_stepper_kwarg_or_extra(run_cfg, "implicit_surf_press", 0.5)),
            "implicit_div_2d_flow": float(_stepper_kwarg_or_extra(run_cfg, "implicit_div_2d_flow", 0.5)),
            "ab_eps": float(_stepper_kwarg_or_extra(run_cfg, "ab_eps", 0.1)),
            "exact_conserv": bool(_stepper_kwarg_or_extra(run_cfg, "exact_conserv", True)),
            "nonlin_free_surf": int(run_cfg.physics.nonlin_free_surf),
            "fwu_correction": run_cfg.physics.transport_correction == "corrected",
            "combined_advection": bool(run_cfg.advection.multi_dim),
            "implicit_vertical_advection": implicit_vertical_advection,
            "dt": float(run_cfg.time.dt),
        },
        "mit_parallel": {
            "nPx": run_cfg.mit_parallel.nPx,
            "nPy": run_cfg.mit_parallel.nPy,
        },
    }
    return _to_plain_python(metadata)


def write_legacy_metadata(
    run_dir: Path,
    run_cfg: RunConfig,
    *,
    args=None,
    mitgcm_only: bool = False,
) -> None:
    """Write outputs/data/<experiment>/{metadata.yaml,provenance.yaml,config_snapshot/}.

    ``metadata.yaml`` comes from :func:`build_run_metadata`. Its
    ``description`` names the run kind: ``"MITgcm-only run: <name>"`` when
    ``mitgcm_only=True``, otherwise the comparison or LOCOMgcm-only text.
    Stage 2 rewrites the file with ``case.extras.description``.
    """
    import sys, platform, dataclasses, yaml as _yaml
    from datetime import datetime

    from defgcm.io.runner.manifest._writer import _to_plain_python

    if mitgcm_only:
        description = f"MITgcm-only run: {run_cfg.run.experiment_name}"
    elif run_cfg.mit_comparison.enabled:
        description = (
            f"Regional LOCOMgcm + MITgcm comparison: {run_cfg.run.experiment_name}"
        )
    else:
        description = f"Regional LOCOMgcm run: {run_cfg.run.experiment_name}"
    metadata = build_run_metadata(run_cfg, description=description, mitgcm_only=mitgcm_only)
    (run_dir / "metadata.yaml").write_text(_yaml.safe_dump(metadata, sort_keys=False))

    # Provenance — git state + environment
    git_state = _collect_git_state()

    try:
        import torch as _torch
        torch_info = {
            "torch_version": str(_torch.__version__),
            "cuda_available": bool(_torch.cuda.is_available()),
        }
        if _torch.cuda.is_available():
            torch_info["cuda_version"] = str(_torch.version.cuda)
            torch_info["gpu_name"] = str(_torch.cuda.get_device_name(0))
    except Exception:
        torch_info = {}

    provenance = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_name": run_cfg.run.experiment_name,
        "git": git_state,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            **torch_info,
        },
        "pipeline": {
            "schema_version": run_cfg.schema_version,
            "layout": "legacy" if getattr(run_cfg.output, "legacy_layout", True) else "stage_subdirs",
        },
        "physics": {
            "implicit_vertical_advection": metadata["physics"]["implicit_vertical_advection"],
        },
    }
    (run_dir / "provenance.yaml").write_text(_yaml.safe_dump(_to_plain_python(provenance), sort_keys=False))

    # config_snapshot/
    snap = run_dir / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    # source_config.yaml = user's input yaml (if reachable via args.config)
    if args is not None and getattr(args, "config", None) is not None:
        src = Path(args.config)
        if src.exists():
            (snap / "source_config.yaml").write_text(src.read_text())
    # resolved_args.yaml = full RunConfig dump
    def _dc_to_dict(o):
        if dataclasses.is_dataclass(o):
            return {f.name: _dc_to_dict(getattr(o, f.name)) for f in dataclasses.fields(o)}
        if isinstance(o, (list, tuple)):
            return [_dc_to_dict(x) for x in o]
        if isinstance(o, dict):
            return {k: _dc_to_dict(v) for k, v in o.items()}
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return o
    (snap / "resolved_args.yaml").write_text(
        _yaml.safe_dump(_to_plain_python(_dc_to_dict(run_cfg)),
                        sort_keys=False, default_flow_style=False)
    )
    # git_state.txt
    (snap / "git_state.txt").write_text(
        f"commit: {git_state.get('commit', 'unknown')}\n"
        f"branch: {git_state.get('branch', 'unknown')}\n"
        f"dirty: {git_state.get('dirty', True)}\n"
        f"probe_status: {git_state.get('probe_status', 'unverified')}\n"
        f"source: {git_state.get('source', 'git')}\n"
    )


_ZARR_SUCCESS_MARKER = "_SUCCESS"


def mark_zarr_incomplete(zarr_path: Path) -> None:
    """Clear the completeness marker before (re)writing a store in place."""
    try:
        (zarr_path / _ZARR_SUCCESS_MARKER).unlink()
    except FileNotFoundError:
        pass


def mark_zarr_complete(zarr_path: Path) -> None:
    """Write the completeness marker — call LAST, after the full store + all
    derived diagnostics are on disk."""
    (zarr_path / _ZARR_SUCCESS_MARKER).touch()


def preprocessed_zarr_is_complete(zarr_path: Path) -> bool:
    """True iff a ``_SUCCESS`` marker sits inside the store.

    A directory artifact like preprocessed.zarr is written IN PLACE — a
    multi-10s-GB store cannot be tmp+rename'd on a near-full disk (that doubles
    peak usage and ENOSPC-fails). To still distinguish a fully-written store
    from one truncated by a crash/OOM/SLURM-kill mid-write, the writer clears
    this marker before writing and re-creates it only after the last byte lands
    — the Hadoop/Spark ``_SUCCESS`` convention. Consumers (dashboard, downstream
    stages) treat a store without ``_SUCCESS`` as incomplete / not trustworthy.
    (zarr ignores the extra 0-byte file.) The manifest stage status is the
    second, authoritative completeness signal.
    """
    return (zarr_path / _ZARR_SUCCESS_MARKER).exists()


def run_helper_preprocess(run_dir: Path) -> Path | None:
    """Generate preprocessed.zarr via helper.preprocess.run_pipeline().

    Compares LOCOMgcm model output (nc_snapshots/day*.nc) vs MIT output
    (mitgcm/run/*.bin), aligning timesteps. Returns zarr structure:

      preprocessed.zarr/
      ├── model/{u,v,w,temp,salt,ssh}_spatial
      ├── mitgcm/{u,v,w,temp,salt,ssh}_spatial
      ├── diff/{u,v,w,temp,salt,ssh}_spatial
      ├── timeseries/hfac.npy
      ├── stats/
      └── diag/    (compute_diag=True)

    Requires `metadata.yaml` at run_dir/ — adapter auto-detects experiment type.
    Pre-req: stage2 + stage3 must have completed (nc_snapshots + mitgcm/run).
    """
    try:
        from helper.preprocess.pipeline import run_pipeline as _hp_run
    except Exception as e:
        raise RuntimeError(f"helper.preprocess unavailable: {e}")

    if not (run_dir / "metadata.yaml").exists():
        raise RuntimeError(f"helper.preprocess requires metadata.yaml at {run_dir}")
    if not (run_dir / "nc_snapshots").exists():
        raise RuntimeError(f"helper.preprocess requires nc_snapshots/ at {run_dir} (stage2 not yet run)")
    if not (run_dir / "mitgcm" / "run").exists():
        raise RuntimeError(f"helper.preprocess requires mitgcm/run/ at {run_dir} (stage3 not yet run)")

    out = run_dir / "preprocessed.zarr"
    # Write in place (a multi-10s-GB store cannot be tmp+rename'd on a near-full
    # disk — that doubles peak usage and ENOSPC-fails). Clear the _SUCCESS
    # marker first and re-create it only after the full write succeeds, so a
    # crash mid-write leaves a store consumers can detect as incomplete
    # (preprocessed_zarr_is_complete).
    mark_zarr_incomplete(out)
    _hp_run(experiment_dir=run_dir, output_zarr=out, compute_diag=True)
    mark_zarr_complete(out)
    return out


def resolve_run_dir_name(run_cfg: RunConfig, run_id: str) -> str:
    """Resolve the subdir name under output.root for this run.

    legacy_layout=True  → run.experiment_name (matches outputs/data/<name>/)
    legacy_layout=False → run_id (run-isolated)
    """
    if getattr(run_cfg.output, "legacy_layout", True):
        return run_cfg.run.experiment_name
    return run_id


def open_manifest_for_stage(args, run_cfg: RunConfig) -> "ManifestWriter":
    """Open ManifestWriter for stage CLI; honors legacy_layout."""
    # Absolute: stage 1a records artifact paths relative to the working
    # directory while later stages resolve them against the run directory, so
    # a relative root (the yaml default) made stage 1b look under a doubled path.
    output_root = Path(args.output or run_cfg.output.root).resolve()
    return ManifestWriter.open(
        run_id=args.run_id,
        output_dir=output_root,
        dir_name=resolve_run_dir_name(run_cfg, args.run_id),
    )


def stage_output_dir_for(stage_name: str, run_dir: Path, run_cfg: RunConfig) -> Path:
    """Resolve per-stage output dir.

    When `run_cfg.output.legacy_layout` is true (default), map stages to the
    following ``outputs/data/<experiment>/...`` subdirectories:

      stage1a_prepare_common  → <run_dir>/preprocessed_cf/      (CF intermediates)
      stage1b_export_to       → <run_dir>/torch_inputs/         (LOCOMgcm IC/OBC/forcing)
      stage1c_export_mit      → <run_dir>/mitgcm/               (input/, code/, *.bin)
      stage2_torch_run        → <run_dir>/                      (legacy writes here)
      stage3_mit_run          → <run_dir>/mitgcm/run/           (MDS output)
      stage4_postproc         → <run_dir>/                      (preprocessed.zarr)
      stage5_compare          → <run_dir>/                      (comparison_metrics.*)

    When `legacy_layout=false`, each stage gets its own subdir
    `<run_dir>/<stage_name>/` (old behavior).
    """
    legacy = getattr(run_cfg.output, "legacy_layout", True)
    if not legacy:
        return run_dir / stage_name

    # CF intermediates + LOCOMgcm inputs go under _pipeline_meta/ — they're new-
    # pipeline-only artifacts, not part of the legacy outputs/data/<exp>/
    # surface that helper.preprocess + run_regional_comparison produce.
    _LEGACY_MAP = {
        "stage1a_prepare_common": run_dir / "_pipeline_meta" / "preprocessed_cf",
        "stage1b_export_to": run_dir / "_pipeline_meta" / "torch_inputs",
        "stage1c_export_mit": run_dir / "mitgcm",
        # stage2/3/4/5 outputs go to the legacy top-level paths; their own
        # intermediate files (logs, generated legacy yaml) live under _pipeline_meta.
        "stage2_torch_run": run_dir / "_pipeline_meta" / "stage2_torch_run",
        "stage3_mit_run": run_dir / "mitgcm" / "run",
        "stage4_postproc": run_dir,           # writes preprocessed.zarr at root
        "stage5_compare": run_dir,            # comparison_metrics.*.json at root
    }
    return _LEGACY_MAP.get(stage_name, run_dir / stage_name)


def build_execution_context(
    stage_name: str,
    run_cfg: RunConfig,
    manifest: ManifestWriter,
    *,
    dry_run: bool = False,
    persist_intermediate: bool = False,
) -> ExecutionContext:
    """Build per-stage ExecutionContext."""
    stage_output_dir = stage_output_dir_for(stage_name, manifest.run_dir, run_cfg)
    stage_output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(stage_name)

    consumer_ctx = ConsumerContext(
        output_dir=stage_output_dir,
        run_id=manifest._run_id,  # type: ignore[attr-defined]
        manifest_writer=manifest,
        logger=logger,
        run_cfg=run_cfg,
    )

    grid = build_grid_from_config(run_cfg)
    source_ctx = SourceContext(
        cache_root=run_cfg.cache.root,
        target_grid_resolver=lambda _gid: grid,
        logger=logger,
    )

    return ExecutionContext(
        run_id=manifest._run_id,  # type: ignore[attr-defined]
        run_cfg=run_cfg,
        stage_name=stage_name,
        output_dir=stage_output_dir,
        source_context=source_ctx,
        consumer_context=consumer_ctx,
        logger=logger,
        persist_intermediate=persist_intermediate,
        dry_run=dry_run,
    )
