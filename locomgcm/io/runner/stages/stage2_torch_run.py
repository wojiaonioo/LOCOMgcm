"""Stage2 LOCOMgcm model run.

Uses the YAML-driven universal runtime. ``case.extras.framework`` is
accepted as an inert compatibility setting.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Sequence

import yaml

from locomgcm.io.runner.config import load_run_config
from locomgcm.io.runner.config._loader import merge_yaml_layers
from locomgcm.io.runner.layer_d import ArtifactKind, ArtifactRef
from locomgcm.io.runner.legacy_bridge import TranslationError, _assert_no_double_track
from locomgcm.io.runner.manifest import ManifestWriter, validate_prereqs

from ._common import build_stage_cli_argparser, open_manifest_for_stage


STAGE_NAME = "stage2_torch_run"
PREREQ_STAGES = ["stage1b_export_to"]
TIMING_ARTIFACT_ENV = "LOCOMGCM_TIMING_ARTIFACT"


def _exit_code(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, int):
        return result
    return 1


def _requested_timing_artifact(stage_out: Path) -> Path | None:
    requested = os.environ.get(TIMING_ARTIFACT_ENV)
    if requested is None:
        return None
    value = requested.strip()
    if value in {"", "1", "true", "True"}:
        return stage_out / "benchmark_timing.json"
    path = Path(value)
    return path if path.is_absolute() else stage_out / path


def _output_cadence_gate(run_cfg) -> None:
    """Fail-loud echo of the LOCOMgcm NetCDF frame cadence — printed to stdout so it lands
    in stage2_stdout.log (where run monitoring looks), not the buffered phase1 log
    (logger.* would bypass the redirect via its basicConfig-bound stderr handler).

    The cadence is driven by output.snapshot_interval_*, NOT nc_output_interval_*
    (which on the LOCOMgcm side only sets the per-.nc-file frame count — yet IS the real
    dump cadence on MIT: same field, two meanings). Echoing what actually takes
    effect lets a mis-set field be caught at a glance instead of drifting silently.
    WARN (not raise): a deliberate split is legal. Silence: LOCOMGCM_SKIP_OUTPUT_GATE=1.
    """
    if os.environ.get("LOCOMGCM_SKIP_OUTPUT_GATE") == "1":
        return
    out = run_cfg.output
    dt = float(run_cfg.time.dt)
    snap_s = out.resolved_snapshot_seconds()
    ncf_s = out.resolved_nc_output_seconds()
    snap_steps = round(snap_s / dt) if dt > 0 else "?"
    print(
        f"[output cadence] LOCOMgcm writes a NetCDF frame every {snap_steps} steps "
        f"(snapshot_interval={snap_s:g}s); nc_output_interval={ncf_s:g}s only sets "
        f"the per-file frame count on the LOCOMgcm side (it IS the dump cadence on MIT).",
        flush=True,
    )
    if dt > 0 and abs(snap_s - ncf_s) > 1e-6 * max(snap_s, ncf_s):
        print(
            f"[output cadence] WARNING: snapshot_interval ({snap_s:g}s) != "
            f"nc_output_interval ({ncf_s:g}s): LOCOMgcm frame cadence follows "
            f"snapshot_interval ({snap_steps} steps/frame). To change LOCOMgcm output "
            f"density edit snapshot_interval_*, NOT nc_output_interval_*. "
            f"(LOCOMGCM_SKIP_OUTPUT_GATE=1 to silence)",
            flush=True,
        )


def _run_framework_in_process(
    *,
    run_cfg,
    manifest: ManifestWriter,
    stage_out: Path,
    logger: logging.Logger,
) -> int:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    is_rank0 = rank == 0
    if is_rank0:
        manifest.mark_running(STAGE_NAME)
    start = time.time()
    log_path = stage_out / (
        "stage2_stdout.log" if is_rank0 else f"stage2_stdout.rank{rank}.log"
    )
    try:
        from experiments.cases._runtime import run_simulation
        from experiments.mitcases.g2_compare import ARTIFACT_SCHEMA
        from experiments.mitcases.g2_history import g2_history_enabled
        from locomgcm.diagnostics.section_transport import (
            SECTION_TRANSPORT_SCHEMA_VERSION,
        )

        g2_path = stage_out / "g2_history.json" if g2_history_enabled() else None
        timing_path = _requested_timing_artifact(stage_out)
        section_config = getattr(
            getattr(run_cfg, "observations", None), "section_transport_u", None
        )
        section_path = (
            stage_out / "section_transport.json" if section_config is not None else None
        )

        with open(log_path, "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            _output_cadence_gate(run_cfg)
            run_kwargs = {"g2_history_path": g2_path}
            if timing_path is not None:
                run_kwargs["timing_artifact_path"] = timing_path
            result = run_simulation(run_cfg, manifest.run_dir, logger, **run_kwargs)
        wallclock = time.time() - start
        exit_code = _exit_code(result)
        if exit_code != 0:
            raise RuntimeError(
                f"framework runner exited with code {exit_code}; see {log_path}"
            )
        if is_rank0:
            if g2_path is not None and not g2_path.is_file():
                raise RuntimeError(
                    f"G2 history was enabled but no artifact exists at {g2_path}"
                )
            artifact = ArtifactRef(
                name="torch_run_framework",
                path=stage_out,
                artifact_kind=ArtifactKind.FILE_DIRECTORY,
                bundle_hash="framework0000",
                schema_version="0.1.0",
                extras={
                    "case_kind": run_cfg.case.kind,
                    "wallclock_seconds": wallclock,
                    "dispatch": "framework",
                    "runner": "experiments.cases._runtime.run_simulation",
                    "log_path": str(log_path),
                },
            )
            manifest.record_artifact(STAGE_NAME, artifact)
            # The side artifacts below are identified by their paths; their
            # bundle_hash is a fixed placeholder, not a content digest.
            if g2_path is not None:
                g2_artifact = ArtifactRef(
                    name="g2_history",
                    path=g2_path,
                    artifact_kind=ArtifactKind.FILE_TEXT,
                    bundle_hash="00000000",
                    schema_version=ARTIFACT_SCHEMA,
                    extras={"role": "zero_perturbation_side_channel"},
                )
                manifest.record_artifact(STAGE_NAME, g2_artifact)
            if timing_path is not None:
                if not timing_path.is_file():
                    raise RuntimeError(
                        f"benchmark timing was requested but no artifact exists at {timing_path}"
                    )
                timing_artifact = ArtifactRef(
                    name="benchmark_timing",
                    path=timing_path,
                    artifact_kind=ArtifactKind.FILE_TEXT,
                    bundle_hash="00000000",
                    schema_version="1.0.0",
                    extras={"role": "steady_state_step_timing"},
                )
                manifest.record_artifact(STAGE_NAME, timing_artifact)
            if section_path is not None:
                if not section_path.is_file():
                    raise RuntimeError(
                        "section transport was requested but no artifact exists at "
                        f"{section_path}"
                    )
                section_artifact = ArtifactRef(
                    name="section_transport",
                    path=section_path,
                    artifact_kind=ArtifactKind.FILE_TEXT,
                    bundle_hash="00000000",
                    schema_version=SECTION_TRANSPORT_SCHEMA_VERSION,
                    extras={"role": "six_hour_section_transport"},
                )
                manifest.record_artifact(STAGE_NAME, section_artifact)
            manifest.mark_completed(STAGE_NAME, wallclock_seconds=wallclock)
        return 0
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed")
        if is_rank0:
            manifest.mark_failed(STAGE_NAME, error=str(e))
        return 5


def main(argv: Sequence[str] | None = None) -> int:
    """Run LOCOMgcm stage 2 via the universal runtime."""
    parser = build_stage_cli_argparser(STAGE_NAME)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(STAGE_NAME)

    if args.config is None:
        parser.error("--config required")

    try:
        run_cfg = load_run_config(args.config, args.overlay, args.sets)
        raw_yaml = merge_yaml_layers(args.config, args.overlay, args.sets)
    except (ValueError, TranslationError) as e:
        # No run directory can be resolved from a configuration that does not
        # load, so there is no manifest to mark; report like every other stage
        # failure (one ERROR line, exit code 5) instead of a traceback.
        logger.error(f"{STAGE_NAME} failed: configuration does not load: {e}")
        return 5
    manifest = open_manifest_for_stage(args, run_cfg)
    # Check double-track consistency for a rerun of this stage alone: stage 1a checks it inside the
    # MITgcm translation, so a stage-2-only rerun with a changed overlay ran
    # on the model.* value while metadata recorded the physics.* value.
    try:
        _assert_no_double_track(raw_yaml)
    except TranslationError as e:
        logger.error(f"{STAGE_NAME} failed: {e}")
        manifest.mark_failed(STAGE_NAME, error=str(e))
        return 5
    if args.dry_run:
        logger.info(
            f"[dry-run] {STAGE_NAME} case.kind={run_cfg.case.kind} dispatch=framework"
        )
        return 0

    if run_cfg.case.kind == "regional":
        validate_prereqs(
            manifest._path,
            STAGE_NAME,
            prereqs=PREREQ_STAGES,
            integrity_check=True,
            force=getattr(args, "force", False),
        )

    from ._common import stage_output_dir_for

    stage_out = stage_output_dir_for(STAGE_NAME, manifest.run_dir, run_cfg)
    stage_out.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))) == 0:
        # config_snapshot/ belongs to stage 1a; this records what the latest
        # stage 2 run actually used, overlays and --set items included.
        (stage_out / "stage2_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "run_id": args.run_id,
                    "config": str(args.config),
                    "overlays": [str(p) for p in args.overlay],
                    "sets": list(args.sets),
                    "merged": raw_yaml,
                },
                sort_keys=False,
            )
        )

    return _run_framework_in_process(
        run_cfg=run_cfg,
        manifest=manifest,
        stage_out=stage_out,
        logger=logger,
    )


if __name__ == "__main__":
    sys.exit(main())
