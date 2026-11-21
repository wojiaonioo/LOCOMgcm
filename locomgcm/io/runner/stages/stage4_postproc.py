"""Stage4 postproc — LOCOMgcm + MIT model outputs → preprocessed.zarr (legacy compat).

Calls `helper.preprocess.pipeline.run_pipeline` which:
  - reads nc_snapshots/day*.nc (LOCOMgcm output)
  - reads mitgcm/run/*.bin via MDS adapter
  - aligns timesteps + writes preprocessed.zarr with model/mitgcm/diff/timeseries groups

Pre-req: stage2 (nc_snapshots/) AND stage3 (mitgcm/run/) must be complete.

With ``mit_comparison.enabled`` false (LOCOMgcm only) the stage requires
only stage2 and calls `helper.preprocess.model_only.build_dashboard_zarr`,
which writes a preprocessed.zarr holding the model and timeseries groups.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence

from locomgcm.io.runner.config import load_run_config
from locomgcm.io.runner.layer_d import ArtifactKind, ArtifactRef
from locomgcm.io.runner.manifest import ManifestWriter, validate_prereqs

from ._common import build_stage_cli_argparser, open_manifest_for_stage


STAGE_NAME = "stage4_postproc"
PREREQ_STAGES = ["stage2_torch_run", "stage3_mit_run"]
REQUIRED_PREREQ_ARTIFACTS = {
    "stage2_torch_run": ("torch_run_framework",),
    "stage3_mit_run": ("mitgcm_raw_mds",),
}
# LOCOMgcm-only mode (mit_comparison.enabled false): no MITgcm output exists.
MODEL_ONLY_PREREQ_STAGES = ["stage2_torch_run"]
MODEL_ONLY_REQUIRED_PREREQ_ARTIFACTS = {
    "stage2_torch_run": ("torch_run_framework",),
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
    model_only = not run_cfg.mit_comparison.enabled
    if args.dry_run:
        mode = "model-only" if model_only else "comparison"
        logger.info(f"[dry-run] {STAGE_NAME} {mode} (prereq check skipped)")
        return 0
    # : orchestrator may pass `--skip-prereq stage2_torch_run` in
    # `--mitgcm-only` mode so the LOCOMgcm-side prereq is bypassed.
    validate_prereqs(
        manifest._path,
        STAGE_NAME,
        prereqs=MODEL_ONLY_PREREQ_STAGES if model_only else PREREQ_STAGES,
        integrity_check=True,
        skip=getattr(args, "skip_prereqs", []) or [],
        force=getattr(args, "force", False),
        required_artifacts=(
            MODEL_ONLY_REQUIRED_PREREQ_ARTIFACTS if model_only else REQUIRED_PREREQ_ARTIFACTS
        ),
    )
    if model_only:
        return _run_model_only(manifest, run_cfg, logger)

    from ._common import run_helper_preprocess
    run_dir = manifest.run_dir

    # helper.preprocess.zarr_writer uses zarr v3 API (Group.create_array).
    # If the current interpreter has zarr < 3, re-exec helper.preprocess

    def _zarr_v3_available() -> bool:
        try:
            import zarr
            return hasattr(zarr.open("/tmp/_z4_probe", mode="w"), "create_array")
        except Exception:
            return False

    manifest.mark_running(STAGE_NAME)
    try:
        if _zarr_v3_available():
            zarr_path = run_helper_preprocess(run_dir)
        else:
            # Subprocess under the same interpreter — use ABSOLUTE paths since
            # cwd is the worktree root (run_dir may be a relative path).
            import subprocess as _sp
            from ._common import mark_zarr_complete, mark_zarr_incomplete
            _py_exe = Path(sys.executable)
            if not _py_exe.exists():
                raise RuntimeError(
                    f"stage4_postproc: sys.executable={_py_exe!r} does not exist; "
                    "cannot launch helper.preprocess subprocess. "
                    "Ensure the correct conda environment is active."
                )
            cuda_py = _py_exe
            abs_run_dir = run_dir.resolve()
            out_zarr = abs_run_dir / "preprocessed.zarr"
            # In-place write + _SUCCESS marker (cleared before, re-created after)
            # so a killed subprocess leaves a store detectable as incomplete.
            mark_zarr_incomplete(out_zarr)
            worktree_root = Path(__file__).resolve().parents[4]
            driver = (
                "from helper.preprocess.pipeline import run_pipeline; "
                f"run_pipeline(experiment_dir=r'{abs_run_dir}', "
                f"output_zarr=r'{out_zarr}', compute_diag=True)"
            )
            logger.info(f"current zarr < v3; running helper.preprocess via {cuda_py} (sys.executable)")
            r = _sp.run([str(cuda_py), "-c", driver], cwd=str(worktree_root),
                        capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    f"helper.preprocess subprocess failed (rc={r.returncode}):\n"
                    f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
                )
            logger.info(r.stdout[-500:])
            mark_zarr_complete(out_zarr)
            zarr_path = out_zarr
        logger.info(f"wrote {zarr_path}")

        artifact = ArtifactRef(
            name="preprocessed_zarr",
            path=zarr_path,
            artifact_kind=ArtifactKind.FILE_ZARR,
            bundle_hash="helper_preprocess",
            schema_version="0.1.0",
            extras={"source": "helper.preprocess.pipeline.run_pipeline"},
        )
        manifest.record_artifact(STAGE_NAME, artifact)
        manifest.mark_completed(STAGE_NAME)
        return 0
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed")
        manifest.mark_failed(STAGE_NAME, error=str(e))
        return 5


def _run_model_only(manifest: ManifestWriter, run_cfg, logger: logging.Logger) -> int:
    """Write the model-only preprocessed.zarr from nc_snapshots/ and the grid tensors.

    Resolution, snapshot cadence and domain centre come from ``run_cfg``, so
    the store does not depend on ``model_input/run_params.yaml``.
    """
    from helper.preprocess.model_only import build_dashboard_zarr

    from ._common import mark_zarr_incomplete

    out = manifest.run_dir / "preprocessed.zarr"
    manifest.mark_running(STAGE_NAME)
    try:
        # The writer creates _SUCCESS last; clear a previous one first so a
        # failure before the writer starts does not leave a stale marker.
        mark_zarr_incomplete(out)
        build_dashboard_zarr(
            manifest.run_dir,
            res=float(run_cfg.grid.resolution_deg),
            snap_seconds=run_cfg.output.resolved_snapshot_seconds(),
            center_lat=float(run_cfg.grid.center_lat),
            center_lon=float(run_cfg.grid.center_lon),
            out=out,
        )
        logger.info(f"wrote {out} (model only)")
        manifest.record_artifact(
            STAGE_NAME,
            ArtifactRef(
                name="preprocessed_zarr",
                path=out,
                artifact_kind=ArtifactKind.FILE_ZARR,
                bundle_hash="helper_preprocess_model_only",
                schema_version="0.1.0",
                extras={
                    "mode": "model_only",
                    "source": "helper.preprocess.model_only.build_dashboard_zarr",
                },
            ),
        )
        manifest.mark_completed(STAGE_NAME)
        return 0
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed")
        manifest.mark_failed(STAGE_NAME, error=str(e))
        return 5


if __name__ == "__main__":
    sys.exit(main())
