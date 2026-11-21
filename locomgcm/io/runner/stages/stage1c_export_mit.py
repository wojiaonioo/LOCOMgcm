"""Stage1c export to MITgcm inputs (binary + namelist + topology)."""

from __future__ import annotations

import logging
import sys
from typing import Sequence

from locomgcm.io.runner.config import load_run_config
from locomgcm.io.runner.execution import run_pipeline_spec
from locomgcm.io.runner.manifest import ManifestWriter, validate_prereqs

from ._common import (
    attach_prereq_zarr_sources,
    open_manifest_for_stage,
    build_execution_context,
    build_sources,
    build_stage_cli_argparser,
)


STAGE_NAME = "stage1c_export_mit"
PREREQ_STAGES = ["stage1a_prepare_common"]
REQUIRED_PREREQ_ARTIFACTS = {
    "stage1a_prepare_common": ("model_input_initial_state_zarr",),
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
        logger.info(f"[dry-run] {STAGE_NAME} (prereq check skipped)")
        return 0
    if not run_cfg.mit_comparison.enabled:
        # Stage 1a skips the MITgcm translation when the comparison is off, so
        # model_input/run_params.yaml lacks the parameters this export reads.
        msg = (
            "the MITgcm export needs mit_comparison.enabled=true; rerun from "
            "stage 1a with --set mit_comparison.enabled=true"
        )
        logger.error(f"{STAGE_NAME} failed: {msg}")
        manifest.mark_failed(STAGE_NAME, error=msg)
        return 5
    validate_prereqs(
        manifest._path,
        STAGE_NAME,
        prereqs=PREREQ_STAGES,
        integrity_check=True,
        force=getattr(args, "force", False),
        required_artifacts=REQUIRED_PREREQ_ARTIFACTS,
    )
    sources = build_sources(run_cfg, cache_root=run_cfg.cache.root, logger=logger)
    attach_prereq_zarr_sources(
        sources, manifest, PREREQ_STAGES,
        cache_root=run_cfg.cache.root, logger=logger, run_cfg=run_cfg,
    )
    exec_ctx = build_execution_context(STAGE_NAME, run_cfg, manifest, dry_run=False)
    manifest.mark_running(STAGE_NAME)
    try:
        for p in run_cfg.pipelines.get(STAGE_NAME, ()):
            artifact = run_pipeline_spec(p, run_cfg, sources, exec_ctx)
            manifest.record_artifact(STAGE_NAME, artifact)
        manifest.mark_completed(STAGE_NAME)
        return 0
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed")
        manifest.mark_failed(STAGE_NAME, error=str(e))
        return 5


if __name__ == "__main__":
    sys.exit(main())
