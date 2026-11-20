"""Stage1a prepare_common CLI.

Reads raw external datasets (HYCOM/ERA5/GEBCO) and writes CF_STANDARD
normalized intermediate zarrs at outputs/<run_id>/stage1a/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from defgcm.io.runner.config import load_run_config
from defgcm.io.runner.config._loader import merge_yaml_layers
from defgcm.io.runner.execution import run_pipeline_spec
from defgcm.io.runner.execution._dask_budget import configure_dask_for_prep
from defgcm.io.runner.layer_b import DATA_SOURCES, DataQuery, SourceContext
from defgcm.io.runner.legacy_bridge import TranslationError
from defgcm.io.runner.model_input import build_model_state, write_model_input

from ._common import (
    build_execution_context,
    open_manifest_for_stage,
    build_sources,
    build_grid_from_config,
    build_stage_cli_argparser,
    write_legacy_metadata,
)


STAGE_NAME = "stage1a_prepare_common"


_PREPARED_ARTIFACT_TO_BUNDLE = {
    "prepared_state_zarr": "state",
    "prepared_boundary_zarr": "boundary",
    "prepared_mesh_zarr": "mesh",
    "prepared_forcing_zarr": "forcing",
    "prepared_tidal_zarr": "tidal",
}


def _bundles_from_artifacts(artifacts, run_cfg, logger):
    """Read just-written prepared zarr artifacts back as DataBundles.

    stage1a writes the expensive prepared zarrs first because downstream
    stages consume those artifacts.  Reusing them here avoids executing the
    raw-source interpolation chain a second time when building ModelInput.
    """

    model_grid = build_grid_from_config(run_cfg)

    def grid_resolver(grid_id: str):
        if grid_id == model_grid.grid_id or grid_id == "model_grid":
            return model_grid
        raise ValueError(f"unknown grid_id {grid_id!r}")

    src_ctx = SourceContext(
        cache_root=run_cfg.cache.root,
        target_grid_resolver=grid_resolver,
        logger=logger,
    )
    bundles = {}
    for artifact in artifacts:
        bundle_name = _PREPARED_ARTIFACT_TO_BUNDLE.get(str(artifact.name))
        if bundle_name is None or artifact.path is None:
            continue
        source = DATA_SOURCES.instantiate("zarr_bundle", src_ctx, path=artifact.path)
        produces = tuple(source.capabilities.produces)
        if len(produces) != 1:
            raise RuntimeError(
                f"prepared artifact {artifact.name} at {artifact.path} "
                f"does not declare exactly one location kind: {produces}"
            )
        bundles[bundle_name] = source.fetch(
            DataQuery(
                location_kind=produces[0],
                variables=source.capabilities.variables,
                time_range=None,
                target_grid=model_grid,
                extras={},
            )
        )
        logger.info("reusing prepared %s artifact for ModelInput", bundle_name)
    return bundles


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_stage_cli_argparser(STAGE_NAME)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger(STAGE_NAME)

    if args.config is None:
        parser.error("--config required for stage1a (no prerequisite stage)")

    run_cfg = load_run_config(args.config, args.overlay, args.sets)

    output_root = args.output or run_cfg.output.root
    manifest = open_manifest_for_stage(args, run_cfg)

    if args.dry_run:
        logger.info(f"[dry-run] would run {STAGE_NAME} for run_id={args.run_id}")
        return 0

    cache_root = run_cfg.cache.root
    cache_root.mkdir(parents=True, exist_ok=True)
    sources = build_sources(run_cfg, cache_root=cache_root, logger=logger)

    exec_ctx = build_execution_context(
        STAGE_NAME, run_cfg, manifest,
        dry_run=args.dry_run,
        persist_intermediate=args.persist_intermediate,
    )

    # The prep pipelines are dask-backed (sources chunk along time); cap dask's
    # worker count from the SLURM --mem budget so streaming stays memory-bounded
    # instead of dask defaulting to ~n_cores concurrent blocks → OOM on big grids.
    configure_dask_for_prep(run_cfg, logger)

    manifest.mark_running(STAGE_NAME)
    try:
        pipelines = run_cfg.pipelines.get(STAGE_NAME, ())
        if not pipelines:
            logger.warning(f"{STAGE_NAME}: no pipelines declared in run_cfg.pipelines")
        prepared_artifacts = []
        for p in pipelines:
            logger.info(f"running pipeline {p.name}")
            artifact = run_pipeline_spec(p, run_cfg, sources, exec_ctx)
            manifest.record_artifact(STAGE_NAME, artifact)
            prepared_artifacts.append(artifact)

        logger.info("building canonical ModelInput")
        raw_yaml = merge_yaml_layers(args.config, args.overlay, args.sets)
        stage1a_bundles = _bundles_from_artifacts(prepared_artifacts, run_cfg, logger)
        model_input = build_model_state(
            run_cfg,
            raw_yaml,
            stage1a_bundles=stage1a_bundles,
        )
        model_input_dir = manifest.run_dir / "_pipeline_meta" / "model_input"
        for artifact in write_model_input(model_input, model_input_dir).values():
            manifest.record_artifact(STAGE_NAME, artifact)
        logger.info(f"wrote canonical ModelInput under {model_input_dir}")

        # Legacy-layout finalize: write top-level metadata files (NOT
        # preprocessed.zarr — that's stage4's post-run job via helper.preprocess).
        # : `--mitgcm-only` (propagated by the orchestrator) flips the
        # metadata.yaml description string to "MITgcm-only run: <name>" so
        # downstream tools can tell the run apart from a full LOCOMgcm+MIT compare.
        if getattr(run_cfg.output, "legacy_layout", True):
            write_legacy_metadata(
                manifest.run_dir,
                run_cfg,
                args=args,
                mitgcm_only=bool(getattr(args, "mitgcm_only", False)),
            )
            logger.info("wrote metadata.yaml / provenance.yaml / config_snapshot/")
        manifest.mark_completed(STAGE_NAME)
        return 0
    except Exception as e:
        if isinstance(e, (TranslationError, ValueError)):
            # Configuration errors carry their own explanation; a traceback
            # adds nothing (the other stages report them the same way).
            logger.error(f"{STAGE_NAME} failed: {e}")
        else:
            logger.exception(f"{STAGE_NAME} failed")
        manifest.mark_failed(STAGE_NAME, error=str(e))
        return 5


if __name__ == "__main__":
    sys.exit(main())
