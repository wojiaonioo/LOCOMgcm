"""Main yaml-driven pipeline orchestrator.

Usage:
    python -m experiments.pipeline_v2.run_pipeline \
        --config experiments/configs/cases/pureocean.yaml \
        --run-id my-run \
        [--stages 1a,1b,1c,2,3,4,5]
        [--mitgcm-only]

Without ``--stages`` the configuration picks the stages: LOCOMgcm only
(1a, 1b, 2, 4) when ``mit_comparison.enabled`` is false, every stage when it
is true. ``--stages all`` always means every stage.

``--mitgcm-only`` skips the LOCOMgcm side (stage1b + stage2) and the
comparison stage (stage5), leaving stage1a + stage1c + stage3 + stage4.
Stage4 receives ``--skip-prereq stage2_torch_run`` so its LOCOMgcm-side prereq
check passes; stage1a flips ``metadata.yaml.description`` to
``"MITgcm-only run: <name>"``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class StageEntry:
    name: str
    cli_module: str
    prerequisites: tuple[str, ...]


STAGES: tuple[StageEntry, ...] = (
    StageEntry("stage1a_prepare_common", "locomgcm.io.runner.stages.stage1a_prepare_common", ()),
    StageEntry("stage1b_export_to", "locomgcm.io.runner.stages.stage1b_export_to", ("stage1a_prepare_common",)),
    StageEntry("stage1c_export_mit", "locomgcm.io.runner.stages.stage1c_export_mit", ("stage1a_prepare_common",)),
    StageEntry("stage2_torch_run", "locomgcm.io.runner.stages.stage2_torch_run", ("stage1b_export_to",)),
    StageEntry("stage3_mit_run", "locomgcm.io.runner.stages.stage3_mit_run", ("stage1c_export_mit",)),
    # stage4 consumes BOTH the LOCOMgcm output (nc_snapshots, stage2) and the MIT
    # output (mitgcm/run, stage3); its prereq list here is aligned with
    # stage4_postproc.PREREQ_STAGES so re-running stage2 correctly invalidates
    # (STALEs) stage4 downstream. (--mitgcm-only filters stage2 out post-sort
    # and passes stage4 --skip-prereq stage2_torch_run.)
    StageEntry("stage4_postproc", "locomgcm.io.runner.stages.stage4_postproc", ("stage2_torch_run", "stage3_mit_run")),
    StageEntry("stage5_compare", "locomgcm.io.runner.stages.stage5_compare", ("stage2_torch_run", "stage4_postproc")),
)

_NAME_ALIASES = {
    "1a": "stage1a_prepare_common",
    "1b": "stage1b_export_to",
    "1c": "stage1c_export_mit",
    "2": "stage2_torch_run",
    "3": "stage3_mit_run",
    "4": "stage4_postproc",
    "5": "stage5_compare",
}

# Default stage set when `--stages` is omitted and `mit_comparison.enabled` is
# false: prepare, export, run LOCOMgcm and write the model-only
# preprocessed.zarr. No stage here needs MITgcm.
_LOCOMGCM_ONLY_STAGES: frozenset[str] = frozenset(
    {
        "stage1a_prepare_common",
        "stage1b_export_to",
        "stage2_torch_run",
        "stage4_postproc",
    }
)

# : stage names retained in `--mitgcm-only` mode. The LOCOMgcm model run
# (stage1b export + stage2 LOCOMgcm run) and LOCOMgcm-vs-MIT compare (stage5) are skipped.
# Stages that exist only for the MITgcm comparison.
_MITGCM_STAGES: frozenset[str] = frozenset(
    {"stage1c_export_mit", "stage3_mit_run", "stage5_compare"}
)

_MITGCM_ONLY_STAGES: frozenset[str] = frozenset(
    {
        "stage1a_prepare_common",
        "stage1c_export_mit",
        "stage3_mit_run",
        "stage4_postproc",
    }
)


def topological_sort(stages: Sequence[StageEntry], requested: set[str] | None = None) -> list[StageEntry]:
    """Return stages in topo order, optionally filtered to requested subset."""
    by_name = {s.name: s for s in stages}
    order: list[StageEntry] = []
    placed: set[str] = set()

    def visit(s: StageEntry) -> None:
        if s.name in placed:
            return
        for prereq in s.prerequisites:
            if prereq in by_name:
                visit(by_name[prereq])
        placed.add(s.name)
        order.append(s)

    for s in stages:
        visit(s)

    if requested:
        return [s for s in order if s.name in requested]
    return order


def parse_stages_arg(stages_str: str) -> set[str]:
    """Parse `--stages 1a,1b,1c,3,4,5` into set of stage names."""
    if stages_str == "all":
        return set()
    parts = [p.strip() for p in stages_str.split(",")]
    return {_NAME_ALIASES.get(p, p) for p in parts}


def _transitive_dependents(seed: set[str], stages: Sequence[StageEntry]) -> set[str]:
    """All stages that (transitively) list any of ``seed`` as a prerequisite.

    Used for STALE propagation: when the stages in ``seed`` re-run, every
    stage downstream of them is now potentially out of date.
    """
    rev: dict[str, set[str]] = {}
    for s in stages:
        for p in s.prerequisites:
            rev.setdefault(p, set()).add(s.name)
    out: set[str] = set()
    frontier = set(seed)
    while frontier:
        nxt: set[str] = set()
        for name in frontier:
            for dep in rev.get(name, ()):
                if dep not in out:
                    out.add(dep)
                    nxt.add(dep)
        frontier = nxt
    return out


def _is_run_status_writer() -> bool:
    """Only global rank 0 writes run-level manifest status in distributed runs."""

    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID"))
    return rank in (None, "", "0")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", dest="sets", default=[])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--stages",
        default=None,
        help="comma-separated stage names or 'all'. Default: 1a,1b,2,4 when "
             "mit_comparison.enabled is false, every stage when it is true.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--persist-intermediate", action="store_true")
    parser.add_argument(
        "--mitgcm-only",
        action="store_true",
        help="MITgcm-only mode (legacy --mitgcm-only equivalent): skip stage1b "
             "+ stage2 + stage5; run only stage1a + stage1c + stage3 + stage4. "
             "Stage4 gets --skip-prereq stage2_torch_run; stage1a flips "
             "metadata.yaml description to 'MITgcm-only run: <name>'.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger("run_pipeline")

    # A configuration that does not load fails the run before any stage
    # starts; otherwise a stage would fail on the same error while the
    # previous outputs and manifest stay in place. Loading only reads files,
    # so --dry-run loads it too: the default stage set depends on it.
    from locomgcm.io.runner.config import load_run_config

    try:
        run_cfg = load_run_config(args.config, args.overlay, args.sets)
    except Exception as e:  # noqa: BLE001 (any load error ends the run)
        logger.error(f"configuration does not load: {e}")
        return 5

    if args.stages is not None:
        requested = parse_stages_arg(args.stages)
    elif args.mitgcm_only or run_cfg.mit_comparison.enabled:
        requested = set()
        why = "--mitgcm-only is set" if args.mitgcm_only else "mit_comparison.enabled is true"
        logger.info(f"orchestrator: --stages not given and {why}; selecting every stage")
    else:
        requested = set(_LOCOMGCM_ONLY_STAGES)
        logger.info(
            "orchestrator: --stages not given and mit_comparison.enabled is false; "
            "running the LOCOMgcm-only stages 1a,1b,2,4"
        )
    ordered_stages = topological_sort(STAGES, requested=requested if requested else None)
    # : when `--mitgcm-only` is set, intersect the (already-resolved) stage
    # list with the MIT-only allow-set. This makes `--mitgcm-only --stages all`
    # behave like the legacy mode, and makes `--mitgcm-only --stages 1a,1b,1c`
    # still skip 1b (the orchestrator-level filter wins over an explicit
    # --stages request — matching the legacy "MIT-only branch supersedes the
    # rest of the run" semantics).
    if args.mitgcm_only:
        before = [s.name for s in ordered_stages]
        ordered_stages = [s for s in ordered_stages if s.name in _MITGCM_ONLY_STAGES]
        skipped = [n for n in before if n not in _MITGCM_ONLY_STAGES]
        if skipped:
            logger.info(f"orchestrator: --mitgcm-only filters out {skipped}")
    # The MITgcm stages cannot run with the comparison off (stage 1a writes no
    # MITgcm parameters then). Refuse now rather than after 1a and 1b have run.
    mit_requested = [s.name for s in ordered_stages if s.name in _MITGCM_STAGES]
    if mit_requested and not run_cfg.mit_comparison.enabled:
        logger.error(
            f"the MITgcm stages {mit_requested} need mit_comparison.enabled=true; "
            "set it (for example --set mit_comparison.enabled=true) or leave them out"
        )
        return 5
    logger.info(f"orchestrator: will run stages {[s.name for s in ordered_stages]}")

    # Build stage argv (passed verbatim except --stages stripped)
    base_argv = [
        "--config", str(args.config),
        "--run-id", args.run_id,
    ]
    for o in args.overlay:
        base_argv += ["--overlay", str(o)]
    for s in args.sets:
        base_argv += ["--set", s]
    if args.output:
        base_argv += ["--output", str(args.output)]
    if args.dry_run:
        base_argv.append("--dry-run")
    if args.persist_intermediate:
        base_argv.append("--persist-intermediate")
    if args.force:
        base_argv.append("--force")
    if args.mitgcm_only:
        # Propagate to each stage CLI (stage1a uses it for the metadata
        # description flip; other stages no-op via build_stage_cli_argparser).
        base_argv.append("--mitgcm-only")

    # Run-level status lifecycle: mark overall_status RUNNING now, COMPLETED /
    # FAILED at the end. Skipped under --dry-run so probe runs never touch disk
    # (and so the monkeypatched orchestrator tests stay side-effect free).
    open_manifest_for_stage = None
    orch_manifest = None
    if not args.dry_run and _is_run_status_writer():
        try:
            from locomgcm.io.runner.stages._common import (
                open_manifest_for_stage as _open_manifest_for_stage,
            )
            open_manifest_for_stage = _open_manifest_for_stage
            orch_manifest = open_manifest_for_stage(args, run_cfg)
            orch_manifest.set_overall_status("RUNNING")
        except Exception as e:  # noqa: BLE001 — status tracking is best-effort
            logger.warning(f"orchestrator: overall_status tracking disabled ({e})")
            open_manifest_for_stage = None
            orch_manifest = None

    for stage in ordered_stages:
        logger.info(f"=== invoking {stage.name} ===")
        try:
            module = importlib.import_module(stage.cli_module)
        except ImportError as e:
            logger.error(f"cannot import {stage.cli_module}: {e}")
            return 1
        stage_argv = list(base_argv)
        # : in --mitgcm-only mode, stage4 must skip the (absent) LOCOMgcm-side
        # prereq stage2_torch_run; stage4's prereq list is
        # ["stage2_torch_run", "stage3_mit_run"], so we pass an explicit
        # --skip-prereq for the LOCOMgcm half.
        if args.mitgcm_only and stage.name == "stage4_postproc":
            stage_argv += ["--skip-prereq", "stage2_torch_run"]
        rc = module.main(stage_argv)
        if rc != 0:
            logger.error(f"{stage.name} failed with exit code {rc}; halting pipeline")
            if orch_manifest is not None:
                try:
                    orch_manifest.set_overall_status("FAILED")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"orchestrator: could not set overall_status=FAILED ({e})")
            return rc

    # Post-run: mark any downstream stage that was COMPLETED but whose upstream
    # just re-ran this invocation as STALE (Make-style invalidation), then
    # finalize overall_status. mark_stale was previously defined but never
    # called from production code.
    if orch_manifest is not None and open_manifest_for_stage is not None:
        try:
            ran = {s.name for s in ordered_stages}
            dependents = _transitive_dependents(ran, STAGES) - ran
            fresh = open_manifest_for_stage(args, run_cfg)  # fresh read post-stages
            for dep in sorted(dependents):
                if fresh.get_stage_status(dep) == "COMPLETED":
                    fresh.mark_stale(dep)
                    logger.info(
                        f"orchestrator: {dep} marked STALE (upstream re-ran; rerun it to refresh)"
                    )
            fresh.set_overall_status("COMPLETED")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"orchestrator: post-run status update failed ({e})")

    logger.info("orchestrator: all stages completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
