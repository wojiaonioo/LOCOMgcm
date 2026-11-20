"""First-step adjoint trace orchestrator entrypoint.

The spawn path is retired.

The original entrypoint translated the new-schema yaml into a flat legacy
yaml and then spawned ``python -m experiments.run_regional_comparison`` to
execute the actual first-step adjoint trace.  That legacy script is gone
(single-path universal runtime); the spawn target no longer exists. The adjoint trace itself has NOT been reimplemented
inside ``pipeline_v2`` yet.

Current status: the yaml-translation half is preserved as scaffolding for
the future ``pipeline_v2``-native reimplementation; calling :func:`main`
writes ``_legacy_adjoint_config.yaml`` and then raises
:class:`NotImplementedError` instead of attempting the (broken) spawn.

Usage (writes legacy yaml only; trace is NOT executed)::

    python -m experiments.pipeline_v2.run_adjoint \
        --config experiments/configs/cases/pureocean.yaml \
        [--overlay OVERLAY.yaml]... \
        [--set key.path=value]... \
        [--output OUTPUT_DIR] \
        [--run-id adjoint]

The yaml translation goes through
:func:`defgcm.io.runner.legacy_bridge.translate_to_legacy_flat_yaml`
with ``target="adjoint"`` (bridge tail emits ``first-step-adjoint-trace=True``).
The 9-line ``first_step_adjoint_*`` translation table in ``legacy_bridge.py``
is kept intentionally so the future reimplementation does not have to
re-derive it.

The adjoint entrypoint is intentionally NOT part of the staged pipeline's
``STAGES`` tuple; it is a peer of ``run_pipeline.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import yaml

from defgcm.io.runner.config import load_run_config
from defgcm.io.runner.config._loader import merge_yaml_layers
from defgcm.io.runner.legacy_bridge import (
    TranslationError,
    translate_to_legacy_flat_yaml,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_adjoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", dest="sets", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "override output dir (default: "
            "<run_cfg.output.root>/<experiment_name>/adjoint)"
        ),
    )
    parser.add_argument("--run-id", default="adjoint")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("run_adjoint")

    run_cfg = load_run_config(args.config, args.overlay, args.sets)
    raw_yaml = merge_yaml_layers(args.config, args.overlay, args.sets)

    if args.output is not None:
        output_dir = Path(args.output)
    else:
        output_dir = (
            Path(run_cfg.output.root)
            / run_cfg.run.experiment_name
            / "adjoint"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        flat = translate_to_legacy_flat_yaml(
            run_cfg,
            raw_yaml,
            target="adjoint",
            output_dir=output_dir,
        )
    except TranslationError as e:
        print(f"run_adjoint: translation failed: {e}", file=sys.stderr)
        return 5

    legacy_yaml = output_dir / "_legacy_adjoint_config.yaml"
    legacy_yaml.write_text(
        yaml.safe_dump(flat, sort_keys=False, default_flow_style=False)
    )
    logger.info(f"adjoint: wrote legacy yaml to {legacy_yaml}")

    #   deleted `experiments/run_regional_comparison.py`
    # — the historical spawn target for the first-step adjoint trace. Trace
    # execution must be reimplemented inside pipeline_v2 (own stage or peer
    # entrypoint that consumes the legacy_yaml scaffold above). Until then,
    # surface the gap explicitly instead of failing with ModuleNotFoundError
    # mid-spawn.
    raise NotImplementedError(
        "first-step-adjoint-trace execution is pending pipeline_v2-native "
        "reimplementation. The legacy spawn target "
        "`experiments.run_regional_comparison` was removed. "
        "The yaml translation scaffold has been written to "
        f"{legacy_yaml}; wire the trace logic onto it when reimplementing."
    )


if __name__ == "__main__":
    sys.exit(main())
