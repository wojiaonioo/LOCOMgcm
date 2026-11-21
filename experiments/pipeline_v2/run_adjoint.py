"""First-step adjoint trace orchestrator entrypoint.

The entrypoint translates the new-schema yaml into a flat legacy yaml. The
first-step adjoint trace itself has no execution path in ``pipeline_v2``:
calling :func:`main` writes ``_legacy_adjoint_config.yaml`` and then raises
:class:`NotImplementedError`.

Usage (writes legacy yaml only; trace is NOT executed)::

    python -m experiments.pipeline_v2.run_adjoint \
        --config experiments/configs/cases/pureocean.yaml \
        [--overlay OVERLAY.yaml]... \
        [--set key.path=value]... \
        [--output OUTPUT_DIR] \
        [--run-id adjoint]

The yaml translation goes through
:func:`locomgcm.io.runner.legacy_bridge.translate_to_legacy_flat_yaml`
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

from locomgcm.io.runner.config import load_run_config
from locomgcm.io.runner.config._loader import merge_yaml_layers
from locomgcm.io.runner.legacy_bridge import (
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

    # The first-step adjoint trace has no execution path in pipeline_v2: it
    # needs its own stage or a peer entry point that consumes the legacy_yaml
    # scaffold above. Stop here with that gap named.
    raise NotImplementedError(
        "first-step-adjoint-trace execution is not implemented in pipeline_v2. "
        "The yaml translation scaffold has been written to "
        f"{legacy_yaml}."
    )


if __name__ == "__main__":
    sys.exit(main())
