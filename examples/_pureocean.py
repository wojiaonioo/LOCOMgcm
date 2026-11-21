"""Shared setup for the pureocean examples.

``prepare`` writes the synthetic raw inputs and runs pipeline stages 1a and 1b
once; ``build`` turns the prepared run directory into a model and its initial
state with the same function that pipeline stage 2 uses.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    # Running ``python examples/<script>.py`` puts only examples/ on sys.path.
    sys.path.insert(0, str(REPO))

CASE = REPO / "experiments" / "configs" / "cases" / "pureocean.yaml"
EXPERIMENT = "pureocean_example"
DEFAULT_WORK = REPO / "outputs" / "examples"


def config_sets(work: Path, steps: int | None = None) -> list[str]:
    """``--set`` overrides that point pureocean at CPU and the synthetic inputs."""
    inputs = work / "synthetic_inputs"
    sets = [
        "run.device=cpu",
        f"run.experiment_name={EXPERIMENT}",
        f"cache.root={work / 'cache'}",
        f"sources.hycom.path={inputs / 'hycom'}",
        f"sources.era5.path={inputs / 'era5'}",
        f"sources.gebco.path={inputs / 'gebco.nc'}",
    ]
    if steps is not None:
        sets.append(f"time.steps={steps}")
    return sets


def _stage_completed(run_dir: Path, stage: str) -> bool:
    manifest = run_dir / "manifest.yaml"
    if not manifest.is_file():
        return False
    stages = (yaml.safe_load(manifest.read_text()) or {}).get("stages", {})
    return (stages.get(stage) or {}).get("status") == "COMPLETED"


def prepare(work: Path, steps: int):
    """Return ``(run_cfg, run_dir)`` for pureocean with prepared inputs under ``work``."""
    from experiments.pipeline_v2.run_pipeline import main as run_pipeline
    from locomgcm.io.runner.config import load_run_config

    import make_synthetic_inputs

    work = work.resolve()
    inputs = work / "synthetic_inputs"
    if not (inputs / "gebco.nc").is_file():
        make_synthetic_inputs.main(["--out", str(inputs)])
    output_root = work / "data"
    run_dir = output_root / EXPERIMENT
    if not _stage_completed(run_dir, "stage1b_export_to"):
        argv = ["--config", str(CASE), "--run-id", EXPERIMENT,
                "--output", str(output_root), "--stages", "1a,1b"]
        for item in config_sets(work):
            argv += ["--set", item]
        if run_pipeline(argv) != 0:
            raise SystemExit("pipeline stages 1a,1b failed")
    return load_run_config(CASE, (), config_sets(work, steps)), run_dir


def build(run_cfg, run_dir: Path, steps: int):
    """Return ``(model, state, time_step_bundle)`` exactly as stage 2 builds them."""
    from experiments.cases import build_model

    return build_model(
        run_cfg, run_dir, steps=steps, logger=logging.getLogger("pureocean_example")
    )
