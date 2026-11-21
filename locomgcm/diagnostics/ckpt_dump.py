"""Opt-in eager checkpoint dumps for step-internal tensor comparison."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from locomgcm.accel.graph_exit import graph_exit
import torch.distributed as dist


_DUMP_DIR = os.environ.get("LOCOMGCM_CKPT_DUMP_DIR")
ENABLED = bool(_DUMP_DIR)
_DUMP_STEPS = (
    frozenset(
        int(value.strip())
        for value in os.environ.get("LOCOMGCM_CKPT_DUMP_STEPS", "2").split(",")
        if value.strip()
    )
    if ENABLED
    else frozenset()
)
# Optional record-name allowlist (comma list).  Empty/unset = every record.  Full
# 3-D state records cost ~48 GB per two-rank job at N=960; station-level probes
# (the 2-D cg2d records) only need their own names.
_DUMP_NAMES = (
    frozenset(
        value.strip()
        for value in os.environ.get("LOCOMGCM_CKPT_DUMP_NAMES", "").split(",")
        if value.strip()
    )
    if ENABLED
    else frozenset()
)
_current_step: int | None = None
_sequence = 0
_captured: dict[str, dict[str, torch.Tensor]] = {}


def begin_step(step_index: int) -> None:
    """Select the completed-step index used by subsequent checkpoint records."""
    global _current_step, _sequence
    _current_step = step_index
    _sequence = 0
    _captured.clear()


def current_step() -> int | None:
    """Return the completed-step index selected by :func:`begin_step`."""
    return _current_step


def is_dump_step() -> bool:
    """Return whether records are enabled for the selected step."""
    return ENABLED and _current_step in _DUMP_STEPS


# Graph-exit discipline: the public entry points are
# traced (a call to a ``torch._dynamo.disable``d function is a graph break even
# when it returns immediately), and only the actual dump work -- reached solely
# when a record is due -- runs behind the disabled boundary, inside a declared
# ``graph_exit()``.  With ``ENABLED`` False (production) Dynamo constant-folds
# the early return and the compiled step contains no call at all.


def capture(name: str, **tensors: torch.Tensor) -> None:
    """Retain helper-internal tensors until their ordered call site records them."""
    if ENABLED:
        _captured[name] = tensors


def captured(name: str) -> dict[str, torch.Tensor]:
    """Return and remove tensors retained at a nested helper boundary."""
    return _captured.pop(name)


def record(name: str, **tensors: torch.Tensor) -> None:
    """Synchronize and save one checkpoint when the current step is selected."""
    if not ENABLED or _current_step not in _DUMP_STEPS:
        return
    if _DUMP_NAMES and name not in _DUMP_NAMES:
        return
    with graph_exit():
        _record_impl(name, **tensors)


@torch._dynamo.disable
def _record_impl(name: str, **tensors: torch.Tensor) -> None:
    global _sequence

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    destination = Path(_DUMP_DIR) / f"rank{rank}"
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: tensor.detach().to("cpu", torch.float64).numpy()
        for key, tensor in tensors.items()
    }
    path = destination / (
        f"step{_current_step:05d}_{_sequence:03d}_{name}.npz"
    )
    np.savez(path, **arrays)
    _sequence += 1
