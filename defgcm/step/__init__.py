"""The three phases of one time step and the contract between them.

    prepare (eager, :mod:`~defgcm.step.plan`)
        -> kernel (traced, ``model.step_kernel``)
        -> finalize (eager, :mod:`~defgcm.step.finalize`)

The boundary separates host preparation from tensor computation and deferred
observation. Verification follows the signed matrix in ``docs/DEVELOPMENT.md``.
"""

from __future__ import annotations

from .contract import (
    DECLARED_EXITS,
    EMPTY_MAPPING,
    BoundaryPayload,
    ClockSample,
    ComponentSample,
    HostPlan,
    HostTimes,
    ShapeContract,
    StepDiagnostics,
    StepOutcome,
    StepPlan,
    StepTimes,
    declared_exit_names,
)
from .carry import OceanCarry, WindowMode, initialize, advance, finalize_step
from .boundary_history import BoundaryHistory
from .plan import build_step_plan

__all__ = [
    "OceanCarry",
    "BoundaryHistory",
    "WindowMode",
    "initialize",
    "advance",
    "BoundaryPayload",
    "ClockSample",
    "ComponentSample",
    "DECLARED_EXITS",
    "EMPTY_MAPPING",
    "HostPlan",
    "HostTimes",
    "ShapeContract",
    "StepDiagnostics",
    "StepOutcome",
    "StepPlan",
    "StepTimes",
    "build_step_plan",
    "declared_exit_names",
    "finalize_step",
]
