from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import ClassVar

import torch

from defgcm.geometry import StructuredCGrid
from defgcm.state import DiagnosticState, PrognosticState, StateTendency


class TendencyStepDomain(StrEnum):
    FAST = "fast"
    SLOW = "slow"
    # Called by the stepper inside its momentum predictor, after the slow
    # tendencies are extrapolated; never part of a tendency assembly.
    PREDICTOR = "predictor"
    # Called by the stepper on the updated tracers, where MITgcm's
    # TRACERS_CORRECTION_STEP adjusts them; never part of a tendency assembly.
    TRACER_CORRECTION = "tracer_correction"


class TendencyPhase(StrEnum):
    """Classification for staggered time stepping.

    MITgcm ``staggerTimeStep=.TRUE.`` evaluates momentum tendencies
    BEFORE the pressure solve and tracer tendencies AFTER velocity
    correction.  MOMENTUM outputs are evaluated with the old velocity u^n;
    TRACER outputs with the pressure-corrected velocity u^{n+1}.
    A module producing both declares a phase for each output.
    """
    MOMENTUM = "momentum"
    TRACER = "tracer"


class MomentumTermClass(StrEnum):
    """Physical class of a momentum term, independent of time-step placement."""

    DYNAMICS = "dynamics"
    DISSIPATION = "dissipation"
    BOTTOM_DRAG = "bottom_drag"
    FORCING = "forcing"
    PRESSURE_GRADIENT = "pressure_gradient"


class TendencyModule(ABC):
    name: str
    stepper_domain: TendencyStepDomain = TendencyStepDomain.FAST
    tendency_phase: TendencyPhase = TendencyPhase.MOMENTUM
    momentum_term_class: ClassVar[MomentumTermClass] = MomentumTermClass.DYNAMICS

    @property
    def output_phases(self) -> dict[str, TendencyPhase]:
        """Output placement; single-phase modules retain their legacy label."""
        return dict.fromkeys(("eta", "u", "v", "temp", "salt"), self.tendency_phase)

    def select_tendency_phase(
        self, tendency: StateTendency, phase: TendencyPhase,
    ) -> StateTendency:
        """Retain only outputs consumed at this evaluation phase."""
        phases = self.output_phases
        return StateTendency(
            eta=tendency.eta if phases.get("eta") == phase else torch.zeros_like(tendency.eta),
            u=tendency.u if phases.get("u") == phase else torch.zeros_like(tendency.u),
            v=tendency.v if phases.get("v") == phase else torch.zeros_like(tendency.v),
            temp=tendency.temp if phases.get("temp") == phase else torch.zeros_like(tendency.temp),
            salt=tendency.salt if phases.get("salt") == phase else torch.zeros_like(tendency.salt),
        )

    @abstractmethod
    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        raise NotImplementedError

    def __call__(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        return self.evaluate(state, diagnostics, grid)

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "DynamicsModule":
        del device, dtype
        return self

    def populate_diagnostic_state(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> None:
        del state, diagnostics, grid


DynamicsModule = TendencyModule
