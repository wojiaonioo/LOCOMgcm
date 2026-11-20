from __future__ import annotations

from dataclasses import dataclass

import torch

from defgcm.geometry import StructuredCGrid
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyStepDomain
from defgcm.dynamics._registry import register_dynamics_module


@dataclass(frozen=True)
@register_dynamics_module("newtonian_eta_damping")
class NewtonianEtaDamping(TendencyModule):
    rate: float
    name: str = "newtonian_eta_damping"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.FAST

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        del diagnostics, grid
        rate = state.eta.data.new_tensor(self.rate)
        return StateTendency(
            eta=-rate * state.eta.data,
            u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data),
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )
