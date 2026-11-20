from __future__ import annotations

from dataclasses import dataclass

from defgcm.state import PrognosticState, blend_states

from .base import StepperModelContext, TimeStepper, tracer_boundary_phases
from defgcm.steppers._registry import register_stepper


@dataclass(frozen=True)
@register_stepper("ssprk3")
class SSPRK3Stepper(TimeStepper):
    name: str = "ssprk3"

    def step(
        self,
        state: PrognosticState,
        model: StepperModelContext,
        dt: float,
        *,
        time_seconds: float = 0.0,
    ) -> PrognosticState:
        """Freeze the accelerator gate snapshot for this step.

        ``defgcm.accel.gate`` resolves the whole registry from the
        environment on every call when no snapshot is active, which puts an
        18-entry re-scan on hot consult sites (e.g. the CG3D preconditioner is
        consulted once per PCG iteration).  Opening the step-scoped snapshot
        here is the "read once per step, frozen" rule: every consumer in
        this step reads the same frozen resolution, and the env becomes
        effective only from the next step.  Nested stepper ``step`` calls
        (split-explicit slow/fast) re-enter the active snapshot instead of
        resolving again.
        """
        from defgcm.accel import step_snapshot

        with step_snapshot():
            return self._step_impl(state, model, dt, time_seconds=time_seconds)

    def _step_impl(
        self,
        state: PrognosticState,
        model: StepperModelContext,
        dt: float,
        *,
        time_seconds: float = 0.0,
    ) -> PrognosticState:
        model.prepare_step_time(time_seconds)
        project, apply_tracers = tracer_boundary_phases(model, state, dt)
        stage0 = project(state, record_budget=False, step_entry=True)

        tendency1, diagnostics1 = model.evaluate_tendency_with_diagnostics(stage0)
        model.accumulate_tendency_diagnostics(diagnostics1, dt / 6.0)

        model.prepare_step_time(time_seconds + dt)
        stage1 = project(tendency1.apply(stage0, dt), record_budget=False)
        tendency2, diagnostics2 = model.evaluate_tendency_with_diagnostics(stage1)
        model.accumulate_tendency_diagnostics(diagnostics2, dt / 6.0)
        stage2_forward = project(tendency2.apply(stage1, dt), record_budget=False)

        model.prepare_step_time(time_seconds + 0.5 * dt)
        stage2 = project(blend_states(stage0, stage2_forward, 0.75, 0.25), record_budget=False)
        tendency3, diagnostics3 = model.evaluate_tendency_with_diagnostics(stage2)
        model.accumulate_tendency_diagnostics(diagnostics3, 2.0 * dt / 3.0)

        model.prepare_step_time(time_seconds + dt)
        stage3_forward = project(tendency3.apply(stage2, dt), record_budget=False)
        return apply_tracers(blend_states(stage0, stage3_forward, 1.0 / 3.0, 2.0 / 3.0))
