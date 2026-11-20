from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from defgcm.geometry import StructuredCGrid
from defgcm.solvers import build_implicit_diffusion_system, tridiag_solve
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from .ab2 import AB2Stepper
from .base import StepperModelContext, TimeStepper, tracer_boundary_phases
from .ssprk import SSPRK3Stepper
from defgcm.steppers._registry import register_stepper


class SplitStepperModelContext(StepperModelContext, Protocol):
    def evaluate_fast_tendency(self, state: PrognosticState) -> StateTendency:
        ...

    def evaluate_slow_tendency(self, state: PrognosticState) -> StateTendency:
        ...

    def evaluate_fast_tendency_with_diagnostics(
        self,
        state: PrognosticState,
    ) -> tuple[StateTendency, DiagnosticState]:
        ...

    def evaluate_slow_tendency_with_diagnostics(
        self,
        state: PrognosticState,
    ) -> tuple[StateTendency, DiagnosticState]:
        ...


class _SubsetContext(StepperModelContext):
    def __init__(self, model: SplitStepperModelContext, mode: str, project) -> None:
        self._model = model
        self._mode = mode
        self._project = project

    def evaluate_tendency(self, state: PrognosticState) -> StateTendency:
        if self._mode == "fast":
            return self._model.evaluate_fast_tendency(state)
        return self._model.evaluate_slow_tendency(state)

    def evaluate_tendency_with_diagnostics(self, state: PrognosticState) -> tuple[StateTendency, DiagnosticState]:
        if self._mode == "fast":
            return self._model.evaluate_fast_tendency_with_diagnostics(state)
        return self._model.evaluate_slow_tendency_with_diagnostics(state)

    def prepare_step_time(self, time_seconds: float) -> None:
        self._model.prepare_step_time(time_seconds)

    def accumulate_tendency_diagnostics(self, diagnostics: DiagnosticState, scale: float) -> None:
        self._model.accumulate_tendency_diagnostics(diagnostics, scale)

    def enforce_boundaries(
        self, state: PrognosticState, *, record_budget: bool = True, initial: bool = False,
        step_entry: bool = False,
    ) -> PrognosticState:
        return self._project(state, record_budget=record_budget, initial=initial,
                             step_entry=step_entry)

    def freeze_boundary_targets(self, state: PrognosticState) -> None:
        # The slow and fast substeps belong to the outer step, whose ``project``
        # already carries the targets evaluated from the outer start state.
        return None


def _apply_implicit_vertical_viscosity(
    field: torch.Tensor,
    grid: StructuredCGrid,
    dt: float,
    viscosity: float,
    recip_hFac: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply implicit vertical viscosity via tridiagonal solve.

    MITgcm equivalent: ``IMPLDIFF`` called from ``dynamics.F:587-599``
    after the Adams-Bashforth prediction step.  Solves

        (I - dt · K · ∂²/∂z²) field_new = field

    using the Thomas algorithm.

    Parameters
    ----------
    field : Tensor [B, nz, ny, nx]  — predicted velocity component
    grid : StructuredCGrid
    dt : float — time step
    viscosity : float — vertical eddy viscosity (m²/s)
    recip_hFac : Tensor [nz, ny, nx], optional — reciprocal partial-cell
        fraction (MITgcm: recip_hFacW for U, recip_hFacS for V).

    Returns
    -------
    field_new : Tensor [B, nz, ny, nx] — implicitly diffused velocity
    """
    a, b, c, d = build_implicit_diffusion_system(
        field, grid.layer_thickness, interface_spacing=grid.vertical_coordinate.interface_spacing,
        diffusivity=viscosity, dt=dt, recip_hFac=recip_hFac,
    )
    return tridiag_solve(a, b, c, d, mitgcm_order=True)


@dataclass(frozen=True)
@register_stepper("split_explicit")
class SplitExplicitStepper(TimeStepper):
    """Split-explicit time stepper with optional implicit vertical viscosity.

    Parameters
    ----------
    fast_substeps : int
        Number of fast (barotropic) substeps per slow step.
    slow_stepper, fast_stepper : TimeStepper
        Sub-steppers for slow and fast tendency domains.
    implicit_viscosity : float
        Vertical eddy viscosity for backward-Euler implicit solve (m²/s).
        Applied AFTER all slow+fast stepping, BEFORE enforce_boundaries,
        matching MITgcm's IMPLDIFF placement (dynamics.F:587-599,
        between TIMESTEP and OBCS_APPLY_UV).
        Default 0.0 disables the implicit solve.
    """
    fast_substeps: int = 4
    slow_stepper: TimeStepper = field(default_factory=SSPRK3Stepper)
    fast_stepper: TimeStepper = field(default_factory=SSPRK3Stepper)
    implicit_viscosity: float = 0.0
    implicit_vertical_viscosity: bool = True
    name: str = "split_explicit"

    def reset_history(self) -> None:
        """The slow and fast steppers own any step-to-step history."""
        self.slow_stepper.reset_history()
        self.fast_stepper.reset_history()

    def step(
        self,
        state: PrognosticState,
        model: SplitStepperModelContext,
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

    # Host orchestration around the slow step and the fast substeps.  Traced, this
    # frame was specialised on the Python clock it hands to the host sampling and
    # to the nested steppers, and recompiled on every new clock under static
    # shapes; the tensor functions it calls still compile as their own frames.
    @torch.compiler.disable(recursive=False, reason="host clock orchestration of the split step")
    def _step_impl(
        self,
        state: PrognosticState,
        model: SplitStepperModelContext,
        dt: float,
        *,
        time_seconds: float = 0.0,
    ) -> PrognosticState:
        if self.fast_substeps < 1:
            raise ValueError("fast_substeps must be positive")

        model.prepare_step_time(time_seconds)
        project, apply_tracers = tracer_boundary_phases(model, state, dt)
        slow_context = _SubsetContext(model, "slow", project)
        fast_context = _SubsetContext(model, "fast", project)

        updated = self.slow_stepper.step(
            project(state, record_budget=False, step_entry=True),
            slow_context,
            dt,
            time_seconds=time_seconds,
        )
        sub_dt = dt / self.fast_substeps
        for substep in range(self.fast_substeps):
            sub_time = time_seconds + substep * sub_dt
            updated = self.fast_stepper.step(updated, fast_context, sub_dt, time_seconds=sub_time)

        # ── Implicit vertical viscosity (MITgcm: IMPLDIFF) ──
        # Applied after all tendency stepping, before final boundary enforce.
        # MITgcm dynamics.F:587-599: IMPLDIFF(gU), IMPLDIFF(gV) on updated
        # velocity (NOT tendency), using recip_hFacW / recip_hFacS.
        if self.implicit_vertical_viscosity and self.implicit_viscosity > 0.0:
            grid = model.grid  # type: ignore[attr-defined]
            u_new = updated.u.data
            v_new = updated.v.data

            recip_hFacW = None
            recip_hFacS = None
            if grid.masks.hFacW is not None:
                hfw = grid.masks.hFacW.to(dtype=u_new.dtype, device=u_new.device)
                recip_hFacW = torch.where(hfw > 0, 1.0 / hfw, torch.zeros_like(hfw))
            if grid.masks.hFacS is not None:
                hfs = grid.masks.hFacS.to(dtype=v_new.dtype, device=v_new.device)
                recip_hFacS = torch.where(hfs > 0, 1.0 / hfs, torch.zeros_like(hfs))

            u_new = _apply_implicit_vertical_viscosity(
                u_new, grid, dt, self.implicit_viscosity,
                recip_hFac=recip_hFacW,
            )
            v_new = _apply_implicit_vertical_viscosity(
                v_new, grid, dt, self.implicit_viscosity,
                recip_hFac=recip_hFacS,
            )

            # Re-apply masks (IMPLDIFF solve may produce tiny values in dry cells)
            if grid.masks.has_3d_masks:
                u_mask = grid.masks.u_float(dtype=u_new.dtype, device=u_new.device)
                v_mask = grid.masks.v_float(dtype=v_new.dtype, device=v_new.device)
                u_new = u_new * u_mask
                v_new = v_new * v_mask

            updated = updated.with_tensors(u=u_new, v=v_new)

        model.prepare_step_time(time_seconds + dt)
        return apply_tracers(updated, record_budget=False)
