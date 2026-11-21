from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Protocol

from locomgcm.state import DiagnosticState, PrognosticState, StateTendency


@dataclass(frozen=True)
class TimeStepBundle:
    """Resolved 4-deltaT values passed through from the config.

    Concrete steppers that do not use the bundle stay on their legacy scalar
    path.  Steppers that do consume these resolved fields directly:
      - dt_mom: momentum tendency time step
      - dt_tracer: tracer tendency time step, future dTtracerLev(k) hook
      - dt_freesurf: free-surface eta and Helmholtz coefficient time step
      - dt_clock: model clock, sampling, IO, and CFL time step
    """

    dt: float
    dt_mom: float
    dt_tracer: float
    dt_freesurf: float
    dt_clock: float

    def __post_init__(self) -> None:
        for name in ("dt", "dt_mom", "dt_tracer", "dt_freesurf", "dt_clock"):
            raw = getattr(self, name)
            value = float(raw)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {raw}")

    @classmethod
    def all_equal(cls, dt: float) -> "TimeStepBundle":
        return cls(
            dt=dt,
            dt_mom=dt,
            dt_tracer=dt,
            dt_freesurf=dt,
            dt_clock=dt,
        )


def coerce_time_step_bundle(dt: float | TimeStepBundle) -> TimeStepBundle:
    if isinstance(dt, TimeStepBundle):
        return dt
    return TimeStepBundle.all_equal(float(dt))


class StepperModelContext(Protocol):
    def prepare_step_time(self, time_seconds: float) -> None:
        ...

    def evaluate_tendency(self, state: PrognosticState) -> StateTendency:
        ...

    def evaluate_tendency_with_diagnostics(self, state: PrognosticState) -> tuple[StateTendency, DiagnosticState]:
        ...

    def accumulate_tendency_diagnostics(self, diagnostics: DiagnosticState, scale: float) -> None:
        ...

    def enforce_boundaries(
        self, state: PrognosticState, *, record_budget: bool = True, initial: bool = False,
        step_entry: bool = False,
    ) -> PrognosticState:
        ...

    def freeze_boundary_targets(self, state: PrognosticState) -> object | None:
        ...


def tracer_boundary_phases(model, state, dt_tracer, *, evaluation=None):
    """Boundary projection at internal stages and tracer advance at the physical step.

    A step calls this once at entry, after its entry samples are installed, on
    its start state.  The Flather velocity targets are evaluated here from that
    state and bound into both phases, so every boundary application of the step
    reuses them and they end with the step (OBCS_CALC once per step,
    OBCS_APPLY_UV several times).
    """
    enforce = model.enforce_boundaries
    if evaluation is not None:
        enforce = partial(enforce, boundary_evaluation=evaluation)
    flather_targets = model.freeze_boundary_targets(state)
    if flather_targets is not None:
        enforce = partial(enforce, flather_targets=flather_targets)
    if getattr(getattr(model, "boundary_condition", None), "uses_orlanski", False):
        # OBCS_CALC reads the step-start fields (do_oceanic_phys.F:322); every
        # application of the step evaluates the same values from them.
        enforce = partial(enforce, orlanski_source=state)
    if getattr(getattr(model, "boundary_condition", None), "uses_stevens", False):
        return (partial(enforce, advance_stevens=False),
                partial(enforce, stevens_source=(state, dt_tracer)))
    return enforce, enforce


class TimeStepper(ABC):
    name: str
    #: Fullgraph model_step requires host sampling outside the numerical kernel.
    #: Legacy steppers sample Python clocks inside step and must run eagerly.
    supports_fullgraph_model_step: bool = False
    #: Tendency domains this stepper actually evaluates (``TendencyStepDomain``
    #: values).  The model refuses to assemble a module whose domain is not in
    #: this set instead of silently never evaluating it (a
    #: stepper owns the placement of physics, a module only declares it).
    #: SSPRK3 and the split-explicit stepper evaluate both; the implicit
    #: free-surface family evaluates the slow (AB2) domain only, which is where
    #: MITgcm puts every explicit momentum term (mom_vecinv.F bottom drag).
    #: A stepper lists ``predictor`` only if its step calls those modules.
    evaluated_domains: frozenset[str] = frozenset({"fast", "slow"})
    #: Registry names of modules this stepper solves for itself (e.g. the
    #: implicit free surface replaces the explicit barotropic continuity and
    #: pressure-gradient modules), so their absence from the evaluated domains
    #: is the design, not a dropped term.
    superseded_module_names: frozenset[str] = frozenset()

    @abstractmethod
    def step(
        self,
        state: PrognosticState,
        model: StepperModelContext,
        dt: float | TimeStepBundle,
        *,
        time_seconds: float = 0.0,
    ) -> PrognosticState:
        raise NotImplementedError

    def reset_history(self) -> None:
        """Release step-to-step values before an independent run.

        ``runner.initialize_fresh_step_state`` calls this on every stepper of
        every fresh run; a resume keeps the history.  A stepper that carries a
        value from one step into the next (AB2's previous tendency, the nested
        steppers of split-explicit) overrides this; a stateless stepper holds
        nothing to release.  The implicit free-surface cache is coupled to
        model-level carry and boundary state and is released by the runner
        through ``release_working_step_state``.
        """
