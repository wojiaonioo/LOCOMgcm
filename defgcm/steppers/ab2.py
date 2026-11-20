"""Adams-Bashforth 2nd order time stepper.

Matches MITgcm's timestep.F AB2 path: single tendency evaluation at the
current state, followed by linear extrapolation of the combined tendency.
The key difference from SSPRK3 is that nonlinear terms (KE gradient, etc.)
are evaluated **once** at u^n and never re-evaluated at intermediate states.

MITgcm flow (timestep.F):
  ① evaluate tendencies at u^n  → G^n
  ② AB2 extrapolate: G_AB = (1 + ab_fac) · G^n − ab_fac · G^{n-1}
  ③ update: u^{n+1} = u^n + dt · G_AB

where ab_fac = 0.5 + ab_eps (default ab_eps=0.1 → ab_fac=0.6).
First step falls back to forward Euler (no G^{n-1} available).

Cross-validated with MITgcm source via implementation:
  - adams_bashforth2.F:60-65,80-87: AB2 algorithm and first-step fallback
  - timestep.F:174-186: AB2 call on combined gU tendency
  - set_defaults.F:317: default abEps=0.01 (our config uses 0.1)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from defgcm.state import PrognosticState, StateTendency

from .base import StepperModelContext, TimeStepper, tracer_boundary_phases
from defgcm.steppers._registry import register_stepper


_AB2_PREV_KEY = "_ab2_prev"
_AB2_META_KEY = "_ab2_prev_meta"


def _tendency_signature(tendency: StateTendency) -> tuple[tuple[str, tuple[int, ...], str, str], ...]:
    return tuple(
        (
            name,
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
        )
        for name, tensor in (
            ("eta", tendency.eta),
            ("u", tendency.u),
            ("v", tendency.v),
            ("temp", tendency.temp),
            ("salt", tendency.salt),
        )
    )


@dataclass(frozen=True)
@register_stepper("ab2")
class AB2Stepper(TimeStepper):
    """Adams-Bashforth 2nd order time stepper.

    Parameters
    ----------
    ab_eps : float
        Damping parameter for 2Δt oscillation suppression.
        AB2 weights become: (1 + 0.5 + ab_eps) for G^n,
        −(0.5 + ab_eps) for G^{n-1}.
        MITgcm default is 0.01; our apple-to-apple config uses 0.1.
    """
    ab_eps: float = 0.1
    name: str = "ab2"

    # Mutable AB2 tendency history inside the frozen dataclass.
    _cache: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        ab_eps = float(self.ab_eps)
        if not math.isfinite(ab_eps) or ab_eps < 0.0:
            raise ValueError(f"ab_eps must be finite and non-negative, got {self.ab_eps}")
        object.__setattr__(self, "ab_eps", ab_eps)

    def reset_history(self) -> None:
        """Clear cached AB2 tendency history before starting an independent run."""
        self._cache.pop(_AB2_PREV_KEY, None)
        self._cache.pop(_AB2_META_KEY, None)

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
        """Advance one time step with Adams-Bashforth 2nd order.

        MITgcm equivalent: the tendency evaluation + AB2 extrapolation
        portion of timestep.F, without the surface pressure gradient
        (which is handled by the fast stepper in split-explicit mode).
        """
        model.prepare_step_time(time_seconds)
        project, apply_tracers = tracer_boundary_phases(model, state, dt)
        bounded = project(state, record_budget=False, step_entry=True)

        # ── Single evaluation at u^n ──
        # MITgcm: mom_vecinv → gU (all slow tendencies evaluated once)
        tendency, diagnostics = model.evaluate_tendency_with_diagnostics(bounded)
        model.accumulate_tendency_diagnostics(diagnostics, dt)

        # ── AB2 extrapolation ──
        # MITgcm timestep.F:174  CALL ADAMS_BASHFORTH2(gU, ...)
        # G_AB = G^n + ab_fac * (G^n - G^{n-1})
        #      = (1 + ab_fac) * G^n - ab_fac * G^{n-1}
        # where ab_fac = 0.5 + ab_eps
        signature = _tendency_signature(tendency)
        prev_meta = self._cache.get(_AB2_META_KEY)
        history_valid = (
            _AB2_PREV_KEY in self._cache
            and prev_meta is not None
            and prev_meta.get("dt") == float(dt)
            and prev_meta.get("signature") == signature
        )
        if history_valid:
            prev = self._cache[_AB2_PREV_KEY]
            ab_fac = 0.5 + self.ab_eps
            # Extrapolate each component individually (StateTendency has
            # no __sub__, so we compute per-field like ImplicitFreeSurfaceStepper)
            tend_eta = tendency.eta + ab_fac * (tendency.eta - prev["eta"])
            tend_u = tendency.u + ab_fac * (tendency.u - prev["u"])
            tend_v = tendency.v + ab_fac * (tendency.v - prev["v"])
            tend_temp = tendency.temp + ab_fac * (tendency.temp - prev["temp"])
            tend_salt = tendency.salt + ab_fac * (tendency.salt - prev["salt"])
            extrapolated = StateTendency(
                eta=tend_eta, u=tend_u, v=tend_v,
                temp=tend_temp, salt=tend_salt,
            )
        else:
            # First step: forward Euler (no G^{n-1})
            # MITgcm: adams_bashforth2.F:60-65 sets AB factor to 0
            extrapolated = tendency

        # Save current tendency for next step's AB2
        # History is part of the differentiated rollout: the next step reads
        # it with weight -ab_fac. Clone its storage without cutting that edge.
        self._cache[_AB2_PREV_KEY] = {
            "eta": tendency.eta.clone(),
            "u": tendency.u.clone(),
            "v": tendency.v.clone(),
            "temp": tendency.temp.clone(),
            "salt": tendency.salt.clone(),
        }
        self._cache[_AB2_META_KEY] = {
            "dt": float(dt),
            "signature": signature,
        }

        # ── Apply tendency ──
        # MITgcm timestep.F:375  gU(i,j) = uVel + dt*gU
        updated = extrapolated.apply(bounded, dt)

        model.prepare_step_time(time_seconds + dt)
        return apply_tracers(updated, record_budget=False)
