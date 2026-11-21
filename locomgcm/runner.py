from __future__ import annotations

import os
from locomgcm.accel.admission import ProgramRequirements, AdmissionRejected, admit, admit_communication
from locomgcm.accel.runtime import resolve_snapshot, selection_scope
from locomgcm.distributed.comm_adjoint import session_for_grid

from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from time import perf_counter

from typing import Any, Callable, Iterator, Literal

import torch
import torch.distributed as dist

from locomgcm.accel import graph_exit as graph_exit_module
from locomgcm.accel.compile_compat import install_compile_compat
from locomgcm.accel.cudagraph_step import mark_cudagraph_step
from locomgcm.accel.preload import preload_requested_extensions
from locomgcm.accel.tensor_context import forward_mode_active, ordinary_tensor, ordinary_tensor_mode
from locomgcm.checkpoint.state import (
    apply_decoded_step_state, borrow_step_inputs, head_step_state, initialize_step_state,
    release_working_step_state,
)
from locomgcm.dynamics.convective_adjustment import convective_adjustment_of
from locomgcm.dynamics.equation_of_state import zero_negative_salinity
from locomgcm.deferred_checks import (
    inspect_final,
    resolve_step_probe,
    require_level_supports_compile,
)
from locomgcm.diagnostics import ckpt_dump
from locomgcm.model import TorchRegionalOceanModel
from locomgcm.state import PrognosticState
from locomgcm.step import build_step_plan, finalize_step
from locomgcm.step.output_ownership import bind_output_ownership
from locomgcm.step.plan import _step_index_tensor
from locomgcm.steppers.base import TimeStepBundle, coerce_time_step_bundle


def _sync_initial_distributed_velocity_faces(
    model: TorchRegionalOceanModel,
    state: PrognosticState,
    *,
    after_boundary: bool = False,
) -> PrognosticState:
    # The sync decides from the topology which faces are shared; a single
    # process on a periodic axis has them too (numerics rule 11).
    from locomgcm.steppers.ifs import faces

    u, v = faces.sync_distributed_shared_velocity_faces(
        state.u.data, state.v.data, model.grid, after_boundary=after_boundary
    )
    if u is state.u.data and v is state.v.data:
        return state
    return state.with_tensors(u=u, v=v)


def _ordinary_run_seed(state):
    # A preceding inference window may hand its final state to this one.
    # Only inference storage needs a copy; ordinary differentiable inputs keep
    # their identity and graph.
    return state.with_tensors(**{
        name: ordinary_tensor(getattr(state, name).data)
        for name in ("eta", "u", "v", "temp", "salt")
    })


def _bound_run_seed(model, state, *, apply_boundaries=True, dt=None):
    state = _ordinary_run_seed(state)
    bounded = state
    if apply_boundaries:
        stevens = {}
        if getattr(model.boundary_condition, "uses_stevens", False):
            bundle = model.config.time_step_bundle() if dt is None else coerce_time_step_bundle(dt)
            # Match step_kernel dispatch: the legacy explicit steppers advance
            # tracers with their single scalar dt, IFS with the tracer clock.
            dt_tracer = (bundle.dt if getattr(model.stepper, "step_kernel", None) is None
                         else bundle.dt_tracer)
            stevens["stevens_source"] = (state, dt_tracer)
        if getattr(model.boundary_condition, "uses_orlanski", False):
            # OBCS_CALC(startTime) on the initial fields (obcs_init_variables.F:386).
            stevens["orlanski_source"] = state
        bounded = model.enforce_boundaries(state, initial=True, **stevens)
    return _sync_initial_distributed_velocity_faces(
        model, bounded, after_boundary=apply_boundaries
    )


@ordinary_tensor_mode()
def initialize_fresh_step_state(
    model: TorchRegionalOceanModel,
    state: PrognosticState,
    *,
    dt: float | TimeStepBundle | None = None,
    pickup_step_state: dict[str, Any] | None = None,
) -> PrognosticState:
    """Bound/synchronise a run seed and seed fresh IFS step state from it.

    Always a new start.  Every cross-step value a previous run, restore or
    window left on the model is released first -- restart history, the
    evaluated inputs, explicit carry, boundary history, the eta-derived
    nonlinear free-surface layer -- with the segment-end release, and every
    stepper's own history with ``TimeStepper.reset_history``; only the
    current-time samples the caller prepared (``prepare_step_time`` runs
    before this) are kept.  Then the initial hydrostatic potential is seeded
    from the unbounded seed (without ``pickup_step_state``, as INI_PRESSURE
    precedes the initial boundary pass), the seed is bounded and the counter
    and the lagged w are seeded from it.  A caller continuing from imported step state
    (restore, pickup) uses :func:`resume_step_state`; the caller's input
    decides which, no flag on the model or stepper does.

    A start from a MITgcm pickup is still a new run (MITgcm's first step is
    ``myIter = nIter0`` for every first-step test); ``pickup_step_state`` is
    the step state it reads (AB2 histories with their start flags, the CD
    state) as the restart codec's mapping, installed over the fresh head.
    """
    if getattr(model.stepper, "name", None) == "implicit_free_surface":
        current = borrow_step_inputs(model.stepper, obc_eta=model.stepper._obc_eta)
        release_working_step_state(model, borrowed_inputs=current)
        from locomgcm.steppers.ifs.nlfs_geometry import SurfaceGeometry, apply_surf_dr

        resting = model.stepper._surface_geometry.resting
        if resting is not None:
            # A previous run left its last surface level on the grid hFac, which
            # prepare and the seed read before the first step installs h(eta0):
            # a fresh run starts on the resting geometry, as the first run did
            # (a continuation installs its saved level instead, checkpoint/nonlinear.py
            # install_nonlinear_geometry).
            apply_surf_dr(model.grid, resting=resting, surface=resting)
            model.grid.refresh_hfac_runtime()
        object.__setattr__(model.stepper, "_surface_geometry", SurfaceGeometry(resting=resting))
        from locomgcm.steppers.ifs.nlfs_geometry import RStarGeometry, install_r_star_hfac
        from locomgcm.steppers.ifs.r_star import resting_r_star_factors

        columns = model.stepper._r_star.columns
        if columns is not None:
            # The r* run left h0Fac*rStarFac on every level: back to factor 1.
            install_r_star_hfac(model.grid, columns, resting_r_star_factors(
                columns.r_low_c, columns.depth_w, columns.depth_s))
        object.__setattr__(model.stepper, "_r_star", RStarGeometry(columns=columns))
    # Boundary history belongs to every Stevens run, including legacy steppers
    # without an IFS cache. Clear it before bounding the new seed.
    from locomgcm.step.boundary_history import boundary_adapter

    adapter = boundary_adapter(model)
    if adapter is not None:
        adapter.reset()
    # Every stepper releases its own step-to-step history (AB2's previous
    # tendency would otherwise extrapolate this run's first step).
    model.stepper.reset_history()
    state = _ordinary_run_seed(state)
    if pickup_step_state is None and getattr(model.stepper, "name", None) == "implicit_free_surface":
        # MITgcm's INI_PRESSURE (fresh starts only, ini_fields.F:27-33) runs in
        # INI_FIELDS (initialise_varia.F:225) on the initial fields, before
        # PACKAGES_INIT_VARIABLES applies the open-boundary values (:263;
        # packages_init_variables.F:631; obcs_init_variables.F:401) and before
        # CONVECTIVE_ADJUSTMENT_INI (initialise_varia.F:290): the seed reads the
        # state as given, with any initial control the caller added (the
        # adjoint build's order; MITgcm adds its controls after INI_PRESSURE,
        # packages_init_variables.F:619, the same values for a zero control).
        model.stepper._seed_total_phi_hyd_picard(model, state)
    bounded = _bound_run_seed(model, state, dt=dt)
    adjustment = convective_adjustment_of(getattr(model, "dynamics_modules", ()))
    if adjustment is not None and adjustment.initial_adjustment:
        # CONVECTIVE_ADJUSTMENT_INI (initialise_varia.F:281-295) adjusts the
        # initial tracers after PACKAGES_INIT_VARIABLES' OBCS_APPLY_TS
        # (obcs_init_variables.F:401), so it follows the initial boundary pass.
        salt = bounded.salt.data
        if model.stepper.check_salinity_for_negative_values and model.grid.nz > 1:
            # convective_adjustment_ini.F:132-145 forms both densities with FIND_RHO_2D,
            # which resets negative salt first (find_rho.F:85-89); resetting the column
            # before mixing is the same, as at the step-end adjustment.
            salt = zero_negative_salinity(salt)
        temp, salt = adjustment.adjust(bounded.temp.data, salt,
                                       grid=model.grid, modules=model.dynamics_modules)
        bounded = bounded.with_tensors(temp=temp, salt=salt)
    if adapter is not None:
        # OBCS_CALC_STEVENS saved the pre-application tracer for the first step.
        adapter.history = adapter.candidate
        adapter.candidate = None
        adapter.initialization_reason = "explicit_new_run"
    initialize_step_state(model, bounded)
    if pickup_step_state is not None:
        head = head_step_state(model.stepper, snapshot=False)
        apply_decoded_step_state(model.stepper, {
            **{key: value for key, value in head.items()
               if key not in ("boundary_history", "nonlinear_geometry")},
            **pickup_step_state})
    return bounded


@ordinary_tensor_mode()
def resume_step_state(
    model: TorchRegionalOceanModel,
    state: PrognosticState,
) -> PrognosticState:
    """Use the saved step boundary without applying physical boundaries twice.

    A checkpoint already contains the accepted, bounded state. Flather's
    additive barotropic correction is not an idempotent assignment, especially
    in masked columns, so applying it here changes the next step's input.
    """
    return _bound_run_seed(model, state, apply_boundaries=False)


@dataclass(frozen=True)
class SimulationResult:
    history: list[dict[str, float]]
    snapshot_steps: list[int]
    snapshot_times: list[float]
    eta_snapshots: torch.Tensor
    u_snapshots: torch.Tensor
    v_snapshots: torch.Tensor
    temp_snapshots: torch.Tensor
    salt_snapshots: torch.Tensor
    final_state: PrognosticState
    metadata: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "history": self.history,
            "snapshot_steps": self.snapshot_steps,
            "snapshot_times": self.snapshot_times,
            "eta_snapshots": self.eta_snapshots,
            "u_snapshots": self.u_snapshots,
            "v_snapshots": self.v_snapshots,
            "temp_snapshots": self.temp_snapshots,
            "salt_snapshots": self.salt_snapshots,
            "final_state": {
                "eta": self.final_state.eta.data.detach().cpu(),
                "u": self.final_state.u.data.detach().cpu(),
                "v": self.final_state.v.data.detach().cpu(),
                "temp": self.final_state.temp.data.detach().cpu(),
                "salt": self.final_state.salt.data.detach().cpu(),
            },
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class StepTimingSummary:
    recorded_steps: int
    warmup_steps: int
    total_seconds: float
    mean_seconds: float
    last_seconds: float | None
    steps_per_second: float | None
    simulated_seconds_per_second: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "recorded_steps": self.recorded_steps,
            "warmup_steps": self.warmup_steps,
            "total_seconds": self.total_seconds,
            "mean_seconds": self.mean_seconds,
            "last_seconds": self.last_seconds,
            "steps_per_second": self.steps_per_second,
            "simulated_seconds_per_second": self.simulated_seconds_per_second,
        }


class StepTimer:
    """Track model-step wall time without synchronizing CUDA every step."""

    def __init__(
        self, *, device: torch.device | str | None = None, warmup_steps: int = 0
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.warmup_steps = warmup_steps
        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.reset()

    def reset(self, *, device: torch.device | str | None = None) -> None:
        if device is not None:
            self._device = torch.device(device)
        self._use_cuda_events = (
            self._device.type == "cuda" and torch.cuda.is_available()
        )
        self._pending_events: deque[tuple[int, torch.cuda.Event, torch.cuda.Event]] = (
            deque()
        )
        self._active_cpu_step: tuple[int, float] | None = None
        self._active_cuda_step: tuple[int, torch.cuda.Event] | None = None
        self._recorded_steps = 0
        self._total_seconds = 0.0
        self._last_seconds: float | None = None

    def start_step(self, step: int) -> None:
        if self._use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream=torch.cuda.current_stream(self._device))
            self._active_cuda_step = (step, start_event)
            return
        self._active_cpu_step = (step, perf_counter())

    def end_step(self, step: int) -> None:
        if self._use_cuda_events:
            if self._active_cuda_step is None or self._active_cuda_step[0] != step:
                raise RuntimeError(
                    "StepTimer.end_step called without a matching start_step"
                )
            start_step, start_event = self._active_cuda_step
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(stream=torch.cuda.current_stream(self._device))
            self._pending_events.append((start_step, start_event, end_event))
            self._active_cuda_step = None
            return
        if self._active_cpu_step is None or self._active_cpu_step[0] != step:
            raise RuntimeError(
                "StepTimer.end_step called without a matching start_step"
            )
        _, start_time = self._active_cpu_step
        self._active_cpu_step = None
        self._record(step, perf_counter() - start_time)

    def poll(self) -> None:
        if not self._use_cuda_events:
            return
        while self._pending_events and self._pending_events[0][2].query():
            step, start_event, end_event = self._pending_events.popleft()
            self._record(step, start_event.elapsed_time(end_event) * 1.0e-3)

    def finalize(self, *, simulated_dt: float | None = None) -> StepTimingSummary:
        if self._use_cuda_events and self._pending_events:
            torch.cuda.synchronize(self._device)
            while self._pending_events:
                step, start_event, end_event = self._pending_events.popleft()
                self._record(step, start_event.elapsed_time(end_event) * 1.0e-3)
        return self.summary(simulated_dt=simulated_dt)

    def summary(self, *, simulated_dt: float | None = None) -> StepTimingSummary:
        self.poll()
        mean_seconds = (
            self._total_seconds / self._recorded_steps if self._recorded_steps else 0.0
        )
        steps_per_second = (
            self._recorded_steps / self._total_seconds
            if self._recorded_steps and self._total_seconds > 0.0
            else None
        )
        simulated_seconds_per_second = (
            self._recorded_steps * simulated_dt / self._total_seconds
            if simulated_dt is not None
            and self._recorded_steps
            and self._total_seconds > 0.0
            else None
        )
        return StepTimingSummary(
            recorded_steps=self._recorded_steps,
            warmup_steps=self.warmup_steps,
            total_seconds=self._total_seconds,
            mean_seconds=mean_seconds,
            last_seconds=self._last_seconds,
            steps_per_second=steps_per_second,
            simulated_seconds_per_second=simulated_seconds_per_second,
        )

    def _record(self, step: int, elapsed_seconds: float) -> None:
        if step < self.warmup_steps:
            return
        self._recorded_steps += 1
        self._total_seconds += elapsed_seconds
        self._last_seconds = elapsed_seconds


class BenchmarkStepTimer(StepTimer):
    """Retain every step sample and cross-check it with one host-clock window."""

    def __init__(self, *, device: torch.device | str, expected_steps: int) -> None:
        if expected_steps < 1:
            raise ValueError("expected_steps must be positive")
        self.expected_steps = int(expected_steps)
        self.per_step: list[tuple[int, float]] = []
        self.synchronized_host_wall_seconds: float | None = None
        self._host_window_start: float | None = None
        self._ended_steps = 0
        super().__init__(device=device, warmup_steps=0)

    @property
    def uses_cuda_events(self) -> bool:
        return self._use_cuda_events

    def reset(self, *, device: torch.device | str | None = None) -> None:
        super().reset(device=device)
        self.per_step = []
        self.synchronized_host_wall_seconds = None
        self._host_window_start = None
        self._ended_steps = 0

    def start_step(self, step: int) -> None:
        if self._host_window_start is None:
            if self._use_cuda_events:
                torch.cuda.synchronize(self._device)
            self._host_window_start = perf_counter()
        super().start_step(step)

    def end_step(self, step: int) -> None:
        super().end_step(step)
        self._ended_steps += 1
        if self._ended_steps > self.expected_steps:
            raise RuntimeError("BenchmarkStepTimer observed more steps than expected")
        if self._ended_steps == self.expected_steps:
            if self._use_cuda_events:
                torch.cuda.synchronize(self._device)
            if self._host_window_start is None:
                raise RuntimeError("benchmark host window was not started")
            self.synchronized_host_wall_seconds = (
                perf_counter() - self._host_window_start
            )
            self.poll()

    def finalize(self, *, simulated_dt: float | None = None) -> StepTimingSummary:
        summary = super().finalize(simulated_dt=simulated_dt)
        if self._ended_steps != self.expected_steps:
            raise RuntimeError(
                f"BenchmarkStepTimer observed {self._ended_steps} steps; "
                f"expected {self.expected_steps}"
            )
        if len(self.per_step) != self.expected_steps:
            raise RuntimeError(
                f"BenchmarkStepTimer recorded {len(self.per_step)} samples; "
                f"expected {self.expected_steps}"
            )
        if self.synchronized_host_wall_seconds is None:
            raise RuntimeError("benchmark host window was not completed")
        return summary

    def _record(self, step: int, elapsed_seconds: float) -> None:
        self.per_step.append((int(step), float(elapsed_seconds)))
        super()._record(step, elapsed_seconds)


CompileTarget = Literal["disabled", "tendency", "model_step"]


#: Dynamo recompile limit, applied whenever the runner compiles.  One-time
#: polymorphism (per-module momentum loop, four boundary sides) needs more than
#: the default 8 variants per frame, and that is a
#: property of the model, not of the posture: the break-allowing posture keeps
#: independent top-level frames and therefore accumulates *more* variants than
#: the fullgraph one, not fewer.  Raising the limit is the answer to shape
#: polymorphism; dynamic shapes are not (they meet an Inductor unbound-symbol
#: defect and measured speed-neutral).  ``fail_on_recompile_limit_hit`` still fires
#: only under ``fullgraph``, where a per-step recompile must be a hard error.
RECOMPILE_LIMIT = 64


@dataclass(frozen=True)
class RunnerCompileConfig:
    target: CompileTarget = "disabled"
    mode: str | None = None
    fullgraph: bool = False
    # Static shapes are the posture.  Dynamic shapes bought one
    # thing -- not exhausting the recompile limit -- and ``RECOMPILE_LIMIT``
    # buys that directly, without Inductor's unbound-symbol defect.
    # ``None`` is torch's automatic-dynamic (static first, symbolic on the
    # first recompile), kept as a diagnostic so a static-shape reading has a
    # same-tree comparison.  ``True`` is refused for a differentiated program
    # (``refuse_dynamic_derivative``).
    dynamic: bool | None = False
    # With ``pin_schedule``
    # the Inductor scheduler's peak-memory reordering pass is switched off
    # while the step compiles (``_INDUCTOR_REORDER_PASSES``).  The ranks
    # compile different graphs, and that pass placed the heavy external
    # kernels on different sides of the halo p2p rounds per rank -- each such
    # kernel was then paid serially by both ranks inside the NCCL wait
    # (+47 ms/step at N=960 on two cards; 277.8 -> 169.0 with the pass off).
    # The driver sets it only on more than one rank (``LOCOMGCM_COMPILE_PIN_SCHEDULE``,
    # experiments/cases/_runtime.py ``_resolve_compile_pin_schedule``).
    pin_schedule: bool = False
    # With ``fullgraph=True`` the first step(s) run EAGERLY so the
    # one-shot host work (a resumed run's Picard seed, cold constant builds,
    # extension loads)
    # never meets the tracer; the compiled step then sees a warm model.  Same
    # numerics either way (the eager step is the reference implementation).
    eager_warmup_steps: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "mode": self.mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "pin_schedule": self.pin_schedule,
            "eager_warmup_steps": self.eager_warmup_steps,
        }


def _dynamic_dispatch(outcome: str, site: str, reason: str | None = None) -> None:
    """Record ``compile_dynamic`` when ``LOCOMGCM_COMPILE_DYNAMIC`` requested it.

    A ``RunnerCompileConfig(dynamic=True)`` built without the switch has no
    request to account for, and the ledger refuses records of a disabled switch.
    """
    from locomgcm.accel import gate, record_dispatch

    if gate("compile_dynamic").effective_value:
        record_dispatch("compile_dynamic", outcome=outcome, reason=reason, variant="dynamic",
                        site=site, backend="dynamo",
                        rank=dist.get_rank() if dist.is_initialized() else 0)


def refuse_dynamic_derivative(
    compile_config: RunnerCompileConfig | None,
    derivative_mode: Literal["none", "vjp", "jvp"],
    *,
    site: str,
) -> None:
    """Refuse a differentiated program that asks for a fully dynamic compiled step.

    ``dynamic=True`` (``LOCOMGCM_COMPILE_DYNAMIC=1``) is a forward-only diagnostic.
    ``derivative_mode`` is the program's stated intent, not the grad mode of the
    moment: a step configured under ``no_grad`` can be differentiated later
    (a checkpoint window enables gradients itself).  Called at ModelRunner.run's
    entry, at the checkpoint window's entry and in ``_configure_step_execution``.
    An f32 ``scs_paper_h160`` step compiled with ``model_step`` and
    differentiated through a checkpoint window compiled 154 graphs with 76
    graph breaks in this posture over hours of cold compile, then failed with
    an Inductor AssertionError in its gradient phase.
    ``None`` (torch's automatic-dynamic posture) is not refused.
    """
    if derivative_mode not in ("none", "vjp", "jvp"):
        raise ValueError(f"derivative_mode must be none, vjp or jvp, not {derivative_mode!r}")
    if (derivative_mode == "none" or compile_config is None
            or compile_config.target == "disabled" or compile_config.dynamic is not True):
        return
    _dynamic_dispatch("inapplicable", site,
                      "UNSUPPORTED_VJP" if derivative_mode == "vjp" else "UNSUPPORTED_JVP")
    raise ValueError(
        "LOCOMGCM_COMPILE_DYNAMIC=1 (RunnerCompileConfig.dynamic=True) is not "
        "supported for a differentiated run: compiling a differentiated step "
        "with fully symbolic shapes has taken hours and failed inside Inductor. "
        "Unset LOCOMGCM_COMPILE_DYNAMIC (or set it to 0) to compile with static "
        "shapes, or run without gradients to keep the dynamic-shape diagnostic."
    )


def refuse_strict_plain_reverse(
    compile_config: RunnerCompileConfig | None,
    derivative_mode: Literal["none", "vjp", "jvp"],
    *,
    site: str,
) -> None:
    """Refuse reverse mode on more than one rank through a strict compiled step
    without a checkpoint window.

    ModelRunner.run is the reverse path without a window.  Its communication
    session then has no window, so each halo round, sum and solve runs eagerly
    at a graph exit (``comm_adjoint`` ``_raw_boundary``, ``forward_round``).
    A compiled step admits communication only as an
    opaque in-graph operator or a declared exit; inside a checkpoint window the
    session's rounds are the in-graph operator.  Under ``fullgraph``
    (``error_on_graph_break``) an exit inside an inlined frame makes Dynamo run
    the calling frames eagerly with its frame callback still set, and the first
    generator frame the callback then meets is a hard error ("Attempt to trace
    generator", torch/_dynamo/convert_frame.py).  A non-hydrostatic
    ``tutorial_deep_convection`` step on 8 ranks (model_step, fullgraph, VJP
    without a window) raised it at step 3, the first traced step, from generator expressions in steppers/assembly_policy.py and
    dynamics/tracers.py; two ranks reproduce it and one rank (no session) does
    not.  Called at ModelRunner.run's entry, beside
    ``refuse_dynamic_derivative``.
    """
    if derivative_mode not in ("none", "vjp", "jvp"):
        raise ValueError(f"derivative_mode must be none, vjp or jvp, not {derivative_mode!r}")
    if (derivative_mode != "vjp" or compile_config is None
            or compile_config.target == "disabled" or not compile_config.fullgraph
            or not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)):
        return
    raise ValueError(
        "LOCOMGCM_COMPILE_FULLGRAPH=1 (RunnerCompileConfig.fullgraph=True) is not "
        "supported for reverse mode on more than one rank without a checkpoint "
        f"window ({site}): the communication session then runs every halo exchange "
        "eagerly at a graph exit, which the single-graph posture cannot trace "
        "(\"Attempt to trace generator\" at the first compiled step). Differentiate "
        "through a checkpoint window, or set LOCOMGCM_COMPILE_FULLGRAPH=0."
    )


#: The Inductor scheduler pass ``RunnerCompileConfig.pin_schedule`` switches
#: off.  Only the peak-memory reordering: it is the pass that placed the heavy
#: external kernels on different sides of the halo rounds per rank
#: (peak off alone 169.0 ms/step, both passes off
#: 211.2, torch defaults 277.8 -- ``reorder_for_locality`` is a net win and
#: stays on).  The pass runs when the traced step is *lowered* (the first call
#: of the compiled function), not when ``torch.compile`` wraps it, so the
#: setting has to hold for the whole step loop and is restored when the loop
#: ends.  A build without the attribute (torch 2.4) has nothing to switch.
_INDUCTOR_REORDER_PASSES = ("reorder_for_peak_memory",)


def _pin_inductor_schedule(enabled: bool) -> tuple[Callable[[], None], dict[str, bool]]:
    """Switch the reordering passes off; return (restore, applied) ."""
    applied: dict[str, bool] = {}
    if not enabled:
        return (lambda: None), applied
    import torch._inductor.config as inductor_config

    previous: dict[str, object] = {}
    for name in _INDUCTOR_REORDER_PASSES:
        if hasattr(inductor_config, name):
            previous[name] = getattr(inductor_config, name)
            setattr(inductor_config, name, False)
            applied[name] = True

    def _restore() -> None:
        for name, value in previous.items():
            setattr(inductor_config, name, value)

    return _restore, applied


def _runner_compile_cache(
    model: TorchRegionalOceanModel,
) -> dict[tuple[object, ...], object]:
    cache = getattr(model, "_model_runner_compile_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_model_runner_compile_cache", cache)
    return cache


def _hold_compile_config(model: TorchRegionalOceanModel, compile_config: RunnerCompileConfig) -> None:
    """Publish the configuration of the compiled step the caller is about to run.

    The checkpoint window reads it (``require_differentiable_window``), so a
    dynamically compiled step configured as forward-only cannot be differentiated
    through a window later.  This is also where a dynamic configuration is handed
    to torch.compile, so its ``compile_dynamic`` hit is recorded here and not
    where the switch is resolved.  The context restores the previous value.
    """
    model._active_compile_config = compile_config
    if compile_config.dynamic is True:
        _dynamic_dispatch("hit", "locomgcm.runner._configure_step_execution")


@bind_output_ownership
@contextmanager
def _configure_step_execution(
    model: TorchRegionalOceanModel,
    compile_config: RunnerCompileConfig | None,
    *,
    derivative_mode: Literal["none", "vjp", "jvp"] = "vjp",
) -> Iterator[tuple[Callable[..., PrognosticState], dict[str, object]]]:
    # ``derivative_mode`` is the caller's derivative intent.  ModelRunner.run
    # passes it from ``track_gradients``; a direct caller that compiles a
    # forward-only step says "none", and an unstated intent counts as reverse.
    metadata: dict[str, object] = {
        "enabled": False,
        "target": "disabled",
        "mode": None,
        "fullgraph": False,
        "dynamic": False,
        "reason": None,
        "fail_on_recompile_limit_hit": False,
        "error_on_graph_break": False,
    }
    if compile_config is None or compile_config.target == "disabled":
        yield model.step_kernel, metadata
        return
    # Before anything compiles.  ModelRunner.run has already refused at its
    # entry; a window opened later over a step configured here as forward-only
    # is refused at the window's entry.
    refuse_dynamic_derivative(compile_config, derivative_mode,
                              site="locomgcm.runner._configure_step_execution")
    if (compile_config.target == "model_step" and compile_config.fullgraph
            and not model.capabilities.supports_fullgraph_model_step):
        metadata.update(
            requested_target=compile_config.target,
            requested_fullgraph=True,
            reason=(f"{type(model.stepper).__name__} does not support fullgraph model_step: "
                    "step samples host clocks; executing eagerly"),
        )
        yield model.step_kernel, metadata
        return
    if not hasattr(torch, "compile"):
        metadata["reason"] = "torch.compile is unavailable in this PyTorch build"
        yield model.step_kernel, metadata
        return
    import sys

    if sys.version_info >= (3, 13):
        metadata["reason"] = "torch.compile (Dynamo) is not supported on Python 3.13+"
        yield model.step_kernel, metadata
        return

    # The strict validation level puts a host branch on a tensor value at every
    # solver call site, which cannot be traced.  Say so here, where the two
    # settings meet, instead of failing deep inside a solver
    # (locomgcm/deferred_checks.py).
    require_level_supports_compile(compile_config.target)
    install_compile_compat()

    # Keep one graph except at the explicitly declared exits.  ``fullgraph``
    # in the config means that contract; on torch>=2.9 it is enforced with
    # ``error_on_graph_break(True)`` (hard error on any undeclared break, the
    # exits opt out with ``graph_exit``) because ``fullgraph=True`` cannot
    # coexist with any exit (locomgcm/accel/graph_exit.py).
    strict_exits = bool(compile_config.fullgraph and graph_exit_module.SUPPORTED)
    compile_kwargs = {
        "mode": compile_config.mode,
        "fullgraph": bool(compile_config.fullgraph and not strict_exits),
        "dynamic": compile_config.dynamic,
    }
    metadata.update(
        {
            "enabled": True,
            "target": compile_config.target,
            "mode": compile_config.mode,
            "fullgraph": compile_config.fullgraph,
            "dynamic": compile_config.dynamic,
            "reason": None,
        }
    )
    dynamo_config = torch._dynamo.config
    fail_setting_available = hasattr(
        dynamo_config, "fail_on_recompile_limit_hit"
    )
    previous_fail_setting = (
        dynamo_config.fail_on_recompile_limit_hit
        if fail_setting_available
        else None
    )
    fail_setting_enabled = bool(compile_config.fullgraph and fail_setting_available)
    # Under the one-graph contract a frame that recompiles on every step must
    # be a hard error, but Dynamo's default limit of 8 is also hit by one-time
    # polymorphism (a momentum-module loop traced once per module, and similar
    # one-time specialisations): a run under the default limit died at frame
    # [15/8] while the recompiles log showed no per-step guard.
    # The limit is raised for every compiling posture (``RECOMPILE_LIMIT``), the
    # hard error only for fullgraph; the recompiles log stays the judge of
    # steady state.
    limit_attrs = ("recompile_limit", "cache_size_limit")
    previous_limits = {
        name: getattr(dynamo_config, name)
        for name in limit_attrs
        if hasattr(dynamo_config, name)
    }
    if fail_setting_enabled:
        dynamo_config.fail_on_recompile_limit_hit = True
    for name, value in previous_limits.items():
        setattr(dynamo_config, name, max(int(value), RECOMPILE_LIMIT))
    metadata["fail_on_recompile_limit_hit"] = fail_setting_enabled
    metadata["recompile_limit"] = (
        max(int(min(previous_limits.values())), RECOMPILE_LIMIT)
        if previous_limits
        else None
    )

    def _restore_recompile_policy() -> None:
        if fail_setting_enabled:
            dynamo_config.fail_on_recompile_limit_hit = previous_fail_setting
        for name, value in previous_limits.items():
            setattr(dynamo_config, name, value)

    metadata["error_on_graph_break"] = strict_exits
    # The reordering passes are off for the lifetime of the
    # compiled step (they run at lowering, inside the loop below), restored
    # together with the recompile policy.
    _restore_schedule, _pinned_passes = _pin_inductor_schedule(
        bool(compile_config.pin_schedule)
    )
    metadata["pin_schedule"] = bool(compile_config.pin_schedule)
    metadata["pinned_inductor_passes"] = sorted(_pinned_passes)

    if compile_config.target == "model_step":
        previous_config = getattr(model, "_active_compile_config", None)
        try:
            cache_key = (
                "model_step",
                compile_config.mode,
                compile_config.fullgraph,
                compile_config.dynamic,
                compile_config.pin_schedule,
            )
            cache = _runner_compile_cache(model)
            compiled_step = cache.get(cache_key)
            if compiled_step is None:
                step_target = (
                    graph_exit_module.strict_graph(model.step_kernel)
                    if strict_exits
                    else model.step_kernel
                )
                compiled_step = torch.compile(step_target, **compile_kwargs)
                cache[cache_key] = compiled_step
            metadata["reason"] = None
            _hold_compile_config(model, compile_config)
            yield compiled_step, metadata
        finally:
            model._active_compile_config = previous_config
            _restore_schedule()
            _restore_recompile_policy()
        return

    if compile_config.target != "tendency":
        _restore_schedule()
        _restore_recompile_policy()
        raise ValueError(f"unsupported compile target: {compile_config.target}")

    method_names = (
        "evaluate_tendency",
        "evaluate_tendency_with_diagnostics",
        "evaluate_fast_tendency",
        "evaluate_fast_tendency_with_diagnostics",
        "evaluate_slow_tendency",
        "evaluate_slow_tendency_with_diagnostics",
    )
    originals: dict[str, object] = {}
    cache = _runner_compile_cache(model)
    previous_config = getattr(model, "_active_compile_config", None)
    try:
        for method_name in method_names:
            original = getattr(model, method_name)
            originals[method_name] = original
            cache_key = (
                "tendency",
                method_name,
                compile_config.mode,
                compile_config.fullgraph,
                compile_config.dynamic,
                compile_config.pin_schedule,
            )
            compiled = cache.get(cache_key)
            if compiled is None:
                compiled = torch.compile(original, **compile_kwargs)
                cache[cache_key] = compiled
            setattr(model, method_name, compiled)
        _hold_compile_config(model, compile_config)
        yield model.step_kernel, metadata
    finally:
        model._active_compile_config = previous_config
        for method_name, original in originals.items():
            setattr(model, method_name, original)
        _restore_schedule()
        _restore_recompile_policy()


def _dynamo_counter_value(category: str, key: str | None = None) -> int:
    counters = torch._dynamo.utils.counters.get(category, {})
    if key is None:
        return int(sum(counters.values()))
    return int(counters.get(key, 0))


def _refuse_unqualified_forward_mode(
    track_gradients: bool, compile_config: RunnerCompileConfig | None
) -> None:
    """Refuse, before any communication or compilation, a forward-mode run this path drops.

    Called while a dual level is open (``forward_mode_active``), which is
    program state: ranks that all open the same dual level raise here together.
    Each refused path returns outputs without the tangent and without an error:
    the untracked run detaches the initial state and runs under inference mode;
    an Inductor-compiled step returns kernel outputs; the communication
    operators and collectives carry first-order reverse derivatives only.
    """
    if not track_gradients:
        raise NotImplementedError(
            "forward-mode derivatives need track_gradients=True: the untracked run "
            "detaches the initial state and runs under inference mode, which drops "
            "the tangents")
    if compile_config is not None and compile_config.target != "disabled":
        raise NotImplementedError(
            f"forward-mode derivatives are not supported in compiled execution "
            f"(compile target {compile_config.target!r}): compiled kernels return "
            f"their outputs without the tangent; run with the compile target disabled")
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        raise NotImplementedError(
            "forward-mode derivatives are single-rank only: the halo exchanges and "
            "reductions provide first-order reverse derivatives only; take "
            "forward-mode derivatives on a single rank")


class ModelRunner:
    def run(
        self,
        model: TorchRegionalOceanModel,
        initial_state: PrognosticState,
        *,
        steps: int,
        dt: float | TimeStepBundle | None = None,
        snapshot_interval: int | None = 1,
        snapshot_steps: list[int] | None = None,
        track_gradients: bool = False,
        time_offset: float = 0.0,
        monitor_interval: int | None = None,
        step_callback: Callable[[int, float, PrognosticState], None] | None = None,
        step_callback_interval: int | None = None,
        collect_snapshots: bool = True,
        step_timer: StepTimer | None = None,
        compile_config: RunnerCompileConfig | None = None,
        benchmark_mode: bool = False,
        requirements: ProgramRequirements | None = None,
        comm_session=None,
        resume: bool = False,
        pickup_step_state: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """Integrate ``steps`` steps from ``initial_state``.

        ``resume=False`` seeds fresh cross-step state from the seed;
        ``resume=True`` continues the step state a restore or pickup installed.
        ``pickup_step_state`` is the step state of a new run started from a
        MITgcm pickup (:func:`initialize_fresh_step_state`).
        """
        if type(resume) is not bool:
            raise TypeError("resume must be an exact bool")
        # The derivative intent of this run, settled before communication
        # admission, extension preload and the seed's boundary pass.
        derivative_mode = "none"
        if track_gradients:
            derivative_mode = ("jvp" if requirements is not None
                               and requirements.derivative_mode == "jvp" else "vjp")
        refuse_dynamic_derivative(compile_config, derivative_mode,
                                  site="locomgcm.runner.ModelRunner.run")
        refuse_strict_plain_reverse(compile_config, derivative_mode,
                                    site="locomgcm.runner.ModelRunner.run")
        if snapshot_steps is not None:
            selected_steps = sorted(set(snapshot_steps))
            if not selected_steps:
                raise ValueError("snapshot_steps must not be empty")
            if selected_steps[0] < 0 or selected_steps[-1] > steps:
                raise ValueError(
                    "snapshot_steps must fall within the integration window"
                )
        else:
            if snapshot_interval is None or snapshot_interval < 1:
                raise ValueError(
                    "snapshot_interval must be positive when snapshot_steps is not provided"
                )
            selected_steps = list(range(0, steps + 1, snapshot_interval))
            if selected_steps[-1] != steps:
                selected_steps.append(steps)
        actual_time_step = (
            coerce_time_step_bundle(dt)
            if dt is not None
            else model.config.time_step_bundle()
        )
        actual_dt = actual_time_step.dt_clock
        if forward_mode_active():
            _refuse_unqualified_forward_mode(track_gradients, compile_config)
        if comm_session is None:
            comm_session = session_for_grid(model.grid)
        self.communication_admission = ()
        if track_gradients and dist.is_initialized() and dist.get_world_size() > 1:
            self.communication_admission = admit_communication(
                comm_session, dtype=model.config.dtype,
                device_type=torch.device(model.config.device).type,
                derivative_mode=requirements.derivative_mode if requirements is not None else "vjp",
            )
        if comm_session is not None:
            if not track_gradients:
                raise AdmissionRejected("program", "vjp", "track_gradients", None,
                                        reason="INVALID_GRAD_MODE")
            # The window's objective pass cannot isolate a graph upstream of
            # the initial state (a caller's custom Function or AOT-compiled
            # graph would run there with materialised zero cotangents); that
            # needs the whole-window boundary of an explicit window VJP
            # (planned with the explicit time program).  Controls enter as leaves.
            upstream = [name for name in ("eta", "u", "v", "temp", "salt")
                        if getattr(initial_state, name).data.grad_fn is not None]
            if upstream:
                raise NotImplementedError(
                    "a differentiable run with a communication session takes leaf "
                    f"initial fields; {upstream} carry an upstream graph, which the "
                    "window's objective pass cannot isolate")
            model = model.with_comm_session(comm_session)
        # Admission is settled before preload, state detachment or compilation.
        # None is the explicitly preserved legacy path, not an inferred demand.
        record = None
        if requirements is not None:
            # The session's control domain owns the agreement; without a
            # session the participants are the whole job.
            record = admit(requirements, resolve_snapshot(), policy="explicit",
                           group=None if comm_session is None else comm_session.control_group)
            if requirements.derivative_mode != "none" and not track_gradients:
                raise AdmissionRejected("program", requirements.derivative_mode,
                    "track_gradients", None, reason="INVALID_GRAD_MODE")
        self.admission_record = record
        context = nullcontext() if track_gradients else torch.inference_mode()
        context = selection_scope(record, context)
        state_seed = (
            initial_state
            if track_gradients
            else initial_state.with_tensors(
                eta=initial_state.eta.data.detach(),
                u=initial_state.u.data.detach(),
                v=initial_state.v.data.detach(),
                temp=initial_state.temp.data.detach(),
                salt=initial_state.salt.data.detach(),
            )
        )
        active_timer = step_timer
        if active_timer is not None:
            active_timer.reset(device=model.config.device)
        with selection_scope(record):
            preload_requested_extensions(device=model.config.device)
        # ``production`` watches the free-surface solve's residual, so a
        # stepper that runs no such solve gives it nothing to look at.  Settle
        # what this run can actually watch before the step loop starts, rather
        # than report every step of such a run clean
        # .
        check_metadata = resolve_step_probe(model.stepper)
        compile_metadata: dict[str, object] = {
            "enabled": False,
            "target": "disabled",
            "mode": None,
            "fullgraph": False,
            "dynamic": False,
            "reason": None,
            "fail_on_recompile_limit_hit": False,
            "error_on_graph_break": False,
        }
        eager_warmup_steps = (
            int(compile_config.eager_warmup_steps)
            if compile_config is not None and compile_config.target != "disabled"
            else 0
        )
        if resume and eager_warmup_steps:
            # The pickup carries the number of accepted steps, including any
            # eager steps before the checkpoint. The run's clock may have an offset.
            eager_warmup_steps = max(0, eager_warmup_steps - int(_step_index_tensor(model.stepper)))
        graph_breaks_before = _dynamo_counter_value("graph_break")
        unique_graphs_before = _dynamo_counter_value("stats", "unique_graphs")
        timing_metadata: dict[str, float | int | None] | None = None
        with ordinary_tensor_mode():
            model.reset_runtime_budget()
        with context:
            model.prepare_step_time(time_offset)
            state = (resume_step_state(model, state_seed) if resume else
                     initialize_fresh_step_state(model, state_seed, dt=actual_time_step,
                                                 pickup_step_state=pickup_step_state))
            from locomgcm.step.carry import supports_explicit_history, initialize, WindowMode

            if supports_explicit_history(model):
                # The same choice as the seed above: a new start rebuilds the
                # history, a resume keeps the restored one.
                mode = (WindowMode.REINITIALIZE_HISTORY if not resume
                        else WindowMode.CONTINUE_DIFFERENTIABLE if track_gradients
                        else WindowMode.FREEZE_PREWINDOW_HISTORY)
                state = initialize(model, state, time_seconds=time_offset, window_mode=mode).state

            request = getattr(model, "_restored_compiled_geometry", None) if steps else None
            restored = None
            if (request is not None and resume and compile_config is not None
                    and compile_config.target == "model_step" and eager_warmup_steps == 0
                    and 0 < request[0] == int(_step_index_tensor(model.stepper))):
                restored = request

            history: list[dict[str, float]] = []
            snapshot_steps = []
            snapshot_times: list[float] = []
            eta_snapshots: list[torch.Tensor] = []
            u_snapshots: list[torch.Tensor] = []
            v_snapshots: list[torch.Tensor] = []
            temp_snapshots: list[torch.Tensor] = []
            salt_snapshots: list[torch.Tensor] = []
            snapshot_step_set = set(selected_steps)
            log_interval = (
                monitor_interval
                if monitor_interval is not None
                else max(1, min(steps // 10, 50))
            )
            cb_interval = (
                step_callback_interval
                if step_callback_interval is not None
                else log_interval
            )
            diag_step_set = (
                set()
                if benchmark_mode
                else snapshot_step_set
                | {step for step in range(0, steps + 1, log_interval)}
                | {steps}
            )

            with _configure_step_execution(model, compile_config,
                                           derivative_mode=derivative_mode) as (
                compiled_step_fn,
                compile_metadata,
            ):
                prepared_restore = restored is not None and compile_metadata["enabled"]
                if prepared_restore:
                    from locomgcm.accel.compile_resources import prepare

                    prepare(model, state, restored[1], convention=restored[2])
                step_fn = compiled_step_fn
                for step in range(steps + 1):
                    step_time = time_offset + step * actual_dt
                    model._prepare_runner_step_time(step_time)

                    if step in diag_step_set:
                        raw_diagnostics = model.compute_diagnostics(
                            state, dt=actual_dt
                        ).scalars
                        diagnostics = {
                            k: float(v.item())
                            if isinstance(v, torch.Tensor)
                            else float(v)
                            for k, v in raw_diagnostics.items()
                        }
                        history.append(
                            {
                                "step": float(step),
                                "time": float(step * actual_dt),
                                **diagnostics,
                            },
                        )
                    else:
                        diagnostics = None

                    if not benchmark_mode and (
                        step % log_interval == 0 or step == steps
                    ):
                        t_hours = step * actual_dt / 3600.0
                        eta = state.eta.data
                        eta_rms = float(eta.pow(2).mean().sqrt().item())
                        ke = (
                            diagnostics.get("kinetic_energy", 0.0)
                            if diagnostics
                            else 0.0
                        )
                        timing_suffix = ""
                        if active_timer is not None:
                            timing_summary = active_timer.summary(
                                simulated_dt=actual_dt
                            )
                            if timing_summary.recorded_steps:
                                timing_suffix = f"  step_wall={1.0e3 * timing_summary.mean_seconds:.1f}ms"
                        print(
                            f"  step {step:>5d}/{steps}  t={t_hours:>7.1f}h  "
                            f"η_rms={eta_rms:.4e}  KE={ke:.4e}{timing_suffix}",
                            flush=True,
                        )
                    if step in snapshot_step_set:
                        snapshot_steps.append(step)
                        snapshot_times.append(float(step * actual_dt))
                        # collect_snapshots=False: the caller streams frames to
                        # disk via step_callback and discards result.*_snapshots,
                        # so skip the O(N_frames) in-memory accumulation (and the
                        # end-of-run torch.stack that transiently doubles it).
                        # snapshot_steps / snapshot_times are still recorded.
                        if collect_snapshots:
                            eta_snapshots.append(state.eta.data.detach().cpu())
                            u_snapshots.append(state.u.data.detach().cpu())
                            v_snapshots.append(state.v.data.detach().cpu())
                            temp_snapshots.append(state.temp.data.detach().cpu())
                            salt_snapshots.append(state.salt.data.detach().cpu())
                    call_callback = (
                        step_callback is not None
                        and (step % cb_interval == 0 or step == steps)
                        and (not benchmark_mode or step == steps)
                    )
                    if call_callback or step == steps:
                        observed = model.observe_state(state) if comm_session is not None else state
                        if call_callback:
                            step_callback(step, step * actual_dt, observed)
                    if step == steps:
                        state = observed
                        model._runner_step_time = None
                        break
                    if active_timer is not None:
                        active_timer.start_step(step)
                    if ckpt_dump.ENABLED:
                        ckpt_dump.begin_step(step + 1)
                    # The three phases of one step.  Only the middle one is ever
                    # compiled; ``build_step_plan`` and ``finalize_step`` are
                    # eager by construction, not by decoration.
                    # ``host.plan`` is the only half the compiled call sees; the
                    # host clocks stay out here.
                    host = build_step_plan(
                        model,
                        state,
                        actual_time_step,
                        time_seconds=step_time,
                    )
                    if prepared_restore:
                        from locomgcm.steppers.regional_cold_start import reseed_regional_kpp_forcing_inputs

                        forcing_inputs = reseed_regional_kpp_forcing_inputs(model, state)
                        if forcing_inputs is not None:
                            host = replace(host, plan=replace(host.plan, kpp_forcing_inputs=forcing_inputs))
                    model._runner_step_time = None
                    # The first ``eager_warmup_steps`` steps run the
                    # eager kernel so one-shot host work never meets the tracer.
                    # Selected BEFORE the generation mark so ``mark_cudagraph_step``
                    # stays the statement immediately preceding the single ``step_fn``
                    # call, as required by the CUDAGraph step-boundary contract.
                    if step < eager_warmup_steps:
                        step_fn = model.step_kernel
                    elif step == eager_warmup_steps:
                        step_fn = compiled_step_fn
                    mark_cudagraph_step()
                    outcome = step_fn(state, host.plan)
                    if active_timer is not None:
                        active_timer.end_step(step)
                        active_timer.poll()
                    # Finalize checks the candidate before publishing it.
                    state = finalize_step(model, outcome, host)
                    if request is not None:
                        # Every accepted step ends the restored boundary, including
                        # a fresh trajectory or a continuation advanced eagerly.
                        if getattr(model, "_restored_compiled_geometry", None) is request:
                            del model._restored_compiled_geometry
                        request = None
                        prepared_restore = False
                    if step == eager_warmup_steps and step_fn is not model.step_kernel:
                        # The warm-up compile is finished: release the Inductor
                        # worker pool (a later recompile recreates it on demand).
                        from torch._inductor.async_compile import shutdown_compile_workers

                        shutdown_compile_workers()

                    # LOCOMGCM_DIV_DIAG=1 — per-step divergence diagnostic (default OFF).
                    # Prints max|var| and argmax (k,j,i) + estimated lon/lat/depth
                    # for each prognostic variable after every time step.
                    # Also logs to LOCOMGCM_DIV_DIAG_LOG if set.
                    if not benchmark_mode and os.environ.get("LOCOMGCM_DIV_DIAG") == "1":
                        _diag_step = step + 1  # step just completed
                        _grd = model.grid
                        _ny, _nx, _nz = _grd.ny, _grd.nx, _grd.nz
                        # Estimate SW-corner lat from center_lat - ny/2*dy, lon from center_lon - nx/2*dx
                        # Using grid metrics: dyG in meters -> degrees via 111320 m/deg
                        _dy_m = float(_grd.metrics.dyG_scalar_1d)
                        _dy_deg = _dy_m / 111320.0
                        # dxF_row gives per-row dx in meters; use middle row for center
                        _mid_row = _ny // 2
                        try:
                            _dx_row = _grd.metrics.dxC_row_for(torch.float64, "cpu")
                            _dx_deg_mid = float(_dx_row[_mid_row]) / (
                                111320.0 * 0.940
                            )  # rough cos(20°)
                        except Exception:
                            _dx_deg_mid = _dy_deg
                        # SW corner estimate (center - half domain)
                        _lat0 = 20.0992 - _ny / 2.0 * _dy_deg
                        _lon0 = 119.8619 - _nx / 2.0 * _dx_deg_mid
                        _depth_centers = (
                            _grd.vertical_coordinate.center_depth.tolist()
                            if hasattr(_grd.vertical_coordinate, "center_depth")
                            else list(range(_nz))
                        )
                        _vars = {
                            "eta": state.eta.data,
                            "u": state.u.data,
                            "v": state.v.data,
                            "temp": state.temp.data,
                            "salt": state.salt.data,
                        }
                        _lines = []
                        for _vn, _vt in _vars.items():
                            _vabs = _vt.abs()
                            _maxabs = float(_vabs.max().item())
                            _isnan = bool(torch.any(torch.isnan(_vt)).item())
                            _isinf = bool(torch.any(torch.isinf(_vt)).item())
                            _flag = " NAN" if _isnan else (" INF" if _isinf else "")
                            if _vt.ndim == 4:
                                _flat = _vabs.reshape(-1)
                                _idx = int(_flat.argmax().item())
                                _B, _NZ, _NY, _NX = _vt.shape
                                _b = _idx // (_NZ * _NY * _NX)
                                _rem = _idx % (_NZ * _NY * _NX)
                                _k = _rem // (_NY * _NX)
                                _rem2 = _rem % (_NY * _NX)
                                _j = _rem2 // _NX
                                _i = _rem2 % _NX
                                _lat = _lat0 + _j * _dy_deg
                                _lon = _lon0 + _i * _dx_deg_mid
                                _dep = (
                                    _depth_centers[_k]
                                    if _k < len(_depth_centers)
                                    else _k
                                )
                                _lines.append(
                                    f"DIV step={_diag_step} var={_vn} maxabs={_maxabs:.6e}"
                                    f" at (k,j,i)=({_k},{_j},{_i})"
                                    f" lon={_lon:.3f} lat={_lat:.3f} depth={_dep:.1f}{_flag}"
                                )
                            else:
                                _flat = _vabs.reshape(-1)
                                _idx = int(_flat.argmax().item())
                                _NX2 = _vt.shape[-1]
                                _j = _idx // _NX2
                                _i = _idx % _NX2
                                _lat = _lat0 + _j * _dy_deg
                                _lon = _lon0 + _i * _dx_deg_mid
                                _lines.append(
                                    f"DIV step={_diag_step} var={_vn} maxabs={_maxabs:.6e}"
                                    f" at (j,i)=({_j},{_i})"
                                    f" lon={_lon:.3f} lat={_lat:.3f}{_flag}"
                                )
                        _out = "\n".join(_lines)
                        print(_out, flush=True)
                        _log_path = os.environ.get("LOCOMGCM_DIV_DIAG_LOG")
                        if _log_path:
                            with open(_log_path, "a") as _lf:
                                _lf.write(_out + "\n")

                    # LOCOMGCM_WIN_DUMP=1 — per-step local field window around the |u| argmax,
                    # for blow-up replay (default OFF).  Tracks the most dangerous point so
                    # the pre-blow-up frames are captured wherever the eruption is, any step.
                    if not benchmark_mode and os.environ.get("LOCOMGCM_WIN_DUMP") == "1":
                        _wstep = step + 1
                        if _wstep % int(os.environ.get("LOCOMGCM_WIN_DUMP_EVERY", "1")) == 0:
                            import numpy as _np

                            _wdir = os.environ.get(
                                "LOCOMGCM_WIN_DUMP_DIR", "diagnostics/window_dumps"
                            )
                            os.makedirs(_wdir, exist_ok=True)
                            _half = int(os.environ.get("LOCOMGCM_WIN_DUMP_HALF", "60"))
                            _wu = state.u.data
                            _wB, _wNZ, _wNY, _wNX = _wu.shape
                            _wflat = _wu.abs().reshape(-1)
                            _wmax = float(_wflat.max().item())
                            _wlin = int(_wflat.argmax().item()) % (_wNZ * _wNY * _wNX)
                            _wk0 = _wlin // (_wNY * _wNX)
                            _wr2 = _wlin % (_wNY * _wNX)
                            _wj0 = _wr2 // _wNX
                            _wi0 = _wr2 % _wNX
                            _wj1, _wj2 = max(0, _wj0 - _half), min(_wNY, _wj0 + _half)
                            _wi1, _wi2 = max(0, _wi0 - _half), min(_wNX, _wi0 + _half)

                            def _wwin(_t):
                                if _t.ndim == 4:
                                    return (
                                        _t[0, :, _wj1:_wj2, _wi1:_wi2]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )
                                if _t.ndim == 3:
                                    return (
                                        _t[0, _wj1:_wj2, _wi1:_wi2]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )
                                return _t[_wj1:_wj2, _wi1:_wi2].detach().cpu().numpy()

                            _np.savez_compressed(
                                os.path.join(_wdir, f"step{_wstep:05d}.npz"),
                                u=_wwin(state.u.data),
                                v=_wwin(state.v.data),
                                temp=_wwin(state.temp.data),
                                salt=_wwin(state.salt.data),
                                eta=_wwin(state.eta.data),
                                j0=_wj0,
                                i0=_wi0,
                                k0=_wk0,
                                j1=_wj1,
                                j2=_wj2,
                                i1=_wi1,
                                i2=_wi2,
                                maxabs_u=_wmax,
                            )

                if active_timer is not None:
                    timing_metadata = active_timer.finalize(
                        simulated_dt=actual_dt
                    ).as_dict()

        compile_metadata["eager_warmup_steps"] = eager_warmup_steps
        compile_metadata["graph_break"] = (
            _dynamo_counter_value("graph_break") - graph_breaks_before
        )
        compile_metadata["unique_graphs"] = (
            _dynamo_counter_value("stats", "unique_graphs") - unique_graphs_before
        )

        # When snapshots were not collected the lists are empty; torch.stack would
        # raise, so return 0-element tensors (the caller that opted out discards them).
        _stack = (
            (lambda xs: torch.stack(xs, dim=0))
            if collect_snapshots
            else (lambda xs: torch.empty(0))
        )
        # Last look at the run's solver-code sink; every step boundary already
        # inspected it, so this only catches a run that ended without one.
        inspect_final()
        # Backend-mixing guard at the end-of-run synchronous point (all ranks
        # reach here after the same step count); see accel.runtime docstring.
        from locomgcm.accel.runtime import verify_dispatch_uniformity
        if comm_session is None:
            verify_dispatch_uniformity()
        else:
            # The raw session is still live: exchange on its control domain,
            # under its failure monitor.
            from locomgcm.distributed.transport_failure import transport_operation

            with transport_operation(comm_session.control_group_name):
                verify_dispatch_uniformity(comm_session.control_group)

        return SimulationResult(
            history=history,
            snapshot_steps=snapshot_steps,
            snapshot_times=snapshot_times,
            eta_snapshots=_stack(eta_snapshots),
            u_snapshots=_stack(u_snapshots),
            v_snapshots=_stack(v_snapshots),
            temp_snapshots=_stack(temp_snapshots),
            salt_snapshots=_stack(salt_snapshots),
            final_state=state,
            metadata={
                "nx": model.grid.nx,
                "ny": model.grid.ny,
                "nz": model.grid.nz,
                "lx": model.config.lx,
                "ly": model.config.ly,
                "depth": model.config.depth,
                "dt": actual_time_step.dt,
                "dt_mom": actual_time_step.dt_mom,
                "dt_tracer": actual_time_step.dt_tracer,
                "dt_freesurf": actual_time_step.dt_freesurf,
                "dt_clock": actual_time_step.dt_clock,
                "gravity": model.config.gravity,
                "stepper": model.stepper.name,
                "west_east_boundary": model.config.west_east_boundary.value,
                "south_north_boundary": model.config.south_north_boundary.value,
                "timing": timing_metadata,
                "compile": compile_metadata,
                # What the step boundary actually watched.  On the record and
                # not only on stderr: a batch run keeps its metadata and loses
                # its stderr, and "production behaved as debug here" has to be
                # answerable afterwards.
                "checks": dict(check_metadata),
            },
        )
