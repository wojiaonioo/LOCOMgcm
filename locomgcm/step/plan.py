"""Phase one: every host decision of one time step, resolved eagerly.

``build_step_plan`` is the single entry point.  It runs the model's eager
pre-step -- clock sampling, boundary-record interpolation, gate freezing, host
validation of the step counter, one-shot constant materialisation -- and then
publishes the result as a :class:`~locomgcm.step.contract.StepPlan` instead of
leaving it as module attributes for the traced step to discover.

Host preparation runs outside the traced region.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from locomgcm.accel import StepGates, freeze_step_snapshot
from locomgcm.accel.tensor_context import ordinary_tensor_mode
from locomgcm.deferred_checks import prepare_solver_codes, strict_checks
from locomgcm.steppers.base import TimeStepBundle, coerce_time_step_bundle

from .contract import (
    EMPTY_MAPPING,
    HostPlan,
    HostTimes,
    ShapeContract,
    StepPlan,
    StepTimes,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from locomgcm.state import PrognosticState


def _step_index_tensor(stepper: Any) -> torch.Tensor:
    """The step counter as a 0-d int64 tensor, whatever the stepper stores.

    ``ImplicitFreeSurfaceStepper`` keeps it as a 0-d tensor already; a
    fresh stepper or a restored checkpoint may hold an exact int, and a stepper
    without the counter at all reports step 0.  Nothing here reads a tensor back
    to the host -- that read belongs to ``_step_index_value`` in the eager
    prepare, which has already run by the time this is called.
    """
    value = getattr(stepper, "_step_index", None)
    value = 0 if value is None else value
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(int(value), dtype=torch.int64)


def _prepare_solver_code_sink(
    gates: StepGates, shapes: ShapeContract, device: Any
) -> None:
    """Give the run its error-code sink, if any solver here can write one.

    Prepare owns this allocation: made eagerly, it is never a buffer the
    traced region allocates -- such a buffer becomes a graph output and is
    reclaimed a generation later.  The call is idempotent; after the first step
    it only checks the device.

    The sink is not a plan field.  Native runtimes publish into it through
    ``deferred_checks.publish_solver_code`` and the step boundary reads it, so
    naming it on the plan would only add a second handle on one buffer.  Only
    the fused vertical solver publishes a code today, and only when its gate is
    on and the level is not ``strict`` (which judges each call instead).
    """
    if strict_checks() or not gates.ivadv:
        return
    prepare_solver_codes(device, shapes)


@ordinary_tensor_mode()
def build_step_plan(
    model: Any,
    state: "PrognosticState",
    bundle: float | TimeStepBundle | None,
    *,
    time_seconds: float,
) -> HostPlan:
    """Resolve one step's host state and freeze it into a plan.

    ``bundle`` accepts the same shapes ``model.step`` does (a bundle, a bare
    ``dt``, or ``None`` for the model's configured bundle) so the runner and the
    direct API build identical plans.

    Returns the :class:`HostPlan`, whose ``.plan`` is the half the kernel is
    given: the host clocks stay on this side of the call, where a Python float
    costs nothing ().
    """
    resolved = (
        model.config.time_step_bundle()
        if bundle is None
        else coerce_time_step_bundle(bundle)
    )
    clock = float(time_seconds)

    from .boundary_history import boundary_adapter

    adapter = boundary_adapter(model)
    if adapter is not None:
        adapter.begin(state, clock, clock + resolved.dt_clock)

    model._check_open_boundary_runtime_support()
    # The eager pre-step: clock sampling and boundary records
    # (``prepare_step_times``), gate freezing (``freeze_step_snapshot``), host
    # validation of the counter (``_validate_step_index_state``) and the
    # one-shot constant builds all happen inside this call.  It hands back the
    # two time samples it took; the traced step reads them off the plan instead
    # of reaching into the model for them.
    boundary, kpp_forcing_inputs = model.prepare_step_host(state, resolved, time_seconds=clock)
    if getattr(model, "_ocean_carry", None) is None:
        model.current_time_seconds = clock

    stepper = getattr(model, "stepper", None)
    gates = getattr(stepper, "gates", None)
    if not isinstance(gates, StepGates):
        # A stepper without a prepare hook never froze the snapshot; the plan
        # owns that decision now, so the traced region reads a frozen view
        # whichever stepper is installed.
        gates = StepGates.from_views(freeze_step_snapshot())

    shapes = ShapeContract.from_grid(model.grid, state)
    _prepare_solver_code_sink(gates, shapes, model.config.device)
    return HostPlan(
        plan=StepPlan(
            gates=gates,
            times=StepTimes.sample(current_seconds=clock, bundle=resolved),
            step_index=_step_index_tensor(stepper),
            shapes=shapes,
            constants=EMPTY_MAPPING,
            boundary=boundary,
            kpp_forcing_inputs=kpp_forcing_inputs,
            boundary_history=(adapter.history.to(state.temp.data.device)
                              if adapter is not None else None),
        ),
        times=HostTimes.sample(current_seconds=clock, bundle=resolved),
    )


__all__ = ["build_step_plan"]
