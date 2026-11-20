"""Phase three: everything that must happen after the traced step, eagerly.

``finalize_step`` owns the step boundary.  It advances the counter, takes
ownership of the retained tensors and drains
the dispatch events queued by the traced region. These operations run
after the compiled step.

The public finalizer in ``carry`` validates the outcome before calling this
publication phase. The runner closes its timer before entering that finalizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from defgcm.accel import drain_pending_dispatch_events

if TYPE_CHECKING:  # pragma: no cover - typing only
    from defgcm.state import PrognosticState

    from .contract import HostPlan, StepOutcome


def finalize_step(
    model: Any, outcome: "StepOutcome", host: "HostPlan"
) -> "PrognosticState":
    """Close one step and hand back the state the next step may consume.

    Takes the :class:`~defgcm.step.contract.HostPlan`, not the kernel's
    half of it: finalize is eager, so the host clocks are readable here, and
    the argument type is what says which side of the boundary this runs on.
    """
    stepper = getattr(model, "stepper", None)

    advance = getattr(stepper, "advance_step_index", None)
    if advance is not None:
        # Advances ``_step_index`` and stamps the promoted momentum
        # wrapper.  It runs BEFORE the ownership transfer so the promoted
        # wrapper is copied in the same pass as every other retained value.
        advance()

    persist = getattr(stepper, "persist_step_outputs", None)
    state = (
        persist(model, outcome.state) if persist is not None else outcome.state
    )

    # The traced region can only queue a dispatch record (``record_dispatch``
    # appends to ``_PENDING_EVENTS`` while compiling); the accounting itself is
    # an observer and belongs out here, at the boundary that produced it.
    drain_pending_dispatch_events()
    return state


__all__ = ["finalize_step"]
