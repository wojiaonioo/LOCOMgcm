"""Owners of state-dependent working values between checkpoint boundaries."""

from contextlib import contextmanager


@contextmanager
def segment_workspace(model):
    """Own working history until a segment has exported its outputs.

    The next segment must install its head. Neither a successful export nor an
    exception leaves the model holding the segment's graph. Outputs returned
    by the segment retain their graph independently for reverse readout.
    """
    from defgcm.checkpoint.state import borrow_step_inputs, release_working_step_state

    # Legacy adapters may provide fixed KPP/forcing inputs directly. They are
    # borrowed from the enclosing window, not produced by this segment.
    borrowed = borrow_step_inputs(model.stepper, obc_eta=None)
    try:
        yield
    finally:
        release_working_step_state(model, borrowed_inputs=borrowed)
