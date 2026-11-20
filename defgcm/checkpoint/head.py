"""Segment heads: what one recomputation segment starts from, kept on the device.

A segmented rollout needs, at the head of every segment, the state the model
holds there, in the same process and for the same model.  The restart codec
(``capture``/``restore``) produces that content for a restart file: host
copies, the per-value checks, provenance stamps and the configuration
comparison.  ``SegmentHeadSpec`` keeps the content and the model writes and
moves the rest to the window: the initial checkpoint passes ``restore``'s
checks once, each new head layout passes ``capture``'s checks once, and the
returned final checkpoint gets ``capture``'s header and checks once.

A head is a ``ModelCheckpoint`` on the model device. ``flatten`` produces an
explicit snapshot; ``readout`` transfers expressions from an ending workspace.
``inject`` consumes an exclusively owned head through the same model writes
as restore, without the restart reader's defensive copy.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_map_only

from .boundary import head_boundary
from .capture import (
    checked_for_resume,
    checkpoint_meta,
    install_checkpoint,
)
from .state import (
    head_step_state,
    require_exact_int,
    validate_checkpoint_structure,
    validate_momentum_wrapper_for_model,
)
from .types import PROGNOSTIC_FIELDS, ModelCheckpoint

_MOMENTUM_KEY = "_momentum_w_nm_half_mit"


class SegmentHeadSpec:
    """The heads of one window: checked once per layout, then moved without checks."""

    def __init__(self, model: Any, initial: ModelCheckpoint):
        self.model = model
        #: ``initial`` after ``restore``'s checks, migrated to the current schema.
        self.initial = checked_for_resume(model, initial, initial.meta.config_fingerprint)
        self.initial_meta = self.initial.meta
        self._layouts: list[tuple] = []

    def to_device(self, head: ModelCheckpoint) -> ModelCheckpoint:
        """``head`` moved to the model device, still attached to the caller's graph.

        Only the device changes; ``inject`` applies ``restore``'s dtype rules.
        """
        device = self.model.config.device
        prognostic, step_state = tree_map_only(
            torch.Tensor, lambda tensor: tensor.to(device=device),
            (head.prognostic, head.step_state))
        return replace(head, prognostic=prognostic, step_state=step_state)

    def flatten(self, model: Any, state: Any, *, global_step: int,
                model_time_seconds: float, snapshot: bool = True) -> ModelCheckpoint:
        """The head at ``state``: ``capture``'s content as device clones that keep their graph."""
        prognostic = {name: (getattr(state, name).data.clone() if snapshot
                            else getattr(state, name).data) for name in PROGNOSTIC_FIELDS}
        step_state = head_step_state(model.stepper, snapshot=snapshot)
        boundary = head_boundary(model, state, snapshot=snapshot)
        if boundary is not None:
            step_state["boundary_history"] = boundary
        from .nonlinear import head_nonlinear_geometry

        geometry = head_nonlinear_geometry(model, snapshot=snapshot)
        if geometry is not None:
            step_state["nonlinear_geometry"] = geometry
        if require_exact_int(step_state["_step_index"], path="step_state['_step_index']") != global_step:
            raise ValueError(
                "segment head clock mismatch: "
                f"step_state['_step_index']={step_state['_step_index']}, global_step={global_step}")
        head = ModelCheckpoint(
            prognostic=prognostic, step_state=step_state,
            meta=replace(self.initial_meta, global_step=global_step,
                         model_time_seconds=float(model_time_seconds), requires_grad=True,
                         state_keys=tuple(sorted(step_state)), migration_records=()))
        layout = _layout(head)
        if layout not in self._layouts:
            # ``capture``'s checks, once for each head layout of the window.
            validate_momentum_wrapper_for_model(step_state, model)
            validate_checkpoint_structure(head)
            self._layouts.append(layout)
        return head

    def readout(self, model, state, **clock):
        """Read expressions at segment end; the workspace relinquishes them."""
        return self.flatten(model, state, snapshot=False, **clock)

    def inject(self, model: Any, head: ModelCheckpoint, *, new_run: bool):
        """Install ``head`` and offer its START sample to the next step's plan.

        The segment has already bound its parameters. Reuse this sample only
        for the immediately following plan; a boundary-history adapter keeps
        plan resampling. The segment clears an unconsumed handoff when its
        workspace exits.
        """
        return install_checkpoint(model, head, new_run=new_run, detach=False, checked=False,
                                  transfer=True, prepare_next_step=True)

    def final(self, model: Any, head: ModelCheckpoint, *, dt: float) -> ModelCheckpoint:
        """The last head as ``capture`` would return it: full header and checks, once."""
        meta = checkpoint_meta(
            model, head.step_state, global_step=head.meta.global_step,
            model_time_seconds=head.meta.model_time_seconds, dt=dt,
            config_fingerprint=self.initial_meta.config_fingerprint, permanent=False,
            compute_device=head.prognostic["u"].device,
            requires_grad=True)
        final = replace(head, meta=meta)
        validate_momentum_wrapper_for_model(final.step_state, model)
        validate_checkpoint_structure(final)
        return final


def _layout(head: ModelCheckpoint) -> tuple:
    """Structure, tensor metadata and constant leaves of a head, its two step clocks blanked."""
    step_state = {**head.step_state, "_step_index": None}
    if isinstance(step_state.get(_MOMENTUM_KEY), dict):
        step_state[_MOMENTUM_KEY] = {**step_state[_MOMENTUM_KEY], "step_index": None}
    leaves, structure = tree_flatten((head.prognostic, step_state))
    return (structure, tuple(
        (tuple(leaf.shape), leaf.dtype, leaf.device, leaf.stride())
        if isinstance(leaf, torch.Tensor) else leaf for leaf in leaves))


__all__ = ["SegmentHeadSpec"]
