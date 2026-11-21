"""Stevens and Orlanski history and the legacy boundary adapter owned by the step layer."""

from dataclasses import dataclass, replace

import torch
from torch.utils import _pytree


@dataclass(frozen=True)
class BoundaryHistory:
    prev_temp: torch.Tensor
    prev_salt: torch.Tensor
    valid: torch.Tensor
    stamp: torch.Tensor
    #: Orlanski storage by side value (``ORLANSKI.h``; layout in
    #: ``locomgcm.boundaries.orlanski``); a side without an entry starts from
    #: ``ORLANSKI_INIT``'s zeros.
    orlanski: dict[str, torch.Tensor]

    @classmethod
    def initial(cls, state, time_seconds=0.):
        return cls(torch.zeros_like(state.temp.data), torch.zeros_like(state.salt.data),
                   torch.zeros((), dtype=torch.bool, device=state.temp.data.device),
                   state.temp.data.new_tensor(time_seconds, dtype=torch.float64), {})

    def to(self, device, dtype=None):
        return BoundaryHistory(self.prev_temp.to(device=device, dtype=dtype).clone(),
                               self.prev_salt.to(device=device, dtype=dtype).clone(),
                               self.valid.to(device=device).clone(),
                               self.stamp.to(device=device).clone(),
                               {side: storage.to(device=device, dtype=dtype).clone()
                                for side, storage in self.orlanski.items()})

    @classmethod
    def sample(cls, state, stamp, orlanski):
        return cls(state.temp.data.clone(), state.salt.data.clone(),
                   torch.ones((), dtype=torch.bool, device=state.temp.data.device),
                   torch.as_tensor(stamp, dtype=torch.float64,
                                   device=state.temp.data.device).clone(), orlanski)


_pytree.register_pytree_node(
    BoundaryHistory,
    lambda history: ([history.prev_temp, history.prev_salt, history.valid, history.stamp,
                      history.orlanski], None),
    lambda leaves, context: BoundaryHistory(*leaves),
)


def validate_boundary_history(history, state):
    for name in ("temp", "salt"):
        value, reference = getattr(history, "prev_" + name), getattr(state, name).data
        if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                or value.dtype != reference.dtype or value.device != reference.device
                or not torch.isfinite(value).all()):
            raise ValueError(f"boundary history prev_{name} shape/dtype/device/finite mismatch")
    for name, dtype in (("valid", torch.bool), ("stamp", torch.float64)):
        value = getattr(history, name)
        if (not isinstance(value, torch.Tensor) or value.shape != () or value.dtype != dtype
                or value.device != state.temp.data.device or not torch.isfinite(value)):
            raise ValueError(f"boundary history {name} must be a finite same-device scalar {dtype}")


@dataclass
class BoundaryEvaluation:
    history: BoundaryHistory
    stamp: torch.Tensor
    candidate: BoundaryHistory | None = None

    def enforce(self, boundary, state, grid, **stevens):
        result, candidate = boundary.enforce_with_history(
            state, grid, history=self.history, stamp=self.stamp, **stevens)
        if candidate is not None:
            self.candidate = candidate
        return result


@dataclass
class BoundaryHistoryAdapter:
    history: BoundaryHistory | None = None
    candidate: BoundaryHistory | None = None
    active_stamp: float | None = None
    initialization_reason: str | None = None
    _legacy_fields: frozenset[str] = frozenset()

    def reset(self):
        """Release all history owned by a completed segment or independent run."""
        self.history = None
        self.candidate = None
        self.active_stamp = None
        self.initialization_reason = None
        self._legacy_fields = frozenset()

    def begin(self, state, time_seconds, future_seconds):
        if self.history is None:
            self.history = BoundaryHistory.initial(state, time_seconds)
        validate_boundary_history(self.history, state)
        if float(self.history.stamp) != time_seconds:
            raise ValueError("boundary history stamp differs from step time; replay requires explicit history")
        self.candidate = None
        self.active_stamp = future_seconds

    def set_legacy(self, name, value):
        if value is None:
            self._legacy_fields = self._legacy_fields - {name}
            if self.history is not None:
                self.history = replace(self.history, **{name: None},
                                       valid=torch.zeros_like(self.history.valid))
            return
        if self.history is None:
            self.history = BoundaryHistory(
                torch.zeros_like(value), torch.zeros_like(value),
                torch.zeros((), dtype=torch.bool, device=value.device),
                torch.zeros((), dtype=torch.float64, device=value.device), {})
        self._legacy_fields = self._legacy_fields | {name}
        valid = bool(self.history.valid) or self._legacy_fields == {"prev_temp", "prev_salt"}
        self.history = replace(self.history, **{name: value},
                               valid=torch.tensor(valid, dtype=torch.bool, device=value.device))


def boundary_adapter(model):
    boundary = getattr(model, "boundary_condition", None)
    return (boundary._history_adapter
            if (getattr(boundary, "uses_stevens", False)
                or getattr(boundary, "uses_orlanski", False)) else None)
