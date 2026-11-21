"""Accepted implicit KPP roots owned by one checkpoint segment.

The window stores physical states with its existing checkpoint slot.  The
metadata contains only the equation identity and Python solve diagnostics;
neither a problem object nor an autograd graph is kept.  Recompute takes the
states in order, and the numerical caller checks each state's original
residual before returning it through its implicit differentiation boundary.
"""

from __future__ import annotations

import copy

import torch


class _SegmentRoots:
    def __init__(self, entries: list, replay: bool) -> None:
        self.entries, self.replay, self.taken = entries, replay, 0

    def take(self):
        if self.taken == len(self.entries):
            raise RuntimeError(
                "the recompute solves more implicit KPP systems than its forward kept"
            )
        entry, self.entries[self.taken] = self.entries[self.taken], None
        self.taken += 1
        return entry


# Like the pressure-root scope, this is read only at the eager solve boundary.
# A module global also reaches the autograd worker that executes the segment.
_SEGMENT_ROOTS: _SegmentRoots | None = None


class segment_roots:
    """Retain or consume the accepted roots of one checkpoint segment.

    Entries have the same tensor-first layout as the existing pressure roots:
    (physical_state, identity, forward_record).  The window replaces the
    tensor by None after attaching it to its store slot, then restores the
    tensor from that slot before replay.  Identity is an immutable tuple of
    scalar equation/configuration/layout information, never a tensor or a
    mutable model object.
    """

    __slots__ = ("_scope", "_previous")

    def __init__(self, entries: list, *, replay: bool) -> None:
        self._scope = _SegmentRoots(entries, replay)
        self._previous: _SegmentRoots | None = None

    def __enter__(self) -> None:
        global _SEGMENT_ROOTS
        self._previous = _SEGMENT_ROOTS
        _SEGMENT_ROOTS = self._scope

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        global _SEGMENT_ROOTS
        _SEGMENT_ROOTS = self._previous
        scope = self._scope
        kept = len(scope.entries)
        if scope.replay or exc_type is not None:
            scope.entries.clear()
        if scope.replay and exc_type is None and scope.taken != kept:
            raise RuntimeError(
                f"the recompute took {scope.taken} of the {kept} implicit KPP "
                "roots its forward kept"
            )
        return False


def segment_active(*, enabled: bool = True) -> bool:
    """Whether a checkpoint segment owns this call's retained-root lifecycle."""
    return bool(enabled and _SEGMENT_ROOTS is not None)


def replay_active(*, enabled: bool = True) -> bool:
    """Whether this call must consume a root from a reverse segment.

    This host-only query lets the caller omit work used solely for Newton's
    initial guess. It does not consume or validate the next root.
    """
    scope = _SEGMENT_ROOTS
    return bool(enabled and scope is not None and scope.replay)


def replay_root(template: torch.Tensor, *, identity: tuple, enabled: bool = True):
    """Take this station's saved state, or None outside a reverse segment.

    Storage strides can change during checkpoint transfer; shape, dtype,
    device and the caller's equation identity must agree.  The state is still
    subject to the caller's original residual and singular-root checks.
    Disabling reuse in both passes preserves ordinary Newton recomputation.
    """
    scope = _SEGMENT_ROOTS
    if not enabled or scope is None or not scope.replay:
        return None
    state, expected, record = scope.take()
    if expected != identity:
        raise RuntimeError(
            "the recompute's implicit KPP equation differs from the one its forward kept"
        )
    if (state.shape != template.shape or state.dtype != template.dtype
            or state.device != template.device):
        raise RuntimeError(
            "the recompute's implicit KPP layout differs from the one its forward kept"
        )
    return state, record


def retain_root(state: torch.Tensor, forward_record: dict, *, identity: tuple,
                enabled: bool = True) -> None:
    """Give a forward segment an owned copy after the original residual passes."""
    scope = _SEGMENT_ROOTS
    if not enabled or scope is None:
        return
    if scope.replay:
        raise RuntimeError("an implicit KPP recompute must consume its saved root")
    scope.entries.append((state.detach().clone(), identity, copy.deepcopy(forward_record)))
