"""Checkpoint scheduling — decides *when* to capture a checkpoint.

This is the SCHEDULE layer of the three-layer checkpoint mechanism
(schedule -> capture -> persist).  It is deliberately decoupled from
*what* is captured (``capture.py``) and *how* it is stored
(``persistence.py`` / ``writer.py``): a schedule only answers the
question "which integration steps should be checkpointed?".

All schedule kinds compile to the same ``frozenset[int]`` of trigger
steps, which the integration loop tests with ``step in trigger_set`` —
mirroring how the existing snapshot machinery already works
(``defgcm/runner.py`` ``snapshot_step_set``).

Supported kinds (see ``build_schedule``):
  * ``interval``    — every N steps (or every N seconds, step-derived)
  * ``window``      — only inside [start, end] seconds, at a cadence
  * ``explicit``    — an explicit set of global steps
  * composite       — the union of any of the above (default when a
                      yaml provides a list of entries)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence, runtime_checkable


def _seconds_to_step(seconds: float, dt: float) -> int:
    """Convert an absolute time (s) to the nearest integration step.

    ``round`` (not floor) so that a cadence that is an exact multiple of
    ``dt`` lands precisely, immune to float drift.
    """
    return int(round(float(seconds) / float(dt)))


@runtime_checkable
class CheckpointSchedule(Protocol):
    """Anything that can name the steps at which to checkpoint."""

    def trigger_steps(self, n_steps: int, dt: float) -> frozenset[int]:
        ...


@dataclass(frozen=True)
class IntervalSchedule:
    """Checkpoint every ``every_steps`` (or ``every_seconds``) steps.

    Exactly one of ``every_steps`` / ``every_seconds`` must be set.
    ``include_first``/``include_last`` add step 0 / the final step.
    """

    every_steps: int | None = None
    every_seconds: float | None = None
    include_first: bool = True
    include_last: bool = True

    def __post_init__(self) -> None:
        if (self.every_steps is None) == (self.every_seconds is None):
            raise ValueError(
                "IntervalSchedule needs exactly one of every_steps / every_seconds"
            )
        if self.every_steps is not None and self.every_steps < 1:
            raise ValueError("every_steps must be >= 1")
        if self.every_seconds is not None and self.every_seconds <= 0:
            raise ValueError("every_seconds must be > 0")

    def trigger_steps(self, n_steps: int, dt: float) -> frozenset[int]:
        stride = (
            int(self.every_steps)
            if self.every_steps is not None
            else _seconds_to_step(self.every_seconds, dt)
        )
        if stride < 1:
            raise ValueError(
                f"interval resolves to stride {stride} (< 1); "
                f"every_seconds={self.every_seconds} smaller than dt={dt}?"
            )
        steps = set(range(0, n_steps + 1, stride))
        if self.include_first:
            steps.add(0)
        else:
            steps.discard(0)
        if self.include_last:
            steps.add(n_steps)
        return frozenset(s for s in steps if 0 <= s <= n_steps)


@dataclass(frozen=True)
class WindowSchedule:
    """Checkpoint only inside a time window, at a cadence.

    ``start_seconds``/``end_seconds`` are absolute model-time bounds
    (inclusive).  ``every_seconds`` (or ``every_steps``) is the in-window
    cadence.  Use several of these (composite) for multi-window dumping,
    e.g. dense around a tidal peak, sparse elsewhere.
    """

    start_seconds: float
    end_seconds: float
    every_steps: int | None = None
    every_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.end_seconds < self.start_seconds:
            raise ValueError("WindowSchedule end_seconds < start_seconds")
        if (self.every_steps is None) == (self.every_seconds is None):
            raise ValueError(
                "WindowSchedule needs exactly one of every_steps / every_seconds"
            )

    def trigger_steps(self, n_steps: int, dt: float) -> frozenset[int]:
        stride = (
            int(self.every_steps)
            if self.every_steps is not None
            else _seconds_to_step(self.every_seconds, dt)
        )
        if stride < 1:
            raise ValueError(f"window cadence resolves to stride {stride} (< 1)")
        lo = max(0, _seconds_to_step(self.start_seconds, dt))
        hi = min(n_steps, _seconds_to_step(self.end_seconds, dt))
        return frozenset(range(lo, hi + 1, stride))


@dataclass(frozen=True)
class ExplicitSchedule:
    """Checkpoint at an explicit set of global steps."""

    steps: tuple[int, ...]

    def trigger_steps(self, n_steps: int, dt: float) -> frozenset[int]:
        return frozenset(s for s in self.steps if 0 <= s <= n_steps)


@dataclass(frozen=True)
class CompositeSchedule:
    """Union of several schedules (natural de-dup via set union)."""

    parts: tuple[CheckpointSchedule, ...]

    def trigger_steps(self, n_steps: int, dt: float) -> frozenset[int]:
        out: set[int] = set()
        for part in self.parts:
            out |= part.trigger_steps(n_steps, dt)
        return frozenset(out)


def build_schedule(entries: Sequence[dict] | None) -> CheckpointSchedule:
    """Compile a list of yaml-style entries into one schedule.

    Each entry is ``{"type": <kind>, ...}``.  An empty / missing list means
    no checkpoint writes.  First/final triggers must be requested explicitly
    through an interval entry's ``include_first``/``include_last`` fields or
    an explicit schedule.
    """
    parts: list[CheckpointSchedule] = []
    for entry in entries or ():
        kind = str(entry.get("type", "interval")).lower()
        if kind == "interval":
            parts.append(
                IntervalSchedule(
                    every_steps=entry.get("every_steps"),
                    every_seconds=entry.get("every_seconds"),
                    include_first=bool(entry.get("include_first", True)),
                    include_last=bool(entry.get("include_last", True)),
                )
            )
        elif kind == "window":
            parts.append(
                WindowSchedule(
                    start_seconds=float(entry["start_seconds"]),
                    end_seconds=float(entry["end_seconds"]),
                    every_steps=entry.get("every_steps"),
                    every_seconds=entry.get("every_seconds"),
                )
            )
        elif kind == "explicit":
            parts.append(ExplicitSchedule(steps=tuple(int(s) for s in entry["steps"])))
        else:
            raise ValueError(f"unknown checkpoint schedule type: {kind!r}")
    if not parts:
        return CompositeSchedule(parts=())
    if len(parts) == 1:
        return parts[0]
    return CompositeSchedule(parts=tuple(parts))


def resolve_trigger_steps(
    entries: Sequence[dict] | None, n_steps: int, dt: float
) -> frozenset[int]:
    """Convenience: build + evaluate in one call."""
    return build_schedule(entries).trigger_steps(n_steps, dt)
