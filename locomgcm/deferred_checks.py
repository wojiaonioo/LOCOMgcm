"""Step-boundary validation with production, debug and strict check levels.

The production level observes solver residuals at the compiled-step boundary.
Debug adds field finiteness checks there; strict checks individual sites.

Levels (``LOCOMGCM_CHECK_LEVEL``, resolved once at import, ``production`` by default)
-----------------------------------------------------------------------------
``production``
    No per-site check runs anywhere.  The step's one probe is
    :func:`inspect_step`, which asks whether the free-surface solve's residual
    -- and, in a non-hydrostatic run, the internal solve's -- came back finite.
    That costs nothing: the PCG loop tests convergence on the host, so both
    numbers are 0-d tensors already sitting on the host side of the boundary
    when finalize reads them.

    Its detection bound is two steps.  A non-finite momentum field enters
    ``div_star`` and the cg2d right-hand side within the same step, so the
    residual is non-finite immediately; a non-finite tracer enters the next
    step's momentum tendency through the equation of state and the hydrostatic
    pressure gradient, so it shows up one step later.  Finalize inspects inside
    the step, and the bound carries one step of implementation margin.

    The known gap: the double-where masks (``cg2d_preconditioner.py:11``,
    ``implicit_free_surface_cg2d.py:1193,1218``, ``step_segments.py:174-178``)
    hold dry points at exactly zero rather than ``0 * NaN``, so a NaN on a dry
    point never reaches the residual.  The same masks keep dry values out of
    every wet result, so such a NaN is numerically unobservable and does not
    propagate; it is visible only on write-out, where the output layer's mask
    handles it.  This is a stated, accepted gap, not an oversight.

``debug``
    The production probe plus one ``isfinite().all()`` per prognostic field
    (eta, u, v, temp, salt) in finalize.  Five reductions per step, eager, at
    the boundary.

``strict``
    Every historical per-site check, raising synchronously at the offending
    call with exact element indices.  A host branch on a tensor value cannot be
    traced, so this level and ``torch.compile`` are mutually exclusive and the
    runner refuses the combination up front rather than degrading quietly
    (:func:`require_level_supports_compile`).

``LOCOMGCM_DEFERRED_CHECKS`` is retired.  For one release cycle ``=0`` still selects
``strict``; any other value is ignored.  Either way the process announces the
substitution once on stderr.

Numerical-failure channel
-------------------------
Native column solvers compute a Thomas error code as part of their own
diagnostic contract, so the code exists whether or not anyone looks at it.
Their opaque runtime implementations publish it with
:func:`publish_solver_code`; traced code never touches the channel, so no
compiled graph reads, saves or guards on it.  A code lands in the slot of its
``(family, field shape)`` pair and accumulates with the kernel's own
within-call rule, the minimum.  A code is a flat index into ``(level, error
type, column)`` of the field it solved, so the minimum keeps the shallowest
level and the earliest error type the window produced rather than a later
consequence of it.  Forward and reverse producers share one slot table, one
reset and one decoder; each family registers its decoder
(:func:`register_solver_code_family`), and this module imports no solver.

A window ends with one judgment that reads and resets every slot.  Who judges
is decided by binding, never by a caller remembering to look:

* A bound owner (:func:`bind_window_owner`; a ``CommSession`` binds itself for
  its differentiation window) receives every failure of the window through its
  ``record_numerical_failure`` and judges collectively in its agreement
  (:func:`window_agreement`).  Nothing raises on one rank inside an owned
  window, so no rank abandons a communication schedule its peers still run.
* Without an owner, a failure raises where it is observed (step boundary,
  :func:`inspect_final`, a call judged synchronously), and a reverse pass that
  touches the channel is judged before ``backward()`` returns, through the
  autograd engine's final callback.  A library node that nests a backward
  (:func:`reverse_segment`) hands that judgment to the enclosing pass, so a
  checkpointed window is read once.  A raised failure ends its window: the
  slots are reset, so a later window in the same process starts clean.

The step boundary reads the slots through an asynchronous copy inspected at a
later boundary -- at most four later, the ring's depth -- so it never stalls the
step loop.  That is a different bound from the residual probe's two steps, and
the two are not interchangeable.  A call with no prepared sink on its device
(a column-level use outside any run), a call under ``strict``, or a call whose
slot table is full is judged at the call."""

from __future__ import annotations

import contextlib
import os
import sys
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from locomgcm.state import PrognosticState
    from locomgcm.step.contract import ShapeContract, StepDiagnostics

__all__ = [
    "CHECK_LEVELS",
    "DEBUG",
    "DeferredCheckError",
    "PRODUCTION",
    "STRICT",
    "SolverCodeAgreement",
    "bind_window_owner",
    "check_level",
    "declared_field_shape",
    "inspect_final",
    "inspect_step",
    "prepare_solver_codes",
    "publish_solver_code",
    "register_solver_code_family",
    "release_window_owner",
    "require_level_supports_compile",
    "resolve_step_probe",
    "reverse_segment",
    "strict_checks",
    "window_agreement",
]

PRODUCTION = "production"
DEBUG = "debug"
STRICT = "strict"
#: The accepted spellings of ``LOCOMGCM_CHECK_LEVEL``, in increasing coverage order.
CHECK_LEVELS: tuple[str, ...] = (PRODUCTION, DEBUG, STRICT)

_LEVEL_ENV = "LOCOMGCM_CHECK_LEVEL"
_RETIRED_ENV = "LOCOMGCM_DEFERRED_CHECKS"

#: The prognostic fields the ``debug`` level probes, in state order.
PROBED_FIELDS: tuple[str, ...] = ("eta", "u", "v", "temp", "salt")

#: How each location extends the centred field, as ``(extra_rows, extra_cols)``.
#: A tracer column sits at a cell centre (``nx`` of them across), a u column on
#: a western face (``nx + 1``) and a v column on a southern face (``ny + 1``).
_LOCATION_EXTENTS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"center": (0, 0), "west_face": (0, 1), "south_face": (1, 0)}
)



class DeferredCheckError(RuntimeError):
    """A step-boundary check found a non-finite value or a solver error code."""

    #: The (family, shape, code) solver entries this error reports; a field,
    #: residual or codec check reports none.
    entries: tuple = ()
    #: A local observation is stamped by its window; agreed peer reports retain
    #: their original rank and never become a new local numerical observation.
    origin_rank: int | None = None
    remote_failure: bool = False
    origin_reports: tuple = ()


# ---------------------------------------------------------------------------
# Level resolution: a run constant, read once at import.
# ---------------------------------------------------------------------------


def _announce_retired_env(detail: str) -> None:
    global _RETIRED_ANNOUNCED
    if _RETIRED_ANNOUNCED:
        return
    _RETIRED_ANNOUNCED = True
    print(
        f"{_RETIRED_ENV} is retired; use {_LEVEL_ENV}="
        f"{'|'.join(CHECK_LEVELS)} instead ({detail})",
        file=sys.stderr,
    )


def _resolve_level() -> str:
    """The level this process runs at, from the environment."""
    requested = (os.environ.get(_LEVEL_ENV) or "").strip().casefold()
    retired = (os.environ.get(_RETIRED_ENV) or "").strip()
    if requested:
        if requested not in CHECK_LEVELS:
            raise ValueError(
                f"{_LEVEL_ENV} must be one of "
                f"{', '.join(CHECK_LEVELS)}; got {requested!r}"
            )
        if retired:
            _announce_retired_env(f"{_LEVEL_ENV}={requested} decides this run")
        return requested
    if retired:
        if retired == "0":
            _announce_retired_env(f"{_RETIRED_ENV}=0 read as {_LEVEL_ENV}={STRICT}")
            return STRICT
        _announce_retired_env(f"{_RETIRED_ENV}={retired} ignored")
    return PRODUCTION


_RETIRED_ANNOUNCED = False
#: Read by :func:`strict_checks` from inside traced code: a module-level bool
#: that never changes during a run is constant-folded at trace time, so the
#: guarded check bodies are absent from the graph rather than skipped in it.
_STRICT = False
_LEVEL = PRODUCTION
#: Whether the step boundary probes the prognostic fields.  True at ``debug``
#: and ``strict``, and at ``production`` on a run with no residual to watch
#: (:func:`resolve_step_probe`).
_FIELD_PROBE = False


def _apply_level(level: str) -> None:
    global _LEVEL, _STRICT, _FIELD_PROBE
    _LEVEL = level
    _STRICT = level == STRICT
    _FIELD_PROBE = level != PRODUCTION


def check_level() -> str:
    """The level this process resolved at import (or at the last test reset)."""
    return _LEVEL


def strict_checks() -> bool:
    """True when the per-site synchronous checks run.

    Every guarded check body in the solvers, the dynamics modules and the open
    boundary reads this.  It is a plain module-level bool, so a traced step
    folds the branch away and the reduction behind it never enters the graph.
    """
    return _STRICT


def require_level_supports_compile(target: str) -> None:
    """Refuse ``strict`` under ``torch.compile`` instead of degrading quietly.

    A strict check is a host branch on a tensor value.  Under tracing that is
    either a graph break or -- with the fullgraph posture the contract asks for
    -- a hard error somewhere deep in a solver, whose message would say nothing
    about the level that caused it.  Say it here instead.
    """
    if not _STRICT:
        return
    raise RuntimeError(
        f"{_LEVEL_ENV}={STRICT} cannot be compiled (target {target!r}): the "
        "strict checks are host branches on tensor values and there is no "
        f"silent fallback.  Run with {_LEVEL_ENV}={DEBUG} to keep whole-field "
        "checks at the step boundary, or disable compilation."
    )


def resolve_step_probe(stepper: Any) -> Mapping[str, Any]:
    """Settle what ``production`` can actually watch on this run.

    Production's probe is the free-surface solve's residual, so a stepper that
    runs no such solve leaves it with nothing to look at: the step boundary
    would inspect an empty diagnostics object every step and report a clean run
    whatever the state did.  The steppers without a phase
    split -- AB2, SSPRK, split-explicit -- are exactly that case; they have no
    ``step_kernel``, and ``model.step_kernel`` reports empty diagnostics for
    them.

    Steppers of that kind are legitimate -- the direct API and the test suite
    drive them, even though every case in ``experiments/configs/cases`` runs
    ``implicit_free_surface`` -- so this does not refuse the run: it gives
    production the only probe that works there, the whole-field check ``debug``
    uses.  Checking nothing silently is the failure this closes; checking the
    fields costs five reductions per step at the boundary, outside the timed
    window, and no observer enters the kernel either way.  ``debug`` and
    ``strict`` already probe the state and are unaffected.

    Returns what it decided, so the caller can put it where a batch run can
    read it afterwards.  The stderr line is once **per run**, not once per
    process: a caller that starts several runs is told about each of them, and
    stderr is the wrong place to look for a batch job's posture anyway, which
    is why the verdict is returned rather than only printed.
    """
    global _FIELD_PROBE
    _FIELD_PROBE = _LEVEL != PRODUCTION
    substituted = (
        _LEVEL == PRODUCTION and getattr(stepper, "step_kernel", None) is None
    )
    if not substituted:
        return MappingProxyType(
            {
                "level": _LEVEL,
                "probe": "fields" if _FIELD_PROBE else "residual",
                "substituted": False,
                "reason": None,
            }
        )
    _FIELD_PROBE = True
    reason = (
        f"{type(stepper).__name__} runs no free-surface solve, so no step "
        "publishes the residual production watches"
    )
    print(
        f"{_LEVEL_ENV}={PRODUCTION} has no residual to watch on this run: "
        f"{reason}.  The step boundary checks the prognostic fields instead "
        f"(five reductions per step, as at {_LEVEL_ENV}={DEBUG}); set "
        f"{_LEVEL_ENV}={DEBUG} to declare that explicitly.",
        file=sys.stderr,
    )
    return MappingProxyType(
        {
            "level": _LEVEL,
            "probe": "fields",
            "substituted": True,
            "reason": reason,
        }
    )


# ---------------------------------------------------------------------------
# The numerical-failure channel.
# ---------------------------------------------------------------------------

#: The kernels' healthy code, and the identity of the minimum they accumulate.
_NO_ERROR = torch.iinfo(torch.int64).max
#: Distinct ``(family, field shape)`` pairs one device sink holds.  A run uses
#: at most ten (three forward and seven reverse sites on one grid); a call
#: beyond the table is judged synchronously rather than dropped.
_SLOT_CAPACITY = 64
#: Pinned staging copies in flight for the step-boundary read.
_RING_DEPTH = 4

_DECODERS: dict[str, Callable[[int, tuple[int, ...]], str]] = {}


class _Sink(NamedTuple):
    codes: torch.Tensor  # int64[_SLOT_CAPACITY], _NO_ERROR when healthy
    index: torch.Tensor  # arange: the scatter index of every slot


class _Copy(NamedTuple):
    buffer: torch.Tensor
    event: Any
    generation: int


_SINK: _Sink | None = None
#: ``(family, shape)`` to slot, for the sink's lifetime: a captured graph keeps
#: writing the slot it was captured with, so a slot is never reassigned.
_SLOTS: dict[tuple[str, tuple[int, ...]], int] = {}
_SLOT_KEYS: list[tuple[str, tuple[int, ...]]] = []
#: Field shape to the location the plan declared there; message text only.
_LOCATIONS: dict[tuple[int, ...], str] = {}
_LOCATION_CONTRACT: Any = None
#: Advanced by every reset, so a staging copy of an ended window never reports.
_GENERATION = 0
_RING_FREE: list[tuple[torch.Tensor, Any]] = []
_RING_PENDING: deque[_Copy] = deque()
_OWNER: weakref.ReferenceType | None = None
_CLAIMED_TASK: int | None = None
_NESTED_REVERSE = 0
_CAPACITY_ANNOUNCED = False


def register_solver_code_family(
    family: str, decode: Callable[[int, tuple[int, ...]], str]
) -> None:
    """Declare a producer family and the decoder of its codes.

    ``decode(code, shape)`` returns the text for one code of a call that solved
    a field of ``shape``.  The family's module registers it at import, next to
    the kernel that defines the code layout; a family nobody registered cannot
    publish.
    """
    existing = _DECODERS.get(family)
    if existing is not None and existing is not decode:
        raise ValueError(f"solver code family {family!r} is already registered")
    _DECODERS[family] = decode


def declared_field_shape(shapes: "ShapeContract", location: str) -> tuple[int, ...]:
    """The extents of the field solved at ``location``, per the run's contract.

    ``center`` is the prognostic tracer shape; ``west_face`` carries one more
    column and ``south_face`` one more row, which is where the u and v solves
    live on a C grid.
    """
    extra_rows, extra_columns = _LOCATION_EXTENTS[location]
    lead = () if shapes.batch is None else (int(shapes.batch),)
    return lead + (
        int(shapes.nz),
        int(shapes.ny) + extra_rows,
        int(shapes.nx) + extra_columns,
    )


def _new_event() -> Any:
    return torch.cuda.Event()


def prepare_solver_codes(device: Any, shapes: "ShapeContract | None" = None) -> None:
    """Make the device sink exist before any producer runs. **Eager: prepare owns this.**

    One tensor per device for the process.  Compiled runtime calls and
    captured graphs write its address, so it is never replaced while the device
    stays the same; a judgment resets it in place.  ``shapes`` only names the
    plan's field locations in decoded messages; any field shape may publish,
    so a second grid in the same process needs no new preparation contract.

    The channel owns the context its buffers are created in: here and nowhere
    else (publication, observation and judgment only write into them), as
    normal tensors whatever autograd mode the caller is in -- the run's final
    read, a judgment's reset or a later gradient window write in place outside
    ``torch.inference_mode``.  A production forward runs in that mode;
    ``build_step_plan`` prepares under ``ordinary_tensor_mode``, but this layer
    does not rely on its caller having left the mode (a caller may prepare from
    inside it) -- and never inside a trace or a CUDA graph capture, where the
    buffer would belong to the graph.
    """
    global _SINK, _LOCATION_CONTRACT
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if _SINK is None or _SINK.codes.device != device:
        if torch.compiler.is_compiling() or (
                device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
            raise RuntimeError(
                "the solver-code sink is created eagerly, outside any trace or CUDA "
                "graph capture: prepare it before the traced or captured step")
        if _SINK is not None:
            # Judge the old device's window before its slots stop being read.
            _judge()
        _RING_FREE.clear()
        _RING_PENDING.clear()
        _SLOTS.clear()
        _SLOT_KEYS.clear()
        with torch.inference_mode(False):
            _SINK = _Sink(
                torch.full((_SLOT_CAPACITY,), _NO_ERROR, dtype=torch.int64, device=device),
                torch.arange(_SLOT_CAPACITY, dtype=torch.int64, device=device),
            )
            if device.type != "cpu":
                for _ in range(_RING_DEPTH):
                    _RING_FREE.append((
                        torch.empty(_SLOT_CAPACITY, dtype=torch.int64).pin_memory(),
                        _new_event(),
                    ))
    if shapes is not None and shapes != _LOCATION_CONTRACT:
        _LOCATION_CONTRACT = shapes
        for location in _LOCATION_EXTENTS:
            shape = declared_field_shape(shapes, location)
            _LOCATIONS[shape] = location
            if len(shape) == 3:
                # Native kernels see a three-dimensional field as a unit batch.
                _LOCATIONS[(1,) + shape] = location


def _slot(family: str, shape: tuple[int, ...]) -> int | None:
    global _CAPACITY_ANNOUNCED
    key = (family, shape)
    slot = _SLOTS.get(key)
    if slot is not None:
        return slot
    if family not in _DECODERS:
        raise ValueError(f"solver code family {family!r} was never registered")
    if len(_SLOT_KEYS) == _SLOT_CAPACITY:
        if not _CAPACITY_ANNOUNCED:
            _CAPACITY_ANNOUNCED = True
            print(
                f"solver code sink holds {_SLOT_CAPACITY} (family, shape) slots; "
                f"{family} on {shape} and later new pairs are judged at the call",
                file=sys.stderr,
            )
        return None
    slot = len(_SLOT_KEYS)
    _SLOTS[key] = slot
    _SLOT_KEYS.append(key)
    return slot


def publish_solver_code(
    family: str, code: torch.Tensor, shape: tuple[int, ...]
) -> None:
    """Publish one call's error code. **Runtime implementations only, never traced.**

    ``code`` is the 0-d int64 the kernel produced, ``_NO_ERROR`` when healthy,
    and ``shape`` is the shape of the field it indexes.  The write is one
    atomic minimum into the call's slot, ordered on the call's stream: no
    reduction, no readback, no branch on the value.  Under ``strict``, with no
    sink on the code's device, or with the slot table full, the call is judged
    here instead, synchronously.
    """
    shape = tuple(int(extent) for extent in shape)
    sink = _SINK
    slot = None
    if not _STRICT and sink is not None and sink.codes.device == code.device:
        slot = _slot(family, shape)
    if slot is None:
        if family not in _DECODERS:
            raise ValueError(f"solver code family {family!r} was never registered")
        value = int(code.item())
        if value != _NO_ERROR:
            _report(_code_error(((family, shape, value),)), codes_read=False)
        return
    sink.codes.scatter_reduce_(
        0, sink.index[slot:slot + 1], code.reshape(1), reduce="amin"
    )
    _claim_reverse()


# --- Owners -----------------------------------------------------------------


def bind_window_owner(owner: Any) -> None:
    """``owner`` judges every numerical failure until it releases the window.

    The owner provides ``record_numerical_failure(error)`` and concludes the
    window through :func:`window_agreement`.  A later binding replaces an
    earlier one; codes an abandoned window left unjudged stay in the slots and
    fail the next owner's agreement rather than vanish.  The binding holds a
    weak reference: an owner that no longer exists owns nothing.
    """
    global _OWNER
    _OWNER = weakref.ref(owner)


def release_window_owner(owner: Any) -> None:
    """End ``owner``'s window; whatever its agreement did not conclude is dropped.

    The owner releases after its collective verdict, or after a collective
    abort that already rejected the window, so the slots it drops belong to a
    window every participant has refused.
    """
    global _OWNER
    if _OWNER is not None and _OWNER() is owner:
        _OWNER = None
        _reset()


def _bound_owner() -> Any:
    return None if _OWNER is None else _OWNER()


class SolverCodeAgreement:
    """One owner's view of the channel at its qualification boundary.

    :meth:`row` is this participant's status row -- the owner's non-negative
    flag word in column 0, then every slot -- for a MAX agreement in which
    every row has a single writer, so no two participants' codes combine.
    :meth:`verdict` reads the agreed rows: whether any row failed, whether any
    row holds a code, and this participant's own slots decoded against its own
    field shapes.  It ends the window for both channels.  The owner never
    learns what a slot holds or how a code is laid out.
    """

    width = 1 + _SLOT_CAPACITY

    def __init__(self) -> None:
        self._sink = _SINK
        self._keys = tuple(_SLOT_KEYS)
        self._sealed = False

    def seal(self) -> None:
        """End attribution after the status row has taken ownership of the codes."""
        if not self._sealed:
            _reset()
            self._sealed = True

    def row(self, flag: int, device: Any, *, out=None) -> torch.Tensor:
        row = torch.empty(self.width, dtype=torch.int64, device=device) if out is None else out
        if row.shape != (self.width,) or row.dtype != torch.int64:
            raise ValueError("invalid solver code status storage")
        row[0].fill_(int(flag))
        if self._sink is None:
            row[1:].fill_(_NO_ERROR)
        else:
            row[1:].copy_(self._sink.codes)
        return row

    def verdict(
        self, rows: list[list[int]], rank: int
    ) -> tuple[bool, bool, DeferredCheckError | None]:
        if any(len(row) != self.width for row in rows):
            raise ValueError("invalid solver code agreement payload")
        coded = any(value != _NO_ERROR for row in rows for value in row[1:])
        failed = coded or any(row[0] for row in rows)
        error = _decode_values(rows[rank][1:], self._keys)
        self.seal()
        return failed, coded, error


def window_agreement() -> SolverCodeAgreement:
    """The channel's participant in an owner's agreement (each one ends the window)."""
    return SolverCodeAgreement()


# --- Judgment ---------------------------------------------------------------


def _reset() -> None:
    global _GENERATION, _CLAIMED_TASK
    _CLAIMED_TASK = None
    _GENERATION += 1
    if _SINK is not None:
        _SINK.codes.fill_(_NO_ERROR)


def _describe(family: str, value: int, shape: tuple[int, ...]) -> str:
    decode = _DECODERS.get(family)
    try:
        if decode is not None:
            return decode(value, shape)
    except Exception:  # noqa: BLE001 - report the raw code instead
        pass
    return (
        f"code {value} does not decode against shape {shape}; rerun with "
        f"{_LEVEL_ENV}={STRICT} for the column and level"
    )


def _entries(values: list[int], keys: Any = None) -> list[tuple[str, tuple[int, ...], int]]:
    keys = _SLOT_KEYS if keys is None else keys
    return [(family, shape, int(value))
            for (family, shape), value in zip(keys, values) if value != _NO_ERROR]


def _entry_line(family: str, shape: tuple[int, ...], value: int) -> str:
    location = _LOCATIONS.get(shape)
    name = family if location is None else f"{family}[{location}]"
    return f"{name}: code={value}, shape={shape}; {_describe(family, value, shape)}"


def _code_error(entries: Any) -> DeferredCheckError | None:
    entries = tuple(entries)
    lines = [_entry_line(*entry) for entry in entries]
    if not lines:
        return None
    error = DeferredCheckError(
        "a native solver reported an error:\n  " + "\n  ".join(lines)
    )
    error.entries = entries
    return error


def window_numerical_error(*errors: Exception | None) -> DeferredCheckError | None:
    """One window's numerical failures as one error, each solver entry once.

    The first failure keeps its object when the others add nothing; otherwise
    a new error carries its text plus the entries and messages it lacks, in
    observation order.  Only ``DeferredCheckError`` values take part.
    """
    found = [error for error in errors if isinstance(error, DeferredCheckError)]
    if not found:
        return None
    local = [error for error in found if not error.remote_failure]
    # The collective agreement gathers peer reports again at completion. A new
    # local observation must not include a previous peer report in its own text.
    found = local or found
    first = found[0]
    entries = list(first.entries)
    texts = [str(first)]
    for error in found[1:]:
        if error is first:
            continue
        extra = [entry for entry in error.entries if entry not in entries]
        if error.entries:
            entries.extend(extra)
            if extra:
                texts.append("the window's native solver codes:\n  "
                             + "\n  ".join(_entry_line(*entry) for entry in extra))
        elif not any(str(error) in text for text in texts):
            texts.append(str(error))
    if len(texts) == 1:
        return first
    merged = DeferredCheckError("\n".join(texts))
    merged.entries = tuple(entries)
    merged.origin_rank = first.origin_rank
    merged.remote_failure = all(error.remote_failure for error in found)
    merged.origin_reports = tuple(dict.fromkeys(
        report for error in found for report in error.origin_reports))
    return merged


def _decode_values(
    values: list[int], keys: Any = None
) -> DeferredCheckError | None:
    return _code_error(_entries(values, keys))


def _report(error: Exception, *, codes_read: bool) -> None:
    """Deliver one failure: to the bound owner, or raise and end the window.

    An unowned window ends here, so the raise also names every slot the
    window holds now (a blocking read on the failure path), after the
    failure itself: the earliest observation comes first, later codes of the
    same window are not dropped by the reset.
    """
    owner = _bound_owner()
    if owner is not None:
        owner.record_numerical_failure(error)
        return
    entries = ([] if codes_read else _collect()) + _live_entries()
    message = str(error)
    extra = [line for line in dict.fromkeys(_entry_line(*entry) for entry in entries)
             if line not in message]
    _reset()
    if not extra:
        raise error
    header = "" if codes_read else "\nthe window's native solver codes:"
    raise DeferredCheckError(message + header + "\n  " + "\n  ".join(extra)) from error


def report_numerical_failure(error: DeferredCheckError) -> None:
    """Deliver a numerical failure found outside the step boundary and the native solvers.

    For a check elsewhere that finds the run's state non-finite (a checkpoint
    head's codec).  Like every failure of the channel it goes to the bound
    owner, which records it and keeps the communication schedule, or raises
    here together with the window's solver codes.
    """
    _report(error, codes_read=False)


def _live_entries() -> list[tuple[str, tuple[int, ...], int]]:
    return [] if _SINK is None else _entries(_SINK.codes.tolist())


def _collect() -> list[tuple[str, tuple[int, ...], int]]:
    """Blocking read: the earliest failing step snapshot, else the live slots."""
    if _SINK is None:
        return []
    first: list[tuple[str, tuple[int, ...], int]] = []
    while _RING_PENDING:
        entry = _RING_PENDING.popleft()
        entry.event.synchronize()
        _RING_FREE.append((entry.buffer, entry.event))
        if not first and entry.generation == _GENERATION:
            first = _entries(entry.buffer.tolist())
    return first or _live_entries()


def _judge() -> None:
    entries = _collect()
    if entries:
        _report(_code_error(entries), codes_read=True)


def _judge_reverse() -> None:
    """The engine's final callback: the claimed reverse pass is complete."""
    global _CLAIMED_TASK
    _CLAIMED_TASK = None
    _judge()


def _claim_reverse() -> None:
    """Judge the window when the reverse pass running this call completes.

    Without an owner, a reverse pass that publishes or observes a code queues
    one judgment on the autograd engine's final callbacks; ``backward()``
    raises it before returning, so no caller can reach the gradient first.  A
    pass nested inside :func:`reverse_segment` leaves the judgment to its
    enclosing pass, which claimed it on entry.
    """
    global _CLAIMED_TASK
    if _NESTED_REVERSE or _bound_owner() is not None:
        return
    task = torch._C._current_graph_task_id()
    if task < 0 or task == _CLAIMED_TASK:
        return
    _CLAIMED_TASK = task
    torch.autograd.Variable._execution_engine.queue_callback(_judge_reverse)


@contextlib.contextmanager
def reverse_segment() -> Iterator[None]:
    """A library node whose backward recomputes and runs a nested backward.

    The enclosing reverse pass claims the window's judgment here, and every
    claim made inside the nested pass defers to it, so a checkpointed window
    with one segment per step is read once, after its last segment, instead of
    once per segment.  The count is restored when the node raises.
    """
    global _NESTED_REVERSE
    _claim_reverse()
    _NESTED_REVERSE += 1
    try:
        yield
    finally:
        _NESTED_REVERSE -= 1


def _inspect_copy(entry: _Copy) -> None:
    _RING_FREE.append((entry.buffer, entry.event))
    if entry.generation != _GENERATION:
        return
    error = _decode_values(entry.buffer.tolist())
    if error is not None:
        _report(error, codes_read=True)


def _observe_solver_codes() -> None:
    """Inspect finished copies, then start one more. **No synchronisation.**

    The sink lives on the device, so reading it is the one part of the
    boundary that can stall, and a stall inside the step loop both costs the
    host its lead over the device and changes what the step timer measures.
    The copy started here is inspected at a later boundary, at most
    :data:`_RING_DEPTH` later: when the ring is empty the oldest copy is waited
    for rather than skipped, so the detection bound holds unconditionally.  On
    a host-only run the sink is host memory and is read directly.
    """
    sink = _SINK
    if sink is None:
        return
    if sink.codes.device.type == "cpu":
        error = _decode_values(sink.codes.tolist())
        if error is not None:
            _report(error, codes_read=True)
        return
    while _RING_PENDING and _RING_PENDING[0].event.query():
        _inspect_copy(_RING_PENDING.popleft())
    if not _RING_FREE:
        entry = _RING_PENDING.popleft()
        entry.event.synchronize()
        _inspect_copy(entry)
    buffer, event = _RING_FREE.pop()
    buffer.copy_(sink.codes, non_blocking=True)
    event.record()
    _RING_PENDING.append(_Copy(buffer, event, _GENERATION))
    # A boundary inside a reverse pass (checkpoint replay) belongs to it.
    _claim_reverse()

# ---------------------------------------------------------------------------
# The step boundary's own checks.
# ---------------------------------------------------------------------------

_RESIDUALS = ("cg2d_residual", "cg3d_residual")


def _inspect_residuals(diagnostics: "StepDiagnostics") -> None:
    # CG2D's publisher waits for the pinned-copy event before constructing these
    # CPU tensors; no in-flight host storage reaches the finiteness read.
    unstable = [
        name
        for name in _RESIDUALS
        if (value := getattr(diagnostics, name, None)) is not None
        and not bool(torch.isfinite(value))
    ]
    if not unstable:
        return
    raise DeferredCheckError(
        "the step's solve returned a non-finite residual "
        f"({', '.join(unstable)}): the prognostic state is no longer finite.  "
        "The residual is the production probe, so the event itself happened at "
        "most 2 steps ago (a tracer reaches the residual through the next "
        f"step's momentum tendency); rerun with {_LEVEL_ENV}={DEBUG} to name "
        f"the field, or {_LEVEL_ENV}={STRICT} for the element index."
    )


def _inspect_fields(state: "PrognosticState") -> None:
    unstable = [
        name
        for name in PROBED_FIELDS
        if not bool(torch.isfinite(getattr(state, name).data).all())
    ]
    if not unstable:
        return
    raise DeferredCheckError(
        f"non-finite prognostic field(s) at the step boundary: "
        f"{', '.join(unstable)}.  Rerun with {_LEVEL_ENV}={STRICT} for the "
        "element index and the producing call site."
    )



def inspect_step(diagnostics: "StepDiagnostics", state: "PrognosticState") -> None:
    """The step's whole verdict. **Eager, and outside the step timer.**

    The runner calls this inside ``finalize_step`` before publication, after
    the timer's window closes.  At ``production`` nothing here synchronises --
    the residuals are 0-d host tensors (CG2D publication has already waited
    for its asynchronous copy's CUDA event),
    and the solver codes are read through an asynchronous copy inspected a step
    later.  ``debug`` adds the five whole-field reductions, which do
    synchronise and are meant to.  A failure goes to the window's owner when
    one is bound; otherwise it raises here, reporting the solver codes the
    window holds as well, and the window ends.
    """
    try:
        _inspect_residuals(diagnostics)
        if _FIELD_PROBE:
            _inspect_fields(state)
    except DeferredCheckError as error:
        _report(error, codes_read=False)
    _observe_solver_codes()


def inspect_final() -> None:
    """Blocking read of every slot and staging copy at the end of a forward run.

    The step-boundary read is asynchronous, so at the end of a run copies may
    still be in flight and the slots may hold a code no copy has carried yet.
    The earliest failing copy is reported, else the live slots.  Without an
    owner a failure raises and resets the slots, so a later run in the same
    process starts clean; with one, the codes stay for its agreement and the
    returned forward values remain candidates until that agreement passes.
    The sink stays allocated: a caller may differentiate the result afterwards.
    """
    _judge()


def _reset_for_tests() -> None:
    """Re-read the level from the environment and drop the channel's state."""
    global _SINK, _RETIRED_ANNOUNCED, _OWNER, _CLAIMED_TASK, _NESTED_REVERSE
    global _GENERATION, _LOCATION_CONTRACT, _CAPACITY_ANNOUNCED
    _SINK = None
    _SLOTS.clear()
    _SLOT_KEYS.clear()
    _LOCATIONS.clear()
    _LOCATION_CONTRACT = None
    _RING_FREE.clear()
    _RING_PENDING.clear()
    _OWNER = None
    _CLAIMED_TASK = None
    _NESTED_REVERSE = 0
    _GENERATION += 1
    _CAPACITY_ANNOUNCED = False
    _RETIRED_ANNOUNCED = False
    _apply_level(_resolve_level())


_apply_level(_resolve_level())
