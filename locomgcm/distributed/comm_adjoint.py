"""Functional transfer/SUM primitives and a single-traversal reverse sweep.

A session records every communication round as an opaque operator whose
autograd formula issues the dual round. A float64 ``link`` tensor threads the
rounds together, so one local backward traversal executes all dual rounds in
reverse forward order and every source expression receives its cotangent in
that same traversal. Its caller enters ``backward`` on every rank. The
standalone operators expose their single-round VJPs to PyTorch; the session
owns participation, order and failure agreement across rounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
from collections.abc import Callable
from typing import Any
import contextlib
import itertools
import weakref
from datetime import timedelta
from functools import wraps

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.utils._pytree import tree_flatten, tree_unflatten

from locomgcm.deferred_checks import (
    DeferredCheckError,
    window_agreement,
    window_numerical_error,
)
from locomgcm.gradient_boundary import GradientBoundary
from locomgcm.accel.graph_exit import graph_exit
from locomgcm.accel.tensor_context import forward_mode_active

from .halo_p2p import HALO_P2P_TAGS, _group_name, _receive_layout, _split_receives, halo_p2p
from .transport_failure import TransportFailure, TransportLifetime, transport_operation, wait_transport


@dataclass(frozen=True)
class RoundPlan:
    peers_send: tuple[int, ...]
    peers_recv: tuple[int, ...]
    tags_send: tuple[int, ...]
    tags_recv: tuple[int, ...]
    group_name: str
    round_id: int
    owner: torch.Tensor | None = field(default=None, compare=False, repr=False)

    def dual(self) -> RoundPlan:
        return RoundPlan(self.peers_recv, self.peers_send,
                         self.tags_recv, self.tags_send,
                         self.group_name, self.round_id)


_PROGRESS: dict[str, int] = {}
_VERDICT_PROGRESS: dict[str, int] = {}

# A CPU tensor carries a run-time owner through Dynamo without specializing
# a Python session id. Only live callers own sessions; autograd keeps weak refs.
_OWNER_IDS = itertools.count(1)
_OWNERS = weakref.WeakValueDictionary()
_CONTROL_GROUPS = weakref.WeakKeyDictionary()


def _objective_noop(*gradients):
    if any(value is not None for value in gradients):
        return False
    from locomgcm.deferred_checks import _bound_owner
    owner = _bound_owner()
    return owner is not None and owner._objective_collecting


@torch.compiler.disable
def _raw_eager_operation(session, function, *args, **kwargs):
    session._raw_eager_active = True
    try:
        return function(*args, **kwargs)
    finally:
        session._raw_eager_active = False


def _window_operation(function):
    @wraps(function)
    def execute(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except Exception as error:
            if self._phase not in ("forward", "replay", "reverse") or self.window._terminal:
                raise
            self.fail_transport(error)
            raise self.transport_error() from error
    return execute


def _control_group(group):
    if group is None:
        return None
    control = _CONTROL_GROUPS.get(group)
    if control is None:
        from .transport_failure import _group_timeout

        ranks = dist.get_process_group_ranks(group)
        # The data group's own timeout: verdicts wait as long as the data
        # rounds they replaced (Td is derived from it), and the first
        # session's rendezvous tolerates the same skew as the data group.
        try:
            timeout = _group_timeout(group, dist.get_backend(group))
        except Exception:  # noqa: BLE001 - no readable timeout: the c10d default
            timeout = c10d.default_pg_timeout
        try:
            control, _ = c10d._new_process_group_helper(
                len(ranks), dist.get_rank(group), ranks, c10d.Backend.GLOO,
                c10d._get_process_group_store(group), "locomgcm-control-" + _group_name(group),
                timeout=timeout)
        except Exception as error:
            raise CommSessionError(
                f"control group rendezvous for {_group_name(group)} failed: "
                f"{type(error).__name__}: {error}") from error
        c10d._world.pg_group_ranks[control] = {rank: index for index, rank in enumerate(ranks)}
        _CONTROL_GROUPS[group] = control
        weakref.finalize(group, _release_control, control)
    return control


def _release_control(control):
    if dist.is_initialized() and control in c10d._world.pg_map:
        dist.destroy_process_group(control)


def active_data_group(group=None):
    """Data domain of the bound window, including untaped atomic solves."""
    from locomgcm.deferred_checks import _bound_owner
    owner = _bound_owner()
    session = getattr(owner, "session", None)
    return group if session is None else session.group


def data_group_for_grid(grid, group=None):
    """Traceable grid binding; constant transport shares the abort domain."""
    session = getattr(grid, "comm_session", None)
    return group if session is None else session.group


def _round_owner(owner):
    if owner is None:  # Standalone primitive: no window bookkeeping.
        return None
    session = _OWNERS.get(int(owner.item()))
    if session is None:
        raise CommSessionError("communication session no longer exists")
    session._require_window()
    return session


def _record_round(session, record, result):
    if session is None:
        return
    # Called by the opaque operator at execution, never while Dynamo traces.
    # Validation follows the issued round; a failure aborts the data domain.
    error = None
    try:
        session._check_template(result)
    except Exception as failure:
        error = failure
    session._plans.append(record.plan)
    session._tape.append(record)
    if error is not None:
        session.fail_transport(error)
        raise session.transport_error() from error


def _progress(group_name: str) -> int:
    """Session data rounds issued on a group, independently of control verdicts."""
    return _PROGRESS.get(group_name, 0)


def _advance(group_name: str, kind: str) -> None:
    """Count one completed round; with the round log enabled, record it."""
    _PROGRESS[group_name] = _progress(group_name) + 1
    if _ROUND_LOG.path is not None:
        _ROUND_LOG.write(group_name, kind)


def _log_verdict(group_name: str, kind: str) -> None:
    """Record a verdict collective in the round log; verdicts are not data rounds."""
    number = _VERDICT_PROGRESS.get(group_name, 0) + 1
    _VERDICT_PROGRESS[group_name] = number
    if _ROUND_LOG.path is not None:
        _ROUND_LOG.write(group_name, "verdict:" + kind, number=number)


class _RoundLog:
    """``LOCOMGCM_COMM_ROUND_LOG=<dir>``: one JSON line per issued round and rank.

    Each line names the window, its phase and the session phase (the
    direction), the rank, the group, the round number on that group and the
    round kind, so two ranks' sequences can be compared round by round.
    """

    def __init__(self):
        import os
        self.path = os.environ.get("LOCOMGCM_COMM_ROUND_LOG") or None
        self.file = None

    def write(self, group_name, kind, *, number=None):
        import json
        import os
        from locomgcm.deferred_checks import _bound_owner
        if self.file is None:
            os.makedirs(self.path, exist_ok=True)
            rank = dist.get_rank() if dist.is_initialized() else 0
            self.file = open(os.path.join(self.path, f"rounds_rank{rank}.jsonl"), "a")
        owner = _bound_owner()
        session = getattr(owner, "session", None)
        self.file.write(json.dumps({
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "window": getattr(owner, "serial", None),
            "window_phase": getattr(owner, "phase", None),
            "session_phase": getattr(session, "_phase", None),
            "group": group_name, "round": _progress(group_name) if number is None else number,
            "kind": kind}) + "\n")
        self.file.flush()


_ROUND_LOG = _RoundLog()


def _transfer(sends, recv_like, peers_send, peers_recv, tags_send, tags_recv, group_name):
    group = c10d._resolve_process_group(group_name) if group_name else None
    group_name = _group_name(active_data_group(group))
    with transport_operation(group_name) as operation:
        return _transfer_data(sends, recv_like, peers_send, peers_recv, tags_send, tags_recv,
                              group_name, operation)


def _transfer_data(sends, recv_like, peers_send, peers_recv, tags_send, tags_recv, group_name,
                   operation=None):
    # An empty name selects the default process group (also the loopback fixture).
    group = c10d._resolve_process_group(group_name) if group_name else None
    reference, count = _receive_layout(sends, recv_like)
    flat = reference.new_empty(count)
    received = _split_receives(flat, recv_like)
    packed_sends = [value.contiguous() for value in sends]
    ops = [dist.P2POp(dist.irecv, value, peer, group=group, tag=tag)
           for value, peer, tag in zip(received, peers_recv, tags_recv)]
    ops += [dist.P2POp(dist.isend, value, peer, group=group, tag=tag)
            for value, peer, tag in zip(packed_sends, peers_send, tags_send)]
    for request in dist.batch_isend_irecv(ops):
        wait_transport(request, operation)
    _advance(group_name, "halo")
    return flat


def _reduce(value, group_name):
    # The empty group name is the default-group/loopback spelling.
    group = c10d._resolve_process_group(group_name) if group_name else None
    group = active_data_group(group)
    group_name = _group_name(group)
    with transport_operation(group_name) as operation:
        out = value.clone()
        wait_transport(dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group, async_op=True),
                       operation)
        _advance(group_name, "sum")
    return out


def _refuse_forward_mode(operator: str) -> None:
    """Refuse a live forward mode before a communication operator issues anything.

    These operators register a backward only (``register_autograd``), and a
    custom operator without a forward rule returns its output without the
    tangent and without an error.  The reading is program state
    (``forward_mode_active``): ranks that all open the same dual level refuse
    at the same operator.
    """
    if forward_mode_active():
        raise NotImplementedError(
            f"{operator}: communication operators provide first-order reverse "
            f"derivatives only; a forward-mode tangent would be dropped; take "
            f"forward-mode derivatives on a single rank")


@torch.library.custom_op(
    "locomgcm::halo_transfer", mutates_args=(),
    tags=HALO_P2P_TAGS,
    schema="(Tensor token, Tensor[] sends, Tensor[] recv_like, int[] peers_send, "
           "int[] peers_recv, int[] tags_send, int[] tags_recv, str group_name, "
           "int round_id) -> (Tensor, Tensor)",
)
def _halo_transfer(
    token: torch.Tensor, sends: list[torch.Tensor], recv_like: list[torch.Tensor],
    peers_send: list[int], peers_recv: list[int], tags_send: list[int],
    tags_recv: list[int], group_name: str, round_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (_transfer(sends, recv_like, peers_send, peers_recv, tags_send, tags_recv, group_name),
            token + 1)


@_halo_transfer.register_fake
def _halo_transfer_fake(token, sends, recv_like, peers_send, peers_recv,
                        tags_send, tags_recv, group_name, round_id):
    reference, count = _receive_layout(sends, recv_like)
    return reference.new_empty(count), torch.empty_like(token)


def halo_transfer(token: torch.Tensor, sends: list[torch.Tensor],
                  recv_like: list[torch.Tensor], plan: RoundPlan
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    _refuse_forward_mode("halo_transfer")
    return _halo_transfer(token, sends, recv_like, list(plan.peers_send),
                          list(plan.peers_recv), list(plan.tags_send),
                          list(plan.tags_recv), plan.group_name, plan.round_id)


def _no_gradient(values) -> list | None:
    """The gradient slot of an ``int[]`` argument: an empty list stays a list node."""
    return [] if len(values) == 0 else None


def _plan_slots(plan: RoundPlan) -> tuple:
    return (_no_gradient(plan.peers_send), _no_gradient(plan.peers_recv),
            _no_gradient(plan.tags_send), _no_gradient(plan.tags_recv), None, None)


def _halo_setup(ctx, inputs, output):
    token, sends, recv_like, ps, pr, ts, tr, group, round_id = inputs
    ctx.plan = RoundPlan(tuple(ps), tuple(pr), tuple(ts), tuple(tr), group, round_id)
    ctx.nsend = len(sends)
    ctx.save_for_backward(token, *sends, *recv_like)


def _halo_backward(ctx, gradient, _token_gradient):
    token, *templates = ctx.saved_tensors
    sends, receives = templates[:ctx.nsend], templates[ctx.nsend:]
    flat, _ = halo_transfer(token, _split_receives(gradient.contiguous(), receives),
                            sends, ctx.plan.dual())
    return (None, _split_receives(flat, sends), [None] * len(receives), *_plan_slots(ctx.plan))


_halo_transfer.register_autograd(_halo_backward, setup_context=_halo_setup)


@torch.library.custom_op(
    "locomgcm::ordered_sum", mutates_args=(),
    tags=HALO_P2P_TAGS,
    schema="(Tensor token, Tensor value, str group_name) -> (Tensor, Tensor)",
)
def _ordered_sum(token: torch.Tensor, value: torch.Tensor, group_name: str
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    return _reduce(value, group_name), token + 1


def ordered_sum(token: torch.Tensor, value: torch.Tensor, group_name: str
                ) -> tuple[torch.Tensor, torch.Tensor]:
    _refuse_forward_mode("ordered_sum")
    return _ordered_sum(token, value, group_name)


@_ordered_sum.register_fake
def _sum_fake(token, value, group_name):
    return torch.empty_like(value), torch.empty_like(token)


def _sum_setup(ctx, inputs, output):
    token, _value, ctx.group_name = inputs
    ctx.save_for_backward(token)


def _sum_backward(ctx, gradient, _token_gradient):
    value, _ = ordered_sum(ctx.saved_tensors[0], gradient, ctx.group_name)
    return None, value, None


_ordered_sum.register_autograd(_sum_backward, setup_context=_sum_setup)


@torch.library.custom_op(
    "locomgcm::halo_round", mutates_args=(),
    tags=HALO_P2P_TAGS,
    schema="(Tensor link, Tensor[] sends, Tensor[] recv_like, int[] peers_send, "
           "int[] peers_recv, int[] tags_send, int[] tags_recv, str group_name, "
           "int round_id, Tensor? owner=None) -> (Tensor, Tensor)",
)
def _halo_round(
    link: torch.Tensor, sends: list[torch.Tensor], recv_like: list[torch.Tensor],
    peers_send: list[int], peers_recv: list[int], tags_send: list[int],
    tags_recv: list[int], group_name: str, round_id: int,
    owner: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A session round: the same transport, chained through a differentiable link."""
    session = _round_owner(owner)
    flat = _transfer(sends, recv_like, peers_send, peers_recv, tags_send, tags_recv, group_name)
    if session is not None:
        round_id = len(session._plans)
    plan = RoundPlan(tuple(peers_send), tuple(peers_recv), tuple(tags_send), tuple(tags_recv),
                     group_name, round_id)
    _record_round(session,
        _Round(plan, "halo", tuple(_Template.of(value) for value in sends),
               tuple(_Template.of(value) for value in recv_like)), _Template.of(flat))
    return flat, link + 1


@_halo_round.register_fake
def _halo_round_fake(link, sends, recv_like, peers_send, peers_recv,
                     tags_send, tags_recv, group_name, round_id, owner=None):
    reference, count = _receive_layout(sends, recv_like)
    return reference.new_empty(count), torch.empty_like(link)


def _halo_round_setup(ctx, inputs, output):
    GradientBoundary.configure(ctx)
    link, sends, recv_like, ps, pr, ts, tr, group, round_id, _owner = inputs
    ctx.plan = RoundPlan(tuple(ps), tuple(pr), tuple(ts), tuple(tr), group, round_id)
    # Layouts only: the dual round needs the shapes, dtypes and devices.
    ctx.link = _Template.of(link)
    ctx.sends = tuple(_Template.of(value) for value in sends)
    ctx.receives = tuple(_Template.of(value) for value in recv_like)


def _halo_round_backward(ctx, gradient, link_gradient):
    if _objective_noop(gradient, link_gradient):
        return (None, [None] * len(ctx.sends), [None] * len(ctx.receives),
                *_plan_slots(ctx.plan),
                *((None,) if len(ctx.needs_input_grad) == 10 else ()))
    # The link cotangent is the reverse token: the dual round consumes and advances it.
    sends = [value.empty() for value in ctx.sends]
    receives = [value.empty() for value in ctx.receives]
    if link_gradient is None:
        link_gradient = ctx.link.zeros()
    if gradient is None:
        reference, count = _receive_layout(sends, receives)
        gradient = reference.new_zeros(count)
    flat, token = halo_transfer(link_gradient.to(torch.int64),
                                _split_receives(gradient.contiguous(), receives),
                                sends, ctx.plan.dual())
    cotangents = [value if source.is_floating_point() or source.is_complex() else None
                  for value, source in zip(_split_receives(flat, sends), sends)]
    return (token.to(ctx.link.dtype), cotangents, [None] * len(receives), *_plan_slots(ctx.plan),
            *((None,) if len(ctx.needs_input_grad) == 10 else ()))


_halo_round.register_autograd(_halo_round_backward, setup_context=_halo_round_setup)


def halo_round(link: torch.Tensor, sends: list[torch.Tensor],
               recv_like: list[torch.Tensor], plan: RoundPlan
               ) -> tuple[torch.Tensor, torch.Tensor]:
    _refuse_forward_mode("halo_round")
    return _halo_round(link, sends, recv_like, list(plan.peers_send),
                       list(plan.peers_recv), list(plan.tags_send),
                       list(plan.tags_recv), plan.group_name, plan.round_id, plan.owner)


@torch.library.custom_op(
    "locomgcm::sum_round", mutates_args=(),
    tags=HALO_P2P_TAGS,
    schema="(Tensor link, Tensor value, str group_name, Tensor? owner=None) -> (Tensor, Tensor)",
)
def _sum_round(link: torch.Tensor, value: torch.Tensor, group_name: str,
               owner: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    session = _round_owner(owner)
    result = _reduce(value, group_name)
    template = (_Template.of(value),)
    plan = RoundPlan((), (), (), (), group_name, 0 if session is None else len(session._plans))
    _record_round(session, _Round(plan, "sum", template, template), template[0])
    return result, link + 1


@_sum_round.register_fake
def _sum_round_fake(link, value, group_name, owner=None):
    return torch.empty_like(value), torch.empty_like(link)


def _sum_round_setup(ctx, inputs, output):
    GradientBoundary.configure(ctx)
    link, value, ctx.group_name, _owner = inputs
    ctx.link, ctx.value = _Template.of(link), _Template.of(value)


def _sum_round_backward(ctx, gradient, link_gradient):
    if _objective_noop(gradient, link_gradient):
        return (None, None, None, *((None,) if len(ctx.needs_input_grad) == 4 else ()))
    if link_gradient is None:
        link_gradient = ctx.link.zeros()
    if gradient is None:
        gradient = ctx.value.zeros()
    cotangent, token = ordered_sum(link_gradient.to(torch.int64), gradient, ctx.group_name)
    return (token.to(ctx.link.dtype), cotangent, None,
            *((None,) if len(ctx.needs_input_grad) == 4 else ()))


_sum_round.register_autograd(_sum_round_backward, setup_context=_sum_round_setup)


def sum_round(link: torch.Tensor, value: torch.Tensor, group_name: str, owner=None
              ) -> tuple[torch.Tensor, torch.Tensor]:
    _refuse_forward_mode("sum_round")
    return _sum_round(link, value, group_name, owner)


class CommSessionError(RuntimeError):
    """An invalid session transition or an all-participant abort."""


@dataclass(frozen=True)
class _Template:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def of(cls, value: torch.Tensor) -> _Template:
        return cls(tuple(value.shape), value.dtype, value.device)

    def zeros(self) -> torch.Tensor:
        return torch.zeros(self.shape, dtype=self.dtype, device=self.device)

    def empty(self) -> torch.Tensor:
        return torch.empty(self.shape, dtype=self.dtype, device=self.device)


@dataclass(frozen=True)
class _Round:
    plan: RoundPlan
    kind: str
    sends: tuple[_Template, ...]
    receives: tuple[_Template, ...]


@dataclass
class _SolveRound:
    """One collective linear solve, including its untaped transpose solve.

    The solver owns its internal halo/reduction sequence on an unbound grid.
    The session orders the complete solve against surrounding model exchanges.
    Keeping that boundary atomic avoids taping the Krylov iterations or adding
    their internal communication to the model's replay plan.
    """

    plan: RoundPlan
    spec: Any
    rhs: _Template
    coefficient_count: int
    solution: torch.Tensor | None = None

    def primal(self, rhs):
        solve = self.spec.primal_solve or self.spec.solve
        with transport_operation(self.plan.group_name, capture_owner=True):
            value = solve(rhs)
            self.solution = value.detach()
            _advance(self.plan.group_name, "solve")
        return value

    def reverse(self, cotangent, needs):
        with transport_operation(self.plan.group_name, capture_owner=True):
            value = self.spec.solve(cotangent.contiguous())
            # All ranks enter coefficient_grads, even those with no local active
            # coefficients: its topology-aware halo exchange is collective work.
            gradients = tuple(self.spec.coefficient_grads(value, self.solution, needs))
            _advance(self.plan.group_name, "solve_transpose")
        return value, gradients


class _SolveEdge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, record, link, rhs, *coefficients):
        GradientBoundary.configure(ctx)
        ctx.record = record
        ctx.save_for_backward(*coefficients)
        return record.primal(rhs), link + 1

    @staticmethod
    def backward(ctx, gradient, link_gradient):
        if _objective_noop(gradient, link_gradient):
            return (None,) * len(ctx.needs_input_grad)
        if torch.is_grad_enabled():
            raise RuntimeError("collective linear solves provide first-order derivatives only")
        # Reading saved tensors checks that coefficients were not overwritten.
        ctx.saved_tensors
        record = ctx.record
        if gradient is None:
            gradient = record.rhs.zeros()
        value, gradients = record.reverse(gradient, ctx.needs_input_grad[3:])
        if link_gradient is None:
            link_gradient = value.new_zeros(1, dtype=torch.float64)
        return (None, link_gradient + 1,
                value if ctx.needs_input_grad[2] else None, *gradients)


@dataclass
class _Checkpoint:
    function: Callable | None
    inputs: tuple[torch.Tensor, ...]
    rounds: tuple[_Round | _SolveRound, ...]
    token: torch.Tensor
    first_round: int
    versions: tuple[int, ...]
    full_outputs: bool = False

    @property
    def plans(self) -> tuple[RoundPlan, ...]:
        return tuple(record.plan for record in self.rounds)


def _segment_schedule(record: _Checkpoint) -> list[tuple[str, _Round | _SolveRound]]:
    return ([("forward", value) for value in record.rounds]
            + [("dual", value) for value in reversed(record.rounds)])


def _plan_difference(recorded, replayed) -> str:
    for index, (before, after) in enumerate(zip(recorded, replayed)):
        if before != after:
            return f"round {index}: recorded {before}, replayed {after}"
    return f"recorded {len(recorded)} rounds, replayed {len(replayed)}"


def _differentiable(values):
    return [value for value in values if value.is_floating_point() or value.is_complex()]


class _SegmentEdge(torch.autograd.Function):
    """A checkpointed segment: replayed and differentiated inside one node."""

    @staticmethod
    def forward(ctx, session_ref, index, link, count, *tensors):
        GradientBoundary.configure(ctx)
        ctx.link_like = (link.shape, link.dtype, link.device)
        ctx.session_ref, ctx.index = session_ref, index
        return (*tensors[count:], link + 1)

    @staticmethod
    def backward(ctx, *gradients):
        if _objective_noop(*gradients):
            return (None,) * len(ctx.needs_input_grad)
        *cotangents, link_gradient = gradients
        if link_gradient is None:
            shape, dtype, device = ctx.link_like
            link_gradient = torch.zeros(shape, dtype=dtype, device=device)
        session = ctx.session_ref()
        if session is None:
            raise CommSessionError("checkpoint owner was released before backward")
        record = session._tape[ctx.index]
        try:
            link, inputs = session._replay(record, cotangents, link_gradient)
        finally:
            # The completed segment retains no host snapshots or callable.
            record.function, record.inputs, record.versions = None, (), ()
        return (None, None, link, None, *inputs, *[None] * len(cotangents))



class CommSession:
    """One differentiation window, with explicit collective participation.

    All ranks call ``backward(loss)``; a constant/zero local loss still drains
    every planned round. Controls receive accumulated cotangents in ``.grad``.
    Every round is chained through one float64 link, so the local backward
    traversal executes each dual round once, in reverse forward order.
    Construct a new session for the next forward window.

    From construction until its ``backward`` ends, the session owns the
    numerical-failure window (``locomgcm.deferred_checks``): every numerical
    failure on this rank is recorded here instead of raised, the planned
    communication continues, and the agreement at the reverse admission (for
    the forward) or after the reverse rejects the window on every rank together.
    """

    def __init__(self, group: dist.ProcessGroup | None):
        self.group = dist.group.WORLD if group is None and dist.is_initialized() else group
        self.group_name = _group_name(self.group)
        self.control_group = _control_group(self.group)
        self.control_group_name = _group_name(self.control_group)
        self.origin_rank = 0 if self.group is None else dist.get_rank()
        self.token = torch.zeros(1, dtype=torch.int64)
        self._plans: list[RoundPlan] = []
        self._tape: list[_Round | _SolveRound | _Checkpoint] = []
        owner_id = next(_OWNER_IDS)
        self._owner = torch.tensor(owner_id, dtype=torch.int64)
        _OWNERS[owner_id] = self
        self._phase = "forward"
        # The reverse link follows the data device; verdicts have a separate CPU domain.
        device = torch.device("cpu")
        if self.group is not None and dist.get_backend(self.group) == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
        self._status = torch.zeros((), dtype=torch.int32)
        participants = 1 if self.group is None else dist.get_world_size(self.group)
        width = window_agreement().width
        with torch.inference_mode(False):
            self._agreement_row = torch.empty(width, dtype=torch.int64)
            self._agreement_payload = torch.empty((participants, width), dtype=torch.int64)
        # The entry link receives the terminal reverse token as its gradient.
        self._entry = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
        self._link = self._entry
        self._reverse_token = torch.zeros_like(self.token)
        self._templates: list[tuple[tuple[int, ...], torch.dtype, torch.device]] = []
        # Window position of the first round in ``rounds``: nonzero while a
        # checkpoint segment records or replays into lists of its own.
        self._round_base = 0
        self._recording = False
        self._raw_eager_active = False
        self.epoch = 0
        self._local_error: Exception | None = None
        self._numerical_error: Exception | None = None
        try:
            self._transport = TransportLifetime(self.group, self.group_name,
                                                control_group=self.control_group,
                                                control_name=self.control_group_name)
        except TransportFailure as failure:
            error = CommSessionError(str(failure))
            error.origin_reports, error.remote_failure = failure.origin_reports, True
            raise error from failure
        from locomgcm.checkpoint.window import WindowOwner
        self.window = WindowOwner(session=self)

    @property
    def transport_broken(self):
        return self._transport.broken.is_set()

    def transport_error(self):
        transport = self._transport.error()
        error = CommSessionError(str(transport))
        error.origin_reports = transport.origin_reports
        error.remote_failure = True
        for note in getattr(transport, "__notes__", ()):
            error.add_note(note)
        return error

    def fail_transport(self, error):
        """Notify all members and abort every wait after a local execution exception."""
        self.record_failure(error)
        self._transport.fail(error)

    def record_numerical_failure(self, error: Exception) -> None:
        """Merge the failure into the window's numerical error; the schedule continues.

        Called by ``locomgcm.deferred_checks`` for every failure observed
        while this session owns the window, including on an autograd replay
        thread.  Raising there would abandon a communication schedule the peers
        still run, and the drain that follows can wait behind a collective
        already issued on the device stream.
        """
        if self._phase not in ("forward", "replay", "reverse"):
            raise CommSessionError("numerical observations require an active communication window") from error
        if getattr(error, "origin_rank", None) is None:
            error.origin_rank = self.origin_rank
        self._numerical_error = window_numerical_error(self._numerical_error, error) or error

    def record_failure(self, error):
        """Seal the cause without turning a peer's report into a local origin."""
        if (getattr(error, "origin_rank", None) is None
                and not getattr(error, "remote_failure", False)):
            error.origin_rank = self.origin_rank
        if self._local_error is None:
            self._local_error = error
        if self.window.cause is None:
            self.window.cause = error

    @_window_operation
    def prepare(self, *receive_templates: torch.Tensor) -> None:
        """Declare the planned receives; each executed round is checked against them.

        Templates follow forward order: flat P2P receives and shaped SUM
        results. Windows without a declaration infer these shapes while
        executing each round.
        """
        if self._plans or self._phase != "forward":
            raise CommSessionError("prepare precedes the forward window")
        self._templates = [(tuple(value.shape), value.dtype, value.device)
                           for value in receive_templates]

    @property
    def rounds(self) -> list[RoundPlan]:
        """Every executed round in order; while a segment records or replays, its own."""
        return self._plans

    def _require_window(self) -> None:
        # Reusing a completed/aborted window would mix two reverse plans.
        if self._phase not in ("forward", "replay"):
            raise CommSessionError("a new forward window requires a new CommSession")

    def _check_template(self, template: _Template) -> None:
        if not self._templates:
            return
        index = self._round_base + len(self._plans)
        if index < len(self._templates) and self._templates[index] != (
                template.shape, template.dtype, template.device):
            raise CommSessionError("receive template differs from the prepared communication plan")

    @_window_operation
    def forward_round(self, sends: list[torch.Tensor], recv_like: list[torch.Tensor],
                      plan: RoundPlan) -> tuple[torch.Tensor, torch.Tensor]:
        if self._raw_boundary() and not self._raw_eager_active:
            with graph_exit():
                return _raw_eager_operation(self, self.forward_round, sends, recv_like, plan)
        # The operator validates and records its owner at execution time.
        with torch.enable_grad():
            flat, self._link = halo_round(self._link, list(sends), list(recv_like),
                                         replace(plan, owner=self._owner, group_name=self.group_name))
        self.token = self.token + 1
        if self._raw_boundary():
            flat, self._link = self.window.guard_raw(flat, self._link)
        return flat, self.token

    def _raw_boundary(self):
        return not self.window._configured and not self._recording and self._phase != "replay"

    def observe(self, outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Expose computed outputs to a raw objective as one numerical boundary.

        The objective pass collects the combined seeds here. The communication
        sweep then differentiates the local and communicated parts together.
        Direct leaf dependencies stay with the objective; checkpoint windows
        already own their output boundary.
        """
        if not self._raw_boundary() or not torch.is_grad_enabled():
            return outputs
        active = [i for i, value in enumerate(outputs) if value.grad_fn is not None]
        if not active:
            return outputs
        self._require_window()
        guarded = self.window.guard_raw(*(outputs[i] for i in active))
        result = list(outputs)
        for i, value in zip(active, guarded, strict=True):
            result[i] = value
        return tuple(result)

    @_window_operation
    def sum(self, value: torch.Tensor) -> torch.Tensor:
        if self._raw_boundary() and not self._raw_eager_active:
            with graph_exit():
                return _raw_eager_operation(self, self.sum, value)
        with torch.enable_grad():
            result, self._link = sum_round(self._link, value, self.group_name, self._owner)
        self.token = self.token + 1
        if self._raw_boundary():
            result, self._link = self.window.guard_raw(result, self._link)
        return result

    @_window_operation
    def linear_solve(self, spec, rhs: torch.Tensor, *coefficients: torch.Tensor):
        """Attach a collective ROOT solve to this window's ordered reverse chain.

        ``spec`` uses the common PCG adjoint contract. Both numerical solves
        must use unbound transport; coefficient_grads must execute the same
        communication on every participant regardless of local gradient needs.
        Called from the solver's eager boundary, outside a compiled step.
        """
        if self._raw_boundary() and not self._raw_eager_active:
            with graph_exit():
                return _raw_eager_operation(self, self.linear_solve, spec, rhs, *coefficients)
        if self._phase not in ("forward", "replay"):
            raise CommSessionError("a new solve requires a forward or replay window")
        # The solve's internal collectives are unbound and run on
        # ``active_data_group()``.  Its failure domain must be this session's
        # group, including allocation between internal rounds.
        if active_data_group() is not self.group:
            raise CommSessionError("a collective solve must run in its session's data group")
        plan = RoundPlan((), (), (), (), self.group_name, len(self._plans))
        record = _SolveRound(plan, spec, _Template.of(rhs), len(coefficients))
        with torch.enable_grad():
            value, self._link = _SolveEdge.apply(record, self._link, rhs, *coefficients)
        self._check_template(_Template.of(value))
        self.token = self.token + 1
        self._plans.append(plan)
        # A segment's recording keeps the round's plan only.  Its replay makes
        # new solve records for the segment's VJP; this one's solution and
        # recipe would otherwise stay until the window ends.  The recording's
        # own graph holds the full record and ends with the segment.
        self._tape.append(replace(record, spec=None, solution=None)
                          if self._recording else record)
        if self._raw_boundary():
            value, self._link = self.window.guard_raw(value, self._link)
        return value

    @_window_operation
    def record_checkpoint(self, function, inputs, *, full_outputs=True):
        """Recompute a pure tensor function and its communication during reverse.

        Inputs contain controls. A model executor consumes its own head handle;
        communication records hold no stored history. Each replay receives a
        fresh execution epoch and must reproduce its communication plan.
        """
        if self._phase != "forward" or self._recording:
            raise CommSessionError("checkpoint requires a forward window")
        tape, plans, token, link = self._tape, self._plans, self.token, self._link
        first_round = len(plans)
        # The segment records into lists of its own, its rounds numbered from
        # zero, as its replay does, and in the replay's posture: every round
        # builds its link edge (``forward_round``) on a link that requires grad,
        # so a compiled step meets the same inputs in both passes and traces
        # once.  The recorded edges end with the segment; the segment edge below
        # carries the chain.
        self._tape, self._plans, self._round_base = [], [], first_round
        self._link, self._recording = link.detach().requires_grad_(), True
        try:
            with torch.no_grad():
                values = function(*inputs)
        except Exception as error:
            # No first or later segment can infer a complete remaining plan.
            self.fail_transport(error)
            raise self.transport_error() from error
        finally:
            segment_rounds, segment_plans = self._tape, self._plans
            self._tape, self._plans, self._round_base = tape, plans, 0
            self._link, self._recording = link, False
        flat, spec = tree_flatten(values)
        if any(not isinstance(value, torch.Tensor) for value in flat):
            raise TypeError("checkpoint outputs must be tensors")
        retained = tuple(inputs)
        record = _Checkpoint(function, retained, tuple(segment_rounds), token,
                             first_round, tuple(value._version for value in retained),
                             full_outputs)
        self._tape.append(record)
        self._plans.extend(segment_plans)
        return values, record

    @_window_operation
    def checkpoint(self, function, *inputs: torch.Tensor):
        """Record a segment and attach an edge consuming its owned output."""
        if self._raw_boundary() and not self._raw_eager_active:
            with graph_exit():
                return _raw_eager_operation(self, self.checkpoint, function, *inputs)
        link = self._link
        values, record = self.record_checkpoint(function, inputs, full_outputs=False)
        flat, spec = tree_flatten(values)
        # Recorded values enter detached: the edge's only graph inputs are the
        # link and the explicit controls, so no recorded expression is revisited.
        with torch.enable_grad():
            *outputs, self._link = _SegmentEdge.apply(weakref.ref(self), len(self._tape) - 1, link, len(inputs), *inputs,
                                                      *[value.detach() for value in _differentiable(flat)])
        if self._raw_boundary():
            *outputs, self._link = self.window.guard_raw(*outputs, self._link)
        # The window's plan list keeps every executed round (counted by callers);
        # a segment's plans carry segment-relative numbers.
        replaced = iter(outputs)
        merged = [next(replaced) if value.is_floating_point() or value.is_complex() else value
                  for value in flat]
        return tree_unflatten(merged, spec)

    def terminal_token(self) -> torch.Tensor:
        return self.token

    def _agree_reverse(self, failed: bool) -> bool:
        if self.transport_broken:
            raise self.transport_error()
        self._status.fill_(int(failed))
        # An uninitialized transport is used only by the one-process fixture.
        if self.control_group is not None:
            with transport_operation(self.control_group_name) as operation:
                wait_transport(dist.all_reduce(self._status, op=dist.ReduceOp.MAX,
                                               group=self.control_group, async_op=True), operation)
        return bool(self._status.item())

    def _own_error(self, error):
        if error is None or getattr(error, "remote_failure", False):
            return None
        origin = getattr(error, "origin_rank", None)
        return error if origin is None or origin == self.origin_rank else None

    def _qualify(self, cleanup=None, *, verdict="admission") -> tuple[bool, str, Exception | None]:
        """The window's agreement: this rank's causes plus the numerical channel.

        It runs at admission, after the whole objective and after the reverse,
        and each run seals the channel's observations. One control-group MAX carries, per origin, a failure flag and the channel's
        status row (each row has one writer, so origins never combine); its one
        read replaces the status read.  When any origin failed, every rank
        gathers every origin's own diagnostics, so each raises the same text.
        The local cause is kept as the cause; the agreed diagnostics are added.
        """
        if self.transport_broken:
            raise self.transport_error()
        if cleanup is not None:
            try:
                self._transport.quiesce()
            except Exception as error:
                self.fail_transport(error)
                raise self.transport_error() from error
        agreement = window_agreement()
        participants = 1 if self.group is None else dist.get_world_size(self.group)
        rank = 0 if self.group is None else dist.get_rank(self.group)
        own_local = self._own_error(self._local_error)
        own_numerical = self._own_error(self._numerical_error)
        # Bit 0: this rank's execution cause; bit 1: its numerical observation.
        flag = (int(own_local is not None and not isinstance(own_local, DeferredCheckError))
                | int(own_numerical is not None or isinstance(own_local, DeferredCheckError)) << 1)
        with transport_operation(self.control_group_name) as operation:
            # The scope includes device-code collection and any blocking host read.
            row = agreement.row(flag, self._status.device, out=self._agreement_row)
            payload = self._agreement_payload.zero_()
            payload[rank].copy_(row)
            agreement.seal()
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as error:
                    self.fail_transport(error)
                    raise self.transport_error() from error
            if self.control_group is not None:
                wait_transport(dist.all_reduce(payload, op=dist.ReduceOp.MAX,
                                               group=self.control_group, async_op=True), operation)
            rows = payload.tolist()
        _log_verdict(self.control_group_name, "completion" if cleanup is not None else verdict)
        failed, coded, codes = agreement.verdict(rows, rank)
        if codes is not None:
            codes.origin_rank = self.origin_rank
        self.window.code_error = codes
        if not failed:
            return False, "", None
        # One numerical error per window and rank, each solver entry once.
        local_numerical = window_numerical_error(own_numerical, own_local, codes)
        if local_numerical is not None:
            local_numerical.origin_rank = self.origin_rank
            self._numerical_error = self.window.numerical_error = local_numerical
        local = None if isinstance(own_local, DeferredCheckError) else own_local
        own = list(dict.fromkeys(
            f"{type(error).__name__}: {error}"
            for error in (local, local_numerical)
            if error is not None))
        reports = [own]
        if self.control_group is not None:
            reports = [None] * participants
            with transport_operation(self.control_group_name):
                dist.all_gather_object(reports, own, group=self.control_group)
        origins = [origin if self.group is None else dist.get_global_rank(self.group, origin)
                   for origin in range(participants)]
        message = "\n".join(f"rank {origin}: {text}"
                            for origin, texts in zip(origins, reports) for text in texts)
        numerical = local_numerical
        if numerical is None and (coded or any(row[0] & 2 for row in rows)):
            # Preserve the source reports without publishing them as this rank's own.
            numerical = DeferredCheckError(message)
            numerical.origin_reports = tuple((origin, tuple(texts))
                                             for origin, texts in zip(origins, reports) if texts)
            numerical.remote_failure = True
            if len(numerical.origin_reports) == 1:
                numerical.origin_rank = numerical.origin_reports[0][0]
        if numerical is not None:
            if local is not None and local is not numerical and numerical.__cause__ is None:
                numerical.__cause__ = local
            return True, message, numerical
        return True, message, own_local or CommSessionError(message)

    def abort_all(self, reason: str) -> None:
        self.window.abort_all(reason)

    def backward(self, loss: torch.Tensor) -> torch.Tensor:
        # A second backward would add new local uses after the communication sweep.
        if self._phase != "forward":
            raise CommSessionError("one backward is supported per CommSession")
        return self.window.backward(loss)

    def _prepare_reverse(self):
        for record in self._tape:
            if isinstance(record, _Checkpoint) and tuple(value._version for value in record.inputs) != record.versions:
                error = CommSessionError("checkpoint controls changed before replay")
                self.fail_transport(error)
                raise self.transport_error() from error
        # The reverse admission is the window's agreement on the forward: a local
        # cause, a recorded numerical failure or a code in the channel on any
        # rank stops the window here on every rank, so no rank replays a
        # failed forward and the failure never depends on that replay.
        self.window.admit()

    def _reverse(self, loss: torch.Tensor, *, reverse=None, before_terminate=None) -> torch.Tensor:
        self._prepare_reverse()
        self._phase = "reverse"
        start = _progress(self.group_name)
        try:
            if reverse is None:
                # Constant objectives still seed every round through the link.
                roots, seeds = [self._link], [torch.zeros_like(self._link)]
                if loss.requires_grad:
                    roots.insert(0, loss)
                    seeds.insert(0, torch.ones_like(loss))
                torch.autograd.backward(roots, seeds)
            else:
                reverse()
        except Exception as error:
            self.fail_transport(error)
            raise self.transport_error() from error
        expected = sum(2 * len(r.rounds) if isinstance(r, _Checkpoint) else 1
                       for r in self._tape)
        executed = _progress(self.group_name) - start
        if executed != expected and self._local_error is None:
            self._local_error = CommSessionError(
                f"reverse executed {executed} transport rounds; planned {expected}")
        if before_terminate is not None:
            try:
                before_terminate()
            except DeferredCheckError as error:
                self.record_numerical_failure(error)
            except Exception as error:
                self.fail_transport(error)
                raise self.transport_error() from error
        self.window.terminate()
        count = self._entry.grad
        self._reverse_token = (torch.zeros_like(self._entry) if count is None else count).to(torch.int64)
        self._phase = "complete"
        return self._reverse_token

    def reverse_rounds(self) -> torch.Tensor:
        # Retrieving a completed terminal token issues zero additional rounds.
        if self._phase == "complete":
            return self._reverse_token
        if self._phase == "forward":
            raise CommSessionError("reverse_rounds follows backward on every participant")
        # An aborted or interrupted sweep belongs to a failed differentiation window.
        raise CommSessionError("the differentiation window has already ended")

    def _schedule(self) -> list[tuple[str, _Round | _SolveRound]]:
        entries: list[tuple[str, _Round | _SolveRound]] = []
        for record in reversed(self._tape):
            if isinstance(record, _Checkpoint):
                entries.extend(_segment_schedule(record))
            else:
                entries.append(("dual", record))
        return entries

    def _replay(self, record: _Checkpoint, cotangents, link_gradient):
        tape, plans, token, link = self._tape, self._plans, self.token, self._link
        entry = torch.zeros_like(link_gradient).requires_grad_()
        # Empty lists, rounds numbered from zero: the recording's posture (checkpoint).
        self._tape, self._plans, self.token, self._link = [], [], record.token, entry
        self._round_base = record.first_round
        self.epoch += 1
        self._phase = "replay"
        try:
            detached = [value.detach().requires_grad_(value.requires_grad)
                        for value in record.inputs]
            with torch.enable_grad():
                values, _ = tree_flatten(record.function(*detached))
                replayed = tuple(self._plans)
                if replayed != record.plans:
                    raise CommSessionError("checkpoint replay changed the communication plan: "
                                           + _plan_difference(record.plans, replayed))
                terminal = self._link
            GradientBoundary.backward(values if record.full_outputs else _differentiable(values),
                                      cotangents, links=((terminal, link_gradient),))
        except Exception as error:
            # A local exception abandons this traversal: a partial tape cannot repair it.
            self.fail_transport(error)
            raise self.transport_error() from error
        finally:
            self._tape, self._plans, self.token, self._link = tape, plans, token, link
            self._round_base = 0
            self._phase = "reverse"
        link_out = torch.zeros_like(link_gradient) if entry.grad is None else entry.grad
        return link_out, [value.grad for value in detached]

    def commit(self, optimizer):
        """Commit one optimizer update after collective derivative qualification."""
        if self.transport_broken:
            raise self.transport_error()
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        invalid = self._phase != "complete" or any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters)
        if self._agree_reverse(invalid):
            self._phase = "aborted"
            raise CommSessionError("optimizer commit requires finite gradients on every rank")
        error = None
        try:
            before = [p.detach().clone() for p in parameters]
            state = deepcopy(optimizer.state_dict())
        except Exception as failure:
            error = failure
        if self._agree_reverse(error is not None):
            self._phase = "aborted"
            raise CommSessionError("optimizer snapshot failed before commit") from error
        try:
            try:
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in parameters):
                    raise CommSessionError("optimizer produced nonfinite parameters")
            except Exception as failure:
                error = failure
            if self._agree_reverse(error is not None):
                raise CommSessionError("participant optimizer update failed") from error
        except Exception as failure:
            with torch.no_grad():
                for parameter, saved in zip(parameters, before):
                    parameter.copy_(saved)
            optimizer.load_state_dict(state)
            self._phase = "aborted"
            raise CommSessionError("optimizer commit failed; local parameters and state restored") from failure
        self._phase = "committed"


def transfer(sends: list[torch.Tensor], recv_like: list[torch.Tensor],
             peers_send: list[int], peers_recv: list[int], tags_send: list[int],
             tags_recv: list[int], group: dist.ProcessGroup | None = None, *,
             session: CommSession | None = None, constant: bool = False) -> list[torch.Tensor]:
    # None explicitly selects the signed, existing inference implementation.
    if session is None:
        return halo_p2p(sends, recv_like, peers_send, peers_recv, tags_send, tags_recv, group=group)
    # The opaque operator assigns the schedule position at execution time.
    plan = RoundPlan(tuple(peers_send), tuple(peers_recv), tuple(tags_send), tuple(tags_recv),
                     session.group_name, 0)
    flat, _ = session.forward_round(sends, recv_like, plan)
    if constant:
        # A declared constant (``exchange(constant=True)``, rank-uniform) keeps its
        # round on the session's link, so it stays in the one issue order both
        # ranks follow; only its received data leaves the graph.  Its dual round
        # still runs, with a zero cotangent.  Sent through the inference transport
        # instead, the constant rode ``halo_p2p``'s own order token, which has no
        # dependency on the link: the compiler interleaved the two chains
        # differently on the two ranks' different graphs and the ranks deadlocked
        # on a coalesced P2P batch (NCCL SeqNum 79).
        flat = flat.detach()
    return _split_receives(flat, recv_like)


# A plain module flag, deliberately not a ``ContextVar``.  ``session_for_grid``
# sits on every halo exchange the traced step issues; Dynamo folds a module
# bool into the trace under a constant guard, whereas ``ContextVar.get()`` is
# a host read it breaks the graph on (``call_method
# UserDefinedObjectVariable(ContextVar) get``) -- 161.9 -> 220.6 ms/step at
# 960^2 on two ranks,   Should a traced frame ever
# be entered while the flag is raised, the guard misses and the frame
# recompiles with the override honoured: visible in the recompile count, never
# a silent bypass.
_constant_transport_active = False


@contextlib.contextmanager
def constant_transport():
    """Route exchanges of state-independent constants to the inference transport.

    A memoized constant (static metrics, masks) is exchanged once and reused, so
    it must not enter a differentiation window: its round would be recorded
    but never replayed, and it carries no cotangent.  Eager only: it wraps cold
    constant builds and the numerical solves of a ROOT solve (cg2d, cg3d),
    never a traced step.  A bound session records such a solve as one round
    (:meth:`CommSession.linear_solve`) in the recording and in the replay
    alike, so its internal exchanges and reductions must not become rounds;
    unbound, they run on ``active_data_group()``, the session's group, and so
    share its failure domain.
    """
    global _constant_transport_active
    previous = _constant_transport_active
    _constant_transport_active = True
    try:
        yield
    finally:
        _constant_transport_active = previous


def session_for_grid(grid, session=None):
    """The grid binding, or the session a caller threaded through explicitly.

    Traceable: one module-bool read and two ``getattr``/identity reads, all
    folded at trace time.  The binding is not per-step state -- ``runner.run``
    resolves it once and installs it on the grid before the step loop -- and
    the only override is ``constant_transport()`` above.
    """
    if _constant_transport_active:
        return None
    return session if session is not None else getattr(grid, "comm_session", None)


def replicated_loss_weight(world_size: int) -> float:
    return 1 / world_size


__all__ = ["RoundPlan", "halo_transfer", "ordered_sum", "halo_round", "sum_round", "CommSession",
           "constant_transport",
           "CommSessionError", "replicated_loss_weight"]
