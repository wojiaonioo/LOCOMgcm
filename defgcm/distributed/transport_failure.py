"""Out-of-band failure of a communication group with an incomplete data round."""

from collections import deque
from contextlib import contextmanager
from datetime import timedelta
import json
import os
import threading
import time
import weakref

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


_ACTIVE = weakref.WeakValueDictionary()
_GROUP_TIMEOUTS = weakref.WeakKeyDictionary()


class TransportFailure(RuntimeError):
    """The group is unusable; restore the window on a newly created group."""


class _Operation:
    """One transport call, open from ``begin`` to ``finish``, with its Works."""

    __slots__ = ("backend", "deadline", "capture_owner", "started", "works")

    def __init__(self, backend, deadline, capture_owner):
        self.backend, self.deadline, self.capture_owner = backend, deadline, capture_owner
        self.started = time.monotonic()
        self.works = []


class _StreamCompletions:
    """Completion of the NCCL operations finished on one CUDA stream.

    Each operation records its own event, taken from a pool, on the current
    stream after its ``wait()`` calls, and keeps its Works: NCCL runs on the
    tensors' device and ``wait()`` orders that device's current stream, which
    need not be this one, so an operation is finished only when its event is
    done and its Works report completion. Work on one stream completes in
    issue order, so the done events form a prefix of ``pending``: a pass
    queries the newest event, bisects only when it is still running, and
    retires from the front while the Works are complete. An operation
    finished later but started no later (an enclosing solve) takes over the
    tail entries before it with their Works: its event completes after
    theirs and its start is not later, so the oldest unfinished start is
    unchanged. The front of ``pending`` is that oldest start.
    """

    __slots__ = ("pending", "free")

    def __init__(self):
        self.pending = deque()  # (event, started, works), in record order, starts increasing
        self.free = []

    def record(self, stream, started, works):
        event = self.free.pop() if self.free else torch.cuda.Event()
        event.record(stream)
        while self.pending and self.pending[-1][1] >= started:
            covered, _, covered_works = self.pending.pop()
            self.free.append(covered)
            works.extend(covered_works)
        self.pending.append((event, started, works))

    def retire(self):
        pending = self.pending
        if not pending:
            return
        if pending[-1][0].query():
            done = len(pending)
        else:
            done, unfinished = 0, len(pending) - 1
            while done < unfinished:
                middle = (done + unfinished) // 2
                if pending[middle][0].query():
                    done = middle + 1
                else:
                    unfinished = middle
        while done and all(work.is_completed() for work in pending[0][2]):
            self.free.append(pending.popleft()[0])
            done -= 1


class TransportLifetime:
    """A store notification interrupts peers even inside a data operation.

    No collective on the damaged group is used to announce or finish failure.
    Data and control groups are locally aborted on every participant. The
    store poison lasts as long as the data group, across successive windows.

    Progress deadline: an operation of the group that stays in flight longer
    than ``deadline`` seconds (half the group timeout by default, or
    ``DEFGCM_TRANSPORT_DEADLINE_S``) fails the group the same way, so a
    divergent schedule ends in TransportFailure before the backend's own
    timeout ends the process. An operation is in flight while open and, on
    NCCL outside capture, until its completion event on its stream is done
    and its Works are complete (``_StreamCompletions``).
    """

    key = "defgcm/transport-failed"

    def __init__(self, group, name, deadline=None, *, control_group=None,
                 control_name=None):
        self.group, self.name = group, name
        self.groups = {name: group}
        if control_group is not None:
            self.groups[control_name] = control_group
        self.rank = 0 if group is None else dist.get_rank()
        self.origins = ((0,) if group is None else
                        tuple(dist.get_process_group_ranks(group)))
        self.reports = {}
        self.cleanup_errors = []
        self.message = None
        self.broken = threading.Event()
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.open = set()   # operations between begin and finish
        self.streams = {}   # (device, stream id) -> _StreamCompletions of finished NCCL operations
        self.store = None if group is None else c10d._get_process_group_store(group)
        self.backend = None if group is None else dist.get_backend(group)
        self.backends = {member: None if member_group is None else dist.get_backend(member_group)
                         for member, member_group in self.groups.items()}
        self.deadline = _deadline(group, self.backend) if deadline is None else deadline
        self.gloo_timeouts = {}
        if self.deadline is not None:
            for member_group in self.groups.values():
                if member_group is not None and dist.get_backend(member_group) == "gloo":
                    timeout = _group_timeout(member_group, "gloo")
                    self.gloo_timeouts[member_group] = timeout
                    c10d._set_pg_timeout(timedelta(seconds=min(timeout.total_seconds(), self.deadline)),
                                         member_group)
        if self.store is not None:
            if self.store.check([self.key]):
                self._read_reports()
                self._break()
                raise self.error()
            self.thread = threading.Thread(target=self._watch, daemon=True,
                                           name="defgcm-transport")
            self.thread.start()
        for group_name in self.groups:
            previous = _ACTIVE.get(group_name)
            if previous is not None and previous is not self:
                # A lifetime displaced here receives no further operation
                # (transport_operation looks its group up by name); one left
                # open, e.g. after a non-owner rejection without abort_all,
                # would otherwise poll the Store and hold its groups until exit.
                previous._retire_displaced()
            _ACTIVE[group_name] = self

    @property
    def in_flight(self):
        """Oldest start of an open call or an unfinished NCCL completion event."""
        with self.lock:
            starts = [operation.started for operation in self.open]
            starts += [stream.pending[0][1] for stream in self.streams.values() if stream.pending]
            return min(starts, default=None)

    def capture_unowned(self, name):
        """Whether a captured NCCL call on ``name`` would be refused.

        A captured collective is retired by the completion event of the
        operation that owns the capture (an enclosing collective solve,
        ``capture_owner``); outside one it could never be retired, so
        :meth:`begin` refuses it.  Gate decisions read the same predicate
        before capturing, so the solve declines instead of failing.
        """
        if self.backends[name] != "nccl":
            return False
        with self.lock:
            return not any(operation.capture_owner for operation in self.open)

    def begin(self, name, *, capture_owner=False):
        backend = self.backends[name]
        # A gloo collective cannot be interrupted once enqueued, so a peer's
        # notification is read from the Store first.  NCCL is aborted by the
        # monitor thread, which polls the Store; the local flag suffices and
        # keeps a Store round trip off every halo, SUM and CG dot product.
        if self.notified() if backend == "gloo" else self.broken.is_set():
            raise self.error()
        if (
            backend == "nccl"
            and torch.cuda.is_current_stream_capturing()
            and self.capture_unowned(name)
        ):
            raise TransportFailure("owned data capture requires an enclosing collective solve")
        operation = _Operation(backend, self.deadline, capture_owner)
        with self.lock:
            self.open.add(operation)
        return operation

    def finish(self, operation):
        # wait() orders the current stream after NCCL; CPU return alone does
        # not retire the operation, its event on that stream does. Captured
        # calls are covered by their enclosing solve's event after replay.
        if operation.backend == "nccl" and not torch.cuda.is_current_stream_capturing():
            stream = torch.cuda.current_stream()
            key = (stream.device_index, stream.stream_id)
            with self.lock:
                completions = self.streams.get(key)
                if completions is None:
                    completions = self.streams[key] = _StreamCompletions()
                completions.record(stream, operation.started, operation.works)
                self.open.discard(operation)
            return
        with self.lock:
            self.open.discard(operation)

    def _retire_completed(self):
        with self.lock:
            for completions in self.streams.values():
                completions.retire()

    def _break(self):
        with self.lock:
            if self.broken.is_set():
                return
            self.broken.set()
        # Do not hold the lock while a backend interrupts outstanding Work.
        if any(group is not None and dist.get_backend(group) == "nccl"
               for group in self.groups.values()):
            from defgcm.accel import cuda_graphs
            try:
                cuda_graphs.release_all()
            except Exception as error:
                self.cleanup_errors.append(f"CUDA graph release: {type(error).__name__}: {error}")
        for group in self.groups.values():
            if group is not None:
                # NCCL abort interrupts device waits. Gloo inherits a no-op
                # group abort: its Work/transport timeouts were bounded above.
                try:
                    group.abort()
                except Exception as error:
                    self.cleanup_errors.append(f"group abort: {type(error).__name__}: {error}")

    def _read_reports(self):
        if self.store is None:
            return
        for rank in self.origins:
            key = f"{self.key}/rank/{rank}"
            if self.store.check([key]):
                self.reports[rank] = json.loads(self.store.get(key).decode())
        self.message = "\n".join(
            f"rank {rank}: {report['type']}: {report['message']}"
            for rank, report in sorted(self.reports.items()))

    def notified(self):
        if self.broken.is_set():
            return True
        if self.store is not None and self.store.check([self.key]):
            self._read_reports()
            self._break()
            return True
        return False

    def _watch(self):
        while not self.stopped.wait(.05):
            try:
                if self.notified():
                    return
                self._retire_completed()
            except Exception as error:
                if not self.notified():
                    self.fail(error)
                return
            started = self.in_flight
            if (started is not None and self.deadline is not None
                    and time.monotonic() - started > self.deadline):
                self.fail(TimeoutError(
                    f"no progress on {self.name} within {self.deadline:.1f} s"))
                return

    def fail(self, error):
        # Abort-induced exceptions on a peer are consequences, not new origins.
        if self.broken.is_set():
            return
        origin = getattr(error, "origin_rank", None)
        origin = self.rank if origin is None else origin
        if origin != self.rank or getattr(error, "remote_failure", False):
            self._break()
            return
        self.reports.setdefault(origin, {
            "rank": origin, "type": type(error).__name__, "message": str(error),
            "fault_monotonic": time.monotonic()})
        self.message = f"rank {origin}: {type(error).__name__}: {error}"
        try:
            if self.store is not None:
                self.store.set(f"{self.key}/rank/{origin}", json.dumps(self.reports[origin]))
                self.store.set(self.key, "failed")
        finally:
            self._break()

    def error(self):
        self._read_reports()
        error = TransportFailure("communication group aborted; no gradients may be committed; "
                                 "create a new process group and window\n" + (self.message or "peer failed"))
        error.origin_reports = tuple(self.reports.items())
        error.remote_failure = True
        if self.cleanup_errors:
            error.add_note(f"cleanup on rank {self.rank}: " + "; ".join(self.cleanup_errors))
        return error

    def quiesce(self):
        """Confirm existing work before the control group's completion verdict."""
        while True:
            if self.notified():
                raise self.error()
            try:
                self._retire_completed()
            except Exception as error:
                if not self.notified():
                    self.fail(error)
                raise self.error() from error
            if self.in_flight is None:
                return
            self.stopped.wait(.01)

    def _retire_displaced(self):
        """Stop the monitor of a lifetime another one replaced. **Eager.**

        No collective, no notification and no timeout restore: the new
        lifetime owns the groups' settings now.
        """
        self.stopped.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)
        for name in self.groups:
            if _ACTIVE.get(name) is self:
                del _ACTIVE[name]

    def close(self):
        error = None
        try:
            self.quiesce()
        except Exception as failure:
            error = failure
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        if not self.broken.is_set():
            for group, timeout in self.gloo_timeouts.items():
                c10d._set_pg_timeout(timeout, group)
        for name in self.groups:
            if _ACTIVE.get(name) is self:
                del _ACTIVE[name]
        if error is not None:
            raise error


def _deadline(group, backend):
    configured = os.environ.get("DEFGCM_TRANSPORT_DEADLINE_S")
    if configured:
        return float(configured)
    if group is None:
        return None
    try:
        return _group_timeout(group, backend).total_seconds() / 2
    except Exception:  # noqa: BLE001 - no readable timeout: no deadline
        return None


def _group_timeout(group, backend):
    timeout = _GROUP_TIMEOUTS.get(group)
    if timeout is None:
        device = "cuda" if backend == "nccl" else "cpu"
        timeout = group._get_backend(torch.device(device)).options._timeout
        _GROUP_TIMEOUTS[group] = timeout
    return timeout


def captured_collectives_unowned(name):
    """Whether a CUDA-graph capture of collectives on group ``name`` would be refused.

    True exactly when :meth:`TransportLifetime.begin` would raise for a captured
    NCCL call: a lifetime is active for the group and no capture-owning
    operation (an enclosing collective solve) is open.  Without a lifetime the
    capture is unguarded and runs.  Every rank constructs lifetimes and opens
    solves at the same program points, so the answer is rank-uniform.
    """
    lifetime = _ACTIVE.get(name)
    return lifetime is not None and lifetime.capture_unowned(name)


@contextmanager
def transport_operation(name, *, capture_owner=False):
    """A partial atomic transport/solve cannot safely be issued a second time.

    The operation is in flight for the lifetime's progress deadline.
    """
    lifetime = _ACTIVE.get(name)
    if lifetime is not None and lifetime.broken.is_set():
        raise lifetime.error()
    operation = None if lifetime is None else lifetime.begin(name, capture_owner=capture_owner)
    try:
        yield operation
    except Exception as error:
        if lifetime is not None:
            if not lifetime.notified():
                lifetime.fail(error)
            raise lifetime.error() from error
        raise
    finally:
        if lifetime is not None and not lifetime.broken.is_set():
            try:
                lifetime.finish(operation)
            except Exception as error:
                if not lifetime.notified():
                    lifetime.fail(error)
                raise lifetime.error() from error


def wait_transport(work, operation):
    """Retain a Work handle after NCCL wait has returned on the host."""
    if work is None:  # Synchronous, single-process transport fixtures.
        return None
    if operation is not None:
        operation.works.append(work)
        if operation.deadline is not None and operation.backend == "gloo":
            remaining = max(.001, operation.deadline - (time.monotonic() - operation.started))
            return work.wait(timedelta(seconds=remaining))
    return work.wait()


def data_all_reduce(value, *, op=dist.ReduceOp.SUM, group=None):
    """An untaped reduction still belongs to its window's data abort domain."""
    from .comm_adjoint import active_data_group
    from .halo_p2p import _group_name
    group = active_data_group(group)
    with transport_operation(_group_name(group)) as operation:
        return wait_transport(dist.all_reduce(value, op=op, group=group, async_op=True),
                              operation)


def data_all_gather(outputs, value, *, group=None):
    """Gather in the active data domain without changing the merge arithmetic."""
    from .comm_adjoint import active_data_group
    from .halo_p2p import _group_name
    group = active_data_group(group)
    with transport_operation(_group_name(group)) as operation:
        return wait_transport(dist.all_gather(outputs, value, group=group, async_op=True),
                              operation)
