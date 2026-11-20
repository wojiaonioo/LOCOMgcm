"""Window byte storage with explicit placement and asynchronous stream events.

The scheduler supplies placement and prefetch requests. Host buffers have an
exact aligned size, are registered once and can be reused by later windows.
The executor consumes a slot with ``take``; the store retains no device alias
of that head. Timing events describe producer readiness, transfers and waits.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import weakref

import torch

#: Byte alignment of each tensor inside a slot's host buffer.
_ALIGN = 512


class _PinnedBuffer:
    """``nbytes`` of page-locked host memory, unlocked when the buffer is dropped."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes
        self.tensor = torch.empty(nbytes, dtype=torch.uint8)
        # Fault the pages in on the intra-op threads before the driver locks them.
        self.tensor.zero_()
        cudart = torch.cuda.cudart()
        torch.cuda.check_error(cudart.cudaHostRegister(self.tensor.data_ptr(), nbytes, 0))
        #: Recorded on the releasing store's copy stream after its last copy
        #: of this buffer; the next user's stream waits on it.
        self.released = None
        self.pending_stream = None
        self.returning = False
        self.keepalive = ()
        self._finalizer = weakref.finalize(self, _unregister, self.tensor)

    def close(self) -> None:
        """Finish the last copy and unregister idle storage before accounting it away."""
        if self.released is not None:
            self.released.synchronize()
        if self._finalizer.alive:
            torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(self.tensor.data_ptr()))
            self._finalizer.detach()
        self.keepalive = ()


def _unregister(tensor: torch.Tensor) -> None:
    try:
        torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())
    except Exception:  # interpreter shutdown: the process releases the lock itself
        pass


class _PinnedPool:
    """The process's page-locked buffers, reused by size across stores and windows.

    One pool per process: a buffer a store releases returns here and any
    later store takes it. A finite host budget bounds all registered bytes,
    including idle buffers and other stores' leases.
    """

    def __init__(self):
        self.free: dict[int, list[_PinnedBuffer]] = {}
        self.busy: set[_PinnedBuffer] = set()
        self.locked_bytes = 0
        self._lock = threading.RLock()

    def _finish_return(self, buffer) -> None:
        self.free.setdefault(buffer.nbytes, []).append(buffer)
        self.busy.remove(buffer)
        buffer.returning = False

    def _reap(self) -> None:
        # A failed event record must not force a CPU wait before the owner can
        # abort a damaged communication group. Keep the lease until the stream
        # has actually finished; a valid recorded event permits ordered reuse.
        for buffer in tuple(self.busy):
            if buffer.pending_stream is not None:
                if not buffer.pending_stream.query():
                    continue
                buffer.pending_stream = None
                buffer.keepalive = ()
            if buffer.returning:
                self._finish_return(buffer)
            elif buffer.keepalive and (buffer.released is None or buffer.released.query()):
                buffer.keepalive = ()
        for buffers in self.free.values():
            for buffer in buffers:
                if buffer.keepalive and (buffer.released is None or buffer.released.query()):
                    buffer.keepalive = ()

    def _trim(self, limit, additional=0) -> None:
        if limit is None:
            return
        while self.locked_bytes + additional > limit:
            buffers = next((items for items in self.free.values() if items), None)
            if buffers is None:
                raise MemoryError(
                    f"page-locked pool needs {self.locked_bytes + additional} registered bytes "
                    f"({self.locked_bytes} active, {additional} additional); host budget is {limit}")
            buffer = buffers[-1]
            buffer.close()
            buffers.pop()
            self.locked_bytes -= buffer.nbytes

    def trim(self, limit) -> None:
        with self._lock:
            self._reap()
            self._trim(limit)

    def check_budget(self, nbytes, held, limit) -> None:
        with self._lock:
            self._reap()
            self._trim(limit)
            idle = sum(size * len(buffers) for size, buffers in self.free.items())
            other_active = self.locked_bytes - idle - held
            required = other_active + nbytes
            if limit is not None and required > limit:
                raise MemoryError(
                    f"checkpoint window needs {nbytes} bytes of page-locked host memory "
                    f"plus {other_active} active bytes from other pool leases "
                    f"({required} total); the pool's host budget is {limit}")

    def take(self, nbytes: int, *, limit=None) -> _PinnedBuffer:
        with self._lock:
            self._reap()
            self._trim(limit)
            free = self.free.get(nbytes)
            if free:
                buffer = free[-1]
                self.busy.add(buffer)
                free.pop()
                return buffer
            self._trim(limit, nbytes)
            buffer = _PinnedBuffer(nbytes)
            try:
                self.busy.add(buffer)
            except BaseException:
                buffer.close()
                raise
            self.locked_bytes += nbytes
            return buffer

    def _fence(self, buffer, stream, keepalive) -> None:
        if stream is not None:
            buffer.released = None
            buffer.pending_stream = stream
        if keepalive:
            buffer.keepalive = (buffer.keepalive, keepalive)
        if stream is not None:
            event = torch.cuda.Event()
            event.record(stream)
            buffer.released = event
            buffer.pending_stream = None

    def hold(self, buffer, stream, *, keepalive) -> None:
        """Keep a failed copy's resources without ending its active host slot."""
        with self._lock:
            if buffer not in self.busy:
                raise RuntimeError("holding a pinned buffer that is not leased")
            self._fence(buffer, stream, keepalive)

    def give(self, buffer: _PinnedBuffer, stream=None, *, keepalive=()) -> None:
        # Each store copies on a stream of its own.  Within one store, later
        # copies queue behind earlier ones; a buffer that moves to another
        # store needs that store's stream to wait for the last copy here.
        with self._lock:
            if buffer not in self.busy:
                raise RuntimeError("returning a pinned buffer that is not leased")
            buffer.returning = True
            self._fence(buffer, stream, keepalive)
            self._finish_return(buffer)

    def release_free(self) -> None:
        """Unlock every buffer no slot holds, after its last copy."""
        with self._lock:
            self._reap()
            for buffers in self.free.values():
                while buffers:
                    buffer = buffers[-1]
                    buffer.close()
                    buffers.pop()
                    self.locked_bytes -= buffer.nbytes


_POOL = _PinnedPool()


def release_pinned_pool() -> None:
    """Unlock the process's idle page-locked buffers (held slots keep theirs)."""
    _POOL.release_free()


def host_head_bytes(tensors) -> int:
    """Page-locked bytes one head occupies in a host slot."""
    return _host_layout(tensors)[1]


def host_layout_bytes(sizes) -> int:
    """Page-locked bytes of one host buffer holding tensors of these byte sizes."""
    return _aligned_layout(sizes)[1]


def _host_layout(tensors) -> tuple[list[int], int]:
    """Aligned byte offset of each tensor in one buffer, and the buffer size."""
    return _aligned_layout([tensor.numel() * tensor.element_size() for tensor in tensors])


def _aligned_layout(sizes) -> tuple[list[int], int]:
    offsets, total = [], 0
    for size in sizes:
        total = -(-total // _ALIGN) * _ALIGN
        offsets.append(total)
        total += size
    return offsets, total


def _versions(tensors) -> tuple[int, ...]:
    return tuple(tensor._version for tensor in tensors)


def _require_dense(tensor: torch.Tensor) -> None:
    """A host slot copies ``numel`` elements in the tensor's own stride order."""
    expected = 1
    for stride, size in sorted((stride, size) for stride, size in zip(tensor.stride(), tensor.shape)
                               if size != 1):
        if stride != expected:
            raise ValueError("a stored segment head tensor must be non-overlapping and dense")
        expected *= size


@dataclass
class _Slot:
    tensors: tuple[torch.Tensor, ...] | None = None    # device or CPU-model slot
    versions: tuple[int, ...] = ()                     # of ``tensors`` when stored
    views: tuple[torch.Tensor, ...] | None = None      # host slot: views of ``buffer``
    buffer: _PinnedBuffer | None = None
    attached: tuple[_PinnedBuffer, ...] = ()           # host slot: leases of ``attach``
    loaded: tuple[torch.Tensor, ...] | None = None     # host slot copied back
    ready: torch.cuda.Event | None = None
    nbytes: int = 0


class CheckpointStore:
    """Byte slots of one window at a time, on device or in pinned host buffers.

    ``device_bytes`` is an entry-point budget passed to the window scheduler.
    The store does not interpret it; ``put`` always requires placement.
    ``host_bytes`` bounds the process pool's active and idle registered bytes
    for this window; ``None`` sets no bound. The buffers come from the process's
    one pinned pool. The scheduler checks a window's host need against this
    bound when the second head is saved.  A head that fits neither is an
    error: a further tier (disk) belongs to a checkpoint schedule, not to
    this store.  A model on the CPU keeps every head where it is.
    """

    def __init__(self, *, device_bytes: int | None = 0, host_bytes: int | None = None):
        for name, value in (("device_bytes", device_bytes), ("host_bytes", host_bytes)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be None or a non-negative int")
        self.device_bytes = device_bytes
        self.host_bytes = host_bytes
        self._device: torch.device | None = None
        self._stream: torch.cuda.Stream | None = None
        self._count = 0
        self._slots: dict[int, _Slot] = {}
        #: Page-locked bytes this store's host slots hold now.
        self.host_bytes_used = 0
        self.device_bytes_used = 0
        #: Tier of every slot of the current window, by slot number ("device" or "host").
        self.placement: dict[int, str] = {}
        self.events: dict[int, dict[str, torch.cuda.Event]] = {}
        self.slot_bytes: dict[int, int] = {}
        #: Page-locked bytes each head of the window occupies in a host slot.
        self.slot_host_bytes: dict[int, int] = {}
        #: Byte size of each tensor ``attach`` added to a slot, in order.
        self.attached_sizes: dict[int, tuple[int, ...]] = {}

    @property
    def pinned_bytes(self) -> int:
        """Page-locked bytes of the process's pool (held and idle)."""
        with _POOL._lock:
            return _POOL.locked_bytes

    # -- window -------------------------------------------------------------
    def reserve(self, count: int, device) -> None:
        """Open a window of ``count`` slots on ``device``; a previous window's slots end here."""
        self.release_all()
        self._count = 0
        self._device = torch.device(device)
        if self._device.type == "cuda":
            _POOL.trim(self.host_bytes)
            if self._device.index is None:
                self._device = torch.device("cuda", torch.cuda.current_device())
            if self._stream is None or self._stream.device != self._device:
                self._stream = torch.cuda.Stream(device=self._device)
        self.placement = {}
        self.events = {}
        self.slot_bytes = {}
        self.slot_host_bytes = {}
        self.attached_sizes = {}
        self._count = count

    def release_all(self) -> None:
        error = None
        for slot in list(self._slots):
            try:
                self.release(slot)
            except BaseException as failure:
                if error is None:
                    error = failure
        if self._stream is not None:
            # Device memory freed by this window may be reused by the compute
            # stream only after every copy that touched it.
            try:
                torch.cuda.current_stream(self._stream.device).wait_stream(self._stream)
                with _POOL._lock:
                    _POOL._reap()
            except BaseException as failure:
                if error is None:
                    error = failure
        if error is not None:
            raise error

    def check_host_budget(self, nbytes: int) -> None:
        """Check the whole window plus other active leases against the pool bound."""
        _POOL.check_budget(nbytes, self.host_bytes_used, self.host_bytes)

    # -- slots ----------------------------------------------------------------
    def put(self, slot: int, tensors, *, placement: str) -> None:
        """Keep ``tensors`` (one head) in ``slot`` until ``release(slot)``."""
        if self._device is None or not 0 <= slot < self._count:
            raise RuntimeError(f"checkpoint store slot {slot} is outside the reserved window")
        if slot in self._slots:
            raise RuntimeError(f"checkpoint store slot {slot} is already filled")
        tensors = tuple(tensor.detach() for tensor in tensors)
        if placement not in ("device", "host"):
            raise ValueError("placement must be device or host")
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        offsets, total = _host_layout(tensors)
        absent = object()
        maps = (self.slot_bytes, self.slot_host_bytes, self.placement, self.events)
        previous = tuple(mapping.get(slot, absent) for mapping in maps)
        host_before, device_before = self.host_bytes_used, self.device_bytes_used
        buffer = None
        try:
            self.slot_bytes[slot], self.slot_host_bytes[slot] = nbytes, total
            if self._device.type != "cuda":
                entry = _Slot(tensors=tensors, versions=_versions(tensors))
                tier = "device"
            else:
                for tensor in tensors:
                    if tensor.device != self._device:
                        raise ValueError(f"segment head tensor on {tensor.device}, store on {self._device}")
                    _require_dense(tensor)
                self.events[slot] = {}
                self._event(slot, "producer_ready", torch.cuda.current_stream(self._device))
                if placement == "device":
                    entry = _Slot(tensors=tensors, versions=_versions(tensors), nbytes=nbytes)
                    self.device_bytes_used += nbytes
                    tier = "device"
                else:
                    buffer = self._take(total)
                    views = tuple(
                        buffer.tensor[offset:offset + tensor.numel() * tensor.element_size()]
                        .view(tensor.dtype).as_strided(tensor.shape, tensor.stride())
                        for offset, tensor in zip(offsets, tensors))
                    self._stream.wait_event(self.events[slot]["producer_ready"])
                    with torch.cuda.stream(self._stream):
                        for view, tensor in zip(views, tensors):
                            # Register before enqueue: a later copy or record
                            # failure must not free an earlier copy's source.
                            tensor.record_stream(self._stream)
                            view.copy_(tensor, non_blocking=True)
                    self._event(slot, "host_ready", self._stream)
                    entry = _Slot(views=views, buffer=buffer)
                    tier = "host"
            self._slots[slot] = entry
            self.placement[slot] = tier
        except BaseException as error:
            self._slots.pop(slot, None)
            self.host_bytes_used, self.device_bytes_used = host_before, device_before
            for mapping, old in zip(maps, previous):
                if old is absent:
                    mapping.pop(slot, None)
                else:
                    mapping[slot] = old
            if buffer is not None:
                try:
                    _POOL.give(buffer, self._stream, keepalive=tensors)
                except BaseException as cleanup:
                    # The pool retains a returning lease even without a valid
                    # event; only actual stream completion makes it reusable.
                    raise error from cleanup
            raise

    def attach(self, slot: int, tensors) -> None:
        """Add ``tensors`` to the head in ``slot``; ``get`` returns them after the head.

        For results a segment produces after its head was stored (its kept cg2d
        solutions).  A device or CPU slot keeps them with their versions.  A host
        slot copies them on the store's stream, after the producing stream's
        work so far, into a second lease from the pool that ``release`` returns
        with the first, and records ``attached_host_ready`` after the copies;
        ``prefetch`` brings them back with the head.
        """
        entry = self._slots[slot]
        tensors = tuple(tensor.detach() for tensor in tensors)
        if not tensors:
            return
        if entry.loaded is not None:
            raise RuntimeError(f"checkpoint store slot {slot} was already copied back")
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        if entry.views is None:
            entry.tensors += tensors
            entry.versions += _versions(tensors)
            if self._device.type == "cuda":
                entry.nbytes += nbytes
                self.device_bytes_used += nbytes
            total = _host_layout(entry.tensors)[1] - self.slot_host_bytes[slot]
        else:
            for tensor in tensors:
                if tensor.device != self._device:
                    raise ValueError(f"attached tensor on {tensor.device}, store on {self._device}")
                _require_dense(tensor)
            offsets, total = _host_layout(tensors)
            buffer = self._take(total)
            views = tuple(
                buffer.tensor[offset:offset + tensor.numel() * tensor.element_size()]
                .view(tensor.dtype).as_strided(tensor.shape, tensor.stride())
                for offset, tensor in zip(offsets, tensors))
            try:
                self._stream.wait_stream(torch.cuda.current_stream(self._device))
                with torch.cuda.stream(self._stream):
                    for view, tensor in zip(views, tensors):
                        tensor.record_stream(self._stream)
                        view.copy_(tensor, non_blocking=True)
            except BaseException:
                self.host_bytes_used -= buffer.nbytes
                _POOL.give(buffer, self._stream, keepalive=tensors)
                raise
            entry.views += views
            entry.attached += (buffer,)
            self._event(slot, "attached_host_ready", self._stream)
        self.slot_bytes[slot] += nbytes
        self.slot_host_bytes[slot] += total
        self.attached_sizes[slot] = self.attached_sizes.get(slot, ()) + tuple(
            tensor.numel() * tensor.element_size() for tensor in tensors)

    def prefetch(self, slot: int) -> None:
        """Start copying a host slot back to the device; any other slot is left alone."""
        entry = self._slots.get(slot)
        if entry is None or entry.views is None or entry.loaded is not None:
            return
        loaded = ()
        try:
            with torch.cuda.stream(self._stream):
                # Allocation and use stay on the copy stream, also on failure.
                loaded = tuple(torch.empty_strided(view.shape, view.stride(), dtype=view.dtype,
                                                   device=self._device) for view in entry.views)
                self._event(slot, "h2d_start", self._stream)
                for target, view in zip(loaded, entry.views):
                    target.copy_(view, non_blocking=True)
            ready = self._event(slot, "h2d_end", self._stream)
        except BaseException as error:
            for buffer in (entry.buffer, *entry.attached):
                try:
                    _POOL.hold(buffer, self._stream, keepalive=loaded)
                except BaseException as cleanup:
                    raise error from cleanup
            raise
        entry.ready = ready
        entry.loaded = loaded

    def get(self, slot: int) -> tuple[torch.Tensor, ...]:
        """The head in ``slot`` on the device, ordered after its copy on the compute stream."""
        entry = self._slots[slot]
        if entry.tensors is not None:
            if _versions(entry.tensors) != entry.versions:
                raise RuntimeError(f"segment head in store slot {slot} was modified in place "
                                   "after the forward pass stored it")
            return entry.tensors
        self.prefetch(slot)
        compute = torch.cuda.current_stream(self._device)
        self._event(slot, "compute_wait_start", compute)
        compute.wait_event(entry.ready)
        self._event(slot, "compute_wait_end", compute)
        for tensor in entry.loaded:
            tensor.record_stream(compute)
        return entry.loaded

    def take(self, slot: int) -> tuple[torch.Tensor, ...]:
        """Transfer the loaded head to the executor and end the slot."""
        tensors = self.get(slot)
        self.release(slot)
        return tensors

    def move(self, slot, placement):
        """Execute a scheduler relocation without retaining a second device head."""
        if self.placement[slot] == placement:
            return
        if placement == "host":
            tensors = self.get(slot)
            self.release(slot)
            self.put(slot, tensors, placement="host")
        elif placement == "device":
            tensors = self.get(slot)
            self.release(slot)
            size = self.slot_bytes[slot]
            self._slots[slot] = _Slot(tensors=tensors, versions=_versions(tensors), nbytes=size)
            self.device_bytes_used += size
            self.placement[slot] = "device"
        else:
            raise ValueError("placement must be device or host")

    def _event(self, slot, name, stream):
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        self.events[slot][name] = event
        return event

    def timings(self):
        """Read timestamps after the caller's measurement synchronization."""
        result = {}
        for slot, events in self.events.items():
            origin = events["producer_ready"]
            result[slot] = {name + "_ms": origin.elapsed_time(event)
                            for name, event in events.items()}
            if "compute_wait_end" in events:
                result[slot]["compute_wait_ms"] = events["compute_wait_start"].elapsed_time(
                    events["compute_wait_end"])
        return result

    def release(self, slot: int) -> None:
        """End ``slot``: its device tensors and its host buffer become reusable."""
        entry = self._slots.pop(slot, None)
        if entry is None:
            return
        self.device_bytes_used -= entry.nbytes
        for buffer in (entry.buffer, *entry.attached):
            if buffer is not None:
                self.host_bytes_used -= buffer.nbytes
                _POOL.give(buffer, self._stream)

    def _take(self, nbytes: int) -> _PinnedBuffer:
        if self.host_bytes is not None and self.host_bytes_used + nbytes > self.host_bytes:
            raise MemoryError(
                f"segment head of {nbytes} bytes exceeds the store's page-locked budget "
                f"({self.host_bytes_used} of {self.host_bytes} bytes held)")
        buffer = _POOL.take(nbytes, limit=self.host_bytes)
        before = self.host_bytes_used
        try:
            if buffer.released is not None:
                self._stream.wait_event(buffer.released)
            self.host_bytes_used += nbytes
            return buffer
        except BaseException:
            self.host_bytes_used = before
            # No new copy used the lease; preserve its previous completion
            # event if the cross-stream wait itself failed.
            _POOL.give(buffer)
            raise


__all__ = ["CheckpointStore", "host_head_bytes", "host_layout_bytes", "release_pinned_pool"]
