"""Window placement and reverse look-ahead, independent of byte storage."""

import torch

from .store import host_layout_bytes


class WindowScheduler:
    """Keep heads on host by default; spend a declared budget in reverse order."""

    def __init__(self, *, device_bytes=0, prefetch_distance=1):
        if device_bytes is not None and (type(device_bytes) is not int or device_bytes < 0):
            raise ValueError("device_bytes must be None or a non-negative integer")
        if type(prefetch_distance) is not int or prefetch_distance < 0:
            raise ValueError("prefetch_distance must be a non-negative integer")
        self.device_bytes = device_bytes
        self.prefetch_distance = prefetch_distance
        self.placement = {}
        #: Page-locked bytes the window's plan needs, set when head 1 is saved.
        self.host_need_bytes = None

    def open(self, store, count, device, steps=None):
        """``steps``: the step count of each segment, when the caller knows it."""
        self.store, self.count = store, count
        self.steps = None if steps is None else tuple(steps)
        self.device = torch.device(device)
        store.reserve(count, device)
        self.placement = {}
        self.host_need_bytes = None
        self._budget = (torch.cuda.get_device_properties(self.device).total_memory // 2
                        if self.device_bytes is None and self.device.type == "cuda"
                        else self.device_bytes or 0)

    def _place(self, slot, size, used):
        # A slot is admitted only if the same budget can cover all heads that
        # reverse visits earlier, so the budget goes to the last segments.
        return "device" if (self.device.type != "cuda" or
                            size * (self.count - slot) <= self._budget - used) else "host"

    def _planned_attached(self, slot):
        """Byte sizes of the tensors segment ``slot`` will attach to its head (``attach``).

        Segment 0's attachments per step times the segment's own step count
        (one kept cg2d solution per step and solve); before segment 0 has run
        there are none to price.  Without the step counts, or when segment 0's
        attachments do not divide into its steps, segment 0's as they are.
        """
        sizes = self.store.attached_sizes.get(0, ())
        if self.steps is None or len(sizes) % self.steps[0]:
            return sizes
        return sizes[:len(sizes) // self.steps[0]] * self.steps[slot]

    def _check_host(self, slot, tensors, size):
        """Refuse a plan whose host heads exceed the store's page-locked budget.

        Run when head 1 is saved: head 0 lacks the history later heads carry,
        head 1 has the layout every later head repeats.  The plan places the
        remaining heads as ``save`` will, so the window fails here, with the
        bytes it needs, and never at a later head.  From head 1 on the need
        includes the heads ``begin_reverse`` moves to the host before it moves
        others back.  ``save`` also runs it at head 0, before any step, as a
        lower bound (forward placement only).  Each later head is priced with
        what its segment will attach (``_planned_attached``): a host head as its
        own lease plus the attachment's, a device head that ``begin_reverse``
        moves down as one lease holding both.
        """
        head = [t.numel() * t.element_size() for t in tensors]
        used, host_need = self.store.device_bytes_used, self.store.host_bytes_used
        forward = {saved: self.store.placement[saved] for saved in range(slot)}
        slot_size, moved_down = {}, {}
        for later in range(slot, self.count):
            attached = self._planned_attached(later)
            slot_size[later] = size + sum(attached)
            forward[later] = self._place(later, slot_size[later], used)
            if forward[later] == "device":
                used += slot_size[later]
                moved_down[later] = host_layout_bytes(head + list(attached))
            else:
                host_need += host_layout_bytes(head) + host_layout_bytes(attached)
        self.host_need_bytes = host_need
        if slot > 0:
            # ``begin_reverse`` re-places by actual size and moves every head
            # out of its device set before moving any in, so the host holds the
            # forward host heads plus every head it moves down at once.  Head 0
            # is priced at its own size; later heads repeat head ``slot``.
            sizes = [self.store.slot_bytes[s] if s < slot else slot_size[s] for s in range(self.count)]
            remaining, desired = self._budget, set()
            for s in reversed(range(self.count)):
                if sizes[s] <= remaining:
                    desired.add(s)
                    remaining -= sizes[s]
            self.host_need_bytes += sum(
                self.store.slot_host_bytes[s] if s < slot else moved_down[s]
                for s in range(self.count) if forward[s] == "device" and s not in desired)
        self.store.check_host_budget(self.host_need_bytes)

    def save(self, slot, tensors):
        size = sum(t.numel() * t.element_size() for t in tensors)
        # Head 0 is saved before the first step.  Later heads carry its keys
        # at the same shapes plus history, so the plan priced at head 0's size
        # is a lower bound on the host need: a budget below it is refused
        # before any segment runs.  The exact check waits for head 1's layout.
        # ponytail: the exact pre-window check needs the later-head layout
        # declared before the first step (planned with the explicit time program).
        # Head 0's price also lacks what its segment will attach (``attach``),
        # which no step has produced yet; a window of one segment has no head
        # 1, so its exact check runs in ``attach``.
        if self.device.type == "cuda" and slot == 0:
            try:
                self._check_host(slot, tensors, size)
            except MemoryError as error:
                raise MemoryError(f"{error}; priced at head 0, which lacks the history "
                                  "later heads carry and what its segment attaches, "
                                  "so this is a lower bound") from error
            self._head_sizes = [t.numel() * t.element_size() for t in tensors]
        if self.device.type == "cuda" and slot == 1:
            self._check_host(slot, tensors, size)
        # The slot also holds what the segment attaches once it has run.
        place = self._place(slot, size + sum(self._planned_attached(slot)),
                            self.store.device_bytes_used)
        self.store.put(slot, tensors, placement=place)
        self.placement[slot] = place

    def begin_reverse(self):
        if self.device.type == "cuda":
            # A later head may have a different layout. Reconcile using actual
            # sizes so an early small head cannot displace a later large one.
            remaining, desired = self._budget, set()
            for slot in reversed(range(self.count)):
                size = self.store.slot_bytes[slot]
                if size <= remaining:
                    desired.add(slot)
                    remaining -= size
            for slot in range(self.count):
                if slot not in desired:
                    self.store.move(slot, "host")
            for slot in sorted(desired, reverse=True):
                self.store.move(slot, "device")
            self.placement = dict(self.store.placement)
        for slot in range(self.count - 1, max(-1, self.count - 1 - self.prefetch_distance), -1):
            self.store.prefetch(slot)

    def attach(self, slot, tensors):
        """Keep a segment's later results with its head (``CheckpointStore.attach``).

        A window of one segment is checked exactly here, before the lease: its
        head is stored, and what it attaches takes a lease of its own on the
        host, or moves down with the head when ``begin_reverse`` finds the two
        above the device budget.
        """
        if self.device.type == "cuda" and self.count == 1:
            sizes = [t.numel() * t.element_size() for t in tensors]
            need = self.store.host_bytes_used
            if self.store.placement[slot] == "host":
                need += host_layout_bytes(sizes)
            elif self.store.slot_bytes[slot] + sum(sizes) > self._budget:
                need += host_layout_bytes(self._head_sizes + sizes)
            self.host_need_bytes = need
            self.store.check_host_budget(need)
        self.store.attach(slot, tensors)

    def take(self, slot):
        tensors = self.store.take(slot)
        for previous in range(slot - 1, max(-1, slot - 1 - self.prefetch_distance), -1):
            self.store.prefetch(previous)
        return tensors

    def description(self):
        return {"policy": "reverse_first", "device_bytes": self.device_bytes,
                "prefetch_distance": self.prefetch_distance,
                "host_need_bytes": self.host_need_bytes,
                "slots": dict(self.placement)}
