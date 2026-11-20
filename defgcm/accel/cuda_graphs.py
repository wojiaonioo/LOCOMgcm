"""Registry of hand-captured CUDA graphs, drained before the communicator dies.

The captured PCG graph contains NCCL collectives by design (lat-lon halo P2P and the
all-reduce dots).  A collective captured into a CUDA graph keeps a reference on
its communicator for as long as the graph is instantiated, and the communicator
cannot be destroyed or aborted until every such graph is gone.  So the graph
must be reset *before* the process group is torn down, and nothing in the
interpreter's exit order guarantees that.  Every ``torch.cuda.CUDAGraph`` that
captures collectives registers here at capture time and
:func:`defgcm.distributed.bootstrap.shutdown_distributed` drains the
registry right after the device sync, ahead of the barrier and the (optional)
``destroy_process_group()``.

The set holds weak references only: an entry disappears with the cache that
owns it, so the registry never extends a graph's lifetime."""

from __future__ import annotations

from typing import Any
import weakref

# TransportLifetime's monitor queries CUDA events from a separate thread.
# Those queries must not invalidate a solver capture on the calling thread.
CAPTURE_ERROR_MODE = "thread_local"

# ``torch.cuda.CUDAGraph`` supports weak references (``__weakrefoffset__`` != 0 on
# torch 2.4 and 2.9); the entries are typed ``Any`` so tests can register a stub.
_LIVE: "weakref.WeakSet[Any]" = weakref.WeakSet()


def register(graph: Any) -> None:
    """Track *graph* (anything with ``reset()``) until it is released or collected."""

    _LIVE.add(graph)


def live_count() -> int:
    """Number of registered graphs that are still alive."""

    return len(_LIVE)


def release_all() -> int:
    """Reset every registered graph that is still alive and return how many."""

    graphs = list(_LIVE)
    _LIVE.clear()
    for graph in graphs:
        graph.reset()
    return len(graphs)
