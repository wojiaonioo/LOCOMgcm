"""Opaque point-to-point halo traffic for compiled distributed execution.

The exchange is a collective: every rank must issue exactly the same sequence of
point-to-point rounds, because NCCL pairs coalesced groups by issue order rather
than by user tag.  A ``torch.library`` custom operation declared with
``mutates_args=()`` is a *pure* operation as far as the compiler is concerned --
``FallbackKernel.has_side_effects()`` is driven purely by schema mutability
(``torch/_inductor/ir.py:5463``), so Inductor deletes a call whose results are
unused and is free to reorder calls that carry no dependency between them.  In
the model the consumer of a received halo is rank dependent (open boundaries,
masks), so one rank's exchange was deleted while its peer still posted it, and
the first unpaired round deadlocked.

The operation therefore takes an ``order_token``: a host tensor, owned by
this module and shared by every call, that the operation mutates.
The mutation makes the schema mutable, which (a) forbids dead-code elimination
and (b) chains the calls through the token so program order is a data
dependency.  ``auto_functionalize`` in torch 2.4 only accepts plain ``Tensor``
mutable arguments and plain ``Tensor`` returns
(``torch/_higher_order_ops/auto_functionalize.py:83-105``), so the receives come
back as one flat buffer that :func:`halo_p2p` slices into per-message views --
the views are contiguous and share the flat buffer's storage, and every caller
treats them read-only.

Set ``LOCOMGCM_HALO_P2P_CUDAGRAPH_UNSAFE=0`` to omit the CUDA-graph safety tag.
The tag defaults to enabled until the distributed CUDA capture test confirms
that the operation can be captured safely.
"""

from __future__ import annotations

import inspect
import os

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d

from locomgcm.accel.tensor_context import forward_mode_active


_CUDA_GRAPH_UNSAFE = getattr(torch._C.Tag, "cudagraph_unsafe", None)
HALO_P2P_TAGS = (
    (_CUDA_GRAPH_UNSAFE,)
    if os.environ.get("LOCOMGCM_HALO_P2P_CUDAGRAPH_UNSAFE", "1") == "1"
    and _CUDA_GRAPH_UNSAFE is not None
    else ()
)

_CUSTOM_OP_KWARGS: dict[str, object] = {
    "mutates_args": ("order_token",),
    "schema": (
        "(Tensor(a!) order_token, Tensor[] sends, Tensor[] recv_like, "
        "int[] peers_send, int[] peers_recv, int[] tags_send, int[] tags_recv, "
        "str group_name) -> Tensor"
    ),
}
if "tags" in inspect.signature(torch.library.custom_op).parameters:
    _CUSTOM_OP_KWARGS["tags"] = HALO_P2P_TAGS

# Created at import, on the host, and never replaced: a token that only exists
# once tracing reaches it would make Dynamo guard on "the holder is still empty"
# and recompile the whole frame on the second call, and the first trace would
# allocate it *inside* the graph and publish a graph-owned tensor to module
# scope.  Its device is irrelevant -- it carries no exchange data, and the
# operation's scheduling device still comes from the halo buffers. Slot 0
# counts collectives, slot 1 counts explicitly ordered compute results. Both
# mutate the same tensor so the compiler keeps their relative issue order.
_ORDER_TOKEN = torch.zeros(2, dtype=torch.int64, device="cpu")


def halo_p2p_issue_count() -> int:
    """Exchanges issued by this process so far.

    Ranks that agree on their exchange sequence agree on this count; a
    divergence is the signature of a deleted or duplicated round.
    """
    return int(_ORDER_TOKEN[0].item())


def _receive_layout(
    sends: list[torch.Tensor], recv_like: list[torch.Tensor]
) -> tuple[torch.Tensor, int]:
    """Reference tensor (dtype/device of the flat receive buffer) and its numel."""
    if not recv_like and not sends:
        raise ValueError("halo_p2p needs at least one send or one receive")
    reference = recv_like[0] if recv_like else sends[0]
    total = 0
    for tensor in recv_like:
        if tensor.dtype != reference.dtype or tensor.device != reference.device:
            raise ValueError(
                "halo_p2p receives must share one dtype and device; got "
                f"{tensor.dtype} on {tensor.device} and "
                f"{reference.dtype} on {reference.device}"
            )
        total += tensor.numel()
    return reference, total


def _split_receives(
    flat: torch.Tensor, recv_like: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Contiguous per-message views into the flat receive buffer."""
    views = []
    offset = 0
    for tensor in recv_like:
        count = tensor.numel()
        views.append(flat[offset : offset + count].view(tensor.shape))
        offset += count
    return views


@torch.library.custom_op("locomgcm::halo_p2p", **_CUSTOM_OP_KWARGS)
def _halo_p2p_op(
    order_token: torch.Tensor,
    sends: list[torch.Tensor],
    recv_like: list[torch.Tensor],
    peers_send: list[int],
    peers_recv: list[int],
    tags_send: list[int],
    tags_recv: list[int],
    group_name: str,
) -> torch.Tensor:
    """Post receives before sends, wait for all requests, return them flattened."""
    if len(sends) != len(peers_send) or len(sends) != len(tags_send):
        raise ValueError("sends, peers_send, and tags_send must have equal lengths")
    if len(recv_like) != len(peers_recv) or len(recv_like) != len(tags_recv):
        raise ValueError("recv_like, peers_recv, and tags_recv must have equal lengths")

    # Declared and performed: this is what pins the round into the graph and
    # into program order (see module docstring).
    order_token[0].add_(1)

    group = (
        distributed_c10d._resolve_process_group(group_name)
        if group_name
        else None
    )
    from .comm_adjoint import active_data_group
    from .transport_failure import transport_operation, wait_transport
    group = active_data_group(group)
    with transport_operation(_group_name(group)) as operation:
        reference, total = _receive_layout(sends, recv_like)
        flat = torch.empty(total, dtype=reference.dtype, device=reference.device)
        received = _split_receives(flat, recv_like)
        ops = [
            dist.P2POp(dist.irecv, tensor, peer, tag=tag, group=group)
            for tensor, peer, tag in zip(received, peers_recv, tags_recv)
        ]
        ops.extend(
            dist.P2POp(dist.isend, tensor, peer, tag=tag, group=group)
            for tensor, peer, tag in zip(sends, peers_send, tags_send)
        )
        for request in dist.batch_isend_irecv(ops):
            wait_transport(request, operation)
    return flat


@_halo_p2p_op.register_fake
def _halo_p2p_fake(
    order_token: torch.Tensor,
    sends: list[torch.Tensor],
    recv_like: list[torch.Tensor],
    peers_send: list[int],
    peers_recv: list[int],
    tags_send: list[int],
    tags_recv: list[int],
    group_name: str,
) -> torch.Tensor:
    del order_token, peers_send, peers_recv, tags_send, tags_recv, group_name
    reference, total = _receive_layout(sends, recv_like)
    return torch.empty(total, dtype=reference.dtype, device=reference.device)


_ALL_REDUCE_KWARGS: dict[str, object] = {
    "mutates_args": ("order_token",),
    "schema": "(Tensor(a!) order_token, Tensor value, str group_name) -> Tensor",
}
if "tags" in inspect.signature(torch.library.custom_op).parameters:
    _ALL_REDUCE_KWARGS["tags"] = HALO_P2P_TAGS


@torch.library.custom_op("locomgcm::ordered_all_reduce_sum", **_ALL_REDUCE_KWARGS)
def _ordered_all_reduce_sum_op(
    order_token: torch.Tensor, value: torch.Tensor, group_name: str
) -> torch.Tensor:
    """SUM all-reduce of ``value`` chained through the halo order token.

    A plain ``dist.all_reduce``
    inside the compiled step is traced as a *functional* collective, which
    carries no dependency on the token-ordered halo rounds, so Inductor was
    free to schedule it before or after them -- and did so differently on the
    two ranks, whose graphs differ (open boundaries live on one tile).  Rank 0
    blocked in a coalesced P2P round while rank 1 blocked in this all-reduce.
    Mutating the same token pins the reduction into the ranks' shared issue
    order, exactly as the P2P rounds are pinned.  The arithmetic is the same
    ``all_reduce(SUM)`` on a clone, so the result is bit-identical.
    """
    order_token[0].add_(1)
    group = (
        distributed_c10d._resolve_process_group(group_name)
        if group_name
        else None
    )
    result = value.clone()
    from .transport_failure import data_all_reduce
    data_all_reduce(result, group=group)
    return result


@_ordered_all_reduce_sum_op.register_fake
def _ordered_all_reduce_sum_fake(
    order_token: torch.Tensor, value: torch.Tensor, group_name: str
) -> torch.Tensor:
    del order_token, group_name
    return torch.empty_like(value)


@torch.library.custom_op(
    "locomgcm::compute_issued",
    **{**_ALL_REDUCE_KWARGS, "schema": "(Tensor(a!) order_token, Tensor result) -> ()"},
)
def _compute_issued_op(order_token: torch.Tensor, result: torch.Tensor) -> None:
    # The result is a compiler dependency, not a host read of device data.
    # This marks issue order; GPU work continues asynchronously.
    del result
    order_token[1].add_(1)


@_compute_issued_op.register_fake
def _compute_issued_fake(order_token: torch.Tensor, result: torch.Tensor) -> None:
    return None


def after_compute(result: torch.Tensor) -> None:
    """Keep an opaque computation ahead of the next ordered collective.

    Its output may otherwise be consumed only much later, allowing Inductor
    to move the computation past intervening communication. Different rank
    graphs can then serialize work that should execute concurrently. The
    result dependency and shared token mutation close that scheduling edge
    without reading device data or adding a communication round.
    """
    _compute_issued_op(_ORDER_TOKEN, result)


_GHOST_KWARGS: dict[str, object] = {"mutates_args": ()}
if "tags" in _CUSTOM_OP_KWARGS:
    _GHOST_KWARGS["tags"] = HALO_P2P_TAGS


@torch.library.custom_op("locomgcm::ghost_after_send", **_GHOST_KWARGS)
def _ghost_after_send_op(order_token: torch.Tensor, ghost: torch.Tensor) -> torch.Tensor:
    # Opaque token read supplies the dependency; only the boundary strip is copied.
    del order_token
    return ghost.clone()


@_ghost_after_send_op.register_fake
def _ghost_after_send_fake(order_token: torch.Tensor, ghost: torch.Tensor) -> torch.Tensor:
    del order_token
    return torch.empty_like(ghost)


def inference_ghost_after_send(ghost: torch.Tensor) -> torch.Tensor | None:
    """A local ghost dependent on preceding sends, or None for the native path.

    Receive outputs already depend on their collective. A send-only round
    needs the same edge for a locally filled ghost; otherwise rank-dependent
    compilation can defer the send until after its peer's following work.
    The dependency only constrains issue order and never reads device data.

    Eager execution is ordered already. Differentiation retains the native
    tensor path: a custom-op identity has no forward-mode rule in PyTorch 2.9.
    """
    if not torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    if (
        torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad.unpack_dual(ghost).tangent is not None
    ):
        return None
    return _ghost_after_send_op(_ORDER_TOKEN, ghost)


def _refuse_forward_mode(operator: str) -> None:
    """Refuse a live forward mode before an inference transport operator issues anything.

    These operators register no autograd at all; with a dual input a custom
    operator returns its output without the tangent and without an error.  The
    reading is program state (``forward_mode_active``): ranks that all open
    the same dual level refuse at the same operator.
    """
    if forward_mode_active():
        raise NotImplementedError(
            f"{operator}: the inference transport has no forward-mode rule; a "
            f"forward-mode tangent would be dropped; take forward-mode derivatives "
            f"on a single rank")


def ordered_all_reduce_sum(
    value: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """``all_reduce(SUM)`` of ``value`` on the halo issue order (see the op)."""
    _refuse_forward_mode("ordered_all_reduce_sum")
    return _ordered_all_reduce_sum_op(_ORDER_TOKEN, value, _group_name(group))


def _group_name(group: dist.ProcessGroup | None) -> str:
    if group is None:
        return ""
    getter = getattr(distributed_c10d, "_get_group_name", None)
    if getter is None:
        getter = distributed_c10d._get_process_group_name
    return str(getter(group))


def halo_p2p(
    sends: list[torch.Tensor],
    recv_like: list[torch.Tensor],
    peers_send: list[int],
    peers_recv: list[int],
    tags_send: list[int],
    tags_recv: list[int],
    group: dist.ProcessGroup | None = None,
) -> list[torch.Tensor]:
    """Run one receive-first P2P round as an opaque, order-pinned operation."""
    _refuse_forward_mode("halo_p2p")
    flat = _halo_p2p_op(
        _ORDER_TOKEN,
        sends,
        recv_like,
        peers_send,
        peers_recv,
        tags_send,
        tags_recv,
        _group_name(group),
    )
    return _split_receives(flat, recv_like)


__all__ = ["HALO_P2P_TAGS", "halo_p2p", "halo_p2p_issue_count", "ordered_all_reduce_sum",
           "after_compute", "inference_ghost_after_send"]
