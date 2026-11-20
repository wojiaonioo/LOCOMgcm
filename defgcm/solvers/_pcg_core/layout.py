"""The shared, topology-free operand predicate for every fused PCG kernel.

One function decides both "may this run" (the dispatcher's per-solve gate) and
"does this run" (each ``custom_op``'s own re-check), so a condition can only be
relaxed in one place and relaxing it there relaxes both
(the shared acceleration dispatcher).

Every condition here is about **device, dtype, contiguity, rank, extent or
grad** -- what the kernels can index.  None of them is about a grid family or a
rank count. A low-level
solver operator must not receive a topology label and judge on it.  A cube
problem is refused here because its coefficients carry a face axis the kernels
cannot index, not because something asked what grid it came from; a 40-rank run
is admitted for the same reason a 1-rank run is, namely that nothing in these
conditions can see the difference.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from defgcm.accel.tensor_context import grad_mode_requires_reference


def layout_reason(
    values: Sequence[Any], supported_dtypes: Sequence[torch.dtype]
) -> str | None:
    """Device / dtype / layout reasons shared by every entry point."""
    if any(not isinstance(value, torch.Tensor) for value in values):
        return "non_tensor"
    if any(value.device.type != "cuda" for value in values):
        return "non_cuda"
    first = values[0]
    if any(value.device != first.device for value in values):
        return "device_mismatch"
    dtype = first.dtype
    if dtype not in supported_dtypes:
        return "dtype"
    if any(value.dtype != dtype for value in values):
        return "dtype_mismatch"
    if any(not value.is_contiguous() for value in values):
        return "non_contiguous"
    return None


def autograd_reason(values: Sequence[Any]) -> str | None:
    """``"autograd"`` for the shared grad-enabled program on every rank.

    Input activity can differ by rank. Retain the operand argument for callers,
    but choose the backend from grad mode rather than those local dependencies.

    NOT part of :func:`layout_reason`, and the distinction is load-bearing.  A
    ``torch.library.custom_op`` implementation runs BELOW the autograd key: by
    the time the op body executes, grad mode is off and its arguments have been
    detached, so this predicate is structurally blind there and would report "no
    reason" on exactly the call it exists to refuse.  (Worse than useless: the
    op still returns a tensor wired to an error node, so the failure surfaces at
    ``.backward()``, far from the cause.)

    It therefore lives where it can see the truth -- the dispatcher's per-solve
    gate, which runs above autograd -- and the caller falls back to the eager
    reference, which is differentiable.  Without this, an ``x0.requires_grad=True`` solve reached the kernels and
    raised "no autograd formula was registered" at backward time.

    The shared policy also declines a live forward mode, which ``no_grad`` does
    not stop: the kernels are custom operators without a forward rule and would
    return the iterate without its tangent.  Inside the ROOT path's Function the
    forward mode is off, so its primal solves keep the kernels.
    """
    if grad_mode_requires_reference():
        return "autograd"
    return None


def scalar_reason(*values: Any) -> str | None:
    """``alpha``/``beta`` must be single-element device tensors."""
    for value in values:
        if value.numel() != 1:
            return "scalar_shape"
    return None


def same_shape_reason(reference: Any, *values: Any) -> str | None:
    """Every vector operand must share the state's shape exactly."""
    shape = tuple(reference.shape)
    for value in values:
        if tuple(value.shape) != shape:
            return "state_shape"
    return None


def halo_shape_reason(
    phi_halo: Any, interior_shape: Sequence[int], halo: int = 1
) -> str | None:
    """The padded block must be the interior grown by ``halo`` on both sides.

    The kernels read ``phi_halo[..., j+-1, i+-1]`` and nothing else, so this is
    the whole contract with whatever exchange produced it -- zero fill, periodic
    wrap, N-rank receive or a rotated cube neighbour.
    """
    expected = tuple(int(extent) + 2 * halo for extent in interior_shape[-2:])
    if tuple(phi_halo.shape[-2:]) != expected:
        return "halo_shape"
    leading_interior = tuple(int(extent) for extent in interior_shape[:-2])
    if tuple(phi_halo.shape[: phi_halo.ndim - 2])[-len(leading_interior) or None:] != (
        leading_interior if leading_interior else ()
    ):
        return "halo_shape"
    return None


__all__ = [
    "autograd_reason",
    "halo_shape_reason",
    "layout_reason",
    "same_shape_reason",
    "scalar_reason",
]
