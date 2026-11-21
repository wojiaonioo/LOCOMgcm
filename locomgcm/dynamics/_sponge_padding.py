from __future__ import annotations

import torch


def pad_target_to_field_shape(target: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """Replicate-pad a boundary target band to a field's horizontal shape.

    Sponge target bands are already sampled from physical OBC values.  Padding
    them to U/V/T field shape is a local broadcasting operation, not a halo
    exchange; distributed rank seams must not affect it.
    """

    if target.shape[-1] < field.shape[-1]:
        target = _replicate_pad_after(target, axis=-1, width=field.shape[-1] - target.shape[-1])
    if target.shape[-2] < field.shape[-2]:
        target = _replicate_pad_after(target, axis=-2, width=field.shape[-2] - target.shape[-2])
    return target


def _replicate_pad_after(tensor: torch.Tensor, *, axis: int, width: int) -> torch.Tensor:
    if width <= 0:
        return tensor
    axis = axis if axis >= 0 else tensor.ndim + axis
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"axis {axis} out of range for tensor ndim {tensor.ndim}")
    if tensor.shape[axis] < 1:
        raise ValueError("cannot replicate-pad an empty tensor axis")
    index = [slice(None)] * tensor.ndim
    index[axis] = slice(-1, None)
    pad = tensor[tuple(index)].expand(
        *tensor.shape[:axis],
        width,
        *tensor.shape[axis + 1:],
    )
    return torch.cat((tensor, pad), dim=axis)
