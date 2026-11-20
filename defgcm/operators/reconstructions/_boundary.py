from __future__ import annotations

import torch


def prepare_obc_tracer(
    values: torch.Tensor | None,
    *,
    tracer: torch.Tensor,
    side_len: int,
    side_name: str,
) -> torch.Tensor | None:
    if values is None:
        return None

    result = values.to(dtype=tracer.dtype, device=tracer.device)
    expected_shape = (*tracer.shape[:-2], side_len)

    if tracer.ndim >= 4 and tuple(result.shape) == expected_shape[1:]:
        result = result.unsqueeze(0)
    if tracer.ndim >= 4 and result.shape[0] == 1 and expected_shape[0] > 1:
        result = result.expand(*expected_shape)

    if tuple(result.shape) != expected_shape:
        raise ValueError(
            f"{side_name} tracer OBC shape {tuple(result.shape)} != expected {expected_shape}"
        )

    return result
