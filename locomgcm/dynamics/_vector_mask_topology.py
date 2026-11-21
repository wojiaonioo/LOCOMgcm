"""Topology helpers for vector tendency masks."""

from __future__ import annotations

import torch


def mask_vector_tendency_for_grid(
    tend_u: torch.Tensor,
    tend_v: torch.Tensor,
    grid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply face-first U/V wet masks when they match the tendency shape."""
    masks = grid.masks
    if masks.u is not None and masks.u.shape == tend_u.shape:
        u_mask = (masks.u > 0).to(dtype=tend_u.dtype, device=tend_u.device)
        tend_u = tend_u * u_mask
    if masks.v is not None and masks.v.shape == tend_v.shape:
        v_mask = (masks.v > 0).to(dtype=tend_v.dtype, device=tend_v.device)
        tend_v = tend_v * v_mask
    return tend_u, tend_v
