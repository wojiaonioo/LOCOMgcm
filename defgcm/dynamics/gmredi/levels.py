"""Vertical neighbours shared by the GM/Redi kernels.

MITgcm's ``km1 = MAX(k-1,1)``, ``kp1 = MIN(k+1,Nr)``, ``maskp1`` (0 at
``k = Nr``) and ``maskUp`` (``model/src/calc_adv_flow.F:112-130``) along
dim -3 of ``[Nr, ...]`` tensors.
"""
from __future__ import annotations

import torch


def previous_level(field: torch.Tensor) -> torch.Tensor:
    """``field(km1)`` with ``km1 = MAX(k-1, 1)``."""
    return torch.cat((field[:1], field[:-1]), dim=0)


def next_level(field: torch.Tensor) -> torch.Tensor:
    """``field(kp1)`` with ``kp1 = MIN(Nr, k+1)``."""
    return torch.cat((field[1:], field[-1:]), dim=0)


def below_bottom_mask(field: torch.Tensor) -> torch.Tensor:
    """``maskp1`` as an ``[Nr, 1, 1]`` column: 1 for ``k < Nr``, 0 at ``k = Nr``."""
    mask = torch.ones((field.shape[0], 1, 1), dtype=field.dtype, device=field.device)
    mask[-1] = 0.0
    return mask


def mask_up(maskC: torch.Tensor) -> torch.Tensor:
    """``maskUp = maskC(k-1)*maskC(k)``, 0 at the surface interface (``calc_adv_flow.F:112-130``)."""
    return torch.cat((torch.zeros_like(maskC[:1]), maskC[:-1] * maskC[1:]), dim=0)
