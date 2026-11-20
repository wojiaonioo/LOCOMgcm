"""Shared sponge-layer weight functions used by tracer, momentum, and barotropic sponge modules."""
from __future__ import annotations

import torch


#: Sponge weight shapes the tracer, momentum and barotropic sponge modules accept:
#: MITgcm obcs_sponge.F ("mitgcm") and a linear ramp across the band ("linear").
SPONGE_MODES = ("mitgcm", "linear")


def band_weights(width: int, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    return torch.arange(width, 0, -1, dtype=dtype, device=device) / float(width)


def mitgcm_boundary_weight(
    width: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    jsl = torch.arange(1, width + 1, dtype=dtype, device=device)
    return (float(width) - jsl) / float(width)


def mitgcm_inverse_relaxation_time(
    width: int,
    *,
    tau_boundary: float,
    tau_inner: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    boundary_weight = mitgcm_boundary_weight(width, dtype=dtype, device=device)
    tau = boundary_weight * tau_boundary + (1.0 - boundary_weight) * tau_inner
    return torch.where(tau != 0.0, 1.0 / tau, torch.zeros_like(tau))
