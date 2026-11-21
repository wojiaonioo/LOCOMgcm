from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_reference_tracer_profile(
    field: torch.Tensor | np.ndarray[Any, Any],
    *,
    dry_tolerance: float = 1.0e-10,
) -> list[float]:
    """Return the per-level wet-cell mean for a tracer field shaped ``(nz, ny, nx)``."""
    if isinstance(field, torch.Tensor):
        if field.ndim != 3:
            raise ValueError(f"expected rank-3 tracer field, got shape {tuple(field.shape)}")
        profile: list[float] = []
        for k in range(field.shape[0]):
            layer = field[k]
            wet = layer.abs() > dry_tolerance
            profile.append(float(layer[wet].mean().item()) if bool(wet.any()) else 0.0)
        return profile

    array = np.asarray(field)
    if array.ndim != 3:
        raise ValueError(f"expected rank-3 tracer field, got shape {array.shape}")

    profile = []
    for k in range(array.shape[0]):
        layer = array[k]
        wet = np.abs(layer) > dry_tolerance
        profile.append(float(layer[wet].mean()) if np.any(wet) else 0.0)
    return profile
