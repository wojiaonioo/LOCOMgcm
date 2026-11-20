"""Diagnose the interface velocity for regional momentum and continuity.

Interface w is MIT upward-positive [B,nz+1,ny,nx] in m/s. Diagnostics use
supplied geometry and masks; tensor dtype and device are retained.
"""
from __future__ import annotations

import torch

from defgcm.geometry import StructuredCGrid
from defgcm.state import PrognosticState


def diagnose_mit_interface_velocity(
    *, state: PrognosticState, grid: StructuredCGrid,
) -> torch.Tensor:
    from defgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

    _, w_iface_to_down, _ = diagnose_vertical_velocity(state, grid)
    return -w_iface_to_down
