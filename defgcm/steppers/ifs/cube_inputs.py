"""Current-step cube surface forcing, replaced as one immutable record.

Fields retain the producer dtype and global compact [6, ny, nx] layout.
Wind stress follows the face-local x/y axes in N/m2; Qnet and Qsw are MIT
upward-positive W/m2, and EmPmR is positive evaporation in m/s. Consumers
place face stress and apply wet masks; this record performs no arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from defgcm.dynamics.surface_flux import SurfaceForcing


@dataclass(frozen=True)
class CubeSurfaceForcing:
    tau_x: torch.Tensor | None
    tau_y: torch.Tensor | None
    qnet: torch.Tensor | None
    qsw: torch.Tensor | None
    empmr: torch.Tensor | None
    penetrating_sw: bool | None
    rho_const_fresh: float | None
    bulk: SurfaceForcing | None


EMPTY_CUBE_SURFACE_FORCING = CubeSurfaceForcing(
    tau_x=None,
    tau_y=None,
    qnet=None,
    qsw=None,
    empmr=None,
    penetrating_sw=None,
    rho_const_fresh=None,
    bulk=None,
)
