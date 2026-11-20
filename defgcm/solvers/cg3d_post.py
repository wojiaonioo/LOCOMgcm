"""POST_CG3D helpers for nonhydrostatic pressure separation."""
from __future__ import annotations

import torch

from defgcm.dycore_parameters import NonhydrostaticParams
from defgcm.geometry import StructuredCGrid


def post_cg3d_separate_dphinh(
    phi_nh_solution: torch.Tensor,
    grid: StructuredCGrid,
    params: NonhydrostaticParams,
    *,
    exact_conserv: bool = True,
) -> torch.Tensor:
    """Return the non-hydrostatic ``dPhiNH`` from the surface wet level.

    ``phi_nh_solution`` is a pressure-potential tensor from CG3D, not a
    vertical velocity field; no w/wVel sign convention applies here.
    """
    if not params.nonhydrostatic or not exact_conserv:
        return torch.zeros(
            phi_nh_solution.shape[0],
            grid.ny,
            grid.nx,
            dtype=phi_nh_solution.dtype,
            device=phi_nh_solution.device,
        )
    if phi_nh_solution.ndim != 4 or phi_nh_solution.shape[1:] != (
        grid.nz,
        grid.ny,
        grid.nx,
    ):
        raise ValueError(
            "phi_nh_solution must be [batch,nz,ny,nx] matching grid, "
            f"got {tuple(phi_nh_solution.shape)}"
        )
    dphi = phi_nh_solution[:, 0].clone()
    if grid.masks.has_3d_masks:
        mask = grid.masks.center_float(
            dtype=dphi.dtype,
            device=dphi.device,
        )[0].unsqueeze(0)
        dphi = dphi * mask
    return dphi
