"""MITgcm TIMESTEP_WVEL port (non-hydrostatic, z-coordinates).

All ``w_vel_mit``/``gW`` inputs and outputs use MIT upward-positive ``wVel``.
LOCOMgcm diagnostic vertical velocity is downward-positive and must be
negated by the stepper before calling this module.
"""
from __future__ import annotations

import torch

from defgcm.dycore_parameters import NonhydrostaticParams
from defgcm.geometry import StructuredCGrid


def _center_mask(grid: StructuredCGrid, ref: torch.Tensor) -> torch.Tensor:
    mask = grid.masks.center_float(dtype=ref.dtype, device=ref.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(grid.nz, -1, -1)
    return mask.unsqueeze(0)


def timestep_wvel(
    w_vel_mit: torch.Tensor,
    gW_ab_weighted: torch.Tensor,
    phi_nh: torch.Tensor | None,
    grid: StructuredCGrid,
    dt_mom: float,
    params: NonhydrostaticParams,
    *,
    implicit_nh_press: float = 1.0,
    implicit_int_grav_wave: bool = False,
    pf_fac_mom: float = 1.0,
    delta_t_tracer_lev: torch.Tensor | None = None,
    dBdr_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step prognostic MIT-convention ``wVel`` using AB-weighted ``gW``."""
    del phi_nh, pf_fac_mom, delta_t_tracer_lev, dBdr_ref
    if w_vel_mit.ndim != 4:
        raise ValueError(f"w_vel_mit must be [batch,nz,ny,nx], got {tuple(w_vel_mit.shape)}")
    if gW_ab_weighted.shape != w_vel_mit.shape:
        raise ValueError("gW_ab_weighted must match w_vel_mit")
    if w_vel_mit.shape[1:] != (grid.nz, grid.ny, grid.nx):
        raise ValueError("w_vel_mit shape must match grid [nz,ny,nx]")
    if params.nh_am2 <= 0:
        raise ValueError(f"nh_am2 must be positive, got {params.nh_am2}")
    if implicit_nh_press != 1.0:
        raise NotImplementedError("implicitNHPress != 1 is not implemented")
    if implicit_int_grav_wave:
        raise NotImplementedError("implicitIntGravWave is not implemented")

    mask_c = _center_mask(grid, w_vel_mit)
    # Batch independent levels: per-level writes compile to a full-column
    # selection at every output cell. The top level still self-masks.
    mask_km1 = torch.cat((mask_c[:, :1], mask_c[:, :-1]), dim=1)
    gWtmp = gW_ab_weighted * mask_c * mask_km1

    gW_history_slot = w_vel_mit.clone()
    w_new = w_vel_mit + float(dt_mom) * float(params.nh_am2) * gWtmp
    return w_new * mask_c, gW_history_slot
