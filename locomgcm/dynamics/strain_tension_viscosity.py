"""MITgcm MOM_HDISSIP strain-tension horizontal viscosity stencil (useStrainTensionVisc).

Kernels only.  ``tension`` is the centre field ``[B, nz, ny, nx]`` of
MOM_CALC_TENSION (times ``maskInC`` under OBCS), ``strain`` the corner field
``[B, nz, ny+1, nx+1]`` of MOM_CALC_STRAIN, zeroed by the caller's corner
``hFacZ`` before use (mom_vecinv.F:335-343).  ``viscAh_D`` multiplies the centre
(tension) fluxes and ``viscAh_Z`` the corner (strain) fluxes (mom_vecinv.F:421-427).
Metrics are the grid's 2-D tensors.  Outputs are U-face ``[B, nz, ny, nx+1]`` and
V-face ``[B, nz, ny+1, nx]`` tendencies in m/s^2, without hFac weighting or wet
masks: MOM_HDISSIP applies neither, and the predictor's wet mask clears dry faces.
The centre fluxes reach the kernels with one halo column (U) or row (V) each side,
exchanged by the caller.  No kernel reads the grid.
"""
from __future__ import annotations

import torch


def _strain_tension_center_flux(
    tension: torch.Tensor,
    visc_d: torch.Tensor,
    *,
    length: torch.Tensor,
) -> torch.Tensor:
    """``length^2 viscAh_D tension`` at C points (``dyF`` for U, ``dxF`` for V)."""
    return ((length * length) * visc_d) * tension


def _strain_tension_corner_flux(
    strain: torch.Tensor,
    visc_z: torch.Tensor,
    *,
    length: torch.Tensor,
) -> torch.Tensor:
    """``length^2 viscAh_Z strain`` at Z points (``dxV`` for U, ``dyU`` for V)."""
    return ((length * length) * visc_z) * strain


def _u_strain_tension_dissipation(
    center_flux_x: torch.Tensor,
    corner_flux: torch.Tensor,
    *,
    recip_dyG: torch.Tensor,
    recip_dxC: torch.Tensor,
) -> torch.Tensor:
    """mom_hdissip.F:45-61 on every U face.

    ``center_flux_x`` ``[..., ny, nx+2]`` carries one x-halo column each side;
    ``corner_flux`` is ``[..., ny+1, nx+1]``.
    """
    return (
        ((recip_dyG * recip_dyG) * recip_dxC)
        * (center_flux_x[..., 1:] - center_flux_x[..., :-1])
        + ((recip_dxC * recip_dxC) * recip_dyG)
        * (corner_flux[..., 1:, :] - corner_flux[..., :-1, :])
    )


def _v_strain_tension_dissipation(
    corner_flux: torch.Tensor,
    center_flux_y: torch.Tensor,
    *,
    recip_dyC: torch.Tensor,
    recip_dxG: torch.Tensor,
) -> torch.Tensor:
    """mom_hdissip.F:63-79 on every V face.

    ``center_flux_y`` ``[..., ny+2, nx]`` carries one y-halo row each side;
    ``corner_flux`` is ``[..., ny+1, nx+1]``.
    """
    return (
        ((recip_dyC * recip_dyC) * recip_dxG)
        * (corner_flux[..., 1:] - corner_flux[..., :-1])
        - ((recip_dxG * recip_dxG) * recip_dyC)
        * (center_flux_y[..., 1:, :] - center_flux_y[..., :-1, :])
    )
