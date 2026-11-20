"""Horizontal flux-form momentum advection kernels (MITgcm ``mom_fluxform.F``).

``u`` sits on U faces ``[B, nz, ny, nx+1]`` and ``v`` on V faces
``[B, nz, ny+1, nx]``, in m s-1.  Transports are volume fluxes in m3 s-1
(velocity times face width, ``drF`` and ``hFac``).  The self fluxes ``Fuu`` and
``Fvv`` sit at tracer centres ``[..., ny, nx]``; the cross fluxes ``Fvu`` and
``Fuv`` sit at corners ``[..., ny+1, nx+1]``.  Arguments named ``*_x`` or
``*_y`` carry the one-cell halo the adapter exchanged along that axis.  No mask
is applied here except the corner support mask the caller passes; the
reciprocal volume carries the dry-face zeros.  Results keep the input dtype.
"""

from __future__ import annotations

import torch


def face_transport(
    velocity: torch.Tensor,
    *,
    face_width: torch.Tensor,
    layer_thickness: torch.Tensor,
    face_hfac: torch.Tensor,
) -> torch.Tensor:
    """Volume transport ``velocity * width * drF * hFac`` through a U or V face."""
    return velocity * face_width * layer_thickness * face_hfac


def u_self_flux(u_trans: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``Fuu(i) = 0.25 * (uTrans(i) + uTrans(i+1)) * (u(i) + u(i+1))`` at centres."""
    return 0.25 * (
        (u_trans[..., :-1] + u_trans[..., 1:])
        * (u[..., :-1] + u[..., 1:])
    )


def v_self_flux(v_trans: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``Fvv(j) = 0.25 * (vTrans(j) + vTrans(j+1)) * (v(j) + v(j+1))`` at centres."""
    return 0.25 * (
        (v_trans[..., :-1, :] + v_trans[..., 1:, :])
        * (v[..., :-1, :] + v[..., 1:, :])
    )


def corner_mean_x(field_x: torch.Tensor, *, ny: int, nx: int) -> torch.Tensor:
    """Average of the two x-neighbours of every corner ``j in [0..ny], i in [0..nx]``."""
    return 0.5 * (
        field_x[..., : ny + 1, : nx + 1]
        + field_x[..., : ny + 1, 1 : nx + 2]
    )


def corner_mean_y(field_y: torch.Tensor, *, ny: int, nx: int) -> torch.Tensor:
    """Average of the two y-neighbours of every corner ``j in [0..ny], i in [0..nx]``."""
    return 0.5 * (
        field_y[..., : ny + 1, : nx + 1]
        + field_y[..., 1 : ny + 2, : nx + 1]
    )


def corner_cross_flux(
    transport_corner: torch.Tensor,
    velocity_corner: torch.Tensor,
    *,
    corner_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross flux at corners, ``+0.0`` where ``corner_mask`` is false."""
    corner_flux = transport_corner * velocity_corner
    return torch.where(corner_mask, corner_flux, torch.zeros_like(corner_flux))


def u_flux_divergence(
    fuu_x: torch.Tensor,
    fvu: torch.Tensor,
    *,
    recip_volume: torch.Tensor,
    nx: int,
) -> torch.Tensor:
    """``-(dFuu/dx + dFvu/dy) * recip_volume`` at U faces."""
    dFuu_dx = fuu_x[..., 1 : nx + 2] - fuu_x[..., : nx + 1]
    dFvu_dy = fvu[..., 1:, :] - fvu[..., :-1, :]
    raw_div = dFuu_dx + dFvu_dy
    return -raw_div * recip_volume


def v_flux_divergence(
    fvv_y: torch.Tensor,
    fuv: torch.Tensor,
    *,
    recip_volume: torch.Tensor,
    ny: int,
) -> torch.Tensor:
    """``-(dFuv/dx + dFvv/dy) * recip_volume`` at V faces."""
    dFvv_dy = fvv_y[..., 1 : ny + 2, :] - fvv_y[..., : ny + 1, :]
    dFuv_dx = fuv[..., 1:] - fuv[..., :-1]
    raw_div = dFuv_dx + dFvv_dy
    return -raw_div * recip_volume
