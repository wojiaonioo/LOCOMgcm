"""Vertical flux-form momentum advection kernels (MITgcm ``mom_u_adv_wu.F``,
``mom_v_adv_wv.F``, ``mom_fluxform.F``).

``w`` is the LOCOMgcm downward-positive interface vertical velocity
``[B, nz+1, ny, nx]`` at tracer columns, interface 0 at the surface; the adapter
negates MIT's upward-positive ``w_interface`` once before calling here.  The
face velocities are the vertical transport ``w * rA`` averaged to the face and
divided by the face's cell area (``mom_calc_rtrans.F:95-102``, ``recip_rAw``
and ``recip_rAs`` of ``mom_fluxform.F``).  ``u``
is on U faces ``[B, nz, ny, nx+1]`` and ``v`` on V faces ``[B, nz, ny+1, nx]``.
Interface fluxes are ``[B, nz+1, ...]`` on the same faces, index 0 the surface
and index nz the bottom, where the flux is zero.  ``center_wet`` is the
``[1, nz, ny, nx]`` wet indicator of tracer cells (1 wet, 0 dry).  Arguments
named ``*_x`` or ``*_y`` carry the one-cell halo the adapter exchanged along
that axis.  Face velocities are cast to the momentum dtype.
"""

from __future__ import annotations

import torch


def u_face_vertical_velocity(
    r_trans_x: torch.Tensor,
    *,
    face_area: torch.Tensor,
    nx: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Transport averaged to each U face and divided by the U-cell area ``rAw``."""
    RU = 0.5 * (r_trans_x[..., : nx + 1] + r_trans_x[..., 1 : nx + 2])
    return torch.where(face_area > 0, RU / face_area, torch.zeros_like(RU)).to(dtype=dtype)


def vertical_transport(w: torch.Tensor, *, cell_area: torch.Tensor) -> torch.Tensor:
    """Vertical volume transport ``rTrans = w * rA`` at tracer columns."""
    return w * cell_area.unsqueeze(0).unsqueeze(0)


def v_face_vertical_velocity(
    r_trans_y: torch.Tensor,
    *,
    face_area: torch.Tensor,
    ny: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Transport averaged to each V face and divided by the V-cell area ``rAs``."""
    RV = 0.5 * (r_trans_y[..., : ny + 1, :] + r_trans_y[..., 1 : ny + 2, :])
    return torch.where(face_area > 0, RV / face_area, torch.zeros_like(RV)).to(dtype=dtype)


def interface_flux(w_face: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """``w_face * velocity`` at interfaces, centred inside, one-sided at the top.

    The surface interface takes the top-level velocity; the bottom interface
    flux is zero.
    """
    nz = velocity.shape[-3]
    velocity_if = torch.zeros(
        (*velocity.shape[:-3], nz + 1, *velocity.shape[-2:]),
        dtype=velocity.dtype, device=velocity.device,
    )
    velocity_if[:, 0, :, :] = velocity[:, 0, :, :]
    if nz > 1:
        velocity_if[:, 1:nz, :, :] = 0.5 * (velocity[:, :-1, :, :] + velocity[:, 1:, :, :])
    velocity_if[:, nz, :, :] = velocity[:, -1, :, :]

    flux = w_face * velocity_if
    flux[:, nz, :, :] = 0.0

    # MITgcm default (without MOM_BOUNDARY_CONSERVE) does NOT zero
    # flux at wet/dry interfaces — it relies on wVel and uFld being
    # masked at dry cells.  Matching that behavior here avoids a
    # systematic bottom-step error at bathymetric transitions where
    # rTransU can be nonzero even when one side is dry.
    return flux


def surface_correction_source(
    r_trans: torch.Tensor, center_wet: torch.Tensor, *, nz: int
) -> torch.Tensor:
    """Transport ``w * rA`` on interior interfaces whose upper cell is dry and lower cell wet."""
    mask_jump = torch.clamp(center_wet[:, 1:, :, :] - center_wet[:, :-1, :, :], min=0.0)
    return r_trans[:, 1:nz, :, :] * mask_jump


def u_surface_correction(
    u: torch.Tensor,
    q_x: torch.Tensor,
    lower_u_faces: torch.Tensor,
    *,
    face_area: torch.Tensor,
    nx: int,
) -> torch.Tensor:
    """``mom_u_adv_wu.F:96-107`` linear free-surface correction on interior interfaces,
    divided by ``rAw`` like the advective flux."""
    batch, nz, ny, nxp1 = u.shape
    correction = u.new_zeros(batch, nz + 1, ny, nxp1)
    q_face = 0.25 * (q_x[:, :, :, : nx + 1] + q_x[:, :, :, 1 : nx + 2])
    correction[:, 1:nz, :, :] = (
        torch.where(face_area > 0, q_face / face_area, torch.zeros_like(q_face)) * lower_u_faces
    )
    return correction


def v_surface_correction(
    v: torch.Tensor,
    q_y: torch.Tensor,
    lower_v_faces: torch.Tensor,
    *,
    face_area: torch.Tensor,
    ny: int,
) -> torch.Tensor:
    """``mom_v_adv_wv.F:94-109`` linear free-surface correction on interior interfaces,
    divided by ``rAs`` like the advective flux."""
    batch, nz, nyp1, nx = v.shape
    correction = v.new_zeros(batch, nz + 1, nyp1, nx)
    q_face = 0.25 * (q_y[:, :, : ny + 1, :] + q_y[:, :, 1 : ny + 2, :])
    correction[:, 1:nz, :, :] = (
        torch.where(face_area > 0, q_face / face_area, torch.zeros_like(q_face)) * lower_v_faces
    )
    return correction


def vertical_flux_divergence(
    flux: torch.Tensor, *, recip_thickness: torch.Tensor
) -> torch.Tensor:
    """``-(F[k+1] - F[k]) * recip_thickness`` at the levels between interfaces."""
    return -(flux[:, 1:, :, :] - flux[:, :-1, :, :]) * recip_thickness
