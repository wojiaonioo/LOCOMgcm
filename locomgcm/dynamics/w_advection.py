"""Advection of vertical momentum in MITgcm ``CALC_GW`` (``calc_gw.F``,
``mom_w_adv_*`` flux form).

``w`` is MIT's upward-positive ``wVel`` ``[B, nz, ny, nx]`` at the W levels of
tracer columns, level k being the top interface of tracer level k; it is used
as given, with MIT ``rkSign = -1`` in the vertical flux difference.  ``u`` and
``v`` are on U and V faces in m s-1.  Geometry: ``dr_f`` ``[1, nz, 1, 1]``,
``dy_g`` ``[1, 1, ny, nx+1]``, ``dx_g`` ``[1, 1, ny+1, nx]``, ``hfac_w``/``hfac_s``
with a leading singleton axis, ``r_a`` ``[1, 1, ny, nx]`` and the W-cell
``recip_volume`` ``[1, nz, ny, nx]`` (0 at dry W cells).  ``w_halo_x``/``w_halo_y``
are ``w[:, 1:]`` with a one-cell ZERO-filled centre halo along that axis; each
is exchanged only when an edge on that axis couples, otherwise None and
unread.  The result is the raw advective ``gW`` in m s-2, zero at level 0.
"""

from __future__ import annotations

import torch


def horizontal_w_advection(
    w: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    dr_f: torch.Tensor,
    dy_g: torch.Tensor,
    dx_g: torch.Tensor,
    hfac_w: torch.Tensor,
    hfac_s: torch.Tensor,
    r_a: torch.Tensor,
    recip_volume: torch.Tensor,
    w_halo_x: torch.Tensor | None,
    w_halo_y: torch.Tensor | None,
    west_couples: bool,
    east_couples: bool,
    south_couples: bool,
    north_couples: bool,
) -> torch.Tensor:
    """Flux-form advective ``gW`` below the surface level."""
    dtype = w.dtype
    device = w.device
    batch, nz, ny, nx = w.shape
    gW = torch.zeros_like(w)

    # Adjacent vertical fluxes replace the loop carry; horizontal halos contain
    # all active levels, so each axis exchanges once rather than nz-1 times.
    w_next = torch.cat((w[:, 1:], w[:, -1:] * 0.0), dim=1)
    tmp_wbar_z = 0.5 * (w + w_next)
    r_trans = 0.5 * (w + w_next) * r_a
    flx_dn = r_trans * tmp_wbar_z
    u_trans = 0.5 * (
        dr_f[:, :-1] * hfac_w[:, :-1] * u[:, :-1]
        + dr_f[:, 1:] * hfac_w[:, 1:] * u[:, 1:]
    ) * dy_g
    v_trans = 0.5 * (
        dr_f[:, :-1] * hfac_s[:, :-1] * v[:, :-1]
        + dr_f[:, 1:] * hfac_s[:, 1:] * v[:, 1:]
    ) * dx_g
    flx_ew = torch.zeros((batch, nz - 1, ny, nx + 1), dtype=dtype, device=device)
    flx_ns = torch.zeros((batch, nz - 1, ny + 1, nx), dtype=dtype, device=device)
    if nx > 1:
        flx_ew[..., 1:nx] = (
            u_trans[..., 1:nx]
            * 0.5
            * (w[:, 1:, :, 1:nx] + w[:, 1:, :, : nx - 1])
        )
    if west_couples:
        flx_ew[..., 0] = u_trans[..., 0] * 0.5 * (w_halo_x[..., 0] + w_halo_x[..., 1])
    if east_couples:
        flx_ew[..., nx] = u_trans[..., nx] * 0.5 * (w_halo_x[..., nx] + w_halo_x[..., nx + 1])
    if ny > 1:
        flx_ns[:, :, 1:ny, :] = (
            v_trans[:, :, 1:ny, :]
            * 0.5
            * (w[:, 1:, 1:ny, :] + w[:, 1:, : ny - 1, :])
        )
    if south_couples:
        flx_ns[:, :, 0, :] = v_trans[:, :, 0, :] * 0.5 * (w_halo_y[:, :, 0, :] + w_halo_y[:, :, 1, :])
    if north_couples:
        flx_ns[:, :, ny, :] = v_trans[:, :, ny, :] * 0.5 * (w_halo_y[:, :, ny, :] + w_halo_y[:, :, ny + 1, :])
    rk_sign = -1.0  # w is MIT upward-positive.
    div_flux = (
        flx_ew[..., 1:] - flx_ew[..., :-1]
        + flx_ns[:, :, 1:, :] - flx_ns[:, :, :-1, :]
        + (flx_dn[:, 1:] - flx_dn[:, :-1]) * rk_sign
    )
    gW[:, 1:] = -div_flux * recip_volume[:, 1:]
    return gW
