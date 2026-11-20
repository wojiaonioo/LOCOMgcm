"""Viscous dissipation of vertical momentum in MITgcm ``CALC_GW``
(``calc_gw.F``): harmonic and biharmonic horizontal fluxes, the vertical
viscous flux and the ``MOM_W_SIDEDRAG`` no-slip side walls.

``w`` is MIT's upward-positive ``wVel`` ``[B, nz, ny, nx]`` at W levels, used
as given with MIT ``rkSign = -1``.  Viscosities: ``visc_h`` (``viscAh_W``) and
``visc4`` (``viscA4_W``) ``[B, nz, ny, nx]`` at W points; ``visc_loc``
``[B, nz-1, ny, nx]`` and ``visc_top`` ``[B, 1, ny, nx]`` from
:func:`calc_gw_vertical_viscosity`.  Geometry: ``dx_c`` ``[ny, nx+1]``,
``dy_c`` ``[ny+1, nx]``, ``dy_g`` ``[ny, nx+1]``, ``dx_g`` ``[ny+1, nx]``,
``x_area`` ``[nz, ny, nx+1]`` and ``y_area`` ``[nz, ny+1, nx]`` (W-face
thickness times width), ``dr_f`` ``[nz]``, ``r_a`` ``[1, ny, nx]`` and the
W-cell ``recip_volume`` ``[1, nz, ny, nx]`` (0 at dry W cells).  Fluxes are
``[B, nz-1, ...]`` for levels 1..nz-1.  Arguments named ``*_halo_x`` or
``*_halo_y`` carry a one-cell ZERO-filled centre halo of levels 1..nz-1 along
that axis; each is exchanged only when an edge on that axis couples,
otherwise None and unread.  ``gw_diss`` is in m s-2, zero at level 0.
"""

from __future__ import annotations

import torch


def calc_gw_vertical_viscosity(
    kappa_ru: torch.Tensor, kappa_rv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """CALC_GW's vertical viscosity at tracer centres (calc_gw.F:407-411, :427-429).

    ``kappa_ru`` / ``kappa_rv`` are ``kappaRU`` / ``kappaRV`` on U / V faces at
    the interfaces, index 0 being the surface (Fortran k = 1) and index nz the
    bottom (Nr+1).  Returns ``(visc_loc, visc_top)``:

    * ``visc_loc[:, k]`` is the viscosity of the flux through the centre of
      tracer level k: the mean of the two U faces and two V faces of the column
      at the level's top and bottom interfaces, summed in the Fortran order;
    * ``visc_top`` is the k=2 upper face (centre of level 1): the horizontal mean
      at the level's bottom interface only; MITgcm does not interpolate
      vertically there because a mixing scheme need not define the viscosity
      at the surface interface (calc_gw.F:423-426).
    """
    ru_w, ru_e = kappa_ru[..., :-1], kappa_ru[..., 1:]
    rv_s, rv_n = kappa_rv[..., :-1, :], kappa_rv[..., 1:, :]
    visc_loc = (
        ru_w[:, :-1] + ru_e[:, :-1] + ru_w[:, 1:] + ru_e[:, 1:]
        + rv_s[:, :-1] + rv_n[:, :-1] + rv_s[:, 1:] + rv_n[:, 1:]
    ) * 0.125
    visc_top = (ru_w[:, 1:2] + ru_e[:, 1:2] + rv_s[:, 1:2] + rv_n[:, 1:2]) * 0.25
    return visc_loc, visc_top


def harmonic_w_fluxes(
    w: torch.Tensor,
    visc_h: torch.Tensor,
    *,
    dx_c: torch.Tensor,
    dy_c: torch.Tensor,
    x_area: torch.Tensor,
    y_area: torch.Tensor,
    w_halo_x: torch.Tensor | None,
    visc_halo_x: torch.Tensor | None,
    w_halo_y: torch.Tensor | None,
    visc_halo_y: torch.Tensor | None,
    west_couples: bool,
    east_couples: bool,
    south_couples: bool,
    north_couples: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Harmonic viscous W fluxes ``-viscAh_W * dw/dx * area`` on W and S faces."""
    dtype = w.dtype
    device = w.device
    batch, nz, ny, nx = w.shape
    # Batch all vertical levels through each halo exchange and local stencil.
    flx_ew = torch.zeros((batch, nz - 1, ny, nx + 1), dtype=dtype, device=device)
    flx_ns = torch.zeros((batch, nz - 1, ny + 1, nx), dtype=dtype, device=device)
    if nx > 1:
        visc_face = 0.5 * (visc_h[:, 1:, :, 1:nx] + visc_h[:, 1:, :, : nx - 1])
        flx_ew[:, :, :, 1:nx] = (
            -visc_face
            * (w[:, 1:, :, 1:nx] - w[:, 1:, :, : nx - 1])
            / dx_c[:, 1:nx].unsqueeze(0)
            * x_area[1:, :, 1:nx].unsqueeze(0)
        )
    if west_couples:
        # expects MIT upward-positive wVel on a center-cell halo
        visc_face = 0.5 * (visc_halo_x[:, :, :, 0] + visc_halo_x[:, :, :, 1])
        flx_ew[:, :, :, 0] = (
            -visc_face
            * (w_halo_x[:, :, :, 1] - w_halo_x[:, :, :, 0])
            / dx_c[:, 0].unsqueeze(0)
            * x_area[1:, :, 0].unsqueeze(0)
        )
    if east_couples:
        # expects MIT upward-positive wVel on a center-cell halo
        visc_face = 0.5 * (visc_halo_x[:, :, :, nx] + visc_halo_x[:, :, :, nx + 1])
        flx_ew[:, :, :, nx] = (
            -visc_face
            * (w_halo_x[:, :, :, nx + 1] - w_halo_x[:, :, :, nx])
            / dx_c[:, nx].unsqueeze(0)
            * x_area[1:, :, nx].unsqueeze(0)
        )
    if ny > 1:
        visc_face = 0.5 * (visc_h[:, 1:, 1:ny, :] + visc_h[:, 1:, : ny - 1, :])
        flx_ns[:, :, 1:ny, :] = (
            -visc_face
            * (w[:, 1:, 1:ny, :] - w[:, 1:, : ny - 1, :])
            / dy_c[1:ny, :]
            * y_area[1:, 1:ny, :].unsqueeze(0)
        )
    if south_couples:
        # expects MIT upward-positive wVel on a center-cell halo
        visc_face = 0.5 * (visc_halo_y[:, :, 0, :] + visc_halo_y[:, :, 1, :])
        flx_ns[:, :, 0, :] = (
            -visc_face
            * (w_halo_y[:, :, 1, :] - w_halo_y[:, :, 0, :])
            / dy_c[0, :]
            * y_area[1:, 0, :].unsqueeze(0)
        )
    if north_couples:
        # expects MIT upward-positive wVel on a center-cell halo
        visc_face = 0.5 * (visc_halo_y[:, :, ny, :] + visc_halo_y[:, :, ny + 1, :])
        flx_ns[:, :, ny, :] = (
            -visc_face
            * (w_halo_y[:, :, ny + 1, :] - w_halo_y[:, :, ny, :])
            / dy_c[ny, :]
            * y_area[1:, ny, :].unsqueeze(0)
        )
    return flx_ew, flx_ns


def w_laplacian(
    w: torch.Tensor,
    *,
    dx_c: torch.Tensor,
    dy_c: torch.Tensor,
    x_area: torch.Tensor,
    y_area: torch.Tensor,
    recip_volume: torch.Tensor,
    w_halo_x: torch.Tensor | None,
    w_halo_y: torch.Tensor | None,
    west_couples: bool,
    east_couples: bool,
    south_couples: bool,
    north_couples: bool,
) -> torch.Tensor:
    """``del2w`` of levels 1..nz-1, the horizontal Laplacian CALC_GW feeds to viscA4_W.

    MIT CALC_GW horizontal biharmonic dissipation of w
    (calc_gw.F:299-392, gated biharmonicVisc=viscA4W!=0, k>1).
    del2w = horizontal Laplacian of w (unit-viscosity gradient-flux
    divergence, same volume normalisation as gw_diss).
    """
    dtype = w.dtype
    device = w.device
    batch, nz, ny, nx = w.shape
    lap_ew = torch.zeros((batch, nz - 1, ny, nx + 1), dtype=dtype, device=device)
    lap_ns = torch.zeros((batch, nz - 1, ny + 1, nx), dtype=dtype, device=device)
    if nx > 1:
        lap_ew[:, :, :, 1:nx] = (
            (w[:, 1:, :, 1:nx] - w[:, 1:, :, : nx - 1])
            / dx_c[:, 1:nx].unsqueeze(0)
            * x_area[1:, :, 1:nx].unsqueeze(0)
        )
    if west_couples:
        # expects MIT upward-positive wVel on a center-cell halo
        lap_ew[:, :, :, 0] = (
            (w_halo_x[:, :, :, 1] - w_halo_x[:, :, :, 0])
            / dx_c[:, 0].unsqueeze(0)
            * x_area[1:, :, 0].unsqueeze(0)
        )
    if east_couples:
        # expects MIT upward-positive wVel on a center-cell halo
        lap_ew[:, :, :, nx] = (
            (w_halo_x[:, :, :, nx + 1] - w_halo_x[:, :, :, nx])
            / dx_c[:, nx].unsqueeze(0)
            * x_area[1:, :, nx].unsqueeze(0)
        )
    if ny > 1:
        lap_ns[:, :, 1:ny, :] = (
            (w[:, 1:, 1:ny, :] - w[:, 1:, : ny - 1, :])
            / dy_c[1:ny, :]
            * y_area[1:, 1:ny, :].unsqueeze(0)
        )
    if south_couples:
        # expects MIT upward-positive wVel on a center-cell halo
        lap_ns[:, :, 0, :] = (
            (w_halo_y[:, :, 1, :] - w_halo_y[:, :, 0, :])
            / dy_c[0, :]
            * y_area[1:, 0, :].unsqueeze(0)
        )
    if north_couples:
        # expects MIT upward-positive wVel on a center-cell halo
        lap_ns[:, :, ny, :] = (
            (w_halo_y[:, :, ny + 1, :] - w_halo_y[:, :, ny, :])
            / dy_c[ny, :]
            * y_area[1:, ny, :].unsqueeze(0)
        )
    return (
        lap_ew[:, :, :, 1:] - lap_ew[:, :, :, :-1]
        + lap_ns[:, :, 1:, :] - lap_ns[:, :, :-1, :]
    ) * recip_volume[:, 1:]


def add_biharmonic_w_fluxes(
    flx_ew: torch.Tensor,
    flx_ns: torch.Tensor,
    del2w: torch.Tensor,
    visc4: torch.Tensor,
    *,
    dx_c: torch.Tensor,
    dy_c: torch.Tensor,
    x_area: torch.Tensor,
    y_area: torch.Tensor,
    del2w_halo_x: torch.Tensor | None,
    visc4_halo_x: torch.Tensor | None,
    del2w_halo_y: torch.Tensor | None,
    visc4_halo_y: torch.Tensor | None,
    west_couples: bool,
    east_couples: bool,
    south_couples: bool,
    north_couples: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add ``+viscA4_W * d(del2w)/dx * area`` to the fluxes, in place, and return them."""
    ny, nx = del2w.shape[-2:]
    # Viscous biharmonic flux: +viscA4_W * d(del2w)/dx * area (calc_gw.F:371,391).
    if nx > 1:
        visc4_face = 0.5 * (visc4[:, 1:, :, 1:nx] + visc4[:, 1:, :, : nx - 1])
        flx_ew[:, :, :, 1:nx] = flx_ew[:, :, :, 1:nx] + (
            visc4_face
            * (del2w[:, :, :, 1:nx] - del2w[:, :, :, : nx - 1])
            / dx_c[:, 1:nx].unsqueeze(0)
            * x_area[1:, :, 1:nx].unsqueeze(0)
        )
    if west_couples:
        visc4_face = 0.5 * (visc4_halo_x[:, :, :, 0] + visc4_halo_x[:, :, :, 1])
        flx_ew[:, :, :, 0] = flx_ew[:, :, :, 0] + (
            visc4_face
            * (del2w_halo_x[:, :, :, 1] - del2w_halo_x[:, :, :, 0])
            / dx_c[:, 0].unsqueeze(0)
            * x_area[1:, :, 0].unsqueeze(0)
        )
    if east_couples:
        visc4_face = 0.5 * (visc4_halo_x[:, :, :, nx] + visc4_halo_x[:, :, :, nx + 1])
        flx_ew[:, :, :, nx] = flx_ew[:, :, :, nx] + (
            visc4_face
            * (del2w_halo_x[:, :, :, nx + 1] - del2w_halo_x[:, :, :, nx])
            / dx_c[:, nx].unsqueeze(0)
            * x_area[1:, :, nx].unsqueeze(0)
        )
    if ny > 1:
        visc4_face = 0.5 * (visc4[:, 1:, 1:ny, :] + visc4[:, 1:, : ny - 1, :])
        flx_ns[:, :, 1:ny, :] = flx_ns[:, :, 1:ny, :] + (
            visc4_face
            * (del2w[:, :, 1:ny, :] - del2w[:, :, : ny - 1, :])
            / dy_c[1:ny, :]
            * y_area[1:, 1:ny, :].unsqueeze(0)
        )
    if south_couples:
        visc4_face = 0.5 * (visc4_halo_y[:, :, 0, :] + visc4_halo_y[:, :, 1, :])
        flx_ns[:, :, 0, :] = flx_ns[:, :, 0, :] + (
            visc4_face
            * (del2w_halo_y[:, :, 1, :] - del2w_halo_y[:, :, 0, :])
            / dy_c[0, :]
            * y_area[1:, 0, :].unsqueeze(0)
        )
    if north_couples:
        visc4_face = 0.5 * (visc4_halo_y[:, :, ny, :] + visc4_halo_y[:, :, ny + 1, :])
        flx_ns[:, :, ny, :] = flx_ns[:, :, ny, :] + (
            visc4_face
            * (del2w_halo_y[:, :, ny + 1, :] - del2w_halo_y[:, :, ny, :])
            / dy_c[ny, :]
            * y_area[1:, ny, :].unsqueeze(0)
        )
    return flx_ew, flx_ns


def viscous_w_tendency(
    w: torch.Tensor,
    flx_ew: torch.Tensor,
    flx_ns: torch.Tensor,
    *,
    visc_loc: torch.Tensor,
    visc_top: torch.Tensor,
    dr_f: torch.Tensor,
    r_a: torch.Tensor,
    recip_volume: torch.Tensor,
) -> torch.Tensor:
    """``gw_diss``: the divergence of the horizontal and vertical viscous W fluxes."""
    nz = w.shape[1]
    gw_diss = torch.zeros_like(w)
    rk_sign = -1.0  # MIT rkSign=-1; do not flip the MIT-upward wVel input.
    w_next = torch.cat((w[:, 2:], w[:, -1:] * 0.0), dim=1)
    flx_dn = (
        -visc_loc[:, 1:]
        * (w_next - w[:, 1:])
        * rk_sign
        / dr_f[1:].view(1, nz - 1, 1, 1)
        * r_a
    )
    flx_top = (
        -visc_top
        * (w[:, 1:2] - w[:, :1])
        * rk_sign
        / dr_f[:1].view(1, 1, 1, 1)
        * r_a
    )
    flx_dis_up = torch.cat((flx_top, flx_dn[:, :-1]), dim=1)
    div_flux = (
        flx_ew[:, :, :, 1:] - flx_ew[:, :, :, :-1]
        + flx_ns[:, :, 1:, :] - flx_ns[:, :, :-1, :]
        + (flx_dn - flx_dis_up) * rk_sign
    )
    gw_diss[:, 1:] = -div_flux * recip_volume[:, 1:]
    return gw_diss


def harmonic_wall_stress(w: torch.Tensor, visc_h: torch.Tensor) -> torch.Tensor:
    """Side-wall stress ``viscAh_W * w`` of levels 1..nz-1."""
    return visc_h[:, 1:] * w[:, 1:]


def biharmonic_wall_stress(
    w: torch.Tensor,
    visc_h: torch.Tensor,
    *,
    visc4: torch.Tensor,
    del2w: torch.Tensor,
) -> torch.Tensor:
    """Side-wall stress ``viscAh_W * w - viscA4_W * del2w`` of levels 1..nz-1."""
    return harmonic_wall_stress(w, visc_h) - visc4[:, 1:] * del2w


def w_side_drag(
    gw_diss: torch.Tensor,
    *,
    wall_stress: torch.Tensor,
    thickness: torch.Tensor,
    thickness_x: torch.Tensor,
    thickness_y: torch.Tensor,
    dx_c: torch.Tensor,
    dy_c: torch.Tensor,
    dy_g: torch.Tensor,
    dx_g: torch.Tensor,
    recip_volume: torch.Tensor,
    side_drag_factor: float,
    zero_west: bool,
    zero_east: bool,
    zero_south: bool,
    zero_north: bool,
) -> torch.Tensor:
    """Subtract MOM_W_SIDEDRAG from ``gw_diss`` in place and return it.

    ``thickness`` is the unmasked ``rThickC_C`` of levels 1..nz-1 and
    ``thickness_x``/``thickness_y`` it with a ZERO-filled one-cell halo.
    ``zero_*`` names an exterior edge whose ZERO fill does not describe a
    solid wall, so its height difference is dropped.  ``wall_stress`` comes
    from :func:`harmonic_wall_stress` or :func:`biharmonic_wall_stress`.
    """
    # MOM_W_SIDEDRAG: every topographic step exposes a lateral wall.
    # Use unmasked rThickC_C, including wet half-cells above a dry lower
    # tracer cell. The receiving W cell is gated by recip_volume.
    height_w = (thickness - thickness_x[..., :-2]).clamp_min(0.0)
    height_e = (thickness - thickness_x[..., 2:]).clamp_min(0.0)
    height_s = (thickness - thickness_y[..., :-2, :]).clamp_min(0.0)
    height_n = (thickness - thickness_y[..., 2:, :]).clamp_min(0.0)
    if zero_west:
        height_w[..., :1] = 0.0
    if zero_east:
        height_e[..., -1:] = 0.0
    if zero_south:
        height_s[..., :1, :] = 0.0
    if zero_north:
        height_n[..., -1:, :] = 0.0
    wall_geometry = (
        height_w * dy_g[..., :-1] / dx_c[..., :-1]
        + height_e * dy_g[..., 1:] / dx_c[..., 1:]
        + height_s * dx_g[:-1, :] / dy_c[:-1, :]
        + height_n * dx_g[1:, :] / dy_c[1:, :]
    )
    # cosPower scaling is outside the supported W-viscosity contract.
    # For cosFac=1, all four walls consume the same Ah*w - A4*del2w.
    gw_diss[:, 1:] = gw_diss[:, 1:] - (
        side_drag_factor * wall_geometry.unsqueeze(0) * wall_stress
        * recip_volume[:, 1:]
    )
    return gw_diss
