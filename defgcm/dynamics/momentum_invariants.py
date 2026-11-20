"""Velocity invariants of the C-grid momentum: divergence, vorticity, strain, tension.

Kernels only: they take face velocities ``u`` ``[B, nz, ny, nx+1]`` and ``v``
``[B, nz, ny+1, nx]`` (or their haloed copies), metrics already shaped to
broadcast against them, and wet fractions passed in.  Divergence and tension sit
at cell centres ``[B, nz, ny, nx]``; vorticity and strain at corners
``[B, nz, ny+1, nx+1]``.  The vorticity is MITgcm's ``vort3`` sign (curl of v
minus curl of u over ``rAz``), in s^-1.  Optional centre masks are the OBCS
interior mask ``maskInC``; ``None`` skips the multiply.  No kernel reads the grid,
exchanges a halo or touches a cache.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ._rounded_product import rounded_multiply


@dataclass(frozen=True)
class DivVortInvariants:
    """Invariants of one velocity pair, read by the coefficient measures and the
    div-vort stencil.  ``vort3_bc``/``strain_bc`` carry MITgcm's dry-corner
    ``sideMaskFac`` scaling; ``vort3`` does not."""

    hDiv: torch.Tensor
    vort3: torch.Tensor
    vort3_bc: torch.Tensor
    strain_bc: torch.Tensor
    tension: torch.Tensor


def _hdiv_from_face_velocities(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    dyG: torch.Tensor,
    dxG: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    recip_rA: torch.Tensor,
    recip_hFacC: torch.Tensor,
    center_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MIT ``mom_calc_hdiv`` tensor kernel from face velocities.

    The caller supplies geometry-specific metrics in broadcast-ready shapes.
    ``dyG``/``dxG`` are the face lengths, ``hFacW/S`` are wet face fractions,
    and ``recip_rA``/``recip_hFacC`` normalize the C-point flux divergence.
    """
    u_flux = u * dyG * hFacW
    div_u = u_flux[..., 1:] - u_flux[..., :-1]
    v_flux = v * dxG * hFacS
    div_v = v_flux[..., 1:, :] - v_flux[..., :-1, :]
    hdiv = (div_u + div_v) * recip_rA * recip_hFacC
    if center_mask is not None:
        hdiv = hdiv * center_mask
    return hdiv


def _vort3_strain_from_curl_terms(
    *,
    curl_v: torch.Tensor,
    curl_u: torch.Tensor,
    recip_rAz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT relvort3/strain tensor kernel from metric-weighted curl terms.

    Geometry-specific exchange and metric multiplication are owned by the
    caller.  This helper applies the shared corner-area normalization and the
    two MIT sign combinations consumed by viscosity coefficient assembly.
    """
    vort3 = recip_rAz * (curl_v - curl_u)
    strain = recip_rAz * (curl_v + curl_u)
    return vort3, strain


def _tension_from_face_velocities(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    dyF: torch.Tensor,
    dxG: torch.Tensor,
    recip_rA: torch.Tensor,
    center_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MIT ``mom_calc_tension`` tensor kernel from face velocities."""
    tension_u = dyF * (u[..., 1:] - u[..., :-1])
    v_dxG = v * dxG
    tension_v = v_dxG[..., 1:, :] - v_dxG[..., :-1, :]
    tension = (tension_u - tension_v) * recip_rA
    if center_mask is not None:
        tension = tension * center_mask
    return tension


def _apply_hfacz_side_mask_to_corner_fields(
    vort3: torch.Tensor,
    strain: torch.Tensor,
    hFacZ: torch.Tensor,
    side_mask_fac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MIT ``vort3BC`` / ``strainBC`` dry-corner side-mask scaling."""
    hFacZ_pos = (hFacZ > 0).to(dtype=vort3.dtype).unsqueeze(0)
    side_factor = vort3.new_tensor(float(side_mask_fac))
    vort3_bc = torch.where(hFacZ_pos > 0, vort3, vort3 * side_factor)
    strain_bc = torch.where(hFacZ_pos > 0, strain, strain * side_factor)
    return vort3_bc, strain_bc


def _vort3_strain_from_haloes(
    *,
    u_ext: torch.Tensor,
    dxC_ext: torch.Tensor,
    v_dyC_ext: torch.Tensor,
    recip_rAz: torch.Tensor,
    ny: int,
    nx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT relvort3/strain at all ``(ny+1)x(nx+1)`` corners from haloed inputs.

    ``u_ext`` ``[B, nz, ny+2, nx+1]`` and ``dxC_ext`` ``[ny+2, nx+1]`` carry one
    j-halo row each side; ``v_dyC_ext`` ``[B, nz, ny+1, nx+2]`` is ``v*dyC`` with
    one i-halo column each side.
    """
    u_dxC_ext = u_ext * dxC_ext.view(1, 1, ny + 2, nx + 1)
    curl_u_ext = u_dxC_ext[..., 1:, :] - u_dxC_ext[..., :-1, :]  # [B, nz, ny+1, nx+1]
    curl_v_ext = v_dyC_ext[..., 1:] - v_dyC_ext[..., :-1]  # [B, nz, ny+1, nx+1]
    return _vort3_strain_from_curl_terms(
        curl_v=curl_v_ext,
        curl_u=curl_u_ext,
        recip_rAz=recip_rAz,
    )


def _hdiv_gradients_from_center(
    hDiv: torch.Tensor,
    *,
    dxC: torch.Tensor,
    dyG: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT ``mom_calc_visc`` hDiv gradients at C-faces."""
    divDx = torch.zeros_like(hDiv)
    divDy = torch.zeros_like(hDiv)
    divDx[..., 1:] = (hDiv[..., 1:] - hDiv[..., :-1]) / dxC
    divDy[..., 1:, :] = (hDiv[..., 1:, :] - hDiv[..., :-1, :]) / dyG
    return divDx, divDy


def _leith_vorticity_gradients_from_masks(
    vort3: torch.Tensor,
    *,
    dxG_ext: torch.Tensor,
    dyG: torch.Tensor,
    maskS: torch.Tensor,
    maskW: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT ``mom_calc_visc`` vorticity gradients with precombined masks."""
    ny = vort3.shape[-2] - 1
    nx = vort3.shape[-1] - 1
    vrtDx = torch.zeros_like(vort3)
    vrtDy = torch.zeros_like(vort3)
    dxG_view = (
        dxG_ext.view(1, 1, ny + 1, 1)
        if dxG_ext.ndim == 1
        else dxG_ext.view(1, 1, ny + 1, nx)
    )
    vrtDx[..., :, :nx] = (
        (vort3[..., :, 1:] - vort3[..., :, :-1])
        / dxG_view
        * maskS.unsqueeze(0)
    )
    vrtDy[..., :ny, :] = (
        (vort3[..., 1:, :] - vort3[..., :-1, :])
        / dyG
        * maskW.unsqueeze(0)
    )
    return vrtDx, vrtDy


def _d_point_vorticity_gradient_measures(
    vrtDx: torch.Tensor,
    vrtDy: torch.Tensor,
    *,
    ny: int,
    nx: int,
    want_sq: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """MIT D-point vorticity-gradient max norm and sum-of-squares measures.

    ``want_sq=False`` skips the sum-of-squares branch and returns ``None`` for
    it; the max norm is unaffected.  Callers gate on
    :meth:`HorizontalViscosity._needs_leith_sq_and_div_measures`.
    """
    vdx0 = vrtDx[..., :-1, :nx]
    vdx1 = vrtDx[..., 1:, :nx]
    vdy0 = vrtDy[..., :ny, :-1]
    vdy1 = vrtDy[..., :ny, 1:]
    vrt_d_max = torch.maximum(vdx0.abs(), vdx1.abs())
    vrt_d_max = torch.maximum(vrt_d_max, vdy0.abs())
    vrt_d_max = torch.maximum(vrt_d_max, vdy1.abs())
    if not want_sq:
        return vrt_d_max, None
    grd_vrt_d_sq = 0.25 * (
        (vdx1.square() + vdx0.square()) + (vdy1.square() + vdy0.square())
    )
    return vrt_d_max, grd_vrt_d_sq


def _d_point_divergence_gradient_measures(
    divDx: torch.Tensor,
    divDy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT D-point divergence-gradient max norm and sum-of-squares measures."""
    divDx_ip1 = torch.zeros_like(divDx)
    divDx_ip1[..., :-1] = divDx[..., 1:]
    divDy_jp1 = torch.zeros_like(divDy)
    divDy_jp1[..., :-1, :] = divDy[..., 1:, :]
    grd_div_d_max = torch.maximum(divDx_ip1.abs(), divDx.abs())
    grd_div_d_max = torch.maximum(grd_div_d_max, divDy_jp1.abs())
    grd_div_d_max = torch.maximum(grd_div_d_max, divDy.abs())
    grd_div_d_sq = 0.25 * (
        (divDx_ip1.square() + divDx.square())
        + (divDy_jp1.square() + divDy.square())
    )
    return grd_div_d_max, grd_div_d_sq


def _d_point_divergence_gradient_measures_from_faces(
    divDx: torch.Tensor,
    divDy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """D-point divergence-gradient measures from seam-aware face gradients."""
    nx = divDy.shape[-1]
    ny = divDx.shape[-2]
    divDx_w = divDx[..., :, :nx]
    divDx_e = divDx[..., :, 1:nx + 1]
    divDy_s = divDy[..., :ny, :]
    divDy_n = divDy[..., 1:ny + 1, :]
    grd_div_d_max = torch.maximum(divDx_e.abs(), divDx_w.abs())
    grd_div_d_max = torch.maximum(grd_div_d_max, divDy_n.abs())
    grd_div_d_max = torch.maximum(grd_div_d_max, divDy_s.abs())
    grd_div_d_sq = 0.25 * (
        (divDx_e.square() + divDx_w.square())
        + (divDy_n.square() + divDy_s.square())
    )
    return grd_div_d_max, grd_div_d_sq


def _z_point_vorticity_gradient_measures(
    vrtDx: torch.Tensor,
    vrtDy: torch.Tensor,
    *,
    want_sq: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """MIT Z-point vorticity-gradient max norm and sum-of-squares measures.

    ``want_sq=False`` skips the sum-of-squares branch (see the D-point analog).
    """
    vrtDx_im = torch.zeros_like(vrtDx)
    vrtDx_im[..., :, 1:] = vrtDx[..., :, :-1]
    vrtDy_jm = torch.zeros_like(vrtDy)
    vrtDy_jm[..., 1:, :] = vrtDy[..., :-1, :]
    vrt_z_max = torch.maximum(vrtDx.abs(), vrtDx_im.abs())
    vrt_z_max = torch.maximum(vrt_z_max, vrtDy.abs())
    vrt_z_max = torch.maximum(vrt_z_max, vrtDy_jm.abs())
    if not want_sq:
        return vrt_z_max, None
    grd_vrt_z_sq = 0.25 * (
        (vrtDx_im.square() + vrtDx.square())
        + (vrtDy_jm.square() + vrtDy.square())
    )
    return vrt_z_max, grd_vrt_z_sq


def _z_point_divergence_gradient_measures(
    divDx: torch.Tensor,
    divDy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT Z-point divergence-gradient max norm and sum-of-squares measures."""
    ny, nx = divDx.shape[-2:]
    d00 = torch.zeros(
        (*divDx.shape[:-2], ny + 1, nx + 1),
        dtype=divDx.dtype,
        device=divDx.device,
    )
    djm = torch.zeros_like(d00)
    e00 = torch.zeros_like(d00)
    eim = torch.zeros_like(d00)
    d00[..., :ny, :nx] = divDx
    djm[..., 1:, :nx] = divDx
    e00[..., :ny, :nx] = divDy
    eim[..., :ny, 1:] = divDy
    grd_div_z_max = torch.maximum(d00.abs(), djm.abs())
    grd_div_z_max = torch.maximum(grd_div_z_max, e00.abs())
    grd_div_z_max = torch.maximum(grd_div_z_max, eim.abs())
    grd_div_z_sq = 0.25 * ((djm.square() + d00.square()) + (eim.square() + e00.square()))
    return grd_div_z_max, grd_div_z_sq


def _smag_strain_magnitude(primary, first, second, third, fourth):
    """Form the MIT D/Z strain norm with separately rounded squared terms."""
    neighbours = (rounded_multiply(first, first) + rounded_multiply(second, second)
                  + rounded_multiply(third, third) + rounded_multiply(fourth, fourth))
    return torch.sqrt(rounded_multiply(primary, primary) + rounded_multiply(neighbours, 0.25) + 1e-30)


def _d_point_smag_strain_magnitude(
    tension: torch.Tensor,
    strain: torch.Tensor,
) -> torch.Tensor:
    """MIT D-point Smagorinsky strain magnitude from tension and strain."""
    return _smag_strain_magnitude(
        tension, strain[..., :-1, 1:], strain[..., 1:, :-1],
        strain[..., :-1, :-1], strain[..., 1:, 1:],
    )


def _z_point_smag_strain_magnitude(
    strain: torch.Tensor,
    tension: torch.Tensor,
) -> torch.Tensor:
    """MIT Z-point Smagorinsky strain magnitude from strain and tension."""
    ny, nx = tension.shape[-2:]
    t00 = torch.zeros(
        (*tension.shape[:-2], ny + 1, nx + 1),
        dtype=tension.dtype,
        device=tension.device,
    )
    tjm = torch.zeros_like(t00)
    tim = torch.zeros_like(t00)
    timjm = torch.zeros_like(t00)
    t00[..., :ny, :nx] = tension
    tjm[..., 1:, :nx] = tension
    tim[..., :ny, 1:] = tension
    timjm[..., 1:, 1:] = tension
    return _smag_strain_magnitude(strain, t00, tjm, tim, timjm)


def _hdiv_gradient_faces_from_haloes(
    hDiv_x: torch.Tensor,
    hDiv_y: torch.Tensor,
    *,
    dxC_row: torch.Tensor,
    dyG: torch.Tensor,
    ny: int,
    nx: int,
    west: bool,
    east: bool,
    south: bool,
    north: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT hDiv gradients on D-neighbour faces, retaining every coupled edge face.

    ``hDiv_x`` ``[..., ny, nx+2]`` and ``hDiv_y`` ``[..., ny+2, nx]`` are ``hDiv``
    with one zero-filled or neighbour halo along x and y.  An edge face is kept
    when its side flag says the edge couples and is zero on a wall or open edge,
    as in :func:`_hdiv_gradients_from_center`.
    """
    raw_x = (
        hDiv_x[..., :, 1:nx + 2] - hDiv_x[..., :, :nx + 1]
    ) / dxC_row.view(1, 1, ny, nx + 1)
    divDx = torch.zeros_like(raw_x)
    divDx[..., :, 1:nx] = raw_x[..., :, 1:nx]
    if west:
        divDx[..., :, 0:1] = raw_x[..., :, 0:1]
    if east:
        divDx[..., :, nx:nx + 1] = raw_x[..., :, nx:nx + 1]

    raw_y = (hDiv_y[..., 1:ny + 2, :] - hDiv_y[..., :ny + 1, :]) / dyG
    divDy = torch.zeros_like(raw_y)
    divDy[..., 1:ny, :] = raw_y[..., 1:ny, :]
    if south:
        divDy[..., 0:1, :] = raw_y[..., 0:1, :]
    if north:
        divDy[..., ny:ny + 1, :] = raw_y[..., ny:ny + 1, :]
    return divDx, divDy


def _z_point_vorticity_gradient_measures_from_haloes(
    vrtDx_ext: torch.Tensor,
    vrtDy_ext: torch.Tensor,
    *,
    ny: int,
    nx: int,
    want_sq: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Z-point vorticity-gradient measures with neighbours across coupled edges.

    ``vrtDx_ext`` ``[..., ny+1, nx+2]`` and ``vrtDy_ext`` ``[..., ny+2, nx+1]``
    carry the neighbour faces: the wrap face on a periodic axis of one tile, the
    neighbour rank's face across a rank seam, zero on a wall or open edge.
    ``want_sq=False`` skips the sum-of-squares branch (see the D-point analog).
    """
    vrtDx_i = vrtDx_ext[..., :, 1:nx + 2]
    vrtDx_im = vrtDx_ext[..., :, :nx + 1]
    vrtDy_j = vrtDy_ext[..., 1:ny + 2, :]
    vrtDy_jm = vrtDy_ext[..., :ny + 1, :]
    vrt_z_max = torch.maximum(vrtDx_i.abs(), vrtDx_im.abs())
    vrt_z_max = torch.maximum(vrt_z_max, vrtDy_j.abs())
    vrt_z_max = torch.maximum(vrt_z_max, vrtDy_jm.abs())
    if not want_sq:
        return vrt_z_max, None
    grd_vrt_z_sq = 0.25 * (
        (vrtDx_im.square() + vrtDx_i.square())
        + (vrtDy_jm.square() + vrtDy_j.square())
    )
    return vrt_z_max, grd_vrt_z_sq


def _z_point_divergence_gradient_measures_from_haloes(
    divDx_ext: torch.Tensor,
    divDy_ext: torch.Tensor,
    *,
    ny: int,
    nx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-point divergence-gradient measures from seam-aware face gradients.

    ``divDx_ext`` ``[..., ny+2, nx+1]`` and ``divDy_ext`` ``[..., ny+1, nx+2]``
    carry one halo row and column of the face gradients.
    """
    d00 = divDx_ext[..., 1:ny + 2, :]
    djm = divDx_ext[..., :ny + 1, :]
    e00 = divDy_ext[..., :, 1:nx + 2]
    eim = divDy_ext[..., :, :nx + 1]
    grd_div_z_max = torch.maximum(d00.abs(), djm.abs())
    grd_div_z_max = torch.maximum(grd_div_z_max, e00.abs())
    grd_div_z_max = torch.maximum(grd_div_z_max, eim.abs())
    grd_div_z_sq = 0.25 * ((djm.square() + d00.square()) + (eim.square() + e00.square()))
    return grd_div_z_max, grd_div_z_sq


def _z_point_smag_strain_magnitude_from_halo(
    strain: torch.Tensor,
    tension_ext: torch.Tensor,
    *,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """Z-point Smagorinsky strain with centre neighbours across coupled edges.

    ``tension_ext`` ``[..., ny+2, nx+2]`` is tension with one halo cell each side.
    """
    t00 = tension_ext[..., 1:ny + 2, 1:nx + 2]
    tjm = tension_ext[..., :ny + 1, 1:nx + 2]
    tim = tension_ext[..., 1:ny + 2, :nx + 1]
    timjm = tension_ext[..., :ny + 1, :nx + 1]
    return _smag_strain_magnitude(strain, t00, tjm, tim, timjm)
