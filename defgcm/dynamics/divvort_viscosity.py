"""MITgcm mom_vi_hdissip div-vort horizontal viscosity stencil.

Kernels only.  Inputs are the centre divergence ``hDiv`` ``[B, nz, ny, nx]`` (or
``hDiv*viscAh_D``), the corner ``hFacZ*vort3`` ``[B, nz, ny+1, nx+1]`` (times
``viscAh_Z`` for a variable viscosity), face reciprocals ``recip_hFacW/S`` and
masks ``maskW/S`` (``hFac > 0``), and 2-D metrics.  Outputs are U-face
``[B, nz, ny, nx+1]`` and V-face ``[B, nz, ny+1, nx]`` tendencies in m/s^2.  The
``*_all_from_haloed`` kernels take ``hDiv`` already exchanged by the caller; the
caller also decides which edge faces couple.  No kernel reads the grid.
"""
from __future__ import annotations

import torch

from .viscosity_geometry import _recip


def _apply_del2uv_interior_masks(
    del2u: torch.Tensor,
    del2v: torch.Tensor,
    maskInW: torch.Tensor | None,
    maskInS: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MIT ``MOM_VI_DEL2UV`` OBCS interior masks to face outputs."""
    if maskInW is not None:
        mInW = maskInW.to(dtype=del2u.dtype, device=del2u.device)
        del2u = del2u * mInW.reshape(
            (1,) * (del2u.ndim - mInW.ndim) + tuple(mInW.shape)
        )
    if maskInS is not None:
        mInS = maskInS.to(dtype=del2v.dtype, device=del2v.device)
        del2v = del2v * mInS.reshape(
            (1,) * (del2v.ndim - mInS.ndim) + tuple(mInS.shape)
        )
    return del2u, del2v


def _divvort_dissipation_from_haloed_hdiv(
    *,
    hDiv_ext: torch.Tensor,
    vort3: torch.Tensor,
    hFacZ: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    dxC: torch.Tensor,
    dyG: torch.Tensor,
    dyC: torch.Tensor,
    dxG: torch.Tensor,
    viscosity: torch.Tensor,
    halo: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the MIT ``mom_vi_hdissip.F`` div-vort stencil on all C-grid faces.

    The caller supplies the geometry-specific pieces: exchanged ``hDiv`` with
    a scalar halo, vorticity/corner hFac, face hFac masks, and 2-D metrics
    already broadcast over any leading batch/level/face axes.
    """
    ny = hFacW.shape[-2]
    nx = hFacS.shape[-1]
    ol = halo

    dHdx = (
        hDiv_ext[..., ol:ol + ny, ol:ol + nx + 1]
        - hDiv_ext[..., ol:ol + ny, ol - 1:ol + nx]
    ) * _recip(dxC)

    hz_v3 = hFacZ * vort3
    dZdy = (hz_v3[..., 1:, :] - hz_v3[..., :-1, :]) * _recip(dyG)
    gU = viscosity * (dHdx - _recip(hFacW) * dZdy)

    dHdy = (
        hDiv_ext[..., ol:ol + ny + 1, ol:ol + nx]
        - hDiv_ext[..., ol - 1:ol + ny, ol:ol + nx]
    ) * _recip(dyC)
    dZdx = (hz_v3[..., :, 1:] - hz_v3[..., :, :-1]) * _recip(dxG)
    gV = viscosity * (_recip(hFacS) * dZdx + dHdy)

    return gU, gV


def _divvort_corner_flux(
    vort3: torch.Tensor,
    hFacZ: torch.Tensor,
) -> torch.Tensor:
    """MIT div-vort corner term ``hFacZ*vort3`` (constant viscosity applied later)."""
    return hFacZ.unsqueeze(0) * vort3


def _divvort_variable_fluxes(
    hDiv: torch.Tensor,
    vort3: torch.Tensor,
    hFacZ: torch.Tensor,
    viscAh_D: torch.Tensor,
    viscAh_Z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT div-vort ``viscAh_D*hDiv`` and ``hFacZ*vort3*viscAh_Z`` flux terms."""
    hDiv_flux = hDiv * viscAh_D
    hz_v3 = hFacZ.unsqueeze(0) * vort3 * viscAh_Z
    return hDiv_flux, hz_v3


def _u_divvort_interior_from_fluxes(
    hDiv_flux: torch.Tensor,
    hz_v3: torch.Tensor,
    recip_hFacW: torch.Tensor,
    maskW: torch.Tensor,
    dxC_row: torch.Tensor,
    dyG: torch.Tensor,
) -> torch.Tensor:
    """MIT ``mom_vi_hdissip`` U-face interior div-vort stencil."""
    ny = hDiv_flux.shape[-2]
    nx = hDiv_flux.shape[-1]
    dxC = (
        dxC_row.view(1, 1, ny, 1)
        if dxC_row.ndim == 1
        else dxC_row[:, 1:nx].view(1, 1, ny, nx - 1)
    )
    dHdx = (
        hDiv_flux[..., 1:] - hDiv_flux[..., :-1]
    ) / dxC
    dZdy = (hz_v3[..., 1:, :] - hz_v3[..., :-1, :]) / dyG
    return (
        dHdx - recip_hFacW[:, :, 1:nx].unsqueeze(0) * dZdy[..., 1:nx]
    ) * maskW[:, :, 1:nx].unsqueeze(0)


def _v_divvort_interior_from_fluxes(
    hDiv_flux: torch.Tensor,
    hz_v3: torch.Tensor,
    recip_hFacS: torch.Tensor,
    maskS: torch.Tensor,
    dyC: torch.Tensor,
    dxG_ext: torch.Tensor,
) -> torch.Tensor:
    """MIT ``mom_vi_hdissip`` V-face interior div-vort stencil."""
    ny = hDiv_flux.shape[-2]
    nx = hDiv_flux.shape[-1]
    dyC_interior = dyC if dyC.ndim == 0 else dyC[1:ny, :]
    dHdy = (
        hDiv_flux[..., 1:, :] - hDiv_flux[..., :-1, :]
    ) / dyC_interior
    dZdx_full = hDiv_flux.new_zeros((*hDiv_flux.shape[:-2], ny + 1, nx))
    dxG_view = (
        dxG_ext.view(1, 1, ny + 1, 1)
        if dxG_ext.ndim == 1
        else dxG_ext.view(1, 1, ny + 1, nx)
    )
    dZdx_full[..., :, :] = (
        hz_v3[..., :, 1:nx + 1] - hz_v3[..., :, :nx]
    ) / dxG_view
    return (
        recip_hFacS[:, 1:ny, :].unsqueeze(0) * dZdx_full[..., 1:ny, :]
        + dHdy
    ) * maskS[:, 1:ny, :].unsqueeze(0)


def _u_divvort_output_from_interior(
    template: torch.Tensor,
    interior: torch.Tensor,
) -> torch.Tensor:
    """Assemble U div-vort dissipation from interior face values."""
    nx = template.shape[-1] - 1
    result = torch.zeros_like(template)
    result[..., 1:nx] = interior
    return result


def _v_divvort_output_from_interior(
    template: torch.Tensor,
    interior: torch.Tensor,
) -> torch.Tensor:
    """Assemble V div-vort dissipation from interior face values."""
    ny = template.shape[-2] - 1
    result = torch.zeros_like(template)
    result[..., 1:ny, :] = interior
    return result


def _u_divvort_output_from_all(
    template: torch.Tensor,
    values: torch.Tensor,
    *,
    west: bool,
    east: bool,
) -> torch.Tensor:
    """Assemble U div-vort output, retaining only true interior/seam faces."""
    nx = template.shape[-1] - 1
    result = torch.zeros_like(template)
    result[..., 1:nx] = values[..., 1:nx]
    if west:
        result[..., 0:1] = values[..., 0:1]
    if east:
        result[..., nx:nx + 1] = values[..., nx:nx + 1]
    return result


def _v_divvort_output_from_all(
    template: torch.Tensor,
    values: torch.Tensor,
    *,
    south: bool,
    north: bool,
) -> torch.Tensor:
    """Assemble V div-vort output, retaining only true interior/seam faces."""
    ny = template.shape[-2] - 1
    result = torch.zeros_like(template)
    result[..., 1:ny, :] = values[..., 1:ny, :]
    if south:
        result[..., 0:1, :] = values[..., 0:1, :]
    if north:
        result[..., ny:ny + 1, :] = values[..., ny:ny + 1, :]
    return result


def _u_divvort_all_from_haloed(
    hDiv_x: torch.Tensor,
    hz_v3: torch.Tensor,
    recip_hFacW: torch.Tensor,
    maskW: torch.Tensor,
    dxC_row: torch.Tensor,
    dyG: torch.Tensor,
    *,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """MIT ``mom_vi_hdissip`` U-face stencil on local interior and seam faces.

    ``hDiv_x`` ``[..., ny, nx+2]`` is the (viscosity-weighted) ``hDiv`` with one
    x-halo column each side.
    """
    dHdx = (
        hDiv_x[..., :, 1:nx + 2] - hDiv_x[..., :, :nx + 1]
    ) / dxC_row.view(1, 1, ny, nx + 1)
    dZdy = (hz_v3[..., 1:, :] - hz_v3[..., :-1, :]) / dyG
    return (
        dHdx - recip_hFacW.unsqueeze(0) * dZdy
    ) * maskW.unsqueeze(0)


def _v_divvort_all_from_haloed(
    hDiv_y: torch.Tensor,
    hz_v3: torch.Tensor,
    recip_hFacS: torch.Tensor,
    maskS: torch.Tensor,
    dyC: torch.Tensor,
    dxG_ext: torch.Tensor,
    *,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """MIT ``mom_vi_hdissip`` V-face stencil on local interior and seam faces.

    ``hDiv_y`` ``[..., ny+2, nx]`` is the (viscosity-weighted) ``hDiv`` with one
    y-halo row each side.
    """
    dHdy = (hDiv_y[..., 1:ny + 2, :] - hDiv_y[..., :ny + 1, :]) / dyC
    dZdx = (
        hz_v3[..., :, 1:nx + 1] - hz_v3[..., :, :nx]
    ) / dxG_ext.view(1, 1, ny + 1, nx)
    return (
        recip_hFacS.unsqueeze(0) * dZdx + dHdy
    ) * maskS.unsqueeze(0)
