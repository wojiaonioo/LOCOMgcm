"""Flux-form horizontal viscosity: MOM_U/V_DEL2U-style strain fluxes and their divergence.

Kernels only.  ``u`` is ``[B, nz, ny, nx+1]``, ``v`` ``[B, nz, ny+1, nx]``.  Zonal
strain fluxes of ``u`` sit at cell centres and are weighted by ``hFacC``;
meridional ones sit at corners and are weighted by ``hFacZ`` (symmetrically for
``v``).  Variable-coefficient fluxes carry ``viscAh_D`` at centres and
``viscAh_Z`` at corners; the biharmonic fluxes subtract ``viscA4_D``/``viscA4_Z``
times the same differences of ``del2u``/``del2v`` (MOM_U/V_DEL2U).  Outputs are face
tendencies; edge faces and corners are kept only where the caller's side flags say
the edge couples.  Haloed inputs (``*_ext``) are exchanged by the caller, in the
order the caller documents.
"""
from __future__ import annotations

import torch

from .lateral_drag import _side_drag_u_2d, _side_drag_v_2d
from .viscosity_geometry import _recip


def _u_center_strain_flux(
    u: torch.Tensor,
    hFacC: torch.Tensor,
    dx: torch.Tensor,
) -> torch.Tensor:
    """MIT U-Laplacian zonal strain flux at C points."""
    return hFacC * (u[..., 1:] - u[..., :-1]) / dx


def _v_center_strain_flux(
    v: torch.Tensor,
    hFacC: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    """MIT V-Laplacian meridional strain flux at C points."""
    return hFacC * (v[..., 1:, :] - v[..., :-1, :]) / dy


def _u_corner_strain_flux(
    u: torch.Tensor,
    hFacZ: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    """MIT U-Laplacian meridional strain flux at Z points."""
    ny = u.shape[-2]
    nx_p1 = u.shape[-1]
    fMer = torch.zeros(
        (*u.shape[:-2], ny + 1, nx_p1),
        dtype=u.dtype,
        device=u.device,
    )
    fMer[..., 1:ny, :] = (
        hFacZ[..., 1:ny, :] * (u[..., 1:, :] - u[..., :-1, :]) / dy
    )
    return fMer


def _v_corner_strain_flux(
    v: torch.Tensor,
    hFacZ: torch.Tensor,
    dx: torch.Tensor,
) -> torch.Tensor:
    """MIT V-Laplacian zonal strain flux at Z points."""
    ny_p1 = v.shape[-2]
    nx = v.shape[-1]
    fZon = torch.zeros(
        (*v.shape[:-2], ny_p1, nx + 1),
        dtype=v.dtype,
        device=v.device,
    )
    fZon[..., 1:nx] = (
        hFacZ[..., 1:nx] * (v[..., 1:] - v[..., :-1]) / dx
    )
    return fZon


def _x_flux_divergence(flux: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """Divergence of an x-staggered flux along the i direction."""
    return (flux[..., 1:] - flux[..., :-1]) / dx


def _y_flux_divergence(flux: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Divergence of a y-staggered flux along the j direction."""
    return (flux[..., 1:, :] - flux[..., :-1, :]) / dy


def _laplacian_flux_divergence_with_hfac(
    primary_div: torch.Tensor,
    cross_div: torch.Tensor,
    hFac: torch.Tensor,
) -> torch.Tensor:
    """Combine Laplacian flux divergences and zero dry faces."""
    safe_hfac = torch.where(hFac > 0, hFac, torch.ones_like(hFac))
    del2 = (primary_div + cross_div) / safe_hfac
    return torch.where(hFac > 0, del2, torch.zeros_like(del2))


def _u_laplacian_output_from_interior(
    primary_div: torch.Tensor,
    interior: torch.Tensor,
) -> torch.Tensor:
    """Assemble U-Laplacian output from interior and edge divergence."""
    nx = primary_div.shape[-1] - 1
    result = torch.zeros_like(primary_div)
    result[..., 1:nx] = interior
    result[..., 0:1] = primary_div[..., 0:1]
    result[..., nx:nx + 1] = primary_div[..., nx:nx + 1]
    return result


def _u_laplacian_output_from_all(
    primary_div: torch.Tensor,
    values: torch.Tensor,
    *,
    west: bool,
    east: bool,
) -> torch.Tensor:
    nx = primary_div.shape[-1] - 1
    result = _u_laplacian_output_from_interior(primary_div, values[..., 1:nx])
    if west:
        result[..., 0:1] = values[..., 0:1]
    if east:
        result[..., nx:nx + 1] = values[..., nx:nx + 1]
    return result


def _v_laplacian_output_from_interior(
    primary_div: torch.Tensor,
    interior: torch.Tensor,
) -> torch.Tensor:
    """Assemble V-Laplacian output from interior and edge divergence."""
    ny = primary_div.shape[-2] - 1
    result = torch.zeros_like(primary_div)
    result[..., 1:ny, :] = interior
    result[..., 0:1, :] = primary_div[..., 0:1, :]
    result[..., ny:ny + 1, :] = primary_div[..., ny:ny + 1, :]
    return result


def _v_laplacian_output_from_all(
    primary_div: torch.Tensor,
    values: torch.Tensor,
    *,
    south: bool,
    north: bool,
) -> torch.Tensor:
    ny = primary_div.shape[-2] - 1
    result = _v_laplacian_output_from_interior(primary_div, values[..., 1:ny, :])
    if south:
        result[..., 0:1, :] = values[..., 0:1, :]
    if north:
        result[..., ny:ny + 1, :] = values[..., ny:ny + 1, :]
    return result


def _u_corner_strain_flux_with_edges(
    u: torch.Tensor,
    u_ext: torch.Tensor,
    hFacZ: torch.Tensor,
    dy: torch.Tensor,
    *,
    south: bool,
    north: bool,
) -> torch.Tensor:
    """Fill U cross-strain fluxes at coupled periodic or partition seams.

    ``u_ext`` ``[..., ny+2, nx+1]`` is ``u`` with one replicated or neighbour
    j-halo row each side.
    """
    fMer = _u_corner_strain_flux(u, hFacZ, dy)
    ny = u.shape[-2]
    all_edges = hFacZ * (u_ext[..., 1:ny + 2, :] - u_ext[..., 0:ny + 1, :]) / dy
    if south:
        fMer[..., 0:1, :] = all_edges[..., 0:1, :]
    if north:
        fMer[..., ny:ny + 1, :] = all_edges[..., ny:ny + 1, :]
    return fMer


def _v_corner_strain_flux_with_edges(
    v: torch.Tensor,
    v_ext: torch.Tensor,
    hFacZ: torch.Tensor,
    dx: torch.Tensor,
    *,
    west: bool,
    east: bool,
) -> torch.Tensor:
    """Fill V cross-strain fluxes at coupled periodic or partition seams.

    ``v_ext`` ``[..., ny+1, nx+2]`` is ``v`` with one replicated or neighbour
    i-halo column each side.
    """
    fZon = _v_corner_strain_flux(v, hFacZ, dx)
    nx = v.shape[-1]
    all_edges = hFacZ * (v_ext[..., 1:nx + 2] - v_ext[..., 0:nx + 1]) / dx
    if west:
        fZon[..., 0:1] = all_edges[..., 0:1]
    if east:
        fZon[..., nx:nx + 1] = all_edges[..., nx:nx + 1]
    return fZon


def _u_viscous_zonal_flux(
    u: torch.Tensor,
    *,
    dyF: torch.Tensor,
    dxF: torch.Tensor,
    hFacC: torch.Tensor,
    visc_d: torch.Tensor,
) -> torch.Tensor:
    """``fZon = dyF hFacC viscAh_D (u(i+1) - u(i)) / dxF`` at C points."""
    return ((dyF * hFacC) * (visc_d * (u[..., 1:] - u[..., :-1]))) / dxF


def _v_viscous_meridional_flux(
    v: torch.Tensor,
    *,
    dxF: torch.Tensor,
    dyF: torch.Tensor,
    hFacC: torch.Tensor,
    visc_d: torch.Tensor,
) -> torch.Tensor:
    """``fMer = dxF hFacC viscAh_D (v(j+1) - v(j)) / dyF`` at C points."""
    return ((dxF * hFacC) * (visc_d * (v[..., 1:, :] - v[..., :-1, :]))) / dyF


def _x_face_difference(flux: torch.Tensor) -> torch.Tensor:
    """``flux(i+1) - flux(i)`` along x (metric flux form: no division)."""
    return flux[..., 1:] - flux[..., :-1]


def _y_face_difference(flux: torch.Tensor) -> torch.Tensor:
    """``flux(j+1) - flux(j)`` along y (metric flux form: no division)."""
    return flux[..., 1:, :] - flux[..., :-1, :]


def _u_strain_flux_metric(
    u: torch.Tensor,
    *,
    dyF: torch.Tensor,
    dxF: torch.Tensor,
    hFacC: torch.Tensor,
) -> torch.Tensor:
    """``fZon = dyF hFacC (u(i+1) - u(i)) / dxF`` at C points."""
    return ((dyF * hFacC) * (u[..., 1:] - u[..., :-1])) / dxF


def _v_strain_flux_metric(
    v: torch.Tensor,
    *,
    dxF: torch.Tensor,
    dyF: torch.Tensor,
    hFacC: torch.Tensor,
) -> torch.Tensor:
    """``fMer = dxF hFacC (v(j+1) - v(j)) / dyF`` at C points."""
    return ((dxF * hFacC) * (v[..., 1:, :] - v[..., :-1, :])) / dyF


def _u_corner_strain_divergence_metric(
    u: torch.Tensor,
    u_ext: torch.Tensor | None,
    *,
    dxV: torch.Tensor,
    dyU: torch.Tensor,
    hFacZ: torch.Tensor,
    ny: int,
    nx: int,
    south: bool,
    north: bool,
) -> torch.Tensor:
    """``fMer(j+1) - fMer(j)`` with ``fMer = dxV hFacZ (u(j) - u(j-1)) / dyU`` at Z
    points; edge corners take ``u_ext`` (``u`` with one j-halo row each side,
    ``None`` unless ``south or north``) and are zero otherwise."""
    fMer = torch.zeros((*u.shape[:-2], ny + 1, nx + 1), dtype=u.dtype, device=u.device)
    fMer[..., 1:ny, :] = (
        (dxV[1:ny] * hFacZ[..., 1:ny, :])
        * (u[..., 1:, :] - u[..., :-1, :])
    ) / dyU[1:ny]
    if south or north:
        all_edges = (
            (dxV * hFacZ)
            * (u_ext[..., 1:ny + 2, :] - u_ext[..., 0:ny + 1, :])
        ) / dyU
        if south:
            fMer[..., 0:1, :] = all_edges[..., 0:1, :]
        if north:
            fMer[..., ny:ny + 1, :] = all_edges[..., ny:ny + 1, :]
    return fMer[..., 1:, :] - fMer[..., :-1, :]


def _v_corner_strain_divergence_metric(
    v: torch.Tensor,
    v_ext: torch.Tensor | None,
    *,
    dxV: torch.Tensor,
    dyU: torch.Tensor,
    hFacZ: torch.Tensor,
    ny: int,
    nx: int,
    west: bool,
    east: bool,
) -> torch.Tensor:
    """``fZon(i+1) - fZon(i)`` with ``fZon = dyU hFacZ (v(i) - v(i-1)) / dxV`` at Z
    points; ``v_ext`` is ``v`` with one i-halo column each side, ``None`` unless
    ``west or east``."""
    fZon = torch.zeros((*v.shape[:-2], ny + 1, nx + 1), dtype=v.dtype, device=v.device)
    fZon[..., 1:nx] = (
        (dyU[..., 1:nx] * hFacZ[..., 1:nx])
        * (v[..., 1:] - v[..., :-1])
    ) / dxV[..., 1:nx]
    if west or east:
        all_edges = (
            (dyU * hFacZ)
            * (v_ext[..., 1:nx + 2] - v_ext[..., 0:nx + 1])
        ) / dxV
        if west:
            fZon[..., 0:1] = all_edges[..., 0:1]
        if east:
            fZon[..., nx:nx + 1] = all_edges[..., nx:nx + 1]
    return fZon[..., 1:] - fZon[..., :-1]


def _u_viscous_meridional_divergence(
    u: torch.Tensor,
    u_ext: torch.Tensor | None,
    *,
    dxV: torch.Tensor,
    dyU: torch.Tensor,
    hFacZ: torch.Tensor,
    visc_z: torch.Tensor,
    ny: int,
    nx: int,
    south: bool,
    north: bool,
) -> torch.Tensor:
    """``fMer(j+1) - fMer(j)`` with ``fMer = dxV hFacZ viscAh_Z (u(j) - u(j-1)) / dyU``;
    ``u_ext`` as in :func:`_u_corner_strain_divergence_metric`."""
    fMer = torch.zeros((*u.shape[:-2], ny + 1, nx + 1), dtype=u.dtype, device=u.device)
    fMer[..., 1:ny, :] = (
        (dxV[1:ny] * hFacZ[..., 1:ny, :])
        * (visc_z[..., 1:ny, :] * (u[..., 1:, :] - u[..., :-1, :]))
    ) / dyU[1:ny]
    if south or north:
        all_edges = (
            (dxV * hFacZ)
            * (visc_z * (u_ext[..., 1:ny + 2, :] - u_ext[..., 0:ny + 1, :]))
        ) / dyU
        if south:
            fMer[..., 0:1, :] = all_edges[..., 0:1, :]
        if north:
            fMer[..., ny:ny + 1, :] = all_edges[..., ny:ny + 1, :]
    return fMer[..., 1:, :] - fMer[..., :-1, :]


def _v_viscous_zonal_divergence(
    v: torch.Tensor,
    v_ext: torch.Tensor | None,
    *,
    dxV: torch.Tensor,
    dyU: torch.Tensor,
    hFacZ: torch.Tensor,
    visc_z: torch.Tensor,
    ny: int,
    nx: int,
    west: bool,
    east: bool,
) -> torch.Tensor:
    """``fZon(i+1) - fZon(i)`` with ``fZon = dyU hFacZ viscAh_Z (v(i) - v(i-1)) / dxV``."""
    fZon = torch.zeros((*v.shape[:-2], ny + 1, nx + 1), dtype=v.dtype, device=v.device)
    fZon[..., 1:nx] = (
        (dyU[..., 1:nx] * hFacZ[..., 1:nx])
        * (visc_z[..., 1:nx] * (v[..., 1:] - v[..., :-1]))
    ) / dxV[..., 1:nx]
    if west or east:
        all_edges = (
            (dyU * hFacZ)
            * (visc_z * (v_ext[..., 1:nx + 2] - v_ext[..., 0:nx + 1]))
        ) / dxV
        if west:
            fZon[..., 0:1] = all_edges[..., 0:1]
        if east:
            fZon[..., nx:nx + 1] = all_edges[..., nx:nx + 1]
    return fZon[..., 1:] - fZon[..., :-1]


def _u_viscous_divergence(
    dFZon: torch.Tensor,
    dFMer: torch.Tensor,
    *,
    hFacW: torch.Tensor,
    rAw: torch.Tensor,
    nx: int,
    west: bool,
    east: bool,
) -> torch.Tensor:
    """``recip_hFacW recip_rAw (dFZon + dFMer)`` on the U faces; an uncoupled edge
    face keeps the zonal difference ``dFZon``."""
    if west or east:
        values = _recip(hFacW) * _recip(rAw) * (dFZon + dFMer)
        return _u_laplacian_output_from_all(dFZon, values, west=west, east=east)
    interior = (_recip(hFacW[..., 1:nx]) * _recip(rAw[..., 1:nx])
                * (dFZon[..., 1:nx] + dFMer[..., 1:nx]))
    return _u_laplacian_output_from_interior(dFZon, interior)


def _v_viscous_divergence(
    dFZon: torch.Tensor,
    dFMer: torch.Tensor,
    *,
    hFacS: torch.Tensor,
    rAs: torch.Tensor,
    ny: int,
    south: bool,
    north: bool,
) -> torch.Tensor:
    """``recip_hFacS recip_rAs (dFZon + dFMer)`` on the V faces; an uncoupled edge
    face keeps the meridional difference ``dFMer``."""
    if south or north:
        values = _recip(hFacS) * _recip(rAs) * (dFZon + dFMer)
        return _v_laplacian_output_from_all(dFMer, values, south=south, north=north)
    interior = (_recip(hFacS[..., 1:ny, :]) * _recip(rAs[..., 1:ny, :])
                * (dFZon[..., 1:ny, :] + dFMer[..., 1:ny, :]))
    return _v_laplacian_output_from_interior(dFMer, interior)


def _u_laplacian_row_divergence(
    dFZon: torch.Tensor,
    dFMer: torch.Tensor,
    *,
    hFacW: torch.Tensor,
    nx: int,
    west: bool,
    east: bool,
) -> torch.Tensor:
    """Row-metric form ``(dFZon + dFMer) / hFacW`` (zero on dry faces); an
    uncoupled edge face keeps ``dFZon``."""
    if west or east:
        values = _laplacian_flux_divergence_with_hfac(dFZon, dFMer, hFacW)
        return _u_laplacian_output_from_all(dFZon, values, west=west, east=east)
    interior = _laplacian_flux_divergence_with_hfac(
        dFZon[..., 1:nx], dFMer[..., 1:nx], hFacW[..., 1:nx],
    )
    return _u_laplacian_output_from_interior(dFZon, interior)


def _v_laplacian_row_divergence(
    dFMer: torch.Tensor,
    dFZon: torch.Tensor,
    *,
    hFacS: torch.Tensor,
    ny: int,
    south: bool,
    north: bool,
) -> torch.Tensor:
    """Row-metric form ``(dFMer + dFZon) / hFacS`` (zero on dry faces); an
    uncoupled edge face keeps ``dFMer``."""
    if south or north:
        values = _laplacian_flux_divergence_with_hfac(dFMer, dFZon, hFacS)
        return _v_laplacian_output_from_all(dFMer, values, south=south, north=north)
    interior = _laplacian_flux_divergence_with_hfac(
        dFMer[..., 1:ny, :], dFZon[..., 1:ny, :], hFacS[..., 1:ny, :],
    )
    return _v_laplacian_output_from_interior(dFMer, interior)


def _u_del2_zonal_flux(
    u: torch.Tensor,
    *,
    dyF: torch.Tensor,
    dxF: torch.Tensor,
    hFacC: torch.Tensor,
    maskInC: torch.Tensor | None,
) -> torch.Tensor:
    """MOM_U_DEL2U ``fZon = dyF hFacC (u(i+1) - u(i)) / dxF * maskInC`` at C points
    (``mom_u_del2u.F:55-63``; ``maskInC`` is ``None`` on a grid without open
    boundaries, where MITgcm's factor is 1)."""
    fZon = _u_strain_flux_metric(u, dyF=dyF, dxF=dxF, hFacC=hFacC)
    if maskInC is not None:
        fZon = fZon * maskInC
    return fZon


def _v_del2_meridional_flux(
    v: torch.Tensor,
    *,
    dxF: torch.Tensor,
    dyF: torch.Tensor,
    hFacC: torch.Tensor,
    maskInC: torch.Tensor | None,
) -> torch.Tensor:
    """MOM_V_DEL2V ``fMer = dxF hFacC (v(j+1) - v(j)) / dyF * maskInC`` at C points
    (``mom_v_del2v.F:69-77``)."""
    fMer = _v_strain_flux_metric(v, dxF=dxF, dyF=dyF, hFacC=hFacC)
    if maskInC is not None:
        fMer = fMer * maskInC
    return fMer


def _u_del2(
    dFZon: torch.Tensor,
    dFMer: torch.Tensor,
    *,
    hFacW: torch.Tensor,
    rAw: torch.Tensor,
    maskW: torch.Tensor,
    nx: int,
    west: bool,
    east: bool,
) -> torch.Tensor:
    """MOM_U_DEL2U ``del2u = recip_hFacW recip_rAw (dFZon + dFMer) maskW``
    (``mom_u_del2u.F:88-95``; the caller applies ``maskInW`` and adds the no-slip
    term)."""
    return _u_viscous_divergence(
        dFZon, dFMer, hFacW=hFacW, rAw=rAw, nx=nx, west=west, east=east,
    ) * maskW


def _v_del2(
    dFZon: torch.Tensor,
    dFMer: torch.Tensor,
    *,
    hFacS: torch.Tensor,
    rAs: torch.Tensor,
    maskS: torch.Tensor,
    ny: int,
    south: bool,
    north: bool,
) -> torch.Tensor:
    """MOM_V_DEL2V ``del2v = recip_hFacS recip_rAs (dFZon + dFMer) maskS``
    (``mom_v_del2v.F:87-94``)."""
    return _v_viscous_divergence(
        dFZon, dFMer, hFacS=hFacS, rAs=rAs, ny=ny, south=south, north=north,
    ) * maskS


def _u_del2_side_term(
    u: torch.Tensor,
    *,
    hFacW: torch.Tensor,
    h0FacW: torch.Tensor,
    h0FacZ: torch.Tensor,
    rAw: torch.Tensor,
    dxV: torch.Tensor,
    dyU: torch.Tensor,
    maskW: torch.Tensor,
    side_drag_factor: float,
) -> torch.Tensor:
    """MOM_U_DEL2U no-slip term (``mom_u_del2u.F:100-121``): the side drag with unit
    viscosity, times ``maskW``.  Resting closed fractions, current ``recip_hFacW``."""
    return _side_drag_u_2d(
        u, hFacW, h0FacW, h0FacZ, rAw, dxV, dyU, 1.0, side_drag_factor,
    ) * maskW


def _v_del2_side_term(
    v: torch.Tensor,
    *,
    hFacS: torch.Tensor,
    h0FacS: torch.Tensor,
    h0FacZ: torch.Tensor,
    rAs: torch.Tensor,
    dyU: torch.Tensor,
    dxV: torch.Tensor,
    maskS: torch.Tensor,
    side_drag_factor: float,
) -> torch.Tensor:
    """MOM_V_DEL2V no-slip term (``mom_v_del2v.F:99-121``)."""
    return _side_drag_v_2d(
        v, hFacS, h0FacS, h0FacZ, rAs, dyU, dxV, 1.0, side_drag_factor,
    ) * maskS


def _u_viscous_zonal_flux_biharmonic(
    u: torch.Tensor,
    del2u: torch.Tensor,
    *,
    dyF: torch.Tensor,
    dxF: torch.Tensor,
    hFacC: torch.Tensor,
    visc_d: torch.Tensor,
    visc4_d: torch.Tensor,
) -> torch.Tensor:
    """``fZon = dyF hFacC (viscAh_D (u(i+1)-u(i)) - viscA4_D (del2u(i+1)-del2u(i))) / dxF``
    at C points (``mom_u_xviscflux.F:56-67`` with the sign folded)."""
    return ((dyF * hFacC) * (visc_d * (u[..., 1:] - u[..., :-1])
                             - visc4_d * (del2u[..., 1:] - del2u[..., :-1]))) / dxF


def _v_viscous_meridional_flux_biharmonic(
    v: torch.Tensor,
    del2v: torch.Tensor,
    *,
    dxF: torch.Tensor,
    dyF: torch.Tensor,
    hFacC: torch.Tensor,
    visc_d: torch.Tensor,
    visc4_d: torch.Tensor,
) -> torch.Tensor:
    """``fMer = dxF hFacC (viscAh_D (v(j+1)-v(j)) - viscA4_D (del2v(j+1)-del2v(j))) / dyF``
    at C points (``mom_v_yviscflux.F:56-71`` with the sign folded)."""
    return ((dxF * hFacC) * (visc_d * (v[..., 1:, :] - v[..., :-1, :])
                             - visc4_d * (del2v[..., 1:, :] - del2v[..., :-1, :]))) / dyF


def _u_viscous_meridional_divergence_biharmonic(
    u: torch.Tensor,
    u_ext: torch.Tensor | None,
    del2u: torch.Tensor,
    del2u_ext: torch.Tensor | None,
    *,
    dxV: torch.Tensor,
    dyU: torch.Tensor,
    hFacZ: torch.Tensor,
    visc_z: torch.Tensor,
    visc4_z: torch.Tensor,
    ny: int,
    nx: int,
    south: bool,
    north: bool,
) -> torch.Tensor:
    """``fMer(j+1) - fMer(j)`` with
    ``fMer = dxV hFacZ (viscAh_Z (u(j)-u(j-1)) - viscA4_Z (del2u(j)-del2u(j-1))) / dyU``
    (``mom_u_yviscflux.F:57-72``); ``u_ext``/``del2u_ext`` carry one j-halo row each
    side, ``None`` unless ``south or north``."""
    fMer = torch.zeros((*u.shape[:-2], ny + 1, nx + 1), dtype=u.dtype, device=u.device)
    fMer[..., 1:ny, :] = (
        (dxV[1:ny] * hFacZ[..., 1:ny, :])
        * (visc_z[..., 1:ny, :] * (u[..., 1:, :] - u[..., :-1, :])
           - visc4_z[..., 1:ny, :] * (del2u[..., 1:, :] - del2u[..., :-1, :]))
    ) / dyU[1:ny]
    if south or north:
        all_edges = (
            (dxV * hFacZ)
            * (visc_z * (u_ext[..., 1:ny + 2, :] - u_ext[..., 0:ny + 1, :])
               - visc4_z * (del2u_ext[..., 1:ny + 2, :] - del2u_ext[..., 0:ny + 1, :]))
        ) / dyU
        if south:
            fMer[..., 0:1, :] = all_edges[..., 0:1, :]
        if north:
            fMer[..., ny:ny + 1, :] = all_edges[..., ny:ny + 1, :]
    return fMer[..., 1:, :] - fMer[..., :-1, :]


def _v_viscous_zonal_divergence_biharmonic(
    v: torch.Tensor,
    v_ext: torch.Tensor | None,
    del2v: torch.Tensor,
    del2v_ext: torch.Tensor | None,
    *,
    dxV: torch.Tensor,
    dyU: torch.Tensor,
    hFacZ: torch.Tensor,
    visc_z: torch.Tensor,
    visc4_z: torch.Tensor,
    ny: int,
    nx: int,
    west: bool,
    east: bool,
) -> torch.Tensor:
    """``fZon(i+1) - fZon(i)`` with
    ``fZon = dyU hFacZ (viscAh_Z (v(i)-v(i-1)) - viscA4_Z (del2v(i)-del2v(i-1))) / dxV``
    (``mom_v_xviscflux.F:57-68``)."""
    fZon = torch.zeros((*v.shape[:-2], ny + 1, nx + 1), dtype=v.dtype, device=v.device)
    fZon[..., 1:nx] = (
        (dyU[..., 1:nx] * hFacZ[..., 1:nx])
        * (visc_z[..., 1:nx] * (v[..., 1:] - v[..., :-1])
           - visc4_z[..., 1:nx] * (del2v[..., 1:] - del2v[..., :-1]))
    ) / dxV[..., 1:nx]
    if west or east:
        all_edges = (
            (dyU * hFacZ)
            * (visc_z * (v_ext[..., 1:nx + 2] - v_ext[..., 0:nx + 1])
               - visc4_z * (del2v_ext[..., 1:nx + 2] - del2v_ext[..., 0:nx + 1]))
        ) / dxV
        if west:
            fZon[..., 0:1] = all_edges[..., 0:1]
        if east:
            fZon[..., nx:nx + 1] = all_edges[..., nx:nx + 1]
    return fZon[..., 1:] - fZon[..., :-1]
