"""MITgcm no-slip lateral (side) drag, ``mom_u_sidedrag.F`` / ``mom_v_sidedrag.F``.

Kernels only, ``sideDragFactor > 0`` branch.  The closed fraction is the resting
face fraction minus the corner fraction (``h0FacW - hFacZ``); the drag divides by
the current face fraction.  U inputs sit on ``[..., ny, nx+1]``, V inputs on
``[..., ny+1, nx]``, corner fractions and the corner metrics ``dxV``/``dyU`` on
``[..., ny+1, nx+1]``.  Outputs are U and V tendencies in m/s^2.  The constant
form takes one scalar viscosity; the variable form takes the corner viscosity
``viscAh_Z`` (or ``viscA4_Z`` applied to ``del2u``, whose sign the caller
reverses).  The caller reads the closed fractions and metrics from the grid.
"""
from __future__ import annotations

import torch

from .viscosity_geometry import _recip


def _side_drag_u_2d(
    u: torch.Tensor,
    hFacW: torch.Tensor,
    h0FacW: torch.Tensor,
    h0FacZ: torch.Tensor,
    rAw: torch.Tensor,
    dxV: torch.Tensor,
    dyU: torch.Tensor,
    viscAh: torch.Tensor | float,
    factor: float,
) -> torch.Tensor:
    """No-slip lateral drag for u in the MITgcm 2-D-metric form.

    Transcribes ``mom_u_sidedrag.F`` (``sideDragFactor>0`` branch, constant
    harmonic ``viscAh``, ``viscA4=0``, ``cosFacU=1``) of a ``NONLIN_FRSURF``
    build (:107-108, :114):

        closedS = h0FacW(i,j) - hFacZ(i,j)
        closedN = h0FacW(i,j) - hFacZ(i,j+1)
        uDrag   = -recip_hFacW * recip_rAw * viscAh * u * sideDragFactor
                  * ( closedS*dxV(i,j  )*recip_dyU(i,j  )
                    + closedN*dxV(i,j+1)*recip_dyU(i,j+1) )

    The closed fraction reads the resting ``h0FacW`` and the corner fraction
    the caller passes (``h0FacZ`` of :func:`_side_drag_closed_fractions`);
    ``recip_hFacW`` is the current one.  On a static grid both fractions are
    the current ``hFacW``/``hFacZ``.

    Shapes (last two dims): ``u/hFacW/h0FacW/rAw = [...,ny,nx+1]`` at
    u-points; ``h0FacZ/dxV/dyU = [...,ny+1,nx+1]`` at corner (vorticity)
    points.  Works for the cube (face axis as batch) and the regional grid
    alike — the caller supplies the per-grid 2-D metrics broadcast over the
    leading ``[B,nz]`` axes.  This replaces the legacy Cartesian ``dy^2``
    side-drag stencil.
    """
    closedS = h0FacW - h0FacZ[..., :-1, :]
    closedN = h0FacW - h0FacZ[..., 1:, :]
    geom = (
        closedS * dxV[..., :-1, :] * _recip(dyU[..., :-1, :])
        + closedN * dxV[..., 1:, :] * _recip(dyU[..., 1:, :])
    )
    return -_recip(hFacW) * _recip(rAw) * viscAh * u * factor * geom


def _side_drag_v_2d(
    v: torch.Tensor,
    hFacS: torch.Tensor,
    h0FacS: torch.Tensor,
    h0FacZ: torch.Tensor,
    rAs: torch.Tensor,
    dyU: torch.Tensor,
    dxV: torch.Tensor,
    viscAh: torch.Tensor | float,
    factor: float,
) -> torch.Tensor:
    """No-slip lateral drag for v (``mom_v_sidedrag.F``, ``sideDragFactor>0``,
    ``NONLIN_FRSURF`` build :104-105, :111).

        closedW = h0FacS(i,j) - hFacZ(i,j)
        closedE = h0FacS(i,j) - hFacZ(i+1,j)
        vDrag   = -recip_hFacS * recip_rAs * viscAh * v * sideDragFactor
                  * ( closedW*dyU(i  ,j)*recip_dxV(i  ,j)
                    + closedE*dyU(i+1,j)*recip_dxV(i+1,j) )

    Resting closed fraction and current ``recip_hFacS``, as in
    :func:`_side_drag_u_2d`.  Shapes: ``v/hFacS/h0FacS/rAs = [...,ny+1,nx]`` at
    v-points; ``h0FacZ/dyU/dxV = [...,ny+1,nx+1]`` at corners.
    """
    closedW = h0FacS - h0FacZ[..., :, :-1]
    closedE = h0FacS - h0FacZ[..., :, 1:]
    geom = (
        closedW * dyU[..., :, :-1] * _recip(dxV[..., :, :-1])
        + closedE * dyU[..., :, 1:] * _recip(dxV[..., :, 1:])
    )
    return -_recip(hFacS) * _recip(rAs) * viscAh * v * factor * geom


def _variable_side_drag_tendencies(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    h0FacW: torch.Tensor,
    h0FacS: torch.Tensor,
    h0FacZ: torch.Tensor,
    dxV: torch.Tensor,
    dyU: torch.Tensor,
    recip_hFacW: torch.Tensor,
    recip_hFacS: torch.Tensor,
    rAw: torch.Tensor,
    rAs: torch.Tensor,
    corner_viscosity: torch.Tensor,
    side_drag_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MOM_U/V_SIDEDRAG with the actual viscosity at each corner.

    ``h0Fac*`` are the closed-fraction operands of the resting faces and
    corners, ``recip_hFac*`` the current reciprocals.
    """
    u_south = (
        (h0FacW - h0FacZ[..., :-1, :]) * dxV[..., :-1, :]
        * _recip(dyU[..., :-1, :])
        * (corner_viscosity[..., :-1, :] * u)
    )
    u_north = (
        (h0FacW - h0FacZ[..., 1:, :]) * dxV[..., 1:, :]
        * _recip(dyU[..., 1:, :])
        * (corner_viscosity[..., 1:, :] * u)
    )
    v_west = (
        (h0FacS - h0FacZ[..., :, :-1]) * dyU[..., :, :-1]
        * _recip(dxV[..., :, :-1])
        * (corner_viscosity[..., :, :-1] * v)
    )
    v_east = (
        (h0FacS - h0FacZ[..., :, 1:]) * dyU[..., :, 1:]
        * _recip(dxV[..., :, 1:])
        * (corner_viscosity[..., :, 1:] * v)
    )
    drag_u = (
        -recip_hFacW * _recip(rAw)
        * (u_south + u_north) * side_drag_factor
    )
    drag_v = (
        -recip_hFacS * _recip(rAs)
        * (v_west + v_east) * side_drag_factor
    )
    return drag_u, drag_v
