"""Statements of MITgcm ``pkg/cd_code/cd_code_scheme.F`` as tensor kernels.

Shapes are LOCOMgcm's, with a leading batch: U-point arrays ``[B, nz, ny, nx+1]``
(west faces; the east face of the last column is stored again), V-point
arrays ``[B, nz, ny+1, nx]`` (south faces; the north face stored again), cell
centres ``[B, ny, nx]``.  MITgcm's ``X(i,j)`` is ``x[..., j-1, i-1]``.  An
argument named ``*_x`` carries one ghost column on each side along x, ``*_y``
one ghost row on each side along y, exchanged by the caller.  Masks are 0/1
in the field dtype; metrics are unmasked.  Each expression keeps the Fortran
order of the cited lines, where equal-precedence operators bind left to right
(``ab15*(S)*0.25`` is ``(ab15*S)*0.25``).  ``phxFac = phyFac`` is
``pfFacMom`` under staggerTimeStep and zero otherwise (``cd_code_scheme.F:77-84``);
the ``*_staggered`` kernels take the hydrostatic term as LOCOMgcm's pressure
gradient tendency ``hpg = -pfFacMom*dPhiHyd``, so ``phyFac*dPhiHydY`` is
``-hpg_v``.  The other two drop the term, which with ``phyFac = 0`` changes
only the sign of a zero.
"""
from __future__ import annotations

import torch


def extrapolated_surface_pressure(
    eta: torch.Tensor,
    eta_nm1: torch.Tensor,
    *,
    bo_surf: float,
    ab15: float,
    ab05: float,
) -> torch.Tensor:
    """``pf = Bo_surf*(ab15*etaN + ab05*etaNm1)``, cd_code_scheme.F:103-104."""
    return bo_surf * (ab15 * eta + ab05 * eta_nm1)


def extrapolated_surface_pressure_no_ab_coriolis(
    eta: torch.Tensor,
    eta_nm1: torch.Tensor,
    *,
    bo_surf: float,
    ab15: float,
    ab05: float,
) -> torch.Tensor:
    """``pf = ab15*(etaN*Bo_surf) + ab05*(etaNm1*Bo_surf)``, cd_code_scheme.F:97-101."""
    return ab15 * (eta * bo_surf) + ab05 * (eta_nm1 * bo_surf)


def v_point_acceleration(
    g_v: torch.Tensor,
    pf_y: torch.Tensor,
    *,
    recip_dyc: torch.Tensor,
    mask_s: torch.Tensor,
) -> torch.Tensor:
    """``af = (gvFld - recip_dyC*(pf(i,j)-pf(i,j-1)))*maskS``, cd_code_scheme.F:113-124.

    ``g_v`` is gVtmp on V points; ``pf_y`` is ``[B, ny+2, nx]``; ``recip_dyc``
    is ``[ny+1, nx]``.  Returns af on V points.
    """
    dpf = pf_y[..., 1:, :] - pf_y[..., :-1, :]
    return (g_v - (recip_dyc * dpf).unsqueeze(-3)) * mask_s


def v_point_acceleration_staggered(
    g_v: torch.Tensor,
    pf_y: torch.Tensor,
    hpg_v: torch.Tensor,
    *,
    recip_dyc: torch.Tensor,
    mask_s: torch.Tensor,
) -> torch.Tensor:
    """``af = (gvFld - (recip_dyC*(pf(i,j)-pf(i,j-1)) + phyFac*dPhiHydY))*maskS``, :113-124."""
    dpf = pf_y[..., 1:, :] - pf_y[..., :-1, :]
    return (g_v - ((recip_dyc * dpf).unsqueeze(-3) - hpg_v)) * mask_s


def u_point_acceleration_staggered(
    g_u: torch.Tensor,
    pf_x: torch.Tensor,
    hpg_u: torch.Tensor,
    *,
    recip_dxc: torch.Tensor,
    mask_w: torch.Tensor,
) -> torch.Tensor:
    """``af = (guFld - (recip_dxC*(pf(i,j)-pf(i-1,j)) + phxFac*dPhiHydX))*maskW``, :175-186."""
    dpf = pf_x[..., 1:] - pf_x[..., :-1]
    return (g_u - ((recip_dxc * dpf).unsqueeze(-3) - hpg_u)) * mask_w


def u_point_acceleration(
    g_u: torch.Tensor,
    pf_x: torch.Tensor,
    *,
    recip_dxc: torch.Tensor,
    mask_w: torch.Tensor,
) -> torch.Tensor:
    """``af = (guFld - recip_dxC*(pf(i,j)-pf(i-1,j)))*maskW``, cd_code_scheme.F:175-186.

    ``g_u`` is gUtmp on U points; ``pf_x`` is ``[B, ny, nx+2]``; ``recip_dxc``
    is ``[ny, nx+1]``.  Returns af on U points.
    """
    dpf = pf_x[..., 1:] - pf_x[..., :-1]
    return (g_u - (recip_dxc * dpf).unsqueeze(-3)) * mask_w


def d_grid_v_step(
    af_v_x: torch.Tensor,
    v_d: torch.Tensor,
    u_cor: torch.Tensor,
    v_x: torch.Tensor,
    v_nm1_x: torch.Tensor,
    *,
    f_u: torch.Tensor,
    mask_w: torch.Tensor,
    ab15: float,
    ab05: float,
    dt_mom: float,
    r_cd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step and relax vVelD on U points, then guCor; cd_code_scheme.F:125-171.

    ``af_v_x``, ``v_x`` and ``v_nm1_x`` are V-point arrays ``[..., ny+1, nx+2]``
    whose column ``i+1`` is MITgcm's ``i``.  ``u_cor`` is the velocity of the
    Coriolis term on U points, ``ab15*uVel + ab05*uNM1`` (lines 136-137) or
    ``uVel`` (line 135).  ``f_u`` is ``[ny, 1]``: on a
    latitude-longitude grid fCori does not vary with i, so MITgcm's
    ``(fCori(i,j)+fCori(i-1,j))*0.5`` is exactly ``fCori(j)``.  ``cfFacMom=1``
    and its product is omitted.  Returns ``(vVelD, guCor)`` on U points.
    """
    ny = v_d.shape[-2]

    def corner_sum(a):
        # (a(i,j)+a(i-1,j+1)) + (a(i-1,j)+a(i,j+1)), lines 129-130 and 153-158;
        # FCoriolis sums the same corners in mom_u_coriolis.F:50-52 order.
        return (a[..., :ny, 1:] + a[..., 1:, :-1]) + (a[..., :ny, :-1] + a[..., 1:, 1:])

    vf = corner_sum(af_v_x) * 0.25 * mask_w - f_u * u_cor
    v_d = v_d + dt_mom * vf
    v_d = (
        r_cd * v_d
        + (1.0 - r_cd) * (ab15 * corner_sum(v_x) * 0.25 + ab05 * corner_sum(v_nm1_x) * 0.25)
    ) * mask_w
    return v_d, f_u * v_d


def d_grid_u_step(
    af_u_y: torch.Tensor,
    u_d: torch.Tensor,
    v_cor: torch.Tensor,
    u_y: torch.Tensor,
    u_nm1_y: torch.Tensor,
    *,
    f_v: torch.Tensor,
    mask_s: torch.Tensor,
    ab15: float,
    ab05: float,
    dt_mom: float,
    r_cd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step and relax uVelD on V points, then gvCor; cd_code_scheme.F:187-233.

    ``af_u_y``, ``u_y`` and ``u_nm1_y`` are U-point arrays ``[..., ny+2, nx+1]``
    whose row ``j+1`` is MITgcm's ``j``.  ``v_cor`` is ``ab15*vVel +
    ab05*vNM1`` (lines 198-199) or ``vVel`` (line 197).  ``f_v`` is
    ``(fCori(i,j)+fCori(i,j-1))*0.5`` as ``[ny+1, 1]``.  The acceleration and
    the velocity average sum the four corners in different orders (lines
    191-192 against 215-216).  Returns ``(uVelD, gvCor)`` on V points.
    """
    # (a(i,j)+a(i+1,j-1)) + (a(i+1,j)+a(i,j-1)), lines 191-192; FCoriolis sums
    # the same corners in mom_v_coriolis.F:50-52 order.
    vf = (
        (af_u_y[..., 1:, :-1] + af_u_y[..., :-1, 1:]) + (af_u_y[..., 1:, 1:] + af_u_y[..., :-1, :-1])
    ) * 0.25 * mask_s + f_v * v_cor
    u_d = u_d + dt_mom * vf

    def corner_sum(a):
        # (a(i,j)+a(i+1,j-1)) + (a(i,j-1)+a(i+1,j)), lines 215-216 and 219-220
        return (a[..., 1:, :-1] + a[..., :-1, 1:]) + (a[..., :-1, :-1] + a[..., 1:, 1:])

    u_d = (
        r_cd * u_d
        + (1.0 - r_cd) * (ab15 * corner_sum(u_y) * 0.25 + ab05 * corner_sum(u_nm1_y) * 0.25)
    ) * mask_s
    return u_d, -(f_v * u_d)
