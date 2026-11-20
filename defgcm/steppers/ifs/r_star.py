"""MITgcm r* vertical coordinate (``select_rStar > 0``): column factors and their uses.

Kernels only: no grid, no halo exchange, no state.  The caller supplies the
halo-extended neighbours a face needs and owns the time layers.

Tensor conventions (MITgcm-native, as the kernels transcribe):

* ``eta`` is etaH, upward-positive, in m, on C points ``[ny, nx]``.
* ``h0_fac_c/w/s`` are the resting fractions ``h0FacC/W/S`` ``[nz, ny, nx]``,
  ``[nz, ny, nx+1]``, ``[nz, ny+1, nx]``; ``dr_f`` is ``drF`` ``[nz]`` in m,
  positive.  ``rF(1) = Ro_surf = 0`` (ocean z coordinate, no ice shelf), so
  ``R_low = -sum_k drF*h0FacC`` and ``Ro_surf - R_low`` is the column depth.
* The r* factors are dimensionless, 1 at rest; a dry column or face keeps 1.
* ``w`` returned by :func:`r_star_vertical_velocity` is LOCOMgcm downward-positive
  on the ``nz+1`` interfaces, zero at the seafloor.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RStarFactors:
    """``rStarFac``, ``rStarExp`` and ``rStarDh*Dt`` of SURFACE.h at C, W and S points.

    Held across steps by the stepper: ``fac_*`` is the factor of the latest
    CALC_R_STAR, ``exp_*`` and ``dhdt_*`` compare it with the one before.
    """

    fac_c: torch.Tensor
    fac_w: torch.Tensor
    fac_s: torch.Tensor
    exp_c: torch.Tensor
    exp_w: torch.Tensor
    exp_s: torch.Tensor
    dhdt_c: torch.Tensor
    dhdt_w: torch.Tensor
    dhdt_s: torch.Tensor


def resting_r_star_factors(
    like_c: torch.Tensor, like_w: torch.Tensor, like_s: torch.Tensor,
) -> RStarFactors:
    """INI_NLFS_VARS: factors and ratios 1, rates 0 (``ini_nlfs_vars.F:80-92``)."""
    return RStarFactors(
        fac_c=torch.ones_like(like_c), fac_w=torch.ones_like(like_w),
        fac_s=torch.ones_like(like_s),
        exp_c=torch.ones_like(like_c), exp_w=torch.ones_like(like_w),
        exp_s=torch.ones_like(like_s),
        dhdt_c=torch.zeros_like(like_c), dhdt_w=torch.zeros_like(like_w),
        dhdt_s=torch.zeros_like(like_s),
    )


def column_sum(h0_fac: torch.Tensor, dr_f: torch.Tensor) -> torch.Tensor:
    """``sum_k drF(k)*h0Fac(k)`` accumulated from k=1 down, as ini_masks_etc.F does."""
    total = torch.zeros_like(h0_fac[0])
    for k in range(h0_fac.shape[0]):
        total = total + dr_f[k] * h0_fac[k]
    return total


def column_low_and_recip_depth(
    h0_fac_c: torch.Tensor, dr_f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``R_low`` and ``recip_Rcol`` of C columns (``ini_masks_etc.F:136-140, :221-231``).

    With ``rF(1) = Ro_surf = 0``: ``R_low = 0 - S``, ``Ro_surf = R_low + S = 0``
    and the column depth ``Ro_surf - R_low`` is ``S``; ``recip_Rcol`` is 0 where
    the depth is not positive.
    """
    depth = column_sum(h0_fac_c, dr_f)
    r_low = -depth
    positive = depth > 0
    recip = torch.where(
        positive, 1.0 / torch.where(positive, depth, torch.ones_like(depth)),
        torch.zeros_like(depth),
    )
    return r_low, recip


def face_column_depth(
    r_low_minus: torch.Tensor, r_low_plus: torch.Tensor,
    h0_fac_face: torch.Tensor, dr_f: torch.Tensor,
) -> torch.Tensor:
    """``rSurfW - rLowW`` (or S) of a face between two C columns.

    ``rLowW = MAX(R_low(i-1), R_low(i))`` and ``rSurfW = rLowW + sum_k drF*h0FacW``
    (``ini_masks_etc.F:332-337, :357-372``); CALC_R_STAR divides by their
    difference (``calc_r_star.F:118``), which is formed here in the same order.
    """
    r_low_face = torch.maximum(r_low_minus, r_low_plus)
    r_surf_face = r_low_face + column_sum(h0_fac_face, dr_f)
    return r_surf_face - r_low_face


def area_weighted_face_mean(
    eta_minus: torch.Tensor, eta_plus: torch.Tensor,
    area_minus: torch.Tensor, area_plus: torch.Tensor, recip_area_face: torch.Tensor,
) -> torch.Tensor:
    """etaH at a W (or S) face, rStarAreaWeight form (``calc_r_star.F:111-138``).

    ``0.5*(eta(i-1)*rA(i-1) + eta(i)*rA(i))*recip_rAw``; ``minus``/``plus`` are
    the C columns before and after the face.
    """
    return 0.5 * (eta_minus * area_minus + eta_plus * area_plus) * recip_area_face


def simple_face_mean(eta_minus: torch.Tensor, eta_plus: torch.Tensor) -> torch.Tensor:
    """etaH at a W (or S) face, simple average (``calc_r_star.F:141-166``)."""
    return 0.5 * (eta_minus + eta_plus)


def calc_r_star(
    *,
    eta: torch.Tensor,
    mean_w: torch.Tensor,
    mean_s: torch.Tensor,
    r_low_c: torch.Tensor,
    recip_depth_c: torch.Tensor,
    depth_w: torch.Tensor,
    depth_s: torch.Tensor,
    wet_c: torch.Tensor,
    wet_w: torch.Tensor,
    wet_s: torch.Tensor,
    previous: RStarFactors,
    dt_free_surf: float,
) -> RStarFactors:
    """MITgcm ``CALC_R_STAR(etaFld)`` (``calc_r_star.F:76-166, :233-247``).

    ``mean_w``/``mean_s`` are etaH at the W ``[ny, nx+1]`` and S ``[ny+1, nx]``
    faces (:func:`area_weighted_face_mean` or :func:`simple_face_mean`, the
    caller's ``rStarAreaWeight``).  ``wet_*`` is ``kSurf <= Nr``.
    ``previous`` holds the factors this call replaces.
    """
    fac_c = torch.where(wet_c, (eta - r_low_c) * recip_depth_c, torch.ones_like(eta))
    safe_w = torch.where(wet_w, depth_w, torch.ones_like(depth_w))
    safe_s = torch.where(wet_s, depth_s, torch.ones_like(depth_s))
    fac_w = torch.where(wet_w, (mean_w + safe_w) / safe_w, torch.ones_like(depth_w))
    fac_s = torch.where(wet_s, (mean_s + safe_s) / safe_s, torch.ones_like(depth_s))
    return RStarFactors(
        fac_c=fac_c, fac_w=fac_w, fac_s=fac_s,
        exp_c=fac_c / previous.fac_c, exp_w=fac_w / previous.fac_w,
        exp_s=fac_s / previous.fac_s,
        dhdt_c=(fac_c - previous.fac_c) / dt_free_surf,
        dhdt_w=(fac_w - previous.fac_w) / dt_free_surf,
        dhdt_s=(fac_s - previous.fac_s) / dt_free_surf,
    )


def r_star_hfac(h0_fac: torch.Tensor, fac: torch.Tensor) -> torch.Tensor:
    """UPDATE_R_STAR: every level ``hFac = h0Fac*rStarFac`` (``update_r_star.F:50-59``)."""
    return h0_fac * fac


def r_star_thickness_rate(d_eta_h_dt: torch.Tensor, recip_depth_c: torch.Tensor) -> torch.Tensor:
    """``rStarDhDt = dEtaHdt*recip_Rcol`` (``integr_continuity.F:245-256``; deep and rho factors 1)."""
    return d_eta_h_dt * recip_depth_c


def r_star_vertical_velocity(
    horizontal_volume_divergence: torch.Tensor,
    *,
    cell_area: torch.Tensor,
    wet_mask: torch.Tensor,
    thickness_rate: torch.Tensor,
    h0_fac_c: torch.Tensor,
    dr_f: torch.Tensor,
) -> torch.Tensor:
    """INTEGRATE_FOR_W, r* branch (``integrate_for_w.F:123-145``), downward-positive.

    MITgcm: ``w(k) = (w(k+1) + conv2d*recip_rA - rStarDhDt*drF(k)*h0FacC(k))*maskC(k)``
    from k=Nr up, ``conv2d = -div``; the surface value is not masked by a level
    above.  Negated term by term (exact), the downward form is
    ``w(k) = (w(k+1) + div/rA + rStarDhDt*drF(k)*h0FacC(k))*maskC(k)``.  The
    divergence is divided by ``rA`` as the z-coordinate producer does
    (``operators.vertical_velocity_continuity``).
    ``horizontal_volume_divergence`` and ``wet_mask`` are ``[*, nz, ny, nx]``,
    ``thickness_rate`` is ``[*, ny, nx]`` and ``h0_fac_c`` ``[nz, ny, nx]``.
    """
    center = horizontal_volume_divergence / cell_area
    nz = center.shape[-3]
    levels = [torch.zeros_like(center[..., 0, :, :])]
    for k in range(nz - 1, -1, -1):
        levels.append(
            (levels[-1] + center[..., k, :, :] + thickness_rate * dr_f[k] * h0_fac_c[k])
            * wet_mask[..., k, :, :]
        )
    return torch.stack(levels[::-1], dim=-3)


def r_star_rescale_tendency(tendency: torch.Tensor, exp_c: torch.Tensor) -> torch.Tensor:
    """FREESURF_RESCALE_G, r* branch: every level ``gTracer /= rStarExpC`` (``freesurf_rescale_g.F:40-47``)."""
    return tendency / exp_c


def r_star_recip_hfac_new(recip_hfac_c: torch.Tensor, exp_c: torch.Tensor) -> torch.Tensor:
    """``recip_hFacNew = recip_hFacC/rStarExpC`` for the implicit tracer stage (``thermodynamics.F:206-216``)."""
    return recip_hfac_c / exp_c
