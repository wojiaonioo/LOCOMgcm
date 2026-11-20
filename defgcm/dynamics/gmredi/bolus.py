"""GM advective form: bolus stream function and residual velocity
(MITgcm ``gmredi_calc_psi_bolus.F``, ``gmredi_slope_psi.F``, ``gmredi_residual_flow.F``).

Slopes are formed at U points (x) and V points (y) between levels ``k-1`` and
``k`` for ``k = 2..Nr`` (``Nr-1`` rows); the stream functions ``GM_PsiX``
(U points) and ``GM_PsiY`` (V points) are ``[Nr, NY, NX]`` W-level arrays in
m^2/s with row 0 (surface) zero.  The residual velocity adds the curl of the
stream function to ``uFld vFld wFld``; ``w`` is MITgcm upward-positive (m/s).
z coordinates make ``wUnit2rVel``, ``rVel2wUnit``, ``deepFac*`` and
``flipSign4LHCoord`` equal to 1.  Index and sign conventions are those of
the package docstring.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .levels import below_bottom_mask, next_level
from .slope_limit import ldd97_depth_factor, tanh_slope_taper

# z coordinates: halfSign = 0.5*gravitySign (gmredi_calc_psi_bolus.F:89, ini_vertical_grid.F:54).
_HALF_SIGN = -0.5


def psi_slope_inputs_x(
    sigma_x: torch.Tensor,
    sigma_r: torch.Tensor,
    *,
    maskW: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(SlopeX, dSigmaDrW)`` numerators at U points, ``gmredi_calc_psi_bolus.F:106-113``.

    Inputs ``[Nr, NY, NX]``; outputs ``[Nr-1, NY, NX]`` for ``k = 2..Nr``, set
    for ``i >= 2-OLx`` and zero in the first column.
    """
    slope = (sigma_x[:-1, :, 1:] + sigma_x[1:, :, 1:]) * 0.5 * maskW[:-1, :, 1:] * maskW[1:, :, 1:]
    dsdr = (sigma_r[1:, :, :-1] + sigma_r[1:, :, 1:]) * _HALF_SIGN * maskW[:-1, :, 1:] * maskW[1:, :, 1:]
    return F.pad(slope, (1, 0)), F.pad(dsdr, (1, 0))


def psi_slope_inputs_y(
    sigma_y: torch.Tensor,
    sigma_r: torch.Tensor,
    *,
    maskS: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(SlopeY, dSigmaDrS)`` numerators at V points, ``gmredi_calc_psi_bolus.F:114-121``."""
    slope = (
        (sigma_y[:-1, 1:, :] + sigma_y[1:, 1:, :]) * 0.5 * maskS[:-1, 1:, :] * maskS[1:, 1:, :]
    )
    dsdr = (
        (sigma_r[1:, :-1, :] + sigma_r[1:, 1:, :]) * _HALF_SIGN * maskS[:-1, 1:, :] * maskS[1:, 1:, :]
    )
    return F.pad(slope, (0, 0, 1, 0)), F.pad(dsdr, (0, 0, 1, 0))


def psi_limited_slope(
    slope: torch.Tensor,
    dsigma_dr: torch.Tensor,
    *,
    small_number: float,
    slope_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slope and cut-off taper for one component, ``gmredi_slope_psi.F:200-227`` (x) and ``:234-261`` (y).

    ``slope_cutoff`` is ``SQRT(GM_slopeSqCutoff)`` (``:80``).  Returns ``(Slope, taper)``.
    """
    dsdr = torch.where(dsigma_dr <= small_number, torch.full_like(dsigma_dr, small_number), dsigma_dr)
    slope = slope / dsdr
    cut = torch.abs(slope) >= slope_cutoff
    slope = torch.where(cut, torch.copysign(torch.full_like(slope, slope_cutoff), slope), slope)
    taper = torch.where(cut, torch.zeros_like(slope), torch.ones_like(slope))
    return slope, taper


def psi_linear_taper(
    slope: torch.Tensor,
    taper: torch.Tensor,
    *,
    max_slope: float,
    slope_cutoff: float,
    small_number: float,
) -> torch.Tensor:
    """'linear', ``gmredi_slope_psi.F:270-288``."""
    slope_mod = torch.abs(slope)
    steep = (slope_mod > max_slope) & (slope_mod < slope_cutoff)
    return torch.where(steep, slope.new_tensor(max_slope) / (slope_mod + small_number), taper)


def psi_gkw91_taper(
    slope: torch.Tensor,
    taper: torch.Tensor,
    *,
    max_slope: float,
    slope_cutoff: float,
    small_number: float,
) -> torch.Tensor:
    """'gkw91', ``gmredi_slope_psi.F:290-312``."""
    slope_mod = torch.abs(slope)
    steep = (slope_mod > max_slope) & (slope_mod < slope_cutoff)
    return torch.where(
        steep, slope.new_tensor(max_slope * max_slope) / (slope * slope + small_number), taper
    )


def psi_dm95_taper(slope: torch.Tensor, *, s_crit: float, s_d: float) -> torch.Tensor:
    """'dm95', ``gmredi_slope_psi.F:314-328``: applied everywhere, overriding the cut-off 0."""
    return tanh_slope_taper(torch.abs(slope), s_crit=s_crit, s_d=s_d)


def psi_ldd97_taper(
    slope: torch.Tensor,
    taper: torch.Tensor,
    *,
    slope_cutoff: float,
    s_crit: float,
    s_d: float,
    r_depth: torch.Tensor,
    l_rho: torch.Tensor,
) -> torch.Tensor:
    """'ldd97', ``gmredi_slope_psi.F:330-374``; ``l_rho`` is ``LrhoW`` for x, ``LrhoS`` for y."""
    slope_mod = torch.abs(slope)
    active = slope_mod < slope_cutoff
    nonzero = slope_mod != 0
    safe = torch.where(nonzero, slope_mod, torch.ones_like(slope_mod))
    r_nondim = torch.where(nonzero, r_depth / (l_rho * safe), torch.ones_like(slope_mod))
    f1 = tanh_slope_taper(slope_mod, s_crit=s_crit, s_d=s_d)
    return torch.where(active, f1 * ldd97_depth_factor(r_nondim), taper)


def bolus_streamfunction(
    slope: torch.Tensor, taper: torch.Tensor, *, background_k: float | torch.Tensor
) -> torch.Tensor:
    """``GM_PsiX`` or ``GM_PsiY`` for ``k = 2..Nr``, ``gmredi_calc_psi_bolus.F:147-190``.

    With unit ``GM_bolFac*`` the factor ``half_K*(fac+fac)`` is ``GM_background_K``
    exactly, or the ``[NY, NX]`` Visbeck field at the face (``:150-153,171-174``).
    Returns ``[Nr, NY, NX]`` with the surface row zero.
    """
    return F.pad(slope * taper * background_k, (0, 0, 0, 0, 1, 0))


def residual_flow(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    psi_x: torch.Tensor,
    psi_y: torch.Tensor,
    *,
    recip_drF: torch.Tensor,
    recip_hFacW: torch.Tensor,
    recip_hFacS: torch.Tensor,
    dxG: torch.Tensor,
    dyG: torch.Tensor,
    recip_rA: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add the bolus velocity, ``gmredi_residual_flow.F:58-98``.

    ``u v w`` are ``[Nr, NY, NX]`` (``w`` upward-positive, row ``k-1`` the
    top of cell ``k``); ``recip_drF`` ``[Nr, 1, 1]``; ``recip_hFacW``,
    ``recip_hFacS`` ``[Nr, NY, NX]`` (0 where dry); ``dxG``, ``dyG``,
    ``recip_rA`` ``[NY, NX]``.  ``w`` changes on ``[1-OL : s+OL-1]`` only.
    """
    maskp1 = below_bottom_mask(psi_x)
    u = u + (next_level(psi_x) * maskp1 - psi_x) * recip_drF * recip_hFacW
    v = v + (next_level(psi_y) * maskp1 - psi_y) * recip_drF * recip_hFacS
    del_psi = (
        dyG[..., :-1, 1:] * psi_x[..., :-1, 1:]
        - dyG[..., :-1, :-1] * psi_x[..., :-1, :-1]
        + dxG[..., 1:, :-1] * psi_y[..., 1:, :-1]
        - dxG[..., :-1, :-1] * psi_y[..., :-1, :-1]
    )
    w = w + F.pad(del_psi * recip_rA[..., :-1, :-1], (0, 1, 0, 1))
    return u, v, w
