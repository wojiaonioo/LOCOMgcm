"""Isoneutral slopes and taper functions for GM/Redi (MITgcm ``gmredi_slope_limit.F``).

Covers the shared "compute slope, cut off, then taper" branch
(``gmredi_slope_limit.F:441-646``), the ac02 branch (``:393-439``) and the gradient interpolation to W, U and V
points that ``gmredi_calc_tensor.F`` does before each call.  ``dSigmaDr`` is
MITgcm's ``gravitySign*sigmaR`` (positive when stable); slopes are
dimensionless; z coordinates make every unit factor (``z2rUnit``,
``wUnit2rVel``, ``convSlopeUnit``) equal to 1, so they are left out.  W-point
inputs hold levels ``k = 2..Nr`` only (``Nr-1`` rows); U and V inputs hold all
``Nr`` levels.  Values outside MITgcm's inner loop range
``[2-OL : s+OL-1]`` are padding that the tensor kernels discard.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .levels import below_bottom_mask, next_level

# z coordinates: model/src/ini_vertical_grid.F:54.
_GRAVITY_SIGN = -1.0
# gmredi_slope_limit.F:124-125 (times z2rUnit or wUnit2rVel, both 1 in z coordinates).
BIG_SLOPE = 1.0e2
# gmredi_calc_tensor.F:166-168.
_LDD97_CSPD = 2.0
_LDD97_LRHO_INF = 15.0e3
_LDD97_LRHO_SUP = 100.0e3


def sigma_gradients_at_w(
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    sigma_r: torch.Tensor,
    *,
    maskC: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients at W points for ``k = 2..Nr`` (``gmredi_calc_tensor.F:395-410,507-511``).

    Inputs are ``[Nr, NY, NX]``; outputs ``[Nr-1, NY, NX]``, row ``k-2`` for
    interface ``k``.  ``dSigmaDx`` and ``dSigmaDy`` are set on the inner ring
    and zero outside it; ``dSigmaDr = gravitySign*sigmaR`` everywhere.
    """
    mask_fk = maskC[:-1] * maskC[1:]
    sx_up, sx_dn = sigma_x[:-1], sigma_x[1:]
    sy_up, sy_dn = sigma_y[:-1], sigma_y[1:]
    dsdx = (
        0.25
        * (
            sx_up[..., 1:-1, 2:]
            + sx_up[..., 1:-1, 1:-1]
            + sx_dn[..., 1:-1, 2:]
            + sx_dn[..., 1:-1, 1:-1]
        )
        * mask_fk[..., 1:-1, 1:-1]
    )
    dsdy = (
        0.25
        * (
            sy_up[..., 2:, 1:-1]
            + sy_up[..., 1:-1, 1:-1]
            + sy_dn[..., 2:, 1:-1]
            + sy_dn[..., 1:-1, 1:-1]
        )
        * mask_fk[..., 1:-1, 1:-1]
    )
    dsdr = _GRAVITY_SIGN * sigma_r[1:]
    return F.pad(dsdx, (1, 1, 1, 1)), F.pad(dsdy, (1, 1, 1, 1)), dsdr


def sigma_gradients_at_u(
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    sigma_r: torch.Tensor,
    *,
    maskW: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients at U points, all levels (``gmredi_calc_tensor.F:731-751``).

    Inputs and outputs are ``[Nr, NY, NX]``; outputs are zero outside the inner ring.
    """
    maskp1 = below_bottom_mask(sigma_r)
    sr_kp1 = next_level(sigma_r)
    mw = maskW[..., 1:-1, 1:-1]
    dsdx = sigma_x[..., 1:-1, 1:-1] * mw
    dsdy = (
        0.25
        * (
            sigma_y[..., 2:, :-2]
            + sigma_y[..., 2:, 1:-1]
            + sigma_y[..., 1:-1, :-2]
            + sigma_y[..., 1:-1, 1:-1]
        )
        * mw
    )
    dsdr = (
        0.25
        * (
            sigma_r[..., 1:-1, :-2]
            + sigma_r[..., 1:-1, 1:-1]
            + (sr_kp1[..., 1:-1, :-2] + sr_kp1[..., 1:-1, 1:-1]) * maskp1
        )
        * mw
        * _GRAVITY_SIGN
    )
    ring = (1, 1, 1, 1)
    return F.pad(dsdx, ring), F.pad(dsdy, ring), F.pad(dsdr, ring)


def sigma_gradients_at_v(
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    sigma_r: torch.Tensor,
    *,
    maskS: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients at V points, all levels (``gmredi_calc_tensor.F:929-948``).

    Inputs and outputs are ``[Nr, NY, NX]``; outputs are zero outside the inner ring.
    """
    maskp1 = below_bottom_mask(sigma_r)
    sr_kp1 = next_level(sigma_r)
    ms = maskS[..., 1:-1, 1:-1]
    dsdx = (
        0.25
        * (
            sigma_x[..., 1:-1, 1:-1]
            + sigma_x[..., 1:-1, 2:]
            + sigma_x[..., :-2, 1:-1]
            + sigma_x[..., :-2, 2:]
        )
        * ms
    )
    dsdy = sigma_y[..., 1:-1, 1:-1] * ms
    dsdr = (
        0.25
        * (
            sigma_r[..., :-2, 1:-1]
            + sigma_r[..., 1:-1, 1:-1]
            + (sr_kp1[..., :-2, 1:-1] + sr_kp1[..., 1:-1, 1:-1]) * maskp1
        )
        * ms
        * _GRAVITY_SIGN
    )
    ring = (1, 1, 1, 1)
    return F.pad(dsdx, ring), F.pad(dsdy, ring), F.pad(dsdr, ring)


def limited_slopes(
    dsigma_dx: torch.Tensor,
    dsigma_dy: torch.Tensor,
    dsigma_dr: torch.Tensor,
    *,
    small_number: float,
    slope_sq_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slopes, squared slope and the cut-off taper (``gmredi_slope_limit.F:461-518``).

    Returns ``(SlopeX, SlopeY, SlopeSqr, taperFct)`` with ``taperFct`` 1, or 0
    where ``SlopeSqr`` reached ``GM_slopeSqCutoff``.  The ``1/dSigmaDr`` and
    ``SIGN(GM_bigSlope, .)`` branches are both evaluated; the divisor is
    replaced by 1 where ``dSigmaDr = 0`` so the discarded branch stays finite
    in reverse mode.
    """
    dsdr = torch.where(
        (dsigma_dr != 0) & (dsigma_dr <= small_number),
        torch.full_like(dsigma_dr, small_number),
        dsigma_dr,
    )
    flat = dsdr == 0
    recip = 1.0 / torch.where(flat, torch.ones_like(dsdr), dsdr)
    big = torch.full_like(dsdr, BIG_SLOPE)
    zero = torch.zeros_like(dsdr)
    slope_x = torch.where(
        flat,
        torch.where(dsigma_dx != 0, torch.copysign(big, dsigma_dx), zero),
        dsigma_dx * recip,
    )
    slope_y = torch.where(
        flat,
        torch.where(dsigma_dy != 0, torch.copysign(big, dsigma_dy), zero),
        dsigma_dy * recip,
    )
    slope_sq = slope_x * slope_x + slope_y * slope_y
    cut = slope_sq >= slope_sq_cutoff
    taper = torch.where(cut, zero, torch.ones_like(dsdr))
    # new_tensor saturates a cutoff beyond the dtype range (default 1e48 in float32) to inf,
    # which only an inf SlopeSqr reaches; full_like would raise before the where selects.
    slope_sq = torch.where(cut, slope_sq.new_tensor(slope_sq_cutoff), slope_sq)
    return slope_x, slope_y, slope_sq, taper


def ac02_slopes(
    dsigma_dx: torch.Tensor,
    dsigma_dy: torch.Tensor,
    dsigma_dr: torch.Tensor,
    *,
    max_slope_sq: float,
    slope_sq_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """'ac02' slopes and taper, its own branch ``gmredi_slope_limit.F:393-439``.

    The true slope without the small-slope approximation (Kxy, Kyx still
    neglected): ``SlopeSqr = |grad_h sigma|^2/|grad sigma|^2``, ``SlopeX =
    dSigmaDx*dSigmaDr/|grad sigma|^2``; all zero where ``|grad sigma| = 0``.
    The taper is gkw91's on that ``SlopeSqr`` and 0 at the cut-off, which
    leaves ``SlopeSqr`` itself unclipped.  Returns ``(SlopeX, SlopeY,
    SlopeSqr, taperFct)``; the divisor is replaced by 1 where it is zero so the
    discarded branch stays finite in reverse mode.
    """
    horiz = dsigma_dx * dsigma_dx + dsigma_dy * dsigma_dy
    mod_sq = horiz + dsigma_dr * dsigma_dr
    flat = mod_sq == 0
    zero = torch.zeros_like(mod_sq)
    recip = 1.0 / torch.where(flat, torch.ones_like(mod_sq), mod_sq)
    slope_sq = torch.where(flat, zero, horiz * recip)
    slope_x = torch.where(flat, zero, dsigma_dx * recip * dsigma_dr)
    slope_y = torch.where(flat, zero, dsigma_dy * recip * dsigma_dr)
    steep = (slope_sq > max_slope_sq) & (slope_sq < slope_sq_cutoff)
    safe = torch.where(steep, slope_sq, torch.ones_like(slope_sq))
    taper = torch.where(
        steep,
        safe.new_tensor(max_slope_sq) / safe,
        torch.where(slope_sq >= slope_sq_cutoff, zero, torch.ones_like(slope_sq)),
    )
    return slope_x, slope_y, slope_sq, taper


def tanh_slope_taper(slope_mod: torch.Tensor, *, s_crit: float, s_d: float) -> torch.Tensor:
    """``op5*(1 + TANH((GM_Scrit - Smod)/GM_Sd))`` (dm95 and ldd97 ``f1``)."""
    return 0.5 * (1.0 + torch.tanh((s_crit - slope_mod) / s_d))


def ldd97_depth_factor(r_nondim: torch.Tensor) -> torch.Tensor:
    """ldd97 ``f2``: 1 for ``Rnondim >= 1``, else ``op5*(1 + SIN(PI*(Rnondim - op5)))``."""
    return torch.where(
        r_nondim >= 1.0,
        torch.ones_like(r_nondim),
        0.5 * (1.0 + torch.sin(math.pi * (r_nondim - 0.5))),
    )


def linear_taper(
    slope_sq: torch.Tensor,
    taper: torch.Tensor,
    *,
    max_slope_sq: float,
    slope_sq_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """'linear' taper, ``gmredi_slope_limit.F:522-537``; returns ``(taperFct, SlopeSqr)``."""
    flat = slope_sq == 0
    steep = (slope_sq > max_slope_sq) & (slope_sq < slope_sq_cutoff)
    safe = torch.where(steep, slope_sq, torch.ones_like(slope_sq))
    taper = torch.where(
        flat,
        torch.ones_like(taper),
        torch.where(steep, torch.sqrt(safe.new_tensor(max_slope_sq) / safe), taper),
    )
    slope_sq = torch.where(steep, torch.clamp(slope_sq, max=BIG_SLOPE * BIG_SLOPE), slope_sq)
    return taper, slope_sq


def gkw91_taper(
    slope_sq: torch.Tensor,
    taper: torch.Tensor,
    *,
    max_slope_sq: float,
    slope_sq_cutoff: float,
) -> torch.Tensor:
    """'gkw91' taper, ``gmredi_slope_limit.F:539-553``."""
    flat = slope_sq == 0
    steep = (slope_sq > max_slope_sq) & (slope_sq < slope_sq_cutoff)
    safe = torch.where(steep, slope_sq, torch.ones_like(slope_sq))
    return torch.where(
        flat,
        torch.ones_like(taper),
        torch.where(steep, safe.new_tensor(max_slope_sq) / safe, taper),
    )


def dm95_taper(
    slope_sq: torch.Tensor,
    taper: torch.Tensor,
    *,
    slope_sq_cutoff: float,
    s_crit: float,
    s_d: float,
) -> torch.Tensor:
    """'dm95' taper, ``gmredi_slope_limit.F:555-568``."""
    flat = slope_sq == 0
    active = ~flat & (slope_sq < slope_sq_cutoff)
    slope_mod = torch.sqrt(torch.where(active, slope_sq, torch.ones_like(slope_sq)))
    return torch.where(
        flat,
        torch.ones_like(taper),
        torch.where(active, tanh_slope_taper(slope_mod, s_crit=s_crit, s_d=s_d), taper),
    )


def ldd97_taper(
    slope_sq: torch.Tensor,
    taper: torch.Tensor,
    *,
    slope_sq_cutoff: float,
    s_crit: float,
    s_d: float,
    r_depth: torch.Tensor,
    l_rho: torch.Tensor,
) -> torch.Tensor:
    """'ldd97' taper, ``gmredi_slope_limit.F:570-592``.

    ``r_depth`` is ``[K, 1, 1]``, positive depth below ``rF(1)`` of the slope
    location; ``l_rho`` ``[NY, NX]`` from :func:`ldd97_rossby_radii`.
    """
    flat = slope_sq == 0
    active = ~flat & (slope_sq < slope_sq_cutoff)
    slope_mod = torch.sqrt(torch.where(active, slope_sq, torch.ones_like(slope_sq)))
    f1 = tanh_slope_taper(slope_mod, s_crit=s_crit, s_d=s_d)
    f2 = ldd97_depth_factor(r_depth / (l_rho * slope_mod))
    return torch.where(flat, torch.ones_like(taper), torch.where(active, f1 * f2, taper))


def _clamped_rossby_radius(f_local: torch.Tensor) -> torch.Tensor:
    nonzero = f_local != 0
    safe = torch.where(nonzero, f_local, torch.ones_like(f_local))
    radius = torch.where(
        nonzero,
        safe.new_tensor(_LDD97_CSPD) / torch.abs(safe),
        torch.full_like(f_local, _LDD97_LRHO_SUP),
    )
    return torch.clamp(radius, min=_LDD97_LRHO_INF, max=_LDD97_LRHO_SUP)


def ldd97_rossby_radii(f_cori: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ldd97 length scales at C, U and V points (``gmredi_calc_tensor.F:163-211``).

    ``f_cori`` is ``fCori`` at cell centres, ``[NY, NX]``.  The first U column
    and the first V row take ``LrhoSup`` as in MITgcm; the V corner
    ``(1-OLx, 1-OLy)``, which MITgcm leaves unset and never reads, takes it too.
    """
    lrho_c = _clamped_rossby_radius(f_cori)
    lrho_w = F.pad(
        _clamped_rossby_radius(0.5 * (f_cori[..., :-1] + f_cori[..., 1:])),
        (1, 0),
        value=_LDD97_LRHO_SUP,
    )
    lrho_s = F.pad(
        _clamped_rossby_radius(0.5 * (f_cori[..., :-1, :] + f_cori[..., 1:, :])),
        (0, 0, 1, 0),
        value=_LDD97_LRHO_SUP,
    )
    return lrho_c, lrho_w, lrho_s
