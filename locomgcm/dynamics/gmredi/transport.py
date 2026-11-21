"""GM/Redi tracer fluxes and vertical diffusivity (MITgcm ``gmredi_x/y/rtransport.F``, ``gmredi_calc_diff.F``).

Each ``add_*`` kernel takes the diffusive flux ``df`` accumulated so far in
``gad_calc_rhs.F`` (area-integrated, tracer*m^3/s, ``[Nr, NY, NX]``) and
returns it with the GM/Redi term subtracted, in MITgcm's order, so the
caller keeps MITgcm's accumulation sequence.  ``tracer`` is ``TracerN``
(time level n).  x fluxes sit at U points, y fluxes at V points, r fluxes at
W points and are positive upward (+r); rows of r fluxes are interfaces.
Points without the full stencil keep ``df`` unchanged; that range is a
superset of MITgcm's ``[iMin:iMax(+1)]`` (``temp_integrate.F:160-163``),
equal to it on the points the interior divergence reads.  Index and sign
conventions are those of the package docstring.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .levels import below_bottom_mask, mask_up, next_level, previous_level


def add_x_diagonal_flux(
    df: torch.Tensor,
    tracer: torch.Tensor,
    *,
    kux: torch.Tensor,
    xA: torch.Tensor,
    recip_dxC: torch.Tensor,
) -> torch.Tensor:
    """``gmredi_xtransport.F:129-137``; ``xA = dyG*drF*hFacW`` ``[Nr, NY, NX]``, ``recip_dxC`` ``[NY, NX]``."""
    flux = xA[..., 1:] * kux[..., 1:] * recip_dxC[..., 1:] * (tracer[..., 1:] - tracer[..., :-1])
    return df - F.pad(flux, (1, 0))


def add_y_diagonal_flux(
    df: torch.Tensor,
    tracer: torch.Tensor,
    *,
    kvy: torch.Tensor,
    yA: torch.Tensor,
    recip_dyC: torch.Tensor,
) -> torch.Tensor:
    """``gmredi_ytransport.F:129-137``; ``yA = dxG*drF*hFacS``."""
    flux = (
        yA[..., 1:, :]
        * kvy[..., 1:, :]
        * recip_dyC[..., 1:, :]
        * (tracer[..., 1:, :] - tracer[..., :-1, :])
    )
    return df - F.pad(flux, (0, 0, 1, 0))


def _vertical_gradient_terms(
    tracer: torch.Tensor,
    maskC: torch.Tensor,
    recip_drC: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-column pieces of the off-diagonal ``dTdz`` (``gmredi_xtransport.F:144-166``).

    Returns ``(upper, lower, recip_drC(k), recip_drC(kp1))`` where a column
    contributes ``upper = maskFk*(T(km1)-T(k))`` and
    ``lower = maskC(k)*maskC(kp1)*maskp1*(T(k)-T(kp1))``.
    """
    upper = mask_up(maskC) * (previous_level(tracer) - tracer)
    lower = maskC * next_level(maskC) * below_bottom_mask(tracer) * (tracer - next_level(tracer))
    return upper, lower, recip_drC, next_level(recip_drC)


def add_x_offdiagonal_flux(
    df: torch.Tensor,
    tracer: torch.Tensor,
    *,
    kuz: torch.Tensor,
    xA: torch.Tensor,
    maskC: torch.Tensor,
    recip_drC: torch.Tensor,
) -> torch.Tensor:
    """``gmredi_xtransport.F:142-192`` (XD defined and ``GM_ExtraDiag``); ``recip_drC`` ``[Nr, 1, 1]``."""
    upper, lower, rdrc_k, rdrc_kp1 = _vertical_gradient_terms(tracer, maskC, recip_drC)
    dtdz = 0.5 * (
        0.5 * rdrc_k * (upper[..., :-1] + upper[..., 1:])
        + 0.5 * rdrc_kp1 * (lower[..., :-1] + lower[..., 1:])
    )
    return df - F.pad(xA[..., 1:] * kuz[..., 1:] * dtdz, (1, 0))


def add_y_offdiagonal_flux(
    df: torch.Tensor,
    tracer: torch.Tensor,
    *,
    kvz: torch.Tensor,
    yA: torch.Tensor,
    maskC: torch.Tensor,
    recip_drC: torch.Tensor,
) -> torch.Tensor:
    """``gmredi_ytransport.F:142-192`` (XD defined and ``GM_ExtraDiag``)."""
    upper, lower, rdrc_k, rdrc_kp1 = _vertical_gradient_terms(tracer, maskC, recip_drC)
    dtdz = 0.5 * (
        0.5 * rdrc_k * (upper[..., :-1, :] + upper[..., 1:, :])
        + 0.5 * rdrc_kp1 * (lower[..., :-1, :] + lower[..., 1:, :])
    )
    return df - F.pad(yA[..., 1:, :] * kvz[..., 1:, :] * dtdz, (0, 0, 1, 0))


def add_r_offdiagonal_flux(
    df: torch.Tensor,
    tracer: torch.Tensor,
    *,
    kwx: torch.Tensor,
    kwy: torch.Tensor,
    maskC: torch.Tensor,
    maskW: torch.Tensor,
    maskS: torch.Tensor,
    recip_dxC: torch.Tensor,
    recip_dyC: torch.Tensor,
    rA: torch.Tensor,
    maskInC: torch.Tensor,
) -> torch.Tensor:
    """``gmredi_rtransport.F:91-153``: interfaces ``k = 2..Nr``, inner ring.

    ``df`` here is the explicit r flux before GM/Redi (zero under
    ``implicitDiffusion`` unless ``diffKr4`` adds to it, ``gad_calc_rhs.F:605-621``).
    """
    rdx = recip_dxC[..., 1:-1, :]
    rdy = recip_dyC[..., :, 1:-1]

    def ddx(t: torch.Tensor, mw: torch.Tensor) -> torch.Tensor:
        t = t[..., 1:-1, :]
        mw = mw[..., 1:-1, :]
        return (
            mw[..., 2:] * rdx[..., 2:] * (t[..., 2:] - t[..., 1:-1])
            + mw[..., 1:-1] * rdx[..., 1:-1] * (t[..., 1:-1] - t[..., :-2])
        )

    def ddy(t: torch.Tensor, ms: torch.Tensor) -> torch.Tensor:
        t = t[..., :, 1:-1]
        ms = ms[..., :, 1:-1]
        return (
            ms[..., 2:, :] * rdy[..., 2:, :] * (t[..., 2:, :] - t[..., 1:-1, :])
            + ms[..., 1:-1, :] * rdy[..., 1:-1, :] * (t[..., 1:-1, :] - t[..., :-2, :])
        )

    t_k, t_km1 = tracer[1:], tracer[:-1]
    dtdx = 0.5 * (0.5 * ddx(t_k, maskW[1:]) + 0.5 * ddx(t_km1, maskW[:-1]))
    dtdy = 0.5 * (0.5 * ddy(t_k, maskS[1:]) + 0.5 * ddy(t_km1, maskS[:-1]))
    inner = (..., slice(1, -1), slice(1, -1))
    flux = (
        rA[inner]
        * maskInC[inner]
        * (kwx[1:][inner] * dtdx + kwy[1:][inner] * dtdy)
        * mask_up(maskC)[1:][inner]
    )
    return df - F.pad(flux, (1, 1, 1, 1, 1, 0))


def add_kwz_to_kappa(kappa: torch.Tensor, kwz: torch.Tensor, *, maskInC: torch.Tensor) -> torch.Tensor:
    """``gmredi_calc_diff.F:49-67`` (``kArg = 0``): ``KappaRx + Kwz*maskInC`` at every interface.

    ``kappa`` is MITgcm's ``kappaRk`` ``[Nr, NY, NX]`` (row ``k-1`` is
    interface ``k``), the same for T and S.  ``Kwz`` is zero outside the
    inner ring, so adding it over the whole array matches MITgcm's
    ``[iMin:iMax]`` range when ``ol = 2``.
    """
    return kappa + kwz * maskInC
