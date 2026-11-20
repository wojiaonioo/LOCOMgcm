"""Visbeck et al. (1997) variable GM/Redi diffusivity (MITgcm ``GM_VISBECK_VARIABLE_K``).

``VisbeckK = alpha*L^2 * <S*N>``, the depth average from the surface down to
``min(bottom centre, GM_Visbeck_depth)`` (at least ``GM_Visbeck_minDepth``) of
the slope times the buoyancy frequency, accumulated over the W points of
``gmredi_calc_tensor.F:412-503`` and limited to ``[minVal_K, maxVal_K]`` on
MITgcm's inner range (``:579-586``); the outer ring stays 0 (``:155-160``).
Inputs follow the package docstring: W-point gradients from
``sigma_gradients_at_w`` (rows ``k = 2..Nr``), ``sigma_r`` all levels,
positive depths below ``rF(1) = 0`` (z coordinates, no ice-shelf top: the
surface level is the top wet level, ``kSurfC = 1``), m and m^2/s.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# z coordinates: model/src/ini_vertical_grid.F:54.
_GRAVITY_SIGN = -1.0


def visbeck_k(
    dsigma_dx: torch.Tensor,
    dsigma_dy: torch.Tensor,
    sigma_r: torch.Tensor,
    *,
    maskC: torch.Tensor,
    r_depth_c: torch.Tensor,
    drc: torch.Tensor,
    alpha: float,
    length: float,
    depth: float,
    min_depth: float,
    max_slope: float,
    min_val_k: float,
    max_val_k: float,
    gravity: float,
    rho_const: float,
) -> torch.Tensor:
    """``VisbeckK`` ``[NY, NX]`` (``gmredi_calc_tensor.F:151-160,412-503,579-586``).

    ``dsigma_dx``, ``dsigma_dy`` ``[Nr-1, NY, NX]`` (row ``k-2`` for interface
    ``k``, zero outside the inner ring); ``sigma_r``, ``maskC`` ``[Nr, NY, NX]``;
    ``r_depth_c`` ``[Nr, 1, 1]`` is ``-rC``; ``drc`` ``[Nr, 1, 1]`` is
    ``drC(1..Nr)``.  ``kLowC`` is the deepest wet level of the column in ``maskC``.
    The ``k`` loop runs from ``Nr`` down to 2 as MITgcm sums.
    """
    nr = sigma_r.shape[0]
    inner = (slice(1, -1), slice(1, -1))
    recip_max_slope = 1.0 / max_slope if max_slope > 0.0 else 0.0
    n2_factor = gravity * (1.0 / rho_const)
    # :441-446 with rTop = Ro_surf = 0: rTop - rC(kLowC) is the bottom centre depth.
    k_low = torch.clamp(maskC.sum(dim=0) - 1.0, min=0.0).to(torch.long)
    bottom_centre_depth = r_depth_c.reshape(-1)[k_low]
    integr = torch.clamp(torch.clamp(bottom_centre_depth[inner], max=depth), min=min_depth)
    # rTop - rC(ks) with ks = 1.
    denom = integr - r_depth_c[0]
    visbeck = torch.zeros_like(integr)
    for k in range(nr - 1, 0, -1):  # MITgcm interface k+1 = Nr..2, the top of level k
        mask_fk = maskC[k - 1] * maskC[k]
        dsdr = torch.clamp(_GRAVITY_SIGN * sigma_r[k], min=0.0)
        # :448-454: distance from the centre above to the integration depth, at most drC(k).
        delta_h = torch.minimum(integr - r_depth_c[k - 1], drc[k]) / denom
        # :459-469: five-point average of the stratification.
        d_sigma_r = (
            dsdr[inner] * 4.0 + dsdr[1:-1, :-2] + dsdr[1:-1, 2:] + dsdr[:-2, 1:-1] + dsdr[2:, 1:-1]
        ) / (4.0 + mask_fk[1:-1, :-2] + mask_fk[1:-1, 2:] + mask_fk[:-2, 1:-1] + mask_fk[2:, 1:-1])
        d_sigma_h_sq = (
            dsigma_dx[k - 1][inner] * dsigma_dx[k - 1][inner]
            + dsigma_dy[k - 1][inner] * dsigma_dy[k - 1][inner]
        )
        sloped = d_sigma_h_sq > 0.0
        d_sigma_h = torch.sqrt(torch.where(sloped, d_sigma_h_sq, torch.ones_like(d_sigma_h_sq)))
        # :482-486: S = |grad_h sigma|/dSigmaR, at most GM_Visbeck_maxSlope.
        steep = d_sigma_r > d_sigma_h * recip_max_slope
        s_loc = torch.where(
            steep,
            d_sigma_h / torch.where(steep, d_sigma_r, torch.ones_like(d_sigma_r)),
            torch.full_like(d_sigma_r, max_slope),
        )
        n2 = n2_factor * d_sigma_r
        stratified = n2 > 0.0
        sn = torch.where(
            sloped & stratified,
            s_loc * torch.sqrt(torch.where(stratified, n2, torch.ones_like(n2))),
            torch.zeros_like(n2),
        )
        # :416-417: only levels whose centre above is shallower than GM_Visbeck_depth.
        active = (mask_fk[inner] != 0) & (r_depth_c[k - 1] < depth)
        visbeck = torch.where(active, visbeck + delta_h * alpha * length * length * sn, visbeck)
    visbeck = torch.clamp(torch.clamp(visbeck, min=min_val_k), max=max_val_k)
    return F.pad(visbeck, (1, 1, 1, 1))
