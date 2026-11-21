"""Klymak and Legg (2010) vertical mixing (MITgcm ``pkg/kl10``).

KL10_CALC (``pkg/kl10/kl10_calc.F``) rebuilds each column's density from
the surface in-situ density and ``sigmaR``, sorts it into a stable profile
(insertion sort of the wet levels) and takes the Ozmidov-scale viscosity
``0.2*dz^2*N`` from the displacement ``dz`` of every sorted parcel.  The
merges into KappaRT, KappaRS, KappaRU and KappaRV are in ``coefficients``
(``add_scheme_coefficient`` and ``add_centre_viscosity``, the forms of
``kl10_calc_diff.F`` and ``kl10_calc_visc.F``).  The dissipation ``KLeps``
is a diagnostic only and is not formed.

Outputs ``KLviscAr`` and ``KLdiffKr`` are centre interface fields
[B, nz, ny, nx]: index k is the top interface of layer k (MITgcm level
K = k + 1); index 0 is ``viscArNr(1)`` and ``max(viscArNr(1), diffKrNrS(1))``
(``kl10_calc.F:89-90, 128-134``).  Neither is multiplied by ``maskC``, as in
MITgcm.  ``rho_surface`` [B, ny, nx] is rhoInSitu of layer 1, kg/m^3;
``sigma_r`` [B, nz, ny, nx] comes from ``density_gradient.vertical_density_gradient``
(MITgcm sign, index 0 unread); ``mask_c`` is the 0/1 centre mask
[nz, ny, nx].  Vertical metrics are positive metres [nz]: ``drc[k]`` and
``recip_drc[k]`` the distance between centres k - 1 and k and its
reciprocal (index 0 unread), ``center_depth`` the depth of each centre
accumulated as ``ini_vertical_grid.F:144-146`` (MITgcm ``rC = -center_depth``).
Background columns ``visc_arnr``, ``diffkr_nr_s`` are [nz] in m^2/s.  dtype
follows the inputs.
"""
from __future__ import annotations

import numpy as np
import torch

from locomgcm.dynamics.viscosity_coefficients import _safe_sqrt_zero_grad

from .density_gradient import RK_SIGN

#: CPP options of ``KL10_OPTIONS.h`` this port accepts (the package header
#: defines none).
KL10_CPP_OPTIONS = ()

#: The mixing efficiency ``0.2`` of ``kl10_calc.F:162`` is a default-real
#: (single precision) literal in a build without ``-r8``; MITgcm multiplies
#: by its double promotion.
MIXING_EFFICIENCY = float(np.float32(0.2))


def _sorted_profile(rho: torch.Tensor, depth: torch.Tensor, wet: torch.Tensor):
    # kl10_calc.F:99-117: for K = 2..Nr, a wet level lighter than the one
    # above moves up past every level it is lighter than (strict <, so equal
    # densities keep their order); rhoS(JJ+1:K) and RS(JJ+1:K) rotate down by
    # one.  The levels above K are already sorted, so the insertion point is
    # the count of them with rhoS <= rhoS(K).
    nz = rho.shape[1]
    for k in range(1, nz):
        x = rho[:, k:k + 1]
        move = (x < rho[:, k - 1:k]) & wet[:, k:k + 1]
        p = (rho[:, :k] <= x).sum(dim=1, keepdim=True)
        j = torch.arange(k + 1, device=rho.device).reshape(1, -1, 1, 1)
        source = torch.where(j < p, j, torch.where(j == p, torch.full_like(j, k), j - 1))
        source = torch.where(move, source, j.expand_as(source))
        rho = torch.cat((torch.gather(rho[:, :k + 1], 1, source), rho[:, k + 1:]), dim=1)
        depth = torch.cat((torch.gather(depth[:, :k + 1], 1, source), depth[:, k + 1:]), dim=1)
    return rho, depth


def kl10_coefficients(
    rho_surface: torch.Tensor,
    sigma_r: torch.Tensor,
    *,
    mask_c: torch.Tensor,
    drc: torch.Tensor,
    recip_drc: torch.Tensor,
    center_depth: torch.Tensor,
    gravity: float,
    mass2r_unit: float,
    visc_max: float,
    visc_arnr: torch.Tensor,
    diffkr_nr_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(KLviscAr, KLdiffKr)`` of ``kl10_calc.F:84-236``.

    ``rhoS(K) = rhoS(K-1) + rkSign*drC(K)*sigmaR(K)`` from ``rhoS(1) =
    rhoInSitu(1)`` (``:85-97``), sorted with the centre depths ``RS``
    (``:99-117``).  Then, from ``b0 = max(-g*mass2rUnit*(rhoS(1) -
    rhoS(2))*recip_drC(2), 0)`` (``:136-137``), for each level the face
    ``N^2`` below, ``buoyFreqf = max(-g*mass2rUnit*(rhoS(K) - rhoS(K+1))
    *recip_drC(K+1), 0)`` (at K = Nr the face above, ``:146-157``), the
    centre ``N^2`` ``(buoyFreqf + b0)*0.5`` and ``KLviscTmp = max(viscArNr(K),
    0.2*(RS(K) - rC(K))**2*sqrt(N^2))`` (``:158-163``); for K >= 2 the face
    value ``min(KLviscMax, 0.5*(KLviscTmp + KLviscTmp(K-1)))`` raised to
    ``viscArNr(K)`` and to ``diffKrNrS(K)`` (``:165-180``).  ``KLviscTmp(1)``'s
    predecessor is ``viscArNr(1)`` (``:89-90``).
    """
    batch, nz, ny, nx = sigma_r.shape
    steps = [rho_surface]
    for k in range(1, nz):
        steps.append(steps[k - 1] + (RK_SIGN * drc[k]) * sigma_r[:, k])
    wet = (mask_c > 0).unsqueeze(0)
    depth = center_depth.reshape(1, nz, 1, 1).expand(batch, nz, ny, nx)
    rho_s, rs = _sorted_profile(torch.stack(steps, dim=1), depth, wet)

    def face_n2(upper, lower, r):
        return torch.clamp(-(gravity * mass2r_unit * (upper - lower) * r), min=0.0)

    visc0 = visc_arnr[0].expand(batch, ny, nx)
    visc = [visc0]
    diff = [torch.maximum(visc0, diffkr_nr_s[0])]
    b0 = face_n2(rho_s[:, 0], rho_s[:, 1], recip_drc[1])
    visc_old = visc0
    for k in range(nz):
        if k < nz - 1:
            buoy_f = face_n2(rho_s[:, k], rho_s[:, k + 1], recip_drc[k + 1])
        else:
            buoy_f = face_n2(rho_s[:, k - 1], rho_s[:, k], recip_drc[k])
        buoy_c = (buoy_f + b0) * 0.5
        dz = rs[:, k] - center_depth[k]
        visc_tmp = torch.maximum(visc_arnr[k], MIXING_EFFICIENCY * dz * dz * _safe_sqrt_zero_grad(buoy_c))
        if k > 0:
            k_temp = torch.clamp(0.5 * (visc_tmp + visc_old), max=visc_max)
            visc.append(torch.maximum(k_temp, visc_arnr[k]))
            diff.append(torch.maximum(k_temp, diffkr_nr_s[k]))
        b0 = buoy_f
        visc_old = visc_tmp
    return torch.stack(visc, dim=1), torch.stack(diff, dim=1)
