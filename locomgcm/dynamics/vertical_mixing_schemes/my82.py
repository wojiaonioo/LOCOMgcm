"""Mellor and Yamada (1982) level-2 vertical mixing (MITgcm ``pkg/my82``).

MY82_CALC (``pkg/my82/my82_calc.F``) turns the Richardson number and its two
factors from ``richardson`` into the MY82 viscosity ``MYviscAr``, diffusivity
``MYdiffKr`` and boundary length scale ``MYhbl``.  The merges into KappaRT,
KappaRS, KappaRU and KappaRV are in ``coefficients``.

``ri``, ``minus_n2`` (MITgcm's ``-N^2``) and ``shear_squared`` cover
interfaces 1..nz-1, shape [B, nz - 1, ny, nx]; outputs are [B, nz, ny, nx]
with index k the top interface of layer k (MITgcm level K = k + 1), and
``hbl`` is [B, ny, nx] in metres, positive.  ``interface_depth`` is LOCOMgcm's
positive depth of the top interface of each layer [nz] in metres; MITgcm's
``rF = -interface_depth`` is formed here.  ``mask_c`` is the 0/1 centre mask
[nz, ny, nx]; ``visc_arnr`` and ``diffkr_nr_s`` are the background columns
[nz].  Coefficients are m^2/s.  dtype follows the inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.dynamics.viscosity_coefficients import _safe_sqrt_zero_grad

#: CPP options of ``MY82_OPTIONS.h`` this port accepts.  MY82_SMOOTH_RI
#: (``#undef`` at ``pkg/my82/MY82_OPTIONS.h:18``, defined by no verification
#: build) is rejected.
MY82_CPP_OPTIONS = ()

#: ``MY82.h:36-40`` parameters of Mellor and Yamada (1982).
A1 = 0.92
A2 = 0.74
B1 = 16.6
B2 = 10.1
C1 = 0.08
#: RiMax, set in ``my82_readparms.F:64`` and not a namelist entry.
RI_MAX = 0.1950

#: ``my82_init_varia.F:32-39``, in its operation order.
_GAM1 = 1.0 / 3.0 - 2.0 * A1 / B1
_GAM2 = (B2 + 6.0 * A1) / B1
ALPHA1 = 3.0 * A2 * _GAM1
ALPHA2 = 3.0 * A2 * (_GAM1 + _GAM2)
BETA1 = A1 * B1 * (_GAM1 - C1)
BETA2 = A1 * (B1 * (_GAM1 - C1) + 6.0 * A1 + 3.0 * A2)
BETA3 = A2 * B1 * _GAM1
BETA4 = A2 * (B1 * (_GAM1 + _GAM2) - 3.0 * A1)


@dataclass(frozen=True)
class My82Parameters:
    """MY_PARM01 values the kernels read (``my82_readparms.F:35-41``): MYhblScale, MYviscMax, MYdiffMax."""

    hbl_scale: float
    visc_max: float
    diff_max: float


def my82_coefficients(
    ri: torch.Tensor,
    minus_n2: torch.Tensor,
    shear_squared: torch.Tensor,
    *,
    mask_c: torch.Tensor,
    interface_depth: torch.Tensor,
    hbl_scale: float,
    visc_max: float,
    diff_max: float,
    visc_arnr: torch.Tensor,
    diffkr_nr_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(MYviscAr, MYdiffKr, MYhbl)`` of MY82_CALC, ``my82_calc.F:74-179``.

    First K loop (K = 2..Nr, ``:92-120``): ``RiTmp = min(Ri, RiMax)``, the
    flux Richardson number, the stability functions ``SH``, ``SM`` and
    ``tke = sqrt(max(0, B1*(SH*GH + SM*GM)))`` with ``GH = -N^2``,
    ``GM = shear^2``.  Second loop (``:126-141``): ``GM = sum tke*rF``,
    ``GH = sum tke`` in K order.  ``MYhbl = -GM/GH*MYhblScale``, zero where
    ``GH = 0`` (``:143-151``).  Third loop over all levels (``:154-179``):
    ``MYhbl*(MYhbl*tke)*SM`` and ``*SH``, floored by the background, capped
    by MYviscMax / MYdiffMax and times ``maskC``.  ``SH``, ``SM`` and ``tke``
    are zero at the surface level (``:81-89``).
    """
    ri_tmp = torch.clamp(ri, max=RI_MAX)
    btmp = BETA1 + BETA4 * ri_tmp
    ri_flux = (btmp - torch.sqrt(btmp * btmp - 4.0 * BETA2 * BETA3 * ri_tmp)) / (2.0 * BETA2)
    sh_tmp = (ALPHA1 - ALPHA2 * ri_flux) / (1.0 - ri_flux)
    sm_inner = sh_tmp * (BETA1 - BETA2 * ri_flux) / (BETA3 - BETA4 * ri_flux)
    # double-where keeps sqrt's infinite slope at zero energy out of the gradient
    tke_inner = _safe_sqrt_zero_grad(torch.clamp(B1 * (sh_tmp * minus_n2 + sm_inner * shear_squared), min=0.0))

    r_f = -interface_depth
    gm = tke_inner.new_zeros((tke_inner.shape[0], *tke_inner.shape[2:]))
    gh = tke_inner.new_zeros((tke_inner.shape[0], *tke_inner.shape[2:]))
    for k in range(1, interface_depth.shape[0]):
        # MITgcm accumulates in K order (my82_calc.F:134-141).
        gm = gm + tke_inner[:, k - 1] * r_f[k]
        gh = gh + tke_inner[:, k - 1]
    calm = gh == 0.0
    # double-where keeps the 0/0 of the unselected branch out of the gradient
    gh_safe = torch.where(calm, torch.ones_like(gh), gh)
    hbl = torch.where(calm, torch.zeros_like(gh), -gm / gh_safe * hbl_scale)

    zero = tke_inner.new_zeros((tke_inner.shape[0], 1, *tke_inner.shape[2:]))
    sh = torch.cat((zero, sh_tmp), dim=1)
    sm = torch.cat((zero, sm_inner), dim=1)
    tke = torch.cat((zero, tke_inner), dim=1)
    hbl_k = hbl.unsqueeze(1)
    tkel = hbl_k * tke
    visc = torch.maximum(hbl_k * tkel * sm, visc_arnr.reshape(-1, 1, 1))
    diff = torch.maximum(hbl_k * tkel * sh, diffkr_nr_s.reshape(-1, 1, 1))
    visc = torch.clamp(visc, max=visc_max) * mask_c
    diff = torch.clamp(diff, max=diff_max) * mask_c
    return visc, diff, hbl
