"""Pacanowski and Philander (1981) vertical mixing (MITgcm ``pkg/pp81``).

PP81_CALC (``pkg/pp81/pp81_calc.F``) turns the gradient Richardson number of
``richardson.richardson_number`` into the PP81 viscosity ``PPviscAr`` and
diffusivity ``PPdiffKr``; ``pp81_ri_limit`` is the RiLimit that
PP81_READPARMS derives (``pkg/pp81/pp81_readparms.F``).  The merges into
KappaRT, KappaRS, KappaRU and KappaRV are in ``coefficients``.

Outputs are centre interface fields [B, nz, ny, nx]: index k is the top
interface of layer k (MITgcm level K = k + 1).  Index 0 holds what
PP81_INIT_VARIA gives every level and PP81_CALC never overwrites
(``pp81_init_varia.F:39-40``): viscArNr(1) and diffKrNrS(1).  ``ri`` covers
interfaces 1..nz-1, shape [B, nz - 1, ny, nx].  ``mask_c`` is the 0/1 centre
mask [nz, ny, nx]; ``visc_arnr`` and ``diffkr_nr_s`` are the background
columns [nz].  Coefficients are m^2/s.  dtype follows the inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: CPP options of ``PP81_OPTIONS.h`` this port accepts.  ALLOW_PP81_LOWERBOUND
#: and PP81_SMOOTH_RI (``#undef`` at ``pkg/pp81/PP81_OPTIONS.h:18,21``, defined
#: by no verification build) are rejected.
PP81_CPP_OPTIONS = ()


@dataclass(frozen=True)
class Pp81Parameters:
    """PP81_PARM01 values the kernels read (``pp81_readparms.F:35-44``): PPnu0, PPalpha, PPnRi, PPviscMax."""

    nu0: float
    alpha: float
    n_ri: int
    visc_max: float


def pp81_ri_limit(*, nu0: float, visc_arnr_surface: float, visc_max: float,
                  n_ri: int, alpha: float) -> float:
    """RiLimit, ``pp81_readparms.F:117-123``; RiLimit is not a namelist entry.

    ``RiLimit = PPnRi`` as a real, then
    ``(((PPnu0 + viscArNr(1))/PPviscMax)**(1/RiLimit) - 1)/PPalpha``.
    """
    ri_limit = float(n_ri)
    return (((nu0 + visc_arnr_surface) / visc_max) ** (1.0 / ri_limit) - 1.0) / alpha


def _integer_power(x: torch.Tensor, n: int) -> torch.Tensor:
    # x**n for a variable integer n >= 1 in the square-and-multiply order of
    # libgcc __powidf2, which gfortran calls; for n <= 3 it equals repeated
    # multiplication.
    y = x if n % 2 else 1.0
    n //= 2
    while n:
        x = x * x
        if n % 2:
            y = y * x
        n //= 2
    return y


def pp81_coefficients(
    ri: torch.Tensor,
    *,
    mask_c: torch.Tensor,
    ri_limit: float,
    alpha: float,
    nu0: float,
    n_ri: int,
    visc_max: float,
    visc_arnr: torch.Tensor,
    diffkr_nr_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(PPviscAr, PPdiffKr)`` of PP81_CALC, ``pp81_calc.F:66-117``.

    Below RiLimit: ``denom = 1 + PPalpha*RiLimit`` and ``PPviscTmp =
    PPviscMax``; otherwise ``denom = 1 + PPalpha*Ri`` and ``PPviscTmp =
    PPnu0/denom**PPnRi`` (``:73-79``).  ``PPviscAr = max(PPviscTmp,
    viscArNr(k))``, ``PPdiffKr = max(PPviscAr/denom, diffKrNrS(k))``
    (``:81-87``), both times ``maskC(K)`` (``:108-115``).
    """
    below = ri < ri_limit
    denom = torch.where(below, torch.full_like(ri, 1.0 + alpha * ri_limit), 1.0 + alpha * ri)
    # a tensor numerator keeps MITgcm's true division; torch evaluates
    # ``float / tensor`` as a reciprocal times the float (one ulp apart)
    visc_tmp = torch.where(below, torch.full_like(ri, visc_max),
                           denom.new_full((), nu0) / _integer_power(denom, n_ri))
    visc = torch.maximum(visc_tmp, visc_arnr[1:].reshape(-1, 1, 1))
    diff = torch.maximum(visc / denom, diffkr_nr_s[1:].reshape(-1, 1, 1))
    visc = visc * mask_c[1:]
    diff = diff * mask_c[1:]
    surface = (visc.shape[0], 1, *visc.shape[2:])
    visc = torch.cat((visc_arnr[0].expand(surface), visc), dim=1)
    diff = torch.cat((diffkr_nr_s[0].expand(surface), diff), dim=1)
    return visc, diff
