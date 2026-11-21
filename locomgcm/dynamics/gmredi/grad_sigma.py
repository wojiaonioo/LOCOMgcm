"""Density gradients that feed the GM/Redi slopes (MITgcm ``model/src/grad_sigma.F``).

Input ``rho_in_situ`` is ``rhoInSitu`` at each level's own reference pressure;
``rho_upper_at_level[k-1]`` (MIT ``k >= 2``) is the density of level ``k-1``
water at level ``k`` reference pressure (``do_oceanic_phys.F:811-823``, z
coordinates).  ``sigmaX`` sits at U points, ``sigmaY`` at V points, ``sigmaR``
at W points in MIT sign (``rkSign = -1``, negative when stable).  Units are
density per metre.  Index conventions are those of the package docstring.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# z coordinates: model/src/ini_vertical_grid.F:53.
_RK_SIGN = -1.0


def grad_sigma(
    rho_in_situ: torch.Tensor,
    rho_upper_at_level: torch.Tensor,
    *,
    maskC: torch.Tensor,
    maskW: torch.Tensor,
    maskS: torch.Tensor,
    recip_dxC: torch.Tensor,
    recip_dyC: torch.Tensor,
    recip_drC: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MITgcm ``grad_sigma.F:51-98`` for all levels (called from ``do_oceanic_phys.F:803-845``).

    Shapes: ``rho_in_situ``, ``rho_upper_at_level``, ``maskC``, ``maskW``,
    ``maskS`` are ``[Nr, NY, NX]``; ``recip_dxC``, ``recip_dyC`` ``[NY, NX]``;
    ``recip_drC`` ``[Nr, 1, 1]``.  Row 0 of ``rho_upper_at_level`` is not read.
    Returns ``(sigmaX, sigmaY, sigmaR)``, each ``[Nr, NY, NX]``.  The first
    column of ``sigmaX``, the first row of ``sigmaY`` and level 1 of
    ``sigmaR`` stay zero, as after ``do_oceanic_phys.F:620-627``.
    """
    # grad_sigma.F:63-69 (recip_deepFacC = 1).
    sigma_x = F.pad(
        maskW[..., 1:] * recip_dxC[..., 1:] * (rho_in_situ[..., 1:] - rho_in_situ[..., :-1]),
        (1, 0),
    )
    # grad_sigma.F:76-82.
    sigma_y = F.pad(
        maskS[..., 1:, :]
        * recip_dyC[..., 1:, :]
        * (rho_in_situ[..., 1:, :] - rho_in_situ[..., :-1, :]),
        (0, 0, 1, 0),
    )
    # grad_sigma.F:84-98: sigmaR(k=1) = 0; rhoKp1 = rhoInSitu(k), rhoKm1 = level k-1 at p(k).
    sigma_r = F.pad(
        maskC[1:]
        * maskC[:-1]
        * recip_drC[1:]
        * _RK_SIGN
        * (rho_in_situ[1:] - rho_upper_at_level[1:]),
        (0, 0, 0, 0, 1, 0),
    )
    return sigma_x, sigma_y, sigma_r
