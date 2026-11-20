"""Vertical density gradient ``sigmaR`` shared by the vertical mixing schemes.

The vertical component of MITgcm GRAD_SIGMA (``model/src/grad_sigma.F``),
which ivdc_kappa (CALC_IVDC) and GGL90 read.  Centre fields are
[B, nz, ny, nx] with level 0 at the surface; the result has the same shape,
index k is the top interface of layer k (MITgcm level K = k + 1) and index
0 holds zero.  Densities are kg/m^3 from one equation-of-state family; only
their difference enters.  ``recip_drc`` is [nz], index k the reciprocal
distance in metres between centres k - 1 and k (MITgcm ``recip_drC(K)``),
index 0 unread.  ``mask_c`` is the 0/1 centre mask [nz, ny, nx].  ``sigma_r``
keeps the MITgcm sign: r points up (rkSign = -1), a stable column has
``sigma_r < 0``.  dtype follows the inputs.
"""
from __future__ import annotations

import torch

#: ``ini_vertical_grid.F:53-54`` for z coordinates.
RK_SIGN = -1.0
GRAVITY_SIGN = -1.0


def vertical_density_gradient(
    rho_in_situ: torch.Tensor,
    rho_upper_at_lower: torch.Tensor,
    *,
    mask_c: torch.Tensor,
    recip_drc: torch.Tensor,
) -> torch.Tensor:
    """``sigmaR`` of GRAD_SIGMA, ``grad_sigma.F:84-98``.

    ``rho_in_situ`` [B, nz, ny, nx] is each layer at its own reference level
    (``rhoInSitu``, the ``rhoKp1`` of ``do_oceanic_phys.F:815``).
    ``rho_upper_at_lower`` [B, nz - 1, ny, nx] holds, at index k - 1, layer
    k - 1 evaluated at the reference level of layer k (``FIND_RHO_2D`` with
    kRef = K, the ``rhoKm1`` of ``do_oceanic_phys.F:818-823``).  Interface 0
    is zero (``:84-89``); interfaces 1..nz-1 are
    ``maskC(K)*maskC(K-1)*recip_drC(K)*rkSign*(rhoKp1 - rhoKm1)`` multiplied
    left to right (``:91-97``).
    """
    inner = (
        mask_c[1:] * mask_c[:-1] * recip_drc[1:].reshape(-1, 1, 1) * RK_SIGN
        * (rho_in_situ[:, 1:] - rho_upper_at_lower)
    )
    surface = inner.new_zeros((inner.shape[0], 1, *inner.shape[2:]))
    return torch.cat((surface, inner), dim=1)
