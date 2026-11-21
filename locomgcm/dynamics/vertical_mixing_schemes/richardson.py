"""Gradient Richardson number shared by PP81 and MY82.

PP81_RI_NUMBER (``pkg/pp81/pp81_ri_number.F``) and MY82_RI_NUMBER
(``pkg/my82/my82_ri_number.F``) are the same computation; MY82 also hands
out the two factors.  Their ``*_SMOOTH_RI`` options are not covered.

Outputs live on interfaces 1..nz-1 only, shape [B, nz - 1, ny, nx], index
k - 1 for the top interface of layer k (MITgcm level K = k + 1); MITgcm
never forms them at the surface.  ``u`` [B, nz, ny, nx + 1] and ``v``
[B, nz, ny + 1, nx] carry both outer faces as in the LOCOMgcm state.
Densities are kg/m^3 from one equation-of-state family, both evaluated at
the reference level of the lower layer.  ``recip_drc`` is [nz], index k the
reciprocal distance in metres between centres k - 1 and k, positive.  The
buoyancy factor keeps MITgcm's sign, ``-N^2`` (negative when stable).
dtype follows the inputs.
"""
from __future__ import annotations

import torch

#: ``epsilon`` of ``pp81_ri_number.F:57`` and ``my82_ri_number.F:62``.
SHEAR_FLOOR = 1.0e-10


def vertical_shear_squared(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    recip_drc: torch.Tensor,
) -> torch.Tensor:
    """``(du/dz)^2 + (dv/dz)^2`` of the centre velocity, ``pp81_ri_number.F:96-104``, ``my82_ri_number.F:99-107``.

    ``tempu = .5*(uVel(I,Km1)+uVel(I+1,Km1) - (uVel(I,K)+uVel(I+1,K)))*recip_drC(K)``,
    ``tempv`` likewise with ``vVel(J)``, ``vVel(J+1)``, then
    ``tempu*tempu + tempv*tempv``.  Unmasked, as in MITgcm.
    """
    r = recip_drc[1:].reshape(-1, 1, 1)
    u_sum = u[..., :-1] + u[..., 1:]
    v_sum = v[..., :-1, :] + v[..., 1:, :]
    tempu = 0.5 * (u_sum[:, :-1] - u_sum[:, 1:]) * r
    tempv = 0.5 * (v_sum[:, :-1] - v_sum[:, 1:]) * r
    return tempu * tempu + tempv * tempv


def negative_buoyancy_frequency(
    rho_in_situ: torch.Tensor,
    rho_upper_at_lower: torch.Tensor,
    *,
    recip_drc: torch.Tensor,
    gravity: float,
    mass2r_unit: float,
) -> torch.Tensor:
    """``-N^2 = gravity*mass2rUnit*(rhoKm1 - rhoK)*recip_drC(K)``, ``my82_ri_number.F:112-113``.

    PP81 forms the negation of the same product (``pp81_ri_number.F:109-110``).
    ``rho_in_situ`` [B, nz, ny, nx] is each layer at its own reference level;
    its layers 1..nz-1 are ``rhoK``, which ``pp81_ri_number.F:88-92`` takes
    from FIND_RHO_2D with kRef = K.  ``rho_upper_at_lower`` [B, nz - 1, ny,
    nx] is layer k - 1 at the reference level of layer k (``rhoKm1``,
    ``:83-87``).
    """
    return (gravity * mass2r_unit * (rho_upper_at_lower - rho_in_situ[:, 1:])
            * recip_drc[1:].reshape(-1, 1, 1))


def richardson_number(minus_n2: torch.Tensor, shear_squared: torch.Tensor) -> torch.Tensor:
    """``RiNumber = N^2/max(shear^2, epsilon)``, ``pp81_ri_number.F:115``, ``my82_ri_number.F:118``.

    The negation of ``-N^2`` is exact, so PP81's ``buoyFreq/max(...)`` and
    MY82's ``-buoyFreq/max(...)`` give the same value.
    """
    return -minus_n2 / torch.clamp(shear_squared, min=SHEAR_FLOOR)
