"""Implicit vertical diffusivity for convection (MITgcm ``ivdc_kappa``).

CALC_IVDC (``model/src/calc_ivdc.F``) and the part of CALC_3D_DIFFUSIVITY
that adds ``ivdc_kappa`` to the background diffusivity of a tracer without
KPP (``model/src/calc_3d_diffusivity.F``).  The input ``sigma_r`` comes from
``density_gradient.vertical_density_gradient``.

Interface fields are [B, nz, ny, nx]; index k is the top interface of layer
k (MITgcm level K = k + 1) and index 0, the sea surface, holds zero in the
count.  ``sigma_r`` keeps the MITgcm sign (stable columns negative).
Diffusivities are m^2/s.  dtype follows the inputs.
"""
from __future__ import annotations

import torch

from .density_gradient import GRAVITY_SIGN

#: Vertical coordinates whose density pairing feeds ``sigma_r``
#: (``do_oceanic_phys.F:812-823``); the pressure-coordinate pairing
#: (``:824-836``) and ``gravitySign = +1`` are not covered.
IVDC_VERTICAL_COORDINATES = ("z",)


def ivdc_convection_count(sigma_r: torch.Tensor) -> torch.Tensor:
    """``IVDConvCount`` of CALC_IVDC, ``calc_ivdc.F:44-53``.

    One where ``-sigmaR*gravitySign > 0`` (statically unstable), zero
    elsewhere, on interfaces 1..nz-1; ``do_oceanic_phys.F:867`` calls it for
    K > 1 only, so interface 0 keeps the zero of ``ini_dynvars.F:93``.
    """
    inner = sigma_r[:, 1:]
    count = torch.where(-inner * GRAVITY_SIGN > 0.0,
                        torch.ones_like(inner), torch.zeros_like(inner))
    return torch.cat((count.new_zeros((count.shape[0], 1, *count.shape[2:])), count), dim=1)


def ivdc_tracer_diffusivity(
    convection_count: torch.Tensor,
    *,
    ivdc_kappa: float,
    background_kr: float | torch.Tensor,
) -> torch.Tensor:
    """``KappaRTr`` of a tracer without KPP, ``calc_3d_diffusivity.F:83-144``.

    ``IVDConvCount*ivdc_kappa + KbryanLewis79`` (``:107-116``), then
    ``+ diffKrNrT(k)`` or ``+ diffKrNrS(k)`` (``:119-144``).  The Bryan-Lewis
    term is zero here: its profile is rejected at assembly, and adding the
    zero changes no value.  ``background_kr`` is the tracer's diffKrNr as a
    scalar or a tensor broadcasting against [B, nz, ny, nx].
    """
    return convection_count * ivdc_kappa + background_kr
