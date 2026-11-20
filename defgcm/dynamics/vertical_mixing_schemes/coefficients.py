"""Merging scheme coefficients into MITgcm's total vertical coefficients.

The statements of CALC_3D_DIFFUSIVITY and CALC_VISCOSITY
(``model/src/calc_3d_diffusivity.F``, ``model/src/calc_viscosity.F``) and of
the per-package ``*_CALC_DIFF`` / ``*_CALC_VISC`` routines that PP81, MY82
and GGL90 share.  The caller starts from the background (``diffKrNrT``,
``diffKrNrS``, ``viscArNr``) and applies the merges in MITgcm's package
order.

Interface fields: index k on the level axis (-3) is the top interface of
layer k (MITgcm level K = k + 1).  Tracer coefficients sit on centres
[B, nz, ny, nx]; ``kappa_ru`` on U faces [B, nz, ny, nx + 1] and
``kappa_rv`` on V faces [B, nz, ny + 1, nx], both outer faces included as
in the LOCOMgcm state.  ``mask_w`` [nz, ny, nx + 1] and ``mask_s``
[nz, ny + 1, nx] are the 0/1 face masks.  Background columns are [nz].
Coefficients are m^2/s.  dtype follows the inputs.
"""
from __future__ import annotations

import torch


def add_scheme_coefficient(
    kappa: torch.Tensor,
    scheme_coefficient: torch.Tensor,
    *,
    background: torch.Tensor,
) -> torch.Tensor:
    """``KappaRx(k) + (X(k) - background(k))`` on every level.

    The tracer merge of PP81, MY82 and GGL90 with ``background = diffKrNrS``
    (``pp81_calc_diff.F:47-61``, ``my82_calc_diff.F:45-59``,
    ``ggl90_calc_diff.F:48-58``), and the GGL90 viscosity merge with
    ``background = viscArNr`` (``ggl90_calc_visc.F:47-59``).  The scheme
    coefficient sits on the same points as ``kappa``.
    """
    return kappa + (scheme_coefficient - background.reshape(-1, 1, 1))


def add_centre_viscosity(
    kappa_ru: torch.Tensor,
    kappa_rv: torch.Tensor,
    centre_viscosity: torch.Tensor,
    *,
    visc_arnr: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The PP81 and MY82 viscosity merge, ``pp81_calc_visc.F:40-54``, ``my82_calc_visc.F:37-51``.

    ``KappaRU = max(KappaRU, KappaRU - viscArNr(k) + maskW*0.5*(X(i) + X(i-1)))``
    and the same on V faces with ``X(j-1)``.  ``centre_viscosity`` [B, nz,
    ny + 2, nx + 2] carries one halo cell on each side, so the outer faces
    average with the neighbouring cell; the caller fills that ring.
    """
    v = visc_arnr.reshape(-1, 1, 1)
    rows = centre_viscosity[..., 1:-1, :]
    cols = centre_viscosity[..., :, 1:-1]
    kappa_ru = torch.maximum(
        kappa_ru, kappa_ru - v + mask_w * 0.5 * (rows[..., 1:] + rows[..., :-1]))
    kappa_rv = torch.maximum(
        kappa_rv, kappa_rv - v + mask_s * 0.5 * (cols[..., 1:, :] + cols[..., :-1, :]))
    return kappa_ru, kappa_rv


def bottom_interface_viscosity(kappa: torch.Tensor) -> torch.Tensor:
    """Append interface Nr+1 equal to interface Nr, ``calc_viscosity.F:121-130``.

    MITgcm does this when PP81, KL10, MY82 or GGL90 is on; the no-slip bottom
    drag of a cell on the last level reads that interface
    (``mom_u_botdrag_coeff.F:72-75,115-128``).  The result has nz + 1 levels.
    """
    return torch.cat((kappa, kappa[..., -1:, :, :]), dim=-3)
