"""GM/Redi tensor components from slopes and tapers (MITgcm ``gmredi_calc_tensor.F``).

Every ``GM_isoFac*``/``GM_bolFac*`` factor is 1, so ``Kredi = GM_isopycK`` and
``Kgm = Kredi + GM_skewflx*GM_background_K`` (m^2/s), as
``gmredi_calc_tensor.F:658-676`` reduces to; with the Visbeck K the
diffusivities arrive as ``[NY, NX]`` tensors at the component's location.  Outputs are
``[Nr, NY, NX]`` and zero outside MITgcm's inner loop range ``[2-OL : s+OL-1]``,
where MITgcm never writes them.  ``Kwx Kwy Kwz`` sit at W points (row 0, the
surface, is zero); ``Kux Kuz`` at U points; ``Kvy Kvz`` at V points.  Index
and sign conventions are those of the package docstring.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# z coordinates: model/src/ini_vertical_grid.F:54.
_GRAVITY_SIGN = -1.0


def _inner(field: torch.Tensor) -> torch.Tensor:
    """Keep MITgcm's inner range ``[2-OL : s+OL-1]`` in both directions, zero the ring."""
    return F.pad(field[..., 1:-1, 1:-1], (1, 1, 1, 1))


def w_tensor(
    slope_x: torch.Tensor,
    slope_y: torch.Tensor,
    slope_sq: torch.Tensor,
    taper: torch.Tensor,
    *,
    kgm: float | torch.Tensor,
    kredi: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``Kwx Kwy Kwz`` (``gmredi_calc_tensor.F:564-570`` times the second step ``:680-682``).

    Inputs are the W-point slopes and taper for ``k = 2..Nr`` (``Nr-1`` rows);
    level 1 is never written by MITgcm and comes out zero.
    """
    kwx = kgm * (-_GRAVITY_SIGN * slope_x * taper)
    kwy = kgm * (-_GRAVITY_SIGN * slope_y * taper)
    kwz = kredi * (slope_sq * taper)
    surface = (0, 0, 0, 0, 1, 0)
    return (
        F.pad(_inner(kwx), surface),
        F.pad(_inner(kwy), surface),
        F.pad(_inner(kwz), surface),
    )


def horizontal_diagonal_diffusivity(
    taper: torch.Tensor,
    *,
    isopyc_k: float | torch.Tensor,
    kmin_horiz: float,
) -> torch.Tensor:
    """``Kux`` from the U-point taper, or ``Kvy`` from the V-point taper, under
    ``GM_NON_UNITY_DIAGONAL`` (``gmredi_calc_tensor.F:785-808,982-1005``)."""
    return _inner(torch.clamp(isopyc_k * taper, min=kmin_horiz))


def constant_horizontal_diffusivity(like: torch.Tensor, *, isopyc_k: float) -> torch.Tensor:
    """``Kux = Kvy = GM_isopycK`` without ``GM_NON_UNITY_DIAGONAL`` (``gmredi_calc_tensor.F:1093-1118``)."""
    return _inner(torch.full_like(like, isopyc_k))


def vertical_offdiagonal_diffusivity(
    slope: torch.Tensor,
    taper: torch.Tensor,
    *,
    isopyc_k: float | torch.Tensor,
    background_k: float | torch.Tensor,
    skewflx: float,
) -> torch.Tensor:
    """``Kuz`` from the U-point ``SlopeX``, or ``Kvz`` from the V-point ``SlopeY``,
    under ``GM_EXTRA_DIAGONAL`` with ``GM_ExtraDiag`` (``gmredi_calc_tensor.F:815-846,1012-1043``)."""
    return _inner(-_GRAVITY_SIGN * (isopyc_k - skewflx * background_k) * slope * taper)
