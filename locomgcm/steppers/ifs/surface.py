"""Surface forcing conversion and first-wet-level centre selection.

Centre fields use [batch, nz, ny, nx], with k increasing downward.
Surface access uses the first wet centre level with 3-D masks and level zero
with 2-D masks. Selection preserves field dtype and bits. Freshwater
volume flux is in m/s; rhoConstFresh converts it to MIT mass flux in kg/m2/s.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from locomgcm.geometry.masks import MaskSet


def real_freshwater_mass_flux(
    dynamics_modules: Sequence[Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor | None, float]:
    """Return raw freshwater forcing in MIT mass-flux convention.

    The MIT-input source preserves ``EmPmRFile`` as the legacy bulk volume flux
    [m/s].  ``WPFluxSurfaceForcing._forcing_fields`` exposes that field directly;
    multiply it once by ``rhoConstFresh`` here to match MIT ``ini_forcing.F:82-94``
    and ``FFIELDS.h:29-36``.  This boundary never reconstructs freshwater from
    the virtual salt tendency.
    """
    for module in dynamics_modules:
        if getattr(module, "name", "") != "wpflux_surface_forcing":
            continue
        _, _, empmr_volume_flux = module._forcing_fields(dtype=dtype, device=device)
        rho_const_fresh = torch.as_tensor(
            float(module.rho_const_fresh), dtype=dtype, device=device
        )
        empmr_mass_flux = empmr_volume_flux * rho_const_fresh
        return empmr_mass_flux, 1.0 / float(module.rho_const)
    return None, 1.0


def first_wet_level(
    masks: MaskSet,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(wet_col, ks)`` — MITgcm's ``kSurfC``, 0-indexed.

    ``ks`` is the first wet centre level of each column and ``wet_col`` marks
    the columns that have one at all; a column without is MITgcm's
    ``ks.GT.Nr``, whose guard (``solve_for_pressure.F:198``) skips the surface
    source term entirely.
    """
    mask_c = masks.center_float(dtype=dtype, device=device)
    has_wet = mask_c > 0
    return has_wet.any(dim=0), has_wet.to(torch.int64).argmax(dim=0)


def surface_level_slice(
    field: torch.Tensor,
    masks: MaskSet,
    *,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """Gather ``field[:, ks]`` per column, zeroing fully dry columns.

    ``field`` is ``[batch, nz, ny, nx]``; the result is ``[batch, ny, nx]``.
    Pure selection — no arithmetic — so the gathered values keep their exact
    bits.
    """
    if not masks.has_3d_masks:
        # ``has_3d_masks`` is exactly ``center.ndim == 3`` (``masks.py``), so
        # this branch is the 2-D-mask case: every column is single-level, ks=0.
        return field[:, 0]
    wet_col, ks = first_wet_level(masks, dtype=field.dtype, device=field.device)
    index = ks.reshape(1, 1, ny, nx).expand(field.shape[0], 1, -1, -1)
    gathered = field.gather(1, index)[:, 0]
    return torch.where(wet_col.unsqueeze(0), gathered, torch.zeros_like(gathered))


def add_at_surface_level(
    field: torch.Tensor,
    addend: torch.Tensor,
    masks: MaskSet,
    *,
    nz: int,
) -> torch.Tensor:
    """Return ``field`` ([B,nz,ny,nx]) with ``addend`` ([B,ny,nx]) added at ks.

    The scatter counterpart of :func:`surface_level_slice`, for MITgcm's
    ``cg3d_b(i,j,ks)`` updates.  Selection via ``torch.where`` rather than a
    multiply, so a dry column never forms ``0 * addend``.
    """
    if not masks.has_3d_masks:
        out = field.clone()
        out[:, 0] = out[:, 0] + addend
        return out
    wet_col, ks = first_wet_level(masks, dtype=field.dtype, device=field.device)
    level = torch.arange(nz, device=field.device).view(nz, 1, 1)
    at_surface = (level == ks.unsqueeze(0)) & wet_col.unsqueeze(0)
    return torch.where(at_surface.unsqueeze(0), field + addend.unsqueeze(1), field)
