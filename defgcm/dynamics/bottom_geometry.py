"""Bottom-cell geometry of U and V face columns for bottom drag
(MITgcm ``mom_u_botdrag_coeff.F``, ``mom_v_botdrag_coeff.F``).

``hfac`` is ``hFacW`` or ``hFacS`` with the vertical level third from the end,
``[..., nz, y, x]``; any leading axes (the face axis of cube grids) are kept.
Level 0 is the surface and levels grow downward.  A column's bottom cell is
its deepest wet level (``hfac > 0``); ``k_bottom`` is that level index with a
singleton level axis, ``[..., 1, y, x]``, and 0 for a fully dry column.
``layer_thickness`` is ``drF`` ``[nz]`` in metres.  Gathered quantities are
two-dimensional bottom values with the singleton level axis.  Results keep
the dtype of ``hfac`` and ``layer_thickness``; level indices are int64.
"""

from __future__ import annotations

import torch


def bottom_level_mask_from_hfac(hfac: torch.Tensor) -> torch.Tensor:
    """Return a mask that is 1 only at the deepest wet level of each column.

    ``hfac`` may have any leading dimensions, with vertical level as the third
    dimension from the end: ``[..., nz, y, x]``.  This is the tensor form of the
    MIT bottom-cell selector used by both regional and cube bottom drag paths.
    """
    if hfac.ndim < 3:
        raise ValueError("hfac must have shape [..., nz, y, x]")
    wet = (hfac > 0).to(dtype=hfac.dtype)
    bottom = torch.zeros_like(wet)
    bottom[..., -1, :, :] = wet[..., -1, :, :]
    if wet.shape[-3] > 1:
        bottom[..., :-1, :, :] = wet[..., :-1, :, :] * (1.0 - wet[..., 1:, :, :])
    return bottom


def bottom_column_levels(
    hfac: torch.Tensor,
    mask: torch.Tensor,
    *,
    layer_thickness: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(k_bottom, has_bottom, recip_drF[k_bottom])`` of every face column.

    ``mask`` is ``bottom_level_mask_from_hfac(hfac)``; ``has_bottom`` is it read
    at ``k_bottom`` (1 for a column with any wet cell, 0 for a fully dry one).
    """
    lead = (1,) * (hfac.ndim - 3)
    levels = torch.arange(
        hfac.shape[-3], dtype=torch.int64, device=hfac.device
    ).view(*lead, -1, 1, 1)
    # amax, not argmax: a fully dry column is all-zero and argmax's tie
    # break is unspecified on device, while amax over k*wet is exactly 0.
    k_bottom = ((hfac > 0).to(torch.int64) * levels).amax(dim=-3, keepdim=True)
    has_bottom = torch.take_along_dim(mask, k_bottom, dim=-3)
    recip_dr_f = 1.0 / layer_thickness.view(*lead, -1, 1, 1).clamp(min=1e-14)
    recip_dr_f_bottom = torch.take_along_dim(recip_dr_f, k_bottom, dim=-3)
    return k_bottom, has_bottom, recip_dr_f_bottom


def bottom_interface_levels(
    k_bottom: torch.Tensor, *, nz: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(bottom_is_internal, k_interface)`` for the interface below each bottom cell.

    A bottom at ``k < nz-1`` reads the interface viscosity one level below,
    element ``k+1``; a bottom on the global floor reads nothing.
    ``k_interface`` is clamped so floor columns still index in range.
    """
    bottom_is_internal = k_bottom < (nz - 1)
    k_interface = (k_bottom + 1).clamp(max=nz - 1)
    return bottom_is_internal, k_interface


def bottom_recip_dr_c(
    k_bottom: torch.Tensor, *, interface_spacing: torch.Tensor, nz: int
) -> torch.Tensor:
    """``1 / drC`` of the interface below each bottom cell (``nz > 1``);
    ``interface_spacing`` is drC at the interior interfaces."""
    recip_dr_c_below = 1.0 / interface_spacing
    view = recip_dr_c_below.view(*([1] * (k_bottom.ndim - 3)), -1, 1, 1)
    return torch.take_along_dim(view, k_bottom.clamp(max=nz - 2), dim=-3)


def bottom_recip_hfac(hfac: torch.Tensor, k_bottom: torch.Tensor) -> torch.Tensor:
    """``1 / hFac`` of the bottom cell, 0 for a dry column.

    Gathering first and reciprocating in 2-D applies the very same
    per-element expression the full-field form applied, hence stays bitwise.
    """
    while hfac.ndim < k_bottom.ndim:
        hfac = hfac.unsqueeze(0)
    hfac_bottom = torch.take_along_dim(hfac, k_bottom, dim=-3)
    return torch.where(
        hfac_bottom > 0,
        1.0 / hfac_bottom.clamp(min=1e-14),
        torch.zeros_like(hfac_bottom),
    )


def bottom_thickness(
    hfac: torch.Tensor, k_bottom: torch.Tensor, *, layer_thickness: torch.Tensor
) -> torch.Tensor:
    """``hFac * drF`` of the bottom cell of every face column (0 for a dry column)."""
    while hfac.ndim < k_bottom.ndim:
        hfac = hfac.unsqueeze(0)
    drf = layer_thickness.view(*([1] * (k_bottom.ndim - 3)), -1, 1, 1)
    return torch.take_along_dim(hfac, k_bottom, dim=-3) * torch.take_along_dim(
        drf, k_bottom, dim=-3
    )
