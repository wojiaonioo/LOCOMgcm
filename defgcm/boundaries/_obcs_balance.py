"""Fixed geometry products for the OBCS transport-balance calculation.

The plan owns no sampled boundary velocity or global reduction. Its tensors
are local face areas, local area sums and normal-face wet masks; every balance
call still evaluates its own inflow and issues both ordered global sums.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any

import torch

from defgcm.accel.constant_cache import ConstantCache


@dataclass(frozen=True)
class ObcsBalanceGeometry:
    """Local west/east/south/north products in the original side order."""

    areas: tuple[torch.Tensor | None, ...]
    area_sums: tuple[torch.Tensor | None, ...]
    correction_masks: tuple[torch.Tensor | None, ...]


def _build_geometry(
    center: torch.Tensor | None,
    thickness: torch.Tensor,
    hfw: torch.Tensor,
    hfs: torch.Tensor,
    dy_g: torch.Tensor | None,
    dx_g: torch.Tensor | None,
    *,
    nx: int,
    ny: int,
    open_sides: tuple[bool, ...],
    present_sides: tuple[bool, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> ObcsBalanceGeometry:
    mask = (center[0].clone() if center is not None
            else torch.ones(ny, nx, dtype=dtype, device=device))
    if open_sides[0]:
        mask[:, 0].zero_()
    if open_sides[1]:
        mask[:, -1].zero_()
    if open_sides[2]:
        mask[0, :].zero_()
    if open_sides[3]:
        mask[-1, :].zero_()
    inner = (
        torch.max(mask[:, 0], mask[:, 1]) if nx >= 2 else mask[:, 0],
        torch.max(mask[:, -2], mask[:, -1]) if nx >= 2 else mask[:, -1],
        torch.max(mask[0, :], mask[1, :]) if ny >= 2 else mask[0, :],
        torch.max(mask[-2, :], mask[-1, :]) if ny >= 2 else mask[-1, :],
    )
    hfac = (hfw[:, :, 1], hfw[:, :, -2], hfs[:, 1, :], hfs[:, -2, :])
    widths = (
        None if dy_g is None else dy_g[:, 1],
        None if dy_g is None else dy_g[:, -2],
        None if dx_g is None else dx_g[1 if ny >= 2 else 0, :],
        None if dx_g is None else dx_g[-2 if ny >= 2 else -1, :],
    )
    dr_f = thickness.to(dtype=dtype, device=device).view(-1, 1)
    areas: list[torch.Tensor | None] = []
    sums: list[torch.Tensor | None] = []
    masks: list[torch.Tensor | None] = []
    for index in range(4):
        if open_sides[index] and present_sides[index]:
            area = dr_f * hfac[index] * widths[index]
            area = area * inner[index].view(1, -1)
            areas.append(area)
            sums.append(area.sum())
        else:
            areas.append(None)
            sums.append(None)
        masks.append((hfac[index] != 0.0).to(dtype=dtype)
                     if present_sides[index] else None)
    return ObcsBalanceGeometry(tuple(areas), tuple(sums), tuple(masks))


def balance_geometry(
    owner: Any,
    center: torch.Tensor | None,
    thickness: torch.Tensor,
    hfw: torch.Tensor,
    hfs: torch.Tensor,
    dy_g: torch.Tensor | None,
    dx_g: torch.Tensor | None,
    *,
    nx: int,
    ny: int,
    open_sides: tuple[bool, ...],
    present_sides: tuple[bool, ...],
    dtype: torch.dtype,
    device: torch.device | str,
    time_varying: bool,
) -> ObcsBalanceGeometry:
    """Reuse inactive versioned geometry; keep active geometry in this graph.

    The memo belongs to the grid and is separated by device, dtype, topology
    and present normal sides. Source replacement or an in-place write replaces
    that key's entry. The source stamp contains weak references and Torch
    mutation counters, so it retains no old control graph and grows with the
    actual layouts rather than with the number of parameter updates.

    Tracing and differentiation transforms use the ordinary tensor expression.
    An inference tensor has no mutation counter and also takes that path.
    """
    def build() -> ObcsBalanceGeometry:
        return _build_geometry(
            center, thickness, hfw, hfs, dy_g, dx_g, nx=nx, ny=ny,
            open_sides=open_sides, present_sides=present_sides,
            dtype=dtype, device=device,
        )

    sources = tuple(value for value in (center, thickness, hfw, hfs, dy_g, dx_g)
                    if value is not None)
    if (torch.compiler.is_compiling() or time_varying
            or torch._C._are_functorch_transforms_active()
            or any(value.requires_grad or value.is_inference()
                   or torch.autograd.forward_ad.unpack_dual(value).tangent is not None
                   for value in sources)):
        return build()
    key = (dtype, str(torch.device(device)), nx, ny, open_sides, present_sides)
    memo = getattr(owner, "_obcs_balance_geometry_memo", None)
    if memo is None:
        memo = {}
        object.__setattr__(owner, "_obcs_balance_geometry_memo", memo)
    entry = memo.get(key)
    if (entry is None or len(entry[0]) != len(sources)
            or any(reference() is not value or version != value._version
                   for (reference, version), value in zip(entry[0], sources))):
        entry = (tuple((weakref.ref(value), value._version) for value in sources),
                 ConstantCache())
        memo[key] = entry
    return entry[1].get_or_build(
        (), build, variant="obcs_balance_geometry",
        site="OBCS.balance_geometry", tensor_only=True,
    )
