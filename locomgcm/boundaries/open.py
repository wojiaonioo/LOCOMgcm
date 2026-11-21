from __future__ import annotations

from locomgcm.sampling import SampleClock, SamplingRule

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, ClassVar

import torch
import torch.distributed as dist

from locomgcm.deferred_checks import strict_checks
from locomgcm.fields import Field, FieldLocation
from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.grid import _side_is_open
from locomgcm.state import PrognosticState, StateTendency

from .base import BoundaryCondition
from .conditions import StructuredBoundaryCondition
from ._obcs_balance import balance_geometry
from .orlanski import OrlanskiParameters, orlanski_side_update
from locomgcm.boundaries._registry import register_boundary_condition


class BoundarySide(StrEnum):
    WEST = "west"
    EAST = "east"
    SOUTH = "south"
    NORTH = "north"


#: The four sides in ``iter(BoundarySide)`` order, frozen at import.
#:
#: Compiled execution constraint: ``for side in BoundarySide:`` inside
#: the compiled step is ``Unsupported method call -- Dynamo does not know how to
#: trace method ``__iter__`` of class ``EnumType``, which breaks the
#: graph once per loop.  Iterating this tuple instead is the same members in the
#: same order, so every loop-carried float accumulation keeps its summation
#: order.
BOUNDARY_SIDES: tuple[BoundarySide, ...] = (
    BoundarySide.WEST,
    BoundarySide.EAST,
    BoundarySide.SOUTH,
    BoundarySide.NORTH,
)


class TracerBoundaryScheme(StrEnum):
    CLAMPED = "clamped"
    GRADIENT = "gradient"
    INFLOW_CLAMPED = "inflow_clamped"
    STEVENS = "stevens"


#: Tracer boundary schemes TracerOpenBoundaryCondition accepts.  STEVENS here is the
#: tracer update only; MITgcm's OBCS_CALC_STEVENS also rebuilds the normal velocity
#: from the interior shear (pkg/obcs/obcs_calc_stevens.F:181-223), which LOCOMgcm does
#: not do, so the MITgcm translator keeps useStevens* unsupported.
TRACER_BOUNDARY_SCHEMES = (
    TracerBoundaryScheme.CLAMPED,
    TracerBoundaryScheme.GRADIENT,
    TracerBoundaryScheme.INFLOW_CLAMPED,
    TracerBoundaryScheme.STEVENS,
)


class OpenBoundaryValueScheme(StrEnum):
    PRESCRIBED = "prescribed"
    ORLANSKI = "orlanski"


#: Where Prescribed3DBoundaryCondition takes one side's u, v, theta and salt
#: values.  PRESCRIBED: its sampled sources, i.e. OBCS_CALC's zero / tRef / sRef
#: or OBCS_PRESCRIBE_READ (pkg/obcs/obcs_calc.F:120-137,407-414).  ORLANSKI:
#: ORLANSKI_<side> on the step-start state (obcs_calc.F:97-103, orlanski_*.F).
OPEN_BOUNDARY_VALUE_SCHEMES = (
    OpenBoundaryValueScheme.PRESCRIBED,
    OpenBoundaryValueScheme.ORLANSKI,
)


def _global_sum_scalar(value: torch.Tensor, *, group=None) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        # Chained through the halo order token so the compiled step issues it
        # in the same position on every rank (halo_p2p.py).
        from locomgcm.distributed.halo_p2p import ordered_all_reduce_sum

        return ordered_all_reduce_sum(value, group=group)
    return value.clone()


@dataclass(frozen=True)
class SideBoundaryTensorSet:
    west: torch.Tensor | None = None
    east: torch.Tensor | None = None
    south: torch.Tensor | None = None
    north: torch.Tensor | None = None

    def get(self, side: BoundarySide) -> torch.Tensor | None:
        return getattr(self, side.value)

    def as_dict(self) -> dict[str, torch.Tensor | None]:
        return {
            "west": self.west,
            "east": self.east,
            "south": self.south,
            "north": self.north,
        }

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "SideBoundaryTensorSet":
        def _move(values: torch.Tensor | None) -> torch.Tensor | None:
            if values is None:
                return None
            kwargs: dict[str, object] = {"device": device}
            if dtype is not None:
                kwargs["dtype"] = dtype
            return values.to(**kwargs)

        return SideBoundaryTensorSet(
            west=_move(self.west),
            east=_move(self.east),
            south=_move(self.south),
            north=_move(self.north),
        )


def compute_obcs_balanced_normal_velocities(
    u_values: SideBoundaryTensorSet,
    v_values: SideBoundaryTensorSet,
    grid: StructuredCGrid,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[SideBoundaryTensorSet, SideBoundaryTensorSet]:
    """Apply MITgcm OBCS_BALANCE_FLOW correction to OBC normal-velocity targets.

    Mirrors the once-per-step OB-array mutation in
    ``references/MITgcm/pkg/obcs/obcs_balance_flow.F:347-456``: integrate net
    inflow across all open-boundary normal faces, divide by total wet face
    area to obtain a uniform correction velocity ``q``, then shift normal
    targets so the integrated transport sums to zero::

        u_west  -> u_west  - q   (positive u into domain → reduce inflow)
        u_east  -> u_east  + q   (positive u out of domain → increase outflow)
        v_south -> v_south - q   (positive v into domain → reduce inflow)
        v_north -> v_north + q   (positive v out of domain → increase outflow)

    Tangential-side arrays pass through unchanged.  Returns NEW
    :class:`SideBoundaryTensorSet` instances; inputs are not mutated.

    Used by both :class:`Prescribed3DBoundaryCondition` (face clamp) and
    :class:`MomentumBoundarySpongeRelaxation` (sponge target relaxation) so
    that BOTH consumers see the SAME balanced targets, matching MIT topology
    where ``OBCS_ADJUST → OBCS_BALANCE_FLOW`` mutates OB arrays once per
    step and downstream readers (``OBCS_APPLY_UV``, ``OBCS_SPONGE_U/V``)
    share the corrected arrays.

    Inner-face mask convention follows ``obcs_init_fixed.F:154``: the inner
    face mask between OB column 0 and interior column 1 is the MAX of the
    adjacent ``maskInC`` values, equivalent to "1 if at least one neighbour
    is interior".  Corner cells (where two OB strips meet) have
    ``maskInC=0`` on both sides so their inner faces drop out of the
    balance integral, ensuring the balanced quantity equals the
    exactConserv telescoping sum exactly.
    """
    q, geometry = _obcs_balance_flow(u_values, v_values, grid, dtype=dtype, device=device)
    if q is None:
        return u_values, v_values
    # q is the uniform correction velocity [m/s].  The branch-free path applies
    # an exact zero correction on a degenerate total area instead of taking the
    # early return, so it returns fresh tensors where the strict path returns
    # the caller's own objects.
    def _shift(values: torch.Tensor | None, sign: float, mask: torch.Tensor | None) -> torch.Tensor | None:
        if values is None:
            return None
        result = values.to(dtype=dtype, device=device)
        correction = sign * q * mask
        if result.ndim == correction.ndim + 1:
            correction = correction.unsqueeze(0)
        return result + correction

    new_u = SideBoundaryTensorSet(
        # obcs_balance_flow.F:413-417, OBWu += flowW*maskW(1+iB)
        west=_shift(u_values.west, -1.0, geometry.correction_masks[0]),
        # obcs_balance_flow.F:393-397, OBEu += flowE*maskW(iB); -2 = east OB
        # normal face (nx-1), matching the balance integral hfw[:,:,-2] above —
        # NOT the outermost face -1 (=nx), which is dry and would zero wet OB.
        east=_shift(u_values.east, +1.0, geometry.correction_masks[1]),
        south=u_values.south,
        north=u_values.north,
    )
    new_v = SideBoundaryTensorSet(
        west=v_values.west,
        east=v_values.east,
        # obcs_balance_flow.F:453-457, OBSv += flowS*maskS(1+jB)
        south=_shift(v_values.south, -1.0, geometry.correction_masks[2]),
        # obcs_balance_flow.F:433-437, OBNv += flowN*maskS(jB); -2 = north OB
        # normal face (ny-1), matching the balance integral hfs[:,-2,:] above —
        # NOT the outermost face -1 (=ny), which is dry and would zero wet OB.
        north=_shift(v_values.north, +1.0, geometry.correction_masks[3]),
    )
    return new_u, new_v


def _obcs_balance_flow(
    u_values: SideBoundaryTensorSet,
    v_values: SideBoundaryTensorSet,
    grid: StructuredCGrid,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    factors: tuple[float, float, float, float] | None = None,
):
    """``(q, geometry)`` of ``OBCS_BALANCE_FLOW``: the uniform correction velocity and the faces.

    The net inflow through the inner normal faces of the open sides (``u_values.west`` /
    ``.east``, ``v_values.south`` / ``.north``: [nz, n], or [B, nz, n], or a depth-uniform
    [B, 1, n]) over their wet area; the areas are drF*hFac*dyG/dxG on the face between the OB
    cell and the interior with MITgcm's ``maskInW/S``, which drops the corners of two open
    sides (``_obcs_balance._build_geometry``, ``obcs_balance_flow.F:118,167,216,265``).
    ``factors`` are ``OBCS_balanceFac{W,E,S,N}`` (``obcs_balance_flow.F:76-85,344-361``):
    a side with factor >= 0 enters the inflow, its area enters weighted by the factor, and it
    is corrected by factor * q; ``None`` is MITgcm's default 1 on every side.  ``q`` is None
    when the strict level finds no open area (the caller keeps its values).
    """
    nx = grid.nx
    ny = grid.ny

    west_open = _side_is_open(BoundarySide.WEST.value, grid)
    east_open = _side_is_open(BoundarySide.EAST.value, grid)
    south_open = _side_is_open(BoundarySide.SOUTH.value, grid)
    north_open = _side_is_open(BoundarySide.NORTH.value, grid)
    present = (u_values.west is not None, u_values.east is not None,
               v_values.south is not None, v_values.north is not None)
    geometry = balance_geometry(
        grid,
        (grid.masks.center_float(dtype=dtype, device=device)
         if grid.masks.has_3d_masks else None),
        grid.layer_thickness,
        grid.hfac_w_for(dtype, device),
        grid.hfac_s_for(dtype, device),
        (grid.metrics.dyG_for(dtype, device)
         if (west_open and present[0]) or (east_open and present[1]) else None),
        (grid.metrics.dxG_for(dtype, device)
         if (south_open and present[2]) or (north_open and present[3]) else None),
        nx=nx, ny=ny,
        open_sides=(west_open, east_open, south_open, north_open),
        present_sides=present, dtype=dtype, device=device,
        time_varying=grid.masks.hfac_time_varying,
    )

    inflow = torch.zeros((), dtype=dtype, device=device)
    total_area = torch.zeros((), dtype=dtype, device=device)

    def enters(index: int) -> bool:
        return factors is None or factors[index] >= 0.0

    def weighted(index: int) -> torch.Tensor:
        area = geometry.area_sums[index]
        return area if factors is None else factors[index] * area

    if west_open and u_values.west is not None and enters(0):
        u_w = u_values.west.to(dtype=dtype, device=device)  # [nz, ny] (or batched)
        face_area_m = geometry.areas[0]
        broadcast_area = face_area_m if u_w.shape == face_area_m.shape else face_area_m.unsqueeze(0)
        inflow = inflow + (u_w * broadcast_area).sum()
        total_area = total_area + weighted(0)

    if east_open and u_values.east is not None and enters(1):
        u_e = u_values.east.to(dtype=dtype, device=device)
        face_area_m = geometry.areas[1]
        broadcast_area = face_area_m if u_e.shape == face_area_m.shape else face_area_m.unsqueeze(0)
        inflow = inflow - (u_e * broadcast_area).sum()
        total_area = total_area + weighted(1)

    if south_open and v_values.south is not None and enters(2):
        v_s = v_values.south.to(dtype=dtype, device=device)
        face_area_m = geometry.areas[2]
        broadcast_area = face_area_m if v_s.shape == face_area_m.shape else face_area_m.unsqueeze(0)
        inflow = inflow + (v_s * broadcast_area).sum()
        total_area = total_area + weighted(2)

    if north_open and v_values.north is not None and enters(3):
        v_n = v_values.north.to(dtype=dtype, device=device)
        face_area_m = geometry.areas[3]
        broadcast_area = face_area_m if v_n.shape == face_area_m.shape else face_area_m.unsqueeze(0)
        inflow = inflow - (v_n * broadcast_area).sum()
        total_area = total_area + weighted(3)

    from locomgcm.distributed.comm_adjoint import data_group_for_grid
    inflow = _global_sum_scalar(inflow, group=data_group_for_grid(grid))
    total_area = _global_sum_scalar(total_area, group=data_group_for_grid(grid))
    degenerate_area = total_area.abs() <= 1e-30
    # The early return reads a tensor value on the host, which the traced step
    # cannot do; it survives only in the ``strict`` level, which is the level
    # that never runs compiled.  Both arms compute the same ``q`` bit for bit
    # whenever the total face area is non-degenerate, which is every posture
    # with an open boundary at all.
    if strict_checks():
        if degenerate_area:
            return None, geometry
        q = inflow / total_area
    else:
        safe_area = torch.where(
            degenerate_area,
            torch.ones_like(total_area),
            total_area,
        )
        q = torch.where(
            degenerate_area,
            torch.zeros_like(inflow),
            inflow / safe_area,
        )
    return q, geometry


@dataclass(frozen=True)
class SideBoundarySchemeSet:
    west: TracerBoundaryScheme = TracerBoundaryScheme.INFLOW_CLAMPED
    east: TracerBoundaryScheme = TracerBoundaryScheme.INFLOW_CLAMPED
    south: TracerBoundaryScheme = TracerBoundaryScheme.INFLOW_CLAMPED
    north: TracerBoundaryScheme = TracerBoundaryScheme.INFLOW_CLAMPED

    def __post_init__(self) -> None:
        for side in BOUNDARY_SIDES:
            object.__setattr__(self, side.value, TracerBoundaryScheme(getattr(self, side.value)))

    def get(self, side: BoundarySide) -> TracerBoundaryScheme:
        return getattr(self, side.value)

    def as_dict(self) -> dict[str, str]:
        return {
            "west": self.west.value,
            "east": self.east.value,
            "south": self.south.value,
            "north": self.north.value,
        }


def _maybe_to(
    value: Any,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> Any:
    if hasattr(value, "to_runtime"):
        return value.to_runtime(device, dtype)
    if hasattr(value, "to"):
        if dtype is None:
            return value.to(device)
        try:
            return value.to(device, dtype)
        except TypeError:
            return value.to(device)
    return value


def expand_side_boundary_tensor(
    values: torch.Tensor,
    side: BoundarySide,
    grid: StructuredCGrid,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    result = values.to(dtype=dtype, device=device)
    n_boundary = grid.ny if side in (BoundarySide.WEST, BoundarySide.EAST) else grid.nx
    expected_shape = (grid.nz, n_boundary)
    # Barotropic input: [n_pts] → expand to [nz, n_pts] (uniform over depth)
    if result.ndim == 1 and result.shape[0] == n_boundary:
        result = result.unsqueeze(0).expand(grid.nz, -1).contiguous()
    if result.ndim == 1:
        result = result.unsqueeze(0)
    if result.ndim == 2:
        result = result.unsqueeze(0)
    if result.ndim != 3:
        raise ValueError(f"{side.value} boundary values must have 1, 2, or 3 dims, got {result.ndim}")
    if tuple(result.shape[1:]) != expected_shape:
        raise ValueError(f"{side.value} boundary values shape {tuple(result.shape[1:])} != {expected_shape}")
    if result.shape[0] == 1 and batch_size > 1:
        result = result.expand(batch_size, -1, -1)
    if result.shape[0] != batch_size:
        raise ValueError(f"{side.value} boundary batch {result.shape[0]} != {batch_size}")
    return result


def side_normal_velocity(state: PrognosticState, side: BoundarySide) -> torch.Tensor:
    if side == BoundarySide.WEST:
        return state.u.data[..., 0]
    if side == BoundarySide.EAST:
        return state.u.data[..., -1]
    if side == BoundarySide.SOUTH:
        return state.v.data[..., 0, :]
    return state.v.data[..., -1, :]


def inflow_mask(state: PrognosticState, side: BoundarySide) -> torch.Tensor:
    normal_velocity = side_normal_velocity(state, side)
    if side in (BoundarySide.WEST, BoundarySide.SOUTH):
        return normal_velocity > 0.0
    return normal_velocity < 0.0


def boundary_cell_values(field: torch.Tensor, side: BoundarySide) -> torch.Tensor:
    if side == BoundarySide.WEST:
        return field[..., 0]
    if side == BoundarySide.EAST:
        return field[..., -1]
    if side == BoundarySide.SOUTH:
        return field[..., 0, :]
    return field[..., -1, :]


def interior_neighbor_values(field: torch.Tensor, side: BoundarySide) -> torch.Tensor:
    if side == BoundarySide.WEST:
        return field[..., 1] if field.shape[-1] > 1 else field[..., 0]
    if side == BoundarySide.EAST:
        return field[..., -2] if field.shape[-1] > 1 else field[..., -1]
    if side == BoundarySide.SOUTH:
        return field[..., 1, :] if field.shape[-2] > 1 else field[..., 0, :]
    return field[..., -2, :] if field.shape[-2] > 1 else field[..., -1, :]


def _wet_interior_copy(
    field: torch.Tensor, applied: torch.Tensor, grid: StructuredCGrid, side: BoundarySide,
) -> torch.Tensor:
    """Zero-gradient boundary value: the interior neighbour where the face between them is wet.

    Where the inner normal face is dry (the neighbour, or the boundary cell, is
    land at that level) nothing is copied and the boundary cell keeps the value
    it holds in ``applied``, the mask the Stevens update applies to its
    increment (``boundary_wet``).  Copying a land neighbour wrote T = S = 0
    into wet boundary cells.
    """
    wet = grid.masks.u if side in (BoundarySide.WEST, BoundarySide.EAST) else grid.masks.v
    face_wet = interior_neighbor_values(wet.to(device=field.device, dtype=torch.bool), side)
    return torch.where(face_wet, interior_neighbor_values(field, side), boundary_cell_values(applied, side))


def _second_interior_neighbor_values(field: torch.Tensor, side: BoundarySide) -> torch.Tensor:
    """Return 2nd interior cell values (used by Stevens phase velocity estimation)."""
    if side == BoundarySide.WEST:
        return field[..., 2] if field.shape[-1] > 2 else field[..., 1] if field.shape[-1] > 1 else field[..., 0]
    if side == BoundarySide.EAST:
        return field[..., -3] if field.shape[-1] > 2 else field[..., -2] if field.shape[-1] > 1 else field[..., -1]
    if side == BoundarySide.SOUTH:
        return field[..., 2, :] if field.shape[-2] > 2 else field[..., 1, :] if field.shape[-2] > 1 else field[..., 0, :]
    return field[..., -3, :] if field.shape[-2] > 2 else field[..., -2, :] if field.shape[-2] > 1 else field[..., -1, :]


def assign_boundary_cell(field: torch.Tensor, side: BoundarySide, values: torch.Tensor) -> torch.Tensor:
    updated = field.clone()
    if side == BoundarySide.WEST:
        updated[..., 0] = values
    elif side == BoundarySide.EAST:
        updated[..., -1] = values
    elif side == BoundarySide.SOUTH:
        updated[..., 0, :] = values
    else:
        updated[..., -1, :] = values
    return updated


def corner_value_for_side(values: torch.Tensor, side: BoundarySide, *, north: bool, east: bool) -> torch.Tensor:
    if side == BoundarySide.WEST:
        return values[..., -1 if north else 0]
    if side == BoundarySide.EAST:
        return values[..., -1 if north else 0]
    if side == BoundarySide.SOUTH:
        return values[..., -1 if east else 0]
    return values[..., -1 if east else 0]


def assign_corner_value(field: torch.Tensor, *, north: bool, east: bool, values: torch.Tensor) -> torch.Tensor:
    updated = field.clone()
    y_index = -1 if north else 0
    x_index = -1 if east else 0
    updated[..., y_index, x_index] = values
    return updated


def boundary_band_target(
    values: torch.Tensor,
    side: BoundarySide,
    grid: StructuredCGrid,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
    width: int,
) -> torch.Tensor:
    expanded = expand_side_boundary_tensor(
        values,
        side,
        grid,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    if side in (BoundarySide.WEST, BoundarySide.EAST):
        return expanded.unsqueeze(-1).expand(-1, -1, -1, width)
    return expanded.unsqueeze(-2).expand(-1, -1, width, -1)


def expand_barotropic_side_boundary_tensor(
    values: torch.Tensor,
    side: BoundarySide,
    grid: StructuredCGrid,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    result = values.to(dtype=dtype, device=device)
    expected_length = grid.ny if side in (BoundarySide.WEST, BoundarySide.EAST) else grid.nx
    if result.ndim == 0:
        result = result.view(1, 1).expand(batch_size, expected_length)
        return result
    if result.ndim == 1:
        if result.shape[0] != expected_length:
            raise ValueError(f"{side.value} barotropic boundary values shape {tuple(result.shape)} != ({expected_length},)")
        result = result.unsqueeze(0)
    elif result.ndim == 2:
        if result.shape[-1] != expected_length:
            raise ValueError(
                f"{side.value} barotropic boundary values shape {tuple(result.shape)} has trailing dim {result.shape[-1]} "
                f"!= {expected_length}"
            )
    else:
        raise ValueError(f"{side.value} barotropic boundary values must have 0, 1, or 2 dims, got {result.ndim}")
    if result.shape[0] == 1 and batch_size > 1:
        result = result.expand(batch_size, -1)
    if result.shape[0] != batch_size:
        raise ValueError(f"{side.value} barotropic boundary batch {result.shape[0]} != {batch_size}")
    return result


def barotropic_boundary_band_target(
    values: torch.Tensor,
    side: BoundarySide,
    grid: StructuredCGrid,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
    width: int,
) -> torch.Tensor:
    expanded = expand_barotropic_side_boundary_tensor(
        values,
        side,
        grid,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    if side in (BoundarySide.WEST, BoundarySide.EAST):
        return expanded.unsqueeze(-1).expand(-1, -1, width)
    return expanded.unsqueeze(-2).expand(-1, width, -1)


@dataclass
@register_boundary_condition("tracer_open")
class TracerOpenBoundaryCondition(BoundaryCondition):

    sampling_rule = SamplingRule(
        SampleClock.END,
        ("active_temp_values", "active_salt_values"),
        entry_clock=SampleClock.START,
        providers=("source", "velocity_boundary"),
    )
    temp_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    salt_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    schemes: SideBoundarySchemeSet = SideBoundarySchemeSet()
    velocity_boundary: BoundaryCondition | None = None
    source: Any | None = None
    time_mode: str = "linear"
    update_interval_seconds: float | None = None
    stevens_relaxation_seconds: float = 21_600.0  # 6 hours default
    dt_seconds: float = 120.0  # model timestep for Stevens advection
    name: str = "tracer_open_boundary"
    supports_open_boundary: bool = True
    supports_barotropic_open_boundary: bool = False
    active_temp_values: SideBoundaryTensorSet = field(init=False, repr=False)
    active_salt_values: SideBoundaryTensorSet = field(init=False, repr=False)
    current_time_seconds: float = field(default=0.0, init=False, repr=False)
    current_sample_metadata: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _history_adapter: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from locomgcm.step.boundary_history import BoundaryHistoryAdapter

        # Manifest kwargs and direct construction share this normalization.
        if isinstance(self.schemes, Mapping):
            self.schemes = SideBoundarySchemeSet(**self.schemes)
        if not isinstance(self.schemes, SideBoundarySchemeSet):
            raise TypeError("schemes must be a SideBoundarySchemeSet or a side mapping")
        for side in BOUNDARY_SIDES:
            if self.schemes.get(side) not in TRACER_BOUNDARY_SCHEMES:
                raise NotImplementedError(
                    f"tracer_open {side.value} scheme {self.schemes.get(side).value!r} is not "
                    f"one of {[scheme.value for scheme in TRACER_BOUNDARY_SCHEMES]}"
                )
        self._history_adapter = BoundaryHistoryAdapter()
        if self.update_interval_seconds is not None and self.update_interval_seconds <= 0.0:
            self.update_interval_seconds = None
        if self.velocity_boundary is None:
            self.velocity_boundary = StructuredBoundaryCondition()
        if getattr(self.velocity_boundary, "uses_orlanski", False):
            # The step publishes the history of the model's boundary condition
            # only; a nested Orlanski boundary would lose its storage.
            raise NotImplementedError("tracer_open with an Orlanski velocity_boundary is not supported")
        self.supports_barotropic_open_boundary = bool(
            getattr(self.velocity_boundary, "supports_barotropic_open_boundary", False)
        )
        self.active_temp_values = self.temp_values
        self.active_salt_values = self.salt_values
        if self.source is not None:
            self.update_model_time(0.0)

    @property
    def uses_stevens(self):
        return any(self.schemes.get(side) == TracerBoundaryScheme.STEVENS for side in BOUNDARY_SIDES)

    @property
    def boundary_history(self):
        return self._history_adapter.history

    @property
    def _prev_temp(self):
        history = self.boundary_history
        return None if history is None else history.prev_temp

    @_prev_temp.setter
    def _prev_temp(self, value):
        self._history_adapter.set_legacy("prev_temp", value)

    @property
    def _prev_salt(self):
        history = self.boundary_history
        return None if history is None else history.prev_salt

    @_prev_salt.setter
    def _prev_salt(self, value):
        self._history_adapter.set_legacy("prev_salt", value)

    def update_model_time(self, model_time_seconds: float) -> None:
        self.current_time_seconds = float(model_time_seconds)
        if self.source is not None:
            sample = self.source.sample(
                model_time_seconds,
                mode=self.time_mode,
                update_interval_seconds=self.update_interval_seconds,
            )
            self.active_temp_values = sample.temp
            self.active_salt_values = sample.salt
            self.current_sample_metadata = sample.interpolation.as_dict()
        # Nested velocity providers are updated once by the common resolver.

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "TracerOpenBoundaryCondition":
        moved = TracerOpenBoundaryCondition(
            temp_values=self.temp_values.to(device, dtype),
            salt_values=self.salt_values.to(device, dtype),
            schemes=self.schemes,
            velocity_boundary=_maybe_to(self.velocity_boundary, device, dtype),
            source=_maybe_to(self.source, device, dtype),
            time_mode=self.time_mode,
            update_interval_seconds=self.update_interval_seconds,
            stevens_relaxation_seconds=self.stevens_relaxation_seconds,
            dt_seconds=self.dt_seconds,
            name=self.name,
            supports_open_boundary=self.supports_open_boundary,
        )
        moved.current_time_seconds = self.current_time_seconds
        if self.boundary_history is not None:
            moved._history_adapter.history = self.boundary_history.to(device, dtype)
            moved._history_adapter.initialization_reason = self._history_adapter.initialization_reason
        if self.source is not None or self.current_time_seconds != 0.0:
            moved.update_model_time(self.current_time_seconds)
        return moved

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "TracerOpenBoundaryCondition":
        return self.to_runtime(device, dtype)

    def enforce_tendency(self, tendency: StateTendency, grid: StructuredCGrid) -> StateTendency:
        """Delegate tendency zeroing to velocity boundary (e.g. FlatherBC maskInW/maskInS)."""
        if self.velocity_boundary is not None and hasattr(self.velocity_boundary, "enforce_tendency"):
            return self.velocity_boundary.enforce_tendency(tendency, grid)
        return tendency

    def enforce(self, state: PrognosticState, grid: StructuredCGrid, *,
                advance_stevens=True, stevens_source=None, flather_targets=None) -> PrognosticState:
        # The stamp labels the Stevens candidate only (enforce_with_history);
        # without Stevens the model clock is not read, so a compiled step
        # holds no guard on its per-step value.
        stamp = None
        if self.uses_stevens:
            stamp = self._history_adapter.active_stamp
            if stamp is None:
                stamp = self.current_time_seconds
        result, candidate = self.enforce_with_history(
            state, grid, history=self.boundary_history, stamp=stamp,
            advance_stevens=advance_stevens, stevens_source=stevens_source,
            flather_targets=flather_targets)
        if candidate is not None:
            self._history_adapter.candidate = candidate
        return result

    def enforce_with_history(self, state, grid, *, history, stamp,
                             advance_stevens=True, stevens_source=None, flather_targets=None):
        """Evaluate from an explicit history and return the pre-tracer candidate.

        Replay callers retain the supplied history and discard candidates from
        unaccepted stages. This method never publishes or stages history.
        ``flather_targets`` are the calling step's targets for a Flather
        velocity boundary (``FlatherBarotropicBoundaryCondition.freeze_targets``).
        """
        from locomgcm.step.boundary_history import BoundaryHistory

        targets = {} if flather_targets is None else {"flather_targets": flather_targets}
        bounded = (self.velocity_boundary.enforce(state, grid, **targets)
                   if self.velocity_boundary is not None else state)
        source, dt_tracer = (state, self.dt_seconds) if stevens_source is None else stevens_source
        temp = self._apply_tracer_field(bounded.temp.data, bounded, grid, self.active_temp_values,
                                        is_temp=True, history=history, stevens_source=source,
                                        advance_stevens=advance_stevens, dt_tracer=dt_tracer)
        salt = self._apply_tracer_field(bounded.salt.data, bounded, grid, self.active_salt_values,
                                        is_temp=False, history=history, stevens_source=source,
                                        advance_stevens=advance_stevens, dt_tracer=dt_tracer)
        # OBCS_STEVENS_SAVE_TRACERS retains the state used to diagnose this tendency.
        candidate = BoundaryHistory.sample(source, stamp, {}) if self.uses_stevens and advance_stevens else None
        return bounded.with_tensors(temp=temp, salt=salt), candidate

    def _apply_tracer_field(
        self,
        field: torch.Tensor,
        state: PrognosticState,
        grid: StructuredCGrid,
        boundary_values: SideBoundaryTensorSet,
        *,
        is_temp: bool,
        history,
        stevens_source,
        advance_stevens: bool,
        dt_tracer: float,
    ) -> torch.Tensor:
        side_updates: dict[BoundarySide, torch.Tensor] = {}
        updated = field.clone()
        # OBCS_APPLY_TS applies north/south first. At a shared corner each
        # Stevens side adds its tendency to the value left by the prior side.
        sides = ((BoundarySide.NORTH, BoundarySide.SOUTH, BoundarySide.EAST, BoundarySide.WEST)
                 if self.uses_stevens else BOUNDARY_SIDES)
        for side in sides:
            side_values = boundary_values.get(side)
            if side_values is None:
                continue
            stevens = self.schemes.get(side) == TracerBoundaryScheme.STEVENS
            if stevens and not advance_stevens:
                continue
            source = stevens_source if stevens else state
            source_field = source.temp.data if is_temp else source.salt.data
            boundary_update = self._side_boundary_update(
                source_field, source, grid, side, side_values, applied=updated,
                is_temp=is_temp, history=history, dt_tracer=dt_tracer)
            if not stevens:
                side_updates[side] = boundary_update
            updated = assign_boundary_cell(updated, side, boundary_update)
        return self._apply_corner_updates(updated, side_updates)

    def _side_boundary_update(
        self,
        field: torch.Tensor,
        state: PrognosticState,
        grid: StructuredCGrid,
        side: BoundarySide,
        side_values: torch.Tensor,
        *,
        applied: torch.Tensor,
        dt_tracer: float,
        is_temp: bool = True,
        history=None,
    ) -> torch.Tensor:
        target = expand_side_boundary_tensor(
            side_values,
            side,
            grid,
            batch_size=field.shape[0],
            dtype=field.dtype,
            device=field.device,
        )
        scheme = self.schemes.get(side)
        if scheme == TracerBoundaryScheme.CLAMPED:
            return target
        if scheme == TracerBoundaryScheme.GRADIENT:
            return _wet_interior_copy(field, applied, grid, side)
        if scheme == TracerBoundaryScheme.INFLOW_CLAMPED:
            return torch.where(inflow_mask(state, side), target, _wet_interior_copy(field, applied, grid, side))
        if scheme == TracerBoundaryScheme.STEVENS:
            return self._stevens_boundary_update(
                field, state, grid, side, target, applied=applied,
                is_temp=is_temp, history=history, dt_tracer=dt_tracer)
        raise ValueError(f"unsupported tracer boundary scheme: {scheme}")

    # -- Stevens (1990) boundary condition --------------------------------------

    def _stevens_boundary_update(
        self,
        field: torch.Tensor,
        state: PrognosticState,
        grid: StructuredCGrid,
        side: BoundarySide,
        target: torch.Tensor,
        *,
        applied: torch.Tensor,
        dt_tracer: float,
        is_temp: bool = True,
        history=None,
    ) -> torch.Tensor:
        """Stevens (1990) tracer OBC from MITgcm obcs_calc_stevens.F.

        Inflow:  relax toward prescribed target
        Outflow: advect tracer out + Orlanski-style phase velocity

        ``OBf = mask * (-(aFac*u_out + c) * dχ/dn - gFac * γ * (χ - χ_BC))``
        ``χ_new = χ + dt * OBf``
        """
        dt = dt_tracer  # dTtracerLev sets the phase speed and advances the tracer.
        gamma = 1.0 / self.stevens_relaxation_seconds if self.stevens_relaxation_seconds > 0 else 0.0

        boundary_val = boundary_cell_values(field, side)   # [B, nz, Nside]
        interior_val = interior_neighbor_values(field, side)  # [B, nz, Nside]

        # MIT's phase estimate spans the two interior centers. The tendency
        # uses the face between the boundary center and its first neighbor.
        if side in (BoundarySide.WEST, BoundarySide.EAST):
            distance = grid.metrics.dxC_for(field.dtype, field.device)
            velocity = state.u.data
            wet = grid.masks.u.to(device=field.device, dtype=torch.bool)
        else:
            distance = grid.metrics.dyC_for(field.dtype, field.device)
            velocity = state.v.data
            wet = grid.masks.v.to(device=field.device, dtype=torch.bool)

        normal = interior_neighbor_values(velocity, side)
        boundary_distance = interior_neighbor_values(distance, side)
        phase_limit = 0.5 * _second_interior_neighbor_values(distance, side) / dt
        boundary_wet = interior_neighbor_values(wet, side)
        phase_wet = _second_interior_neighbor_values(wet, side)

        # Express both speed and tracer differences along the outward normal.
        # This reverses the west/south velocity together with its gradient.
        if side in (BoundarySide.WEST, BoundarySide.SOUTH):
            outward = -normal
            inflow = normal >= 0
        else:
            outward = normal
            inflow = normal < 0

        # OBCS_INIT_VARIABLES zeros the Stevens tracer history on a fresh start.
        previous = torch.zeros_like(interior_val)
        prev_tracer = None if history is None else (history.prev_temp if is_temp else history.prev_salt)
        if prev_tracer is not None:
            previous = torch.where(history.valid, interior_neighbor_values(prev_tracer, side), previous)
        space = interior_val - _second_interior_neighbor_values(field, side)
        available = phase_wet & (space != 0)
        denominator = torch.where(available, space, torch.ones_like(space))
        phase = -phase_limit * (interior_val - previous) / denominator
        phase_speed = torch.where(available, torch.minimum(phase_limit, phase.clamp_min(0)), phase_limit)

        gradient = (boundary_val - interior_val) / boundary_distance
        tendency = -(outward.clamp_min(0) + phase_speed) * gradient
        tendency = tendency - inflow.to(field.dtype) * gamma * (boundary_val - target)
        increment = dt * torch.where(boundary_wet, tendency, torch.zeros_like(tendency))
        return boundary_cell_values(applied, side) + increment

    def _apply_corner_updates(
        self,
        field: torch.Tensor,
        side_updates: dict[BoundarySide, torch.Tensor],
    ) -> torch.Tensor:
        updated = field
        corners = (
            (False, False, (BoundarySide.WEST, BoundarySide.SOUTH)),
            (False, True, (BoundarySide.EAST, BoundarySide.SOUTH)),
            (True, False, (BoundarySide.WEST, BoundarySide.NORTH)),
            (True, True, (BoundarySide.EAST, BoundarySide.NORTH)),
        )
        for north, east, sides in corners:
            corner_updates = [
                corner_value_for_side(side_updates[side], side, north=north, east=east)
                for side in sides
                if side in side_updates
            ]
            if not corner_updates or (len(corner_updates) == 1 and self.uses_stevens):
                continue
            corner_value = (corner_updates[0] if len(corner_updates) == 1
                            else torch.stack(corner_updates, dim=0).mean(dim=0))
            updated = assign_corner_value(updated, north=north, east=east, values=corner_value)
        return updated


def _barotropic_depth_along_side(
    grid: StructuredCGrid,
    side: BoundarySide,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Flather's H: the wet depth sum(hFac*drF) of the inner normal face the target is written on.

    The face (u index 1 / -2 west / east, v row 1 / -2 south / north) and the geometry
    (``hfac_w_for`` / ``hfac_s_for``, the level the grid holds at this call) are those of the
    depth mean the target replaces (:func:`_apply_dual_barotropic_target_to_side`), so the
    radiated transport is H * ubar on that face.  A dry face has depth 0.  Under a nonlinear
    free surface the target frozen at step entry and the step's last application would read
    different surface levels; the model refuses that pairing
    (``ImplicitFreeSurfaceStepper.prepare_step_host``, ``ifs.admission``).
    """
    thickness = grid.layer_thickness_for(dtype, device).reshape(-1, 1)
    if side == BoundarySide.WEST:
        return (grid.hfac_w_for(dtype, device)[..., 1] * thickness).sum(dim=0)
    if side == BoundarySide.EAST:
        return (grid.hfac_w_for(dtype, device)[..., -2] * thickness).sum(dim=0)
    if side == BoundarySide.SOUTH:
        return (grid.hfac_s_for(dtype, device)[..., 1, :] * thickness).sum(dim=0)
    return (grid.hfac_s_for(dtype, device)[..., -2, :] * thickness).sum(dim=0)


OB_INDEX_MODES: tuple[str, ...] = ("full_edge", "wet_only")


@dataclass(frozen=True)
class ObcsAlongTables:
    """LOCOMgcm's analogue of MITgcm's per-column / per-row open-boundary tables.

    MITgcm's open boundary is not an edge but an index table: ``OB_Jsouth(i)`` /
    ``OB_Jnorth(i)`` carry ``Nx`` entries and ``OB_Iwest(j)`` / ``OB_Ieast(j)``
    carry ``Ny``.  An entry of ``0`` is ``OB_indexUnset``
    (``references/MITgcm/pkg/obcs/obcs_readparms.F:238``); such a column is never
    translated into a tile index (``:245-250`` initialise the whole table to
    ``OB_indexUnset``, and only entries the namelist sets are overwritten), so
    every OBCS action is guarded by
    ``IF ( OB_J*(i,bi,bj) .NE. OB_indexNone )`` — for the momentum sponge at
    ``pkg/obcs/obcs_sponge.F:67 / :102 / :137`` (u) and ``:267 / :302 / :337``
    (v).  ``BoundaryTopology`` is a per-axis boolean and cannot express that
    table, which is why it lives in its own object.

    Each side holds either

    * a ``bool`` tensor of shape ``(n_tracer,)`` — ``True`` where an open
      boundary exists (``n_tracer`` is ``nx`` for south/north, ``ny`` for
      west/east); or
    * ``None``, meaning **this side is not gated at all**.

    ``None`` is not "all True": consumers must short-circuit on it so that the
    default ``full_edge`` policy is bit-for-bit the pre-gate behaviour rather
    than an all-true :func:`torch.where`.
    """

    west: torch.Tensor | None = None
    east: torch.Tensor | None = None
    south: torch.Tensor | None = None
    north: torch.Tensor | None = None
    #: which :data:`OB_INDEX_MODES` policy produced these tables (provenance for
    #: error messages and for the export-side agreement assertion).
    mode: str = "full_edge"

    def get(self, side: BoundarySide) -> torch.Tensor | None:
        return getattr(self, side.value)

    def gate_for_axis(
        self,
        side: BoundarySide,
        *,
        length: int,
        device: torch.device | str,
        seam_entry: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """The side's table stretched onto a tensor axis of ``length`` entries.

        Returns ``None`` when the side is not gated, so the caller can skip the
        selection entirely.

        Staggered faces carry one slot more than there are tracer columns (u is
        ``[ny, nx+1]``, v is ``[ny+1, nx]``).  That last face is the first face of
        the next column along the side: the next tile's, or on a periodic axis this
        tile's own first column.  MITgcm gives it that column's table entry: its
        momentum sponge loops ``i = 0..sNx+1`` (``model/src/dynamics.F:191``,
        ``pkg/obcs/obcs_sponge.F:66-67``) and the tile table at ``i = sNx+1``
        holds the global entry ``OB_Jnorth(1+MOD(iG-1+Nx,Nx))`` (EXCH1 branch,
        ``pkg/obcs/obcs_readparms.F:812-830``), so a closed entry skips the slot.
        ``seam_entry`` is that column's entry when the face couples to another
        tile (the caller exchanges it); otherwise the slot takes this table's
        first entry, MITgcm's wrap for a single tile.  On a non-periodic edge the
        slot is the domain-edge face, whose velocity the open boundary prescribes
        or the wall mask zeroes, so only the coupled case reaches the state.

        Any other length is a shape error, not something to broadcast around.
        """
        table = self.get(side)
        if table is None:
            return None
        n_tracer = int(table.shape[0])
        if length == n_tracer:
            return table.to(device=device)
        if length == n_tracer + 1:
            first = table[:1] if seam_entry is None else seam_entry.reshape(1).to(torch.bool)
            return torch.cat([table, first.to(device=table.device)]).to(device=device)
        raise ValueError(
            f"{side.value}: gated axis of length {length} matches neither the "
            f"{n_tracer} tracer columns along that side nor the {n_tracer + 1} "
            "slots of the staggered face"
        )


def _along_wet_from_center_mask(
    center: torch.Tensor, side: BoundarySide
) -> torch.Tensor:
    """Wetness of the outermost tracer column/row of ``side``.

    🔴 This is the *only* definition of along-boundary wetness in LOCOMgcm.
    Do not re-derive it from ``Depth`` / ``bathymetry``: ``hFacMin`` clamping can
    leave ``Depth > 0`` on a column whose every cell is dry, so the two
    definitions are not equivalent in general.  Both consumers (the momentum
    sponge at runtime, the ``data.obcs`` export) reach this function with the
    same ``grid.masks.center``.
    """
    if center.ndim == 3:
        # Slice the boundary row/column FIRST, then reduce over depth: the
        # reduction then costs O(nz * n_along) instead of O(nz * ny * nx).
        if side == BoundarySide.WEST:
            line = center[:, :, 0]
        elif side == BoundarySide.EAST:
            line = center[:, :, -1]
        elif side == BoundarySide.SOUTH:
            line = center[:, 0, :]
        else:
            line = center[:, -1, :]
        return line.to(torch.bool).any(dim=0)
    if center.ndim != 2:
        raise ValueError(
            f"center mask must be 2-D (ny, nx) or 3-D (nz, ny, nx), got shape "
            f"{tuple(center.shape)}"
        )
    if side == BoundarySide.WEST:
        line = center[:, 0]
    elif side == BoundarySide.EAST:
        line = center[:, -1]
    elif side == BoundarySide.SOUTH:
        line = center[0, :]
    else:
        line = center[-1, :]
    return line.to(torch.bool)


#: ``boundary.sides`` spells sides as MITgcm namelist letters; LOCOMgcm spells
#: them as :class:`BoundarySide` values.  One map, so the runtime gate and the
#: ``data.obcs`` export cannot disagree about what ``"N"`` means.
_OB_SIDE_LETTERS: dict[str, BoundarySide] = {
    "N": BoundarySide.NORTH,
    "S": BoundarySide.SOUTH,
    "E": BoundarySide.EAST,
    "W": BoundarySide.WEST,
}


def normalize_ob_sides(sides: Iterable[BoundarySide | str]) -> tuple[BoundarySide, ...]:
    """Parse a side collection written either as letters or as full names.

    Accepts ``("N", "E", "S", "W")`` (the ``boundary.sides`` spelling) and
    ``(BoundarySide.NORTH, ...)`` / ``("north", ...)`` alike, so both consumers of
    :func:`resolve_ob_along_masks` can be handed the *same* case field verbatim.
    Order and duplicates are irrelevant to the caller, so the result is
    deduplicated in :class:`BoundarySide` declaration order.
    """
    wanted: set[BoundarySide] = set()
    for entry in sides:
        if isinstance(entry, BoundarySide):
            wanted.add(entry)
            continue
        text = str(entry)
        letter = _OB_SIDE_LETTERS.get(text.upper())
        if letter is not None:
            wanted.add(letter)
            continue
        try:
            wanted.add(BoundarySide(text.lower()))
        except ValueError as exc:
            raise ValueError(
                f"unknown open-boundary side {entry!r}; expected one of "
                f"{sorted(_OB_SIDE_LETTERS)} or "
                f"{[side.value for side in BoundarySide]}"
            ) from exc
    return tuple(side for side in BoundarySide if side in wanted)


#: The dynamics modules gated by MITgcm's OB index tables (obcs_sponge.F:67 for u,
#: :467 for theta): one OB_Jnorth/OB_Jsouth/OB_Ieast/OB_Iwest table per side serves both.
OB_GATED_SPONGE_MODULES: tuple[str, ...] = (
    "momentum_boundary_sponge_relaxation",
    "tracer_boundary_sponge_relaxation",
)


def sponge_ob_policy(module_kwargs: Mapping[str, Any], boundary: Any) -> tuple[str, tuple]:
    """``(ob_index_mode, ob_sides)`` a sponge module applies: its own kwargs, else the
    case's ``boundary.ob_index_mode`` and ``boundary.sides`` (the defaults
    ``enrich_manifest`` injects, ``sides: []`` reading as all four)."""
    mode = str(module_kwargs.get("ob_index_mode", getattr(boundary, "ob_index_mode", "full_edge")))
    sides = module_kwargs.get("ob_sides")
    if sides is None:
        sides = tuple(getattr(boundary, "sides", None) or ("N", "E", "S", "W"))
    return mode, tuple(sides)


def case_ob_policy(dynamics_entries: Iterable[Any], boundary: Any) -> tuple[str, tuple[BoundarySide, ...]]:
    """The one OB index policy of a case, as its sponge modules apply it.

    MITgcm has one table per side for every sponge, so sponges that gate on different
    tables have no MITgcm counterpart and are refused.  The sides of a ``full_edge``
    policy gate nothing and are not compared.  Without a sponge module the boundary
    fields are the policy.  Read by the ``data.obcs`` export, so the tables it writes
    are the ones the runtime sponges gate on, module overrides included.
    """
    policies: dict[str, tuple[str, tuple[BoundarySide, ...]]] = {}
    for entry in dynamics_entries:
        if entry.name not in OB_GATED_SPONGE_MODULES:
            continue
        mode, sides = sponge_ob_policy(entry.kwargs, boundary)
        policies[entry.name] = (mode, () if mode == "full_edge" else normalize_ob_sides(sides))
    if not policies:
        mode, sides = sponge_ob_policy({}, boundary)
        return mode, () if mode == "full_edge" else normalize_ob_sides(sides)
    distinct = set(policies.values())
    if len(distinct) > 1:
        raise ValueError(
            "the sponge modules gate on different open-boundary index tables "
            f"({policies}); MITgcm has one OB_J*/OB_I* table per side for the momentum "
            "and the tracer sponge (pkg/obcs/obcs_sponge.F:67, :467)"
        )
    return distinct.pop()


def resolve_ob_along_masks_from_center_mask(
    mode: str,
    center_mask: torch.Tensor | None,
    sides: Iterable[BoundarySide | str],
    *,
    device: torch.device | str | None = None,
) -> ObcsAlongTables:
    """:func:`resolve_ob_along_masks` for callers that hold the mask, not the grid.

    Used by the ``data.obcs`` export, which reads ``mask_center`` out of
    ``_pipeline_meta/model_input/grid/tensors.npz`` (written by
    ``io/runner/model_input.py`` from ``grid.masks.center``) — an equivalent mask
    produced by the same factory from the same config as the one the runtime
    gates on, not literally the same array.  Reading it — rather than rebuilding
    a grid or re-thresholding bathymetry — is what keeps the runtime gate and
    the exported namelist on a single wetness definition.
    """
    if mode not in OB_INDEX_MODES:
        raise ValueError(
            f"ob_index_mode must be one of {list(OB_INDEX_MODES)}, got {mode!r}"
        )
    if mode == "full_edge":
        # Every side ungated.  No geometry is read on this path at all: the
        # default policy must not depend on the mask even by accident.
        return ObcsAlongTables(mode=mode)

    if center_mask is None:
        raise ValueError(
            "ob_index_mode='wet_only' needs the center wet mask; got None"
        )
    wanted = set(normalize_ob_sides(sides))
    tables: dict[str, torch.Tensor | None] = {}
    for side in BoundarySide:
        if side not in wanted:
            # Side not declared in boundary.sides -> MITgcm writes no OB table for
            # it -> ungated here too, i.e. this side keeps its pre-gate behaviour.
            tables[side.value] = None
            continue
        along = _along_wet_from_center_mask(center_mask, side)
        tables[side.value] = along if device is None else along.to(device=device)
    return ObcsAlongTables(mode=mode, **tables)


def resolve_ob_along_masks(
    mode: str,
    grid: StructuredCGrid,
    sides: Iterable[BoundarySide | str],
    *,
    device: torch.device | str | None = None,
) -> ObcsAlongTables:
    """Resolve the per-side along-boundary open/closed tables for ``mode``.

    ``mode`` is a **case policy**, carried by ``boundary.ob_index_mode``, not a
    property of the geometry:

    ``full_edge``
        the whole side is open — MITgcm's ``OB_Jsouth = 720*1`` form, which
        the SCS reference configuration's ``data.obcs`` uses on three sides of a
        domain that does have dry boundary cells.  Every side comes back
        ``None`` and no geometry is read.
    ``wet_only``
        a column/row is open where its outermost tracer cell is ocean and closed
        where it is land — the form generated by the SCS_CLM reference leg's
        ``Input/OB_index_fixed.txt``, whose zero set equals ``{Depth <= 0}`` on
        the boundary row/column element-for-element on all three of its open
        sides (382 / 562 / 5 entries, symmetric difference 0).

    Geometry is therefore read **only** under ``wet_only``, and only because the
    case configuration asked for it.  A future ``explicit`` mode carrying literal
    per-side tables slots in here without touching either consumer.
    ``# ponytail:`` ceiling = arbitrary per-column tables / open boundaries on
    interior rows; upgrade path = a third mode plus per-side table fields.

    Args:
      mode: one of :data:`OB_INDEX_MODES`.
      grid: source of ``masks.center`` — the single wetness definition.  Not
        touched at all when ``mode == "full_edge"``.
      sides: the sides that carry an open boundary, as ``boundary.sides`` spells
        them.  Sides outside this set come back ``None`` (ungated), because
        MITgcm writes no OB table for them.  🔴 Both consumers must pass the
        SAME case field — deriving this set from the grid topology on one side
        and from ``boundary.sides`` on the other is how the two ends silently
        disagree about which columns are open.
      device: device of the returned tables; ``None`` keeps the mask's own.
    """
    center = None if mode == "full_edge" else grid.masks.center
    return resolve_ob_along_masks_from_center_mask(
        mode, center, sides, device=device
    )


def _apply_barotropic_target_to_side(
    field: torch.Tensor,
    side: BoundarySide,
    target: torch.Tensor,
    layer_thickness: torch.Tensor,
) -> torch.Tensor:
    thickness = layer_thickness.to(dtype=field.dtype, device=field.device).view(1, -1, 1)
    if side == BoundarySide.WEST:
        face = field[..., 0]
        barotropic = (face * thickness).sum(dim=1) / thickness.sum(dim=1)
        updated_face = face + (target - barotropic).unsqueeze(1)
        result = field.clone()
        result[..., 0] = updated_face
        return result
    if side == BoundarySide.EAST:
        face = field[..., -1]
        barotropic = (face * thickness).sum(dim=1) / thickness.sum(dim=1)
        updated_face = face + (target - barotropic).unsqueeze(1)
        result = field.clone()
        result[..., -1] = updated_face
        return result
    if side == BoundarySide.SOUTH:
        face = field[..., 0, :]
        barotropic = (face * thickness).sum(dim=1) / thickness.sum(dim=1)
        updated_face = face + (target - barotropic).unsqueeze(1)
        result = field.clone()
        result[..., 0, :] = updated_face
        return result
    face = field[..., -1, :]
    barotropic = (face * thickness).sum(dim=1) / thickness.sum(dim=1)
    updated_face = face + (target - barotropic).unsqueeze(1)
    result = field.clone()
    result[..., -1, :] = updated_face
    return result


def _wet_column_mean(face: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
    """Depth mean of ``face`` [B, nz, n] weighted by the wet thickness ``wet`` [nz, n].

    Dry layers are selected out before the sum (``where``, not a product: a dry layer may
    hold NaN or a fill value), and a face with no wet layer has no column: its mean is 0 by
    selection, so the division never sees a zero depth.
    """
    depth = wet.sum(dim=0)
    has_column = depth > 0
    safe_depth = torch.where(has_column, depth, torch.ones_like(depth))
    selected = torch.where(wet > 0, face, torch.zeros_like(face))
    return torch.where(has_column, (selected * wet).sum(dim=1) / safe_depth, torch.zeros_like(face[:, 0]))


def _apply_dual_barotropic_target_to_side(
    field: torch.Tensor,
    side: BoundarySide,
    target: torch.Tensor,
    layer_thickness: torch.Tensor,
    uv_apply_fac: float,
    mask: torch.Tensor,
    face_hfac: torch.Tensor,
) -> torch.Tensor:
    """Replace the depth mean of the inner normal face by ``target``, keep the shear.

    The mean is weighted by the face's wet thickness ``hFac * drF`` (``face_hfac``
    is hFacW or hFacS, the level the grid holds at this call), the depth weights of
    the OBCS balance's face areas (``_obcs_balance._build_geometry``) and of MITgcm
    ``obcs_balance_flow.F:118,167,216,265``.  Weighting by drF over all nz layers counted the dry layers
    below a shallow face, so the mean removed was a fraction of the real one and
    every application added the target again.  With the wet weights one
    application sets the wet-column mean to ``target`` for any wet column.  The written
    faces are selected by ``mask`` (``where``): a dry face is 0 whatever it held.
    """
    thickness = layer_thickness.to(dtype=field.dtype, device=field.device).reshape(-1, 1)
    hfac = face_hfac.to(dtype=field.dtype, device=field.device)
    result = field.clone()

    if side == BoundarySide.WEST:
        face_inner = field[..., 1]
        barotropic = _wet_column_mean(face_inner, hfac[..., 1] * thickness)
        updated_inner = face_inner + (target - barotropic).unsqueeze(1)
        m = mask[..., 1]
        result[..., 1] = torch.where(m > 0, updated_inner * m, torch.zeros_like(updated_inner))
        result[..., 0] = torch.where(m > 0, updated_inner * m * uv_apply_fac, torch.zeros_like(updated_inner))
        
    elif side == BoundarySide.EAST:
        face_inner = field[..., -2]
        barotropic = _wet_column_mean(face_inner, hfac[..., -2] * thickness)
        updated_inner = face_inner + (target - barotropic).unsqueeze(1)
        m = mask[..., -2]
        result[..., -2] = torch.where(m > 0, updated_inner * m, torch.zeros_like(updated_inner))
        result[..., -1] = torch.where(m > 0, updated_inner * m * uv_apply_fac, torch.zeros_like(updated_inner))
        
    elif side == BoundarySide.SOUTH:
        face_inner = field[..., 1, :]
        barotropic = _wet_column_mean(face_inner, hfac[..., 1, :] * thickness)
        updated_inner = face_inner + (target - barotropic).unsqueeze(1)
        m = mask[..., 1, :]
        result[..., 1, :] = torch.where(m > 0, updated_inner * m, torch.zeros_like(updated_inner))
        result[..., 0, :] = torch.where(m > 0, updated_inner * m * uv_apply_fac, torch.zeros_like(updated_inner))
        
    elif side == BoundarySide.NORTH:
        face_inner = field[..., -2, :]
        barotropic = _wet_column_mean(face_inner, hfac[..., -2, :] * thickness)
        updated_inner = face_inner + (target - barotropic).unsqueeze(1)
        m = mask[..., -2, :]
        result[..., -2, :] = torch.where(m > 0, updated_inner * m, torch.zeros_like(updated_inner))
        result[..., -1, :] = torch.where(m > 0, updated_inner * m * uv_apply_fac, torch.zeros_like(updated_inner))
        
    return result


@dataclass
@register_boundary_condition("flather_barotropic")
class FlatherBarotropicBoundaryCondition(BoundaryCondition):
    """Flather open-boundary condition with optional OBCSbalance.

    When ``balance_obc`` is True, a uniform velocity correction is applied
    to all participating open boundaries so that the net barotropic
    transport across all OB faces is zero (volume conservation).

    MITgcm reference: ``pkg/obcs/obcs_balance_flow.F``
    """

    sampling_rule = SamplingRule(
        SampleClock.END,
        ("_active_eta_values", "_active_normal_velocity_values", "_active_tangential_velocity_values"),
        entry_clock=SampleClock.START,
        providers=("source", "fallback_boundary"),
    )

    gravity: float = 9.81
    uv_apply_fac: float = 1.0
    eta_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    normal_velocity_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    tangential_velocity_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    fallback_boundary: BoundaryCondition | None = None
    source: object = None  # BarotropicOpenBoundarySource (kept as object to avoid circular import)
    time_mode: str = "linear"
    update_interval_seconds: float | None = None
    balance_obc: bool = False
    balance_factors: dict[str, float] | None = None
    name: str = "flather_barotropic_open_boundary"
    supports_open_boundary: bool = True
    supports_barotropic_open_boundary: bool = True
    current_time_seconds: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fallback_boundary is None:
            self.fallback_boundary = StructuredBoundaryCondition()
        # Initialize active values from static values
        self._active_eta_values: SideBoundaryTensorSet = self.eta_values
        self._active_normal_velocity_values: SideBoundaryTensorSet = self.normal_velocity_values
        self._active_tangential_velocity_values: SideBoundaryTensorSet = self.tangential_velocity_values
        if self.source is not None:
            self.update_model_time(0.0)
        # Default balance factors: all sides participate equally
        if self.balance_factors is None:
            self._balance_factors: dict[str, float] = {
                s.value: 1.0 for s in BoundarySide
            }
        else:
            self._balance_factors = dict(self.balance_factors)
        negative = {side: fac for side, fac in self._balance_factors.items() if fac < 0.0}
        if negative:
            # MITgcm balances a side with OBCS_balanceFac < 0 on its own
            # (obcs_balance_flow.F:137-147); the shared flow implements the joint balance only.
            raise ValueError(f"Flather balance_factors {negative}: a negative factor (separate per-side "
                             "balance) is not implemented")

    def update_model_time(self, model_time_seconds: float) -> None:
        """Update boundary data from time-varying source."""
        self.current_time_seconds = float(model_time_seconds)
        if self.source is None:
            return
        sample = self.source.sample(
            model_time_seconds,
            mode=self.time_mode,
            update_interval_seconds=self.update_interval_seconds,
        )
        self._active_eta_values = sample.eta
        self._active_normal_velocity_values = sample.normal_velocity
        self._active_tangential_velocity_values = getattr(sample, "tangential_velocity", SideBoundaryTensorSet())

    def freeze_targets(
        self, state: PrognosticState, grid: StructuredCGrid,
    ) -> dict[BoundarySide, tuple[torch.Tensor, float]]:
        """The Flather normal-velocity targets of one step, from its start state.

        A step evaluates them once, when it creates its boundary phases
        (``steppers.base.tracer_boundary_phases``), and passes them to every
        ``enforce`` inside the step, as MITgcm computes OBCS values once per
        step from the step-start state (``OBCS_CALC``, do_oceanic_phys.F:322)
        and applies them several times (``OBCS_APPLY_UV``, dynamics.F:610 and
        momentum_correction_step.F:95).  Re-evaluating after the pressure
        correction from the CG-solved eta makes transport and eta inconsistent
        and grows exponentially on the exactConserv path.  The result is held
        by the caller only: nothing is stored here, so an ``enforce`` outside a
        step evaluates its own targets.

        The formula reads eta and the sampled boundary values only.  The
        fallback is a velocity boundary (structured by default, writing u and
        v), so it is applied by ``enforce`` alone: a step applies it, and
        exchanges a split periodic face, as often as before.
        """
        return self._normal_targets(state.eta.data, grid)

    def _normal_targets(
        self, eta: torch.Tensor, grid: StructuredCGrid,
    ) -> dict[BoundarySide, tuple[torch.Tensor, float]]:
        gravity = eta.new_tensor(self.gravity)
        side_targets: dict[BoundarySide, tuple[torch.Tensor, float]] = {}
        for side in BoundarySide:
            if not _side_is_open(side.value, grid):
                continue
            # MITgcm samples the adjacent interior eta when forming the open-
            # boundary free-surface signal (e.g. obcs_calc_stevens.F uses
            # etaN(i-1,j) / etaN(i+1,j) / etaN(i,j±1)), then applies the
            # prescribed boundary eta separately on the boundary strip.
            eta_internal = interior_neighbor_values(eta, side)
            eta_external = self._side_target(
                self._active_eta_values,
                side,
                grid,
                batch_size=eta.shape[0],
                dtype=eta.dtype,
                device=eta.device,
            )
            normal_external = self._side_target(
                self._active_normal_velocity_values,
                side,
                grid,
                batch_size=eta.shape[0],
                dtype=eta.dtype,
                device=eta.device,
            )
            # H is the written face's wet depth; a dry face has no column and its target is 0
            # by selection (no clamp decides a physical value, no division by 0).
            depth = _barotropic_depth_along_side(grid, side, dtype=eta.dtype, device=eta.device)
            wet = depth > 0
            safe_depth = torch.where(wet, depth, torch.ones_like(depth))
            radiation_coefficient = torch.sqrt(gravity / safe_depth).view(1, -1)
            sign = -1.0 if side in (BoundarySide.WEST, BoundarySide.SOUTH) else 1.0

            normal_target = torch.where(
                wet.view(1, -1),
                normal_external + sign * radiation_coefficient * (eta_internal - eta_external),
                torch.zeros_like(normal_external))

            side_targets[side] = (normal_target, sign)
        return side_targets

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "FlatherBarotropicBoundaryCondition":
        moved = FlatherBarotropicBoundaryCondition(
            gravity=self.gravity,
            uv_apply_fac=self.uv_apply_fac,
            eta_values=self.eta_values.to(device, dtype),
            normal_velocity_values=self.normal_velocity_values.to(device, dtype),
            tangential_velocity_values=self.tangential_velocity_values.to(device, dtype),
            fallback_boundary=_maybe_to(self.fallback_boundary, device, dtype),
            source=_maybe_to(self.source, device, dtype),
            time_mode=self.time_mode,
            update_interval_seconds=self.update_interval_seconds,
            balance_obc=self.balance_obc,
            balance_factors=dict(self._balance_factors),
            name=self.name,
            supports_open_boundary=self.supports_open_boundary,
            supports_barotropic_open_boundary=self.supports_barotropic_open_boundary,
        )
        moved.current_time_seconds = self.current_time_seconds
        if self.source is not None or self.current_time_seconds != 0.0:
            moved.update_model_time(self.current_time_seconds)
        return moved

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "FlatherBarotropicBoundaryCondition":
        return self.to_runtime(device, dtype)

    def _side_target(
        self,
        values: SideBoundaryTensorSet | None,
        side: BoundarySide,
        grid: StructuredCGrid,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        target = values.get(side) if values is not None else None
        if target is None:
            target_length = grid.ny if side in (BoundarySide.WEST, BoundarySide.EAST) else grid.nx
            return torch.zeros((batch_size, target_length), dtype=dtype, device=device)
        return expand_barotropic_side_boundary_tensor(
            target,
            side,
            grid,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

    def enforce_tendency(self, tendency: StateTendency, grid: StructuredCGrid) -> StateTendency:
        """Zero velocity and eta tendencies at open-boundary faces.

        Mimics MITgcm maskInW/maskInS/maskInC which zero tendencies at
        boundary points that will be overwritten by OBCS_APPLY.
        Zeroes BOTH outer face AND dual-point inner face for normal velocity.
        Also zeroes eta tendency at boundary cells to prevent the
        Flather-continuity feedback loop.
        """
        eta = tendency.eta.clone()
        u = tendency.u.clone()
        v = tendency.v.clone()
        for side in BoundarySide:
            if not _side_is_open(side.value, grid):
                continue
            if side == BoundarySide.SOUTH:
                eta[..., 0, :] = 0.0
                u[..., 0, :] = 0.0
                v[..., 0, :] = 0.0
                v[..., 1, :] = 0.0
            elif side == BoundarySide.NORTH:
                eta[..., -1, :] = 0.0
                u[..., -1, :] = 0.0
                v[..., -1, :] = 0.0
                v[..., -2, :] = 0.0
            elif side == BoundarySide.WEST:
                eta[..., :, 0] = 0.0
                u[..., 0] = 0.0
                u[..., 1] = 0.0
                v[..., 0] = 0.0
            elif side == BoundarySide.EAST:
                eta[..., :, -1] = 0.0
                u[..., -1] = 0.0
                u[..., -2] = 0.0
                v[..., -1] = 0.0
        return StateTendency(
            eta=eta,
            u=u,
            v=v,
            temp=tendency.temp,
            salt=tendency.salt,
        )

    def enforce(
        self, state: PrognosticState, grid: StructuredCGrid, *,
        flather_targets: dict[BoundarySide, tuple[torch.Tensor, float]] | None = None,
    ) -> PrognosticState:
        """Apply the boundary; ``flather_targets`` are the calling step's (``freeze_targets``).

        Without them the targets are evaluated from this call's state, which is
        what a boundary application outside a step (model build, a run seed, a
        diagnostic) does.
        """
        bounded = self.fallback_boundary.enforce(state, grid) if self.fallback_boundary is not None else state
        eta = bounded.eta.data
        u = bounded.u.data.clone()
        v = bounded.v.data.clone()

        u_mask = grid.masks.u_float(dtype=u.dtype, device=u.device) if grid.masks.has_3d_masks else None
        v_mask = grid.masks.v_float(dtype=v.dtype, device=v.device) if grid.masks.has_3d_masks else None

        # ---- Pass 1: Tangential Clamping ----
        for side in BoundarySide:
            if not _side_is_open(side.value, grid):
                continue
            tangential_external = self._side_target(
                self._active_tangential_velocity_values,
                side,
                grid,
                batch_size=eta.shape[0],
                dtype=eta.dtype,
                device=eta.device,
            )
            # Pad tangential_external by 1 element at the end to match staggered field lengths (nx+1, ny+1)
            tangential_padded = torch.nn.functional.pad(tangential_external, (0, 1), mode='replicate')
            
            # Tangential assignment
            if side == BoundarySide.SOUTH:
                m = 1.0 if u_mask is None else u_mask[..., 0, :]
                u[..., 0, :] = tangential_padded * m
            elif side == BoundarySide.NORTH:
                m = 1.0 if u_mask is None else u_mask[..., -1, :]
                u[..., -1, :] = tangential_padded * m
            elif side == BoundarySide.WEST:
                m = 1.0 if v_mask is None else v_mask[..., 0]
                v[..., 0] = tangential_padded * m
            elif side == BoundarySide.EAST:
                m = 1.0 if v_mask is None else v_mask[..., -1]
                v[..., -1] = tangential_padded * m

        # ---- Pass 2: Flather targets for all open sides ----
        # A step passes the targets it evaluated from its start state, so every
        # application inside the step writes the same values, as OBCS_APPLY_UV
        # writes the OBWu that OBCS_CALC computed.  The balance below edits
        # this dict, never the step's.
        side_targets = (dict(flather_targets) if flather_targets is not None
                        else self._normal_targets(state.eta.data, grid))

        # ---- OBCSbalance: uniform velocity correction ----
        # MITgcm obcs_balance_flow.F: compute net inFlow (positive = water
        # entering domain), then redistribute a uniform correction so that
        # net inFlow = 0.
        #
        # Sign convention (MITgcm C-grid):
        #   West/South: positive velocity = into domain  → inFlow += Q
        #   East/North: positive velocity = out of domain → inFlow -= Q
        #
        # Correction sign:
        #   East/North: OBEu/OBNv += +inFlow_per_area * fac  (add outflow)
        #   West/South: OBWu/OBSv -= +inFlow_per_area * fac  (reduce inflow)
        if self.balance_obc and side_targets:
            # OBCS_BALANCE_FLOW through the shared implementation the prescribed
            # boundary and the momentum sponge use: inner normal faces, MITgcm's
            # maskInW/S (no corner of two open sides), OBCS_balanceFac weights.
            # A Flather target is depth uniform, so the transport is the target
            # times the face's wet area and the correction shifts the target.
            order = (BoundarySide.WEST, BoundarySide.EAST, BoundarySide.SOUTH, BoundarySide.NORTH)
            factors = tuple(float(self._balance_factors.get(s.value, 1.0)) for s in order)
            depth_uniform = {s: (side_targets[s][0].unsqueeze(-2) if s in side_targets else None)
                             for s in order}
            q, geometry = _obcs_balance_flow(
                SideBoundaryTensorSet(west=depth_uniform[BoundarySide.WEST],
                                      east=depth_uniform[BoundarySide.EAST]),
                SideBoundaryTensorSet(south=depth_uniform[BoundarySide.SOUTH],
                                      north=depth_uniform[BoundarySide.NORTH]),
                grid, dtype=eta.dtype, device=eta.device, factors=factors)
            if q is not None:
                for index, side in enumerate(order):
                    if side not in side_targets or factors[index] == 0.0:
                        continue
                    normal_target, sign = side_targets[side]
                    # obcs_balance_flow.F:393-457: E/N += fac*q, W/S -= fac*q, on wet faces
                    shift = (1.0 if side in (BoundarySide.EAST, BoundarySide.NORTH) else -1.0) * factors[index] * q
                    wet = geometry.correction_masks[index].amax(dim=0) > 0
                    side_targets[side] = (torch.where(wet, normal_target + shift, normal_target), sign)

        # ---- Pass 3: Apply Normal (balanced) targets using Double-Point override ----
        u_mask = grid.masks.u_float(dtype=u.dtype, device=u.device) if grid.masks.has_3d_masks else torch.ones_like(u)
        v_mask = grid.masks.v_float(dtype=v.dtype, device=v.device) if grid.masks.has_3d_masks else torch.ones_like(v)
        for side, (normal_target, sign) in side_targets.items():
            if side in (BoundarySide.WEST, BoundarySide.EAST):
                u = _apply_dual_barotropic_target_to_side(u, side, normal_target, grid.layer_thickness, self.uv_apply_fac, u_mask,
                                                          grid.hfac_w_for(u.dtype, u.device))
            else:
                v = _apply_dual_barotropic_target_to_side(v, side, normal_target, grid.layer_thickness, self.uv_apply_fac, v_mask,
                                                          grid.hfac_s_for(v.dtype, v.device))
        # ---- Pass 4: Clamp boundary eta to external target ----
        # Prevents Flather-continuity feedback: boundary cell eta is set to
        # external target rather than evolving freely from continuity.
        eta_out = eta.clone()
        for side in BoundarySide:
            if not _side_is_open(side.value, grid):
                continue
            eta_ext = self._side_target(
                self._active_eta_values,
                side,
                grid,
                batch_size=eta.shape[0],
                dtype=eta.dtype,
                device=eta.device,
            )
            if side == BoundarySide.SOUTH:
                eta_out[..., 0, :] = eta_ext
            elif side == BoundarySide.NORTH:
                eta_out[..., -1, :] = eta_ext
            elif side == BoundarySide.WEST:
                eta_out[..., :, 0] = eta_ext
            elif side == BoundarySide.EAST:
                eta_out[..., :, -1] = eta_ext

        return bounded.with_tensors(eta=eta_out, u=u, v=v)


#: Tangential-velocity zeroing passes of the prescribed boundary: none, or the
#: obcs_apply_uv.F pre-pass over the full staggered range or truncated to nx, ny.
TANGENTIAL_ZERO_MODES = ("none", "phase1_full", "phase1_truncated")


@dataclass
@register_boundary_condition("prescribed_3d")
class Prescribed3DBoundaryCondition(BoundaryCondition):
    """Rigid 3D prescribed open-boundary condition.

    Mimics MITgcm's useOBCSprescribe=.TRUE. by overwriting exact 3D values
    for u, v, temp, and salt at the boundaries every time step.  Boundary eta
    is NOT prescribed — MITgcm's OBCS_APPLY_ETA is commented out, so boundary
    η stays 0 via maskInC (matching the inline note in ``enforce``).
    Normal velocities are dual-clamped to maintain exact transport.
    """

    accepts_tidal_config: ClassVar[bool] = True

    sampling_rule = SamplingRule(
        SampleClock.END,
        ("_active_eta_values", "_active_u_values", "_active_v_values",
         "_active_temp_values", "_active_salt_values", "_balanced_u_values",
         "_balanced_v_values", "_prepared_balanced_targets"),
        entry_clock=SampleClock.START,
        providers=("tracer_source", "barotropic_source", "fallback_boundary", "tidal_config"),
    )
    tracer_source: Any | None = None
    barotropic_source: Any | None = None
    time_mode: str = "linear"
    update_interval_seconds: float | None = None
    uv_apply_fac: float = 1.0
    fallback_boundary: BoundaryCondition | None = None
    name: str = "prescribed_3d_boundary"
    supports_open_boundary: bool = True
    supports_barotropic_open_boundary: bool = True
    tangential_zero_mode: str = 'none'
    apply_tangential_velocity: bool = False
    # OBCS Tides: barotropic tidal velocity overlay (MITgcm OBCS_ADD_TIDES).
    # When set, tidal harmonic velocities are added to balanced OB targets
    # AFTER OBCS_BALANCE_FLOW, matching pkg/obcs/obcs_adjust.F:48-51.
    tidal_config: Any | None = None
    # MITgcm useOBCSbalance flag — when True, redistribute net inFlow as
    # uniform velocity correction across the four open boundaries.  MITgcm
    # default is .FALSE.; exp4 verification case relies on the .FALSE. default
    # (no explicit `useOBCSbalance` in data.obcs).  Setting this True with a
    # non-zero net inFlow leaks ~1 Sv into the v field at S/N OB rows even
    # when SOUTH/NORTH have prescribed v=0, distorting the long-term attractor.
    # Set to False to match exp4-style runs; keep True for cases that match
    # MITgcm useOBCSbalance=.TRUE. (e.g. linfs_30d_ocean_jmd95 production).
    enforce_obcs_balance: bool = True
    # Value scheme of each side, one of OPEN_BOUNDARY_VALUE_SCHEMES; a side not
    # named takes PRESCRIBED.  Normalised to all four sides at construction.
    value_schemes: Mapping[str, str] = field(default_factory=dict)
    # OBCS_PARM02 with abEps and deltaT (OrlanskiParameters); given exactly when
    # a side is ORLANSKI.
    orlanski: OrlanskiParameters | Mapping[str, Any] | None = None

    current_time_seconds: float = field(default=0.0, init=False, repr=False)
    _history_adapter: object = field(default=None, init=False, repr=False)
    
    _active_eta_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet, init=False, repr=False)
    _active_u_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet, init=False, repr=False)
    _active_v_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet, init=False, repr=False)
    _active_temp_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet, init=False, repr=False)
    _active_salt_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet, init=False, repr=False)

    # OBCS_BALANCE_FLOW cache: balanced normal-velocity targets are computed
    # lazily on first enforce() per step and reused for subsequent enforce()
    # calls within the same time-key (initial / predicted / corrected /
    # final).  Invalidated when ``update_model_time`` advances the active
    # source samples.  Mirrors MIT one-per-step balance topology: targets
    # are mutated once then consumed by both OBCS_APPLY_UV (face clamp) and
    # OBCS_SPONGE_U/V (sponge tendency).
    _balanced_u_values: SideBoundaryTensorSet | None = field(default=None, init=False, repr=False)
    _balanced_v_values: SideBoundaryTensorSet | None = field(default=None, init=False, repr=False)
    _balance_cache_key: tuple | None = field(default=None, init=False, repr=False)
    # The eager pre-step's answer, as ``(bal_u, bal_v, dtype, device)``.
    # ``None`` = nothing prepared for this clock => the legacy in-step compute.
    _prepared_balanced_targets: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # The removed 'inline_full' / 'inline_truncated' modes zeroed tangential
        # velocity inside the side loop, the mixed order obcs_apply_uv.F excludes
        # (normal writes must win at corners); a config naming one fails here.
        if self.tangential_zero_mode not in TANGENTIAL_ZERO_MODES:
            raise ValueError(
                f"tangential_zero_mode must be one of {TANGENTIAL_ZERO_MODES}, "
                f"got {self.tangential_zero_mode!r}"
            )
        if self.fallback_boundary is None:
            self.fallback_boundary = StructuredBoundaryCondition()
        from locomgcm.step.boundary_history import BoundaryHistoryAdapter

        schemes = {side: OpenBoundaryValueScheme.PRESCRIBED for side in BOUNDARY_SIDES}
        for side, scheme in dict(self.value_schemes).items():
            if scheme not in OPEN_BOUNDARY_VALUE_SCHEMES:
                raise ValueError(
                    f"prescribed_3d value scheme {scheme!r} for side {side!r} is not one of "
                    f"{[name.value for name in OPEN_BOUNDARY_VALUE_SCHEMES]}"
                )
            schemes[BoundarySide(side)] = OpenBoundaryValueScheme(scheme)
        self.value_schemes = schemes
        if isinstance(self.orlanski, Mapping):
            self.orlanski = OrlanskiParameters(**self.orlanski)
        if self.uses_orlanski and self.orlanski is None:
            raise ValueError("prescribed_3d has an Orlanski side but no orlanski parameters")
        if not self.uses_orlanski and self.orlanski is not None:
            raise ValueError("prescribed_3d orlanski parameters are given but no side is Orlanski")
        if self.uses_orlanski and self.tidal_config is not None:
            raise NotImplementedError("prescribed_3d tides with an Orlanski side are not supported")
        self._history_adapter = BoundaryHistoryAdapter()

        if self.tracer_source is not None or self.barotropic_source is not None:
            self.update_model_time(0.0)

    @property
    def uses_orlanski(self) -> bool:
        return any(scheme == OpenBoundaryValueScheme.ORLANSKI for scheme in self.value_schemes.values())

    @property
    def boundary_history(self):
        return self._history_adapter.history

    def update_model_time(self, model_time_seconds: float) -> None:
        self.current_time_seconds = float(model_time_seconds)
        if self.tracer_source is not None:
            sample = self.tracer_source.sample(
                model_time_seconds,
                mode=self.time_mode,
                update_interval_seconds=self.update_interval_seconds,
            )
            self._active_u_values = getattr(sample, "u", SideBoundaryTensorSet())
            self._active_v_values = getattr(sample, "v", SideBoundaryTensorSet())
            self._active_temp_values = sample.temp
            self._active_salt_values = sample.salt

        if self.barotropic_source is not None:
            bsample = self.barotropic_source.sample(
                model_time_seconds,
                mode=self.time_mode,
                update_interval_seconds=self.update_interval_seconds,
            )
            self._active_eta_values = bsample.eta

        # Active-source values changed → invalidate balanced-targets cache.
        self._balanced_u_values = None
        self._balanced_v_values = None
        self._balance_cache_key = None
        self._prepared_balanced_targets = None

    def prepare_balanced_targets(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Eager pre-step: resolve the balanced targets for the installed clock.

        Compiled execution observed four graph breaks at
        ``open.py:1627:_get_balanced_targets``).  The in-step memo is keyed on
        ``(id(active_u), id(active_v), current_time_seconds, str(dtype),
        str(device))``: ``id()`` of a Python object is untraceable and the float
        clock bakes a fresh guard every step, so the whole
        ``compute_obcs_balanced_normal_velocities`` body was re-traced inside the
        compiled step.  Producing it here -- from the same sampled side values,
        grid, dtype and device, through the same call -- keeps every bit and
        leaves the step a plain attribute read.  Called by
        ``TorchRegionalOceanModel.prepare_step_times`` for both clocks; the
        current sample is installed by the eager pre-step and the future one
        arrives on the traced step's own signature (``StepPlan.boundary``).
        """
        bal_u, bal_v = self._compute_balanced_targets(
            grid, dtype=dtype, device=device
        )
        self._prepared_balanced_targets = (bal_u, bal_v, dtype, device)

    def _get_balanced_targets(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> tuple[SideBoundaryTensorSet, SideBoundaryTensorSet]:
        """Return (balanced_u, balanced_v) for this clock.

        Serves :meth:`prepare_balanced_targets`' answer when the eager pre-step
        produced one at this dtype/device; otherwise falls back to the legacy
        in-step compute (direct ``enforce`` callers -- tests, API users -- never
        run the pre-step).
        """
        prepared = self._prepared_balanced_targets
        if (
            prepared is not None
            and prepared[2] == dtype
            and prepared[3] == device
        ):
            return prepared[0], prepared[1]
        return self._compute_balanced_targets(grid, dtype=dtype, device=device)

    @torch._dynamo.disable
    def _compute_balanced_targets(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> tuple[SideBoundaryTensorSet, SideBoundaryTensorSet]:
        """Compute once per (active values, time, dtype, device).

        When :attr:`enforce_obcs_balance` is ``False`` returns the raw active
        values unchanged so the OBCSbalance projection is fully bypassed.

        Opaque to Dynamo for the same reason as
        ``MomentumBoundarySpongeRelaxation._compute_balanced_targets``: an
        ``id()`` and float-clock memo key, reached from the traced ``enforce``
        of the steppers that have no eager prepare.
        """
        if not self.enforce_obcs_balance:
            u, v = self._active_u_values, self._active_v_values
            if self.tidal_config is not None:
                from .tidal import add_tidal_velocities
                u, v = add_tidal_velocities(u, v, grid, self.current_time_seconds, self.tidal_config)
            return u, v
        cache_key = (
            id(self._active_u_values),
            id(self._active_v_values),
            self.current_time_seconds,
            str(dtype),
            str(device),
        )
        if self._balance_cache_key == cache_key and self._balanced_u_values is not None:
            return self._balanced_u_values, self._balanced_v_values
        bal_u, bal_v = compute_obcs_balanced_normal_velocities(
            self._active_u_values,
            self._active_v_values,
            grid,
            dtype=dtype,
            device=device,
        )
        # OBCS_ADD_TIDES: post-balance tidal overlay (obcs_adjust.F:48-51)
        if self.tidal_config is not None:
            from .tidal import add_tidal_velocities
            bal_u, bal_v = add_tidal_velocities(
                bal_u, bal_v, grid, self.current_time_seconds, self.tidal_config,
            )
        self._balanced_u_values = bal_u
        self._balanced_v_values = bal_v
        self._balance_cache_key = cache_key
        return bal_u, bal_v

    def _runtime_fields(self, device: torch.device | str, dtype: torch.dtype | None) -> dict[str, Any]:
        """The init fields holding device data, placed on ``device``/``dtype``.

        The single owner of placement for this boundary and its subclasses:
        a subclass extends this mapping with its own fields and does not
        override :meth:`to_runtime`, so no inherited field is left behind.
        """
        return {
            "tracer_source": _maybe_to(self.tracer_source, device, dtype),
            "barotropic_source": _maybe_to(self.barotropic_source, device, dtype),
            "fallback_boundary": _maybe_to(self.fallback_boundary, device, dtype),
            "tidal_config": self.tidal_config.to(device, dtype) if self.tidal_config is not None else None,
        }

    def to_runtime(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "Prescribed3DBoundaryCondition":
        # ``replace`` keeps the concrete class and every other init field.
        moved = replace(self, **self._runtime_fields(device, dtype))
        moved.current_time_seconds = self.current_time_seconds
        # Force re-evaluation on new device
        if moved.tracer_source is not None or moved.barotropic_source is not None or moved.current_time_seconds != 0.0:
            moved.update_model_time(moved.current_time_seconds)
        return moved

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "Prescribed3DBoundaryCondition":
        return self.to_runtime(device, dtype)
        
    def _side_target_3d(self, values: SideBoundaryTensorSet, side: BoundarySide, grid: StructuredCGrid, batch_size: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor | None:
        target = values.get(side)
        if target is None:
            return None
        return expand_side_boundary_tensor(target, side, grid, batch_size=batch_size, dtype=dtype, device=device)

    def _side_target_2d(self, values: SideBoundaryTensorSet, side: BoundarySide, grid: StructuredCGrid, batch_size: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor | None:
        target = values.get(side)
        if target is None:
            return None
        return expand_barotropic_side_boundary_tensor(target, side, grid, batch_size=batch_size, dtype=dtype, device=device)

    def enforce_tendency(self, tendency: StateTendency, grid: StructuredCGrid) -> StateTendency:
        eta = tendency.eta.clone()
        u = tendency.u.clone()
        v = tendency.v.clone()
        temp = tendency.temp.clone()
        salt = tendency.salt.clone()
        
        # 60 ``aten::_local_scalar_dense`` per step came from here, the largest
        # single non-PCG site.  ``x[slice] = 0.0`` on a
        # CUDA tensor is ``copy_to -> Tensor.fill_(cpu 0-d tensor)``, and
        # ``fill_`` reads that scalar back with ``value.item()`` because the
        # devices differ -- one host read per assignment, 20 assignments per
        # call, three calls per step.  ``.zero_()`` on the same view issues the
        # same ``aten::fill_`` kernel with the same +0.0 and no read.
        # ``BOUNDARY_SIDES`` replaces ``EnumType.__iter__``, same
        # members in the same order.
        for side in BOUNDARY_SIDES:
            if not _side_is_open(side.value, grid):
                continue
            if side == BoundarySide.SOUTH:
                eta[..., 0, :].zero_()
                u[..., 0, :].zero_()
                v[..., 0, :].zero_()
                temp[..., 0, :].zero_()
                salt[..., 0, :].zero_()
            elif side == BoundarySide.NORTH:
                eta[..., -1, :].zero_()
                u[..., -1, :].zero_()
                v[..., -1, :].zero_()
                temp[..., -1, :].zero_()
                salt[..., -1, :].zero_()
            elif side == BoundarySide.WEST:
                eta[..., :, 0].zero_()
                u[..., 0].zero_()
                v[..., 0].zero_()
                temp[..., :, 0].zero_()
                salt[..., :, 0].zero_()
            elif side == BoundarySide.EAST:
                eta[..., :, -1].zero_()
                u[..., -1].zero_()
                v[..., -1].zero_()
                temp[..., :, -1].zero_()
                salt[..., :, -1].zero_()

        return StateTendency(eta=eta, u=u, v=v, temp=temp, salt=salt)

    def enforce(self, state: PrognosticState, grid: StructuredCGrid, *,
                orlanski_source: PrognosticState | None = None) -> PrognosticState:
        """Apply the open-boundary values.

        With an Orlanski side, ``orlanski_source`` is the step-start state that
        this step's OBCS_CALC reads; an application without it computes nothing
        and returns ``state`` as it is (see :meth:`enforce_with_history`).
        """
        if not self.uses_orlanski:
            return self._enforce_values(state, grid, orlanski_values={})
        stamp = self._history_adapter.active_stamp
        if stamp is None:
            stamp = self.current_time_seconds
        result, candidate = self.enforce_with_history(
            state, grid, history=self.boundary_history, stamp=stamp, orlanski_source=orlanski_source)
        if candidate is not None:
            self._history_adapter.candidate = candidate
        return result

    def enforce_with_history(self, state, grid, *, history, stamp, orlanski_source):
        """Evaluate the Orlanski sides from an explicit history; return the state and the candidate.

        ``orlanski_source`` is the state OBCS_CALC reads (do_oceanic_phys.F:322
        passes the step-start fields; obcs_init_variables.F:386 the initial
        ones).  ``None`` is an application with no OBCS_CALC, such as the
        step-entry one, which has no MITgcm counterpart: the state already holds
        the values the previous step applied, so it is returned unchanged.
        The candidate carries the new Orlanski storage; the caller publishes it
        when the step is accepted.  Nothing is stored here.
        """
        from locomgcm.step.boundary_history import BoundaryHistory

        if orlanski_source is None:
            return state, None
        values, storage = self._orlanski_values(orlanski_source, grid, history)
        return (self._enforce_values(state, grid, orlanski_values=values),
                BoundaryHistory.sample(orlanski_source, stamp, storage))

    def _orlanski_values(self, source: PrognosticState, grid: StructuredCGrid, history):
        """ORLANSKI_<side> of every Orlanski side on ``source``.

        Returns ``({side: (u, v, temp, salt)}, {side value: storage})``, the
        values [B, nz, n] and the storage of ``locomgcm.boundaries.orlanski``.
        Points and spacings follow orlanski_<side>.F (LOCOMgcm index i is MITgcm index i+1, u has nx+1 columns, v ny+1 rows).
        """
        u, v = source.u.data, source.v.data
        temp, salt = source.temp.data, source.salt.data
        dtype, device = u.dtype, u.device
        ny, nx = grid.ny, grid.nx
        metrics = grid.metrics
        values: dict[BoundarySide, tuple[torch.Tensor, ...]] = {}
        storage: dict[str, torch.Tensor] = {}
        for side in BOUNDARY_SIDES:
            if self.value_schemes[side] != OpenBoundaryValueScheme.ORLANSKI:
                continue
            if side == BoundarySide.EAST:
                dxF = metrics.dxF_for(dtype, device)
                dxV = metrics.dxV_for(dtype, device)
                dxC = metrics.dxC_for(dtype, device)
                points = (
                    (u[..., -2], u[..., -3], u[..., -4], dxF[:, -3], dxF[:, -2]),
                    (v[..., :ny, -1], v[..., :ny, -2], v[..., :ny, -3], dxV[:ny, -3], dxV[:ny, -2]),
                    (temp[..., -1], temp[..., -2], temp[..., -3], dxC[:, -3], dxC[:, -2]),
                    (salt[..., -1], salt[..., -2], salt[..., -3], dxC[:, -3], dxC[:, -2]),
                )
            elif side == BoundarySide.WEST:
                dxF = metrics.dxF_for(dtype, device)
                dxV = metrics.dxV_for(dtype, device)
                dxC = metrics.dxC_for(dtype, device)
                points = (
                    (u[..., 1], u[..., 2], u[..., 3], dxF[:, 2], dxF[:, 1]),
                    (v[..., :ny, 0], v[..., :ny, 1], v[..., :ny, 2], dxV[:ny, 2], dxV[:ny, 1]),
                    (temp[..., 0], temp[..., 1], temp[..., 2], dxC[:, 2], dxC[:, 1]),
                    (salt[..., 0], salt[..., 1], salt[..., 2], dxC[:, 2], dxC[:, 1]),
                )
            elif side == BoundarySide.NORTH:
                dyU = metrics.dyU_for(dtype, device)
                dyF = metrics.dyF_for(dtype, device)
                dyC = metrics.dyC_for(dtype, device)
                points = (
                    (u[..., -1, :nx], u[..., -2, :nx], u[..., -3, :nx], dyU[-3, :nx], dyU[-2, :nx]),
                    (v[..., -2, :], v[..., -3, :], v[..., -4, :], dyF[-3, :], dyF[-2, :]),
                    (temp[..., -1, :], temp[..., -2, :], temp[..., -3, :], dyC[-3, :], dyC[-2, :]),
                    (salt[..., -1, :], salt[..., -2, :], salt[..., -3, :], dyC[-3, :], dyC[-2, :]),
                )
            elif side == BoundarySide.SOUTH:
                dyU = metrics.dyU_for(dtype, device)
                dyF = metrics.dyF_for(dtype, device)
                dyC = metrics.dyC_for(dtype, device)
                points = (
                    (u[..., 0, :nx], u[..., 1, :nx], u[..., 2, :nx], dyU[2, :nx], dyU[1, :nx]),
                    (v[..., 1, :], v[..., 2, :], v[..., 3, :], dyF[2, :], dyF[1, :]),
                    (temp[..., 0, :], temp[..., 1, :], temp[..., 2, :], dyC[2, :], dyC[1, :]),
                    (salt[..., 0, :], salt[..., 1, :], salt[..., 2, :], dyC[2, :], dyC[1, :]),
                )
            boundary = torch.stack([point[0] for point in points])
            previous = None if history is None else history.orlanski.get(side.value)
            if previous is None:
                # ORLANSKI_INIT (orlanski_init.F:29-82) zeroes the storage.
                previous = boundary.new_zeros((5, *boundary.shape))
            side_values, storage[side.value] = orlanski_side_update(
                side=side.value,
                boundary=boundary,
                first=torch.stack([point[1] for point in points]),
                second=torch.stack([point[2] for point in points]),
                spacing=torch.stack([point[3] for point in points])[:, None, None, :],
                inverse_spacing=1.0 / torch.stack([point[4] for point in points])[:, None, None, :],
                storage=previous,
                params=self.orlanski,
            )
            values[side] = tuple(side_values.unbind(0))
        return values, storage

    def _side_values(
        self, orlanski_values: dict[BoundarySide, tuple[torch.Tensor, ...]],
    ) -> tuple[SideBoundaryTensorSet, ...]:
        """The u, v, temp and salt side sets: each side from its value scheme."""
        sides: dict[str, list] = {"u": [], "v": [], "temp": [], "salt": []}
        for side in BOUNDARY_SIDES:
            scheme = self.value_schemes[side]
            if scheme == OpenBoundaryValueScheme.PRESCRIBED:
                per_field = (self._active_u_values.get(side), self._active_v_values.get(side),
                             self._active_temp_values.get(side), self._active_salt_values.get(side))
            elif scheme == OpenBoundaryValueScheme.ORLANSKI:
                per_field = orlanski_values[side]
            for name, value in zip(sides, per_field):
                sides[name].append((side.value, value))
        return tuple(SideBoundaryTensorSet(**dict(entries)) for entries in sides.values())

    def _enforce_values(
        self, state: PrognosticState, grid: StructuredCGrid, *,
        orlanski_values: dict[BoundarySide, tuple[torch.Tensor, ...]],
    ) -> PrognosticState:
        bounded = self.fallback_boundary.enforce(state, grid) if self.fallback_boundary is not None else state
        eta = bounded.eta.data.clone()
        u = bounded.u.data.clone()
        v = bounded.v.data.clone()
        temp = bounded.temp.data.clone()
        salt = bounded.salt.data.clone()

        u_mask = grid.masks.u_float(dtype=u.dtype, device=u.device) if grid.masks.has_3d_masks else torch.ones_like(u)
        v_mask = grid.masks.v_float(dtype=v.dtype, device=v.device) if grid.masks.has_3d_masks else torch.ones_like(v)

        batch_size = eta.shape[0]

        # OBCS_BALANCE_FLOW once per step: compute (or reuse cached) balanced
        # normal-velocity targets BEFORE writing any face value.  Both this
        # face clamp and any downstream sponge tendency consume the same
        # ``balanced_*_values`` so they see exactly the same mass-balanced
        # OB arrays, matching MIT topology in
        # ``references/MITgcm/pkg/obcs/obcs_adjust.F:42-45`` →
        # ``obcs_balance_flow.F:347-456``.  When ``enforce_obcs_balance``
        # is False the helper returns the raw active values unchanged.
        if self.uses_orlanski:
            # OBCS_ADJUST balances the OB arrays OBCS_CALC just filled, Orlanski
            # sides included (do_oceanic_phys.F:595, obcs_adjust.F:42-45).
            u_values, v_values, temp_values, salt_values = self._side_values(orlanski_values)
            if self.enforce_obcs_balance:
                balanced_u_values, balanced_v_values = compute_obcs_balanced_normal_velocities(
                    u_values, v_values, grid, dtype=u.dtype, device=u.device)
            else:
                balanced_u_values, balanced_v_values = u_values, v_values
        else:
            balanced_u_values, balanced_v_values = self._get_balanced_targets(
                grid, dtype=u.dtype, device=u.device,
            )
            temp_values, salt_values = self._active_temp_values, self._active_salt_values

        # Gap-L: MITgcm obcs_apply_uv.F processes sides N→S→E→W.
        # At corner cells where two OB strips overlap, the last side to
        # write determines the final value.  Match MITgcm's order.
        _MITGCM_SIDE_ORDER = [
            BoundarySide.NORTH, BoundarySide.SOUTH,
            BoundarySide.EAST, BoundarySide.WEST,
        ]
        # ---- Tangential zero writes (MITgcm obcs_apply_uv.F ) ----
        # MITgcm zeros tangential velocity at OB rows/cols when no tangential
        # OB file is provided.  These cells are masked by maskInW/maskInS = 0,
        # so the write is dynamically inert.  Modes:
        #   'none'              — skip (current default, dynamically equivalent)
        #   'phase1_full'       — MITgcm-correct: pre-pass, full staggered range
        #   'phase1_truncated'  — pre-pass, truncated (:nx / :ny) for diagnosis
        if self.tangential_zero_mode.startswith('phase1'):
            _full = self.tangential_zero_mode == 'phase1_full'
            nx = grid.nx  # number of center cells in x
            ny = grid.ny  # number of center cells in y
            # North tangential: MITgcm uFld(i, Jobc=Ny, k) = 0
            # → LOCOMgcm u[..., ny-1, :] (all nx+1 u-columns if full)
            if _side_is_open(BoundarySide.NORTH.value, grid):
                if _full:
                    u[..., -1, :] = 0.0
                else:
                    u[..., -1, :nx] = 0.0
            # South tangential: MITgcm uFld(i, Jobc=1, k) = 0
            # → LOCOMgcm u[..., 0, :]
            if _side_is_open(BoundarySide.SOUTH.value, grid):
                if _full:
                    u[..., 0, :] = 0.0
                else:
                    u[..., 0, :nx] = 0.0
            # East tangential: MITgcm vFld(Iobc=Nx, j, k) = 0
            # → LOCOMgcm v[..., :, nx-1] = v[..., :, -1] (all ny+1 v-rows if full)
            if _side_is_open(BoundarySide.EAST.value, grid):
                if _full:
                    v[..., :, -1] = 0.0
                else:
                    v[..., :ny, -1] = 0.0
            # West tangential: MITgcm vFld(Iobc=1, j, k) = 0
            # → LOCOMgcm v[..., :, 0]
            if _side_is_open(BoundarySide.WEST.value, grid):
                if _full:
                    v[..., :, 0] = 0.0
                else:
                    v[..., :ny, 0] = 0.0

        # Tangential velocity injection — runs BEFORE
        # the side loop's normal/tracer writes so that NORMAL writes at
        # the corner overwrite tangential writes (matching MIT's two-phase
        # obcs_apply_uv.F: tangential phase lines 84-141, then normal phase
        # lines 143-214). When mixed inline with normal writes, the WEST
        # tangential (last in _MITGCM_SIDE_ORDER) was overwriting SOUTH
        # normal at the SW corner (j=0, i=0): v_mask=0 at the boundary
        # face zeroed out the SOUTH normal's correctly-computed -1.33 m/s.
        for tside in _MITGCM_SIDE_ORDER:
            if not _side_is_open(tside.value, grid):
                continue
            # obcs_apply_uv.F:84-141 writes every side's tangential value;
            # an Orlanski side radiates it and reads it back in the next
            # ORLANSKI_<side>, so that side writes it in any case.
            if not (self.apply_tangential_velocity
                    or self.value_schemes[tside] == OpenBoundaryValueScheme.ORLANSKI):
                continue
            if tside == BoundarySide.NORTH:
                u_ext = self._side_target_3d(balanced_u_values, tside, grid, batch_size, u.dtype, u.device)
                if u_ext is not None:
                    u[..., -1, :grid.nx] = u_ext * u_mask[..., -1, :grid.nx]
            elif tside == BoundarySide.SOUTH:
                u_ext = self._side_target_3d(balanced_u_values, tside, grid, batch_size, u.dtype, u.device)
                if u_ext is not None:
                    u[..., 0, :grid.nx] = u_ext * u_mask[..., 0, :grid.nx]
            elif tside == BoundarySide.EAST:
                v_ext = self._side_target_3d(balanced_v_values, tside, grid, batch_size, v.dtype, v.device)
                if v_ext is not None:
                    v[..., :grid.ny, -1] = v_ext * v_mask[..., :grid.ny, -1]
            elif tside == BoundarySide.WEST:
                v_ext = self._side_target_3d(balanced_v_values, tside, grid, batch_size, v.dtype, v.device)
                if v_ext is not None:
                    v[..., :grid.ny, 0] = v_ext * v_mask[..., :grid.ny, 0]

        for side in _MITGCM_SIDE_ORDER:
            if not _side_is_open(side.value, grid):
                continue
                
            eta_ext = self._side_target_2d(self._active_eta_values, side, grid, batch_size, eta.dtype, eta.device)
            # Normal velocity face writes consume the once-per-step balanced
            # targets so the prescribed face value already includes the
            # OBCS_BALANCE_FLOW correction (no separate post-write q apply).
            u_ext = self._side_target_3d(balanced_u_values, side, grid, batch_size, u.dtype, u.device)
            v_ext = self._side_target_3d(balanced_v_values, side, grid, batch_size, v.dtype, v.device)
            temp_ext = self._side_target_3d(temp_values, side, grid, batch_size, temp.dtype, temp.device)
            salt_ext = self._side_target_3d(salt_values, side, grid, batch_size, salt.dtype, salt.device)

            if side == BoundarySide.WEST:
                # MITgcm: OB_Iw=1, Iobc=1
                # NOTE: eta NOT prescribed — MITgcm OBCS_APPLY_ETA is
                # commented out; boundary η stays 0 via maskInC in the
                # pressure solver and exactConserv path.
                if u_ext is not None:
                    # Normal: uFld(Iobc+1) = OBWu * maskW(Iobc+1)   → u[..., 1]
                    #    Fac: uFld(Iobc)   = OBWu * maskW(Iobc+1)*fac → u[..., 0]
                    inner_mask = u_mask[..., 1]
                    u[..., 1] = u_ext * inner_mask
                    u[..., 0] = u_ext * self.uv_apply_fac * inner_mask
                # NOTE: WEST tangential v injection moved to phase-1 pre-loop
                # (above) to mirror MIT's two-phase obcs_apply_uv.F.
                if temp_ext is not None:
                    temp[..., :, 0] = temp_ext
                if salt_ext is not None:
                    salt[..., :, 0] = salt_ext
            elif side == BoundarySide.EAST:
                # MITgcm: OB_Ie=Nx, Iobc=Nx
                if u_ext is not None:
                    # Normal: uFld(Iobc) = OBEu * maskW(Iobc)   → u[..., -2]
                    #    Fac: uFld(Iobc+1) = OBEu * maskW(Iobc)*fac → u[..., -1]
                    inner_mask = u_mask[..., -2]
                    u[..., -2] = u_ext * inner_mask
                    u[..., -1] = u_ext * self.uv_apply_fac * inner_mask
                # NOTE: EAST tangential v injection moved to phase-1 pre-loop.
                if temp_ext is not None:
                    temp[..., :, -1] = temp_ext
                if salt_ext is not None:
                    salt[..., :, -1] = salt_ext
            elif side == BoundarySide.SOUTH:
                # MITgcm: OB_Js=1, Jobc=1
                if v_ext is not None:
                    # Normal: vFld(Jobc+1) = OBSv * maskS(Jobc+1)   → v[..., 1, :]
                    #    Fac: vFld(Jobc)   = OBSv * maskS(Jobc+1)*fac → v[..., 0, :]
                    inner_mask = v_mask[..., 1, :]
                    v[..., 1, :] = v_ext * inner_mask
                    v[..., 0, :] = v_ext * self.uv_apply_fac * inner_mask
                # NOTE: SOUTH tangential u injection moved to phase-1 pre-loop.
                if temp_ext is not None:
                    temp[..., 0, :] = temp_ext
                if salt_ext is not None:
                    salt[..., 0, :] = salt_ext
            elif side == BoundarySide.NORTH:
                # MITgcm: OB_Jn=Ny, Jobc=Ny
                if v_ext is not None:
                    # Normal: vFld(Jobc) = OBNv * maskS(Jobc)   → v[..., -2, :]
                    #    Fac: vFld(Jobc+1) = OBNv * maskS(Jobc)*fac → v[..., -1, :]
                    inner_mask = v_mask[..., -2, :]
                    v[..., -2, :] = v_ext * inner_mask
                    v[..., -1, :] = v_ext * self.uv_apply_fac * inner_mask
                # NOTE: NORTH tangential u injection moved to phase-1 pre-loop.
                if temp_ext is not None:
                    temp[..., -1, :] = temp_ext
                if salt_ext is not None:
                    salt[..., -1, :] = salt_ext
                    
        # OBCSbalance projection has been moved upstream into
        # ``_get_balanced_targets`` (called once per step at the top of this
        # method) so the prescribed face writes above already use balanced
        # normal-velocity targets.  This matches MIT topology where
        # ``OBCS_BALANCE_FLOW`` mutates the OBNxx arrays once (in
        # ``do_oceanic_phys.F:588-596 → obcs_adjust.F:42-45``) and
        # downstream consumers (``OBCS_APPLY_UV``, ``OBCS_SPONGE_U/V``)
        # share the corrected arrays — there is no second post-write
        # correction step.

        # On a wall or open axis, u[..., -1] and v[..., -1, :] are interior
        # boundary faces. Preserve the prescribed or Flather boundary values;
        # copying the opposite edge would overwrite them. Boundary-face
        # ownership follows the grid topology.

        return bounded.with_tensors(eta=eta, u=u, v=v, temp=temp, salt=salt)
