from __future__ import annotations

import torch

from defgcm.boundaries.open import (
    BoundarySide,
    SideBoundaryTensorSet,
    boundary_cell_values,
    boundary_band_target,
    inflow_mask,
    side_normal_velocity,
)
from defgcm.geometry import StructuredCGrid
from defgcm.state import PrognosticState


def edge_mismatch_metrics(field: torch.Tensor) -> dict[str, float]:
    if field.ndim < 2:
        raise ValueError("field must have at least two spatial dimensions")

    values = field
    left = values[..., :, 0]
    right = values[..., :, -1]
    south = values[..., 0, :]
    north = values[..., -1, :]

    if values.shape[-1] > 1:
        west_inner = values[..., :, 0] - values[..., :, 1]
        east_inner = values[..., :, -1] - values[..., :, -2]
    else:
        west_inner = torch.zeros_like(left)
        east_inner = torch.zeros_like(right)

    if values.shape[-2] > 1:
        south_inner = values[..., 0, :] - values[..., 1, :]
        north_inner = values[..., -1, :] - values[..., -2, :]
    else:
        south_inner = torch.zeros_like(south)
        north_inner = torch.zeros_like(north)

    def _abs_mean(tensor: torch.Tensor) -> float:
        return float(tensor.abs().mean().item())

    def _abs_max(tensor: torch.Tensor) -> float:
        return float(tensor.abs().max().item())

    return {
        "left_right_abs_mean": _abs_mean(left - right),
        "left_right_abs_max": _abs_max(left - right),
        "south_north_abs_mean": _abs_mean(south - north),
        "south_north_abs_max": _abs_max(south - north),
        "west_inner_abs_mean": _abs_mean(west_inner),
        "west_inner_abs_max": _abs_max(west_inner),
        "east_inner_abs_mean": _abs_mean(east_inner),
        "east_inner_abs_max": _abs_max(east_inner),
        "south_inner_abs_mean": _abs_mean(south_inner),
        "south_inner_abs_max": _abs_max(south_inner),
        "north_inner_abs_mean": _abs_mean(north_inner),
        "north_inner_abs_max": _abs_max(north_inner),
    }


def boundary_band_rmse(
    field: torch.Tensor,
    boundary_values: SideBoundaryTensorSet,
    grid: StructuredCGrid,
    *,
    width: int,
) -> dict[str, float]:
    if width < 1:
        raise ValueError("width must be positive")

    metrics: dict[str, float] = {}
    for side in BoundarySide:
        values = boundary_values.get(side)
        if values is None:
            continue
        side_width = min(width, grid.nx if side in (BoundarySide.WEST, BoundarySide.EAST) else grid.ny)
        target = boundary_band_target(
            values,
            side,
            grid,
            batch_size=field.shape[0],
            dtype=field.dtype,
            device=field.device,
            width=side_width,
        )
        if side == BoundarySide.WEST:
            band = field[..., :side_width]
        elif side == BoundarySide.EAST:
            band = field[..., -side_width:]
        elif side == BoundarySide.SOUTH:
            band = field[..., :side_width, :]
        else:
            band = field[..., -side_width:, :]
        metrics[f"{side.value}_rmse"] = float(torch.sqrt((band - target).square().mean()).item())
    return metrics


def _side_face_area(state: PrognosticState, grid: StructuredCGrid, side: BoundarySide) -> torch.Tensor:
    """Edge length times layer thickness of the faces ``side_normal_velocity`` reads.

    The edge is the grid's ``dyG`` (u faces) or ``dxG`` (v faces) at that face
    column or row, so non-uniform spacing is taken face by face.
    """
    dtype, device = state.u.data.dtype, state.u.data.device
    thickness = grid.layer_thickness.to(dtype=dtype, device=device).view(1, grid.nz, 1)
    if side == BoundarySide.WEST:
        edge = grid.metrics.dyG_for(dtype, device)[:, 0]
    elif side == BoundarySide.EAST:
        edge = grid.metrics.dyG_for(dtype, device)[:, state.u.data.shape[-1] - 1]
    elif side == BoundarySide.SOUTH:
        edge = grid.metrics.dxG_for(dtype, device)[0, :]
    else:
        edge = grid.metrics.dxG_for(dtype, device)[state.v.data.shape[-2] - 1, :]
    return edge.view(1, 1, -1) * thickness


def boundary_flow_metrics(state: PrognosticState, grid: StructuredCGrid) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for side in BoundarySide:
        normal_velocity = side_normal_velocity(state, side)
        inflow = inflow_mask(state, side)
        face_area = _side_face_area(state, grid, side)
        signed_transport = normal_velocity * face_area
        metrics[f"{side.value}_net_normal_transport"] = float(signed_transport.sum().item())
        metrics[f"{side.value}_inflow_fraction"] = float(inflow.to(dtype=normal_velocity.dtype).mean().item())
        metrics[f"{side.value}_outflow_fraction"] = float((~inflow).to(dtype=normal_velocity.dtype).mean().item())
    return metrics


def outward_normal_velocity(state: PrognosticState, side: BoundarySide) -> torch.Tensor:
    normal_velocity = side_normal_velocity(state, side)
    if side in (BoundarySide.WEST, BoundarySide.SOUTH):
        return -normal_velocity
    return normal_velocity


def boundary_budget_metrics(state: PrognosticState, grid: StructuredCGrid) -> dict[str, object]:
    transport_net_by_side: dict[str, float] = {}
    transport_in_by_side: dict[str, float] = {}
    transport_out_by_side: dict[str, float] = {}
    inflow_fraction_by_side: dict[str, float] = {}
    outflow_fraction_by_side: dict[str, float] = {}
    temp_flux_by_side: dict[str, float] = {}
    salt_flux_by_side: dict[str, float] = {}

    for side in BoundarySide:
        normal_velocity = outward_normal_velocity(state, side)
        face_area = _side_face_area(state, grid, side)
        transport = normal_velocity * face_area
        inflow = transport < 0.0
        temp_boundary = boundary_cell_values(state.temp.data, side)
        salt_boundary = boundary_cell_values(state.salt.data, side)
        temp_flux = transport * temp_boundary
        salt_flux = transport * salt_boundary

        transport_net_by_side[side.value] = float(transport.sum().item())
        transport_in_by_side[side.value] = float((-transport.clamp_max(0.0)).sum().item())
        transport_out_by_side[side.value] = float(transport.clamp_min(0.0).sum().item())
        inflow_fraction_by_side[side.value] = float(inflow.to(dtype=normal_velocity.dtype).mean().item())
        outflow_fraction_by_side[side.value] = float((~inflow).to(dtype=normal_velocity.dtype).mean().item())
        temp_flux_by_side[side.value] = float(temp_flux.sum().item())
        salt_flux_by_side[side.value] = float(salt_flux.sum().item())

    return {
        "transport_net_by_side": transport_net_by_side,
        "transport_in_by_side": transport_in_by_side,
        "transport_out_by_side": transport_out_by_side,
        "inflow_fraction_by_side": inflow_fraction_by_side,
        "outflow_fraction_by_side": outflow_fraction_by_side,
        "temp_flux_by_side": temp_flux_by_side,
        "salt_flux_by_side": salt_flux_by_side,
        "q_net": float(sum(transport_net_by_side.values())),
        "q_in": float(sum(transport_in_by_side.values())),
        "q_out": float(sum(transport_out_by_side.values())),
        "temp_flux_net": float(sum(temp_flux_by_side.values())),
        "salt_flux_net": float(sum(salt_flux_by_side.values())),
    }
