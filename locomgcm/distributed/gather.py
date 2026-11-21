"""Gather rank-local C-grid state tiles back to a global state."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from locomgcm.distributed.layout import ParallelLayout
from locomgcm.state import PrognosticState


def assemble_prognostic_state_tiles(
    tiles: list[PrognosticState] | tuple[PrognosticState, ...],
    layouts: list[ParallelLayout] | tuple[ParallelLayout, ...],
) -> PrognosticState:
    """Assemble rank-local prognostic state tiles into one global C-grid state."""

    if len(tiles) != len(layouts):
        raise ValueError("tiles and layouts must have the same length")
    if not tiles:
        raise ValueError("at least one state tile is required")

    first = tiles[0]
    first_layout = layouts[0]
    eta = _empty_global_center_like(first.eta.data, first_layout)
    temp = _empty_global_center_like(first.temp.data, first_layout)
    salt = _empty_global_center_like(first.salt.data, first_layout)
    u = _empty_global_u_like(first.u.data, first_layout)
    v = _empty_global_v_like(first.v.data, first_layout)
    filled = torch.zeros(
        (first_layout.global_ny, first_layout.global_nx),
        dtype=torch.bool,
        device=first.eta.data.device,
    )

    for tile, layout in zip(tiles, layouts, strict=True):
        _check_tile_shapes(tile, layout, first)
        y = slice(layout.y_start, layout.y_stop)
        x = slice(layout.x_start, layout.x_stop)
        y_v = slice(layout.y_start, layout.y_stop + 1)
        x_u = slice(layout.x_start, layout.x_stop + 1)
        eta[..., y, x] = tile.eta.data
        temp[..., y, x] = tile.temp.data
        salt[..., y, x] = tile.salt.data
        u[..., y, x_u] = tile.u.data
        v[..., y_v, x] = tile.v.data
        filled[y, x] = True

    if not bool(filled.all().item()):
        raise ValueError("layouts do not cover the full global center domain")
    return first.with_tensors(eta=eta, u=u, v=v, temp=temp, salt=salt)


def require_whole_job_layout(layout: ParallelLayout) -> None:
    """These gathers address rank 0 of the whole job with the layout's ranks.

    Inside a window bound to a smaller process group the non-members never
    enter the gather and the layout's rank numbers are not the group's; that
    output needs its members, root and layout adapted together, which is not
    built.  Refused before any collective.
    """
    from locomgcm.distributed.comm_adjoint import active_data_group

    group = active_data_group(None)
    if (group is not None and group is not dist.group.WORLD) or layout.world_size != dist.get_world_size():
        raise NotImplementedError(
            "rank-0 output gathers the whole job's layout; inside a window on a "
            "process subgroup it is not supported")


def gather_prognostic_state_to_rank0(
    state: PrognosticState,
    layout: ParallelLayout,
) -> PrognosticState | None:
    """Gather a rank-local state to rank 0 and assemble the global state there."""

    if not dist.is_available() or not dist.is_initialized() or layout.world_size == 1:
        return state

    require_whole_job_layout(layout)
    payload = (layout, _state_payload(state))
    gathered: list[tuple[ParallelLayout, dict[str, torch.Tensor]] | None] | None
    gathered = [None for _ in range(layout.world_size)] if layout.rank == 0 else None
    dist.gather_object(payload, object_gather_list=gathered, dst=0)
    if layout.rank != 0:
        return None
    if gathered is None or any(item is None for item in gathered):
        raise RuntimeError("state gather to rank 0 returned incomplete payloads")
    layouts: list[ParallelLayout] = []
    states: list[PrognosticState] = []
    for item in gathered:
        if item is None:
            raise RuntimeError("state gather to rank 0 returned an empty payload")
        item_layout, tensors = item
        layouts.append(item_layout)
        states.append(_state_from_payload(state, tensors))
    return assemble_prognostic_state_tiles(states, layouts)


def _state_payload(state: PrognosticState) -> dict[str, torch.Tensor]:
    return {
        "eta": state.eta.data.detach().cpu().contiguous(),
        "u": state.u.data.detach().cpu().contiguous(),
        "v": state.v.data.detach().cpu().contiguous(),
        "temp": state.temp.data.detach().cpu().contiguous(),
        "salt": state.salt.data.detach().cpu().contiguous(),
    }


def _state_from_payload(
    template: PrognosticState,
    tensors: dict[str, torch.Tensor],
) -> PrognosticState:
    return template.with_tensors(
        eta=tensors["eta"],
        u=tensors["u"],
        v=tensors["v"],
        temp=tensors["temp"],
        salt=tensors["salt"],
    )


def _empty_global_center_like(tensor: torch.Tensor, layout: ParallelLayout) -> torch.Tensor:
    return tensor.new_empty((*tensor.shape[:-2], layout.global_ny, layout.global_nx))


def _empty_global_u_like(tensor: torch.Tensor, layout: ParallelLayout) -> torch.Tensor:
    return tensor.new_empty((*tensor.shape[:-2], layout.global_ny, layout.global_nx + 1))


def _empty_global_v_like(tensor: torch.Tensor, layout: ParallelLayout) -> torch.Tensor:
    return tensor.new_empty((*tensor.shape[:-2], layout.global_ny + 1, layout.global_nx))


def _check_tile_shapes(
    tile: PrognosticState,
    layout: ParallelLayout,
    first: PrognosticState,
) -> None:
    prefix_center = first.eta.data.shape[:-2]
    prefix_scalar = first.temp.data.shape[:-2]
    prefix_uv = first.u.data.shape[:-2]
    expected: dict[str, tuple[int, ...]] = {
        "eta": (*prefix_center, layout.ny, layout.nx),
        "temp": (*prefix_scalar, layout.ny, layout.nx),
        "salt": (*prefix_scalar, layout.ny, layout.nx),
        "u": (*prefix_uv, layout.ny, layout.nx + 1),
        "v": (*prefix_uv, layout.ny + 1, layout.nx),
    }
    actual: dict[str, Any] = {
        "eta": tuple(tile.eta.data.shape),
        "temp": tuple(tile.temp.data.shape),
        "salt": tuple(tile.salt.data.shape),
        "u": tuple(tile.u.data.shape),
        "v": tuple(tile.v.data.shape),
    }
    for name, shape in actual.items():
        if shape != expected[name]:
            raise ValueError(
                f"rank {layout.rank} {name} shape {shape} != {expected[name]}"
            )
