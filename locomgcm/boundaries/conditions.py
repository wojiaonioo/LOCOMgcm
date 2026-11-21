from __future__ import annotations

from locomgcm.sampling import SampleClock, SamplingRule

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

from locomgcm.distributed.comm_adjoint import session_for_grid, transfer
from locomgcm.geometry import BoundaryType, StructuredCGrid
from locomgcm.state import PrognosticState

from .base import BoundaryCondition
from locomgcm.boundaries._registry import register_boundary_condition


def _copy_lower_face_to_upper(
    lower_face: torch.Tensor,
    upper_face: torch.Tensor,
) -> None:
    """Periodic axis: the upper edge face is face 0 stored a second time.

    The tile owns faces ``0 .. n-1``; face ``n`` takes face 0's value, the way
    MITgcm fills the east (north) halo face from face 1 by EXCH
    (``eesupp/src/exch_rx_recv_get_x.template:265-266``).  Face 0 keeps its
    own value, so a stale or differently computed face ``n`` never leaks into
    it (an average did, e.g. the sponge target that face ``n`` replicates from
    face ``n-1``).
    """
    upper_face.copy_(lower_face)


def _zero_boundary_faces(
    lower_face: torch.Tensor,
    upper_face: torch.Tensor,
) -> None:
    lower_face.zero_()
    upper_face.zero_()


def _leave_boundary_faces(
    lower_face: torch.Tensor,
    upper_face: torch.Tensor,
) -> None:
    del lower_face, upper_face


_BoundaryFacePolicy = Callable[[torch.Tensor, torch.Tensor], None]
_BOUNDARY_FACE_POLICIES: dict[BoundaryType, _BoundaryFacePolicy] = {
    BoundaryType.PERIODIC: _copy_lower_face_to_upper,
    BoundaryType.WALL: _zero_boundary_faces,
    BoundaryType.OPEN: _leave_boundary_faces,
}


def _apply_boundary_face_policy(
    boundary_type: BoundaryType,
    lower_face: torch.Tensor,
    upper_face: torch.Tensor,
) -> None:
    try:
        policy = _BOUNDARY_FACE_POLICIES[boundary_type]
    except KeyError as exc:
        raise NotImplementedError(f"Unsupported boundary type: {boundary_type!r}") from exc
    policy(lower_face, upper_face)


def _apply_distributed_axis_boundary_policy(
    boundary_type: BoundaryType,
    lower_face: torch.Tensor,
    upper_face: torch.Tensor,
    *,
    grid: StructuredCGrid,
    axis: str,
) -> None:
    layout = getattr(grid, "parallel_layout", None)
    parts = int(getattr(layout, "nPx" if axis == "x" else "nPy", 1)) if layout is not None else 1
    if parts <= 1:
        _apply_boundary_face_policy(boundary_type, lower_face, upper_face)
        return

    if boundary_type == BoundaryType.PERIODIC:
        _copy_distributed_periodic_global_face(
            lower_face,
            upper_face,
            grid=grid,
            axis=axis,
        )
        return

    if boundary_type == BoundaryType.WALL:
        if _rank_owns_global_lower_side(grid, axis):
            lower_face.zero_()
        if _rank_owns_global_upper_side(grid, axis):
            upper_face.zero_()
        return

    if boundary_type == BoundaryType.OPEN:
        return

    raise NotImplementedError(f"Unsupported boundary type: {boundary_type!r}")


def _copy_distributed_periodic_global_face(
    lower_face: torch.Tensor,
    upper_face: torch.Tensor,
    *,
    grid: StructuredCGrid,
    axis: str,
) -> None:
    """The split periodic axis: the upper owner takes the lower owner's face 0.

    Same policy as :func:`_copy_lower_face_to_upper` across the two ranks that
    hold the global edges.  The round stays the symmetric exchange both edge
    ranks issued before, so the step's communication is unchanged; the lower
    owner does not use the face it receives.
    """
    layout = getattr(grid, "parallel_layout", None)
    if layout is None or not dist.is_available() or not dist.is_initialized():
        return

    lower_owner, upper_owner = _global_axis_edge_ranks(grid, axis)
    rank = int(getattr(layout, "rank", dist.get_rank()))
    if rank not in (lower_owner, upper_owner):
        return

    local_face = lower_face if rank == lower_owner else upper_face
    peer = upper_owner if rank == lower_owner else lower_owner
    if peer == rank:
        _copy_lower_face_to_upper(lower_face, upper_face)
        return

    send = local_face.contiguous()
    recv = transfer([send], [send], [peer], [peer], [0], [0],
                    session=session_for_grid(grid))[0]
    if rank == upper_owner:
        upper_face.copy_(recv)


def _rank_owns_global_lower_side(grid: StructuredCGrid, axis: str) -> bool:
    layout = getattr(grid, "parallel_layout", None)
    if layout is None:
        return True
    tile = int(getattr(layout, "tile_i" if axis == "x" else "tile_j", 0))
    return tile == 0


def _rank_owns_global_upper_side(grid: StructuredCGrid, axis: str) -> bool:
    layout = getattr(grid, "parallel_layout", None)
    if layout is None:
        return True
    parts = int(getattr(layout, "nPx" if axis == "x" else "nPy", 1))
    tile = int(getattr(layout, "tile_i" if axis == "x" else "tile_j", 0))
    return tile == parts - 1


def _global_axis_edge_ranks(grid: StructuredCGrid, axis: str) -> tuple[int, int]:
    layout = getattr(grid, "parallel_layout", None)
    if layout is None:
        return 0, 0
    nPx = int(getattr(layout, "nPx", 1))
    nPy = int(getattr(layout, "nPy", 1))
    tile_i = int(getattr(layout, "tile_i", 0))
    tile_j = int(getattr(layout, "tile_j", 0))
    if axis == "x":
        return tile_j * nPx, tile_j * nPx + (nPx - 1)
    return tile_i, (nPy - 1) * nPx + tile_i


@dataclass(frozen=True)
@register_boundary_condition("passthrough")
class PassthroughBoundaryCondition(BoundaryCondition):

    sampling_rule = SamplingRule(SampleClock.INDEPENDENT)
    name: str = "passthrough_boundary"
    # Topology capability (locomgcm/grid_support.py).
    # The cube path's boundary condition (closed global domain, no-op enforce;
    # build_cube_model uses it) — the Structured SEAM policy would raise.
    grid_support = ("latlon", "cube")

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "PassthroughBoundaryCondition":
        del device, dtype
        return self

    def enforce(self, state: PrognosticState, grid: StructuredCGrid) -> PrognosticState:
        del grid
        return state


@dataclass(frozen=True)
@register_boundary_condition("structured")
class StructuredBoundaryCondition(BoundaryCondition):

    sampling_rule = SamplingRule(SampleClock.INDEPENDENT)
    name: str = "structured_boundary"

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "StructuredBoundaryCondition":
        del device, dtype
        return self

    def enforce(self, state: PrognosticState, grid: StructuredCGrid) -> PrognosticState:
        u = state.u.data.clone()
        v = state.v.data.clone()

        _apply_distributed_axis_boundary_policy(
            grid.boundary_topology.west_east,
            u[..., 0],
            u[..., -1],
            grid=grid,
            axis="x",
        )
        _apply_distributed_axis_boundary_policy(
            grid.boundary_topology.south_north,
            v[..., 0, :],
            v[..., -1, :],
            grid=grid,
            axis="y",
        )

        return state.with_tensors(u=u, v=v)
