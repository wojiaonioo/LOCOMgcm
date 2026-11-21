"""Issue a whole step's halo requests as one packed round per dependency stage.

A step of the model issues tens of independent :func:`~locomgcm.geometry.exchange.exchange`
calls, each of which costs at least one point-to-point round -- and, when one
peer serves two semantic halo sides, several (see ``_distributed_p2p_rounds``).
The rounds are serialised by the order token, so the step pays a latency for
each one even though almost none of them depend on each other.

An *epoch* is a set of requests declared together.  Instead of walking them one
at a time, the epoch collects every strip bound for a given peer -- across all
requests in the epoch -- into a single message, so the whole set costs one round
per peer per direction.  The bytes on the wire are the same bytes the sequential
path would have sent, concatenated; the received strips are therefore identical
cell for cell with a plain ``exchange()`` loop.

Stages
------
``exchange()`` pads x first and then y, and the y strips are cut from the
*x-padded* tensor: the corner cells of a two-axis halo reach a rank through its
y-neighbour, carrying the x-ghost values that neighbour had already received.
That is a real data dependency, not an ordering convention -- the corner of an
OBC-filled x edge is a function of the y-neighbour's own boundary values, which
this rank does not hold -- so it cannot be folded into the same message.

An epoch therefore has two stages:

``stage 0``
    every request's x work, plus the y work of requests with no x halo;
``stage 1``
    the y work of requests that do have an x halo.

Each stage costs at most one packed round, so an epoch whose requests are all
single-axis (the majority of a step's traffic) costs exactly one round and
advances the order token exactly once.  ``EpochPlan.stage_count`` is the number
of dependency stages, fixed when the plan is built; it is an upper bound on the
rounds, because a stage in which no rank has a distributed peer -- a y stage on
a 1-column decomposition, say -- issues nothing at all, exactly as
``_pad_axis_distributed`` issues nothing when its spec lists come out empty.
That skip is uniform for the same reason the existing path's is: a Cartesian
neighbour table is symmetric, so either every rank has a peer on an axis or none
does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from locomgcm.geometry.exchange import (
    CGridLocation,
    FillSpec,
    HaloExtentV1,
    ObcEdges,
    _assemble_padded_axis,
    _distributed_halo_specs,
    _distributed_neighbour_rank,
    _halo_slot_recvs,
    _halo_slot_sends,
    _has_distributed_transport,
    _pad_axis,
    _packed_halo_round,
    _store_received_halos,
)
from locomgcm.geometry.grid import BoundaryType, Grid


_X_AXIS_ROLE = 0
_Y_AXIS_ROLE = 1


@dataclass(frozen=True)
class _EpochTopology:
    """Grid facts an epoch resolves once, outside anything that gets traced.

    ``_distributed_neighbour_rank`` and ``_has_distributed_transport`` reach
    into the grid's neighbour mapping with ``hasattr``/``getattr``, which torch
    2.4's Dynamo declines to trace on a dict.  Reading them here, before
    :func:`issue` touches a tensor, keeps the traced body free of Python
    container inspection -- the peers arrive as plain ints.  The values are a
    property of the decomposition, not of the data, so hoisting them changes
    nothing about what is computed.
    """

    x_peers: tuple[int | None, int | None]
    y_peers: tuple[int | None, int | None]
    distributed_x: bool
    distributed_y: bool


def _epoch_topology(grid: Grid) -> _EpochTopology:
    return _EpochTopology(
        x_peers=(
            _distributed_neighbour_rank(grid, "west"),
            _distributed_neighbour_rank(grid, "east"),
        ),
        y_peers=(
            _distributed_neighbour_rank(grid, "south"),
            _distributed_neighbour_rank(grid, "north"),
        ),
        distributed_x=_has_distributed_transport(grid, ("west", "east")),
        distributed_y=_has_distributed_transport(grid, ("south", "north")),
    )


@dataclass(frozen=True)
class EpochRequest:
    """One ``exchange()`` call, declared before the data exists.

    ``shape`` and ``dtype`` are part of the declaration rather than read off the
    tensor at issue time: the packed layout is a function of them, and pinning
    them here is what makes the layout a build-time constant that every rank
    computes identically.  ``issue`` rejects a tensor that disagrees.

    ``obc_key`` selects this request's open-boundary values out of the mapping
    passed to :func:`issue`; ``None`` means the request takes no ``obc`` (the
    ``OBC_VALUE`` fill then falls back to REPLICATE exactly as it does in
    ``exchange()``).
    """

    location: CGridLocation
    extent: HaloExtentV1
    fill: FillSpec
    dtype: torch.dtype
    shape: tuple[int, ...]
    obc_key: str | None = None

    def __post_init__(self) -> None:
        if len(self.shape) < 2:
            raise ValueError(
                "EpochRequest.shape needs horizontal [..., j, i] dimensions; "
                "1-D metric rows are not part of an epoch"
            )

    @property
    def pads_x(self) -> bool:
        return bool(self.extent.west_left_extent or self.extent.east_right_extent)

    @property
    def pads_y(self) -> bool:
        return bool(self.extent.south_bottom_extent or self.extent.north_top_extent)


@dataclass(frozen=True)
class EpochPlan:
    """A frozen set of requests issued together, built once per configuration."""

    requests: tuple[EpochRequest, ...]

    def __init__(self, requests: Sequence[EpochRequest]) -> None:
        frozen = tuple(requests)
        if not frozen:
            raise ValueError("an epoch needs at least one request")
        for request in frozen:
            if not isinstance(request, EpochRequest):
                raise TypeError("EpochPlan takes EpochRequest entries")
        object.__setattr__(self, "requests", frozen)

    @property
    def stage_count(self) -> int:
        """Dependency stages: 2 once any request pads both axes, else 1.

        An upper bound on the packed rounds :func:`issue` will run -- a stage
        with no distributed peers on this grid costs none (see the module
        docstring).
        """
        return 2 if any(r.pads_x and r.pads_y for r in self.requests) else 1

    def __len__(self) -> int:
        return len(self.requests)


def issue(
    plan: EpochPlan,
    tensors: Sequence[torch.Tensor],
    *,
    grid: Grid,
    obc: Mapping[str, ObcEdges] | None = None,
    topology: _EpochTopology | None = None,
) -> list[torch.Tensor]:
    """Pad every request in ``plan``, one packed round per stage.

    Returns one padded tensor per request, in plan order, each bit-identical to
    the ``exchange()`` call the request declares.

    ``topology`` may be pre-resolved with :func:`_epoch_topology` when ``issue``
    is called from inside a compiled region; see :class:`_EpochTopology`.
    """
    if len(tensors) != len(plan.requests):
        raise ValueError(
            f"epoch has {len(plan.requests)} requests but got {len(tensors)} tensors"
        )
    boundary = grid.boundary_topology
    if BoundaryType.SEAM in (boundary.west_east, boundary.south_north):
        raise NotImplementedError(
            "cubed-sphere (SEAM) transport has its own cross-face backend; "
            "an epoch covers the per-axis regional flow only"
        )
    for index, (request, tensor) in enumerate(zip(plan.requests, tensors)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"epoch tensor {index} is not a torch.Tensor")
        if tuple(tensor.shape) != request.shape or tensor.dtype != request.dtype:
            raise ValueError(
                f"epoch tensor {index} is {tuple(tensor.shape)}/{tensor.dtype}, "
                f"plan declares {request.shape}/{request.dtype}"
            )

    if topology is None:
        topology = _epoch_topology(grid)
    distributed_x = topology.distributed_x
    distributed_y = topology.distributed_y

    stage_zero: list[tuple[int, int, object]] = []
    stage_one_pending: list[int] = []
    outputs: list[torch.Tensor | None] = [None] * len(plan.requests)
    working: list[torch.Tensor] = list(tensors)

    for index, request in enumerate(plan.requests):
        edges = _request_obc(request, obc)
        if request.pads_x:
            if distributed_x:
                stage_zero.append(
                    (
                        index,
                        _X_AXIS_ROLE,
                        _x_specs(working[index], request, grid, edges, topology),
                    )
                )
            else:
                working[index] = _x_local(working[index], request, grid, edges)
        elif request.pads_y:
            if distributed_y:
                stage_zero.append(
                    (
                        index,
                        _Y_AXIS_ROLE,
                        _y_specs(working[index], request, grid, edges, topology),
                    )
                )
            else:
                working[index] = _y_local(working[index], request, grid, edges)
        else:
            outputs[index] = working[index].clone()

    _run_stage(stage_zero)
    for index, role, specs in stage_zero:
        request = plan.requests[index]
        if role == _X_AXIS_ROLE:
            working[index] = _assemble_padded_axis(
                working[index],
                specs=specs,
                before=request.extent.west_left_extent,
                after=request.extent.east_right_extent,
            )
        else:
            working[index] = _assemble_padded_axis(
                working[index],
                specs=specs,
                before=request.extent.south_bottom_extent,
                after=request.extent.north_top_extent,
            )

    for index, request in enumerate(plan.requests):
        if outputs[index] is not None:
            continue
        if request.pads_x and request.pads_y:
            stage_one_pending.append(index)
        else:
            outputs[index] = working[index]

    stage_one: list[tuple[int, int, object]] = []
    for index in stage_one_pending:
        request = plan.requests[index]
        edges = _request_obc(request, obc)
        if distributed_y:
            stage_one.append(
                (
                    index,
                    _Y_AXIS_ROLE,
                    _y_specs(working[index], request, grid, edges, topology),
                )
            )
        else:
            outputs[index] = _y_local(working[index], request, grid, edges)

    _run_stage(stage_one)
    for index, _role, specs in stage_one:
        request = plan.requests[index]
        outputs[index] = _assemble_padded_axis(
            working[index],
            specs=specs,
            before=request.extent.south_bottom_extent,
            after=request.extent.north_top_extent,
        )

    return [tensor for tensor in outputs if tensor is not None]


def _run_stage(stage: list[tuple[int, int, object]]) -> None:
    """One packed round for a stage: at most one message per peer per direction."""
    sends: list[tuple[int, tuple[int, ...], torch.Tensor]] = []
    recvs: list[tuple[int, tuple[int, ...], torch.Tensor]] = []
    for index, role, specs in stage:
        sends.extend(_halo_slot_sends(specs, index, role))
        recvs.extend(_halo_slot_recvs(specs, index, role))
    if not sends and not recvs:
        return
    received = _packed_halo_round(sends, recvs)
    offset = 0
    for _index, _role, specs in stage:
        count = len(specs.recv_specs)
        _store_received_halos(specs, received[offset : offset + count])
        offset += count


def _request_obc(
    request: EpochRequest, obc: Mapping[str, ObcEdges] | None
) -> ObcEdges | None:
    if request.obc_key is None:
        return None
    if obc is None or request.obc_key not in obc:
        raise KeyError(
            f"epoch request declares obc_key={request.obc_key!r} but issue() "
            "was not given those edge values"
        )
    return obc[request.obc_key]


def _x_specs(
    tensor,
    request: EpochRequest,
    grid: Grid,
    edges: ObcEdges | None,
    topology: _EpochTopology,
):
    return _distributed_halo_specs(
        tensor,
        grid=grid,
        location=request.location,
        axis=-1,
        before=request.extent.west_left_extent,
        after=request.extent.east_right_extent,
        before_side="west",
        after_side="east",
        boundary_type=grid.boundary_topology.west_east,
        fill=request.fill,
        obc=edges,
        peers=topology.x_peers,
    )


def _y_specs(
    tensor,
    request: EpochRequest,
    grid: Grid,
    edges: ObcEdges | None,
    topology: _EpochTopology,
):
    return _distributed_halo_specs(
        tensor,
        grid=grid,
        location=request.location,
        axis=-2,
        before=request.extent.south_bottom_extent,
        after=request.extent.north_top_extent,
        before_side="south",
        after_side="north",
        boundary_type=grid.boundary_topology.south_north,
        fill=request.fill,
        obc=edges,
        x_extent=(request.extent.west_left_extent, request.extent.east_right_extent),
        x_boundary_type=grid.boundary_topology.west_east,
        peers=topology.y_peers,
    )


def _x_local(tensor, request: EpochRequest, grid: Grid, edges: ObcEdges | None):
    return _pad_axis(
        tensor,
        grid=grid,
        location=request.location,
        axis=-1,
        before=request.extent.west_left_extent,
        after=request.extent.east_right_extent,
        before_side="west",
        after_side="east",
        boundary_type=grid.boundary_topology.west_east,
        fill=request.fill,
        obc=edges,
    )


def _y_local(tensor, request: EpochRequest, grid: Grid, edges: ObcEdges | None):
    return _pad_axis(
        tensor,
        grid=grid,
        location=request.location,
        axis=-2,
        before=request.extent.south_bottom_extent,
        after=request.extent.north_top_extent,
        before_side="south",
        after_side="north",
        boundary_type=grid.boundary_topology.south_north,
        fill=request.fill,
        obc=edges,
        x_extent=(request.extent.west_left_extent, request.extent.east_right_extent),
        x_boundary_type=grid.boundary_topology.west_east,
    )


__all__ = ["EpochPlan", "EpochRequest", "issue"]
