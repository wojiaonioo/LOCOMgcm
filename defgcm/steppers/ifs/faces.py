"""Regional shared-face ownership and open-strip seam synchronization.

U and V occupy [batch, nz, ny, nx+1] and [batch, nz, ny+1, nx], respectively,
in m/s. Synchronization copies owner values without arithmetic or dtype
conversion; the regional grid operator decides sharing from grid topology.
"""
from __future__ import annotations

import torch

from defgcm.distributed.comm_adjoint import CommSession
from defgcm.distributed.layout import ParallelLayout
from defgcm.geometry import StructuredCGrid
from defgcm.geometry.grid import BoundaryTopology
from defgcm.operators.c_grid import synchronize_regional_face_values


def open_boundary_crosses_rank_seam(
    *,
    face_count: int,
    parallel_layout: ParallelLayout | None,
    boundary_topology: BoundaryTopology,
) -> bool:
    """Whether global open strips cross a regional rank seam (rank-uniform)."""
    if face_count != 1:
        return False
    layout = parallel_layout
    if layout is None:
        return False
    # open_boundary() is rank-local: interior tiles must not skip the exchange.
    topology = boundary_topology
    return ((topology.open_x and layout.nPy > 1)
            or (topology.open_y and layout.nPx > 1))


def sync_distributed_shared_velocity_faces(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
    *,
    session: CommSession | None = None,
    after_boundary: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give every duplicated regional C-grid seam face its owner's value.

    A tile owns u faces ``0 .. nx-1`` and v rows ``0 .. ny-1``; its last u face
    (v row) is face 0 of the tile to the east (north).  Whether that face is
    shared is decided by the topology, per numerics rule 11: the axis couples
    its edge faces (``grid.edge_face_couples``, a periodic axis) or a
    distributed neighbour sits on that side.  The owner is that neighbour, or
    this tile when the periodic axis is not split (a single process included);
    then the copy is local.  MITgcm fills the same face as the east (north) halo
    from the neighbour's first face by EXCH
    (``eesupp/src/exch_rx_recv_get_x.template:265-266``).

    ``after_boundary`` marks an immediate follow-up to model boundary
    enforcement: crossing open strips have already synchronized inside that
    operation, so no second exchange is issued. Other domains retain their
    existing sync and observation points. The static topology and session
    binding fold at trace time. Cube seams use the rotating cube exchange.
    """
    if after_boundary and open_boundary_crosses_rank_seam(
        face_count=getattr(grid, "face_count", 1),
        parallel_layout=getattr(grid, "parallel_layout", None),
        boundary_topology=grid.boundary_topology,
    ):
        return u, v
    return synchronize_regional_face_values(u, v, grid, session=session)
