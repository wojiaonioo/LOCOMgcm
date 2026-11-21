"""Publish a bounded regional state and its synchronized C-grid velocities.

State tensors keep their input units and dtype. Boundary application precedes
U/V shared-face synchronization; the returned boundary state and synchronized
velocity pair remain distinct values for tracer and continuity consumers.
"""
from __future__ import annotations

from locomgcm.state import PrognosticState

from .faces import sync_distributed_shared_velocity_faces
from .trace import _dist_trace


def apply_boundary_and_sync(
    state: PrognosticState, grid, *, apply_boundary,
    boundary_begin, boundary_done, sync_begin, sync_done,
) -> tuple:
    _dist_trace(boundary_begin)
    bounded = apply_boundary(state, record_budget=False)
    _dist_trace(boundary_done)
    u, v = bounded.u.data, bounded.v.data
    _dist_trace(sync_begin)
    u, v = sync_distributed_shared_velocity_faces(u, v, grid, after_boundary=True)
    _dist_trace(sync_done)
    return (bounded, u, v)
