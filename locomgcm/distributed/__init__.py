"""Distributed execution helpers for horizontal domain decomposition."""
from .comm_adjoint import CommSession, CommSessionError, RoundPlan, halo_transfer, ordered_sum, replicated_loss_weight

from .comm_adjoint import (CommSession, CommSessionError, RoundPlan,
                           halo_transfer, ordered_sum, replicated_loss_weight)
from .bootstrap import DistributedContext, LaunchEnvironment, detect_launch_environment, init_distributed
from .gather import assemble_prognostic_state_tiles, gather_prognostic_state_to_rank0
from .halo_p2p import HALO_P2P_TAGS, halo_p2p
from .layout import (
    CartesianNeighbor,
    ParallelLayout,
    all_layouts,
    assemble_global_from_tiles,
    axis_chunks,
    build_cartesian_neighbors,
    layout_from_rank,
    minimum_tile_extents,
    require_minimum_tile_extent,
    split_owned_region,
)
from .reductions import HaloTrim, global_dot, global_sum, owned_interior

_LOCALIZE_EXPORTS = frozenset(
    {
        "localize_boundary_bundle_source",
        "localize_bulk_source",
        "localize_ic_tensors",
        "localize_prognostic_state",
        "localize_surface_flux_source",
        "localize_structured_grid",
        "localize_tidal_config",
        "localize_wind_source",
    }
)


def __getattr__(name: str):
    if name not in _LOCALIZE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import localize

    value = getattr(localize, name)
    globals()[name] = value
    return value

__all__ = [
    "CommSession", "CommSessionError", "RoundPlan",
    "halo_transfer", "ordered_sum", "replicated_loss_weight",
    "CartesianNeighbor",
    "DistributedContext",
    "HaloTrim",
    "HALO_P2P_TAGS",
    "LaunchEnvironment",
    "ParallelLayout",
    "all_layouts",
    "assemble_prognostic_state_tiles",
    "assemble_global_from_tiles",
    "axis_chunks",
    "build_cartesian_neighbors",
    "detect_launch_environment",
    "global_dot",
    "global_sum",
    "halo_p2p",
    "gather_prognostic_state_to_rank0",
    "init_distributed",
    "layout_from_rank",
    "minimum_tile_extents",
    "require_minimum_tile_extent",
    "localize_boundary_bundle_source",
    "localize_bulk_source",
    "localize_ic_tensors",
    "localize_prognostic_state",
    "localize_surface_flux_source",
    "localize_structured_grid",
    "localize_tidal_config",
    "localize_wind_source",
    "owned_interior",
    "split_owned_region",
]
