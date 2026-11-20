"""2-D horizontal domain decomposition helpers.

LOCOMgcm keeps vertical columns local.  These helpers split and reassemble
only the last two tensor dimensions ``[..., j, i]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch


Side = Literal[
    "west",
    "east",
    "south",
    "north",
    "southwest",
    "southeast",
    "northwest",
    "northeast",
]


@dataclass(frozen=True)
class CartesianNeighbor:
    """Neighbour rank metadata for a Cartesian tile adjacency."""

    rank: int
    tile_i: int
    tile_j: int
    offset_i: int
    offset_j: int


@dataclass(frozen=True)
class ParallelLayout:
    """Owned horizontal tile for one distributed rank."""

    nPx: int
    nPy: int
    rank: int
    local_rank: int
    world_size: int
    global_nx: int
    global_ny: int
    tile_i: int
    tile_j: int
    x_start: int
    x_stop: int
    y_start: int
    y_stop: int

    @property
    def nx(self) -> int:
        return self.x_stop - self.x_start

    @property
    def ny(self) -> int:
        return self.y_stop - self.y_start

    @property
    def tile_offset_i(self) -> int:
        return self.x_start

    @property
    def tile_offset_j(self) -> int:
        return self.y_start

    @property
    def owned_slices(self) -> tuple[slice, slice]:
        return (slice(self.y_start, self.y_stop), slice(self.x_start, self.x_stop))


def axis_chunks(length: int, parts: int) -> tuple[tuple[int, int], ...]:
    """Return balanced half-open chunks covering ``range(length)``."""

    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    if isinstance(parts, bool) or not isinstance(parts, int) or parts <= 0:
        raise ValueError("parts must be a positive integer")
    base, remainder = divmod(length, parts)
    chunks: list[tuple[int, int]] = []
    start = 0
    for part in range(parts):
        width = base + (1 if part < remainder else 0)
        stop = start + width
        chunks.append((start, stop))
        start = stop
    return tuple(chunks)


def minimum_tile_extents(
    *, global_nx: int, global_ny: int, nPx: int, nPy: int
) -> tuple[int, int]:
    """Return the narrowest (x, y) tile this decomposition produces.

    ``axis_chunks`` is balanced, so the narrowest chunk is the floor division;
    ranks beyond ``length`` get width zero.
    """
    return (global_nx // nPx, global_ny // nPy)


def require_minimum_tile_extent(
    *,
    global_nx: int,
    global_ny: int,
    nPx: int,
    nPy: int,
    minimum: int,
    reason: str,
) -> None:
    """Reject a decomposition that slices any tile below ``minimum`` cells.

    Horizontal stencils have a smallest tile they can be evaluated on, and a
    decomposition finer than that does not merely lose accuracy: the rank whose
    tile is too small takes an early return that skips the exchanges its
    neighbours are still posting, and the run DEADLOCKS with no error message.
    Refusing the layout is
    the only outcome that is both loud and early.

    The verdict is computed from the global decomposition, so every rank reaches
    the same conclusion from the same numbers without a collective -- uniform by
    construction, and safe to call before the process group exists.

    ``minimum`` is passed in rather than fixed here: it is a property of the
    operators a run will execute, not of the layout abstraction, and a layout
    built for a model with no wide-stencil advection is legitimately allowed to
    be narrower.
    """
    min_nx, min_ny = minimum_tile_extents(
        global_nx=global_nx, global_ny=global_ny, nPx=nPx, nPy=nPy
    )
    offenders = [
        (name, parts, minimum_extent, axis_length, knob)
        for name, parts, minimum_extent, axis_length, knob in (
            ("x", nPx, min_nx, global_nx, "nPx"),
            ("y", nPy, min_ny, global_ny, "nPy"),
        )
        if minimum_extent < minimum
    ]
    if not offenders:
        return
    detail = "; ".join(
        f"{name}: {axis_length} cells over {parts} ranks leaves {minimum_extent} "
        f"per tile (need >= {minimum}; use {knob} <= {axis_length // minimum})"
        for name, parts, minimum_extent, axis_length, knob in offenders
    )
    raise ValueError(
        f"decomposition {nPx}x{nPy} slices {global_nx}x{global_ny} too finely for "
        f"{reason}: {detail}. A tile below the stencil width deadlocks the run "
        f"rather than failing, so the layout is refused here instead."
    )


def layout_from_rank(
    *,
    global_nx: int,
    global_ny: int,
    nPx: int,
    nPy: int,
    rank: int,
    local_rank: int = 0,
    world_size: int | None = None,
) -> ParallelLayout:
    """Build a rank-local 2-D horizontal layout.

    Rank order is x-fast, matching MITgcm's ``nPx``/``nPy`` convention:
    ``rank = tile_j * nPx + tile_i``.
    """

    expected_world = nPx * nPy
    if world_size is None:
        world_size = expected_world
    if world_size != expected_world:
        raise ValueError(
            f"world_size={world_size} must equal nPx*nPy={expected_world}"
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} out of range for world_size={world_size}")

    tile_i = rank % nPx
    tile_j = rank // nPx
    x0, x1 = axis_chunks(global_nx, nPx)[tile_i]
    y0, y1 = axis_chunks(global_ny, nPy)[tile_j]
    return ParallelLayout(
        nPx=nPx,
        nPy=nPy,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        global_nx=global_nx,
        global_ny=global_ny,
        tile_i=tile_i,
        tile_j=tile_j,
        x_start=x0,
        x_stop=x1,
        y_start=y0,
        y_stop=y1,
    )


def all_layouts(
    *,
    global_nx: int,
    global_ny: int,
    nPx: int,
    nPy: int,
) -> tuple[ParallelLayout, ...]:
    """Return layouts for all ranks in x-fast order."""

    return tuple(
        layout_from_rank(
            global_nx=global_nx,
            global_ny=global_ny,
            nPx=nPx,
            nPy=nPy,
            rank=rank,
            local_rank=rank,
        )
        for rank in range(nPx * nPy)
    )


_NEIGHBOUR_OFFSETS: Mapping[Side, tuple[int, int]] = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
    "southwest": (-1, -1),
    "southeast": (1, -1),
    "northwest": (-1, 1),
    "northeast": (1, 1),
}


def build_cartesian_neighbors(
    *,
    nPx: int,
    nPy: int,
    rank: int,
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> dict[Side, CartesianNeighbor | None]:
    """Return the 8-neighbour Cartesian table for ``rank``."""

    layout = layout_from_rank(
        global_nx=nPx,
        global_ny=nPy,
        nPx=nPx,
        nPy=nPy,
        rank=rank,
    )
    neighbours: dict[Side, CartesianNeighbor | None] = {}
    for side, (di, dj) in _NEIGHBOUR_OFFSETS.items():
        tile_i = layout.tile_i + di
        tile_j = layout.tile_j + dj

        if tile_i < 0 or tile_i >= nPx:
            if not periodic_x:
                neighbours[side] = None
                continue
            tile_i %= nPx
        if tile_j < 0 or tile_j >= nPy:
            if not periodic_y:
                neighbours[side] = None
                continue
            tile_j %= nPy

        neighbour_rank = tile_j * nPx + tile_i
        neighbours[side] = CartesianNeighbor(
            rank=neighbour_rank,
            tile_i=tile_i,
            tile_j=tile_j,
            offset_i=di,
            offset_j=dj,
        )
    return neighbours


def split_owned_region(field: torch.Tensor, layout: ParallelLayout) -> torch.Tensor:
    """Extract this rank's owned ``[..., y, x]`` region from a global tensor."""

    _check_global_field(field, layout)
    y_slice, x_slice = layout.owned_slices
    return field[..., y_slice, x_slice].contiguous()


def assemble_global_from_tiles(
    tiles: Sequence[torch.Tensor],
    layouts: Sequence[ParallelLayout],
) -> torch.Tensor:
    """Reassemble rank-local owned tiles into a global tensor."""

    if len(tiles) != len(layouts):
        raise ValueError("tiles and layouts must have the same length")
    if not tiles:
        raise ValueError("at least one tile is required")

    first = tiles[0]
    first_layout = layouts[0]
    out = first.new_empty(
        (*first.shape[:-2], first_layout.global_ny, first_layout.global_nx)
    )
    filled = torch.zeros(
        (first_layout.global_ny, first_layout.global_nx),
        dtype=torch.bool,
        device=first.device,
    )
    for tile, layout in zip(tiles, layouts, strict=True):
        expected = (*first.shape[:-2], layout.ny, layout.nx)
        if tuple(tile.shape) != expected:
            raise ValueError(
                f"tile rank {layout.rank} has shape {tuple(tile.shape)}, "
                f"expected {expected}"
            )
        y_slice, x_slice = layout.owned_slices
        out[..., y_slice, x_slice] = tile
        filled[y_slice, x_slice] = True

    if not bool(filled.all().item()):
        raise ValueError("layouts do not cover the full global domain")
    return out


def _check_global_field(field: torch.Tensor, layout: ParallelLayout) -> None:
    if field.ndim < 2:
        raise ValueError("field must have at least horizontal [y, x] dimensions")
    if tuple(field.shape[-2:]) != (layout.global_ny, layout.global_nx):
        raise ValueError(
            f"field horizontal shape {tuple(field.shape[-2:])} does not match "
            f"global layout {(layout.global_ny, layout.global_nx)}"
        )
