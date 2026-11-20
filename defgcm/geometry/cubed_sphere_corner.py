"""Cube-corner helpers for MITgcm-style cubed-sphere grids."""

from __future__ import annotations

from typing import Literal

import torch

from .grid import Grid


_FACE_COUNT = 6
_SIDES = ("west", "east", "south", "north")

Side = Literal["west", "east", "south", "north"]


def build_cube_corner_indices(n: int) -> tuple[tuple[int, int, int], ...]:
    """Return one representative triple for each distinct cube vertex.

    LOCOMgcm stores 0-indexed ``(face, i, j)`` triples.  These eight
    entries represent the eight physical 3-valent cube vertices, not the
    24 face-local face corners.  Each cube vertex has three face-corner
    equivalents via ``tile_connectivity``; only one representative per
    equivalence class is listed here.

    ``n`` is the per-face cell count of the grid being built (32 for cs32,
    192 for cs192); the corner stagger has ``n+1`` points, so ``n`` indexes
    the east/north face corner.  Only the index magnitude scales with
    resolution — the face/edge topology is resolution-independent.

    MIT sources:
    - ``pkg/exch2/W2_EXCH2_TOPOLOGY.h:38-80`` defines the face-edge topology
      arrays consumed by EXCH2 corner logic.
    - ``pkg/exch2/W2_EXCH2_PARAMS.h:78-83`` documents the packed
      ``facet_link(edge, face)=XX.edge`` face-edge adjacency.
    - ``pkg/exch2/w2_set_cs6_facets.F:58-78`` hard-codes the 6-face cube
      adjacency used by cs32x15.
    - Face-0 plus face-1 representatives double-count two equivalence classes; this list enumerates all eight distinct classes.
    """

    if n < 1:
        raise ValueError(f"face size n must be positive, got {n}")
    return (
        (0, 0, 0),
        (0, n, 0),
        (0, 0, n),
        (0, n, n),
        (4, 0, 0),
        (1, n, 0),
        (5, 0, 0),
        (1, n, n),
    )


def cube_corner_mask(grid: Grid) -> torch.Tensor:
    """Return a corner-staggered mask with true values at cube vertices."""

    device = grid.metrics.cell_area.device
    mask = torch.zeros(
        (_FACE_COUNT, grid.ny + 1, grid.nx + 1),
        dtype=torch.bool,
        device=device,
    )
    if getattr(grid, "face_count", 1) != _FACE_COUNT:
        return mask

    mask[:, 0, 0] = True
    mask[:, 0, grid.nx] = True
    mask[:, grid.ny, 0] = True
    mask[:, grid.ny, grid.nx] = True
    return mask


def fill_cs_corner_tr_rl(field: torch.Tensor, mode: int, grid: Grid) -> torch.Tensor:
    """Fill scalar corner halos using MIT ``FILL_CS_CORNER_TR_RL`` mode 1 or 2.

    ``mode=1`` follows ``fill_cs_corner_tr_rl.F:162-189`` for X-gradient
    preparation.  ``mode=2`` follows ``fill_cs_corner_tr_rl.F:232-259`` for
    Y-gradient preparation.  If ``field`` is unpadded with shape
    ``(6, ..., ny, nx)``, a one-cell MIT-aware side halo is built first using
    ``grid.tile_connectivity``; if it is already halo-padded, the input shape
    is preserved.
    """

    if not isinstance(field, torch.Tensor):
        raise TypeError("field must be a torch.Tensor")
    if mode not in (1, 2):
        raise ValueError("mode must be 1 or 2")
    if getattr(grid, "face_count", 1) != _FACE_COUNT:
        return field

    ctx = getattr(grid, "_cube_step_ctx", None)
    if ctx is not None:
        # Face-local: the grid is rank-local (leading face dim n_local < 6),
        # so the in-tensor side-halo build (_prepare_scalar_halo, which asserts a
        # 6-face leading dim and reads neighbour faces directly) cannot run.  The
        # OL=1 side halo is bitwise-identical to the proven distributed scalar
        # exchange (verified for all 24 seams and rotations), and the corner fill
        # is purely in-face, so it runs unchanged per owned local face.  The
        # exchange also writes the four corner cells, but _fill_tr_face_corner
        # overwrites exactly those four (reading only side strips), so they are
        # harmless.  The side halo is mode-independent; only the corner fill
        # depends on mode.
        from defgcm.geometry.exchange import (
            HaloExtentV1,
            exchange_cubed_sphere_scalar_distributed,
        )

        if tuple(field.shape[-2:]) == (grid.ny, grid.nx):
            # Unpadded (tracer GAD path): build the OL=1 side halo via the proven
            # distributed scalar exchange, then fill corners in-face (unchanged).
            out = exchange_cubed_sphere_scalar_distributed(
                field, grid=grid, location="T",
                extent=HaloExtentV1.uniform(1), ctx=ctx,
            )
            olc = 1
        else:
            # Pre-padded (KPP: ``sc(temp)`` = distributed OL=4 side exchange already
            # ran).  The side strips carry the correct cross-face data, so the corner
            # fill is purely in-face per owned local face at the existing halo width
            # -- byte-identical to the single-rank ``_prepare_existing_halo`` path.
            extra_y = field.shape[-2] - grid.ny
            extra_x = field.shape[-1] - grid.nx
            if extra_x != extra_y or extra_x < 2 or extra_x % 2:
                raise ValueError(
                    "distributed corner fill needs unpadded or symmetric-even halo"
                )
            olc = extra_x // 2
            out = field.clone()
        for local_face in range(len(ctx.local_face_ids)):
            _fill_tr_face_corner(
                out, face=local_face, mode=mode, nx=grid.nx, ny=grid.ny, olx=olc, oly=olc
            )
        return out

    out, olx, oly = _prepare_scalar_halo(field, grid)
    for face in range(_FACE_COUNT):
        _fill_tr_face_corner(out, face=face, mode=mode, nx=grid.nx, ny=grid.ny, olx=olx, oly=oly)
    return out


def fill_cs_corner_uv_rl(
    field_u: torch.Tensor,
    field_v: torch.Tensor,
    grid: Grid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill C-grid vector corner halos using MIT ``FILL_CS_CORNER_UV_RL``.

    The staggered corner formulas follow ``fill_cs_corner_uv_rl.F:124-182``.
    Vector signs are derived from the two face seam rotations that meet at
    each corner, matching the MIT ``withSigns=.TRUE.`` SW/NE sign changes.
    """

    if not isinstance(field_u, torch.Tensor) or not isinstance(field_v, torch.Tensor):
        raise TypeError("field_u and field_v must be torch.Tensor instances")
    if getattr(grid, "face_count", 1) != _FACE_COUNT:
        return field_u, field_v
    if tuple(field_u.shape) != tuple(field_v.shape):
        raise ValueError("field_u and field_v must have the same halo shape")

    out_u, olx, oly = _prepare_existing_halo(field_u, grid)
    out_v, volx, voly = _prepare_existing_halo(field_v, grid)
    if (volx, voly) != (olx, oly):
        raise ValueError("field_u and field_v halo widths must match")

    for face in range(_FACE_COUNT):
        _fill_uv_face_corner(
            out_u,
            out_v,
            face=face,
            grid=grid,
            nx=grid.nx,
            ny=grid.ny,
            olx=olx,
            oly=oly,
        )
    return out_u, out_v


def _prepare_scalar_halo(field: torch.Tensor, grid: Grid) -> tuple[torch.Tensor, int, int]:
    if field.ndim < 3:
        raise ValueError("cubed-sphere fields must have shape (6, ..., j, i)")
    if field.shape[0] != _FACE_COUNT:
        raise ValueError(f"cubed-sphere fields must have leading face dimension {_FACE_COUNT}")
    if tuple(field.shape[-2:]) == (grid.ny, grid.nx):
        return _pad_scalar_center_halo(field, grid), 1, 1
    return _prepare_existing_halo(field, grid)


def _prepare_existing_halo(field: torch.Tensor, grid: Grid) -> tuple[torch.Tensor, int, int]:
    if field.ndim < 3:
        raise ValueError("cubed-sphere fields must have shape (6, ..., j, i)")
    if field.shape[0] != _FACE_COUNT:
        raise ValueError(f"cubed-sphere fields must have leading face dimension {_FACE_COUNT}")

    extra_y = field.shape[-2] - grid.ny
    extra_x = field.shape[-1] - grid.nx
    if extra_x < 2 or extra_y < 2 or extra_x % 2 or extra_y % 2:
        raise ValueError(
            "field must be unpadded (6, ..., ny, nx) or symmetrically "
            "halo-padded around the last two dimensions"
        )
    return field.clone(), extra_x // 2, extra_y // 2


def _pad_scalar_center_halo(field: torch.Tensor, grid: Grid) -> torch.Tensor:
    if len(grid.tile_connectivity) != _FACE_COUNT * len(_SIDES):
        raise ValueError("cubed-sphere side halo fill requires 24 tile_connectivity entries")

    out = field.new_empty((*field.shape[:-2], grid.ny + 2, grid.nx + 2))
    out[(slice(None), Ellipsis, slice(1, grid.ny + 1), slice(1, grid.nx + 1))] = field

    for face in range(_FACE_COUNT):
        _copy_scalar_side_halo(out, field, grid, face=face, side="west")
        _copy_scalar_side_halo(out, field, grid, face=face, side="east")
        _copy_scalar_side_halo(out, field, grid, face=face, side="south")
        _copy_scalar_side_halo(out, field, grid, face=face, side="north")
    return out


def _copy_scalar_side_halo(
    out: torch.Tensor,
    field: torch.Tensor,
    grid: Grid,
    *,
    face: int,
    side: Side,
) -> None:
    neighbour_face, neighbour_side, rotation = grid.tile_connectivity[(face, side)]
    values = _source_edge(field, face=neighbour_face, side=neighbour_side)
    if rotation in (90, 270):
        values = values.flip(-1)

    if side == "west":
        out[(face, Ellipsis, slice(1, grid.ny + 1), 0)] = values
    elif side == "east":
        out[(face, Ellipsis, slice(1, grid.ny + 1), grid.nx + 1)] = values
    elif side == "south":
        out[(face, Ellipsis, 0, slice(1, grid.nx + 1))] = values
    elif side == "north":
        out[(face, Ellipsis, grid.ny + 1, slice(1, grid.nx + 1))] = values
    else:
        raise ValueError(f"unsupported side {side!r}")


def _source_edge(field: torch.Tensor, *, face: int, side: Side) -> torch.Tensor:
    if side == "west":
        return field[(face, Ellipsis, slice(None), 0)]
    if side == "east":
        return field[(face, Ellipsis, slice(None), -1)]
    if side == "south":
        return field[(face, Ellipsis, 0, slice(None))]
    if side == "north":
        return field[(face, Ellipsis, -1, slice(None))]
    raise ValueError(f"unsupported side {side!r}")


def _fill_tr_face_corner(
    out: torch.Tensor,
    *,
    face: int,
    mode: int,
    nx: int,
    ny: int,
    olx: int,
    oly: int,
) -> None:
    i0 = olx
    j0 = oly
    for j in range(1, oly + 1):
        for i in range(1, olx + 1):
            if mode == 1:
                out[(face, Ellipsis, j0 - j, i0 - i)] = out[
                    (face, Ellipsis, j0 + i - 1, i0 - j)
                ]
                out[(face, Ellipsis, j0 - j, i0 + nx + i - 1)] = out[
                    (face, Ellipsis, j0 + i - 1, i0 + nx + j - 1)
                ]
                out[(face, Ellipsis, j0 + ny + j - 1, i0 - i)] = out[
                    (face, Ellipsis, j0 + ny - i, i0 - j)
                ]
                out[(face, Ellipsis, j0 + ny + j - 1, i0 + nx + i - 1)] = out[
                    (face, Ellipsis, j0 + ny - i, i0 + nx + j - 1)
                ]
            else:
                out[(face, Ellipsis, j0 - j, i0 - i)] = out[
                    (face, Ellipsis, j0 - i, i0 + j - 1)
                ]
                out[(face, Ellipsis, j0 - j, i0 + nx + i - 1)] = out[
                    (face, Ellipsis, j0 - i, i0 + nx - j)
                ]
                out[(face, Ellipsis, j0 + ny + j - 1, i0 - i)] = out[
                    (face, Ellipsis, j0 + ny + i - 1, i0 + j - 1)
                ]
                out[(face, Ellipsis, j0 + ny + j - 1, i0 + nx + i - 1)] = out[
                    (face, Ellipsis, j0 + ny + i - 1, i0 + nx - j)
                ]


def _fill_uv_face_corner(
    out_u: torch.Tensor,
    out_v: torch.Tensor,
    *,
    face: int,
    grid: Grid,
    nx: int,
    ny: int,
    olx: int,
    oly: int,
) -> None:
    i0 = olx
    j0 = oly
    sw_sign = _corner_vector_sign(grid, face, "west", "south")
    ne_sign = _corner_vector_sign(grid, face, "east", "north")

    for j in range(1, oly + 1):
        for i in range(1, olx + 1):
            out_u[(face, Ellipsis, j0 - j, i0 - i)] = sw_sign * out_v[
                (face, Ellipsis, j0 + i, i0 - j)
            ]
            out_v[(face, Ellipsis, j0 - j, i0 - i)] = sw_sign * out_u[
                (face, Ellipsis, j0 - i, i0 + j)
            ]

            if i >= 2:
                out_u[(face, Ellipsis, j0 - j, i0 + nx + i - 1)] = out_v[
                    (face, Ellipsis, j0 + i - 1, i0 + nx + j - 1)
                ]
                out_u[(face, Ellipsis, j0 + ny + j - 1, i0 + nx + i - 1)] = (
                    ne_sign
                    * out_v[(face, Ellipsis, j0 + ny + 1 - i, i0 + nx + j - 1)]
                )

            out_v[(face, Ellipsis, j0 - j, i0 + nx + i - 1)] = out_u[
                (face, Ellipsis, j0 - i, i0 + nx - j)
            ]
            out_u[(face, Ellipsis, j0 + ny + j - 1, i0 - i)] = out_v[
                (face, Ellipsis, j0 + ny - i, i0 - j)
            ]

            if j >= 2:
                out_v[(face, Ellipsis, j0 + ny + j - 1, i0 - i)] = out_u[
                    (face, Ellipsis, j0 + ny + i - 1, i0 + j - 1)
                ]
                out_v[(face, Ellipsis, j0 + ny + j - 1, i0 + nx + i - 1)] = (
                    ne_sign
                    * out_u[(face, Ellipsis, j0 + ny + i - 1, i0 + nx + 1 - j)]
                )


def _corner_vector_sign(grid: Grid, face: int, side_a: Side, side_b: Side) -> int:
    rotations = (
        grid.face_seam_rotation(face, side_a),
        grid.face_seam_rotation(face, side_b),
    )
    rotated_count = sum(rotation in (90, 270) for rotation in rotations)
    return -1 if rotated_count % 2 else 1


__all__ = [
    "build_cube_corner_indices",
    "cube_corner_mask",
    "fill_cs_corner_tr_rl",
    "fill_cs_corner_uv_rl",
]
