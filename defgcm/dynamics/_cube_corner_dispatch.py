"""Shared face-local cube-corner dispatch helpers for dynamics kernels."""

from __future__ import annotations

import torch


def _has_cube_corner_override(grid: object) -> bool:
    return bool(getattr(grid, "cube_corner_indices", ()))


def _cs_face_axis(field: torch.Tensor, grid: object) -> int | None:
    ctx = getattr(grid, "_cube_step_ctx", None)
    if ctx is not None:
        # Localized cube tensors keep the leading face axis, including R=6.
        return 0 if field.ndim >= 3 and field.shape[0] == len(ctx.local_face_ids) else None
    face_count = int(getattr(grid, "face_count", 1))
    if face_count <= 1:
        return None
    if field.ndim >= 3 and field.shape[0] == face_count:
        return 0
    if field.ndim >= 3 and field.shape[-3] == face_count:
        return field.ndim - 3
    for axis, size in enumerate(field.shape[:-2]):
        if size == face_count:
            return axis
    return None


def _cs_face_first(field: torch.Tensor, grid: object) -> torch.Tensor:
    face_axis = _cs_face_axis(field, grid)
    if face_axis is None:
        raise ValueError(
            "cubed-sphere corner override requires a face dimension before "
            f"the horizontal axes, got shape {tuple(field.shape)}"
        )
    return field.movedim(face_axis, 0)


def _cube_corner_positions(grid: object) -> tuple[tuple[int, int, int, int], ...]:
    """Return owned corners as (global face, local face, j, i).

    The input ``cube_corner_indices`` stores one representative for each of
    the eight physical cube vertices.  This expands each representative
    through ``tile_connectivity`` to the three incident face corners used by
    MIT Family B in-place overrides, including ``mom_calc_relvort3.F`` and
    ``mom_calc_hfacz.F`` cube-corner branches.  It follows the same
    face-local dispatch shape as the Family A corner halo helpers
    modeled on ``fill_cs_corner_tr_rl.F`` / ``fill_cs_corner_uv_rl.F``.
    """
    nx = getattr(grid, "nx")
    ny = getattr(grid, "ny")
    positions: set[tuple[int, int, int]] = set()
    for face, i, j in getattr(grid, "cube_corner_indices", ()):
        positions.add((face, j, i))
        sides: list[str] = []
        if i == 0:
            sides.append("west")
        elif i == nx:
            sides.append("east")
        if j == 0:
            sides.append("south")
        elif j == ny:
            sides.append("north")
        for side in sides:
            neighbour_face, neighbour_side, rotation = grid.tile_connectivity[(face, side)]
            t = j if side in ("west", "east") else i
            t_neighbour = nx - t if rotation in (90, 270) else t
            if neighbour_side == "west":
                positions.add((neighbour_face, t_neighbour, 0))
            elif neighbour_side == "east":
                positions.add((neighbour_face, t_neighbour, nx))
            elif neighbour_side == "south":
                positions.add((neighbour_face, 0, t_neighbour))
            elif neighbour_side == "north":
                positions.add((neighbour_face, ny, t_neighbour))
            else:
                raise ValueError(f"unsupported cube side {neighbour_side!r}")
    ctx = getattr(grid, "_cube_step_ctx", None)
    if ctx is None:
        return tuple((face, face, j, i) for face, j, i in sorted(positions))
    return tuple(
        (face, ctx.global_to_local[face], j, i)
        for face, j, i in sorted(positions)
        if face in ctx.global_to_local
    )


def _cube_corner_name(grid: object, j: int, i: int) -> str:
    """Map a face-local corner index to the MIT SW/SE/NW/NE branch name.

    These labels match the branch names in MIT cube-corner stencils such as
    ``mom_calc_relvort3.F`` and ``mom_calc_hfacz.F``.  The helper is kept
    shared so relvort3 and hFacZ use the same face-local corner classifier.
    """
    if j == 0 and i == 0:
        return "sw"
    if j == 0 and i == getattr(grid, "nx"):
        return "se"
    if j == getattr(grid, "ny") and i == 0:
        return "nw"
    if j == getattr(grid, "ny") and i == getattr(grid, "nx"):
        return "ne"
    raise ValueError(f"not a cube face corner: j={j}, i={i}")


def _at_clamped(field: torch.Tensor, j: int, i: int) -> torch.Tensor:
    """Read a face-local neighbour with clamped semantics.

    This intentionally mirrors the current Family B self-consistency path:
    MIT formulas from ``mom_calc_relvort3.F`` / ``mom_calc_hfacz.F`` are
    evaluated on face-local values; there is no full Family A halo wiring
    yet.  Indices outside the face are clamped to the nearest in-face edge.
    """
    jj = min(max(j, 0), field.shape[-2] - 1)
    ii = min(max(i, 0), field.shape[-1] - 1)
    return field[..., jj, ii]
