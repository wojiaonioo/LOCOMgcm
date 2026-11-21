"""Topology adapters for flux-form momentum advection."""

from __future__ import annotations

import torch

from locomgcm.geometry import StructuredCGrid

from ._cube_corner_dispatch import (
    _at_clamped,
    _cs_face_first,
    _cube_corner_name,
    _cube_corner_positions,
    _has_cube_corner_override,
)


def _ordered_corner_sum3(
    face: int,
    corner: str,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    if corner == "sw":
        return (a + b) + c
    if corner == "se":
        if face == 1:
            return (b + c) + a
        if face == 3:
            return (c + a) + b
        return (a + b) + c
    if corner == "nw":
        if face == 0:
            return (c + a) + b
        if face == 2:
            return (b + a) + c
        return (a + b) + c
    if corner == "ne" and face % 2 == 0:
        return (b + c) + a
    return (a + b) + c


def corner_flux3_value(
    face: int,
    corner: str,
    *,
    v_left: torch.Tensor,
    v_right: torch.Tensor,
    u_south: torch.Tensor,
    u_north: torch.Tensor,
) -> torch.Tensor:
    if corner == "sw":
        a = v_right * u_north
        b = v_left * u_north
        c = v_right * u_south
    elif corner == "se":
        a = v_left * u_north
        b = v_right * u_north
        c = v_left * u_south
    elif corner == "nw":
        a = v_right * u_south
        b = v_left * u_south
        c = v_right * u_north
    elif corner == "ne":
        a = v_left * u_south
        b = v_right * u_south
        c = v_left * u_north
    else:
        raise ValueError(f"unsupported cube corner {corner!r}")
    return _ordered_corner_sum3(face, corner, a, b, c) / 3.0


def apply_corner_flux3_override(
    corner_flux: torch.Tensor,
    transport: torch.Tensor,
    velocity: torch.Tensor,
    grid: StructuredCGrid,
    *,
    flux_kind: str,
) -> torch.Tensor:
    if not _has_cube_corner_override(grid):
        return corner_flux

    result = corner_flux.clone()
    result_faces = _cs_face_first(result, grid)
    transport_faces = _cs_face_first(transport, grid)
    velocity_faces = _cs_face_first(velocity, grid)
    positions = _cube_corner_positions(grid)

    for face, local_face, j, i in positions:
        out_face = result_faces[local_face]
        transport_face = transport_faces[local_face]
        velocity_face = velocity_faces[local_face]
        corner = _cube_corner_name(grid, j, i)
        if flux_kind == "vu":
            v_left = _at_clamped(transport_face, j, i - 1)
            v_right = _at_clamped(transport_face, j, i)
            u_south = _at_clamped(velocity_face, j - 1, i)
            u_north = _at_clamped(velocity_face, j, i)
        elif flux_kind == "uv":
            u_south = _at_clamped(transport_face, j - 1, i)
            u_north = _at_clamped(transport_face, j, i)
            v_left = _at_clamped(velocity_face, j, i - 1)
            v_right = _at_clamped(velocity_face, j, i)
        else:
            raise ValueError(f"unsupported corner flux kind {flux_kind!r}")
        out_face[..., j, i] = corner_flux3_value(
            face,
            corner,
            v_left=v_left,
            v_right=v_right,
            u_south=u_south,
            u_north=u_north,
        )

    return result
