"""Topology adapters for hFacZ corner construction."""

from __future__ import annotations

import torch

from defgcm.geometry import StructuredCGrid

from ._cube_corner_dispatch import (
    _at_clamped,
    _cs_face_first,
    _cube_corner_name,
    _cube_corner_positions,
    _has_cube_corner_override,
)


def apply_hfacz_corner_min3_override(
    hFacZ: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    grid: StructuredCGrid,
    *,
    option: int,
) -> torch.Tensor:
    """Apply MIT hZoption=1 valence-3 corner min override when present."""
    if option != 1 or not _has_cube_corner_override(grid):
        return hFacZ

    result = hFacZ.clone()
    result_faces = _cs_face_first(result, grid)
    hFacW_faces = _cs_face_first(hFacW, grid)
    hFacS_faces = _cs_face_first(hFacS, grid)

    for _face, local_face, j, i in _cube_corner_positions(grid):
        out_face = result_faces[local_face]
        hFacW_face = hFacW_faces[local_face]
        hFacS_face = hFacS_faces[local_face]
        corner = _cube_corner_name(grid, j, i)

        hfac_w = _at_clamped(hFacW_face, j, i)
        hfac_w_south = _at_clamped(hFacW_face, j - 1, i)
        if corner in ("sw", "nw"):
            hfac_s = _at_clamped(hFacS_face, j, i)
        else:
            hfac_s = _at_clamped(hFacS_face, j, i - 1)
        out_face[..., j, i] = torch.minimum(torch.minimum(hfac_w, hfac_w_south), hfac_s)

    return result


def build_hfacz_for_grid(
    grid: StructuredCGrid,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build cube vorticity-point hFacZ with the MIT valence-3 corner override."""
    from defgcm.dynamics.viscosity_geometry import _compute_hFacZ_4neighbor

    if grid.masks.hFacW is None or grid.masks.hFacS is None:
        hFacW = torch.ones(
            grid.face_count,
            grid.nz,
            grid.ny,
            grid.nx + 1,
            dtype=dtype,
            device=device,
        )
        hFacS = torch.ones(
            grid.face_count,
            grid.nz,
            grid.ny + 1,
            grid.nx,
            dtype=dtype,
            device=device,
        )
    else:
        hFacW = grid.masks.hFacW.to(dtype=dtype, device=device)
        hFacS = grid.masks.hFacS.to(dtype=dtype, device=device)

    return _compute_hFacZ_4neighbor(
        hFacW,
        hFacS,
        grid,
        hZoption=1,
    )
