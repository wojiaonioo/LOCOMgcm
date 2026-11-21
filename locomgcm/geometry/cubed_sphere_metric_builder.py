"""Build cubed-sphere metrics from MITgcm cs32 face geometry binaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from .metrics import MetricSet
from .vertical_coordinates import ZLevelCoordinate


FACE_COUNT = 6

_CENTER_FIELDS = frozenset(("xC", "yC", "dxF", "dyF", "rA", "angleCS", "angleSN"))
_CORNER_FIELDS = frozenset(("xG", "yG", "dxV", "dyU", "rAz"))
_U_FACE_FIELDS = frozenset(("dxC", "rAw", "dyG"))
_V_FACE_FIELDS = frozenset(("dyC", "rAs", "dxG"))
_SIDES = ("west", "east", "south", "north")


def _grid_field_names() -> tuple[str, ...]:
    from locomgcm.io.cs32x15_loader import GRID_FIELD_NAMES

    return GRID_FIELD_NAMES


def calc_grid_angles(
    yG: torch.Tensor, dxG: torch.Tensor, dyG: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MITgcm CALC_GRID_ANGLES: cosine and sine of grid north against geographic north.

    Port of ``calc_grid_angles.F:68-110`` for the cell centres of each face.
    Inputs are raw face arrays ``(..., n+1, n+1)`` indexed ``[j, i]`` as in the
    face files: ``yG`` corner latitudes in degrees, ``dyG`` at the west face of
    cell ``(j, i)`` in rows ``0..n-1``, ``dxG`` at the south face in columns
    ``0..n-1``; the extra row and column hold the east and north edges, which
    MITgcm reads from the shared corners and edges of the neighbouring face.
    Returns ``(angleCS, angleSN)`` in the same ``(..., n+1, n+1)`` layout as the
    face files' records 17 and 18; only ``[..., :n, :n]`` are cell centres, the
    extra row and column are zero.
    """
    deg2rad = 2.0 * 3.14159265358979323844 / 360.0
    n = yG.shape[-1] - 1
    dyg = dyG[..., :n, :]
    dxg = dxG[..., :, :n]
    u_pseudo = torch.where(
        dyg > 0.0,
        -(yG[..., :n, :] - yG[..., 1:, :]) * deg2rad
        / torch.where(dyg > 0.0, dyg, torch.ones_like(dyg)),
        torch.zeros_like(dyg),
    )
    v_pseudo = torch.where(
        dxg > 0.0,
        (yG[..., :, :n] - yG[..., :, 1:]) * deg2rad
        / torch.where(dxg > 0.0, dxg, torch.ones_like(dxg)),
        torch.zeros_like(dxg),
    )
    u_c = 0.5 * (u_pseudo[..., :, :n] + u_pseudo[..., :, 1:])
    v_c = 0.5 * (v_pseudo[..., :n, :] + v_pseudo[..., 1:, :])
    norm = torch.sqrt(u_c * u_c + v_c * v_c)
    recip = torch.where(
        norm != 0.0, 1.0 / torch.where(norm != 0.0, norm, torch.ones_like(norm)), norm,
    )
    angle_cs = torch.zeros_like(yG)
    angle_sn = torch.zeros_like(yG)
    angle_cs[..., :n, :n] = u_c * recip
    angle_sn[..., :n, :n] = -v_c * recip
    return angle_cs, angle_sn


@dataclass(frozen=True)
class CubedSphereMetricSet(MetricSet):
    """`MetricSet` carrying stagger-aware raw cubed-sphere face geometry.

    `face_geometry` stores the 18 cs32 fields after MIT stagger slicing:
    cell-centre fields are `(6, 32, 32)`, corner fields `(6, 33, 33)`,
    U-face fields `(6, 32, 33)`, and V-face fields `(6, 33, 32)`.
    `angleSN` remains in MIT's face-local signed convention; across seams it
    is compared under a sign flip rather than same-sign averaging.
    """

    face_geometry: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def _dyG_tensor(self) -> torch.Tensor:
        """Genuine per-cell u-face ``dyG`` (``[face, ny, nx+1]`` = ``(6, 32, 33)``
        for cs32), replacing the lossy ``dyG_scalar_1d = dyG.mean()`` collapse
        (``build_face_metric_set``).
        """
        return self.face_geometry["dyG"]

    def _dyC_tensor(self) -> torch.Tensor:
        """Genuine per-cell v-face ``dyC`` (``[face, ny+1, nx]`` = ``(6, 33, 32)``
        for cs32) — the meridional centre-to-centre distance used in y-gradient
        denominators, replacing the uniform ``dyG_scalar_1d`` broadcast
        (``MetricSet._dyC_tensor``).
        Distinct from ``dy_center`` (= ``dyF``, the cell meridional width).
        """
        return self.face_geometry["dyC"]

    def _dxC_tensor(self) -> torch.Tensor:
        """Genuine per-cell u-face ``dxC`` (``[face, ny, nx+1]`` = ``(6, 32, 33)``
        for cs32) — the zonal centre-to-centre distance used in x-gradient
        denominators.
        """
        return self.face_geometry["dxC"]

    def _dxG_tensor(self) -> torch.Tensor:
        """Genuine per-cell v-face ``dxG`` (``[face, ny+1, nx]`` = ``(6, 33, 32)``
        for cs32) — the zonal cell-edge width at v-face stagger.
        """
        return self.face_geometry["dxG"]

    def _rAz_tensor(self) -> torch.Tensor:
        """Genuine per-cell corner ``rAz`` (``[face, ny+1, nx+1]`` = ``(6, 33, 33)``
        for cs32) — the vorticity-point cell area.
        """
        return self.face_geometry["rAz"]


def _infer_face_size(geometry_dict: Mapping[str, torch.Tensor]) -> int:
    """Derive the per-face cell count `n` from the raw `(6, n+1, n+1)` geometry.

    `load_face_geometry` returns every field at the geometry size `n+1`, so the
    face size is read from `rA` (always present, always `(6, n+1, n+1)`) rather
    than assuming the cs32 constant.  This is what makes the metric builder
    resolution-generic: cs32 yields `n=32`, cs192 yields `n=192`.
    """
    anchor = geometry_dict.get("rA")
    if not isinstance(anchor, torch.Tensor):
        raise KeyError("geometry dict missing required field 'rA'")
    if anchor.ndim != 3 or anchor.shape[0] != FACE_COUNT or anchor.shape[1] != anchor.shape[2]:
        raise ValueError(
            f"rA must have shape (6, n+1, n+1), got {tuple(anchor.shape)}"
        )
    n = anchor.shape[1] - 1
    if n < 1:
        raise ValueError(f"inferred face size n must be positive, got {n}")
    return n


def _validate_geometry_dict(
    geometry_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    field_names = _grid_field_names()
    missing = [name for name in field_names if name not in geometry_dict]
    if missing:
        raise KeyError(f"missing cubed-sphere geometry fields: {missing}")

    geometry_size = _infer_face_size(geometry_dict) + 1
    validated: dict[str, torch.Tensor] = {}
    for name in field_names:
        tensor = geometry_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
        if tuple(tensor.shape) != (FACE_COUNT, geometry_size, geometry_size):
            raise ValueError(
                f"{name} must have shape "
                f"({FACE_COUNT}, {geometry_size}, {geometry_size}), "
                f"got {tuple(tensor.shape)}"
            )
        validated[name] = tensor.contiguous()
    return validated


def _slice_face_geometry(
    geometry_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    n = _infer_face_size(geometry_dict)
    geometry_size = n + 1
    sliced: dict[str, torch.Tensor] = {}
    for name in _grid_field_names():
        tensor = geometry_dict[name]
        if name in _CENTER_FIELDS:
            sliced[name] = tensor[:, :n, :n].contiguous()
        elif name in _CORNER_FIELDS:
            sliced[name] = tensor.contiguous()
        elif name in _U_FACE_FIELDS:
            sliced[name] = tensor[:, :n, :geometry_size].contiguous()
        elif name in _V_FACE_FIELDS:
            sliced[name] = tensor[:, :geometry_size, :n].contiguous()
        else:
            raise KeyError(f"unclassified cubed-sphere geometry field: {name}")
    return sliced


def build_face_metric_set(
    geometry_dict: Mapping[str, torch.Tensor],
    vertical_coord: ZLevelCoordinate,
) -> MetricSet:
    """Build a face-aware `MetricSet` from `load_face_geometry` output.

    Stagger slicing follows `ini_curvilinear_grid.F:294-350`:
    cell-centre fields use the inner 32x32 block, corner fields keep the full
    33x33 block, `dxC/rAw/dyG` keep the U-face extra column, and
    `dyC/rAs/dxG` keep the V-face extra row.
    """

    geometry = _validate_geometry_dict(geometry_dict)
    n = _infer_face_size(geometry)
    face_geometry = _slice_face_geometry(geometry)
    dtype = face_geometry["rA"].dtype
    device = face_geometry["rA"].device
    depth = vertical_coord.interface_depth[-1].to(dtype=dtype, device=device)

    dx_center = face_geometry["dxF"]
    dy_center = face_geometry["dyF"]
    cell_area = face_geometry["rA"]
    depth_center = torch.full_like(cell_area, depth)
    cell_volume = cell_area * depth
    face_area_u = face_geometry["dyG"] * depth
    face_area_v = face_geometry["dxG"] * depth

    return CubedSphereMetricSet(
        dx_scalar=float(dx_center.mean().item()),
        dy_scalar=float(dy_center.mean().item()),
        dx_center=dx_center,
        dy_center=dy_center,
        cell_area=cell_area,
        cell_volume=cell_volume,
        depth_center=depth_center,
        face_area_u=face_area_u,
        face_area_v=face_area_v,
        dxG_row=face_geometry["dxG"][:, :n, :],
        dxC_row=face_geometry["dxC"][:, :, :n],
        dyG_scalar_1d=float(face_geometry["dyG"].mean().item()),
        dxG_row_ext=face_geometry["dxG"],
        rAz_row=face_geometry["rAz"],
        face_geometry=face_geometry,
    )


def build_tile_connectivity() -> Mapping[tuple[int, str], tuple[int, str, int]]:
    """Return MITgcm cs32 6-face topology as 0-indexed LOCOMgcm metadata.

    Source: `pkg/exch2/w2_set_cs6_facets.F:58-78` defines `facet_link` in MIT
    face numbering. LOCOMgcm stores `face = MIT face - 1`. Rotation codes
    are derived from `pkg/exch2/w2_set_f2f_index.F:126-184`: identity
    `pij=[[1,0],[0,1]]` is 0 degrees, `[[0,-1],[1,0]]` is 90 degrees, and
    `[[0,1],[-1,0]]` is 270 degrees.
    """

    connectivity: dict[tuple[int, str], tuple[int, str, int]] = {
        (0, "north"): (2, "west", 90),
        (0, "south"): (5, "north", 0),
        (0, "east"): (1, "west", 0),
        (0, "west"): (4, "north", 270),
        (1, "north"): (2, "south", 0),
        (1, "south"): (5, "east", 90),
        (1, "east"): (3, "south", 270),
        (1, "west"): (0, "east", 0),
        (2, "north"): (4, "west", 90),
        (2, "south"): (1, "north", 0),
        (2, "east"): (3, "west", 0),
        (2, "west"): (0, "north", 270),
        (3, "north"): (4, "south", 0),
        (3, "south"): (1, "east", 90),
        (3, "east"): (5, "south", 270),
        (3, "west"): (2, "east", 0),
        (4, "north"): (0, "west", 90),
        (4, "south"): (3, "north", 0),
        (4, "east"): (5, "west", 0),
        (4, "west"): (2, "north", 270),
        (5, "north"): (0, "south", 0),
        (5, "south"): (3, "east", 90),
        (5, "east"): (1, "south", 270),
        (5, "west"): (4, "east", 0),
    }
    _validate_tile_connectivity(connectivity)
    return connectivity


def _validate_tile_connectivity(
    connectivity: Mapping[tuple[int, str], tuple[int, str, int]],
) -> None:
    if len(connectivity) != FACE_COUNT * len(_SIDES):
        raise ValueError(f"expected 24 tile links, got {len(connectivity)}")

    for face in range(FACE_COUNT):
        for side in _SIDES:
            key = (face, side)
            if key not in connectivity:
                raise ValueError(f"missing tile link {key}")
            neighbour_face, neighbour_side, rotation = connectivity[key]
            if not 0 <= neighbour_face < FACE_COUNT:
                raise ValueError(f"{key} has invalid neighbour face {neighbour_face}")
            if neighbour_side not in _SIDES:
                raise ValueError(f"{key} has invalid neighbour side {neighbour_side!r}")
            if rotation not in (0, 90, 180, 270):
                raise ValueError(f"{key} has invalid rotation {rotation}")

            reverse = connectivity.get((neighbour_face, neighbour_side))
            if reverse is None:
                raise ValueError(f"{key} reverse link is missing")
            reverse_face, reverse_side, reverse_rotation = reverse
            if (reverse_face, reverse_side) != key:
                raise ValueError(f"{key} reverse link points to {(reverse_face, reverse_side)}")
            if (rotation + reverse_rotation) % 360 != 0:
                raise ValueError(f"{key} reverse rotation is not inverse")


__all__ = [
    "CubedSphereMetricSet",
    "build_face_metric_set",
    "build_tile_connectivity",
]
