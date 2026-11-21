"""Cubed-sphere bathymetry ingestion and hFac mask construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .masks import MaskSet
from .vertical_coordinates import ZLevelCoordinate


FACE_COUNT = 6
FACE_SIZE = 32


@dataclass(frozen=True)
class CubedSphereMaskSet(MaskSet):
    """`MaskSet` variant for face-leading, depth-dependent cubed-sphere masks."""

    @property
    def has_3d_masks(self) -> bool:
        return (
            self.center.ndim >= 3
            and self.hFacC is not None
            and self.hFacW is not None
            and self.hFacS is not None
        )


def project_bathy_to_cubed_sphere(
    bathy_path: Path,
    *,
    n: int = FACE_SIZE,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Load a MIT compact `bathy.bin` as LOCOMgcm positive-depth face bathymetry.

    MIT input bathymetry stores ocean as negative elevation and land as zero
    or positive elevation. LOCOMgcm stores ocean depth as positive and land
    as zero, so ingestion applies `where(bathy < 0, -bathy, 0)` and clamps the
    result to the documented non-negative depth contract.

    `n` is the per-face cell count of the compact file (32 for cs32, 192 for
    cs192). The compact layout is `(n, 6*n)`; the default `n=32` reproduces the
    cs32x15 bathymetry ingestion byte-for-byte.

    `dtype` note: the file is read as float64 and this is the last chance to
    keep it that way. A caller that feeds the result to the mask builder must
    leave `dtype` at float64 — the hFac closure test that decides wet/dry has
    no tolerance, so a run-dtype projection moves the topology and widening it
    back afterwards cannot recover the low bits (`grid.py`
    `_GEOMETRY_DTYPE_NOTE`).
    """
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype)!r}")
    if n < 1:
        raise ValueError(f"face size n must be positive, got {n}")

    from locomgcm.io.cs32x15_loader import (
        decompose_compact_to_faces,
        load_compact_field,
    )

    compact = load_compact_field(
        Path(bathy_path), n_records=1, dtype=">f8", shape=(n, FACE_COUNT * n)
    )
    bathy_mit = decompose_compact_to_faces(compact, n=n).to(dtype=dtype)
    bathy_to = torch.where(
        bathy_mit < 0.0,
        -bathy_mit,
        torch.zeros_like(bathy_mit),
    )
    return torch.clamp(bathy_to, min=0.0)


def build_cubed_sphere_mask_set(
    bathy_faces: torch.Tensor,
    vertical_coord: ZLevelCoordinate,
    *,
    hfac_min: float = 0.1,
    hfac_min_dr: float = 20.0,
) -> MaskSet:
    """Build a face-aware `MaskSet` from positive-depth cubed-sphere bathy.

    Input bathymetry must be `(6, ny, nx)` with positive ocean depth and zero
    land. The returned masks are face-leading: `center/hFacC` are
    `(6, nz, ny, nx)`, `u/hFacW` are `(6, nz, ny, nx+1)`, and `v/hFacS` are
    `(6, nz, ny+1, nx)`. Isolated wet cells are not pruned.
    """
    if not isinstance(bathy_faces, torch.Tensor):
        raise TypeError(f"bathy_faces must be a torch.Tensor, got {type(bathy_faces)!r}")
    if bathy_faces.ndim != 3:
        raise ValueError(f"expected bathy_faces shape (6, ny, nx), got {tuple(bathy_faces.shape)}")
    if bathy_faces.shape[0] != FACE_COUNT:
        raise ValueError(f"expected {FACE_COUNT} faces, got {bathy_faces.shape[0]}")
    if bathy_faces.shape[1] < 1 or bathy_faces.shape[2] < 1:
        raise ValueError(f"bathy face dimensions must be positive, got {tuple(bathy_faces.shape)}")

    iface = vertical_coord.interface_depth
    device = iface.device
    bathy = torch.clamp(bathy_faces.to(dtype=torch.float64, device=device), min=0.0)

    face_masks = [
        MaskSet.from_bathymetry_3d(
            bathy[face],
            iface,
            device=device,
            hfac_min=hfac_min,
            hfac_min_dz=hfac_min_dr,
            boundary_topology=None,
        )
        for face in range(FACE_COUNT)
    ]

    return CubedSphereMaskSet(
        center=torch.stack([mask.center for mask in face_masks], dim=0),
        u=torch.stack([mask.u for mask in face_masks], dim=0),
        v=torch.stack([mask.v for mask in face_masks], dim=0),
        hFacC=_stack_required(face_masks, "hFacC"),
        hFacW=_stack_required(face_masks, "hFacW"),
        hFacS=_stack_required(face_masks, "hFacS"),
        maskInC=_stack_required(face_masks, "maskInC"),
        maskInW=_stack_required(face_masks, "maskInW"),
        maskInS=_stack_required(face_masks, "maskInS"),
    )


def _stack_required(face_masks: list[MaskSet], field_name: str) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for mask in face_masks:
        tensor = getattr(mask, field_name)
        if tensor is None:
            raise RuntimeError(f"MaskSet.from_bathymetry_3d returned {field_name}=None")
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)


__all__ = [
    "CubedSphereMaskSet",
    "build_cubed_sphere_mask_set",
    "project_bathy_to_cubed_sphere",
]
