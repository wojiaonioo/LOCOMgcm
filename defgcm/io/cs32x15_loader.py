"""Load MITgcm cs32x15 compact binaries without LOCOMgcm convention conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

GRID_FIELD_NAMES: tuple[str, ...] = (
    "xC",
    "yC",
    "dxF",
    "dyF",
    "rA",
    "xG",
    "yG",
    "dxV",
    "dyU",
    "rAz",
    "dxC",
    "dyC",
    "rAw",
    "rAs",
    "dxG",
    "dyG",
    "angleCS",
    "angleSN",
)

N_FACES = 6
FACE_SIZE = 32
FACE_GEOMETRY_SIZE = FACE_SIZE + 1

__all__ = [
    "GRID_FIELD_NAMES",
    "decompose_compact_to_faces",
    "load_compact_field",
    "load_face_geometry",
    "load_proj_2uEvN",
    "load_tile_geometry",
]


def _read_native_array(
    path: Path,
    dtype: str | np.dtype,
    expected_values: int,
) -> np.ndarray:
    path = Path(path)
    raw_dtype = np.dtype(dtype)
    raw = np.fromfile(path, dtype=raw_dtype)
    if raw.size != expected_values:
        raise ValueError(
            f"{path}: expected {expected_values} values of {raw_dtype}, got {raw.size}"
        )
    return raw.astype(raw_dtype.newbyteorder("="), copy=False)


def _validate_compact_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"shape must be a 2-D tuple, got {shape!r}")
    ny, nx = shape
    if ny <= 0 or nx <= 0:
        raise ValueError(f"shape entries must be positive, got {shape!r}")
    return ny, nx


def load_face_geometry(
    grid_dir: Path,
    n: int = FACE_SIZE,
    *,
    stem: str = "grid_cs32",
) -> dict[str, torch.Tensor]:
    """Read six `<stem>.face00?.bin` files as raw 18-field face geometry.

    The field order is the MITgcm `ini_curvilinear_grid.F` order:
    `xC yC dxF dyF rA xG yG dxV dyU rAz dxC dyC rAw rAs dxG dyG
    angleCS angleSN`. Each returned tensor has shape `(6, n+1, n+1)` and dtype
    `torch.float64`. This function intentionally performs no stagger-aware
    slicing and no LOCOMgcm sign or convention conversion.

    `n` is the per-face cell count (32 for cs32, 192 for cs192). The default
    `n=32` / `stem="grid_cs32"` reproduces the cs32x15 baseline byte-for-byte.
    """
    if n < 1:
        raise ValueError(f"face size n must be positive, got {n}")
    grid_dir = Path(grid_dir)
    geometry_size = n + 1
    expected_values = len(GRID_FIELD_NAMES) * geometry_size * geometry_size
    face_arrays: list[np.ndarray] = []

    for face in range(1, N_FACES + 1):
        path = grid_dir / f"{stem}.face{face:03d}.bin"
        arr = _read_native_array(path, ">f8", expected_values).reshape(
            len(GRID_FIELD_NAMES),
            geometry_size,
            geometry_size,
        )
        face_arrays.append(arr)

    stacked = np.stack(face_arrays, axis=0)
    return {
        name: torch.from_numpy(np.ascontiguousarray(stacked[:, index, :, :]))
        for index, name in enumerate(GRID_FIELD_NAMES)
    }


def load_tile_geometry(grid_dir: Path, n: int) -> dict[str, torch.Tensor]:
    """Read six ``tile00?.mitgrid`` files and compute the grid angles.

    MITgcm reads these when ``horizGridFile`` is empty
    (``ini_curvilinear_grid.F:278-284``): the first 16 fields of
    ``GRID_FIELD_NAMES``, each ``(n+1, n+1)`` big-endian float64, and no angles,
    which CALC_GRID_ANGLES then computes from ``yG``, ``dxG``, ``dyG``
    (``ini_curvilinear_grid.F:343-354, 417``).  Returns the same 18-field
    ``(6, n+1, n+1)`` float64 dictionary as ``load_face_geometry``.
    """
    from defgcm.geometry.cubed_sphere_metric_builder import calc_grid_angles

    if n < 1:
        raise ValueError(f"face size n must be positive, got {n}")
    grid_dir = Path(grid_dir)
    geometry_size = n + 1
    read_names = GRID_FIELD_NAMES[:16]
    face_arrays = [
        _read_native_array(
            grid_dir / f"tile{face:03d}.mitgrid", ">f8",
            len(read_names) * geometry_size * geometry_size,
        ).reshape(len(read_names), geometry_size, geometry_size)
        for face in range(1, N_FACES + 1)
    ]
    stacked = np.stack(face_arrays, axis=0)
    geometry = {
        name: torch.from_numpy(np.ascontiguousarray(stacked[:, index, :, :]))
        for index, name in enumerate(read_names)
    }
    geometry["angleCS"], geometry["angleSN"] = calc_grid_angles(
        geometry["yG"], geometry["dxG"], geometry["dyG"],
    )
    return geometry


def load_compact_field(
    path: Path,
    n_records: int = 1,
    *,
    dtype: str = ">f8",
    shape: tuple[int, int] = (FACE_SIZE, N_FACES * FACE_SIZE),
) -> torch.Tensor:
    """Read a cs32x15 non-grid binary in compact face-stacked layout.

    The per-record layout is `(32, 192)` in Python `[j, i]` order, with the six
    32x32 faces stacked along the last axis. This is MITgcm exch2 compact I/O,
    not the logical 12-tile `(384, 16)` interpretation. The loader is neutral:
    it preserves MIT file signs such as negative-ocean bathymetry and Qnet's
    positive-upward cooling convention for later ingestion code to convert.
    """
    if n_records <= 0:
        raise ValueError(f"n_records must be positive, got {n_records}")

    ny, nx = _validate_compact_shape(shape)
    expected_values = n_records * ny * nx
    arr = _read_native_array(Path(path), dtype, expected_values).reshape(
        n_records,
        ny,
        nx,
    )
    if n_records == 1:
        arr = arr[0]
    return torch.from_numpy(np.ascontiguousarray(arr))


def decompose_compact_to_faces(
    compact: torch.Tensor,
    n: int = FACE_SIZE,
) -> torch.Tensor:
    """Decompose compact `(*, n, 6*n)` fields into `(*, 6, n, n)` faces.

    Faces are stacked along the compact I axis as `k*n:(k+1)*n` for face
    indices 0..5, matching MITgcm exch2 face-stacked I/O. This function only
    reshapes the compact layout and does not apply stagger-aware slicing,
    vector rotation, or sign conversion. The default `n=32` reproduces the
    cs32x15 layout exactly.
    """
    if n < 1:
        raise ValueError(f"face size n must be positive, got {n}")
    if compact.ndim < 2:
        raise ValueError(
            f"compact tensor must have at least 2 dimensions, got {compact.ndim}"
        )
    if tuple(compact.shape[-2:]) != (n, N_FACES * n):
        raise ValueError(
            "compact tensor must end with shape "
            f"({n}, {N_FACES * n}), got {tuple(compact.shape[-2:])}"
        )

    leading = tuple(compact.shape[:-2])
    reshaped = compact.reshape(*leading, n, N_FACES, n)
    return reshaped.transpose(-3, -2).contiguous()


def load_proj_2uEvN(
    path: Path,
    n: int = FACE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read `proj_cs32_2uEvN.bin` as `(angleCos, angleSin)` face tensors.

    The file stores two big-endian float64 records, each logically `(6, n, n)`.
    The record payload follows MITgcm/MATLAB `[x, face, y]` order, so it is
    returned as face-major `(6, n, n)`. These are a cos/sin pair for one
    angle per cell, not a four-entry rotation matrix.

    `n` is the per-face cell count (32 for cs32). The default `n=32`
    reproduces the cs32x15 baseline byte-for-byte.
    """
    expected_values = 2 * N_FACES * n * n
    arr = _read_native_array(Path(path), ">f8", expected_values).reshape(
        2,
        n,
        N_FACES,
        n,
    )
    arr = np.ascontiguousarray(arr.transpose(0, 2, 1, 3))
    return torch.from_numpy(arr[0]), torch.from_numpy(arr[1])
