"""Single accessor for a store's per-variable time series.

The ``{group}/{var}_spatial`` zarr arrays (chunks ``(1[,1], ny, nx)``) are the
only copy of the data.  The writer does not also stream them into a
``timeseries/{group}_{var}.npy`` sidecar: that byte-identical duplicate would
double the store (~150 GB at 1696x960x90 x 41 frames).  Every consumer reads one
frame at a time (``arr[t]``), which a chunked zarr array serves directly.

``open_series`` returns the zarr array when present and falls back to a legacy
memory-mapped sidecar for stores written before the change, so both read the
same way: ``.shape``/``.ndim``/``.dtype``, ``arr[t]`` -> ``np.ndarray``,
``np.asarray(arr)`` for the (rare) whole-series consumer.

Point extraction ``arr[:, k, j, i]`` reads one horizontal chunk per frame
(nt chunks) instead of nt pages of a memmap; at 160x160 that is trivial, at
960x1696 it is ~7 MB per frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr


def open_series(store_path: Path | str, group: str, var: str) -> Any | None:
    """Frame-indexable ``(nt[, nz], ny, nx)`` series for ``group``/``var``.

    Returns ``None`` when the store holds neither the zarr array nor a legacy
    ``.npy`` sidecar for the pair.
    """
    store_path = Path(store_path)
    try:
        root = zarr.open_group(str(store_path), mode="r")
        return root[f"{group}/{var}_spatial"]
    except (KeyError, FileNotFoundError, ValueError):
        return _open_legacy_series(store_path, group, var)


def _open_legacy_series(store_path: Path, group: str, var: str) -> Any | None:
    """Read the legacy ``timeseries/<group>_<var>.npy`` sidecar layout."""
    legacy = store_path / "timeseries" / f"{group}_{var}.npy"
    if legacy.exists():
        return np.load(legacy, mmap_mode="r")
    return None


def series_nt(store_path: Path | str, group: str = "model") -> int:
    """Number of frames in ``group`` (0 when the group holds no series)."""
    for var in ("temp", "salt", "u", "v", "w", "ssh"):
        arr = open_series(store_path, group, var)
        if arr is not None:
            return int(arr.shape[0])
    return 0
