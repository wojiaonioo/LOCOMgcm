"""Shared helpers for MIT binary writers.

NUMERICAL CONTRACT: big-endian float64 default; atomic .tmp+rename;
.meta companion in MITgcm Fortran namelist format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


def write_bin_atomic(
    array: np.ndarray,
    path: Path,
    *,
    dtype: Literal["float64", "float32"] = "float64",
    endianness: Literal["big", "little"] = "big",
) -> Path:
    """Atomically write a numpy array as MITgcm-style binary file.

    Algorithm:
      1. Ensure array dtype + endianness per arguments
      2. Write to `.tmp` file via numpy.tofile (Fortran-flat order: native C
         contiguous → MITgcm reads Fortran-flat after _RD_TILE_RX reshape)
      3. Rename `.tmp` → final

    Returns the final path.  The written file is not read back and no digest
    is computed.

    Invariants:
      - default dtype = float64, endianness = 'big' (matches MITgcm _RD_TILE_R8)
      - .tmp + rename pattern; partial crash leaves no garbage at final path
    """
    if dtype == "float64":
        np_dtype = np.dtype(">f8" if endianness == "big" else "<f8")
    elif dtype == "float32":
        np_dtype = np.dtype(">f4" if endianness == "big" else "<f4")
    else:
        raise ValueError(f"unsupported dtype {dtype!r}")

    # Ensure contiguous + correct dtype
    arr_to_write = np.ascontiguousarray(array.astype(np_dtype))

    tmp = path.parent / (path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy.tofile writes in C order (row-major); for MITgcm Fortran-flat,
    # caller is responsible for transposing before passing.
    arr_to_write.tofile(tmp)

    if path.exists():
        path.unlink()
    tmp.rename(path)
    return path


def write_meta_companion(
    path: Path,
    *,
    dim_list: list[tuple[int, int, int]],
    dataprec: Literal["float32", "float64"],
    nrecords: int,
    time_step_number: int = 0,
) -> Path:
    """Write MITgcm .meta companion file in canonical namelist format.

    Format follows the upstream MITgcmutils ``mds.py:parsemeta`` reader:

        nDims = [ <N> ];
        dimList = [
           <dim1_global_size>, <dim1_local_start_1based>, <dim1_local_end_1based>,
           ...
        ];
        dataprec = [ '<float32|float64>' ];
        nrecords = [ <N> ];
        timeStepNumber = [ <N> ];

    `dim_list` is a list of (global_size, local_start_1based, local_end_1based)
    triplets, one per dim. For single-tile / full-domain files: local_start=1,
    local_end=global_size.

    NOTE: this is the format real MITgcm meta files use; a
    (global_start, global_end, local_size) form is wrong.
    """
    lines: list[str] = []
    lines.append(f" nDims = [ {len(dim_list):>3d} ];")
    lines.append(" dimList = [")
    for i, (gsize, lstart, lend) in enumerate(dim_list):
        sep = "," if i < len(dim_list) - 1 else ""
        lines.append(f" {gsize:>5d}, {lstart:>4d}, {lend:>4d}{sep}")
    lines.append(" ];")
    lines.append(f" dataprec = [ '{dataprec}' ];")
    lines.append(f" nrecords = [ {nrecords:>10d} ];")
    lines.append(f" timeStepNumber = [ {time_step_number:>10d} ];")

    text = "\n".join(lines) + "\n"
    tmp = path.parent / (path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text)
    if path.exists():
        path.unlink()
    tmp.rename(path)
    return path
