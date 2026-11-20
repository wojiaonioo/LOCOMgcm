"""Read one MITgcm field at one step across all tiles, reassemble.

PROFILE S keystone: concat order + endianness + dtype + halo strip are
all silent-failure modes if wrong.

MITgcm binary file layout per tile (`.data`):
  - Fortran-order multi-dimensional array of (nrecords, nz, sNy, sNx)
  - dtype = float32 or float64 per .meta `dataprec`
  - Big-endian byte order (MITgcm convention; _RD_TILE_R8 reads big-endian)

After reading, the per-tile array has shape (nrecords, nz, sNy, sNx) in
C-order numpy. We need to:
  1. Read each tile (IPx, IPy)
  2. Reassemble into (nrecords, nz, ny_global, nx_global)
  3. Strip halo if requested (.meta usually doesn't include halo;
     output files have only interior cells per MITgcm convention)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._meta_parser import parse_meta
from ._tile_index import TileIndex, parse_filename


@dataclass(frozen=True)
class MdsField:
    """A reassembled global field."""

    name: str
    step: int
    array: np.ndarray  # shape (nrecords, nz, ny_global, nx_global)
    dataprec: str  # "float32" | "float64"


def _read_tile(
    data_path: Path,
    *,
    dataprec: str,
    nrecords: int,
    nz: int,
    s_ny: int,
    s_nx: int,
) -> np.ndarray:
    """Read one tile's .data file (big-endian) and reshape to C-order array.

    File layout: Fortran-flat (nrecords × nz × sNy × sNx) — but numpy.fromfile
    reads sequentially; we reshape with the canonical order (nrecords, nz,
    sNy, sNx) which produces C-order. This matches MITgcm output convention
    when read by MITgcmutils.mds.rdmds.
    """
    if dataprec == "float32":
        np_dtype = np.dtype(">f4")
    elif dataprec == "float64":
        np_dtype = np.dtype(">f8")
    else:
        raise ValueError(f"unsupported dataprec {dataprec!r}")

    raw = np.fromfile(data_path, dtype=np_dtype)
    expected = nrecords * nz * s_ny * s_nx
    if raw.size != expected:
        raise ValueError(
            f"{data_path}: expected {expected} values "
            f"(nrecords={nrecords}, nz={nz}, sNy={s_ny}, sNx={s_nx}), got {raw.size}"
        )

    arr = raw.reshape((nrecords, nz, s_ny, s_nx))
    # Convert to native byte order so downstream ops work
    return arr.astype(arr.dtype.newbyteorder("="))


def read_multi_tile_field(
    run_dir: Path,
    field_stem: str,
    step: int | None,
    tile_index: TileIndex,
) -> MdsField:
    """Read one field at one step across all tiles, reassemble into global array.

    Args:
      run_dir: MITgcm run directory containing tile files
      field_stem: variable name prefix (e.g. 'U', 'T', 'hFacC')
      step: time step number (10 = matches '0000000010' in filename);
            pass None for time-invariant fields (e.g. hFacC)
      tile_index: discovered TileIndex from discover_tile_index()

    Returns:
      MdsField with global array of shape (nrecords, nz, ny_global, nx_global)
    """
    # Find one sample tile to read .meta
    if step is None:
        sample_pattern = f"{field_stem}.001.001.meta"
    else:
        sample_pattern = f"{field_stem}.{step:010d}.001.001.meta"
    sample_metas = list(run_dir.glob(sample_pattern))
    if not sample_metas:
        # Try single-tile time-variant (no .001.001 suffix)
        if step is not None:
            sample_metas = list(run_dir.glob(f"{field_stem}.{step:010d}.meta"))
    if not sample_metas:
        raise ValueError(
            f"read_multi_tile_field: no .meta found for {field_stem} step={step} in {run_dir}"
        )

    sample_meta = parse_meta(sample_metas[0])
    dataprec = sample_meta["dataprec"]
    nrecords = sample_meta["nrecords"]
    nz = tile_index.nz
    s_nx = tile_index.s_nx
    s_ny = tile_index.s_ny

    # Allocate global array (nrecords, nz, ny_global, nx_global)
    np_dtype = np.dtype(">f4" if dataprec == "float32" else ">f8").newbyteorder("=")
    global_arr = np.empty(
        (nrecords, nz, tile_index.ny_global, tile_index.nx_global),
        dtype=np_dtype,
    )

    # Read each tile + place into global
    for ipy in range(1, tile_index.n_py + 1):
        for ipx in range(1, tile_index.n_px + 1):
            # Build tile path
            if tile_index.is_single_tile:
                if step is None:
                    data_path = run_dir / f"{field_stem}.001.001.data"
                    if not data_path.exists():
                        data_path = run_dir / f"{field_stem}.data"
                else:
                    data_path = run_dir / f"{field_stem}.{step:010d}.001.001.data"
                    if not data_path.exists():
                        data_path = run_dir / f"{field_stem}.{step:010d}.data"
            else:
                if step is None:
                    data_path = run_dir / f"{field_stem}.{ipx:03d}.{ipy:03d}.data"
                else:
                    data_path = run_dir / f"{field_stem}.{step:010d}.{ipx:03d}.{ipy:03d}.data"

            tile_arr = _read_tile(
                data_path,
                dataprec=dataprec,
                nrecords=nrecords,
                nz=nz,
                s_ny=s_ny,
                s_nx=s_nx,
            )

            # Place into global array at tile (ipy, ipx) location
            y_start = (ipy - 1) * s_ny
            y_end = y_start + s_ny
            x_start = (ipx - 1) * s_nx
            x_end = x_start + s_nx
            global_arr[:, :, y_start:y_end, x_start:x_end] = tile_arr

    return MdsField(
        name=field_stem,
        step=step or 0,
        array=global_arr,
        dataprec=dataprec,
    )
