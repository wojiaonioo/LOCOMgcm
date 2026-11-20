"""Faithful readers for MITgcm-native binaries and namelists (SCS_CLM case).

Scope
-----
This module does **exactly one thing**: read MIT-side bytes and hand back the
raw numbers MITgcm itself holds.  It performs **no** sign conversion, no unit
conversion, no axis reordering beyond the MDS record layout, and no masking.
Convention translation to LOCOMgcm (``w`` sign, ``rC`` vs ``center_depth``,
bathymetry sign, ...) is the job of the downstream adapter -- see docs/NUMERICS.md
for the authoritative convention table.

Byte order -- READ THIS FIRST
-----------------------------
MITgcm here is compiled with ``-convert big_endian`` (see
``references/SCS_CLM/build.sh`` / ``build_console.log``).  **Every** Fortran
unformatted write -- MDS ``.data`` output, ``pickup.*.data``, and the
hand-rolled ``input/*.bin`` forcing files alike -- is therefore **big-endian**.
numpy must read them as ``>f4`` / ``>f8``.

    KNOWN TRAP: if a read "succeeds" but the values come out as ~1e308,
    ~1e-38, or otherwise absurd garbage, suspect byte order *before* suspecting
    the model.  Runtime evidence for this case:
    ``>f4`` on ``STD/XC.data`` -> 99.0208.. 128.979 degE (correct);
    ``<f4`` on the same bytes  -> ~3.4e13 (garbage).

Record layout
-------------
MDS files are a flat sequence of records; each record is one horizontal slab
of ``ny * nx`` values in Fortran x-fastest order, which reshapes to C-order
``(ny, nx)`` directly.  A 3-D field is ``nz`` consecutive records (k=1 at the
surface, k increasing downward).  ``pickup`` files concatenate **all 3-D
fields first** (``nz`` records each) and **then all 2-D fields** (1 record
each) -- verified against ``references/MITgcm/model/src/write_pickup.F``
lines 244-362.

Precision is per-file, from the ``.meta`` companion: this case writes grid /
snapshot / diagnostic files as ``float32`` (``readBinaryPrec=32``) but the
pickup as ``float64`` (pickups are always ``_RL``).  Do not assume one
precision for the whole case.

Deliberately depends on numpy + stdlib only -- no torch, no defgcm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


__all__ = [
    "MetaInfo",
    "PickupReader",
    "dtype_of",
    "open_pickup",
    "parse_meta",
    "parse_namelist",
    "read_grid",
    "read_mds",
]


#: 2-D fields the main MITgcm pickup may contain, in write order.  Source:
#: ``references/MITgcm/model/src/write_pickup.F`` §"Write 2-D fields".  Used to
#: split a pickup's ``fldList`` into 3-D and 2-D groups so ``nz`` can be
#: recovered (pickup ``.meta`` declares ``nDims=2`` and never states ``nz``).
_PICKUP_2D_FIELDS: frozenset[str] = frozenset(
    {"EtaN", "Phi_rLow", "dPhiNH", "dEtaHdt", "EtaH"}
)

#: Horizontal grid descriptors in an MITgcm ``STD``/output directory, 2-D.
_GRID_2D_FIELDS: tuple[str, ...] = (
    "XC", "YC", "XG", "YG", "Depth", "RAC", "DXC", "DXG", "DYC", "DYG",
)
#: Vertical grid descriptors, 1-D (``RF``/``DRC`` have ``nz+1`` entries).
_GRID_1D_FIELDS: tuple[str, ...] = ("RC", "RF", "DRF", "DRC")
#: Partial-cell fraction fields, 3-D and large (262 MB each at 720x684x133).
_GRID_3D_FIELDS: tuple[str, ...] = ("hFacC", "hFacW", "hFacS")

_DATAPREC_TO_DTYPE: Mapping[str, str] = {"float32": ">f4", "float64": ">f8"}


# --------------------------------------------------------------------------
# .meta parsing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaInfo:
    """Parsed MITgcm ``.meta`` companion file.

    All values are MIT-side as written; nothing is converted.

    Attributes:
      path: the ``.meta`` file this came from.
      n_dims: ``nDims`` -- 2 for horizontal slabs, 3 for volumes and for the
        degenerate ``1 x 1 x nz`` vertical descriptors (``RC``, ``DRF``, ...).
      dim_list: one ``(global_size, local_start_1based, local_end_1based)``
        triple per dimension, in MIT order (x, y, z).
      dataprec: ``'float32'`` or ``'float64'``.  This is the *file* precision;
        big-endianness is implicit (see module docstring).
      nrecords: number of ``ny x nx`` records in the file.
      time_step_number: ``timeStepNumber`` (iteration index), ``None`` if the
        file is time-invariant (grid descriptors omit the field).
      time_interval: ``timeInterval`` values in seconds; MITgcm writes one
        number for a snapshot and two (window start/end) for a time average.
        ``None`` when absent.
      fld_list: ``fldList`` entries with the fixed-width blank padding
        stripped, in file order.  ``None`` when absent (single-field files).
      n_flds: ``nFlds``.  ``None`` when absent.
      missing_value: ``missingValue`` (diagnostics files), ``None`` when absent.
    """

    path: Path
    n_dims: int
    dim_list: tuple[tuple[int, int, int], ...]
    dataprec: str
    nrecords: int
    time_step_number: int | None = None
    time_interval: tuple[float, ...] | None = None
    fld_list: tuple[str, ...] | None = None
    n_flds: int | None = None
    missing_value: float | None = None

    @property
    def nx(self) -> int:
        """Global x (longitude) size."""
        return self.dim_list[0][0]

    @property
    def ny(self) -> int:
        """Global y (latitude) size."""
        return self.dim_list[1][0]

    @property
    def nz(self) -> int | None:
        """Global z size, or ``None`` for a 2-D (``nDims=2``) file.

        Note ``nz`` is the *declared* third dimension: for the vertical
        descriptors (``RC``, ``RF``, ``DRF``, ``DRC``) MITgcm writes
        ``nDims=3`` with ``nx=ny=1``, so ``nz`` is the array length there
        (``nz+1`` for the interface-based ``RF``/``DRC``).
        """
        return self.dim_list[2][0] if self.n_dims >= 3 else None

    @property
    def dims(self) -> tuple[int, int, int | None]:
        """``(nx, ny, nz)``; ``nz is None`` for a 2-D file."""
        return (self.nx, self.ny, self.nz)

    @property
    def dtype(self) -> np.dtype:
        """numpy dtype to read this file with -- always big-endian."""
        return dtype_of(self.dataprec)

    @property
    def record_shape(self) -> tuple[int, ...]:
        """Shape of ONE record, C-order.

        ``(nz, ny, nx)`` for a volume, ``(ny, nx)`` for a horizontal slab, and
        ``(nz,)`` for the degenerate ``1 x 1 x nz`` vertical descriptors.
        """
        if self.n_dims >= 3:
            nz = self.nz
            assert nz is not None  # n_dims >= 3 guarantees a third triple
            if self.nx == 1 and self.ny == 1:
                return (nz,)
            return (nz, self.ny, self.nx)
        return (self.ny, self.nx)

    @property
    def full_shape(self) -> tuple[int, ...]:
        """Shape of the whole file; a leading record axis iff ``nrecords > 1``."""
        if self.nrecords > 1:
            return (self.nrecords, *self.record_shape)
        return self.record_shape


def _meta_block(text: str, name: str) -> str | None:
    """Return the raw text inside ``name = [ ... ];``, or ``None`` if absent."""
    match = re.search(rf"\b{name}\s*=\s*\[(.*?)\]\s*;", text, flags=re.S)
    return match.group(1) if match else None


def parse_meta(path: str | Path) -> MetaInfo:
    """Parse an MITgcm ``.meta`` companion file.

    Self-contained by design: this module must not import the defgcm
    trunk's MDS parser (``defgcm/io/mds_v2/_meta_parser.py`` is a useful
    reference but the case-private layer stays standalone).

    Conventions: the parsed numbers are MIT-side verbatim.  ``dataprec`` names
    the file's precision only -- the byte order is always big-endian and is
    supplied by :func:`dtype_of`, never by the ``.meta``.

    Args:
      path: path to the ``.meta`` file.

    Returns:
      A :class:`MetaInfo`.

    Raises:
      ValueError: on a malformed or unsupported ``.meta``.
    """
    meta_path = Path(path)
    text = meta_path.read_text()

    n_dims_raw = _meta_block(text, "nDims")
    if n_dims_raw is None:
        raise ValueError(f"{meta_path}: missing required field 'nDims'")
    try:
        n_dims = int(n_dims_raw.strip())
    except ValueError as exc:
        raise ValueError(f"{meta_path}: nDims not parseable: {n_dims_raw!r}") from exc

    dim_raw = _meta_block(text, "dimList")
    if dim_raw is None:
        raise ValueError(f"{meta_path}: missing required field 'dimList'")
    ints = [int(tok) for tok in re.findall(r"-?\d+", dim_raw)]
    if len(ints) != 3 * n_dims:
        raise ValueError(
            f"{meta_path}: dimList holds {len(ints)} integers; expected "
            f"{3 * n_dims} (3 per dim x nDims={n_dims})"
        )
    dim_list = tuple(
        (ints[3 * i], ints[3 * i + 1], ints[3 * i + 2]) for i in range(n_dims)
    )

    prec_raw = _meta_block(text, "dataprec")
    if prec_raw is None:
        raise ValueError(f"{meta_path}: missing required field 'dataprec'")
    prec_match = re.search(r"'([^']+)'", prec_raw)
    if prec_match is None:
        raise ValueError(f"{meta_path}: dataprec has no quoted string: {prec_raw!r}")
    dataprec = prec_match.group(1).strip()
    if dataprec not in _DATAPREC_TO_DTYPE:
        raise ValueError(
            f"{meta_path}: unsupported dataprec {dataprec!r} "
            f"(expected one of {sorted(_DATAPREC_TO_DTYPE)})"
        )

    nrec_raw = _meta_block(text, "nrecords")
    if nrec_raw is None:
        raise ValueError(f"{meta_path}: missing required field 'nrecords'")
    nrecords = int(re.findall(r"-?\d+", nrec_raw)[0])

    def _opt_int(name: str) -> int | None:
        raw = _meta_block(text, name)
        if raw is None:
            return None
        found = re.findall(r"-?\d+", raw)
        return int(found[0]) if found else None

    def _opt_floats(name: str) -> tuple[float, ...] | None:
        raw = _meta_block(text, name)
        if raw is None:
            return None
        found = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eEdD][-+]?[0-9]+)?", raw)
        if not found:
            return None
        return tuple(float(tok.replace("D", "E").replace("d", "e")) for tok in found)

    time_interval = _opt_floats("timeInterval")
    missing_raw = _opt_floats("missingValue")
    missing_value = missing_raw[0] if missing_raw else None

    fld_list: tuple[str, ...] | None = None
    fld_match = re.search(r"\bfldList\s*=\s*\{(.*?)\}\s*;", text, flags=re.S)
    if fld_match is not None:
        fld_list = tuple(
            name.strip() for name in re.findall(r"'([^']*)'", fld_match.group(1))
        )

    n_flds = _opt_int("nFlds")
    if fld_list is not None and n_flds is not None and len(fld_list) != n_flds:
        raise ValueError(
            f"{meta_path}: nFlds={n_flds} but fldList holds {len(fld_list)} names"
        )

    return MetaInfo(
        path=meta_path,
        n_dims=n_dims,
        dim_list=dim_list,
        dataprec=dataprec,
        nrecords=nrecords,
        time_step_number=_opt_int("timeStepNumber"),
        time_interval=time_interval,
        fld_list=fld_list,
        n_flds=n_flds,
        missing_value=missing_value,
    )


def dtype_of(dataprec: str) -> np.dtype:
    """Map an MITgcm ``dataprec`` string to the numpy dtype that reads it.

    Convention: **always big-endian**.  This case's MITgcm is built with
    ``-convert big_endian``, so ``'float32' -> '>f4'`` and
    ``'float64' -> '>f8'``.  There is deliberately no little-endian branch --
    a native-order read of these files yields ~1e308 garbage, not an error.

    Args:
      dataprec: ``'float32'`` or ``'float64'`` (as written in a ``.meta``).

    Returns:
      ``np.dtype('>f4')`` or ``np.dtype('>f8')``.

    Raises:
      ValueError: for any other precision string.
    """
    try:
        return np.dtype(_DATAPREC_TO_DTYPE[dataprec])
    except KeyError:
        raise ValueError(
            f"unsupported dataprec {dataprec!r} "
            f"(expected one of {sorted(_DATAPREC_TO_DTYPE)})"
        ) from None


# --------------------------------------------------------------------------
# MDS .data reading
# --------------------------------------------------------------------------


def read_mds(
    data_path: str | Path,
    meta_path: str | Path | None = None,
    shape: Sequence[int] | None = None,
    dataprec: str | None = None,
) -> np.ndarray:
    """Read one MITgcm MDS ``.data`` file in full.

    Data conventions:
      * Byte order is big-endian on disk (module docstring); the returned
        array is byte-swapped to **native** order so downstream numpy /
        torch code behaves, but the *values* are bit-identical to MIT's.
      * Layout is C-order ``(nz, ny, nx)`` for a volume, ``(ny, nx)`` for a
        horizontal slab, ``(nz,)`` for a vertical descriptor, with a leading
        record axis prepended iff ``nrecords > 1``.
      * k=0 is the surface and k increases downward (MIT k=1 surface, minus
        the 1-based offset).  No sign or unit conversion is applied -- e.g.
        ``RC`` comes back negative (MIT r-coordinate, upward-positive) and
        ``Depth`` comes back positive.

    Resolution order for shape/dtype -- **explicit arguments win**: whatever
    you pass in ``shape`` / ``dataprec`` overrides the ``.meta``.  Anything you
    leave as ``None`` is filled from ``meta_path``, or from the sibling
    ``<stem>.meta`` when ``meta_path`` is ``None`` and that file exists.

    Args:
      data_path: the ``.data`` (or ``.bin``) file.
      meta_path: explicit ``.meta``; ``None`` auto-discovers the sibling and
        silently tolerates its absence.
      shape: element shape override, e.g. ``(133, 684, 720)``.
      dataprec: precision override, ``'float32'`` or ``'float64'``.

    Returns:
      A native-byte-order ``np.ndarray`` of the resolved shape.

    Raises:
      ValueError: if neither a ``.meta`` nor explicit ``shape``/``dataprec``
        is available, or if the resolved element count does not match the file
        size exactly.  A size mismatch is almost always a wrong ``dataprec``
        (4 vs 8 bytes) or a wrong ``shape`` -- and if the size *does* match but
        the numbers look like ~1e308, suspect byte order.
    """
    data_file = Path(data_path)
    if not data_file.is_file():
        raise ValueError(f"{data_file}: no such MDS data file")

    meta: MetaInfo | None = None
    if meta_path is not None:
        meta = parse_meta(meta_path)
    else:
        sibling = data_file.with_suffix(".meta")
        if sibling.is_file():
            meta = parse_meta(sibling)

    resolved_prec = dataprec if dataprec is not None else (meta.dataprec if meta else None)
    if resolved_prec is None:
        raise ValueError(
            f"{data_file}: no .meta found and no explicit dataprec given; "
            "pass dataprec='float32'/'float64'"
        )
    dtype = dtype_of(resolved_prec)

    if shape is not None:
        resolved_shape: tuple[int, ...] = tuple(int(n) for n in shape)
    elif meta is not None:
        resolved_shape = meta.full_shape
    else:
        raise ValueError(
            f"{data_file}: no .meta found and no explicit shape given; "
            "pass shape=(nz, ny, nx) or (ny, nx) or (nz,)"
        )

    n_expected = int(np.prod(resolved_shape)) if resolved_shape else 1
    n_bytes = data_file.stat().st_size
    bytes_expected = n_expected * dtype.itemsize
    if n_bytes != bytes_expected:
        raise ValueError(
            f"{data_file}: size mismatch -- file is {n_bytes} bytes but "
            f"shape={resolved_shape} x dataprec={resolved_prec} "
            f"({dtype.itemsize} B/elem) needs {bytes_expected} bytes "
            f"({n_expected} elements). Check dataprec (float32 vs float64) and "
            "shape; if both look right, the file may not be an MDS record "
            "sequence. (A wrong *byte order* does NOT change the size -- it "
            "shows up as ~1e308 garbage values instead.)"
        )

    raw = np.fromfile(data_file, dtype=dtype)
    # .astype to the same-width native dtype performs the byte swap in one
    # pass; values are preserved bit-for-bit.
    return raw.reshape(resolved_shape).astype(dtype.newbyteorder("="))


# --------------------------------------------------------------------------
# pickup
# --------------------------------------------------------------------------


def _infer_pickup_layout(
    fld_list: Sequence[str],
    nrecords: int,
    nz: int | None,
) -> tuple[dict[str, tuple[int, int]], int]:
    """Split a pickup ``fldList`` into records; recover ``nz`` if not given.

    MITgcm writes all 3-D fields first (``nz`` records each) then all 2-D
    fields (1 record each) -- ``write_pickup.F`` lines 244-362.  The pickup
    ``.meta`` declares ``nDims=2`` and never states ``nz``, so ``nz`` is
    recovered from the record budget once the trailing 2-D fields are
    identified by name against :data:`_PICKUP_2D_FIELDS`.

    Returns:
      ``(fields, nz)`` where ``fields`` maps name -> ``(start_rec, n_levels)``.
    """
    n_2d = 0
    for name in reversed(list(fld_list)):
        if name in _PICKUP_2D_FIELDS:
            n_2d += 1
        else:
            break
    n_3d = len(fld_list) - n_2d

    if nz is None:
        if n_3d == 0:
            nz = 1
        else:
            remainder = nrecords - n_2d
            if remainder <= 0 or remainder % n_3d != 0:
                raise ValueError(
                    f"cannot infer nz: nrecords={nrecords}, {n_3d} 3-D fields, "
                    f"{n_2d} 2-D fields -> ({nrecords} - {n_2d}) is not a "
                    f"positive multiple of {n_3d}. Pass nz explicitly."
                )
            nz = remainder // n_3d

    expected = n_3d * nz + n_2d
    if expected != nrecords:
        raise ValueError(
            f"pickup record budget mismatch: {n_3d} 3-D fields x nz={nz} + "
            f"{n_2d} 2-D fields = {expected} records, but .meta says "
            f"nrecords={nrecords}"
        )

    fields: dict[str, tuple[int, int]] = {}
    cursor = 0
    for index, name in enumerate(fld_list):
        n_levels = 1 if index >= n_3d else nz
        fields[name] = (cursor, n_levels)
        cursor += n_levels
    return fields, nz


class PickupReader:
    """Lazy reader over an MITgcm ``pickup.*.data`` file.

   The SCS_CLM pickup is 3.7 GB, so the file is opened with ``np.memmap`` and
    only the requested field is ever materialised (a 133-level field is
    ~524 MB as float64).

    Data conventions:
      * On-disk dtype is big-endian ``float64`` (pickups are always ``_RL``,
        independent of ``writeBinaryPrec``).  :meth:`read` returns
        **native**-byte-order float64 with bit-identical values.
      * Shapes are C-order ``(nz, ny, nx)`` for 3-D fields and ``(ny, nx)`` for
        2-D fields; k=0 is the surface, k increasing downward.
      * Values are MIT-side verbatim.  In particular ``Uvel``/``Vvel`` are on
        the MIT C-grid west/south faces, ``EtaN`` is MIT's free surface, and no
        LOCOMgcm sign convention (docs/NUMERICS.md) has been applied.  Land cells
        hold whatever MITgcm left there -- mask with ``hFacC``/``hFacW``/
        ``hFacS``, not by value.
    """

    def __init__(self, path: Path, meta: MetaInfo, nz: int | None = None) -> None:
        if meta.fld_list is None:
            raise ValueError(
                f"{meta.path}: pickup .meta has no fldList; cannot split records"
            )
        self.path: Path = path
        self.meta: MetaInfo = meta
        self.ny: int = meta.ny
        self.nx: int = meta.nx
        self.fields: dict[str, tuple[int, int]]
        self.fields, self.nz = _infer_pickup_layout(meta.fld_list, meta.nrecords, nz)

        dtype = meta.dtype
        expected_bytes = meta.nrecords * meta.ny * meta.nx * dtype.itemsize
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{path}: size mismatch -- file is {actual_bytes} bytes but "
                f"nrecords={meta.nrecords} x ny={meta.ny} x nx={meta.nx} x "
                f"{dtype.itemsize} B = {expected_bytes} bytes. Check the "
                "pickup dataprec (pickups are float64 even when "
                "writeBinaryPrec=32)."
            )
        self._mm = np.memmap(
            path, dtype=dtype, mode="r", shape=(meta.nrecords, meta.ny, meta.nx)
        )

    @property
    def time_step(self) -> int | None:
        """``timeStepNumber`` (iteration index) from the ``.meta``."""
        return self.meta.time_step_number

    @property
    def time_interval(self) -> tuple[float, ...] | None:
        """``timeInterval`` in seconds from the ``.meta`` (model time)."""
        return self.meta.time_interval

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names in file order."""
        return tuple(self.fields)

    def read(self, name: str) -> np.ndarray:
        """Materialise one pickup field.

        Args:
          name: field name as it appears in ``fldList`` (already stripped of
            MITgcm's fixed-width blank padding), e.g. ``'Theta'``, ``'EtaN'``.

        Returns:
          Native-byte-order float64 array, ``(nz, ny, nx)`` for a 3-D field or
          ``(ny, nx)`` for a 2-D field.  Values are MIT-side verbatim -- no
          sign flip, no masking (see class docstring).

        Raises:
          KeyError: if ``name`` is not in this pickup.
        """
        if name not in self.fields:
            raise KeyError(
                f"{self.path}: no field {name!r}; available: "
                f"{', '.join(self.fields)}"
            )
        start, n_levels = self.fields[name]
        block = self._mm[start : start + n_levels]
        native = self.meta.dtype.newbyteorder("=")
        arr = np.asarray(block).astype(native)
        return arr[0] if n_levels == 1 else arr

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __repr__(self) -> str:
        return (
            f"PickupReader({self.path.name}, nz={self.nz}, ny={self.ny}, "
            f"nx={self.nx}, step={self.time_step}, "
            f"fields={len(self.fields)})"
        )


def open_pickup(
    path: str | Path,
    meta_path: str | Path | None = None,
    nz: int | None = None,
) -> PickupReader:
    """Open an MITgcm ``pickup.*.data`` lazily.

    Nothing is read into memory here: the file is memory-mapped and fields are
    materialised one at a time by :meth:`PickupReader.read`.

    Data conventions: see :class:`PickupReader` -- big-endian float64 on disk,
    native float64 out, ``(nz, ny, nx)`` C-order, k=0 at the surface, MIT-side
    values with no convention translation.

    Args:
      path: the ``pickup.<iter>.data`` file.
      meta_path: explicit ``.meta``; ``None`` auto-discovers the sibling
        ``pickup.<iter>.meta``, which is **required** (its ``fldList`` is the
        only record map).
      nz: override the vertical size.  By default ``nz`` is recovered from the
        record budget: pickup ``.meta`` declares ``nDims=2`` and never states
        ``nz``, so the trailing 2-D fields are identified by name and
        ``nz = (nrecords - n_2d) / n_3d``.

    Returns:
      A :class:`PickupReader`.

    Raises:
      ValueError: if the ``.meta`` is missing, has no ``fldList``, or the
        record budget is inconsistent with the file size.
    """
    data_file = Path(path)
    if not data_file.is_file():
        raise ValueError(f"{data_file}: no such pickup file")
    if meta_path is None:
        candidate = data_file.with_suffix(".meta")
        if not candidate.is_file():
            raise ValueError(
                f"{data_file}: no sibling .meta ({candidate.name}); a pickup "
                "cannot be decoded without its fldList"
            )
        meta_path = candidate
    return PickupReader(data_file, parse_meta(meta_path), nz=nz)


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------


def read_grid(std_dir: str | Path, lazy_3d: bool = True) -> dict[str, np.ndarray]:
    """Read the MITgcm grid descriptors written to an output directory.

    Reads, when present: ``XC YC XG YG`` (cell-centre / cell-corner
    longitude-latitude), ``RC RF DRF DRC`` (vertical), ``Depth``, ``hFacC
    hFacW hFacS`` (partial-cell fractions), ``RAC``, and ``DXC DXG DYC DYG``
    (metric distances).  Missing files are skipped rather than raising, since
    which descriptors MITgcm dumps depends on the build.

    Data conventions -- **MIT-side, untranslated**:
      * ``XC``/``YC``/``XG``/``YG``: degrees east / north, shape ``(ny, nx)``.
      * ``RC``/``RF``: r-coordinate, **upward-positive**, hence *negative*
        below the surface (``RC[0] ~ -2.5 m``).  LOCOMgcm's ``center_depth``
        / ``interface_depth`` are downward-positive -- convert with
        ``z = -depth`` at the consumer (docs/NUMERICS.md).  ``RF``/``DRC`` carry
        ``nz+1`` entries.
      * ``DRF``: layer thickness, positive, shape ``(nz,)`` -- same sign as
        LOCOMgcm, no conversion.
      * ``Depth``: MITgcm *runtime* ocean depth, **positive** metres (matching
        LOCOMgcm).  Note this is NOT the sign convention of the input file
        ``bathyFile`` (``SCS_Dep.bin``), which is negative below sea level.
      * ``hFacC``/``hFacW``/``hFacS``: dimensionless open fraction in
        ``[0, 1]``, shape ``(nz, ny, nx)``, ``0`` on land.  These are the
        authoritative wet mask -- do not infer wet cells from field values.
      * ``RAC``: cell area, m^2.  ``DXC``/``DXG``/``DYC``/``DYG``: metres.

    Args:
      std_dir: directory holding the ``*.data``/``*.meta`` pairs (e.g.
        ``references/SCS_CLM/STD``).
      lazy_3d: when ``True`` (default) the three ``hFac`` volumes are returned
        as read-only **big-endian** ``np.memmap`` views instead of being read
        into memory -- 262 MB each at 720x684x133.  Comparisons and slicing
        work unchanged; call ``np.asarray(x, dtype=np.float64)`` to
        materialise in native order.  When ``False`` they are read eagerly in
        native order like everything else.

    Returns:
      ``dict`` mapping field name to array.  Only fields whose ``.data`` file
      exists are present.

    Raises:
      ValueError: if ``std_dir`` is not a directory, or a present file fails
        its size check.
    """
    root = Path(std_dir)
    if not root.is_dir():
        raise ValueError(f"{root}: not a directory")

    out: dict[str, np.ndarray] = {}
    for name in (*_GRID_2D_FIELDS, *_GRID_1D_FIELDS):
        data_file = root / f"{name}.data"
        if data_file.is_file():
            out[name] = read_mds(data_file)

    for name in _GRID_3D_FIELDS:
        data_file = root / f"{name}.data"
        if not data_file.is_file():
            continue
        if not lazy_3d:
            out[name] = read_mds(data_file)
            continue
        meta = parse_meta(root / f"{name}.meta")
        shape = meta.full_shape
        expected = int(np.prod(shape)) * meta.dtype.itemsize
        actual = data_file.stat().st_size
        if actual != expected:
            raise ValueError(
                f"{data_file}: size mismatch -- file is {actual} bytes but "
                f"shape={shape} x dataprec={meta.dataprec} needs {expected}"
            )
        # Big-endian memmap: values are correct, dtype byte order is '>'.
        out[name] = np.memmap(data_file, dtype=meta.dtype, mode="r", shape=shape)

    return out


# --------------------------------------------------------------------------
# Fortran namelist
# --------------------------------------------------------------------------

_GROUP_START_RE = re.compile(r"^[&$](\w+)\s*$")
_GROUP_END_TOKENS = frozenset({"&", "&end", "/", "$", "$end"})
_ASSIGN_RE = re.compile(r"([A-Za-z_]\w*)\s*(\([^)]*\))?\s*=")
_INT_RE = re.compile(r"^[-+]?\d+$")
_FLOAT_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?$")
_REPEAT_RE = re.compile(r"^(\d+)\*(.*)$", flags=re.S)
_TRUE_TOKENS = frozenset({".true.", ".t.", "t", "true"})
_FALSE_TOKENS = frozenset({".false.", ".f.", "f", "false"})


def _is_comment(line: str) -> bool:
    """Whole-line comment test for MITgcm namelists.

    MITgcm's ``data`` parser rejects trailing inline comments (a ``#`` after a
    value triggers ``forrtl-17``), so only whole-line comments exist:
      * first non-blank char is ``#`` or ``!``;
      * fixed-form Fortran ``C`` comment -- an uppercase ``C`` **not** followed
        by an identifier character, so ``C--  note`` is a comment while
        ``Cd = 1.0`` is an assignment (and lowercase ``cg2dMaxIters`` is never
        mistaken for one).
    """
    stripped = line.lstrip()
    if not stripped:
        return True
    head = stripped[0]
    if head in "#!":
        return True
    if head == "C":
        return len(stripped) == 1 or not (stripped[1].isalnum() or stripped[1] == "_")
    return False


def _mask_quotes(text: str) -> str:
    """Return ``text`` with single-quoted spans blanked out, length preserved.

    Used so that assignment / comma scanning never trips on characters inside
    a Fortran string literal.  Doubled quotes (``''``) escape a quote.
    """
    out = list(text)
    in_quote = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'":
            if in_quote and i + 1 < n and text[i + 1] == "'":
                out[i] = out[i + 1] = "x"
                i += 2
                continue
            in_quote = not in_quote
        elif in_quote and ch != "\n":
            out[i] = "x"
        i += 1
    return "".join(out)


def _split_values(text: str) -> list[str]:
    """Split a namelist value blob on top-level commas (quote-aware)."""
    masked = _mask_quotes(text)
    tokens: list[str] = []
    start = 0
    for index, ch in enumerate(masked):
        if ch == ",":
            tokens.append(text[start:index])
            start = index + 1
    tokens.append(text[start:])
    return [tok.strip() for tok in tokens if tok.strip()]


def _parse_scalar(token: str) -> Any:
    """Convert one namelist token to a Python value.

    Handles quoted strings, Fortran logicals (``.TRUE.`` / ``.T.``), integers,
    and Fortran reals including the trailing-point (``43200.``), exponent
    (``1.E-4``), and double-precision (``1.0d0``) spellings.  Anything else is
    returned as the raw stripped string rather than guessed at.
    """
    token = token.strip()
    if not token:
        return ""
    if token.startswith("'") and token.endswith("'") and len(token) >= 2:
        return token[1:-1].replace("''", "'").strip()
    lowered = token.lower()
    if lowered in _TRUE_TOKENS:
        return True
    if lowered in _FALSE_TOKENS:
        return False
    if _INT_RE.match(token):
        return int(token)
    if _FLOAT_RE.match(token):
        return float(token.replace("D", "E").replace("d", "e"))
    return token


def _parse_value(text: str) -> Any:
    """Parse a namelist right-hand side into a scalar or a list.

    Expands MITgcm's repeat syntax ``N*V`` (e.g. ``562*0`` -> 562 zeros,
    ``158*-1`` -> 158 minus-ones).  Returns a bare scalar when the RHS holds
    exactly one value, a ``list`` otherwise.
    """
    values: list[Any] = []
    for token in _split_values(text):
        repeat = _REPEAT_RE.match(token)
        if repeat is not None and not token.startswith("'"):
            count = int(repeat.group(1))
            values.extend([_parse_scalar(repeat.group(2))] * count)
        else:
            values.append(_parse_scalar(token))
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def parse_namelist(path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse an MITgcm Fortran namelist (``data``, ``data.obcs``, ``data.exf``).

    Handles the dialect MITgcm actually emits:
      * groups open with ``&NAME`` (or ``$NAME``) on its own line and close
        with a lone ``&``, ``&end``, ``/`` or ``$end``;
      * several ``key = value`` pairs may share one line (``data.exf``:
        ``ustressperiod = 0.0,   ustressfile = 'exf_ustress.bin',``);
      * one value list may span many lines (``PARM01.tRef`` is 133 numbers
        over 23 lines);
      * the repeat syntax ``N*V`` is expanded, so ``OB_Jnorth = 562*0, 158*-1``
        yields 720 integers -- the per-column boundary index array;
      * whole-line comments start with ``#``, ``!``, or fixed-form ``C``;
        MITgcm does not accept trailing inline comments and neither does this
        parser;
      * ``'quoted strings'`` become ``str``, ``.TRUE.``/``.FALSE.`` become
        ``bool``, and Fortran reals (``1.E-4``, ``43200.``, ``1.0d0``) become
        ``float``.

    No unit or convention translation is applied: values are exactly what
    MITgcm reads.  Note in particular that ``PARM05.bathyFile`` names a file
    whose ocean depths are **negative** while runtime ``Depth`` is positive
    (docs/NUMERICS.md).

    Args:
      path: the namelist file.

    Returns:
      ``{group_name: {key: value}}``, group and key spelling preserved as
      written.  A value is a scalar when the right-hand side holds one item and
      a ``list`` when it holds several.  An array-indexed assignment
      (``key(1,2) = ...``) is keyed by its verbatim spelling with whitespace
      removed, so nothing is silently overwritten.

    Raises:
      ValueError: on an unterminated group.
    """
    text = Path(path).read_text(errors="replace")

    groups: dict[str, dict[str, Any]] = {}
    current: str | None = None
    body: list[str] = []

    def _flush(group: str, lines: list[str]) -> None:
        blob = "\n".join(lines)
        masked = _mask_quotes(blob)
        matches = list(_ASSIGN_RE.finditer(masked))
        entries: dict[str, Any] = groups.setdefault(group, {})
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(blob)
            key = match.group(1)
            if match.group(2):
                key += re.sub(r"\s+", "", match.group(2))
            entries[key] = _parse_value(blob[match.end() : end])

    for raw_line in text.splitlines():
        if _is_comment(raw_line):
            continue
        stripped = raw_line.strip()
        if current is None:
            start = _GROUP_START_RE.match(stripped)
            if start is not None:
                current = start.group(1)
                body = []
            continue
        if stripped in _GROUP_END_TOKENS:
            _flush(current, body)
            current = None
            body = []
            continue
        body.append(raw_line)

    if current is not None:
        raise ValueError(
            f"{path}: namelist group &{current} is never closed "
            "(expected a lone '&', '&end' or '/')"
        )
    return groups
