"""Bitwise verification of two pipeline run dirs.

The YAML-driven pipeline uses this as a reproducible artifact equality
gate. Files are compared directly, byte against byte (no digests); the
generated regional pipeline uses MITgcm ``readBinaryPrec=64``. NetCDF
snapshots whose bytes differ are opened without CF decoding and compared
variable by variable on their stored bits.

Usage::

    python -m experiments.pipeline_v2.verify_bitwise BASELINE NEW

Exit codes: ``0`` all artifacts bitwise-identical; ``1`` mismatch (details to
stdout); ``2`` missing artifacts on either side; ``3`` invocation error.

Artifacts compared:

* ``mitgcm/run/*.data`` — raw MITgcm binary, byte for byte; a mismatch
  reports the offset of the first differing byte.
  Skipped silently when ``mitgcm/run/`` is absent on *both* sides
  (the external-MIT verification shape). Asymmetric absence is flagged.
* ``mitgcm/run/*.meta`` — text diff (timestamps stripped); same skip rule.
* ``nc_snapshots/day*.nc`` — raw bytes first; when they differ, both files
  must hold the same dimensions, the same data variables and coordinates,
  and for each variable the same dims, dtype and attributes (compared as
  strictly as the values, see ``_same_value``), and no element may differ in
  its stored bits (+0 and -0 differ; NaNs are equal only with the same bits;
  strings compare by value). The report names the first differing variable
  and index and counts the differing values. Global attributes are not
  compared, so a file that differs only there counts as matched.

Every file set must be the same on both sides.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_CHUNK_BYTES = 1 << 20


def _first_byte_difference(a: Path, b: Path) -> int | None:
    """Offset of the first byte where ``a`` and ``b`` differ, or ``None``.

    When one file is a strict prefix of the other, the offset is the length of
    the shorter file.
    """
    offset = 0
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            chunk_a = fa.read(_CHUNK_BYTES)
            chunk_b = fb.read(_CHUNK_BYTES)
            if chunk_a != chunk_b:
                common = min(len(chunk_a), len(chunk_b))
                unequal = np.flatnonzero(
                    np.frombuffer(chunk_a, np.uint8, common)
                    != np.frombuffer(chunk_b, np.uint8, common)
                )
                return offset + (int(unequal[0]) if unequal.size else common)
            if not chunk_a:
                return None
            offset += len(chunk_a)


def differing_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mask of the elements of same-shape, same-dtype ``a`` and ``b`` whose stored bits differ.

    Every bit counts: +0 and -0 differ, and two NaNs are equal only when their
    sign and payload are. Object arrays (variable-length strings) compare by value.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(f"bit comparison needs one shape and dtype: {a.shape} {a.dtype}, {b.shape} {b.dtype}")
    if a.dtype.hasobject:
        return np.asarray(a != b, dtype=bool).reshape(a.shape)
    width = a.dtype.itemsize
    raw_a = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape + (width,))
    raw_b = np.ascontiguousarray(b).view(np.uint8).reshape(b.shape + (width,))
    return (raw_a != raw_b).any(axis=-1)


def _first_index(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.unravel_index(int(np.flatnonzero(mask)[0]), mask.shape))


def format_pair(x, y) -> str:
    """``x vs y`` for two differing elements, with their bits when they print alike (NaN payloads)."""
    text = f"{x!r} vs {y!r}"
    if repr(x) == repr(y) and isinstance(x, np.generic) and x.dtype.kind in "fc":
        bits = [np.asarray(v).reshape(1).view(np.uint8)[::-1].tobytes().hex() for v in (x, y)]
        text += f", bits 0x{bits[0]} vs 0x{bits[1]}"
    return text


def _same_value(x, y) -> bool:
    """Attribute values ``x`` and ``y`` hold the same type, shape and bits.

    Policy: a variable's attributes are compared as strictly as its values.
    They are written by the same writer as the reference, and some of them
    (``_FillValue``, units, scale) say how the values are read, so the
    comparison does not decide which differences are harmless: float32 1.0
    differs from float64 1.0, NaN fill values compare by bits, and text such
    as units or long_name must match exactly.  A writer change that alters
    them is reviewed and signed as a new reference.
    """
    if x is None or y is None:
        return x is y
    a, b = np.asarray(x), np.asarray(y)
    return a.shape == b.shape and a.dtype == b.dtype and not differing_bits(a, b).any()


def _is_text(dtype: np.dtype) -> bool:
    """Unicode or object (variable-length string) data, whose in-memory width is not stored."""
    return dtype.kind in "OU"


def _enumerate(directory: Path, glob: str) -> list[Path]:
    return sorted(p for p in directory.glob(glob) if p.is_file())


def _only_lines(where: str, base_names: set[str], new_names: set[str]) -> list[str]:
    only_base = sorted(base_names - new_names)
    only_new = sorted(new_names - base_names)
    lines = []
    if only_base:
        lines.append(f"only in baseline: {where}{only_base[:5]}{'...' if len(only_base) > 5 else ''} (n={len(only_base)})")
    if only_new:
        lines.append(f"only in new: {where}{only_new[:5]}{'...' if len(only_new) > 5 else ''} (n={len(only_new)})")
    return lines


def _strip_meta_timestamps(text: str) -> str:
    """MITgcm .meta files include a timestamp; strip it for fair text-diff."""
    return re.sub(r"^\s*timeStepNumber.*$", "", text, flags=re.MULTILINE)


def _compare_mit_data_files(baseline: Path, new: Path) -> tuple[int, int, list[str]]:
    """Compare mitgcm/run/*.data between two run dirs.

    Returns ``(matched, mismatched, error_lines)``. If ``mitgcm/run/`` is
    absent on *both* sides (the external-MIT verification shape), the check is
    a silent no-op. Asymmetric absence is still flagged as an error.
    """
    base_dir = baseline / "mitgcm" / "run"
    new_dir = new / "mitgcm" / "run"
    base_present = base_dir.is_dir()
    new_present = new_dir.is_dir()
    if not base_present and not new_present:
        return 0, 0, []
    if base_present != new_present:
        which = "baseline" if base_present else "new"
        return 0, 0, [f"asymmetric mitgcm/run/: present only on {which}"]

    base_files = {p.name: p for p in _enumerate(base_dir, "*.data")}
    new_files = {p.name: p for p in _enumerate(new_dir, "*.data")}

    errors = _only_lines("", set(base_files), set(new_files))
    matched = 0
    mismatched = 0
    for name in sorted(set(base_files) & set(new_files)):
        offset = _first_byte_difference(base_files[name], new_files[name])
        if offset is None:
            matched += 1
        else:
            mismatched += 1
            errors.append(f"bytes differ: mitgcm/run/{name} first at byte offset {offset}")

    return matched, mismatched, errors


def _compare_mit_meta_files(baseline: Path, new: Path) -> tuple[int, int, list[str]]:
    base_dir = baseline / "mitgcm" / "run"
    new_dir = new / "mitgcm" / "run"
    base_present = base_dir.is_dir()
    new_present = new_dir.is_dir()
    if not base_present and not new_present:
        return 0, 0, []
    if base_present != new_present:
        return 0, 0, []  # already flagged by _compare_mit_data_files; don't double-report
    base_files = {p.name: p for p in _enumerate(base_dir, "*.meta")}
    new_files = {p.name: p for p in _enumerate(new_dir, "*.meta")}
    matched = 0
    mismatched = 0
    errors = _only_lines("", set(base_files), set(new_files))
    for name in sorted(set(base_files) & set(new_files)):
        a = _strip_meta_timestamps(base_files[name].read_text())
        b = _strip_meta_timestamps(new_files[name].read_text())
        if a == b:
            matched += 1
        else:
            mismatched += 1
            errors.append(f"meta mismatch (after timestamp strip): mitgcm/run/{name}")
    return matched, mismatched, errors


def _compare_nc_file(base: Path, new: Path) -> tuple[list[str], int, str | None]:
    """``(structure differences, differing values, first difference)`` of two NetCDF files.

    The files are read without CF decoding, so stored values, dtypes and
    encoding attributes (``_FillValue``, units, scale) are compared as written.
    String variables compare by value: their fixed width in memory follows the
    longest string read, not the file.  A variable that cannot be compared is
    a structure difference; the values counted in other variables are kept.
    """
    import xarray as xr

    structure: list[str] = []
    differing, first = 0, None
    with xr.open_dataset(base, decode_cf=False) as ds_a, xr.open_dataset(new, decode_cf=False) as ds_b:
        if dict(ds_a.sizes) != dict(ds_b.sizes):
            structure.append(f"dimensions {dict(ds_a.sizes)} vs {dict(ds_b.sizes)}")
        structure += _only_lines("variables ", set(ds_a.data_vars), set(ds_b.data_vars))
        structure += _only_lines("coordinates ", set(ds_a.coords), set(ds_b.coords))
        for name in sorted(set(ds_a.variables) & set(ds_b.variables)):
            va, vb = ds_a.variables[name], ds_b.variables[name]
            text = _is_text(va.dtype) and _is_text(vb.dtype)
            if va.dims != vb.dims or va.shape != vb.shape or (va.dtype != vb.dtype and not text):
                structure.append(f"{name}: {va.dims} {va.shape} {va.dtype} vs {vb.dims} {vb.shape} {vb.dtype}")
                continue
            try:
                attributes = [key for key in sorted(set(va.attrs) | set(vb.attrs))
                              if not _same_value(va.attrs.get(key), vb.attrs.get(key))]
                if attributes:
                    structure.append(f"{name}: attributes {attributes} differ")
                a, b = va.values, vb.values
                if text:
                    a, b = a.astype(object), b.astype(object)
                mask = differing_bits(a, b)
            except Exception as exc:  # noqa: BLE001 - reported as a difference of this variable
                structure.append(f"{name}: cannot compare ({type(exc).__name__}: {exc})")
                continue
            count = int(mask.sum())
            if count == 0:
                continue
            differing += count
            if first is None:
                index = _first_index(mask)
                first = f"{name} at index {index} ({format_pair(a[index], b[index])})"
                if np.issubdtype(a.dtype, np.number):
                    delta = np.abs(a[mask].astype(np.float64) - b[mask].astype(np.float64))
                    finite = delta[np.isfinite(delta)]
                    if finite.size:
                        first += f", max|Δ| over the variable's differing values {finite.max():.3e}"
    return structure, differing, first


def _compare_nc_snapshots(baseline: Path, new: Path) -> tuple[int, int, list[str], int]:
    """``(matched, mismatched, error_lines, differing values)`` of the two nc_snapshots/ sets."""
    base_dir = baseline / "nc_snapshots"
    new_dir = new / "nc_snapshots"
    if not base_dir.is_dir() or not new_dir.is_dir():
        return 0, 0, [f"missing nc_snapshots/ on baseline or new ({base_dir}, {new_dir})"], 0
    base_files = {p.name: p for p in _enumerate(base_dir, "day*.nc")}
    new_files = {p.name: p for p in _enumerate(new_dir, "day*.nc")}
    errors = _only_lines("nc_snapshots/", set(base_files), set(new_files))
    matched = 0
    mismatched = 0
    total = 0
    for name in sorted(set(base_files) & set(new_files)):
        if _first_byte_difference(base_files[name], new_files[name]) is None:
            matched += 1
            continue
        try:
            structure, differing, first = _compare_nc_file(base_files[name], new_files[name])
        except Exception as exc:  # pragma: no cover - an unreadable file is a mismatch
            structure, differing, first = [f"cannot compare ({type(exc).__name__}: {exc})"], 0, None
        total += differing
        if not structure and differing == 0:
            matched += 1
            continue
        mismatched += 1
        parts = structure + ([f"{differing} differing values, first {first}"] if differing else [])
        errors.append(f"nc_snapshots/{name} differs: {'; '.join(parts)}")
    return matched, mismatched, errors, total


def compare_run_dirs(baseline: Path, new: Path) -> dict:
    """Compare every artifact of two run dirs.

    Returns ``status`` (the exit code of :func:`main`), ``differing_values``
    (NetCDF elements whose stored bits differ) and ``lines`` (the report).
    """
    if not baseline.is_dir() or not new.is_dir():
        which = baseline if not baseline.is_dir() else new
        return {"status": 2, "differing_values": 0, "lines": [f"FAIL: {which} not a directory"]}

    mit_data_ok, mit_data_bad, mit_data_err = _compare_mit_data_files(baseline, new)
    mit_meta_ok, mit_meta_bad, mit_meta_err = _compare_mit_meta_files(baseline, new)
    nc_ok, nc_bad, nc_err, differing = _compare_nc_snapshots(baseline, new)

    lines = []
    mit_absent_both = (
        not (baseline / "mitgcm" / "run").is_dir()
        and not (new / "mitgcm" / "run").is_dir()
    )
    if mit_absent_both:
        lines.append("MIT *.data : (skipped — mitgcm/run/ absent on both sides)")
        lines.append("MIT *.meta : (skipped — mitgcm/run/ absent on both sides)")
    else:
        lines.append(f"MIT *.data : {mit_data_ok} matched, {mit_data_bad} mismatched")
        lines.append(f"MIT *.meta : {mit_meta_ok} matched, {mit_meta_bad} mismatched")
    lines.append(f"nc_snapshots/day*.nc : {nc_ok} matched, {nc_bad} mismatched, "
                 f"{differing} differing values")
    all_err = mit_data_err + mit_meta_err + nc_err
    if all_err:
        lines.append("ERRORS:")
        lines += [f"  - {line}" for line in all_err]
        status = 1
    elif mit_data_ok + nc_ok == 0:
        lines.append("FAIL: nothing compared (empty artifact sets)")
        status = 2
    else:
        lines.append("RESULT: bitwise-identical ✓")
        status = 0
    return {"status": status, "differing_values": differing, "lines": lines}


def verify_run_dirs(baseline: Path, new: Path) -> int:
    print(f"baseline: {baseline}")
    print(f"new:      {new}")
    print()
    result = compare_run_dirs(baseline, new)
    for line in result["lines"]:
        print(line)
    return result["status"]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify_bitwise")
    parser.add_argument("baseline", type=Path, help="reference run dir (legacy or prior pipeline)")
    parser.add_argument("new", type=Path, help="new run dir to verify against baseline")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return verify_run_dirs(args.baseline.resolve(), args.new.resolve())


if __name__ == "__main__":
    sys.exit(main())
