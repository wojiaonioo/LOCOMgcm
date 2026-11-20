"""Atomic zarr write helper.

Pattern: write to `.tmp` directory, then atomic rename to final path.
This ensures partial writes on crash do not leave a corrupt zarr at the
final path.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping as MappingABC
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from defgcm.io.runner.layer_a import BundleMeta


def serialize_bundle_meta(meta: BundleMeta) -> str:
    """Serialize BundleMeta to a JSON string for storage in zarr.attrs.

    Includes provenance and stringifies opaque metadata values that are not
    directly JSON-serializable.
    """
    d = {
        "schema_version": meta.schema_version,
        "location_kind": meta.location_kind.value,
        "convention": meta.convention.value,
        "variables": [
            {
                "name": v.name,
                "cf_standard_name": v.cf_standard_name,
                "units": v.units,
                "dtype": v.dtype,
                "dims": list(v.dims),
            }
            for v in meta.variables
        ],
        "time_range": (
            None if meta.time_range is None
            else [meta.time_range[0].isoformat(), meta.time_range[1].isoformat()]
        ),
        "target_grid_id": meta.target_grid_id,
        "provenance": [
            {
                "kind": s.kind,
                "name": s.name,
                "timestamp": s.timestamp.isoformat(),
                "bundle_hash": s.bundle_hash,
                # params may not be JSON-serializable; stringify
                "params": {k: str(v) for k, v in s.params.items()},
            }
            for s in meta.provenance
        ],
        "extras": _json_attr_value(meta.extras),
    }
    return json.dumps(d, sort_keys=True)


def _json_attr_value(value: Any) -> Any:
    """Convert BundleMeta opaque values to JSON-safe attrs."""
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.generic):
        return _json_attr_value(value.item())
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MappingABC):
        return {
            str(key): _json_attr_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_json_attr_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(
            (_json_attr_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, np.ndarray):
        return _json_attr_value(value.tolist())
    return repr(value)


def _rechunk_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    """Make execution and storage chunks safe for a complete Zarr write.

    The prep operator chain is dask-backed (chunked along time for streaming).
    interp/regrid (xr.interp along z and lat/lon) split those dims into
    non-uniform dask chunks (e.g. z -> (12,11,12,11)); zarr rejects non-uniform
    chunks. Rechunk every non-time dim to a single chunk and keep time at 1, so
    the write still streams one time-step at a time (bounded memory) but each
    on-disk chunk is uniform. Preserve compatible storage chunks inherited
    from a source, but replace any dimension that would let multiple dask
    tasks write the same Zarr chunk. No-op for eager (numpy) datasets.
    """
    is_dask = any(
        v.chunks is not None for v in ds.variables.values()
    )
    if not is_dask:
        return ds
    spec = {d: (1 if d == "time" else -1) for d in ds.dims}
    ds = ds.chunk(spec)
    for var in ds.variables.values():
        stored = var.encoding.get("chunks")
        if var.chunks is None or stored is None:
            continue
        if isinstance(stored, int | np.integer):
            stored = (int(stored),) * var.ndim
        # Let xarray handle malformed encodings as it does for eager arrays.
        if len(stored) != var.ndim:
            continue
        # Only interior task boundaries must coincide with storage boundaries;
        # the last chunk may be partial in this full-array mode="w" write.
        safe = tuple(
            blocks[0] if any(b % size for b in blocks[:-1]) else size
            for blocks, size in zip(var.chunks, stored, strict=True)
        )
        var.encoding = {**var.encoding, "chunks": safe}
    return ds


def atomic_write_zarr(ds: xr.Dataset, final_path: Path) -> Path:
    """Atomically write `ds` to `final_path` as zarr.

    Algorithm:
      1. Write to `final_path.with_suffix('.zarr.tmp')`
      2. Atomic rename via shutil.move (POSIX semantics on same fs)

    Returns final_path.
    """
    tmp_path = final_path.parent / (final_path.name + ".tmp")
    # Remove any prior tmp dir from a crashed previous run
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    # dask-backed bundles stream chunk-by-chunk here (bounded memory); rechunk to
    # uniform chunks first so zarr accepts the interp/regrid-split dims.
    ds = _rechunk_for_zarr(ds)
    ds.to_zarr(tmp_path, mode="w", consolidated=False)
    if final_path.exists():
        shutil.rmtree(final_path)
    tmp_path.rename(final_path)
    return final_path
