"""Layer A — DataBundle invariant validators.

Called from DataBundle.__post_init__. Pure functions of (xr.Dataset, BundleMeta).

Seven invariants are enforced here (listed in :func:`validate_bundle`).
"""

from __future__ import annotations

import xarray as xr

from ._types import BundleMeta, KNOWN_SCHEMA_VERSIONS, LocationKind


# Canonical dim order for gridded 3D ocean bundles. Variables may omit any
# subset of these dims but cannot reorder them.
_GRIDDED_3D_OCEAN_DIM_ORDER: tuple[str, ...] = (
    "time",
    "z",
    "latitude",
    "longitude",
)
_SINGLE_GRID_CF_LOCATION_KINDS: frozenset[LocationKind] = frozenset(
    {
        LocationKind.GRIDDED_3D_OCEAN,
        LocationKind.GRIDDED_2D_SURFACE,
        LocationKind.GRIDDED_3D_ATM,
    }
)


def validate_bundle(data: xr.Dataset, meta: BundleMeta) -> None:
    """Run all DataBundle invariants. Raises ValueError on first violation.

    Invariants enforced:
      1. variable names match
      2. variable dtypes match
      3. variable dims match
      4. schema_version known
      5. CF coords present for gridded location_kinds
      6. canonical dim order for GRIDDED_3D_OCEAN
      7. provenance non-empty
    """
    _check_variable_names(data, meta)
    _check_variable_dtypes(data, meta)
    _check_variable_dims(data, meta)
    _check_schema_version(meta)
    _check_cf_coords(data, meta)
    _check_gridded_3d_dim_order(meta)
    _check_provenance_non_empty(meta)


def _check_variable_names(data: xr.Dataset, meta: BundleMeta) -> None:
    """Invariant 1: meta.variables names must equal data.data_vars."""
    declared_names = [v.name for v in meta.variables]
    duplicates = sorted(
        name for name in set(declared_names)
        if declared_names.count(name) > 1
    )
    if duplicates:
        raise ValueError(
            f"BundleMeta.variables contains duplicate names: {duplicates}"
        )

    meta_names = set(declared_names)
    data_names = set(data.data_vars)
    if meta_names != data_names:
        missing_in_data = meta_names - data_names
        extra_in_data = data_names - meta_names
        raise ValueError(
            f"BundleMeta.variables names {sorted(meta_names)} != "
            f"Dataset.data_vars {sorted(data_names)}; "
            f"missing in data: {sorted(missing_in_data)}, "
            f"extra in data: {sorted(extra_in_data)}"
        )


def _check_variable_dtypes(data: xr.Dataset, meta: BundleMeta) -> None:
    """Invariant 2: each variable's data dtype matches spec dtype."""
    for var in meta.variables:
        actual = str(data[var.name].dtype)
        if actual != var.dtype:
            raise ValueError(
                f"variable {var.name!r} dtype {actual} != spec {var.dtype}"
            )


def _check_variable_dims(data: xr.Dataset, meta: BundleMeta) -> None:
    """Invariant 3: each variable's data dims match spec dims (order + names)."""
    for var in meta.variables:
        actual = tuple(data[var.name].dims)
        if actual != var.dims:
            raise ValueError(
                f"variable {var.name!r} dims {actual} != spec {var.dims}"
            )


def _check_schema_version(meta: BundleMeta) -> None:
    """Invariant 4: schema_version must be known."""
    if meta.schema_version not in KNOWN_SCHEMA_VERSIONS:
        raise ValueError(
            f"unknown schema_version {meta.schema_version!r}; "
            f"known: {sorted(KNOWN_SCHEMA_VERSIONS)}"
        )


def _check_cf_coords(data: xr.Dataset, meta: BundleMeta) -> None:
    """Invariant 5: gridded location_kinds require latitude + longitude coords.

    Only the single-grid GRIDDED_* kinds are constrained. BOUNDARY_TIMESERIES,
    MESH, SCATTERED_*, and TIDAL_HARMONICS are intentionally exempt: tidal
    harmonics carry u/v on two distinct staggered grids (lat_u/lon_u vs
    lat_v/lon_v) plus a constituent axis, which cannot be expressed as a single
    pair of `latitude`/`longitude` coords. Do NOT add TIDAL_HARMONICS here.
    """
    if meta.location_kind not in _SINGLE_GRID_CF_LOCATION_KINDS:
        return

    for coord in ("latitude", "longitude"):
        if coord not in data.coords:
            raise ValueError(
                f"location_kind={meta.location_kind.value} bundle missing "
                f"CF coord {coord!r} (data.coords={sorted(data.coords)})"
            )
        if coord not in data.dims or tuple(data.coords[coord].dims) != (coord,):
            raise ValueError(
                f"location_kind={meta.location_kind.value} requires CF coord "
                f"{coord!r} to be a one-dimensional dimension coordinate; "
                f"got dims {tuple(data.coords[coord].dims)}"
            )

    # Arakawa-C staggered u/v live on offset horizontal dims (longitude_u /
    # latitude_v — see model_input.py); accept those alongside the canonical
    # coord so every variable is still required to be georeferenced on both
    # horizontal axes.
    accepted_dims = {
        "latitude": ("latitude", "latitude_v"),
        "longitude": ("longitude", "longitude_u"),
    }
    for var in meta.variables:
        for coord, accepted in accepted_dims.items():
            if not any(dim in var.dims for dim in accepted):
                raise ValueError(
                    f"variable {var.name!r} dims {var.dims} do not use "
                    f"required CF coord {coord!r} (or staggered variant "
                    f"{accepted[1]!r}) for "
                    f"location_kind={meta.location_kind.value}"
                )


def _check_gridded_3d_dim_order(meta: BundleMeta) -> None:
    """Invariant 6: GRIDDED_3D_OCEAN variables follow canonical dim order.

    Variables may omit any subset of (time, z, latitude, longitude) but cannot
    reorder. E.g. a 2D variable like 'eta' may have dims (time, latitude,
    longitude) — still in canonical order, just no 'z'.
    """
    if meta.location_kind != LocationKind.GRIDDED_3D_OCEAN:
        return
    canonical = _GRIDDED_3D_OCEAN_DIM_ORDER
    for var in meta.variables:
        # Filter canonical to dims actually in var.dims, preserve canonical order.
        expected_subset = tuple(d for d in canonical if d in var.dims)
        # Filter var.dims to those in canonical (ignore custom dims like 'mask_id').
        actual_canonical = tuple(d for d in var.dims if d in canonical)
        if actual_canonical != expected_subset:
            raise ValueError(
                f"variable {var.name!r} dim order {var.dims} violates "
                f"GRIDDED_3D_OCEAN canonical {canonical}: "
                f"got canonical-subset {actual_canonical}, expected {expected_subset}"
            )


def _check_provenance_non_empty(meta: BundleMeta) -> None:
    """Invariant 7: every bundle must have at least one provenance step."""
    if len(meta.provenance) < 1:
        raise ValueError(
            "BundleMeta.provenance is empty; every bundle must have at least "
            "one ProvenanceStep (typically kind='source' from the originating Layer B)"
        )
