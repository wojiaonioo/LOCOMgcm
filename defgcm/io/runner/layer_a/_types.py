"""Layer A — DataBundle types.

Universal data carrier for the simulation pipeline.

This module defines:
  - LocationKind enum (12 values)
  - ConventionTag enum (4 values)
  - VariableSpec frozen dataclass
  - ProvenanceStep frozen dataclass with frozen params
  - BundleMeta frozen dataclass with frozen extras
  - DataBundle frozen dataclass with __post_init__ invariant checks

Conventions: w / depth / z / heat_flux / salt_flux sign conventions live in
ConventionTag enum values. ConventionConverter (Layer C) handles
inter-convention dispatch; Layer A only carries the tag.
"""

from __future__ import annotations

import re
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Mapping

import xarray as xr


class LocationKind(Enum):
    """Spatial / temporal kind of a bundle payload."""

    GRIDDED_3D_OCEAN = "gridded_3d_ocean"
    GRIDDED_2D_SURFACE = "gridded_2d_surface"
    GRIDDED_3D_ATM = "gridded_3d_atm"
    SCATTERED_PROFILE = "scattered_profile"
    SCATTERED_POINT = "scattered_point"
    ALONG_TRACK = "along_track"
    SWATH = "swath"
    TRAJECTORY = "trajectory"
    COUPLED_STATE = "coupled_state"
    MESH = "mesh"
    BOUNDARY_TIMESERIES = "boundary_timeseries"
    # Tidal harmonic constants (amplitude/phase per constituent). The u/v
    # fields live on two distinct staggered grids (lat_u/lon_u, lat_v/lon_v)
    # plus a constituent axis, so they do not fit the single latitude/longitude
    # CF-coord invariant of the GRIDDED_* kinds — exempt in _check_cf_coords.
    TIDAL_HARMONICS = "tidal_harmonics"


class ConventionTag(Enum):
    """Sign / layout convention of a bundle payload (carries LOCOMgcm ↔ MIT mismatch)."""

    TORCH_OCEAN = "torchocean"
    MITGCM = "mitgcm"
    CF_STANDARD = "cf_standard"
    OPAQUE = "opaque"


KNOWN_SCHEMA_VERSIONS: frozenset[str] = frozenset({"0.1.0"})
_VALID_PROVENANCE_KINDS: frozenset[str] = frozenset({"source", "operator"})
_VALID_BUNDLE_HASH_RE = re.compile(r"^[0-9a-fA-F]{8}([0-9a-fA-F]{56})?$")


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return value


def _freeze_mapping_value(value: Any, field_name: str) -> Any:
    if isinstance(value, MappingABC):
        return _freeze_mapping(value, field_name)
    if isinstance(value, list | tuple):
        return tuple(
            _freeze_mapping_value(item, f"{field_name}[]")
            for item in value
        )
    if isinstance(value, set | frozenset):
        return frozenset(
            _freeze_mapping_value(item, f"{field_name}[]")
            for item in value
        )
    return value


def _freeze_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise TypeError(f"{field_name} must be a mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be nonempty strings")
        frozen[key] = _freeze_mapping_value(item, f"{field_name}.{key}")
    return MappingProxyType(frozen)


def _validate_bundle_hash(value: Any) -> str:
    raw = _nonempty_string(value, "ProvenanceStep.bundle_hash")
    if _VALID_BUNDLE_HASH_RE.fullmatch(raw) is None:
        raise ValueError(
            "ProvenanceStep.bundle_hash must be an 8- or 64-character hex string"
        )
    return raw.lower()


@dataclass(frozen=True)
class VariableSpec:
    """Strongly-typed declaration of a payload variable.

    name: short local name (e.g. 'temperature')
    cf_standard_name: CF-1.10 standard name (e.g. 'sea_water_potential_temperature')
    units: CF-compliant unit string (e.g. 'degC')
    dtype: numpy dtype string ('float64' / 'float32' / 'int32' / 'bool')
    dims: tuple of dim names (e.g. ('time', 'z', 'latitude', 'longitude'))
    """

    name: str
    cf_standard_name: str
    units: str
    dtype: str
    dims: tuple[str, ...]


@dataclass(frozen=True)
class ProvenanceStep:
    """One entry in the bundle history.

    kind: 'source' (initial fetch) or 'operator' (subsequent transform)
    name: plugin class name (e.g. 'HycomNetcdfSource', 'GridInterpolator')
    params: call params (hashable repr; values should be primitives or tuples)
    timestamp: when this step produced the bundle
    bundle_hash: provenance-only label; producers write the placeholder
                 "00000000" (no digest is computed). 64-char values in
                 older artifacts still validate. Nothing compares this field.
    """

    kind: Literal["source", "operator"]
    name: str
    params: Mapping[str, Any]
    timestamp: datetime
    bundle_hash: str

    def __post_init__(self) -> None:
        if self.kind not in _VALID_PROVENANCE_KINDS:
            raise ValueError(
                "ProvenanceStep.kind must be one of "
                f"{sorted(_VALID_PROVENANCE_KINDS)}"
            )
        object.__setattr__(self, "name", _nonempty_string(self.name, "ProvenanceStep.name"))
        if not isinstance(self.timestamp, datetime):
            raise TypeError("ProvenanceStep.timestamp must be a datetime")
        object.__setattr__(
            self,
            "params",
            _freeze_mapping(self.params, "ProvenanceStep.params"),
        )
        object.__setattr__(self, "bundle_hash", _validate_bundle_hash(self.bundle_hash))


@dataclass(frozen=True)
class BundleMeta:
    """Strongly-typed metadata for a DataBundle.

    schema_version: must be in KNOWN_SCHEMA_VERSIONS
    location_kind: enum tagging spatial/temporal kind
    convention: enum tagging sign/layout convention
    variables: tuple of VariableSpec, 1-to-1 with bundle.data.data_vars
    time_range: (t0, t1) interval if time-varying; None for time-invariant
    target_grid_id: canonical grid id once regridded; None for source-native
    provenance: chain of ProvenanceStep from source through operators
    extras: plugin-specific opaque metadata (e.g. convention_event, chunking)
    """

    schema_version: str
    location_kind: LocationKind
    convention: ConventionTag
    variables: tuple[VariableSpec, ...]
    time_range: tuple[datetime, datetime] | None
    target_grid_id: str | None
    provenance: tuple[ProvenanceStep, ...]
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.location_kind, LocationKind):
            raise TypeError("BundleMeta.location_kind must be a LocationKind")
        if not isinstance(self.convention, ConventionTag):
            raise TypeError("BundleMeta.convention must be a ConventionTag")

        variables = tuple(self.variables)
        for idx, var in enumerate(variables):
            if not isinstance(var, VariableSpec):
                raise TypeError(f"BundleMeta.variables[{idx}] must be a VariableSpec")
        object.__setattr__(self, "variables", variables)

        provenance = tuple(self.provenance)
        for idx, step in enumerate(provenance):
            if not isinstance(step, ProvenanceStep):
                raise TypeError(f"BundleMeta.provenance[{idx}] must be a ProvenanceStep")
        object.__setattr__(self, "provenance", provenance)

        if self.time_range is not None:
            if len(self.time_range) != 2:
                raise ValueError("BundleMeta.time_range must contain exactly two datetimes")
            start, end = self.time_range
            if not isinstance(start, datetime) or not isinstance(end, datetime):
                raise TypeError("BundleMeta.time_range entries must be datetime values")
            if start > end:
                raise ValueError("BundleMeta.time_range start must be <= end")
            object.__setattr__(self, "time_range", (start, end))

        if self.target_grid_id is not None:
            object.__setattr__(
                self,
                "target_grid_id",
                _nonempty_string(self.target_grid_id, "BundleMeta.target_grid_id"),
            )
        object.__setattr__(self, "extras", _freeze_mapping(self.extras, "BundleMeta.extras"))


@dataclass(frozen=True)
class DataBundle:
    """Universal data carrier — frozen wrapper around xarray.Dataset.

    Construction validates invariants (see _invariants.validate_bundle):
      1. meta.variables names match data.data_vars
      2. meta.variables[i].dtype matches data[name].dtype
      3. meta.variables[i].dims matches data[name].dims
      4. schema_version in KNOWN_SCHEMA_VERSIONS
      5. GRIDDED_* location_kinds require 'latitude' / 'longitude' coords
      6. GRIDDED_3D_OCEAN dim order canonical (time, z, latitude, longitude)
      7. provenance non-empty
    """

    data: xr.Dataset
    meta: BundleMeta

    def __post_init__(self) -> None:
        # Validation deferred to _invariants module to keep this class focused.
        from . import _invariants

        _invariants.validate_bundle(self.data, self.meta)
