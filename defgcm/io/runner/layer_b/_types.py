"""Layer B — DataSource public types."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from logging import Logger
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from defgcm.io.runner.layer_a import (
    ConventionTag,
    DataBundle,
    LocationKind,
)


class TimeCoverage(Enum):
    """How a DataSource declares its time coverage strategy."""

    FILE_GLOB_SCAN = "file_glob_scan"
    INSTANT = "instant"
    INTERVAL = "interval"
    ON_DEMAND = "on_demand"


def _coerce_enum(value: Any, enum_cls: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except Exception as exc:
        raise TypeError(f"{field_name} must be a {enum_cls.__name__}") from exc


def _coerce_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a positive integer")
    try:
        out = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a positive integer") from exc
    if out <= 0:
        raise ValueError(f"{field_name} must be positive")
    return int(out)


def _coerce_finite_float(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a finite numeric value")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a finite numeric value") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    if positive and out <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return out


def _string_tuple(
    values: Any,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be an iterable of strings, not a string")
    try:
        out = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings") from exc
    if not allow_empty and not out:
        raise ValueError(f"{field_name} must be nonempty")
    for idx, value in enumerate(out):
        if not isinstance(value, str) or not value:
            raise TypeError(f"{field_name}[{idx}] must be a nonempty string")
    if len(set(out)) != len(out):
        raise ValueError(f"{field_name} must not contain duplicates")
    return out


def _freeze_extras(value: Any) -> Any:
    if isinstance(value, MappingABC):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("DataQuery.extras keys must be nonempty strings")
            frozen[key] = _freeze_extras(item)
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_extras(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze_extras(item) for item in sorted(value, key=repr))
    return value


@dataclass(frozen=True)
class SourceCapabilities:
    """What a DataSource plugin can produce.

    produces:                tuple of LocationKind values the source can output
    variables:               tuple of CF standard_name (or local short names)
                              the source provides
    time_coverage_strategy:  TimeCoverage enum (how the source knows what time
                              range it covers)
    convention:              raw convention of the source's data (OPAQUE if
                              undeclared)
    requires_target_grid:    True if fetch() needs target_grid to operate
                              (e.g. analytical sources, neural-net inference)
    """

    produces: tuple[LocationKind, ...]
    variables: tuple[str, ...]
    time_coverage_strategy: TimeCoverage
    convention: ConventionTag
    requires_target_grid: bool

    def __post_init__(self) -> None:
        produces = tuple(
            _coerce_enum(value, LocationKind, f"produces[{idx}]")
            for idx, value in enumerate(self.produces)
        )
        if not produces:
            raise ValueError("SourceCapabilities.produces must be nonempty")
        if len(set(produces)) != len(produces):
            raise ValueError("SourceCapabilities.produces must not contain duplicates")
        object.__setattr__(self, "produces", produces)

        object.__setattr__(
            self,
            "variables",
            _string_tuple(
                self.variables,
                "SourceCapabilities.variables",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "time_coverage_strategy",
            _coerce_enum(
                self.time_coverage_strategy,
                TimeCoverage,
                "SourceCapabilities.time_coverage_strategy",
            ),
        )
        object.__setattr__(
            self,
            "convention",
            _coerce_enum(
                self.convention,
                ConventionTag,
                "SourceCapabilities.convention",
            ),
        )
        if type(self.requires_target_grid) is not bool:
            raise TypeError("SourceCapabilities.requires_target_grid must be bool")


@dataclass(frozen=True)
class Grid:
    """Minimal forward-declared geometry handle."""

    nx: int
    ny: int
    nz: int
    center_lat: float
    center_lon: float
    resolution_deg: float
    depth_m: float
    vertical_layer_thicknesses: tuple[float, ...]
    grid_id: str  # canonical short id like 'micro_16x16x4'

    def __post_init__(self) -> None:
        nx = _coerce_positive_int(self.nx, "Grid.nx")
        ny = _coerce_positive_int(self.ny, "Grid.ny")
        nz = _coerce_positive_int(self.nz, "Grid.nz")
        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "nz", nz)
        object.__setattr__(
            self,
            "center_lat",
            _coerce_finite_float(self.center_lat, "Grid.center_lat"),
        )
        object.__setattr__(
            self,
            "center_lon",
            _coerce_finite_float(self.center_lon, "Grid.center_lon"),
        )
        object.__setattr__(
            self,
            "resolution_deg",
            _coerce_finite_float(
                self.resolution_deg,
                "Grid.resolution_deg",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "depth_m",
            _coerce_finite_float(self.depth_m, "Grid.depth_m", positive=True),
        )
        try:
            thicknesses = tuple(
                _coerce_finite_float(
                    value,
                    "Grid.vertical_layer_thicknesses",
                    positive=True,
                )
                for value in self.vertical_layer_thicknesses
            )
        except TypeError as exc:
            raise TypeError("Grid.vertical_layer_thicknesses must be iterable") from exc
        if len(thicknesses) != nz:
            raise ValueError(
                "Grid.vertical_layer_thicknesses length must match Grid.nz"
            )
        object.__setattr__(self, "vertical_layer_thicknesses", thicknesses)
        if not isinstance(self.grid_id, str) or not self.grid_id:
            raise TypeError("Grid.grid_id must be a nonempty string")


# Type alias for forward-compat with future credentials provider.
CredentialsProvider = Any  # placeholder


@dataclass(frozen=True)
class SourceContext:
    """Dependency-injection container passed to source constructors.

    cache_root:           shared cache directory
    target_grid_resolver: grid_id → Grid lookup
    logger:               per-source named logger
    credentials:          future auth provider (currently None)
    """

    cache_root: Path
    target_grid_resolver: Callable[[str], Grid]
    logger: Logger
    credentials: CredentialsProvider | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        if not callable(self.target_grid_resolver):
            raise TypeError("SourceContext.target_grid_resolver must be callable")
        if not callable(getattr(self.logger, "info", None)):
            raise TypeError("SourceContext.logger must provide an info() method")


@dataclass(frozen=True)
class DataQuery:
    """Source.fetch input — fully specifies one fetch call.

    location_kind:  which LocationKind the caller wants
    variables:      subset of capabilities.variables to retrieve
    time_range:     (t0, t1) inclusive interval (or None for time-invariant)
    target_grid:    grid to interpolate to (None = source-native)
    extras:         plugin-specific (e.g. sides=['N','E'] for boundary fetches)
    """

    location_kind: LocationKind
    variables: tuple[str, ...]
    time_range: tuple[datetime, datetime] | None
    target_grid: Grid | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "location_kind",
            _coerce_enum(self.location_kind, LocationKind, "DataQuery.location_kind"),
        )
        object.__setattr__(
            self,
            "variables",
            _string_tuple(self.variables, "DataQuery.variables", allow_empty=True),
        )
        if self.time_range is not None:
            try:
                t0, t1 = self.time_range
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "DataQuery.time_range must be a (start, end) pair"
                ) from exc
            if not isinstance(t0, datetime) or not isinstance(t1, datetime):
                raise TypeError("DataQuery.time_range values must be datetime instances")
            if t1 < t0:
                raise ValueError("DataQuery.time_range end must be >= start")
            object.__setattr__(self, "time_range", (t0, t1))
        if self.target_grid is not None and not isinstance(self.target_grid, Grid):
            raise TypeError("DataQuery.target_grid must be a Grid or None")
        extras = {} if self.extras is None else self.extras
        object.__setattr__(self, "extras", _freeze_extras(extras))


def validate_query_against_capabilities(
    query: DataQuery,
    capabilities: SourceCapabilities,
    *,
    source_name: str = "source",
) -> None:
    """Validate shared pre-fetch rules that are expressible in capabilities."""
    if not isinstance(query, DataQuery):
        raise TypeError("query must be a DataQuery")
    if not isinstance(capabilities, SourceCapabilities):
        raise TypeError("capabilities must be a SourceCapabilities")
    if query.location_kind not in capabilities.produces:
        raise ValueError(
            f"{source_name}: query location_kind {query.location_kind.value!r} "
            f"is not produced by source"
        )
    if capabilities.variables:
        unknown = tuple(v for v in query.variables if v not in capabilities.variables)
        if unknown:
            raise ValueError(
                f"{source_name}: query variables not advertised by source: {unknown}"
            )
    if capabilities.requires_target_grid and query.target_grid is None:
        raise ValueError(f"{source_name}: query requires target_grid")


class DataSource(Protocol):
    """Layer B plugin protocol.

    Every plugin class registered into DATA_SOURCES MUST have:
      - capabilities class attribute (SourceCapabilities frozen dataclass)
      - __init__(self, ctx: SourceContext, **config) constructor
      - fetch(self, query: DataQuery) -> DataBundle method
    """

    capabilities: SourceCapabilities

    def __init__(self, ctx: SourceContext, **config: Any) -> None: ...
    def fetch(self, query: DataQuery) -> DataBundle: ...
