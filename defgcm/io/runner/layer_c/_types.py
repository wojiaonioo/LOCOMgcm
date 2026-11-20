"""Public types for preprocessing operators and mergers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from defgcm.io.runner.layer_a import (
    ConventionTag,
    DataBundle,
    LocationKind,
)


@dataclass(frozen=True)
class OperatorCapabilities:
    """What an Operator (1→1) accepts/produces.

    accepts:               LocationKinds the operator can transform
    produces:              LocationKinds in the output (usually 1)
    convention_in:         required input convention; None = any
    convention_out:        output convention; None = passthrough (same as input)
    preserves_variables:   does it keep all variables?
    is_pure:               True if no side effects (default; almost always)
    """

    accepts: tuple[LocationKind, ...]
    produces: tuple[LocationKind, ...]
    convention_in: ConventionTag | None
    convention_out: ConventionTag | None
    preserves_variables: bool
    is_pure: bool = True


@dataclass(frozen=True)
class MergerCapabilities:
    """What a Merger (N→1) accepts/produces."""

    accepts_primary: tuple[LocationKind, ...]
    accepts_secondary: tuple[LocationKind, ...]
    produces: tuple[LocationKind, ...]
    convention_in: ConventionTag | None
    convention_out: ConventionTag | None
    n_secondaries: tuple[int, ...] | None = None


class Operator(Protocol):
    """Pure 1→1 transform on DataBundle."""

    capabilities: OperatorCapabilities

    def __init__(self, **config: Any) -> None: ...
    def __call__(self, bundle: DataBundle) -> DataBundle: ...


class Merger(Protocol):
    """N→1 merge of primary + secondaries into a single bundle."""

    capabilities: MergerCapabilities

    def __init__(self, **config: Any) -> None: ...
    def __call__(self, primary: DataBundle, *secondaries: DataBundle) -> DataBundle: ...
