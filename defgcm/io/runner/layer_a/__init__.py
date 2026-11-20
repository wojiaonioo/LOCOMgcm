"""Layer A — DataBundle public re-exports.

Public API:
    LocationKind, ConventionTag enums
    VariableSpec, ProvenanceStep, BundleMeta, DataBundle frozen dataclasses
    KNOWN_SCHEMA_VERSIONS frozen set
"""

from ._types import (
    KNOWN_SCHEMA_VERSIONS,
    BundleMeta,
    ConventionTag,
    DataBundle,
    LocationKind,
    ProvenanceStep,
    VariableSpec,
)

__all__ = [
    "BundleMeta",
    "ConventionTag",
    "DataBundle",
    "KNOWN_SCHEMA_VERSIONS",
    "LocationKind",
    "ProvenanceStep",
    "VariableSpec",
]
