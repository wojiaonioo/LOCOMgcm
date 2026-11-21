"""Layer B — DataSource plugins.

Importing this package triggers eager registration of all built-in sources
via `sources/__init__.py`.

Public API:
    DataSource Protocol
    SourceCapabilities / SourceContext / DataQuery / Grid frozen dataclasses
    TimeCoverage enum
    DATA_SOURCES PluginRegistry instance
    register_source decorator
"""

from ._registry import DATA_SOURCES, register_source
from ._types import (
    CredentialsProvider,
    DataQuery,
    DataSource,
    Grid,
    SourceCapabilities,
    SourceContext,
    TimeCoverage,
    validate_query_against_capabilities,
)

# Trigger source-plugin registrations
from . import sources  # noqa: F401

__all__ = [
    "DATA_SOURCES",
    "CredentialsProvider",
    "DataQuery",
    "DataSource",
    "Grid",
    "SourceCapabilities",
    "SourceContext",
    "TimeCoverage",
    "register_source",
    "validate_query_against_capabilities",
]
