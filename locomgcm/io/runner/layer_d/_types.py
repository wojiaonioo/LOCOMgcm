"""Public types for model-input consumer adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Any, Mapping, Protocol

from locomgcm.io.runner.layer_a import (
    ConventionTag,
    DataBundle,
    LocationKind,
)


class ArtifactKind(Enum):
    """How a consumer's output is realized."""

    FILE_ZARR = "file_zarr"
    FILE_NETCDF = "file_netcdf"
    FILE_BINARY = "file_binary"        # MIT .bin
    FILE_TEXT = "file_text"             # MIT SIZE.h / namelist
    FILE_DIRECTORY = "file_directory"   # output-dir-shaped artifact (legacy runner)
    DB_INGESTION = "db_ingestion"
    RUNTIME_PORT = "runtime_port"
    IN_MEMORY = "in_memory"


@dataclass(frozen=True)
class ConsumerCapabilities:
    """What a ConsumerAdapter accepts and produces."""

    accepts_location_kinds: tuple[LocationKind, ...]
    required_convention: ConventionTag
    required_variables: tuple[str, ...]
    artifact_kind: ArtifactKind
    artifact_schema_version: str


@dataclass(frozen=True)
class ConsumerContext:
    """DI container passed to ConsumerAdapter constructors.

    output_dir:        per-stage output root; consumer restricted to subtree
    run_id:            run identifier
    manifest_writer:   framework-injected; consumer calls back to record
                       artifact (or returns ArtifactRef for framework to log)
    logger:            per-consumer named logger
    run_cfg:           full RunConfig — used by namelist consumers to access
                       physics / time / mit_comparison.effective_* etc.
    extras:            consumer-specific (e.g. stage_name)
    """

    output_dir: Path
    run_id: str
    manifest_writer: Any = None  # ManifestWriter
    logger: Logger | None = None
    run_cfg: Any = None  # RunConfig; populated by build_execution_context
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to an artifact produced by a consumer."""

    name: str
    path: Path | None
    artifact_kind: ArtifactKind
    bundle_hash: str
    schema_version: str
    extras: dict = field(default_factory=dict)


class ConsumerAdapter(Protocol):
    """Layer D plugin protocol."""

    capabilities: ConsumerCapabilities

    def __init__(self, ctx: ConsumerContext, **config: Any) -> None: ...
    def consume(self, bundle: DataBundle) -> ArtifactRef: ...
