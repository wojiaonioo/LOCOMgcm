"""Manifest yaml schema (per stage entry, artifact entry, top-level run header)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping as MappingABC
from typing import Any, Mapping


class StageStatus(Enum):
    """Stage status; STALE marks downstream invalidation."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STALE = "STALE"  # input changed, must rerun before COMPLETED-equivalent


RUN_LEVEL_STATUSES = frozenset(
    {
        StageStatus.PENDING.value,
        StageStatus.RUNNING.value,
        StageStatus.COMPLETED.value,
        StageStatus.FAILED.value,
    }
)


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return value


def _freeze_manifest_value(value: Any, field_name: str) -> Any:
    if isinstance(value, MappingABC):
        return _freeze_manifest_mapping(value, field_name)
    if isinstance(value, list | tuple):
        return tuple(
            _freeze_manifest_value(item, f"{field_name}[]")
            for item in value
        )
    return value


def _freeze_manifest_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise TypeError(f"{field_name} must be a mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be nonempty strings")
        frozen[key] = _freeze_manifest_value(item, f"{field_name}.{key}")
    return MappingProxyType(frozen)


def _validate_manifest_path(path: str) -> str:
    raw = _nonempty_string(path, "ArtifactEntry.path")
    parsed = Path(raw)
    if not parsed.is_absolute() and any(part == ".." for part in parsed.parts):
        raise ValueError("ArtifactEntry.path must not escape the run directory")
    return raw


def _coerce_stage_status(status: Any) -> StageStatus:
    if isinstance(status, StageStatus):
        return status
    if isinstance(status, str):
        try:
            return StageStatus(status)
        except ValueError as exc:
            raise ValueError(f"StageEntry.status must be one of {[s.value for s in StageStatus]}") from exc
    raise TypeError("StageEntry.status must be a StageStatus or status string")


def validate_run_level_status(status: Any) -> str:
    raw = _nonempty_string(status, "overall_status")
    if raw not in RUN_LEVEL_STATUSES:
        raise ValueError(f"overall_status must be one of {sorted(RUN_LEVEL_STATUSES)}")
    return raw


@dataclass(frozen=True)
class ArtifactEntry:
    name: str
    path: str          # relative to run dir, or absolute for external artifacts
    artifact_kind: str
    bundle_hash: str
    schema_version: str
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, "ArtifactEntry.name"))
        object.__setattr__(self, "path", _validate_manifest_path(self.path))
        object.__setattr__(
            self,
            "artifact_kind",
            _nonempty_string(self.artifact_kind, "ArtifactEntry.artifact_kind"),
        )
        object.__setattr__(
            self,
            "bundle_hash",
            _nonempty_string(self.bundle_hash, "ArtifactEntry.bundle_hash"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _nonempty_string(self.schema_version, "ArtifactEntry.schema_version"),
        )
        object.__setattr__(self, "extras", _freeze_manifest_mapping(self.extras, "ArtifactEntry.extras"))


@dataclass(frozen=True)
class StageEntry:
    status: StageStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    pid: int | None = None
    hostname: str | None = None
    wallclock_seconds: float | None = None
    artifacts: tuple[ArtifactEntry, ...] = ()
    error: str | None = None
    inputs_resolved: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _coerce_stage_status(self.status))
        artifacts: list[ArtifactEntry] = []
        for idx, artifact in enumerate(self.artifacts):
            if isinstance(artifact, ArtifactEntry):
                artifacts.append(artifact)
            elif isinstance(artifact, MappingABC):
                artifacts.append(ArtifactEntry(**artifact))
            else:
                raise TypeError(f"StageEntry.artifacts[{idx}] must be an ArtifactEntry or mapping")
        object.__setattr__(self, "artifacts", tuple(artifacts))
        if self.pid is not None and (not isinstance(self.pid, int) or self.pid < 0):
            raise ValueError("StageEntry.pid must be a nonnegative integer or None")
        if self.wallclock_seconds is not None:
            wallclock = float(self.wallclock_seconds)
            if not math.isfinite(wallclock) or wallclock < 0.0:
                raise ValueError("StageEntry.wallclock_seconds must be finite and nonnegative")
            object.__setattr__(self, "wallclock_seconds", wallclock)
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("StageEntry.error must be a string or None")
        object.__setattr__(
            self,
            "inputs_resolved",
            _freeze_manifest_mapping(self.inputs_resolved, "StageEntry.inputs_resolved"),
        )
