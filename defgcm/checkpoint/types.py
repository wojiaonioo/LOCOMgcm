"""Data structures for checkpoint capture, persistence and restoration.

``ModelCheckpoint`` contains:
  * ``prognostic``: eta, u, v, temperature and salinity. Capture copies these
    fields to CPU memory while preserving their dtypes.
  * ``step_state``: the restart state, including AB2 history, solver caches,
    boundary history and typed values carried between steps.
  * ``meta``: the schema, model clock, configuration and state-key metadata
    used to validate and restore the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch

CHECKPOINT_SCHEMA_VERSION = 5

CHECKPOINT_META_FIELDS = frozenset({
    "schema_version",
    "global_step",
    "model_time_seconds",
    "dt",
    "mom_forcing_out_ab",
    "physics_flags",
    "config_fingerprint",
    "conventions",
    "git_head",
    "created_utc",
    "permanent",
    "requires_grad",
    "state_keys",
})

#: Prognostic field names, in canonical order.
PROGNOSTIC_FIELDS = ("eta", "u", "v", "temp", "salt")


class CheckpointSchemaVersionError(ValueError):
    """Base error for a checkpoint schema that cannot be imported."""


class LegacyCheckpointSchemaVersionError(CheckpointSchemaVersionError):
    """A pre-v3 checkpoint was supplied; schema migration is intentionally absent."""


class UnknownCheckpointSchemaVersionError(CheckpointSchemaVersionError):
    """A checkpoint from an unknown newer schema was supplied."""


def validate_checkpoint_schema_version(value: Any) -> int:
    """Require the exact current schema without coercion or migration."""
    if type(value) is not int:
        raise TypeError(
            "checkpoint meta.schema_version must be an exact int, "
            f"got {type(value).__name__}"
        )
    if value == CHECKPOINT_SCHEMA_VERSION:
        return value
    if value < CHECKPOINT_SCHEMA_VERSION:
        if value == 2:
            raise LegacyCheckpointSchemaVersionError(
                "checkpoint schema 2 predates the required mom_forcing_out_ab "
                "marker for _ab2_prev_tendency; refusing to infer whether the "
                "saved AB2 history includes momentum forcing. Re-create the "
                "checkpoint or re-convert its source pickup with the current "
                "converter. Alternatively, first process it with a tool that "
                "requires an explicit legacy compatibility mode; direct resume "
                "does not guess legacy semantics"
            )
        raise LegacyCheckpointSchemaVersionError(
            f"checkpoint schema {value} is obsolete; expected exactly "
            f"{CHECKPOINT_SCHEMA_VERSION}. Re-create or re-convert the checkpoint; "
            "automatic migration is not supported"
        )
    raise UnknownCheckpointSchemaVersionError(
        f"checkpoint schema {value} is unknown; expected exactly "
        f"{CHECKPOINT_SCHEMA_VERSION}"
    )


def require_nonempty_fingerprint(value: Any, *, path: str) -> str | dict[str, Any]:
    """Require a config identity: a config mapping or a label string.

    ``capture.config_fingerprint`` returns the mapping; callers that build a
    model without a run config pass a label, and older checkpoints hold a
    digest string.
    """
    if type(value) is str and value:
        return value
    if type(value) is dict and value and all(type(key) is str for key in value):
        return value
    raise ValueError(
        f"{path} must be a non-empty string or a non-empty dict with string keys"
    )


@dataclass(frozen=True)
class CheckpointMeta:
    """Self-describing header for a checkpoint (mirrored to meta.json)."""

    schema_version: int
    global_step: int
    model_time_seconds: float
    dt: float
    mom_forcing_out_ab: bool
    physics_flags: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str | dict[str, Any] = ""
    conventions: dict[str, str] = field(default_factory=dict)
    git_head: str = "unknown"
    created_utc: str = ""
    permanent: bool = False
    requires_grad: bool = False
    state_keys: tuple[str, ...] = ()
    migration_records: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **({"migration_records": list(self.migration_records)} if self.migration_records else {}),
            "schema_version": self.schema_version,
            "global_step": self.global_step,
            "model_time_seconds": self.model_time_seconds,
            "dt": self.dt,
            "mom_forcing_out_ab": self.mom_forcing_out_ab,
            "physics_flags": dict(self.physics_flags),
            "config_fingerprint": self.config_fingerprint,
            "conventions": dict(self.conventions),
            "git_head": self.git_head,
            "created_utc": self.created_utc,
            "permanent": self.permanent,
            "requires_grad": self.requires_grad,
            "state_keys": list(self.state_keys),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, legacy_v3: bool = False) -> "CheckpointMeta":
        if type(d) is not dict:
            raise TypeError(f"checkpoint meta must be a dict, got {type(d).__name__}")
        if "schema_version" not in d:
            raise ValueError("checkpoint meta is missing required field 'schema_version'")
        # Check the version before the v3 field set so legacy inputs receive
        # their dedicated no-migration error even when old metadata is sparse.
        _validate_version(d["schema_version"], legacy_v3)
        fields = frozenset(d)
        expected = CHECKPOINT_META_FIELDS
        if not legacy_v3 and "migration_records" in fields:
            expected = expected | {"migration_records"}
        if fields != expected:
            missing = sorted(CHECKPOINT_META_FIELDS - fields)
            extra = sorted(fields - CHECKPOINT_META_FIELDS)
            raise ValueError(
                "checkpoint meta fields must match schema v3 exactly; "
                f"missing={missing}, extra={extra}"
            )
        if type(d["state_keys"]) is not list:
            raise TypeError(
                "checkpoint meta.state_keys must be a list of unique strings, "
                f"got {type(d['state_keys']).__name__}"
            )
        meta = cls(
            schema_version=d["schema_version"],
            global_step=d["global_step"],
            model_time_seconds=d["model_time_seconds"],
            dt=d["dt"],
            mom_forcing_out_ab=d["mom_forcing_out_ab"],
            physics_flags=d["physics_flags"],
            config_fingerprint=d["config_fingerprint"],
            conventions=d["conventions"],
            git_head=d["git_head"],
            created_utc=d["created_utc"],
            permanent=d["permanent"],
            requires_grad=d["requires_grad"],
            state_keys=tuple(d["state_keys"]),
            migration_records=tuple(d.get("migration_records", ())),
        )
        validate_checkpoint_meta(meta, legacy_v3=legacy_v3)
        return meta


def _validate_version(version, legacy_v3):
    if legacy_v3:
        if type(version) is not int or version != 3:
            raise ValueError("legacy validation requires exact schema 3")
    else:
        validate_checkpoint_schema_version(version)


def validate_checkpoint_meta(meta: CheckpointMeta, *, legacy_v3: bool = False) -> None:
    """Validate an in-memory v3 header without repairing any value."""
    if not isinstance(meta, CheckpointMeta):
        raise TypeError(
            f"checkpoint meta must be CheckpointMeta, got {type(meta).__name__}"
        )
    _validate_version(meta.schema_version, legacy_v3)
    if type(meta.migration_records) is not tuple or any(
        type(record) is not dict for record in meta.migration_records
    ):
        raise TypeError("checkpoint migration_records must be a tuple of records")
    if legacy_v3 and meta.migration_records:
        raise ValueError("schema 3 cannot contain migration_records")
    if type(meta.global_step) is not int:
        raise TypeError(
            "checkpoint meta.global_step must be an exact int, "
            f"got {type(meta.global_step).__name__}"
        )
    for name in ("model_time_seconds", "dt"):
        value = getattr(meta, name)
        if type(value) is not float or not math.isfinite(value):
            raise TypeError(
                f"checkpoint meta.{name} must be a finite exact float, "
                f"got {value!r}"
            )
    if type(meta.mom_forcing_out_ab) is not bool:
        raise TypeError(
            "checkpoint meta.mom_forcing_out_ab must be an exact bool, "
            f"got {type(meta.mom_forcing_out_ab).__name__}"
        )
    if type(meta.physics_flags) is not dict or any(
        type(key) is not str for key in meta.physics_flags
    ):
        raise TypeError("checkpoint meta.physics_flags must be a dict with string keys")
    require_nonempty_fingerprint(
        meta.config_fingerprint, path="checkpoint meta.config_fingerprint"
    )
    if type(meta.conventions) is not dict or any(
        type(key) is not str or type(value) is not str
        for key, value in meta.conventions.items()
    ):
        raise TypeError("checkpoint meta.conventions must be a dict[str, str]")
    for name in ("git_head", "created_utc"):
        if type(getattr(meta, name)) is not str:
            raise TypeError(f"checkpoint meta.{name} must be an exact string")
    for name in ("permanent", "requires_grad"):
        if type(getattr(meta, name)) is not bool:
            raise TypeError(f"checkpoint meta.{name} must be an exact bool")
    if type(meta.state_keys) is not tuple or any(
        type(key) is not str for key in meta.state_keys
    ):
        raise TypeError("checkpoint meta.state_keys must be a tuple of strings")
    if len(meta.state_keys) != len(set(meta.state_keys)):
        raise ValueError("checkpoint meta.state_keys must not contain duplicates")


@dataclass(frozen=True)
class ModelCheckpoint:
    """A complete, self-consistent snapshot of a running model.

    ``prognostic`` maps field name -> CPU tensor; ``step_state`` is the
    closed restart state; ``meta`` is the validated header.
    """

    prognostic: dict[str, torch.Tensor]
    step_state: dict[str, Any]
    meta: CheckpointMeta

    def __post_init__(self) -> None:
        missing = [f for f in PROGNOSTIC_FIELDS if f not in self.prognostic]
        if missing:
            raise ValueError(f"ModelCheckpoint missing prognostic fields: {missing}")
