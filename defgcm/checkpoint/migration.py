"""Explicit, validated migration of the disk-v3 restart format."""

from dataclasses import replace
from copy import deepcopy

from .state import _host_clone, validate_checkpoint_structure
from .types import ModelCheckpoint


def migrate_checkpoint_v3_to_current(checkpoint):
    from .types import CHECKPOINT_SCHEMA_VERSION

    validate_checkpoint_structure(checkpoint, legacy_v3=True)
    record = {"source_version": 3, "target_version": CHECKPOINT_SCHEMA_VERSION,
              "boundary_history": {"valid": False, "reason": "missing_legacy_boundary_history"}}
    candidate = ModelCheckpoint(
        prognostic=_host_clone(checkpoint.prognostic, detach=False, path="prognostic"),
        step_state=_host_clone(checkpoint.step_state, detach=False, path="step_state"),
        meta=replace(deepcopy(checkpoint.meta), schema_version=CHECKPOINT_SCHEMA_VERSION,
                     migration_records=(record,)))
    validate_checkpoint_structure(candidate)
    return candidate
