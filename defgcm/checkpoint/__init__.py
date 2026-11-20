"""LOCOMgcm checkpoint mechanism.

A decoupled three-layer design:

    schedule  (when)  -> capture (what) -> persist (how)

Public surface — most users only need ``CheckpointManager``:

    from defgcm.checkpoint import CheckpointManager
    mgr = CheckpointManager.from_run_config(run_cfg, run_dir)

Lower-level pieces (capture/restore a state by hand, list checkpoints,
build a custom schedule) are exported too for tests and tooling.

Name collision — ``capture`` (footgun): this package re-exports the *function*
``capture`` (from ``capture.py``), whose name matches the ``capture`` submodule.
The re-exported function shadows the submodule at the package-attribute level,
so ALL of the following bind the FUNCTION, not the module::

    from defgcm.checkpoint import capture            # -> function
    from defgcm.checkpoint import capture as cm      # -> function
    import defgcm.checkpoint.capture as cm           # -> function
    defgcm.checkpoint.capture                        # -> function

To obtain the ``capture`` *submodule* object, bypass attribute lookup::

    import importlib
    capture_module = importlib.import_module("defgcm.checkpoint.capture")
    # or: import sys; sys.modules["defgcm.checkpoint.capture"]

``capture(..., detach=False)`` and ``restore(..., detach=False)`` keep state
and history derivatives. ``recompute.checkpointed_rollout`` recomputes
differentiable production windows from segment heads: ``head`` keeps the
codec's content on the device and checks it once per window, ``store`` holds
the heads between the passes (device slots, then exact-size page-locked host
slots behind a copy stream).
"""

from __future__ import annotations

from .capture import (
    capture,
    config_fingerprint,
    export_step_state,
    import_step_state,
    physics_flags,
    restore,
)
from .manager import CheckpointManager, ResumePlan
from .persistence import (
    CheckpointRef,
    find_checkpoint,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint_atomic,
)
from .schedule import (
    CheckpointSchedule,
    CompositeSchedule,
    ExplicitSchedule,
    IntervalSchedule,
    WindowSchedule,
    build_schedule,
    resolve_trigger_steps,
)
from .state import RESTART_STATE_KEYS_V2, RESTART_STATE_KEYS_V3, RESTART_STATE_KEYS_V5, initialize_step_state
from .store import CheckpointStore
from .migration import migrate_checkpoint_v3_to_current
from .types import (
    CHECKPOINT_SCHEMA_VERSION,
    PROGNOSTIC_FIELDS,
    CheckpointMeta,
    CheckpointSchemaVersionError,
    LegacyCheckpointSchemaVersionError,
    ModelCheckpoint,
    UnknownCheckpointSchemaVersionError,
)
from .writer import (
    AsyncCheckpointWriter,
    SyncCheckpointWriter,
    make_writer,
)

__all__ = [
    # facade
    "CheckpointManager",
    "ResumePlan",
    # capture/restore
    "capture",
    "restore",
    "export_step_state",
    "import_step_state",
    "physics_flags",
    "config_fingerprint",
    "initialize_step_state",
    "RESTART_STATE_KEYS_V2",
    "RESTART_STATE_KEYS_V3",
    "RESTART_STATE_KEYS_V5",
    "migrate_checkpoint_v3_to_current",
    # segment heads between the passes of a recomputed window
    "CheckpointStore",
    # types
    "ModelCheckpoint",
    "CheckpointMeta",
    "CHECKPOINT_SCHEMA_VERSION",
    "PROGNOSTIC_FIELDS",
    "CheckpointSchemaVersionError",
    "LegacyCheckpointSchemaVersionError",
    "UnknownCheckpointSchemaVersionError",
    # schedule
    "CheckpointSchedule",
    "IntervalSchedule",
    "WindowSchedule",
    "ExplicitSchedule",
    "CompositeSchedule",
    "build_schedule",
    "resolve_trigger_steps",
    # persistence
    "save_checkpoint_atomic",
    "load_checkpoint",
    "list_checkpoints",
    "latest_checkpoint",
    "find_checkpoint",
    "CheckpointRef",
    # writer
    "AsyncCheckpointWriter",
    "SyncCheckpointWriter",
    "make_writer",
]
