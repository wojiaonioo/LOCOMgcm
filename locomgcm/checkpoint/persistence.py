"""Checkpoint persistence — *how* a checkpoint is stored and recovered.

Storage contract: a checkpoint is complete only when its ``_SUCCESS`` marker
exists; a bare atomic rename is not sufficient:

    <root>/
    ├── manifest.jsonl                 # append-only log, one line / checkpoint
    ├── latest -> ckpt_stepNNNNNNNNNN  # symlink to newest *complete* ckpt
    └── ckpt_step0000000123/
        ├── payload.pt                 # torch.save: prognostic + step_state
        ├── meta.json                  # human-readable header (CheckpointMeta)
        └── _SUCCESS                    # written LAST; presence == complete

Write sequence (so a crash never leaves a half checkpoint that loads):
  1. write payload + meta into ``<dir>.tmp.<pid>``
  2. fsync files + the tmp dir
  3. atomically rename tmp dir -> final dir
  4. touch ``_SUCCESS``
  5. repoint ``latest`` symlink atomically
  6. append one manifest.jsonl line

Payload format is ``torch.save`` of a plain dict.  Rationale: exact
float64 round-trip, native handling of the nested tensor/dict/tuple/scalar
structure inside ``step_state``, and no extra dependency.  All inputs are
self-produced (never untrusted), so pickle-safety is a non-issue here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .types import CheckpointMeta, ModelCheckpoint
from .state import validate_checkpoint_structure

_CKPT_DIR_RE = re.compile(r"^ckpt_step(\d{10})$")
_PAYLOAD = "payload.pt"
_META = "meta.json"
_SUCCESS = "_SUCCESS"
_MANIFEST = "manifest.jsonl"
_LATEST = "latest"


def checkpoint_dirname(global_step: int) -> str:
    """``ckpt_step0000000123`` — 10-digit zero pad (matches MIT Eta.* naming)."""
    return f"ckpt_step{int(global_step):010d}"


def _fsync_path(path: Path) -> None:
    """Best-effort fsync of a file or directory (durability of the rename)."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def save_checkpoint_atomic(root: Path, ckpt: ModelCheckpoint) -> Path:
    """Persist ``ckpt`` under ``root`` atomically; return the final dir.

    Caller is expected to have already moved tensors to host memory
    (the capture/snapshot phase) so this can run on a background thread.
    """
    validate_checkpoint_structure(ckpt)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / checkpoint_dirname(ckpt.meta.global_step)
    tmp = root / f"{final.name}.tmp.{os.getpid()}"
    if tmp.exists():  # stale tmp from a previous crashed attempt
        _rmtree(tmp)
    tmp.mkdir(parents=True)

    payload = {
        "prognostic": ckpt.prognostic,
        "step_state": ckpt.step_state,
        "meta": ckpt.meta.to_dict(),
    }
    payload_path = tmp / _PAYLOAD
    torch.save(payload, payload_path)
    _fsync_path(payload_path)

    meta_path = tmp / _META
    meta_path.write_text(json.dumps(ckpt.meta.to_dict(), indent=2, sort_keys=True))
    _fsync_path(meta_path)
    _fsync_path(tmp)

    if final.exists():  # overwrite (e.g. re-running the same step) — replace
        _rmtree(final)
    os.replace(tmp, final)
    _fsync_path(root)

    (final / _SUCCESS).write_text("")
    _fsync_path(final)

    _repoint_latest(root, final)
    _append_manifest(root, ckpt.meta, final)
    return final


def _repoint_latest(root: Path, final: Path) -> None:
    link = root / _LATEST
    tmp_link = root / f"{_LATEST}.tmp.{os.getpid()}"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(final.name, tmp_link)  # relative target
    os.replace(tmp_link, link)


def _append_manifest(root: Path, meta: CheckpointMeta, final: Path) -> None:
    line = json.dumps(
        {**meta.to_dict(), "dir": final.name}, sort_keys=True
    )
    with open(root / _MANIFEST, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


@dataclass(frozen=True)
class CheckpointRef:
    """A discovered, *complete* checkpoint on disk."""

    global_step: int
    path: Path

    @property
    def is_complete(self) -> bool:
        return (self.path / _SUCCESS).exists()


def list_checkpoints(root: Path) -> list[CheckpointRef]:
    """All complete checkpoints under ``root``, ascending by step."""
    root = Path(root)
    if not root.exists():
        return []
    refs: list[CheckpointRef] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = _CKPT_DIR_RE.match(child.name)
        if not m:
            continue
        ref = CheckpointRef(global_step=int(m.group(1)), path=child)
        if ref.is_complete:
            refs.append(ref)
    refs.sort(key=lambda r: r.global_step)
    return refs


def latest_checkpoint(root: Path) -> CheckpointRef | None:
    refs = list_checkpoints(root)
    return refs[-1] if refs else None


def find_checkpoint(root: Path, global_step: int) -> CheckpointRef | None:
    for ref in list_checkpoints(root):
        if ref.global_step == int(global_step):
            return ref
    return None


def load_checkpoint(path: Path, *, map_location: Any = "cpu") -> ModelCheckpoint:
    """Load a checkpoint directory.  Raises if it is not complete."""
    path = Path(path)
    if not (path / _SUCCESS).exists():
        raise RuntimeError(
            f"checkpoint {path} has no _SUCCESS marker — incomplete or corrupt"
        )
    payload = torch.load(path / _PAYLOAD, map_location=map_location, weights_only=False)
    if type(payload) is not dict:
        raise TypeError(
            f"checkpoint payload must be a dict, got {type(payload).__name__}"
        )
    expected_fields = frozenset({"prognostic", "step_state", "meta"})
    payload_fields = frozenset(payload)
    if payload_fields != expected_fields:
        missing = sorted(expected_fields - payload_fields)
        extra = sorted(payload_fields - expected_fields)
        raise ValueError(
            "checkpoint payload fields must match exactly; "
            f"missing={missing}, extra={extra}"
        )
    legacy_v3 = (type(payload["meta"]) is dict
                 and type(payload["meta"].get("schema_version")) is int
                 and payload["meta"]["schema_version"] == 3)
    meta = CheckpointMeta.from_dict(payload["meta"], legacy_v3=legacy_v3)
    ckpt = ModelCheckpoint(
        prognostic=payload["prognostic"],
        step_state=payload["step_state"],
        meta=meta,
    )
    if legacy_v3:
        from .migration import migrate_checkpoint_v3_to_current

        return migrate_checkpoint_v3_to_current(ckpt)
    validate_checkpoint_structure(ckpt)
    return ckpt


def prune_checkpoints(
    root: Path,
    *,
    keep_last: int | None,
    permanent_steps: set[int],
) -> list[int]:
    """Rolling cleanup: keep the newest ``keep_last`` + any permanent ones.

    Returns the list of pruned steps.  Never deletes the single newest
    checkpoint, and always leaves at least one complete checkpoint alive
    (MITgcm rolling ckptA/ckptB safety).
    """
    if keep_last is None:
        return []
    refs = list_checkpoints(root)
    if len(refs) <= keep_last:
        return []
    # Candidates for deletion: oldest first, excluding the protected tail
    # and any permanent steps.
    protected = set(r.global_step for r in refs[-keep_last:])
    pruned: list[int] = []
    for ref in refs[:-keep_last] if keep_last > 0 else refs[:-1]:
        if ref.global_step in protected or ref.global_step in permanent_steps:
            continue
        _rmtree(ref.path)
        pruned.append(ref.global_step)
    return pruned
