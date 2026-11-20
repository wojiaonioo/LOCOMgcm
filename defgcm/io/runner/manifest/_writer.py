"""ManifestWriter: atomic flock + rename.

Lock the STABLE `.manifest.lock` file (NOT the .tmp), then write to .tmp, then
atomic rename.
"""

from __future__ import annotations

import fcntl
import os
import socket
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from ._schema import StageStatus, validate_run_level_status


def _to_plain_python(obj: Any) -> Any:
    """Recursively convert numpy types / pathlib paths / enums to plain
    Python types so yaml.safe_dump doesn't fail on np.str_, np.int64, Path,
    or Enum instances coming from consumer extras.
    """
    try:
        import numpy as _np
        np_str = _np.str_
        np_generic = _np.generic
    except Exception:
        np_str = None
        np_generic = None

    if isinstance(obj, dict):
        return {_to_plain_key(k): _to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_python(v) for v in obj]
    if np_str is not None and isinstance(obj, np_str):
        return str(obj)
    if np_generic is not None and isinstance(obj, np_generic):
        # np.int64, np.float64, np.bool_ etc. → native via .item()
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _to_plain_key(obj: Any) -> str:
    plain = _to_plain_python(obj)
    if isinstance(plain, str):
        return plain
    return str(plain)


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists on THIS host.

    Used only for orphan recovery. Pids are per-host, so callers MUST first
    confirm the recorded hostname matches the current host before trusting
    this result — a pid alive on another node looks dead here.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but is owned by another user
    except OSError:
        return True  # uncertain → assume alive (never falsely fail a stage)
    return True


class ManifestWriter:
    """Writer for `${OUTPUT_ROOT}/${run_id}/manifest.yaml`.

    Concurrency model:
      - All write/read serialized via flock on `${run_dir}/.manifest.lock`
        (stable path, NOT the tmp file)
      - Write itself uses `.yaml.tmp` + atomic rename

    Context-manager friendly:
      with ManifestWriter.open(run_id, output_dir) as mw:
          mw.mark_running("stage1a_prepare_common")
          ...
          mw.mark_completed("stage1a_prepare_common")
    """

    def __init__(self, run_dir: Path, run_id: str, manifest_yaml: dict | None = None) -> None:
        self._run_dir = run_dir
        self._run_id = run_id
        self._path = run_dir / "manifest.yaml"
        self._lock_path = run_dir / ".manifest.lock"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path.touch(exist_ok=True)
        if manifest_yaml is not None:
            self._cache = manifest_yaml
        elif self._path.exists():
            # Read under the shared flock — a concurrent writer's .tmp+rename
            # is atomic, but acquiring LOCK_SH here also serialises against
            # any writer mid-_atomic_write, and we retry on a transient
            # parse error (defensive: rename should make this impossible,
            # but a stale/partial file from an interrupted run can exist).
            self._cache = None
            for _attempt in range(3):
                try:
                    with open(self._lock_path, "r") as lock_f:
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_SH)
                        try:
                            with open(self._path) as f:
                                self._cache = yaml.safe_load(f) or {}
                        finally:
                            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                    break
                except yaml.YAMLError:
                    import time as _t
                    _t.sleep(0.2)
            if self._cache is None:
                # Last attempt without catching — surface the real error
                with open(self._path) as f:
                    self._cache = yaml.safe_load(f) or {}
            # Heal stages left RUNNING by a process that died before reaching
            # mark_completed/mark_failed (crash, OOM, SLURM kill). Hostname-
            # guarded so a stage genuinely running on another node is never
            # falsely failed (pids are per-host).
            self._recover_orphaned_running()
        else:
            self._cache = {
                "schema_version": "0.1.0",
                "run_id": run_id,
                "created_at": datetime.utcnow().isoformat(),
                "last_updated_at": datetime.utcnow().isoformat(),
                "overall_status": "PENDING",
                "stages": {},
            }
            self._atomic_write()

    @classmethod
    def open(cls, run_id: str, output_dir: Path, *, dir_name: str | None = None) -> "ManifestWriter":
        """Open manifest at `<output_dir>/<dir_name>/manifest.yaml`.

        `dir_name` overrides `run_id` for legacy layout where run_dir name =
        experiment_name. Falls back to run_id when not provided.
        """
        run_dir = output_dir / (dir_name or run_id)
        return cls(run_dir, run_id)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def _artifact_path_for_manifest(self, path: Any) -> str | None:
        if path is None:
            return None
        artifact_path = Path(path)
        if artifact_path.is_absolute():
            try:
                return str(
                    artifact_path.resolve(strict=False).relative_to(
                        self._run_dir.resolve(strict=False)
                    )
                )
            except ValueError:
                return str(artifact_path)
        if any(part == ".." for part in artifact_path.parts):
            raise ValueError("manifest artifact paths must not escape the run directory")
        return str(artifact_path)

    def _atomic_write(self, current_stage: str | None = None) -> None:
        """Lock STABLE .manifest.lock, NOT the tmp file.

        When ``current_stage`` is provided, re-read disk *under the
        exclusive lock* and merge stage entries authored by other processes
        before writing.  This makes the read-modify-write atomic and closes
        the TOCTOU window that could lose stage entries written by a
        concurrent process (e.g. stage2 on the GPU node vs stage3 on the CPU node).
        """
        self._cache["last_updated_at"] = datetime.utcnow().isoformat()
        with open(self._lock_path, "r+") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                # Merge under the exclusive lock so we never clobber
                # stage entries written by a concurrent writer between our
                # last _merge call and this lock acquisition.
                if current_stage is not None and self._path.exists():
                    try:
                        with open(self._path) as f:
                            disk = yaml.safe_load(f) or {}
                        disk_stages = disk.get("stages", {}) or {}
                        cache_stages = self._cache.setdefault("stages", {})
                        for stage_name, disk_entry in disk_stages.items():
                            if stage_name == current_stage:
                                continue
                            # Always prefer the disk entry for stages we are
                            # not currently authoring — it reflects any writes
                            # made by other processes after our cache snapshot.
                            cache_stages[stage_name] = disk_entry
                    except (yaml.YAMLError, OSError):
                        pass
                sanitized = _to_plain_python(self._cache)
                tmp_path = self._path.with_suffix(".yaml.tmp")
                with open(tmp_path, "w") as f:
                    yaml.safe_dump(sanitized, f, sort_keys=False)
                    f.flush()
                    os.fsync(f.fileno())
                tmp_path.rename(self._path)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _merge_other_stages_from_disk(self, current_stage: str) -> None:
        """Pre-flight cache warm-up before acquiring the write lock.

        This remains a best-effort unlocked read to populate self._cache with
        other stages' state before we mutate the current stage.  It reduces
        the chance of clobbering concurrent writes by shrinking the window
        between cache snapshot and lock acquisition.  The authoritative merge
        happens *under the exclusive lock* inside ``_atomic_write``,
        so this unlocked read is NOT the last safety net — it is only a
        performance optimisation that avoids holding the lock during the read.

        The current stage is always authored locally so we never let disk
        overwrite it.
        """
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                disk = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError):
            # Treat unreadable disk state as "nothing extra to merge";
            # the locked merge in _atomic_write will retry.
            return
        disk_stages = disk.get("stages", {}) or {}
        cache_stages = self._cache.setdefault("stages", {})
        for stage_name, disk_entry in disk_stages.items():
            if stage_name == current_stage:
                continue
            cache_stages[stage_name] = disk_entry

    def _recover_orphaned_running(self) -> None:
        """Transition RUNNING stages whose process is provably dead → FAILED.

        A stage records ``(pid, hostname)`` at mark_running time. If, on
        re-open, a RUNNING entry's hostname matches this host but its pid is no
        longer alive, the process died (crash / SLURM kill / OOM) before
        reaching a terminal status — so we mark it FAILED, making the status
        honest and the stage retriable.

        Cross-host safety: pids are per-host. When the recorded hostname
        differs from this host (or is absent), liveness cannot be inferred from
        the pid, so the entry is left untouched — a stage genuinely running on
        another node must never be falsely failed.
        """
        this_host = socket.gethostname()
        orphaned: list[tuple[str, int]] = []
        for name, entry in ((self._cache or {}).get("stages", {}) or {}).items():
            if not isinstance(entry, dict):
                continue
            if entry.get("status") != StageStatus.RUNNING.value:
                continue
            host = entry.get("hostname")
            pid = entry.get("pid")
            if not host or host != this_host or pid is None:
                continue
            if _pid_alive(int(pid)):
                continue
            orphaned.append((name, int(pid)))
        for name, pid in orphaned:
            self.mark_failed(
                name,
                error=(
                    f"orphaned: status was RUNNING but pid {pid} on {this_host} "
                    f"is no longer alive (crash/kill before completion); "
                    f"recovered at manifest open"
                ),
            )

    # Stage status mutators

    def _set_stage_status(self, stage: str, status: StageStatus, **kwargs: Any) -> None:
        self._merge_other_stages_from_disk(current_stage=stage)
        stages = self._cache.setdefault("stages", {})
        entry = stages.setdefault(stage, {})
        entry["status"] = status.value
        for k, v in kwargs.items():
            entry[k] = v
        self._atomic_write(current_stage=stage)

    def mark_pending(self, stage: str) -> None:
        self._set_stage_status(stage, StageStatus.PENDING)

    def mark_running(self, stage: str) -> None:
        # Clear stale artifacts from a prior run of this stage — record_artifact
        # appends, so without this a re-run would accumulate phantom entries
        # (e.g. artifacts whose source files no longer exist after a yaml edit).
        stages = self._cache.setdefault("stages", {})
        entry = stages.setdefault(stage, {})
        entry["artifacts"] = []
        entry.pop("error", None)
        self._set_stage_status(
            stage,
            StageStatus.RUNNING,
            started_at=datetime.utcnow().isoformat(),
            pid=os.getpid(),
            hostname=socket.gethostname(),
        )

    def mark_completed(self, stage: str, wallclock_seconds: float | None = None) -> None:
        kwargs: dict[str, Any] = {
            "completed_at": datetime.utcnow().isoformat(),
            "pid": None,
        }
        if wallclock_seconds is not None:
            kwargs["wallclock_seconds"] = wallclock_seconds
        self._set_stage_status(stage, StageStatus.COMPLETED, **kwargs)

    def mark_failed(self, stage: str, error: str | None = None) -> None:
        self._set_stage_status(
            stage,
            StageStatus.FAILED,
            completed_at=datetime.utcnow().isoformat(),
            pid=None,
            error=error,
        )

    def mark_stale(self, stage: str) -> None:
        """Downstream invalidation via STALE status."""
        self._set_stage_status(stage, StageStatus.STALE)

    def set_overall_status(self, status: str) -> None:
        """Set the run-level ``overall_status`` (PENDING/RUNNING/COMPLETED/FAILED).

        Called by the orchestrator at pipeline start/end. Re-reads disk first
        (via the merge path) so a concurrent stage write is not clobbered.
        The final merge is performed under the exclusive lock in _atomic_write.
        """
        status = validate_run_level_status(status)
        self._merge_other_stages_from_disk(current_stage="__overall__")
        self._cache["overall_status"] = status
        # Pass current_stage="__overall__" so the locked merge in _atomic_write
        # absorbs any stage updates written by concurrent processes without
        # clobbering them.
        self._atomic_write(current_stage="__overall__")

    def record_artifact(self, stage: str, artifact_ref: Any) -> None:
        self._merge_other_stages_from_disk(current_stage=stage)
        stages = self._cache.setdefault("stages", {})
        entry = stages.setdefault(stage, {"status": "RUNNING", "artifacts": []})
        artifacts = entry.setdefault("artifacts", [])
        # Serialize ArtifactRef-like to dict
        if hasattr(artifact_ref, "name"):
            art_dict = {
                "name": artifact_ref.name,
                "path": self._artifact_path_for_manifest(artifact_ref.path),
                "artifact_kind": (
                    artifact_ref.artifact_kind.value
                    if hasattr(artifact_ref.artifact_kind, "value")
                    else str(artifact_ref.artifact_kind)
                ),
                "bundle_hash": artifact_ref.bundle_hash,
                "schema_version": artifact_ref.schema_version,
                "extras": dict(artifact_ref.extras) if hasattr(artifact_ref, "extras") else {},
            }
        else:
            art_dict = dict(artifact_ref)
            if "path" in art_dict:
                art_dict["path"] = self._artifact_path_for_manifest(art_dict["path"])
        artifacts.append(art_dict)
        self._atomic_write(current_stage=stage)

    def get_stage_status(self, stage: str) -> str | None:
        return self._cache.get("stages", {}).get(stage, {}).get("status")

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        pass  # writes are eager; nothing to close
