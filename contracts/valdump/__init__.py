"""Environment-neutral producer and loader for validation dumps.

Callers retain their result objects and provide detached host NumPy arrays
when the tap is armed. A candidate call is serialized in memory, including
the NPY headers, and checked against the byte cap before any payload file is
opened. An admitted call is published as one complete unit.

The manifest records instrumented source paths, the tap module and an advisory
Git commit. The submitting source revision and its uncommitted differences
identify the code used for a run. This module depends only on the Python
standard library and NumPy.
"""

from __future__ import annotations

import copy
import io
import json
import os
import re
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


_STEP_FILE = re.compile(r"^step(\d{5})_(.+)\.npy$")
_RESERVED_MANIFEST_KEYS = frozenset(
    {
        "case_tag",
        "case_yaml",
        "git_commit",
        "git_commit_status",
        "created_utc",
        "steps_dumped",
        "dump_stride",
        "shape",
        "dtype",
        "provenance",
        "valdump",
        "status",
        "disabled_reason",
        "extras",
    }
)


@dataclass(frozen=True)
class TapSpec:
    """Static declaration of one operator dump interface.

    ``variables`` and ``static_fields`` are exact filename stems. ``shape_field``
    selects the per-call array reported by the manifest’s top-level ``shape`` key.
    """

    variables: tuple[str, ...]
    static_fields: tuple[str, ...] = ()
    shape_field: str | None = None
    source_files: tuple[str | os.PathLike[str], ...] = ()
    max_bytes: int = 20 * 1024**3
    stride: int = 1
    keep_first: int = 5
    marker: str = "VALDUMP"
    format_version: str = "1.2"

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("TapSpec.variables must not be empty")
        names = (*self.variables, *self.static_fields)
        if len(set(names)) != len(names):
            raise ValueError("variable and static field names must be unique")
        if any(not name or "/" in name or "\\" in name for name in names):
            raise ValueError("dump field names must be non-empty filename stems")
        if self.shape_field is not None and self.shape_field not in self.variables:
            raise ValueError("shape_field must name a declared variable")
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if self.stride < 1:
            raise ValueError("stride must be at least one")
        if self.keep_first < 0:
            raise ValueError("keep_first must be non-negative")


@dataclass(frozen=True)
class LoadedDump:
    """Manifest and arrays loaded from one valdump directory."""

    root: Path
    manifest: dict[str, Any]
    calls: dict[int, dict[str, np.ndarray]]
    statics: dict[str, np.ndarray]


def _companion_prefix(env_var: str) -> str:
    return env_var[:-4] if env_var.endswith("_DIR") else env_var


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def _env_max_bytes(prefix: str, default: int) -> int:
    raw_bytes = os.environ.get(f"{prefix}_MAX_BYTES")
    if raw_bytes not in (None, ""):
        return int(raw_bytes)
    raw_gb = os.environ.get(f"{prefix}_MAX_GB")
    if raw_gb in (None, ""):
        return default
    return int(float(raw_gb) * 1024**3)


def _git_dir(repo_root: Path) -> Path | None:
    marker = repo_root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        text = marker.read_text(encoding="utf-8", errors="replace").strip()
        if text.startswith("gitdir:"):
            path = Path(text.split(":", 1)[1].strip())
            return path if path.is_absolute() else (repo_root / path).resolve()
    return None


def _find_repo_root(start: Path) -> Path | None:
    candidate = start if start.is_dir() else start.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _read_advisory_git_commit(start: Path) -> str | None:
    repo_root = _find_repo_root(start)
    if repo_root is None:
        return None
    git_dir = _git_dir(repo_root)
    if git_dir is None:
        return None
    try:
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.split(":", 1)[1].strip()
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text(encoding="ascii").strip() or None
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="ascii").splitlines():
                if line and not line.startswith(("#", "^")):
                    value, name = line.split(" ", 1)
                    if name == ref:
                        return value
    except (OSError, ValueError):
        return None
    return None


def _source_record(path: Path, *, relative_to: Path | None) -> dict[str, str]:
    # strict=True keeps the old failure on a declared source that does not exist.
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(relative_to)) if relative_to else str(resolved)
    except ValueError:
        display = str(resolved)
    return {"path": display}


def _serialize_array(name: str, value: Any) -> tuple[bytes, tuple[int, ...]]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(value).__name__}")
    if value.dtype != np.float64:
        raise TypeError(f"{name} must be float64 before capture, got {value.dtype}")
    detached_copy = np.array(value, dtype=np.float64, copy=True, order="K")
    stream = io.BytesIO()
    np.save(stream, detached_copy, allow_pickle=False)
    return stream.getvalue(), tuple(detached_copy.shape)


class DumpTap:
    """An environment-gated, observation-only dump producer."""

    def __init__(
        self,
        *,
        root: Path | None,
        spec: TapSpec,
        env_var: str,
        case_tag: str | None,
        case_yaml: str | None,
        stride: int,
        max_bytes: int,
        nsteps_env: str | None,
        rank: int,
    ) -> None:
        self._root = root
        self.spec = spec
        self._env_var = env_var
        self._case_tag = case_tag
        self._case_yaml = case_yaml
        self._stride = stride
        self._max_bytes = max_bytes
        self._nsteps_env = nsteps_env
        self._rank = rank
        self._lock = threading.RLock()
        self._armed = False
        self._disabled = False
        self._disabled_reason: str | None = None
        self._warning_emitted = False
        self._calls_seen = 0
        self._steps_dumped: list[int] = []
        self._bytes_written = 0
        self._truncated_at_call: int | None = None
        self._admission_checks = 0
        self._admission_rejections = 0
        self._shape: list[int] | None = None
        self._variable_shapes_by_call: dict[str, dict[str, list[int]]] = {}
        self._static_shapes: dict[str, list[int]] = {}
        self._metadata: dict[str, Any] = {}
        self._provenance: dict[str, Any] | None = None
        self._git_commit: str | None = None
        self._created_utc: str | None = None

    @classmethod
    def from_env(cls, env_var: str, *, spec: TapSpec) -> DumpTap:
        raw = os.environ.get(env_var)
        prefix = _companion_prefix(env_var)
        root: Path | None = None
        rank = int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0")
        if raw:
            root = Path(raw).expanduser().absolute()
            if rank != 0:
                root /= f"rank{rank:02d}"
        stride = max(1, _env_int(f"{prefix}_STRIDE", spec.stride))
        max_bytes = _env_max_bytes(prefix, spec.max_bytes)
        if max_bytes < 0:
            raise ValueError(f"{prefix}_MAX_BYTES/MAX_GB must be non-negative")
        case_tag = os.environ.get(f"{prefix}_CASE_TAG")
        if case_tag is None and raw:
            case_tag = Path(raw).expanduser().absolute().name
        return cls(
            root=root,
            spec=spec,
            env_var=env_var,
            case_tag=case_tag,
            case_yaml=os.environ.get(f"{prefix}_CASE_YAML"),
            stride=stride,
            max_bytes=max_bytes,
            nsteps_env=os.environ.get(f"{prefix}_NSTEPS"),
            rank=rank,
        )

    @property
    def enabled(self) -> bool:
        return self._root is not None and not self._disabled

    def _ensure_identity(self) -> None:
        if self._provenance is not None:
            return
        tap_path = Path(__file__).resolve()
        repo_root = _find_repo_root(tap_path)
        source_records = [
            _source_record(Path(path), relative_to=repo_root)
            for path in self.spec.source_files
        ]
        self._provenance = {
            "instrumented_sources": source_records,
            "tap_module": _source_record(tap_path, relative_to=repo_root),
        }
        self._git_commit = _read_advisory_git_commit(tap_path)
        self._created_utc = datetime.now(timezone.utc).isoformat()

    def _update_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        if metadata is None:
            return
        overlap = _RESERVED_MANIFEST_KEYS.intersection(metadata)
        if overlap:
            raise ValueError(f"metadata cannot replace reserved manifest keys: {sorted(overlap)}")
        copied = copy.deepcopy(dict(metadata))
        json.dumps(copied)
        if self._metadata and copied != self._metadata:
            raise ValueError("run-level valdump metadata changed after the first capture")
        self._metadata = copied

    def _should_dump(self, sequence: int) -> bool:
        return sequence <= self.spec.keep_first or sequence % self._stride == 0

    def _admit(self, payload_bytes: int) -> bool:
        self._admission_checks += 1
        return self._bytes_written + payload_bytes <= self._max_bytes

    def _publish_payload(self, payload: Mapping[str, bytes]) -> None:
        assert self._root is not None
        self._root.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        staged: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            for name, content in payload.items():
                final = self._root / name
                temporary = self._root / f".{name}.{token}.tmp"
                temporary.write_bytes(content)
                staged.append((temporary, final))
            for temporary, final in staged:
                os.replace(temporary, final)
                published.append(final)
        except Exception:
            for temporary, _ in staged:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            for path in published:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def _manifest(self) -> dict[str, Any]:
        calls_per_step = None
        if self._nsteps_env:
            try:
                calls_per_step = self._calls_seen / float(self._nsteps_env)
            except (TypeError, ValueError, ZeroDivisionError):
                calls_per_step = None
        pressure_shapes = {
            sequence: shapes["pressure_dbar"]
            for sequence, shapes in self._variable_shapes_by_call.items()
            if "pressure_dbar" in shapes
        }
        pressure_shape = next(iter(pressure_shapes.values()), None)
        manifest: dict[str, Any] = {
            "case_tag": self._case_tag,
            "case_yaml": self._case_yaml,
            "git_commit": self._git_commit,
            "git_commit_status": "advisory",
            "created_utc": self._created_utc,
            "steps_dumped": list(self._steps_dumped),
            "dump_stride": self._stride,
            "shape": self._shape,
            "dtype": "float64",
            "provenance": copy.deepcopy(self._provenance),
            "valdump": {
                "format_version": self.spec.format_version,
                "env_var": self._env_var,
                "variables": list(self.spec.variables),
                "static_fields": list(self.spec.static_fields),
                "shape_field": self.spec.shape_field,
            },
            "status": "disabled" if self._disabled else "armed",
            "disabled_reason": self._disabled_reason,
            "extras": {
                "calls_seen": self._calls_seen,
                "calls_dumped": len(self._steps_dumped),
                "calls_per_step": calls_per_step,
                "calls_per_step_note": (
                    "calls_seen / nsteps_env; meaningful only in the final manifest"
                ),
                "nsteps_env": self._nsteps_env,
                "rank": self._rank,
                "bytes_written": self._bytes_written,
                "max_bytes": self._max_bytes,
                "truncated_at_call": self._truncated_at_call,
                "admission_checks": self._admission_checks,
                "admission_rejections": self._admission_rejections,
                "variable_shapes_by_call": copy.deepcopy(
                    self._variable_shapes_by_call
                ),
                "static_shapes": copy.deepcopy(self._static_shapes),
                "pressure_shape": copy.deepcopy(pressure_shape),
                "pressure_shape_by_call": copy.deepcopy(pressure_shapes),
                "wet_mask_shape": copy.deepcopy(self._static_shapes.get("wet_mask")),
                "z_center_shape": copy.deepcopy(self._static_shapes.get("z_center")),
            },
        }
        manifest.update(copy.deepcopy(self._metadata))
        return manifest

    def _write_manifest(self) -> None:
        assert self._root is not None
        self._root.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(self._manifest(), indent=2) + "\n").encode("utf-8")
        temporary = self._root / f".manifest.json.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(content)
            os.replace(temporary, self._root / "manifest.json")
        finally:
            temporary.unlink(missing_ok=True)

    def _disable_for_budget(self, sequence: int, payload_bytes: int) -> None:
        self._admission_rejections += 1
        self._truncated_at_call = sequence
        self._disabled = True
        self._disabled_reason = "byte_budget"
        if not self._warning_emitted:
            sys.stderr.write(
                f"[{self.spec.marker}] byte budget {self._max_bytes} would be exceeded "
                f"by call {sequence} ({self._bytes_written} written + {payload_bytes} "
                "exact serialized bytes); dump disabled\n"
            )
            sys.stderr.flush()
            self._warning_emitted = True

    def capture(
        self,
        result: Any,
        *,
        arrays: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
        statics: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    ) -> Any:
        """Observe one operator call and always return ``result`` unchanged."""

        if not self.enabled:
            return result
        with self._lock:
            if not self.enabled:
                return result
            self._calls_seen += 1
            sequence = self._calls_seen
            if not self._should_dump(sequence):
                if self._armed and sequence % 10 == 0:
                    self._write_manifest()
                return result
            if arrays is None:
                raise ValueError("armed capture requires arrays")
            if callable(arrays):
                arrays = arrays()
            missing = set(self.spec.variables).difference(arrays)
            extra = set(arrays).difference(self.spec.variables)
            if missing or extra:
                raise ValueError(
                    f"per-call fields differ from TapSpec: missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )

            self._ensure_identity()
            if callable(metadata):
                metadata = metadata()
            self._update_metadata(metadata)
            payload: dict[str, bytes] = {}
            call_shapes: dict[str, list[int]] = {}
            for name in self.spec.variables:
                content, shape = _serialize_array(name, arrays[name])
                payload[f"step{sequence:05d}_{name}.npy"] = content
                call_shapes[name] = list(shape)
            shape_field = self.spec.shape_field or self.spec.variables[0]
            candidate_shape = call_shapes[shape_field]

            candidate_static_shapes: dict[str, list[int]] = {}
            if not self._armed:
                if statics is None:
                    statics = {}
                if callable(statics):
                    statics = statics()
                missing_static = set(self.spec.static_fields).difference(statics)
                extra_static = set(statics).difference(self.spec.static_fields)
                if missing_static or extra_static:
                    raise ValueError(
                        "static fields differ from TapSpec: "
                        f"missing={sorted(missing_static)}, extra={sorted(extra_static)}"
                    )
                for name in self.spec.static_fields:
                    content, shape = _serialize_array(name, statics[name])
                    payload[f"{name}.npy"] = content
                    candidate_static_shapes[name] = list(shape)

            payload_bytes = sum(len(content) for content in payload.values())
            if not self._admit(payload_bytes):
                if self._shape is None:
                    self._shape = candidate_shape
                self._disable_for_budget(sequence, payload_bytes)
                self._write_manifest()
                return result

            self._publish_payload(payload)
            if self._shape is None:
                self._shape = candidate_shape
            self._bytes_written += payload_bytes
            self._steps_dumped.append(sequence)
            self._variable_shapes_by_call[str(sequence)] = call_shapes
            if candidate_static_shapes:
                self._static_shapes = candidate_static_shapes
            first_fire = not self._armed
            self._armed = True
            if first_fire:
                sys.stderr.write(
                    f"[{self.spec.marker}] armed dir={self._root} "
                    f"case_tag={self._case_tag} shape={self._shape} "
                    f"stride={self._stride} rank={self._rank}\n"
                )
                sys.stderr.flush()
            self._write_manifest()
            return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "armed": self._armed,
                "disabled_reason": self._disabled_reason,
                "calls_seen": self._calls_seen,
                "calls_dumped": len(self._steps_dumped),
                "bytes_written": self._bytes_written,
                "max_bytes": self._max_bytes,
                "truncated_at_call": self._truncated_at_call,
                "admission_checks": self._admission_checks,
                "admission_rejections": self._admission_rejections,
            }


def load_dump(
    root: str | os.PathLike[str], *, mmap_mode: str | None = None
) -> LoadedDump:
    """Load a valdump manifest and all arrays without importing a compute chassis."""

    root_path = Path(root)
    manifest = json.loads((root_path / "manifest.json").read_text(encoding="utf-8"))
    calls: dict[int, dict[str, np.ndarray]] = {}
    statics: dict[str, np.ndarray] = {}
    for path in sorted(root_path.glob("*.npy")):
        match = _STEP_FILE.match(path.name)
        array = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
        if match is None:
            statics[path.stem] = array
        else:
            sequence = int(match.group(1))
            calls.setdefault(sequence, {})[match.group(2)] = array
    return LoadedDump(root=root_path, manifest=manifest, calls=calls, statics=statics)


__all__ = ["DumpTap", "LoadedDump", "TapSpec", "load_dump"]
