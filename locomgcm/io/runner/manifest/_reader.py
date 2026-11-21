"""ManifestReader: read-only manifest access for prereq validation."""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class ManifestReader:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._path = run_dir / "manifest.yaml"
        self._lock_path = run_dir / ".manifest.lock"
        if not self._path.exists():
            raise FileNotFoundError(f"ManifestReader: no manifest at {self._path}")
        self._lock_path.touch(exist_ok=True)
        # Shared lock, blocking read
        with open(self._lock_path, "r+") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_SH)
            try:
                with open(self._path) as f:
                    self._content = yaml.safe_load(f) or {}
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    @classmethod
    def open(cls, run_id: str, output_dir: Path, *, dir_name: str | None = None) -> "ManifestReader":
        return cls(output_dir / (dir_name or run_id))

    def get_stage(self, name: str) -> dict[str, Any] | None:
        return self._content.get("stages", {}).get(name)

    def get_stage_status(self, name: str) -> str | None:
        s = self.get_stage(name)
        return s.get("status") if s else None

    def get_artifact(self, stage: str, artifact_name: str) -> dict[str, Any] | None:
        s = self.get_stage(stage)
        if not s:
            return None
        for art in s.get("artifacts", []):
            if art.get("name") == artifact_name:
                return art
        return None

    def overall_status(self) -> str:
        return self._content.get("overall_status", "PENDING")


class PrereqError(Exception):
    """Raised when validate_prereqs fails."""


def _resolve_artifact_path(manifest_path: Path, artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def validate_prereqs(
    manifest_path: Path,
    stage: str,
    prereqs: list[str],
    integrity_check: bool = True,
    skip: list[str] | tuple[str, ...] | set[str] | None = None,
    force: bool = False,
    required_artifacts: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Validate prerequisite stages; integrity_check defaults to True.

    Raises PrereqError with remediation hint if any prereq stage is not
    COMPLETED, if a caller-declared required artifact name is absent, or if a
    recorded artifact path is missing.

    : ``skip`` (optional) is the set of prereq stage names to ignore.
    The orchestrator passes ``skip=["stage2_torch_run"]`` to stage4 in
    ``--mitgcm-only`` mode so the missing LOCOMgcm-side prereq does not abort the
    stage4 validation. Skipped names are filtered out before any manifest
    lookup; unknown entries are silently ignored (no error if a stage has
    no such prereq in the first place).

    ``force`` (``--force``): when True, a prereq marked STALE (upstream
    re-ran since this stage last completed) is accepted instead of blocking —
    the operator knowingly proceeds on possibly-outdated upstream output. The
    artifact integrity check still applies. FAILED / PENDING / missing prereqs
    are NOT overridden by force (those are genuine, not just stale).

    ``required_artifacts`` optionally maps prereq stage names to artifact names
    that must be present. The generic manifest layer cannot infer the required
    artifact set for pipeline-configured stages, so callers with fixed stage
    contracts can opt in without breaking dynamic pipeline prerequisites.
    """
    if not manifest_path.exists():
        raise PrereqError(f"Manifest not found at {manifest_path}; run prerequisite stage(s) first")

    with open(manifest_path) as f:
        content = yaml.safe_load(f) or {}

    stages = content.get("stages", {})
    errors: list[str] = []
    skip_set = set(skip or ())
    required = {
        prereq: set(names)
        for prereq, names in (required_artifacts or {}).items()
    }

    for prereq in prereqs:
        if prereq in skip_set:
            continue
        entry = stages.get(prereq)
        if entry is None:
            errors.append(f"  ✗ {prereq}: not present in manifest (must be COMPLETED)")
            continue
        status = entry.get("status")
        if status == "STALE":
            if not force:
                errors.append(f"  ✗ {prereq}: STALE (upstream changed; rerun required, or --force to override)")
                continue
            # --force: knowingly proceed despite stale upstream; still integrity-checked below.
        elif status != "COMPLETED":
            errors.append(f"  ✗ {prereq}: status={status} (must be COMPLETED)")
            continue

        artifacts = entry.get("artifacts", []) or []
        if not isinstance(artifacts, list):
            errors.append(f"  ✗ {prereq}: artifacts must be a list")
            continue
        required_names = required.get(prereq, set())
        seen_names: set[str] = set()

        # Integrity check
        for art in artifacts:
            if not isinstance(art, dict):
                errors.append(f"  ✗ {prereq}: artifact entry must be a mapping")
                continue
            art_name = art.get("name")
            if isinstance(art_name, str):
                seen_names.add(art_name)
            p = art.get("path")
            if p is None:
                if art_name in required_names:
                    errors.append(f"  ✗ {prereq}: required artifact {art_name} has no path")
                continue
            if not isinstance(p, str) or not p:
                errors.append(f"  ✗ {prereq}: artifact {art_name} has invalid path {p!r}")
                continue
            if integrity_check:
                resolved = _resolve_artifact_path(manifest_path, p)
                if not resolved.exists():
                    errors.append(f"  ✗ {prereq}: artifact {art_name} missing at {resolved}")

        for missing_name in sorted(required_names - seen_names):
            errors.append(f"  ✗ {prereq}: required artifact {missing_name} not present in manifest")

    if errors:
        msg = f"PREREQUISITE CHECK FAILED for {stage}\n\n" + "\n".join(errors) + (
            "\n\nRemediation: rerun the failed prerequisite stage(s) before retrying."
        )
        raise PrereqError(msg)
