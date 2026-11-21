"""Print a run's pipeline state at a glance — no SLURM, no model imports.

Answers the questions an operator (or a fresh AI session) otherwise has to dig
out of manifest.yaml + mtimes + pids by hand:
  - which stages are COMPLETED / RUNNING / FAILED / STALE / not-started;
  - is a RUNNING stage actually alive, or a dead-pid zombie (host-aware);
  - is preprocessed.zarr complete (_SUCCESS marker) and fresh (newer than the
    LOCOMgcm output it derives from), or stale → needs a postproc rerun;
  - the PASS/FAIL verdict, or that the run has no MITgcm comparison;
  - the exact command to refresh whatever is stale/missing.

Usage:
    python -m experiments.pipeline_v2.show_run_status <run-id> [--output outputs/data]
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_STAGE_ORDER = [
    "stage1a_prepare_common", "stage1b_export_to", "stage1c_export_mit",
    "stage2_torch_run", "stage3_mit_run", "stage4_postproc", "stage5_compare",
]

# A RUNNING entry is considered "stale" when it is on another host and its
# started_at is more than this many seconds old (other-host check) OR when its
# recorded pid is dead on THIS host.  Cross-node PIDs cannot be checked, so we
# use wall-clock age as the proxy.
_STALE_OTHER_HOST_SECONDS = 30 * 60  # 30 minutes


def _pid_alive_here(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _entry_age_seconds(entry: dict) -> float | None:
    """Return the age of a stage entry in seconds, or None if unparseable."""
    started = entry.get("started_at")
    if not started:
        return None
    try:
        # ISO 8601 without timezone → assume UTC
        dt = datetime.fromisoformat(str(started))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - dt).total_seconds()
    except (ValueError, TypeError):
        return None


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _newest_nc_mtime(run_dir: Path) -> float:
    nc_dir = run_dir / "nc_snapshots"
    if not nc_dir.is_dir():
        return 0.0
    return max((_mtime(p) for p in nc_dir.glob("day*.nc")), default=0.0)


def _comparison_off(run_dir: Path) -> bool:
    """True when metadata.yaml records a run without the MITgcm comparison."""
    meta_path = run_dir / "metadata.yaml"
    if not meta_path.exists():
        return False
    meta = yaml.safe_load(meta_path.read_text()) or {}
    return meta.get("type") == "locomgcm_only" or "reference" not in (meta.get("sources") or {})


def _fmt_pid(entry: dict, this_host: str) -> str:
    """Return a human-readable pid/host string for a stage entry.

    For RUNNING stages this includes liveness and staleness annotations
    without mutating the manifest.
    """
    status = entry.get("status")
    pid = entry.get("pid")
    host = entry.get("hostname")
    if status != "RUNNING" or pid is None:
        return "-"
    if host and host != this_host:
        return f"pid {pid}@{host} (other host)"
    return f"pid {pid} {'ALIVE' if _pid_alive_here(int(pid)) else 'DEAD(zombie)'}"


def _effective_display_status(entry: dict, this_host: str) -> str:
    """Return the display status for a stage entry.

    For RUNNING stages we annotate stale/suspect entries without mutating the
    manifest:
    - Same host + pid dead → "RUNNING (stale? pid dead)"
    - Other host + entry older than _STALE_OTHER_HOST_SECONDS →
      "RUNNING (stale? other host, started <age> ago)"
    - Other host + entry is recent → "RUNNING" (may genuinely be running)

    All other statuses are returned as-is.
    """
    stored = entry.get("status", "?")
    if stored != "RUNNING":
        return stored

    pid = entry.get("pid")
    host = entry.get("hostname")

    if pid is None:
        return stored

    if host and host != this_host:
        age = _entry_age_seconds(entry)
        if age is not None and age > _STALE_OTHER_HOST_SECONDS:
            age_min = int(age // 60)
            return f"RUNNING (stale? other host, started {age_min}m ago)"
        # Either recent or age unknown — show as plain RUNNING; _fmt_pid
        # already appends "(other host)" to the pid column.
        return stored

    # Same host — we can check pid liveness directly.
    if not _pid_alive_here(int(pid)):
        return "RUNNING (stale? pid dead)"

    return stored


def _derive_effective_overall_status(stages: dict, stored_overall: str, this_host: str) -> str:
    """Derive a display-layer overall status from stage entries.

    The stored overall_status is shown verbatim alongside the derived value.
    When the stored value is RUNNING but all known stage entries are in
    terminal or stale states (none are genuinely RUNNING), we annotate it as
    "(stale?)" to alert the operator.

    This is display-only; the manifest is never mutated.
    """
    if stored_overall != "RUNNING":
        return stored_overall

    # Check if any stage entry is genuinely RUNNING (alive or unknown liveness)
    for entry in stages.values():
        if not isinstance(entry, dict):
            continue
        eff = _effective_display_status(entry, this_host)
        if eff == "RUNNING":
            # At least one stage appears genuinely RUNNING — trust stored value.
            return stored_overall

    # No genuinely RUNNING stage found; all are terminal or annotated stale.
    # The orchestrator itself may have died without writing FAILED.
    return "RUNNING (stale? no live stage processes found)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="show_run_status")
    ap.add_argument("run_id")
    # Default to an absolute path derived from this script's location so the
    # tool works correctly when invoked from any working directory.
    _default_output = Path(__file__).resolve().parents[2] / "outputs" / "data"
    ap.add_argument("--output", type=Path, default=_default_output)
    args = ap.parse_args(argv)

    run_dir = args.output / args.run_id
    manifest_path = run_dir / "manifest.yaml"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2

    doc = yaml.safe_load(manifest_path.read_text()) or {}
    stages = doc.get("stages", {}) or {}
    this_host = socket.gethostname()

    stored_overall = doc.get("overall_status", "?")
    effective_overall = _derive_effective_overall_status(stages, stored_overall, this_host)
    if effective_overall != stored_overall:
        print(f"Run: {args.run_id}   overall_status: {effective_overall}  (stored: {stored_overall})")
    else:
        print(f"Run: {args.run_id}   overall_status: {stored_overall}")
    print(f"dir: {run_dir}")
    print()
    print(f"  {'stage':<24} {'status':<36} {'pid/host':<26} completed_at")
    suggestions: list[str] = []
    model_stages_failed = False
    for name in _STAGE_ORDER:
        if name not in stages:
            continue
        e = stages[name]
        st = e.get("status", "?")
        eff_st = _effective_display_status(e, this_host)
        print(f"  {name:<24} {eff_st:<36} {_fmt_pid(e, this_host):<26} {e.get('completed_at', '-')}")
        if st in ("FAILED", "STALE"):
            suggestions.append(f"{name} is {st}")
        # Track whether any model-execution stage failed.
        if st == "FAILED" and name in ("stage2_torch_run", "stage3_mit_run"):
            model_stages_failed = True

    # preprocessed.zarr completeness + freshness
    zarr = run_dir / "preprocessed.zarr"
    print()
    if zarr.is_dir():
        complete = (zarr / "_SUCCESS").exists()
        z_mt, nc_mt = _mtime(zarr / "zarr.json"), _newest_nc_mtime(run_dir)
        stale = nc_mt > 0 and z_mt > 0 and nc_mt > z_mt
        flag = "complete" if complete else "INCOMPLETE (no _SUCCESS)"
        fresh = "STALE — LOCOMgcm output is newer → rerun postproc" if stale else "fresh"
        print(f"  preprocessed.zarr: {flag}, {fresh}")
        if not complete or stale:
            suggestions.append("preprocessed.zarr stale/incomplete")
    else:
        print("  preprocessed.zarr: absent")

    comparison_off = _comparison_off(run_dir)
    summary = run_dir / "comparison_metrics.summary.json"
    if comparison_off:
        print("  comparison: off")
    elif summary.exists():
        import json
        try:
            verdict = json.loads(summary.read_text()).get("verdict", "?")
        except Exception:
            verdict = "?"
        print(f"  verdict: {verdict}")
    else:
        print("  verdict: (no comparison_metrics.summary.json)")

    if suggestions:
        print()
        print("  needs attention: " + "; ".join(suggestions))
        # Only suggest a postproc-only rerun when model stages are not
        # FAILED. When stage2 (or stage3) failed, running postproc-only would
        # hit the prereq check and fail with a confusing error. Without the
        # comparison there is no stage 5 (it would be refused).
        stage2_status = stages.get("stage2_torch_run", {}).get("status")
        if model_stages_failed:
            print(f"  → fix and rerun model: run_case.sh resume {args.run_id}")
        elif stage2_status == "COMPLETED" and comparison_off:
            print(f"  → refresh postproc:  STAGES=4 run_case.sh resume {args.run_id}")
        elif stage2_status == "COMPLETED":
            print(f"  → refresh postproc:  METRICS_FULL=1 STAGES=4,5 run_case.sh resume {args.run_id}")
        else:
            # stage2 not completed (PENDING / STALE / RUNNING / absent) — full resume
            print(f"  → run full pipeline: run_case.sh resume {args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
