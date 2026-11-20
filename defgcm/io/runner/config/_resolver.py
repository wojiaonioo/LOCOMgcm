"""Default run_id computation."""

from __future__ import annotations

from typing import Any


def compute_run_id(run_cfg: Any) -> str:
    """Default run_id format: {exp_name}-{start_date}.

    Uses run.run_id override if present.
    """
    if hasattr(run_cfg, "run") and run_cfg.run.run_id is not None:
        run_id = run_cfg.run.run_id
        if not isinstance(run_id, str):
            raise TypeError(
                f"run.run_id must be str or None, got {type(run_id).__name__}"
            )
        if not run_id or run_id.strip() != run_id:
            raise ValueError("run.run_id must be a nonempty path component")
        if run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise ValueError(f"run.run_id must be a single path component, got {run_id!r}")
        return run_id
    exp = run_cfg.run.experiment_name
    date_str = run_cfg.run.start_date.isoformat() if hasattr(run_cfg.run.start_date, "isoformat") else str(run_cfg.run.start_date)
    return f"{exp}-{date_str}"
