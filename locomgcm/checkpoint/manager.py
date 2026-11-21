"""CheckpointManager — the one object the integration loop talks to.

This is the easy-to-use facade that composes the three decoupled layers
(schedule -> capture -> persist) behind a tiny surface:

    mgr = CheckpointManager.from_run_config(run_cfg, run_dir)   # or None
    plan = mgr and mgr.resolve_resume(model, n_steps)           # resume?
    if plan: state, time_off, start = mgr.apply_resume(model, plan)
    ...
    mgr.maybe_capture(model, state, global_step=g, model_time=t, dt=dt,
                      n_steps=n_steps)                           # in the loop
    mgr.close()                                                 # drain async

Everything physics-specific lives behind ``capture`` and the closed restart
state codec; everything I/O lives behind ``writer``.  The manager only
orchestrates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.distributed as dist

from locomgcm.state import PrognosticState

from .capture import (
    assert_resume_compatible as _assert_resume_compatible,
    capture as _capture_state,
    config_fingerprint as _config_fingerprint,
    restore as _restore,
)
from .persistence import find_checkpoint, latest_checkpoint, load_checkpoint
from .schedule import build_schedule
from .types import ModelCheckpoint, require_nonempty_fingerprint
from .writer import CheckpointWriter, make_writer


@dataclass(frozen=True)
class ResumePlan:
    """Resolved resume target: which checkpoint, at which global step."""

    checkpoint: ModelCheckpoint
    start_step: int
    model_time_seconds: float
    source_path: Path


class CheckpointManager:
    """Ties schedule + capture + async writer for one run."""

    def __init__(
        self,
        *,
        root: Path,
        schedule_entries: list[dict] | None,
        writer: CheckpointWriter,
        config_fingerprint: str | dict[str, Any],
        permanent_steps: set[int],
        resume_directive: str | int | None,
        resume_root: Path | None = None,
    ) -> None:
        require_nonempty_fingerprint(
            config_fingerprint, path="CheckpointManager config_fingerprint"
        )
        self.root = Path(root)
        # Where a resume *reads* from — defaults to this run's own dir, but may
        # point at another run so you can fork a new output from a prior run's
        # checkpoint without clobbering either side's diagnostics.
        self._resume_root = Path(resume_root) if resume_root is not None else self.root
        self._schedule = build_schedule(schedule_entries)
        self._writer = writer
        self._fingerprint = config_fingerprint
        self._permanent_steps = set(permanent_steps)
        self._resume_directive = resume_directive
        self._trigger_cache: dict[int, frozenset[int]] = {}

    # -- construction --------------------------------------------------------
    @classmethod
    def from_run_config(
        cls,
        run_cfg: Any,
        run_dir: Path,
        *,
        rank: int | None = None,
    ) -> "CheckpointManager | None":
        """Build from ``run_cfg.checkpoint``; ``None`` when disabled.

        A resume is honored even if capture is disabled, so you can resume
        without writing new checkpoints — but the common case enables both.
        Distributed stage2 callers pass ``rank`` so each process writes and
        resumes its own rank-local state under ``checkpoints/rankXXXXX/``.
        """
        cc = getattr(run_cfg, "checkpoint", None)
        if cc is None:
            return None
        resume = _normalize_resume(getattr(cc, "resume", "none"))
        if not cc.enabled and resume is None:
            return None

        root = _checkpoint_root(run_dir, rank=rank)
        resume_dir = getattr(cc, "resume_dir", None)
        resume_root = _checkpoint_root(resume_dir, rank=rank) if resume_dir else None
        fingerprint = _config_fingerprint(run_cfg)
        writer = make_writer(
            root,
            backend=cc.backend,
            keep_last=cc.keep_last,
            permanent_steps=set(cc.permanent_steps),
            max_inflight=cc.max_inflight,
        )
        return cls(
            root=root,
            schedule_entries=[dict(e) for e in cc.schedule],
            writer=writer,
            config_fingerprint=fingerprint,
            permanent_steps=set(cc.permanent_steps),
            resume_directive=resume,
            resume_root=resume_root,
        )

    # -- schedule ------------------------------------------------------------
    def trigger_steps(self, n_steps: int, dt: float) -> frozenset[int]:
        if n_steps not in self._trigger_cache:
            steps = set(self._schedule.trigger_steps(n_steps, dt))
            steps |= {s for s in self._permanent_steps if 0 <= s <= n_steps}
            self._trigger_cache[n_steps] = frozenset(steps)
        return self._trigger_cache[n_steps]

    # -- resume --------------------------------------------------------------
    def resolve_resume(self, model: Any, n_steps: int) -> ResumePlan | None:
        """Locate + load + validate the checkpoint named by the directive."""
        if self._resume_directive is None:
            return None
        if self._resume_directive == "latest":
            ref = latest_checkpoint(self._resume_root)
        else:
            ref = find_checkpoint(self._resume_root, int(self._resume_directive))
        layout = model.grid.parallel_layout
        if layout is not None and layout.world_size > 1:
            generations = [None] * dist.get_world_size()
            dist.all_gather_object(generations, None if ref is None else ref.global_step)
            if None in generations or len(set(generations)) != 1:
                raise ValueError(f"checkpoint generation differs across ranks: {generations}")
        if ref is None:
            raise FileNotFoundError(
                f"resume requested ({self._resume_directive!r}) but no matching "
                f"complete checkpoint under {self._resume_root}"
            )
        ckpt = load_checkpoint(ref.path)
        _assert_resume_compatible(ckpt, model, self._fingerprint)
        if ckpt.meta.global_step > n_steps:
            raise ValueError(
                f"checkpoint step {ckpt.meta.global_step} exceeds target "
                f"n_steps {n_steps}; nothing to integrate"
            )
        return ResumePlan(
            checkpoint=ckpt,
            start_step=ckpt.meta.global_step,
            model_time_seconds=ckpt.meta.model_time_seconds,
            source_path=ref.path,
        )

    def apply_resume(self, model: Any, plan: ResumePlan) -> PrognosticState:
        """Inject the resume checkpoint into ``model``; return the seed state."""
        return _restore(
            model,
            plan.checkpoint,
            config_fingerprint=self._fingerprint,
        )

    # -- capture -------------------------------------------------------------
    def maybe_capture(
        self,
        model: Any,
        state: PrognosticState,
        *,
        global_step: int,
        model_time: float,
        dt: float,
        n_steps: int,
        detach: bool = True,
    ) -> bool:
        """Capture + submit if ``global_step`` is a trigger step.

        Returns whether a checkpoint was taken.  Cheap no-op otherwise.
        """
        if global_step not in self.trigger_steps(n_steps, dt):
            return False
        ckpt = _capture_state(
            model,
            state,
            global_step=global_step,
            model_time_seconds=model_time,
            dt=dt,
            config_fingerprint=self._fingerprint,
            permanent=global_step in self._permanent_steps,
            detach=detach,
        )
        self._writer.submit(ckpt)
        return True

    # -- lifecycle -----------------------------------------------------------
    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def _normalize_resume(value: Any) -> str | int | None:
    """``none``/None/'' -> None; ``latest`` -> 'latest'; ``<int>`` -> int."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("", "none", "off", "false", "no"):
        return None
    if s in ("latest", "last"):
        return "latest"
    try:
        return int(s)
    except ValueError as exc:
        raise ValueError(
            f"checkpoint.resume must be 'none' | 'latest' | <int step>, got {value!r}"
        ) from exc


def _checkpoint_root(run_dir: str | Path, *, rank: int | None = None) -> Path:
    root = Path(run_dir) / "checkpoints"
    if rank is None:
        return root
    return root / f"rank{int(rank):05d}"
