"""Async checkpoint writer — the persist phase, off the hot path.

Two-phase model (CheckFreq, FAST'21; PyTorch DCP async_save):

  * snapshot phase (caller / main thread): ``capture()`` already moved
    tensors GPU->host, producing a device-decoupled ``ModelCheckpoint``.
  * persist phase (this writer, background thread): ``torch.save`` +
    atomic rename + ``_SUCCESS`` + manifest + rolling prune.

Back-pressure bounds memory: at most ``max_inflight`` persist jobs may be
queued; submitting beyond that blocks on the oldest, exactly like the
existing NetCDF writer's ``MAX_QUEUE`` (nc_snapshot.py).  A ``sync``
backend is provided for debugging / tiny cases / strict ordering.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from .persistence import prune_checkpoints, save_checkpoint_atomic
from .types import ModelCheckpoint


class CheckpointWriter(Protocol):
    def submit(self, ckpt: ModelCheckpoint) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class _BaseWriter:
    def __init__(
        self,
        root: Path,
        *,
        keep_last: int | None = 3,
        permanent_steps: set[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.keep_last = keep_last
        self.permanent_steps = set(permanent_steps or ())

    def _persist(self, ckpt: ModelCheckpoint) -> Path:
        path = save_checkpoint_atomic(self.root, ckpt)
        # Prune only AFTER the new checkpoint's _SUCCESS is on disk, so a
        # complete checkpoint is always alive (MITgcm rolling safety).
        prune_checkpoints(
            self.root,
            keep_last=self.keep_last,
            permanent_steps=self.permanent_steps,
        )
        return path


class SyncCheckpointWriter(_BaseWriter):
    """Blocking writer — persists inline.  Deterministic, for tests/tiny runs."""

    def submit(self, ckpt: ModelCheckpoint) -> None:
        self._persist(ckpt)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class AsyncCheckpointWriter(_BaseWriter):
    """Background-thread writer with bounded in-flight back-pressure."""

    def __init__(
        self,
        root: Path,
        *,
        keep_last: int | None = 3,
        permanent_steps: set[int] | None = None,
        max_inflight: int = 2,
    ) -> None:
        super().__init__(root, keep_last=keep_last, permanent_steps=permanent_steps)
        if max_inflight < 1:
            raise ValueError("max_inflight must be >= 1")
        self.max_inflight = max_inflight
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt-writer")
        self._pending: deque[Future] = deque()

    def submit(self, ckpt: ModelCheckpoint) -> None:
        # Back-pressure: block on the oldest job if the queue is full.  This
        # also surfaces any background exception promptly (result() re-raises).
        while len(self._pending) >= self.max_inflight:
            self._pending.popleft().result()
        self._pending.append(self._exec.submit(self._persist, ckpt))

    def flush(self) -> None:
        while self._pending:
            self._pending.popleft().result()

    def close(self) -> None:
        self.flush()
        self._exec.shutdown(wait=True)

    def __enter__(self) -> "AsyncCheckpointWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_writer(
    root: Path,
    *,
    backend: str = "thread",
    keep_last: int | None = 3,
    permanent_steps: set[int] | None = None,
    max_inflight: int = 2,
) -> CheckpointWriter:
    """Factory: ``thread`` (async) | ``sync``."""
    backend = (backend or "thread").lower()
    if backend in ("sync", "blocking"):
        return SyncCheckpointWriter(
            root, keep_last=keep_last, permanent_steps=permanent_steps
        )
    if backend in ("thread", "async"):
        return AsyncCheckpointWriter(
            root,
            keep_last=keep_last,
            permanent_steps=permanent_steps,
            max_inflight=max_inflight,
        )
    raise ValueError(f"unknown checkpoint writer backend: {backend!r}")
