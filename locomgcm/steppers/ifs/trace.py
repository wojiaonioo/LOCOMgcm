"""Distributed messages, stream traces and checkpoint stations for IFS.

Payload tensors retain their producer's sign, grid location, masks, units and
dtype. The callback holder owns the host step index for stream traces and ``R4_TRACK``
diagnostics. The distributed message gate is resolved at import time.
Checkpoint stations consume the staged predictor and CG2D RHS payloads in
call order; the diagnostics backend saves arrays as host float64.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from locomgcm.diagnostics import ckpt_dump


# Resolved once at import: ``os.environ.get`` inside the compiled
# step body is a Dynamo constant-fold failure (one graph break per call site).
_DIST_TRACE_ENABLED = os.environ.get("LOCOMGCM_DIST_TRACE") == "1"


def _dist_trace(message: str) -> None:
    if not _DIST_TRACE_ENABLED:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[rank{rank}] implicit_fs: {message}", flush=True)


@dataclass
class StreamTrace:
    hook: Optional[Callable[..., None]] = None
    step_index: int = 0

    def set_stream_trace_hook(self, hook: Optional[Callable[..., None]]) -> None:
        """Install a non-checkpointed debug trace hook for case adapters."""
        self.hook = hook

    def emit_stream_trace(self, name: str, **payload: Any) -> None:
        """Emit optional step-internal tensors to an external alignment tool."""
        if ckpt_dump.ENABLED:
            if name == "cube.predictor_after_explicit_pg_mask":
                ckpt_dump.capture(
                    "ustar", u=payload["u_star"], v=payload["v_star"]
                )
            elif name == "cube.predictor_after_impldiff":
                ckpt_dump.capture(
                    "implicit", u=payload["u_star"], v=payload["v_star"]
                )
            elif name == "cube.cg2d_rhs":
                ckpt_dump.capture("cg_rhs", rhs=payload["rhs"])
        hook = self.hook
        if hook is None:
            return
        hook(
            name,
            step_index=int(self.step_index),
            **payload,
        )


def record_checkpoint(name: str, **tensors: torch.Tensor) -> None:
    if name in ("ustar", "implicit", "cg_rhs"):
        tensors = ckpt_dump.captured(name)
    ckpt_dump.record(name, **tensors)


def track_surface_solution(eta_new, *, step_index):
    _step_idx = int(step_index)
    _eta_abs = eta_new[0].abs()
    _eta_max = _eta_abs.max().item()
    _amax_idx = int(_eta_abs.flatten().argmax().item())
    _nx = eta_new.shape[-1]
    _amax_j = _amax_idx // _nx
    _amax_i = _amax_idx % _nx
    print(f"[r4-track] step={_step_idx:03d} cg2d_eta_new abs_max={_eta_max:.3e} @(j={_amax_j},i={_amax_i})", flush=True)
