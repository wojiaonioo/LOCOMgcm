"""Unified control plane for LOCOMgcm's verified accelerator clients."""

from .launch import (
    clear_launch_decisions,
    format_launch_decisions,
    launch_decisions,
    record_launch_decision,
)
from .cudagraph_step import mark_cudagraph_step
from .preload import preload_requested_extensions
from .registry import AcceleratorSpec, REGISTRY
from .runtime import (
    StepGates,
    consumer_snapshot,
    current_rank,
    dispatch_scope,
    drain_pending_dispatch_events,
    emit_summaries,
    freeze_step_snapshot,
    format_summary,
    gate,
    in_accelerated_dispatch,
    parse_bool,
    record_cache_event,
    record_dispatch,
    record_selection,
    selection_scope,
    requested,
    reset_runtime,
    resolve_snapshot,
    step_snapshot,
    step_snapshot_active,
    summaries,
)
from .schema import GateView

__all__ = [
    "AcceleratorSpec",
    "GateView",
    "REGISTRY",
    "StepGates",
    "clear_launch_decisions",
    "consumer_snapshot",
    "current_rank",
    "dispatch_scope",
    "drain_pending_dispatch_events",
    "emit_summaries",
    "format_launch_decisions",
    "format_summary",
    "freeze_step_snapshot",
    "gate",
    "in_accelerated_dispatch",
    "launch_decisions",
    "mark_cudagraph_step",
    "parse_bool",
    "preload_requested_extensions",
    "record_cache_event",
    "record_dispatch",
    "record_selection",
    "selection_scope",
    "record_launch_decision",
    "requested",
    "reset_runtime",
    "resolve_snapshot",
    "step_snapshot",
    "step_snapshot_active",
    "summaries",
]
