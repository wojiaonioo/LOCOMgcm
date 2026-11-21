"""Execution framework — runs Layer B → C → D pipelines."""

from ._runner import (
    ExecutionContext,
    framework_call_consumer,
    framework_call_merger,
    framework_call_operator,
    run_pipeline_spec,
)

__all__ = [
    "ExecutionContext",
    "framework_call_consumer",
    "framework_call_merger",
    "framework_call_operator",
    "run_pipeline_spec",
]
