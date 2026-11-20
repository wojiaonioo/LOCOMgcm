"""Declared graph exits for a compiled model step.

The PCG host loop, checkpoint diagnostics and step-boundary output may leave
the Dynamo graph. On supported PyTorch versions, strict_graph enables
error_on_graph_break around the step and graph_exit disables it within these
declared regions. fullgraph=True takes precedence over this setting, so the
runner uses error_on_graph_break for selective exits. When that API is absent,
these context helpers leave execution unchanged."""

from __future__ import annotations

import contextlib

import torch

_ERROR_ON_GRAPH_BREAK = getattr(torch._dynamo, "error_on_graph_break", None)

#: True when this torch build can enforce "hard error on undeclared break".
SUPPORTED: bool = _ERROR_ON_GRAPH_BREAK is not None


def graph_exit():
    """Permit a graph break inside the compiled step at a declared exit."""
    if _ERROR_ON_GRAPH_BREAK is None:
        return contextlib.nullcontext()
    return torch._dynamo.error_on_graph_break(False)


def strict_graph(fn):
    """Wrap ``fn`` so tracing it errors on any undeclared graph break."""
    if _ERROR_ON_GRAPH_BREAK is None:
        return fn
    return torch._dynamo.error_on_graph_break(True)(fn)


__all__ = ["SUPPORTED", "graph_exit", "strict_graph"]
