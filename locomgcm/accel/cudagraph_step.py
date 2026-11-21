"""CUDAGraph logical-step boundary helper."""

from __future__ import annotations

import torch

try:
    from torch._inductor import config as _inductor_config
except (AttributeError, ImportError):  # pragma: no cover - older torch
    _inductor_config = None

try:
    from torch import compiler as _torch_compiler
except (AttributeError, ImportError):  # pragma: no cover - older torch
    _torch_compiler = None


def mark_cudagraph_step() -> None:
    """Mark one logical simulation generation as at the step boundary.

    The mark groups compiled partitions within a step but does not preserve tensors retained across generations.
    """
    if _inductor_config is None or _torch_compiler is None:
        return
    try:
        enabled = _inductor_config.triton.cudagraphs
        mark_step_begin = _torch_compiler.cudagraph_mark_step_begin
    except AttributeError:  # pragma: no cover - older torch
        return
    if enabled:
        mark_step_begin()
