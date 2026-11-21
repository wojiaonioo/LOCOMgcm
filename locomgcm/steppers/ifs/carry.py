"""Retained tensor storage across IFS step boundaries.

Tensor shapes, masks, units and signs follow their producers. Carry writes
retain the graph in grad mode; clones preserve tensor dtype and device.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import torch


def cast_tensor(value, *, dtype, device):
    return value.to(dtype=dtype, device=device)


def _carry(value: torch.Tensor) -> torch.Tensor:
    """Cross-step retention: keep the tape when a gradient is being tracked.

    The stepper carries previous-step interface ``w``, the AB2 tracer
    tendency, ``etaH`` and hydrostatic potential into the next step.
    Detaching them cuts every
    multi-step derivative at the step boundary (with detaching, AD agreed
    with central FD to 7e-11 over one step and drifted to 3.3% over ten).  Under ``torch.inference_mode`` / ``no_grad`` -- every production
    run -- this is exactly ``detach`` and bytes are unchanged.
    """
    return value if torch.is_grad_enabled() else value.detach()


def _clone_retained_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_retained_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_retained_value(item) for item in value)
    if dataclasses.is_dataclass(value):
        replacements = {}
        for item in dataclasses.fields(value):
            if not item.init:
                continue
            original = getattr(value, item.name)
            cloned = _clone_retained_value(original)
            if cloned is not original:
                replacements[item.name] = cloned
        return dataclasses.replace(value, **replacements) if replacements else value
    return value
