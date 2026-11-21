"""Diagnostic view of native CUDA launch decisions.

Every CUDA package derives its launch configuration natively through the
shared ``locomgcm/solvers/_cuda_launch`` C++ header (CUDA occupancy API).
This module is the cross-package surface those packages push their resolved
decisions into; it performs no derivation of its own, so a second source of
tuning truth cannot exist here.

Row semantics: extension load records only a ``<kernel>@load`` marker row
(``{"phase": "loaded"}``); the unsuffixed and ``@<dtype>`` rows are recorded
by the launch paths for configurations they actually resolved, so the view
never claims a resource basis for a dtype variant that did not execute
(dtype-parametrised packages; the single-dtype packages still push their one
config at load).

There is deliberately no fallback path: a package that never pushes simply
has no row in the view, which the audit tooling reports as a missing
decision rather than silently substituting one.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, Mapping


_DECISIONS: dict[tuple[str, str], Mapping[str, Any]] = {}


def record_launch_decision(
    *, kernel: str, device: str, decision: Mapping[str, Any]
) -> None:
    """Store one native launch decision for ``(kernel, device)``.

    ``decision`` is the dict produced by the native layer's pybind
    ``to_python`` conversion (or an equivalent mapping).  Re-recording the
    same key overwrites: the native layer caches per device, so a repeat
    push carries the same content.
    """
    if not kernel or not device:
        raise ValueError("kernel and device must be non-empty")
    _DECISIONS[(str(kernel), str(device))] = MappingProxyType(dict(decision))


def launch_decisions() -> dict[tuple[str, str], dict[str, Any]]:
    """Return a copy of every recorded native launch decision."""
    return {key: dict(value) for key, value in _DECISIONS.items()}


def clear_launch_decisions() -> None:
    _DECISIONS.clear()


def format_launch_decisions() -> list[str]:
    """One machine-readable line per decision, for the ignition audit."""
    lines = []
    for (kernel, device), decision in sorted(_DECISIONS.items()):
        payload = {"kernel": kernel, "device": device, **decision}
        lines.append(
            "LOCOMGCM_LAUNCH "
            + json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        )
    return lines
