"""Unified-runtime dispatcher for the tridiagonal CUDA kernel.

Enablement comes from the accelerator gate, never from a private environment
read, and every outcome is recorded once against the unified ignition ledger.
Unsupported inputs fall back to the frozen eager solver before the extension is
loaded; once a supported call is selected, a launch failure is recorded and
re-raised rather than hidden behind a fallback.
"""

from __future__ import annotations

import torch
from locomgcm.accel.tensor_context import grad_mode_requires_reference

from locomgcm.accel import current_rank, dispatch_scope, gate, record_dispatch

from ..tridiag import forward_mode_requested, tridiag_solve as _eager_tridiag_solve
from ._fast import SUPPORTED_DTYPES, SUPPORTED_NZ, tridiag_solve_cuda_direct


_KEY = "tridiag_cuda"
_VARIANT = "thomas"
_SITE = "tridiag_solve"
_BACKEND = "cuda"


def _record(outcome: str, *, reason: str | None = None, count: int = 1) -> None:
    record_dispatch(
        _KEY,
        outcome=outcome,
        variant=_VARIANT,
        site=_SITE,
        backend=_BACKEND,
        rank=current_rank(),
        reason=reason,
        count=count,
    )


def _decline_reason(values: tuple[torch.Tensor, ...]) -> str | None:
    if not all(isinstance(value, torch.Tensor) for value in values):
        return "non_tensor"
    reference = values[0]
    if any(value.layout != torch.strided for value in values):
        return "layout"
    if any(value.device.type != "cuda" for value in values):
        return "non_cuda"
    if any(value.device != reference.device for value in values):
        return "device_mismatch"
    if reference.dtype not in SUPPORTED_DTYPES:
        return "dtype"
    if any(value.dtype != reference.dtype for value in values):
        return "dtype"
    if any(value.shape != reference.shape for value in values):
        return "shape_mismatch"
    if reference.ndim != 4 or reference.shape[-3] > SUPPORTED_NZ:
        return "unsupported_shape"
    if any(size == 0 for size in reference.shape):
        return "empty_shape"
    if any(stride < 0 for value in values for stride in value.stride()):
        return "negative_stride"
    if grad_mode_requires_reference():
        return "autograd"
    return None


def tridiag_solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    *,
    mitgcm_order: bool = False,
) -> torch.Tensor:
    """Use the native kernel when enabled and applicable, else frozen eager."""
    # This package entry is public as well: let the common solver split duals
    # before its two primal solves return here for native dispatch.
    if forward_mode_requested(a, b, c, d):
        return _eager_tridiag_solve(a, b, c, d, mitgcm_order=mitgcm_order)
    if not gate(_KEY).effective_value:
        return _eager_tridiag_solve(a, b, c, d, mitgcm_order=mitgcm_order)

    with dispatch_scope(_KEY):
        values = (a, b, c, d)
        reason = _decline_reason(values)
        if reason is not None:
            _record("inapplicable", reason=reason)
            return _eager_tridiag_solve(a, b, c, d, mitgcm_order=mitgcm_order)
        try:
            output = tridiag_solve_cuda_direct(a, b, c, d, mitgcm_order)
        except Exception as error:
            _record("error", reason=type(error).__name__)
            raise
        _record("hit")
        return output


__all__ = [
    "SUPPORTED_NZ",
    "tridiag_solve",
]
