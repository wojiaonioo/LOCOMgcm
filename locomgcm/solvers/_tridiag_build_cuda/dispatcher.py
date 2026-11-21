"""Unified-runtime dispatcher for CUDA implicit-diffusion system assembly.

Enablement comes from the accelerator gate, never from a private environment
read, and every outcome is recorded once against the unified ignition ledger.
"""

from __future__ import annotations

import torch

from locomgcm.accel import current_rank, dispatch_scope, gate, record_dispatch

from ..tridiag import build_implicit_diffusion_system as _eager_build
from ._fast import SUPPORTED_NZ, direct_decline_reason, tridiag_build_cuda


_KEY = "tridiag_build_cuda"
_VARIANT = "implicit-diffusion-system"
_SITE = "build_implicit_diffusion_system"
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


def build_implicit_diffusion_system(
    tracer: torch.Tensor,
    layer_thickness: torch.Tensor,
    *,
    interface_spacing: torch.Tensor,
    diffusivity: float | torch.Tensor,
    dt: float,
    recip_hFac: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the CUDA kernel when applicable, otherwise call the frozen reference."""
    if not gate(_KEY).effective_value:
        return _eager_build(
            tracer,
            layer_thickness,
            interface_spacing=interface_spacing,
            diffusivity=diffusivity,
            dt=dt,
            recip_hFac=recip_hFac,
        )

    with dispatch_scope(_KEY):
        # ONE predicate, shared with the strict entry: the gate can never
        # accept an input the kernel would reject, nor decline one it serves.
        reason = direct_decline_reason(
            tracer, layer_thickness, interface_spacing, diffusivity, dt, recip_hFac
        )
        if reason is not None:
            _record("inapplicable", reason=reason)
            return _eager_build(
                tracer,
                layer_thickness,
                interface_spacing=interface_spacing,
                diffusivity=diffusivity,
                dt=dt,
                recip_hFac=recip_hFac,
            )

        try:
            result = tridiag_build_cuda(
                tracer, layer_thickness, interface_spacing, diffusivity, dt, recip_hFac
            )
        except Exception as error:
            _record("error", reason=type(error).__name__)
            raise
        _record("hit")
        return result


__all__ = [
    "SUPPORTED_NZ",
    "build_implicit_diffusion_system",
]
