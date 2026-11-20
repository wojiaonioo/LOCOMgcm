"""Unified-runtime dispatcher for the DST3/DST3-FL transport-flux core.

Enablement comes from the accelerator gate, never from a private environment
read, and every outcome is recorded once against the unified ignition ledger.

The consumer boundary is the stencil-flux selection
``operators/fluxes/tracer_flux.py:horizontal_stencil_flux``; one wiring
point therefore serves all six call sites (the region single-patch adapter's X
and Y passes plus the four cube tile-adapter sites).  The fallback calls the
scheme's eager math kernel in ``operators/fluxes/tracer_reconstruction.py``.

Distributed note:
the decline predicate below is rank-local by design and reads only local tensor
device/dtype/shape, which are SPMD-uniform on a homogeneous run.  It contains no
collective, so it cannot wedge; divergence, if a heterogeneous setup ever
produced it, is caught at the end-of-run ``verify_dispatch_uniformity`` barrier
(``defgcm/accel/runtime.py:343``, called from ``runner.py:760``).
"""

from __future__ import annotations

import torch

from defgcm.accel import current_rank, dispatch_scope, gate, record_dispatch

from ._fast import direct_decline_reason, dst3_transport_flux_values_cuda_direct


_KEY = "dst3_cuda"
_SITE = "horizontal_stencil_flux"
_BACKEND = "cuda"


def _record(
    *, outcome: str, variant: str, reason: str | None = None, count: int = 1
) -> None:
    record_dispatch(
        _KEY,
        outcome=outcome,
        variant=variant,
        site=_SITE,
        backend=_BACKEND,
        rank=current_rank(),
        reason=reason,
        count=count,
    )


def dst3_transport_flux_values_cuda(
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    Tm2: torch.Tensor,
    Tm1: torch.Tensor,
    T0: torch.Tensor,
    Tp1: torch.Tensor,
    mask_im1: torch.Tensor,
    mask_i: torch.Tensor,
    mask_ip1: torch.Tensor,
    u_trans: torch.Tensor,
    u_cfl: torch.Tensor,
    *,
    with_limiter: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the native core, or its exact eager fallback, and record which."""
    from ..tracer_reconstruction import dst3_transport_flux, dst3fl_transport_flux

    values = (
        T_left,
        T_right,
        Tm2,
        Tm1,
        T0,
        Tp1,
        mask_im1,
        mask_i,
        mask_ip1,
        u_trans,
        u_cfl,
    )
    # The limiter branch is a different arithmetic path (Sweby bound + clamp),
    # so it is a distinct dispatch variant rather than a parameter of one.
    variant = "limiter" if with_limiter is True else "plain"
    reference = dst3fl_transport_flux if with_limiter is True else dst3_transport_flux

    # The production boundary already asked ``requested``; this second read makes
    # a direct call with the gate off run the reference instead of tripping the
    # ledger's "dispatched while disabled" guard.
    if not gate(_KEY).effective_value:
        return reference(*values)

    with dispatch_scope(_KEY):
        # ONE predicate, shared with the strict entry point: the gate can never
        # accept an input the kernel would reject, nor decline one it serves.
        reason = direct_decline_reason(values, with_limiter=with_limiter)
        if reason is not None:
            _record(outcome="inapplicable", variant=variant, reason=reason)
            return reference(*values)

        try:
            upwind_flux, flux_dst = dst3_transport_flux_values_cuda_direct(
                *values, with_limiter=with_limiter
            )
        except Exception as error:
            # Rank-local fail-fast is the repository convention for numerical
            # cores; a silent fallback here would move
            # this kernel into the backend-mixing family.
            _record(outcome="error", variant=variant, reason=type(error).__name__)
            raise
        # Runtime custom-op launches own forward/backward hit accounting.
        return upwind_flux, flux_dst


__all__ = ["dst3_transport_flux_values_cuda"]
