"""Unified-runtime dispatcher for the fused KPP STATEKPP kernel.

Enablement comes from the accelerator gate, never from a private environment
read, and every outcome is recorded once against the unified ignition ledger.
The kernel serves JMD95 only; the ``kpp_calc`` consumer keeps every other EOS
family on the eager path, so ``eos_type`` never reaches this module.  It
evaluates dbloc as the JMD95 divided difference, so it serves
``dbloc_form="divided_difference"`` only; the MITgcm form (``"statekpp"``) is
declined as ``inapplicable:dbloc_form`` and runs the eager reference.
"""

from __future__ import annotations

import torch

from locomgcm.accel import current_rank, dispatch_scope, gate, record_dispatch
from locomgcm.dynamics.kpp import kpp_statekpp as _eager_kpp_statekpp

from ._fast import (
    _PROMOTED_DTYPES, direct_decline_reason, kpp_statekpp_cuda_direct,
)


_KEY = "kpp_statekpp_cuda"
_VARIANT = "jmd95"
_SITE = "kpp_statekpp"
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


def kpp_statekpp_cuda(
    theta: torch.Tensor,
    salt: torch.Tensor,
    layer_pressure_pa: torch.Tensor,
    *,
    gravity: float,
    rho_const: float,
    dbloc_form: str,
):
    """Use the fused CUDA kernel when enabled/applicable, else the reference.

    The eager fallback runs inside :func:`dispatch_scope`, so the consumer
    boundary in ``kpp_calc`` sees ``in_accelerated_dispatch()`` and cannot
    dispatch a second time.
    """
    eager_keywords = {
        "gravity": gravity,
        "rho_const": rho_const,
        "eos_type": "jmd95",
        "dbloc_form": dbloc_form,
    }
    if not gate(_KEY).effective_value:
        return _eager_kpp_statekpp(
            theta, salt, layer_pressure_pa, **eager_keywords
        )

    with dispatch_scope(_KEY):
        if dbloc_form != "divided_difference":
            # ponytail: the kernel computes the divided difference only; the MITgcm
            # subtraction form would need a second kernel branch
            # and its own GPU acceptance before this decline can become a hit.
            _record("inapplicable", reason="dbloc_form")
            return _eager_kpp_statekpp(
                theta, salt, layer_pressure_pa, **eager_keywords
            )
        # ONE predicate, shared with the strict entry; ``widen=True`` because
        # float32 is widened below before the launch.
        reason = direct_decline_reason(
            theta,
            salt,
            layer_pressure_pa,
            gravity=gravity,
            rho_const=rho_const,
            widen=True,
        )
        if reason is not None:
            _record("inapplicable", reason=reason)
            return _eager_kpp_statekpp(
                theta, salt, layer_pressure_pa, **eager_keywords
            )

        # One semantics for the buoyancy chain, whatever the run precision.  The
        # eager reference evaluates all of STATEKPP in float64 and narrows only
        # its five outputs, because dbloc/dbsfc subtract density anomalies of
        # order 10 kg/m^3 whose difference in a near-neutral column is five
        # decades smaller (narrowing first costs 1.6e-02 relative on dbloc
        # there).  Widening here, and narrowing in the same place eager does,
        # makes this path bitwise the fallback instead of a second answer.
        out_dtype = theta.dtype
        widened = out_dtype in _PROMOTED_DTYPES
        try:
            result = kpp_statekpp_cuda_direct(
                theta.to(torch.float64) if widened else theta,
                salt.to(torch.float64) if widened else salt,
                layer_pressure_pa.to(torch.float64) if widened else layer_pressure_pa,
                gravity=gravity,
                rho_const=rho_const,
            )
        except Exception as error:
            _record("error", reason=type(error).__name__)
            raise
        if widened:
            result = tuple(value.to(out_dtype) for value in result)
        # Distinguish a widened hit from a native float64 one in the ledger:
        # "float32 and float64 are one semantics" is what this whole change
        # claims, and which steps actually took the widened path is its only
        # run-time evidence.  ``record_dispatch`` only *requires* a reason on a
        # non-hit, so carrying one here costs nothing.
        _record("hit", reason="widened_to_float64" if widened else None)
        return result


__all__ = [
    "kpp_statekpp_cuda",
]
