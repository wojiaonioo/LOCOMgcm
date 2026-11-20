"""Unified-runtime dispatcher for the native KPP bldepth kernel.

Enablement comes from the accelerator gate, never from a private environment
read, and every outcome is recorded once against the unified ignition ledger.
"""

from __future__ import annotations

import torch

from defgcm.accel import current_rank, dispatch_scope, gate, record_dispatch
from defgcm.dynamics.kpp import kpp_bldepth as _eager_kpp_bldepth

from ._fast import (
    SUPPORTED_NZ,
    SUPPORTED_TABLE_SHAPE,
    direct_decline_reason,
    kpp_bldepth_cuda_direct,
)


_KEY = "kpp_bldepth_cuda"
_VARIANT = "bldepth"
_SITE = "kpp_bldepth"
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


def kpp_bldepth_cuda(
    kmtj: torch.Tensor,
    dvsq: torch.Tensor,
    dbloc: torch.Tensor,
    Ritop: torch.Tensor,
    ustar: torch.Tensor,
    bo: torch.Tensor,
    bosol: torch.Tensor,
    coriol: torch.Tensor,
    zgrid: torch.Tensor,
    hwide: torch.Tensor,
    wmt: torch.Tensor,
    wst: torch.Tensor,
    *,
    Ricr: float,
    epsilon: float,
    Vtc: float,
    cekman: float,
    cmonob: float,
    minKPPhbl: float,
    vonk: float,
    phepsi: float = 1.0e-10,
    limit_hbl_stable: bool = True,
    jwtype: int = 2,
    penetrating_sw: bool = True,
    sign_zero: str = "signbit",
    **wscale_overrides,
):
    """Use the native kernel when enabled/applicable, else the frozen reference."""
    eager_arguments = (
        kmtj,
        dvsq,
        dbloc,
        Ritop,
        ustar,
        bo,
        bosol,
        coriol,
        zgrid,
        hwide,
        wmt,
        wst,
    )
    eager_keywords = {
        "Ricr": Ricr,
        "epsilon": epsilon,
        "Vtc": Vtc,
        "cekman": cekman,
        "cmonob": cmonob,
        "minKPPhbl": minKPPhbl,
        "vonk": vonk,
        "phepsi": phepsi,
        "limit_hbl_stable": limit_hbl_stable,
        "jwtype": jwtype,
        "penetrating_sw": penetrating_sw,
        "sign_zero": sign_zero,
        **wscale_overrides,
    }
    if not gate(_KEY).effective_value:
        return _eager_kpp_bldepth(*eager_arguments, **eager_keywords)

    with dispatch_scope(_KEY):
        # ONE predicate, shared with the strict entry: the gate can never
        # accept an input the kernel would reject, nor decline one it serves.
        reason = direct_decline_reason(
            eager_arguments,
            limit_hbl_stable=limit_hbl_stable,
            jwtype=jwtype,
            scalar_parameters=(
                Ricr,
                epsilon,
                Vtc,
                cekman,
                cmonob,
                minKPPhbl,
                vonk,
                phepsi,
            ),
            wscale_overrides=wscale_overrides,
        )
        if reason is not None:
            _record("inapplicable", reason=reason)
            return _eager_kpp_bldepth(*eager_arguments, **eager_keywords)

        try:
            output = kpp_bldepth_cuda_direct(*eager_arguments, **eager_keywords)
        except Exception as error:
            _record("error", reason=type(error).__name__)
            raise
        # The custom op records real launches, including compiled execution.
        return output


__all__ = [
    "SUPPORTED_NZ",
    "SUPPORTED_TABLE_SHAPE",
    "kpp_bldepth_cuda",
]
