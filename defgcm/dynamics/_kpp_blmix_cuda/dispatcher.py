"""Unified-runtime dispatcher for the native CUDA KPP blmix candidate.

Enablement comes from the accelerator gate, never from a private environment
read; an invalid spelling is refused centrally by the gate grammar.  Every
outcome is recorded once against the unified ignition ledger.

The profile uses a native first-order VJP; velocity-scale preparation remains
on the Torch tape. Active geometry or profile scalar parameters use Torch.
"""

from __future__ import annotations

import torch

from defgcm.accel import current_rank, dispatch_scope, gate, record_dispatch
from defgcm.dynamics.kpp import (
    _fortran_sign,
    _gather_mit,
    kpp_blmix as _eager_kpp_blmix,
    kpp_wscale,
)

from ._fast import direct_decline_reason, kpp_blmix_cuda as _native_kpp_blmix


_KEY = "kpp_blmix_cuda"
_VARIANT = "blmix"
_SITE = "kpp_blmix"
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


def _prepare_wscale_inputs(
    ustar: torch.Tensor,
    bfsfc: torch.Tensor,
    hbl: torch.Tensor,
    stable: torch.Tensor,
    kbl: torch.Tensor,
    diffus: torch.Tensor,
    zgrid: torch.Tensor,
    hwide: torch.Tensor,
    wmt: torch.Tensor,
    wst: torch.Tensor,
    *,
    epsilon: float,
    phepsi: float,
    sign_zero: str = "signbit",
    **wscale_overrides,
) -> tuple[torch.Tensor, ...]:
    """Evaluate the frozen batched wscale path before the native profile kernel.

    The velocity-scale SIGN clip runs here, before the kernel, so ``sign_zero``
    (kpp.py ``KPP_SIGN_ZERO_MODES``) is applied on the Torch side.
    """
    nz = diffus.shape[1] - 2

    sigma = stable * 1.0 + (1.0 - stable) * epsilon
    wm_hbl, ws_hbl = kpp_wscale(sigma, hbl, ustar, bfsfc, wmt, wst, **wscale_overrides)
    positive_wm = _fortran_sign(1.0, wm_hbl, sign_zero)
    positive_ws = _fortran_sign(1.0, ws_hbl, sign_zero)
    wm_hbl = positive_wm * torch.clamp(torch.abs(wm_hbl), min=phepsi)
    ws_hbl = positive_ws * torch.clamp(torch.abs(ws_hbl), min=phepsi)

    kis = torch.arange(1, nz + 1, device=ustar.device)
    sigb = (-zgrid[kis] + 0.5 * hwide[kis]).reshape((-1, 1, 1)) / hbl
    sigma_levels = stable * sigb + (1.0 - stable) * torch.clamp(sigb, max=epsilon)
    wm_levels, ws_levels = kpp_wscale(
        sigma_levels, hbl, ustar, bfsfc, wmt, wst, **wscale_overrides
    )

    zg_kbl_minus_one = _gather_mit(zgrid, kbl - 1)
    sig_d = -zg_kbl_minus_one / hbl
    sigma_d = stable * sig_d + (1.0 - stable) * torch.clamp(sig_d, max=epsilon)
    wm_d, ws_d = kpp_wscale(sigma_d, hbl, ustar, bfsfc, wmt, wst, **wscale_overrides)
    return wm_hbl, ws_hbl, wm_levels, ws_levels, wm_d, ws_d


def kpp_blmix(
    ustar: torch.Tensor,
    bfsfc: torch.Tensor,
    hbl: torch.Tensor,
    stable: torch.Tensor,
    casea: torch.Tensor,
    kbl: torch.Tensor,
    diffus: torch.Tensor,
    zgrid: torch.Tensor,
    hwide: torch.Tensor,
    wmt: torch.Tensor,
    wst: torch.Tensor,
    *,
    epsilon: float,
    conc1: float,
    cg: float,
    phepsi: float = 1.0e-10,
    sign_zero: str = "signbit",
    **wscale_overrides,
):
    """Run native CUDA blmix when enabled, otherwise use the frozen reference."""
    eager_arguments = (
        ustar,
        bfsfc,
        hbl,
        stable,
        casea,
        kbl,
        diffus,
        zgrid,
        hwide,
        wmt,
        wst,
    )
    eager_keywords = {
        "epsilon": epsilon,
        "conc1": conc1,
        "cg": cg,
        "phepsi": phepsi,
        "sign_zero": sign_zero,
        **wscale_overrides,
    }
    if not gate(_KEY).effective_value:
        return _eager_kpp_blmix(*eager_arguments, **eager_keywords)

    with dispatch_scope(_KEY):
        # ONE predicate, shared with the strict entry.  This call reads the
        # table form; the strict entry reads the prepared form of the same
        # predicate once the velocity scales exist.
        reason = direct_decline_reason(
            eager_arguments,
            scalar_parameters=(epsilon, conc1, cg, phepsi),
            prepared=False,
        )
        if reason is not None:
            _record("inapplicable", reason=reason)
            return _eager_kpp_blmix(*eager_arguments, **eager_keywords)

        try:
            velocity_scales = _prepare_wscale_inputs(
                ustar,
                bfsfc,
                hbl,
                stable,
                kbl,
                diffus,
                zgrid,
                hwide,
                wmt,
                wst,
                epsilon=epsilon,
                phepsi=phepsi,
                sign_zero=sign_zero,
                **wscale_overrides,
            )
            native_values = (
                ustar,
                bfsfc,
                hbl,
                stable,
                casea,
                kbl,
                diffus,
                zgrid,
                hwide,
                *velocity_scales,
            )
            result = _native_kpp_blmix(
                native_values,
                conc1=conc1,
                cg=cg,
                phepsi=phepsi,
            )
        except Exception as error:
            _record("error", reason=type(error).__name__)
            raise
        _record("hit")
        return result


__all__ = [
    "kpp_blmix",
]
