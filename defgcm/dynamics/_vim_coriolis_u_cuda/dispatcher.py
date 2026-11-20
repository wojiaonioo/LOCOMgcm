"""Unified-runtime dispatcher for vim coriolis u.

Enablement comes from the accelerator gate, never from a private environment
read, and every outcome is recorded once against the unified ignition ledger.
The fallback re-enters the eager reference inside :func:`dispatch_scope`, so the
frozen reference runs instead of dispatching a second time -- the control plane's
single recursion-guard mechanism, not a private flag.

Distributed note: the decline predicate below is rank-local by design and reads
only local tensor device/dtype/shape and the program's shared grad mode, which
are SPMD-uniform on a homogeneous run. It contains no collective; divergence, if a
heterogeneous setup ever produced it, is caught at the end-of-run
``verify_dispatch_uniformity`` barrier.
"""

from __future__ import annotations

import torch

from defgcm.accel import current_rank, dispatch_scope, gate, record_dispatch

from ._fast import direct_decline_reason, u_face_coriolis_from_haloed_v_row_metrics_cuda_direct


_KEY = "vim_coriolis_u"
_SITE = "_u_face_vorticity_flux"
_BACKEND = "cuda"


def consumer_reachable(*, select_vort_scheme: int, split_coriolis: bool) -> bool:
    """Whether the selected VIM formula uses this U-face stencil.

    Scheme 1 uses it for vorticity. Split Coriolis also uses it for the
    planetary term; combined scheme 4 has its own transport-weighted stencil.
    The model route and dispatch expectations both read this predicate.
    """
    return split_coriolis or select_vort_scheme == 1


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


def _reference(
    variant: str,
    v_ext: torch.Tensor,
    corner_value: torch.Tensor,
    *,
    dxG_south: torch.Tensor,
    dxG_north: torch.Tensor,
    recip_dxC_u: torch.Tensor,
    hFacZ: torch.Tensor | None,
    face_mask: torch.Tensor | None,
    edge_pair_mask: torch.Tensor | None,
    edge_face_couples: bool,
) -> torch.Tensor:
    """Torch reference of the native core for one ledger variant.

    ``variant`` picks the row-form vBar average (``planetary`` without hFac,
    ``vorticity`` hFacZ-weighted), then the shared result kernel applies the
    corner average, ``recip_dxC`` and the edge and face masks.
    """
    from defgcm.dynamics.vorticity_flux import (
        _u_face_coriolis_from_vbar_row,
        _u_vbar_hfacz_row,
        _u_vbar_no_hfac_row,
    )

    ny = recip_dxC_u.shape[-2]
    nx = corner_value.shape[-1] - 1
    if variant == "planetary":
        v_bar = _u_vbar_no_hfac_row(
            v_ext, dxG_south=dxG_south, dxG_north=dxG_north, ny=ny, nx=nx,
        )
    elif variant == "vorticity":
        v_bar = _u_vbar_hfacz_row(
            v_ext, hFacZ, dxG_south=dxG_south, dxG_north=dxG_north, ny=ny, nx=nx,
        )
    return _u_face_coriolis_from_vbar_row(
        v_bar,
        corner_value,
        recip_dxC_u,
        face_mask=face_mask,
        edge_face_couples=edge_face_couples,
        edge_pair_mask=edge_pair_mask,
    )


def u_face_coriolis_from_haloed_v_row_metrics_cuda(
    v_ext: torch.Tensor,
    corner_value: torch.Tensor,
    *,
    dxG_south: torch.Tensor,
    dxG_north: torch.Tensor,
    recip_dxC_u: torch.Tensor,
    hFacZ: torch.Tensor | None = None,
    face_mask: torch.Tensor | None = None,
    edge_pair_mask: torch.Tensor | None = None,
    edge_face_couples: bool = True,
) -> torch.Tensor:
    """Use the native core, or its exact eager fallback, and record which."""
    values = (
        v_ext,
        corner_value,
        dxG_south,
        dxG_north,
        recip_dxC_u,
        hFacZ,
        face_mask,
        edge_pair_mask,
    )
    # Two live arithmetic branches share this boundary (spec vim_coriolis_u.yaml
    # ``variant_present``/``variant_absent``): absent hFacZ is the planetary
    # quarter-sum, present is the hFacZ-weighted average.  They are separate
    # ledger variants for the same reason DST3 splits limiter/plain -- different
    # arithmetic, so "the kernel ignited" has to be provable for each.
    variant = "vorticity" if hFacZ is not None else "planetary"

    # GENERATED-BODY ADAPTATION: the generated body
    # spliced the eight values in positionally (``f(*values, ...)``).  Both the
    # eager reference and the strict entry take six of them KEYWORD-ONLY, so
    # every call raised TypeError -- fallback and native path alike.  Keep the
    # keyword form here; ``values`` stays a tuple only for the predicate, whose
    # contract is positional.
    def _eager() -> torch.Tensor:
        return _reference(
            variant,
            v_ext,
            corner_value,
            dxG_south=dxG_south,
            dxG_north=dxG_north,
            recip_dxC_u=recip_dxC_u,
            hFacZ=hFacZ,
            face_mask=face_mask,
            edge_face_couples=edge_face_couples,
            edge_pair_mask=edge_pair_mask,
        )

    # The production boundary already asked ``requested``; this second read makes
    # a direct call with the gate off run the reference instead of tripping the
    # ledger's "dispatched while disabled" guard.
    if not gate(_KEY).effective_value:
        return _eager()

    with dispatch_scope(_KEY):
        # ONE predicate, shared with the strict entry point: the gate can never
        # accept an input the kernel would reject, nor decline one it serves.
        reason = direct_decline_reason(values, edge_face_couples=edge_face_couples)
        if reason is not None:
            _record(outcome="inapplicable", variant=variant, reason=reason)
            return _eager()

        try:
            result = u_face_coriolis_from_haloed_v_row_metrics_cuda_direct(
                v_ext,
                corner_value,
                dxG_south=dxG_south,
                dxG_north=dxG_north,
                recip_dxC_u=recip_dxC_u,
                hFacZ=hFacZ,
                face_mask=face_mask,
                edge_pair_mask=edge_pair_mask,
                edge_face_couples=edge_face_couples,
            )
        except Exception as error:
            # Rank-local fail-fast is the repository convention for numerical
            # cores; a silent fallback here would move this kernel into the
            # backend-mixing family.
            _record(outcome="error", variant=variant, reason=type(error).__name__)
            raise
        _record(outcome="hit", variant=variant)
        return result


__all__ = ["u_face_coriolis_from_haloed_v_row_metrics_cuda"]
