"""Unified gate, custom-op mount, and consumer orchestration for the fused
horizontal (X then Y) multidimensional tracer-advection passes.

Enablement comes from the accelerator gate, never from a private environment
read, and every outcome is recorded once against the unified ignition ledger.

The consumer boundary is
``dynamics/tracers.py:HorizontalTracerAdvection._multidim_advective_tendency``
in its ``horizontal_only=True`` form -- the production hand-off to the implicit
vertical solve.  :func:`fused_horizontal_multidim_tendency` returns ``None`` for
every configuration it cannot reproduce bitwise, and the caller then runs the
untouched eager body.

Halo policy: the four ``exchange`` calls of each directional pass stay here, in
Python, byte-for-byte the ones ``tracer_flux._stencil_face_flux_{x,y}`` posts: the
same topology-driven fill (``_dst3_halo_fill``), the same face-mask ghosts
(``_dst3_face_mask_halo``) and the same DST3 splice range
(``_dst3_splice_range``), so a coupled edge face (periodic wrap or distributed
seam) takes the DST3 value on both paths (numerics rule 11).  The
kernel only reads an already-exchanged r=2 halo and never communicates.  The
stencil-width backstop is evaluated BEFORE any exchange of either pass, so a
distributed tile too narrow for the stencil fails with zero traffic posted
(``tracer_flux._require_stencil_width``).

Two predicates, two subjects, each a single source:
  * :func:`_scope_decline_reason` -- can this fused orchestration stand in for
    the eager body at all?  Evaluated before any exchange.
  * ``_fast.direct_decline_reason`` -- can the kernel serve these tensors?
    Shared verbatim with the strict entry point ``_fast._check_direct``.

Distributed note:
both predicates read the shared grad mode, module configuration and
local tensor device/dtype/shape, which are SPMD-uniform on a homogeneous run.
Neither contains a collective, so neither can wedge; divergence, if a
heterogeneous setup ever produced it, is caught at the end-of-run
``verify_dispatch_uniformity`` barrier.
"""

# NOTE: no ``from __future__ import annotations`` here.  ``torch.library``
# infers the custom op's schema from the runtime annotations of
# ``tracer_adv_fused_pass``; stringised annotations make that inference fail.

from typing import Any

import torch
from defgcm.accel.tensor_context import grad_mode_requires_reference

from defgcm.accel import current_rank, dispatch_scope, gate, record_dispatch

from ._fast import _load_extension, direct_decline_reason, tracer_adv_pass_cuda_direct


_KEY = "tracer_adv_fused_cuda"
_SITE = "_multidim_advective_tendency"
_BACKEND = "cuda"
_SUPPORTED_DTYPES = (torch.float64, torch.float32)


def load_extension() -> Any:
    """Load and return the fused tracer-advection CUDA extension."""
    return _load_extension()


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


@torch.library.custom_op("defgcm::tracer_adv_fused_pass", mutates_args=())
def tracer_adv_fused_pass(
    t_wide: torch.Tensor,
    t_left: torch.Tensor,
    t_right: torch.Tensor,
    mask_wide: torch.Tensor,
    vel_face: torch.Tensor,
    vel_trans: torch.Tensor,
    recip_dc: torch.Tensor,
    recip_vol: torch.Tensor,
    mask_in_c: torch.Tensor,
    tracer_orig: torch.Tensor,
    local_t: torch.Tensor,
    dt: float,
    dst_start: int,
    dst_stop: int,
    with_limiter: bool,
    has_mask_in_c: bool,
    emit_tendency: bool,
    axis: str,
) -> torch.Tensor:
    values = (
        t_wide,
        t_left,
        t_right,
        mask_wide,
        vel_face,
        vel_trans,
        recip_dc,
        recip_vol,
        mask_in_c,
        tracer_orig,
        local_t,
    )
    return tracer_adv_pass_cuda_direct(
        values,
        dt=dt,
        dst_start=dst_start,
        dst_stop=dst_stop,
        with_limiter=with_limiter,
        has_mask_in_c=has_mask_in_c,
        emit_tendency=emit_tendency,
        axis=axis,
    )


@tracer_adv_fused_pass.register_fake
def _tracer_adv_fused_pass_fake(
    t_wide, t_left, t_right, mask_wide, vel_face, vel_trans, recip_dc,
    recip_vol, mask_in_c, tracer_orig, local_t, dt, dst_start, dst_stop,
    with_limiter, has_mask_in_c, emit_tendency, axis,
):
    del t_wide, t_left, t_right, mask_wide, vel_face, vel_trans, recip_dc
    del recip_vol, mask_in_c, local_t, dt, dst_start, dst_stop
    del with_limiter, has_mask_in_c, emit_tendency, axis
    return torch.empty_like(tracer_orig)


def tracer_adv_pass_dispatch(
    values: tuple[torch.Tensor, ...],
    *,
    dt: float,
    dst_start: int,
    dst_stop: int,
    with_limiter: bool,
    has_mask_in_c: bool,
    emit_tendency: bool,
    axis: str,
) -> torch.Tensor | None:
    """Return one fused pass, or ``None`` as a counted request-open fallback."""
    if not gate(_KEY).effective_value:
        return None
    variant = f"{axis}-{'limiter' if with_limiter else 'plain'}"
    # The scope is held only around the native call: the eager fallback this
    # ``None`` hands back runs at the caller, outside it, so an unrelated
    # accelerator on that path is not suppressed.
    with dispatch_scope(_KEY):
        # ONE predicate, shared with the strict entry point: the gate can never
        # accept an input the kernel would reject, nor decline one it serves.
        reason = direct_decline_reason(
            values,
            axis=axis,
            with_limiter=with_limiter,
            emit_tendency=emit_tendency,
            has_mask_in_c=has_mask_in_c,
            dst_start=dst_start,
            dst_stop=dst_stop,
        )
        if reason is not None:
            _record(outcome="inapplicable", variant=variant, reason=reason)
            return None
        try:
            output = tracer_adv_fused_pass(
                *values,
                float(dt),
                int(dst_start),
                int(dst_stop),
                bool(with_limiter),
                bool(has_mask_in_c),
                bool(emit_tendency),
                axis,
            )
        except Exception as error:
            _record(outcome="error", variant=variant, reason=type(error).__name__)
            raise
        _record(outcome="hit", variant=variant)
        return output


# ---------------------------------------------------------------------------
# Consumer orchestration
# ---------------------------------------------------------------------------


def _as_four_dimensional(value: torch.Tensor) -> torch.Tensor:
    while value.ndim < 4:
        value = value.unsqueeze(0)
    return value


def _scope_decline_reason(
    module: Any,
    tracer: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    grid: Any,
    dt: float,
) -> str | None:
    """First eager behaviour this orchestration cannot reproduce, or ``None``.

    Every rule names an eager branch the fused path does not implement, so a
    ``None`` verdict is the claim "the eager body reduces exactly to the two
    fused passes below".
    """
    from defgcm.dynamics import tracers as _tracers
    from defgcm.operators.fluxes.tracer_flux import TracerFluxScheme

    # Configuration rules first, device rules after: the configuration ones are
    # device-independent, so a CPU harness can observe every one of them, and a
    # ledger row then names the most specific unmet rule rather than "non_cuda".
    if module.scheme not in (TracerFluxScheme.DST3, TracerFluxScheme.DST3_FL):
        return "scheme"
    if not (dt > 0):
        return "timestep"
    # The cube tile adapter reaches a different flux path entirely.
    if int(getattr(grid, "face_count", 1)) != 1:
        return "cube"
    # tracers.py:731-732 / 801-802 wrap the face flux in an extra OBC upwind
    # override; the kernel does not implement it.
    topology = grid.boundary_topology
    has_obcs = topology.open_x or topology.open_y
    if has_obcs and module.obcs_upwind and topology.open_x:
        return "obcs_upwind_x"
    if has_obcs and module.obcs_upwind and topology.open_y:
        return "obcs_upwind_y"
    # The debug/capture/leaf-inject hooks all need materialised intermediates.
    if _tracers._MULTIDIM_DEBUG_STATE["enabled"]:
        return "debug_state"
    if _tracers._MULTIDIM_CAPTURE["enabled"]:
        return "capture"
    if _tracers._MULTIDIM_LEAF_INJECT["enabled"]:
        return "leaf_inject"
    if tracer.ndim != 4 or u.ndim != 4 or v.ndim != 4:
        return "rank"
    if tracer.device.type != "cuda":
        return "non_cuda"
    if tracer.dtype not in _SUPPORTED_DTYPES:
        return "dtype"
    if u.dtype != tracer.dtype or v.dtype != tracer.dtype:
        return "velocity_dtype"
    if u.device != tracer.device or v.device != tracer.device:
        return "device_mismatch"
    if grad_mode_requires_reference():
        return "autograd"
    return None


def _mask_loc(
    face_hfac: torch.Tensor | None,
    face_in_mask: torch.Tensor | None,
    velocity: torch.Tensor,
    tracer: torch.Tensor,
) -> torch.Tensor:
    """MITgcm ``maskLocW/S = _maskW/S * maskInW/S`` (gad_advection.F:321-323).

    Mirrors ``tracer_flux.py:320-333`` (X) and ``tracer_flux.py:464-473`` (Y)
    operation for operation.
    """
    if face_hfac is None:
        return torch.ones_like(velocity)
    hfac = face_hfac.to(dtype=tracer.dtype, device=tracer.device)
    mask = (hfac > 0).to(dtype=tracer.dtype)
    if face_in_mask is not None:
        mask = mask * face_in_mask.to(dtype=tracer.dtype, device=tracer.device)
    return mask


def fused_horizontal_multidim_tendency(
    module: Any,
    tracer: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    grid: Any,
    dt: float,
) -> torch.Tensor | None:
    """Fused ``horizontal_only`` X→Y tendency, or ``None`` to run eager.

    The production wiring point in ``tracers.py`` has already resolved the gate
    before importing this package, so in production these guards are a second
    read of the same snapshot (``consumer_snapshot`` reuses the active one).
    They stay because this is also the entry a direct caller or a test uses,
    and such a call must not dispatch while the switch is off.
    """
    from defgcm.accel import (
        consumer_snapshot,
        in_accelerated_dispatch,
        requested,
    )

    if in_accelerated_dispatch():
        return None
    with consumer_snapshot():
        if not requested((_KEY,)):
            return None
        return _run_horizontal(module, tracer, u, v, grid, dt)


def _run_horizontal(
    module: Any,
    tracer: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    grid: Any,
    dt: float,
) -> torch.Tensor | None:
    from defgcm.dynamics.tracers import (
        _calc_adv_flow_transports_mit_order,
    )
    from defgcm.geometry.exchange import exchange
    from defgcm.geometry.metric_bundle import build_tracer_metric_bundle
    from defgcm.distributed.halo_p2p import after_compute
    from defgcm.operators.fluxes.tracer_flux import (
        TracerFluxScheme,
        _X_LEFT_EXTENT,
        _X_RIGHT_EXTENT,
        _Y_BOTTOM_EXTENT,
        _Y_TOP_EXTENT,
        _cell_stencil_extent,
        _dst3_face_mask_halo,
        _dst3_halo_fill,
        _dst3_splice_range,
        _require_stencil_width,
    )

    scope_reason = _scope_decline_reason(module, tracer, u, v, grid, dt)
    if scope_reason is not None:
        _record(outcome="inapplicable", variant="scope", reason=scope_reason)
        return None

    nx, ny = int(grid.nx), int(grid.ny)
    layout = getattr(grid, "parallel_layout", None)
    order_with_collectives = layout is not None and layout.world_size > 1
    # Both width verdicts are taken before ANY exchange of either pass: a
    # distributed tile too narrow raises here with zero traffic posted, which is
    # the posture tracer_flux._require_stencil_width documents.  A single patch
    # too narrow returns False and the eager full-upwind fallback runs instead.
    if not _require_stencil_width(grid, nx, axis="x", halo=2):
        _record(
            outcome="inapplicable", variant="scope", reason="stencil_width_x"
        )
        return None
    if not _require_stencil_width(grid, ny, axis="y", halo=2):
        _record(
            outcome="inapplicable", variant="scope", reason="stencil_width_y"
        )
        return None

    with_limiter = module.scheme == TracerFluxScheme.DST3_FL
    dtype, device = tracer.dtype, tracer.device

    # tracers.py:663-696 verbatim: same bundle, same hFac guards, same MIT
    # transport operation order.
    bundle = build_tracer_metric_bundle(grid, dtype, device)
    hFacW = bundle.hFacW
    hFacS = bundle.hFacS
    recip_local_volume = bundle.recip_volume

    topology = grid.boundary_topology
    maskInC = None
    if topology.open_x or topology.open_y:
        from defgcm.geometry.grid import _compute_interior_mask

        maskInC_2d, _, _ = _compute_interior_mask(grid, dtype, device)
        maskInC = maskInC_2d.unsqueeze(0).unsqueeze(0)

    u_trans, v_trans = _calc_adv_flow_transports_mit_order(
        u, v, bundle.dyG, bundle.dxG, bundle.layer_thickness,
        hFacW=hFacW, hFacS=hFacS,
    )

    has_mask_in_c = maskInC is not None
    # The kernel never dereferences the placeholder; its rank and dtype are all
    # the native shape check reads when has_mask_in_c is False.
    mask_in_c = (
        maskInC
        if has_mask_in_c
        else torch.zeros((1, 1, 1, 1), dtype=dtype, device=device)
    )
    recip_vol_4d = _as_four_dimensional(recip_local_volume)
    # Halo fills, face-mask ghosts and the DST3 splice range come from the same
    # tracer_flux helpers the eager adapters call, so the seam semantics
    # (numerics rule 11) have one source for both paths.
    fill_x = _dst3_halo_fill(grid, "x")
    fill_y = _dst3_halo_fill(grid, "y")
    dst_start_x, dst_stop_x = _dst3_splice_range(grid, "x")
    dst_start_y, dst_stop_y = _dst3_splice_range(grid, "y")

    # ---- Pass 1: X-direction (tracer_flux.py _stencil_face_flux_x + tracers.py:734-763) ---
    mask_loc_w = _mask_loc(grid.masks.hFacW, grid.masks.maskInW, u, tracer)
    t_left = exchange(
        tracer, grid=grid, location="T",
        extent=_X_LEFT_EXTENT, fill=fill_x,
    )
    t_right = exchange(
        tracer, grid=grid, location="T",
        extent=_X_RIGHT_EXTENT, fill=fill_x,
    )
    tracer_wide_x = exchange(
        tracer, grid=grid, location="T",
        extent=_cell_stencil_extent("x", 2), fill=fill_x,
    )
    mask_wide_x = _dst3_face_mask_halo(mask_loc_w, grid, "x", halo=2)
    recip_dxC = (1.0 / grid.metrics.dxC_for(dtype, device)).view(1, 1, ny, nx + 1)

    # tracers.py: localT = tracer.clone() is a value-identical copy that the
    # eager passes never mutate in place, so the fused path reads ``tracer``
    # directly as the X pass base.
    local_x = tracer_adv_pass_dispatch(
        (
            tracer_wide_x,
            t_left,
            t_right,
            _as_four_dimensional(mask_wide_x),
            u,
            u_trans,
            recip_dxC,
            recip_vol_4d,
            mask_in_c,
            tracer,
            tracer,
        ),
        dt=dt,
        dst_start=dst_start_x,
        dst_stop=dst_stop_x,
        with_limiter=with_limiter,
        has_mask_in_c=has_mask_in_c,
        emit_tendency=False,
        axis="x",
    )
    if local_x is None:
        return None
    if order_with_collectives:
        after_compute(local_x)

    # ---- Pass 2: Y-direction (tracer_flux.py _stencil_face_flux_y + tracers.py:804-848) ---
    mask_loc_s = _mask_loc(grid.masks.hFacS, grid.masks.maskInS, v, tracer)
    t_bottom = exchange(
        local_x, grid=grid, location="T",
        extent=_Y_BOTTOM_EXTENT, fill=fill_y,
    )
    t_top = exchange(
        local_x, grid=grid, location="T",
        extent=_Y_TOP_EXTENT, fill=fill_y,
    )
    tracer_wide_y = exchange(
        local_x, grid=grid, location="T",
        extent=_cell_stencil_extent("y", 2), fill=fill_y,
    )
    mask_wide_y = _dst3_face_mask_halo(mask_loc_s, grid, "y", halo=2)
    recip_dyC = (1.0 / grid.metrics.dyC_for(dtype, device)).view(1, 1, ny + 1, nx)

    tendency = tracer_adv_pass_dispatch(
        (
            tracer_wide_y,
            t_bottom,
            t_top,
            _as_four_dimensional(mask_wide_y),
            v,
            v_trans,
            recip_dyC,
            recip_vol_4d,
            mask_in_c,
            tracer,
            local_x,
        ),
        dt=dt,
        dst_start=dst_start_y,
        dst_stop=dst_stop_y,
        with_limiter=with_limiter,
        has_mask_in_c=has_mask_in_c,
        emit_tendency=True,
        axis="y",
    )
    if order_with_collectives and tendency is not None:
        after_compute(tendency)
    return tendency


__all__ = [
    "direct_decline_reason",
    "fused_horizontal_multidim_tendency",
    "load_extension",
    "tracer_adv_fused_pass",
    "tracer_adv_pass_dispatch",
]
