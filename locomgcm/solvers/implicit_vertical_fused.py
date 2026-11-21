"""Fused band-assembly and Thomas vertical-line solver.

The public eager path in :mod:`locomgcm.solvers.implicit_vertical` remains
the numerical reference.  This module provides one explicit, no-fallback
acceleration candidate for well-conditioned physical systems:

``cuda``
    A native CUDA column-tile kernel. Adjacent threads issue coalesced loads
    for one level at a time while each thread retains an independent vertical
    recurrence. Forward-sweep ``c'`` uses a level-major global workspace;
    ``d'`` uses the final output buffer and is overwritten in-place by the
    backward sweep.

The candidate spells the elimination division as reciprocal-then-multiply,
matching MITgcm's source-level operation sequence.  Compiler contraction can
still make the accelerated result differ from eager by floating-point order.
The tracer entry supplies a first-order native transpose and physical-input
pullback. It never silently falls back.
It reproduces the eager solver's per-level pivot and quotient diagnostic
contract on device, in either dtype.  The native runtime publishes every
forward and reverse code to the numerical-failure channel
(``locomgcm.deferred_checks``); this module neither reads nor writes it.

Precision
---------
``float64`` is the aligned production path and the reference for every bitwise
comparison against MITgcm.

``float32`` was added as a **performance experiment only** and is not a
validated numerical path.  The former fp64-only contract has been relaxed on
explicit request: the kernel now selects the pivot solvability threshold by
dtype (``8*DBL_EPSILON`` for float64, ``8*FLT_EPSILON`` for float32) instead of
baking the float64 literal.  The float32 threshold is the mechanical eps
rescaling of the float64 one; it has **not** been shown to separate solvable
from unsolvable columns at float32 conditioning, no float32 result from this
solver has been validated against MITgcm or against the eager reference, and a
float32 solve of a stiff vertical system may return a plausible-looking but
wrong answer without tripping the diagnostic contract.  Do not read float32
output as physics.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Literal

import torch

from locomgcm.accel import current_rank, dispatch_scope, gate, record_dispatch

from ._implicit_vertical_cuda._fast import SUPPORTED_DTYPES
from .implicit_vertical import (
    MomentumWNmHalfMIT,
    TracerVerticalMetrics,
    TracerWNpHalfMIT,
    UVerticalMetrics,
    VVerticalMetrics,
    _coefficient,
    _float_time_step,
    _horizontal_view,
    _require_target_field,
    _scalar_or_vertical_dt,
    _validate_physical_w,
    _validate_tracer_metrics,
    _validate_u_metrics,
    _validate_v_metrics,
    _vertical_view,
    reverse_mode_requested,
)


FusedBackend = Literal["cuda"]
_KEY = "ivadv"
_BACKEND = "cuda"


def configured_fused_vertical_backend() -> FusedBackend | None:
    """Return the production backend resolved by the unified accelerator gate.

    The switch is owned by :mod:`locomgcm.accel`: this reads the value frozen
    by the active step snapshot rather than the environment, so every consumer
    within one step sees the same decision.  The production switch stays
    disabled unless the gate resolves to true, and the unified grammar rejects
    every unsupported spelling, so a run configured for a removed arm cannot
    silently execute the eager path.
    """
    return _BACKEND if gate(_KEY).effective_value else None


def fused_vertical_backend_available(backend: FusedBackend) -> bool:
    """Report whether the explicit candidate can be attempted in this process."""
    if backend == "cuda":
        return torch.cuda.is_available()
    raise ValueError(f"unknown fused vertical backend {backend!r}")


def _record(
    outcome: str, *, variant: str, site: str, reason: str | None = None
) -> None:
    record_dispatch(
        _KEY,
        outcome=outcome,
        variant=variant,
        site=site,
        backend=_BACKEND,
        rank=current_rank(),
        reason=reason,
        count=1,
    )


@contextmanager
def _accounted_dispatch(*, variant: str, site: str) -> Iterator[None]:
    """Post one solver entry outcome to the unified ignition ledger.

    A completed solve is one ``hit``.  Every raise this entry already produces
    -- argument validation, the CUDA device/dtype precondition, and a solver
    code judged at the call (``strict``, or no prepared sink) -- is posted as
    ``error`` carrying the exception class name and then re-raised unchanged.  ``inapplicable`` is
    never used: this client declares ``unmet_policy='raise'``, under which the
    unified runtime would turn such a record into an unmet-gate error of its
    own, and this solver has no fallback arm to decline into.
    """
    with dispatch_scope(_KEY):
        try:
            yield
        except Exception as error:
            _record("error", variant=variant, site=site, reason=type(error).__name__)
            raise
        _record("hit", variant=variant, site=site)


def _as_field4(value: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    if value.ndim == 3:
        return value.unsqueeze(0), True
    if value.ndim == 4:
        return value, False
    raise RuntimeError(
        f"fused vertical {name} must be [N,ny,nx] or [batch,N,ny,nx]; "
        f"got {tuple(value.shape)}"
    )


def _expand4(
    value: torch.Tensor, shape: tuple[int, int, int, int], name: str
) -> torch.Tensor:
    while value.ndim < 4:
        value = value.unsqueeze(0)
    if value.ndim != 4:
        raise RuntimeError(
            f"fused vertical {name} cannot broadcast from {value.ndim}D"
        )
    try:
        return value.expand(shape)
    except RuntimeError as error:
        raise RuntimeError(
            f"fused vertical {name} shape {tuple(value.shape)} cannot broadcast "
            f"to {shape}"
        ) from error


class _TracerPhysicalSolve(torch.autograd.Function):
    """Native primal with a first-order physical-input pullback.

    The native transpose reconstructs rows through the shared primal accessor.
    The physical pullback writes requested cotangents without band tensors.
    Forward-mode continues to use the public reference dispatcher.
    """

    @staticmethod
    def forward(*inputs):
        *arguments, advection_scheme, diagnostic_contract = inputs
        return _cuda_tracer_direct(
            tuple(value.detach() for value in arguments),
            advection_scheme=advection_scheme,
            diagnostic_contract=diagnostic_contract,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:-2], output[0])
        ctx.advection_scheme = inputs[-2]
        ctx.mark_non_differentiable(output[1])

    @staticmethod
    def backward(ctx, grad_solution, _grad_error):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "native tracer solve supports first-order reverse differentiation only; "
                "use the reference solve for higher-order derivatives"
            )
        *arguments, solution = ctx.saved_tensors
        bars = _cuda_tracer_backward(
            tuple(arguments), solution, grad_solution, ctx.needs_input_grad[:10],
            advection_scheme=ctx.advection_scheme,
        )
        return *bars, None, None



def _cuda_tracer_direct(
    arguments: tuple[torch.Tensor, ...], *, advection_scheme: int,
    diagnostic_contract: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse tracer precursor construction, band assembly, and Thomas solve."""
    if reverse_mode_requested(*arguments):
        return _TracerPhysicalSolve.apply(*arguments, advection_scheme, diagnostic_contract)
    from ._implicit_vertical_cuda import tracer_cuda

    values, squeeze_batch = _tracer_cuda_values(arguments)
    rhs4 = values[0]
    if rhs4.device.type != "cuda" or rhs4.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            "native fused vertical CUDA dispatch requires a CUDA float64 or "
            "float32 rhs"
        )
    output, error_code = tracer_cuda(
        values, advection_scheme=advection_scheme, diagnostic_contract=diagnostic_contract
    )
    result = output.squeeze(0) if squeeze_batch else output
    return result, error_code


def _tracer_cuda_values(arguments):
    (
        rhs,
        w_data,
        dt,
        diffusivity,
        recip_hfac,
        mask,
        area,
        reciprocal_area,
        recip_dr_f,
        recip_dr_c,
    ) = arguments
    rhs4, squeeze_batch = _as_field4(rhs, "rhs")
    batch, nz, ny, nx = (int(value) for value in rhs4.shape)
    shape = (batch, nz, ny, nx)
    values = (
        rhs4,
        _expand4(dt, shape, "dt"),
        _expand4(diffusivity, shape, "diffusivity"),
        _expand4(recip_hfac, shape, "recip_hfac"),
        _expand4(mask, shape, "mask"),
        _expand4(_horizontal_view(area), (batch, 1, ny, nx), "area"),
        _expand4(
            _horizontal_view(reciprocal_area),
            (batch, 1, ny, nx),
            "reciprocal_area",
        ),
        _expand4(
            _vertical_view(recip_dr_f, rhs.ndim), shape, "recip_dr_f"
        ),
        _expand4(
            _vertical_view(recip_dr_c, rhs.ndim),
            (batch, max(nz - 1, 0), ny, nx),
            "recip_dr_c",
        ),
        _expand4(w_data, (batch, nz + 1, ny, nx), "w"),
    )
    return values, squeeze_batch


def _cuda_tracer_backward(arguments, solution, grad_solution, needs, *, advection_scheme):
    from ._implicit_vertical_cuda._fast import implicit_vertical_tracer_bwd

    values, _ = _tracer_cuda_values(arguments)
    solution4, _ = _as_field4(solution, "solution")
    seed4, _ = _as_field4(grad_solution, "grad_solution")
    bars, _error_code = implicit_vertical_tracer_bwd(
        list(values), solution4, seed4, list(needs), advection_scheme, True,
    )
    # The opaque runtime publishes the code; no channel state enters the
    # mathematical graph or AOT's saved values.
    # Native outputs are expanded physical cotangents; reduce only active inputs.
    result = []
    for index, (bar, original) in enumerate(zip(bars, arguments)):
        if bar is None:
            result.append(None)
        elif index in (6, 7):
            result.append(bar.squeeze(-3).sum_to_size(original.shape))
        elif index in (8, 9):
            result.append(bar.sum(dim=(0, 2, 3)))
        else:
            result.append(bar.sum_to_size(original.shape))
    return tuple(result)


def _cuda_momentum_direct(
    arguments: tuple[torch.Tensor | bool, ...],
    *,
    diagnostic_contract: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse momentum precursor construction, band assembly, and Thomas solve."""
    from ._implicit_vertical_cuda import momentum_cuda

    (
        rhs,
        w_left,
        w_right,
        dt,
        viscosity,
        recip_hfac,
        mask,
        area_left,
        area_right,
        mask_left,
        mask_right,
        reciprocal_face_area,
        recip_dr_f,
        recip_dr_c,
        upwind_shear,
    ) = arguments
    assert isinstance(rhs, torch.Tensor)
    assert isinstance(upwind_shear, bool)
    tensors = (
        w_left,
        w_right,
        dt,
        viscosity,
        recip_hfac,
        mask,
        area_left,
        area_right,
        mask_left,
        mask_right,
        reciprocal_face_area,
        recip_dr_f,
        recip_dr_c,
    )
    assert all(isinstance(value, torch.Tensor) for value in tensors)
    rhs4, squeeze_batch = _as_field4(rhs, "rhs")
    if rhs4.device.type != "cuda" or rhs4.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            "native fused vertical CUDA dispatch requires a CUDA float64 or "
            "float32 rhs"
        )
    batch, nz, ny, nx = (int(value) for value in rhs4.shape)
    shape = (batch, nz, ny, nx)
    values = (
        rhs4,
        _expand4(dt, shape, "dt"),
        _expand4(viscosity, shape, "viscosity"),
        _expand4(recip_hfac, shape, "recip_hfac"),
        _expand4(mask, shape, "mask"),
        _expand4(w_left, (batch, nz + 1, ny, nx), "w_left"),
        _expand4(w_right, (batch, nz + 1, ny, nx), "w_right"),
        _expand4(
            _horizontal_view(area_left), (batch, 1, ny, nx), "area_left"
        ),
        _expand4(
            _horizontal_view(area_right), (batch, 1, ny, nx), "area_right"
        ),
        _expand4(mask_left, shape, "mask_left"),
        _expand4(mask_right, shape, "mask_right"),
        _expand4(
            _horizontal_view(reciprocal_face_area),
            (batch, 1, ny, nx),
            "reciprocal_face_area",
        ),
        _expand4(
            _vertical_view(recip_dr_f, rhs.ndim), shape, "recip_dr_f"
        ),
        _expand4(
            _vertical_view(recip_dr_c, rhs.ndim),
            (batch, max(nz - 1, 0), ny, nx),
            "recip_dr_c",
        ),
    )
    output, error_code = momentum_cuda(
        values,
        upwind=upwind_shear,
        diagnostic_contract=diagnostic_contract,
    )
    result = output.squeeze(0) if squeeze_batch else output
    return result, error_code


def solve_tracer_band_thomas_fused(
    *,
    rhs: torch.Tensor,
    advection_scheme: int,
    w: TracerWNpHalfMIT,
    dt_lev: float | torch.Tensor,
    diffusivity: float | torch.Tensor,
    metrics: TracerVerticalMetrics,
    backend: FusedBackend,
    _diagnostic_contract: bool = True,
) -> torch.Tensor:
    """Assemble and solve one tracer system through an explicit candidate."""
    with _accounted_dispatch(
        variant="tracer", site="solve_tracer_band_thomas_fused"
    ):
        _require_target_field("rhs", rhs)
        _validate_tracer_metrics(rhs, metrics)
        if not isinstance(w, TracerWNpHalfMIT):
            raise TypeError("w must be a TracerWNpHalfMIT object")
        _validate_physical_w(
            name="w",
            data=w.data,
            center_field=rhs,
            mask_c=metrics.mask_c,
        )
        dt = _scalar_or_vertical_dt(dt_lev, field=rhs, name="dt_lev")
        kappa = _coefficient(diffusivity, field=rhs, name="diffusivity")
        arguments = (
            rhs,
            w.data,
            dt,
            kappa,
            metrics.recip_hfac_matrix_c,
            metrics.mask_c,
            metrics.rA,
            metrics.recip_rA,
            metrics.recip_drF,
            metrics.recip_drC,
        )
        if backend != "cuda":
            raise ValueError(f"unknown fused vertical backend {backend!r}")
        result, _error_code = _cuda_tracer_direct(
            arguments, advection_scheme=advection_scheme,
            diagnostic_contract=_diagnostic_contract
        )
        return result


def _momentum_target_views(
    *,
    rhs: torch.Tensor,
    w_data: torch.Tensor,
    viscosity: torch.Tensor,
    row_recip_hfac: torch.Tensor,
    row_mask: torch.Tensor,
    mask_c: torch.Tensor,
    r_a: torch.Tensor,
    reciprocal_face_area: torch.Tensor,
    axis: int,
) -> tuple[torch.Tensor, ...]:
    target = [slice(None)] * rhs.ndim
    target[axis] = slice(1, -1)
    key = tuple(target)
    rhs_target = rhs[key]
    viscosity_target = viscosity if viscosity.ndim == 0 else viscosity[key]
    row_recip_target = row_recip_hfac[key[-row_recip_hfac.ndim :]]
    row_mask_target = row_mask[key[-row_mask.ndim :]]
    reciprocal_area_target = reciprocal_face_area[key[-reciprocal_face_area.ndim :]]

    if axis == -1:
        w_left = w_data[..., :-1]
        w_right = w_data[..., 1:]
        mask_left = mask_c[..., :-1]
        mask_right = mask_c[..., 1:]
        area_left = r_a[..., :-1]
        area_right = r_a[..., 1:]
    elif axis == -2:
        w_left = w_data[..., :-1, :]
        w_right = w_data[..., 1:, :]
        mask_left = mask_c[..., :-1, :]
        mask_right = mask_c[..., 1:, :]
        area_left = r_a[..., :-1, :]
        area_right = r_a[..., 1:, :]
    else:  # pragma: no cover - only the two staggered axes are callers
        raise ValueError(f"unsupported momentum face axis {axis}")
    return (
        rhs_target,
        w_left,
        w_right,
        viscosity_target,
        row_recip_target,
        row_mask_target,
        area_left,
        area_right,
        mask_left,
        mask_right,
        reciprocal_area_target,
    )


def _solve_momentum_band_thomas_fused(
    *,
    rhs: torch.Tensor,
    w: MomentumWNmHalfMIT,
    dt_mom: float,
    viscosity: float | torch.Tensor,
    metrics: UVerticalMetrics | VVerticalMetrics,
    upwind_shear: bool,
    backend: FusedBackend,
    location: Literal["u", "v"],
    diagnostic_contract: bool,
) -> torch.Tensor:
    _require_target_field(f"rhs_{location}", rhs)
    if location == "u":
        assert isinstance(metrics, UVerticalMetrics)
        _validate_u_metrics(rhs, metrics)
        row_recip_hfac = metrics.recip_hfac_w
        row_mask = metrics.mask_w
        reciprocal_face_area = metrics.recip_rAw
        axis = -1
    else:
        assert isinstance(metrics, VVerticalMetrics)
        _validate_v_metrics(rhs, metrics)
        row_recip_hfac = metrics.recip_hfac_s
        row_mask = metrics.mask_s
        reciprocal_face_area = metrics.recip_rAs
        axis = -2
    _validate_physical_w(
        name="w",
        data=w.data,
        center_field=metrics.mask_c,
        mask_c=metrics.mask_c,
        field_prefix=tuple(rhs.shape[:-3]),
    )
    if type(upwind_shear) is not bool:
        raise TypeError(
            f"upwind_shear must be a bool; got {type(upwind_shear).__name__}"
        )
    dt = _float_time_step(dt_mom, field=rhs, name="dt_mom")
    kappa = _coefficient(viscosity, field=rhs, name="viscosity")
    views = _momentum_target_views(
        rhs=rhs,
        w_data=w.data,
        viscosity=kappa,
        row_recip_hfac=row_recip_hfac,
        row_mask=row_mask,
        mask_c=metrics.mask_c,
        r_a=metrics.rA,
        reciprocal_face_area=reciprocal_face_area,
        axis=axis,
    )
    (
        rhs_target,
        w_left,
        w_right,
        viscosity_target,
        row_recip_target,
        row_mask_target,
        area_left,
        area_right,
        mask_left,
        mask_right,
        reciprocal_area_target,
    ) = views
    arguments = (
        rhs_target,
        w_left,
        w_right,
        dt,
        viscosity_target,
        row_recip_target,
        row_mask_target,
        area_left,
        area_right,
        mask_left,
        mask_right,
        reciprocal_area_target,
        metrics.recip_drF,
        metrics.recip_drC,
        upwind_shear,
    )
    if backend != "cuda":
        raise ValueError(f"unknown fused vertical backend {backend!r}")
    result, _error_code = _cuda_momentum_direct(
        arguments, diagnostic_contract=diagnostic_contract
    )
    return result


def solve_u_band_thomas_fused(
    *,
    rhs_u: torch.Tensor,
    w: MomentumWNmHalfMIT,
    dt_mom: float,
    viscosity: float | torch.Tensor,
    metrics: UVerticalMetrics,
    upwind_shear: bool,
    backend: FusedBackend,
    _diagnostic_contract: bool = True,
) -> torch.Tensor:
    """Solve a prepared, halo-extended U system and return its physical slice."""
    with _accounted_dispatch(variant="u", site="solve_u_band_thomas_fused"):
        if not isinstance(metrics, UVerticalMetrics):
            raise TypeError("metrics must be a UVerticalMetrics object")
        return _solve_momentum_band_thomas_fused(
            rhs=rhs_u,
            w=w,
            dt_mom=dt_mom,
            viscosity=viscosity,
            metrics=metrics,
            upwind_shear=upwind_shear,
            backend=backend,
            location="u",
            diagnostic_contract=_diagnostic_contract,
        )


def solve_v_band_thomas_fused(
    *,
    rhs_v: torch.Tensor,
    w: MomentumWNmHalfMIT,
    dt_mom: float,
    viscosity: float | torch.Tensor,
    metrics: VVerticalMetrics,
    upwind_shear: bool,
    backend: FusedBackend,
    _diagnostic_contract: bool = True,
) -> torch.Tensor:
    """Solve a prepared, halo-extended V system and return its physical slice."""
    with _accounted_dispatch(variant="v", site="solve_v_band_thomas_fused"):
        if not isinstance(metrics, VVerticalMetrics):
            raise TypeError("metrics must be a VVerticalMetrics object")
        return _solve_momentum_band_thomas_fused(
            rhs=rhs_v,
            w=w,
            dt_mom=dt_mom,
            viscosity=viscosity,
            metrics=metrics,
            upwind_shear=upwind_shear,
            backend=backend,
            location="v",
            diagnostic_contract=_diagnostic_contract,
        )


__all__ = [
    "FusedBackend",
    "configured_fused_vertical_backend",
    "fused_vertical_backend_available",
    "solve_tracer_band_thomas_fused",
    "solve_u_band_thomas_fused",
    "solve_v_band_thomas_fused",
]
