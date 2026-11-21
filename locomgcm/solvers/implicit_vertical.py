"""Implicit vertical-advection tridiagonal systems.

The physical vertical velocity accepted here is the complete MITgcm field:
surface-first, upward-positive, and measured in m/s.  Its surface value may be
non-zero.  The bottom value and values on internal wet-dry interfaces must be
zero.  Operator-specific endpoint projection is performed by each assembler;
the physical wrapper is never modified or zeroed in place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from locomgcm.accel.tensor_context import extent_mismatch, grad_mode_requires_reference

from locomgcm.deferred_checks import strict_checks
from locomgcm.solvers.tridiag import forward_mode_requested

_RK_SIGN = -1.0


def _device_scalar(value: float, field: torch.Tensor) -> torch.Tensor:
    """A 0-d ``field``-typed tensor holding ``value``, built without a host copy.

    Without it there are 2 waits at ``_float_time_step`` + 2 at
    ``_scalar_or_vertical_dt`` per step.  ``field.new_tensor(v)`` materialises
    the scalar on the HOST and then copies it in, which is
    ``cudaMemcpyAsync`` + ``cudaStreamSynchronize`` -- a pipeline drain per
    call.  ``torch.full`` passes the value as a launch argument to
    ``aten::fill_`` and touches the host not at all.

    Bit-neutral: both spellings round the Python double to ``field.dtype``
    exactly once, with the same round-to-nearest, and produce a detached leaf.
    """
    return torch.full((), value, dtype=field.dtype, device=field.device)


class TridiagonalSolveError(RuntimeError):
    """Raised when a system is outside the strict no-pivot Thomas contract."""


@dataclass(frozen=True)
class TridiagonalSystem:
    """A batched tridiagonal system with vertical axis ``dim=-3``.

    ``lower``, ``main``, ``upper``, and ``rhs`` have identical
    ``[..., N, ny, nx]`` shape, dtype, and device.  The first lower row and
    last upper row are exactly zero.  The object carries no offsets, bandwidth,
    drag parameters, location string, or solver registry.
    """

    lower: torch.Tensor
    main: torch.Tensor
    upper: torch.Tensor
    rhs: torch.Tensor


@dataclass(frozen=True)
class MomentumWNmHalfMIT:
    """Momentum-time physical ``w`` at ``n-1/2`` in MIT convention.

    ``data`` has shape ``[..., N+1, ny, nx]``, is surface-first, has units
    m/s, and is upward-positive. The surface value may be non-zero.  The
    bottom value and every internal wet-dry interface value must be zero.
    ``rkSign=-1`` is not applied to this state; it appears only in operators.

    **Tensor only.**  The step builds it from the lagged w the tendencies read
    (``EvaluationInputs.w_interface``); it is not stored across steps and
    carries no step number.  A per-step Python integer on traced state is
    guarded per step by torch 2.9 whether or not anything reads it, which is
    why the step stamp it once had cost 15.5 s a step on the production
    configuration before it moved off the object.
    """

    data: torch.Tensor


@dataclass(frozen=True)
class TracerWNpHalfMIT:
    """Tracer-time physical ``w`` at ``n+1/2`` in MIT convention.

    ``data`` has shape ``[..., N+1, ny, nx]``, is surface-first, has units
    m/s, and is upward-positive.  The surface value may be non-zero.  The
    bottom value and every internal wet-dry interface value must be zero.
    ``rkSign=-1`` is not applied to this state; it appears only in operators.

    Tensor only, for the same reason as :class:`MomentumWNmHalfMIT` and with
    less to say about it: this wrapper is step-local and is built inside the
    traced step, so a Python integer on it is a per-step guard on the frame that
    builds it.  It is not checkpoint state and never was.
    """

    data: torch.Tensor


@dataclass(frozen=True)
class TracerVerticalMetrics:
    """C-point metrics for ``[..., N, ny, nx]`` tracer rows.

    Row fields ``recip_hfac_matrix_c``, ``recip_hfac_current_c``, and
    ``mask_c`` have shape ``[..., N, ny, nx]``.  ``rA``, ``recip_rA``, and
    ``mask_in_c`` have shape ``[..., ny, nx]``.  Metric leading dimensions must
    align with trailing grid-leading dimensions of the tracer; omitted batch
    dimensions are allowed, but location dimensions are not implicitly
    expanded.  ``recip_drF`` has exact shape ``[N]`` and ``recip_drC`` has
    exact shape ``[N-1]`` for the internal interfaces.  All tensors have the
    target dtype and device.
    """

    recip_hfac_matrix_c: torch.Tensor
    recip_hfac_current_c: torch.Tensor
    mask_c: torch.Tensor
    rA: torch.Tensor
    recip_rA: torch.Tensor
    recip_drF: torch.Tensor
    recip_drC: torch.Tensor
    mask_in_c: torch.Tensor


@dataclass(frozen=True)
class UVerticalMetrics:
    """W-point metrics for ``[..., N, ny, nx+1]`` U rows.

    ``recip_hfac_w`` and ``mask_w`` have the U-field shape.  ``rAw`` and
    ``recip_rAw`` have shape ``[..., ny, nx+1]``.  The source C-point ``rA``
    and ``mask_c`` have shapes ``[..., ny, nx]`` and ``[..., N, ny, nx]``;
    their west-neighbour relation must follow the topology-aware metric
    factory.  ``recip_drF`` has exact shape ``[N]`` and ``recip_drC`` has
    exact shape ``[N-1]``.  Metric leading dimensions align with trailing
    grid-leading dimensions of the U field; omitted batch dimensions are
    allowed.  All tensors have the target dtype and device.
    """

    recip_hfac_w: torch.Tensor
    mask_w: torch.Tensor
    mask_c: torch.Tensor
    rA: torch.Tensor
    rAw: torch.Tensor
    recip_rAw: torch.Tensor
    recip_drF: torch.Tensor
    recip_drC: torch.Tensor


@dataclass(frozen=True)
class VVerticalMetrics:
    """S-point metrics for ``[..., N, ny+1, nx]`` V rows.

    ``recip_hfac_s`` and ``mask_s`` have the V-field shape.  ``rAs`` and
    ``recip_rAs`` have shape ``[..., ny+1, nx]``.  The source C-point ``rA``
    and ``mask_c`` have shapes ``[..., ny, nx]`` and ``[..., N, ny, nx]``;
    their south-neighbour relation must follow the topology-aware metric
    factory.  ``recip_drF`` has exact shape ``[N]`` and ``recip_drC`` has
    exact shape ``[N-1]``.  Metric leading dimensions align with trailing
    grid-leading dimensions of the V field; omitted batch dimensions are
    allowed.  All tensors have the target dtype and device.
    """

    recip_hfac_s: torch.Tensor
    mask_s: torch.Tensor
    mask_c: torch.Tensor
    rA: torch.Tensor
    rAs: torch.Tensor
    recip_rAs: torch.Tensor
    recip_drF: torch.Tensor
    recip_drC: torch.Tensor


def _first_true_index(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in torch.nonzero(mask, as_tuple=False)[0].tolist())


def _require_finite(name: str, value: torch.Tensor) -> None:
    """Non-finite guard, in the historical spelling. **Strict level only.**

    Under ``production``/``debug`` the body below is not skipped at run time --
    it is absent from the traced step.  ``strict_checks()`` reads a module-level
    bool fixed for the run, so Dynamo folds the branch away and neither the
    ``isfinite`` reduction nor the host branch behind it enters the graph.
    Those levels are covered by the
    step boundary's own probe instead (locomgcm/deferred_checks.py).
    """
    if not strict_checks():
        return
    bad = ~torch.isfinite(value)
    if torch.any(bad):
        raise ValueError(
            f"{name} contains a non-finite value at index {_first_true_index(bad)}"
        )


def _require_target_field(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < 3:
        raise ValueError(f"{name} must have shape [..., N, ny, nx]; got {tuple(value.shape)}")
    if value.shape[-3] < 1 or value.shape[-2] < 1 or value.shape[-1] < 1:
        raise ValueError(
            f"{name} dimensions N, ny, and nx must be positive; got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype; got {value.dtype}")
    _require_finite(name, value)


def _require_tensor(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}; got {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}; got {value.device}")
    _require_finite(name, value)


def _require_metric_tensor(
    name: str,
    value: torch.Tensor,
    *,
    suffix: tuple[int, ...],
    field_prefix: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    suffix_length = len(suffix)
    if value.ndim < suffix_length or tuple(value.shape[-suffix_length:]) != suffix:
        raise ValueError(f"{name} must end with shape {suffix}; got {tuple(value.shape)}")
    metric_prefix = tuple(value.shape[:-suffix_length])
    aligned_prefix = field_prefix[-len(metric_prefix) :] if metric_prefix else ()
    if len(metric_prefix) > len(field_prefix) or any(
        extent_mismatch(metric_size, field_size)
        for metric_size, field_size in zip(metric_prefix, aligned_prefix, strict=True)
    ):
        raise ValueError(
            f"{name} leading shape {metric_prefix} cannot align with field leading "
            f"shape {field_prefix}"
        )
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}; got {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}; got {value.device}")
    _require_finite(name, value)


def _vertical_view(metric: torch.Tensor, field_ndim: int) -> torch.Tensor:
    return metric.view((1,) * (field_ndim - 3) + (metric.shape[0], 1, 1))


def _horizontal_view(metric: torch.Tensor) -> torch.Tensor:
    return metric.unsqueeze(-3)


def _scalar_or_vertical_dt(
    value: float | torch.Tensor,
    *,
    field: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype != field.dtype:
            raise TypeError(f"{name} must have dtype {field.dtype}; got {value.dtype}")
        if value.device != field.device:
            raise ValueError(f"{name} must be on device {field.device}; got {value.device}")
        if value.ndim == 0:
            result = value
        elif tuple(value.shape) == (field.shape[-3],):
            result = _vertical_view(value, field.ndim)
        else:
            raise ValueError(
                f"{name} must be scalar or have shape ({field.shape[-3]},); "
                f"got {tuple(value.shape)}"
            )
        if strict_checks() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        return result
    if type(value) is not float:
        raise TypeError(f"{name} must be a float or torch.Tensor; got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value}")
    return _device_scalar(value, field)


def _coefficient(
    value: float | torch.Tensor,
    *,
    field: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype != field.dtype:
            raise TypeError(f"{name} must have dtype {field.dtype}; got {value.dtype}")
        if value.device != field.device:
            raise ValueError(f"{name} must be on device {field.device}; got {value.device}")
        if value.ndim != 0 and tuple(value.shape) != tuple(field.shape):
            raise ValueError(
                f"{name} tensor must be scalar or have interface shape {tuple(field.shape)}; "
                f"got {tuple(value.shape)}"
            )
        _require_finite(name, value)
        return value
    if type(value) is not float:
        raise TypeError(f"{name} must be a float or torch.Tensor; got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value}")
    return _device_scalar(value, field)


def _validate_tracer_metrics(field: torch.Tensor, metrics: TracerVerticalMetrics) -> None:
    if not isinstance(metrics, TracerVerticalMetrics):
        raise TypeError("metrics must be a TracerVerticalMetrics object")
    field_prefix = tuple(field.shape[:-3])
    row_suffix = tuple(field.shape[-3:])
    horizontal_suffix = tuple(field.shape[-2:])
    for name in ("recip_hfac_matrix_c", "recip_hfac_current_c", "mask_c"):
        _require_metric_tensor(
            f"metrics.{name}",
            getattr(metrics, name),
            suffix=row_suffix,
            field_prefix=field_prefix,
            dtype=field.dtype,
            device=field.device,
        )
    for name in ("rA", "recip_rA", "mask_in_c"):
        _require_metric_tensor(
            f"metrics.{name}",
            getattr(metrics, name),
            suffix=horizontal_suffix,
            field_prefix=field_prefix,
            dtype=field.dtype,
            device=field.device,
        )
    for name, shape in (
        ("recip_drF", (field.shape[-3],)),
        ("recip_drC", (field.shape[-3] - 1,)),
    ):
        _require_tensor(
            f"metrics.{name}",
            getattr(metrics, name),
            shape=shape,
            dtype=field.dtype,
            device=field.device,
        )


def _validate_physical_w(
    *,
    name: str,
    data: torch.Tensor,
    center_field: torch.Tensor,
    mask_c: torch.Tensor,
    field_prefix: tuple[int, ...] | None = None,
) -> None:
    """Shape/dtype/device and endpoint contract on a physical ``w`` wrapper.

    There is no step stamp to cross-check here any more.  The wrappers carry
    tensors only, and the one comparison the momentum stamp existed for
    -- "is this cached wrapper the one this step should consume, or a stale one
    a restart left behind" -- runs once per step against the stepper's host
    bookkeeping, in ``ImplicitFreeSurfaceStepper._validate_step_index_state``.
    """
    leading_shape = tuple(center_field.shape[:-3]) if field_prefix is None else field_prefix
    expected_shape = (
        leading_shape
        + (center_field.shape[-3] + 1,)
        + tuple(center_field.shape[-2:])
    )
    _require_tensor(
        f"{name}.data",
        data,
        shape=expected_shape,
        dtype=center_field.dtype,
        device=center_field.device,
    )

    # Endpoint-projection contract on the physical w wrapper. **Strict level
    # only**: both checks are whole-field comparisons followed by a host branch,
    # so under ``production``/``debug`` they are folded out of the traced step
    # rather than executed and ignored.
    if not strict_checks():
        return

    bottom_bad = data[..., -1, :, :] != 0
    if torch.any(bottom_bad):
        column = _first_true_index(bottom_bad)
        raise ValueError(
            f"{name}.data bottom interface q={center_field.shape[-3]} is non-zero "
            f"at column {column}"
        )

    if center_field.shape[-3] > 1:
        wet = mask_c != 0
        wet_dry = wet[..., :-1, :, :] != wet[..., 1:, :, :]
        bad = wet_dry & (data[..., 1:-1, :, :] != 0)
        if torch.any(bad):
            index = _first_true_index(bad)
            q_position = len(index) - 3
            interface = index[q_position] + 1
            column = index[:q_position] + index[q_position + 1 :]
            raise ValueError(
                f"{name}.data internal wet-dry interface q={interface} is non-zero "
                f"at column {column}"
            )


def tracer_vertical_transport_correction(
    *,
    tracer_n: torch.Tensor,
    w: TracerWNpHalfMIT,
    metrics: TracerVerticalMetrics,
) -> torch.Tensor:
    """Return the MIT continuity correction for one tracer at time ``n``.

    The physical surface value in ``w`` is retained in the wrapper but both
    correction-transport endpoints are projected to zero.  Internal transport
    carries the two adjacent C masks.  No direct vertical tracer flux is
    included here.
    """
    _require_target_field("tracer_n", tracer_n)
    _validate_tracer_metrics(tracer_n, metrics)
    if not isinstance(w, TracerWNpHalfMIT):
        raise TypeError("w must be a TracerWNpHalfMIT object")
    _validate_physical_w(
        name="w",
        data=w.data,
        center_field=tracer_n,
        mask_c=metrics.mask_c,
    )

    transport = torch.zeros_like(w.data)
    if tracer_n.shape[-3] > 1:
        transport[..., 1:-1, :, :] = (
            w.data[..., 1:-1, :, :]
            * _horizontal_view(metrics.rA)
            * metrics.mask_c[..., :-1, :, :]
            * metrics.mask_c[..., 1:, :, :]
        )

    divergence = transport[..., 1:, :, :] - transport[..., :-1, :, :]
    row_scale = (
        metrics.recip_hfac_current_c
        * _vertical_view(metrics.recip_drF, tracer_n.ndim)
        * _horizontal_view(metrics.recip_rA)
    )
    return (
        row_scale
        * tracer_n
        * divergence
        * _RK_SIGN
        * _horizontal_view(metrics.mask_in_c)
    )


def tracer_interface_center(
    *,
    rhs: torch.Tensor,
    w: TracerWNpHalfMIT,
    metrics: TracerVerticalMetrics,
) -> torch.Tensor:
    """MIT ``rCenter = 0.5*rTrans*recip_rA*rkSign`` on the nz-1 internal interfaces.

    ``rTrans = w*rA*maskC(k-1)`` carries the mask of the cell above each
    interface (gad_implicit_r.F:196-203); w is MIT upward-positive and rhs
    fixes its expected shape.  The result has shape ``[..., nz-1, ny, nx]``.
    """
    if not isinstance(w, TracerWNpHalfMIT):
        raise TypeError("w must be a TracerWNpHalfMIT object")
    _validate_physical_w(
        name="w",
        data=w.data,
        center_field=rhs,
        mask_c=metrics.mask_c,
    )
    w_data = w.data
    # implicit_vertical_high_order.high_order_interface_transport writes the
    # same rTrans with a where on the wet cell above; the two differ only in
    # the sign of zero on dry interfaces.
    transport = torch.zeros_like(w_data)
    if w_data.shape[-3] > 2:
        transport[..., 1:-1, :, :] = (
            w_data[..., 1:-1, :, :]
            * _horizontal_view(metrics.rA)
            * metrics.mask_c[..., :-1, :, :]
        )
    center = 0.5 * transport * _horizontal_view(metrics.recip_rA) * _RK_SIGN
    return center[..., 1:-1, :, :]


def assemble_tracer_implicit_vertical_system(
    *,
    rhs: torch.Tensor,
    center: torch.Tensor,
    upwind_fraction: float | torch.Tensor,
    dt_lev: float | torch.Tensor,
    diffusivity: float | torch.Tensor,
    metrics: TracerVerticalMetrics,
) -> TridiagonalSystem:
    """Assemble the combined tracer diffusion and three-band advection matrix.

    center is ``rCenter`` from :func:`tracer_interface_center`.  The scheme
    enters only through ``upwind_fraction``, MITgcm's
    ``rUpwind = ABS(rCenter)*upwindFac`` (``1 - rLimit`` in
    gad_dst2u1_impl_r.F:83-84): 1 for first-order upwind, 0 for a scheme
    without an upwind part.
    """
    _require_target_field("rhs", rhs)
    _validate_tracer_metrics(rhs, metrics)
    dt = _scalar_or_vertical_dt(dt_lev, field=rhs, name="dt_lev")
    kappa = _coefficient(diffusivity, field=rhs, name="diffusivity")

    recip_dr_f = _vertical_view(metrics.recip_drF, rhs.ndim)
    row_scale = metrics.recip_hfac_matrix_c * recip_dr_f
    lam = dt * row_scale

    mu_lower = torch.zeros_like(rhs)
    mu_upper = torch.zeros_like(rhs)
    if rhs.shape[-3] > 1:
        interface_kappa = kappa if kappa.ndim == 0 else kappa[..., 1:, :, :]
        interface_diffusion = interface_kappa * _vertical_view(metrics.recip_drC, rhs.ndim)
        mu_lower[..., 1:, :, :] = (
            lam[..., 1:, :, :]
            * metrics.mask_c[..., :-1, :, :]
            * interface_diffusion
        )
        mu_upper[..., :-1, :, :] = (
            lam[..., :-1, :, :]
            * metrics.mask_c[..., 1:, :, :]
            * interface_diffusion
        )

    # Preserve MITgcm's observable assembly order: materialize the negative
    # diffusion bands, form c5d = 1 - (b5d + d5d), then apply the interface
    # updates in increasing k order: lower and upper are updated once; main[k]
    # first takes interface k's subtraction, then interface k+1's addition.
    # MIT's interface loop runs k=Nr..1 (gad_implicit_r.F:186), so its main[k]
    # takes interface k+1's addition first; the two orders differ by rounding
    # without structure.  Whole-band slices replace the per-interface loop so
    # reverse mode does not keep one full band copy per interface.
    lower = -mu_lower
    upper = -mu_upper
    main = 1.0 - (lower + upper)
    # center/upwind live on the internal interfaces; interface q lies between
    # rows q and q+1.
    upwind = torch.abs(center) * upwind_fraction
    lam_below = lam[..., 1:, :, :]
    lam_above = lam[..., :-1, :, :]
    lower[..., 1:, :, :] = lower[..., 1:, :, :] - lam_below * (center + upwind)
    main[..., 1:, :, :] = main[..., 1:, :, :] - lam_below * (center - upwind)
    main[..., :-1, :, :] = main[..., :-1, :, :] + lam_above * (center + upwind)
    upper[..., :-1, :, :] = upper[..., :-1, :, :] + lam_above * (center - upwind)
    lower[..., 0, :, :] = 0.0
    upper[..., -1, :, :] = 0.0
    return TridiagonalSystem(lower=lower, main=main, upper=upper, rhs=rhs.clone())


def _validate_u_metrics(field: torch.Tensor, metrics: UVerticalMetrics) -> None:
    if not isinstance(metrics, UVerticalMetrics):
        raise TypeError("metrics must be a UVerticalMetrics object")
    prefix = tuple(field.shape[:-3])
    nz, ny, nx_plus_one = field.shape[-3:]
    if nx_plus_one < 2:
        raise ValueError(f"rhs_u must include nx+1 U faces; got shape {tuple(field.shape)}")
    for name, suffix in (
        ("recip_hfac_w", (nz, ny, nx_plus_one)),
        ("mask_w", (nz, ny, nx_plus_one)),
        ("mask_c", (nz, ny, nx_plus_one - 1)),
        ("rA", (ny, nx_plus_one - 1)),
        ("rAw", (ny, nx_plus_one)),
        ("recip_rAw", (ny, nx_plus_one)),
    ):
        _require_metric_tensor(
            f"metrics.{name}",
            getattr(metrics, name),
            suffix=suffix,
            field_prefix=prefix,
            dtype=field.dtype,
            device=field.device,
        )
    for name, shape in (("recip_drF", (nz,)), ("recip_drC", (nz - 1,))):
        _require_tensor(
            f"metrics.{name}",
            getattr(metrics, name),
            shape=shape,
            dtype=field.dtype,
            device=field.device,
        )


def _validate_v_metrics(field: torch.Tensor, metrics: VVerticalMetrics) -> None:
    if not isinstance(metrics, VVerticalMetrics):
        raise TypeError("metrics must be a VVerticalMetrics object")
    prefix = tuple(field.shape[:-3])
    nz, ny_plus_one, nx = field.shape[-3:]
    if ny_plus_one < 2:
        raise ValueError(f"rhs_v must include ny+1 V faces; got shape {tuple(field.shape)}")
    for name, suffix in (
        ("recip_hfac_s", (nz, ny_plus_one, nx)),
        ("mask_s", (nz, ny_plus_one, nx)),
        ("mask_c", (nz, ny_plus_one - 1, nx)),
        ("rA", (ny_plus_one - 1, nx)),
        ("rAs", (ny_plus_one, nx)),
        ("recip_rAs", (ny_plus_one, nx)),
    ):
        _require_metric_tensor(
            f"metrics.{name}",
            getattr(metrics, name),
            suffix=suffix,
            field_prefix=prefix,
            dtype=field.dtype,
            device=field.device,
        )
    for name, shape in (("recip_drF", (nz,)), ("recip_drC", (nz - 1,))):
        _require_tensor(
            f"metrics.{name}",
            getattr(metrics, name),
            shape=shape,
            dtype=field.dtype,
            device=field.device,
        )


def _float_time_step(value: float, *, field: torch.Tensor, name: str) -> torch.Tensor:
    if type(value) is not float:
        raise TypeError(f"{name} must be a float; got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value}")
    return _device_scalar(value, field)


def _momentum_center_and_upwind(
    *,
    transport: torch.Tensor,
    reciprocal_face_area: torch.Tensor,
    dt: torch.Tensor,
    upwind_shear: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = (
        0.5
        * dt
        * transport
        * _horizontal_view(reciprocal_face_area)
        * _RK_SIGN
    )
    upwind = torch.abs(center) if upwind_shear else torch.zeros_like(center)
    return center, upwind


def _assemble_momentum_bands(
    *,
    rhs: torch.Tensor,
    dt: torch.Tensor,
    viscosity: torch.Tensor,
    row_recip_hfac: torch.Tensor,
    row_mask: torch.Tensor,
    recip_dr_f: torch.Tensor,
    recip_dr_c: torch.Tensor,
    center: torch.Tensor,
    upwind: torch.Tensor,
) -> TridiagonalSystem:
    row_scale = row_recip_hfac * recip_dr_f
    mu_lower = torch.zeros_like(rhs)
    mu_upper = torch.zeros_like(rhs)
    if rhs.shape[-3] > 1:
        interface_kappa = viscosity if viscosity.ndim == 0 else viscosity[..., 1:, :, :]
        interface_viscosity = interface_kappa * recip_dr_c
        mu_lower[..., 1:, :, :] = (
            dt
            * row_scale[..., 1:, :, :]
            * interface_viscosity
            * row_mask[..., :-1, :, :]
        )
        mu_upper[..., :-1, :, :] = (
            dt
            * row_scale[..., :-1, :, :]
            * interface_viscosity
            * row_mask[..., 1:, :, :]
        )

    # mom_{u,v}_implicit_r.F use the same two-stage ordering as the tracer
    # path: diffusion first, parenthesized main diagonal second, then in-place
    # vertical-interface advection updates.
    lower = -mu_lower
    upper = -mu_upper
    main = 1.0 - (lower + upper)
    if rhs.shape[-3] > 1:
        center_q = center[..., 1:rhs.shape[-3], :, :]
        upwind_q = upwind[..., 1:rhs.shape[-3], :, :]
        lower[..., 1:, :, :] = lower[..., 1:, :, :] - (
            row_scale[..., 1:, :, :] * (center_q + upwind_q)
        )
        # Each interior row receives its lower-side addition before its
        # upper-side subtraction, exactly as in the ascending interface loop.
        main[..., 1:, :, :] = main[..., 1:, :, :] + (
            row_scale[..., 1:, :, :] * (center_q + upwind_q)
        )
        main[..., :-1, :, :] = main[..., :-1, :, :] - (
            row_scale[..., :-1, :, :] * (center_q - upwind_q)
        )
        upper[..., :-1, :, :] = upper[..., :-1, :, :] + (
            row_scale[..., :-1, :, :] * (center_q - upwind_q)
        )
    lower[..., 0, :, :] = 0.0
    upper[..., -1, :, :] = 0.0
    return TridiagonalSystem(lower=lower, main=main, upper=upper, rhs=rhs.clone())


def assemble_u_implicit_vertical_system(
    *,
    rhs_u: torch.Tensor,
    w: MomentumWNmHalfMIT,
    dt_mom: float,
    viscosity: float | torch.Tensor,
    metrics: UVerticalMetrics,
    upwind_shear: bool,
) -> TridiagonalSystem:
    """Assemble the combined W-point viscosity and vertical-advection matrix."""
    _require_target_field("rhs_u", rhs_u)
    _validate_u_metrics(rhs_u, metrics)
    if not isinstance(w, MomentumWNmHalfMIT):
        raise TypeError("w must be a MomentumWNmHalfMIT object")
    _validate_physical_w(
        name="w",
        data=w.data,
        center_field=metrics.mask_c,
        mask_c=metrics.mask_c,
        field_prefix=tuple(rhs_u.shape[:-3]),
    )
    if type(upwind_shear) is not bool:
        raise TypeError(f"upwind_shear must be a bool; got {type(upwind_shear).__name__}")
    dt = _float_time_step(dt_mom, field=rhs_u, name="dt_mom")
    kappa = _coefficient(viscosity, field=rhs_u, name="viscosity")

    transport = torch.zeros(
        tuple(rhs_u.shape[:-3])
        + (rhs_u.shape[-3] + 1, rhs_u.shape[-2], rhs_u.shape[-1]),
        dtype=rhs_u.dtype,
        device=rhs_u.device,
    )
    if rhs_u.shape[-3] > 1:
        weighted_w = (
            w.data[..., 1:-1, :, :]
            * _horizontal_view(metrics.rA)
            * metrics.mask_c[..., :-1, :, :]
        )
        zero_column = torch.zeros_like(weighted_w[..., :1])
        current = torch.cat((weighted_w, zero_column), dim=-1)
        west = torch.cat((zero_column, weighted_w), dim=-1)
        transport[..., 1:-1, :, :] = 0.5 * (current + west)

    center, upwind = _momentum_center_and_upwind(
        transport=transport,
        reciprocal_face_area=metrics.recip_rAw,
        dt=dt,
        upwind_shear=upwind_shear,
    )
    return _assemble_momentum_bands(
        rhs=rhs_u,
        dt=dt,
        viscosity=kappa,
        row_recip_hfac=metrics.recip_hfac_w,
        row_mask=metrics.mask_w,
        recip_dr_f=_vertical_view(metrics.recip_drF, rhs_u.ndim),
        recip_dr_c=_vertical_view(metrics.recip_drC, rhs_u.ndim),
        center=center,
        upwind=upwind,
    )


def assemble_v_implicit_vertical_system(
    *,
    rhs_v: torch.Tensor,
    w: MomentumWNmHalfMIT,
    dt_mom: float,
    viscosity: float | torch.Tensor,
    metrics: VVerticalMetrics,
    upwind_shear: bool,
) -> TridiagonalSystem:
    """Assemble the combined S-point viscosity and vertical-advection matrix."""
    _require_target_field("rhs_v", rhs_v)
    _validate_v_metrics(rhs_v, metrics)
    if not isinstance(w, MomentumWNmHalfMIT):
        raise TypeError("w must be a MomentumWNmHalfMIT object")
    _validate_physical_w(
        name="w",
        data=w.data,
        center_field=metrics.mask_c,
        mask_c=metrics.mask_c,
        field_prefix=tuple(rhs_v.shape[:-3]),
    )
    if type(upwind_shear) is not bool:
        raise TypeError(f"upwind_shear must be a bool; got {type(upwind_shear).__name__}")
    dt = _float_time_step(dt_mom, field=rhs_v, name="dt_mom")
    kappa = _coefficient(viscosity, field=rhs_v, name="viscosity")

    transport = torch.zeros(
        tuple(rhs_v.shape[:-3])
        + (rhs_v.shape[-3] + 1, rhs_v.shape[-2], rhs_v.shape[-1]),
        dtype=rhs_v.dtype,
        device=rhs_v.device,
    )
    if rhs_v.shape[-3] > 1:
        weighted_w = (
            w.data[..., 1:-1, :, :]
            * _horizontal_view(metrics.rA)
            * metrics.mask_c[..., :-1, :, :]
        )
        zero_row = torch.zeros_like(weighted_w[..., :1, :])
        current = torch.cat((weighted_w, zero_row), dim=-2)
        south = torch.cat((zero_row, weighted_w), dim=-2)
        transport[..., 1:-1, :, :] = 0.5 * (current + south)

    center, upwind = _momentum_center_and_upwind(
        transport=transport,
        reciprocal_face_area=metrics.recip_rAs,
        dt=dt,
        upwind_shear=upwind_shear,
    )
    return _assemble_momentum_bands(
        rhs=rhs_v,
        dt=dt,
        viscosity=kappa,
        row_recip_hfac=metrics.recip_hfac_s,
        row_mask=metrics.mask_s,
        recip_dr_f=_vertical_view(metrics.recip_drF, rhs_v.ndim),
        recip_dr_c=_vertical_view(metrics.recip_drC, rhs_v.ndim),
        center=center,
        upwind=upwind,
    )


def _report_solver_failure(message: str) -> None:
    """All numerical observers report to the owner; shape/API errors still raise."""
    from locomgcm.deferred_checks import report_numerical_failure
    report_numerical_failure(TridiagonalSolveError(message))


def _raise_solver_column(message: str, bad: torch.Tensor, level: int) -> None:
    column = _first_true_index(bad)
    _report_solver_failure(f"{message} at column {column}, level k={level}")


def _validate_system(system: TridiagonalSystem) -> None:
    if not isinstance(system, TridiagonalSystem):
        raise TridiagonalSolveError("system must be a TridiagonalSystem object")
    try:
        _require_target_field("system.rhs", system.rhs)
        shape = tuple(system.rhs.shape)
        for name in ("lower", "main", "upper"):
            _require_tensor(
                f"system.{name}",
                getattr(system, name),
                shape=shape,
                dtype=system.rhs.dtype,
                device=system.rhs.device,
            )
    except (TypeError, ValueError) as error:
        raise TridiagonalSolveError(str(error)) from error

    # Corner-row contract. **Strict level only** -- two whole-field comparisons
    # and two host branches, folded out of the traced step otherwise.  The
    # short-circuit structure is the parent's, verbatim: the upper mask is only
    # compared when the lower one passes (the comparison
    # order on the failure path is part of the historical behaviour).
    if not strict_checks():
        return
    lower_boundary = system.lower[..., 0, :, :] != 0
    if torch.any(lower_boundary):
        _raise_solver_column("system.lower first row must be exactly zero", lower_boundary, 0)
    upper_boundary = system.upper[..., -1, :, :] != 0
    if torch.any(upper_boundary):
        _raise_solver_column(
            "system.upper last row must be exactly zero",
            upper_boundary,
            system.rhs.shape[-3] - 1,
        )


def _solve_tridiagonal_mitgcm_order(system: TridiagonalSystem) -> torch.Tensor:
    """Solve the four bands without capturing their caller's mutable caches."""
    native = _native_mitgcm_band_solve(system)
    if native is not None:
        return native
    return _mitgcm_order_rows(system.lower, system.main, system.upper, system.rhs)


def _native_mitgcm_band_solve(system: TridiagonalSystem, *, transpose=False):
    """Select the IVADV four-band primitive only for a first-order CUDA solve.

    Transform requests and unsupported metadata keep the reference recurrence;
    no differentiation dependency is detached outside the analytic boundary.
    """
    from locomgcm.accel import requested

    values = (system.lower, system.main, system.upper, system.rhs)
    if not requested(("ivadv",)):
        return None
    if torch._C._are_functorch_transforms_active() or forward_mode_requested(*values):
        return None
    if grad_mode_requires_reference():
        return None
    rhs = system.rhs
    if rhs.device.type != "cuda" or rhs.dtype not in (torch.float32, torch.float64):
        return None
    if rhs.ndim not in (3, 4) or not 1 <= rhs.shape[-3] <= 90:
        return None
    if any(value.shape != rhs.shape or value.dtype != rhs.dtype or value.device != rhs.device
           or any(size == 0 for size in value.shape)
           for value in values):
        return None
    # Torch tensors have non-negative strides; the native ABI rechecks them.
    # A speculative backward seed has no static stride, and the kernel accepts
    # its actual strides without requiring a contiguous copy.
    from ._implicit_vertical_cuda import band_cuda

    prepared = tuple(value.detach().unsqueeze(0) if rhs.ndim == 3 else value.detach()
                     for value in values)
    # The opaque runtime publishes diagnostics outside the mathematical graph.
    result, _error = band_cuda(prepared, transpose=transpose)
    return result.squeeze(0) if rhs.ndim == 3 else result


def _mitgcm_order_rows(
    lower: torch.Tensor, main: torch.Tensor, upper: torch.Tensor, rhs: torch.Tensor,
) -> torch.Tensor:
    """Mirror MITgcm's reciprocal-then-multiply Thomas FP sequence.

    Each eliminated row is an immutable value. Repeated writes into a full
    column tensor force AOTAutograd to build slice/scatter chains, including
    for the transpose solve. Lists retain the identical level recurrence and
    rounding order without those full-field mutation intermediates.
    """
    nz = rhs.shape[-3]
    reciprocal = 1.0 / main[..., 0, :, :]
    c_prime = [upper[..., 0, :, :] * reciprocal]
    d_prime = [rhs[..., 0, :, :] * reciprocal]

    for k in range(1, nz):
        pivot = main[..., k, :, :] - lower[..., k, :, :] * c_prime[-1]
        reciprocal = 1.0 / pivot
        c_prime.append(upper[..., k, :, :] * reciprocal)
        d_prime.append((rhs[..., k, :, :] - lower[..., k, :, :] * d_prime[-1]) * reciprocal)

    result = [d_prime[-1]]
    for k in range(nz - 2, -1, -1):
        result.append(d_prime[k] - c_prime[k] * result[-1])
    return torch.stack(result[::-1], dim=-3)


# Reuse the compiled recurrence across tracer/momentum callers and checkpoint
# retraces instead of inlining 90 levels into each surrounding graph. This is
# only a compiler hint: eager/autograd semantics are unchanged. Older Torch
# builds without hierarchical compilation still compile the same function flat.
if hasattr(torch.compiler, "nested_compile_region"):
    _mitgcm_order_rows = torch.compiler.nested_compile_region(_mitgcm_order_rows)


def _shift_down(t: torch.Tensor) -> torch.Tensor:
    """``out[k] = t[k-1]`` along the vertical axis; slot 0 zeroed."""
    zero = torch.zeros_like(t[..., :1, :, :])
    return torch.cat([zero, t[..., :-1, :, :]], dim=-3)


def _shift_up(t: torch.Tensor) -> torch.Tensor:
    """``out[k] = t[k+1]`` along the vertical axis; last slot zeroed."""
    zero = torch.zeros_like(t[..., :1, :, :])
    return torch.cat([t[..., 1:, :, :], zero], dim=-3)


class _MitgcmOrderThomasAdjoint(torch.autograd.Function):
    r"""Analytic adjoint of ``x = A^{-1} rhs`` around the MITgcm-order sweep.

    Run the same recurrence with grad mode off and supply the reverse mode of
    a linear solve analytically, as in
    ``solvers/tridiag.py::_TridiagSolveAdjoint``. The functional row storage
    avoids compiler mutation chains without changing this derivative rule::

        rhs_bar   = A^{-T} x_bar        (A^T: sub = upper shifted down,
                                                super = lower shifted up)
        main_bar  = -rhs_bar * x
        lower_bar = -rhs_bar * x[k-1]
        upper_bar = -rhs_bar * x[k+1]

    The forward bytes are those of the no-gradient path; the tape holds
    ``{lower, main, upper, x}`` instead of ``~2*nz`` intermediates.  The
    transpose solve goes through this same boundary, so higher derivatives
    stay available.

    **Reverse mode only**, for the reason given on
    ``solvers/tridiag.py`` ``_TridiagSolveAdjoint``: a class-level ``jvp``
    makes torch 2.9's Dynamo break the graph at every solve, so forward mode is
    served by the plain sweep (:func:`_solve_tridiagonal_mitgcm_order`) that
    forward AD tapes itself, selected in
    :func:`_solve_mitgcm_order_differentiable`.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(lower, main, upper, rhs, reference=False):  # type: ignore[override]
        if reference:
            return _mitgcm_order_rows(lower, main, upper, rhs)
        return _solve_tridiagonal_mitgcm_order(
            TridiagonalSystem(lower=lower, main=main, upper=upper, rhs=rhs)
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        lower, main, upper, _, *options = inputs
        ctx.reference = bool(options and options[0])
        ctx.input_count = len(inputs)
        ctx.save_for_backward(lower, main, upper, output)

    @staticmethod
    def backward(ctx, grad_x):  # type: ignore[override]
        lower, main, upper, x = ctx.saved_tensors
        if torch.is_grad_enabled():
            # The saved solution participates in later reverse traversals too.
            # Once higher order is requested, this original node also stays on
            # the reference recurrence when those traversals reach it again.
            ctx.reference = True
        rhs_bar = None
        if not ctx.reference and not torch.is_grad_enabled():
            rhs_bar = _native_mitgcm_band_solve(
                TridiagonalSystem(lower, main, upper, grad_x), transpose=True)
        if rhs_bar is None:
            # High-order reverse stays on the recursive reference boundary.
            rhs_bar = _MitgcmOrderThomasAdjoint.apply(
                _shift_down(upper), main, _shift_up(lower), grad_x.contiguous(), True)
        grad_lower = grad_main = grad_upper = grad_rhs = None
        if ctx.needs_input_grad[0]:
            grad_lower = -rhs_bar * _shift_down(x)
        if ctx.needs_input_grad[1]:
            grad_main = -rhs_bar * x
        if ctx.needs_input_grad[2]:
            grad_upper = -rhs_bar * _shift_up(x)
        if ctx.needs_input_grad[3]:
            grad_rhs = rhs_bar
        return (grad_lower, grad_main, grad_upper, grad_rhs, *([None] * (ctx.input_count - 4)))


def _solve_mitgcm_order_differentiable(system: TridiagonalSystem) -> torch.Tensor:
    """MITgcm-order solve; the analytic boundary whenever grad mode is enabled.

    Forward mode takes the plain sweep, which tapes itself; the analytic
    adjoint is the reverse-mode path and the ``Function`` therefore declares no
    ``jvp`` (see :class:`_MitgcmOrderThomasAdjoint`).  All three branches run
    the same recurrence, so the primal is bit-identical in each.
    """
    operands = (system.lower, system.main, system.upper, system.rhs)
    if forward_mode_requested(*operands):
        return _solve_tridiagonal_mitgcm_order(system)
    if grad_mode_requires_reference():
        return _MitgcmOrderThomasAdjoint.apply(
            system.lower, system.main, system.upper, system.rhs
        )
    return _solve_tridiagonal_mitgcm_order(system)


def band_solve_pullback(system, solution, grad_solution, needs=(True, True, True, True)):
    """First-order band/RHS cotangents of an attached solve."""
    rhs_bar = _native_mitgcm_band_solve(
        TridiagonalSystem(system.lower, system.main, system.upper, grad_solution), transpose=True)
    if rhs_bar is None:
        rhs_bar = _mitgcm_order_rows(
            _shift_down(system.upper), system.main, _shift_up(system.lower),
            grad_solution.contiguous())
    return (
        -rhs_bar * _shift_down(solution) if needs[0] else None,
        -rhs_bar * solution if needs[1] else None,
        -rhs_bar * _shift_up(solution) if needs[2] else None,
        rhs_bar if needs[3] else None,
    )


class _AttachedBandSolveAdjoint(torch.autograd.Function):
    r"""Tape the analytic adjoint of ``x = A^{-1} rhs`` onto a solution made elsewhere.

    The fused band kernels (``implicit_vertical_fused``) assemble and solve in
    one launch and expose no derivative, and the reverse mode of a linear solve
    does not need one from them: ``x`` arrives detached from the kernel, the
    bands arrive from the ordinary assembly (``assemble_*_implicit_vertical_system``,
    which tapes itself back to ``w``, the coefficient and the right-hand side),
    and this Function supplies the same four formulas as
    :class:`_MitgcmOrderThomasAdjoint`::

        rhs_bar   = A^{-T} x_bar        (A^T: sub = upper shifted down,
                                                super = lower shifted up)
        main_bar  = -rhs_bar * x
        lower_bar = -rhs_bar * x[k-1]
        upper_bar = -rhs_bar * x[k+1]

    The forward is the identity on ``x`` (a view), so the primal bytes are the
    kernel's.  ``x`` must be the solution of exactly the system whose bands are
    passed; the caller (``dynamics/implicit_vertical_advection.py``) guarantees
    it by handing the kernel and the assembly the same physical inputs.

    **Reverse mode only.**  The transpose solve uses the IVADV four-band kernel
    when admitted, otherwise the reference sweep. Nothing higher than first
    order is supported through this path (backward rejects
    create_graph explicitly), and forward mode never reaches here:
    :func:`reverse_mode_requested` is false under a tangent and the caller then
    takes the taped assembly and sweep.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(lower, main, upper, rhs, x):  # type: ignore[override]
        return x.view_as(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        lower, main, upper, _, x = inputs
        ctx.save_for_backward(lower, main, upper, x)

    @staticmethod
    def backward(ctx, grad_x):  # type: ignore[override]
        if torch.is_grad_enabled():
            raise RuntimeError(
                "attached band solve supports first-order reverse differentiation only; "
                "use the reference solve for higher-order derivatives"
            )
        lower, main, upper, x = ctx.saved_tensors
        bars = band_solve_pullback(
            TridiagonalSystem(lower=lower, main=main, upper=upper, rhs=grad_x),
            x, grad_x, ctx.needs_input_grad[:4],
        )
        return (*(bar if needed else None for bar, needed in zip(bars, ctx.needs_input_grad)), None)


def reverse_mode_requested(*values: object) -> bool:
    """True when a gradient is being taped through any tensor operand.

    False under a forward-mode tangent (:func:`forward_mode_requested`), so the
    two predicates partition the differentiable cases the way
    :func:`_solve_mitgcm_order_differentiable` does; non-tensor operands (a
    scalar coefficient, a float ``dt``) never request anything.
    """
    tensors = tuple(value for value in values if isinstance(value, torch.Tensor))
    if not tensors or not torch.is_grad_enabled():
        return False
    if not any(tensor.requires_grad for tensor in tensors):
        return False
    return not forward_mode_requested(*tensors)


def attach_band_solve_adjoint(system: TridiagonalSystem, solution: torch.Tensor) -> torch.Tensor:
    """Return ``solution`` carrying the analytic adjoint of ``system``.

    ``solution`` is a detached solve of ``system`` (a fused kernel's output);
    the returned tensor has the same bytes and a ``grad_fn`` that reaches the
    system's bands and right-hand side (see :class:`_AttachedBandSolveAdjoint`).
    """
    return _AttachedBandSolveAdjoint.apply(
        system.lower, system.main, system.upper, system.rhs, solution
    )


def _preflight_thomas(system: TridiagonalSystem, nz: int) -> None:
    """The no-pivot Thomas preflight, in the historical spelling.

    **Strict level only, and pure observer**: it recomputes the elimination in
    order to reject an unsolvable column before the solve runs, and the solve
    (:func:`_solve_tridiagonal_mitgcm_order`) recomputes every quantity from
    the system itself, so removing this changes no result bit.  Eleven of the
    whole-field reductions it issues per vertical level are inside the loop
    below;  booked this module's share of them under ``deferred_checks.py``
    (29.6 ms and 295 launches per step for the module as a whole).
    """
    epsilon = torch.finfo(system.rhs.dtype).eps

    pivot = system.main[..., 0, :, :]
    scale = torch.maximum(torch.ones_like(pivot), torch.abs(system.main[..., 0, :, :]))
    tau = (8.0 * epsilon) * scale
    nonfinite_pivot = ~torch.isfinite(pivot)
    if torch.any(nonfinite_pivot):
        _raise_solver_column("Thomas pivot is non-finite", nonfinite_pivot, 0)
    unresolved = torch.abs(pivot) <= tau
    if torch.any(unresolved):
        _raise_solver_column(
            "Thomas pivot is not numerically resolvable under |p_k| <= 8*eps*s_k",
            unresolved,
            0,
        )

    c_prime = system.upper[..., 0, :, :] / pivot
    nonfinite_quotient = ~torch.isfinite(c_prime)
    if torch.any(nonfinite_quotient):
        _raise_solver_column(
            "Thomas elimination quotient c_prime is non-finite", nonfinite_quotient, 0
        )
    d_prime = system.rhs[..., 0, :, :] / pivot
    nonfinite_rhs_quotient = ~torch.isfinite(d_prime)
    if torch.any(nonfinite_rhs_quotient):
        _raise_solver_column(
            "Thomas elimination quotient d_prime is non-finite",
            nonfinite_rhs_quotient,
            0,
        )

    for k in range(1, nz):
        elimination_product = system.lower[..., k, :, :] * c_prime
        pivot = system.main[..., k, :, :] - elimination_product
        scale = torch.maximum(
            torch.ones_like(pivot),
            torch.abs(system.main[..., k, :, :]) + torch.abs(elimination_product),
        )
        tau = (8.0 * epsilon) * scale
        nonfinite_pivot = ~torch.isfinite(pivot)
        if torch.any(nonfinite_pivot):
            _raise_solver_column("Thomas pivot is non-finite", nonfinite_pivot, k)
        unresolved = torch.abs(pivot) <= tau
        if torch.any(unresolved):
            _raise_solver_column(
                "Thomas pivot is not numerically resolvable under |p_k| <= 8*eps*s_k",
                unresolved,
                k,
            )

        multiplier = system.lower[..., k, :, :] / pivot
        nonfinite_multiplier = ~torch.isfinite(multiplier)
        if torch.any(nonfinite_multiplier):
            _raise_solver_column(
                "Thomas elimination quotient a_k/p_k is non-finite",
                nonfinite_multiplier,
                k,
            )
        c_prime = system.upper[..., k, :, :] / pivot
        nonfinite_quotient = ~torch.isfinite(c_prime)
        if torch.any(nonfinite_quotient):
            _raise_solver_column(
                "Thomas elimination quotient c_prime is non-finite",
                nonfinite_quotient,
                k,
            )
        d_prime = (
            system.rhs[..., k, :, :] - system.lower[..., k, :, :] * d_prime
        ) / pivot
        nonfinite_rhs_quotient = ~torch.isfinite(d_prime)
        if torch.any(nonfinite_rhs_quotient):
            _raise_solver_column(
                "Thomas elimination quotient d_prime is non-finite",
                nonfinite_rhs_quotient,
                k,
            )


def solve_tridiagonal_system(system: TridiagonalSystem) -> torch.Tensor:
    """Solve in MITgcm FP order, after the no-pivot Thomas preflight if asked.

    A rejected pivot is described only as a Thomas breakdown; this function
    does not run a dense or pivoted solver and therefore does not infer that the
    mathematical matrix is singular.  The preflight and the result check are
    the ``strict`` level's; the other two levels reach the same verdict at the
    step boundary (locomgcm/deferred_checks.py).
    """
    _validate_system(system)
    nz = system.rhs.shape[-3]
    if strict_checks():
        _preflight_thomas(system, nz)

    if nz == 1:
        # Preserve the identity fast path's exact clone while selecting a
        # non-identity scalar solve without a data-dependent host branch.
        reciprocal = 1.0 / system.main
        solved = system.rhs * reciprocal
        result = torch.where(system.main == 1.0, system.rhs.clone(), solved)
    else:
        try:
            result = _solve_mitgcm_order_differentiable(system)
        except RuntimeError as error:
            raise TridiagonalSolveError(f"MITgcm-order Thomas solve failed: {error}") from error
    if strict_checks():
        bad_result = ~torch.isfinite(result)
        if torch.any(bad_result):
            index = _first_true_index(bad_result)
            level_position = len(index) - 3
            level = index[level_position]
            column = index[:level_position] + index[level_position + 1 :]
            _report_solver_failure(
                f"MITgcm-order Thomas solve produced a non-finite result at column {column}, "
                f"level k={level}"
            )
    return result
