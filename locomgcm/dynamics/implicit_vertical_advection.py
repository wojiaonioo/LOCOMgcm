"""Execution adapters for implicit vertical advection.

The physical vertical velocity crossing this module boundary is stored in the
MITgcm convention: surface-first, upward-positive, and in metres per second.
``diagnose_vertical_velocity`` produces LOCOMgcm downward-positive velocity,
so regional producers negate it exactly once here.  Cubed-sphere producers use
the shared continuity operator directly in upward-positive convention.

The solver assemblers apply their own ``rkSign=-1`` and endpoint transport
projection.  This module does not apply ``rkSign`` and does not zero the
physical surface velocity.  Momentum U/V neighbour columns are supplied through
the repository's topology-aware scalar exchange before calling the fixed solver
API; closed physical boundaries receive zero ghosts, while periodic, distributed,
and cube seams receive their actual neighbouring C columns.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, fields, replace

import torch

from locomgcm.dynamics.base import TendencyModule, TendencyPhase, TendencyStepDomain
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1
from locomgcm.geometry.grid import _compute_interior_mask
from locomgcm.geometry.halo_exchange import HaloExchange
from locomgcm.geometry.metric_bundle import build_vertshear_metric_bundle
from locomgcm.operators import (
    face_volume_transport,
    horizontal_volume_divergence_from_transports,
    vertical_velocity_continuity,
)
from locomgcm.operators.fluxes.tracer_flux import TracerFluxScheme, tracer_scheme_spec
from locomgcm.solvers import (
    MomentumWNmHalfMIT,
    TracerVerticalMetrics,
    TracerWNpHalfMIT,
    TridiagonalSystem,
    UVerticalMetrics,
    VVerticalMetrics,
    assemble_tracer_implicit_vertical_system,
    assemble_u_implicit_vertical_system,
    assemble_v_implicit_vertical_system,
    configured_fused_vertical_backend,
    solve_tracer_band_thomas_fused,
    solve_tridiagonal_system,
    solve_u_band_thomas_fused,
    solve_v_band_thomas_fused,
    tracer_vertical_transport_correction,
)
from locomgcm.solvers.implicit_vertical import (
    attach_band_solve_adjoint,
    reverse_mode_requested,
    tracer_interface_center,
)
from locomgcm.solvers.implicit_vertical_schemes import (
    dst3_interface_weights,
    dst3fl_interface_weights,
    upwind3_interface_weights,
)
from locomgcm.solvers.tridiag import forward_mode_requested
from locomgcm.state import PrognosticState, StateTendency


#: Vertical tracer schemes the implicit column solve takes (MITgcm
#: ``tempVertAdvScheme`` with ``tempImplVertAdv``, dispatched in
#: ``GAD_IMPLICIT_R``, gad_implicit_r.F:214-251).  Each has its own branch in
#: :func:`solve_tracer_implicit_vertical_advection`; names and MITgcm codes are
#: those of ``TRACER_ADVECTION_SCHEMES``.
IMPLICIT_VERTICAL_TRACER_SCHEMES = (
    TracerFluxScheme.UPWIND,
    TracerFluxScheme.CENTERED,
    TracerFluxScheme.UPWIND3,
    TracerFluxScheme.DST2,
    TracerFluxScheme.DST3,
    TracerFluxScheme.DST3_FL,
)


_U_CENTER_EXTENT = HaloExtentV1(
    west_left_extent=1,
    east_right_extent=1,
    south_bottom_extent=0,
    north_top_extent=0,
)
_V_CENTER_EXTENT = HaloExtentV1(
    west_left_extent=0,
    east_right_extent=0,
    south_bottom_extent=1,
    north_top_extent=1,
)


def _positive_reciprocal(value: torch.Tensor) -> torch.Tensor:
    positive = value > 0
    # Dry geometry has zero reciprocal and zero derivative. Masking only the
    # quotient evaluates 1/0 first, so its backward still produces 0*inf.
    denominator = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, 1.0 / denominator, torch.zeros_like(value))


def tracer_hfac_factors(
    *, field: torch.Tensor, hfac_c: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the C-cell wet mask and reciprocal for the supplied hFac layer.

    hFac is dimensionless. None supplies full cells in the field's dtype
    and trailing [nz, ny, nx] shape; callers choose the geometry time layer.
    """
    dtype = field.dtype
    device = field.device
    if hfac_c is None:
        metric_shape = tuple(field.shape[-3:])
        hfac_c = torch.ones(metric_shape, dtype=dtype, device=device)
    else:
        hfac_c = hfac_c.to(dtype=dtype, device=device)
    mask_c = (hfac_c > 0).to(dtype=dtype)
    recip_hfac_c = _positive_reciprocal(hfac_c)
    return mask_c, recip_hfac_c


def _local_cube_grid(grid: Any, cube_ctx: Any) -> Any:
    if getattr(grid, "face_count", 1) != 6 or cube_ctx is None:
        return grid
    from locomgcm.geometry.cube_localize import localize_cube_grid

    return localize_cube_grid(grid, cube_ctx)


def diagnose_mit_vertical_velocity(
    state: PrognosticState,
    grid: Any,
    *,
    cube_ctx: Any = None,
) -> torch.Tensor:
    """Diagnose complete interface ``w`` in MIT upward-positive convention.

    The returned shape is ``[..., N+1, ny, nx]``.  The physical surface value
    is retained; continuity supplies an exact zero bottom interface and zero
    transport at internal wet-dry interfaces.
    """
    if getattr(grid, "face_count", 1) != 6:
        from locomgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

        _, w_to_downward, _ = diagnose_vertical_velocity(state, grid)
        return -w_to_downward

    local_grid = _local_cube_grid(grid, cube_ctx)
    dtype = state.u.data.dtype
    device = state.u.data.device
    nz = int(local_grid.nz)
    drf = local_grid.layer_thickness_for(dtype, device).view(1, nz, 1, 1)
    dyg = local_grid.metrics.dyG_for(dtype, device).unsqueeze(1)
    dxg = local_grid.metrics.dxG_for(dtype, device).unsqueeze(1)
    hfac_w = local_grid.masks.hFacW.to(dtype=dtype, device=device)
    hfac_s = local_grid.masks.hFacS.to(dtype=dtype, device=device)
    hfac_c = local_grid.masks.hFacC.to(dtype=dtype, device=device)
    u_transport, v_transport = face_volume_transport(
        state.u.data,
        state.v.data,
        drf,
        dyg,
        dxg,
        hFacW=hfac_w,
        hFacS=hfac_s,
    )
    wet = (hfac_c > 0).to(dtype=dtype)
    horizontal_divergence = horizontal_volume_divergence_from_transports(
        u_transport,
        v_transport,
        wet_mask=wet,
    )
    cell_area = local_grid.metrics.cell_area_for(dtype, device)
    if cell_area.ndim == horizontal_divergence.ndim - 1:
        cell_area = cell_area.unsqueeze(-3)
    return vertical_velocity_continuity(
        horizontal_divergence,
        cell_area=cell_area,
        wet_mask=wet,
        form="velocity",
        convention="upward",
    )


def make_momentum_w_nm_half_mit(
    state: PrognosticState,
    grid: Any,
    cube_ctx: Any = None,
) -> MomentumWNmHalfMIT:
    """Diagnose momentum-time MIT-upward ``w`` from bounded U/V.

    ``initialize_step_state`` stores its ``data`` as the fresh run's lagged w
    (``_w_interface_prev``), the one copy the tendencies and the momentum solve
    both read.
    """
    return MomentumWNmHalfMIT(
        data=diagnose_mit_vertical_velocity(state, grid, cube_ctx=cube_ctx),
    )


def make_tracer_w_np_half_mit(
    state: PrognosticState,
    grid: Any,
    cube_ctx: Any = None,
) -> TracerWNpHalfMIT:
    """Build step-local tracer-time MIT-upward ``w`` from corrected U/V."""
    return TracerWNpHalfMIT(
        data=diagnose_mit_vertical_velocity(state, grid, cube_ctx=cube_ctx),
    )


def _recip_dr_c(grid: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    center_depth = grid.center_depth.to(dtype=dtype, device=device)
    if center_depth.numel() <= 1:
        return center_depth.new_empty((0,))
    return 1.0 / torch.diff(center_depth)


def build_tracer_vertical_metrics(
    field: torch.Tensor,
    grid: Any,
    *,
    mask_c_current: torch.Tensor,
    recip_hfac_matrix_c: torch.Tensor,
    recip_hfac_current_c: torch.Tensor,
) -> TracerVerticalMetrics:
    """Build C-column metrics with explicitly selected hFac time layers."""
    dtype = field.dtype
    device = field.device
    r_a = grid.metrics.cell_area_for(dtype, device)
    recip_r_a = _positive_reciprocal(r_a)
    if grid.boundary_topology.open_x or grid.boundary_topology.open_y:
        mask_in_c, _, _ = _compute_interior_mask(grid, dtype, device)
    else:
        mask_in_c = torch.ones_like(r_a)
    drf = grid.layer_thickness_for(dtype, device)
    return TracerVerticalMetrics(
        recip_hfac_matrix_c=recip_hfac_matrix_c,
        recip_hfac_current_c=recip_hfac_current_c,
        mask_c=mask_c_current,
        rA=r_a,
        recip_rA=recip_r_a,
        recip_drF=1.0 / drf,
        recip_drC=_recip_dr_c(grid, dtype=dtype, device=device),
        mask_in_c=mask_in_c,
    )


def _face_area_from_reciprocal(reciprocal: torch.Tensor) -> torch.Tensor:
    return _positive_reciprocal(reciprocal)


def build_u_vertical_metrics(field: torch.Tensor, grid: Any) -> UVerticalMetrics:
    """Build W-column metrics, including topology-aware face-area reciprocals."""
    bundle = build_vertshear_metric_bundle(grid, field.dtype, field.device)
    recip_hfac = bundle.recip_hFacW
    if recip_hfac is None:
        recip_hfac = torch.ones(tuple(field.shape[-3:]), dtype=field.dtype, device=field.device)
    mask_w = (recip_hfac > 0).to(dtype=field.dtype)
    mask_c = bundle.maskC
    if mask_c is None:
        mask_c = torch.ones(
            (field.shape[-3], field.shape[-2], field.shape[-1] - 1),
            dtype=field.dtype,
            device=field.device,
        )
    return UVerticalMetrics(
        recip_hfac_w=recip_hfac,
        mask_w=mask_w,
        mask_c=mask_c,
        rA=bundle.rA,
        rAw=_face_area_from_reciprocal(bundle.recip_rAw),
        recip_rAw=bundle.recip_rAw,
        recip_drF=bundle.recip_drF,
        recip_drC=_recip_dr_c(grid, dtype=field.dtype, device=field.device),
    )


def build_v_vertical_metrics(field: torch.Tensor, grid: Any) -> VVerticalMetrics:
    """Build S-column metrics, including topology-aware face-area reciprocals."""
    bundle = build_vertshear_metric_bundle(grid, field.dtype, field.device)
    recip_hfac = bundle.recip_hFacS
    if recip_hfac is None:
        recip_hfac = torch.ones(tuple(field.shape[-3:]), dtype=field.dtype, device=field.device)
    mask_s = (recip_hfac > 0).to(dtype=field.dtype)
    mask_c = bundle.maskC
    if mask_c is None:
        mask_c = torch.ones(
            (field.shape[-3], field.shape[-2] - 1, field.shape[-1]),
            dtype=field.dtype,
            device=field.device,
        )
    return VVerticalMetrics(
        recip_hfac_s=recip_hfac,
        mask_s=mask_s,
        mask_c=mask_c,
        rA=bundle.rA,
        rAs=_face_area_from_reciprocal(bundle.recip_rAs),
        recip_rAs=bundle.recip_rAs,
        recip_drF=bundle.recip_drF,
        recip_drC=_recip_dr_c(grid, dtype=field.dtype, device=field.device),
    )


def _pad_face_field(field: torch.Tensor, *, axis: int) -> torch.Tensor:
    shape = list(field.shape)
    shape[axis] += 2
    result = field.new_zeros(shape)
    index = [slice(None)] * field.ndim
    index[axis] = slice(1, -1)
    result[tuple(index)] = field
    return result


def _slice_system(system: TridiagonalSystem, *, axis: int) -> TridiagonalSystem:
    index = [slice(None)] * system.rhs.ndim
    index[axis] = slice(1, -1)
    key = tuple(index)
    return TridiagonalSystem(
        lower=system.lower[key],
        main=system.main[key],
        upper=system.upper[key],
        rhs=system.rhs[key],
    )


def _pad_coefficient(value: float | torch.Tensor, *, axis: int) -> float | torch.Tensor:
    if isinstance(value, torch.Tensor) and value.ndim != 0:
        return _pad_face_field(value, axis=axis)
    return value


def _detached(value: float | torch.Tensor) -> float | torch.Tensor:
    return value.detach() if isinstance(value, torch.Tensor) else value


def _tensor_operands(*values: object) -> tuple[torch.Tensor, ...]:
    return tuple(value for value in values if isinstance(value, torch.Tensor))


def _metric_operands(metrics) -> tuple[torch.Tensor, ...]:
    return tuple(getattr(metrics, item.name) for item in fields(metrics))


def _detached_metrics(metrics):
    """Detach only the native primal copy; live metrics still build the bands."""
    return replace(metrics, **{item.name: getattr(metrics, item.name).detach()
                               for item in fields(metrics)})


def _fused_backend_for(*operands: object) -> str | None:
    """The fused backend unless a forward-mode tangent is present.

    The fused kernels have no forward-mode rule; with a tangent on any operand
    the taped assembly and sweep are the differentiable path, exactly as in
    ``solvers.implicit_vertical._solve_mitgcm_order_differentiable``.
    """
    backend = configured_fused_vertical_backend()
    if backend is not None and forward_mode_requested(*_tensor_operands(*operands)):
        return None
    return backend



def solve_u_implicit_vertical_advection(
    *,
    rhs_u: torch.Tensor,
    w: MomentumWNmHalfMIT,
    dt_mom: float,
    viscosity: float | torch.Tensor,
    grid: Any,
    upwind_shear: bool,
) -> torch.Tensor:
    """Exchange C neighbours, assemble one U combined system, and solve once."""
    metrics = build_u_vertical_metrics(rhs_u, grid)
    exchange = HaloExchange(grid)
    w_extended = MomentumWNmHalfMIT(
        data=exchange.center(
            w.data,
            location="U",
            extent=_U_CENTER_EXTENT,
            fill=FillSpec.ZERO,
        ),
    )
    rhs_extended = _pad_face_field(rhs_u, axis=-1)
    extended_metrics = UVerticalMetrics(
        recip_hfac_w=_pad_face_field(metrics.recip_hfac_w, axis=-1),
        mask_w=_pad_face_field(metrics.mask_w, axis=-1),
        mask_c=exchange.center(
            metrics.mask_c,
            location="U",
            extent=_U_CENTER_EXTENT,
            fill=FillSpec.ZERO,
            constant=True,
        ),
        rA=exchange.center(
            metrics.rA,
            location="U",
            extent=_U_CENTER_EXTENT,
            fill=FillSpec.ZERO,
            constant=True,
        ),
        rAw=_pad_face_field(metrics.rAw, axis=-1),
        recip_rAw=_pad_face_field(metrics.recip_rAw, axis=-1),
        recip_drF=metrics.recip_drF,
        recip_drC=metrics.recip_drC,
    )
    viscosity_padded = _pad_coefficient(viscosity, axis=-1)
    operands = (rhs_extended, w_extended.data, viscosity_padded, dt_mom,
                *_metric_operands(extended_metrics))
    backend = _fused_backend_for(*operands)
    if backend is not None:
        # Fused band+Thomas: the accelerated kernels assemble the bands inside
        # the kernel, so they take the prepared physical inputs directly and
        # must be reached before the eager assembly below.
        if reverse_mode_requested(*operands):
            # The kernel exposes no derivative: solve it detached, then tape the
            # analytic adjoint onto the solution through the taped assembly of
            # the same system (solvers.implicit_vertical.attach_band_solve_adjoint).
            solution = solve_u_band_thomas_fused(
                rhs_u=rhs_extended.detach(),
                w=MomentumWNmHalfMIT(w_extended.data.detach()),
                dt_mom=_detached(dt_mom),
                viscosity=_detached(viscosity_padded),
                metrics=_detached_metrics(extended_metrics),
                upwind_shear=upwind_shear,
                backend=backend,
            )
            extended_system = assemble_u_implicit_vertical_system(
                rhs_u=rhs_extended,
                w=w_extended,
                dt_mom=dt_mom,
                viscosity=viscosity_padded,
                metrics=extended_metrics,
                upwind_shear=upwind_shear,
            )
            return attach_band_solve_adjoint(
                _slice_system(extended_system, axis=-1), solution
            )
        return solve_u_band_thomas_fused(
            rhs_u=rhs_extended,
            w=w_extended,
            dt_mom=dt_mom,
            viscosity=viscosity_padded,
            metrics=extended_metrics,
            upwind_shear=upwind_shear,
            backend=backend,
        )
    extended_system = assemble_u_implicit_vertical_system(
        rhs_u=rhs_extended,
        w=w_extended,
        dt_mom=dt_mom,
        viscosity=viscosity_padded,
        metrics=extended_metrics,
        upwind_shear=upwind_shear,
    )
    return solve_tridiagonal_system(_slice_system(extended_system, axis=-1))


def solve_v_implicit_vertical_advection(
    *,
    rhs_v: torch.Tensor,
    w: MomentumWNmHalfMIT,
    dt_mom: float,
    viscosity: float | torch.Tensor,
    grid: Any,
    upwind_shear: bool,
) -> torch.Tensor:
    """Exchange C neighbours, assemble one V combined system, and solve once."""
    metrics = build_v_vertical_metrics(rhs_v, grid)
    exchange = HaloExchange(grid)
    w_extended = MomentumWNmHalfMIT(
        data=exchange.center(
            w.data,
            location="V",
            extent=_V_CENTER_EXTENT,
            fill=FillSpec.ZERO,
        ),
    )
    rhs_extended = _pad_face_field(rhs_v, axis=-2)
    extended_metrics = VVerticalMetrics(
        recip_hfac_s=_pad_face_field(metrics.recip_hfac_s, axis=-2),
        mask_s=_pad_face_field(metrics.mask_s, axis=-2),
        mask_c=exchange.center(
            metrics.mask_c,
            location="V",
            extent=_V_CENTER_EXTENT,
            fill=FillSpec.ZERO,
            constant=True,
        ),
        rA=exchange.center(
            metrics.rA,
            location="V",
            extent=_V_CENTER_EXTENT,
            fill=FillSpec.ZERO,
            constant=True,
        ),
        rAs=_pad_face_field(metrics.rAs, axis=-2),
        recip_rAs=_pad_face_field(metrics.recip_rAs, axis=-2),
        recip_drF=metrics.recip_drF,
        recip_drC=metrics.recip_drC,
    )
    viscosity_padded = _pad_coefficient(viscosity, axis=-2)
    operands = (rhs_extended, w_extended.data, viscosity_padded, dt_mom,
                *_metric_operands(extended_metrics))
    backend = _fused_backend_for(*operands)
    if backend is not None:
        if reverse_mode_requested(*operands):
            solution = solve_v_band_thomas_fused(
                rhs_v=rhs_extended.detach(),
                w=MomentumWNmHalfMIT(w_extended.data.detach()),
                dt_mom=_detached(dt_mom),
                viscosity=_detached(viscosity_padded),
                metrics=_detached_metrics(extended_metrics),
                upwind_shear=upwind_shear,
                backend=backend,
            )
            extended_system = assemble_v_implicit_vertical_system(
                rhs_v=rhs_extended,
                w=w_extended,
                dt_mom=dt_mom,
                viscosity=viscosity_padded,
                metrics=extended_metrics,
                upwind_shear=upwind_shear,
            )
            return attach_band_solve_adjoint(
                _slice_system(extended_system, axis=-2), solution
            )
        return solve_v_band_thomas_fused(
            rhs_v=rhs_extended,
            w=w_extended,
            dt_mom=dt_mom,
            viscosity=viscosity_padded,
            metrics=extended_metrics,
            upwind_shear=upwind_shear,
            backend=backend,
        )
    extended_system = assemble_v_implicit_vertical_system(
        rhs_v=rhs_extended,
        w=w_extended,
        dt_mom=dt_mom,
        viscosity=viscosity_padded,
        metrics=extended_metrics,
        upwind_shear=upwind_shear,
    )
    return solve_tridiagonal_system(_slice_system(extended_system, axis=-2))


def solve_tracer_vertical_diffusion(
    field: torch.Tensor,
    grid: Any,
    dt_tracer: float,
    diffusivity: float | torch.Tensor,
    *,
    recip_hFac: torch.Tensor | None,
) -> torch.Tensor:
    """Solve a center-tracer IMPLDIFF column with explicit partial-cell input.

    The caller selects the reciprocal hFacC time layer; None explicitly
    selects columns without partial-cell factors.
    """
    from locomgcm.solvers import build_implicit_diffusion_system, tridiag_solve

    a, b, c, d = build_implicit_diffusion_system(
        field,
        grid.layer_thickness,
        interface_spacing=grid.vertical_coordinate.interface_spacing,
        diffusivity=diffusivity,
        dt=dt_tracer,
        recip_hFac=recip_hFac,
    )
    return tridiag_solve(a, b, c, d, mitgcm_order=True)


@dataclass
class TracerVerticalContinuity(TendencyModule):
    """Explicit GAD continuity source with MIT upward-positive interface w.

    The old tracer multiplies the known transport divergence, so this source
    participates in tendency AB and its raw history (gad_calc_rhs.F).
    """

    w: TracerWNpHalfMIT
    tracer_names: tuple[str, ...]
    name = "tracer_vertical_continuity"
    stepper_domain = TendencyStepDomain.SLOW
    tendency_phase = TendencyPhase.TRACER
    ab2_eligible = True

    def evaluate(self, state, diagnostics, grid) -> StateTendency:
        tracers = {}
        mask_c_current = None
        recip_hfac_current = None
        for name in ("temp", "salt"):
            tracer = getattr(state, name).data
            if name in self.tracer_names:
                w = self.w
                if recip_hfac_current is None:
                    mask_c_current, recip_hfac_current = tracer_hfac_factors(
                        field=tracer, hfac_c=getattr(grid.masks, "hFacC", None),
                    )
                tracers[name] = tracer_vertical_transport_correction(
                    tracer_n=tracer, w=w,
                    metrics=build_tracer_vertical_metrics(
                        tracer, grid, mask_c_current=mask_c_current,
                        recip_hfac_matrix_c=recip_hfac_current,
                        recip_hfac_current_c=recip_hfac_current,
                    ),
                )
            else:
                tracers[name] = torch.zeros_like(tracer)
        return StateTendency(
            eta=torch.zeros_like(state.eta.data), u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data), temp=tracers["temp"], salt=tracers["salt"],
        )


def add_tracer_vertical_continuity(
    *,
    tracer_n: torch.Tensor,
    rhs: torch.Tensor,
    w: TracerWNpHalfMIT,
    dt_lev: float | torch.Tensor,
    grid: Any,
    mask_c_current: torch.Tensor,
    recip_hfac_matrix_c: torch.Tensor,
) -> torch.Tensor:
    """Add raw C for Euler tracers, preserving the original state-add order.

    w is MIT upward-positive; layer thickness is positive. AB tracers include
    this source in their tendency instead and must not call this helper.
    recip_hfac_matrix_c is the column-matrix reciprocal, including the future
    surface under NLFS. The wet mask retains the current geometry.
    """
    metrics = build_tracer_vertical_metrics(
        tracer_n, grid, mask_c_current=mask_c_current,
        recip_hfac_matrix_c=recip_hfac_matrix_c,
        recip_hfac_current_c=recip_hfac_matrix_c,
    )
    correction = tracer_vertical_transport_correction(
        tracer_n=tracer_n,
        w=w,
        metrics=metrics,
    )
    dt = dt_lev
    if not isinstance(dt, torch.Tensor):
        # 2 waits per step here otherwise.
        # ``new_tensor`` builds the scalar on the host and copies it in
        # (``cudaMemcpyAsync`` + ``cudaStreamSynchronize``); ``torch.full``
        # passes it as a launch argument.  Same dtype, same rounding, same bits.
        dt = torch.full(
            (), float(dt), dtype=tracer_n.dtype, device=tracer_n.device
        )
    if dt.ndim == 1:
        dt = dt.view(*([1] * (tracer_n.ndim - 3)), dt.shape[0], 1, 1)
    return rhs + dt * correction


def _solve_tracer_three_band(*, rhs, w, dt_lev, diffusivity, metrics, upwind_fraction):
    """Reference three-band assembly and MITgcm-order solve of one tracer column."""
    center = tracer_interface_center(rhs=rhs, w=w, metrics=metrics)
    system = assemble_tracer_implicit_vertical_system(
        rhs=rhs,
        center=center,
        upwind_fraction=upwind_fraction,
        dt_lev=dt_lev,
        diffusivity=diffusivity,
        metrics=metrics,
    )
    return solve_tridiagonal_system(system)


def solve_tracer_implicit_vertical_advection(
    *,
    rhs: torch.Tensor,
    advection_scheme: TracerFluxScheme,
    limiter_tracer: torch.Tensor,
    w: TracerWNpHalfMIT,
    dt_lev: float | torch.Tensor,
    diffusivity: float | torch.Tensor,
    grid: Any,
    mask_c_current: torch.Tensor,
    recip_hfac_matrix_c: torch.Tensor,
    recip_hfac_current_c: torch.Tensor,
) -> torch.Tensor:
    """Solve a complete tracer RHS; explicit GAD sources are assembled upstream.

    w uses MIT upward-positive units and layer thickness is positive. The
    caller selects the scheme (a member of ``IMPLICIT_VERTICAL_TRACER_SCHEMES``)
    and, for dst3_fl, its limiter time layer.
    """
    scheme = advection_scheme
    metrics = build_tracer_vertical_metrics(
        rhs, grid, mask_c_current=mask_c_current,
        recip_hfac_matrix_c=recip_hfac_matrix_c,
        recip_hfac_current_c=recip_hfac_current_c,
    )
    if scheme is TracerFluxScheme.UPWIND:
        operands = (rhs, w.data, diffusivity, dt_lev, *_metric_operands(metrics))
        backend = _fused_backend_for(*operands)
        if backend is not None:
            return solve_tracer_band_thomas_fused(
                rhs=rhs,
                advection_scheme=tracer_scheme_spec(scheme).mit_code,
                w=w,
                dt_lev=dt_lev,
                diffusivity=diffusivity,
                metrics=metrics,
                backend=backend,
            )
        return _solve_tracer_three_band(
            rhs=rhs, w=w, dt_lev=dt_lev, diffusivity=diffusivity, metrics=metrics,
            upwind_fraction=1.0,  # gad_dst2u1_impl_r.F:69-70, rLimit=0
        )
    elif scheme is TracerFluxScheme.CENTERED or scheme is TracerFluxScheme.DST2:
        # Code 2 has no upwind part (gad_c2_impl_r.F:57-83); code 20 is
        # GAD_DST2U1_IMPL_R with rLimit=1, rUpwind = |rCenter|*0
        # (gad_dst2u1_impl_r.F:69-70, 83-84).
        return _solve_tracer_three_band(
            rhs=rhs, w=w, dt_lev=dt_lev, diffusivity=diffusivity, metrics=metrics,
            upwind_fraction=0.0,
        )
    elif (scheme is TracerFluxScheme.UPWIND3 or scheme is TracerFluxScheme.DST3
          or scheme is TracerFluxScheme.DST3_FL):
        from locomgcm.solvers.implicit_vertical_high_order import (
            assemble_tracer_high_order_vertical_system,
            column_scales,
            high_order_interface_transport,
            interface_courant_number,
        )
        from locomgcm.solvers.pentadiag import pentadiag_solve

        dt, rdf, rdc = column_scales(
            rhs=rhs, dt_lev=dt_lev,
            recip_dr_f=metrics.recip_drF, recip_dr_c=metrics.recip_drC,
        )
        transport, center, absolute = high_order_interface_transport(
            w_mit=w.data, mask_c=metrics.mask_c, r_a=metrics.rA, recip_r_a=metrics.recip_rA,
        )
        if scheme is TracerFluxScheme.UPWIND3:
            high, upwind = upwind3_interface_weights(absolute=absolute)
        elif scheme is TracerFluxScheme.DST3:
            cfl = interface_courant_number(
                dt=dt, transport=transport, recip_r_a=metrics.recip_rA, rdc=rdc,
            )
            high, upwind = dst3_interface_weights(cfl=cfl, absolute=absolute)
        elif scheme is TracerFluxScheme.DST3_FL:
            cfl = interface_courant_number(
                dt=dt, transport=transport, recip_r_a=metrics.recip_rA, rdc=rdc,
            )
            high, upwind = dst3fl_interface_weights(
                cfl=cfl, absolute=absolute, transport=transport,
                limiter_tracer=limiter_tracer, mask_c=metrics.mask_c,
            )
        bands = assemble_tracer_high_order_vertical_system(
            rhs=rhs, dt=dt, rdf=rdf, rdc=rdc, diffusivity=diffusivity,
            mask_c=metrics.mask_c, recip_hfac=metrics.recip_hfac_matrix_c,
            center=center, absolute=absolute, high=high, upwind=upwind,
        )
        return pentadiag_solve(*bands)
    else:
        raise ValueError(
            f"implicit vertical tracer advection does not implement scheme "
            f"{advection_scheme!r}; it takes {', '.join(IMPLICIT_VERTICAL_TRACER_SCHEMES)}"
        )


__all__ = [
    "IMPLICIT_VERTICAL_TRACER_SCHEMES",
    "TracerVerticalContinuity",
    "add_tracer_vertical_continuity",
    "build_tracer_vertical_metrics",
    "build_u_vertical_metrics",
    "build_v_vertical_metrics",
    "diagnose_mit_vertical_velocity",
    "make_momentum_w_nm_half_mit",
    "make_tracer_w_np_half_mit",
    "solve_tracer_implicit_vertical_advection",
    "solve_tracer_vertical_diffusion",
    "tracer_hfac_factors",
    "solve_u_implicit_vertical_advection",
    "solve_v_implicit_vertical_advection",
]
