"""Cubed-sphere implicit free-surface step implementation."""

from __future__ import annotations

import os
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from locomgcm.dynamics import AssemblyBucket, AssemblyPolicy, MomentumTermClass
from locomgcm.dynamics.implicit_vertical_advection import solve_tracer_vertical_diffusion
from locomgcm.dynamics.wind_stress import surface_wind_tendency_from_face_stress
from locomgcm.fields import Field
from locomgcm.geometry.grid import top_layer_thickness_value
from locomgcm.operators import (
    active_face_velocity_from_transport_area,
    divergence_volume_transport_to_center_mit,
    face_volume_transport,
    horizontal_volume_divergence_from_transports,
    vertical_velocity_continuity,
    vertical_diffusive_flux,
    vertical_flux_divergence,
)
from locomgcm.state import DiagnosticState, EvaluationInputs, PrognosticState

from .base import TimeStepBundle
from .ifs.tendency import MomentumABHistory, ab2_factor, extrapolate_ab2, retain_cube_momentum_ab
from .split_explicit import SplitStepperModelContext
from .step_segments import (
    PredictorDivergence,
    compute_predictor_divergence,
)

if TYPE_CHECKING:
    from locomgcm.dynamics.kpp_mixing import KppStatic


@dataclass(frozen=True)
class CubeStaticState:
    """Run-owned cube modules, viscosity constants and KPP geometry.

    fcorig is the corner Coriolis field in s-1; KPP tensors keep their
    prepared dtype and rank-local face layout. Viscosity is in m2/s and
    rho_const in kg/m3. Module injection precedes host preparation.
    """

    fcorig: torch.Tensor | None
    momentum: Any
    viscosity: Any
    visc_ah: float
    no_slip_sides: bool
    side_drag_factor: float
    kpp_static: KppStatic | None
    kpp_static_owner: tuple[Any, ...] | None
    rho_const: float
    eos: Any
    bottom_drag: Any

    @property
    def mass2r_unit(self) -> float:
        """MITgcm ``mass2rUnit = recip_rhoConst`` (ini_parms.F:763, :1573), the one
        reciprocal of ``rho_const`` the cube step multiplies by."""
        return 1.0 / self.rho_const


#: Stepper fields the cube step reads, with the values it implements.  The readers
#: are step_cube and its helpers here, the shared predictor, free-surface solve,
#: correction and Helmholtz coefficients it calls, the cube branch of
#: ``prepare_step_host`` and the model assembly.  ``None`` takes every value the field
#: itself accepts; a tuple takes only those values.  A field not listed has no reader
#: on the cube step and keeps its default.  ``refuse_unimplemented_cube_kwargs``
#: applies this at the cubed-sphere model assembly, and the MITgcm translator reads it.
CUBE_STEP_KWARGS: dict[str, tuple[Any, ...] | None] = {
    "gravity": None,
    "implicit_surf_press": None,
    # An unstepped tracer keeps its step-start value (_hold_unstepped_tracers).
    "temp_stepping": None,
    "salt_stepping": None,
    "implicit_div_2d_flow": None,
    "momentum_pressure_forcing": None,
    "ab_eps": None,
    "exact_conserv": None,
    # The cg2d segment has no EmPmR input on the cube and raises when this is set.
    "use_real_freshwater_flux": (False,),
    "implicit_viscosity": None,
    "implicit_vertical_viscosity": None,
    "implicit_tracer_diffusion": None,
    # No implicit vertical advection column on the cube; the momentum flag raises
    # in _predict_velocity without the MomentumWNmHalfMIT wrapper.
    "temperature_implicit_vertical_advection": (False,),
    "salinity_implicit_vertical_advection": (False,),
    "momentum_implicit_vertical_advection": (False,),
    "tracer_vertical_diffusivity_kr": None,
    "tracer_diffusivity_kh": None,
    # Linear free surface only: _get_coefficients switches coefficients for
    # nonlin_free_surf > 2, and the cube step has no surface-thickness geometry.
    "nonlin_free_surf": (0,),
    "select_r_star": (0,),
    # Read by the assembly (resolve_vector_invariant_momentum) against the core.
    "vector_invariant_momentum": None,
    # The cube tracer step has no Adams-Bashforth history.
    "ab2_on_temperature_tendency": (False,),
    "ab2_on_salinity_tendency": (False,),
    "ab2_on_tracer_tendency": (None, False),
    "stagger_tracer_stepping": None,
    "mom_dissip_in_ab": None,
    # The cube AssemblyPolicy takes cube_neutral's False, not this field
    # (AssemblyPolicy.cube); cube_assembly_policy is the cube's override.
    "mom_forcing_out_ab": (False,),
    "cube_assembly_policy": None,
    "implicit_int_grav_wave": None,
    "cg_max_iters": None,
    "cg_tolerance": None,
    "cg_tolerance_mode": None,
    "cg_use_min_res_sol": None,
    "kpp_enabled": None,
    "penetrating_sw": None,
    "kpp_ricr": None,
    "kpp_min_hbl": None,
    "rotation_period": None,
    "kpp_estimate_uref": None,
    "kpp_smooth_shsq": None,
    "kpp_smooth_dbloc": None,
    "kpp_sign_zero": None,
    "kpp_surface_buoyancy": None,
    "kpp_dbloc_form": None,
    "kpp_use_double_diff": None,
    "cube_tracer_dst3fl": None,
    "cube_enable_hpg": None,
    # JMD95P is not wired on the cube (_resolve_cube_eos raises).
    "cube_eos_dynamic_pressure": (False,),
}


def refuse_unimplemented_cube_kwargs(stepper: Any) -> None:
    """Refuse the stepper fields a cubed-sphere run would drop (``CUBE_STEP_KWARGS``)."""
    refused = []
    for spec in dataclasses.fields(stepper):
        if not spec.init:
            continue
        value = getattr(stepper, spec.name)
        if spec.name not in CUBE_STEP_KWARGS:
            default = (
                spec.default_factory()
                if spec.default is dataclasses.MISSING
                else spec.default
            )
            if value != default:
                refused.append(
                    f"{spec.name}={value!r} (no reader on the cube step; only the "
                    f"default {default!r})"
                )
            continue
        admitted = CUBE_STEP_KWARGS[spec.name]
        if admitted is not None and value not in admitted:
            refused.append(f"{spec.name}={value!r} (the cube step implements {admitted})")
    if refused:
        raise NotImplementedError(
            "the cubed-sphere step does not implement " + "; ".join(refused)
        )


INITIAL_CUBE_STATIC_STATE = CubeStaticState(
    fcorig=None,
    momentum=None,
    viscosity=None,
    visc_ah=3.0e5,
    no_slip_sides=True,
    side_drag_factor=2.0,
    kpp_static=None,
    kpp_static_owner=None,
    rho_const=1035.0,
    eos=None,
    bottom_drag=None,
)


def _clone_cube_retained_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {
            key: _clone_cube_retained_value(item) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_cube_retained_value(item) for item in value)
    if dataclasses.is_dataclass(value):
        replacements = {}
        for item in dataclasses.fields(value):
            if not item.init:
                continue
            original = getattr(value, item.name)
            cloned = _clone_cube_retained_value(original)
            if cloned is not original:
                replacements[item.name] = cloned
        return dataclasses.replace(value, **replacements) if replacements else value
    return value


@dataclass(frozen=True)
class CubeStaticContext:
    """Prepared cube geometry for one grid/context and dtype/device.

    Built by :func:`build_cube_static_context` for the run's dtype/device.
    Modules, physical constants and KPP tensors are owned by CubeStaticState.

    ``top_layer_thickness`` is the host float the surface forcing and the wind
    stress divide by; it comes from the grid's own construction-time constant
    (``geometry/grid.py`` ``top_layer_thickness_value``) rather than a
    ``.item()`` in the step.

    """

    dtype: torch.dtype
    device: torch.device
    nz: int
    ny: int
    nx: int
    top_layer_thickness: float
    drF: torch.Tensor
    dyG: torch.Tensor
    dxG: torch.Tensor
    hFacW: torch.Tensor
    hFacS: torch.Tensor
    hFacC: torch.Tensor
    u_mask: torch.Tensor
    v_mask: torch.Tensor
    c_mask: torch.Tensor
    surf: torch.Tensor
    recip_hFacC: torch.Tensor
    rA: torch.Tensor


@dataclass(frozen=True)
class CubeStepEntry:
    u: torch.Tensor
    v: torch.Tensor
    eta: torch.Tensor
    temp: torch.Tensor
    salt: torch.Tensor
    dtype: torch.dtype
    device: torch.device
    nz: int
    ny: int
    nx: int
    drF: torch.Tensor
    dyG: torch.Tensor
    dxG: torch.Tensor
    hFacW: torch.Tensor
    hFacS: torch.Tensor
    hFacC: torch.Tensor
    u_mask: torch.Tensor
    v_mask: torch.Tensor
    c_mask: torch.Tensor
    surf: torch.Tensor
    recip_hFacC: torch.Tensor
    rA: torch.Tensor
    u_trans: torch.Tensor
    v_trans: torch.Tensor
    hdiv_vol: torch.Tensor
    etaH_prev: torch.Tensor


@dataclass(frozen=True)
class CubeMomentumModules:
    momentum: Any
    viscosity: Any
    #: Corner Coriolis ``fCoriG`` -- resolved with the two modules because the
    #: three were one lazy block (``_resolve_cube_momentum_modules``).
    fcorig: torch.Tensor | None = None


@dataclass(frozen=True)
class CubeExactConservUpdate:
    eta_final: torch.Tensor
    etaH_new: torch.Tensor
    u_transport_corrected: torch.Tensor | None
    v_transport_corrected: torch.Tensor | None
    div_corrected: torch.Tensor | None
    d_eta: torch.Tensor | None


@dataclass(frozen=True)
class CubeTracerUpdate:
    temp: torch.Tensor
    salt: torch.Tensor
    # The tracers after the explicit updates, before the implicit column solves.
    explicit_temp: torch.Tensor
    explicit_salt: torch.Tensor


@dataclass(frozen=True)
class CubeMomentumExtrapolation:
    u: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class CubeMomentumTendencies:
    u: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class CubeMomentumModuleTendencies:
    momentum: CubeMomentumTendencies
    viscosity: CubeMomentumTendencies


@dataclass(frozen=True)
class CubeMomentumBuckets:
    ab2_u: torch.Tensor | None = None
    ab2_v: torch.Tensor | None = None
    post_u: torch.Tensor | None = None
    post_v: torch.Tensor | None = None


@dataclass(frozen=True)
class CubeStepResult:
    state: PrognosticState


def _cube_assembly_policy(stepper: Any, bottom_drag_module: Any) -> AssemblyPolicy:
    """Build the shared policy with explicit cube-neutral compatibility inputs."""
    neutral = AssemblyPolicy.cube_neutral(
        bottom_drag_in_ab=not bool(bottom_drag_module.implicit)
    ).with_overrides(
        {
            "mom_dissip_in_ab": stepper.mom_dissip_in_ab,
            "implicit_int_grav_wave": stepper.implicit_int_grav_wave,
        }
    )
    configured = stepper.cube_assembly_policy
    if configured is None:
        return neutral
    if isinstance(configured, AssemblyPolicy):
        return configured
    return neutral.with_overrides(configured)


def _add_cube_momentum_tendency(
    buckets: CubeMomentumBuckets,
    bucket: AssemblyBucket,
    tendency: CubeMomentumTendencies | None,
    *,
    surface_only: bool = False,
) -> CubeMomentumBuckets:
    if tendency is None:
        return buckets
    if bucket == AssemblyBucket.AB2:
        current_u, current_v = buckets.ab2_u, buckets.ab2_v
    else:
        current_u, current_v = buckets.post_u, buckets.post_v
    if current_u is None or current_v is None:
        next_u, next_v = tendency.u, tendency.v
    elif surface_only:
        next_u = current_u.clone()
        next_v = current_v.clone()
        next_u[:, 0] = next_u[:, 0] + tendency.u[:, 0]
        next_v[:, 0] = next_v[:, 0] + tendency.v[:, 0]
    else:
        next_u = current_u + tendency.u
        next_v = current_v + tendency.v
    if bucket == AssemblyBucket.AB2:
        return CubeMomentumBuckets(
            ab2_u=next_u,
            ab2_v=next_v,
            post_u=buckets.post_u,
            post_v=buckets.post_v,
        )
    return CubeMomentumBuckets(
        ab2_u=buckets.ab2_u,
        ab2_v=buckets.ab2_v,
        post_u=next_u,
        post_v=next_v,
    )


def build_cube_static_context(
    grid: Any,
    stepper: Any,
    ctx: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> CubeStaticContext:
    """Prepare cube geometry and resolve run-owned physics eagerly.

    Geometry follows the global or rank-local face layout; drF has no face
    axis. Supplied momentum and viscosity modules remain the injected objects.
    Resolved physics and KPP static tensors are published to CubeStaticState;
    the context retains geometry for the traced step.
    """
    from locomgcm.geometry.cube_decomposition import scatter_faces

    nz, ny, nx = grid.nz, grid.ny, grid.nx

    drF = grid.layer_thickness_for(dtype, device).view(1, nz, 1, 1)
    dyG = grid.metrics.dyG_for(dtype, device).unsqueeze(1)
    dxG = grid.metrics.dxG_row_ext_for(dtype, device).unsqueeze(1)
    hFacW = grid.masks.hFacW.to(dtype=dtype, device=device)
    hFacS = grid.masks.hFacS.to(dtype=dtype, device=device)
    hFacC = grid.masks.hFacC.to(dtype=dtype, device=device)

    if ctx is not None:
        # Multi-rank: scatter every face-varying tensor from [6,...] to
        # [n_local,...] so they match the local state layout.  drF is
        # [1,nz,1,1] (not face-indexed) and stays global.
        # dyG/dxG have an unsqueeze(1) applied above; scatter on the [6,...]
        # base and re-unsqueeze, otherwise scatter_faces (which expects
        # [face,...]) won't index correctly.
        dyG = scatter_faces(grid.metrics.dyG_for(dtype, device), ctx).unsqueeze(1)
        dxG = scatter_faces(grid.metrics.dxG_row_ext_for(dtype, device), ctx).unsqueeze(1)
        hFacW = scatter_faces(hFacW, ctx)
        hFacS = scatter_faces(hFacS, ctx)
        hFacC = scatter_faces(hFacC, ctx)
        rA = scatter_faces(grid.metrics.cell_area_for(dtype, device), ctx)
    else:
        rA = grid.metrics.cell_area_for(dtype, device)

    u_mask = (hFacW > 0).to(dtype)
    v_mask = (hFacS > 0).to(dtype)
    c_mask = (hFacC > 0).to(dtype)
    surf = c_mask[:, 0]
    recip_hFacC = torch.where(hFacC > 0, 1.0 / hFacC, torch.zeros_like(hFacC))

    modules = _resolve_cube_momentum_modules(stepper, grid, dtype, device)
    # Warm the module memos on the grid object the step will hand the modules
    # (the rank-local grid under a context), through the same hook the lat-lon
    # prepare runs on ``model.dynamics_modules``: a cube run's modules live in
    # CubeStaticState and may be another object set, and a cold build inside
    # the traced step is a graph break that outlives the cold step.
    from locomgcm.geometry.cube_localize import localize_cube_grid

    module_grid = localize_cube_grid(grid, ctx) if ctx is not None else grid
    for module in (
        modules.momentum,
        modules.viscosity,
        stepper.cube_static_state.bottom_drag,
    ):
        prepare_constants = getattr(module, "prepare_host_constants", None)
        if prepare_constants is not None:
            prepare_constants(module_grid, dtype=dtype, device=device)
    from locomgcm.dynamics.equation_of_state import validate_dynamic_pressure_composition

    validate_dynamic_pressure_composition(modules=(), stepper=stepper, face_count=6)
    eos = _resolve_cube_eos(stepper, gravity=float(stepper.gravity))
    if bool(getattr(stepper, "kpp_enabled", False)):
        prepare_cube_kpp_static(stepper, grid, ctx, dtype, device)
    object.__setattr__(
        stepper, "cube_static_state",
        dataclasses.replace(
            stepper.cube_static_state,
            fcorig=modules.fcorig,
            momentum=modules.momentum,
            viscosity=modules.viscosity,
            eos=eos,
        ),
    )
    if getattr(grid, "advection_topology", None) == "mit_12tile":
        # The tile GAD schedule's host index tables, built here so the traced
        # tracer advection reads a warm memo (goal 4).  The tables are keyed on
        # the face size, ``grid.mit_tiling`` and ``local_face_ids`` and owned by
        # ``dynamics/_gad_advection_topology`` rather than by this context,
        # because the route from here to their consumer runs through
        # ``dynamics/gad_advection.py``.  Same pure function either way, so
        # warming it changes only which phase pays for the build.
        from locomgcm.dynamics._gad_advection_topology import warm_cube_tile_tables

        local_face_ids = (
            ctx.local_face_ids if ctx is not None else tuple(range(grid.face_count))
        )
        warm_cube_tile_tables(
            tuple(local_face_ids), nx=grid.nx, ny=grid.ny, tiling=grid.mit_tiling
        )
    return CubeStaticContext(
        dtype=dtype,
        device=device,
        nz=nz,
        ny=ny,
        nx=nx,
        top_layer_thickness=top_layer_thickness_value(grid),
        drF=drF,
        dyG=dyG,
        dxG=dxG,
        hFacW=hFacW,
        hFacS=hFacS,
        hFacC=hFacC,
        u_mask=u_mask,
        v_mask=v_mask,
        c_mask=c_mask,
        surf=surf,
        recip_hFacC=recip_hFacC,
        rA=rA,
    )


def resolve_cube_static_context(
    stepper: Any,
    grid: Any,
    ctx: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> CubeStaticContext:
    """The run's :class:`CubeStaticContext`, built on first use.

    Kept on a plain stepper attribute for the run's fixed geometry.

    The guard is the one thing that can legitimately change under a live
    stepper: which grid object and rank context the constants were derived from,
    plus the run dtype/device.  A checkpoint restart, a device migration or
    ``model.with_comm_session`` builds a new grid object, so the context is
    rebuilt rather than serving constants derived from the old geometry.

    The key retains and compares the grid and context objects by identity.
    Retaining them prevents address reuse from serving geometry belonging to
    a released grid. Comparing with ``is`` also avoids tracing ``id()`` calls.
    """
    keyed = getattr(stepper, "_cube_static_key", None)
    if keyed is not None:
        cached_grid, cached_ctx, cached_dtype, cached_device, cached = keyed
        if (
            cached_grid is grid
            and cached_ctx is ctx
            and cached_dtype == dtype
            and cached_device == device
        ):
            return cached
    static = build_cube_static_context(grid, stepper, ctx, dtype, device)
    # One tuple, holding the two keyed objects alongside the product: the
    # reference that makes the identity comparison meaningful and the answer it
    # guards cannot get out of step with each other.  Same shape as
    # ``prepare_cube_kpp_static``'s owner record (``(grid, ctx, …, static)``).
    object.__setattr__(
        stepper, "_cube_static_key", (grid, ctx, dtype, device, static)
    )
    object.__setattr__(stepper, "_cube_static_context", static)
    return static


def _initial_cube_etaH(
    stepper: Any,
    eta: torch.Tensor,
    hdiv_vol: torch.Tensor,
    rA: torch.Tensor,
    surf: torch.Tensor,
    dt_freesurf: float,
) -> torch.Tensor:
    """The exactConserv surface state ``etaH`` of a step with no predecessor.

    MITgcm starts ``etaH`` from ``etaN``; with ``implicDiv2DFlow < 1`` the
    first step also carries the explicit share of the surface divergence.
    One expression, used by the entry's fallback branch and by the seed, so
    the two cannot drift apart.
    """
    if stepper.exact_conserv and stepper.implicit_div_2d_flow != 1.0:
        d_eta_h_dt = -hdiv_vol.sum(dim=1) / rA
        return (
            eta
            + (1.0 - stepper.implicit_div_2d_flow) * dt_freesurf * d_eta_h_dt
        ) * surf
    return eta * surf


def seed_cube_cross_step_state(
    stepper: Any,
    state: PrognosticState,
    grid: Any,
    bundle: TimeStepBundle,
    ctx: Any = None,
) -> None:
    """Give a fresh cube run the cross-step history its first step would
    otherwise build inline.  **Eager (initialisation).**

    Before the first step, this seeds ``_etaH`` through the entry's expression
    (:func:`_initial_cube_etaH`, on the bounded state the first step receives)
    and an AB2 history of zeros with a false ``valid`` flag, which
    :func:`_compute_cube_momentum_extrapolation` selects away with
    ``torch.where`` -- bit-for-bit the forward-Euler branch.  Both keys are in
    the restart allowlist, so a step-zero checkpoint now carries them and a
    segment restored from it starts warm.  Keys already present are left alone:
    a restart import has published its own history.

    The zeros follow the state's autograd posture (``requires_grad`` of the
    controls) so a seeded segment head and a warm step present the same tensor
    guards; the gradient they receive is identically zero.

    A missing host velocity readout is diagnosed from the prepared state's
    velocity in MIT upward-positive m/s, with the producer dtype and local
    face layout. The cube kernel replaces it with each step's entry-velocity readout.
    The prepare hook calls this idempotently after resolving the static context.
    """
    u = state.u.data
    v = state.v.data
    dtype, device = u.dtype, u.device
    static = resolve_cube_static_context(stepper, grid, ctx, dtype, device)
    entry = None
    if stepper._continuity.eta_h is None or stepper.cube_w_interface_readout is None:
        entry = _prepare_cube_step_entry(stepper, state, grid, bundle, ctx, static)
    if stepper._continuity.eta_h is None:
        object.__setattr__(stepper, "_continuity", stepper._continuity.with_eta_h(entry.etaH_prev))
    if stepper.ab_eps > 0.0 and stepper._momentum_ab is None:
        object.__setattr__(stepper, "_momentum_ab", MomentumABHistory(
            torch.zeros_like(u, requires_grad=u.requires_grad),
            torch.zeros_like(v, requires_grad=v.requires_grad),
            valid=torch.zeros((), dtype=torch.bool, device=device),
        ))
    if stepper.cube_w_interface_readout is None:
        object.__setattr__(
            stepper, "cube_w_interface_readout",
            vertical_velocity_continuity(
                entry.hdiv_vol,
                cell_area=static.rA.unsqueeze(1),
                form="velocity",
                convention="upward",
            ).detach(),
        )


def _prepare_cube_step_entry(
    stepper: Any,
    state: PrognosticState,
    grid: Any,
    bundle: TimeStepBundle,
    ctx: Any = None,
    static: CubeStaticContext | None = None,
) -> CubeStepEntry:
    """Collect the state-dependent half of one cube step's entry record.

    Geometry, masks and reciprocals come from CubeStaticContext, including
    multi-rank face localization. This function assembles the state-dependent
    prognostic tensors, face transports, divergence and etaH history.

    *static* is required in practice -- ``step_cube`` always passes it -- and
    defaults to ``None`` only so the two harness call sites that build an entry
    without a run context keep working; that branch builds the context itself,
    with the same arguments, so the numbers do not depend on which way in.
    """
    u = state.u.data
    v = state.v.data
    eta = state.eta.data
    temp = state.temp.data
    salt = state.salt.data
    dtype, device = u.dtype, u.device
    if static is None:
        static = build_cube_static_context(grid, stepper, ctx, dtype, device)
    nz, ny, nx = static.nz, static.ny, static.nx
    drF = static.drF
    dyG = static.dyG
    dxG = static.dxG
    hFacW = static.hFacW
    hFacS = static.hFacS
    hFacC = static.hFacC
    u_mask = static.u_mask
    v_mask = static.v_mask
    c_mask = static.c_mask
    surf = static.surf
    recip_hFacC = static.recip_hFacC
    rA = static.rA

    u_trans, v_trans = face_volume_transport(
        u,
        v,
        drF,
        dyG,
        dxG,
        hFacW=hFacW,
        hFacS=hFacS,
    )
    hdiv_vol = horizontal_volume_divergence_from_transports(
        u_trans,
        v_trans,
        wet_mask=c_mask,
    )
    etaH_prev = stepper._continuity.eta_h
    if etaH_prev is None:
        # A step with no predecessor and no seed (a harness entry built without
        # ``initialize_fresh_step_state``); a run seeds the key before its
        # first step (``seed_cube_cross_step_state``) through this same
        # expression, so the traced step never takes this branch.
        etaH_prev = _initial_cube_etaH(
            stepper, eta, hdiv_vol, rA, surf, bundle.dt_freesurf
        )

    return CubeStepEntry(
        u=u,
        v=v,
        eta=eta,
        temp=temp,
        salt=salt,
        dtype=dtype,
        device=device,
        nz=nz,
        ny=ny,
        nx=nx,
        drF=drF,
        dyG=dyG,
        dxG=dxG,
        hFacW=hFacW,
        hFacS=hFacS,
        hFacC=hFacC,
        u_mask=u_mask,
        v_mask=v_mask,
        c_mask=c_mask,
        surf=surf,
        recip_hFacC=recip_hFacC,
        rA=rA,
        u_trans=u_trans,
        v_trans=v_trans,
        hdiv_vol=hdiv_vol,
        etaH_prev=etaH_prev,
    )


def _resolve_cube_momentum_modules(
    stepper: Any,
    grid: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> CubeMomentumModules:
    """Resolve momentum, viscosity and corner Coriolis for host preparation.

    Injected modules win over the configured defaults. Host preparation
    publishes the resolved values to CubeStaticState.
    """
    from locomgcm.dynamics.horizontal_viscosity import HorizontalViscosity
    from locomgcm.dynamics.vector_invariant_momentum import (
        VectorInvariantMomentumAdvection,
    )
    from locomgcm.dynamics._cube_momentum import build_fcorig_corner
    from locomgcm.constants import omega_from_rotation_period

    momentum = stepper.cube_static_state.momentum
    if momentum is None:
        momentum = VectorInvariantMomentumAdvection(
            select_vort_scheme=1,
            select_ke_scheme=0,
            enable_vertical_shear=True,
            split_coriolis=True,
        )

    # The corner f follows the momentum module's rotation period.
    fcorig = stepper.cube_static_state.fcorig
    if fcorig is None:
        fcorig = build_fcorig_corner(
            grid, dtype, device,
            omega=omega_from_rotation_period(momentum.rotation_period),
        )

    viscosity = stepper.cube_static_state.viscosity
    if viscosity is None:
        # MITgcm-aligned cube dissipation: div-vort harmonic form
        # (``useStrainTensionVisc=.FALSE.``) + no-slip lateral drag
        # (``no_slip_sides=.TRUE.``, ``sideDragFactor=2``).
        viscosity = HorizontalViscosity(
            viscosity=stepper.cube_static_state.visc_ah,
            viscosity_scheme="divvort",
            no_slip_sides=bool(stepper.cube_static_state.no_slip_sides),
            side_drag_factor=float(stepper.cube_static_state.side_drag_factor),
        )

    return CubeMomentumModules(
        momentum=momentum, viscosity=viscosity, fcorig=fcorig
    )


def _resolve_cube_eos(stepper: Any, *, gravity: float) -> Any:
    """Resolve the cube equation of state from run-owned inputs, eagerly.

    ``cube_eos_dynamic_pressure`` selects JMD95P, which needs the lagged
    ``totPhiHyd`` (baroclinic + ``g*eta``) feedback cache and a first-step
    Picard seed that the cube does not carry; the model's assembly rule
    (``validate_dynamic_pressure_composition``) refuses it, and the cube's
    host preparation applies the same rule for a stepper prepared without a
    model.
    """
    eos = stepper.cube_static_state.eos
    if eos is not None:
        return eos
    from locomgcm.dynamics.equation_of_state import JMD95EquationOfState

    # Grid-independent, config-driven EOS shared with the lat-lon path -- the
    # cube does not hardcode JMD95Z.  The dynamic-pressure knob is refused
    # before this runs (validate_dynamic_pressure_composition).
    return JMD95EquationOfState(
        gravity=gravity,
        reference_density=float(stepper.cube_static_state.rho_const),
        dynamic_pressure=False,
    )


def _compute_cube_depth_integrated_divergence(
    u_face: torch.Tensor,
    v_face: torch.Tensor,
    grid: Any,
    *,
    drF: torch.Tensor,
    dyG: torch.Tensor,
    dxG: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    ctx: Any = None,
    rA: Any = None,
    wet_mask: Any = None,
) -> PredictorDivergence:
    """Thin shim over ``step_segments.compute_predictor_divergence``.

    The depth-integrate -> seam-reconcile -> diverge chain lives in the
    topology-blind segment function.  This wrapper exists because the chain has
    a second consumer (``_compute_cube_exact_conserv_update``): delegating keeps
    that call site untouched and automatically same-source.  The shim can go
    once the exactConserv stage becomes its own segment.

    ``stream_trace=None`` is deliberate: the segment's
    ``cube.predictor_transport_divergence`` stream trace belongs to the
    *predictor* consumer only (never the exactConserv path), and this shim has no stepper in scope.

    *rA* and *wet_mask* are optional pre-computed local cell area and centre
    wet mask (see ``divergence_volume_transport_to_center_mit``).  When ctx is
    given (multi-rank), callers should pass the locally-scattered tensors so the
    divergence kernel does not re-read the global ``[6,...]`` values from grid.
    """
    return compute_predictor_divergence(
        u_face,
        v_face,
        grid,
        stream_trace=None,
        layer_thickness=drF,
        dyG=dyG,
        dxG=dxG,
        hFacW=hFacW,
        hFacS=hFacS,
        rA=rA,
        wet_mask=wet_mask,
        ctx=ctx,
    )


def _recover_face_tracer(flux: torch.Tensor, transport: torch.Tensor) -> torch.Tensor:
    return torch.where(
        transport != 0.0,
        flux / torch.where(transport != 0.0, transport, torch.ones_like(transport)),
        torch.zeros_like(flux),
    )


def _emit_cube_gad_tracer_trace(
    stepper: Any,
    tracer_name: str,
    gad: dict[str, torch.Tensor],
    *,
    hdiv_vol: torch.Tensor,
    wflux: torch.Tensor,
    u_transport: torch.Tensor,
    v_transport: torch.Tensor,
    drF: torch.Tensor,
    hFacC: torch.Tensor,
    nz: int,
) -> None:
    stepper.stream_trace.emit_stream_trace(
        "tracer.horizontal_advection",
        tracer=tracer_name,
        horizontal_scheme=gad.get("horizontal_scheme"),
        flux_sidecar_kind=gad.get("flux_sidecar_kind"),
        transport_u=u_transport,
        transport_v=v_transport,
        tracer_u=_recover_face_tracer(gad["afx"], u_transport),
        tracer_v=_recover_face_tracer(gad["afy"], v_transport),
        flux_u=gad["afx"],
        flux_v=gad["afy"],
        divergence=-(gad["horizontal_tendency"] * drF * hFacC),
        tendency=gad["horizontal_tendency"],
    )
    stepper.stream_trace.emit_stream_trace(
        "tracer.vertical_advection",
        tracer=tracer_name,
        horizontal_divergence_volume=hdiv_vol,
        vertical_volume_flux=wflux,
        vertical_tracer_flux=gad["fver"],
        vertical_tracer_flux_diag=gad["fver"][:, :nz],
        vertical_flux_tendency=gad["vertical_tendency"],
        surface_dilution=torch.zeros_like(gad["vertical_tendency"][:, 0]),
        tendency=gad["vertical_tendency"],
    )


def _emit_regional_horizontal_tracer_trace(
    stepper: Any,
    tracer: torch.Tensor,
    tracer_name: str,
    tendency: torch.Tensor,
    *,
    u_velocity: torch.Tensor,
    v_velocity: torch.Tensor,
    u_transport: torch.Tensor,
    v_transport: torch.Tensor,
    grid: Any,
) -> None:
    from locomgcm.operators.reconstructions import (
        center_to_u_upwind,
        center_to_v_upwind,
    )

    f_u = center_to_u_upwind(tracer, u_velocity, grid)
    f_v = center_to_v_upwind(tracer, v_velocity, grid)
    flux_u = u_transport * f_u
    flux_v = v_transport * f_v
    div = divergence_volume_transport_to_center_mit(flux_u, flux_v, grid)
    stepper.stream_trace.emit_stream_trace(
        "tracer.horizontal_advection",
        tracer=tracer_name,
        flux_u=flux_u,
        flux_v=flux_v,
        divergence=div,
        tendency=tendency,
    )


@torch._dynamo.disable
def prepare_cube_kpp_static(
    stepper: Any,
    grid: Any,
    ctx: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> "KppStatic":
    """Prepare KPP static tensors in CubeStaticState outside traced code.

    Explicitly injected tensors retain their owner. The owner tuple records
    grid, rank context, precision and device so only this builder's own
    tensors are rebuilt when those inputs change.
    """
    from locomgcm.dynamics.kpp_mixing import build_kpp_static_from_grid

    static = stepper.cube_static_state.kpp_static
    owner = stepper.cube_static_state.kpp_static_owner
    if static is not None:
        # Preserve explicitly injected static tensors; only invalidate our own build.
        if owner is None or static is not owner[-1]:
            return static
        if (owner[0] is grid and owner[1] is ctx
                and owner[2:4] == (dtype, device)):
            return static

    # Multi-rank cube KPP: localize the grid to this rank's owned faces so
    # the static masks and Coriolis field have the same face ownership as state.
    if ctx is not None and grid.face_count == 6:
        from locomgcm.geometry.cube_localize import localize_cube_grid
        kpp_grid = localize_cube_grid(grid, ctx)
    else:
        kpp_grid = grid

    from locomgcm.constants import omega_from_rotation_period

    lt = kpp_grid.layer_thickness_for(dtype, device)
    static = build_kpp_static_from_grid(
        kpp_grid, lt, dtype=dtype, device=device,
        omega=omega_from_rotation_period(stepper.rotation_period),
    )
    # Retain grid/context owners so identity checks cannot reuse their ids.
    object.__setattr__(
        stepper, "cube_static_state",
        dataclasses.replace(
            stepper.cube_static_state,
            kpp_static=static,
            kpp_static_owner=(grid, ctx, dtype, device, static),
        ),
    )
    return static


def _compute_cube_kpp_fields(
    stepper: Any,
    temp: torch.Tensor,
    salt: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    grid: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
    ctx: Any = None,
):
    """Consume prepared static tensors for per-step cube KPP diffusivities.

    Gated by ``_cube_kpp_enabled``.  Surface forcing: wind stress from
    ``_cube_surface_forcing.tau_x/tau_y`` (sfU/sfV = tau*mass2rUnit) and net heat flux from
    ``_cube_surface_forcing.qnet`` (MIT Qnet, ``(6, ny, nx)``).

    A produced SurfaceForcing bundle carries sfT/sfS and the explicit
    penetrating_sw mode. Raw adapters use the record's mode or the stepper
    setting: disabled shortwave deposits Qnet at the surface; enabled mode
    splits Qsw for penetration and KPP, including at instantaneous zero flux.
    EmPmR is the bulk freshwater flux [m/s], converted to [kg/m²/s] via
    rhoConstFresh for the raw salt adapter. Returns the :class:`KppFields`.
    """
    from locomgcm.dynamics.kpp_mixing import (
        KPP_STOCK_PARAMS,
        cube_kpp_fields,
    )
    from locomgcm.dynamics.surface_flux import (
        surface_heat_forcing,
        surface_salt_forcing,
    )

    rho0 = float(stepper.cube_static_state.rho_const)
    grav = float(getattr(stepper, "gravity", 9.81))
    cp = float(KPP_STOCK_PARAMS["heat_capacity_cp"])

    # Multi-rank cube KPP: localize the grid to this rank's owned faces so
    # the cube halo exchanges inside cube_kpp_fields route to the distributed P2P
    # path (same face-local pattern as the tracer GAD).  ctx=None -> global grid unchanged.
    if ctx is not None and grid.face_count == 6:
        from locomgcm.geometry.cube_localize import localize_cube_grid
        kpp_grid = localize_cube_grid(grid, ctx)
    else:
        kpp_grid = grid

    static = stepper.cube_static_state.kpp_static
    if static is None:
        raise RuntimeError(
            "Cube KPP static tensors not prepared: call "
            "prepare_cube_kpp_static(...) before computing cube KPP fields"
        )

    surface_inputs = stepper._cube_surface_forcing
    tau_x = surface_inputs.tau_x
    tau_y = surface_inputs.tau_y
    qnet = surface_inputs.qnet
    if tau_x is None or tau_y is None or qnet is None:
        raise ValueError("KPP enabled but _cube_tau_x/_cube_tau_y/_cube_qnet not set")
    qnet = qnet.to(dtype=dtype, device=device)
    # surfaceForcingU = fu*mass2rUnit (external_forcing_surf.F:222)
    mass2r_unit = stepper.cube_static_state.mass2r_unit
    sfU = tau_x.to(dtype=dtype, device=device) * mass2r_unit
    sfV = tau_y.to(dtype=dtype, device=device) * mass2r_unit
    forcing = surface_inputs.bulk
    if forcing is not None:
        # Production consume the one produced SurfaceForcing bundle.
        qsw_c = forcing.qsw
        penetrating_sw = forcing.penetrating_sw
        sfT = forcing.sfT.to(dtype=dtype, device=device)
        sfS = forcing.sfS.to(dtype=dtype, device=device)
    else:
        penetrating_sw = (
            stepper.penetrating_sw
            if surface_inputs.penetrating_sw is None else surface_inputs.penetrating_sw
        )
        qsw_c = surface_inputs.qsw
        if qsw_c is None and not penetrating_sw:
            sfT = -qnet / (cp * rho0)
        else:
            if qsw_c is None:
                qsw_c = torch.zeros_like(qnet)
            sfT = surface_heat_forcing(qnet, qsw_c.to(dtype=dtype, device=device),
                                       heat_capacity=cp, rho_const=rho0,
                                       penetrating_sw=penetrating_sw)
        # Salt: virtual salt flux from freshwater EmPmR (bulk flux [m/s] ×
        # rhoConstFresh → [kg/m²/s]); absent ⟹ sfS=0 (bitwise).
        empmr_c = surface_inputs.empmr
        if empmr_c is None:
            sfS = torch.zeros_like(sfT)
        else:
            rho_fresh = float(
                rho0 if surface_inputs.rho_const_fresh is None
                else surface_inputs.rho_const_fresh
            )
            sfS = surface_salt_forcing(
                empmr_c.to(dtype=dtype, device=device) * rho_fresh,
                rho_const=rho0,
                surface_salinity=salt.select(dim=-3, index=0),
            )
    # Same parameter source as the lat-lon step (data.kpp overrides, grid-derived
    # minKPPhbl, viscAr background); the EOS is the cube's own.
    from locomgcm.steppers.ifs.kpp_inputs import kpp_driver_params

    params = kpp_driver_params(
        stepper._kpp_run_values, rho_const=rho0,
        eos_module=_resolve_cube_eos(stepper, gravity=grav),
        min_hbl_default=static.min_hbl_default,
    )
    params["penetrating_sw"] = penetrating_sw
    # KPP surface buoyancy forcing is the COMPLETE forcing: sfT/sfS -> bo, sfU/sfV ->
    # ustar, and net shortwave qsw -> bosol. The explicit mode controls whether
    # the radiative contribution is consumed, independently of zero Qsw.
    qsw_kpp = qsw_c.to(dtype=dtype, device=device) if qsw_c is not None else None

    # The surface forcing cache knobs are GLOBAL [6,ny,nx]; temp/salt/u/v from the
    # step entry are already LOCAL [n_local,...].  Scatter the forcing to match so
    # cube_kpp_fields runs face-local.  (Mirrors _apply_cube_wind_stress.)
    if ctx is not None:
        from locomgcm.geometry.cube_decomposition import scatter_faces
        sfT = scatter_faces(sfT, ctx)
        sfS = scatter_faces(sfS, ctx)
        sfU = scatter_faces(sfU, ctx)
        sfV = scatter_faces(sfV, ctx)
        if qsw_kpp is not None:
            qsw_kpp = scatter_faces(qsw_kpp, ctx)

    diffkr = float(stepper.tracer_vertical_diffusivity_kr)
    return cube_kpp_fields(temp, salt, u, v, sfT, sfS, sfU, sfV, qsw_kpp,
                           grid=kpp_grid, static=static, params=params,
                           background_diffusivity_s=diffkr, background_diffusivity_t=diffkr)


def _resolve_cube_dst3fl(stepper: Any) -> bool:
    """Resolve the cube DST3-FL tracer-advection switch by priority.

    env ``LOCOMGCM_CUBE_TRACER_DST3FL`` (highest, when explicitly set) > stepper field
    ``cube_tracer_dst3fl`` (default True, the regular config surface via
    model.stepper.kwargs).  There is no ``_cube_tracer_dst3fl`` shadow cache
    tier: the field is the single source of truth (all cube yaml
    already carry it in stepper.kwargs), env override kept for op-probe harnesses.
    """
    env_val = os.environ.get("LOCOMGCM_CUBE_TRACER_DST3FL")
    if env_val is not None:
        return env_val != "0"
    return bool(getattr(stepper, "cube_tracer_dst3fl", True))


def _compute_cube_tracer_update(
    stepper: Any,
    temp: torch.Tensor,
    salt: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    u_new: torch.Tensor,
    v_new: torch.Tensor,
    grid: Any,
    *,
    dt_tracer: float,
    drF: torch.Tensor,
    dyG: torch.Tensor,
    dxG: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    hFacC: torch.Tensor,
    c_mask: torch.Tensor,
    recip_hFacC: torch.Tensor,
    nz: int,
    ctx: Any = None,
    kpp: Any = None,
) -> CubeTracerUpdate:
    """Compute cube tracer advection update.

    *kpp* is this step's KPP bundle, required when kpp_enabled is true.
    Background kr is independent of KPP: without KPP it is applied explicitly
    from TracerN or implicitly from the updated RHS, according to the diffusion
    gate. The bundle arrives as an argument rather than a cache lookup.

    When *ctx* is provided and ``grid.face_count == 6`` the GAD advection runs
    FACE-LOCAL: the grid is localized to the rank's owned faces and the
    GAD retains the source grid's declared advection topology.
    ``mit_12tile`` cuts each owned face into its ``mit_tiling`` tiles and runs the original global
    face pass schedule; ``six_face_approx`` uses the existing whole-face path.
    Both exchange cross-face halos through the distributed exchange helpers
    and commit only rank-local fields, without gathering global tracers.
    """
    from locomgcm.dynamics.gad_advection import (
        cube_horizontal_tracer_diffusion,
        gad_advection_first_order_cube,
    )
    from locomgcm.dynamics.tracers import (
        horizontal_tracer_advection_from_volume_transports,
    )
    from locomgcm.operators.fluxes.tracer_flux import TracerFluxScheme

    if stepper.stagger_tracer_stepping and grid.face_count == 6:
        from locomgcm.dynamics._vector_seam_topology import (
            reconcile_vector_redundant_edges_for_grid,
        )

        # Use the same reconciled seam velocities as predictor/exactConserv
        # before forming w and horizontal transport. MIT INTEGR_CONTINUITY
        # likewise uses one uFld/vFld pair for the free surface and wVel.
        u_adv, v_adv = reconcile_vector_redundant_edges_for_grid(
            u_new, v_new, grid, ctx=ctx,
        )
    elif stepper.stagger_tracer_stepping:
        u_adv, v_adv = u_new, v_new
    else:
        u_adv, v_adv = u, v
    u_vel, v_vel = active_face_velocity_from_transport_area(
        u_adv,
        v_adv,
        drF * hFacW * dyG,
        drF * hFacS * dxG,
    )
    u_trans3d, v_trans3d = face_volume_transport(
        u_adv,
        v_adv,
        drF,
        dyG,
        dxG,
        hFacW=hFacW,
        hFacS=hFacS,
    )
    stepper.stream_trace.emit_stream_trace(
        "cube.tracer_advection_velocity",
        u_adv=u_adv,
        v_adv=v_adv,
        u_transport=u_trans3d,
        v_transport=v_trans3d,
    )

    # Cube tracer DST3-FL scheme (horizontal + vertical).
    # Priority resolve (see ``_resolve_cube_dst3fl``): env LOCOMGCM_CUBE_TRACER_DST3FL
    # (highest, when explicitly set) > cache knob _cube_tracer_dst3fl (harness/driver
    # --dst3, when present) > stepper config field cube_tracer_dst3fl (default True).
    # ONE knob turns on BOTH horizontal AND vertical DST3-FL (= MIT tempAdvScheme=33
    # + tempVertAdvScheme=33; gad_dst3fl_adv_{x,y,r}.F + Sweby limiter).  DST3-FL is
    # wired on ALL cube paths: mit_12tile, single-card six_face, AND the multi-card
    # face-local branch (2-rank == 1-rank bitwise verified).  Default ON ->
    # cube matches region + MIT production; set False (or env=0) for the
    # 1st-order upwind byte-id baseline.
    _dst3_on = _resolve_cube_dst3fl(stepper)
    cube_h_scheme = (
        TracerFluxScheme.DST3_FL if _dst3_on else TracerFluxScheme.UPWIND
    )

    # Cube horizontal tracer diffusion (full MIT GAD_DIFF
    # flux form).  Cube had NO horizontal diffusion (region has the module;
    # p125 uses diffKh=100 m2/s).  Knob = ``_cube_tracer_diffKh`` (m2/s), default
    # 0.0 -> the block below adds NOTHING (byte-id no-op).  >0 -> MIT-faithful
    # Laplacian diffusion via the shared full-flux-form core (cube-only adoption;
    # region keeps its simplified bitwise-MIT-for-regular-grids _diffuse — the
    # two forms diverge on 2-D metrics, NOT FP-floor, so region is left untouched,
    # is_cube-leaf style like momentum-adv).
    # Read from the stepper field (single source of truth
    # physics.diffusivity, injected by the cube runtime), not a cache shadow.
    _cube_diffKh = float(stepper.tracer_diffusivity_kh)

    if grid.face_count == 6 and ctx is not None:
        # Face-local: preserve the declared GAD topology while scattering
        # metrics and masks; halo exchanges retain cross-rank coupling.
        from locomgcm.geometry.cube_localize import localize_cube_grid

        local_grid = localize_cube_grid(grid, ctx)

        hdiv_vol = horizontal_volume_divergence_from_transports(
            u_trans3d,
            v_trans3d,
            wet_mask=c_mask,
        )
        # Cube tracer R-pass wflux via the geometry-neutral continuity core:
        # form='volume_flux' (no /rA, m3/s) + convention='upward' reproduces MIT
        # upward-positive wflux[k]=wflux[k+1]-div_h[k] (negated to LOCOMgcm-down inside GAD).
        wflux = vertical_velocity_continuity(
            hdiv_vol, form="volume_flux", convention="upward",
        )

        # Both GAD topologies consume the same scheme and face velocities;
        # the localized grid carries ownership for their halo exchanges.
        temp_gad = gad_advection_first_order_cube(
            tracer=temp,
            u_transport=u_trans3d,
            v_transport=v_trans3d,
            vertical_volume_flux=wflux,
            grid=local_grid,
            dt=dt_tracer,
            h_scheme=cube_h_scheme,
            v_scheme=cube_h_scheme,
            u_vel=u_vel,
            v_vel=v_vel,
        )
        salt_gad = gad_advection_first_order_cube(
            tracer=salt,
            u_transport=u_trans3d,
            v_transport=v_trans3d,
            vertical_volume_flux=wflux,
            grid=local_grid,
            dt=dt_tracer,
            h_scheme=cube_h_scheme,
            v_scheme=cube_h_scheme,
            u_vel=u_vel,
            v_vel=v_vel,
        )
        # Do NOT return here -- fall through to the shared KPP vertical-mixing
        # block so the face-local KPP diffusivities/ghat/surface-flux are applied to
        # the tracer (previously multi-rank returned advection-only, skipping KPP).
        temp_new = temp_gad["final"]
        salt_new = salt_gad["final"]
    elif grid.face_count == 6:
        hdiv_vol = horizontal_volume_divergence_from_transports(
            u_trans3d,
            v_trans3d,
            wet_mask=c_mask,
        )
        # Cube tracer R-pass wflux via the geometry-neutral continuity core:
        # form='volume_flux' (no /rA, m3/s) + convention='upward' reproduces MIT
        # upward-positive wflux[k]=wflux[k+1]-div_h[k] (negated to LOCOMgcm-down inside GAD).
        wflux = vertical_velocity_continuity(
            hdiv_vol, form="volume_flux", convention="upward",
        )

        temp_gad = gad_advection_first_order_cube(
            tracer=temp,
            u_transport=u_trans3d,
            v_transport=v_trans3d,
            vertical_volume_flux=wflux,
            grid=grid,
            dt=dt_tracer,
            h_scheme=cube_h_scheme,
            v_scheme=cube_h_scheme,
            u_vel=u_vel,
            v_vel=v_vel,
        )
        temp_tendency = temp_gad["total_tendency"]
        _emit_cube_gad_tracer_trace(
            stepper,
            "T",
            temp_gad,
            hdiv_vol=hdiv_vol,
            wflux=wflux,
            u_transport=u_trans3d,
            v_transport=v_trans3d,
            drF=drF,
            hFacC=hFacC,
            nz=nz,
        )
        stepper.stream_trace.emit_stream_trace(
            "tracer.total_tendency",
            tracer="T",
            horizontal_tendency=temp_gad["horizontal_tendency"],
            vertical_tendency=temp_gad["vertical_tendency"],
            total_tendency=temp_tendency,
        )
        temp_new = temp_gad["final"]

        salt_gad = gad_advection_first_order_cube(
            tracer=salt,
            u_transport=u_trans3d,
            v_transport=v_trans3d,
            vertical_volume_flux=wflux,
            grid=grid,
            dt=dt_tracer,
            h_scheme=cube_h_scheme,
            v_scheme=cube_h_scheme,
            u_vel=u_vel,
            v_vel=v_vel,
        )
        salt_tendency = salt_gad["total_tendency"]
        _emit_cube_gad_tracer_trace(
            stepper,
            "S",
            salt_gad,
            hdiv_vol=hdiv_vol,
            wflux=wflux,
            u_transport=u_trans3d,
            v_transport=v_trans3d,
            drF=drF,
            hFacC=hFacC,
            nz=nz,
        )
        stepper.stream_trace.emit_stream_trace(
            "tracer.total_tendency",
            tracer="S",
            horizontal_tendency=salt_gad["horizontal_tendency"],
            vertical_tendency=salt_gad["vertical_tendency"],
            total_tendency=salt_tendency,
        )
        salt_new = salt_gad["final"]
    else:
        temp_hadv = horizontal_tracer_advection_from_volume_transports(
            temp,
            u_vel,
            v_vel,
            u_trans3d,
            v_trans3d,
            grid,
            recip_hFacC=recip_hFacC,
            layer_thickness=drF,
        )
        salt_hadv = horizontal_tracer_advection_from_volume_transports(
            salt,
            u_vel,
            v_vel,
            u_trans3d,
            v_trans3d,
            grid,
            recip_hFacC=recip_hFacC,
            layer_thickness=drF,
        )
        if stepper.stream_trace.hook is not None:
            _emit_regional_horizontal_tracer_trace(
                stepper,
                temp,
                "T",
                temp_hadv,
                u_velocity=u_vel,
                v_velocity=v_vel,
                u_transport=u_trans3d,
                v_transport=v_trans3d,
                grid=grid,
            )
            _emit_regional_horizontal_tracer_trace(
                stepper,
                salt,
                "S",
                salt_hadv,
                u_velocity=u_vel,
                v_velocity=v_vel,
                u_transport=u_trans3d,
                v_transport=v_trans3d,
                grid=grid,
            )
        temp_new = (temp + dt_tracer * temp_hadv) * c_mask
        salt_new = (salt + dt_tracer * salt_hadv) * c_mask
        stepper.stream_trace.emit_stream_trace(
            "tracer.total_tendency",
            tracer="T",
            horizontal_tendency=temp_hadv,
            vertical_tendency=torch.zeros_like(temp_hadv),
            total_tendency=temp_hadv * c_mask,
        )
        stepper.stream_trace.emit_stream_trace(
            "tracer.total_tendency",
            tracer="S",
            horizontal_tendency=salt_hadv,
            vertical_tendency=torch.zeros_like(salt_hadv),
            total_tendency=salt_hadv * c_mask,
        )

    # Cube horizontal tracer diffusion (full MIT GAD_DIFF flux form), gated by
    # ``_cube_tracer_diffKh`` (read above).  diffKh == 0.0 -> add NOTHING (the
    # default; byte-id no-op vs the prior cube, which had no horizontal diffusion).
    # >0 -> temp_new += dt * diff(temp) * c_mask (and salt), after horizontal
    # advection (both DST3 and upwind branches converge here).
    #
    # Multi-card (ctx is not None): run on the rank-LOCAL grid (``local_grid``,
    # built above by the face-local GAD branch — ``localize_cube_grid`` is
    # memoized, so this re-fetch is free).  Step-start ``temp``/``salt`` and
    # ``c_mask`` are rank-local, and the local grid carries ``_cube_step_ctx`` so the
    # tracer corner-halo (``fill_cs_corner_tr_rl``) + the metric cross-face
    # exchanges (``exchange_hfac_pair`` for dyG/dxG, recip_dxC/recip_dyC,
    # hFacW/hFacS) inside ``cube_horizontal_tracer_diffusion`` self-dispatch to the
    # distributed P2P seam — the SAME routing the six_face DST3 path uses.
    # Both branches diffuse TracerN, as GAD_CALC_RHS does before tracer integration.
    if _cube_diffKh != 0.0:
        if ctx is not None:
            from locomgcm.geometry.cube_localize import localize_cube_grid
            _diff_grid = localize_cube_grid(grid, ctx)
        else:
            _diff_grid = grid
        temp_new = temp_new + dt_tracer * cube_horizontal_tracer_diffusion(
            temp, _diff_grid, _cube_diffKh,
        ) * c_mask
        salt_new = salt_new + dt_tracer * cube_horizontal_tracer_diffusion(
            salt, _diff_grid, _cube_diffKh,
        ) * c_mask

    diffkr = float(stepper.tracer_vertical_diffusivity_kr)
    if stepper.kpp_enabled and kpp is None:
        raise RuntimeError("cube tracer mixing requires the configured KPP bundle")
    if not stepper.kpp_enabled and kpp is not None:
        raise ValueError("cube tracer mixing received KPP fields while KPP is disabled")

    # KPP vertical mixing of T/S: explicit ghat nonlocal counter-gradient
    # transport (MIT kpp_transport_t + gad_calc_rhs divergence) THEN implicit
    # vertical diffusion (MIT IMPLDIFF with KPPdiffKzT/KzS), matching MIT's
    # explicit-then-implicit order.
    if stepper.kpp_enabled:
        from locomgcm.dynamics.kpp_mixing import kpp_ghat_tendency
        # Surface heat forcing (MIT EXTERNAL_FORCING_T, apply_forcing.F): the
        # (non-shortwave) sfT at the surface.  gT(1) += sfT*recip_drF[0]*recip_hFacC[0].
        # The top layer's thickness as a host constant the grid owns from
        # construction (``ZLevelCoordinate.top_layer_thickness_value``), not a
        # ``.item()`` off the memoised ``drF`` product, which would be a host
        # read inside the step.  Measured, not assumed: 406 call sites across
        # the cube suites (float64 and float32, both fixtures) agree exactly, and
        # the two spellings can only differ when the consuming dtype is NARROWER
        # than the grid's, which no cube path does (the runtime grid is built at
        # the run dtype).
        recip_drF0 = 1.0 / top_layer_thickness_value(grid)
        heat = torch.zeros_like(temp_new)
        heat[:, 0] = kpp.surf_forc_t * recip_drF0 * recip_hFacC[:, 0]
        temp_new = temp_new + dt_tracer * heat
        # Surface salt forcing (MIT EXTERNAL_FORCING_S): gS(1) += sfS*recip_drF[0]*
        # recip_hFacC[0], the EmPmR virtual salt flux; surf_forc_s=0 ⟹ no-op
        # (bitwise).
        salt_sfc = torch.zeros_like(salt_new)
        salt_sfc[:, 0] = kpp.surf_forc_s * recip_drF0 * recip_hFacC[:, 0]
        salt_new = salt_new + dt_tracer * salt_sfc
        # Penetrating shortwave : Qsw absorbed over
        # depth.  Reuses the per-face shortwave_penetration_tendency (z=dim0) over the
        # cube [6,nz,ny,nx] layout; the physical mode controls this contribution.
        surface_inputs = stepper._cube_surface_forcing
        qsw_pen = surface_inputs.qsw
        if (stepper.penetrating_sw if surface_inputs.penetrating_sw is None
                else surface_inputs.penetrating_sw):
            if qsw_pen is None:
                qsw_pen = torch.zeros_like(surface_inputs.qnet)
            from locomgcm.dynamics.kpp_mixing import KPP_STOCK_PARAMS
            from locomgcm.dynamics.surface_flux import shortwave_penetration_tendency
            cp_pen = float(KPP_STOCK_PARAMS["heat_capacity_cp"])
            rho0_pen = float(stepper.cube_static_state.rho_const)
            qsw_pen = qsw_pen.to(dtype=temp_new.dtype, device=temp_new.device)
            # qsw input is GLOBAL [6,ny,nx]; temp_new/hFacC are LOCAL multi-rank.
            if ctx is not None:
                from locomgcm.geometry.cube_decomposition import scatter_faces
                qsw_pen = scatter_faces(qsw_pen, ctx)
            drF_1d = drF[0, :, 0, 0]
            iface = torch.cat([drF.new_zeros(1), torch.cumsum(drF_1d, 0)])
            # shortwave_penetration_tendency is layout-agnostic over dim=-3, so the
            # cube [6,nz,ny,nx] hFacC + [6,ny,nx] qsw go in directly (no per-face
            # loop); per-column, no halo => single- and multi-card face-local safe.
            temp_new = temp_new + dt_tracer * shortwave_penetration_tendency(
                qsw_pen, iface, drF_1d, hFacC,
                heat_capacity=cp_pen, rho_const=rho0_pen, jwtype=2,
            )
        # ghat nonlocal counter-gradient (explicit), before the implicit solve, with
        # KPPdiffKzT/S, which include the background (kpp_transport_t.F).
        # Uses the COMPLETE temperature forcing surf_forc_t_nl (= sfT minus the
        # shortwave heating the mixing layer, MIT kpp_transport_t.F); salt has no
        # shortwave term (kpp_transport_s.F).
        temp_new = temp_new + dt_tracer * kpp_ghat_tendency(
            kpp.ghat, kpp.diff_kzt, kpp.surf_forc_t_nl, drF, recip_hFacC,
        )
        salt_new = salt_new + dt_tracer * kpp_ghat_tendency(
            kpp.ghat, kpp.diff_kzs, kpp.surf_forc_s, drF, recip_hFacC,
        )
        # The implicit solve takes KappaRT/S = KPPdiffKzT/S (kpp_calc_diff_t.F:50),
        # which already hold the background diffKrT/S (cube_kpp_fields).
        explicit_temp, explicit_salt = temp_new, salt_new
        # Pass the rank-LOCAL recip_hFacC: ``grid`` here is the global cube
        # grid, but temp_new/diff_kzt are face-local in multi-rank.
        temp_new = solve_tracer_vertical_diffusion(
            field=temp_new, grid=grid, dt_tracer=dt_tracer,
            diffusivity=kpp.diff_kzt, recip_hFac=recip_hFacC,
        ) * c_mask
        salt_new = solve_tracer_vertical_diffusion(
            field=salt_new, grid=grid, dt_tracer=dt_tracer,
            diffusivity=kpp.diff_kzs, recip_hFac=recip_hFacC,
        ) * c_mask
        stepper.stream_trace.emit_stream_trace(
            "cube.tracer_kpp", temp=temp_new, salt=salt_new,
        )
    elif stepper.implicit_tracer_diffusion and diffkr != 0.0:
        explicit_temp, explicit_salt = temp_new, salt_new
        temp_new = solve_tracer_vertical_diffusion(
            field=temp_new, grid=grid, dt_tracer=dt_tracer,
            diffusivity=diffkr, recip_hFac=recip_hFacC,
        ) * c_mask
        salt_new = solve_tracer_vertical_diffusion(
            field=salt_new, grid=grid, dt_tracer=dt_tracer,
            diffusivity=diffkr, recip_hFac=recip_hFacC,
        ) * c_mask
    elif diffkr != 0.0:
        # GAD_CALC_RHS passes the old T/S to GAD_DIFF_R, even after the
        # multidimensional advection pass. The downward-positive flux and its
        # negative divergence account for the two MIT rkSign=-1 factors.
        # Zero interface fluxes at wet/dry joins; rA cancels against cell volume.
        # All hFac/mask inputs here are rank-local, including multi-rank cubes.
        temp_flux = vertical_diffusive_flux(
            temp, grid.vertical_coordinate.interface_spacing, diffusivity=diffkr, mask=c_mask,
        )
        salt_flux = vertical_diffusive_flux(
            salt, grid.vertical_coordinate.interface_spacing, diffusivity=diffkr, mask=c_mask,
        )
        temp_diff = vertical_flux_divergence(temp_flux, grid.layer_thickness) * recip_hFacC
        salt_diff = vertical_flux_divergence(salt_flux, grid.layer_thickness) * recip_hFacC
        temp_new = temp_new + dt_tracer * torch.where(
            hFacC > 0, temp_diff, torch.zeros_like(temp_diff),
        )
        salt_new = salt_new + dt_tracer * torch.where(
            hFacC > 0, salt_diff, torch.zeros_like(salt_diff),
        )
        explicit_temp, explicit_salt = temp_new, salt_new
    else:
        explicit_temp, explicit_salt = temp_new, salt_new

    return CubeTracerUpdate(
        temp=temp_new, salt=salt_new,
        explicit_temp=explicit_temp, explicit_salt=explicit_salt,
    )


def _compute_cube_momentum_extrapolation(
    stepper: Any,
    tend_u: torch.Tensor,
    tend_v: torch.Tensor,
    *,
    post_tend_u: torch.Tensor | None,
    post_tend_v: torch.Tensor | None,
) -> CubeMomentumExtrapolation:
    """Apply AB2 momentum extrapolation, then the policy-selected post bucket.

    The history retains the raw eligible tendencies before the post bucket.
    """
    if stepper.ab_eps > 0.0 and stepper._momentum_ab is not None:
        prev = stepper._momentum_ab
        ab_fac = ab2_factor(stepper.ab_eps)
        ex_u = extrapolate_ab2(tend_u, prev.u, factor=ab_fac)
        ex_v = extrapolate_ab2(tend_v, prev.v, factor=ab_fac)
        # A seeded history (``seed_cube_cross_step_state``) carries zeros and a
        # false ``valid`` flag: the first step must take the forward-Euler
        # branch, which is what MITgcm's ``AB_START`` does on step 1 and what
        # the absent-history branch below does. The select uses the tensor
        # flag, so the same graph serves the first step and every later one.
        # Bit-neutral: ``where`` returns the untouched ``tend`` on the false branch.
        valid = prev.valid
        if valid is not None:
            ex_u = torch.where(valid, ex_u, tend_u)
            ex_v = torch.where(valid, ex_v, tend_v)
    else:
        ex_u, ex_v = tend_u, tend_v

    if stepper.ab_eps > 0.0:
        history = retain_cube_momentum_ab(tend_u, tend_v)
        object.__setattr__(stepper, "_momentum_ab", history)

    stepper.stream_trace.emit_stream_trace(
        "momentum.post_ab",
        g_u=ex_u,
        g_v=ex_v,
    )

    if post_tend_u is not None and post_tend_v is not None:
        ex_u = ex_u + post_tend_u
        ex_v = ex_v + post_tend_v

    return CubeMomentumExtrapolation(u=ex_u, v=ex_v)


def _compute_cube_momentum_module_tendencies(
    stepper: Any,
    state: PrognosticState,
    grid: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
    hdiv_vol: torch.Tensor,
    rA: torch.Tensor,
    u_trans: torch.Tensor,
    v_trans: torch.Tensor,
    ctx: Any = None,
) -> CubeMomentumModuleTendencies:
    """Evaluate the momentum/viscosity modules and stage the continuity ``w``.

    For the MULTI-RANK path (ctx is not None, face-local): evaluate
    VIM.evaluate() and viscosity.evaluate() directly on the rank-local state and
    a localized grid (``localize_cube_grid``: masks/metrics scattered to owned
    faces, ``_cube_step_ctx`` attached).  The operators' cross-face halo
    exchanges route to the distributed P2P backend.

    This face-local approach is bitwise-equal to the former gather-evaluate-
    scatter because:
      - momentum/viscosity are per-face stencils (no cross-face reduction), so
        each owned face's tendency depends only on that face's haloed inputs.
      - the distributed halo exchanges are proven bitwise == the single-rank
        rot90 halos, so each face sees identical halo values.
      - it is literally the single-rank code path below, restricted to the
        rank's owned faces.

    Momentum consumes live w through EvaluationInputs; the host diagnostic
    retains the detached rank-local interface velocity.
    """
    from locomgcm.geometry.cube_localize import localize_cube_grid

    static_state = stepper.cube_static_state

    # MIT INTEGRATE_FOR_W free-surface bottom-up continuity
    # w(k)=w(k+1)+conv2d*recip_rA (conv2d=-hdiv): the geometry-neutral core divides
    # by rA per layer (MIT-faithful conv2d*recip_rA ordering, tighter than the old
    # column-sum-then-divide). The velocity is MIT upward-positive for
    # VIM vertical shear / NH Coriolis.
    w_interface = vertical_velocity_continuity(
        hdiv_vol,
        cell_area=rA.unsqueeze(1),
        form="velocity",
        convention="upward",
    )
    # The host attribute holds a detached rank-local diagnostic readout. Momentum
    # consumes the live tensor through EvaluationInputs.

    w_readout = w_interface.detach()
    object.__setattr__(stepper, "cube_w_interface_readout", w_readout.detach())
    stepper.stream_trace.emit_stream_trace(
        "cube.continuity_from_entry_velocity",
        u_transport=u_trans,
        v_transport=v_trans,
        horizontal_divergence_volume=hdiv_vol,
        w_interface_upward=w_interface,
    )

    resolved_3d_coriolis = stepper._resolved_3d_coriolis
    diag = DiagnosticState(inputs=EvaluationInputs(
        w_interface=w_interface,
        select_3d_cori_scheme=resolved_3d_coriolis.select_3d_cori_scheme,
        f_cori_cos=resolved_3d_coriolis.f_cori_cos,
        momentum_pressure_forcing=stepper.momentum_pressure_forcing,
    ))
    if ctx is not None:
        # Face-local: evaluate momentum/viscosity directly on the rank-local
        # state + a localized grid (masks/metrics scattered to owned faces, ctx
        # attached so the operators' cross-face halo exchanges route distributed
        # P2P).  No gather: momentum/viscosity are per-face stencils, so the local
        # tendency is bitwise-equal to the former gather-evaluate-scatter (same
        # operator code, same halo values via the proven distributed exchanges).
        # This branch is the single-rank path below, restricted to a subdomain.
        local_grid = localize_cube_grid(grid, ctx)
        # w_interface is already rank-local; VIM vertical shear reads it as
        # this evaluation's input (MIT upward-positive convention preserved).
        tend_mom = static_state.momentum.evaluate(state, diag, local_grid)
        tend_visc = static_state.viscosity.evaluate(state, diag, local_grid)

        stepper.stream_trace.emit_stream_trace(
            "cube.momentum_modules",
            tend_mom_u=tend_mom.u,
            tend_mom_v=tend_mom.v,
            tend_visc_u=tend_visc.u,
            tend_visc_v=tend_visc.v,
        )
        return CubeMomentumModuleTendencies(
            momentum=CubeMomentumTendencies(u=tend_mom.u, v=tend_mom.v),
            viscosity=CubeMomentumTendencies(u=tend_visc.u, v=tend_visc.v),
        )

    tend_mom = static_state.momentum.evaluate(state, diag, grid)
    tend_visc = static_state.viscosity.evaluate(state, diag, grid)

    stepper.stream_trace.emit_stream_trace(
        "cube.momentum_modules",
        tend_mom_u=tend_mom.u,
        tend_mom_v=tend_mom.v,
        tend_visc_u=tend_visc.u,
        tend_visc_v=tend_visc.v,
    )
    return CubeMomentumModuleTendencies(
        momentum=CubeMomentumTendencies(u=tend_mom.u, v=tend_mom.v),
        viscosity=CubeMomentumTendencies(u=tend_visc.u, v=tend_visc.v),
    )


def _apply_cube_hydrostatic_pressure_gradient(
    stepper: Any,
    temp: torch.Tensor,
    salt: torch.Tensor,
    grid: Any,
    *,
    gravity: float,
    hFacC: torch.Tensor,
    ctx: Any = None,
) -> CubeMomentumTendencies | None:
    """Compute the current-state hydrostatic pressure-gradient tendency.

    When *ctx* is provided (multi-rank), the local *hFacC* is forwarded
    to ``cube_phi_hyd`` (overriding ``grid.masks.hFacC``) and to
    ``cube_hydrostatic_pressure_gradient`` (distributed exchange + scattered
    dxC/dyC metrics).  ctx=None: unchanged single-rank path.

    phi_hyd is the step-local potential used by the pressure-gradient tendency.
    """
    from locomgcm.dynamics._cube_momentum import (
        cube_hydrostatic_pressure_gradient,
        cube_phi_hyd,
    )

    # Gate reads the regular stepper field ``cube_enable_hpg``
    # (model.stepper.kwargs); there is no ``_cube_enable_hpg`` cache key.
    if not getattr(stepper, "cube_enable_hpg", True):
        return None

    if not stepper.momentum_pressure_forcing:
        return None

    eos = stepper.cube_static_state.eos
    phi_hyd = cube_phi_hyd(
        temp,
        salt,
        grid,
        gravity=gravity,
        reference_density=eos.reference_density,
        hFacC=hFacC,  # local [n_local,...] when ctx given; else grid default
        eos=eos,
    )
    hpg_u, hpg_v = cube_hydrostatic_pressure_gradient(
        phi_hyd,
        grid,
        hFacC=hFacC,
        ctx=ctx,
    )
    stepper.stream_trace.emit_stream_trace(
        "cube.hydrostatic_pressure_gradient",
        phi_hyd=phi_hyd,
        hpg_u=hpg_u,
        hpg_v=hpg_v,
    )
    return CubeMomentumTendencies(u=hpg_u, v=hpg_v)


def _apply_cube_bottom_drag(
    stepper: Any,
    state: PrognosticState,
    grid: Any,
    *,
    ctx: Any = None,
) -> CubeMomentumTendencies | None:
    """Compute the shared BottomDrag tendency for policy placement.

    When *ctx* is provided (multi-rank), bottom drag is column-local (no
    horizontal exchange), so ``BottomDrag.evaluate`` runs directly on the
    rank-local faces via a ``localize_cube_grid`` view (masks/metrics sliced to
    owned faces) — bitwise-identical to the single-rank path, without the old
    all_gather of every prognostic field to global face order.
    ctx=None: unchanged single-rank path.
    """
    # The cube dynamics modules are assembled by the registry in
    # the cube runtime (build_cube_model) from the regular config surface
    # (single source of truth) and stored in CubeStaticState; the step-loop
    # orchestration below just consumes the injected template.  Gating reads the
    # module fields, not _cube_* shadow cache keys.  The template
    # carries the RAW no_slip_bottom, whereas the old code baked ns_bot_active into
    # BottomDrag(no_slip_bottom=ns_bot_active); this is bitwise-identical because
    # BottomDrag.evaluate re-ANDs no_slip_bottom with vertical_viscosity>0 at both
    # consumption points (bottom_drag.py:409 / linear_no_slip_..._coefficient:138),
    # so (ns_bot_active AND viscAr>0) == (no_slip_bottom AND viscAr>0).
    module = stepper.cube_static_state.bottom_drag
    # ``BottomDrag.active`` is the same gate (linear or quadratic coefficient
    # non-zero, or no-slip bottom with viscAr > 0) decided at construction as a
    # Python bool.  Reading ``module.linear`` here would be a ``Tensor.item()``
    # inside the traced step.
    if module.active:
        bottom_drag = module
        if ctx is not None:
            # Bottom drag is column-local (no horizontal exchange), so evaluate
            # on the rank-local faces directly — same localize pattern as the
            # momentum/tracer tendencies above.  The old gather-to-global path
            # all_gather'd all 5 prognostic fields (~4.5 GB/step at cs768) only so
            # evaluate could read grid.masks.hFacW/S globally; localize_cube_grid
            # scatters those masks to the owned faces (pure slice, bitwise-equal),
            # and the tendency comes out already in local face layout.
            from locomgcm.geometry.cube_localize import localize_cube_grid
            local_grid = localize_cube_grid(grid, ctx)
            drag_tend = bottom_drag.evaluate(state, DiagnosticState(), local_grid)
            drag_u = drag_tend.u
            drag_v = drag_tend.v
        else:
            drag_tend = bottom_drag.evaluate(state, DiagnosticState(), grid)
            drag_u = drag_tend.u
            drag_v = drag_tend.v
        stepper.stream_trace.emit_stream_trace(
            "cube.bottom_drag",
            drag_u=drag_u,
            drag_v=drag_v,
        )
        return CubeMomentumTendencies(u=drag_u, v=drag_v)
    return None


def _apply_cube_wind_stress(
    stepper: Any,
    grid: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    ctx: Any = None,
) -> CubeMomentumTendencies | None:
    """Compute cube surface wind stress at C-grid face points.

    When *ctx* is provided (multi-rank), the global ``[6,...]`` wind stress
    tensors in ``_cube_surface_forcing.tau_x/tau_y`` are scattered to the local
    ``[n_local,...]`` layout before use, so they match the locally-shaped
    ``hFacW``/``hFacS`` and tend tensors.  ctx=None: unchanged global path.
    """
    tau_x = stepper._cube_surface_forcing.tau_x
    tau_y = stepper._cube_surface_forcing.tau_y
    if tau_x is None or tau_y is None:
        return None

    rho0 = float(stepper.cube_static_state.rho_const)
    # Same host constant, same source as the tracer surface forcing above; the
    # inline ``float(grid.layer_thickness[0])`` was the second ``.item()`` cut.
    drF0 = top_layer_thickness_value(grid)
    tx = tau_x.to(dtype=dtype, device=device)
    ty = tau_y.to(dtype=dtype, device=device)

    # Tau tensors are global [6,...]; scatter to local when ctx given.
    if ctx is not None:
        from locomgcm.geometry.cube_decomposition import scatter_faces
        tx = scatter_faces(tx, ctx)
        ty = scatter_faces(ty, ctx)

    wind_u, wind_v = surface_wind_tendency_from_face_stress(
        tx,
        ty,
        hFacW0=hFacW[:, 0],
        hFacS0=hFacS[:, 0],
        rho0=rho0,
        drF0=drF0,
    )
    tend_u = torch.zeros(
        (wind_u.shape[0], grid.nz, *wind_u.shape[1:]),
        dtype=dtype,
        device=device,
    )
    tend_v = torch.zeros(
        (wind_v.shape[0], grid.nz, *wind_v.shape[1:]),
        dtype=dtype,
        device=device,
    )
    tend_u[:, 0] = wind_u
    tend_v[:, 0] = wind_v
    stepper.stream_trace.emit_stream_trace(
        "cube.wind_stress",
        wind_u=wind_u,
        wind_v=wind_v,
    )
    return CubeMomentumTendencies(u=tend_u, v=tend_v)


def _compute_cube_momentum_forcing(
    stepper: Any,
    state: PrognosticState,
    grid: Any,
    *,
    temp: torch.Tensor,
    salt: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    gravity: float,
    hdiv_vol: torch.Tensor,
    rA: torch.Tensor,
    u_trans: torch.Tensor,
    v_trans: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    hFacC: torch.Tensor,
    ctx: Any = None,
) -> CubeMomentumExtrapolation:
    """Assemble every cube momentum term through the shared policy.

    Prepared physics comes from CubeStaticState; histories use their owners.
    """
    module_tendencies = _compute_cube_momentum_module_tendencies(
        stepper,
        state,
        grid,
        dtype=dtype,
        device=device,
        hdiv_vol=hdiv_vol,
        rA=rA,
        u_trans=u_trans,
        v_trans=v_trans,
        ctx=ctx,
    )
    bottom_drag_module = stepper.cube_static_state.bottom_drag
    policy = _cube_assembly_policy(stepper, bottom_drag_module)
    buckets = _add_cube_momentum_tendency(
        CubeMomentumBuckets(),
        policy.bucket_of(stepper.cube_static_state.momentum),
        module_tendencies.momentum,
    )
    buckets = _add_cube_momentum_tendency(
        buckets,
        policy.bucket_of(stepper.cube_static_state.viscosity),
        module_tendencies.viscosity,
    )

    drag_tendency = _apply_cube_bottom_drag(
        stepper,
        state,
        grid,
        ctx=ctx,
    )
    buckets = _add_cube_momentum_tendency(
        buckets,
        policy.bucket_of(bottom_drag_module),
        drag_tendency,
    )

    wind_tendency = _apply_cube_wind_stress(
        stepper,
        grid,
        dtype=dtype,
        device=device,
        hFacW=hFacW,
        hFacS=hFacS,
        ctx=ctx,
    )
    buckets = _add_cube_momentum_tendency(
        buckets,
        policy.bucket_for(MomentumTermClass.FORCING),
        wind_tendency,
        surface_only=True,
    )

    hpg_tendency = _apply_cube_hydrostatic_pressure_gradient(
        stepper,
        temp,
        salt,
        grid,
        gravity=gravity,
        hFacC=hFacC,
        ctx=ctx,
    )
    buckets = _add_cube_momentum_tendency(
        buckets,
        policy.bucket_for(MomentumTermClass.PRESSURE_GRADIENT),
        hpg_tendency,
    )
    if buckets.ab2_u is None or buckets.ab2_v is None:
        raise RuntimeError("cube momentum assembly produced an empty AB2 bucket")

    extrapolated = _compute_cube_momentum_extrapolation(
        stepper,
        buckets.ab2_u,
        buckets.ab2_v,
        post_tend_u=buckets.post_u,
        post_tend_v=buckets.post_v,
    )
    stepper.stream_trace.emit_stream_trace(
        "cube.momentum_tendency_after_ab",
        tend_u=buckets.ab2_u,
        tend_v=buckets.ab2_v,
        ex_u=extrapolated.u,
        ex_v=extrapolated.v,
    )
    return extrapolated


def _compute_cube_exact_conserv_update(
    stepper: Any,
    eta_new: torch.Tensor,
    etaH_prev: torch.Tensor,
    u_new: torch.Tensor,
    v_new: torch.Tensor,
    grid: Any,
    *,
    dt_freesurf: float,
    beta: float,
    drF: torch.Tensor,
    dyG: torch.Tensor,
    dxG: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    ctx: Any = None,
    rA: Any = None,
    wet_mask: Any = None,
) -> CubeExactConservUpdate:
    """Recompute eta/etaH from corrected transports when exactConserv is active.

    *rA* and *wet_mask* are forwarded to ``_compute_cube_depth_integrated_divergence``
    (see that function for the multi-rank motivation).
    """
    if not stepper.exact_conserv:
        return CubeExactConservUpdate(
            eta_final=eta_new,
            etaH_new=eta_new,
            u_transport_corrected=None,
            v_transport_corrected=None,
            div_corrected=None,
            d_eta=None,
        )

    corrected = _compute_cube_depth_integrated_divergence(
        u_new,
        v_new,
        grid,
        drF=drF,
        dyG=dyG,
        dxG=dxG,
        hFacW=hFacW,
        hFacS=hFacS,
        ctx=ctx,
        rA=rA,
        wet_mask=wet_mask,
    )
    d_eta = -dt_freesurf * corrected.divergence
    eta_final = etaH_prev + beta * d_eta
    etaH_new = eta_final if beta == 1.0 else eta_final + (1.0 - beta) * d_eta
    return CubeExactConservUpdate(
        eta_final=eta_final,
        etaH_new=etaH_new,
        u_transport_corrected=corrected.u_transport,
        v_transport_corrected=corrected.v_transport,
        div_corrected=corrected.divergence,
        d_eta=d_eta,
    )


def _apply_cube_exact_conserv_update(
    stepper: Any,
    eta_new: torch.Tensor,
    etaH_prev: torch.Tensor,
    u_new: torch.Tensor,
    v_new: torch.Tensor,
    grid: Any,
    *,
    dt_freesurf: float,
    beta: float,
    drF: torch.Tensor,
    dyG: torch.Tensor,
    dxG: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    ctx: Any = None,
    rA: Any = None,
    wet_mask: Any = None,
) -> CubeExactConservUpdate:
    """Apply exactConserv eta/etaH update and publish the refreshed etaH.

    *rA* and *wet_mask* are forwarded to the divergence kernel.  When ctx is
    given (multi-rank), pass the locally-scattered cell area and centre mask so
    the kernel does not re-read the global ``[6,...]`` values from grid.

    The *etaH* cache stores the local ``[n_local, ny, nx]`` tensor in the
    multi-rank path; the harness gathers it before writing checkpoints.
    """
    exact = _compute_cube_exact_conserv_update(
        stepper,
        eta_new,
        etaH_prev,
        u_new,
        v_new,
        grid,
        dt_freesurf=dt_freesurf,
        beta=beta,
        drF=drF,
        dyG=dyG,
        dxG=dxG,
        hFacW=hFacW,
        hFacS=hFacS,
        ctx=ctx,
        rA=rA,
        wet_mask=wet_mask,
    )
    if exact.div_corrected is not None:
        stepper.stream_trace.emit_stream_trace(
            "cube.exact_conserv",
            u_transport_corrected=exact.u_transport_corrected,
            v_transport_corrected=exact.v_transport_corrected,
            div_corrected=exact.div_corrected,
            d_eta=exact.d_eta,
            eta_final=exact.eta_final,
            eta_H_new=exact.etaH_new,
        )
    # etaH is stored in local [n_local,ny,nx] shape when ctx is
    # given.  The multi-rank harness gathers it back to [6,...] before writing
    # checkpoint npz files (gather_snapshot).  ctx=None: unchanged global [6,...].
    #
    # The ``.clone()`` stays here, at the producer: it is what detaches the
    # retained history from the step's graph-owned tensor while preserving
    # its autograd connection.
    etaH_new = exact.etaH_new.clone()
    object.__setattr__(stepper, "_continuity", stepper._continuity.with_eta_h(etaH_new))
    return exact


def _assemble_cube_step_result(
    stepper: Any,
    state: PrognosticState,
    grid: Any,
    *,
    eta_final: torch.Tensor,
    u_new: torch.Tensor,
    v_new: torch.Tensor,
    temp_new: torch.Tensor,
    salt_new: torch.Tensor,
    u_mask: torch.Tensor,
    v_mask: torch.Tensor,
    ctx: Any = None,
) -> CubeStepResult:
    """Reconcile cube redundant edges and assemble the updated state.

    When *ctx* is provided the redundant-edge reconciliation exchange is routed
    to the distributed path.  ctx=None retains the single-rank path.
    """
    from locomgcm.dynamics._vector_seam_topology import (
        reconcile_vector_redundant_edges_for_grid,
    )

    # MITgcm keeps the redundant cube velocity edge as exchanged halo state.
    # Refresh it after exactConserv/tracers so the next predictor cannot use
    # an independently evolved shared-edge DOF.
    if grid.face_count == 6:
        u_new, v_new = reconcile_vector_redundant_edges_for_grid(
            u_new,
            v_new,
            grid,
            u_mask=u_mask,
            v_mask=v_mask,
            ctx=ctx,
        )
        stepper.stream_trace.emit_stream_trace(
            "cube.reconciled_redundant_edges",
            u_new=u_new,
            v_new=v_new,
        )

    result = state.with_tensors(
        eta=eta_final, u=u_new, v=v_new, temp=temp_new, salt=salt_new,
    )
    stepper.stream_trace.emit_stream_trace(
        "cube.result",
        eta=eta_final,
        u=u_new,
        v=v_new,
        temp=temp_new,
        salt=salt_new,
    )
    # The counter advances once, in
    # ``ImplicitFreeSurfaceStepper.advance_step_index``, which the eager step
    # epilogue calls immediately after this function returns.  Same value at
    # every later read; bumping here as well would double-count.
    return CubeStepResult(state=result)


def step_cube(
    self: Any,
    state: PrognosticState,
    model: SplitStepperModelContext,
    bundle: TimeStepBundle,
    *,
    time_seconds: float = 0.0,
) -> tuple[PrognosticState, torch.Tensor, torch.Tensor]:
    """Self-contained cubed-sphere (global) semi-implicit step.

    Wires the core global-ocean forecast loop on the FACE-FIRST cube state
    ``[6, nz, ny, nx]`` / ``u[6,nz,ny,nx+1]`` / ``v[6,nz,ny+1,nx]`` /
    ``eta[6,ny,nx]``:

      1. Momentum tendency: cube vector-invariant advection + Coriolis
         (``VectorInvariantMomentumAdvection._evaluate_cube``) + harmonic
         horizontal viscosity (``HorizontalViscosity._evaluate_cube``) +
         BAROCLINIC hydrostatic pressure gradient (JMD95Z EOS +
         ``calc_phi_hyd`` vertical integral + seam-aware ``-∇phiHyd``) +
         surface wind stress.  AB2 extrapolation (ab_eps).
      2. Predict ``u* = u^n + dt*(tend - (1-α)·g·∇η^n)``.
      3. Predicted transport divergence (face-aware nz-sum of
         ``u*·drF·hFacW·dyG``).
      4. Helmholtz RHS = ``rA·(etaH - β·dt_fs·div)``; solve cube cg2d
         (``_assemble_helmholtz_coefficients`` + ``cg2d_solve``).
      5. Velocity correction ``u^{n+1} = u* - α·g·dt·∇η^{n+1}`` (masked).
      6. exactConserv η recompute from corrected transport divergence.
      7. Horizontal tracer advection (MITgcm volume-transport upwind) with
         the corrected velocity; ``T += dt_tracer·tend``.

    Configured viscosity and bottom drag:
      * Implicit vertical viscosity ``viscAr`` (``stepper.implicit_viscosity``): MIT
        IMPLDIFF backward-Euler Thomas column solve on u*/v* (step 2b).
      * Linear bottom drag ``r_bot`` (``CubeStaticState.bottom_drag.linear``):
        ``-r·u·recip_hFac·recip_drF`` at the deepest wet level (step 1,
        into AB2).  Default 0 = MIT bottomDragLinear (no-op).
      * Quadratic bottom drag ``q_bot`` (``CubeStaticState.bottom_drag.quadratic``):
        adds ``q·SQRT(KE(i)+KE(i-1))`` to ``cDrag`` at the deepest wet level,
        with ``KE`` the unweighted cell-center kinetic energy averaged across
        the cube seam via ``exchange``.  Matches MITgcm
        ``mom_u_botdrag_coeff.F`` (``selectBotDragQuadr=0``, ``inp_KE=.TRUE.``):
        ``cDrag += bottomDragQuadratic·SQRT(KE(i,j)+KE(i-1,j))``.  Default 0 = off.
      * Implicit bottom drag switch (``BottomDrag.implicit``):
        when True, the bottom-drag tendency is computed by a backward-Euler
        update ``u_new = u/(1+dt·rate)``.  The cube-neutral assembly policy
        defaults this tendency to POST-AB2; explicit drag defaults to AB2.
        ``cube_assembly_policy.bottom_drag_in_ab`` can override either placement.
        This differs from MITgcm's exact implicit treatment, which folds the
        bottom-drag rate into the vertical tridiagonal (IMPLDIFF) solve; the
        post-AB2 form is an unconditionally-stable approximation, NOT yet
        op-level numerically validated against a MIT dump.  Default False =
        explicit drag into AB2 (bitwise-identical to the pre-switch path).

    SIMPLIFICATIONS (flagged — deferred for the first smoke):
      * Vertical tracer ADVECTION is ON (continuity-diagnosed w, flux-form
        upwind — required for a stable baroclinic cube: the horizontal-only
        tracer scheme spuriously piles T/S at the high-divergence valence-3
        seam corners and the column runs away).  Implicit vertical tracer
        DIFFUSION follows implicit_tracer_diffusion when KPP is disabled;
        KPP uses implicit KzT/KzS plus the background kr. Without the implicit
        gate, background kr contributes explicit diffusion from the old T/S.
      * Non-hydrostatic CG3D OFF (hydrostatic only).
      * nonlinFreeSurf / r* rescaling OFF (linear free surface).
      * Surface heat/salt flux + restoring is off by default; applied (KPP
        surface forcing + EmPmR / penetrating shortwave) when KPP and the
        current surface flux inputs are configured.
      * No AB2 on tracer tendency; boundaries are the cube seams (closed
        global domain, no OBCS).
    """
    grid = model.grid  # type: ignore[attr-defined]
    dt_mom = bundle.dt_mom
    dt_tracer = bundle.dt_tracer
    dt_freesurf = bundle.dt_freesurf
    alpha = self.implicit_surf_press
    beta = self.implicit_div_2d_flow
    g = self.gravity

    # Read the per-rank cube context from the model object (set by the
    # multi-rank harness).  ctx=None -> single-rank path unchanged.
    ctx = getattr(model, "cube_ctx", None)
    # Geometry is memoized for this grid/context. CubeStaticState owns the
    # modules, physical constants and KPP static tensors prepared with it.

    static = resolve_cube_static_context(
        self, grid, ctx, state.temp.data.dtype, state.temp.data.device
    )
    # -- metrics / masks (face-aware) ------------------------------------
    # MIT initialises etaH during INITIALISE_VARIA via INTEGR_CONTINUITY on the
    # initial velocity field, before the first SOLVE_FOR_PRESSURE.  This matters
    # for non-zero velocity ICs when implicDiv2DFlow != 1.
    # The scatter of masks/metrics from global [6,...] to local
    # [n_local,...] is the static context's, built with the same ctx.
    entry = _prepare_cube_step_entry(
        self, state, grid, bundle, ctx=ctx, static=static
    )
    u = entry.u
    v = entry.v
    eta = entry.eta
    temp = entry.temp
    salt = entry.salt
    dtype = entry.dtype
    device = entry.device
    nz = entry.nz
    ny = entry.ny
    nx = entry.nx
    drF = entry.drF
    dyG = entry.dyG
    dxG = entry.dxG
    hFacW = entry.hFacW
    hFacS = entry.hFacS
    hFacC = entry.hFacC
    u_mask = entry.u_mask
    v_mask = entry.v_mask
    c_mask = entry.c_mask
    surf = entry.surf
    recip_hFacC = entry.recip_hFacC
    rA = entry.rA
    u_trans = entry.u_trans
    v_trans = entry.v_trans
    hdiv_vol = entry.hdiv_vol
    etaH_prev = entry.etaH_prev
    self.stream_trace.emit_stream_trace(
        "cube.entry",
        eta=eta,
        u=u,
        v=v,
        temp=temp,
        salt=salt,
        etaH_prev=etaH_prev,
    )
    if self.stream_trace.hook is not None:
        # The cube has no open boundary or freezing stage, so the entry state
        # is the bounded one; w is LOCOMgcm downward-positive, as the regional station.
        self.stream_trace.emit_stream_trace(
            "step.post_oceanic_physics",
            u=u,
            v=v,
            w=vertical_velocity_continuity(
                hdiv_vol, cell_area=rA.unsqueeze(1), form="velocity", convention="downward",
            ),
            temp=temp,
            salt=salt,
        )

    # ===================================================================
    # 0. KPP vertical mixing: diagnose diffusivities from the current state
    #    (pure diagnostic, no cross-step cache).  Consumed by the implicit
    #    vertical viscosity (predictor) and tracer vertical diffusion below.
    # ===================================================================
    # KPP gating reads the regular stepper field ``kpp_enabled``, the same
    # source as the regional path (yaml value in
    # model.stepper.kwargs.kpp_enabled).
    kpp = (
        _compute_cube_kpp_fields(
            self, temp, salt, u, v, grid, dtype=dtype, device=device, ctx=ctx,
        )
        if bool(getattr(self, "kpp_enabled", False))
        else None
    )
    # Guard: the surface heat /
    # shortwave / freshwater flux ingest lives ENTIRELY inside the KPP block of
    # ``_compute_cube_tracer_update``.  With KPP off, ``set_cube_surface_fluxes``
    # would set qnet/qsw/empmr yet they would be silently
    # dropped.  Raise loudly instead of computing a wrong forcing-free step.
    if kpp is None:
        _stale_fluxes = [k for k, value in (
            ("_cube_qnet", self._cube_surface_forcing.qnet),
            ("_cube_qsw", self._cube_surface_forcing.qsw),
            ("_cube_empmr", self._cube_surface_forcing.empmr),
        ) if value is not None]
        if _stale_fluxes:
            raise ValueError(
                f"cube surface heat/salt fluxes require KPP: _cube_kpp_enabled is off but "
                f"{_stale_fluxes} are set — the flux ingest path lives inside the KPP block "
                "and they would be silently ignored. "
                "Enable KPP or clear set_cube_surface_fluxes(None)."
            )

    # ===================================================================
    # 1. Momentum tendency (advection + Coriolis + viscosity + wind).
    # ===================================================================
    momentum_forcing = _compute_cube_momentum_forcing(
        self,
        state,
        grid,
        temp=temp,
        salt=salt,
        dtype=dtype,
        device=device,
        gravity=g,
        hdiv_vol=hdiv_vol,
        rA=rA,
        u_trans=u_trans,
        v_trans=v_trans,
        hFacW=hFacW,
        hFacS=hFacS,
        hFacC=hFacC,
        ctx=ctx,
    )
    ex_u = momentum_forcing.u
    ex_v = momentum_forcing.v

    u_star, v_star = self._predict_velocity(
        u,
        v,
        eta,
        ex_u,
        ex_v,
        grid,
        alpha=alpha,
        dt_mom=dt_mom,
        field_viscosity=kpp,
        dtype=dtype,
        device=device,
        hFacW=hFacW,
        hFacS=hFacS,
        u_mask=u_mask,
        v_mask=v_mask,
        ctx=ctx,
    )

    # ===================================================================
    # 3. Predicted transport divergence (face-aware nz reduction).
    # ===================================================================
    predicted = compute_predictor_divergence(
        u_star,
        v_star,
        grid,
        stream_trace=self.stream_trace,
        layer_thickness=drF,
        dyG=dyG,
        dxG=dxG,
        hFacW=hFacW,
        hFacS=hFacS,
        rA=rA,
        wet_mask=c_mask,
        ctx=ctx,
    )
    div_star = predicted.divergence                               # [n_local|6,ny,nx]

    # ===================================================================
    # 4. Helmholtz RHS + cg2d solve.  x0 = eta*surf is the
    #    cube's own initial-guess policy (dry cells forced to 0, unlike
    #    the regional raw/maskInC forms), and the two solver diagnostics are
    #    returned with the state.
    # ===================================================================
    # Pass and validate cg_tolerance_mode explicitly. A generic context manager
    # around the PCG exit prevents torch 2.9 from breaking the graph inside it.
    eta_new, cg2d_iters, cg2d_residual = self._solve_free_surface(
        eta,
        etaH_prev,
        div_star,
        grid,
        dt_mom=dt_mom,
        dt_freesurf=dt_freesurf,
        beta=beta,
        dtype=dtype,
        device=device,
        rA=rA,
        surface_mask=surf,
        x0_mask=surf,
        # The cube grid rejects nonhydrostatic=True at construction
        # (implicit_free_surface.py validation), so MITgcm's
        # ``oldFreeSurfTerm`` half can never arise here.  Declared
        # explicitly rather than defaulted (the explicit-input requirement).
        eta_source=etaH_prev,
        # The cube step has no EmPmR input for the cg2d source term, so a cube
        # run with the switch on raises in the segment instead of solving
        # without the term (the explicit-input requirement).
        use_real_freshwater_flux=self.use_real_freshwater_flux,
        empmr_mass_flux=None,
        ctx=ctx,
    )

    u_new, v_new = self._correct_surface_velocity(
        eta_new,
        u_star,
        v_star,
        grid,
        alpha=alpha,
        dt_mom=dt_mom,
        dtype=dtype,
        device=device,
        u_mask=u_mask,
        v_mask=v_mask,
        ctx=ctx,
    )
    self.stream_trace.emit_stream_trace(
        "step.post_momentum_correction",
        u=u_new,
        v=v_new,
        eta=eta_new,
    )

    # ===================================================================
    # 6. exactConserv: recompute η from the corrected transport divergence.
    # ===================================================================
    exact = _apply_cube_exact_conserv_update(
        self,
        eta_new,
        etaH_prev,
        u_new,
        v_new,
        grid,
        dt_freesurf=dt_freesurf,
        beta=beta,
        drF=drF,
        dyG=dyG,
        dxG=dxG,
        hFacW=hFacW,
        hFacS=hFacS,
        ctx=ctx,
        rA=rA,
        wet_mask=c_mask,
    )
    eta_final = exact.eta_final
    if self.stream_trace.hook is not None:
        # INTEGR_CONTINUITY reads uVel, vVel after the exchange that ends
        # MOMENTUM_CORRECTION_STEP (momentum_correction_step.F:128), i.e. with
        # the shared seam faces reconciled; w is LOCOMgcm downward-positive.
        from locomgcm.dynamics._vector_seam_topology import (
            reconcile_vector_redundant_edges_for_grid,
        )

        u_cont, v_cont = reconcile_vector_redundant_edges_for_grid(
            u_new, v_new, grid, ctx=ctx,
        )
        u_cont_trans, v_cont_trans = face_volume_transport(
            u_cont, v_cont, drF, dyG, dxG, hFacW=hFacW, hFacS=hFacS,
        )
        self.stream_trace.emit_stream_trace(
            "step.post_integr_continuity",
            w=vertical_velocity_continuity(
                horizontal_volume_divergence_from_transports(
                    u_cont_trans, v_cont_trans, wet_mask=c_mask,
                ),
                cell_area=rA.unsqueeze(1),
                form="velocity",
                convention="downward",
            ),
            eta=eta_final,
        )

    # ===================================================================
    # 7. Horizontal tracer advection (MIT volume-transport upwind form).
    #    staggerTimeStep=T: tracers see corrected u^{n+1}.
    #    staggerTimeStep=F: tracers see current-step u^n (synchronous).
    # ===================================================================
    tracer_update = _compute_cube_tracer_update(
        self,
        temp,
        salt,
        u,
        v,
        u_new,
        v_new,
        grid,
        dt_tracer=dt_tracer,
        drF=drF,
        dyG=dyG,
        dxG=dxG,
        hFacW=hFacW,
        hFacS=hFacS,
        hFacC=hFacC,
        c_mask=c_mask,
        recip_hFacC=recip_hFacC,
        nz=nz,
        ctx=ctx,
        kpp=kpp,
    )
    temp_new, salt_new = self._hold_unstepped_tracers(
        bounded=state, temp_new=tracer_update.temp, salt_new=tracer_update.salt,
    )
    if self.stream_trace.hook is not None:
        # Published after the hold: MITgcm never integrates an unstepped tracer
        # (thermodynamics.F:317-337), so neither stage station shows an update.
        explicit_temp, explicit_salt = self._hold_unstepped_tracers(
            bounded=state,
            temp_new=tracer_update.explicit_temp,
            salt_new=tracer_update.explicit_salt,
        )
        self.stream_trace.emit_stream_trace(
            "tracer.after_explicit", temp=explicit_temp, salt=explicit_salt,
        )
        self.stream_trace.emit_stream_trace(
            "tracer.after_implicit", temp=temp_new, salt=salt_new,
        )

    # ===================================================================
    # 8. Assemble the updated state.
    # ===================================================================
    result = _assemble_cube_step_result(
        self,
        state,
        grid,
        eta_final=eta_final,
        u_new=u_new,
        v_new=v_new,
        temp_new=temp_new,
        salt_new=salt_new,
        u_mask=u_mask,
        v_mask=v_mask,
        ctx=ctx,
    )
    return result.state, cg2d_iters, cg2d_residual
