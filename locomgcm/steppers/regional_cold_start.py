"""Warm start of the regional implicit free-surface step, done in prepare.

The compiled regional step traces on its first call.  What that call finds
different from the calls after it turns into a guard that fails later (a
retrace) or into trace pieces the step keeps (graph breaks): histories first
published during the step, a monitoring budget that changes type, constants
the step builds on first use. ``ImplicitFreeSurfaceStepper.prepare_step_host``
resolves these before the first call, eagerly.

It works on what the stepper passes in: the stepper object (its explicit histories,
flags, and three methods of the step body:
``_nonlin_free_surf_active``, ``_evaluation_inputs`` and
``_evaluate_slow_tendency_phase``), the model and the state.  It does not
import the stepper module; the cube counterpart is
``implicit_free_surface_cube.seed_cube_cross_step_state``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from .ifs.freshwater import ContinuityHistory
from typing import Any

import torch

from locomgcm.accel.tensor_context import caller_tensor_mode
from locomgcm.distributed.comm_adjoint import constant_transport
from locomgcm.dynamics import AssemblyPolicy
from locomgcm.state import PrognosticState
from locomgcm.sampling import sampling_rule
from .ifs.kpp_inputs import kpp_eos_module, seed_kpp_forcing_inputs

#: Fields produced by each regional KPP forcing source. Only these fields
#: receive NaN placeholders before the compiled step consumes the record.
KPP_HARVEST_FIELDS: dict[str, tuple[str, ...]] = {
    "era5_wind_forcing": ("surf_forc_u", "surf_forc_v"),
    "era5_bulk_forcing": ("surf_forc_t", "surf_qsw", "penetrating_sw", "surf_forc_s"),
    "wpflux_surface_forcing": ("surf_forc_t", "surf_qsw", "penetrating_sw", "surf_forc_s"),
    "surface_heat_salt_flux": ("penetrating_sw", "surf_forc_t", "surf_forc_s"),
}
#: The field holding rhoConst on each KPP forcing source.
KPP_SOURCE_RHO_CONST: dict[str, str] = {
    "era5_wind_forcing": "reference_density",
    "era5_bulk_forcing": "rho_const",
    "wpflux_surface_forcing": "rho_const",
    "surface_heat_salt_flux": "reference_density",
}
#: Modules that put a surface flux or stress into a regional model.  With none
#: of them MITgcm's surfaceForcingU/V/T/S stay zero (ini_ffields.F:45-48,
#: external_forcing_surf.F:144-145,222-233) and KPP reads those zeros;
#: ``surface_climate_relaxation`` is added onto those fields by the harvest.
SURFACE_FORCING_MODULES = (
    "era5_bulk_forcing", "era5_wind_forcing", "surface_wind_stress",
    "surface_heat_salt_flux", "wpflux_surface_forcing",
)


def _constant_prewarm_applies(stepper: Any, model: Any) -> bool:
    """Whether the constant prewarm applies to this run.

    Not for a model whose histories live in an explicit carry
    (``_ocean_carry``) or a nonlinear free surface, whose step rewrites the
    geometry before its first read.  Such a run builds its constants in the
    first traced step and keeps its second-step retrace, which traces on warm
    constants; a run continued from a pickup then traces the cold variant again
    at its first step, another graph partition than the donor's later steps.
    """
    if getattr(model, "_ocean_carry", None) is not None:
        return False
    if stepper._nonlin_free_surf_active(model.grid):
        # ponytail: the nonlinear free surface also adds _hFacSurf and _h0Fac*
        # in its first step and rescales the geometry before the first
        # enforce; warming it needs that time-layer block in prepare.
        return False
    if stepper._package_modules.seaice is not None:
        # The sea-ice module's tracer forcing exists only after the step's
        # own ice advance (do_oceanic_phys.F:376-474), so no tracer evaluation
        # can run before the first step.
        return False
    return True


def _warm_start_applies(stepper: Any, model: Any) -> bool:
    """Whether the cross-step seed applies: the constant prewarm's runs except
    a Stevens boundary, whose step runs on a copy of the histories that only an
    accepted step publishes.  The prewarm itself publishes no history, so a
    Stevens run warms its constants all the same."""
    from locomgcm.step.boundary_history import boundary_adapter

    if boundary_adapter(model) is not None:
        # ponytail: ``model.step_kernel`` hands a Stevens-boundary step a copy
        # of its histories and publishes it only when finalize accepts the step,
        # so a restart key seeded in prepare would outlive a rejected step.
        # Seeding the copy is the way to lift it.
        return False
    return _constant_prewarm_applies(stepper, model)


def _hold_sponge_budget_as_tensors(model: Any, state: PrognosticState) -> None:
    """Hold the tracer sponge budget integrals as 0-d tensors.

    The step body accumulates them (``model.accumulate_tendency_diagnostics``)
    inside the traced region.  They start, and restart after
    ``reset_runtime_budget``, as the float 0.0.  The same value as a tensor of
    the rate's dtype and device (``tracer_sponge.py`` computes the rate in the
    state's) is the type every later step sees; ``tensor(0.0) + x`` and
    ``0.0 + x`` round alike.  Without a tracer sponge the rates are absent, the
    integrals stay 0.0 and the guard holds, so the budget is left alone.
    """
    budget = getattr(model, "boundary_budget", None)
    if budget is None or not any(
        getattr(module, "name", None) == "tracer_boundary_sponge"
        for module in getattr(model, "dynamics_modules", ())
    ):
        return
    for name in ("temp_sponge_integral", "salt_sponge_integral"):
        value = getattr(budget, name, None)
        if isinstance(value, float):
            setattr(budget, name, torch.tensor(
                value, dtype=state.temp.data.dtype, device=state.temp.data.device,
            ))


def prepare_regional_kpp_background(stepper, grid, dtype, device):
    """Prime the grid's scalar diffusivities for a single haloed tracer column block."""
    if grid.face_count != 1 or not getattr(stepper, "kpp_enabled", False):
        return
    # IFS binds its configured background as a scalar for both S and T.
    background = float(stepper.tracer_vertical_diffusivity_kr)
    from locomgcm.dynamics.kpp_mixing import _OL, _background_diffusivity_fields

    sample = torch.empty((), dtype=dtype, device=device).expand(
        grid.nz, grid.ny + 2 * _OL, grid.nx + 2 * _OL,
    )
    _background_diffusivity_fields(grid, stepper._kpp_static_inputs.static, sample,
                                   background, background)


def _prewarm_step_constants(stepper: Any, model: Any, state: PrognosticState) -> None:
    """Build the constants the regional step body builds on first use.

    Every cold constant build goes through ``accel.constant_cache._build_eager``,
    which Dynamo does not trace; a first traced call that meets one breaks that
    frame and every frame above it, for the rest of the run.  The constants
    are keyed by grid, dtype, device and shape, never by the state's values, so
    running once the code that looks them up builds the entries the traced step
    then finds.  On the h160 fixture three places build them: the slow
    tendency evaluation, run
    through the step's own dispatch with the step's own evaluation inputs, the
    KPP background diffusivity fields, and the metric bundle of the implicit
    vertical advection.  Results are discarded, under ``no_grad`` and the
    ``force_eager`` stance (compiled tendency methods, the runner's
    ``tendency`` target, are not traced here); nothing is written to the grid.

    Its halo exchanges go to the inference transport
    (``comm_adjoint.constant_transport()``, as cold constant builds do).  On a
    grid bound to a ``CommSession`` they otherwise entered the first
    checkpoint segment's recorded plan, and that segment's recompute, where
    this runs no more, replayed a different plan: ``CommSessionError`` on two
    gloo ranks.
    """
    grid = model.grid
    dtype, device = state.u.data.dtype, state.u.data.device
    done = getattr(stepper, "_prewarmed_constants_for", None)
    if done is not None and done[0] is grid and done[1] == dtype and done[2] == device:
        return
    if (
        not _constant_prewarm_applies(stepper, model)
        or grid.face_count == 6
    ):
        # The cube resolves its constants in ``resolve_cube_static_context``.
        return
    # prepare_step_host has resolved 3-D Coriolis and the hydrostatic pressure
    # seed. The shared input installer supplies both, plus the initial/lagged
    # vertical velocity, for NH just as it does in the real tendency phase.
    assembly_policy = stepper._resolved_assembly_policy or AssemblyPolicy.regional(stepper)
    with (
        _preserve_sampled_inputs(model), torch.no_grad(),
        torch.compiler.set_stance("force_eager"), constant_transport(),
    ):
        inputs = stepper._evaluation_inputs(model, state, grid, None, keep=False)
        stepper._evaluate_slow_tendency_phase(
            model, state, kpp=None, assembly_policy=assembly_policy, inputs=inputs,
        )
        if stepper.stagger_tracer_stepping:
            # The momentum-only prewarm above does not visit tracer constants
            # such as global wet area. Resolve the existing tracer dispatch as
            # well so its first real evaluation does not build a memo in-graph.
            split_ab2 = (stepper.ab2_on_temperature_tendency
                         or stepper.ab2_on_salinity_tendency)
            implicit_tracers = tuple(name for name, enabled in (
                ("temp", stepper.temperature_implicit_vertical_advection),
                ("salt", stepper.salinity_implicit_vertical_advection),
            ) if enabled)
            if implicit_tracers:
                from .ifs import tendency as ifs_tendency

                tracer_sources = stepper._tracer_tendency_sources(model)
                if split_ab2:
                    tracer_ab_modules, tracer_post_modules = AssemblyPolicy.split_tracers(
                        tracer_sources.modules)
                else:
                    tracer_ab_modules, tracer_post_modules = tracer_sources.modules, ()
                ifs_tendency._evaluate_tracer_tendency_for_implicit_vertical_advection(
                    state,
                    sources=tracer_sources,
                    implicit_tracer_names=implicit_tracers,
                    ab2_modules=tracer_ab_modules,
                    post_modules=tracer_post_modules,
                    inputs=inputs,
                )
                del tracer_sources, tracer_ab_modules, tracer_post_modules
            elif split_ab2:
                model.evaluate_tracer_tendency_split_ab2(state, inputs=inputs)
            else:
                model.evaluate_tracer_tendency_with_diagnostics(state, inputs=inputs)
        if stepper._kpp_static_inputs.static is not None:
            prepare_regional_kpp_background(stepper, grid, dtype, device)
        if (
            stepper.temperature_implicit_vertical_advection
            or stepper.salinity_implicit_vertical_advection
            or stepper.momentum_implicit_vertical_advection
        ):
            from locomgcm.geometry.metric_bundle import build_vertshear_metric_bundle

            build_vertshear_metric_bundle(grid, dtype, device)
    object.__setattr__(stepper, "_prewarmed_constants_for", (grid, dtype, device))


@contextmanager
def _preserve_sampled_inputs(model):
    """Constant discovery must not publish its no-grad forcing products.

    The runner already sampled START before prepare. A cold tendency pass can
    populate derived sampled fields (notably wind face stresses); leaving those
    installed changes the first plan from None to Tensor and can retain a
    detached value for an active forcing control. Restore the declared inputs
    and producer-owned transient caches so every real step computes its own
    products in its caller's grad mode.
    """
    saved = [(obj, {name: getattr(obj, name) for name in (
        sampling_rule(obj).inputs + getattr(obj, "prewarm_cache_fields", ())
    )})
             for obj in model._time_state_objects()]
    try:
        yield
    finally:
        for obj, attributes in saved:
            for name, value in attributes.items():
                object.__setattr__(obj, name, value)


def _seed_cross_step_state(
    stepper: Any, model: Any, state: PrognosticState, *, dt_freesurf: float,
) -> None:
    """Initialize regional histories and KPP forcing inputs before the first call.

    Present histories are left alone (a restart import published its own).
    Each owner receives the value the step itself would use:

    * ``_etaH`` and ``_w_interface_prev``: the step's absent-key expressions
      (eta plus the explicit initial divergence, and ``-w``) on
      the state the step receives, bounded under the clock just sampled, which
      is the clock of the step's first ``enforce_boundaries``.
    * ``_ab2_prev_tendency``: zeros with false validity flags, which the AB2
      branch selects away with ``torch.where``; the result is the
      forward-Euler step the absent-key branch took.
    * KPP input fields: NaN placeholders expose missing producer fields when
      the step collects its local forcing record.

    Prepare runs this on the state the first step then receives, in the grad
    mode of that step: ``_etaH`` carries the graph of the state it was bounded
    from.  Every caller does (``model.step``, the runner, ``step/carry.py``,
    ``checkpointed_rollout``), and a restore pops these keys before the next
    prepare.
    """
    if not _warm_start_applies(stepper, model):
        return
    grid = model.grid
    eta_h = stepper._continuity.eta_h
    w_interface_prev = stepper._continuity.w_interface_prev
    if eta_h is None or w_interface_prev is None:
        from .ifs.prepare import diagnose_mit_interface_velocity

        bounded = model.enforce_boundaries(state, record_budget=False)
        if eta_h is None:
            eta_h = stepper._initial_eta_h(bounded, grid, dt_freesurf)
            # When the boundaries leave eta unchanged, ``bounded.eta`` is the
            # step's own eta input; two names on one storage would make the
            # first step's inputs aliases (a separate graph), so copy then.
            if eta_h.untyped_storage().data_ptr() == state.eta.data.untyped_storage().data_ptr():
                eta_h = eta_h.clone()
        if w_interface_prev is None:
            w_up = diagnose_mit_interface_velocity(state=bounded, grid=grid)
            # The step's ``_carry``: the tape is kept under grad mode only.
            w_interface_prev = w_up if torch.is_grad_enabled() else w_up.detach()
        object.__setattr__(stepper, "_continuity", ContinuityHistory(eta_h, w_interface_prev))
    if stepper.ab_eps > 0.0 and stepper._momentum_ab is None:
        # A later history is the previous step's tendency, which under grad
        # mode requires grad exactly when a state field does (every floating
        # input of a ``checkpointed_rollout`` segment head does); the zeros take
        # the same flag so every head meets one guard.  The gradient they
        # receive is zero and stays on them.
        requires_grad = torch.is_grad_enabled() and any(
            getattr(state, name).data.requires_grad
            for name in ("eta", "u", "v", "temp", "salt")
        )
        device = state.u.data.device
        # One flag tensor per entry: a later history has separate copies
        # (``persist_step_outputs``), and shared ones would alias.
        history = {
            "u": torch.zeros_like(state.u.data, requires_grad=requires_grad),
            "v": torch.zeros_like(state.v.data, requires_grad=requires_grad),
            "mom_valid": torch.zeros((), dtype=torch.bool, device=device),
        }
        for name, enabled in (
            ("temp", stepper.ab2_on_temperature_tendency),
            ("salt", stepper.ab2_on_salinity_tendency),
        ):
            if enabled:
                history[name] = torch.zeros_like(
                    getattr(state, name).data, requires_grad=requires_grad
                )
                history[name + "_valid"] = torch.zeros((), dtype=torch.bool, device=device)
        from .ifs.tendency import MomentumABHistory

        object.__setattr__(stepper, "_momentum_ab", MomentumABHistory(**history))
    # Reserve the remaining regional slots before the first compiled write.
    # None keeps the existing first-step branches and their gradients intact;
    # after that step the same keys hold the actual published histories. Do
    # not seed numerical zeros in place of a missing AB or NH history.
    if (stepper.stagger_tracer_stepping and stepper.ab_eps > 0.0
            and (stepper.ab2_on_temperature_tendency
                 or stepper.ab2_on_salinity_tendency)
            and hasattr(model, "evaluate_tracer_tendency_split_ab2")):
        if stepper._tracer_ab is None:
            from .ifs.tracer_stage import TracerABHistory

            object.__setattr__(stepper, "_tracer_ab", TracerABHistory(None, None, True))
    if (stepper.nonhydrostatic.nonhydrostatic
            and stepper.external_pnh_provider is None):
        object.__setattr__(stepper, "_nh_history", replace(
            stepper._nh_history,
            phi_nh_present=True, gw_nm1_present=True, w_vel_present=True,
        ))
    if stepper.reads_kpp_forcing:
        # Match the kernel's tensor mode: a no-grad run steps under inference
        # mode while prepare runs ordinary (``build_step_plan``).
        with caller_tensor_mode():
            object.__setattr__(stepper, "_kpp_forcing_inputs", _seed_kpp_forcing_inputs(stepper, model, state))


def reseed_regional_kpp_forcing_inputs(model: Any, state: PrognosticState):
    """Rebuild a compiled regional restart's KPP placeholders in kernel mode."""
    from locomgcm.step.boundary_history import boundary_adapter

    stepper = model.stepper
    if (getattr(model.grid, "face_count", 1) == 1
            and hasattr(stepper, "OWNED_STEP_FIELDS")
            and boundary_adapter(model) is None
            and getattr(model, "_ocean_carry", None) is None
            and stepper.reads_kpp_forcing):
        with caller_tensor_mode():
            forcing_inputs = _seed_kpp_forcing_inputs(stepper, model, state)
            object.__setattr__(stepper, "_kpp_forcing_inputs", forcing_inputs)
        return forcing_inputs
    return None


def _seed_kpp_forcing_inputs(stepper: Any, model: Any, state: PrognosticState):
    """Reserve fields from the same producer choice as the regional harvest.

    Missing producers retain their explicit input fields. NaN placeholders
    expose any selected producer that fails to supply its fields.  A model
    without any ``SURFACE_FORCING_MODULES`` gets zero fields and the EOS
    module's reference density, the rhoConst the translator writes there.
    """
    grid = model.grid
    fields: list[str] = []
    rho_const = None
    penetrating_sw = stepper.penetrating_sw
    for name, producer, _, _ in stepper._kpp_forcing_producers():
        if name != "era5_wind_forcing":
            produces = name != "era5_bulk_forcing" or (
                hasattr(producer, "compute_surface_fluxes")
                or getattr(producer, "_current_forcing", None) is not None
            )
            if not produces:
                continue
        fields.extend(KPP_HARVEST_FIELDS[name])
        if name == "era5_wind_forcing" or rho_const is None:
            rho_const = float(getattr(producer, KPP_SOURCE_RHO_CONST[name]))
        if name != "era5_wind_forcing":
            penetrating_sw = (
                False if name == "surface_heat_salt_flux"
                else bool(getattr(producer, "penetrating_sw", stepper.penetrating_sw))
            )
    fill = float("nan")
    if not fields:
        modules = getattr(model, "dynamics_modules", ()) or ()
        if any(getattr(m, "name", "") in SURFACE_FORCING_MODULES for m in modules):
            return stepper._kpp_forcing_inputs
        fields = ["surf_forc_u", "surf_forc_v", "surf_forc_t", "surf_forc_s"]
        fill = 0.0
        rho_const = float(kpp_eos_module(model).reference_density)
    if stepper._kpp_static_inputs.rho_const is None:
        object.__setattr__(stepper, "_kpp_static_inputs", replace(
            stepper._kpp_static_inputs, rho_const=rho_const,
        ))
    return seed_kpp_forcing_inputs(
        stepper._kpp_forcing_inputs, fields=tuple(fields), fill=fill,
        penetrating_sw=penetrating_sw, template=state.temp.data, ny=grid.ny, nx=grid.nx,
    )
