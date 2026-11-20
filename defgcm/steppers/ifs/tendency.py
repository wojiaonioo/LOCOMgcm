"""Tendency assembly, GAD bucket placement and momentum AB histories.

Evaluators use regional C-grid centres and U/V faces. AB extrapolation,
raw-history retention and tracer-memory surface rescaling use the caller's
time levels and dtype. Histories also retain face-first cube layouts.
EvaluationInputs supplies MIT upward-positive lagged interface w.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Sequence

import torch

from defgcm.dynamics import AssemblyPolicy
from defgcm.dynamics.assembly_policy import AssemblyBucket
from defgcm.dynamics.base import TendencyPhase
from defgcm.dynamics.kpp_mixing import KppNonlocalTransport
from defgcm.state import DiagnosticState, EvaluationInputs, PrognosticState, StateTendency
from defgcm.steppers.implicit_free_surface_nonlin import rescale_tracer_tendency_freesurf
from .r_star import r_star_rescale_tendency
from .tracer_stage import TracerABHistory

from .admission import _composition_module_name


@dataclass(frozen=True)
class TendencySources:
    grid: Any
    modules: tuple[Any, ...]
    populate_diagnostics: Callable
    enforce_tendency: Callable


@dataclass(frozen=True)
class MomentumABHistory:
    u: torch.Tensor
    v: torch.Tensor
    temp: torch.Tensor | None = None
    salt: torch.Tensor | None = None
    mom_valid: torch.Tensor | None = None
    temp_valid: torch.Tensor | None = None
    salt_valid: torch.Tensor | None = None
    valid: torch.Tensor | None = None

    def with_velocities(self, u: torch.Tensor, v: torch.Tensor) -> MomentumABHistory:
        return replace(self, u=u, v=v)


def encode_momentum_ab_history(history: MomentumABHistory | None):
    if history is None:
        return {}
    return {"_ab2_prev_tendency": {
        item.name: getattr(history, item.name)
        for item in fields(MomentumABHistory)
        if getattr(history, item.name) is not None}}


def decode_momentum_ab_history(payload):
    if "_ab2_prev_tendency" not in payload:
        return None
    return MomentumABHistory(**payload["_ab2_prev_tendency"])


def ab2_factor(ab_eps):
    return 0.5 + ab_eps


def extrapolate_ab2(current, previous, *, factor):
    return current + factor * (current - previous)


def select_valid_ab2(*, valid, extrapolated, current):
    return torch.where(valid, extrapolated, current)


def extrapolate_valid_ab2(*, valid, current, previous, factor):
    return select_valid_ab2(
        valid=valid,
        extrapolated=extrapolate_ab2(current, previous, factor=factor),
        current=current,
    )


def assemble_momentum_ab_history(*, cache_dict, device):
    # The seeded flags, true from the first completed step on (names
    # as in the explicit carry's restart payload).
    flags = ["mom_valid"] + [n + "_valid" for n in ("temp", "salt") if n in cache_dict]
    for flag in flags:
        cache_dict[flag] = torch.ones((), dtype=torch.bool, device=device)
    return MomentumABHistory(**cache_dict)


def assemble_carry_momentum_ab_history(*, u, v, mom_valid):
    return {"mom_u": u, "mom_v": v, "mom_valid": torch.ones_like(mom_valid)}


def assemble_carry_tracer_ab_history(*, name, tendency, valid):
    return {name: tendency, name + "_valid": torch.ones_like(valid)}


def retain_momentum_ab(tend_u, tend_v) -> tuple:
    return tend_u.clone(), tend_v.clone()


def rescale_tracer_ab_surface(history: MomentumABHistory | TracerABHistory, *, hfac_c_now, surface_hfac_c):
    memory = {}
    for name in ("temp", "salt"):
        value = getattr(history, name)
        if value is not None:
            memory[name] = rescale_tracer_tendency_freesurf(
                value, hFacC_now=hfac_c_now, hFac_surfC=surface_hfac_c,
            )
    return replace(history, temp=memory.get("temp"), salt=memory.get("salt"))


def rescale_tracer_ab_r_star(history: MomentumABHistory | TracerABHistory, *, exp_c):
    """FREESURF_RESCALE_G r* form on the tracer AB memory (temp_integrate.F, the gtNm1 call)."""
    memory = {}
    for name in ("temp", "salt"):
        value = getattr(history, name)
        if value is not None:
            memory[name] = r_star_rescale_tendency(value, exp_c)
    return replace(history, temp=memory.get("temp"), salt=memory.get("salt"))


def retain_cube_momentum_ab(tend_u, tend_v):
    u, v = retain_momentum_ab(tend_u, tend_v)
    return MomentumABHistory(
        u, v,
        valid=torch.ones((), dtype=torch.bool, device=tend_u.device),
    )


def _evaluate_momentum_module_for_implicit_vertical_advection(
    module: Any,
    name: str,
    state: PrognosticState,
    diagnostics: DiagnosticState,
    grid: Any,
    kpp: Any,
) -> StateTendency:
    """Evaluate one momentum module with enabled-path physical hand-offs."""
    if name == "bottom_drag" and bool(getattr(module, "no_slip_bottom", True)):
        evaluator = getattr(module, "evaluate_with_total_vertical_viscosity", None)
        if not callable(evaluator):
            raise TypeError(
                "bottom_drag lacks evaluate_with_total_vertical_viscosity for "
                "momentum implicit vertical advection"
            )
        background = float(getattr(module, "vertical_viscosity", 0.0))
        if kpp is None:
            total_u = torch.full_like(state.u.data, background)
            total_v = torch.full_like(state.v.data, background)
        else:
            total_u = getattr(kpp, "kappa_ru", None)
            total_v = getattr(kpp, "kappa_rv", None)
            if not isinstance(total_u, torch.Tensor) or not isinstance(
                total_v, torch.Tensor
            ):
                raise TypeError(
                    "KPP momentum fields must expose tensor kappa_ru/kappa_rv"
                )
        return evaluator(
            state,
            diagnostics,
            grid,
            total_vertical_viscosity_u=total_u,
            total_vertical_viscosity_v=total_v,
        )
    if name in {"era5_wind_forcing", "surface_wind_stress"}:
        evaluator = getattr(module, "evaluate_local_surface", None)
        if not callable(evaluator):
            raise TypeError(
                f"{name} lacks evaluate_local_surface for momentum implicit "
                "vertical advection"
            )
        return evaluator(state, diagnostics, grid)
    return module.evaluate(state, diagnostics, grid)


def _evaluate_momentum_tendency_for_implicit_vertical_advection(
    state: PrognosticState,
    *,
    sources: TendencySources,
    kpp: Any,
    assembly_policy: AssemblyPolicy,
    inputs: EvaluationInputs | None = None,
) -> tuple[StateTendency, StateTendency | None, DiagnosticState]:
    """Mirror the model AB split while injecting no-slip/wind hand-offs."""
    grid = sources.grid
    state.validate_against(grid)
    modules = sources.modules
    diagnostics = DiagnosticState(inputs=inputs)
    ab2_modules, post_modules = assembly_policy.split(modules)
    sources.populate_diagnostics(modules, state, diagnostics)

    def evaluate(modules_to_evaluate: Sequence[Any]) -> StateTendency:
        accumulator = StateTendency.accumulator(state)
        for module in modules_to_evaluate:
            module_tendency = _evaluate_momentum_module_for_implicit_vertical_advection(
                module, _composition_module_name(module), state, diagnostics, grid, kpp,
            )
            accumulator.add(
                module.select_tendency_phase(module_tendency, TendencyPhase.MOMENTUM)
            )
        return sources.enforce_tendency(
            accumulator.finish(), grid
        )

    ab2_tendency = evaluate(ab2_modules)
    post_tendency = evaluate(post_modules) if post_modules else None
    return ab2_tendency, post_tendency, diagnostics


def _add_kpp_nonlocal_to_tracer_bucket(
    ab2_tendency, *, state, kpp, temp_diffusivity, salt_diffusivity, layer_thickness, grid, tracer_names,
):
    """Use the shared GAD placement for both regional tracer stepping paths."""
    module = KppNonlocalTransport(kpp, layer_thickness, tracer_names, temp_diffusivity, salt_diffusivity)
    if AssemblyPolicy.tracer_bucket_of(module) != AssemblyBucket.AB2:
        raise ValueError("KPP nonlocal transport must belong to the GAD AB bucket")
    nonlocal_tendency = module.evaluate(state, DiagnosticState(), grid)
    return StateTendency(
        eta=ab2_tendency.eta, u=ab2_tendency.u, v=ab2_tendency.v,
        temp=(ab2_tendency.temp + nonlocal_tendency.temp
              if "temp" in tracer_names else ab2_tendency.temp),
        salt=(ab2_tendency.salt + nonlocal_tendency.salt
              if "salt" in tracer_names else ab2_tendency.salt),
    ), nonlocal_tendency


def _evaluate_tracer_tendency_for_implicit_vertical_advection(
    state: PrognosticState,
    *,
    sources: TendencySources,
    implicit_tracer_names: tuple[str, ...],
    ab2_modules: tuple[Any, ...],
    post_modules: tuple[Any, ...],
    inputs: EvaluationInputs | None = None,
) -> tuple[StateTendency, StateTendency | None, DiagnosticState]:
    """Evaluate tracer modules with per-name horizontal-only advection seams."""
    grid = sources.grid
    state.validate_against(grid)
    modules = sources.modules
    diagnostics = DiagnosticState(inputs=inputs)
    sources.populate_diagnostics(modules, state, diagnostics)

    def evaluate(modules_to_evaluate: Sequence[Any]) -> StateTendency:
        accumulator = StateTendency.accumulator(state)
        for module in modules_to_evaluate:
            if _composition_module_name(module) == "horizontal_tracer_advection":
                module_tendency = module.evaluate_for_implicit_vertical_advection(
                    state,
                    diagnostics,
                    grid,
                    implicit_tracer_names=implicit_tracer_names,
                )
            else:
                module_tendency = module.evaluate(state, diagnostics, grid)
            accumulator.add(module.select_tendency_phase(module_tendency, TendencyPhase.TRACER))
        return sources.enforce_tendency(
            accumulator.finish(), grid
        )

    ab2_tendency = evaluate(ab2_modules)
    post_tendency = evaluate(post_modules) if post_modules else None
    return ab2_tendency, post_tendency, diagnostics


def add_post_ab2(tend_u, tend_v, tend_temp, tend_salt, post_ab2_tendency) -> tuple:
    # 1b'. Add post-AB2 tendencies (forcing/sponge; HPG in staggered/IIGW regime).
    #      MITgcm timestep.F:208-226 (post-AB momentum forcing/dissipation),
    #      timestep.F:320-325 (HPG added after AB when staggerTimeStep
    #      OR implicitIntGravWave).  In synchronous+non-IIGW mode the HPG
    #      policy places the HPG module in AB2, so it does not appear in
    #      ``post_ab2_tendency``.
    tend_u = tend_u + post_ab2_tendency.u
    tend_v = tend_v + post_ab2_tendency.v
    tend_temp = tend_temp + post_ab2_tendency.temp
    tend_salt = tend_salt + post_ab2_tendency.salt

    return (tend_u, tend_v, tend_temp, tend_salt)


def momentum_w_from_evaluation(evaluation_inputs):
    from defgcm.solvers import MomentumWNmHalfMIT

    # The momentum solve reads the lagged w the tendencies read: one
    # stored tensor (MIT has one wVel).  It reads it through a view
    # taken here, before the tendency evaluation, so the reverse pass
    # sums the solve's contributions on the view and adds them to w as
    # one term after every tendency contribution: T + P, the grouping
    # the two stored copies (tendency w and momentum wrapper, each
    # summed apart) gave.  Without the view the solve, later in the
    # step, would be added first: ((P + t_n) + ...) + t_1.
    return MomentumWNmHalfMIT(
        data=evaluation_inputs.w_interface.view_as(evaluation_inputs.w_interface))


def rescale_kpp_nonlocal(kpp_nonlocal, *, tracer_names, hfac_c_now, surface_hfac_c):
    # Keep geometry out of a comprehension closure in the time program.
    scaled_nonlocal = {
        name: rescale_tracer_tendency_freesurf(
            getattr(kpp_nonlocal, name),
            hFacC_now=hfac_c_now, hFac_surfC=surface_hfac_c,
        )
        for name in tracer_names
    }
    return replace(kpp_nonlocal, **scaled_nonlocal)


def clone_tracer_ab(tendency):
    return tendency.clone()


def add_tendency_term(tendency, extra):
    return tendency + extra
