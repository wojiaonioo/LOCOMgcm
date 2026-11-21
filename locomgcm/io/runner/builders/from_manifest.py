"""Build LOCOMgcm runtime models from yaml model manifests."""

from __future__ import annotations

from locomgcm.sampling import sampling_components

from collections.abc import Callable, Mapping
import inspect
import math
from typing import Any

import torch

import locomgcm.boundaries  # noqa: F401 - eager registration side effects
import locomgcm.dynamics  # noqa: F401 - eager registration side effects
import locomgcm.steppers  # noqa: F401 - eager registration side effects
from locomgcm import RegionalOceanConfig, TorchRegionalOceanModel
from locomgcm.boundaries._registry import BOUNDARY_CONDITIONS
from locomgcm.dynamics._registry import DYNAMICS_MODULES
from locomgcm.dynamics import AssemblyPolicy
from locomgcm.dynamics.base import MomentumTermClass, TendencyModule, TendencyPhase
from locomgcm.constants import ROTATION_PERIOD_EARTH
from locomgcm.dynamics.equation_of_state import EOS_TYPES
from locomgcm.dycore_parameters import NonhydrostaticParams
from locomgcm.geometry import BoundaryType, StructuredCGrid
from locomgcm.io.runner.config import RunConfig
from locomgcm.io.runner.config.vertical_diffusion import (
    assemble_vertical_diffusion, resolved_vertical_diffusion,
)
from locomgcm.operators import TracerFluxScheme
from locomgcm.operators.fluxes.tracer_flux import TRACER_SCHEME_BY_NAME
from locomgcm.state import PrognosticState
from locomgcm.steppers._registry import STEPPERS
from locomgcm.steppers.ifs.admission import (
    validate_implicit_vertical_advection_composition,
)


class ManifestBuildError(RuntimeError):
    """Raised when ModelManifest cannot be assembled into a model."""


def _available_names(registry: Any) -> list[str]:
    return sorted(registry.items().keys())


def _instantiate_dynamics(entry: Any, *, registry: Any = DYNAMICS_MODULES) -> TendencyModule:
    """Look up entry.name in DYNAMICS_MODULES; instantiate with entry.kwargs."""
    kwargs = dict(entry.kwargs)
    try:
        module_cls = registry.get(entry.name)
        side_channels = sorted(
            set(kwargs).intersection(AssemblyPolicy.placement_side_channels())
        )
        if side_channels and getattr(
            module_cls,
            "tendency_phase",
            TendencyPhase.MOMENTUM,
        ) != TendencyPhase.TRACER:
            term_class = module_cls.momentum_term_class
            declarations = ", ".join(
                f"{name}={kwargs[name]!r}" for name in side_channels
            )
            raise ManifestBuildError(
                f"dynamics module {entry.name!r} config declares {declarations}, "
                "which bypasses AssemblyPolicy while the policy owns placement "
                f"for MomentumTermClass.{term_class.name}. Remove the per-module "
                "placement field(s) and express the decision with "
                "model.stepper.kwargs AssemblyPolicy inputs."
            )
        return registry.instantiate(entry.name, **kwargs)
    except KeyError as exc:
        raise ManifestBuildError(
            f"dynamics module '{entry.name}' not registered. "
            f"Available: {_available_names(registry)}"
        ) from exc
    except TypeError as exc:
        raise ManifestBuildError(
            f"dynamics module '{entry.name}' kwargs incompatible: {exc}"
        ) from exc


def _forcing_provider(forcing_source: Any | None) -> Callable[[float], Any] | None:
    if forcing_source is None:
        return None
    if hasattr(forcing_source, "sample"):
        return forcing_source.sample
    if callable(forcing_source):
        return forcing_source
    raise ManifestBuildError(
        "forcing_source must be callable or expose a sample(time_seconds) method"
    )


def _normalize_nonhydrostatic_params(
    value: Any,
) -> NonhydrostaticParams | None:
    """Normalize a stepper NH value for canonical conflict decisions."""
    if value is None:
        return None
    if isinstance(value, NonhydrostaticParams):
        normalized = value
    elif isinstance(value, Mapping):
        try:
            normalized = NonhydrostaticParams(**value)
        except TypeError as exc:
            raise ManifestBuildError(
                "model.stepper.kwargs.nonhydrostatic could not normalize to "
                "NonhydrostaticParams"
            ) from exc
    else:
        normalized = NonhydrostaticParams(nonhydrostatic=value)
    if type(normalized.nonhydrostatic) is not bool:
        raise ManifestBuildError(
            "model.stepper.kwargs.nonhydrostatic must normalize to "
            "NonhydrostaticParams with a bool nonhydrostatic field, got "
            f"{type(normalized.nonhydrostatic).__name__}"
        )
    return normalized


def _instantiate_stepper(
    manifest_stepper: Any,
    *,
    nonhydrostatic_config: Any | None = None,
    implicit_viscosity: float | None = None,
    implicit_vertical_viscosity: bool | None = None,
    forcing_source: Any | None = None,
    rotation_period: float = ROTATION_PERIOD_EARTH,
) -> Any:
    if manifest_stepper.name is None:
        raise ManifestBuildError("model.stepper.name is required")
    kwargs = dict(manifest_stepper.kwargs)
    provider = _forcing_provider(forcing_source)
    try:
        stepper_cls = STEPPERS.get(manifest_stepper.name)
        configured = kwargs.get("nonhydrostatic")
        normalized_nonhydrostatic = (
            _normalize_nonhydrostatic_params(configured)
            if nonhydrostatic_config is not None
            else None
        )
        if (
            nonhydrostatic_config is not None
            and not nonhydrostatic_config.enabled
        ):
            if (
                normalized_nonhydrostatic is not None
                and normalized_nonhydrostatic.nonhydrostatic
            ):
                raise ManifestBuildError(
                    "nonhydrostatic dual-entry mismatch: "
                    f"physics.nonhydrostatic={nonhydrostatic_config!r} but "
                    "model.stepper.kwargs.nonhydrostatic="
                    f"{normalized_nonhydrostatic!r}"
                )
        if (
            nonhydrostatic_config is not None
            and nonhydrostatic_config.enabled
        ):
            nonhydrostatic = NonhydrostaticParams(
                nonhydrostatic=nonhydrostatic_config.enabled,
                cg3d_max_iters=nonhydrostatic_config.cg3d_max_iters,
                cg3d_target_residual=nonhydrostatic_config.cg3d_target_residual,
                cg3d_target_res_w_unit=(
                    nonhydrostatic_config.cg3d_target_res_w_unit
                ),
                nh_am2=nonhydrostatic_config.nh_am2,
                select_3d_cori_scheme=(
                    nonhydrostatic_config.select_3d_cori_scheme
                ),
                select_cori_map=nonhydrostatic_config.select_cori_map,
                f_prime=nonhydrostatic_config.f_prime,
                rotation_period=rotation_period,
            )
            if (
                normalized_nonhydrostatic is not None
                and normalized_nonhydrostatic != nonhydrostatic
            ):
                raise ManifestBuildError(
                    "nonhydrostatic dual-entry mismatch: "
                    f"physics.nonhydrostatic={nonhydrostatic_config!r} but "
                    "model.stepper.kwargs.nonhydrostatic="
                    f"{normalized_nonhydrostatic!r}"
                )
            kwargs["nonhydrostatic"] = nonhydrostatic
        if "rotation_period" in inspect.signature(stepper_cls.__init__).parameters:
            if "rotation_period" in kwargs:
                raise ManifestBuildError(
                    "model.stepper.kwargs.rotation_period: set physics.rotation_period, "
                    "the one rotation period the momentum, NonhydrostaticParams, KPP and "
                    "the $coriolis_* profiles read"
                )
            kwargs["rotation_period"] = rotation_period
        if implicit_viscosity is not None:
            # Momentum twin of the shared tracer assembly. MITgcm's viscAz/viscAr
            # (PARM01) reaches LOCOMgcm only as ``physics.implicit_viscosity``
            # -- experiments/mitcases/translate.py:1169-1170 maps both keywords
            # there and never emits a stepper kwarg -- but the stepper's
            # ``implicit_viscosity`` field (implicit_free_surface.py:872) had no
            # producer on this path, so every translated MIT case ran with
            # viscAr = 0.  That closes the KPP background floor
            # (``visc_arnr`` at kpp.py:882) and the KappaRU floor
            # (kpp_mixing.py:445-452), i.e. no implicit vertical viscosity at
            # all.  Fourth confirmed case of the single-source contract's
            # "assembly layer silently drops a switch LOCOMgcm implements"
            # class (the namelist producer-consumer contract).
            signature = inspect.signature(stepper_cls.__init__)
            if "implicit_viscosity" not in signature.parameters:
                if (
                    implicit_viscosity != 0.0
                    and implicit_vertical_viscosity is not False
                ):
                    raise ManifestBuildError(
                        f"physics.implicit_viscosity={implicit_viscosity!r} requires a "
                        "stepper with an implicit_viscosity consumer"
                    )
            else:
                configured = kwargs.get("implicit_viscosity")
                if configured is not None and not math.isclose(
                    float(configured),
                    float(implicit_viscosity),
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                ):
                    raise ManifestBuildError(
                        "implicit viscosity dual-entry mismatch: "
                        f"physics.implicit_viscosity={implicit_viscosity!r} but "
                        f"model.stepper.kwargs gives {configured!r}"
                    )
                kwargs["implicit_viscosity"] = float(implicit_viscosity)
        if implicit_vertical_viscosity is not None:
            signature = inspect.signature(stepper_cls.__init__)
            if "implicit_vertical_viscosity" not in signature.parameters:
                if (
                    implicit_vertical_viscosity
                    and float(implicit_viscosity or 0.0) > 0.0
                ):
                    raise ManifestBuildError(
                        "physics.implicit_vertical_viscosity=True requires a "
                        "stepper with an implicit_vertical_viscosity consumer"
                    )
            else:
                configured = kwargs.get("implicit_vertical_viscosity")
                if (
                    configured is not None
                    and configured != implicit_vertical_viscosity
                ):
                    raise ManifestBuildError(
                        "implicit vertical viscosity gate dual-entry mismatch: "
                        "physics.implicit_vertical_viscosity="
                        f"{implicit_vertical_viscosity!r} but model.stepper.kwargs "
                        f"gives {configured!r}"
                    )
                kwargs["implicit_vertical_viscosity"] = implicit_vertical_viscosity
        if provider is not None and "external_pnh_provider" not in kwargs:
            sig = inspect.signature(stepper_cls.__init__)
            if "external_pnh_provider" in sig.parameters:
                kwargs["external_pnh_provider"] = provider
        return STEPPERS.instantiate(
            manifest_stepper.name,
            **kwargs,
        )
    except KeyError as exc:
        raise ManifestBuildError(
            f"stepper '{manifest_stepper.name}' not registered. "
            f"Available: {_available_names(STEPPERS)}"
        ) from exc
    except TypeError as exc:
        raise ManifestBuildError(
            f"stepper '{manifest_stepper.name}' kwargs incompatible: {exc}"
        ) from exc


#: dynamics modules that diagnose the density the hydrostatic pressure gradient integrates
_EQUATION_OF_STATE_MODULES = tuple(dict.fromkeys(name for _type, name, _dynamic in EOS_TYPES))


_GM_RESIDUAL_FLOW_CONSUMERS = ("horizontal_tracer_advection", "vertical_tracer_transport")


def _bind_gmredi(dynamics: list[TendencyModule], grid: StructuredCGrid) -> None:
    """Give each gmredi module the model's equation of state; refuse the cube and ldd97's halo fCori gap.

    GM/Redi evaluates the model's equation of state at shifted reference levels
    (MITgcm do_oceanic_phys.F:759-823), so it reads the one EOS module of the
    manifest instead of a copy of its parameters.  On the cube grad_sigma.F:59-62,72-75
    fills the corner halos, which the gmredi package does not port.
    """
    from locomgcm.dynamics.gmredi.module import GMRedi

    gm_modules = [module for module in dynamics if isinstance(module, GMRedi)]
    # GM_AdvForm: every tracer advection consumer adds the bolus velocity
    # (thermodynamics.F:258-276), and none does without it.
    adv_form = any(module.adv_form for module in gm_modules)
    for module in dynamics:
        if (getattr(module, "name", None) in _GM_RESIDUAL_FLOW_CONSUMERS
                and bool(module.gm_residual_flow) != adv_form):
            raise ManifestBuildError(
                f"{module.name}.gm_residual_flow={module.gm_residual_flow} but the manifest "
                f"{'has' if adv_form else 'has no'} gmredi with adv_form; the tracer advection "
                "velocity is uVel+bolus exactly when GM_AdvForm is on (thermodynamics.F:258-276)"
            )
    if not gm_modules:
        return
    eos_modules = [m for m in dynamics if getattr(m, "name", None) in _EQUATION_OF_STATE_MODULES]
    if len(eos_modules) != 1:
        raise ManifestBuildError(
            f"gmredi needs exactly one equation-of-state module, got {len(eos_modules)}"
        )
    if BoundaryType.SEAM in (grid.boundary_topology.west_east, grid.boundary_topology.south_north):
        raise ManifestBuildError(
            "gmredi on the cubed sphere needs the corner fills of MITgcm grad_sigma.F:59-62,72-75"
        )
    for module in gm_modules:
        profile = module.coriolis_profile_c
        if profile is not None and grid.edge_face_couples("y") and len(set(profile)) > 1:
            # ldd97 reads fCori one row beyond the edge at the edge V faces
            # (gmredi_calc_tensor.F:192-210).  MITgcm's halo fCori comes from the halo
            # yC (ini_cori.F:55-100, exchanged only for selectCoriMap=3); a periodic y
            # exchange gives the opposite edge's row instead, which differs unless fCori
            # is constant in y, and a wet seam face reads it.
            # On a periodic axis face 0 is the seam; the module drops the duplicate face ny
            # and takes the north seam face from the exchange.
            if bool((grid.hfac_s_for(torch.float64, "cpu")[..., 0, :] > 0).any()):
                raise ManifestBuildError(
                    "gmredi ldd97 on a periodic-y grid with wet edge V faces and fCori varying "
                    "in y: MITgcm takes the halo fCori from the halo yC (ini_cori.F:55-100), "
                    "not from the wrapped rows"
                )
        module.equation_of_state = eos_modules[0]


def _refuse_dropped_terms(dynamics: list[TendencyModule], stepper: Any) -> None:
    """Module and stepper pairings that run without error but drop or misplace a term.

    The hydrostatic pressure gradient comes only from the hydrostatic_pressure_gradient
    module (hydrostatic_pressure.py, HydrostaticPressureGradient.evaluate): with an
    equation of state and pressure forcing on, a manifest without it computes the
    density and never applies its gradient, which MITgcm adds whenever
    momPressureForcing is on, on a single layer too (calc_phi_hyd.F:624-630,
    timestep.F:322-326).  surface_heat_salt_flux adds the linear free-surface tracer
    correction (CALC_WSURF_TR) whenever lin_fs_conserve_tr is set, whatever the free
    surface (surface_flux.py), which MITgcm's CONFIG_CHECK refuses with a nonlinear free
    surface (config_check.F:699).  gmredi's Kwz reaches the tracers only through the
    stepper's implicit column diffusivity (implicit_free_surface.py,
    _gm_kwz_contributions), which MITgcm requires of GM/Redi (gmredi_check.F:137-141).
    """
    names = {getattr(module, "name", None) for module in dynamics}
    if (bool(getattr(stepper, "momentum_pressure_forcing", False))
            and names & set(_EQUATION_OF_STATE_MODULES)
            and "hydrostatic_pressure_gradient" not in names):
        raise ManifestBuildError(
            "an equation of state with momentum_pressure_forcing on but no "
            "hydrostatic_pressure_gradient module: the density would be computed and "
            "its pressure gradient dropped; add the module, or turn "
            "momentum_pressure_forcing off (MITgcm model/src/timestep.F:322-326)"
        )
    # tempStepping / saltStepping .FALSE. skip the whole TEMP_INTEGRATE /
    # SALT_INTEGRATE (thermodynamics.F:317-337); a module still naming that
    # tracer would compute a term the stepper then discards.
    unstepped = {
        name for name, switch in (("temp", "temp_stepping"), ("salt", "salt_stepping"))
        if not getattr(stepper, switch, True)
    }
    for module in dynamics:
        named = unstepped.intersection(getattr(module, "tracer_names", ()))
        if named:
            raise ManifestBuildError(
                f"module {getattr(module, 'name', type(module).__name__)!r} names "
                f"{sorted(named)} in tracer_names, but the stepper does not step "
                f"{'them' if len(named) > 1 else 'it'} (temp_stepping / salt_stepping; "
                "MITgcm skips TEMP_INTEGRATE / SALT_INTEGRATE, model/src/thermodynamics.F:317-337)"
            )
    nlfs = int(getattr(stepper, "nonlin_free_surf", 0) or 0)
    if nlfs != 0 and any(bool(getattr(module, "lin_fs_conserve_tr", False)) for module in dynamics):
        raise ManifestBuildError(
            f"lin_fs_conserve_tr with nonlin_free_surf={nlfs}: the linear free-surface "
            "tracer correction would be added to a nonlinear free surface; MITgcm's "
            "CONFIG_CHECK refuses the pair (model/src/config_check.F:699)"
        )
    if "gmredi" in names and not bool(getattr(stepper, "implicit_tracer_diffusion", False)):
        raise ManifestBuildError(
            "gmredi without implicit_tracer_diffusion: its Kwz would be dropped; MITgcm "
            "stops with 'GM/Redi needs implicitDiffusion=.true.' (pkg/gmredi/gmredi_check.F:137-141)"
        )


def _refuse_unread_upwind_shear(
    dynamics: list[TendencyModule], stepper: Any, grid: StructuredCGrid
) -> None:
    """The stepper's upwind_shear with an explicit vector-invariant vertical shear.

    MITgcm's upwindShear selects the vertical-shear flux of MOM_VI_U/V_VERTSHEAR
    (mom_vi_u_vertshear.F:113-127), which MOM_VECINV calls when momImplVertAdv is
    off (mom_vecinv.F:823-831).  LOCOMgcm reads the stepper's upwind_shear only in the
    implicit vertical momentum advection; the explicit shear is the
    vector-invariant module's, whose SHEAR_SCHEMES has no upwind form, so the
    stepper value would be dropped.  One layer has no vertical shear in either model.
    """
    if (grid.nz <= 1 or not bool(getattr(stepper, "upwind_shear", False))
            or bool(getattr(stepper, "momentum_implicit_vertical_advection", False))):
        return
    for module in dynamics:
        if (getattr(module, "name", None) == "vector_invariant_momentum"
                and bool(module.enable_vertical_shear)):
            raise ManifestBuildError(
                "model.stepper.kwargs.upwind_shear=true with the explicit vertical shear of "
                "vector_invariant_momentum_advection: the explicit shear has no upwind form "
                "(SHEAR_SCHEMES), so MITgcm's upwindShear (mom_vi_u_vertshear.F:113-127) "
                "would be dropped; use implicit vertical momentum advection or "
                "upwind_shear=false"
            )


def _refuse_unread_center_spacing(
    dynamics: list[TendencyModule], stepper: Any, grid: StructuredCGrid
) -> None:
    """A given centre spacing (MITgcm delRc, ZLevelCoordinate.center_spacing_given)
    with a consumer that still derives drC or rC from the layer thicknesses: KPP
    (dynamics/kpp.py drC_values, kpp_mixing.py reference_layer_centres) and the
    reference pressure of the pressure-dependent equations of state
    (equation_of_state.py reference_layer_centres)."""
    if not grid.vertical_coordinate.center_spacing_given:
        return
    readers = [
        name for name in (getattr(module, "name", None) for module in dynamics)
        if name in _EQUATION_OF_STATE_MODULES and name != "linear_equation_of_state"
    ]
    if bool(getattr(stepper, "kpp_enabled", False)):
        readers.append("kpp")
    if readers:
        raise ManifestBuildError(
            f"grid.vertical_center_spacing (delRc) with {readers}: these still derive "
            "the centre spacing or centre depths from the layer thicknesses"
        )


def _refuse_multidim_adams_bashforth_vertical(
    dynamics: list[TendencyModule], stepper: Any, grid: StructuredCGrid
) -> None:
    """A multi-dimensional tracer sweep with an explicit Adams-Bashforth-type vertical step on several layers.

    MITgcm's multi-dimensional sweep takes the vertical flux at an interior interface
    from the upwind, DST2, flux-limit, DST3, DST3-FL, OS7MP or PPM/PQM kernels and
    stops on any other scheme (pkg/generic_advdiff/gad_advection.F:987-1018); of the
    schemes LOCOMgcm names those are the Adams-Bashforth ones, centered (2), upwind3 (3)
    and centered4 (4), read from TracerAdvectionScheme.adams_bashforth.  That branch
    runs only for interfaces k > 1 (the surface interface has zero transport, :961-973),
    so one layer runs in MITgcm and is not refused here; LOCOMgcm's R pass then has no
    interior interface either (operators/vertical.py:268, vertical_advective_flux).  An
    Adams-Bashforth horizontal scheme turns the sweep off in MITgcm
    (gad_init_fixed.F:128-137); the module's ``_runs_multi_dim`` reads that declaration
    together with multi_dim and dt.  The whole vertical step sits inside
    ``IF (.NOT.implicitAdvection)`` (gad_advection.F:875), which temp_integrate.F:277-278
    and salt_integrate.F:275-276 set per tracer from tempImplVertAdv and
    saltImplVertAdv; a tracer the stepper advects implicitly gets LOCOMgcm's
    horizontal-only hand-off (HorizontalTracerAdvection.evaluate_for_implicit_vertical_advection)
    and its vertical scheme is checked by the implicit admission, so only tracers
    stepped explicitly are refused.
    """
    if grid.nz <= 1:
        return
    implicit = {
        tracer
        for tracer, flag in (
            ("temp", "temperature_implicit_vertical_advection"),
            ("salt", "salinity_implicit_vertical_advection"),
        )
        if bool(getattr(stepper, flag, False))
    }
    for module in dynamics:
        if getattr(module, "name", None) != "horizontal_tracer_advection":
            continue
        explicit = [tracer for tracer in module.tracer_names if tracer not in implicit]
        vertical = TRACER_SCHEME_BY_NAME[TracerFluxScheme(module.vertical_scheme)]
        if explicit and module._runs_multi_dim() and vertical.adams_bashforth:
            raise ManifestBuildError(
                f"horizontal_tracer_advection multi_dim with vertical_scheme={vertical.name!r} "
                f"(MITgcm {vertical.mit_code}) on {grid.nz} layers for the explicitly stepped "
                f"tracer(s) {explicit}: MITgcm's multi-dimensional sweep stops on an "
                "Adams-Bashforth vertical scheme at an interior interface "
                "(pkg/generic_advdiff/gad_advection.F:987-1018); use upwind, dst3 or dst3_fl "
                "for the vertical scheme, multi_dim=False, or implicit vertical advection for "
                "those tracers"
            )


def _route_explicit_vertical_viscosity(
    dynamics: list[TendencyModule],
    *,
    coefficient: float,
    implicit: bool,
    implicit_solve: bool,
) -> None:
    """Ensure the vertical viscosity is one momentum dissipation term.

    MITgcm has one viscAr, applied explicitly or in the implicit vertical solve
    (set_defaults.F:210, implicitViscosity).  With the gate true the implicit solve
    applies the stepper's viscosity and KPP's kappaRU/RV, so a vertical_mixing module
    that also mixes momentum would add the vertical viscosity a second time as an
    explicit tendency (the implicit stepper supersedes only continuity and the
    pressure gradient); the gate-false route keeps exactly one momentum mixer.
    ``implicit_solve`` says whether the stepper has that solve at all: the explicit
    steppers have no implicit_vertical_viscosity field, take their vertical viscosity
    from the module, and keep the historical early return of ``implicit``'s default.
    """
    existing = [
        module
        for module in dynamics
        if getattr(module, "name", None) == "vertical_mixing"
        and bool(getattr(module, "mix_momentum", False))
    ]
    if implicit:
        if existing and implicit_solve:
            raise ManifestBuildError(
                "vertical_mixing mixes momentum while implicit_vertical_viscosity is on: "
                "the implicit vertical solve already applies the vertical viscosity "
                "(and KPP's), so the module would apply it a second time; set "
                "mix_momentum=False, or turn implicit_vertical_viscosity off "
                "(MITgcm has one viscAr, model/src/set_defaults.F:210)"
            )
        return
    if coefficient <= 0.0:
        return

    if existing:
        if len(existing) != 1:
            raise ManifestBuildError(
                "explicit vertical viscosity must have exactly one momentum mixer"
            )
        module = existing[0]
        if module.tendency_phase != TendencyPhase.MOMENTUM:
            raise ManifestBuildError(
                "explicit vertical viscosity must use tendency_phase='momentum'"
            )
        if not math.isclose(
            float(module.effective_viscosity),
            coefficient,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ManifestBuildError(
                "explicit vertical viscosity coefficient mismatch: "
                f"physics.implicit_viscosity={coefficient!r} but "
                f"vertical_mixing.effective_viscosity={module.effective_viscosity!r}"
            )
        return

    module = DYNAMICS_MODULES.instantiate(
        "vertical_mixing",
        tracer_names=(),
        diffusivity=0.0,
        mix_momentum=True,
        viscosity=coefficient,
        tendency_phase=TendencyPhase.MOMENTUM,
    )
    insert_at = next(
        (
            index
            for index, candidate in enumerate(dynamics)
            if candidate.momentum_term_class
            in {MomentumTermClass.BOTTOM_DRAG, MomentumTermClass.FORCING}
        ),
        len(dynamics),
    )
    dynamics.insert(insert_at, module)


def _instantiate_boundary_condition(manifest_bc: Any, *, obcs_source: Any | None) -> Any:
    if manifest_bc.name is None:
        raise ManifestBuildError("model.boundary_condition.name is required")

    kwargs = dict(manifest_bc.kwargs)
    if manifest_bc.name in {"prescribed_3d", "tracer_open"}:
        if obcs_source is None:
            raise ManifestBuildError(
                f"boundary_condition '{manifest_bc.name}' requires obcs_source argument"
            )
        source_kwarg = "tracer_source" if manifest_bc.name == "prescribed_3d" else "source"
        kwargs.setdefault(source_kwarg, obcs_source)

    try:
        return BOUNDARY_CONDITIONS.instantiate(manifest_bc.name, **kwargs)
    except KeyError as exc:
        raise ManifestBuildError(
            f"boundary_condition '{manifest_bc.name}' not registered. "
            f"Available: {_available_names(BOUNDARY_CONDITIONS)}"
        ) from exc
    except TypeError as exc:
        raise ManifestBuildError(
            f"boundary_condition '{manifest_bc.name}' kwargs incompatible: {exc}"
        ) from exc


def _manifest_tracer_schemes(
    run_cfg: RunConfig,
) -> tuple[Any, Any, Any, Any]:
    horizontal: dict[str, Any] = {
        "temp": run_cfg.advection.tracer,
        "salt": run_cfg.advection.tracer,
    }
    vertical: dict[str, Any] = {
        "temp": run_cfg.advection.vertical,
        "salt": run_cfg.advection.vertical,
    }
    for entry in run_cfg.model.dynamics:
        if entry.name != "horizontal_tracer_advection":
            continue
        kwargs = dict(entry.kwargs)
        tracer_names = kwargs.get("tracer_names", ("temp", "salt"))
        if isinstance(tracer_names, str):
            tracer_names = (tracer_names,)
        normalized_names = {
            "temp" if str(name).lower() == "temperature" else
            "salt" if str(name).lower() == "salinity" else str(name).lower()
            for name in tracer_names
        }
        for tracer in ("temp", "salt"):
            if tracer not in normalized_names:
                continue
            if "scheme" in kwargs:
                horizontal[tracer] = kwargs["scheme"]
            if "vertical_scheme" in kwargs:
                vertical[tracer] = kwargs["vertical_scheme"]
    return horizontal["temp"], horizontal["salt"], vertical["temp"], vertical["salt"]


def _validate_manifest_implicit_vertical_advection(run_cfg: RunConfig) -> None:
    manifest = run_cfg.model
    stepper_kwargs = dict(manifest.stepper.kwargs)
    canonical = run_cfg.physics.implicit_vertical_advection
    temp_h, salt_h, temp_v, salt_v = _manifest_tracer_schemes(run_cfg)
    raw_nonhydrostatic = stepper_kwargs.get(
        "nonhydrostatic", run_cfg.physics.nonhydrostatic.enabled
    )
    if isinstance(raw_nonhydrostatic, Mapping):
        nonhydrostatic = raw_nonhydrostatic.get("nonhydrostatic", False)
    else:
        nonhydrostatic = getattr(
            raw_nonhydrostatic, "nonhydrostatic", raw_nonhydrostatic
        )
    # A declared free-surface level that never reaches the stepper is a silent
    # downgrade to the linear free surface: the manifest says nonlinFreeSurf=3
    # while ImplicitFreeSurfaceStepper runs level 0 and reports nothing.  The
    # declaration site (physics.*) and the consumer (stepper kwargs) must agree.

    _declared_nlfs = int(run_cfg.physics.nonlin_free_surf)
    _stepper_nlfs = int(stepper_kwargs.get("nonlin_free_surf", 0))
    if _declared_nlfs != _stepper_nlfs:
        raise ValueError(
            f"physics.nonlin_free_surf={_declared_nlfs} but "
            f"model.stepper.kwargs.nonlin_free_surf={_stepper_nlfs}: the stepper "
            "is the only consumer, so the declared level would be silently "
            "ignored.  Set both, or neither"
        )

    validate_implicit_vertical_advection_composition(
        temperature_implicit_vertical_advection=stepper_kwargs.get(
            "temperature_implicit_vertical_advection", canonical.temperature
        ),
        salinity_implicit_vertical_advection=stepper_kwargs.get(
            "salinity_implicit_vertical_advection", canonical.salinity
        ),
        momentum_implicit_vertical_advection=stepper_kwargs.get(
            "momentum_implicit_vertical_advection", canonical.momentum
        ),
        temperature_horizontal_advection_scheme=temp_h,
        salinity_horizontal_advection_scheme=salt_h,
        temperature_vertical_advection_scheme=temp_v,
        salinity_vertical_advection_scheme=salt_v,
        stepper_name=manifest.stepper.name,
        stagger_tracer_stepping=stepper_kwargs.get(
            "stagger_tracer_stepping", False
        ),
        nonlin_free_surf=stepper_kwargs.get(
            "nonlin_free_surf", run_cfg.physics.nonlin_free_surf
        ),
        select_r_star=stepper_kwargs.get(
            "select_r_star", run_cfg.physics.free_surface.select_r_star
        ),
        nonhydrostatic=nonhydrostatic,
        grid_kind=run_cfg.grid.kind,
        physics_enable_vertical_shear=run_cfg.physics.enable_vertical_shear,
        dynamics_modules=manifest.dynamics,
        source="build_model_from_manifest",
    )


def _resolve_device(run_cfg: RunConfig) -> str:
    if run_cfg.run.device != "auto":
        return str(run_cfg.run.device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _extra_float(
    extras: Mapping[str, Any],
    *names: str,
    default: float,
) -> float:
    for name in names:
        if name in extras and extras[name] is not None:
            return float(extras[name])
    return float(default)


def _grid_extent(grid: StructuredCGrid) -> tuple[float, float]:
    metrics = grid.metrics
    lx = float(metrics.dx_center[0, :].sum().item()) if grid.ny > 0 else 0.0
    ly = float(metrics.dy_center[:, 0].sum().item()) if grid.nx > 0 else 0.0
    return lx, ly


def _grid_depth(grid: StructuredCGrid, fallback: float) -> float:
    try:
        depth = float(grid.layer_thickness.sum().item())
    except (AttributeError, RuntimeError, TypeError):
        depth = float(fallback)
    return depth if depth > 0.0 else float(fallback)


def _boundary_type(value: str) -> BoundaryType:
    try:
        return BoundaryType(value)
    except ValueError as exc:
        raise ManifestBuildError(
            "boundary.type must be one of "
            f"{[item.value for item in BoundaryType]}, got {value!r}"
        ) from exc


def _config_boundary_types(
    run_cfg: RunConfig,
    grid: StructuredCGrid,
) -> tuple[BoundaryType, BoundaryType]:
    topology = getattr(grid, "boundary_topology", None)
    if topology is not None:
        return topology.west_east, topology.south_north

    boundary = _boundary_type(str(run_cfg.boundary.type))
    return boundary, boundary


def _tracer_flux_scheme(run_cfg: RunConfig) -> TracerFluxScheme:
    value = str(run_cfg.advection.tracer)
    try:
        return TracerFluxScheme(value)
    except ValueError as exc:
        raise ManifestBuildError(
            f"advection.tracer {run_cfg.advection.tracer!r} is not supported by "
            f"TracerFluxScheme; available: {[item.value for item in TracerFluxScheme]}"
        ) from exc


def _resolve_reference_density(run_cfg: RunConfig, extras: Mapping[str, Any]) -> float:
    """Single-source rhoConst: ``physics.reference_density`` wins.

    ``physics.reference_density`` is the canonical face. The legacy channel
    (``case.extras.reference_density`` / ``rho_const``) keeps working when the
    physics face is unset, but writing BOTH with different values raises — two
    config surfaces for one physical quantity drift apart silently
    (docs/NUMERICS.md).
    """
    legacy = None
    for name in ("reference_density", "rho_const"):
        if name in extras and extras[name] is not None:
            legacy = float(extras[name])
            break
    phys = run_cfg.physics.reference_density
    if phys is not None:
        if legacy is not None and not math.isclose(
            float(phys), legacy, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ManifestBuildError(
                f"reference_density dual-entry mismatch: physics.reference_density="
                f"{float(phys)!r} but case.extras gives {legacy!r}. "
                "physics.reference_density is the single source — "
                "remove the extras entry or set them equal."
            )
        return float(phys)
    if legacy is not None:
        return legacy
    return 1027.0


def runtime_config_fields(run_cfg: RunConfig) -> dict[str, Any]:
    """The ``RegionalOceanConfig`` fields that do not depend on the grid.

    ``model.default_dynamics`` builds a manifest-less model's modules from them;
    the MITgcm data export reads the same values (``layer_d/_mit_switches.py``).
    """
    extras = dict(run_cfg.case.extras or {})
    return {
        "nx": run_cfg.grid.nx,
        "ny": run_cfg.grid.ny,
        "nz": run_cfg.grid.nz,
        "gravity": _extra_float(extras, "gravity", default=9.81),
        "reference_density": _resolve_reference_density(run_cfg, extras),
        "thermal_expansion": _extra_float(extras, "thermal_expansion", "alpha_t", default=2.0e-4),
        "haline_contraction": _extra_float(extras, "haline_contraction", "beta_s", default=0.0),
        "reference_temperature": _extra_float(extras, "reference_temperature", "t_ref", default=0.0),
        "reference_salinity": _extra_float(extras, "reference_salinity", "s_ref", default=35.0),
        "vertical_diffusivity": resolved_vertical_diffusion(run_cfg).stepper_diffusivity_kr,
        "coriolis_parameter": _extra_float(extras, "coriolis_parameter", "f0", default=0.0),
        "default_dt": float(run_cfg.time.dt),
        "dt_mom": float(run_cfg.time.dt_mom),
        "dt_tracer": float(run_cfg.time.dt_tracer),
        "dt_freesurf": float(run_cfg.time.dt_freesurf),
        "dt_clock": float(run_cfg.time.dt_clock),
        # run.precision selects the model config dtype.  The caller already
        # builds grid / IC tensors at this precision (experiments/cases/
        # _runtime.py resolves it once), so hardcoding float64 here silently
        # cast a float32 run back to float64 -- the switch was accepted and
        # then had no effect.  Default float64 keeps existing runs unchanged.
        "dtype": _resolve_dtype(run_cfg),
        "device": _resolve_device(run_cfg),
        "tracer_advection_scheme": _tracer_flux_scheme(run_cfg),
    }


def _build_runtime_config(run_cfg: RunConfig, grid: StructuredCGrid) -> RegionalOceanConfig:
    """Build RegionalOceanConfig from RunConfig fields and the supplied grid."""
    lx, ly = _grid_extent(grid)
    west_east_boundary, south_north_boundary = _config_boundary_types(run_cfg, grid)
    return RegionalOceanConfig(
        lx=lx,
        ly=ly,
        depth=_grid_depth(grid, run_cfg.grid.depth_m),
        west_east_boundary=west_east_boundary,
        south_north_boundary=south_north_boundary,
        **runtime_config_fields(run_cfg),
    )


def _required_ic_tensor(ic_bundle: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    try:
        return ic_bundle[name]
    except KeyError as exc:
        raise ManifestBuildError(f"ic_bundle missing required tensor '{name}'") from exc


def _resolve_dtype(run_cfg: RunConfig) -> torch.dtype:
    """``run.precision`` -> torch dtype.  Unknown values are rejected at config
    load (RunIdentity.__post_init__); guarded here too so a builder reached by
    another path cannot silently fall back to float64."""
    precision = getattr(run_cfg.run, "precision", "float64")
    if precision == "float64":
        return torch.float64
    if precision == "float32":
        return torch.float32
    raise ValueError(
        f"unsupported run.precision {precision!r}; expected 'float64' or 'float32'"
    )


def apply_initial_theta_freezing(
    temperature: torch.Tensor, *, run_cfg: RunConfig,
) -> torch.Tensor:
    """Apply MITgcm ini_theta.F:129-147 after input-precision conversion.

    Both initial-condition flags are required config fields.  The source-coded
    -1.9 degC limit applies to every level and halo cell, without a wet mask.
    """
    if (
        run_cfg.initial_condition.check_ini_temp
        and run_cfg.initial_condition.allow_freezing
    ):
        return torch.clamp(temperature, min=-1.9)
    return temperature


def build_model_from_manifest(
    run_cfg: RunConfig,
    *,
    grid: StructuredCGrid,
    ic_bundle: Mapping[str, torch.Tensor],
    obcs_source: Any | None = None,
    forcing_source: Any | None = None,
) -> tuple[TorchRegionalOceanModel, PrognosticState]:
    """Universal model assembly from yaml manifest.

    Args:
        run_cfg: typed config; must have non-empty run_cfg.model.
        grid: pre-built StructuredCGrid (caller's responsibility).
        ic_bundle: maps tensor name to tensor; required keys at minimum:
            "temperature", "salinity". Optional: "u", "v", "eta".
            Tensor shape and dtype must match model's config dtype/device after
            TorchRegionalOceanModel.create_state coercion.
        obcs_source: pre-built sample-capable boundary source. May be None; if
            None and boundary_condition.name resolves to a class requiring an
            OBC source, ManifestBuildError is raised.
        forcing_source: optional pre-built forcing source. When the selected
            stepper accepts ``external_pnh_provider``, this source is bound
            through its ``sample(time_seconds)`` method or used directly if
            callable.

    Returns:
        (model, state)

    Raises:
        ManifestBuildError: manifest is empty, registry names are unknown,
            kwargs are invalid, or required runtime inputs are missing.
    """
    manifest = run_cfg.model
    if manifest.is_empty():
        raise ManifestBuildError(
            "run_cfg.model is empty. yaml must declare a 'model:' block "
            "with dynamics/stepper/boundary_condition for framework path."
        )

    _validate_manifest_implicit_vertical_advection(run_cfg)
    diffusion = assemble_vertical_diffusion(run_cfg)
    dynamics = [_instantiate_dynamics(entry) for entry in diffusion.dynamics]
    from locomgcm.dynamics.era5_bulk_forcing import ERA5BulkForcing
    from locomgcm.dynamics.era5_wind_forcing import ERA5WindForcing
    bulk_producers = [m for m in dynamics if isinstance(m, ERA5BulkForcing) and m.use_atm_wind]
    for module in dynamics:
        if isinstance(module, ERA5WindForcing) and module.use_bulk_stress:
            if len(bulk_producers) != 1:
                raise ManifestBuildError("wind-derived stress requires exactly one ERA5 bulk producer")
            module.bulk_forcing = bulk_producers[0]
    _bind_gmredi(dynamics, grid)
    stepper = _instantiate_stepper(
        diffusion.stepper,
        nonhydrostatic_config=run_cfg.physics.nonhydrostatic,
        rotation_period=float(run_cfg.physics.rotation_period),
        implicit_viscosity=run_cfg.physics.implicit_viscosity,
        implicit_vertical_viscosity=run_cfg.physics.implicit_vertical_viscosity,
        forcing_source=forcing_source,
    )
    _refuse_dropped_terms(dynamics, stepper)
    _refuse_unread_center_spacing(dynamics, stepper, grid)
    _refuse_multidim_adams_bashforth_vertical(dynamics, stepper, grid)
    _refuse_unread_upwind_shear(dynamics, stepper, grid)
    coefficient = (
        float(run_cfg.physics.implicit_viscosity)
        if run_cfg.physics.implicit_viscosity is not None
        else float(getattr(stepper, "implicit_viscosity", 0.0))
    )
    implicit = (
        bool(run_cfg.physics.implicit_vertical_viscosity)
        if run_cfg.physics.implicit_vertical_viscosity is not None
        else bool(getattr(stepper, "implicit_vertical_viscosity", True))
    )
    _route_explicit_vertical_viscosity(
        dynamics,
        coefficient=coefficient,
        implicit=implicit,
        implicit_solve=hasattr(stepper, "implicit_vertical_viscosity"),
    )
    boundary_condition = _instantiate_boundary_condition(
        manifest.boundary_condition,
        obcs_source=obcs_source,
    )

    sampling_components(boundary_condition, dynamics)
    config = _build_runtime_config(run_cfg, grid)
    model = TorchRegionalOceanModel(
        config,
        grid=grid,
        dynamics_modules=dynamics,
        stepper=stepper,
        boundary_condition=boundary_condition,
        tracer_multi_dim_advection=bool(run_cfg.advection.multi_dim),
    )

    temperature = apply_initial_theta_freezing(
        _required_ic_tensor(ic_bundle, "temperature"), run_cfg=run_cfg,
    )

    state = model.create_state(
        temp=temperature,
        salt=_required_ic_tensor(ic_bundle, "salinity"),
        eta=ic_bundle.get("eta"),
        u=ic_bundle.get("u"),
        v=ic_bundle.get("v"),
    )
    return model, state
