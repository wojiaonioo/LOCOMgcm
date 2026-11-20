"""Check and normalize implicit free-surface configuration and composition values.

Composition, runtime, execution-protocol, solver and option-pair checks only
inspect inputs and raise. Configuration validation returns the normalized
convective configuration; resolve_tracer_ab_flags normalizes tracer AB flags.
resolve_vector_invariant_momentum returns the selected momentum form.
resolve_volume_freshwater writes _volume_freshwater on forcing modules.
Assembly validation logs KPP sign-zero settings to the stepper logger.

Flags and scheme codes are dimensionless; viscosity inputs use m^2/s.
Scalar finiteness checks retain PyTorch's default tensor dtype.
Admission does not evaluate gridded fields or masks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from defgcm.dynamics import AssemblyPolicy
from defgcm.dynamics.kpp import KPP_SURFACE_BUOYANCY_FORMS, check_kpp_sign_zero
from defgcm.dynamics.implicit_vertical_advection import IMPLICIT_VERTICAL_TRACER_SCHEMES
from defgcm.dynamics.kpp_convective_implicit import ConvectiveImplicitConfig
from defgcm.operators.fluxes.tracer_flux import (
    TRACER_SCHEME_BY_NAME,
    TracerFluxScheme,
)
from defgcm.seaice.step_state import seaice_module

from .kpp_inputs import SURFACE_STRESS_MODULES


@dataclass(frozen=True)
class ImplicitVerticalAdvectionFlags:
    temperature: bool
    salinity: bool
    momentum: bool

    @property
    def active(self) -> bool:
        return self.temperature or self.salinity or self.momentum


@dataclass(frozen=True)
class ImplicitVerticalAdvectionComposition:
    flags: ImplicitVerticalAdvectionFlags
    stepper_name: str
    stagger_tracer_stepping: bool
    nonlin_free_surf: int
    select_r_star: int
    nonhydrostatic: Any
    grid_kind: str | None
    dynamics_modules: Sequence[Any] | None


@dataclass(frozen=True)
class ImplicitVerticalAdvectionExecution:
    populate_module_diagnostics: Any
    accumulate_tendency_diagnostics: Any
    enforce_tendency: Any
    missing_required_slow_modules: tuple[str, ...]


#: MITgcm nonlinFreeSurf levels the stepper implements with select_rStar = 0; the
#: configuration value is the MITgcm code.  Its branches are 0, >=1, >=2 and >2; 4
#: there adds etaH to the hydrostatic integral (calc_phi_hyd.F:96), which this
#: stepper does not do.
NONLIN_FREE_SURF_LEVELS = (0, 1, 2, 3)
#: MITgcm select_rStar values the stepper implements: 0, the surface-layer
#: UPDATE_SURF_DR geometry, and 2, the r* geometry of every level
#: (CALC_R_STAR/UPDATE_R_STAR).
SELECT_R_STAR_VALUES = (0, 2)
#: The one nonlinFreeSurf level select_rStar = 2 runs with: config_check.F:754 warns
#: on any other, and level 4 takes the r* hydrostatic gradient
#: (calc_grad_phi_hyd.F:63-83).
R_STAR_NONLIN_FREE_SURF_LEVEL = 4

_COMPOSITION_MISSING = object()


def _composition_module_name(module: Any) -> str:
    name = getattr(module, "name", None)
    if name is None:
        name = type(module).__name__
    return str(name).strip().lower()


def _composition_module_path(module: Any, index: int) -> str:
    name = _composition_module_name(module)
    if isinstance(getattr(module, "kwargs", None), Mapping):
        return f"model.dynamics[{index}:{name}]"
    return f"model.dynamics_modules[{index}:{name}]"


def _composition_module_value(
    module: Any,
    key: str,
    *,
    default: Any = _COMPOSITION_MISSING,
) -> Any:
    kwargs = getattr(module, "kwargs", None)
    if isinstance(kwargs, Mapping):
        return kwargs.get(key, default)
    return getattr(module, key, default)


def _composition_tracer_names(module: Any, name: str) -> tuple[str, ...]:
    default = (
        ("temp", "salt")
        if name in {"horizontal_tracer_advection", "vertical_tracer_transport"}
        else ()
    )
    raw = _composition_module_value(module, "tracer_names", default=default)
    if isinstance(raw, str):
        raw = (raw,)
    try:
        values = tuple(str(value).strip().lower() for value in raw)
    except TypeError:
        return ()
    normalized = []
    for value in values:
        if value == "temperature":
            value = "temp"
        elif value == "salinity":
            value = "salt"
        normalized.append(value)
    return tuple(normalized)


def _tracer_scheme(value: Any) -> TracerFluxScheme | None:
    """The declared scheme (``TRACER_ADVECTION_SCHEMES``) a configured value names, else None.

    The value is a scheme name or member.
    """
    raw = getattr(value, "value", value)
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in TRACER_SCHEME_BY_NAME:
            return TracerFluxScheme(token)
    return None


def validate_rotation_period(stepper: Any, modules: Sequence[Any]) -> None:
    """Refuse a model whose momentum, W Coriolis and KPP f use different rotation periods.

    MITgcm has one ``rotationPeriod`` (``ini_parms.F:487-490``); the canonical
    assembly hands ``physics.rotation_period`` to each holder checked here.
    """
    periods = {"stepper.rotation_period (KPP f)": float(stepper.rotation_period)}
    if stepper.nonhydrostatic.nonhydrostatic:
        periods["stepper.nonhydrostatic.rotation_period (W Coriolis)"] = float(
            stepper.nonhydrostatic.rotation_period
        )
    for module in modules:
        period = getattr(module, "rotation_period", None)
        if period is not None:
            periods[f"model.dynamics[{_composition_module_name(module)}].rotation_period"] = float(period)
    if len(set(periods.values())) > 1:
        raise ValueError(
            f"rotation periods disagree: {periods}; MITgcm has one rotationPeriod "
            "(ini_parms.F:487-490), set physics.rotation_period"
        )


def validate_negative_salinity_check(stepper: Any, modules: Sequence[Any], *, face_count: int) -> None:
    """Refuse CHECK_SALINITY_FOR_NEGATIVE_VALUES where its port does not reach.

    The stepper resets salt at the step's rhoInSitu (do_oceanic_phys.F:758-765) and
    before the convective adjustment.  Not ported: the cube step; the readers of the
    unreset surface salt in EXTERNAL_FORCING_SURF, which precedes the reset (SSS
    relaxation forcing_surf_relax.F:87,103, convertFW2Salt=-1 external_forcing_surf.F:316,
    real freshwater with a nonlinear free surface :284); and INI_PRESSURE, which resets
    the initial salt when storePhiHyd4Phys (ini_pressure.F:112, set_parms.F:304).
    """
    if not stepper.check_salinity_for_negative_values:
        return
    refused = []
    if face_count != 1:
        refused.append("the cubed-sphere step")
    for module in modules:
        name = _composition_module_name(module)
        if name == "surface_climate_relaxation" and module.tau_salt > 0.0:
            refused.append(f"{name} with tau_salt > 0 (SSS relaxation reads the unreset salt)")
        if getattr(module, "convert_fw2salt", None) == -1.0:
            refused.append(f"{name} with convert_fw2salt = -1 (reads the unreset surface salt)")
        if name == "mdjwf_equation_of_state" or (
            name == "jmd95_equation_of_state" and module.dynamic_pressure
        ):
            refused.append(f"{name} with dynamic pressure (INI_PRESSURE resets the initial salt)")
    if stepper.use_real_freshwater_flux and stepper.nonlin_free_surf > 0:
        refused.append("use_real_freshwater_flux with nonlin_free_surf > 0 (PmEpR term reads the unreset surface salt)")
    if refused:
        raise ValueError(
            "check_salinity_for_negative_values (CHECK_SALINITY_FOR_NEGATIVE_VALUES) is not "
            f"ported with: {'; '.join(refused)}"
        )


def validate_tracer_advection_scheme(
    *, tracer, horizontal, vertical, implicit, convective_implicit,
) -> None:
    """Check the scheme pairs that cross modules: the implicit column and KPP's convective column.

    Each scheme alone is checked where it is declared (``HorizontalTracerAdvection``
    against ``HORIZONTAL_TRACER_SCHEMES`` and ``VERTICAL_TRACER_SCHEMES``).
    """
    vertical_scheme = _tracer_scheme(vertical)
    if implicit and convective_implicit and vertical_scheme is not TracerFluxScheme.UPWIND:
        raise ValueError(
            "KPP convective implicit requires vertical advection scheme upwind; "
            f"its Newton column does not implement {vertical!r} ({tracer})"
        )


def validate_tracer_ab_scheme(*, tracer: str, horizontal: Any, tendency_ab: bool) -> None:
    """Refuse a tracer whose tendency AB disagrees with its horizontal scheme's declaration.

    MITgcm extrapolates a stepped tracer's tendency exactly when its horizontal
    scheme is 2, 3 or 4 (``AdamsBashforthGt = tempStepping``,
    ``gad_init_fixed.F:143-157``), the schemes that declare ``adams_bashforth``
    in ``TRACER_ADVECTION_SCHEMES``; the stepper's tracer AB is that tendency
    form (``doAB_onGtGs``), so its ``ab2_on_<tracer>_tendency`` must name the
    same value.
    """
    declared = TRACER_SCHEME_BY_NAME[_tracer_scheme(horizontal)].adams_bashforth
    if tendency_ab != declared:
        raise ValueError(
            f"{tracer}: the stepper's tendency AB is {tendency_ab} but its horizontal "
            f"advection scheme {_tracer_scheme(horizontal).value!r} declares "
            f"adams_bashforth={declared} (MITgcm AdamsBashforthGt/Gs, "
            "gad_init_fixed.F:143-157)"
        )


def validate_implicit_vertical_advection_composition(
    *,
    temperature_implicit_vertical_advection: Any,
    salinity_implicit_vertical_advection: Any,
    momentum_implicit_vertical_advection: Any,
    temperature_horizontal_advection_scheme: Any = None,
    salinity_horizontal_advection_scheme: Any = None,
    temperature_vertical_advection_scheme: Any = None,
    salinity_vertical_advection_scheme: Any = None,
    stepper_name: str | None = None,
    stagger_tracer_stepping: Any = None,
    nonlin_free_surf: Any = None,
    select_r_star: Any = None,
    nonhydrostatic: Any = None,
    grid_kind: str | None = None,
    physics_enable_vertical_shear: Any = None,
    dynamics_modules: Sequence[Any] | None = None,
    source: str = "implicit vertical advection composition",
) -> None:
    """Validate the declared composition without mutating modules or state."""
    flags = {
        "temperature": temperature_implicit_vertical_advection,
        "salinity": salinity_implicit_vertical_advection,
        "momentum": momentum_implicit_vertical_advection,
    }
    errors: list[str] = []
    for tracer, value in flags.items():
        if type(value) is not bool:
            errors.append(
                f"stepper.{tracer}_implicit_vertical_advection must be bool, "
                f"got {type(value).__name__}"
            )
    if errors:
        raise ValueError(f"{source}:\n- " + "\n- ".join(errors))
    schemes = {
        "temperature": (
            temperature_horizontal_advection_scheme,
            temperature_vertical_advection_scheme,
        ),
        "salinity": (
            salinity_horizontal_advection_scheme,
            salinity_vertical_advection_scheme,
        ),
    }
    carried = {"temp", "salt"} if dynamics_modules is None else {
        tracer for module in dynamics_modules
        if _composition_module_name(module) == "horizontal_tracer_advection"
        for tracer in _composition_tracer_names(module, "horizontal_tracer_advection")
    }
    for tracer, (horizontal, vertical) in schemes.items():
        if {"temperature": "temp", "salinity": "salt"}[tracer] not in carried:
            continue
        validate_tracer_advection_scheme(
            tracer=tracer, horizontal=horizontal, vertical=vertical,
            implicit=flags[tracer], convective_implicit=False,
        )
    if not any(flags.values()):
        return

    if stepper_name is None:
        errors.append("model.stepper.name is required when an implicit flag is true")
    elif str(stepper_name) != "implicit_free_surface":
        errors.append(
            "model.stepper.name must be 'implicit_free_surface' when an implicit "
            f"flag is true, got {stepper_name!r}"
        )

    if type(stagger_tracer_stepping) is not bool:
        errors.append(
            "model.stepper.kwargs.stagger_tracer_stepping must be an explicit bool "
            "when an implicit flag is true"
        )
    elif not stagger_tracer_stepping:
        errors.append(
            "model.stepper.kwargs.stagger_tracer_stepping=false selects the "
            "unsupported synchronous tracer time path"
        )
    if type(nonlin_free_surf) is bool or not isinstance(nonlin_free_surf, int):
        errors.append(
            "model.stepper.kwargs.nonlin_free_surf must be an integer when an "
            "implicit flag is true"
        )
    elif (
        nonlin_free_surf != 0
        and str(grid_kind).lower() in {"cube", "cube_global", "cubed_sphere"}
    ):
        errors.append(
            f"model.stepper.kwargs.nonlin_free_surf={nonlin_free_surf} selects an "
            "unsupported nonlinear/r*/sigma time path"
        )
    if type(select_r_star) is bool or not isinstance(select_r_star, int):
        errors.append(
            "model.stepper.kwargs.select_r_star must be an integer when an "
            "implicit flag is true"
        )
    if type(nonhydrostatic) is not bool:
        errors.append(
            "model.stepper.kwargs.nonhydrostatic.nonhydrostatic must be bool when "
            "an implicit flag is true"
        )

    normalized_grid_kind = None if grid_kind is None else str(grid_kind).lower()
    if normalized_grid_kind in {"cube", "cube_global", "cubed_sphere"}:
        normalized_grid_kind = "cube"
    elif normalized_grid_kind in {"regional", "latlon", "mit_verification"}:
        normalized_grid_kind = "latlon"
    else:
        errors.append(
            f"grid.kind must resolve to latlon or cube, got {grid_kind!r}"
        )
    if normalized_grid_kind == "cube" and nonhydrostatic is True:
        errors.append(
            "grid.kind=cubed_sphere conflicts with "
            "model.stepper.kwargs.nonhydrostatic.nonhydrostatic=true"
        )

    for tracer, (horizontal, vertical) in schemes.items():
        enabled = flags[tracer]
        for direction, value in (("horizontal", horizontal), ("vertical", vertical)):
            if value is None:
                if enabled:
                    config_path = (
                        "advection.tracer/model.dynamics[*:horizontal_tracer_advection]"
                        ".kwargs.scheme"
                        if direction == "horizontal"
                        else "advection.vertical/model.dynamics[*:horizontal_tracer_advection]"
                        ".kwargs.vertical_scheme"
                    )
                    errors.append(
                        f"{config_path}: effective {tracer} {direction} advection "
                        "scheme is missing"
                    )
                continue
            if _tracer_scheme(value) is None:
                errors.append(
                    "model.dynamics[*:horizontal_tracer_advection].kwargs."
                    f"{'scheme' if direction == 'horizontal' else 'vertical_scheme'}: "
                    f"effective {tracer} {direction} advection scheme {value!r} is unknown"
                )
        horizontal_scheme = _tracer_scheme(horizontal)
        vertical_scheme = _tracer_scheme(vertical)
        if (enabled and vertical_scheme is not None
                and vertical_scheme not in IMPLICIT_VERTICAL_TRACER_SCHEMES):
            errors.append(
                f"effective {tracer} vertical advection scheme must be one of "
                f"{', '.join(IMPLICIT_VERTICAL_TRACER_SCHEMES)} (IMPLICIT_VERTICAL_TRACER_SCHEMES) "
                f"when {tracer}_implicit_vertical_advection=true, got {vertical_scheme}"
            )
        if enabled and (
            horizontal_scheme in (TracerFluxScheme.UPWIND3, TracerFluxScheme.CENTERED4)
            or (horizontal_scheme is TracerFluxScheme.CENTERED and nonlin_free_surf == 0)
        ):
            errors.append(
                f"effective {tracer} horizontal advection scheme "
                f"{horizontal_scheme} requires unsupported raw AB history correction"
            )

    if dynamics_modules is None:
        errors.append(
            "model.dynamics_modules composition is required when an implicit flag is true"
        )
        modules: tuple[Any, ...] = ()
    else:
        modules = tuple(dynamics_modules)

    vi_modules: list[tuple[Any, str, Any]] = []
    explicit_momentum_paths: list[str] = []
    implicit_drag_paths: list[str] = []
    missing_bottom_background_paths: list[str] = []
    linfs_conserve_paths: list[str] = []
    fused_tracer_paths: dict[str, list[str]] = {"temp": [], "salt": []}
    standalone_tracer_paths: dict[str, list[str]] = {"temp": [], "salt": []}
    for index, module in enumerate(modules):
        name = _composition_module_name(module)
        path = _composition_module_path(module, index)
        if name in {
            "vector_invariant_momentum",
            "vector_invariant_momentum_advection",
        }:
            shear = _composition_module_value(
                module, "enable_vertical_shear", default=True
            )
            vi_modules.append((module, path, shear))
        elif name in {"momentum_advection", "flux_form_momentum"}:
            explicit_momentum_paths.append(path)
        elif name == "bottom_drag":
            implicit_drag = _composition_module_value(
                module, "implicit", default=False
            )
            if implicit_drag is True:
                implicit_drag_paths.append(path)
            no_slip_bottom = _composition_module_value(
                module, "no_slip_bottom", default=True
            )
            vertical_viscosity = _composition_module_value(
                module, "vertical_viscosity", default=0.0
            )
            if flags["momentum"] and no_slip_bottom is True:
                if (
                    isinstance(vertical_viscosity, bool)
                    or not isinstance(vertical_viscosity, (int, float))
                    or not torch.isfinite(torch.tensor(float(vertical_viscosity)))
                    or float(vertical_viscosity) <= 0.0
                ):
                    missing_bottom_background_paths.append(
                        f"{path}.vertical_viscosity"
                    )

        if name == "surface_heat_salt_flux" and _composition_module_value(
            module, "lin_fs_conserve_tr", default=False
        ) is True:
            linfs_conserve_paths.append(f"{path}.lin_fs_conserve_tr=true")

        # Known limitation: active GM residual/bolus-w has no canonical module
        # or field in this repository.  Do not invent an alias-based detector
        # here; the module must be defined first.

        tracer_names = _composition_tracer_names(module, name)
        if name == "horizontal_tracer_advection" and _composition_module_value(
            module, "combined_with_vtt", default=False
        ) is True:
            for tracer in ("temp", "salt"):
                if tracer in tracer_names:
                    fused_tracer_paths[tracer].append(path)
        if name == "vertical_tracer_transport":
            for tracer in ("temp", "salt"):
                if tracer in tracer_names:
                    standalone_tracer_paths[tracer].append(path)

    if physics_enable_vertical_shear is not None:
        if type(physics_enable_vertical_shear) is not bool:
            errors.append(
                "physics.enable_vertical_shear must be bool, got "
                f"{type(physics_enable_vertical_shear).__name__}"
            )
        if len(vi_modules) == 1 and type(vi_modules[0][2]) is bool:
            if physics_enable_vertical_shear != vi_modules[0][2]:
                errors.append(
                    "physics.enable_vertical_shear and "
                    f"{vi_modules[0][1]}.enable_vertical_shear disagree"
                )

    if flags["momentum"]:
        if len(vi_modules) != 1:
            errors.append(
                "momentum_implicit_vertical_advection=true requires exactly one "
                f"vector-invariant core, found {len(vi_modules)}"
            )
        for _, path, shear in vi_modules:
            if type(shear) is not bool:
                errors.append(
                    f"{path}.enable_vertical_shear must be bool, got "
                    f"{type(shear).__name__}"
                )
            elif shear:
                errors.append(
                    f"{path}.enable_vertical_shear=true conflicts with "
                    "momentum_implicit_vertical_advection=true"
                )
        if explicit_momentum_paths:
            errors.append(
                "momentum_implicit_vertical_advection=true conflicts with explicit "
                "momentum route(s): " + ", ".join(explicit_momentum_paths)
            )
        if implicit_drag_paths:
            errors.append(
                "momentum_implicit_vertical_advection=true conflicts with implicit "
                "bottom drag route(s): " + ", ".join(implicit_drag_paths)
            )
        if missing_bottom_background_paths:
            errors.append(
                "momentum_implicit_vertical_advection=true with no-slip bottom "
                "requires explicit global-bottom background viscosity at: "
                + ", ".join(missing_bottom_background_paths)
            )
        if physics_enable_vertical_shear is True:
                errors.append(
                    "physics.enable_vertical_shear=true conflicts with "
                    "momentum_implicit_vertical_advection=true"
                )

    for tracer, flag_name in (
        ("temp", "temperature_implicit_vertical_advection"),
        ("salt", "salinity_implicit_vertical_advection"),
    ):
        enabled = flags["temperature" if tracer == "temp" else "salinity"]
        if not enabled:
            continue
        if fused_tracer_paths[tracer]:
            errors.append(
                f"{flag_name}=true conflicts with combined_with_vtt module(s): "
                + ", ".join(fused_tracer_paths[tracer])
            )
        if standalone_tracer_paths[tracer]:
            errors.append(
                f"{flag_name}=true conflicts with VerticalTracerTransport module(s): "
                + ", ".join(standalone_tracer_paths[tracer])
            )

    if (flags["temperature"] or flags["salinity"]) and linfs_conserve_paths:
        errors.append(
            "implicit tracer vertical advection requires linFSConserveTr=false; "
            "the combination with the global surface correction is not validated: "
            + ", ".join(linfs_conserve_paths)
        )

    if errors:
        raise ValueError(f"{source}:\n- " + "\n- ".join(errors))

    if normalized_grid_kind == "cube":
        enabled_names = ", ".join(name for name, enabled in flags.items() if enabled)
        raise NotImplementedError(
            f"{source}: cube implicit vertical advection is not implemented "
            f"(enabled: {enabled_names})"
        )


def resolve_vector_invariant_momentum(
    vector_invariant_momentum: Any,
    nonlin_free_surf: int,
    modules: Sequence[Any],
) -> bool | None:
    """MITgcm ``vectorInvariantMomentum`` of an assembled composition.

    The ``nonlin_free_surf >= 2`` momentum tendency rescale is taken outside the
    vector-invariant form only (``timestep.F:275-276``).  A
    ``VectorInvariantMomentumAdvection`` core names True; a ``FluxFormMomentum``
    or ``MomentumAdvection`` core names False.  A composition without a core (a
    bare ``FCoriolis`` serves both forms, MITgcm ``momAdvection=.FALSE.``) names
    neither and takes ``stepper.vector_invariant_momentum``.  Two cores of
    different forms, or a declaration that contradicts the core, raise; so does
    ``nonlin_free_surf >= 2`` with the form named by neither.  Returns None only
    when the rescale branch is not taken and nothing names the form.
    """
    from defgcm.dynamics.flux_form_momentum import FluxFormMomentum
    from defgcm.dynamics.momentum_advection import MomentumAdvection
    from defgcm.dynamics.vector_invariant_momentum import VectorInvariantMomentumAdvection

    named = set()
    for module in modules:
        if isinstance(module, VectorInvariantMomentumAdvection):
            named.add(True)
        elif isinstance(module, (FluxFormMomentum, MomentumAdvection)):
            named.add(False)
    if len(named) > 1:
        raise ValueError(
            "the composition holds a vector-invariant and a flux-form momentum core"
        )
    declared = vector_invariant_momentum
    if declared is not None and type(declared) is not bool:
        raise TypeError(
            "model.stepper.kwargs.vector_invariant_momentum must be bool or omitted, got "
            f"{type(declared).__name__}"
        )
    core = next(iter(named)) if named else None
    if declared is not None and core is not None and declared != core:
        raise ValueError(
            f"model.stepper.kwargs.vector_invariant_momentum={declared} contradicts the "
            f"{'vector-invariant' if core else 'flux-form'} momentum core of the composition"
        )
    resolved = core if core is not None else declared
    if resolved is None and nonlin_free_surf >= 2:
        raise ValueError(
            f"nonlin_free_surf={nonlin_free_surf} rescales the momentum tendency "
            "only outside the vector-invariant form (MITgcm timestep.F:275-276), and this "
            "composition has no momentum core naming the form: declare "
            "model.stepper.kwargs.vector_invariant_momentum (MITgcm vectorInvariantMomentum)"
        )
    return resolved


def resolve_volume_freshwater(
    nonlin_free_surf: int,
    use_real_freshwater_flux: bool,
    modules: Sequence[Any],
) -> bool:
    """MITgcm's ``(nonlinFreeSurf>0 .AND. useRealFreshWaterFlux)`` surface branch.

    The freshwater then changes the column height: the salt forcing reads
    ``PmEpR`` and EmPmR adds no virtual salt flux
    (``external_forcing_surf.F:261-288``).  Every ``wpflux_surface_forcing``
    module of the composition is told the branch; the stepper passes PmEpR
    to the evaluations exactly when it is True.
    """
    volume = bool(nonlin_free_surf >= 1 and use_real_freshwater_flux)
    for module in modules:
        if getattr(module, "name", "") == "wpflux_surface_forcing":
            object.__setattr__(module, "_volume_freshwater", volume)
    return volume


#: The surface tracer-forcing producers the KPP and GGL90 forcing harvest reads,
#: the first of them in the composition.
KPP_TRACER_PRODUCERS = ("era5_bulk_forcing", "wpflux_surface_forcing", "surface_heat_salt_flux")


@dataclass(frozen=True)
class PackageModules:
    """The composition's modules the regional step reads by package, found once at model assembly.

    ``None`` is a package the assembled composition does not carry.  ``surface_stress``
    is the first module of ``SURFACE_STRESS_MODULES``; ``kpp_wind`` the first
    ``era5_wind_forcing`` and ``kpp_tracer`` the first of ``KPP_TRACER_PRODUCERS``,
    the producers of the KPP and GGL90 forcing harvest.
    """

    gmredi: Any
    seaice: Any
    surface_relaxation: Any
    surface_stress: Any
    kpp_wind: Any
    kpp_tracer: Any


def resolve_package_modules(model: Any) -> PackageModules:
    modules = getattr(model, "dynamics_modules", ()) or ()

    def first(names: tuple[str, ...]) -> Any:
        return next((m for m in modules if getattr(m, "name", "") in names), None)

    return PackageModules(
        gmredi=first(("gmredi",)),
        seaice=seaice_module(model),
        surface_relaxation=first(("surface_climate_relaxation",)),
        surface_stress=first(SURFACE_STRESS_MODULES),
        kpp_wind=first(("era5_wind_forcing",)),
        kpp_tracer=first(KPP_TRACER_PRODUCERS),
    )


def _runtime_tracer_schemes(
    modules: Sequence[Any],
) -> tuple[Any, Any, Any, Any]:
    horizontal: dict[str, Any] = {"temp": None, "salt": None}
    vertical: dict[str, Any] = {"temp": None, "salt": None}
    for module in modules:
        name = _composition_module_name(module)
        if name != "horizontal_tracer_advection":
            continue
        tracer_names = _composition_tracer_names(module, name)
        for tracer in ("temp", "salt"):
            if tracer not in tracer_names:
                continue
            horizontal[tracer] = _composition_module_value(module, "scheme", default=None)
            vertical[tracer] = _composition_module_value(
                module, "vertical_scheme", default=None
            )
    return horizontal["temp"], horizontal["salt"], vertical["temp"], vertical["salt"]


def _validate_runtime_implicit_vertical_advection(
    composition: ImplicitVerticalAdvectionComposition,
    *,
    source: str,
) -> None:
    flags = composition.flags
    modules = composition.dynamics_modules
    module_sequence = () if modules is None else tuple(modules)
    temp_h, salt_h, temp_v, salt_v = _runtime_tracer_schemes(module_sequence)
    validate_implicit_vertical_advection_composition(
        temperature_implicit_vertical_advection=(
            flags.temperature
        ),
        salinity_implicit_vertical_advection=(
            flags.salinity
        ),
        momentum_implicit_vertical_advection=(
            flags.momentum
        ),
        temperature_horizontal_advection_scheme=temp_h,
        salinity_horizontal_advection_scheme=salt_h,
        temperature_vertical_advection_scheme=temp_v,
        salinity_vertical_advection_scheme=salt_v,
        stepper_name=composition.stepper_name,
        stagger_tracer_stepping=composition.stagger_tracer_stepping,
        nonlin_free_surf=composition.nonlin_free_surf,
        select_r_star=composition.select_r_star,
        nonhydrostatic=composition.nonhydrostatic,
        grid_kind=composition.grid_kind,
        dynamics_modules=modules,
        source=source,
    )


def _refuse_flather_nonlinear_free_surface(nonlin_free_surf: int, flather: Any) -> None:
    """Refuse a Flather velocity boundary with ``nonlin_free_surf > 0``.

    ``flather`` is the model's Flather boundary, alone or inside tracer_open
    (``TorchRegionalOceanModel._find_flather_bc``), or None.  Called from
    ``ImplicitFreeSurfaceStepper.prepare_step_host``, the host phase every
    regional step passes through (``build_step_plan`` and the direct
    ``step``/``step_outcome``), before any state is touched.
    """
    if nonlin_free_surf > 0 and flather is not None:
        raise NotImplementedError(
            f"a Flather barotropic open boundary with nonlin_free_surf={nonlin_free_surf} is not "
            "supported: the step freezes the Flather target with H, the inner-face wet depth, read "
            "from the lagged surface hFac, while the last application of the step takes its "
            "wet-column mean on the hFac installed for the new surface level, so the target and "
            "the mean come from different surface geometries and no discretisation of this pairing "
            "is defined. Use a linear free surface (nonlin_free_surf=0) or the prescribed_3d "
            "velocity boundary"
        )


def _refuse_stepping_switches_unimplemented(
    *,
    mom_stepping: bool,
    open_boundary: bool,
    nonhydrostatic: bool,
    external_pnh: bool,
) -> None:
    """Refuse a switched-off stepping switch outside the regional path written for it.

    With momStepping=.FALSE. MITgcm leaves uVel and vVel untouched each step and
    skips SOLVE_FOR_PRESSURE (forward_step.F:786-795, :897-903, :911-917); the
    regional step keeps them and skips the predictor, the CG2D solve and the
    correction.  Continuity runs whenever calc_wVelocity = momStepping .OR.
    exactConserv (set_parms.F:105, forward_step.F:924-930): with exactConserv it
    moves etaN and etaH from the fixed flow (integr_continuity.F:207-229,
    update_etah.F), as the regional step's continuity does whatever mom_stepping
    is; without it etaN keeps its value.  With tempStepping or saltStepping
    .FALSE. MITgcm skips TEMP_INTEGRATE or SALT_INTEGRATE
    (thermodynamics.F:317-337); the regional step holds that tracer.  The cube
    step declares its own (CUBE_STEP_KWARGS).  Not implemented: with
    momStepping off, open boundaries, which MITgcm then never applies to uVel,
    vVel (OBCS_APPLY_UV sits in MOMENTUM_CORRECTION_STEP,
    momentum_correction_step.F:93-96) while the regional step enforces them; a
    nonhydrostatic or external
    nonhydrostatic pressure, which MITgcm switches off (set_parms.F:88).
    Called from ``ImplicitFreeSurfaceStepper.prepare_step_host``.
    """
    if mom_stepping:
        return
    refused = [
        name for name, on in (
            ("open boundaries", open_boundary),
            ("a nonhydrostatic pressure", nonhydrostatic),
            ("an external nonhydrostatic pressure provider", external_pnh),
        ) if on
    ]
    if refused:
        raise NotImplementedError(
            f"mom_stepping=False with {', '.join(refused)} is not implemented: only the "
            "regional step without open boundaries keeps u and v as MITgcm's "
            "momStepping=.FALSE. does (forward_step.F:786-795, :897-917)"
        )


def _refuse_r_star_grid(*, select_r_star: int, cube: bool, open_boundary: bool, has_hfac: bool) -> None:
    """The grids the r* geometry is not ported to; called from ``prepare_step_host``.

    The cube step has no nonlinear free-surface path (its r* also exchanges the
    factors as a vector and fills the corner regions, calc_r_star.F:200-226);
    open boundaries need OBCS_APPLY_R_STAR (calc_r_star.F:164-170); without the
    C/W/S hFac there is no geometry to rescale.
    """
    if select_r_star != 2:
        return
    refused = [name for name, on in (
        ("the cube step", cube), ("open boundaries", open_boundary),
        ("a grid without C/W/S hFac", not has_hfac),
    ) if on]
    if refused:
        raise NotImplementedError(
            f"select_r_star=2 is not implemented with {', '.join(refused)}"
        )


def _refuse_r_star_unread_ke_scheme(
    *, select_r_star: int, vector_invariant: bool | None, ke_schemes: tuple[int, ...],
) -> None:
    """The r* face average needs MITgcm's global selectKEscheme when the form is vector-invariant.

    rStarAreaWeight is off for vectorInvariantMomentum with selectKEscheme 1 or 3
    (calc_r_star.F:64-68), whether or not momentum steps.  LOCOMgcm carries the KE
    choice only on vector_invariant_momentum modules, so a vector-invariant form
    without such a module, or modules that disagree, leave it unreadable.
    Called from ``prepare_step_host``.
    """
    if select_r_star != 2 or not vector_invariant:
        return
    if len(set(ke_schemes)) != 1:
        raise NotImplementedError(
            "select_r_star=2 with vector_invariant_momentum=True needs exactly one "
            "selectKEscheme for rStarAreaWeight (calc_r_star.F:64-68); the "
            f"vector_invariant_momentum modules carry {sorted(set(ke_schemes))}"
        )


def _require_implicit_vertical_advection_execution_protocol(
    execution: ImplicitVerticalAdvectionExecution,
) -> None:
    """Reject enabled execution contexts that cannot run the physical hand-offs."""
    required_methods = {
        "_populate_module_diagnostics": execution.populate_module_diagnostics,
        "accumulate_tendency_diagnostics": execution.accumulate_tendency_diagnostics,
    }
    missing = [name for name, method in required_methods.items() if not callable(method)]
    if not callable(execution.enforce_tendency):
        missing.append("boundary_condition.enforce_tendency")
    missing.extend(execution.missing_required_slow_modules)
    if missing:
        raise TypeError(
            "implicit vertical advection requires the formal model execution "
            "protocol; missing: " + ", ".join(missing)
        )


def validate_implicit_free_surface_configuration(
    *,
    kpp_tracer_substeps: int,
    kpp_enabled: bool,
    stagger_tracer_stepping: bool,
    select_r_star: int,
    nh_mom_advection: bool,
    nh_mom_viscosity: bool,
    temperature_implicit_vertical_advection: bool,
    salinity_implicit_vertical_advection: bool,
    momentum_implicit_vertical_advection: bool,
    implicit_vertical_viscosity: bool,
    use_real_freshwater_flux: bool,
    upwind_shear: bool,
    kpp_estimate_uref: bool,
    kpp_smooth_shsq: bool,
    kpp_smooth_dbloc: bool,
    kpp_sign_zero: str,
    kpp_surface_buoyancy: str,
    kpp_dbloc_form: str,
    kpp_convective_implicit: Any,
) -> ConvectiveImplicitConfig | None:
    if type(kpp_tracer_substeps) is not int or kpp_tracer_substeps < 1:
        raise ValueError("kpp_tracer_substeps must be a positive integer")
    if kpp_tracer_substeps > 1 and not (kpp_enabled and stagger_tracer_stepping):
        raise ValueError("KPP tracer substeps require KPP and staggered tracers")
    # Only the select_rStar=0 surface-layer geometry is
    # implemented (compute_surface_hfac).  MITgcm branches to UPDATE_R_STAR
    # for select_rStar > 0 (forward_step.F:829 at MITgcm 27256b2d) and rescales hFacC/W/S at
    # EVERY level (update_r_star.F:50-59); accepting the flag here would run
    # a hybrid of r* tracer rescaling and z-level surface geometry that
    # silently diverges from the reference model.  Unconditional:
    # a conjunction with nonlin_free_surf left nlfs=0 ∧ r*>=1 accepted yet
    # consumer-less, and the manifest validator only reaches its own
    # select_r_star check when an implicit-vertical flag is set — direct
    # construction is the gate that always fires.  Negative values are
    # equally meaningless, so anything nonzero is refused.
    # Lift this gate only together with all-level geometry and tracer
    # rescaling support for UPDATE_R_STAR.
    if isinstance(select_r_star, bool) or select_r_star not in SELECT_R_STAR_VALUES:
        raise NotImplementedError(
            f"select_r_star={select_r_star} is not one of the implemented values "
            f"{SELECT_R_STAR_VALUES} (0: surface-layer UPDATE_SURF_DR; 2: r* geometry "
            "of every level, UPDATE_R_STAR)"
        )
    for name, value in (
        ("nh_mom_advection", nh_mom_advection),
        ("nh_mom_viscosity", nh_mom_viscosity),
        ("temperature_implicit_vertical_advection", temperature_implicit_vertical_advection),
        ("salinity_implicit_vertical_advection", salinity_implicit_vertical_advection),
        ("momentum_implicit_vertical_advection", momentum_implicit_vertical_advection),
        ("implicit_vertical_viscosity", implicit_vertical_viscosity),
        ("use_real_freshwater_flux", use_real_freshwater_flux),
        ("upwind_shear", upwind_shear),
        ("kpp_estimate_uref", kpp_estimate_uref),
        ("kpp_smooth_shsq", kpp_smooth_shsq),
        ("kpp_smooth_dbloc", kpp_smooth_dbloc),
    ):
        if type(value) is not bool:
            raise ValueError(
                f"ImplicitFreeSurfaceStepper.{name} must be bool, got "
                f"{type(value).__name__}"
            )
    check_kpp_sign_zero(kpp_sign_zero)
    if kpp_surface_buoyancy not in KPP_SURFACE_BUOYANCY_FORMS:
        raise ValueError(
            "ImplicitFreeSurfaceStepper.kpp_surface_buoyancy must be one of "
            f"{KPP_SURFACE_BUOYANCY_FORMS}, got {kpp_surface_buoyancy!r}"
        )
    from defgcm.dynamics.kpp import check_kpp_dbloc_form

    check_kpp_dbloc_form(kpp_dbloc_form)
    return ConvectiveImplicitConfig.from_mapping(kpp_convective_implicit)


def validate_r_star_configuration(
    *,
    select_r_star: int,
    nonlin_free_surf: int,
    mom_stepping: bool,
    use_real_freshwater_flux: bool,
    kpp_enabled: bool,
    temperature_implicit_vertical_advection: bool,
    salinity_implicit_vertical_advection: bool,
    momentum_implicit_vertical_advection: bool,
    implicit_div_2d_flow: float,
    nonhydrostatic: bool,
) -> None:
    """Pair select_r_star with nonlin_free_surf and refuse the r* combinations not ported.

    select_r_star = 2 runs with R_STAR_NONLIN_FREE_SURF_LEVEL only, and that level
    only under r*.  Each refusal below names the MITgcm r* statements LOCOMgcm does
    not run yet.
    """
    if (select_r_star == 2) != (nonlin_free_surf == R_STAR_NONLIN_FREE_SURF_LEVEL):
        raise NotImplementedError(
            f"select_r_star={select_r_star} with nonlin_free_surf={nonlin_free_surf}: "
            f"the r* geometry runs with nonlin_free_surf={R_STAR_NONLIN_FREE_SURF_LEVEL} "
            "only and that level only under r* (MITgcm config_check.F:754, "
            "calc_phi_hyd.F:96)"
        )
    if select_r_star != 2:
        return
    refused = [
        reason for on, reason in (
            (mom_stepping, "mom_stepping=True (the r* momentum terms: timestep.F:277-283, "
             "mom_fluxform.F:556-564, mom_calc_rtrans.F:111-155, calc_grad_phi_hyd.F:63-83)"),
            (use_real_freshwater_flux, "use_real_freshwater_flux=True (the fresh-water "
             "dEtaHdt in rStarDhDt, integr_continuity.F:187-189, 245-256)"),
            (kpp_enabled, "KPP (its nonlocal and column terms take the surface-level "
             "rescale only)"),
            (temperature_implicit_vertical_advection or salinity_implicit_vertical_advection
             or momentum_implicit_vertical_advection,
             "implicit vertical advection (the r* volume of gad_implicit_r.F:384-390)"),
            (float(implicit_div_2d_flow) != 1.0, f"implicit_div_2d_flow={implicit_div_2d_flow!r} "
             "(the initial etaH reads the surface w, which r* makes zero)"),
            (nonhydrostatic, "the nonhydrostatic solver (cg3d.F:114-175 r* branches)"),
        ) if on
    ]
    if refused:
        raise NotImplementedError(
            "select_r_star=2 is not implemented with " + "; ".join(refused)
        )


def validate_implicit_free_surface_assembly(
    *,
    convective: ConvectiveImplicitConfig | None,
    kpp_enabled: bool,
    stagger_tracer_stepping: bool,
    kpp_tracer_substeps: int,
    temperature_implicit_vertical_advection: bool,
    salinity_implicit_vertical_advection: bool,
    kpp_dbloc_form: str,
    nonlin_free_surf: int,
    kpp_coriolis_profile_c: Any,
    kpp_sign_zero: str,
    cube_assembly_policy: AssemblyPolicy | Mapping[str, bool] | None,
    mom_forcing_out_ab: Any,
) -> None:
    if isinstance(nonlin_free_surf, bool) or nonlin_free_surf not in (
            *NONLIN_FREE_SURF_LEVELS, R_STAR_NONLIN_FREE_SURF_LEVEL):
        raise NotImplementedError(
            f"nonlin_free_surf={nonlin_free_surf!r} is not one of the implemented levels "
            f"{NONLIN_FREE_SURF_LEVELS} (select_r_star=0) or "
            f"{R_STAR_NONLIN_FREE_SURF_LEVEL} (select_r_star=2)"
        )
    if convective is not None and not (
        kpp_enabled
        and stagger_tracer_stepping
        and kpp_tracer_substeps == 1
        and temperature_implicit_vertical_advection
        and salinity_implicit_vertical_advection
    ):
        raise ValueError(
            "kpp_convective_implicit requires KPP, staggered tracer stepping, one KPP "
            "tracer substep and implicit vertical advection of temperature and salinity"
        )
    if convective is not None and kpp_dbloc_form != "divided_difference":
        raise ValueError(
            "kpp_convective_implicit requires kpp_dbloc_form='divided_difference' "
            "(its column coefficients use the JMD95 divided difference)"
        )
    if convective is not None and nonlin_free_surf != 0:
        raise ValueError(
            "kpp_convective_implicit with nonlin_free_surf!=0 is unsupported: "
            "the convective column solve does not take recip_hFacNew"
        )
    if isinstance(kpp_coriolis_profile_c, str):
        raise ValueError(
            "ImplicitFreeSurfaceStepper.kpp_coriolis_profile_c must be the resolved "
            f"profile, got the unresolved {kpp_coriolis_profile_c!r}"
        )
    if kpp_enabled:
        logging.getLogger("defgcm.steppers.implicit_free_surface").info(
            "KPP Fortran SIGN of a -0.0 argument: kpp_sign_zero=%s",
            kpp_sign_zero,
        )
    if isinstance(cube_assembly_policy, Mapping):
        AssemblyPolicy.cube_neutral(bottom_drag_in_ab=True).with_overrides(
            cube_assembly_policy
        )
    elif cube_assembly_policy is not None and not isinstance(
        cube_assembly_policy, AssemblyPolicy
    ):
        raise ValueError(
            "ImplicitFreeSurfaceStepper.cube_assembly_policy must be an "
            f"AssemblyPolicy, mapping, or None, got "
            f"{type(cube_assembly_policy).__name__}"
        )
    if type(mom_forcing_out_ab) is not bool:
        raise ValueError(
            "model.stepper.kwargs.mom_forcing_out_ab must be explicitly "
            "declared as true or false for ImplicitFreeSurfaceStepper"
        )


def resolve_tracer_ab_flags(
    *,
    ab2_on_tracer_tendency: bool | None,
    ab2_on_temperature_tendency: bool | None,
    ab2_on_salinity_tendency: bool | None,
) -> tuple[bool, bool]:
    tracer_ab = ab2_on_tracer_tendency
    temperature_ab = ab2_on_temperature_tendency
    salinity_ab = ab2_on_salinity_tendency
    if temperature_ab is None and salinity_ab is None:
        if type(tracer_ab) is not bool:
            raise ValueError(
                "model.stepper.kwargs must explicitly declare both "
                "ab2_on_temperature_tendency and ab2_on_salinity_tendency, "
                "or the legacy ab2_on_tracer_tendency spelling"
            )
        temperature_ab = tracer_ab
        salinity_ab = tracer_ab
    elif temperature_ab is None or salinity_ab is None:
        raise ValueError(
            "model.stepper.kwargs must declare ab2_on_temperature_tendency "
            "and ab2_on_salinity_tendency together"
        )
    elif type(temperature_ab) is not bool or type(salinity_ab) is not bool:
        raise ValueError(
            "ab2_on_temperature_tendency and ab2_on_salinity_tendency "
            "must both be bool"
        )
    if tracer_ab is not None and type(tracer_ab) is not bool:
        raise ValueError(
            "model.stepper.kwargs.ab2_on_tracer_tendency must be bool when "
            "the legacy spelling is used"
        )
    return temperature_ab, salinity_ab


def validate_solver_tolerance_modes(
    *,
    cg_tolerance_mode: str,
    cg3d_tolerance_mode: str,
) -> None:
    from defgcm.solvers.cg2d import CG_TOLERANCE_MODES
    from defgcm.steppers.implicit_free_surface_cg2d import CG2D_STOP_CRITERIA

    if cg_tolerance_mode not in CG2D_STOP_CRITERIA:
        raise ValueError(
            "ImplicitFreeSurfaceStepper.cg_tolerance_mode must be one of "
            f"{CG2D_STOP_CRITERIA}, got {cg_tolerance_mode!r}"
        )
    if cg3d_tolerance_mode not in CG_TOLERANCE_MODES:
        raise ValueError(
            "ImplicitFreeSurfaceStepper.cg3d_tolerance_mode must be one of "
            f"{CG_TOLERANCE_MODES}, got {cg3d_tolerance_mode!r}"
        )


def _refuse_unimplemented_pairs(
    *,
    kpp_enabled: bool,
    implicit_vertical_viscosity: bool,
    implicit_div_2d_flow: float,
    exact_conserv: bool,
    nonlin_free_surf: int,
    nonhydrostatic: Any,
) -> None:
    """Pairs of this stepper's switches that MITgcm stops on and LOCOMgcm has no design for.

    Each would run without error and drop a term or pair two inconsistent steps:
    KPP's viscosity kappaRU/RV reaches the momentum only in the implicit vertical solve
    (step_segments.py:297-302, :319), so with the explicit vertical viscosity it is
    dropped; MITgcm's KPP_CHECK asks for implicitViscosity (kpp_check.F:141-145).  The
    exact-conservation and nonlinear free-surface pairs are CONFIG_CHECK's
    (config_check.F:718, :726, :874); LOCOMgcm's CG3D operator is the linear
    free-surface one (_get_cg3d_coefficients).  casegen's build-stage rules cite the
    raise lines below.
    """
    if kpp_enabled and not implicit_vertical_viscosity:
        raise ValueError(
            "KPP with implicit_vertical_viscosity=False: KPP's viscosity reaches the "
            "momentum only through the implicit vertical solve and would be dropped "
            "(MITgcm KPP_CHECK needs implicitViscosity, pkg/kpp/kpp_check.F:141-145)"
        )
    if float(implicit_div_2d_flow) != 1.0 and not exact_conserv:
        raise ValueError(
            f"implicit_div_2d_flow={implicit_div_2d_flow!r} < 1 with "
            "exact_conserv=False: MITgcm's CONFIG_CHECK requires exactConserv when "
            "implicDiv2DFlow < 1 (model/src/config_check.F:718)"
        )
    nlfs = int(nonlin_free_surf or 0)
    if nlfs != 0 and not exact_conserv:
        raise ValueError(
            f"nonlin_free_surf={nlfs} with exact_conserv=False: MITgcm's CONFIG_CHECK "
            "refuses a nonlinear free surface without exactConserv "
            "(model/src/config_check.F:726)"
        )
    if nlfs != 0 and bool(nonhydrostatic):
        raise ValueError(
            f"nonlin_free_surf={nlfs} with the nonhydrostatic solver: the CG3D "
            "coefficients are those of the linear free surface, and MITgcm's "
            "CONFIG_CHECK has no nonlinear free surface in the nonhydrostatic code "
            "(model/src/config_check.F:874)"
        )


def _check_nonhydrostatic_implicit_factors(
    *,
    nonhydrostatic: Any,
    implicit_surf_press: float,
    implicit_div_2d_flow: float,
) -> None:
    """The CG3D chain runs implicitNHPress = implicDiv2DFlow = 1 only.

    MITgcm scales the CG3D operator by ``implicitNHPress*implicDiv2DFlow``
    (ini_cg3d.F:106-107, :119-120, :150-151) and the PRE_CG3D w term by
    ``implicDiv2DFlow`` (pre_cg3d.F:190, :208-209), and ``implicitNHPress``
    defaults to ``implicSurfPress`` (ini_parms.F:485-486).  LOCOMgcm's
    CG3D operator (solvers/cg3d_operator.py) and w term (solvers/cg3d_rhs.py)
    omit both factors and TIMESTEP_WVEL has no explicit NH gradient
    (solvers/cg3d_timestep_w.py), which is exact only when both are 1,
    while the free-surface solve and the CG3D seed run this stepper's
    values.  The stepper fields are the one source of the two factors, and
    their defaults (0.5) are not what the nonhydrostatic chain runs, so a
    case that leaves either one out would pair two different factors.
    """
    if not bool(nonhydrostatic):
        return
    wrong = {
        name: value
        for name, value in (
            ("implicit_surf_press", implicit_surf_press),
            ("implicit_div_2d_flow", implicit_div_2d_flow),
        )
        if float(value) != 1.0
    }
    if wrong:
        listed = ", ".join(f"{name}={value!r}" for name, value in wrong.items())
        raise ValueError(
            "nonhydrostatic=True needs implicit_surf_press = "
            f"implicit_div_2d_flow = 1.0, got {listed} (the stepper default "
            "is 0.5).  The CG3D operator and PRE_CG3D w term implement "
            "MITgcm's implicitNHPress*implicDiv2DFlow = 1 only "
            "(ini_cg3d.F:106-107, pre_cg3d.F:190), with implicitNHPress = "
            "implicSurfPress (ini_parms.F:485-486), while the free-surface "
            "solve would run the value above.  Set both to 1.0 in "
            "model.stepper.kwargs."
        )
