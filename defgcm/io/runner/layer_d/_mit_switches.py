"""Single source of truth for MITgcm namelist switches that LOCOMgcm also implements.

Why this module exists
----------------------
The MIT-export consumers write namelists for the reference model, while the LOCOMgcm
runtime configures itself from the same case yaml.  Historically the export side
wrote **hardcoded constants** while the LOCOMgcm side read config, with nothing linking
the two.  Every such pair is a silent cross-model divergence waiting to happen:

* ``OBCSfixTopo`` — export hardcoded
  ``.TRUE.``; once the LOCOMgcm side became configurable with MIT's own default
  (``.FALSE.``) the two models built **different geometry** at the open
  boundary.
* ``hFacMin`` — export hardcoded 0.3 against a LOCOMgcm yaml of 0.2; MIT step-0
  dump zeroed wet-but-thin cells, ``max|ΔT| = 29 K``.

Both were found by numerical forensics *after the fact*.  This table plus the
assembly-time gate in :func:`assert_switches_agree` turns that class of bug into
an export-time exception.

The two properties this module maintains
----------------------------------------
**Agreement** (load-bearing).  The value written into the reference model's
namelist equals the value the LOCOMgcm runtime will act on, for the same
``run_cfg``.  :func:`render_switch_lines` produces the line and
:func:`assert_switches_agree` re-parses the assembled file to prove it, so an
f-string edit that bypasses the renderer is caught at export time rather than
surfacing as a numerical mismatch hundreds of steps into a comparison.

**Default hygiene** (secondary).  With a case yaml that declares nothing, LOCOMgcm
should behave the way MITgcm behaves out of the box.  Where that is not
achievable, the divergence is *declared* in the table
(:attr:`MitSwitch.to_default` plus :attr:`MitSwitch.default_divergence_note`)
rather than left implicit.  Note the two properties are independent: a declared
default divergence is harmless precisely because Agreement still holds — the
export writes LOCOMgcm's value, not MIT's.

What a :class:`MitSwitch` binds together
----------------------------------------
``mit_name``     the namelist keyword as MITgcm spells it;
``mit_default``  **MITgcm's own default**, not LOCOMgcm's preference;
``mit_source``   the ``path:line`` under ``references/MITgcm`` proving
                 ``mit_default``;
``to_path``      the dotted ``RunConfig`` path the LOCOMgcm runtime reads, or a key
                 into :data:`_CUSTOM_READERS` when the LOCOMgcm-side value is derived
                 from several native fields rather than mirrored by one leaf;
``to_default``   what :func:`switch_value` returns for a config that declares
                 nothing — pinned here so that *moving* a default is a red test
                 rather than a silent behaviour change.

🔴 Setting ``to_default`` to the historically-exported value to "preserve
compatibility" is exactly how the OBCSfixTopo bug was born.  A case that needs
the non-default value declares it in its own yaml.

Field names deliberately mirror the WebUI parameter registry entries
(``mit_default`` / ``evidence`` / ``to.path``) so that if the registry-driven
compiler ever takes over namelist generation, this table merges into it rather
than being rewritten.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Any


@dataclass(frozen=True)
class MitSwitch:
    """One MITgcm logical switch that LOCOMgcm also implements.

    Parameters
    ----------
    mit_name
        Namelist keyword, MITgcm spelling (e.g. ``OBCSfixTopo``).
    namelist
        Which file it is written to (``data.obcs`` / ``data.exf`` / ...).
        Used to scope the assembly-time gate to the file being written.
    mit_default
        MITgcm's own default value, per ``mit_source``.  A ``bool`` renders as a
        Fortran logical, an ``int`` as a bare integer — MITgcm namelists carry
        both kinds, and some of the switches that matter most are integers
        (``momForcingOutAB``).
    mit_source
        ``path:line`` inside ``references/MITgcm`` proving ``mit_default``.
    to_path
        Dotted ``RunConfig`` path the LOCOMgcm runtime reads (a *mirror* switch), or a
        key of :data:`_CUSTOM_READERS` (a *derived* switch), or ``None`` when LOCOMgcm
        has no counterpart at all — then the switch is emitted at ``mit_default``
        and there is nothing to cross-check.
    to_default
        Value :func:`switch_value` yields for a ``RunConfig`` that declares
        nothing about this switch.  Pinned so that relocating a schema default
        fails the gate instead of quietly changing what the reference model runs.
    rationale
        Why this switch is safe to emit the way it is — read by humans during
        review, and quoted by the gate's failure message.
    default_divergence_note
        Required when ``to_default != mit_default``: why LOCOMgcm's
        out-of-the-box behaviour is allowed to differ from MITgcm's.
    """

    mit_name: str
    namelist: str
    mit_default: bool | int
    mit_source: str
    to_path: str | None
    to_default: bool | int
    rationale: str
    default_divergence_note: str | None = None

    def __post_init__(self) -> None:
        # Enforced at import, not only under pytest: a divergent default that
        # nobody wrote down is the OBCSfixTopo failure mode in miniature.
        if self.to_default != self.mit_default and not self.default_divergence_note:
            raise ValueError(
                f"{self.mit_name}: to_default={self.to_default} differs from MITgcm's "
                f"default {self.mit_default} ({self.mit_source}) with no "
                "default_divergence_note explaining why."
            )
        if self.to_default == self.mit_default and self.default_divergence_note:
            raise ValueError(
                f"{self.mit_name}: default_divergence_note is set but the defaults "
                "agree — a stale note is worse than none."
            )


def _resolve(run_cfg: Any, dotted: str) -> Any:
    node: Any = run_cfg
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


#: Boundary conditions that overwrite the open-boundary strip from a boundary
#: data source, i.e. LOCOMgcm's counterpart of ``useOBCSprescribe``.  Kept
#: identical to the set in ``io/runner/builders/from_manifest.py:121``, which is
#: what actually decides whether an ``obcs_source`` is wired in at build time.
_PRESCRIBING_BOUNDARY_CONDITIONS = frozenset({"prescribed_3d", "tracer_open"})

#: Dynamics modules that implement the OBCS sponge band.  Identical to the
#: ``targets`` set in ``_assert_obcs_sponge_consistency``
#: (``consumers/mit_binary.py:1421-1424``), which cross-checks their timescales
#: against the same ``data.obcs`` this module writes the switch into.
_SPONGE_DYNAMICS = frozenset(
    {
        "momentum_boundary_sponge_relaxation",
        "tracer_boundary_sponge_relaxation",
    }
)


def _prescribe_enabled(cfg: Any) -> bool:
    return str(getattr(cfg.model.boundary_condition, "name", None)) in (
        _PRESCRIBING_BOUNDARY_CONDITIONS
    )


def _sponge_enabled(cfg: Any) -> bool:
    return any(name in _SPONGE_DYNAMICS for _where, name, _cls, _kw in _runtime_components(cfg))


#: The field of each registered boundary condition that switches LOCOMgcm's
#: OBCS_BALANCE_FLOW port (``boundaries/open.py`` ``_obcs_balance_flow``), or
#: ``None`` when the condition never balances.
_OBCS_BALANCE_FIELDS = {
    "passthrough": None,
    "structured": None,
    "flather_barotropic": "balance_obc",
    "prescribed_3d": "enforce_obcs_balance",
    "internal_wave_open": "enforce_obcs_balance",
}

#: Boundary conditions that balance only through a nested velocity boundary,
#: and the field holding it: ``tracer_open`` enforces its ``velocity_boundary``
#: each step (``TracerOpenBoundaryCondition.enforce``) and without one runs a
#: ``StructuredBoundaryCondition``, which does not balance.
_NESTED_BOUNDARY_FIELDS = {"tracer_open": "velocity_boundary"}


def _balance_switch(name: str, value_of: Any) -> bool:
    """The balance switch of boundary condition ``name``; ``value_of(field)`` reads its fields."""
    if name in _NESTED_BOUNDARY_FIELDS:
        nested = value_of(_NESTED_BOUNDARY_FIELDS[name])
        return False if nested is None else _boundary_object_balances(nested)
    if name not in _OBCS_BALANCE_FIELDS:
        raise ValueError(
            f"useOBCSbalance: boundary condition {name!r} has no entry in "
            f"_OBCS_BALANCE_FIELDS or _NESTED_BOUNDARY_FIELDS "
            f"({', '.join([*_OBCS_BALANCE_FIELDS, *_NESTED_BOUNDARY_FIELDS])})"
        )
    field_name = _OBCS_BALANCE_FIELDS[name]
    return False if field_name is None else bool(value_of(field_name))


def _boundary_object_balances(bc: Any) -> bool:
    """The balance switch of an assembled boundary condition object."""
    import defgcm.boundaries  # noqa: F401  (registers every boundary condition)
    from defgcm.boundaries._registry import BOUNDARY_CONDITIONS

    names = [name for name in BOUNDARY_CONDITIONS.list_names()
             if BOUNDARY_CONDITIONS.get(name) is type(bc)]
    if len(names) != 1:
        raise ValueError(f"useOBCSbalance: {type(bc).__name__} is registered as {names or 'nothing'}")
    return _balance_switch(names[0], lambda field_name: getattr(bc, field_name))


def _obcs_balance_enabled(cfg: Any) -> bool:
    """Resolve ``useOBCSbalance`` from the boundary condition's own balance switch.

    ``prescribed_3d`` and ``internal_wave_open`` (which accepts only False) hold
    ``enforce_obcs_balance``, ``flather_barotropic`` holds ``balance_obc``; the
    value is the yaml kwarg when given, else the class default.  ``tracer_open``
    takes the switch of its nested ``velocity_boundary`` object, False without
    one.  A boundary condition that never balances, or none at all, gives False.
    """
    bc = cfg.model.boundary_condition
    name = getattr(bc, "name", None)
    if name is None:
        return False
    import defgcm.boundaries  # noqa: F401  (registers every boundary condition)
    from defgcm.boundaries._registry import BOUNDARY_CONDITIONS

    cls = BOUNDARY_CONDITIONS.get(name) if (
        name in _OBCS_BALANCE_FIELDS or name in _NESTED_BOUNDARY_FIELDS) else None
    kwargs = getattr(bc, "kwargs", None)
    return _balance_switch(name, lambda field_name: _component_field(cls, kwargs, field_name))


def _component_field(cls: type, kwargs: Any, name: str) -> Any:
    """``name`` as a configured component holds it: the yaml kwarg, else the class default."""
    kw = dict(kwargs or {})
    if name in kw:
        return kw[name]
    for f in fields(cls):
        if f.name == name:
            if f.default is not MISSING:
                return f.default
            if f.default_factory is not MISSING:
                return f.default_factory()
    raise ValueError(f"{cls.__name__} has no default for {name!r} and the yaml omits it")


def _kpp_enabled(cfg: Any) -> bool:
    """Resolve ``useKPP`` from the stepper's native KPP gate."""
    stepper_kw = dict(getattr(cfg.model.stepper, "kwargs", {}) or {})
    return bool(stepper_kw.get("kpp_enabled", False))


#: ``ImplicitFreeSurfaceStepper`` field defaults for the two time-stepping
#: switches, i.e. what the stepper runs when the case yaml omits the kwarg.
#: Keep these equal to the stepper dataclass defaults.
_IFS_EXACT_CONSERV_DEFAULT = True
_IFS_STAGGER_TRACER_STEPPING_DEFAULT = False


def _exact_conserv_enabled(cfg: Any) -> bool:
    """Resolve ``exactConserv`` from ``model.stepper.kwargs.exact_conserv``."""
    stepper_kw = dict(getattr(cfg.model.stepper, "kwargs", {}) or {})
    return bool(stepper_kw.get("exact_conserv", _IFS_EXACT_CONSERV_DEFAULT))


def _stagger_time_step_enabled(cfg: Any) -> bool:
    """Resolve ``staggerTimeStep`` from ``model.stepper.kwargs.stagger_tracer_stepping``."""
    stepper_kw = dict(getattr(cfg.model.stepper, "kwargs", {}) or {})
    return bool(
        stepper_kw.get("stagger_tracer_stepping", _IFS_STAGGER_TRACER_STEPPING_DEFAULT)
    )


# ------------------------------------------------------------ tracer stepping
#
# MITgcm's tempStepping / saltStepping switch one tracer's whole equation off:
# TEMP_INTEGRATE / SALT_INTEGRATE are skipped (thermodynamics.F:314, :325),
# with the implicit vertical diffusion inside them (temp_integrate.F:499,
# salt_integrate.F:497), and the tracer's advection and forcing flags are
# cleared (set_parms.F:242-247).  OBCS_APPLY_TS still writes the open-boundary
# values either way (thermodynamics.F:355-357).  The LOCOMgcm stepper has no
# such switch: a tracer is stepped by the dynamics modules whose tracer_names
# contain it, which is how the MITgcm-to-LOCOMgcm translator writes these
# two switches (experiments/mitcases/translate.py, rows temp_stepping and
# salt_stepping: "'temp' joins the prognostic tracer set").

#: Registered tracer-phase dynamics modules without a ``tracer_names`` field,
#: and whether each can change a tracer.  DiagnosticVerticalVelocity only fills
#: the w diagnostics and returns a zero tendency.  The three surface-forcing
#: modules each have a heat path and a freshwater/salt path, set by kwargs
#: (surface_heat_salt_flux) or by a data source (era5_bulk_forcing,
#: wpflux_surface_forcing).  The readers do not interpret those settings, which
#: would copy the modules' evaluate logic here, and count either tracer as
#: possibly changed; that can only refuse an export, never write a wrong value.
#: Every tracer-phase module must declare tracer_names or have an entry here.
_TRACER_MODULES_WITHOUT_TRACER_NAMES: dict[str, bool] = {
    "diagnostic_vertical_velocity": False,
    "surface_heat_salt_flux": True,
    "era5_bulk_forcing": True,
    "wpflux_surface_forcing": True,
}

#: Stepper kwargs read with a False default by :func:`_stepper_tracer_changers`;
#: ImplicitFreeSurfaceStepper also defaults both to False.
_STEPPER_TRACER_KWARGS = ("kpp_enabled", "implicit_tracer_diffusion")

_STEPPING_KEYWORD = {"temp": "tempStepping", "salt": "saltStepping"}
_IMPLICIT_ADVECTION_LEAF = {"temp": "temperature", "salt": "salinity"}


def _dynamics_class_tracer_names(cls: type) -> Any:
    """Default of ``cls.tracer_names``, or ``MISSING`` when the class has no such field."""
    if not is_dataclass(cls):
        return MISSING
    for f in fields(cls):
        if f.name == "tracer_names":
            if f.default is not MISSING:
                return f.default
            if f.default_factory is not MISSING:
                return f.default_factory()
    return MISSING


def _stepper_tracer_changers(cfg: Any, tracer: str) -> list[str]:
    """Stepper-level processes that change ``tracer`` outside the dynamics modules.

    Each is read where the stepper builder reads it: KPP from the stepper kwargs,
    the implicit tracer diffusion from the shared resolved scientific definition,
    and the implicit vertical advection from physics.implicit_vertical_advection
    (``implicit_vertical_advection_stepper_kwargs``).  KPP and the implicit
    diffusion act on both tracers, each implicit vertical advection flag on its
    own tracer (the tracer predictor of ``ImplicitFreeSurfaceStepper``).
    """
    from defgcm.io.runner._canonical_helpers import (
        implicit_vertical_advection_stepper_kwargs,
    )

    stepper_kw = dict(getattr(cfg.model.stepper, "kwargs", {}) or {})
    active: dict[str, Any] = {
        "model.stepper.kwargs.kpp_enabled": stepper_kw.get("kpp_enabled", False)
    }
    from defgcm.io.runner.config.vertical_diffusion import resolved_vertical_diffusion
    from defgcm.io.runner.legacy_bridge import TranslationError

    diffusion = resolved_vertical_diffusion(cfg, error_type=TranslationError)
    active[diffusion.implicit_sources[0]] = diffusion.implicit
    leaf = _IMPLICIT_ADVECTION_LEAF[tracer]
    active[f"physics.implicit_vertical_advection.{leaf}"] = (
        implicit_vertical_advection_stepper_kwargs(cfg)[f"{leaf}_implicit_vertical_advection"]
    )
    return [path for path, on in active.items() if bool(on)]


def _tracer_stepping(cfg: Any, tracer: str) -> bool:
    """Resolve ``tempStepping`` (``tracer="temp"``) or ``saltStepping`` (``"salt"``).

    True when a ``model.dynamics`` module lists the tracer in ``tracer_names``
    (the dataclass default when the yaml omits the kwarg); the modules test the
    same membership (``"temp" in self.tracer_names``).  False when nothing in
    LOCOMgcm changes the tracer.  Raises otherwise: LOCOMgcm then changes the
    tracer only through a surface-forcing module or the processes of
    :func:`_stepper_tracer_changers`, and neither MITgcm value runs that, since
    .FALSE. freezes the tracer and .TRUE. also advects and diffuses it.

    A config without a stepper is synthetic (no LOCOMgcm run can start from
    it) and gets MITgcm's own default, as in :func:`_mom_forcing_out_ab`.
    """
    stepper = cfg.model.stepper
    if getattr(stepper, "name", None) is None:
        return True
    changers: list[str] = []
    for where, name, cls, kwargs in _runtime_components(cfg):
        default_names = _dynamics_class_tracer_names(cls)
        if default_names is not MISSING:
            if tracer in kwargs.get("tracer_names", default_names):
                return True
        elif _TRACER_MODULES_WITHOUT_TRACER_NAMES.get(name, False):
            changers.append(where)
    changers += _stepper_tracer_changers(cfg, tracer)
    if not changers:
        return False
    keyword = _STEPPING_KEYWORD[tracer]
    raise ValueError(
        f"{keyword}: no model.dynamics module lists {tracer!r} in tracer_names, "
        f"but {', '.join(changers)} can still change {tracer} in LOCOMgcm. "
        f"MITgcm has no value for that: {keyword}=.FALSE. freezes the tracer "
        "(thermodynamics.F:314/325 skip its integration, implicit diffusion "
        f"included) and .TRUE. also advects and diffuses it. List {tracer!r} in "
        "the tracer modules' tracer_names, or remove what changes it."
    )


def _temp_stepping(cfg: Any) -> bool:
    return _tracer_stepping(cfg, "temp")


def _salt_stepping(cfg: Any) -> bool:
    return _tracer_stepping(cfg, "salt")


#: The dynamics module that implements MITgcm's bottom friction
#: (``mom_u_botdrag_coeff.F``): ``bottomDragLinear`` + ``bottomDragQuadratic``
#: + the no-slip term ``2*kappaRU*recip_drC`` (``dynamics/bottom_drag.py``).
_BOTTOM_DRAG_DYNAMICS = "bottom_drag"


def _dynamics_module_kwargs(cfg: Any, name: str) -> dict[str, Any] | None:
    """kwargs of the assembled component called ``name``, or ``None`` when LOCOMgcm has none."""
    return _component_kwargs(cfg, name)


def bottom_drag_module_kwargs(cfg: Any) -> dict[str, Any] | None:
    """kwargs of the ``bottom_drag`` dynamics entry, or ``None`` when LOCOMgcm has none."""
    return _dynamics_module_kwargs(cfg, _BOTTOM_DRAG_DYNAMICS)


def _no_slip_bottom_enabled(cfg: Any) -> bool:
    """Resolve ``no_slip_bottom`` from the bottom-drag module's own gate.

    ``BottomDrag`` adds MIT's no-slip friction ``2*viscAr*recip_drC`` to the drag
    coefficient only when ``no_slip_bottom`` (dataclass default True) **and**
    ``vertical_viscosity > 0`` (``dynamics/bottom_drag.py:303``, ``:625``); the
    stepper refuses a momentum-IVADV case that sets the first without the second
    (``steppers/implicit_free_surface.py:661-676``).  The effective LOCOMgcm switch is
    therefore the conjunction, and a config with no bottom-drag module has no
    bottom friction at all — MITgcm must get ``.FALSE.`` in both cases.
    """
    kwargs = bottom_drag_module_kwargs(cfg)
    if kwargs is None:
        return False
    vertical_viscosity = float(kwargs.get("vertical_viscosity", 0.0) or 0.0)
    return bool(kwargs.get("no_slip_bottom", True)) and vertical_viscosity > 0.0


def bottom_drag_coefficients(cfg: Any) -> tuple[float, float]:
    """``(bottomDragLinear, bottomDragQuadratic)`` as LOCOMgcm's bottom-drag module holds them.

    Not a :class:`MitSwitch` (those are logical/integer switches), but emitted by
    the ``data`` consumer next to ``no_slip_bottom`` so the three bottom-friction
    terms of ``mom_u_botdrag_coeff.F:104-127`` come from one place.  No module
    (or no config) ⇒ ``(0, 0)``: a LOCOMgcm run without bottom drag must not hand the
    reference model any.
    """
    kwargs = bottom_drag_module_kwargs(cfg) if cfg is not None else None
    if kwargs is None:
        return 0.0, 0.0
    return (
        float(kwargs.get("linear", 0.0) or 0.0),
        float(kwargs.get("quadratic", 0.0) or 0.0),
    )


def _tracer_diffusion_sum(cfg: Any, tracer: str, field_name: str) -> float:
    """Sum of ``field_name`` over the ``horizontal_tracer_diffusion`` modules listing ``tracer``.

    Each module applies its coefficients only to the tracers in its own
    ``tracer_names`` (dataclass default ``("temp",)``), and the modules'
    tendencies add.  No module listing the tracer (or no config) gives 0: a LOCOMgcm
    run that does not diffuse the tracer must not hand the reference model any.
    """
    if cfg is None:
        return 0.0
    from defgcm.dynamics.horizontal_tracer_diffusion import HorizontalTracerDiffusion

    total = 0.0
    for _where, name, _cls, kwargs in _runtime_components(cfg):
        if name != "horizontal_tracer_diffusion":
            continue
        module = HorizontalTracerDiffusion(**kwargs)
        if tracer in module.tracer_names:
            total += float(getattr(module, field_name))
    return total


def tracer_biharmonic_diffusivity(cfg: Any, tracer: str) -> float:
    """``diffK4T`` (``tracer="temp"``) or ``diffK4S`` (``"salt"``) as LOCOMgcm applies it.

    MITgcm reads diffK4T for temperature and diffK4S for salinity
    (temp_integrate.F:341,356, salt_integrate.F:339,354), each with its own zero
    default (set_defaults.F:155-156).
    """
    return _tracer_diffusion_sum(cfg, tracer, "diffusivity_k4")


def tracer_harmonic_diffusivity(cfg: Any, tracer: str) -> float:
    """``diffKhT`` (``tracer="temp"``) or ``diffKhS`` (``"salt"``) as LOCOMgcm applies it.

    The default assembly of a manifest without dynamics has no horizontal tracer
    diffusion, so it gives 0 as well.
    """
    return _tracer_diffusion_sum(cfg, tracer, "diffusivity")


# ------------------------------------------------------------ PARM01 constants
#
# rhoConst, gravity, the equation of state and the horizontal viscosity and
# diffusivity are reals the data consumer writes from the components LOCOMgcm
# assembles, each read as the component holds it (the yaml kwarg, else the class
# default; a manifest without dynamics runs the default density components).
# None of rhoConst, rhoNil, gravity or eosType is a template constant or a run
# parameter with its own default.

#: The rhoConst field of each dynamics module that holds one.  The linear EOS's
#: ``reference_density`` and the pressure gradient's ``rho_nil`` are rhoNil.
_RHO_CONST_FIELDS = {
    "hydrostatic_pressure_gradient": "reference_density",
    "jmd95_equation_of_state": "reference_density",
    "mdjwf_equation_of_state": "reference_density",
    "surface_heat_salt_flux": "reference_density",
    "surface_wind_stress": "reference_density",
    "vertical_mixing": "reference_density",
    "era5_wind_forcing": "reference_density",
    "era5_bulk_forcing": "rho_const",
    "wpflux_surface_forcing": "rho_const",
}


def _runtime_components(
    cfg: Any, reference_profiles: tuple[Any, Any] | None = None,
) -> list[tuple[str, str, type, dict[str, Any]]]:
    """``(where, name, class, kwargs)`` of the dynamics components LOCOMgcm assembles.

    Every reader of a dynamics component in this file goes through here.  A
    manifest without dynamics runs ``TorchRegionalOceanModel``'s default
    assembly, ``model.default_dynamics`` on the config of
    ``from_manifest.runtime_config_fields``; its kwargs are the built modules'
    fields.  With ``reference_profiles`` (the runtime tRef/sRef,
    ``_canonical_helpers.resolve_reference_profiles``) the modules of
    ``REFERENCE_PROFILE_MODULES`` take them unless their kwargs set them, as
    ``enrich_manifest`` does.  A config without a stepper is synthetic and has
    no default assembly.
    """
    import defgcm.dynamics  # noqa: F401  (registers every dynamics module)
    from defgcm.dynamics._registry import DYNAMICS_MODULES

    entries = tuple(cfg.model.dynamics or ())
    if not entries:
        if getattr(cfg.model.stepper, "name", None) is None:
            return []
        from defgcm.io.runner._canonical_helpers import stepper_select_r_star
        from defgcm.io.runner.builders.from_manifest import runtime_config_fields
        from defgcm.model import RegionalOceanConfig, default_dynamics

        modules = default_dynamics(RegionalOceanConfig(**runtime_config_fields(cfg)),
                                   select_r_star=stepper_select_r_star(
                                       cfg.model.stepper.name, dict(cfg.model.stepper.kwargs or {})))
        return [
            (f"default {module.name}", module.name, type(module),
             {f.name: getattr(module, f.name) for f in fields(module) if f.init})
            for module in modules
        ]
    from defgcm.io.runner._canonical_helpers import (
        REFERENCE_PROFILE_MODULES,
        tracer_vertical_scheme_default,
    )

    components = []
    for entry in entries:
        kwargs = dict(entry.kwargs or {})
        if reference_profiles is not None and entry.name in REFERENCE_PROFILE_MODULES:
            kwargs.setdefault("reference_temperature", reference_profiles[0])
            kwargs.setdefault("reference_salinity", reference_profiles[1])
        if entry.name == "horizontal_tracer_advection":
            kwargs.setdefault("vertical_scheme", tracer_vertical_scheme_default(cfg))
        try:
            cls = DYNAMICS_MODULES.get(entry.name)
        except KeyError as exc:
            raise ValueError(
                f"model.dynamics entry {entry.name!r} is not a registered dynamics module"
            ) from exc
        components.append((f"model.dynamics[{entry.name}]", entry.name, cls, kwargs))
    return components


def _component_kwargs(cfg: Any, name: str) -> dict[str, Any] | None:
    """kwargs of the assembled component called ``name``, or ``None`` when LOCOMgcm has none."""
    for _where, component, _cls, kwargs in _runtime_components(cfg):
        if component == name:
            return dict(kwargs)
    return None


def _uses_rho_const(name: str, cls: type, kwargs: dict[str, Any]) -> bool:
    """Whether the component reads its rhoConst field in this configuration.

    ``vertical_mixing`` reads ``reference_density`` only to turn a surface
    stress into the momentum flux of its momentum mixing
    (``vertical_mixing.py`` ``_momentum_tendency`` and ``evaluate``).
    """
    if name != "vertical_mixing":
        return True
    viscosity = _component_field(cls, kwargs, "viscosity")
    if viscosity is None:
        viscosity = _component_field(cls, kwargs, "diffusivity")
    return (
        bool(_component_field(cls, kwargs, "mix_momentum"))
        and float(viscosity) != 0.0
        and (float(_component_field(cls, kwargs, "surface_stress_x")) != 0.0
             or float(_component_field(cls, kwargs, "surface_stress_y")) != 0.0)
    )


def _one_value(keyword: str, values: dict[str, float]) -> float:
    """The single value every holder agrees on; MITgcm has one ``keyword``."""
    distinct = set(values.values())
    if len(distinct) != 1:
        held = ", ".join(f"{where}={value!r}" for where, value in values.items())
        raise ValueError(
            f"{keyword}: MITgcm has one value, LOCOMgcm's components hold "
            f"{held or 'none'}; make them one value"
        )
    return distinct.pop()


def rho_const_value(cfg: Any) -> float:
    """``rhoConst`` [kg m-3] as the components of ``_RHO_CONST_FIELDS`` that read it hold it."""
    values = {
        f"{where}.{_RHO_CONST_FIELDS[name]}": float(
            _component_field(cls, kwargs, _RHO_CONST_FIELDS[name]))
        for where, name, cls, kwargs in _runtime_components(cfg)
        if name in _RHO_CONST_FIELDS and _uses_rho_const(name, cls, kwargs)
    }
    return _one_value("rhoConst", values)


def gravity_value(cfg: Any) -> float:
    """``gravity`` [m s-2] as the stepper, dynamics components and boundary condition hold it.

    Every one of them whose class has a ``gravity`` field counts.
    """
    import defgcm.boundaries  # noqa: F401  (registers every boundary condition)
    import defgcm.steppers  # noqa: F401  (registers every stepper)
    from defgcm.boundaries._registry import BOUNDARY_CONDITIONS
    from defgcm.steppers._registry import STEPPERS

    holders = []
    for where, registry, entry in (("model.stepper", STEPPERS, cfg.model.stepper),
                                   ("model.boundary_condition", BOUNDARY_CONDITIONS,
                                    cfg.model.boundary_condition)):
        name = getattr(entry, "name", None)
        if name is not None:
            holders.append((where, registry.get(name), entry.kwargs))
    holders += [(where, cls, kwargs) for where, _name, cls, kwargs in _runtime_components(cfg)]
    values = {
        f"{where}.gravity": float(_component_field(cls, kwargs, "gravity"))
        for where, cls, kwargs in holders
        if is_dataclass(cls) and any(f.name == "gravity" for f in fields(cls))
    }
    return _one_value("gravity", values)


def _reference_profile(value: Any, nz: int, where: str) -> list[float]:
    """A module's reference temperature or salinity as MITgcm's ``nz`` level values."""
    import torch

    profile = torch.as_tensor(value, dtype=torch.float64).reshape(-1).tolist()
    if len(profile) == 1:
        return profile * nz
    if len(profile) != nz:
        raise ValueError(f"{where}: {len(profile)} reference levels for nz={nz}")
    return profile


def eos_namelist_values(cfg: Any, *, reference_profiles: tuple[Any, Any]) -> dict[str, Any]:
    """``eosType``; for LINEAR ``tAlpha``, ``sBeta``, ``rhoNil``, ``tRef``, ``sRef``;
    for MDJWF ``selectP_inEOS_Zc``.

    The equation-of-state module of ``EOS_TYPES`` gives the type.  Without one
    the hydrostatic pressure gradient computes its own linear density
    (``hydrostatic_pressure.py`` ``density_anomaly``), which is MITgcm LINEAR
    with that module's coefficients and reference temperature and salinity
    (find_rho.F:98-108).  ``reference_profiles`` is the runtime tRef/sRef a
    module takes when its kwargs do not set them.  rhoNil enters MITgcm only
    through LINEAR (find_rho.F:101-108,943-946; set_ref_state.F:203-208).
    """
    from defgcm.dynamics.equation_of_state import EOS_TYPES, mit_eos_type

    eos_modules = {name for _type, name, _dynamic in EOS_TYPES}
    components = {name: (where, cls, kwargs)
                  for where, name, cls, kwargs in _runtime_components(cfg, reference_profiles)}
    present = [name for name in components if name in eos_modules]
    if len(present) > 1:
        raise ValueError(f"eosType: model.dynamics declares {present}; MITgcm has one equation of state")
    if present:
        name = present[0]
        where, cls, kwargs = components[name]
        dynamic = bool(_component_field(cls, kwargs, "dynamic_pressure")) if any(
            f.name == "dynamic_pressure" for f in fields(cls)) else False
        eos_type = mit_eos_type(name, dynamic if name == "jmd95_equation_of_state" else False)
        if eos_type != "LINEAR":
            values: dict[str, Any] = {"eosType": eos_type}
            if eos_type == "MDJWF":
                # MDJWF's pressure choice is selectP_inEOS_Zc (set_parms.F:275-282,
                # pressure_for_eos.F:73-95); JMD95Z/P carry it in eosType.
                values["selectP_inEOS_Zc"] = 2 if dynamic else 0
            return values
        rho_nil = _component_field(cls, kwargs, "reference_density")
    elif "hydrostatic_pressure_gradient" in components:
        where, cls, kwargs = components["hydrostatic_pressure_gradient"]
        rho_nil = _component_field(cls, kwargs, "rho_nil")
        if rho_nil is None:
            rho_nil = _component_field(cls, kwargs, "reference_density")
    else:
        raise ValueError(
            "eosType: model.dynamics has neither an equation-of-state module nor "
            "hydrostatic_pressure_gradient, so LOCOMgcm computes no density"
        )
    nz = int(cfg.grid.nz)
    return {"eosType": "LINEAR",
            "tAlpha": float(_component_field(cls, kwargs, "thermal_expansion")),
            "sBeta": float(_component_field(cls, kwargs, "haline_contraction")),
            "rhoNil": float(rho_nil),
            "tRef": _reference_profile(
                _component_field(cls, kwargs, "reference_temperature"), nz, where),
            "sRef": _reference_profile(
                _component_field(cls, kwargs, "reference_salinity"), nz, where)}


# ------------------------------------------------------------ horizontal viscosity
#
# Not MitSwitch rows either: most of these are reals, and the ones that only
# matter when LOCOMgcm applies the term are written only then, so a case that sets
# none of them keeps a line-for-line identical ``data``.  Without them, e.g. the
# Gong viscAhGridMin floor (about 2 m^2/s against a 0.01 base) would exist only
# on the LOCOMgcm side.

#: The dynamics module that implements MITgcm's horizontal viscosity
#: (``dynamics/horizontal_diffusion.py:HorizontalViscosity``).
_HORIZONTAL_VISCOSITY_DYNAMICS = "horizontal_viscosity"

#: ``HorizontalViscosity`` kwargs of LOCOMgcm's variable-viscosity schemes
#: (``leith_smag``, ``leith_smag_flux``) that map one to
#: one onto a MITgcm PARM01 keyword of ``mom_calc_visc.F``, with the test for
#: "LOCOMgcm applies this term".  A term that is off is not written and MITgcm keeps
#: its own default: coefficients 0, useFullLeith .FALSE., grid floor 0 and grid
#: cap 1e21 (set_defaults.F:122-148, :206).  LOCOMgcm reads a grid limit <= 0 as "no
#: limit" (``_apply_grid_visc_floor_cap``), so writing a 0 cap would turn
#: LOCOMgcm's "no cap" into a MITgcm cap of zero.  LOCOMgcm clamps the biharmonic viscosity
#: only inside its biharmonic operator (``_biharmonic_active``), so the two
#: biharmonic limits are written only when that operator runs.
_VARIABLE_VISCOSITY_KEYWORDS: tuple[tuple[str, str, str], ...] = (
    ("visc_c2_leith_d", "viscC2leithD", "coefficient"),
    ("use_full_leith", "useFullLeith", "flag"),
    ("visc_c4_leith", "viscC4leith", "coefficient"),
    ("visc_c4_leith_d", "viscC4leithD", "coefficient"),
    ("visc_c4_smag", "viscC4smag", "coefficient"),
    ("visc_ah_grid_min", "viscAhGridMin", "limit"),
    ("visc_ah_grid_max", "viscAhGridMax", "limit"),
    ("visc_a4_grid_min", "viscA4GridMin", "biharmonic_limit"),
    ("visc_a4_grid_max", "viscA4GridMax", "biharmonic_limit"),
)

#: MITgcm's own sideDragFactor (set_defaults.F:135).
_MIT_SIDE_DRAG_FACTOR_DEFAULT = 2.0

#: Keywords that make MITgcm's useVariableVisc true in what the data template
#: writes (set_parms.F:132-138; the export writes none of viscAhGrid,
#: viscA4Grid, viscC2LeithQG or the viscA*file inputs).  Only then does MITgcm
#: call MOM_CALC_VISC (mom_vecinv.F:373, mom_fluxform.F:453), where the grid
#: limits act; the only other reader, the sideDragFactor<=0 side-drag form
#: (mom_u_sidedrag.F:58-76), is refused by viscosity_namelist_values.
_USE_VARIABLE_VISC_KEYWORDS = (
    "viscC2Leith",
    "viscC2smag",
    "viscC2leithD",
    "viscC4leith",
    "viscC4leithD",
    "viscC4smag",
)

#: Keywords that make MITgcm's useHarmonicVisc true in what the data template
#: writes (set_parms.F:141-145; the export writes none of viscAhD, viscAhZ,
#: viscAhGrid, viscC2LeithQG or the viscAh*file inputs).  The vector-invariant
#: momentum passes it to MOM_VI_HDISSIP (mom_vecinv.F:429-434), which skips the
#: harmonic dissipation, grid floor included, when it is false
#: (mom_vi_hdissip.F:45).  The flux form applies viscAh_D without the flag
#: (mom_fluxform.F:612-618, mom_u_xviscflux.F:53-62), and MOM_CALC_VISC applies
#: the floor to viscAh_D either way (mom_calc_visc.F:514-517).
_USE_HARMONIC_VISC_KEYWORDS = (
    "viscAh",
    "viscC2Leith",
    "viscC2smag",
    "viscC2leithD",
)

#: Where the data namelist carries each ``HorizontalViscosity`` field.
#: The keys must equal the dataclass fields, so a new LOCOMgcm viscosity knob
#: needs its MITgcm spelling here; at export time :func:`horizontal_viscosity_module` refuses a case that sets
#: an unlisted field away from its default.
_HORIZONTAL_VISCOSITY_CARRIERS: dict[str, str] = {
    "viscosity": "viscAh, horizontal_viscosity_value unless mit_comparison overrides it on purpose",
    "no_slip_sides": "no_slip_sides, from run_params; checked",
    "viscosity_scheme": "selects the template's Leith/Smag block",
    "visc_c2_leith": "viscC2Leith, from run_params; checked",
    "visc_c2_smag": "viscC2smag, from run_params; checked",
    "viscosity_a4": "viscA4, from mit_comparison; checked for every scheme (the cube refuses a nonzero value)",
    "side_drag_factor": "sideDragFactor",
    **{to_name: mit_name for to_name, mit_name, _ in _VARIABLE_VISCOSITY_KEYWORDS},
    "dt": "must equal deltaTmom while a grid limit acts",
    "name": "module identity",
    "stepper_domain": "assembly placement",
}


def horizontal_viscosity_module(cfg: Any) -> Any | None:
    """The ``HorizontalViscosity`` LOCOMgcm builds for this case, or ``None`` without one.

    Built from the same yaml kwargs ``from_manifest._instantiate_dynamics``
    passes, so every read below sees the dataclass defaults LOCOMgcm runs with.
    Raises for a field the data namelist has no carrier for, set away from its
    default: LOCOMgcm would apply it and the reference model would not.
    """
    kwargs = (
        _dynamics_module_kwargs(cfg, _HORIZONTAL_VISCOSITY_DYNAMICS) if cfg is not None else None
    )
    if kwargs is None:
        return None
    from defgcm.dynamics.horizontal_viscosity import HorizontalViscosity

    try:
        module = HorizontalViscosity(**kwargs)
    except TypeError as exc:
        raise ValueError(f"model.dynamics.horizontal_viscosity.kwargs: {exc}") from exc
    uncarried = [
        f"{f.name}={getattr(module, f.name)!r}"
        for f in fields(module)
        if f.name not in _HORIZONTAL_VISCOSITY_CARRIERS and getattr(module, f.name) != f.default
    ]
    if uncarried:
        raise ValueError(
            "horizontal_viscosity sets " + ", ".join(uncarried) + ", which the "
            "MITgcm data namelist does not carry; LOCOMgcm would apply it and "
            "the reference model would not. Give it a carrier in "
            "layer_d/_mit_switches.py (_HORIZONTAL_VISCOSITY_CARRIERS) first."
        )
    return module


def horizontal_viscosity_value(cfg: Any) -> float:
    """``viscAh`` as LOCOMgcm's ``horizontal_viscosity`` module holds it; 0 without one.

    The default assembly of a manifest without dynamics has no horizontal
    viscosity.
    """
    kwargs = _dynamics_module_kwargs(cfg, _HORIZONTAL_VISCOSITY_DYNAMICS)
    if kwargs is None:
        return 0.0
    from defgcm.dynamics.horizontal_viscosity import HorizontalViscosity

    return float(HorizontalViscosity(**kwargs).viscosity)


def viscosity_namelist_values(cfg: Any) -> dict[str, float | bool]:
    """PARM01 keywords the data template adds for LOCOMgcm's horizontal viscosity.

    ``sideDragFactor`` when LOCOMgcm's factor differs from MITgcm's default, and each
    :data:`_VARIABLE_VISCOSITY_KEYWORDS` term that LOCOMgcm's ``leith_smag`` or
    ``leith_smag_flux`` scheme applies.  Outside them LOCOMgcm ignores those kwargs
    (``HorizontalViscosity.evaluate``), so nothing is written for them.  Empty
    without a horizontal_viscosity module.  ``useAreaViscLength`` is never
    written: LOCOMgcm's limiter lengths are MITgcm's default ``.FALSE.`` form
    (``_grid_clamp_length_scales``, mom_init_fixed.F:96-129).
    """
    module = horizontal_viscosity_module(cfg)
    if module is None:
        return {}
    values: dict[str, float | bool] = {}
    factor = float(module.side_drag_factor)
    if bool(module.no_slip_sides) and factor <= 0.0:
        raise ValueError(
            f"horizontal_viscosity.side_drag_factor={factor!r} with no_slip_sides: "
            "MITgcm reads sideDragFactor<=0 as its old side-drag form, a constant "
            "viscosity times a fixed 2 (mom_u_sidedrag.F:58-96), while LOCOMgcm "
            "multiplies its drag by the factor itself. The namelist cannot express "
            "that."
        )
    if factor != _MIT_SIDE_DRAG_FACTOR_DEFAULT:
        values["sideDragFactor"] = factor
    from defgcm.dynamics.horizontal_viscosity import VARIABLE_VISCOSITY_SCHEMES

    if module.viscosity_scheme not in VARIABLE_VISCOSITY_SCHEMES:
        return values
    biharmonic = module._biharmonic_active()
    for to_name, mit_name, kind in _VARIABLE_VISCOSITY_KEYWORDS:
        value = getattr(module, to_name)
        if kind == "flag":
            if bool(value):
                values[mit_name] = True
        elif kind == "limit" or (kind == "biharmonic_limit" and biharmonic):
            if float(value) > 0.0:
                values[mit_name] = float(value)
        elif kind == "coefficient" and float(value) != 0.0:
            values[mit_name] = float(value)
    return values


def _namelist_real(value: float) -> str:
    """Shortest text that reads back as exactly ``value``, Fortran spelling."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"a MITgcm namelist real must be finite, got {value!r}")
    return repr(value).replace("e", "E")


def render_viscosity_lines(run_cfg: Any) -> str:
    """Render :func:`viscosity_namelist_values` as PARM01 lines, in table order."""
    return "".join(
        f" {name}={fortran_logical(value) if type(value) is bool else _namelist_real(value)},\n"
        for name, value in viscosity_namelist_values(run_cfg).items()
    )


def _namelist_values(text: str, keyword: str) -> list[float | bool]:
    """Every value ``text`` assigns to ``keyword``; ``#`` comment lines are skipped."""
    values: list[float | bool] = []
    for raw in re.findall(
        rf"^[ \t]*{re.escape(keyword)}[ \t]*=[ \t]*([^,\n]+?)[ \t]*,",
        text,
        re.MULTILINE | re.IGNORECASE,
    ):
        logical = re.fullmatch(r"\.(TRUE|FALSE)\.", raw, re.IGNORECASE)
        if logical is not None:
            values.append(logical.group(1).upper() == "TRUE")
        else:
            values.append(float(raw.replace("D", "E").replace("d", "e")))
    return values


def assert_viscosity_agrees(run_cfg: Any, text: str) -> None:
    """Assembly-time gate: the written ``data`` holds LOCOMgcm's horizontal viscosity.

    Re-parses the text, like :func:`assert_switches_agree`, and raises unless

    * each :func:`viscosity_namelist_values` keyword appears once with LOCOMgcm's value
      and the keywords of terms LOCOMgcm does not apply are absent;
    * viscC2Leith and viscC2smag equal the coefficients LOCOMgcm runs (zero outside
      ``leith_smag`` and ``leith_smag_flux``).  The template takes them from run_params, which falls
      back to ``physics.*`` when the module omits them;
    * no_slip_sides is the module's own value.  The module defaults it to True
      while run_params falls back to ``physics.no_slip_sides`` (False) when the
      module kwargs omit it;
    * ``leith_smag_flux`` runs with flux-form momentum, whose stencil it is;
    * a biharmonic base LOCOMgcm applies under ``leith_smag`` is the namelist's viscA4;
    * a grid limit LOCOMgcm applies also acts in MITgcm: MITgcm applies the limits
      inside MOM_CALC_VISC, which it calls only with a nonzero Leith/Smag
      keyword (:data:`_USE_VARIABLE_VISC_KEYWORDS`), and scales them with
      deltaTmom (mom_calc_visc.F:167, mom_init_fixed.F:43) where LOCOMgcm uses the
      module's dt; under vectorInvariantMomentum a harmonic limit also needs
      MITgcm's harmonic dissipation to run (:data:`_USE_HARMONIC_VISC_KEYWORDS`).

    viscAh is not compared: ``mit_comparison``'s overrides exist to give the
    reference model a different base viscosity on purpose; without one the
    export writes ``horizontal_viscosity_value``.
    """
    module = horizontal_viscosity_module(run_cfg)
    if module is None:
        return
    problems: list[str] = []
    found_sides = _namelist_values(text, "no_slip_sides")
    if found_sides != [bool(module.no_slip_sides)]:
        problems.append(
            f"no_slip_sides: data says {found_sides or 'nothing'}, LOCOMgcm's "
            f"horizontal_viscosity runs no_slip_sides={module.no_slip_sides!r}; the "
            "module defaults it to True while run_params falls back to "
            "physics.no_slip_sides, so declare it in the module kwargs"
        )
    expected = viscosity_namelist_values(run_cfg)
    for mit_name in ("sideDragFactor", *(m for _, m, _ in _VARIABLE_VISCOSITY_KEYWORDS)):
        found = _namelist_values(text, mit_name)
        want = expected.get(mit_name)
        if want is None and found:
            problems.append(
                f"{mit_name}: data says {found} but LOCOMgcm does not apply that term"
            )
        elif want is not None and found != [want]:
            problems.append(
                f"{mit_name}: data says {found or 'nothing'}, LOCOMgcm applies {want!r}"
            )
    from defgcm.dynamics.horizontal_viscosity import VARIABLE_VISCOSITY_SCHEMES

    leith = module.viscosity_scheme in VARIABLE_VISCOSITY_SCHEMES
    if (module.viscosity_scheme == "leith_smag_flux"
            and _namelist_values(text, "vectorInvariantMomentum") == [True]):
        problems.append(
            "viscosity_scheme 'leith_smag_flux' is the flux-form stencil "
            "(mom_fluxform.F:612-618), but data sets vectorInvariantMomentum, under "
            "which MITgcm applies the div-vort form (mom_vi_hdissip.F); use 'leith_smag'"
        )
    for to_name, mit_name in (("visc_c2_leith", "viscC2Leith"), ("visc_c2_smag", "viscC2smag")):
        want = float(getattr(module, to_name)) if leith else 0.0
        found = _namelist_values(text, mit_name)
        if found != [want]:
            problems.append(
                f"{mit_name}: data says {found or 'nothing'}, LOCOMgcm runs {want!r} "
                f"(horizontal_viscosity.{to_name}, scheme {module.viscosity_scheme!r})"
            )
    # Every scheme applies the constant biharmonic viscosity (visc/17-18).
    visc_a4 = float(module.viscosity_a4)
    found_a4 = _namelist_values(text, "viscA4")
    if visc_a4 != 0.0 and found_a4 != [visc_a4]:
        problems.append(
            f"viscA4: data says {found_a4 or 'nothing'}, LOCOMgcm's biharmonic "
            f"base is horizontal_viscosity.viscosity_a4={visc_a4!r}; set "
            "mit_comparison.viscA4 to the same value"
        )
    if leith:
        # When LOCOMgcm applies each limit: HorizontalViscosity._apply_grid_visc_floor_cap
        # and, inside the biharmonic operator only, _apply_grid_visca4_floor_cap.
        harmonic_limit = module.visc_ah_grid_min > 0.0 or module.visc_ah_grid_max > 0.0
        biharmonic_limit = module._biharmonic_active() and (
            module.visc_a4_grid_min > 0.0 or module.visc_a4_grid_max > 0.0
        )
        if harmonic_limit or biharmonic_limit:
            if not any(
                value != 0.0
                for name in _USE_VARIABLE_VISC_KEYWORDS
                for value in _namelist_values(text, name)
            ):
                problems.append(
                    "grid viscosity limits: LOCOMgcm applies them, but no "
                    "Leith/Smagorinsky keyword is nonzero, so MITgcm's "
                    "useVariableVisc is false (set_parms.F:132-138) and it never "
                    "calls MOM_CALC_VISC, where the limits act"
                )
            if (
                harmonic_limit
                and _namelist_values(text, "vectorInvariantMomentum") == [True]
                and not any(
                    value != 0.0
                    for name in _USE_HARMONIC_VISC_KEYWORDS
                    for value in _namelist_values(text, name)
                )
            ):
                problems.append(
                    "harmonic grid limits: LOCOMgcm applies them, but viscAh and "
                    "every harmonic Leith/Smagorinsky keyword are zero, so MITgcm's "
                    "useHarmonicVisc is false (set_parms.F:141-145) and its "
                    "vector-invariant momentum skips the harmonic dissipation, "
                    "floor included (mom_vi_hdissip.F:45)"
                )
            found_dt = _namelist_values(text, "deltaTmom")
            if found_dt != [float(module.dt)]:
                problems.append(
                    f"grid viscosity limits: LOCOMgcm scales them with "
                    f"horizontal_viscosity.dt={module.dt!r}, MITgcm with deltaTmom="
                    f"{found_dt or 'nothing'}"
                )
    if problems:
        raise ValueError(
            "MIT/TO horizontal viscosity divergence while writing data: the "
            "reference model would not run the viscosity LOCOMgcm runs.\n  "
            + "\n  ".join(problems)
        )


# ------------------------------------------------------------ Adams-Bashforth
#
# PARM03 abEps is a real, so it is not a MitSwitch row either.  A constant 0.1
# in the data template would disagree with
# scs_nscs_500m_hycom_2014_gong_mitcheck, which runs the stepper at ab_eps 0.02.

#: ``ImplicitFreeSurfaceStepper.ab_eps`` field default, i.e. what the stepper
#: runs when the case YAML omits the kwarg. Keep this equal to the dataclass default.
_IFS_AB_EPS_DEFAULT = 0.1


def ab_eps_value(cfg: Any) -> float:
    """``abEps`` as LOCOMgcm's stepper runs it: ``model.stepper.kwargs.ab_eps``.

    Both models extrapolate a tendency as ``G + (0.5 + eps) * (G - G_prev)``
    and use the plain tendency on the first step of a cold start.  MITgcm does
    it in ADAMS_BASHFORTH2 (adams_bashforth2.F:61-65, :84-86), called for
    momentum, for the tracers it steps with AB on the tendency and, when
    nonhydrostatic, for gW; LOCOMgcm in the stepper's momentum and tracer AB2
    and in ``compute_gw``.  For ``ab_eps <= 0`` the stepper skips the momentum
    and tracer extrapolation altogether (its ``self.ab_eps > 0.0`` gates) and
    ``compute_gw`` takes no gW extrapolation either; MITgcm runs that scheme with
    ``abEps = -0.5`` (``abFac = 0``), not with ``abEps = ab_eps``, and this
    exporter writes the stepper's value, so the value is refused.  A config
    without the kwarg, or no config at all, gets the stepper's default.
    """
    stepper = getattr(getattr(cfg, "model", None), "stepper", None)
    stepper_kw = dict(getattr(stepper, "kwargs", {}) or {})
    value = float(stepper_kw.get("ab_eps", _IFS_AB_EPS_DEFAULT))
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(
            f"model.stepper.kwargs.ab_eps={value!r}: for ab_eps <= 0 LOCOMgcm "
            "steps momentum, tracers and gW forward in time without the Adams-Bashforth "
            "extrapolation, which MITgcm runs with abEps = -0.5 (abFac = 0.5 + abEps, "
            "adams_bashforth2.F:64), not with abEps = ab_eps as this exporter writes. "
            "Use ab_eps > 0 for a case exported to MITgcm."
        )
    return value


def assert_ab_eps_agrees(run_cfg: Any, text: str) -> None:
    """Assembly-time gate: the written ``data`` assigns abEps once, LOCOMgcm's value.

    Re-parses the text like :func:`assert_viscosity_agrees`, so a template that
    writes its own constant instead of the value passed to it fails the export.
    """
    want = ab_eps_value(run_cfg)
    found = _namelist_values(text, "abEps")
    if found != [want]:
        raise ValueError(
            f"MIT/TO Adams-Bashforth divergence while writing data: abEps is "
            f"{found or 'absent'}, LOCOMgcm's stepper runs ab_eps={want!r} "
            "(model.stepper.kwargs.ab_eps, or the stepper default when omitted)."
        )


# EXF_NML_01 windstressmax is a real, so it is not a MitSwitch row either.  If
# data.exf did not write it, MITgcm would clip the A-grid stress at its own
# default 2 N m-2 (exf_mapfields.F:216-233, exf_readparms.F:601) while LOCOMgcm
# applies no clip at all.

#: The dynamics module that consumes the stress EXF would read.
_WIND_FORCING_DYNAMICS = "era5_wind_forcing"

#: ``ERA5WindForcing.windstressmax`` field default, i.e. what the module runs when
#: the case YAML omits the kwarg. Keep this equal to the dataclass default.
_ERA5_WINDSTRESSMAX_DEFAULT = 2.0

#: data.exf value for a module run without the clip (``windstressmax: .inf``).
#: MITgcm has no switch to turn the clip off; no finite stress exceeds the
#: largest double, so exf_mapfields.F leaves every finite value alone.
_WINDSTRESSMAX_OFF = sys.float_info.max


def windstressmax_namelist_value(value: Any) -> float:
    """The data.exf value for the module's ``windstressmax`` (``inf`` = no clip)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0.0:
        raise ValueError(
            f"era5_wind_forcing windstressmax={value!r}: expected a positive "
            "stress bound [N m-2], or .inf to turn the clip off"
        )
    return _WINDSTRESSMAX_OFF if math.isinf(value) else float(value)


def windstressmax_value(cfg: Any) -> float:
    """EXF ``windstressmax`` as LOCOMgcm's wind module runs it.

    Read from ``model.dynamics[era5_wind_forcing].kwargs.windstressmax``; a module
    that omits the kwarg, a config without the module, or no config at all gets
    the module default.  ``inf`` (the clip turned off) is written as the largest
    double.
    """
    kwargs = (
        _dynamics_module_kwargs(cfg, _WIND_FORCING_DYNAMICS)
        if getattr(getattr(cfg, "model", None), "dynamics", None) is not None
        else None
    )
    return windstressmax_namelist_value(
        (kwargs or {}).get("windstressmax", _ERA5_WINDSTRESSMAX_DEFAULT)
    )


def render_windstressmax(value: float) -> str:
    """``windstressmax`` in the spelling both data.exf writers use."""
    return _namelist_real(value)


def assert_windstressmax_agrees(run_cfg: Any, text: str) -> None:
    """Assembly-time gate: the written ``data.exf`` assigns windstressmax once.

    Re-parses the text like :func:`assert_ab_eps_agrees`, so a template that drops
    the keyword (MITgcm would then clip at its default) or writes a constant
    fails the export.
    """
    want = windstressmax_value(run_cfg)
    found = _namelist_values(text, "windstressmax")
    if found != [want]:
        raise ValueError(
            f"MIT/TO wind-stress clip divergence while writing data.exf: "
            f"windstressmax is {found or 'absent'}, LOCOMgcm's era5_wind_forcing "
            f"clips at {want!r} (model.dynamics[era5_wind_forcing].kwargs."
            "windstressmax, or the module default when omitted)."
        )


def _mom_forcing_out_ab(cfg: Any) -> int:
    """Resolve ``momForcingOutAB`` from the stepper's own regime declaration.

    MITgcm reaches this integer two ways: written directly, or resolved from the
    single ``forcing_In_AB`` logical (``ini_parms.F:1095-1097``).  The export used
    to take the second route with a hardcoded ``.FALSE.``, which pins
    ``momForcingOutAB`` **and** ``tracForcingOutAB`` together — see the
    ``tracForcingOutAB`` row for why that coupling is unrepresentable here.

    ``ImplicitFreeSurfaceStepper.mom_forcing_out_ab`` has no behavioural default
    (``implicit_free_surface.py:899``): ``__post_init__`` demands an explicit
    declaration, because the two regimes place momentum forcing on opposite sides
    of the AB2 extrapolation (``timestep.F:139/209/238``).  So this reader refuses
    to guess for a config that names a stepper and then omits the kwarg — LOCOMgcm would
    refuse to construct it, and picking a value here would hand the reference model
    a regime nobody chose.  A config with no stepper at all is synthetic (no LOCOMgcm run
    can start from it) and falls back to MITgcm's own default.
    """
    stepper = cfg.model.stepper
    declared = dict(getattr(stepper, "kwargs", {}) or {}).get("mom_forcing_out_ab")
    if declared is not None:
        return 1 if bool(declared) else 0
    if getattr(stepper, "name", None) is None:
        return 0  # MITgcm's own default, via forcing_In_AB=.TRUE.
    raise ValueError(
        "model.stepper.kwargs.mom_forcing_out_ab is not declared, so the momentum "
        "Adams-Bashforth forcing regime is undefined. LOCOMgcm refuses to build "
        "the stepper without it (implicit_free_surface.py:899) and the MITgcm "
        "export cannot invent a momForcingOutAB for the reference model either. "
        "Declare it in the case yaml: true == MIT momForcingOutAB=1 (forcing added "
        "after AB2), false == 0 (forcing inside the AB2 history)."
    )


#: Readers for switches whose LOCOMgcm-side value is not a single ``RunConfig``
#: attribute.  Keyed by the pseudo-``to_path`` used in :data:`SWITCHES`.
#:
#: These are not indirection for its own sake: each one encodes a real fact about
#: where the LOCOMgcm runtime actually reads the behaviour from, which is *not* always a
#: yaml leaf.  Writing them here keeps that knowledge next to the MIT default it
#: must agree with.  Membership in this mapping is what makes a switch *derived*
#: (:func:`is_derived`), which in turn is what obliges the gate to prove the
#: predicate with an on/off control pair rather than a single default check.
_CUSTOM_READERS = {
    "model_boundary_condition_obcs_balance": _obcs_balance_enabled,
    "model_stepper_kpp_enabled": _kpp_enabled,
    "boundary_sponge_enabled": _sponge_enabled,
    "boundary_prescribe_enabled": _prescribe_enabled,
    "model_stepper_mom_forcing_out_ab": _mom_forcing_out_ab,
    "dynamics_bottom_drag_no_slip": _no_slip_bottom_enabled,
    "model_stepper_exact_conserv": _exact_conserv_enabled,
    "model_stepper_stagger_tracer_stepping": _stagger_time_step_enabled,
    "dynamics_temp_stepping": _temp_stepping,
    "dynamics_salt_stepping": _salt_stepping,
}


#: Switches whose value is emitted from ``run_cfg`` rather than hardcoded.
#:
#: Adding a row here is what makes a switch *configurable*.  Writers must
#: represent enabled switches here or document their fixed policy explicitly.
#:
#: Row order is the namelist emission order, kept identical to the original
#: hardcoded block so that a case whose switches all resolve to the historical
#: values still produces a line-for-line identical ``data.obcs``.
SWITCHES: tuple[MitSwitch, ...] = (
    MitSwitch(
        mit_name="OBCSfixTopo",
        namelist="data.obcs",
        mit_default=False,
        mit_source="pkg/obcs/obcs_readparms.F:291",
        to_path="boundary.obcs_fix_topo",
        to_default=False,
        rationale=(
            "Shallows open-boundary strip columns to min(self, interior neighbour) "
            "before hFac/metrics. TO implements the identical transform in "
            "defgcm/geometry/masks.py:obcs_fix_topo_and_clamp, so the two models "
            "build different bathymetry whenever the flags disagree. The entire body "
            "of pkg/obcs/obcs_check_depths.F sits inside IF (OBCSfixTopo) at :31."
        ),
    ),
    MitSwitch(
        mit_name="useOBCSprescribe",
        namelist="data.obcs",
        mit_default=False,
        mit_source="pkg/obcs/obcs_readparms.F:267",
        to_path="boundary_prescribe_enabled",  # derived; see _CUSTOM_READERS
        to_default=False,
        rationale=(
            "MIT overwrites boundary u/v/T/S from the OB*File binaries each record. "
            "TO's counterpart is the boundary condition itself: PrescribedBoundary3D "
            "(boundaries/open.py:1496-1502, whose docstring names this switch) and "
            "tracer_open are exactly the two that require an obcs_source at build "
            "time (io/runner/builders/from_manifest.py:121-127); every other "
            "registered condition lets the boundary free-evolve, which is what "
            "useOBCSprescribe=.FALSE. means. Note boundary.scheme is NOT this "
            "predicate: it maps to the legacy tracer-boundary-scheme "
            "(io/runner/legacy_bridge.py:202-209) and says nothing about whether the "
            "OB strip is overwritten."
        ),
    ),
    MitSwitch(
        mit_name="useOBCSbalance",
        namelist="data.obcs",
        mit_default=False,
        mit_source="pkg/obcs/obcs_readparms.F:283",
        to_path="model_boundary_condition_obcs_balance",  # derived; see _CUSTOM_READERS
        to_default=False,
        rationale=(
            "MIT obcs_balance_flow.F enforces zero net inflow by a uniform velocity "
            "correction. TO implements it in boundaries/open.py _obcs_balance_flow "
            "(obcs_balance_flow.F:347-456), switched by PrescribedBoundary3D."
            "enforce_obcs_balance and FlatherBarotropicBoundaryCondition.balance_obc "
            "(InternalWaveOpenBoundary declares enforce_obcs_balance and accepts only "
            "False); the reader takes that value and gives False for a boundary "
            "condition that never balances."
        ),
    ),
    MitSwitch(
        mit_name="useOBCSsponge",
        namelist="data.obcs",
        mit_default=False,
        mit_source="pkg/obcs/obcs_readparms.F:269",
        to_path="boundary_sponge_enabled",  # derived; see _CUSTOM_READERS
        to_default=False,
        rationale=(
            "MIT relaxes u/v/T/S inside spongeThickness cells of each open boundary. "
            "TO builds that band only when a momentum/tracer boundary-sponge dynamics "
            "module is present, which is the same predicate "
            "_assert_obcs_sponge_consistency already keys off "
            "(consumers/mit_binary.py:1421-1424) when it cross-checks the relax "
            "timescales written into this very namelist. boundary.sponge_width is the "
            "band *width* fed to spongeThickness, not the on/off switch: a case can "
            "leave the schema default 3 in place and still build no sponge module."
        ),
    ),
    MitSwitch(
        mit_name="useKPP",
        namelist="data.pkg",
        mit_default=False,
        mit_source="model/src/packages_boot.F:121",
        to_path="model_stepper_kpp_enabled",  # derived; see _CUSTOM_READERS
        to_default=False,
        rationale=(
            "MITgcm executes pkg/kpp only when useKPP is enabled. LOCOMgcm "
            "executes the corresponding per-step mixing work when "
            "model.stepper.kwargs.kpp_enabled is true, so benchmark exports must "
            "derive the package switch from the same gate to compare comparable "
            "steady-state step workloads."
        ),
    ),
    # ---------------------------------------------------------------- `data`
    # PARM03 Adams-Bashforth forcing placement.
    #
    # 🔴 These are emitted as the two integers rather than via MITgcm's
    # ``forcing_In_AB`` logical **on purpose**.  That logical is a convenience
    # that resolves BOTH integers at once (ini_parms.F:1095-1101), so it cannot
    # express LOCOMgcm's actual configuration: LOCOMgcm's momentum side is a per-case
    # declaration while its tracer side is hardcoded post-AB2.  A hardcoded
    # ``forcing_In_AB=.FALSE.`` would give both LOCOMgcm's values only when the
    # case declares ``mom_forcing_out_ab: true``.  All three names are PARM03 namelist
    # variables (ini_parms.F:279), so writing the integers costs nothing.
    MitSwitch(
        mit_name="momForcingOutAB",
        namelist="data",
        mit_default=0,
        mit_source="model/src/ini_parms.F:1095-1097 (resolved from forcing_In_AB=.TRUE. @:969)",
        to_path="model_stepper_mom_forcing_out_ab",  # derived; see _CUSTOM_READERS
        to_default=0,
        rationale=(
            "1 keeps momentum forcing out of the guNm1/gvNm1 AB2 history and adds "
            "it afterwards; 0 lets it enter the history (timestep.F:139/209/238). "
            "TO declares the regime per case in "
            "model.stepper.kwargs.mom_forcing_out_ab (implicit_free_surface.py:899) "
            "with no behavioural default, so a hardcoded export value is a silent "
            "cross-model divergence for any case that picks the other regime — "
            "scs_clm_hotstart.yaml:563 already declares false."
        ),
    ),
    MitSwitch(
        mit_name="tracForcingOutAB",
        namelist="data",
        mit_default=0,
        mit_source="model/src/ini_parms.F:1099-1101 (resolved from forcing_In_AB=.TRUE. @:969)",
        to_path=None,
        to_default=1,
        rationale=(
            "1 adds the surface tracer forcing after the Adams-Bashforth call, 0 "
            "before it (temp_integrate.F:368 vs :398, salt_integrate.F:366 vs "
            ":396). LOCOMgcm has no knob here: the post-AB2 placement is "
            "hardcoded across the tracer path — surface_flux.py:529, "
            "tracer_sponge.py:68, era5_bulk_forcing.py:69, model.py:644 all cite "
            "tracForcingOutAB=1 as the regime they implement. Note this is NOT "
            "stepper.ab2_on_tracer_tendency, which corresponds to MIT's "
            "AdamsBashforthGt (whether the AB call happens at all) and leaves the "
            "forcing placement untouched."
        ),
        default_divergence_note=(
            "TO implements post-AB2 tracer forcing unconditionally while MITgcm "
            "defaults to in-AB. Emitting MIT's default would make the reference "
            "model add gtForc on the other side of the AB2 extrapolation, so the "
            "export states TO's regime explicitly. This is exactly why the two "
            "integers are written separately instead of via forcing_In_AB: that "
            "single logical cannot give momForcingOutAB=0 and tracForcingOutAB=1 "
            "at the same time."
        ),
    ),
    MitSwitch(
        mit_name="no_slip_bottom",
        namelist="data",
        mit_default=True,
        mit_source="model/src/set_defaults.F:133",
        to_path="dynamics_bottom_drag_no_slip",  # derived; see _CUSTOM_READERS
        to_default=False,
        rationale=(
            "MITgcm adds the no-slip bottom friction 2*kappaRU*recip_drC to the "
            "bottom drag coefficient when no_slip_bottom is set "
            "(mom_u_botdrag_coeff.F:68,122-127). LOCOMgcm's BottomDrag module "
            "implements the same term from its own no_slip_bottom and "
            "vertical_viscosity kwargs (dynamics/bottom_drag.py:285-308). If the "
            "data consumer did not pass the switch, _write_data's default would "
            "write .FALSE. for every case while LOCOMgcm cases with "
            "vertical_viscosity>0 apply the friction: the reference model would run "
            "with less bottom friction than LOCOMgcm on every shelf."
        ),
        default_divergence_note=(
            "A RunConfig that declares no bottom_drag module has no bottom friction "
            "in LOCOMgcm at all, and BottomDrag's own vertical_viscosity default is 0.0, "
            "which makes its no_slip_bottom=True default inert (bottom_drag.py:303). "
            "MITgcm defaults to .TRUE. because its viscAr is always a live "
            "coefficient; emitting that default would give the reference model a "
            "friction term LOCOMgcm does not have."
        ),
    ),
    # PARM01 time-stepping switches, written from the stepper kwargs rather than
    # as .TRUE. constants.
    MitSwitch(
        mit_name="exactConserv",
        namelist="data",
        mit_default=False,
        mit_source="model/src/set_defaults.F:254",
        to_path="model_stepper_exact_conserv",  # derived; see _CUSTOM_READERS
        to_default=True,
        rationale=(
            "With exactConserv MITgcm recomputes etaN from the divergence of the "
            "corrected transport (integr_continuity.F:90,207,316,325), derives "
            "calc_wVelocity from it (set_parms.F:105) and chooses the free-surface "
            "term of the nonhydrostatic solve (solve_for_pressure.F:90,274-277,560). "
            "LOCOMgcm's ImplicitFreeSurfaceStepper.exact_conserv gates the same "
            "eta recompute after the velocity correction and the same "
            "old_free_surf_term choice, so the two models integrate eta differently "
            "whenever the flags disagree."
        ),
        default_divergence_note=(
            "ImplicitFreeSurfaceStepper defaults exact_conserv=True while MITgcm "
            "defaults .FALSE.. The export writes TO's resolved value, so a case "
            "that omits the kwarg gets .TRUE. on both sides; a case compared "
            "against a MITgcm run without exactConserv declares exact_conserv: "
            "false in its own yaml."
        ),
    ),
    MitSwitch(
        mit_name="staggerTimeStep",
        namelist="data",
        mit_default=False,
        mit_source="model/src/set_defaults.F:183",
        to_path="model_stepper_stagger_tracer_stepping",  # derived; see _CUSTOM_READERS
        to_default=False,
        rationale=(
            "MITgcm steps the tracers before the momentum solve when "
            "staggerTimeStep is off (forward_step.F:724) and after the momentum "
            "correction when it is on (forward_step.F:975); with a nonlinear free "
            "surface and real freshwater flux it also sets PmEpR to the current "
            "step's -EmPmR, where synchronous stepping keeps the previous "
            "continuity's (external_forcing_surf.F:158-166), and it reorders the halo "
            "exchanges (do_fields_blocking_exchanges.F:52, "
            "do_stagger_fields_exchanges.F:37). LOCOMgcm's "
            "ImplicitFreeSurfaceStepper.stagger_tracer_stepping selects the same "
            "tracer phase and feeds AssemblyPolicy.stagger_tracer_stepping. With "
            "a .TRUE. constant, a case omitting the kwarg would run LOCOMgcm "
            "synchronous and MITgcm staggered with no signal."
        ),
    ),
    # PARM01 tracer equations, written from the tracer modules' tracer names
    # rather than as .TRUE. constants.
    MitSwitch(
        mit_name="tempStepping",
        namelist="data",
        mit_default=True,
        mit_source="model/src/set_defaults.F:194",
        to_path="dynamics_temp_stepping",  # derived; see _CUSTOM_READERS
        to_default=True,
        rationale=(
            "With tempStepping off MITgcm skips TEMP_INTEGRATE (thermodynamics.F:314) "
            "and the implicit vertical diffusion inside it (temp_integrate.F:499), "
            "and clears tempAdvection/tempVertDiff4/tempForcing "
            "(set_parms.F:242-244). LOCOMgcm steps temperature through the "
            "dynamics modules whose tracer_names contain 'temp', the form the "
            "MITgcm-to-LOCOMgcm translator gives tempStepping, so the reader "
            "resolves .TRUE. when such a module exists, .FALSE. when nothing "
            "changes temperature, and refuses a case where only KPP, the implicit "
            "tracer diffusion, the temperature implicit vertical advection or a "
            "surface-forcing module could change it."
        ),
    ),
    MitSwitch(
        mit_name="saltStepping",
        namelist="data",
        mit_default=True,
        mit_source="model/src/set_defaults.F:198",
        to_path="dynamics_salt_stepping",  # derived; see _CUSTOM_READERS
        to_default=True,
        rationale=(
            "With saltStepping off MITgcm skips SALT_INTEGRATE (thermodynamics.F:325) "
            "and the implicit vertical diffusion inside it (salt_integrate.F:497), "
            "clears saltAdvection/saltVertDiff4/saltForcing (set_parms.F:245-247) "
            "and, with tempStepping also off, drops GAD (packages_boot.F:236). "
            "LOCOMgcm steps salinity through the dynamics modules whose "
            "tracer_names contain 'salt'; the reader resolves it as for "
            "tempStepping, with the salinity implicit vertical advection flag."
        ),
    ),
)


def is_derived(switch: MitSwitch) -> bool:
    """True when the LOCOMgcm-side value is computed from several native fields.

    A derived switch cannot be checked by comparing one schema default against
    MITgcm's, so the gate demands an on/off control pair instead.
    """
    return switch.to_path in _CUSTOM_READERS


def fortran_logical(value: bool) -> str:
    """Render a python bool as a MITgcm namelist logical.

    Rejects non-bool input outright: a truthy string or an int silently
    formatting as ``.TRUE.`` is precisely the kind of quiet mistake this module
    exists to prevent.
    """
    if type(value) is not bool:
        raise TypeError(
            f"MITgcm logical must be a bool, got {type(value).__name__}: {value!r}"
        )
    return ".TRUE." if value else ".FALSE."


def fortran_value(value: bool | int) -> str:
    """Render a switch value the way its MITgcm namelist entry spells it.

    ``bool`` -> logical, ``int`` -> bare integer.  ``bool`` is checked first
    because it subclasses ``int``: without that order every logical would be
    emitted as ``0``/``1``, which MITgcm's namelist reader rejects for a LOGICAL.
    """
    if type(value) is bool:
        return fortran_logical(value)
    if type(value) is int:
        return str(value)
    raise TypeError(
        f"MITgcm namelist value must be a bool or int, got "
        f"{type(value).__name__}: {value!r}"
    )


def switch_value(run_cfg: Any, switch: MitSwitch) -> bool:
    """Resolve one switch's value from ``run_cfg``.

    Falls back to ``switch.to_default`` when ``run_cfg`` is absent or the path is
    missing, so a consumer invoked without a full config still emits the value LOCOMgcm
    itself would act on rather than an arbitrary constant.
    """
    if switch.to_path is None or run_cfg is None:
        return switch.to_default
    reader = _CUSTOM_READERS.get(switch.to_path)
    try:
        value = reader(run_cfg) if reader is not None else _resolve(run_cfg, switch.to_path)
    except AttributeError:
        return switch.to_default
    # Match the declared kind rather than coercing: an integer switch flattened
    # to a bool would emit ``.TRUE.`` where MITgcm expects ``1``.
    return value if type(switch.mit_default) is int else bool(value)


def switches_for(namelist: str) -> tuple[MitSwitch, ...]:
    """All switches written into ``namelist``."""
    return tuple(s for s in SWITCHES if s.namelist == namelist)


def render_switch_lines(run_cfg: Any, namelist: str) -> str:
    """Render every switch of ``namelist`` as namelist lines, in table order."""
    return "".join(
        f" {s.mit_name}={fortran_value(switch_value(run_cfg, s))},\n"
        for s in switches_for(namelist)
    )


def assert_switches_agree(run_cfg: Any, namelist: str, text: str) -> None:
    """Assembly-time gate: the namelist just rendered must match ``run_cfg``.

    Re-parses the emitted text rather than trusting the renderer, so an f-string
    edit that bypasses :func:`render_switch_lines` — the exact failure mode that
    produced the OBCSfixTopo divergence — is caught here instead of surfacing as
    a numerical mismatch hundreds of steps into a comparison run.

    Same shape and intent as ``_assert_obcs_sponge_consistency`` in
    ``consumers/mit_binary.py``, which guards the sponge timescales.
    """
    problems: list[str] = []
    for switch in switches_for(namelist):
        expected = switch_value(run_cfg, switch)
        is_logical = type(switch.mit_default) is bool
        pattern = r"\.(TRUE|FALSE)\." if is_logical else r"(-?\d+)"
        found = re.search(
            rf"^\s*{re.escape(switch.mit_name)}\s*=\s*{pattern}\s*,",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
        if found is None:
            problems.append(
                f"{switch.mit_name}: expected {fortran_value(expected)} but the "
                f"keyword is absent from the rendered {namelist}"
            )
            continue
        emitted = found.group(1).upper() == "TRUE" if is_logical else int(found.group(1))
        if emitted != expected:
            problems.append(
                f"{switch.mit_name}: {namelist} says {fortran_value(emitted)} but "
                f"LOCOMgcm ({switch.to_path}) resolves to {fortran_value(expected)}"
                f" — {switch.rationale}"
            )
    if problems:
        raise ValueError(
            f"MIT/TO switch divergence while writing {namelist}: the exported "
            "reference model would not be configured like the LOCOMgcm run it is "
            "compared against.\n  " + "\n  ".join(problems)
        )


# ------------------------------------------------------------ data PARM01-PARM04
#
# Every key the data consumer writes into PARM01 to PARM04 has a row in
# DATA_KEYS: where LOCOMgcm holds the value and how it is read.  A row either
# reads the assembled components (``_runtime_components``, the stepper, the
# ModelInput the model was built from) or states a constant together with the
# fact about LOCOMgcm that makes it the model's value.  After writing, the
# consumer re-parses the file (:func:`assert_data_keys_agree`): a key without a
# row, a missing key, or a value that does not read back as the model's value
# refuses the export.


class _Absent:
    def __repr__(self) -> str:
        return "ABSENT"


#: A row's value when the key must not be written (LOCOMgcm has no such term).
ABSENT = _Absent()


@dataclass(frozen=True)
class DataView:
    """What the rows read: the run config, the ModelInput run parameters, the
    consumer's resolved export arguments, and the equation of state."""

    run_cfg: Any
    params: dict
    args: Any
    eos: dict


@dataclass(frozen=True)
class DataKey:
    group: str
    key: str
    holder: str
    value: Any                   # Callable[[DataView], Any]: the model's value or ABSENT
    mit_default: Any = ABSENT    # MITgcm's default: the key may be omitted when equal


def _stepper_entry(cfg: Any) -> tuple[type, dict[str, Any]]:
    import defgcm.steppers  # noqa: F401  (registers every stepper)
    from defgcm.steppers._registry import STEPPERS

    stepper = cfg.model.stepper
    return STEPPERS.get(stepper.name), dict(getattr(stepper, "kwargs", {}) or {})


def stepper_field(cfg: Any, name: str) -> Any:
    """``name`` as the run's stepper holds it (yaml kwarg, else class default)."""
    cls, kwargs = _stepper_entry(cfg)
    return _component_field(cls, kwargs, name)


def _require_free_surface_stepper(cfg: Any) -> None:
    from defgcm.steppers.implicit_free_surface import ImplicitFreeSurfaceStepper

    cls, _ = _stepper_entry(cfg)
    if not issubclass(cls, ImplicitFreeSurfaceStepper):
        raise ValueError(
            f"rigidLid, implicitFreeSurface: the data export describes MITgcm's implicit "
            f"free surface; LOCOMgcm runs {cls.__name__}, which MITgcm has no namelist for"
        )


def momentum_advection_form(cfg: Any) -> str:
    """``vector_invariant`` or ``flux_form``, from the assembled momentum core.

    ``MomentumAdvection`` (the default assembly's core) and ``FluxFormMomentum``
    are flux form; a bare Coriolis module names the flux form as before.
    """
    names = {name for _where, name, _cls, _kw in _runtime_components(cfg)}
    has_vi = "vector_invariant_momentum_advection" in names
    has_flux = bool(names & {"flux_form_momentum", "momentum_advection"})
    if has_vi and has_flux:
        raise ValueError(
            "model.dynamics contains both vector_invariant_momentum_advection and a "
            "flux-form momentum core; choose one momentum core"
        )
    if has_vi:
        return "vector_invariant"
    if has_flux or "f_coriolis" in names:
        return "flux_form"
    return "vector_invariant"


def select_ke_scheme_value(cfg: Any) -> int:
    """``selectKEscheme``: the vector-invariant core's KE scheme.

    MITgcm reads it in mom_vecinv.F; the flux-form core uses KE scheme 2
    whatever the key says (mom_fluxform.F:329), so without a vector-invariant
    core the export writes MITgcm's default 0.
    """
    kwargs = _component_kwargs(cfg, "vector_invariant_momentum_advection")
    if kwargs is None:
        return 0
    from defgcm.dynamics.vector_invariant_momentum import VectorInvariantMomentumAdvection

    return int(_component_field(VectorInvariantMomentumAdvection, kwargs, "select_ke_scheme"))


def no_slip_sides_value(cfg: Any) -> bool:
    """``no_slip_sides``: the horizontal viscosity module's; False without one."""
    module = horizontal_viscosity_module(cfg)
    return False if module is None else bool(module.no_slip_sides)


def leith_smag_coefficients(cfg: Any) -> tuple[float, float]:
    """``(viscC2Leith, viscC2smag)`` the module applies (0 outside the Leith/Smag schemes)."""
    from defgcm.dynamics.horizontal_viscosity import VARIABLE_VISCOSITY_SCHEMES

    module = horizontal_viscosity_module(cfg)
    if module is None or module.viscosity_scheme not in VARIABLE_VISCOSITY_SCHEMES:
        return 0.0, 0.0
    return float(module.visc_c2_leith), float(module.visc_c2_smag)


def convective_adjustment_frequency(cfg: Any) -> float:
    """``cAdjFreq``: the convective adjustment component's c_adj_freq; 0 without one."""
    kwargs = _component_kwargs(cfg, "convective_adjustment")
    if kwargs is None:
        return 0.0
    from defgcm.dynamics.convective_adjustment import ConvectiveAdjustment

    return float(_component_field(ConvectiveAdjustment, kwargs, "c_adj_freq"))


def cg2d_namelist_values(cfg: Any) -> dict[str, float | int]:
    """PARM02 CG2D keys from the stepper's cg_max_iters, cg_tolerance and stop test.

    mit_normalized is cg2dTargetResidual with cg2dTargetResWunit <= 0, mit_w_unit
    is cg2dTargetResWunit > 0 (ini_cg2d.F:148-161, cg2d.F:117-127, :204, :337);
    LOCOMgcm's absolute test has no MITgcm spelling.
    """
    mode = str(stepper_field(cfg, "cg_tolerance_mode"))
    tolerance = float(stepper_field(cfg, "cg_tolerance"))
    values: dict[str, float | int] = {"cg2dMaxIters": int(stepper_field(cfg, "cg_max_iters"))}
    if mode == "mit_normalized":
        values["cg2dTargetResidual"] = tolerance
    elif mode == "mit_w_unit":
        values["cg2dTargetResWunit"] = tolerance
    else:
        raise ValueError(
            f"cg2dTargetResidual: the stepper's cg_tolerance_mode={mode!r} has no MITgcm "
            "spelling (mit_normalized or mit_w_unit)"
        )
    return values


def _tracer_advection_component(cfg: Any, tracer: str) -> tuple[str, type, dict] | None:
    found = [(where, cls, kwargs) for where, name, cls, kwargs in _runtime_components(cfg)
             if name == "horizontal_tracer_advection"
             and tracer in _component_field(cls, kwargs, "tracer_names")]
    if len(found) > 1:
        raise ValueError(f"tempAdvScheme/saltAdvScheme: {tracer} is advected by {len(found)} modules")
    return found[0] if found else None


def tracer_advection_values(cfg: Any, tracer: str) -> dict[str, Any]:
    """``AdvScheme``, ``VertAdvScheme`` (codes) and ``multi_dim`` of ``tracer``.

    The horizontal scheme and multi-dimension choice are the advecting
    module's.  The vertical scheme is its ``vertical_scheme`` when it does the
    vertical pass (multi_dim with dt, combined_with_vtt, or the stepper's
    implicit column, which takes the same field), else the
    ``vertical_tracer_transport`` module's.  A tracer nothing advects gives
    None for all three.
    """
    from defgcm.io.runner._canonical_helpers import implicit_vertical_advection_stepper_kwargs
    from defgcm.operators.fluxes.tracer_flux import TRACER_SCHEME_BY_NAME

    def code(value: Any) -> int:
        return TRACER_SCHEME_BY_NAME[str(getattr(value, "value", value)).lower()].mit_code

    component = _tracer_advection_component(cfg, tracer)
    if component is None:
        return {"horizontal": None, "vertical": None, "multi_dim": None}
    _where, cls, kwargs = component
    multi_dim = bool(_component_field(cls, kwargs, "multi_dim"))
    leaf = _IMPLICIT_ADVECTION_LEAF[tracer]
    implicit = implicit_vertical_advection_stepper_kwargs(cfg)[f"{leaf}_implicit_vertical_advection"]
    hta_vertical = (implicit or bool(_component_field(cls, kwargs, "combined_with_vtt"))
                    or (multi_dim and float(_component_field(cls, kwargs, "dt")) > 0.0))
    vertical = None
    if hta_vertical:
        vertical = code(_component_field(cls, kwargs, "vertical_scheme"))
    else:
        for _w, name, vcls, vkwargs in _runtime_components(cfg):
            if name == "vertical_tracer_transport" and tracer in _component_field(vcls, vkwargs, "tracer_names"):
                vertical = code(_component_field(vcls, vkwargs, "vertical_advection_scheme"))
    return {"horizontal": code(_component_field(cls, kwargs, "scheme")),
            "vertical": vertical, "multi_dim": multi_dim}


def tracer_advection_namelist_values(cfg: Any) -> dict[str, Any]:
    """tempAdvScheme, saltAdvScheme, temp/saltVertAdvScheme and multiDimAdvection.

    ``mit_comparison`` overrides of the scheme and of multiDimAdvection keep
    their meaning (``MitComparisonConfig``).  MITgcm has one multiDimAdvection.
    A stepped tracer nothing advects is refused: MITgcm would advect it.
    """
    comparison = cfg.mit_comparison
    values: dict[str, Any] = {}
    multi = {}
    for tracer, prefix in (("temp", "temp"), ("salt", "salt")):
        resolved = tracer_advection_values(cfg, tracer)
        if resolved["horizontal"] is None:
            if _tracer_stepping(cfg, tracer):
                raise ValueError(
                    f"{prefix}AdvScheme: LOCOMgcm steps {tracer} without advecting it, "
                    "MITgcm would advect it"
                )
            continue
        values[f"{prefix}AdvScheme"] = comparison.effective_tracer_scheme(resolved["horizontal"])
        if resolved["vertical"] is None:
            # An omitted key is not "no vertical advection": MITgcm sets it to
            # the horizontal scheme (ini_parms.F:509) and advects vertically
            # (temp_integrate.F:289, gad_calc_rhs.F:513, :550).
            raise ValueError(
                f"{prefix}VertAdvScheme: LOCOMgcm advects {tracer} horizontally with no "
                "vertical advection (horizontal_tracer_advection does no vertical pass "
                "and no vertical_tracer_transport lists it); MITgcm would advect it vertically"
            )
        values[f"{prefix}VertAdvScheme"] = resolved["vertical"]
        multi[tracer] = comparison.effective_multi_dim(resolved["multi_dim"])
    if len(set(multi.values())) > 1:
        raise ValueError(f"multiDimAdvection: MITgcm has one value, LOCOMgcm's modules hold {multi}")
    values["multiDimAdvection"] = next(iter(multi.values())) if multi else True
    return values


def _reference_profile_value(view: DataView, key: str) -> list[float]:
    if key in view.eos:
        return list(view.eos[key])
    if view.eos["eosType"] == "LINEAR":
        raise ValueError(f"{key}: a LINEAR equation of state without its reference profile")
    import numpy as np
    import torch

    from defgcm.io.reference_profile import compute_reference_tracer_profile

    raw = view.args.init_temp if key == "tRef" else view.args.init_salt
    field = raw if isinstance(raw, torch.Tensor) else np.asarray(raw)
    nz = int(view.run_cfg.grid.nz)
    if field.ndim == 3:
        profile = [float(value) for value in compute_reference_tracer_profile(field)]
        if key == "sRef":
            profile = [35.0 if value == 0.0 else value for value in profile]
    elif field.ndim == 1:
        profile = [float(value) for value in field]
    else:
        profile = [20.0 if key == "tRef" else 35.0] * nz
    return profile[:nz]


def _eos_value(key: str):
    return lambda v: v.eos.get(key, ABSENT)


def _visc_extra(key: str):
    return lambda v: viscosity_namelist_values(v.run_cfg).get(key, ABSENT)


def _adv(key: str):
    return lambda v: tracer_advection_namelist_values(v.run_cfg).get(key, ABSENT)


def _nh_only(value):
    return lambda v: value(v) if v.args.nonhydrostatic else ABSENT


def _diffusion(key: str):
    def read(v: DataView) -> Any:
        from defgcm.io.runner.layer_d.consumers.mit_binary import vertical_diffusion_export_args

        return vertical_diffusion_export_args(v.run_cfg)[0][key]
    return read


def _switch(name: str):
    return lambda v: switch_value(v.run_cfg, next(s for s in SWITCHES if s.mit_name == name))


def _free_surface_constant(value: bool):
    def read(v: DataView) -> bool:
        _require_free_surface_stepper(v.run_cfg)
        return value
    return read


def _implicit_flag(leaf: str):
    def read(v: DataView) -> bool:
        from defgcm.io.runner._canonical_helpers import implicit_vertical_advection_stepper_kwargs

        return implicit_vertical_advection_stepper_kwargs(v.run_cfg)[f"{leaf}_implicit_vertical_advection"]
    return read


_P1, _P2, _P3, _P4 = "PARM01", "PARM02", "PARM03", "PARM04"
_MODEL_INPUT = "the ModelInput the model is built from (run_params)"
_OUTPUT_POLICY = ("comparison output policy: the run writes no LOCOMgcm state through it, "
                  "and stage 4 reads MITgcm's float64 global output")
DATA_KEYS: tuple[DataKey, ...] = (
    DataKey(_P1, "tRef", "LINEAR: the source component's reference_temperature.  Other types: "
            "no LOCOMgcm component reads a reference profile (density comes from the EOS "
            "module); MITgcm's set_ref_state input, the initial wet-cell mean",
            lambda v: _reference_profile_value(v, "tRef")),
    DataKey(_P1, "sRef", "as tRef, reference_salinity", lambda v: _reference_profile_value(v, "sRef")),
    DataKey(_P1, "no_slip_sides", "horizontal_viscosity.no_slip_sides",
            lambda v: no_slip_sides_value(v.run_cfg)),
    DataKey(_P1, "no_slip_bottom", "bottom_drag (no_slip_bottom and vertical_viscosity)",
            _switch("no_slip_bottom")),
    DataKey(_P1, "bottomDragLinear", "bottom_drag.linear",
            lambda v: bottom_drag_coefficients(v.run_cfg)[0]),
    DataKey(_P1, "bottomDragQuadratic", "bottom_drag.quadratic",
            lambda v: bottom_drag_coefficients(v.run_cfg)[1]),
    DataKey(_P1, "selectKEscheme", "vector_invariant_momentum_advection.select_ke_scheme",
            lambda v: select_ke_scheme_value(v.run_cfg)),
    DataKey(_P1, "viscAh", "horizontal_viscosity.viscosity, unless mit_comparison overrides it",
            lambda v: v.run_cfg.mit_comparison.effective_viscosity(horizontal_viscosity_value(v.run_cfg))),
    DataKey(_P1, "viscA4", "mit_comparison.viscA4 (comparison setting; checked against "
            "horizontal_viscosity.viscosity_a4 by assert_viscosity_agrees)",
            lambda v: abs(float(v.run_cfg.mit_comparison.effective_viscA4(v.run_cfg)))),
    DataKey(_P1, "viscC2Leith", "horizontal_viscosity.visc_c2_leith under Leith/Smag",
            lambda v: leith_smag_coefficients(v.run_cfg)[0]),
    DataKey(_P1, "viscC2smag", "horizontal_viscosity.visc_c2_smag under Leith/Smag",
            lambda v: leith_smag_coefficients(v.run_cfg)[1]),
    *(DataKey(_P1, mit, "horizontal_viscosity (viscosity_namelist_values)", _visc_extra(mit))
      for mit in ("sideDragFactor", *(m for _, m, _ in _VARIABLE_VISCOSITY_KEYWORDS))),
    DataKey(_P1, "viscAz", "the stepper's implicit_viscosity (legacy_bridge._implicit_viscosity)",
            lambda v: float(v.params["implicit_viscosity"])),
    DataKey(_P1, "implicitViscosity", "physics.implicit_vertical_viscosity, the stepper's gate",
            lambda v: not bool(v.params.get("mitgcm_explicit_viscosity", False))),
    DataKey(_P1, "diffKhT", "horizontal_tracer_diffusion.diffusivity of temp, unless overridden",
            lambda v: v.run_cfg.mit_comparison.effective_diffusivity(
                tracer_harmonic_diffusivity(v.run_cfg, "temp"))),
    DataKey(_P1, "diffKhS", "as diffKhT, salt",
            lambda v: v.run_cfg.mit_comparison.effective_diffusivity(
                tracer_harmonic_diffusivity(v.run_cfg, "salt"))),
    DataKey(_P1, "diffKzT", "the resolved vertical diffusion (vertical_diffusion_export_args)",
            _diffusion("tracer_vertical_diffusivity_kr")),
    DataKey(_P1, "diffKzS", "as diffKzT", _diffusion("tracer_vertical_diffusivity_kr")),
    DataKey(_P1, "implicitDiffusion", "the resolved vertical diffusion",
            _diffusion("implicit_diffusion")),
    DataKey(_P1, "diffK4T", "horizontal_tracer_diffusion.diffusivity_k4 of temp",
            lambda v: tracer_biharmonic_diffusivity(v.run_cfg, "temp"), mit_default=0.0),
    DataKey(_P1, "diffK4S", "as diffK4T, salt",
            lambda v: tracer_biharmonic_diffusivity(v.run_cfg, "salt"), mit_default=0.0),
    DataKey(_P1, "rigidLid", "constant: the stepper is ImplicitFreeSurfaceStepper, a free surface",
            _free_surface_constant(False)),
    DataKey(_P1, "implicitFreeSurface", "constant: the stepper is ImplicitFreeSurfaceStepper",
            _free_surface_constant(True)),
    DataKey(_P1, "exactConserv", "stepper.exact_conserv", _switch("exactConserv")),
    DataKey(_P1, "staggerTimeStep", "stepper.stagger_tracer_stepping", _switch("staggerTimeStep")),
    DataKey(_P1, "linFSConserveTr", "physics.free_surface.lin_fs_conserve_tr",
            lambda v: bool(v.params.get("lin_fs_conserve_tr", False))),
    DataKey(_P1, "nonHydrostatic", "physics.nonhydrostatic, the stepper's NonhydrostaticParams",
            lambda v: bool(v.args.nonhydrostatic)),
    DataKey(_P1, "implicSurfPress", "stepper.implicit_surf_press",
            lambda v: float(stepper_field(v.run_cfg, "implicit_surf_press")), mit_default=1.0),
    DataKey(_P1, "implicDiv2DFlow", "stepper.implicit_div_2d_flow",
            lambda v: float(stepper_field(v.run_cfg, "implicit_div_2d_flow")), mit_default=1.0),
    DataKey(_P1, "implicitNHPress", "stepper.implicit_surf_press (LOCOMgcm's NH factor, "
            "ini_parms.F:485-486)", _nh_only(lambda v: float(stepper_field(v.run_cfg, "implicit_surf_press")))),
    DataKey(_P1, "selectNHfreeSurf", "constant: LOCOMgcm's NH solve has only the 0 form",
            _nh_only(lambda v: 0)),
    DataKey(_P1, "nh_Am2", "physics.nonhydrostatic.nh_am2", _nh_only(lambda v: float(v.args.nh_am2))),
    DataKey(_P1, "eosType", "the equation-of-state component (eos_namelist_values)", _eos_value("eosType")),
    DataKey(_P1, "tAlpha", "the LINEAR source component", _eos_value("tAlpha")),
    DataKey(_P1, "sBeta", "the LINEAR source component", _eos_value("sBeta")),
    DataKey(_P1, "rhoNil", "the LINEAR source component", _eos_value("rhoNil")),
    DataKey(_P1, "selectP_inEOS_Zc", "mdjwf_equation_of_state.dynamic_pressure (0 static, 2 dynamic)",
            _eos_value("selectP_inEOS_Zc"), mit_default=2),
    DataKey(_P1, "f0", "Cartesian only, refused by the consumer", lambda v: ABSENT),
    DataKey(_P1, "beta", "Cartesian only, refused by the consumer", lambda v: ABSENT),
    DataKey(_P1, "rhoConst", "rho_const_value", lambda v: rho_const_value(v.run_cfg)),
    DataKey(_P1, "gravity", "gravity_value", lambda v: gravity_value(v.run_cfg)),
    DataKey(_P1, "tempStepping", "the components listing temp", _switch("tempStepping")),
    DataKey(_P1, "saltStepping", "the components listing salt", _switch("saltStepping")),
    DataKey(_P1, "momStepping", "stepper.mom_stepping",
            lambda v: bool(stepper_field(v.run_cfg, "mom_stepping"))),
    DataKey(_P1, "tempImplVertAdv", "physics.implicit_vertical_advection.temperature",
            _implicit_flag("temperature")),
    DataKey(_P1, "saltImplVertAdv", "physics.implicit_vertical_advection.salinity",
            _implicit_flag("salinity")),
    DataKey(_P1, "momImplVertAdv", "physics.implicit_vertical_advection.momentum",
            _implicit_flag("momentum")),
    DataKey(_P1, "vectorInvariantMomentum", "the assembled momentum core",
            lambda v: momentum_advection_form(v.run_cfg) == "vector_invariant"),
    DataKey(_P1, "tempAdvScheme", "horizontal_tracer_advection.scheme of temp", _adv("tempAdvScheme")),
    DataKey(_P1, "saltAdvScheme", "horizontal_tracer_advection.scheme of salt", _adv("saltAdvScheme")),
    DataKey(_P1, "tempVertAdvScheme", "the vertical pass of temp", _adv("tempVertAdvScheme")),
    DataKey(_P1, "saltVertAdvScheme", "the vertical pass of salt", _adv("saltVertAdvScheme")),
    DataKey(_P1, "multiDimAdvection", "horizontal_tracer_advection.multi_dim", _adv("multiDimAdvection")),
    DataKey(_P1, "readBinaryPrec", "constant: this export writes its binaries as float64",
            lambda v: 64),
    DataKey(_P1, "writeBinaryPrec", f"constant: {_OUTPUT_POLICY}", lambda v: 64),
    DataKey(_P1, "useSingleCpuIO", f"constant: {_OUTPUT_POLICY}", lambda v: True),
    DataKey(_P1, "hFacMin", f"hfac_min of {_MODEL_INPUT}", lambda v: float(v.args.hfac_min)),
    DataKey(_P1, "hFacMinDz", f"hfac_min_dz of {_MODEL_INPUT}", lambda v: float(v.args.hfac_min_dz)),
    DataKey(_P2, "cg2dMaxIters", "stepper.cg_max_iters",
            lambda v: cg2d_namelist_values(v.run_cfg)["cg2dMaxIters"]),
    DataKey(_P2, "cg2dTargetResidual", "stepper.cg_tolerance under mit_normalized",
            lambda v: cg2d_namelist_values(v.run_cfg).get("cg2dTargetResidual", ABSENT)),
    DataKey(_P2, "cg2dTargetResWunit", "stepper.cg_tolerance under mit_w_unit",
            lambda v: cg2d_namelist_values(v.run_cfg).get("cg2dTargetResWunit", ABSENT)),
    DataKey(_P2, "cg3dMaxIters", "physics.nonhydrostatic.cg3d_max_iters",
            lambda v: int(v.args.cg3d_max_iters)),
    DataKey(_P2, "cg3dTargetResidual", "physics.nonhydrostatic.cg3d_target_residual",
            lambda v: float(v.args.cg3d_target_residual)),
    DataKey(_P2, "cg3dTargetResWunit", "physics.nonhydrostatic.cg3d_target_res_w_unit",
            _nh_only(lambda v: float(v.args.cg3d_target_res_w_unit))),
    DataKey(_P3, "startTime", "constant: LOCOMgcm's clock starts at 0, MITgcm's startTime",
            lambda v: 0.0),
    DataKey(_P3, "endTime", "the run length (run_cfg.time, whole MITgcm steps)",
            lambda v: float(v.args.total_seconds)),
    DataKey(_P3, "deltaT", "time.dt, the model's step (RegionalOceanConfig.default_dt)",
            lambda v: float(v.run_cfg.time.dt)),
    DataKey(_P3, "deltaTmom", "time.dt_mom", lambda v: float(v.args.dt_mom_seconds)),
    DataKey(_P3, "deltaTtracer", "time.dt_tracer", lambda v: float(v.args.dt_tracer_seconds)),
    DataKey(_P3, "deltaTFreeSurf", "time.dt_freesurf", lambda v: float(v.args.dt_freesurf_seconds)),
    DataKey(_P3, "deltaTClock", "time.dt_clock", lambda v: float(v.args.dt_clock_seconds)),
    DataKey(_P3, "cAdjFreq", "convective_adjustment.c_adj_freq",
            lambda v: convective_adjustment_frequency(v.run_cfg)),
    DataKey(_P3, "abEps", "stepper.ab_eps", lambda v: ab_eps_value(v.run_cfg)),
    DataKey(_P3, "momForcingOutAB", "stepper.mom_forcing_out_ab", _switch("momForcingOutAB")),
    DataKey(_P3, "tracForcingOutAB", "the stepper's post-AB tracer forcing", _switch("tracForcingOutAB")),
    DataKey(_P3, "pChkptFreq", f"constant: {_OUTPUT_POLICY}", lambda v: 0.0),
    DataKey(_P3, "chkptFreq", f"constant: {_OUTPUT_POLICY}", lambda v: 0.0),
    DataKey(_P3, "dumpFreq", "the run's output interval (whole MITgcm clock steps)",
            lambda v: float(v.args.dump_freq_seconds)),
    DataKey(_P3, "monitorFreq", f"constant: 100 clock steps, {_OUTPUT_POLICY}",
            lambda v: float(v.args.dt_clock_seconds) * 100),
    DataKey(_P3, "monitorSelect", f"constant: {_OUTPUT_POLICY}", lambda v: 2),
    DataKey(_P3, "pickupStrictlyMatch", f"constant: {_OUTPUT_POLICY}", lambda v: False),
    DataKey(_P4, "usingSphericalPolarGrid", f"the spherical grid of {_MODEL_INPUT}",
            lambda v: not v.args.use_cartesian),
    DataKey(_P4, "usingCartesianGrid", "refused by the consumer", lambda v: ABSENT),
    DataKey(_P4, "delX", f"resolution_deg of {_MODEL_INPUT}",
            lambda v: [float(v.args.resolution_deg)] * int(v.args.nx)),
    DataKey(_P4, "delY", f"resolution_deg of {_MODEL_INPUT}",
            lambda v: [float(v.args.resolution_deg)] * int(v.args.ny)),
    DataKey(_P4, "delZ", f"the layer thicknesses of {_MODEL_INPUT}",
            lambda v: ([float(x) for x in v.args.layer_thickness] if v.args.layer_thickness
                       else [float(v.args.depth_m) / int(v.args.nz)] * int(v.args.nz))),
    DataKey(_P4, "ygOrigin", f"the grid origin of {_MODEL_INPUT}", lambda v: float(v.args.yg_origin)),
    DataKey(_P4, "xgOrigin", f"the grid origin of {_MODEL_INPUT}", lambda v: float(v.args.xg_origin)),
)
_DATA_KEYS_BY_NAME = {(entry.group, entry.key.lower()): entry for entry in DATA_KEYS}


def _parse_namelist_items(text: str) -> list[Any]:
    values: list[Any] = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        count, star, rest = item.partition("*")
        repeat, item = (int(count), rest.strip()) if star and count.strip().isdigit() else (1, item)
        logical = re.fullmatch(r"\.(TRUE|FALSE)\.", item, re.IGNORECASE)
        if logical is not None:
            value: Any = logical.group(1).upper() == "TRUE"
        elif item.startswith("'"):
            value = item.strip("'")
        else:
            value = float(item.replace("D", "E").replace("d", "e"))
        values += [value] * repeat
    return values


def parse_data_groups(text: str) -> dict[str, dict[str, list[Any]]]:
    """``{group: {key: values}}`` of a data namelist; ``#`` lines are comments."""
    groups: dict[str, dict[str, list[Any]]] = {}
    current = key = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("&"):
            name = line[1:].strip().upper()
            current = name or None
            if current:
                if current in groups:
                    # MITgcm reads the first group of a name (ini_parms.F:937, one
                    # READ per group); a second one would be checked instead.
                    raise ValueError(f"data namelist repeats group &{current}")
                groups[current] = {}
            continue
        if current is None:
            continue
        match = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.*)$", line)
        if match is not None:
            key, line = match.group(1), match.group(2)
            groups[current][key] = []
        groups[current][key] += _parse_namelist_items(line)
    return groups


def _as_values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _same_values(found: list[Any], expected: list[Any]) -> bool:
    if len(found) != len(expected):
        return False
    for a, b in zip(found, expected):
        if isinstance(a, bool) != isinstance(b, bool):
            return False      # MITgcm types the key; .TRUE. is not the integer 1
        if isinstance(b, bool) or isinstance(b, str):
            if a != b:
                return False
        elif float(a) != float(b):
            return False
    return True


def assert_data_keys_agree(view: DataView, text: str) -> None:
    """Refuse a data file whose PARM01-PARM04 keys are not exactly the model's values."""
    groups = parse_data_groups(text)
    problems: list[str] = []
    for group in (_P1, _P2, _P3, _P4):
        for key in groups.get(group, {}):
            if (group, key.lower()) not in _DATA_KEYS_BY_NAME:
                problems.append(f"{group}.{key}: written but has no row in DATA_KEYS")
    written = {(group, key.lower()): values
               for group, keys in groups.items() for key, values in keys.items()}
    for entry in DATA_KEYS:
        expected = entry.value(view)
        found = written.get((entry.group, entry.key.lower()))
        if found is None:
            if expected is ABSENT or (entry.mit_default is not ABSENT
                                      and _same_values([entry.mit_default], _as_values(expected))):
                continue
            problems.append(f"{entry.key}: LOCOMgcm holds {expected!r} ({entry.holder}) but data omits it")
        elif expected is ABSENT:
            problems.append(f"{entry.key}: data says {found} but LOCOMgcm has no such term ({entry.holder})")
        elif not _same_values(found, _as_values(expected)):
            problems.append(f"{entry.key}: data says {found} but LOCOMgcm holds {expected!r} ({entry.holder})")
    if problems:
        raise ValueError(
            "MITgcm data export disagrees with the LOCOMgcm model it describes:\n  "
            + "\n  ".join(problems)
        )
