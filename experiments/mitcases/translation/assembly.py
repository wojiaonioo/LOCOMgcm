"""The yaml assembly: model modules, stepper, initial condition source, derived decisions."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from pathlib import Path
from typing import Any
from experiments.mitcases.code_inventory import DEFER, MAP, UNSUPPORTED
from locomgcm.dynamics._registry import DYNAMICS_MODULES
from locomgcm.dynamics.cd_scheme import CD_CODE_OPTIONS, CD_MOMENTUM_FORMS, CD_STEPPING_MODES
from locomgcm.dynamics.coriolis import FLUX_CORI_SCHEMES
from locomgcm.dynamics.vector_invariant_momentum import CORI_SCHEMES
from locomgcm.steppers._registry import STEPPERS
from locomgcm.io.runner.config._schema import AdvectionConfig
from locomgcm.io.runner._canonical_helpers import complete_stepper_owned_kwargs
from locomgcm.io.runner.builders.from_manifest import (
    ManifestBuildError,
    _refuse_unread_center_spacing,
)
from locomgcm.steppers.ifs.admission import _refuse_stepping_switches_unimplemented
from locomgcm.operators.fluxes.tracer_flux import TRACER_SCHEME_BY_NAME

from .core import (
    InvalidParameter,
    Record,
    TranslationError,
    UnknownParameter,
    _ASSEMBLY,
    _Assembly,
    _FORCING_SOURCE_NAME,
    _IC_SOURCE_NAME,
    _MESH_SOURCE_NAME,
    _MISSING,
    _OBCS_SIDES,
    _OBCS_SOURCE_NAME,
    _dtype_of,
    _entry,
    _record_gap,
    _set_dotted,
)
from .rules import (
    Ctx,
    _TO_GRID_MODE,
    _coriolis_profile,
)
from .cpp import (
    _SIZE_KEYS,
    _reference_mit_root,
    _resolve_penetrating_sw,
    parse_cpp_macro_states,
)
from .convert import (
    _ADV_SCHEME_MAP,
    _EOS_MODULES,
    _MIT_AB_ADV_SCHEMES,
    _MIT_DEFAULT_ADV_SCHEME,
    _MIT_DEFAULT_GRAVITY,
    _MIT_DEFAULT_SREF,
    _MIT_DEFAULT_TREF,
    _effective_mit_tracer_adv_scheme,
    _eos_pressure_kwargs,
    _fill_ref,
    _master_dt,
    _mit_delta_t_mom,
    _mit_selectp_in_eos_zc,
    _mit_start_clock,
    _mit_vort_scheme,
    _tracer_tendency_ab_gates,
    _tracer_tendency_ab_writes,
)
from .vmix import (
    _vertical_mixing_parameters,
)
from .gmredi import (
    _gmredi_entry,
)
from .obcs import (
    _wire_obcs_analytic,
    _wire_obcs_binary,
)
from .exf import (
    _wire_core_surface_forcing,
    _wire_exf,
    _wire_surface_climatology,
)

# --------------------------------------------------------------------------
# yaml assembly
# --------------------------------------------------------------------------

#: eosType default (model/src/set_defaults.F:175).
_MIT_DEFAULT_EOS = "LINEAR"

#: tempAdvScheme / multiDimAdvection defaults
#: (model/src/set_defaults.F:226, :228).  LOCOMgcm's own schema defaults
#: differ (dst3_fl / true), so the assembly states MIT's value explicitly
#: rather than letting the two ends drift apart in silence.
_MIT_DEFAULT_MULTI_DIM = True


def _record_silent_divergences(ctx: Ctx, asm: _Assembly, record: Record) -> None:
    """Record bounded MIT/LOCOMgcm semantic divergences after defaults resolve.

    Tracer forcing order is carried by the surface module ab2_eligible flag.
    The separate doAB_onGtGs derivation selects tracer tendency integration.
    """
    del asm
    real_freshwater = bool(ctx.data("PARM01", "userealfreshwaterflux", False))
    if real_freshwater:
        record.silent_divergences.append(
            {
                "mit_key": "useRealFreshWaterFlux",
                "namelist_block": "PARM01",
                "mit_behavior": {
                    "value": True,
                    "description": (
                        "real freshwater flux changes free-surface volume, the "
                        "CG2D RHS, and the surface vertical-momentum flux"
                    ),
                    "sources": [
                        "model/src/external_forcing_surf.F:149-166,258-288",
                        "model/src/solve_for_pressure.F:146-153",
                        "model/src/pre_cg3d.F:88-110",
                    ],
                },
                "to_behavior": {
                    "value": True,
                    "description": (
                        "the integrated-eta CG2D RHS and exactConserv etaH volume "
                        "change are mapped; the surface vertical-momentum effect "
                        "remains omitted"
                    ),
                    "sources": [
                        "locomgcm/steppers/step_segments.py:613-748",
                        "locomgcm/steppers/implicit_free_surface.py:824-847,"
                        "1941-1953,2434-2490,2778-2794",
                    ],
                },
                "applicability": {
                    "when": "useRealFreshWaterFlux=.TRUE.",
                    "resolved": True,
                },
                "note": (
                    "The remaining divergence is limited to the surface "
                    "vertical-momentum coupling in PRE_CG3D; the integrated-eta "
                    "volume path is implemented"
                ),
            }
        )

def _mit_cori_scheme(ctx: Ctx) -> int:
    """MITgcm's effective selectCoriScheme (ini_parms.F:661-692).

    Unset, it is 1 with useJamartWetPoints and 0 without (:661-663);
    useEnergyConservingCoriolis would add 2 in the flux form (:664-666) but has
    no row, so a case setting it stops at UnknownParameter.  Set, it is its own
    value, and MITgcm stops when useJamartWetPoints is also on and the value is
    not 1 (vector-invariant, :668-674) or not 1 or 3 (flux form, :684-691).
    """
    jamart = bool(ctx.data("PARM01", "usejamartwetpoints", False))
    selected = ctx.data("PARM01", "selectcorischeme")
    if selected is None:
        return 1 if jamart else 0
    code = int(selected)
    vector_invariant = bool(ctx.data("PARM01", "vectorinvariantmomentum", False))
    if jamart and code not in ((1,) if vector_invariant else (1, 3)):
        raise InvalidParameter(
            f"selectCoriScheme={code} conflicts with useJamartWetPoints: MITgcm stops "
            "(ini_parms.F:668-691)"
        )
    return code


def _cori_scheme_kwargs(
    ctx: Ctx, record: Record, module: str, *, vector_invariant: bool
) -> dict[str, Any]:
    """``select_cori_scheme`` for the momentum module ``module`` holding the Coriolis term.

    The vector-invariant term is MOM_VI_CORIOLIS (mom_vi_coriolis.F:58-189;
    VectorInvariantMomentumAdvection, CORI_SCHEMES), the flux-form term
    MOM_U/V_CORIOLIS (mom_u_coriolis.F:44-91; FCoriolis, FLUX_CORI_SCHEMES),
    the configuration value being the MITgcm code.  Written only when the
    namelist sets selectCoriScheme or useJamartWetPoints.  A code outside the
    module's tuple, or a non-zero code for a module without the field (the
    flux-form bundle before it passes the field to its FCoriolis), is a gap.
    """
    if ctx.data("PARM01", "selectcorischeme") is None and ctx.data(
        "PARM01", "usejamartwetpoints"
    ) is None:
        return {}
    code = _mit_cori_scheme(ctx)
    schemes, declared = (
        (CORI_SCHEMES, "vector_invariant_momentum.CORI_SCHEMES")
        if vector_invariant
        else (FLUX_CORI_SCHEMES, "coriolis.FLUX_CORI_SCHEMES")
    )
    row = {
        "source": "input/data:PARM01.selectCoriScheme/useJamartWetPoints",
        "parameter": f"<derived:selectCoriScheme of {module}>",
        "value": code,
    }
    if module == "f_coriolis" and vector_invariant:
        # MITgcm keeps MOM_VI_CORIOLIS with momAdvection off (mom_vecinv.F:680-700);
        # the bare term the assembly emits is the flux-form stencil.
        _record_gap(record, row, "selectCoriScheme with vector-invariant momentum and "
                    "momAdvection=.FALSE.: the bare Coriolis term f_coriolis is the "
                    "flux-form stencil, not MOM_VI_CORIOLIS")
        return {}
    if code not in {c for _, c in schemes}:
        _record_gap(record, row, (
            f"selectCoriScheme={code}: {module} implements "
            f"{', '.join(f'{c} ({n})' for n, c in schemes)} ({declared})"
        ))
        return {}
    if "select_cori_scheme" not in {f.name for f in fields(DYNAMICS_MODULES.get(module))}:
        if code != 0:
            _record_gap(record, row, (
                f"selectCoriScheme={code}: {module} has no select_cori_scheme field "
                "(its Coriolis term is selectCoriScheme 0)"
            ))
        return {}
    return {"select_cori_scheme": code}


def _cd_code_options(case_dir: Path) -> list[str]:
    """The CD_CODE_OPTIONS members the reference build defines.

    The case's code/CD_CODE_OPTIONS.h replaces the package header
    pkg/cd_code/CD_CODE_OPTIONS.h (both state each macro, the package one
    undefining both); a macro other than the two and the include guard is
    unknown.
    """
    path = case_dir / "code" / "CD_CODE_OPTIONS.h"
    if not path.is_file():
        path = _reference_mit_root() / "pkg" / "cd_code" / "CD_CODE_OPTIONS.h"
    states = parse_cpp_macro_states(path, predefined=frozenset({"ALLOW_CD_CODE"}))
    unknown = sorted(set(states) - set(CD_CODE_OPTIONS) - {"ALLOW_CD_CODE", "CD_CODE_OPTIONS_H"})
    if unknown:
        raise UnknownParameter(unknown[0], str(path))
    return [macro for macro in CD_CODE_OPTIONS if states.get(macro, False)]


def _convective_adjustment_entry(
    asm: _Assembly, ctx: Ctx, record: Record, case_dir: Path, *, tracers: bool
) -> dict[str, Any] | None:
    """The convective_adjustment module for a staged nonzero cAdjFreq (start/3).

    deltaTClock is its own value, else deltaT (ini_parms.F:1043-1059); a negative
    cAdjFreq becomes deltaTClock (:1104-1106).  The build's CPP_OPTIONS.h guards
    the per-step call with INCLUDE_CONVECT_CALL (tracers_correction_step.F:114-118;
    with neither it nor INCLUDE_CONVECT_INI_CALL MITgcm stops, config_check.F:62-82)
    and the initial-state call with INCLUDE_CONVECT_INI_CALL, made only when
    startTime = baseTime (initialise_varia.F:280-295, the clock of
    :func:`_mit_start_clock`): written as initial_adjustment, which
    initialize_fresh_step_state applies to the initial tracers.  The module adjusts
    the stepped tracers with the composition's equation of state, assembled only
    with tracer stepping.
    """
    header = case_dir / "code" / "CPP_OPTIONS.h"
    if not header.is_file():
        header = _reference_mit_root() / "model" / "inc" / "CPP_OPTIONS.h"
    macros = parse_cpp_macro_states(header)
    per_step = macros.get("INCLUDE_CONVECT_CALL", False)
    initial = macros.get("INCLUDE_CONVECT_INI_CALL", False)
    if not (per_step or initial):
        raise InvalidParameter(
            "cAdjFreq is nonzero with INCLUDE_CONVECT_CALL and INCLUDE_CONVECT_INI_CALL "
            "undefined: MITgcm stops (config_check.F:62-82)"
        )
    refused = []
    if not per_step:
        refused.append("INCLUDE_CONVECT_CALL undefined: only the initial adjustment runs "
                       "(config_check.F:64-72), which the module has no mode for")
    if not tracers:
        refused.append("no tracer stepping: no equation-of-state module is assembled")
    if refused:
        _record_gap(
            record,
            {"source": "input/data:PARM03.cAdjFreq",
             "parameter": "<derived:convective_adjustment composition>",
             "value": asm.staged("c_adj_freq")},
            "cAdjFreq with " + "; ".join(refused),
        )
        return None
    clock = ctx.data("PARM03", "deltatclock")
    dt_clock = float(clock) if clock is not None else _master_dt(ctx)
    c_adj_freq = float(asm.staged("c_adj_freq"))
    kwargs: dict[str, Any] = {
        "c_adj_freq": dt_clock if c_adj_freq < 0.0 else c_adj_freq,
        "dt_clock": dt_clock,
    }
    start, _, base = _mit_start_clock(ctx)
    if initial and start == base:
        kwargs["initial_adjustment"] = True
    return _entry("convective_adjustment", **kwargs)


def _check_advection_section(yaml_tree: dict[str, Any], record: Record) -> None:
    """The run-config loader builds AdvectionConfig from the advection section as
    below (config/_loader.py) and the dataclass refuses a scheme the run config does
    not carry (config/_schema.py AdvectionConfig.__post_init__), which the trial
    construction of the modules does not reach; the section is constructed here and
    its refusal recorded unsupported."""
    section = yaml_tree.get("advection") or {}
    try:
        AdvectionConfig(
            vertical=section.get("vertical"),
            tracer=section.get("tracer", "dst3_fl"),
            multi_dim=bool(section.get("multi_dim", True)),
        )
    except ValueError as exc:
        _record_gap(
            record,
            {"source": "derived:advection", "parameter": "<derived:advection section>",
             "value": dict(section)},
            f"ValueError: {exc} (config/_schema.py AdvectionConfig)",
        )


def _check_center_spacing_readers(
    yaml_tree: dict[str, Any], dynamics: list[dict[str, Any]], record: Record
) -> None:
    """A mapped delRc with a module that still derives drC or rC from the layer
    thicknesses (KPP, a pressure-dependent equation of state) is refused by the
    model assembly (from_manifest._refuse_unread_center_spacing), which the trial
    construction does not reach; that declaration is evaluated here on the
    translated entries and its refusal recorded unsupported."""
    try:
        _refuse_unread_center_spacing(
            [SimpleNamespace(name=entry["name"]) for entry in dynamics],
            SimpleNamespace(kpp_enabled=bool(
                yaml_tree["model"]["stepper"]["kwargs"].get("kpp_enabled", False))),
            SimpleNamespace(vertical_coordinate=SimpleNamespace(center_spacing_given=True)),
        )
    except ManifestBuildError as exc:
        _record_gap(
            record,
            {"source": "input/data:PARM04.delRc",
             "parameter": "<derived:vertical_center_spacing readers>", "value": None},
            f"{type(exc).__name__}: {exc}",
        )


def _wire_stepping_switches(yaml_tree: dict[str, Any], asm: _Assembly, record: Record) -> None:
    """momStepping, tempStepping, saltStepping -> the stepper's mom_stepping,
    temp_stepping, salt_stepping (trc4/2, trc4/6), each written as resolved (unset
    .TRUE.: set_defaults.F:192, :194, :198).

    momStepping=.FALSE. skips DYNAMICS, SOLVE_FOR_PRESSURE and
    MOMENTUM_CORRECTION_STEP, so uVel and vVel keep their values
    (forward_step.F:786-795, :897-917); etaN keeps its value too unless
    exactConserv keeps INTEGR_CONTINUITY running (calc_wVelocity = momStepping
    .OR. exactConserv, set_parms.F:105; forward_step.F:924-930), which then moves
    etaN and etaH from the fixed flow.  tempStepping or saltStepping .FALSE. skips
    TEMP_INTEGRATE or SALT_INTEGRATE (thermodynamics.F:317-337).  The modules with
    tracer_names list the stepped tracers only, which the assembly requires
    (from_manifest._refuse_dropped_terms).  With momStepping off the stepper
    refuses open boundaries, a nonhydrostatic pressure and an external
    nonhydrostatic pressure provider in its prepare_step_host, which the trial
    construction does not reach, so that declaration
    (admission._refuse_stepping_switches_unimplemented) is evaluated here on the
    translated configuration and its refusal recorded unsupported.  The
    translator wires no external nonhydrostatic pressure provider, hence
    external_pnh=False.
    """
    stepper_kwargs = yaml_tree["model"]["stepper"]["kwargs"]
    switches = {key: asm.flag(key) for key in ("mom_stepping", "temp_stepping", "salt_stepping")}
    stepper_kwargs.update(switches)
    extras = yaml_tree["case"]["extras"]
    try:
        _refuse_stepping_switches_unimplemented(
            mom_stepping=switches["mom_stepping"],
            open_boundary="open" in (extras.get("west_east_boundary"), extras.get("south_north_boundary")),
            nonhydrostatic=bool(asm.yaml("physics.nonhydrostatic.enabled", False)),
            external_pnh=False,
        )
    except NotImplementedError as exc:
        _record_gap(
            record,
            {"source": "input/data:PARM01.momStepping",
             "parameter": "<derived:stepping switches composition>", "value": switches},
            f"NotImplementedError: {exc}",
        )


def _cd_scheme_entry(
    asm: _Assembly, record: Record, case_dir: Path, profile: Any, *,
    vector_invariant: bool, coriolis: bool,
) -> dict[str, Any] | None:
    """The cd_scheme entry of useCDscheme, or None with the gap recorded.

    tauCD, rCD and epsAB_CD are the namelist's values or MITgcm's unset ones
    (set_defaults.F:328, ini_parms.F:965-966), which the module resolves as
    ini_parms.F:1114-1118; ``profile`` is the centre fCori of the flux form.  The
    momentum form and the stepping regime must be in the module's
    CD_MOMENTUM_FORMS and CD_STEPPING_MODES; useCoriolis=.FALSE. makes MITgcm's
    cfFacMom 0 (cd_code_scheme.F:169, :231) while the module keeps it 1.  The rest of admit_cd_scheme (grid, open boundaries,
    implicit viscosity, implicit drag) needs the built model and refuses there.
    """
    form = "vector_invariant" if vector_invariant else "flux_form"
    stepping = (
        "staggered" if asm.yaml("model.stepper.kwargs.stagger_tracer_stepping", False)
        else "synchronous"
    )
    refused = []
    if form not in CD_MOMENTUM_FORMS:
        refused.append(f"{form} momentum (cd_scheme CD_MOMENTUM_FORMS {CD_MOMENTUM_FORMS})")
    if stepping not in CD_STEPPING_MODES:
        refused.append(f"{stepping} stepping (cd_scheme CD_STEPPING_MODES {CD_STEPPING_MODES})")
    if not coriolis:
        refused.append("useCoriolis=.FALSE. (MITgcm's cfFacMom is 0, cd_scheme's 1)")
    if refused:
        _record_gap(
            record,
            {"source": "input/data:PARM01.useCDscheme",
             "parameter": "<derived:cd_scheme composition>", "value": True},
            "useCDscheme with " + "; ".join(refused),
        )
        return None
    return _entry(
        "cd_scheme",
        coriolis_profile_c=profile,
        tau_cd=float(asm.staged("tau_cd", 0.0)),
        r_cd=float(asm.staged("r_cd", -1.0)),
        eps_ab_cd=asm.staged("eps_ab_cd"),
        cpp_defines=_cd_code_options(case_dir),
    )


def _metric_term_kwargs(asm: _Assembly, ctx: Ctx, record: Record) -> dict[str, Any]:
    """The spherical-polar metric (curvature) kwargs the flux-form bundle needs.

    ``selectMetricTerms`` is unset in every translated case, so it resolves through the
    backward-compatibility branch to ``metricTerms``, whose default is ``.TRUE.``
    (``model/src/ini_parms.F:346,716-719``) -- i.e. 1.  ``set_parms.F:54-57``
    forces it back to 0 on Cartesian grids and ``set_parms.F:65-68`` on
    curvilinear ones; ``set_parms.F:101`` zeroes it when ``momAdvection=.FALSE.``
    With it >= 1 the flux-form bundle adds the metric terms to gU/gV
    (``pkg/mom_fluxform/mom_fluxform.F:752-760,1046-1054``).

    LOCOMgcm's ``SphericalMetric`` implements the ``selectMetricTerms=1``
    analytic-``tan(phi)`` branch (``mom_u_metric_sphere.F`` /
    ``mom_v_metric_sphere.F:44-56``) and reads the two tan profiles as kwargs.
    ``FluxFormMomentum`` takes both as required keywords (the explicit-input requirement), so
    the result always carries both keys: the profiles when MITgcm computes the
    metric term, ``None`` for both when it does not (``selectMetricTerms=0``).
   The profiles are the runtime's ``$tan_phi_u`` / ``$tan_phi_v``
    sentinels over ``grid.center_lat`` / ``grid.resolution_deg`` -- the same
    uniform-latitude construction ``$coriolis_profile_c`` already relies on.

   The unrepresentable variants (the grid-spacing-gradient estimate
    ``selectMetricTerms>=2``, and the cylindrical kernels) record an
    ``UNSUPPORTED`` gap, which withholds the case yaml entirely, rather than a
    defer row: a defer would leave the terms silently absent, which is the exact
    failure this function exists to prevent.
    """
    raw = ctx.data("PARM01", "selectmetricterms", None)
    if raw is None:
        select = 1 if bool(ctx.data("PARM01", "metricterms", True)) else 0
    else:
        select = int(raw)
    if ctx.grid_family in ("cartesian", "curvilinear"):
        select = 0
    if not asm.flag("mom_advection"):
        select = 0
    if select == 0:
        return {"tan_phi_u": None, "tan_phi_v": None}
    if ctx.grid_family != "spherical_polar" or select != 1:
        gap = (
            f"selectMetricTerms={select} on a {ctx.grid_family} grid: LOCOMgcm "
            "implements only the analytic tan(phi) spherical branch "
            "(mom_u/v_metric_sphere.F, selectMetricTerms=1)"
        )
        record.add(
            source="derived:PARM01.selectMetricTerms",
            parameter="<derived:select_metric_terms>",
            value=select,
            disposition=UNSUPPORTED,
            gap=gap,
        )
        # The UNSUPPORTED row withholds the case yaml, so this pair is never
        # built into a module.
        return {"tan_phi_u": None, "tan_phi_v": None}
    record.add(
        source="derived:PARM01.selectMetricTerms",
        parameter="<derived:select_metric_terms>",
        value=select,
        disposition=MAP,
        target=[
            "model.dynamics.flux_form_momentum.tan_phi_u",
            "model.dynamics.flux_form_momentum.tan_phi_v",
        ],
        convert=(
            "metricTerms default .TRUE. -> selectMetricTerms=1 on a spherical "
            "grid -> analytic tan(phi) profiles at U/V points"
        ),
    )
    kwargs: dict[str, Any] = {"tan_phi_u": "$tan_phi_u", "tan_phi_v": "$tan_phi_v"}
    if asm.staged("recip_rsphere") is not None:
        # rSphere set in the namelist (_f_rsphere); unset is MITgcm's 6370e3
        # (ini_parms.F:1355), the module default.
        kwargs["recip_rsphere"] = asm.staged("recip_rsphere")
    return kwargs


def _horizontal_tracer_diffusion(
    asm: _Assembly, yaml_tree: dict[str, Any], tracers: tuple[str, ...], notes: list[str]
) -> list[dict[str, Any]]:
    """HorizontalTracerDiffusion entries for the stepped tracers' own coefficients.

    MITgcm diffuses temperature with diffKhT, diffK4T and salinity with diffKhS,
    diffK4S (temp_integrate.F:341, :356; salt_integrate.F:339, :354), each 0 when
    omitted (set_defaults.F:153-156).  The module diffuses each of its
    tracer_names with one diffusivity and one diffusivity_k4
    (horizontal_tracer_diffusion.py:248-252, :266-269), so tracers with equal
    coefficients share an instance and a tracer with others gets its own; a tracer
    with both zero gets none.  physics.diffusivity (diffKhT) is the second spelling
    of the module's diffusivity (config/_double_track.py:27, checked at load by
    legacy_bridge._assert_no_double_track); when an instance carries another value
    it is left out, a case may write one spelling only (_double_track.py:1-9).
    """
    coefficients = {
        "temp": (float(asm.yaml("physics.diffusivity", 0.0)),
                 float(asm.staged("diffusivity_k4_temp", 0.0))),
        "salt": (float(asm.staged("diffusivity_kh_salt", 0.0)),
                 float(asm.staged("diffusivity_k4_salt", 0.0))),
    }
    groups: dict[tuple[float, float], list[str]] = {}
    for tracer in tracers:
        if coefficients[tracer] != (0.0, 0.0):
            groups.setdefault(coefficients[tracer], []).append(tracer)
    entries = []
    for (kh, k4), names in groups.items():
        # diffusivity_k4 only when set, so a case without it keeps its bytes
        k4_kwargs = {"diffusivity_k4": k4} if k4 != 0.0 else {}
        entries.append(
            _entry("horizontal_tracer_diffusion", diffusivity=kh, tracer_names=names, **k4_kwargs)
        )
    declared = asm.yaml("physics.diffusivity")
    if declared is not None and any(kh != float(declared) for kh, _ in groups):
        del yaml_tree["physics"]["diffusivity"]
        notes.append(
            "temperature and salinity horizontal diffusion differ: one "
            "horizontal_tracer_diffusion per coefficient pair, physics.diffusivity left out"
        )
    return entries


def _build_model_assembly(
    yaml_tree: dict[str, Any], case_dir: Path, ctx: Ctx, record: Record
) -> None:
    """Turn the staged switches into ``model.dynamics``.

    One shared derivation, no case names: every branch is a switch the namelist
    set (or a cited MITgcm default), and the grid family only chooses which
    Coriolis map MITgcm itself would have chosen.  Each emitted entry lands in
    ``mapping.derived`` with its rationale, so the audit target stays the
    registry record rather than this function.
    """
    asm = _Assembly(yaml_tree.pop(_ASSEMBLY, {}), yaml_tree)
    penetrating_sw, sw_header = _resolve_penetrating_sw(
        case_dir, asm.staged("select_penetrating_sw")
    )
    _set_dotted(yaml_tree, "model.stepper.kwargs.penetrating_sw", penetrating_sw)
    record.derived.append({
        "source": str(sw_header),
        "target": "model.stepper.kwargs.penetrating_sw",
        "value": penetrating_sw,
        "rationale": (
            "SHORTWAVE_HEATING and selectPenetratingSW>=1; the oceanic unset "
            "default follows ini_parms.F:653-657. Shared by the surface forcing "
            "producer, tracer penetration and all KPP paths; independent of Qsw values."
        ),
    })
    dynamics: list[dict[str, Any]] = []
    notes: list[str] = []

    # PARM04 silence is resolved before INI_GRID.  Retain that cited value for
    # Cartesian coordinates and, for spherical grids, for Tier-0 longitude/
    # latitude reconstruction without changing any explicit-origin emission.
    if ctx.grid_family in ("cartesian", "spherical_polar"):
        for axis in ("x", "y"):
            if ctx.data("PARM04", f"{axis}gorigin") is not None:
                continue
            origin = asm.get(f"{axis}g_origin")
            unit = "m" if ctx.grid_family == "cartesian" else "deg"
            _set_dotted(
                yaml_tree,
                f"case.extras.mitcase.{axis}g_origin_{unit}",
                origin,
            )

    # Flux-form assemblies carry only the centre Coriolis profile at runtime;
    # Tier-0 fCoriG needs the ny+1 corner values (ini_cori.F:77-78).  Retain
    # the same cited construction _coriolis_profile builds for the model, so
    # ny=1 columns keep a verified corner profile instead of an extrapolation.
    if ctx.grid_family in ("cartesian", "cylindrical"):
        corner_profile = _coriolis_profile(asm, ctx, corner=True)
        if isinstance(corner_profile, tuple):
            _set_dotted(
                yaml_tree,
                "case.extras.mitcase.coriolis_profile_g",
                [float(v) for v in corner_profile],
            )

    gravity = float(asm.yaml("model.stepper.kwargs.gravity", _MIT_DEFAULT_GRAVITY))
    # rhoNil scales the linear EOS (find_rho.F:101-108, rhoNil*(sBeta*dS - tAlpha*dT) +
    # rhoNil - rhoConst); rhoConst is the Boussinesq reference everything else reads.
    rho_nil = asm.number("rho_nil")
    rho = rho_nil if asm.staged("reference_density") is None else asm.number("reference_density")
    mom = asm.flag("mom_stepping")
    implicit_vertical_viscosity = asm.flag("implicit_vertical_viscosity")
    _set_dotted(
        yaml_tree,
        "physics.implicit_vertical_viscosity",
        implicit_vertical_viscosity,
    )
    _vertical_mixing_parameters(asm, yaml_tree, case_dir, record)
    temp_stepping = asm.flag("temp_stepping")
    salt_stepping = asm.flag("salt_stepping")
    if ctx.data("PARM03", "doab_ongtgs") is None:
        do_ab_on_gt_gs = asm.flag("do_ab_on_gt_gs")
        temperature_ab, salinity_ab = _tracer_tendency_ab_gates(ctx)
        for target, resolved in _tracer_tendency_ab_writes(
            do_ab_on_gt_gs=do_ab_on_gt_gs,
            temperature=temperature_ab,
            salinity=salinity_ab,
        ).items():
            _set_dotted(yaml_tree, target, resolved)
    wind, surface_flux = _wire_core_surface_forcing(
        yaml_tree,
        case_dir,
        ctx,
        asm,
        record,
        reference_density=rho,
        mom_stepping=mom,
        temp_stepping=temp_stepping,
        salt_stepping=salt_stepping,
    )
    if asm.staged("use_exf", False):
        if wind is not None or surface_flux is not None:
            _record_gap(
                record,
                {"source": "input/data:PARM05 + input/data.pkg:PACKAGES.useEXF",
                 "parameter": "<derived:core forcing files with EXF>", "value": None},
                "core surface-forcing files beside pkg/exf: EXF overwrites fu, fv, Qnet "
                "and EmPmR after EXTERNAL_FIELDS_LOAD reads them (load_fields_driver.F:"
                "209-236, exf_mapfields.F); a mit_exf_flux source is the only surface "
                "source of a run (_canonical_helpers.enrich_manifest)",
            )
        else:
            wind, surface_flux = _wire_exf(
                yaml_tree, case_dir, ctx, asm, record, reference_density=rho,
                mom_stepping=mom, temp_stepping=temp_stepping, salt_stepping=salt_stepping,
            )
    obcs_upwind = _wire_obcs_analytic(yaml_tree, ctx, asm, record)
    if obcs_upwind is None and asm.staged("analytic_obcs") is None:
        obcs_upwind = _wire_obcs_binary(yaml_tree, case_dir, ctx, asm, record)

    # --- free surface: implicitFreeSurface is guaranteed by its own rule -----
    dynamics.append(_entry("linear_free_surface_continuity"))
    dynamics.append(_entry("linear_free_surface_pressure_gradient", gravity=gravity))

    # --- momentum -----------------------------------------------------------
    if mom:
        advective = asm.flag("mom_advection")
        coriolis = asm.flag("use_coriolis")
        vector_invariant = asm.flag("vector_invariant")
        # The NLFS momentum rescale reads the form itself (timestep.F:275-276),
        # also when momAdvection=.FALSE. leaves no core naming it.
        _set_dotted(yaml_tree, "model.stepper.kwargs.vector_invariant_momentum", vector_invariant)
        profile = _coriolis_profile(asm, ctx, corner=vector_invariant)
        if profile is None:
            notes.append(
                "beta-plane Coriolis needs a uniform delY to place yC; "
                "falling back to the f-plane intercept"
            )
            record.add(
                source="derived:model.dynamics",
                parameter="<derived:coriolis_map>",
                value=None,
                disposition=DEFER,
                reason=(
                    "beta-plane Coriolis profile needs a uniform delY "
                    "(delXfile/delYfile case); f-plane f0 used instead"
                ),
            )
            profile = (asm.number("f0"),) * (ctx.ny + 1 if vector_invariant else ctx.ny)
        key = "coriolis_profile_g" if vector_invariant else "coriolis_profile_c"
        cd_scheme = bool(asm.staged("use_cd_scheme", False))
        # useCDscheme takes the Coriolis term off the C grid (mom_fluxform.F:1082,
        # mom_vecinv.F:680-682); cd_scheme adds its own (_cd_scheme_entry).
        c_grid_coriolis = coriolis and not cd_scheme
        cori_kwargs: dict[str, Any] = {key: profile} if c_grid_coriolis else {}
        if advective:
            name = (
                "vector_invariant_momentum_advection"
                if vector_invariant
                else "flux_form_momentum"
            )
            # Spherical metric terms are a flux-form-only bundle member:
            # mom_vecinv.F calls no MOM_U/V_METRIC_SPHERE at all (only the NH
            # pair at mom_vecinv.F:902-903) because the vector-invariant form
            # carries the curvature inside its vorticity / KE-gradient
            # discretisation.
            mom_kwargs = dict(cori_kwargs)
            if c_grid_coriolis:
                mom_kwargs.update(
                    _cori_scheme_kwargs(ctx, record, name, vector_invariant=vector_invariant)
                )
            if vector_invariant:
                # selectVortScheme and the corner vorticity reconstruction as MITgcm
                # resolves them (_mit_vort_scheme; mom_vecinv.F:741-790), written
                # whether or not the namelist sets them; the grids a combination
                # does not support are refused at model assembly
                # (vector_invariant_momentum.validate_vorticity_composition).
                # selectKEscheme is staged only when set; unset is MITgcm's 0
                # (set_defaults.F:239), the module default.
                high_order = asm.flag("high_order_vorticity")
                upwind = asm.flag("upwind_vorticity")
                mom_kwargs["select_vort_scheme"] = _mit_vort_scheme(asm)
                mom_kwargs["vort_reconstruction"] = (
                    "high_order" if high_order else "upwind" if upwind else "centered"
                )
                if asm.staged("select_ke_scheme") is not None:
                    mom_kwargs["select_ke_scheme"] = asm.staged("select_ke_scheme")
                    # the declaration the MITgcm export reads (_double_track.py)
                    _set_dotted(yaml_tree, "physics.ke_scheme", asm.staged("select_ke_scheme"))
            else:
                # FluxFormMomentum requires the Coriolis profile declared;
                # None is useCoriolis=.FALSE. (the explicit-input requirement).
                mom_kwargs.setdefault("coriolis_profile_c", None)
                mom_kwargs.update(_metric_term_kwargs(asm, ctx, record))
            dynamics.append(_entry(name, **mom_kwargs))
        elif c_grid_coriolis:
            # momAdvection=.FALSE. still leaves the Coriolis term standing
            # (set_parms.F:88-95 gates them independently), and the flux-form
            # bundle has no advection-off switch, so the bare term is used.
            dynamics.append(_entry(
                "f_coriolis", **cori_kwargs,
                **_cori_scheme_kwargs(ctx, record, "f_coriolis", vector_invariant=vector_invariant),
            ))
        if cd_scheme:
            cd_entry = _cd_scheme_entry(
                asm, record, case_dir, profile, vector_invariant=vector_invariant, coriolis=coriolis
            )
            if cd_entry is not None:
                dynamics.append(cd_entry)
        if asm.flag("mom_viscosity"):
            # no_slip_sides defaults .TRUE. in MITgcm but false in the LOCOMgcm
            # schema, so the MIT value is written back into physics.* rather
            # than only into the module kwargs: the two ends must not disagree.
            if asm.yaml("physics.no_slip_sides") is None:
                _set_dotted(yaml_tree, "physics.no_slip_sides", asm.flag("no_slip_sides"))
            scheme = asm.yaml("physics.viscosity_scheme")
            if scheme is None:
                # No Leith/Smagorinsky coefficient: the constant viscosity in
                # the momentum form's own stencil.  Vector-invariant momentum
                # calls MOM_VI_HDISSIP (mom_vecinv.F:429-434), whose constant
                # branch is the div-vort form (mom_vi_hdissip.F:73-103) =
                # 'divvort' (HorizontalViscosity._divvort_dissipation); the
                # flux form takes the flux-divergence Laplacian
                # (mom_fluxform.F:612-618) = 'constant' (_u_laplacian).  Both
                # add MOM_U/V_SIDEDRAG for no_slip_sides (mom_vecinv.F:468-475, :578-585)
                # = _u_side_drag / _v_side_drag.  useStrainTensionVisc, the
                # third stencil, has a row that refuses it.
                scheme = "divvort" if vector_invariant else "constant"
            # Staged only when the namelist sets them; an absent keyword is
            # MITgcm's default, which is the module default (0, .FALSE., and
            # for the caps 1e21, a cap no viscosity reaches = LOCOMgcm's 0.0).
            variable_visc = {
                key: asm.staged(key)
                for key in (
                    "visc_c2_leith_d", "use_full_leith", "visc_c4_leith",
                    "visc_c4_leith_d", "visc_c4_smag", "visc_ah_grid_max",
                    "visc_a4_grid_max", "viscosity_a4",
                )
                if asm.staged(key) is not None
            }
            if "visc_ah_grid_max" in variable_visc or "visc_a4_grid_max" in variable_visc:
                # The caps scale with 1/deltaTmom (mom_init_fixed.F:42-43,
                # mom_calc_visc.F:166-167); the module scales them with dt.
                variable_visc["dt"] = _mit_delta_t_mom(ctx)
            dynamics.append(
                _entry(
                    "horizontal_viscosity",
                    viscosity=float(asm.yaml("physics.viscosity", 0.0)),
                    no_slip_sides=bool(asm.yaml("physics.no_slip_sides", False)),
                    viscosity_scheme=str(scheme),
                    visc_c2_leith=float(asm.yaml("physics.visc_c2_leith", 0.0)),
                    visc_c2_smag=float(asm.yaml("physics.visc_c2_smag", 0.0)),
                    **variable_visc,
                )
            )
            vertical_viscosity = float(
                asm.yaml("physics.implicit_viscosity", 0.0)
            )
            if not implicit_vertical_viscosity and vertical_viscosity > 0.0:
                dynamics.append(
                    _entry(
                        "vertical_mixing",
                        tracer_names=(),
                        diffusivity=0.0,
                        mix_momentum=True,
                        viscosity=vertical_viscosity,
                        tendency_phase="momentum",
                    )
                )
            # MIT's explicit bottom drag is part of momViscosity even when
            # momAdvection is disabled. MOM_FLUXFORM still computes KE scheme
            # 2; MOM_VECINV uses selectKEscheme independently of advection.
            dynamics.append(
                _entry(
                    "bottom_drag",
                    linear=asm.number("bottom_drag_linear"),
                    quadratic=asm.number("bottom_drag_quadratic"),
                    no_slip_bottom=asm.flag("no_slip_bottom"),
                    vertical_viscosity=float(asm.yaml("physics.implicit_viscosity", 0.0)),
                    ke_scheme=(
                        int(asm.staged("select_ke_scheme", 0))
                        if vector_invariant else 2
                    ),
                )
            )
        if wind is not None:
            dynamics.append(wind)
    elif asm.staged("zonal_wind_file") or asm.staged("merid_wind_file"):
        notes.append(
            "a surface wind stress file is declared but momStepping=.FALSE., so "
            "MITgcm never applies it (apply_forcing.F is reached from the "
            "momentum step); no forcing source is wired"
        )

    # --- tracers ------------------------------------------------------------
    tracers = tuple(
        name for name, enabled in (("temp", temp_stepping), ("salt", salt_stepping)) if enabled
    )
    if tracers:
        advected_tracers = tuple(
            tracer
            for tracer, enabled in (
                ("temp", temp_stepping and asm.flag("temp_advection")),
                ("salt", salt_stepping and asm.flag("salt_advection")),
            )
            if enabled
        )
        if advected_tracers:
            # MIT's advection defaults differ from the LOCOMgcm schema's
            # (dst3_fl, MIT code 33), so an absent tempAdvScheme is written back
            # into advection.tracer rather than only into the module kwargs --
            # otherwise the yaml block and the module would disagree.  The MIT
            # code is not written: cfg2/6 removed advection.mitgcm_adv_scheme and
            # the MIT side reads TRACER_SCHEME_BY_NAME[advection.tracer].mit_code
            # (MitComparisonConfig.effective_tracer_scheme).
            horizontal_schemes = {
                tracer: _effective_mit_tracer_adv_scheme(ctx, tracer)
                for tracer in advected_tracers
            }
            distinct_horizontal_schemes = set(horizontal_schemes.values())
            if len(distinct_horizontal_schemes) != 1:
                _record_gap(
                    record,
                    {
                        "source": "derived:gad_init_fixed.F:125-137",
                        "parameter": "<derived:shared horizontal tracer scheme>",
                        "value": horizontal_schemes,
                    },
                    "LOCOMgcm carries one horizontal tracer scheme but the "
                    "effective MIT temperature and salinity schemes differ",
                )
            # TracerAdvectionScheme.halo is the reach of LOCOMgcm's face stencil, which
            # takes the centred branch on every cell.  A MITgcm tile carries OLx, OLy
            # overlap cells and takes a one-sided oscillation indicator on the
            # outermost one (gad_osc_hat_x.F:27-28, :48-94; gad_osc_hat_y.F:27-28,
            # :48-94), so where the reach exceeds the overlap its result depends on
            # the tile layout.  MITgcm stops below its own minimum, the scheme's
            # GAD_advScheme_OlMin (gad_init_fixed.F:44-60, :169-176;
            # gad_check.F:115-118), which equals halo for every scheme but ppm_weno
            # (42): minimum 3, reach 4, because its hat loop from 1-OLx+2 calls
            # GAD_OSC_MUL_X(ix,+2) over ohat(ix-2..ix+2) (gad_ppm_hat_x.F:64, :111;
            # gad_osc_mul_x.F:33) and the flux loop from 1-OLx+3 reads that hat
            # (gad_ppm_flx_x.F:63; y: gad_ppm_hat_y.F:65, :112, gad_ppm_flx_y.F:63).
            # pqm_weno (52) starts its hat loop at 1-OLx+3 (gad_pqm_hat_x.F:72),
            # so it never reads the one-sided cell.
            overlap = min(int(ctx.size_h["OLx"]), int(ctx.size_h["OLy"]))
            beyond_overlap = {
                tracer: code
                for tracer, code in horizontal_schemes.items()
                if code in _ADV_SCHEME_MAP
                and TRACER_SCHEME_BY_NAME[_ADV_SCHEME_MAP[code]].halo > overlap
            }
            if beyond_overlap:
                _record_gap(
                    record,
                    {
                        "source": "derived:gad_osc_hat_x.F:48-94",
                        "parameter": "<derived:tracer stencil beyond tile overlap>",
                        "value": {
                            **beyond_overlap,
                            "OLx": ctx.size_h["OLx"],
                            "OLy": ctx.size_h["OLy"],
                        },
                    },
                    "the horizontal tracer stencil reaches past the tile overlap, "
                    "where MITgcm takes a one-sided oscillation indicator, so its "
                    "result depends on the tile layout",
                )
            mit_scheme = next(iter(horizontal_schemes.values()))
            scheme = asm.yaml("advection.tracer")
            if len(distinct_horizontal_schemes) == 1 and mit_scheme in _ADV_SCHEME_MAP:
                scheme = _ADV_SCHEME_MAP[mit_scheme]
                _set_dotted(yaml_tree, "advection.tracer", scheme)
            elif scheme is None:
                scheme = _ADV_SCHEME_MAP[_MIT_DEFAULT_ADV_SCHEME]
                _set_dotted(yaml_tree, "advection.tracer", scheme)
                notes.append(
                    f"tempAdvScheme absent: MIT default {_MIT_DEFAULT_ADV_SCHEME} "
                    f"-> {scheme} (set_defaults.F:226)"
                )
            requested_multi_dim = asm.yaml("advection.multi_dim")
            if requested_multi_dim is None:
                requested_multi_dim = _MIT_DEFAULT_MULTI_DIM
            forced_non_multidim = mit_scheme in _MIT_AB_ADV_SCHEMES
            multi_dim = bool(requested_multi_dim) and not forced_non_multidim
            _set_dotted(yaml_tree, "advection.multi_dim", bool(requested_multi_dim))
            tracer_kwargs: dict[str, Any] = {
                "tracer_names": list(advected_tracers),
                "scheme": str(scheme),
                "multi_dim": bool(multi_dim),
                "dt": float(asm.yaml("time.dt", 0.0)),
            }
            implicit_columns = all(
                ctx.data("PARM01", f"{tracer}implvertadv", False)
                for tracer in advected_tracers
            )
            if not multi_dim and not implicit_columns:
                # Multi-dimensional advection off (multiDimAdvection=.FALSE., or
                # forced by gad_init_fixed.F:125-137): GAD_CALC_RHS takes the
                # horizontal and vertical fluxes of each scheme on the same state
                # in one RHS (gad_calc_rhs.F:769-781), the combined route of
                # HorizontalTracerAdvection for every horizontal and vertical scheme.
                tracer_kwargs["combined_with_vtt"] = True
                notes.append(
                    "multi-dimensional advection off ("
                    + (f"gad_init_fixed.F:125-137 for scheme {mit_scheme}"
                       if forced_non_multidim else "multiDimAdvection=.FALSE.")
                    + "): horizontal and vertical fluxes in one RHS, "
                    "gad_calc_rhs.F:769-781 (combined_with_vtt)"
                )
            if obcs_upwind is not None:
                tracer_kwargs["obcs_upwind"] = obcs_upwind
            dynamics.append(_entry("horizontal_tracer_advection", **tracer_kwargs))
        dynamics.extend(_horizontal_tracer_diffusion(asm, yaml_tree, tracers, notes))
        if asm.staged("use_gmredi"):
            # GAD_CALC_RHS adds the GM fluxes to the diffusive flux of every
            # tracer (gad_calc_rhs.F:343-351,472-480,624-632)
            dynamics.append(_gmredi_entry(asm, ctx, case_dir, record, tracers,
                                          gravity=gravity, rho_const=rho))
        vertical_diffusivity = float(asm.yaml("physics.tracer.diffusivity_kr", 0.0))
        if vertical_diffusivity != 0.0:
            implicit_diffusion = asm.yaml(
                "physics.tracer.implicit_vertical_diffusion", _MISSING
            )
            if implicit_diffusion is _MISSING:
                implicit_diffusion = asm.flag("implicit_vertical_diffusion")
            if not bool(implicit_diffusion):
                vertical_mixers = [
                    entry for entry in dynamics if entry["name"] == "vertical_mixing"
                ]
                if len(vertical_mixers) > 1:
                    raise TranslationError(
                        "explicit vertical tracer diffusion found multiple "
                        "vertical_mixing entries"
                    )
                if vertical_mixers:
                    vertical_mixers[0]["kwargs"].update(
                        tracer_names=list(tracers),
                        diffusivity=vertical_diffusivity,
                    )
                else:
                    dynamics.append(
                        _entry(
                            "vertical_mixing",
                            tracer_names=list(tracers),
                            diffusivity=vertical_diffusivity,
                        )
                    )
        relaxation = _wire_surface_climatology(
            yaml_tree, case_dir, ctx, asm, record,
            temp_stepping=temp_stepping, salt_stepping=salt_stepping,
        )
        if relaxation is not None:
            dynamics.append(relaxation)
        if surface_flux is not None:
            dynamics.append(surface_flux)
        # The equation-of-state module carries eosType (EOS_TYPES); an absent
        # eosType is MITgcm's LINEAR (set_defaults.F:175).
        eos_type = asm.staged("eos_type")
        if eos_type is None:
            eos_type = _MIT_DEFAULT_EOS
            notes.append(
                f"eosType absent: MIT default {eos_type} -> {_EOS_MODULES[eos_type]} "
                "(set_defaults.F:175)"
            )
        eos_module = _EOS_MODULES[eos_type]
        if eos_module == "linear_equation_of_state":
            eos_kwargs: dict[str, Any] = {
                "reference_density": rho_nil,
                "thermal_expansion": asm.number("t_alpha"),
                "haline_contraction": asm.number("s_beta"),
            }
        else:
            eos_kwargs = {"gravity": gravity, "reference_density": rho}
            eos_kwargs.update(_eos_pressure_kwargs(_mit_selectp_in_eos_zc(ctx)))
        dynamics.append(_entry(eos_module, **eos_kwargs))
        hpg_kwargs: dict[str, Any] = {"gravity": gravity, "reference_density": rho}
        if eos_module == "linear_equation_of_state":
            hpg_kwargs["thermal_expansion"] = asm.number("t_alpha")
            hpg_kwargs["haline_contraction"] = asm.number("s_beta")
            if rho_nil != rho:
                # unwritten, rho_nil is reference_density (hydrostatic_pressure.py:68-72)
                hpg_kwargs["rho_nil"] = rho_nil
        if mom and asm.staged("pload_file"):
            hpg_kwargs["surface_pressure_load"] = True
        dynamics.append(_entry("hydrostatic_pressure_gradient", **hpg_kwargs))

    if asm.staged("c_adj_freq") is not None:
        adjustment = _convective_adjustment_entry(
            asm, ctx, record, case_dir, tracers=bool(tracers)
        )
        if adjustment is not None:
            dynamics.append(adjustment)
    pload_file = asm.staged("pload_file")
    if pload_file and not mom:
        notes.append(
            "pLoadFile is read but momStepping=.FALSE.: phi0surf reaches the state only "
            "through CALC_PHI_HYD in DYNAMICS (dynamics.F:482, forward_step.F:786-792), "
            "so no consumer is wired"
        )
    elif pload_file and not tracers:
        # MITgcm still adds phi0surf to dPhiHydX (calc_grad_phi_hyd.F:127-133); the
        # pressure-gradient module that would carry it is built with the tracers.
        _record_gap(
            record,
            {"source": "input/data:PARM05.pLoadFile",
             "parameter": "<derived:surface_pressure_load consumer>", "value": pload_file},
            "pLoadFile with momStepping and no tracer stepping: the hydrostatic pressure "
            "gradient that carries phi0surf is not assembled",
        )

    dynamics.append(_entry("diagnostic_vertical_velocity"))

    # The stepper refuses to guess this one (implicit_free_surface.py:1018);
    # MITgcm derives it from forcing_In_AB when momForcingOutAB is unset
    # (ini_parms.F:1095-1097), so the same derivation is applied here.
    stepper_kwargs = (
        yaml_tree.setdefault("model", {})
        .setdefault("stepper", {})
        .setdefault("kwargs", {})
    )
    if bool(asm.yaml("physics.nonhydrostatic.enabled", False)):
        # CALC_GW consumes these independently of the U/V module list.
        stepper_kwargs["nh_mom_advection"] = asm.flag("mom_advection")
        stepper_kwargs["nh_mom_viscosity"] = asm.flag("mom_viscosity")
    check_ini_temp = asm.flag("check_ini_temp")
    if "allow_freezing" in stepper_kwargs:
        allow_freezing = bool(stepper_kwargs["allow_freezing"])
    else:
        allow_freezing = asm.flag("allow_freezing")
    _set_dotted(yaml_tree, "initial_condition.check_ini_temp", check_ini_temp)
    _set_dotted(yaml_tree, "initial_condition.allow_freezing", allow_freezing)
    for key in (
        "implicit_surf_press",
        "implicit_div_2d_flow",
        "exact_conserv",
        "ab_eps",
    ):
        if key not in stepper_kwargs:
            stepper_kwargs[key] = asm.get(key)
    use_real_freshwater_flux = asm.staged("use_real_freshwater_flux")
    if (
        use_real_freshwater_flux is not None
        and "use_real_freshwater_flux" not in stepper_kwargs
    ):
        stepper_kwargs["use_real_freshwater_flux"] = bool(use_real_freshwater_flux)
    if "mom_forcing_out_ab" not in stepper_kwargs:
        stepper_kwargs["mom_forcing_out_ab"] = not asm.flag("forcing_in_ab")
        notes.append(
            "momForcingOutAB absent: derived from forcing_In_AB "
            f"({asm.flag('forcing_in_ab')}) per ini_parms.F:1095-1097"
        )
    if stepper_kwargs.get("kpp_enabled", False):
        if "kpp_ricr" not in stepper_kwargs:
            stepper_kwargs["kpp_ricr"] = asm.get("kpp_ricr")
        if "kpp_min_hbl" not in stepper_kwargs:
            # None preserves the stepper's grid-derived -rC(1) path.
            assert asm.get("kpp_min_hbl") is None
        if ctx.grid_family == "cartesian":
            f0 = asm.number("f0")
            beta = asm.number("beta")
            if beta == 0.0:
                stepper_kwargs.setdefault("kpp_coriolis", f0)
            else:
                _record_gap(
                    record,
                    {
                        "source": "input/data:PARM01.beta",
                        "parameter": "<derived:kpp_coriolis capability>",
                        "value": beta,
                    },
                    (
                        "KPP static fields have no beta-plane path; a scalar f0 "
                        "approximation is forbidden when beta is nonzero"
                    ),
                )

    # The builder refuses a nameless boundary condition
    # (builders/from_manifest.py:118).  Authorised analytic and prescribed
    # binary OBCS were wired above; all other cases retain the no-op condition,
    # with their defer/gap row declaring the approximation.
    yaml_tree["model"].setdefault("boundary_condition", {"name": "passthrough"})
    _wire_stepping_switches(yaml_tree, asm, record)
    _check_advection_section(yaml_tree, record)
    if asm.yaml("grid.vertical_center_spacing"):
        _check_center_spacing_readers(yaml_tree, dynamics, record)

    yaml_tree.setdefault("model", {})["dynamics"] = dynamics
    record.derived.append(
        {
            "target": "model.dynamics",
            "value": [entry["name"] for entry in dynamics],
            "rationale": (
                "assembled from the staged momStepping / momAdvection / "
                "momViscosity / useCoriolis / vectorInvariantMomentum / "
                "tempStepping / saltStepping / tempAdvection / saltAdvection "
                "switches plus the "
                "already-mapped physics.* and advection.* blocks; the grid "
                "family only selects the Coriolis map MITgcm derives at "
                "set_parms.F:73-81"
            ),
            "mit_defaults_applied": asm.defaults_used,
            "notes": notes,
        }
    )
    _record_silent_divergences(ctx, asm, record)


#: Modules whose data source the run attaches when it builds them
#: (locomgcm/io/runner/_canonical_helpers.py, the enrichment loop that sets each
#: one's ``source`` or target values); they are constructed by the run, not here.
_RUN_SOURCED_MODULES = frozenset({
    "era5_wind_forcing", "era5_bulk_forcing", "wpflux_surface_forcing",
    "surface_climate_relaxation",
    "tracer_boundary_sponge_relaxation", "momentum_boundary_sponge_relaxation",
})
#: Modules the run attaches data to only when a kwarg declares it: the pressure
#: gradient gets ``pload`` beside surface_pressure_load (the same enrichment loop).
_RUN_SOURCED_WHEN = {"hydrostatic_pressure_gradient": "surface_pressure_load"}


def _run_sourced(entry: dict[str, Any]) -> bool:
    flag = _RUN_SOURCED_WHEN.get(entry["name"])
    return entry["name"] in _RUN_SOURCED_MODULES or (
        flag is not None and bool(entry.get("kwargs", {}).get(flag, False))
    )


def _try_construct(yaml_tree: dict[str, Any], record: Record) -> None:
    """Construct every ``model.dynamics`` entry and the stepper from their kwargs.

    The modules check their own configuration at construction (scheme tuples,
    fields the selected branch does not read, the stepper's admitted levels), so
    their refusal is the capability decision; its message is the gap.  Refusals
    that need the grid stay with the model build.  A TypeError is the translator's
    kwargs not fitting the constructor (a required field it does not write, a key
    the module does not have): a defect of this module's assembly, raised as a
    TranslationError naming the entry, so translate_case returns STATUS_ERROR for
    this case alone.
    """
    model = yaml_tree["model"]
    stepper_name = model["stepper"]["name"]
    stepper_kwargs = model["stepper"].get("kwargs", {})
    built = [(entry["name"], DYNAMICS_MODULES, entry.get("kwargs", {}))
             for entry in model["dynamics"] if not _run_sourced(entry)]
    built.append((stepper_name, STEPPERS, stepper_kwargs))
    for name, registry, kwargs in built:
        try:
            if registry is DYNAMICS_MODULES:
                # The fields the run's assembly writes from the stepper (enrich_manifest).
                kwargs = dict(kwargs)
                complete_stepper_owned_kwargs(name, kwargs, stepper_name=stepper_name,
                                              stepper_kwargs=stepper_kwargs)
            registry.instantiate(name, **kwargs)
        except (ValueError, NotImplementedError) as exc:
            _record_gap(
                record,
                {"source": f"derived:model construction of {name}",
                 "parameter": f"<construct:{name}>", "value": None},
                f"{type(exc).__name__}: {exc}",
            )
        except TypeError as exc:
            raise TranslationError(
                f"model construction of {name}: {exc}; the translated kwargs do not fit "
                "its constructor"
            ) from exc


def _wire_initial_condition_source(
    yaml_tree: dict[str, Any], case_dir: Path, ctx: Ctx, record: Record
) -> None:
    """Point the stage1a ``state`` lane at this case's ``input/`` binaries.

    The source block already holds whichever ``PARM05`` file names the namelist
    declared; this adds the directory and the precision that make them readable,
    pins the undeclared roles to ``None`` so the source's exp4-shaped defaults
    cannot be picked up by accident, and declares the one-step pipeline that
    turns the bundle into the canonical state.
    """
    prec = int(ctx.data("PARM01", "readbinaryprec", 32) or 32)
    source: dict[str, Any] = yaml_tree.setdefault("sources", {}).setdefault(
        _IC_SOURCE_NAME, {}
    )
    for kwarg in ("bathy_file", "temp_file", "salt_file", "u_file", "v_file", "eta_file"):
        source.setdefault(kwarg, None)
    source.setdefault("t_ref", _fill_ref(None, ctx.nz, _MIT_DEFAULT_TREF, "tref"))
    source.setdefault("s_ref", _fill_ref(None, ctx.nz, _MIT_DEFAULT_SREF, "sref"))
    yaml_tree["case"]["extras"].setdefault("obcs_sides", list(_OBCS_SIDES))
    source["type"] = "mit_input_binary_ic"
    source["mit_verification_dir"] = str(case_dir.resolve())
    source["read_binary_prec"] = prec
    mesh_source = yaml_tree["sources"].setdefault(_MESH_SOURCE_NAME, {})
    mesh_source.update(
        {
            "type": "mit_input_binary_mesh",
            "mit_verification_dir": source["mit_verification_dir"],
            "read_binary_prec": prec,
            "bathy_file": source["bathy_file"],
        }
    )
    boundary_source = (
        _OBCS_SOURCE_NAME
        if _OBCS_SOURCE_NAME in yaml_tree.get("sources", {})
        else _IC_SOURCE_NAME
    )
    boundary_vars = ["temperature", "salinity", "eta", "u", "v"]
    if boundary_source == _OBCS_SOURCE_NAME:
        boundary_vars.append("w")
    boundary_operators: list[dict[str, Any]] = [{"op": "subset_vars", "vars": boundary_vars}]
    if boundary_source == _OBCS_SOURCE_NAME:
        # mit_obcs_binary yields the MITGCM convention; ConventionConverter reaches
        # CF_STANDARD from it through LOCOMGCM, having no direct table
        # (layer_c/operators/convention_convert.py:58, _DISPATCH).
        boundary_operators.append({"op": "convention_convert", "to": "locomgcm"})
    boundary_operators.append({"op": "convention_convert", "to": "cf_standard"})
    if boundary_source == _IC_SOURCE_NAME:
        boundary_operators.append(
            {"op": "subset_boundary", "sides": list(_OBCS_SIDES), "width": 3}
        )
    yaml_tree["pipelines"] = {
        "stage1a_prepare_common": [
            {
                "name": "state",
                "primary_source": _IC_SOURCE_NAME,
                "operators": [
                    {"op": "subset_vars", "vars": ["temperature", "salinity", "eta", "u", "v"]},
                    {"op": "convention_convert", "to": "cf_standard"},
                ],
                "consumer": "prepared_state_zarr",
            },
            {
                # no convention_convert here: prepared_mesh_zarr is the one
                # stage1a consumer that requires LOCOMGCM, which is what the
                # source already produces (positive depth).
                "name": "mesh",
                "primary_source": _MESH_SOURCE_NAME,
                "consumer": "prepared_mesh_zarr",
            },
            {
                "name": "boundary",
                "primary_source": boundary_source,
                "operators": boundary_operators,
                "consumer": "prepared_boundary_zarr",
            },
        ],
        "stage1b_export_to": [
            {
                "name": "to_obc",
                "primary_source": "stage1a_boundary",
                "operators": [{"op": "convention_convert", "to": "locomgcm"}],
                "consumer": "locomgcm_obc_zarr",
            }
        ],
    }
    if _FORCING_SOURCE_NAME in yaml_tree["sources"]:
        # stage1a's canonical builder raises unless every dynamics module that
        # takes a runtime source has one (_canonical_helpers.py:540-543), and
        # its wind source comes from the 'forcing' lane bundle.  stage2 reads
        # the same source directly (case.extras.forcing_source), so the two ends
        # read the same bytes with the same kwargs.
        yaml_tree["pipelines"]["stage1a_prepare_common"].append(
            {
                "name": "forcing",
                "primary_source": _FORCING_SOURCE_NAME,
                "operators": [{"op": "convention_convert", "to": "cf_standard"}],
                "consumer": "prepared_forcing_zarr",
            }
        )
    yaml_tree["case"]["extras"]["obcs_source"] = "zarr_bundle"
    yaml_tree["case"]["extras"]["obcs_source_extras"] = {
        "stage1b_consumer": "locomgcm_obc_zarr"
    }
    record.derived.append(
        {
            "target": f"sources.{_IC_SOURCE_NAME} + pipelines.stage1a_prepare_common",
            "value": (
                "mit_input_binary_ic -> prepared_state_zarr + prepared_mesh_zarr; "
                f"{boundary_source} -> prepared_boundary_zarr"
            ),
            "rationale": (
                "the initial state is read straight from the case's own "
                f"input/*.bin at readBinaryPrec={prec} ({_dtype_of(prec)}, "
                "big-endian). Neither lane regrids -- the "
                "bytes are already on the model grid, so an interpolation step "
                "would only add error.  Bathymetry rides the 'mesh' lane "
                "because that is the only one whose reader is "
                "convention-aware (_positive_depth_from_mesh); the 'state' "
                "lane drops it, since _ic_tensor_map would take a CF elevation "
                "for a depth.  convention_convert re-tags LOCOMGCM -> "
                "CF_STANDARD for the stage1a consumers and is undone on the "
                "mesh read, so the depth that reaches the grid builder is the "
                "one MITgcm would have used.  Undeclared PARM05 roles are "
                "pinned to null so the reader applies MITgcm's own fallback "
                "(tRef/sRef, zero velocity, flat bottom at "
                "ini_depths.F:91-102) instead of an exp4 file name.  "
                "mit_verification_dir is an absolute path resolved at "
                "translation time: regenerate the yaml on another host"
            ),
        }
    )


_DERIVED_DECISIONS = [
    {
        "target": "case.kind",
        "value": "regional",
        "rationale": (
            "'mit_verification' is not usable for a LOCOMgcm-only run: setting "
            "case.mit_verification_dir makes the runtime mirror MIT dump steps "
            "and abort when <dir>/run holds no Eta dumps "
            "(experiments/cases/_runtime.py:1198-1295).  'regional' keeps the "
            "normal stage1a -> stage1b -> stage2 chain, and the case's own "
            "input/*.bin enter through the stage1a 'state' lane instead"
        ),
    },
    {
        "target": "case.extras.case_type",
        "value": "mitgcm_comparison",
        "rationale": (
            "stage4 postproc dispatches on this key; 'mitgcm_comparison' is the "
            "registered HycomAdapter value (adapters/hycom.py:14)"
        ),
    },
    {
        "target": "boundary.type + case.extras.{west_east,south_north}_boundary",
        "value": "wall + periodic",
        "rationale": (
            "the grid topology is periodic because MITgcm's process grid is "
            "doubly periodic by default (eesupp/src/ini_procs.F:124-125) -- only "
            "CAN_PREVENT_{X,Y}_PERIODICITY plus notUsing{X,Y}Periodicity walls "
            "an edge, and the translator and binary ingest read no code/ CPP options.  It is carried by "
            "case.extras, which the grid builder prefers "
            "(builders/grid_factory.py:58-72), because boundary.type itself "
            "cannot say 'periodic': the legacy bridge accepts only open|wall "
            "there (legacy_bridge.py:195).  'wall' is the honest value for that "
            "field while pkg/obcs is deferred -- a case that is closed is closed "
            "by its bathyFile land ring, and one with open boundaries needs the "
            "still-deferred data.obcs mapping"
        ),
    },
    {
        "target": "run.precision",
        "value": "float64 (loader default; key not emitted)",
        "rationale": (
            "the key is deliberately absent: stage1a's legacy bridge rejects any "
            "yaml path it does not claim (legacy_bridge.py:963-971) and "
            "'run.precision' is claimed only on the cube path, so emitting it "
            "would make every case.kind=regional case fail at stage1a.  The "
            "loader default is float64 either way (_loader.py:260)"
        ),
    },
    {
        "target": "mit_comparison.enabled",
        "value": False,
        "rationale": (
            "The translator emits LOCOMgcm-side configuration only; during MIT execution, "
            "with enabled=false the driver's default stage set is 1a,1b,2"
        ),
    },
    {
        "target": "model.stepper.name",
        "value": "implicit_free_surface",
        "rationale": (
            "the stepper that owns the mapped kwargs (gravity / implicSurfPress / "
            "implicDiv2DFlow / abEps / exactConserv / staggerTimeStep / cg2d*); "
            "the model.dynamics list beside it is built by _build_model_assembly"
        ),
    },
]


def _base_yaml(case: str, size_h: dict[str, int], grid: str) -> dict[str, Any]:
    tree: dict[str, Any] = {
        "schema_version": "0.2.0",
        "case": {
            "kind": "regional",
            "extras": {
                "case_type": "mitgcm_comparison",
                "case_name": case,
                "description": f"MITgcm verification case {case} translated to LOCOMgcm",
                # Derived from the grid family, never from the case name; the
                # builders' default for a missing key is 'cartesian'.
                **(
                    {"grid_mode": _TO_GRID_MODE[grid]}
                    if grid in _TO_GRID_MODE
                    else {}
                ),
                "west_east_boundary": "periodic",
                "south_north_boundary": "periodic",
                "mitcase": {"grid_family": grid, "source_case": case},
            },
        },
        "run": {"experiment_name": f"mitcase_{case}"},
        "grid": {"nx": size_h["nx"], "ny": size_h["ny"], "nz": size_h["nz"]},
        "mit_parallel": {
            "nPx": size_h["nPx"],
            "nPy": size_h["nPy"],
            "sNx": size_h["sNx"],
            "sNy": size_h["sNy"],
            "OLx": size_h["OLx"],
            "OLy": size_h["OLy"],
        },
        "boundary": {"type": "wall"},
        "mit_comparison": {"enabled": False},
        "model": {"stepper": {"name": "implicit_free_surface", "kwargs": {}}},
        # translator-internal staging; removed by _build_model_assembly
        _ASSEMBLY: {},
    }
    tree["case"]["extras"]["mitcase"]["size_h"] = {k: size_h[k] for k in _SIZE_KEYS}
    return tree
