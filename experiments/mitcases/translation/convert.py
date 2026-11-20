"""Conversion functions with cross-parameter context, used by the rule tables and the assembly."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from experiments.mitcases.code_inventory import MAP
from defgcm.constants import ROTATION_PERIOD_EARTH
from defgcm.dycore_parameters import resolve_3d_coriolis_controls
from defgcm.dynamics.bulk_formulae import CEN2KEL
from defgcm.dynamics.equation_of_state import EOS_TYPES
from defgcm.dynamics.implicit_vertical_advection import IMPLICIT_VERTICAL_TRACER_SCHEMES
from defgcm.geometry.vertical_coordinates import layer_thickness_from_center_spacing
from defgcm.operators.fluxes.tracer_flux import (
    HORIZONTAL_TRACER_SCHEMES,
    TRACER_ADVECTION_SCHEMES,
    VERTICAL_TRACER_SCHEMES,
)

from .core import (
    ANALYTIC_OBCS,
    CAMPAIGN_DISABLED,
    Deferred,
    Gap,
    InvalidParameter,
    Record,
    Skipped,
    TranslationError,
    _ASSEMBLY,
    _Assembly,
    _CLIMATOLOGY_SOURCE_NAME,
    _FORCING_SOURCE_NAME,
    _FORCING_SOURCE_TYPE,
    _IC_SOURCE_NAME,
    _MISSING,
    _OBCS_SOURCE_NAME,
    _OBCS_SOURCE_TYPE,
    _record_gap,
    _set_dotted,
)
from .rules import (
    Convert,
    Ctx,
    Rule,
    _as_list,
    _is_zero,
    _map,
    _mit_cori_map,
)
from .namelist import (
    _mit_namelist,
    _mit_unset,
)
from .cpp import (
    _reference_mit_root,
    parse_cpp_macro_states,
)

# --------------------------------------------------------------------------
# conversion functions with cross-parameter context
# --------------------------------------------------------------------------

#: MITgcm eosType -> LOCOMgcm equation-of-state module, as the model declares it
#: (EOS_TYPES).  JMD95Z and JMD95P name one module; its dynamic_pressure follows
#: selectP_inEOS_Zc (_eos_pressure_kwargs), whose default set_parms.F:275-282
#: derives from the eosType name.
_EOS_MODULES = {eos_type: module for eos_type, module, _dynamic in EOS_TYPES}

# set_parms.F:275-282: the eosType names whose selectP_inEOS_Zc default is 2.
_MIT_DYNAMIC_PRESSURE_EOS_TYPES = frozenset(
    {"JMD95P", "UNESCO", "MDJWF", "TEOS10"}
)

# MITgcm source-coded default (references/MITgcm/model/src/set_defaults.F:101).
_MIT_DEFAULT_GRAVITY = 9.81
_MIT_DEFAULT_CG3D_MAX_ITERS = 150
_MIT_DEFAULT_CG3D_TARGET_RESIDUAL = 1.0e-7
_MIT_DEFAULT_CG3D_TARGET_RES_W_UNIT = -1.0
# CG2D controls (model/src/set_defaults.F:286, :288, :289).
_MIT_DEFAULT_CG2D_MAX_ITERS = 150
_MIT_DEFAULT_CG2D_TARGET_RESIDUAL = 1.0e-7
_MIT_DEFAULT_CG2D_TARGET_RES_W_UNIT = -1.0
# hFacMin (model/src/set_defaults.F:180).
_MIT_DEFAULT_HFAC_MIN = 1.0

# MITgcm tracer advection scheme codes -> LOCOMgcm scheme names, as the model
# declares them (TRACER_ADVECTION_SCHEMES, the only name-to-code table) for the
# horizontal face fluxes, the explicit vertical fluxes and the implicit column
# solve.  A code outside a table is a capability gap, not a substitution.
_ADV_SCHEME_MAP = {
    s.mit_code: s.name for s in TRACER_ADVECTION_SCHEMES if s.name in HORIZONTAL_TRACER_SCHEMES
}
_VERTICAL_ADV_SCHEME_MAP = {
    s.mit_code: s.name for s in TRACER_ADVECTION_SCHEMES if s.name in VERTICAL_TRACER_SCHEMES
}
_IMPLICIT_VERTICAL_ADV_SCHEME_MAP = {
    s.mit_code: s.name
    for s in TRACER_ADVECTION_SCHEMES
    if s.name in IMPLICIT_VERTICAL_TRACER_SCHEMES
}
_MIT_DEFAULT_ADV_SCHEME = 2

# The schemes whose declaration sets adams_bashforth (MITgcm 2, 3, 4).  Only these
# retain the temperature or salinity tendency in the Adams-Bashforth history
# (gad_init_fixed.F:148-157; the two decisions are independent and use
# tempStepping / saltStepping respectively), and each active tracer using one of
# them as horizontal scheme loses the multi-dimensional route
# (gad_init_fixed.F:125-137).  A code MITgcm accepts without a declaration here
# (80, 81) is in neither MITgcm list.
_MIT_AB_ADV_SCHEMES = frozenset(s.mit_code for s in TRACER_ADVECTION_SCHEMES if s.adams_bashforth)

# Scheme numbers MITgcm itself accepts, i.e. exactly the set registered by the
# GAD_ADVSCHEME_SET calls at gad_init_fixed.F:44-60.  GAD_ADVSCHEME_GET returns
# a negative overlap for anything outside it (gad_advscheme.F:143-147), and
# GAD_INIT_FIXED then stops the run for each of the four tracer scheme slots
# (gad_init_fixed.F:87-115) -- before tempStepping / tempAdvection are read at
# :118-163.  Membership here is therefore about whether MITgcm can run the case
# at all; whether LOCOMgcm owns a kernel is the separate _ADV_SCHEME_MAP
# question.
_MIT_VALID_ADV_SCHEMES = frozenset(
    {1, 2, 3, 4, 7, 20, 30, 33, 40, 41, 42, 50, 51, 52, 77, 80, 81}
)


def _reject_unrunnable_mit_adv_scheme(
    code: int, tracer: str, *, vertical: bool
) -> None:
    """Refuse a scheme number MITgcm would stop on, whatever the activity flags.

    GAD_INIT_FIXED validates tempAdvScheme, tempVertAdvScheme, saltAdvScheme and
    saltVertAdvScheme unconditionally (gad_init_fixed.F:87-115); tempStepping /
    tempAdvection only enter afterwards, at :118-163.  Without this check an
    unrunnable number reaches the AB eligibility membership test and comes back
    out as a merely ineligible tracer, so LOCOMgcm would emit a case MITgcm
    refuses to start.
    """
    if code in _MIT_VALID_ADV_SCHEMES:
        return
    slot = f"{tracer}VertAdvScheme" if vertical else f"{tracer}AdvScheme"
    raise InvalidParameter(
        f"MIT {slot}={code} is not a registered advection scheme number "
        "(gad_init_fixed.F:44-60); MITgcm stops on it at gad_init_fixed.F:87-115"
    )


def _effective_mit_tracer_adv_scheme(
    ctx: Ctx,
    tracer: str,
    *,
    vertical: bool = False,
) -> int:
    horizontal = ctx.data("PARM01", f"{tracer}advscheme")
    try:
        horizontal_code = (
            _MIT_DEFAULT_ADV_SCHEME if horizontal is None else int(horizontal)
        )
    except (TypeError, ValueError) as exc:
        raise InvalidParameter(
            f"invalid MIT {tracer} tracer advection scheme {horizontal!r}"
        ) from exc
    _reject_unrunnable_mit_adv_scheme(horizontal_code, tracer, vertical=False)
    if not vertical:
        return horizontal_code
    vertical_value = ctx.data("PARM01", f"{tracer}vertadvscheme")
    try:
        if vertical_value is None:
            # ini_parms.F:509-510 resolves the unset 0 (_MIT_UNSET) to the horizontal
            # code before GAD_INIT_FIXED validates the vertical slot.
            return horizontal_code
        vertical_code = int(vertical_value)
    except (TypeError, ValueError) as exc:
        raise InvalidParameter(
            f"invalid MIT {tracer} vertical tracer advection scheme "
            f"{vertical_value!r}"
        ) from exc
    _reject_unrunnable_mit_adv_scheme(vertical_code, tracer, vertical=True)
    return vertical_code


def _f_eos(value: Any, ctx: Ctx) -> dict[str, Any]:
    eos_type = str(value).strip().upper()
    if eos_type not in _EOS_MODULES:
        raise Gap(f"eosType={value!r} is not in LOCOMgcm's EOS_TYPES (equation_of_state.py)")
    return {f"{_ASSEMBLY}.eos_type": eos_type}


def _mit_selectp_in_eos_zc(ctx: Ctx) -> int | None:
    """selectP_inEOS_Zc as SET_PARMS leaves it in z coordinates (set_parms.F:275-298): unset
    (_MIT_UNSET) is 2 for the dynamic-pressure eosTypes and 0 for the others (:275-282); a
    written integer 0 to 3 is kept, MIN(value, 2) without nonHydrostatic (:289-290); any other
    written value is None, on which MITgcm stops (:283-288).  The -1 of pressure coordinates
    (:300) never reaches a translation: a buoyancyRelation other than OCEANIC is refused
    (_f_buoyancy)."""
    eos_type = str(ctx.data("PARM01", "eostype", "LINEAR")).strip().upper()
    value = ctx.data("PARM01", "selectp_ineos_zc")
    if value is None:
        return 2 if eos_type in _MIT_DYNAMIC_PRESSURE_EOS_TYPES else 0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1, 2, 3):
        return None
    if not bool(ctx.data("PARM01", "nonhydrostatic", False)):
        return min(int(value), 2)
    return int(value)


def _f_selectp_in_eos_zc(value: Any, ctx: Ctx) -> dict[str, Any]:
    """selectP_inEOS_Zc, resolved by :func:`_mit_selectp_in_eos_zc`: 0 (pressure from
    rhoConst*phiRef) and 2 (from the hydrostatic potential) are the EOS modules'
    dynamic_pressure false and true; MITgcm's stops, 1 and 3 are refused."""
    selector = _mit_selectp_in_eos_zc(ctx)
    if selector is None:
        raise Gap(f"selectP_inEOS_Zc={value!r}: MITgcm stops outside the integers 0 to 3 "
                  "(set_parms.F:283-288)")
    eos_type = str(ctx.data("PARM01", "eostype", "LINEAR")).strip().upper()
    if eos_type in {"LINEAR", "POLY3"} and selector != 0:
        raise Gap(
            f"selectP_inEOS_Zc={selector} is invalid with eosType={eos_type}: MITgcm stops "
            "(set_parms.F:292-298)"
        )
    if selector == 1:
        raise Gap(
            "selectP_inEOS_Zc=1 takes the EOS pressure from pRef4EOS, the reference pressure "
            "iterated to hydrostatic balance with rho(tRef, sRef) (set_ref_state.F:178-187, "
            "pressure_for_eos.F:90-98); LOCOMgcm's EOS modules have no such pressure"
        )
    if selector == 3:
        raise Gap(
            "selectP_inEOS_Zc=3 with nonHydrostatic requests the non-hydrostatic total "
            "pressure (set_parms.F:289-290 keep it), which LOCOMgcm's EOS modules cannot express"
        )
    return {f"{_ASSEMBLY}.selectp_in_eos_zc": selector}


def _eos_pressure_kwargs(selector: int | None) -> dict[str, bool]:
    """The EOS modules' pressure switch for MITgcm's resolved selectP_inEOS_Zc (0 or 2; the row
    refuses the others).  False is the EOS dataclass default: omitting it keeps Z/LINEAR YAML
    bytes unchanged while selector 2 must be explicit to activate the P branch."""
    return {"dynamic_pressure": True} if selector == 2 else {}


def _f_layers(value: Any, ctx: Ctx) -> dict[str, Any]:
    layers = _as_list(value)
    if any(v <= 0.0 for v in layers):
        raise Gap("non-positive layer thickness (pressure-coordinate or masked delR)")
    if len(layers) != ctx.nz:
        raise Gap(f"delR/delZ length {len(layers)} != SIZE.h Nr {ctx.nz}")
    return {
        "grid.vertical_layer_thicknesses": layers,
        "grid.depth_m": math.fsum(layers),
    }


def _f_delrc(value: Any, ctx: Ctx) -> dict[str, Any]:
    """delRc -> grid.vertical_center_spacing, MITgcm's drC(1..Nr+1) (gmredi2/4).

    MITgcm takes drC = delRc (ini_vertical_grid.F:103-107); without delR it also
    derives drF from delRc (:88-101), which layer_thickness_from_center_spacing
    reproduces and is written as the layer thicknesses.  Without delRc MITgcm takes
    drC(1) = delR(1)/2, drC(k) = (delR(k-1)+delR(k))/2, drC(Nr+1) = delR(Nr)/2
    (:120-128), the spacing LOCOMgcm derives from the layer thicknesses
    (vertical_coordinates.py mit_layer_centres); a delRc equal to those midpoints
    stays a skip, so a case keeps its bytes and its KPP or pressure-dependent EOS,
    which refuse a given centre spacing (_check_center_spacing_readers).
    """
    centres = _as_list(value)
    if len(centres) != ctx.nz + 1:
        raise Gap(f"delRc length {len(centres)} != Nr+1 = {ctx.nz + 1}")
    layers = ctx.data("PARM04", "delr")
    if layers is None:
        layers = ctx.data("PARM04", "delz")
    if layers is None:
        drf = list(layer_thickness_from_center_spacing(centres))
        return {
            "grid.vertical_center_spacing": centres,
            "grid.vertical_layer_thicknesses": drf,
            "grid.depth_m": math.fsum(drf),
        }
    drf = _as_list(layers)
    midpoints = (
        [0.5 * drf[0]]
        + [0.5 * (drf[k - 1] + drf[k]) for k in range(1, len(drf))]
        + [0.5 * drf[-1]]
    )
    if centres == midpoints:
        raise Skipped("delRc equals the delR midpoints (ini_vertical_grid.F:120-128), "
                      "the spacing LOCOMgcm derives")
    return {"grid.vertical_center_spacing": centres}


def _f_tracer_adv(value: Any, ctx: Ctx) -> dict[str, Any]:
    try:
        code = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidParameter(
            f"invalid MIT tracer advection scheme {value!r}"
        ) from exc
    if code not in _ADV_SCHEME_MAP:
        raise Gap(f"MIT tracer advection scheme {code} has no TO counterpart")
    return {"advection.tracer": _ADV_SCHEME_MAP[code]}


def _tracer_adv_rule(tracer: str) -> Rule:
    """Map an advection scheme only when MIT actually advects that tracer."""

    sibling = "salt" if tracer == "temp" else "temp"

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not bool(ctx.data("PARM01", f"{tracer}stepping", True)) or not bool(
            ctx.data("PARM01", f"{tracer}advection", True)
        ):
            # The scheme still feeds the AB gate, but selects no TO advection kernel.
            _effective_mit_tracer_adv_scheme(ctx, tracer)
            sibling_active = bool(
                ctx.data("PARM01", f"{sibling}stepping", True)
            ) and bool(ctx.data("PARM01", f"{sibling}advection", True))
            if tracer == "temp" and sibling_active:
                sibling_code = _effective_mit_tracer_adv_scheme(ctx, sibling)
                if sibling_code not in _ADV_SCHEME_MAP:
                    raise Gap(
                        f"MIT tracer advection scheme {sibling_code} has no TO counterpart"
                    )
                # Preserve the advection block's established insertion point;
                # assembly later resolves the same active-salinity value.
                return {"advection.tracer": _ADV_SCHEME_MAP[sibling_code]}
            raise Skipped(
                f"{tracer} tracer advection is inactive; scheme only feeds the AB gate"
            )
        code = _effective_mit_tracer_adv_scheme(ctx, tracer)
        if tracer == "salt":
            sibling_active = bool(
                ctx.data("PARM01", f"{sibling}stepping", True)
            ) and bool(ctx.data("PARM01", f"{sibling}advection", True))
            if sibling_active:
                sibling_value = ctx.data("PARM01", f"{sibling}advscheme")
                sibling_code = (
                    _MIT_DEFAULT_ADV_SCHEME
                    if sibling_value is None
                    else _effective_mit_tracer_adv_scheme(ctx, sibling)
                )
                sibling_description = (
                    f"tempAdvScheme MIT default {_MIT_DEFAULT_ADV_SCHEME!r}"
                    if sibling_value is None
                    else "tempAdvScheme"
                )
                if sibling_code != code:
                    raise Gap(
                        f"differs from {sibling_description}: TO carries one "
                        "horizontal tracer advection scheme"
                    )
                raise Skipped(
                    f"equals {sibling_description} (TO carries one horizontal "
                    "tracer advection scheme)"
                )
        if code not in _ADV_SCHEME_MAP:
            raise Gap(f"MIT tracer advection scheme {code} has no TO counterpart")
        return {"advection.tracer": _ADV_SCHEME_MAP[code]}

    return Rule(
        MAP,
        target="advection.tracer",
        convert=(
            "MIT scheme code -> TO enum"
            if tracer == "temp"
            else "whitelisted when equal to tempAdvScheme"
        ),
        fn=fn,
    )


def _f_vertical_adv(value: Any, ctx: Ctx) -> dict[str, Any]:
    """The vertical scheme of the active tracers: the implicit column solve's
    table where ``<tracer>ImplVertAdv`` is set (GAD_IMPLICIT_R), else the explicit
    vertical flux table."""
    code = int(value)
    for tracer in ("temp", "salt"):
        if not (ctx.data("PARM01", f"{tracer}stepping", True)
                and ctx.data("PARM01", f"{tracer}advection", True)):
            continue
        effective = _effective_mit_tracer_adv_scheme(ctx, tracer, vertical=True)
        if effective != code:
            raise Gap("TO carries one vertical tracer advection scheme; active T/S formats differ")
        if ctx.data("PARM01", f"{tracer}implvertadv", False):
            if code not in _IMPLICIT_VERTICAL_ADV_SCHEME_MAP:
                raise Gap(
                    f"MIT vertical advection scheme {code} with {tracer}ImplVertAdv has no "
                    "LOCOMgcm implicit column scheme (IMPLICIT_VERTICAL_TRACER_SCHEMES)"
                )
        elif code not in _VERTICAL_ADV_SCHEME_MAP:
            raise Gap(
                f"MIT vertical advection scheme {code} has no LOCOMgcm explicit vertical "
                "scheme (VERTICAL_TRACER_SCHEMES)"
            )
    return {
        "advection.vertical": _VERTICAL_ADV_SCHEME_MAP.get(code)
        or _IMPLICIT_VERTICAL_ADV_SCHEME_MAP[code]
    }


def _vertical_adv_rule(tracer: str) -> Rule:
    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not (ctx.data("PARM01", f"{tracer}stepping", True)
                and ctx.data("PARM01", f"{tracer}advection", True)):
            raise Skipped(f"{tracer} tracer advection is inactive")
        code = _effective_mit_tracer_adv_scheme(ctx, tracer, vertical=True)
        mapped = _f_vertical_adv(code, ctx)
        if (tracer == "salt" and ctx.data("PARM01", "tempstepping", True)
                and ctx.data("PARM01", "tempadvection", True)
                and ctx.data("PARM01", "tempvertadvscheme") is not None):
            raise Skipped("equals the effective temperature vertical advection scheme")
        return mapped

    return _map("advection.vertical", "MIT resolved vertical code -> TO enum", fn)


def _tracer_tendency_ab_gates(ctx: Ctx) -> tuple[bool, bool]:
    """Derive MIT's independent AdamsBashforthGt/Gs tendency flags."""

    temperature = bool(ctx.data("PARM01", "tempstepping", True)) and (
        _effective_mit_tracer_adv_scheme(ctx, "temp")
        in _MIT_AB_ADV_SCHEMES
    )
    salinity = bool(ctx.data("PARM01", "saltstepping", True)) and (
        _effective_mit_tracer_adv_scheme(ctx, "salt")
        in _MIT_AB_ADV_SCHEMES
    )
    return temperature, salinity


def _tracer_tendency_ab_writes(
    *,
    do_ab_on_gt_gs: bool,
    temperature: bool,
    salinity: bool,
) -> dict[str, Any]:
    """Emit a lossless tracer-AB configuration with a byte-stable legacy case."""

    if not do_ab_on_gt_gs:
        return {"model.stepper.kwargs.ab2_on_tracer_tendency": False}
    if temperature and salinity:
        # Explicit legacy input remains a lossless spelling when both gates agree.
        return {"model.stepper.kwargs.ab2_on_tracer_tendency": True}
    writes = {
        "model.stepper.kwargs.ab2_on_temperature_tendency": temperature,
        "model.stepper.kwargs.ab2_on_salinity_tendency": salinity,
    }
    if temperature == salinity:
        # Preserve audit readers of the explicit legacy spelling.  The stepper
        # gives the independent pair precedence, so this is not a physics gate.
        writes["model.stepper.kwargs.ab2_on_tracer_tendency"] = True
    return writes


def _f_do_ab_on_gt_gs(value: Any, ctx: Ctx) -> dict[str, Any]:
    if type(value) is not bool:
        raise InvalidParameter(
            f"invalid MIT doAB_onGtGs logical value {value!r}"
        )
    temperature, salinity = _tracer_tendency_ab_gates(ctx)
    if not value and (temperature or salinity):
        raise Gap(
            "doAB_onGtGs=.FALSE. selects AB on T/S fields "
            "(gad_init_fixed.F:158-162), which LOCOMgcm cannot express"
        )
    return _tracer_tendency_ab_writes(
        do_ab_on_gt_gs=value,
        temperature=temperature,
        salinity=salinity,
    )


def _matches(
    peer_group: str,
    peer_key: str,
    peer_label: str,
    reason: str,
    *,
    peer_default: Any = _MISSING,
) -> Convert:
    """Salt-side knob: whitelist it when it equals its temperature-side twin."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        peer = ctx.data(peer_group, peer_key, _MISSING)
        if peer is _MISSING or peer is None:
            if peer_default is _MISSING:
                raise Skipped(f"equals {peer_label} ({reason})")
            peer = peer_default
            peer_description = f"{peer_label} MIT default {peer!r}"
        else:
            peer_description = peer_label
        if peer == value:
            raise Skipped(f"equals {peer_description} ({reason})")
        raise Gap(f"differs from {peer_description}: {reason}")

    return fn


def _f_convert_fw2salt(value: Any, ctx: Ctx) -> dict[str, Any]:
    """Preserve MIT's uniform value or local-surface-salinity sentinel."""
    del ctx
    return {f"{_ASSEMBLY}.convert_fw2salt": float(value)}


def _f_real_freshwater_flux(value: Any, ctx: Ctx) -> dict[str, Any]:
    """Map MIT's real-freshwater volume gate to the integrated-eta stepper."""
    del ctx
    if value is False:
        raise Skipped("real freshwater flux is off: no-op")
    return {f"{_ASSEMBLY}.use_real_freshwater_flux": True}


def _f_cg2d_target_residual(value: Any, ctx: Ctx) -> dict[str, Any]:
    target_res_w_unit = ctx.data("PARM02", "cg2dtargetreswunit", -1.0)
    if float(target_res_w_unit) > 0.0:
        raise Skipped(
            "dimensionless target is inactive because cg2dTargetResWunit > 0"
        )
    return {
        "model.stepper.kwargs.cg_tolerance": float(value),
        "model.stepper.kwargs.cg_tolerance_mode": "mit_normalized",
    }


def _f_cg2d_target_res_w_unit(value: Any, ctx: Ctx) -> dict[str, Any]:
    """cg2dTargetResWunit: <= 0 is the normalised test, > 0 the W-unit test.

    MITgcm normalises the RHS exactly when cg2dTargetResWunit <= 0 and then
    stops on cg2dTargetResidual (ini_cg2d.F:148-151, cg2d.F:117-127), the test
    :func:`_f_cg2d_target_residual` maps.  A positive value stops CG2D at
    ||r|| < cg2dNorm * W * implicDiv2DFlow / deltaTMom * globalArea /
    SQRT(n2dWetPts) on the un-normalised RHS (ini_cg2d.F:152-161, cg2d.F:204,
    :337), the stepper's mit_w_unit stop (one of CG2D_STOP_CRITERIA), which
    takes W in m/s as cg_tolerance and forms the area terms from the grid
    (implicit_free_surface_cg2d.cg2d_w_unit_tolerance).
    """
    del ctx
    if float(value) <= 0.0:
        raise Skipped(
            "cg2dTargetResWunit <= 0: MITgcm normalises the RHS and stops on "
            "cg2dTargetResidual (ini_cg2d.F:148-151), which that row maps"
        )
    return {
        "model.stepper.kwargs.cg_tolerance": float(value),
        "model.stepper.kwargs.cg_tolerance_mode": "mit_w_unit",
    }


def _f_cg3d_target_residual(value: Any, ctx: Ctx) -> dict[str, Any]:
    result = {"physics.nonhydrostatic.cg3d_target_residual": float(value)}
    target_res_w_unit = ctx.data(
        "PARM02", "cg3dtargetreswunit", _MIT_DEFAULT_CG3D_TARGET_RES_W_UNIT
    )
    if (
        bool(ctx.data("PARM01", "nonhydrostatic", False))
        and float(target_res_w_unit) <= 0.0
    ):
        result["model.stepper.kwargs.cg3d_tolerance_mode"] = "mit_normalized"
    return result


def _cg2d_nsa_selected(ctx: Ctx) -> bool:
    """useNSACGSolver selects CG2D_NSA unless useSRCGSolver takes the branch before it
    (solve_for_pressure.F:288-311); the call also needs ALLOW_CG2D_NSA (#undef in
    model/inc/CPP_OPTIONS.h:123), a build macro the code inventory rules."""
    return bool(ctx.data("PARM02", "usensacgsolver", False)) and not bool(
        ctx.data("PARM02", "usesrcgsolver", False)
    )


def _f_use_nsa_cg_solver(value: Any, ctx: Ctx) -> dict[str, Any]:
    if not value:
        raise Skipped("useNSACGSolver off: SOLVE_FOR_PRESSURE calls CG2D "
                      "(solve_for_pressure.F:288-311), the stepper's solve")
    if not _cg2d_nsa_selected(ctx):
        raise Skipped("useSRCGSolver takes the branch before CG2D_NSA "
                      "(solve_for_pressure.F:288-298)")
    raise Skipped(
        "CG2D_NSA stops on the residual test of CG2D (cg2d_nsa.F:250-251; cg2d.F:204, :337) "
        "and differs by the forced cg2dMinItersNSA iterations, which that row decides"
    )


def _f_cg2d_min_iters_nsa(value: Any, ctx: Ctx) -> dict[str, Any]:
    if int(value) == 0:
        raise Skipped("cg2dMinItersNSA = 0, its default (set_defaults.F:287): no forced iteration")
    if not _cg2d_nsa_selected(ctx):
        raise Skipped("read only by CG2D_NSA, which useNSACGSolver off or useSRCGSolver on "
                      "leaves uncalled (solve_for_pressure.F:288-311)")
    raise Gap(
        "CG2D_NSA iterates while it2d <= cg2dMinItersNSA whatever the residual "
        "(cg2d_nsa.F:250-251); the stepper's CG2D has no minimum iteration count"
    )


def _f_cg2d_use_min_res_sol(value: Any, ctx: Ctx) -> dict[str, Any]:
    del ctx
    return {"model.stepper.kwargs.cg_use_min_res_sol": int(value) >= 1}


def _mit_cg2d_use_min_res_sol_default(ctx: Ctx) -> bool:
    """cg2dUseMinResSol unset: 1 on a Cartesian grid with blank topoFile and
    bathyFile, else 0 (ini_parms.F:1585-1588)."""
    blank = lambda key: not str(ctx.data("PARM05", key, "") or "").strip()  # noqa: E731
    return ctx.grid_family == "cartesian" and blank("topofile") and blank("bathyfile")


def _apply_mit_cg2d_defaults(
    yaml_tree: dict[str, Any], ctx: Ctx, record: Record
) -> None:
    """Write MITgcm's CG2D controls the namelist leaves out.

    The stepper's own defaults are cg_max_iters=1000 and cg_tolerance=1e-13
    (steppers/implicit_free_surface.py ImplicitFreeSurfaceStepper); MITgcm's are
    cg2dMaxIters=150 and cg2dTargetResidual=1e-7 (set_defaults.F:286, :288), the
    latter a test on the normalised residual while cg2dTargetResWunit <= 0
    (default -1, :289; ini_cg2d.F:148-151), the mit_normalized mode.  A positive
    cg2dTargetResWunit is decided by its own row.  An absent cg2dUseMinResSol is
    resolved by MITgcm's grid and topography condition and always written.
    """
    stepper_kwargs = (
        yaml_tree.setdefault("model", {}).setdefault("stepper", {}).setdefault("kwargs", {})
    )
    defaults: dict[str, Any] = {}
    if ctx.data("PARM02", "cg2dmaxiters") is None:
        defaults["model.stepper.kwargs.cg_max_iters"] = _MIT_DEFAULT_CG2D_MAX_ITERS
    target_res_w_unit = float(
        ctx.data("PARM02", "cg2dtargetreswunit", _MIT_DEFAULT_CG2D_TARGET_RES_W_UNIT)
    )
    if ctx.data("PARM02", "cg2dtargetresidual") is None and target_res_w_unit <= 0.0:
        defaults["model.stepper.kwargs.cg_tolerance"] = _MIT_DEFAULT_CG2D_TARGET_RESIDUAL
        defaults["model.stepper.kwargs.cg_tolerance_mode"] = "mit_normalized"
    for dotted, value in defaults.items():
        stepper_kwargs[dotted.rsplit(".", 1)[1]] = value
    if defaults:
        record.derived.append(
            {
                "target": sorted(defaults),
                "value": defaults,
                "rationale": (
                    "absent CG2D controls resolve the MITgcm defaults cg2dMaxIters=150 "
                    "and cg2dTargetResidual=1e-7 on the normalised residual "
                    "(model/src/set_defaults.F:286, :288-289; ini_cg2d.F:148-151)"
                ),
            }
        )
    if ctx.data("PARM02", "cg2duseminressol") is None:
        use_min_res_sol = _mit_cg2d_use_min_res_sol_default(ctx)
        stepper_kwargs["cg_use_min_res_sol"] = use_min_res_sol
        record.derived.append(
            {
                "target": ["model.stepper.kwargs.cg_use_min_res_sol"],
                "value": {"model.stepper.kwargs.cg_use_min_res_sol": use_min_res_sol},
                "rationale": (
                    "absent cg2dUseMinResSol is 1 on a Cartesian grid with blank topoFile "
                    "and bathyFile, else 0 (model/src/ini_parms.F:1585-1588)"
                ),
            }
        )


def _apply_mit_hfac_min_default(
    yaml_tree: dict[str, Any], ctx: Ctx, record: Record
) -> None:
    """Write MITgcm's hFacMin when the namelist leaves it out.

    The grid builders' own default is 0.2 (io/runner/model_input.py:955);
    MITgcm's is 1.0 (set_defaults.F:180), read by ini_masks_etc.F:108.
    hFacMinDr and hFacMinDz keep their own rows (an unset one is 0 on both ends,
    ini_parms.F:644-646).
    """
    if ctx.data("PARM01", "hfacmin") is not None:
        return
    _set_dotted(yaml_tree, "case.extras.hfac_min", _MIT_DEFAULT_HFAC_MIN)
    record.derived.append(
        {
            "target": ["case.extras.hfac_min"],
            "value": {"case.extras.hfac_min": _MIT_DEFAULT_HFAC_MIN},
            "rationale": "absent hFacMin is the MITgcm default 1.0 (model/src/set_defaults.F:180)",
        }
    )


def _apply_mit_cg3d_defaults(
    yaml_tree: dict[str, Any], ctx: Ctx, record: Record
) -> None:
    """Resolve source defaults only for translated non-hydrostatic cases."""
    if not bool(ctx.data("PARM01", "nonhydrostatic", False)):
        return

    nonhydrostatic = yaml_tree.setdefault("physics", {}).setdefault(
        "nonhydrostatic", {}
    )
    defaults: dict[str, Any] = {}
    if ctx.data("PARM02", "cg3dmaxiters") is None:
        nonhydrostatic["cg3d_max_iters"] = _MIT_DEFAULT_CG3D_MAX_ITERS
        defaults["physics.nonhydrostatic.cg3d_max_iters"] = (
            _MIT_DEFAULT_CG3D_MAX_ITERS
        )
    if ctx.data("PARM02", "cg3dtargetresidual") is None:
        nonhydrostatic["cg3d_target_residual"] = (
            _MIT_DEFAULT_CG3D_TARGET_RESIDUAL
        )
        defaults["physics.nonhydrostatic.cg3d_target_residual"] = (
            _MIT_DEFAULT_CG3D_TARGET_RESIDUAL
        )

    target_res_w_unit = float(
        ctx.data(
            "PARM02",
            "cg3dtargetreswunit",
            _MIT_DEFAULT_CG3D_TARGET_RES_W_UNIT,
        )
    )
    if target_res_w_unit <= 0.0:
        stepper_kwargs = (
            yaml_tree.setdefault("model", {})
            .setdefault("stepper", {})
            .setdefault("kwargs", {})
        )
        stepper_kwargs["cg3d_tolerance_mode"] = "mit_normalized"
        if ctx.data("PARM02", "cg3dtargetresidual") is None:
            defaults["model.stepper.kwargs.cg3d_tolerance_mode"] = (
                "mit_normalized"
            )

    if defaults:
        record.derived.append(
            {
                "target": sorted(defaults),
                "value": defaults,
                "rationale": (
                    "nonHydrostatic=.TRUE. with absent CG3D controls resolves "
                    "the MITgcm defaults cg3dMaxIters=150, "
                    "cg3dTargetResidual=1e-7, and cg3dTargetResWunit=-1 "
                    "(model/src/set_defaults.F:293-295)"
                ),
            }
        )


def _apply_mit_3d_coriolis_controls(
    yaml_tree: dict[str, Any], ctx: Ctx, record: Record
) -> None:
    """Emit MIT-effective 3-D Coriolis controls for NH translations only."""
    if not bool(ctx.data("PARM01", "nonhydrostatic", False)):
        return

    raw_scheme = ctx.data("PARM01", "select3dcorischeme")
    raw_map = int(ctx.data("PARM01", "selectcorimap", -1))
    f_prime = float(ctx.data("PARM01", "fprime", 0.0))
    try:
        resolved = resolve_3d_coriolis_controls(
            nonhydrostatic=True,
            select_3d_cori_scheme=(
                None if raw_scheme is None else int(raw_scheme)
            ),
            use_3d_coriolis=bool(ctx.data("PARM01", "use3dcoriolis", True)),
            select_cori_map=raw_map,
            f_prime=f_prime,
            grid_is_spherical=(ctx.grid_family not in ("cartesian", "cylindrical")),
            use_coriolis=bool(ctx.data("PARM01", "usecoriolis", True)),
        )
    except (TypeError, ValueError) as exc:
        raise TranslationError(f"invalid 3-D Coriolis controls: {exc}") from exc
    if (
        resolved.select_cori_map == 2
        and ctx.grid_family in ("cartesian", "cylindrical")
    ):
        raise TranslationError(
            "selectCoriMap=2 on a Cartesian/cylindrical grid is inexpressible: "
            "MIT interprets yC as degrees but LOCOMgcm's grid coordinates are metres"
        )

    nonhydrostatic = yaml_tree.setdefault("physics", {}).setdefault(
        "nonhydrostatic", {}
    )
    nonhydrostatic.update(
        select_3d_cori_scheme=resolved.select_3d_cori_scheme,
        select_cori_map=resolved.select_cori_map,
        f_prime=f_prime,
    )
    record.derived.append(
        {
            "target": [
                "physics.nonhydrostatic.f_prime",
                "physics.nonhydrostatic.select_3d_cori_scheme",
                "physics.nonhydrostatic.select_cori_map",
            ],
            "value": {
                "f_prime": f_prime,
                "select_3d_cori_scheme": resolved.select_3d_cori_scheme,
                "select_cori_map": resolved.select_cori_map,
            },
            "source": (
                "ini_parms.F:342-345,705-714; "
                "set_parms.F:73-85,95; ini_cori.F:55-100"
            ),
            "rationale": (
                "resolve UNSET selector through use3dCoriolis, derive the map "
                "from grid family when unset, then apply the map/fPrime and "
                "useCoriolis effective gates before runtime binding"
            ),
        }
    )


def _f_free_surface(value: Any, ctx: Ctx) -> dict[str, Any]:
    if value is True:
        raise Skipped("TO stepper is an implicit free surface")
    raise Gap("implicitFreeSurface=.FALSE. (rigid lid / explicit free surface)")


def _f_rigid_lid(value: Any, ctx: Ctx) -> dict[str, Any]:
    if value is False:
        raise Skipped("rigid lid off")
    raise Gap("rigid lid has no TO counterpart")


def _f_gbaro(value: Any, ctx: Ctx) -> dict[str, Any]:
    # MITgcm default gravity = 9.81 (references/MITgcm/model/src/set_defaults.F:101).
    gravity = float(ctx.data("PARM01", "gravity", _MIT_DEFAULT_GRAVITY))
    if gravity == float(value):
        raise Skipped(f"gBaro equals gravity ({gravity})")
    raise Gap(f"split barotropic gravity (gBaro={value} != gravity={gravity})")


def _f_implicit_visc(value: Any, ctx: Ctx) -> dict[str, Any]:
    del ctx
    return {f"{_ASSEMBLY}.implicit_vertical_viscosity": bool(value)}


def _f_nonzero_gap(reason: str) -> Convert:
    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if _is_zero(value):
            raise Skipped(f"{reason} (value is zero: no-op)")
        raise Gap(reason)

    return fn


def _f_cadj_freq(value: Any, ctx: Ctx) -> dict[str, Any]:
    """cAdjFreq -> the convective_adjustment module (start/3).

    MITgcm calls CONVECTIVE_ADJUSTMENT on the updated tracers in
    TRACERS_CORRECTION_STEP (tracers_correction_step.F:114-118) when cAdjFreq is
    nonzero; the module is assembled by _convective_adjustment_entry.  The model
    refuses the module with ivdc_kappa, which stops MITgcm (ini_parms.F:1107-1112),
    or with KPP, PP81, MY82 or KL10 at construction (columns.validate_convection).
    Outside the verbatim mode the casegen bases translate under the case-scope rule
    (CAMPAIGN_DISABLED["cadjfreq"]), so the row raises the rule's gap text and
    _record_gap records it disabled.
    """
    reason = CAMPAIGN_DISABLED["cadjfreq"].gap
    if _is_zero(value):
        raise Skipped(f"{reason} (value is zero: no-op)")
    if not ctx.verbatim:
        raise Gap(reason)
    return {f"{_ASSEMBLY}.c_adj_freq": float(value)}


def _mit_start_clock(ctx: Ctx) -> tuple[float, int, float]:
    """MITgcm's (startTime, nIter0, baseTime) as ini_parms.F:1121-1133 resolves them.

    Neither startTime nor nIter0 set: startTime = baseTime, nIter0 = 0; startTime unset:
    baseTime + deltaTClock*nIter0; nIter0 unset: NINT((startTime - baseTime)/deltaTClock);
    both set with baseTime 0: baseTime = startTime - deltaTClock*nIter0.  deltaTClock is its
    own value, else deltaT and its fallbacks (:1043-1059).  "Unset" is MITgcm's unset value,
    written or not (_MIT_UNSET: startTime UNSET_RL, nIter0 -1, baseTime 0, deltaTClock 0), which
    the namelists the rules read leave out.  LOCOMgcm's time zero is this startTime: every
    time-dependent source the translation writes carries it as start_time.
    """
    start, n_iter0 = ctx.data("PARM03", "starttime"), ctx.data("PARM03", "niter0")
    base = ctx.data("PARM03", "basetime")
    clock = ctx.data("PARM03", "deltatclock")
    dt_clock = float(clock) if clock is not None else float(_master_dt(ctx))
    base_value = 0.0 if base is None else float(base)
    if start is None and n_iter0 is None:
        return base_value, 0, base_value
    if start is None:
        return base_value + dt_clock * int(n_iter0), int(n_iter0), base_value
    if n_iter0 is None:
        steps = (float(start) - base_value) / dt_clock
        return float(start), int(math.copysign(math.floor(abs(steps) + 0.5), steps)), base_value
    if base is None:
        return float(start), int(n_iter0), float(start) - dt_clock * int(n_iter0)
    return float(start), int(n_iter0), base_value


def _staged_namelist_value(ctx: Ctx, file: str, group: str, key: str, default: Any) -> Any:
    """``key`` of ``group`` in the staged namelist ``file``; ``default`` when unset or absent."""
    if file in ctx.namelists:
        return ctx.namelists[file].get(group, {}).get(key, default)
    path = ctx.case_dir / "input" / file
    if not path.is_file():
        return default
    return _mit_namelist(path).get(group, {}).get(key, default)


#: Records READ_PICKUP reads from a field-list pickup (AB2 branch, read_pickup.F:251-463) that a
#: LOCOMgcm pickup start does not read (defgcm/io/mit_pickup.py PROGNOSTIC_RECORDS and
#: AB_HISTORY_RECORDS), each with the condition under which MITgcm reads it and the first step
#: then uses it.  The other records the pickup holds are not read (dEtaHdt without
#: useRealFreshWaterFlux is recomputed at myIter = nIter0, integr_continuity.F:163-170).
_PICKUP_RECORDS_UNREAD = (
    ("UEulerM, VEulerM", lambda c: _staged_namelist_value(
        c, "data.gmredi", "GM_PARM01", "gm_inmomasstress", False) is True,
     "GM_InMomAsStress (read_pickup.F:260-263)"),
    ("TempNm1, SaltNm1", lambda c: c.data("PARM03", "doab_ontands", False) is True,
     "AdamsBashforth_T/S under doAB_onTandS (gad_init_fixed.F:143-163, read_pickup.F:378-387)"),
    ("GwNm1, Phi_NHyd", lambda c: c.data("PARM01", "nonhydrostatic", False) is True
     or c.data("PARM01", "implicitintgravwave", False) is True,
     "nonHydrostatic, use3Dsolver (set_parms.F:104; read_pickup.F:391-392, 412-413)"),
    ("QH_GwNm1", lambda c: c.data("PARM01", "quasihydrostatic", False) is True
     and c.data("PARM01", "staggertimestep", False) is True,
     "quasiHydrostatic with staggerTimeStep (read_pickup.F:398-399)"),
    ("PhiHyd", lambda c: (_mit_selectp_in_eos_zc(c) or 0) >= 2,
     "storePhiHyd4Phys = selectP_inEOS_Zc >= 2 (set_parms.F:304; read_pickup.F:407-408), the "
     "first step's EOS pressure, selectP_inEOS_Zc as _mit_selectp_in_eos_zc resolves it "
     "(set_parms.F:275-298)"),
    ("SmagDiff", lambda c: float(c.data("PARM01", "smag3d_diffcoeff", 0.0) or 0.0) > 0.0
     and c.data("PARM01", "staggertimestep", False) is not True,
     "smag3D_diffCoeff > 0 without staggerTimeStep (read_pickup.F:421-423)"),
    ("AddMass", lambda c: int(c.data("PARM01", "selectaddfluid", 0) or 0) >= 1,
     "selectAddFluid >= 1 (read_pickup.F:429-430)"),
    ("FricHeat", lambda c: c.data("PARM01", "addfrictionheating", False) is True
     and c.data("PARM01", "staggertimestep", False) is not True,
     "addFrictionHeating without staggerTimeStep (read_pickup.F:436-437)"),
    ("Phi_rLow", lambda c: str(c.data("PARM01", "buoyancyrelation", "OCEANIC")).strip().upper()
     in ("OCEANICP", "ATMOSPHERIC") and _staged_namelist_value(
         c, "data.pkg", "PACKAGES", "useseaice", False) is True,
     "usingPCoords (ini_parms.F:450-455) with useSEAICE (read_pickup.F:446-447)"),
    ("dPhiNH", lambda c: int(c.data("PARM01", "selectnhfreesurf", 0) or 0) >= 1,
     "selectNHfreeSurf >= 1 (read_pickup.F:451-452)"),
    ("dEtaHdt", lambda c: c.data("PARM01", "userealfreshwaterflux", False) is True,
     "useRealFreshWaterFlux: PmEpR from it at myIter = nIter0 (integr_continuity.F:141-162)"),
    ("EtaH", lambda c: float(c.data("PARM01", "nonlinfreesurf", 0) or 0) > 0,
     "nonlinFreeSurf > 0 (read_pickup.F:460-461)"),
)

#: Package pickups MITgcm reads at a pickup start that LOCOMgcm does not read (it reads
#: pickup_cd, initial_condition.mit_pickup.cd_data), with the condition and its site.
_PACKAGE_PICKUPS_UNREAD = (
    ("pickup_seaice", lambda c, n, s: _staged_namelist_value(
        c, "data.pkg", "PACKAGES", "useseaice", False) is True,
     "useSEAICE (seaice_init_varia.F:251-254)"),
    ("pickup_ic", lambda c, n, s: _staged_namelist_value(
        c, "data.pkg", "PACKAGES", "usethsice", False) is True and int(_staged_namelist_value(
            c, "data.ice", "THSICE_PARM01", "starticemodel", 0) or 0) <= 0,
     "useThSIce with startIceModel <= 0 (thsice_ini_vars.F:112-116)"),
    ("pickup_ptracers", lambda c, n, s: _staged_namelist_value(
        c, "data.pkg", "PACKAGES", "useptracers", False) is True and (n, bool(s)) > (int(
            _staged_namelist_value(c, "data.ptracers", "PTRACERS_PARM01", "ptracers_iter0", 0)
            or 0), False),
     "usePTRACERS with nIter0 > PTRACERS_Iter0, or equal with pickupSuff "
     "(ptracers_init_varia.F:124-128)"),
    ("pickup_ggl90", lambda c, n, s: _staged_namelist_value(
        c, "data.pkg", "PACKAGES", "useggl90", False) is True,
     "useGGL90 (ggl90_init_varia.F:131-132)"),
    ("pickup_gmredi", lambda c, n, s: _staged_namelist_value(
        c, "data.pkg", "PACKAGES", "usegmredi", False) is True and any(
            _staged_namelist_value(c, "data.gmredi", "GM_PARM01", key, False) is True
            for key in ("gm_usebatesk3d", "gm_usegeom")),
     "useGMRedi with GM_useBatesK3d or GM_useGEOM (gmredi_init_varia.F:167-171)"),
    ("pickup_orlanski*, pickup_stevens*", lambda c, n, s: _staged_namelist_value(
        c, "data.pkg", "PACKAGES", "useobcs", False) is True and any(
            _staged_namelist_value(c, "data.obcs", "OBCS_PARM01", f"use{kind}{side}", False) is True
            for kind in ("orlanski", "stevens") for side in ("north", "south", "east", "west")),
     "useOBCS with an Orlanski or Stevens side (obcs_init_variables.F:358-360, "
     "obcs_read_pickup.F:58-260)"),
    ("pickup_somT, pickup_somS", lambda c, n, s: any(
        int(c.data("PARM01", key, 0) or 0) in (80, 81) for key in ("tempadvscheme", "saltadvscheme")),
     "second-order-moment advection (gad_init_varia.F:58-69)"),
)


def _mit_pickup_file(input_dir: Path, name: str) -> Path | None:
    """The global file MDS_READ_FIELD opens for ``name``: the name itself, else name.data
    (mdsio_read_field.F:229-255); None when neither exists (one file per tile)."""
    for candidate in (input_dir / name, input_dir / f"{name}.data"):
        if candidate.is_file():
            return candidate
    return None


def _f_mit_start(value: Any, ctx: Ctx) -> dict[str, Any]:
    """Start the run where MITgcm starts it: from the IC files or from a pickup.

    INI_FIELDS calls READ_PICKUP(nIter0) unless startTime = baseTime, nIter0 = 0 and pickupSuff
    is blank (ini_fields.F:27-38), the clock resolved by :func:`_mit_start_clock`.  An IC start
    whose startTime is not zero only moves the time origin, which the time-dependent sources
    carry (start_time).  A pickup start writes initial_condition.mit_pickup: the field-list
    pickup ``pickup.<suffix>`` (pickupSuff, else nIter0 as I10.10, read_pickup.F:82-91) and its
    meta file, nIter0, and with useCDscheme the pickup_cd file CD_CODE_READ_PICKUP reads at
    the same start (cd_code_ini_vars.F:52-54, cd_code_read_pickup.F:68-75).  Pickup content
    LOCOMgcm does not read (:data:`_PICKUP_RECORDS_UNREAD`, :data:`_PACKAGE_PICKUPS_UNREAD`) is
    unsupported; the combinations its pickup start refuses (_runtime._start_from_mit_pickup) are
    refused by the model build that bin/casegen fixed runs before admitting a translation.
    """
    del value
    start, n_iter0, base = _mit_start_clock(ctx)
    suffix = str(ctx.data("PARM03", "pickupsuff", "") or "")[:10].rstrip()
    if start == base and n_iter0 == 0 and not suffix:
        raise Skipped(
            f"IC start (ini_fields.F:27-38): startTime = baseTime = {start!r}, nIter0 = 0, "
            "pickupSuff blank; a non-zero startTime is the start_time of the time-dependent "
            "sources (_mit_start_clock)")
    refused = []
    if ctx.data("PARM01", "usepickupbeforec54", False) is True:
        refused.append("usePickupBeforeC54: the positional pickup of read_pickup.F:160-250 and "
                       "cd_code_read_pickup.F:71-72; LOCOMgcm reads a field-list pickup")
    if not suffix and int(ctx.data("PARM03", "rwsuffixtype", 0) or 0) != 0:
        refused.append("rwSuffixType != 0 names the pickup by RW_GET_SUFFIX (read_pickup.F:83-86)")
    header = ctx.case_dir / "code" / "CPP_OPTIONS.h"
    if not header.is_file():
        header = _reference_mit_root() / "model" / "inc" / "CPP_OPTIONS.h"
    if parse_cpp_macro_states(header).get("ALLOW_ADAMSBASHFORTH_3", False):
        refused.append("ALLOW_ADAMSBASHFORTH_3: READ_PICKUP reads the AB3 histories "
                       "(read_pickup.F:272-364); LOCOMgcm's pickup history is AB2")
    refused += [f"{names} under {basis}" for names, read, basis in _PICKUP_RECORDS_UNREAD
                if read(ctx)]
    refused += [f"{stem}.<suffix> under {basis}" for stem, read, basis in _PACKAGE_PICKUPS_UNREAD
                if read(ctx, n_iter0, suffix)]
    cd = ctx.data("PARM01", "usecdscheme", False) is True
    name = f"pickup.{suffix or f'{n_iter0:010d}'}"
    input_dir = ctx.case_dir / "input"
    meta = input_dir / f"{name}.meta"
    data = _mit_pickup_file(input_dir, name)
    if not meta.is_file() or "fldList" not in meta.read_text():
        refused.append(f"{name}.meta without a field list: read_pickup.F:122-160 reads it "
                       "positionally or stops")
    if data is None:
        refused.append(f"no global {name} or {name}.data (LOCOMgcm reads one global file)")
    cd_data = _mit_pickup_file(input_dir, f"pickup_cd.{suffix or f'{n_iter0:010d}'}") if cd else None
    if cd and cd_data is None:
        refused.append("useCDscheme without a global pickup_cd file")
    if refused:
        raise Gap(f"start from MITgcm pickup {name} (nIter0={n_iter0}): " + "; ".join(refused))
    writes = {"initial_condition.mit_pickup.meta": str(meta.resolve()),
              "initial_condition.mit_pickup.data": str(data.resolve()),
              "initial_condition.mit_pickup.n_iter0": n_iter0}
    if cd:
        writes["initial_condition.mit_pickup.cd_data"] = str(cd_data.resolve())
    return writes


def _f_use_pickup_before_c54(value: Any, ctx: Ctx) -> dict[str, Any]:
    """usePickupBeforeC54 selects the positional pickup layout (read_pickup.F:160-250); MITgcm
    reads it only at a pickup start, which :func:`_f_mit_start` refuses with it."""
    if value is not True:
        raise Skipped("field-list pickup layout (the default, set_defaults.F:225)")
    return _f_mit_start(value, ctx)


def _f_end_time(value: Any, ctx: Ctx) -> dict[str, Any]:
    if ctx.data("PARM03", "ntimesteps") is not None:
        raise Skipped("nTimeSteps already carries time.steps")
    # nTimeSteps = NINT((endTime - startTime)/deltaTClock), ini_parms.F:1141-1142
    return {"time.seconds": float(value) - _mit_start_clock(ctx)[0]}


def _refuse_unclocked_inputs(yaml_tree: dict[str, Any], ctx: Ctx, record: Record) -> None:
    """Refuse a time-dependent input read at LOCOMgcm time t when MITgcm's startTime is not zero.

    MITgcm samples records at myTime = startTime + deltaTClock*(iLoop-1) (forward_step.F:429-430):
    the core forcing (external_fields_load.F:69-72) and the prescribed OBCS files
    (obcs_fields_load.F:105-108) through GET_PERIODIC_INTERVAL, an open-boundary routine at
    myTime + deltaTClock (do_oceanic_phys.F:322).  The climatology and EXF sources carry the
    startTime of :func:`_mit_start_clock` as start_time; the periodic core-forcing source, the
    periodic prescribed OBCS source and an analytic open boundary read t itself.
    """
    start = _mit_start_clock(ctx)[0]
    if start == 0.0:
        return
    sources = yaml_tree.get("sources", {})
    forcing = sources.get(_FORCING_SOURCE_NAME, {})
    obcs = sources.get(_OBCS_SOURCE_NAME, {})
    boundary = yaml_tree.get("model", {}).get("boundary_condition", {}).get("name")
    for name, read_at_t in (
        (f"sources.{_FORCING_SOURCE_NAME} (periodic records)",
         forcing.get("type") == _FORCING_SOURCE_TYPE
         and forcing.get("forcing_period_seconds") is not None),
        (f"sources.{_OBCS_SOURCE_NAME} (periodic records)",
         obcs.get("type") == _OBCS_SOURCE_TYPE and obcs.get("periodic_external_forcing") is True),
        (f"model.boundary_condition {boundary}",
         boundary in {spec["plugin"] for spec in ANALYTIC_OBCS}),
    ):
        if read_at_t:
            _record_gap(
                record,
                {"source": "input/data:PARM03 (startTime, nIter0, baseTime)",
                 "parameter": "<derived:start time of a time-dependent input>", "value": start},
                f"{name} is read at LOCOMgcm time t and has no start_time; MITgcm reads it at "
                f"myTime = startTime + t with startTime={start!r}",
            )


# MITgcm's own resolution order for the master step (ini_parms.F:1043-1046):
# deltaT, then deltaTClock, then deltaTtracer, then deltaTMom, then
# deltaTFreeSurf.  Fixed here so the result never depends on namelist line
# order, and kept in MITgcm's order so an asynchronous case that names only
# some of them resolves the way MITgcm resolves it.
_DT_FALLBACK = (
    "deltat",
    "deltatclock",
    "deltattracer",
    "deltatmom",
    "deltatfreesurf",
)


def _master_dt(ctx: Ctx) -> float | None:
    for key in _DT_FALLBACK:
        value = ctx.data("PARM03", key)
        if value is not None:
            return float(value)
    return None


def _mit_delta_t_mom(ctx: Ctx) -> float | None:
    """MITgcm's deltaTMom: its own value, else deltaT (model/src/ini_parms.F:1057)."""
    mom = ctx.data("PARM03", "deltatmom")
    return float(mom) if mom is not None else _master_dt(ctx)


def _apply_mit_time_defaults(
    yaml_tree: dict[str, Any], ctx: Ctx, record: Record
) -> None:
    """State ``dt_freesurf`` explicitly, because the two defaults disagree.

    When ``deltaTFreeSurf`` is absent MITgcm resolves it to ``deltaTMom``
    (``model/src/ini_parms.F:1068`` -- the in-source comment there says the
    *appropriate* default would have been ``deltaTtracer`` and that it is kept
    only for backwards compatibility).  LOCOMgcm's schema instead resolves a
    missing ``time.dt_freesurf`` to ``time.dt``
    (``defgcm/io/runner/config/_schema.py:255-256``).

    The two agree exactly for a synchronous case and disagree by the whole
    asynchronous ratio otherwise: ``exp2`` (deltaTmom=2400, deltaTClock=108000)
    would take its free surface at 45x MITgcm's step.  So the value is written
    out rather than left to whichever default happens to fire -- the same
    policy :data:`_MIT_ASSEMBLY_DEFAULTS` applies to the model assembly.
    """
    if ctx.data("PARM03", "deltatfreesurf") is not None:
        return
    resolved = _mit_delta_t_mom(ctx)
    if resolved is None:
        return
    _set_dotted(yaml_tree, "time.dt_freesurf", resolved)
    record.derived.append(
        {
            "target": "time.dt_freesurf",
            "value": resolved,
            "rationale": (
                "deltaTFreeSurf is absent; MITgcm resolves it to deltaTMom "
                "(model/src/ini_parms.F:1068) while LOCOMgcm's schema would "
                "resolve it to time.dt (config/_schema.py:255-256).  Identical "
                "for a synchronous case, off by the asynchronous ratio "
                "otherwise, so it is stated"
            ),
        }
    )


def _f_dt_alias(target: str, key: str) -> Convert:
    """A specific deltaT* also seeds time.dt when no earlier fallback is present."""
    rank = _DT_FALLBACK.index(key)

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        writes = {target: float(value)}
        if all(ctx.data("PARM03", k) is None for k in _DT_FALLBACK[:rank]):
            writes["time.dt"] = float(value)
        return writes

    return fn


def _f_monitor_freq(value: Any, ctx: Ctx) -> dict[str, Any]:
    dt = _master_dt(ctx)
    if dt is None:
        raise Deferred("monitorFreq needs a deltaT* to become a step interval")
    if dt <= 0.0 or float(value) <= 0.0:
        raise Skipped("monitor cadence disabled")
    steps = int(round(float(value) / dt))
    if steps < 1:
        raise Skipped("monitorFreq shorter than one step")
    return {"output.monitor_interval_steps": steps}


def _f_dump_freq(value: Any, ctx: Ctx) -> dict[str, Any]:
    seconds = float(value)
    if seconds <= 0.0:
        raise Skipped("snapshot dump disabled")
    return {
        "output.snapshot_interval_seconds": seconds,
        "output.nc_output_interval_seconds": seconds,
    }


def _f_bool_from_int(target: str) -> Convert:
    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        return {target: bool(value)}

    return fn


def _f_leith(target: str) -> Convert:
    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if float(value) == 0.0:
            raise Skipped("coefficient is zero: leaves physics.viscosity_scheme alone")
        # The variable viscosity enters the stencil of the momentum form:
        # mom_vi_hdissip.F's div-vort form under vectorInvariantMomentum, the
        # flux-divergence Laplacian otherwise (mom_fluxform.F:612-618); MITgcm's
        # default is the flux form (set_defaults.F:193).
        vector_invariant = bool(ctx.data("PARM01", "vectorinvariantmomentum", False))
        scheme = "leith_smag" if vector_invariant else "leith_smag_flux"
        return {target: float(value), "physics.viscosity_scheme": scheme}

    return fn


def _f_visca4(value: Any, ctx: Ctx) -> dict[str, Any]:
    """viscA4 -> HorizontalViscosity.viscosity_a4, the constant biharmonic viscosity.

    MITgcm applies it with the flux-form viscous fluxes (mom_fluxform.F:601-669),
    MOM_VI_DEL2UV and MOM_VI_HDISSIP's del^4 block (mom_vecinv.F:400-434), and in
    the base of MOM_CALC_VISC's viscA4_D/Z (mom_calc_visc.F); the module applies it
    in the matching scheme: ``divvort`` through ``_divvort_constant_biharmonic``,
    ``constant`` through ``_flux_form_constant_biharmonic``, the
    Leith/Smagorinsky schemes in their biharmonic chains.  The module refuses it on
    the cubed sphere (``_refuse_cube_biharmonic``).
    """
    del ctx
    if float(value) == 0.0:
        raise Skipped("viscA4 is zero: no biharmonic viscosity (the module default)")
    return {f"{_ASSEMBLY}.viscosity_a4": float(value)}


def _f_tracer_coefficient(staged: str, label: str) -> Convert:
    """One tracer's horizontal diffusion coefficient, staged for the assembly."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del ctx
        if float(value) == 0.0:
            raise Skipped(f"{label} is zero, its MIT default: no such diffusion of the tracer")
        return {f"{_ASSEMBLY}.{staged}": float(value)}

    return fn


def _f_ivdc_kappa(value: Any, ctx: Ctx) -> dict[str, Any]:
    """ivdc_kappa -> ImplicitFreeSurfaceStepper.ivdc_kappa (m2 s-1, no conversion).

    MITgcm sets IVDConvCount = 1 on an interface k > 1 where
    -sigmaR*gravitySign > 0 (calc_ivdc.F:44-53, called at do_oceanic_phys.F:868-875
    when calcConvect = ivdc_kappa /= 0, :270) from the step-start FIND_RHO_2D
    densities at kRef = k (:811-823), and adds IVDConvCount*ivdc_kappa to
    KappaRT/RS of a tracer without KPP (calc_3d_diffusivity.F:83-116).  The
    stepper forms the same count from the same densities and adds the same term
    (implicit_free_surface.py _prepare_step_forcing, steppers/ifs/columns.py
    ivdc_tracer_diffusivity_at_step).  MITgcm requires implicitDiffusion
    (ini_parms.F:738-744) and so does the stepper at construction
    (columns.validate_vertical_mixing), so implicitDiffusion is written beside
    it; the vertical-diffusion resolver reads that and
    physics.tracer.implicit_vertical_diffusion as one declaration and requires
    them equal.

    Outside the verbatim mode the casegen bases translate under the case-scope rule
    (CAMPAIGN_DISABLED["ivdc_kappa"]): both models run with ivdc off, as the
    frozen casegen MITgcm truth did, so the row raises the rule's gap text and
    _record_gap records it disabled.
    """
    if _is_zero(value):
        raise Skipped("ivdc_kappa is zero: calcConvect is false (do_oceanic_phys.F:270)")
    if not ctx.verbatim:
        raise Gap(CAMPAIGN_DISABLED["ivdc_kappa"].gap)
    return {
        "model.stepper.kwargs.ivdc_kappa": float(value),
        "model.stepper.kwargs.implicit_tracer_diffusion": bool(
            ctx.data("PARM01", "implicitdiffusion", False)
        ),
    }


def _clim_relax_tau(tracer: str) -> Convert:
    """tau{Theta,Salt}ClimRelax -> surface_climate_relaxation.tau_{theta,salt} (s, no conversion).

    MITgcm restores the surface tracer when tempForcing (saltForcing) and the
    time scale is positive (set_parms.F:250-253), with lambda = 1/tau inside
    |yC| <= latBandClimRelax (ini_forcing.F:46-61), as surfaceForcingT/S
    (forcing_surf_relax.F:91-104, external_forcing_surf.F:173-177).  The
    surface_climate_relaxation module computes the same flux
    (defgcm/dynamics/surface_relaxation.py); :func:`_wire_surface_climatology`
    emits it with its mit_surface_climatology source.  Outside the verbatim mode
    the casegen bases translate under the case-scope rule
    (CAMPAIGN_DISABLED["tau<tracer>climrelax"]), as for ivdc_kappa.
    """
    key = f"tau{tracer}climrelax"

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if _is_zero(value):
            raise Skipped(f"{CAMPAIGN_DISABLED[key].gap} (value is zero: no-op)")
        if not ctx.verbatim:
            raise Gap(CAMPAIGN_DISABLED[key].gap)
        if float(value) < 0.0:
            raise Skipped(
                f"negative time scale: do{tracer.capitalize()}ClimRelax is false "
                "(set_parms.F:250-253)"
            )
        return {f"{_ASSEMBLY}.tau_{tracer}_clim_relax": float(value)}

    return fn


def _f_lat_band_clim_relax(value: Any, ctx: Ctx) -> dict[str, Any]:
    """latBandClimRelax -> surface_climate_relaxation.lat_band (yC units, no conversion).

    ini_forcing.F:50-61 compares ABS(yC) with it, degrees on a spherical grid;
    the module compares the grid's centre latitudes.
    """
    if not ctx.verbatim:
        raise Gap("latitude-banded climatology relaxation")
    return {f"{_ASSEMBLY}.lat_band_clim_relax": float(value)}


def _clim_file(key: str, role: str) -> Rule:
    """thetaClimFile/saltClimFile -> the mit_surface_climatology source.

    Outside the verbatim mode the row stays the recorded-only binary input it
    was (:func:`_binary_input`).  A blank name reads no file
    (ini_forcing.F:100-105).
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not ctx.verbatim:
            raise Deferred(f"binary ingest ({role}); path and dtype recorded only")
        if value is None or not str(value).strip():
            raise Skipped(f"{role} file name is blank; MITgcm reads no file (ini_forcing.F:100-105)")
        return {f"{_ASSEMBLY}.{key}": str(value)}

    return Rule(
        MAP,
        target=f"sources.{_CLIMATOLOGY_SOURCE_NAME}.{key} <- {key}",
        convert=f"MIT {role} file name, read at readBinaryPrec (big-endian)",
        fn=fn,
    )


def _f_use_cd_scheme(value: Any, ctx: Ctx) -> dict[str, Any]:
    """useCDscheme -> the cd_scheme module (pkg/cd_code; defgcm/dynamics/cd_scheme).

    MITgcm then steps a D-grid velocity in CD_CODE_SCHEME, called from TIMESTEP
    after Adams-Bashforth (timestep.F:228-270), adds its Coriolis terms, and
    drops the C-grid Coriolis term (mom_fluxform.F:1082, mom_vecinv.F:680-682).
    The momentum assembly emits cd_scheme (_cd_scheme_entry) and no C-grid
    Coriolis profile.  Outside the verbatim mode the casegen bases translate
    under the case-scope rule (CAMPAIGN_DISABLED["usecdscheme"]), as for ivdc_kappa.
    """
    if value is False:
        raise Skipped("C-D scheme momentum time stepping (switch is .FALSE.: no-op)")
    if not ctx.verbatim:
        raise Gap(CAMPAIGN_DISABLED["usecdscheme"].gap)
    return {f"{_ASSEMBLY}.use_cd_scheme": True}


def _f_use_gmredi(value: Any, ctx: Ctx) -> dict[str, Any]:
    """useGMRedi -> the gmredi tracer module (pkg/gmredi; defgcm/dynamics/gmredi).

    MITgcm then forms the GM/Redi tensor from the step-start density gradients
    (do_oceanic_phys.F:759-1046), adds its x, y and r fluxes to every tracer's
    diffusive flux (gad_calc_rhs.F:343-351,472-480,624-632) and Kwz to KappaRT/RS
    after KPP (calc_3d_diffusivity.F:202-209).  data.gmredi is read key by key
    (GMREDI_RULES) and the assembly emits gmredi (_gmredi_entry); the stepper adds
    the module's Kwz to the column diffusivities.  Outside the verbatim mode the
    casegen bases translate under the case-scope rule (CAMPAIGN_DISABLED["usegmredi"]),
    as for ivdc_kappa.
    """
    del value
    if not ctx.verbatim:
        raise Gap(CAMPAIGN_DISABLED["usegmredi"].gap)
    return {f"{_ASSEMBLY}.use_gmredi": True}


def _cd_parameter(key: str, reason_off: str) -> Convert:
    """A data PARM03 CD parameter: staged for cd_scheme when the scheme runs, else a no-op.

    MITgcm reads tauCD, rCD and epsAB_CD only under useCDscheme (ini_parms.F:1114-1118,
    cd_code_scheme.F); the module resolves the unset values the same way.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not (ctx.verbatim and ctx.data("PARM01", "usecdscheme", False) is True):
            raise Skipped(reason_off)
        return {f"{_ASSEMBLY}.{key}": float(value)}

    return fn


#: set_parms.F:132-139: useVariableVisc, which alone makes MITgcm call
#: MOM_CALC_VISC (mom_fluxform.F:453-458, mom_vecinv.F:373-378), is true when one
#: of these PARM01 coefficients is nonzero or one of these PARM05 files is set.
_VARIABLE_VISC_COEFFICIENTS = (
    "viscahgrid", "visca4grid", "viscc2smag", "viscc4smag", "viscc2leith",
    "viscc2leithd", "viscc2leithqg", "viscc4leith", "viscc4leithd",
)
_VARIABLE_VISC_FILES = ("viscahdfile", "viscahzfile", "visca4dfile", "visca4zfile")


def _use_variable_visc(ctx: Ctx) -> bool:
    files = ctx.namelists["data"].get("PARM05", {})
    return any(float(ctx.data("PARM01", key, 0.0)) != 0.0 for key in _VARIABLE_VISC_COEFFICIENTS) or any(
        str(files.get(key) or "").strip() for key in _VARIABLE_VISC_FILES
    )


_CONSTANT_VISC_SKIP = (
    "useVariableVisc is false (set_parms.F:132-139), so MITgcm does not call "
    "MOM_CALC_VISC (mom_fluxform.F:453-458, mom_vecinv.F:373-378)"
)


def _f_grid_cap(key: str, mit_name: str) -> Convert:
    """viscAhGridMax / viscA4GridMax -> the module's grid cap.

    MITgcm caps the variable viscosity at ``<cap> * L2rdt`` (harmonic) or
    ``<cap> * L4rdt`` (biharmonic) inside MOM_CALC_VISC (mom_calc_visc.F:516-517,
    :530-531, :663-664, :677-678), with L2rdt = 0.25 L2 / deltaTmom and L4rdt =
    0.03125 L2**2 / deltaTmom (mom_calc_visc.F:386, mom_init_fixed.F:42-43,
    :113-114); ``_apply_grid_visc_floor_cap`` / ``_apply_grid_visca4_floor_cap``
    are the same clamps on the same lengths.  LOCOMgcm reads a cap <= 0 as no cap
    while MITgcm would clamp the viscosity to it, so only a positive cap maps.
    The MITgcm default 1e21 (set_defaults.F:123, :142) is a cap no viscosity
    reaches, i.e. LOCOMgcm's module default 0.0.

    MOM_CALC_VISC is the caps' only reader (mom_calc_visc.F:516, :530, :663,
    :677) apart from viscA4GridMax in the side drag's sideDragFactor <= 0 branch
    (mom_u_sidedrag.F:58, :75; mom_v_sidedrag.F:58, :75-77), and sideDragFactor has no
    row here (default 2, set_defaults.F:135).  So with useVariableVisc false the
    cap is a no-op and is not passed to a constant-viscosity scheme.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not _use_variable_visc(ctx):
            raise Skipped(f"{_CONSTANT_VISC_SKIP}, the only reader of {mit_name}")
        if float(value) <= 0.0:
            raise Gap(
                f"{mit_name}={value!r} clamps MITgcm's viscosity to zero or below; "
                "LOCOMgcm reads a cap <= 0 as no cap"
            )
        return {f"{_ASSEMBLY}.{key}": float(value)}

    return fn


def _f_use_full_leith(value: Any, ctx: Ctx) -> dict[str, Any]:
    """useFullLeith is read in MOM_CALC_VISC only (mom_calc_visc.F:204, :406, :561)."""
    if not _use_variable_visc(ctx):
        raise Skipped(f"{_CONSTANT_VISC_SKIP}, the only reader of useFullLeith")
    return {f"{_ASSEMBLY}.use_full_leith": value}


def _f_vector_invariant_scheme(
    mit_name: str, schemes: tuple[tuple[str, int], ...], key: str, *, declared: str
) -> Convert:
    """A vector-invariant scheme selector whose configuration value is the MITgcm code.

    The codes LOCOMgcm implements are those of the module's tuple ``schemes``
    (``declared`` names it); MITgcm reads the selector only with
    vectorInvariantMomentum (mom_vecinv.F; set_parms.F:184-196 for
    selectVortScheme, mom_vecinv.F:284 and mom_vi_u/v_vertshear.F:46 for
    selectKEscheme), so a flux-form case skips it.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not bool(ctx.data("PARM01", "vectorinvariantmomentum", False)):
            raise Skipped(f"{mit_name} is read only with vectorInvariantMomentum")
        code = int(value)
        if code not in {c for _, c in schemes}:
            raise Gap(
                f"{mit_name}={code}: LOCOMgcm's vector-invariant momentum implements "
                f"{', '.join(f'{c} ({n})' for n, c in schemes)} ({declared})"
            )
        return {f"{_ASSEMBLY}.{key}": code}

    return fn


def _f_sadourny_coriolis(value: Any, ctx: Ctx) -> dict[str, Any]:
    """SadournyCoriolis, the backward-compatible spelling of selectVortScheme=2.

    INI_PARMS sets an unset selectVortScheme to 2 and stops on any other value
    (ini_parms.F:694-704), whatever the momentum form; SET_PARMS then ignores
    selectVortScheme outside vectorInvariantMomentum (set_parms.F:185-195).  The
    momentum assembly resolves the staged switch (_mit_vort_scheme).
    """
    if value is False:
        raise Skipped("SadournyCoriolis=.FALSE. (ini_parms.F:344), selectVortScheme unchanged")
    explicit = ctx.data("PARM01", "selectvortscheme")
    if explicit is not None and int(explicit) != 2:
        raise Gap(
            f"SadournyCoriolis=.TRUE. conflicts with selectVortScheme={explicit}: "
            "MITgcm stops (ini_parms.F:697-702)"
        )
    if not bool(ctx.data("PARM01", "vectorinvariantmomentum", False)):
        raise Skipped(
            "SadournyCoriolis only sets selectVortScheme, which MITgcm ignores "
            "without vectorInvariantMomentum (set_parms.F:191-195)"
        )
    return {f"{_ASSEMBLY}.sadourny_coriolis": True}


def _vorticity_reconstruction_switch(mit_name: str, key: str, other: str) -> Rule:
    """highOrderVorticity or upwindVorticity, staged for vort_reconstruction.

    MITgcm reads both only in mom_vecinv.F:741-790 (MOM_VI_U/V_CORIOLIS_C4 in
    place of MOM_VI_U/V_CORIOLIS) and in the selectVortScheme default
    (set_parms.F:185-190); CONFIG_CHECK stops when both are on
    (config_check.F:945-951), which is refused here.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not bool(ctx.data("PARM01", "vectorinvariantmomentum", False)):
            raise Skipped(f"{mit_name} is read only with vectorInvariantMomentum (mom_vecinv.F)")
        if value is True and ctx.data("PARM01", other.lower(), False) is True:
            raise Gap(
                f"{mit_name} conflicts with {other}: MITgcm stops (config_check.F:945-951)"
            )
        return {f"{_ASSEMBLY}.{key}": bool(value)}

    return Rule(
        MAP,
        target="model.dynamics.vector_invariant_momentum_advection.vort_reconstruction",
        convert=f"{mit_name} -> vort_reconstruction (VORT_RECONSTRUCTIONS)",
        fn=fn,
    )


def _mit_vort_scheme(asm: _Assembly) -> int:
    """selectVortScheme as MITgcm resolves it in a vector-invariant case.

    The namelist value; unset, 2 under SadournyCoriolis (ini_parms.F:694-696),
    else 0 with upwindVorticity or highOrderVorticity and 1 otherwise
    (set_parms.F:185-190).
    """
    explicit = asm.staged("select_vort_scheme")
    if explicit is not None:
        return int(explicit)
    if asm.flag("sadourny_coriolis"):
        return 2
    if asm.flag("upwind_vorticity") or asm.flag("high_order_vorticity"):
        return 0
    return 1


def _f_buoyancy(value: Any, ctx: Ctx) -> dict[str, Any]:
    if str(value).strip().upper() == "OCEANIC":
        raise Skipped("oceanic buoyancy relation (TO default)")
    raise Gap(f"buoyancyRelation={value!r} (atmosphere / pressure coordinate)")


# MITgcm fills an unset reference level with the level above it, seeded from
# 20 degC / 30 psu for an ocean run (model/src/load_ref_files.F:45-53, :76-84).
_MIT_DEFAULT_TREF = 20.0
_MIT_DEFAULT_SREF = 30.0


def _fill_ref(value: Any, nz: int, seed: float, key: str) -> list[float]:
    """tRef or sRef (``key``) down to Nr the way ``load_ref_files.F:45-53, :76-84`` fills it: a
    level past the list or at MITgcm's unset value takes the level above, the top one ``seed``."""
    values = _as_list(value)[:nz] if value is not None else []
    filled: list[Any] = []
    above: Any = seed
    for k in range(nz):
        level = values[k] if k < len(values) else None
        if level is None or _mit_unset("data", "PARM01", key, level):
            level = above
        filled.append(level)
        above = level
    return filled


def _f_sref(value: Any, ctx: Ctx) -> dict[str, Any]:
    """sRef -> the IC source's per-level ``s_ref`` profile (same shape as tRef)."""
    return {
        f"sources.{_IC_SOURCE_NAME}.s_ref": _fill_ref(value, ctx.nz, _MIT_DEFAULT_SREF, "sref")
    }


def _f_rsphere(value: Any, ctx: Ctx) -> dict[str, Any]:
    """rSphere -> the spherical-polar grid's radius and the metric terms' 1/a.

    MITgcm reads rSphere in the spherical-polar grid (ini_spherical_polar_grid.F:
    103-235), as recip_rSphere = 1/rSphere (ini_parms.F:1361-1365) in the
    flux-form metric terms (mom_u_metric_sphere.F:51, mom_v_metric_sphere.F:48),
    and in the curvilinear rescale (ini_curvilinear_grid.F:390-391).  LOCOMgcm's
    spherical grid takes case.extras.r_sphere (model_input.py
    _build_grid_from_tensors, MetricSet.spherical_polar: the same dxF, dyF, dxG,
    dyG, rA, rAz formulas) and SphericalMetric takes recip_rsphere.  On a
    Cartesian or cylindrical grid neither reader runs; its remaining readers
    (deepAtmosphere factors, NH and quasi-hydrostatic metric terms, GM, sea
    ice) have rows of their own.  On a curvilinear grid rSphere rescales the
    grid file only when radius_fromHorizGrid is set to another value
    (ini_parms.F:1358-1360, ini_curvilinear_grid.F:390-413), feeds
    CALC_GRID_ANGLES only when horizGridFile holds no angles
    (ini_curvilinear_grid.F:343-354, calc_grid_angles.F:60-117) and the Coriolis
    map only on a beta plane (ini_cori.F:70-72).
    """
    if ctx.grid_family == "spherical_polar":
        radius = float(value)
        return {"case.extras.r_sphere": radius, f"{_ASSEMBLY}.recip_rsphere": 1.0 / radius}
    if ctx.grid_family in ("cartesian", "cylindrical"):
        raise Skipped(
            f"rSphere has no reader on a {ctx.grid_family} grid "
            "(ini_spherical_polar_grid.F and the spherical metric terms only)"
        )
    from_grid = ctx.data("PARM04", "radius_fromhorizgrid")
    rescales = from_grid is not None and float(from_grid) != float(value)
    angles_read = bool(str(ctx.data("PARM04", "horizgridfile", "")).strip())
    if not rescales and angles_read and _mit_cori_map(ctx) != 1:
        raise Skipped(
            "on this curvilinear grid rSphere does not rescale the grid file "
            "(radius_fromHorizGrid unset or equal, ini_curvilinear_grid.F:390), the angles "
            "come from horizGridFile (ini_curvilinear_grid.F:343-351) and the Coriolis map is "
            "not a beta plane (ini_cori.F:70-72); its remaining readers have rows of their own"
        )
    raise Gap(
        "rSphere on this curvilinear grid rescales the grid file by "
        "rSphere/radius_fromHorizGrid (ini_curvilinear_grid.F:390-413), sets the angles "
        "of a grid file without them (calc_grid_angles.F:60-117) or scales a beta-plane "
        "Coriolis map (ini_cori.F:70-72); LOCOMgcm's cubed-sphere grid takes its distances "
        "and angles from the tile files only"
    )


def _f_rotation_period(value: Any, ctx: Ctx) -> dict[str, Any]:
    """rotationPeriod -> physics.rotation_period, the run's one rotation period.

    MITgcm sets omega = 2 PI / rotationPeriod (ini_parms.F:487-495, PI of
    PARAMS.h:16) and reads it only in the spherical Coriolis map,
    selectCoriMap = 2 (ini_cori.F:84-99: fCori, fCoriG, fCoriCos).  A translated
    case reaches that map only on the spherical-polar grid.  The canonical
    assembly hands physics.rotation_period to the momentum modules,
    NonhydrostaticParams, the stepper (KPP's f) and the $coriolis_* profiles
    (cfg/15, _canonical_helpers.runtime_kwargs and enrich_manifest), and refuses a
    rotation_period written in module or stepper kwargs.
    """
    cori_map = _mit_cori_map(ctx)
    if cori_map != 2:
        raise Skipped(
            f"omega is read only by the spherical Coriolis map (ini_cori.F:84-99); "
            f"this case uses selectCoriMap={cori_map}"
        )
    if float(value) == ROTATION_PERIOD_EARTH:
        raise Skipped("rotationPeriod equals LOCOMgcm's ROTATION_PERIOD_EARTH (constants.py:18)")
    return {"physics.rotation_period": float(value)}


def _f_celsius2k(value: Any, ctx: Ctx) -> dict[str, Any]:
    """celsius2K: MITgcm's Celsius-to-Kelvin constant (set_defaults.F:276).

    No core routine reads it; its readers are packages (the EXF atempconst
    default, exf_readparms.F:385, the sea-ice and thsice thermodynamics), each
    with rows of its own.  LOCOMgcm's conversion is the constant CEN2KEL
    (defgcm/dynamics/bulk_formulae.py:28).
    """
    del ctx
    if float(value) == CEN2KEL:
        raise Skipped("celsius2K equals LOCOMgcm's CEN2KEL (bulk_formulae.py:28)")
    raise Gap(
        f"celsius2K={value!r}: LOCOMgcm's Celsius-to-Kelvin conversion is the constant "
        f"CEN2KEL = {CEN2KEL:g} (defgcm/dynamics/bulk_formulae.py:28), not configurable"
    )


def _f_select_bot_drag_quadr(value: Any, ctx: Ctx) -> dict[str, Any]:
    """selectBotDragQuadr: the quadratic bottom-drag formulation.

    MITgcm resolves the default -1 to 0 when bottomDragQuadratic or zRoughBot is
    set (ini_parms.F:548-553) and otherwise computes no quadratic drag; 0 averages
    the kinetic energy to the velocity point, 1 and 2 are other forms
    (mom_u_botdrag_coeff.F:133-212).  BottomDrag implements only 0
    (bottom_drag_coefficients.py, bottom_drag.py BottomDrag) and its quadratic
    coefficient comes from bottomDragQuadratic, so -1 is the same model.
    """
    del ctx
    if int(value) in (-1, 0):
        raise Skipped(
            f"selectBotDragQuadr={int(value)}: BottomDrag's only form (KE averaged to the "
            "velocity point, mom_u_botdrag_coeff.F:133-160); -1 resolves to it when a "
            "quadratic coefficient is set (ini_parms.F:548-553)"
        )
    raise Gap(
        f"selectBotDragQuadr={int(value)}: LOCOMgcm's BottomDrag implements only form 0 "
        "(mom_u_botdrag_coeff.F:161-210 are the others)"
    )


def _f_radius_from_horiz_grid(value: Any, ctx: Ctx) -> dict[str, Any]:
    """radius_fromHorizGrid: read only for a curvilinear grid.

    It replaces an unset rSphere and scales the grid file by rSphere /
    radius_fromHorizGrid only when usingCurvilinearGrid (ini_parms.F:1350-1360,
    ini_curvilinear_grid.F:390-391).  With rSphere unset or equal to it the grid
    file is used as read (ini_curvilinear_grid.F:390 compares the two), as
    LOCOMgcm's cubed-sphere grid uses its tile files, and rSphere holds the same
    value for its own readers (its row decides them).
    """
    if ctx.grid_family == "curvilinear":
        r_sphere = ctx.data("PARM04", "rsphere")
        if r_sphere is None or float(r_sphere) == float(value):
            raise Skipped(
                "rSphere is unset or equals radius_fromHorizGrid, so the grid file is "
                "not rescaled (ini_parms.F:1350-1353, ini_curvilinear_grid.F:390)"
            )
        raise Gap(
            f"radius_fromHorizGrid={value!r} with rSphere={r_sphere!r} rescales the "
            "curvilinear grid file by rSphere/radius_fromHorizGrid "
            "(ini_curvilinear_grid.F:390-413); LOCOMgcm's cubed-sphere grid takes "
            "its distances from the tile files only"
        )
    raise Skipped(
        f"radius_fromHorizGrid is read only for a curvilinear grid "
        f"(ini_parms.F:1350-1360); this is a {ctx.grid_family} grid"
    )
