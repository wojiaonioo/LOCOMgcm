"""The shared mapping tables of input/data.pkg, input/data.kpp and input/data.cal."""

from __future__ import annotations

from typing import Any
from experiments.mitcases.code_inventory import MAP
from defgcm.dynamics.kpp_mixing import KPP_STOCK_PARAMS

from .core import (
    CAMPAIGN_DISABLED,
    Gap,
    Skipped,
    _ASSEMBLY,
    _EXF_SOURCE_NAME,
)
from .rules import (
    Convert,
    Ctx,
    Rule,
    _defer,
    _false_or_gap,
    _gap,
    _map,
    _skip,
    _true_or_gap,
    _use_obcs_rule,
)
from .convert import (
    _f_use_gmredi,
)
from .vmix import (
    _vertical_mixing_package,
)

# --------------------------------------------------------------------------
# THE SHARED MAPPING TABLE -- ``input/data.pkg`` (&PACKAGES)
# --------------------------------------------------------------------------
#
# Every ``use<Pkg>`` defaults to .FALSE. in MITgcm
# (references/MITgcm/model/src/packages_boot.F:113-145), so a package that is
# not switched on here is never read -- which is what lets the auxiliary
# ``data.<pkg>`` namelists be whitelisted wholesale when their flag is off.

_USE_EXF_GAP = (
    "pkg/exf outside the verbatim mode: the casegen bases translate without the "
    "EXF sources, which are wired in the verbatim mode only (_wire_exf); data.exf "
    "is read key by key (EXF_RULES)"
)


def _f_use_exf(value: Any, ctx: Ctx) -> dict[str, Any]:
    """useEXF -> the mit_exf or mit_exf_flux source and its consumers (:func:`_wire_exf`)."""
    del value
    if not ctx.verbatim:
        raise Gap(_USE_EXF_GAP)
    return {f"{_ASSEMBLY}.use_exf": True}


def _exf_enabled(ctx: Ctx) -> bool:
    return ctx.namelists.get("data.pkg", {}).get("PACKAGES", {}).get("useexf") is True


def _cal_entry(key: str, name: str) -> Convert:
    """A pkg/cal value: EXF record times read it, nothing else in an ocean run does.

    pkg/cal assigns no model time parameter; outside EXF its dates set pickup
    and dump times only (do_write_pickup.F:66-73, cal_time2dump.F:34).
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not (ctx.verbatim and _exf_enabled(ctx)):
            raise Skipped(
                f"{name}: pkg/cal sets no model time parameter; only EXF record times "
                "read its dates (exf_getffield_start.F:80-104), and EXF is not wired here"
            )
        return {f"{_ASSEMBLY}.{key}": value}

    return fn


PACKAGE_RULES: dict[str, Rule] = {
    "usediagnostics": _skip("MIT diagnostics output; no effect on the state"),
    "usemnc": _skip("MIT NetCDF output layer"),
    "usemonitor": _skip("MIT monitor output layer"),
    "useobcs": _use_obcs_rule(),
    "usekpp": _map(
        "model.stepper.kwargs.kpp_enabled",
        "MIT useKPP -> implicit_free_surface regional KPP path",
    ),
    "useexf": Rule(
        MAP, target=f"sources.{_EXF_SOURCE_NAME} + its consumers (_wire_exf)",
        convert="verbatim mode: mit_exf or mit_exf_flux by ALLOW_ATM_TEMP; outside it a gap",
        fn=_f_use_exf,
    ),
    "usecal": Rule(
        MAP, target=f"sources.{_EXF_SOURCE_NAME}.use_cal",
        convert="EXF record times from data.cal dates (exf_getffield_start.F:80-104)",
        fn=_cal_entry("use_cal", "useCAL"),
    ),
    "useshap_filt": _defer("Shapiro filter; TO counterpart not wired"),
    "usezonal_filt": _gap("zonal filter (pkg/zonal_filt)"),
    # reason: the gap the case-scope rule disables outside the verbatim mode, which the
    # file-level accounting of data.gmredi matches (_account_auxiliary_namelists)
    "usegmredi": Rule(
        MAP, target="model.dynamics <- gmredi",
        convert="verbatim mode: the gmredi module; outside it the case-scope rule",
        reason=CAMPAIGN_DISABLED["usegmredi"].gap, fn=_f_use_gmredi,
    ),
    "useseaice": _gap("dynamic sea ice (out of scope)"),
    "usethsice": _gap("thermodynamic sea ice (out of scope)"),
    "useshelfice": _gap("ice shelf (pkg/shelfice)"),
    "usestreamice": _gap("ice stream / ice sheet (pkg/streamice)"),
    "useaim": _gap("atmospheric intermediate model (pkg/aim)"),
    "useatm_phys": _gap("atmospheric physics (pkg/atm_phys)"),
    "usefizhi": _gap("atmospheric physics (pkg/fizhi)"),
    "usegridalt": _gap("alternate atmospheric vertical grid (pkg/gridalt)"),
    "usecheapaml": _gap("cheap atmospheric mixed layer (pkg/cheapaml)"),
    "useland": _gap("land surface model (pkg/land)"),
    "useptracers": _gap("passive tracers (pkg/ptracers)"),
    "usegchem": _gap("geochemistry driver (pkg/gchem)"),
    "usedic": _gap("dissolved inorganic carbon (pkg/dic)"),
    "usebling": _gap("BLING biogeochemistry (pkg/bling)"),
    "usecfc": _gap("CFC tracers (pkg/cfc)"),
    "userbcs": _gap("relaxed boundary conditions (pkg/rbcs)"),
    "usesalt_plume": _gap("salt plume parameterisation (pkg/salt_plume)"),
    "usedown_slope": _gap("down-slope transport (pkg/down_slope)"),
    "usebbl": _gap("bottom boundary layer (pkg/bbl)"),
    "useicefront": _gap("ice front (pkg/icefront)"),
    "useggl90": _vertical_mixing_package("ggl90"),
    "usekl10": _vertical_mixing_package("kl10"),
    "usemy82": _vertical_mixing_package("my82"),
    "useopps": _gap("OPPS convection (pkg/opps)"),
    "usepp81": _vertical_mixing_package("pp81"),
    # the data.pkg switch is useBulkForce (model/src/packages_boot.F:61)
    "usebulkforce": _gap(
        "bulk formula forcing (pkg/bulk_force); LOCOMgcm's bulk formulae are "
        "pkg/exf's (dynamics/bulk_formulae.py)"
    ),
    # FRAZIL_TENDENCY_APPLY_T adds to gT (external_forcing.F:554-555,
    # apply_forcing.F:696-697); LOCOMgcm's only frazil is inside its sea-ice growth.
    "usefrazil": _gap("frazil heat tendency (pkg/frazil)"),
    "usestic": _gap("sea-ice thermodynamics variant (pkg/stic)"),
    # FLT_MAIN (forward_step.F:1131-1135) and FLT_INIT_FIXED/_VARIA interpolate
    # uVel, vVel, wVel, theta, salt and etaN into FLT.h (flt_up.F:130-136,
    # flt_runga2.F, flt_runga4.F; FLT_BILINEAR/TRILINEAR never assign their
    # field argument) and write nothing back; every FLT_NML key of data.flt sets
    # an FLT.h variable (flt_readparms.F:28-31).
    "useflt": _skip(
        "floats (pkg/flt): read the state, feed nothing back into it"
    ),
    "uselayers": _gap("layers diagnostics (pkg/layers)"),
    "useoffline": _gap("offline tracer transport (pkg/offline)"),
    "usematrix": _gap("transport matrix (pkg/matrix)"),
    "useprofiles": _gap("profile cost (pkg/profiles)"),
    "useobsfit": _gap("observation fit (pkg/obsfit)"),
    "usesbo": _gap("solid body / ocean-bottom diagnostics (pkg/sbo)"),
    "useebm": _gap("energy balance atmosphere (pkg/ebm)"),
    "usempi": _skip("MIT MPI switch"),
    "usemypackage": _gap("template package (pkg/mypackage)"),
    "usesmooth": _gap("smoothing operator (pkg/smooth)"),
    "useexch2": _gap("exch2 tile topology (cube / multi-tile exchange)"),
    "usecpl": _gap("coupler component"),
    # Adjoint and optimisation parameters retain their explicit deferred rules.
    "usegrdchk": _defer("gradient check; adjoint axis"),
    "usectrl": _defer("control variables; adjoint axis"),
    "usecost": _defer("cost function; adjoint axis"),
    "useautodiff": _defer("autodiff store/restore; adjoint axis"),
    "useoptim": _defer("offline optimiser; adjoint axis"),
    "useecco": _defer("ECCO cost terms; adjoint axis"),
    "usedivided": _defer("divided adjoint; adjoint axis"),
}


def _kpp_stock_or_gap(name: str, reason: str) -> Rule:
    """A KPP constant the model fixes in kpp_mixing.KPP_STOCK_PARAMS.

    Restating the fixed value (Rrho0 1.9, dsfmax 1e-2, the MIT defaults of
    kpp_readparms.F:142-143) is a skip; any other value is a gap.
    """
    fixed = float(KPP_STOCK_PARAMS[name])

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del ctx
        if float(value) == fixed:
            raise Skipped(f"{reason} restates the value KPP fixes ({fixed!r}, kpp_readparms.F:142-143)")
        raise Gap(reason)

    return Rule(MAP, target=None, convert="the fixed KPP value -> no-op, else capability gap", fn=fn)


# --------------------------------------------------------------------------
# THE SHARED MAPPING TABLE -- ``input/data.kpp``
# --------------------------------------------------------------------------

KPP_RULES: dict[tuple[str, str], Rule] = {
    ("KPP_PARM01", "kpp_freq"): _gap("KPP diagnostic recomputation frequency"),
    ("KPP_PARM01", "kpp_dumpfreq"): _skip("MIT KPP snapshot output cadence"),
    ("KPP_PARM01", "kpp_tavefreq"): _gap("retired MIT KPP time-average cadence"),
    ("KPP_PARM01", "kppwritestate"): _skip("MIT KPP state output only"),
    # kpp_transport_t.F:69-82 and kpp_transport_s.F:86-95: with useGMRedi the
    # non-local flux multiplies KPPdiffKzT/S + Kwz; without it the switch is a
    # no-op in MITgcm and in the stepper (_kpp_nonlocal_diffusivities).
    ("KPP_PARM01", "kpp_ghatusetotaldiffus"): _map(
        "model.stepper.kwargs.kpp_ghat_use_total_diffus",
        "MIT KPP_ghatUseTotalDiffus -> regional KPP non-local diffusivity",
        lambda value, _ctx: {"model.stepper.kwargs.kpp_ghat_use_total_diffus": bool(value)},
    ),
    # kpp_calc.F:542-553 calls KPP_DOUBLEDIFF (kpp_routines.F:1962-2088) under
    # KPPuseDoubleDiff; EXCLUDE_KPP_DOUBLEDIFF with it set stops MITgcm
    # (kpp_check.F:170-177) and is a deferred macro row here.
    ("KPP_PARM01", "kppusedoublediff"): _map(
        "model.stepper.kwargs.kpp_use_double_diff",
        "MIT KPPuseDoubleDiff -> regional KPP KPP_DOUBLEDIFF",
        lambda value, _ctx: {"model.stepper.kwargs.kpp_use_double_diff": bool(value)},
    ),
    ("KPP_PARM01", "limithblstable"): _true_or_gap(
        "KPP stable-boundary-layer limit"
    ),
    ("KPP_PARM01", "kppuseswfrac3d"): _false_or_gap(
        "KPP three-dimensional shortwave fraction"
    ),
    ("KPP_PARM01", "minkpphbl"): _map(
        "model.stepper.kwargs.kpp_min_hbl",
        "MIT minKPPhbl metres -> regional KPP minKPPhbl",
    ),
    ("KPP_PARM01", "epsln"): _gap("KPP epsln parameter has no TO consumer"),
    ("KPP_PARM01", "phepsi"): _gap("KPP phepsi is not run-configurable"),
    ("KPP_PARM01", "epsilon"): _gap("KPP epsilon is not run-configurable"),
    ("KPP_PARM01", "vonk"): _gap("KPP von Karman constant is not run-configurable"),
    ("KPP_PARM01", "db_dz"): _gap("KPP dB_dz parameter has no TO consumer"),
    ("KPP_PARM01", "conc1"): _gap("KPP conc1 is not run-configurable"),
    ("KPP_PARM01", "conam"): _gap("KPP conam lookup-table constant"),
    ("KPP_PARM01", "concm"): _gap("KPP concm lookup-table constant"),
    ("KPP_PARM01", "conc2"): _gap("KPP conc2 lookup-table constant"),
    ("KPP_PARM01", "zetam"): _gap("KPP zetam lookup-table bound"),
    ("KPP_PARM01", "conas"): _gap("KPP conas lookup-table constant"),
    ("KPP_PARM01", "concs"): _gap("KPP concs lookup-table constant"),
    ("KPP_PARM01", "conc3"): _gap("KPP conc3 lookup-table constant"),
    ("KPP_PARM01", "zetas"): _gap("KPP zetas lookup-table bound"),
    ("KPP_PARM01", "ricr"): _map(
        "model.stepper.kwargs.kpp_ricr",
        "MIT Ricr identity -> regional KPP Ricr",
    ),
    ("KPP_PARM01", "cekman"): _gap("KPP cekman is not run-configurable"),
    ("KPP_PARM01", "cmonob"): _gap("KPP cmonob is not run-configurable"),
    ("KPP_PARM01", "concv"): _gap("KPP concv/Vtc derivation is not run-configurable"),
    ("KPP_PARM01", "hbf"): _gap("KPP shortwave hbf coefficient"),
    ("KPP_PARM01", "zmin"): _gap("KPP zmin lookup-table bound"),
    ("KPP_PARM01", "zmax"): _gap("KPP zmax lookup-table bound"),
    ("KPP_PARM01", "umin"): _gap("KPP umin lookup-table bound"),
    ("KPP_PARM01", "umax"): _gap("KPP umax lookup-table bound"),
    ("KPP_PARM01", "num_v_smooth_ri"): _gap(
        "KPP Richardson-number smoothing count is not run-configurable"
    ),
    ("KPP_PARM01", "riinfty"): _gap("KPP Riinfty is not run-configurable"),
    ("KPP_PARM01", "bvsqcon"): _gap("KPP BVSQcon is not run-configurable"),
    ("KPP_PARM01", "difm0"): _gap("KPP difm0 is not run-configurable"),
    ("KPP_PARM01", "difs0"): _gap("KPP difs0 is not run-configurable"),
    ("KPP_PARM01", "dift0"): _gap("KPP dift0 is not run-configurable"),
    ("KPP_PARM01", "difmcon"): _gap("KPP difmcon is not run-configurable"),
    ("KPP_PARM01", "difscon"): _gap("KPP difscon is not run-configurable"),
    ("KPP_PARM01", "diftcon"): _gap("KPP diftcon is not run-configurable"),
    ("KPP_PARM01", "rrho0"): _kpp_stock_or_gap("Rrho0", "KPP double-diffusion Rrho0 parameter"),
    ("KPP_PARM01", "dsfmax"): _kpp_stock_or_gap("dsfmax", "KPP double-diffusion dsfmax parameter"),
    ("KPP_PARM01", "cstar"): _gap("KPP cstar/cg derivation is not run-configurable"),
    ("KPP_PARM01", "kppmixingmaps"): _false_or_gap("retired MIT KPP mixing maps"),
    ("KPP_PARM01", "num_v_smooth_bv"): _gap("retired MIT KPP BV smoothing"),
    ("KPP_PARM01", "num_z_smooth_sh"): _gap("retired MIT KPP shear z smoothing"),
    ("KPP_PARM01", "num_m_smooth_sh"): _gap("retired MIT KPP shear map smoothing"),
}


# --------------------------------------------------------------------------
# THE SHARED MAPPING TABLE -- ``input/data.cal`` (pkg/cal)
# --------------------------------------------------------------------------
#
# pkg/cal assigns no model time parameter (no startTime, nIter0, deltaT or
# nTimeSteps in pkg/cal/*.F).  Its dates reach the model state only through
# its consumers, each with its own row: EXF record times
# (exf_getffield_start.F:80-104, exf_getffieldrec.F:80-198, exf_set_fld.F:
# 133-150) and the OBCS fields EXF loads (obcs_exf_load.F:284-300).  The core
# uses it for pickup and NetCDF dump times only (do_write_pickup.F:66-73,
# ini_mnc_vars.F:206), pkg/diagnostics for output times
# (diagnostics_switch_onoff.F:112,202).  CAL_NML is cal_readparms.F:59-62.

CAL_RULES: dict[tuple[str, str], Rule] = {
    ("CAL_NML", "thecalendar"): Rule(
        MAP, target=f"sources.{_EXF_SOURCE_NAME}.calendar",
        convert="as read (cal_set.F:99-110); the EXF sources take gregorian only",
        fn=_cal_entry("cal_calendar", "TheCalendar"),
    ),
    ("CAL_NML", "startdate_1"): Rule(
        MAP, target=f"sources.{_EXF_SOURCE_NAME}.start_date_1",
        convert="yyyymmdd of the model start (exf_getffield_start.F:99-103)",
        fn=_cal_entry("cal_start_date_1", "startDate_1"),
    ),
    ("CAL_NML", "startdate_2"): Rule(
        MAP, target=f"sources.{_EXF_SOURCE_NAME}.start_date_2",
        convert="hhmmss of the model start (exf_getffield_start.F:99-103)",
        fn=_cal_entry("cal_start_date_2", "startDate_2"),
    ),
    ("CAL_NML", "calendardumps"): _skip(
        "output dump times on calendar months (cal_time2dump.F:34)"
    ),
}
