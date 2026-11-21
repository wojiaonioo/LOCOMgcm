"""The shared mapping table of input/data.gmredi and the gmredi model entry."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Any, Callable
from experiments.mitcases.code_inventory import MAP
from locomgcm.dynamics.gmredi.calc_tensor import GM_TAPER_SCHEMES

from .core import (
    Gap,
    Record,
    Skipped,
    _ASSEMBLY,
    _Assembly,
    _MISSING,
    _entry,
    _record_gap,
)
from .rules import (
    Ctx,
    Rule,
    _coriolis_profile,
    _false_or_gap,
    _is_zero,
    _skip,
)
from .cpp import (
    _gmredi_cpp_states,
)

# --------------------------------------------------------------------------
# THE SHARED MAPPING TABLE -- ``input/data.gmredi`` (pkg/gmredi)
# --------------------------------------------------------------------------
#
# GM_PARM01 is gmredi_readparms.F:52-91, its defaults :110-195; an unset
# GM_isopycK (-999) then takes GM_background_K (:223).  A key the gmredi module
# carries maps onto its field (locomgcm/dynamics/gmredi/module.py GMRedi), which
# refuses at construction what it does not cover (_try_construct).  A switch of a
# branch the module lacks is a skip at its MIT default and unsupported otherwise;
# a parameter read only inside such a branch, or only by one taper scheme, is a
# skip while that branch is off; a factor or coefficient file is a skip when
# blank (factors 1, gmredi_init_fixed.F:46-49,75-76; constant K) and unsupported
# otherwise.  The module covers constant K without factor files
# (locomgcm/dynamics/gmredi/calc_tensor.py).

_GM_VARIABLE_K_GAP = (
    "variable GM/Redi diffusivity (Bates gmredi_calc_tensor.F:309-328, QG Leith "
    ":344-357, GEOMETRIC :365-376); of the variable K the gmredi module computes "
    "the Visbeck K only"
)
_GM_FILE_GAP = (
    "GM/Redi diffusivity factor or coefficient file (gmredi_init_fixed.F:62-131, "
    "gmredi_init_varia.F:86-165); the gmredi module has constant GM_isopycK and "
    "GM_background_K only"
)


def _gm(ctx: Ctx, key: str, default: Any) -> Any:
    """A data.gmredi GM_PARM01 value as read (lowercase key), else ``default``."""
    return ctx.namelists.get("data.gmredi", {}).get("GM_PARM01", {}).get(key, default)


def _gm_taper(ctx: Ctx) -> str:
    """GM_taper_scheme as Fortran compares it: blank-padded, so trailing blanks drop."""
    return str(_gm(ctx, "gm_taper_scheme", " ")).rstrip()


def _gm_leith_qg_on(ctx: Ctx) -> bool:
    """GM_useLeithQG after gmredi_readparms.F:234-240 resets it when viscC2LeithQG is zero."""
    return _gm(ctx, "gm_useleithqg", False) is True and not _is_zero(
        ctx.data("PARM01", "viscc2leithqg", 0.0)
    )


def _gm_bvp_on(ctx: Ctx) -> bool:
    """GMREDI_CALC_PSI_BVP runs only under GM_BOLUS_BVP with GM_AdvForm (gmredi_calc_tensor.F:609-631)."""
    return (
        _gm(ctx, "gm_usebvp", False) is True
        and _gm(ctx, "gm_advform", False) is True
        and _gmredi_cpp_states(ctx.case_dir)[0].get("GM_BOLUS_BVP", False)
    )


def _gm_field(field: str, cast: Callable[[Any], Any], line: str) -> Rule:
    """A data.gmredi key the gmredi module carries as ``field`` (MIT default at ``line``)."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del ctx
        return {f"{_ASSEMBLY}.gm_{field}": cast(value)}

    return Rule(
        MAP, target=f"model.dynamics.gmredi.{field}",
        convert=f"identity (default gmredi_readparms.F:{line})", fn=fn,
    )


def _f_gm_taper_scheme(value: Any, ctx: Ctx) -> dict[str, Any]:
    """GM_taper_scheme -> gmredi.taper_scheme when GM_TAPER_SCHEMES declares it.

    ' ' and '' are the blank default (no taper); 'orig', 'clipping', 'fm07',
    'ac02' and 'stableGmAdjTap' are the other branches of
    gmredi_slope_limit.F:171-440,594-639.
    """
    del ctx
    scheme = str(value).rstrip()
    if scheme not in GM_TAPER_SCHEMES:
        raise Gap(
            f"GM_taper_scheme={scheme!r} (gmredi_slope_limit.F:171-440,594-639); the "
            f"gmredi module declares GM_TAPER_SCHEMES {GM_TAPER_SCHEMES}"
        )
    return {f"{_ASSEMBLY}.gm_taper_scheme": scheme}


def _f_gm_visbeck_alpha(value: Any, ctx: Ctx) -> dict[str, Any]:
    """GM_Visbeck_alpha -> gmredi.visbeck_alpha (default 0, gmredi_readparms.F:151).

    A nonzero value without GM_VISBECK_VARIABLE_K in the GMREDI_OPTIONS.h the build
    compiles stops MITgcm (gmredi_check.F:166-176); with it the module computes the
    Visbeck K (gmredi_calc_tensor.F:416-503,579-597) and refuses at construction the
    values it does not cover (alpha < 0, GM_isoFac_calcK without GM_NON_UNITY_DIAGONAL).
    """
    alpha = float(value)
    if alpha != 0.0 and not _gmredi_cpp_states(ctx.case_dir)[0].get("GM_VISBECK_VARIABLE_K", False):
        raise Gap(
            "GM_Visbeck_alpha nonzero without GM_VISBECK_VARIABLE_K stops MITgcm "
            "(gmredi_check.F:166-176)"
        )
    return {f"{_ASSEMBLY}.gm_visbeck_alpha": alpha}


def _gm_switch(active: Callable[[Ctx], bool], branch: str, gap: str) -> Rule:
    """A data.gmredi switch: .FALSE. is the MIT default, .TRUE. acts only under ``branch``."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if value is False:
            raise Skipped("restates the MIT default .FALSE. (gmredi_readparms.F:110-195)")
        if not active(ctx):
            raise Skipped(f".TRUE. but MITgcm reads it only {branch}, which does not hold")
        raise Gap(gap)

    return Rule(
        MAP, target=None, convert=f".FALSE., or {branch} not holding -> no-op, else capability gap",
        fn=fn,
    )


def _gm_read_only_under(active: Callable[[Ctx], bool], branch: str, gap: str) -> Rule:
    """A data.gmredi parameter MITgcm reads only under ``branch``."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not active(ctx):
            raise Skipped(f"read only {branch}, which does not hold")
        raise Gap(gap)

    return Rule(MAP, target=None, convert=f"no-op unless {branch}, else capability gap", fn=fn)


def _gm_file(gap: str) -> Rule:
    """A data.gmredi file name: blank is MITgcm's "no file"."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not str(value).strip():
            raise Skipped("blank, the MIT default (gmredi_readparms.F:127-132,199-200): no file is read")
        raise Gap(gap)

    return Rule(MAP, target=None, convert="blank -> no file, else capability gap", fn=fn)


GMREDI_RULES: dict[tuple[str, str], Rule] = {
    ("GM_PARM01", "gm_advform"): _gm_field("adv_form", bool, "110"),
    ("GM_PARM01", "gm_isopyck"): _gm_field("isopyc_k", float, "113; -999 -> GM_background_K at :223"),
    ("GM_PARM01", "gm_background_k"): _gm_field("background_k", float, "114"),
    ("GM_PARM01", "gm_maxslope"): _gm_field("max_slope", float, "115"),
    ("GM_PARM01", "gm_kmin_horiz"): _gm_field("kmin_horiz", float, "116"),
    ("GM_PARM01", "gm_small_number"): _gm_field("small_number", float, "117"),
    ("GM_PARM01", "gm_slopesqcutoff"): _gm_field("slope_sq_cutoff", float, "118"),
    ("GM_PARM01", "gm_scrit"): _gm_field("s_crit", float, "123"),
    ("GM_PARM01", "gm_sd"): _gm_field("s_d", float, "124"),
    ("GM_PARM01", "gm_taper_scheme"): Rule(
        MAP, target="model.dynamics.gmredi.taper_scheme",
        convert="trailing blanks dropped; GM_TAPER_SCHEMES, else capability gap",
        fn=_f_gm_taper_scheme,
    ),
    ("GM_PARM01", "gm_advseparate"): _gm_switch(
        lambda ctx: _gm(ctx, "gm_advform", False) is True,
        "with GM_AdvForm (gmredi_xtransport.F:195, gmredi_rtransport.F:159, "
        "gmredi_residual_flow.F:58)",
        "separate bolus advection of the GM advective form (gmredi_xtransport.F:"
        "194-236); the gmredi module has the residual-velocity form only",
    ),
    ("GM_PARM01", "gm_inmomasstress"): _false_or_gap(
        "GM bolus transport as a momentum stress (calc_eddy_stress.F, "
        "taueddy_tendency_apply.F; ALLOW_EDDYPSI, gmredi_check.F:394-416)"
    ),
    ("GM_PARM01", "gm_usebvp"): _gm_switch(
        _gm_bvp_on,
        "with GM_AdvForm and GM_BOLUS_BVP defined (gmredi_calc_tensor.F:609-631)",
        "boundary-value-problem bolus streamfunction (gmredi_calc_psi_bvp.F)",
    ),
    ("GM_PARM01", "gm_usesubmeso"): _false_or_gap(
        "sub-mesoscale bolus streamfunction (submeso_calc_psi.F, called at "
        "gmredi_calc_tensor.F:638; with GM_AdvForm off gmredi_check.F:387-392 stops)"
    ),
    ("GM_PARM01", "gm_visbeck_alpha"): Rule(
        MAP, target="model.dynamics.gmredi.visbeck_alpha",
        convert="identity (default gmredi_readparms.F:151); nonzero without "
                "GM_VISBECK_VARIABLE_K -> capability gap (gmredi_check.F:166-176)",
        fn=_f_gm_visbeck_alpha,
    ),
    ("GM_PARM01", "gm_usebatesk3d"): _false_or_gap(_GM_VARIABLE_K_GAP),
    ("GM_PARM01", "gm_useleithqg"): _gm_switch(
        _gm_leith_qg_on,
        "with viscC2LeithQG nonzero (gmredi_readparms.F:234-240 resets it)",
        _GM_VARIABLE_K_GAP,
    ),
    ("GM_PARM01", "gm_usegeom"): _false_or_gap(_GM_VARIABLE_K_GAP),
    ("GM_PARM01", "gm_isofac_calck"): _gm_field("iso_fac_calc_k", float, "125"),
    ("GM_PARM01", "gm_mnc"): _skip("GM/Redi NetCDF output switch (gmredi_output.F:67)"),
}

for _key in ("gm_iso2dfile", "gm_iso1dfile", "gm_bol2dfile", "gm_bol1dfile",
             "gm_k3dredifile", "gm_k3dgmfile"):
    GMREDI_RULES[("GM_PARM01", _key)] = _gm_file(_GM_FILE_GAP)
for _key in ("gm_background_k3dfile", "gm_isopyck3dfile"):
    GMREDI_RULES[("GM_PARM01", _key)] = _gm_file(
        "retired GM/Redi file parameter: any name stops MITgcm (gmredi_readparms.F:288-316)"
    )
for _key in ("gm_factrl2dz", "gm_factrl2ml", "gm_maxtranslay"):
    GMREDI_RULES[("GM_PARM01", _key)] = _gm_read_only_under(
        lambda ctx: _gm_taper(ctx) == "fm07",
        "by the fm07 taper (gmredi_slope_limit.F:229-392)",
        "fm07 transition-layer taper (gmredi_slope_limit.F:229-392); not in GM_TAPER_SCHEMES",
    )
for _key in ("gm_bvp_cmin", "gm_bvp_modenumber"):
    GMREDI_RULES[("GM_PARM01", _key)] = _gm_read_only_under(
        _gm_bvp_on,
        "by the BVP streamfunction (gmredi_readparms.F:274-277, gmredi_calc_psi_bvp.F:116-214)",
        "boundary-value-problem bolus streamfunction (gmredi_calc_psi_bvp.F)",
    )
for _key in ("submeso_ceff", "submeso_invtau", "submeso_lfmin", "submeso_lmax"):
    GMREDI_RULES[("GM_PARM01", _key)] = _gm_read_only_under(
        lambda ctx: _gm(ctx, "gm_usesubmeso", False) is True,
        "with GM_useSubMeso (submeso_calc_psi.F:96-197)",
        "sub-mesoscale bolus streamfunction (submeso_calc_psi.F)",
    )
# The Visbeck parameters the module carries (read when GM_Visbeck_alpha > 0, written
# resolved either way); GM_Visbeck_maxSlope unset is GM_maxSlope (:225-227).
for _key, _field, _line in (
    ("gm_visbeck_length", "visbeck_length", "152"),
    ("gm_visbeck_depth", "visbeck_depth", "153"),
    ("gm_visbeck_mindepth", "visbeck_min_depth", "154"),
    ("gm_visbeck_maxslope", "visbeck_max_slope", "155; unset -> GM_maxSlope at :225-227"),
    ("gm_visbeck_minval_k", "visbeck_min_val_k", "156"),
    ("gm_visbeck_maxval_k", "visbeck_max_val_k", "157"),
):
    GMREDI_RULES[("GM_PARM01", _key)] = _gm_field(_field, float, _line)
for _key in ("gm_bates_smooth", "gm_bates_use_constk", "gm_bates_beta_eq_0",
             "gm_bates_thicksheet", "gm_bates_surfk", "gm_bates_constredi",
             "gm_bates_gamma", "gm_bates_b1", "gm_bates_eadymindepth",
             "gm_bates_eadymaxdepth", "gm_bates_lambda", "gm_bates_smallk",
             "gm_bates_maxk", "gm_bates_constk", "gm_bates_maxc", "gm_bates_rmax",
             "gm_bates_rmin", "gm_bates_mincori", "gm_bates_minn2",
             "gm_bates_surfmindepth", "gm_bates_vecfreq", "gm_bates_minrenorm",
             "gm_bates_maxrenorm"):
    GMREDI_RULES[("GM_PARM01", _key)] = _gm_read_only_under(
        lambda ctx: _gm(ctx, "gm_usebatesk3d", False) is True,
        "with GM_useBatesK3d (gmredi_calc_bates_k.F, called at gmredi_calc_tensor.F:309-328)",
        _GM_VARIABLE_K_GAP,
    )
for _key in ("geom_lmbda", "geom_alpha", "geom_ini_eke", "geom_diffkh_eke",
             "geom_vert_struc", "geom_vert_struc_min", "geom_vert_struc_max",
             "geom_minval_k", "geom_maxval_k"):
    GMREDI_RULES[("GM_PARM01", _key)] = _gm_read_only_under(
        lambda ctx: _gm(ctx, "gm_usegeom", False) is True,
        "with GM_useGEOM (gmredi_calc_geom.F, called at gmredi_calc_tensor.F:365-376; "
        "GEOM_K3d stays zero otherwise, gmredi_init_varia.F:128)",
        _GM_VARIABLE_K_GAP,
    )




#: GMREDI_OPTIONS.h macros the gmredi module carries, and their fields.
_GMREDI_CPP_FIELDS = {
    "GM_NON_UNITY_DIAGONAL": "non_unity_diagonal",
    "GM_EXTRA_DIAGONAL": "extra_diagonal",
    "GM_EXCLUDE_TAPERING": "exclude_tapering",
}


def _gmredi_entry(
    asm: _Assembly, ctx: Ctx, case_dir: Path, record: Record, tracers: tuple[str, ...],
    *, gravity: float, rho_const: float,
) -> dict[str, Any]:
    """The gmredi entry of useGMRedi from data.gmredi (GMREDI_RULES) and GMREDI_OPTIONS.h.

    ``tracers`` are the stepped tracers (tempStepping, saltStepping; unset is
    .TRUE., set_defaults.F:194,198): MITgcm adds the GM fluxes inside
    TEMP_INTEGRATE / SALT_INTEGRATE, called only for a stepped tracer
    (thermodynamics.F:317-337), and the module takes them as ``tracer_names``.
    The entry is assembled with the tracers, so the set is never empty; with no
    tracer stepping GM reaches no state in MITgcm either (its fluxes, Kwz and the
    residual velocity are consumed only by the tracer integration; the momentum
    route GM_InMomAsStress is a gap row of its own).

    ``coriolis_profile_c`` is the cell-centre fCori of ini_cori.F, which the ldd97
    taper reads for its Rossby radius (gmredi_calc_tensor.F:163-211) whatever
    useCoriolis says; it is the momentum bundles' map (_coriolis_profile, the
    ``$coriolis_profile_c`` sentinel on a spherical grid, resolved from
    physics.rotation_period), and None for every other taper, which the module
    requires.  A beta-plane map without a uniform delY, which the momentum path
    defers to f0, is a gap here.

    The Visbeck parameters and GM_isoFac_calcK are written resolved whatever
    GM_Visbeck_alpha is (defaults gmredi_readparms.F:125,151-157; GM_Visbeck_maxSlope
    unset is GM_maxSlope, :225-227); ``gravity`` and ``rho_const`` are the assembly's
    resolved gravity and rhoConst (unset rhoNil, ini_parms.F:483), which the Visbeck
    N^2 reads (gmredi_calc_tensor.F:488).  ldd97 or the Visbeck K with dry cells at
    the top (useShelfIce, or a topoFile) is a gap: MITgcm stops for ldd97
    (gmredi_check.F:460-472) and only warns for Visbeck (:485-489), whose module
    takes rTop = 0 and kSurfC = 1 (locomgcm/dynamics/gmredi/visbeck.py).

    GM_isopycK unset, or -999, is GM_background_K (gmredi_readparms.F:113,223).
    The macros come from the header the build compiles (_gmredi_cpp_states):
    GM_NON_UNITY_DIAGONAL selects the tapered Kux, Kvy floored at GM_Kmin_horiz
    (gmredi_calc_tensor.F:785-808,982-1005) over constant GM_isopycK (:1093-1118),
    GM_EXTRA_DIAGONAL compiles Kuz, Kvz (:810-847,1007-1044) and the off-diagonal
    x, y fluxes (gmredi_xtransport.F:142-192), GM_EXCLUDE_TAPERING leaves a STOP in
    the taper branch (gmredi_slope_limit.F:441-445).  The module refuses at
    construction what it does not cover.  MITgcm stops without implicitDiffusion
    (gmredi_check.F:137-141); Kwz then has no consumer in LOCOMgcm either, the
    stepper adding it only to an implicit column diffusivity.
    """
    background_k = asm.number("gm_background_k")
    isopyc_k = asm.staged("gm_isopyc_k")
    if isopyc_k is None:
        isopyc_k = background_k
    states, header = _gmredi_cpp_states(case_dir)
    taper_scheme = str(asm.get("gm_taper_scheme"))
    profile = _coriolis_profile(asm, ctx, corner=False) if taper_scheme == "ldd97" else None
    if taper_scheme == "ldd97" and profile is None:
        _record_gap(
            record,
            {"source": "data.gmredi:GM_PARM01.GM_taper_scheme",
             "parameter": "<derived:gmredi ldd97 fCori>", "value": taper_scheme},
            "ldd97 reads the cell-centre fCori (gmredi_calc_tensor.F:163-211); the "
            "beta-plane map needs a uniform delY to place yC (ini_cori.F:77)",
        )
    visbeck_alpha = asm.number("gm_visbeck_alpha")
    dry_top = [
        key for key, on in (
            ("useShelfIce", ctx.namelists.get("data.pkg", {}).get("PACKAGES", {}).get("useshelfice") is True),
            ("topoFile", bool(str(ctx.data("PARM05", "topofile", "") or "").strip())),
        ) if on
    ]
    readers = [name for name, on in (("ldd97", taper_scheme == "ldd97"),
                                     ("Visbeck K", visbeck_alpha != 0.0)) if on]
    if dry_top and readers:
        _record_gap(
            record,
            {"source": "data.gmredi:GM_PARM01",
             "parameter": "<derived:gmredi with dry cells at the top>",
             "value": {"readers": readers, "dry_top": dry_top}},
            f"{' and '.join(readers)} with {' and '.join(dry_top)}: MITgcm stops for ldd97 "
            "(gmredi_check.F:460-472) and only warns for Visbeck (:485-489); the module's "
            "Visbeck K takes rTop = 0 and kSurfC = 1 (locomgcm/dynamics/gmredi/visbeck.py)",
        )
    kwargs: dict[str, Any] = {
        "taper_scheme": taper_scheme,
        "adv_form": asm.flag("gm_adv_form"),
        "isopyc_k": float(isopyc_k),
        "background_k": background_k,
        "max_slope": asm.number("gm_max_slope"),
        "kmin_horiz": asm.number("gm_kmin_horiz"),
        "small_number": asm.number("gm_small_number"),
        "slope_sq_cutoff": asm.number("gm_slope_sq_cutoff"),
        "s_crit": asm.number("gm_s_crit"),
        "s_d": asm.number("gm_s_d"),
        **{field: states.get(macro, False) for macro, field in _GMREDI_CPP_FIELDS.items()},
        "visbeck_alpha": visbeck_alpha,
        "visbeck_length": asm.number("gm_visbeck_length"),
        "visbeck_depth": asm.number("gm_visbeck_depth"),
        "visbeck_min_depth": asm.number("gm_visbeck_min_depth"),
        "visbeck_max_slope": float(asm.staged("gm_visbeck_max_slope", asm.number("gm_max_slope"))),
        "visbeck_min_val_k": asm.number("gm_visbeck_min_val_k"),
        "visbeck_max_val_k": asm.number("gm_visbeck_max_val_k"),
        "iso_fac_calc_k": asm.number("gm_iso_fac_calc_k"),
        "gravity": float(gravity),
        "rho_const": float(rho_const),
        "tracer_names": list(tracers),
        "coriolis_profile_c": profile,
    }
    record.derived.append({
        "source": header,
        "target": "model.dynamics.gmredi",
        "value": dict(kwargs),
        "rationale": (
            "data.gmredi after MITgcm's defaults, GM_isopycK unset -> GM_background_K "
            "(gmredi_readparms.F:113,223); GM_NON_UNITY_DIAGONAL, GM_EXTRA_DIAGONAL, "
            "GM_EXCLUDE_TAPERING from the GMREDI_OPTIONS.h the build compiles; tracer_names "
            "the stepped tracers (thermodynamics.F:317-337); coriolis_profile_c the "
            "ini_cori.F fCori for ldd97 (gmredi_calc_tensor.F:163-211), None otherwise; "
            "GM_Visbeck_* and GM_isoFac_calcK after gmredi_readparms.F:125,151-157,225-227; "
            "gravity and rhoConst of the assembly (gmredi_calc_tensor.F:488)"
        ),
    })
    implicit = asm.yaml("physics.tracer.implicit_vertical_diffusion", _MISSING)
    if implicit is _MISSING:
        implicit = asm.flag("implicit_vertical_diffusion")
    if not implicit:
        _record_gap(
            record,
            {"source": "input/data:PARM01.implicitDiffusion",
             "parameter": "<derived:gmredi without implicitDiffusion>", "value": False},
            "GM/Redi without implicitDiffusion stops MITgcm (gmredi_check.F:137-141); "
            "the stepper adds Kwz only to an implicit column diffusivity",
        )
    return _entry("gmredi", **kwargs)
