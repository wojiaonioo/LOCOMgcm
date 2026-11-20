"""Reading the staged namelists and MITgcm's unset values."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from experiments.cases.scs_clm.mit_native_io import parse_namelist

from .core import (
    _lookup_key,
)

#: EXF_NML_02 fields with a file and a record clock (exf_readparms.F:87-142);
#: the same fields carry an input grid in EXF_NML_04 (exf_readparms.F:201-259).
_EXF_FIELDS = (
    "hflux", "atemp", "aqh", "hs", "hl", "sflux", "evap", "precip", "snowprecip",
    "runoff", "saltflx", "ustress", "vstress", "uwind", "vwind", "wspeed",
    "swflux", "lwflux", "swdown", "lwdown", "apressure", "tidepot", "areamask",
    "climsst", "climsss", "climustr", "climvstr",
)


def _exf_prefix(field: str) -> str:
    """Key prefix of an EXF_NML_02 field: ``hs_file``, ``hs_period`` (exf_readparms.F:94-97)."""
    return f"{field}_" if field in ("hs", "hl") else field


#: MITgcm's unset values UNSET_RL (= UNSET_RS) and UNSET_I (eesupp/inc/EEPARAMS.h:80-85).
_UNSET_RL = 1.234567e5
_UNSET_I = 123456789


@dataclass(frozen=True)
class _Unset:
    """The value a namelist parameter holds until the namelist sets it, which MITgcm compares with
    after reading to decide that the parameter was not given, and where it does.  ``per_element``
    marks an array whose unset entries a resolver fills one by one (tRef and sRef from the level
    above, OB_J*/OB_I* from OB_single*); any other array is unset only as a whole."""

    value: float
    source: str
    per_element: bool = False


def _unset_table() -> dict[tuple[str, str, str], _Unset]:
    rl, i = _UNSET_RL, _UNSET_I
    table = {
        ("data", "PARM01", "gbaro"): _Unset(rl, "model/src/ini_parms.F:482"),
        ("data", "PARM01", "rhoconst"): _Unset(rl, "model/src/ini_parms.F:483"),
        ("data", "PARM01", "rhoconstfresh"): _Unset(rl, "model/src/ini_parms.F:484"),
        ("data", "PARM01", "tempvertadvscheme"): _Unset(0, "model/src/ini_parms.F:509"),
        ("data", "PARM01", "saltvertadvscheme"): _Unset(0, "model/src/ini_parms.F:510"),
        ("data", "PARM01", "viscaz"): _Unset(rl, "model/src/ini_parms.F:520-524"),
        ("data", "PARM01", "viscar"): _Unset(rl, "model/src/ini_parms.F:520-524"),
        ("data", "PARM01", "diffkzt"): _Unset(rl, "model/src/ini_parms.F:567-571"),
        ("data", "PARM01", "diffkrt"): _Unset(rl, "model/src/ini_parms.F:567-571"),
        ("data", "PARM01", "diffkrnrt"): _Unset(rl, "model/src/ini_parms.F:572-575"),
        ("data", "PARM01", "diffkzs"): _Unset(rl, "model/src/ini_parms.F:599-603"),
        ("data", "PARM01", "diffkrs"): _Unset(rl, "model/src/ini_parms.F:599-603"),
        ("data", "PARM01", "diffkrnrs"): _Unset(rl, "model/src/ini_parms.F:604-607"),
        ("data", "PARM01", "hfacmindz"): _Unset(rl, "model/src/ini_parms.F:641-646"),
        ("data", "PARM01", "hfacmindr"): _Unset(rl, "model/src/ini_parms.F:641-646"),
        ("data", "PARM01", "convertfw2salt"): _Unset(rl, "model/src/ini_parms.F:648-652"),
        ("data", "PARM01", "selectpenetratingsw"): _Unset(i, "model/src/ini_parms.F:653-657"),
        ("data", "PARM01", "selectcorischeme"): _Unset(i, "model/src/ini_parms.F:661-667"),
        ("data", "PARM01", "selectvortscheme"): _Unset(
            i, "model/src/ini_parms.F:696, set_parms.F:185-190"),
        ("data", "PARM01", "select3dcorischeme"): _Unset(i, "model/src/ini_parms.F:705-708"),
        ("data", "PARM01", "selectmetricterms"): _Unset(i, "model/src/ini_parms.F:716-719"),
        ("data", "PARM01", "talpha"): _Unset(rl, "model/src/ini_eos.F:85"),
        ("data", "PARM01", "sbeta"): _Unset(rl, "model/src/ini_eos.F:86"),
        ("data", "PARM01", "selectp_ineos_zc"): _Unset(i, "model/src/set_parms.F:275"),
        ("data", "PARM01", "temp_evprrn"): _Unset(rl, "model/src/external_forcing_surf.F:268"),
        ("data", "PARM01", "tref"): _Unset(rl, "model/src/load_ref_files.F:50-52", True),
        ("data", "PARM01", "sref"): _Unset(rl, "model/src/load_ref_files.F:81-83", True),
        ("data", "PARM02", "cg2duseminressol"): _Unset(i, "model/src/ini_parms.F:1585-1588"),
        ("data", "PARM03", "deltat"): _Unset(0, "model/src/ini_parms.F:1043-1047"),
        ("data", "PARM03", "deltatclock"): _Unset(0, "model/src/ini_parms.F:1043, :1059"),
        ("data", "PARM03", "deltattracer"): _Unset(0, "model/src/ini_parms.F:1044, :1058"),
        ("data", "PARM03", "deltatmom"): _Unset(0, "model/src/ini_parms.F:1045, :1057"),
        ("data", "PARM03", "deltatfreesurf"): _Unset(0, "model/src/ini_parms.F:1046, :1068"),
        ("data", "PARM03", "momforcingoutab"): _Unset(i, "model/src/ini_parms.F:1095-1097"),
        ("data", "PARM03", "tracforcingoutab"): _Unset(i, "model/src/ini_parms.F:1099-1101"),
        ("data", "PARM03", "epsab_cd"): _Unset(rl, "model/src/ini_parms.F:1118"),
        ("data", "PARM03", "starttime"): _Unset(rl, "model/src/ini_parms.F:1121-1127"),
        ("data", "PARM03", "niter0"): _Unset(-1, "model/src/ini_parms.F:1121-1130"),
        ("data", "PARM03", "basetime"): _Unset(0, "model/src/ini_parms.F:1131"),
        ("data", "PARM03", "latbandclimrelax"): _Unset(
            rl, "model/src/set_parms.F:267-270, load_grid_spacing.F:172-174"),
        ("data", "PARM03", "alph_ab"): _Unset(rl, "model/src/config_check.F:190"),
        ("data", "PARM03", "beta_ab"): _Unset(rl, "model/src/config_check.F:190"),
        ("data", "PARM04", "dxspacing"): _Unset(rl, "model/src/ini_parms.F:1295-1308"),
        ("data", "PARM04", "dyspacing"): _Unset(rl, "model/src/ini_parms.F:1311-1324"),
        ("data", "PARM04", "delx"): _Unset(rl, "model/src/ini_parms.F:1294, load_grid_spacing.F:118"),
        ("data", "PARM04", "dely"): _Unset(rl, "model/src/ini_parms.F:1310, load_grid_spacing.F:143"),
        ("data", "PARM04", "rsphere"): _Unset(rl, "model/src/ini_parms.F:1350-1357"),
        ("data", "PARM04", "radius_fromhorizgrid"): _Unset(rl, "model/src/ini_parms.F:1358-1360"),
        ("data", "PARM04", "top_pres"): _Unset(rl, "model/src/ini_parms.F:1388"),
        ("data", "PARM04", "xgorigin"): _Unset(rl, "model/src/ini_parms.F:1391"),
        ("data", "PARM04", "ygorigin"): _Unset(rl, "model/src/ini_parms.F:1392-1400"),
        ("data", "PARM04", "delrc"): _Unset(rl, "model/src/ini_parms.F:1411-1430"),
        ("data", "PARM04", "delz"): _Unset(rl, "model/src/ini_parms.F:1433-1438"),
        ("data", "PARM04", "delr"): _Unset(rl, "model/src/ini_parms.F:1433-1438"),
        ("data.gmredi", "GM_PARM01", "gm_isopyck"): _Unset(-999, "pkg/gmredi/gmredi_readparms.F:223"),
        ("data.gmredi", "GM_PARM01", "gm_visbeck_maxslope"): _Unset(
            rl, "pkg/gmredi/gmredi_readparms.F:225-227"),
        ("data.ggl90", "GGL90_PARM01", "ggl90tkebottom"): _Unset(
            rl, "pkg/ggl90/ggl90_readparms.F:251-253"),
        ("data.kpp", "KPP_PARM01", "minkpphbl"): _Unset(rl, "pkg/kpp/kpp_init_fixed.F:162-164"),
        ("data.exf", "EXF_NML_01", "exf_iprec_obcs"): _Unset(i, "pkg/exf/exf_readparms.F:1022"),
    }
    for key, line in (("num_v_smooth_bv", "201"), ("num_z_smooth_sh", "207"),
                      ("num_m_smooth_sh", "213")):
        table[("data.kpp", "KPP_PARM01", key)] = _Unset(i, f"pkg/kpp/kpp_readparms.F:{line}")
    for field in _EXF_FIELDS:
        table[("data.exf", "EXF_NML_02", f"{_exf_prefix(field)}starttime")] = _Unset(
            rl, "pkg/exf/exf_getffield_start.F:63-64")
    for side in "nsew":
        for prefix in ("obcs", "siob"):
            table[("data.exf", "EXF_NML_OBCS", f"{prefix}{side}starttime")] = _Unset(
                rl, "pkg/exf/exf_getffield_start.F:63-64")
        for suffix, unset in (("startdate1", i), ("startdate2", i), ("period", rl),
                              ("repcycle", rl)):
            table[("data.exf", "EXF_NML_OBCS", f"siob{side}{suffix}")] = _Unset(
                unset, "pkg/exf/exf_readparms.F:1005-1020")
    for index_key, single_key in (("ob_jnorth", "ob_singlejnorth"), ("ob_jsouth", "ob_singlejsouth"),
                                  ("ob_ieast", "ob_singleieast"), ("ob_iwest", "ob_singleiwest")):
        table[("data.obcs", "OBCS_PARM01", index_key)] = _Unset(
            0, "pkg/obcs/obcs_readparms.F:238,244-251,604-665", True)
        table[("data.obcs", "OBCS_PARM01", single_key)] = _Unset(
            0, "pkg/obcs/obcs_readparms.F:238,252-255,604-665")
    return table


#: (namelist file, group, key) -> MITgcm's unset value.  An entry that writes it is no entry for
#: MITgcm, so the namelists the rules read drop it (_mit_namelist) and its row is a skip
#: (_translate_namelist): a rule's derived default applies whether the key is absent or unset.
_MIT_UNSET = _unset_table()


def _mit_unset(file: str, group: str, key: str, value: Any) -> bool:
    """Whether ``value`` of namelist entry ``key`` in ``group`` of ``file`` is MITgcm's unset
    value (_MIT_UNSET): a number equal to it, or an array whose every entry is."""
    entry = _MIT_UNSET.get((file, group.upper(), _lookup_key(key)))
    if entry is None or value is None:
        return False
    values = value if isinstance(value, list) else [value]
    return all(isinstance(item, (int, float)) and not isinstance(item, bool)
               and float(item) == float(entry.value) for item in values)


def _mit_namelist(path: Path) -> dict[str, dict[str, Any]]:
    """The namelist ``path`` as the rules read it: groups upper-cased, keys lower-cased, and the
    entries holding MITgcm's unset value dropped (:func:`_mit_unset`)."""
    return {
        group: {key: value for key, value in entries.items()
                if not _mit_unset(path.name, group, key, value)}
        for group, entries in _upper_groups(parse_namelist(path)).items()
    }


def _upper_groups(parsed: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        group.upper(): {key.lower(): value for key, value in entries.items()}
        for group, entries in parsed.items()
    }
