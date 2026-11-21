"""EXF and surface forcing: the shared mapping table of input/data.exf and the forcing wiring."""

from __future__ import annotations

from dataclasses import field, fields
from pathlib import Path
from typing import Any
from experiments.mitcases.code_inventory import MAP
from locomgcm.dynamics.bulk_formulae import CEN2KEL, EXF_BULK_DEFAULTS
from locomgcm.dynamics.era5_wind_forcing import WINDSTRESSMAX_EXF_DEFAULT
from locomgcm.io.runner._canonical_helpers import (
    ASSEMBLY_SOURCE_TYPES,
    _EXF_BULK_FIELDS,
    _EXF_FLUX_FIELDS,
)
from locomgcm.io.exf_source import EXF_OPTIONS_UNDEFINED
from locomgcm.io.runner.layer_b import DATA_SOURCES

from .core import (
    Gap,
    Record,
    Skipped,
    _ASSEMBLY,
    _Assembly,
    _CLIMATOLOGY_SOURCE_NAME,
    _CLIMATOLOGY_SOURCE_TYPE,
    _EXF_SOURCE_NAME,
    _FORCING_SOURCE_NAME,
    _FORCING_SOURCE_TYPE,
    _MISSING,
    _entry,
    _record_gap,
)
from .rules import (
    Ctx,
    Rule,
    _as_list,
    _gap,
    _skip,
)
from .namelist import (
    _EXF_FIELDS,
    _exf_prefix,
)
from .cpp import (
    _exf_macros,
)
from .convert import (
    _mit_start_clock,
    _staged_namelist_value,
)

# --------------------------------------------------------------------------
# THE SHARED MAPPING TABLE -- ``input/data.exf`` (pkg/exf)
# --------------------------------------------------------------------------
#
# LOCOMgcm carries the EXF Large-Pond kernel (locomgcm/dynamics/bulk_formulae.py
# exf_bulk_fluxes: exf_bulkformulae.F, the radiation of exf_radiation.F and the
# net assembly of exf_getforcing.F) and the EXF stress mapping
# (era5_wind_forcing), but its only atmospheric-state source is the ERA5
# NetCDF archive (locomgcm/io/runner/_canonical_helpers.py build_bulk_source): no
# source reads EXF binary field files on EXF's record clock.  So ``useEXF`` is
# unsupported (PACKAGE_RULES), and every key of the data.exf namelists
# (pkg/exf/exf_readparms.F:67-281) has a row here: an entry that restates the
# MIT default, or a value LOCOMgcm's kernel fixes, is a skip; any other value is
# unsupported and names the EXF capability LOCOMgcm lacks.  A field attribute
# (record clock, input grid) is read only for a field that has a file
# (exf_set_fld.F:118-121, exf_init_fld.F:98), so it is a skip for a field
# without one.

_EXF_FILE_GAP = (
    "EXF field file the LOCOMgcm EXF sources do not read: mit_exf reads atemp, aqh, "
    "uwind, vwind, swdown, lwdown, precip, snowprecip and runoff, mit_exf_flux "
    "hflux, sflux, ustress, vstress and swflux (locomgcm/io/runner/_canonical_helpers.py "
    "_EXF_BULK_FIELDS, _EXF_FLUX_FIELDS)"
)
_EXF_CLOCK_GAP = (
    "record clock of an EXF field file the LOCOMgcm EXF sources do not read "
    "(exf_getffieldrec.F:80-266); see that field's file row"
)
_EXF_INTERP_GAP = (
    "EXF interpolation from an input lat-lon grid (USE_EXF_INTERPOLATION, "
    "exf_readparms.F:989-995, exf_interp.F); the EXF sources read model-grid "
    "records and refuse the macro (locomgcm/io/exf_source.py EXF_OPTIONS_UNDEFINED)"
)
_EXF_STRESS_GAP = (
    "EXF stress files on the C grid or rotated from the A grid (exf_readparms.F:"
    "1045-1053, exf_mapfields.F:216-284); mit_exf_flux reads cell-centred, "
    "unrotated stress (_canonical_helpers.exf_flux_sources)"
)
_EXF_OBCS_GAP = (
    "EXF record clock of the open-boundary files (obcs_exf_load.F, "
    "exf_set_obcs.F); LOCOMgcm's mit_obcs_binary has no EXF clock"
)
_EXF_YEARLY_GAP = (
    "yearly EXF files with calendar-year records (exf_getyearlyfieldname.F, "
    "exf_getffieldrec.F:151-191); the EXF sources read one file per field"
)
_EXF_RELAX_GAP = (
    "surface relaxation to the EXF climatology (exf_readparms.F:1066-1091 sets "
    "tauThetaClimRelax/tauSaltClimRelax from it): EXF samples climsst/climsss on "
    "its own record clock, floors climsst at climtempfreeze (exf_getclim.F:88-99) "
    "and scales by exf_outscal_sst/sss (exf_mapfields.F:298-309), while "
    "surface_climate_relaxation's mit_surface_climatology source carries the core "
    "forcing clock only (locomgcm/io/forcing_records.py SurfaceClimatologySource)"
)
_EXF_ZENITH_GAP = (
    "zenith-angle albedo and incoming radiation (ALLOW_ZENITHANGLE, "
    "exf_zenithangle.F); the bulk kernel uses the constant exf_albedo and the "
    "EXF sources refuse the macro (locomgcm/io/exf_source.py EXF_OPTIONS_UNDEFINED)"
)
_EXF_ICE_GAP = (
    "sea-ice and snow coefficient of EXF, read by pkg/seaice and pkg/thsice and "
    "by the snow-melt heat term (exf_mapfields.F:144-170); no LOCOMgcm ocean-model "
    "counterpart"
)
_EXF_SCALE_GAP = (
    "scaling or offset of an EXF field the LOCOMgcm EXF sources do not read "
    "(exf_set_fld.F:303-310, exf_mapfields.F:100-130)"
)
_EXF_TREND_GAP = (
    "trend removed from an EXF field the LOCOMgcm EXF sources do not read "
    "(exf_set_fld.F:307-310, exf_mapfields.F:105-130)"
)
_EXF_CONST_GAP = (
    "constant of an EXF field the LOCOMgcm EXF sources do not read "
    "(exf_init_fld.F:85-94)"
)

#: data.exf names of the bulk constants mit_exf carries: the fields of
#: ExfBulkConstants, whose EXF_BULK_DEFAULTS are the exf_readparms.F:329-370,603
#: defaults; hq is not a namelist entry but ht (exf_readparms.F:1044).
_EXF_CONSTANT_NAMES = {item.name.lower(): item.name for item in fields(EXF_BULK_DEFAULTS)}

#: Fields the LOCOMgcm EXF sources read (bulk and flux path).
_EXF_SOURCE_FIELDS = (*_EXF_BULK_FIELDS, *_EXF_FLUX_FIELDS)

#: Sea-ice and snow parameters of EXF_NML_01 with their MIT defaults.
_EXF_ICE_DEFAULTS = {
    "usestabilityfct_overice": (False, "320"),
    "flami": (334000.0, "334"),
    "cvapor_fac_ice": (11637800.0, "337"),
    "cvapor_exp_ice": (5897.8, "338"),
    "exf_icecd": (1.63e-3, "364"),
    "exf_icece": (1.63e-3, "365"),
    "exf_icech": (1.63e-3, "366"),
    "ice_emissivity": (0.95, "371"),
    "snow_emissivity": (0.95, "372"),
}


def _exf_has_file(ctx: Ctx, field: str) -> bool:
    files = ctx.namelists.get("data.exf", {}).get("EXF_NML_02", {})
    return bool(str(files.get(f"{_exf_prefix(field)}file", "")).strip())


def _exf_file(field: str) -> Rule:
    """An EXF field file: blank is MITgcm's "no file", any name is unread by LOCOMgcm."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not str(value).strip():
            raise Skipped(
                f"blank {field} file, the MIT default (exf_readparms.F:609-636): "
                "EXF reads no file for it"
            )
        raise Gap(_EXF_FILE_GAP)

    return Rule(MAP, target=None, convert="blank -> no file, else capability gap", fn=fn)


def _exf_attribute(
    fields: tuple[str, ...], what: str, gap: str, default: Any = _MISSING, line: str = ""
) -> Rule:
    """A record-clock or input-grid attribute of the files of ``fields``.

    ``default`` is the MIT default an explicit entry may restate; ``_MISSING``
    marks an attribute whose every explicit value matters (UNSET defaults and
    defaults derived from other parameters).
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not any(_exf_has_file(ctx, field) for field in fields):
            raise Skipped(
                f"{what} of {'/'.join(fields)}, which has no file: EXF never "
                "reads it (exf_set_fld.F:118-121)"
            )
        if default is not _MISSING and value == default:
            raise Skipped(f"{what} restates the MIT default {default!r} (exf_readparms.F:{line})")
        raise Gap(gap)

    return Rule(
        MAP, target=None, convert="no file or MIT default -> no-op, else capability gap", fn=fn
    )


def _exf_default(default: Any, line: str, gap: str) -> Rule:
    """A no-op at its MIT default, a capability gap otherwise."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if value == default:
            raise Skipped(f"restates the MIT default {default!r} (exf_readparms.F:{line})")
        raise Gap(gap)

    return Rule(MAP, target=None, convert="MIT default -> no-op, else capability gap", fn=fn)


def _exf_stage(key: str, convert: str, *, line: str) -> Rule:
    """A data.exf value the EXF source config carries as read (staged under ``exf.<key>``)."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del ctx
        return {f"{_ASSEMBLY}.exf.{key}": value}

    return Rule(MAP, target=f"sources.{_EXF_SOURCE_NAME} <- {key} (exf_readparms.F:{line})",
                convert=convert, fn=fn)


def _exf_field_stage(field: str, attr: str) -> Rule:
    """An attribute of a field the EXF sources read: staged as ``exf_field.<field>.<attr>``.

    A blank file name is EXF's "no file" (exf_set_fld.F:118-121): the field
    keeps its constant.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del ctx
        if attr == "file":
            if not str(value).strip():
                raise Skipped(f"blank {field} file: EXF holds {field}const (exf_init_fld.F:85-94)")
            value = str(value).strip()
        return {f"{_ASSEMBLY}.exf_field.{field}.{attr}": value}

    return Rule(MAP, target=f"sources.{_EXF_SOURCE_NAME}.fields.{field}.{attr}",
                convert="as read; unset entries take the exf_readparms.F defaults in _wire_exf",
                fn=fn)


def _exf_bulk_build(ctx: Ctx) -> bool:
    """The build's EXF path, as :func:`_wire_exf` selects it: ALLOW_ATM_TEMP -> mit_exf."""
    return "ALLOW_ATM_TEMP" in _exf_macros(ctx.case_dir)


def _exf_flux_only(stage: Rule, default: float, line: str, gap: str) -> Rule:
    """``stage`` on the direct-flux path (mit_exf_flux carries the value).

    On the bulk path MITgcm still applies the value to the fluxes the bulk
    formulae produce, while mit_exf carries none: the default is a no-op, any
    other value a capability gap.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not _exf_bulk_build(ctx):
            return stage.fn(value, ctx)
        if float(value) == default:
            raise Skipped(f"bulk path: restates the MIT default {default!r} "
                          f"(exf_readparms.F:{line}), which mit_exf leaves out")
        raise Gap(gap)

    return Rule(MAP, target=stage.target,
                convert=f"{stage.convert}; bulk path: MIT default -> no-op, else capability gap",
                fn=fn)


def _f_exf_runoff_const(value: Any, ctx: Ctx) -> dict[str, Any]:
    """runoffconst: mit_exf reads it; mit_exf_flux has no runoff.

    Under ALLOW_RUNOFF EXF fills runoff from it (exf_init_varia.F:357-366) and
    subtracts it from sflux on both paths (exf_getforcing.F:312-314).
    """
    if _exf_bulk_build(ctx):
        return _exf_field_stage("runoff", "const").fn(value, ctx)
    if "ALLOW_RUNOFF" not in _exf_macros(ctx.case_dir):
        raise Skipped("direct-flux path without ALLOW_RUNOFF: EXF neither fills nor subtracts "
                      "runoff (exf_init_varia.F:357-366, exf_getforcing.F:312-314)")
    if float(value) == 0.0:
        raise Skipped("direct-flux path: restates the MIT default 0.0 (exf_readparms.F:441)")
    raise Gap(
        "runoffconst on the direct-flux path: MITgcm subtracts runoff from sflux under "
        "ALLOW_RUNOFF (exf_getforcing.F:312-314); mit_exf_flux carries no runoff"
    )


def _exf_clim_relax_on(ctx: Ctx, field: str) -> bool:
    entries = ctx.namelists.get("data.exf", {}).get("EXF_NML_02", {})
    return float(entries.get(f"{field}taurelax", 0.0) or 0.0) > 0.0


def _exf_clim(field: str, what: str) -> Rule:
    """A climsst/climsss entry, read only by the EXF restoring towards that field.

    ``<field>TauRelax`` decides: zero leaves the restoring time scale at zero
    and no tendency reads the field (exf_readparms.F:1066-1091,
    set_parms.F:250-253); positive is the unsupported EXF restoring.  The
    field's other entries follow that row.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del value
        on = _exf_clim_relax_on(ctx, field)
        if what == "restoring time scale" and on:
            raise Gap(_EXF_RELAX_GAP)
        raise Skipped(
            f"{what} of {field}, read only by the EXF restoring towards it; "
            f"{field}TauRelax {'is the unsupported row' if on else 'is zero, so no tendency reads it'}"
            " (exf_readparms.F:1066-1091, set_parms.F:250-253)"
        )

    return Rule(MAP, target=None, convert="decided on <field>TauRelax", fn=fn)


def _f_exf_use_atm_wind(value: Any, ctx: Ctx) -> dict[str, Any]:
    if value is True:
        return {f"{_ASSEMBLY}.exf.use_atm_wind": True}
    raise Gap(
        "useAtmWind=.FALSE.: stress read from EXF files beside the bulk heat fluxes "
        "(exf_bulkformulae.F:230-237); era5_bulk_forcing has the branch "
        "(use_atm_wind=False) but mit_exf carries no stress files"
    )


def _exf_constant_field(field: str) -> Rule:
    """``<field>const``: 0 by default, atemp celsius2K, apressure surf_pRef (exf_readparms.F:378-571)."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        default = 0.0
        if field == "atemp":
            default = float(ctx.data("PARM01", "celsius2k", 273.15))  # set_defaults.F:276
        elif field == "apressure":
            default = float(ctx.data("PARM01", "surf_pref", 101325.0))  # set_defaults.F:103
        if float(value) == default:
            raise Skipped(f"restates the MIT default {default!r} (exf_readparms.F:378-571)")
        raise Gap(_EXF_CONST_GAP)

    return Rule(MAP, target=None, convert="MIT default -> no-op, else capability gap", fn=fn)


def _exf_obcs(default: Any = _MISSING, line: str = "") -> Rule:
    """An EXF_NML_OBCS entry: read only when useOBCS is on."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if ctx.namelists.get("data.pkg", {}).get("PACKAGES", {}).get("useobcs") is not True:
            raise Skipped(
                "EXF reads EXF_NML_OBCS only when useOBCS is on (exf_readparms.F:997-1004)"
            )
        if default is not _MISSING and value == default:
            raise Skipped(f"restates the MIT default {default!r} (exf_readparms.F:{line})")
        raise Gap(_EXF_OBCS_GAP)

    return Rule(MAP, target=None, convert="useOBCS off or MIT default -> no-op, else capability gap", fn=fn)


EXF_RULES: dict[tuple[str, str], Rule] = {
    ("EXF_NML_01", "exf_verbose"): _skip("EXF message verbosity"),
    ("EXF_NML_01", "exf_debuglev"): _skip(
        "EXF message level; from debLevC on the range check runs every step "
        "(exf_getforcing.F:346-348), and it only stops the run"
    ),
    ("EXF_NML_01", "exf_monfreq"): _skip("EXF monitor output cadence"),
    ("EXF_NML_01", "exf_adjmonfreq"): _skip("EXF adjoint monitor output cadence"),
    ("EXF_NML_01", "exf_adjmonselect"): _skip("EXF adjoint monitor output selector"),
    ("EXF_NML_01", "diags_opoceweighted"): _skip("weighting of EXF diagnostics output"),
    ("EXF_NML_01", "useexfcheckrange"): _skip(
        "range check that stops the run on out-of-range input and changes no "
        "field (exf_getforcing.F:346-348, exf_check_range.F:216)"
    ),
    ("EXF_NML_01", "exf_yftype"): _skip(
        "retired; any value but 'RL' stops the run (exf_readparms.F:1037-1039)"
    ),
    ("EXF_NML_01", "repeatperiod"): _exf_stage(
        "repeat_period", "s; the default rep_cycle of every field", line="600"
    ),
    ("EXF_NML_01", "exf_iprec"): _exf_stage(
        "exf_iprec", "bits of the big-endian field files", line="676"
    ),
    ("EXF_NML_01", "exf_iprec_obcs"): _exf_obcs(),
    ("EXF_NML_01", "useexfyearlyfields"): _exf_default(False, "679", _EXF_YEARLY_GAP),
    ("EXF_NML_01", "twodigityear"): _exf_default(False, "680", _EXF_YEARLY_GAP),
    ("EXF_NML_01", "readstressonagrid"): _exf_attribute(
        ("ustress", "vstress"), "A-grid stress switch", _EXF_STRESS_GAP, False, "310"
    ),
    ("EXF_NML_01", "rotatestressonagrid"): _exf_attribute(
        ("ustress", "vstress"), "A-grid stress rotation", _EXF_STRESS_GAP, False, "311"
    ),
    ("EXF_NML_01", "readstressoncgrid"): _exf_attribute(
        ("ustress", "vstress"), "C-grid stress switch", _EXF_STRESS_GAP, False, "312"
    ),
    ("EXF_NML_01", "useatmwind"): Rule(
        MAP, target="model.dynamics[era5_bulk_forcing].use_atm_wind",
        convert=".TRUE. -> stress from the wind, else capability gap",
        fn=_f_exf_use_atm_wind,
    ),
    ("EXF_NML_01", "userelativewind"): _exf_default(
        False, "318",
        "wind relative to the ocean surface velocity (exf_wind.F); LOCOMgcm's "
        "kernel takes the absolute wind",
    ),
    ("EXF_NML_01", "nonegativeevap"): _exf_stage(
        "no_negative_evap", "-> era5_bulk_forcing.no_negative_evap", line="319"
    ),
    ("EXF_NML_01", "windstressmax"): _exf_stage(
        "windstressmax", "N m-2 -> era5_wind_forcing.windstressmax (exf_mapfields.F:216-284)",
        line="601",
    ),
    ("EXF_NML_01", "sstextrapol"): _exf_default(
        0.0, "342",
        "SST extrapolated from the two top levels (exf_getforcing.F:174-177); "
        "LOCOMgcm's bulk takes the top-level temperature "
        "(era5_bulk_forcing.py compute_surface_fluxes)",
    ),
    ("EXF_NML_01", "climtempfreeze"): _exf_clim("climsst", "floor"),
    ("EXF_NML_01", "select_zenalbedo"): _exf_default(0, "308", _EXF_ZENITH_GAP),
    ("EXF_NML_01", "useexfzenincoming"): _exf_default(False, "309", _EXF_ZENITH_GAP),
    ("EXF_NML_02", "runoftempfile"): _exf_file("runoftemp"),
    ("EXF_NML_02", "climssttaurelax"): _exf_clim("climsst", "restoring time scale"),
    ("EXF_NML_02", "climssstaurelax"): _exf_clim("climsss", "restoring time scale"),
    ("EXF_NML_02", "areamasktaurelax"): _exf_default(
        0.0, "536", "sea-ice fraction relaxation (EXF_SEAICE_FRACTION); no LOCOMgcm counterpart"
    ),
    ("EXF_NML_02", "climustrtaurelax"): _exf_default(
        0.0, "560", "stress relaxation (ALLOW_CLIMSTRESS_RELAXATION); no LOCOMgcm counterpart"
    ),
    ("EXF_NML_02", "climvstrtaurelax"): _exf_default(
        0.0, "568", "stress relaxation (ALLOW_CLIMSTRESS_RELAXATION); no LOCOMgcm counterpart"
    ),
    ("EXF_NML_03", "exf_inscal_sst"): _skip(
        "read by no EXF routine; its default is commented out (exf_readparms.F:694-695)"
    ),
    ("EXF_NML_03", "exf_inscal_sss"): _skip(
        "read by no EXF routine; its default is commented out (exf_readparms.F:694-695)"
    ),
    ("EXF_NML_03", "exf_offset_atemp"): _exf_stage(
        "exf_offset_atemp", "K added to atemp (exf_set_fld.F:303-306)", line="697"
    ),
    ("EXF_NML_04", "exf_output_interp"): _skip("writes the interpolated fields to files"),
    ("EXF_NML_OBCS", "useobcsyearlyfields"): _exf_obcs(False, "573"),
}

for _key in ("cdrag_8", "cdragmax", "umax"):
    EXF_RULES[("EXF_NML_01", _key)] = _gap(
        "Large and Yeager (2009) drag (ALLOW_DRAG_LARGEYEAGER09, "
        "exf_readparms.F:346-354); LOCOMgcm's kernel has no such branch"
    )
for _key, _name in _EXF_CONSTANT_NAMES.items():
    EXF_RULES[("EXF_NML_01", _key)] = _exf_stage(
        f"constants.{_name}", "-> mit_exf constants, the era5_bulk_forcing kernel's "
        "ExfBulkConstants (_canonical_helpers.bulk_kernel_constants)", line="329-370,603",
    )
for _key, (_default, _line) in _EXF_ICE_DEFAULTS.items():
    EXF_RULES[("EXF_NML_01", _key)] = _exf_default(_default, _line, _EXF_ICE_GAP)
for _field in _EXF_FIELDS:
    _prefix = _exf_prefix(_field)
    _suffixes = ("file", "startdate1", "startdate2", "period", "starttime", "repcycle")
    if _field in _EXF_SOURCE_FIELDS:
        for _suffix in _suffixes:
            EXF_RULES[("EXF_NML_02", f"{_prefix}{_suffix}")] = _exf_field_stage(_field, _suffix)
    elif _field in ("climsst", "climsss"):
        for _suffix in _suffixes:
            EXF_RULES[("EXF_NML_02", f"{_prefix}{_suffix}")] = _exf_clim(_field, _suffix)
    else:
        EXF_RULES[("EXF_NML_02", f"{_prefix}file")] = _exf_file(_field)
        for _suffix, _default in (
            ("startdate1", 0), ("startdate2", 0), ("period", 0.0),
            ("starttime", _MISSING), ("repcycle", _MISSING),
        ):
            EXF_RULES[("EXF_NML_02", f"{_prefix}{_suffix}")] = _exf_attribute(
                (_field,), _suffix, _EXF_CLOCK_GAP, _default, "374-673"
            )
    for _suffix in ("lon0", "lon_inc", "lat0", "lat_inc", "nlon", "nlat", "interpmethod"):
        EXF_RULES[("EXF_NML_04", f"{_field}_{_suffix}")] = _exf_attribute(
            (_field,), "input grid", _EXF_INTERP_GAP
        )
for _field in (*_EXF_FIELDS, "runoftemp"):
    if _field in _EXF_SOURCE_FIELDS:
        EXF_RULES[("EXF_NML_03", f"{_field}const")] = (
            Rule(MAP, target=f"sources.{_EXF_SOURCE_NAME}.fields.runoff.const (mit_exf)",
                 convert="bulk path: as read; direct-flux path: no ALLOW_RUNOFF or 0 -> no-op, "
                         "else capability gap",
                 fn=_f_exf_runoff_const)
            if _field == "runoff" else _exf_field_stage(_field, "const"))
        for _key, _line in (("intercept", "379,414"), ("slope", "380,415")):
            _trend = _exf_field_stage(_field, f"remove_{_key}")
            if _field in ("hflux", "sflux"):
                # Taken off whenever the field has no file (exf_mapfields.F:105-113 Qnet,
                # :122-130 EmPmR), so also off the bulk-formula flux.
                _trend = _exf_flux_only(
                    _trend, 0.0, _line,
                    f"{_field}_exfremo_{_key} on the bulk path: MITgcm takes the trend off the "
                    "bulk-formula flux (exf_mapfields.F:105-113, :122-130); mit_exf has no such "
                    "term")
            EXF_RULES[("EXF_NML_03", f"{_field}_exfremo_{_key}")] = _trend
        continue
    if _field in ("climsst", "climsss"):
        for _key in (f"{_field}const", f"{_field}_exfremo_intercept", f"{_field}_exfremo_slope"):
            EXF_RULES[("EXF_NML_03", _key)] = _exf_clim(_field, _key)
        continue
    EXF_RULES[("EXF_NML_03", f"{_exf_prefix(_field)}const")] = _exf_constant_field(_field)
    EXF_RULES[("EXF_NML_03", f"{_field}_exfremo_intercept")] = _exf_default(
        0.0, "379-571", _EXF_TREND_GAP
    )
    EXF_RULES[("EXF_NML_03", f"{_field}_exfremo_slope")] = _exf_default(
        0.0, "379-571", _EXF_TREND_GAP
    )
for _field in (
    "hflux", "sflux", "evap", "ustress", "vstress", "uwind", "vwind", "wspeed",
    "atemp", "aqh", "hs", "hl", "swflux", "lwflux", "precip", "runoff",
    "apressure", "snowprecip", "runoftemp", "saltflx", "swdown", "lwdown",
    "climsst", "climsss", "climustr", "climvstr", "tidepot", "areamask",
):
    if _field in _EXF_SOURCE_FIELDS:
        EXF_RULES[("EXF_NML_03", f"exf_inscal_{_field}")] = _exf_field_stage(_field, "inscale")
    elif _field in ("climsst", "climsss"):
        EXF_RULES[("EXF_NML_03", f"exf_inscal_{_field}")] = _exf_clim(_field, "input scale")
    else:
        EXF_RULES[("EXF_NML_03", f"exf_inscal_{_field}")] = _exf_default(
            1.0, "683-713", _EXF_SCALE_GAP)
for _field in (
    "hflux", "sflux", "ustress", "vstress", "swflux", "sst", "sss", "apressure",
    "tidepot", "areamask",
):
    if _field in ("hflux", "sflux", "swflux", "ustress", "vstress"):
        EXF_RULES[("EXF_NML_03", f"exf_outscal_{_field}")] = _exf_flux_only(
            _exf_stage(f"outscal.{_field}",
                       "-> mit_exf_flux exf_outscal_<field> (exf_mapfields.F:100-284)",
                       line="716-725"),
            1.0, "716-725",
            f"exf_outscal_{_field} on the bulk path: MITgcm scales the bulk-formula "
            f"{_field} too (exf_mapfields.F:102-127, 237-287, 293); mit_exf carries no output "
            "scale")
    elif _field in ("sst", "sss"):
        EXF_RULES[("EXF_NML_03", f"exf_outscal_{_field}")] = _exf_clim(
            f"clim{_field}", "output scale")
    else:
        EXF_RULES[("EXF_NML_03", f"exf_outscal_{_field}")] = _exf_default(
            1.0, "716-725", _EXF_SCALE_GAP)
for _side in "nsew":
    EXF_RULES[("EXF_NML_OBCS", f"obcs{_side}startdate1")] = _exf_obcs(0, "574-585")
    EXF_RULES[("EXF_NML_OBCS", f"obcs{_side}startdate2")] = _exf_obcs(0, "574-585")
    EXF_RULES[("EXF_NML_OBCS", f"obcs{_side}period")] = _exf_obcs(0.0, "574-585")
    EXF_RULES[("EXF_NML_OBCS", f"obcs{_side}starttime")] = _exf_obcs()
    EXF_RULES[("EXF_NML_OBCS", f"obcs{_side}repcycle")] = _exf_obcs()
    for _suffix in ("startdate1", "startdate2", "starttime", "period", "repcycle"):
        EXF_RULES[("EXF_NML_OBCS", f"siob{_side}{_suffix}")] = _exf_obcs()




def _mit_convert_fw2salt(ctx: Ctx) -> float:
    """Resolve MIT's explicit value or conditional default without approximation."""
    explicit = ctx.data("PARM01", "convertfw2salt", _MISSING)
    if explicit is not _MISSING and explicit is not None:
        return float(explicit)
    real_freshwater = bool(ctx.data("PARM01", "userealfreshwaterflux", False))
    select_add_fluid = int(ctx.data("PARM01", "selectaddfluid", 0) or 0)
    return -1.0 if real_freshwater or select_add_fluid >= 1 else 35.0


def _resolved_convert_fw2salt(asm: _Assembly, ctx: Ctx) -> float:
    """Apply MIT's conditional default after all PARM01 values are available.

    references/MITgcm/model/src/ini_parms.F:648-652
    """
    explicit = asm.staged("convert_fw2salt")
    if explicit is not None:
        return float(explicit)
    mit_resolved = _mit_convert_fw2salt(ctx)
    resolved = mit_resolved
    asm.defaults_used["convert_fw2salt"] = (
        f"MIT {mit_resolved!r} -> LOCOMgcm {resolved!r} "
        "(model/src/ini_parms.F:648-652; conditional on "
        "useRealFreshWaterFlux/selectAddFluid)"
    )
    return resolved


def _trac_forcing_out_ab(asm: _Assembly) -> int:
    """tracForcingOutAB, or its forcing_In_AB default (ini_parms.F:1099-1101)."""
    trac_forcing_out_ab = asm.staged("trac_forcing_out_ab")
    if trac_forcing_out_ab is None:
        trac_forcing_out_ab = int(not asm.flag("forcing_in_ab"))
        asm.defaults_used["trac_forcing_out_ab"] = (
            f"{trac_forcing_out_ab} (model/src/ini_parms.F:1099-1101; "
            "conditional on forcing_In_AB)"
        )
    return trac_forcing_out_ab


def _unregistered_source(record: Record, source_type: str, target: str) -> bool:
    """Record a gap when the run reads no ``source_type`` source.

    Stage 1a instantiates the Layer B sources (DATA_SOURCES) by plugin name and
    leaves the types the assembly reads itself, ASSEMBLY_SOURCE_TYPES, to the
    assembly (locomgcm/io/runner/stages/_common.py build_sources); a type in
    neither declaration stops the run at preparation.
    """
    if source_type in DATA_SOURCES.list_names() or source_type in ASSEMBLY_SOURCE_TYPES:
        return False
    _record_gap(
        record,
        {"source": f"derived:{target}", "parameter": f"<derived:{source_type} source>",
         "value": None},
        f"source type {source_type!r} is neither a Layer B plugin "
        "(locomgcm/io/runner/layer_b DATA_SOURCES) nor read by the assembly "
        "(locomgcm/io/runner/_canonical_helpers.py ASSEMBLY_SOURCE_TYPES); stage 1a "
        "stops on it (locomgcm/io/runner/stages/_common.py build_sources)",
    )
    return True


def _wire_surface_climatology(
    yaml_tree: dict[str, Any],
    case_dir: Path,
    ctx: Ctx,
    asm: _Assembly,
    record: Record,
    *,
    temp_stepping: bool,
    salt_stepping: bool,
) -> dict[str, Any] | None:
    """Wire core surface T/S restoring onto surface_climate_relaxation.

    The staged time scales exist only in the verbatim mode and only when
    positive (:func:`_clim_relax_tau`).  A tracer MITgcm does not step gets
    tau 0, since its restoring term never reaches a tendency.  The SST/SSS
    records follow the core forcing clock (externForcingPeriod/Cycle under
    periodicExternalForcing, else record 1; external_fields_load.F:63-72,
    249-264, ini_forcing.F:100-105), which the source reproduces with
    GET_PERIODIC_INTERVAL at myTime = startTime + t.  latBandClimRelax unset
    is 180 on a spherical or curvilinear grid (set_parms.F:265-270, run before
    INI_GRID) and 3*sum(delY) on a Cartesian one (load_grid_spacing.F:166-175).
    """
    tau_theta = asm.staged("tau_theta_clim_relax") if temp_stepping else None
    tau_salt = asm.staged("tau_salt_clim_relax") if salt_stepping else None
    if tau_theta is None and tau_salt is None:
        return None
    lat_band = asm.staged("lat_band_clim_relax")
    if lat_band is None:
        if ctx.grid_family in ("spherical_polar", "curvilinear"):
            lat_band = 180.0
        else:
            spacing = ctx.data("PARM04", "dely")
            if spacing is None:
                spacing = ctx.data("PARM04", "dyspacing")
            if spacing is None:
                _record_gap(
                    record,
                    {"source": "input/data:PARM03.latBandClimRelax",
                     "parameter": "<derived:latBandClimRelax default>", "value": None},
                    "latBandClimRelax default 3*sum(delY) needs inline delY "
                    "(load_grid_spacing.F:166-175)",
                )
                return None
            values = _as_list(spacing)
            lat_band = 3.0 * sum(values * ctx.ny if len(values) == 1 else values)
        asm.defaults_used["lat_band_clim_relax"] = f"{lat_band!r} (set_parms.F:265-270, load_grid_spacing.F:166-175)"
    if _unregistered_source(record, _CLIMATOLOGY_SOURCE_TYPE, f"sources.{_CLIMATOLOGY_SOURCE_NAME}"):
        return None
    periodic = asm.flag("periodic_external_forcing")
    source = {
        "type": _CLIMATOLOGY_SOURCE_TYPE,
        "path": str((case_dir / "input").resolve()),
        "read_binary_prec": int(ctx.data("PARM01", "readbinaryprec", 32) or 32),
        "theta_clim_file": asm.staged("theta_clim_file") or "",
        "salt_clim_file": asm.staged("salt_clim_file") or "",
        "periodic_external_forcing": periodic,
        "extern_forcing_period": asm.number("extern_forcing_period"),
        "extern_forcing_cycle": asm.number("extern_forcing_cycle"),
        "start_time": _mit_start_clock(ctx)[0],
    }
    yaml_tree.setdefault("sources", {})[_CLIMATOLOGY_SOURCE_NAME] = source
    kwargs = {
        "tau_theta": float(tau_theta or 0.0),
        "tau_salt": float(tau_salt or 0.0),
        "lat_band": float(lat_band),
        "ab2_eligible": _trac_forcing_out_ab(asm) == 0,
        "restore_under_ice": _restore_under_ice(ctx),
    }
    record.derived.append(
        {
            "target": (
                f"sources.{_CLIMATOLOGY_SOURCE_NAME} + "
                "model.dynamics[surface_climate_relaxation]"
            ),
            "value": {**{k: v for k, v in source.items() if k != "type"}, **kwargs},
            "rationale": (
                "FORCING_SURF_RELAX adds -lambda*(theta(k=1)-SST)*drF(1)*hFacC(1) to "
                "surfaceForcingT (forcing_surf_relax.F:91-104) with lambda = 1/tau "
                "inside |yC| <= latBandClimRelax (ini_forcing.F:46-61), before the "
                "other surface fluxes (external_forcing_surf.F:173-177), applied "
                "with them under tracForcingOutAB (temp_integrate.F:368-416).  "
                "surface_climate_relaxation forms the same flux from SST/SSS its "
                "mit_surface_climatology source samples at step start on the core "
                "clock (external_fields_load.F:63-72,249-264; non-periodic: record "
                "1, ini_forcing.F:100-105)"
            ),
        }
    )
    return _entry("surface_climate_relaxation", **kwargs)


def _restore_under_ice(ctx: Ctx) -> bool:
    """surface_climate_relaxation.restore_under_ice.

    False exactly when FORCING_SURF_RELAX scales the restoring flux by 1-AREA:
    useSEAICE .AND. .NOT.SEAICErestoreUnderIce (forcing_surf_relax.F:75-85),
    SEAICErestoreUnderIce .FALSE. when unset (seaice_readparms.F:264).
    """
    seaice = _staged_namelist_value(ctx, "data.pkg", "PACKAGES", "useseaice", False) is True
    under_ice = _staged_namelist_value(
        ctx, "data.seaice", "SEAICE_PARM01", "seaicerestoreunderice", False) is True
    return not seaice or under_ice


def _surface_tracer_kwargs(asm: _Assembly, ctx: Ctx, reference_density: float) -> dict[str, Any]:
    """Constants of the surface tracer forcing consumers (wpflux, bulk).

    HeatCapacity_Cp, rhoConst, rhoConstFresh (rhoConst when unset,
    ini_parms.F:484), convertFW2Salt, salt_EvPrRn, the shortwave penetration
    and tracForcingOutAB as ab2_eligible.
    """
    heat_capacity = asm.number("heat_capacity")
    convert_fw2salt = _resolved_convert_fw2salt(asm, ctx)
    salt_evprn = asm.number("salt_evprn")
    rho_const_fresh_value = asm.staged("rho_const_fresh")
    if rho_const_fresh_value is None:
        rho_const_fresh = reference_density
        asm.defaults_used["rho_const_fresh"] = (
            f"{rho_const_fresh!r} (model/src/ini_parms.F:484: "
            "rhoConstFresh=rhoConst when unset)"
        )
    else:
        rho_const_fresh = float(rho_const_fresh_value)
    return {
        "ab2_eligible": _trac_forcing_out_ab(asm) == 0,
        "heat_capacity": heat_capacity,
        "rho_const": reference_density,
        "rho_const_fresh": rho_const_fresh,
        "convert_fw2salt": convert_fw2salt,
        "salt_evprn": salt_evprn,
        "penetrating_sw": asm.yaml("model.stepper.kwargs.penetrating_sw"),
    }


#: Macros the bulk kernel's discretisation assumes (exf_bulkformulae.F, exf_radiation.F:52-78,
#: exf_getforcing.F:260-316) and the ones that would replace a computed flux by a read one.
_EXF_BULK_MACROS = (
    "ALLOW_ATM_WIND", "ALLOW_BULKFORMULAE", "ALLOW_DOWNWARD_RADIATION",
    "EXF_LWDOWN_WITH_EMISSIVITY",
)
_EXF_READ_FLUX_MACROS = ("ALLOW_READ_TURBFLUXES", "EXF_READ_EVAP")


def _exf_field_config(ctx: Ctx, asm: _Assembly, field: str, item: dict[str, Any]) -> dict[str, Any]:
    """One ``fields`` entry: the staged data.exf values, else their exf_readparms.F defaults.

    const 0, atemp celsius2K (:378-571, set_defaults.F:276); period and the
    start dates 0, start time UNSET -> 0 (exf_getffield_start.F:62-63);
    rep_cycle repeatPeriod (:942-970); inscale 1 (:683-713); no trend.
    """
    const = float(ctx.data("PARM01", "celsius2k", CEN2KEL)) if field == "atemp" else 0.0
    return {
        "file": str(item.get("file", "")),
        "const": float(item.get("const", const)),
        "period": float(item.get("period", 0.0)),
        "start_time": float(item.get("starttime", 0.0)),
        "startdate1": int(item.get("startdate1", 0)),
        "startdate2": int(item.get("startdate2", 0)),
        "rep_cycle": float(item.get("repcycle", asm.staged("exf", {}).get("repeat_period", 0.0))),
        "inscale": float(item.get("inscale", 1.0)),
        "remove_intercept": float(item.get("remove_intercept", 0.0)),
        "remove_slope": float(item.get("remove_slope", 0.0)),
    }


def _wire_exf(
    yaml_tree: dict[str, Any],
    case_dir: Path,
    ctx: Ctx,
    asm: _Assembly,
    record: Record,
    *,
    reference_density: float,
    mom_stepping: bool,
    temp_stepping: bool,
    salt_stepping: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Wire pkg/exf onto the mit_exf or mit_exf_flux source and its consumers.

    The build's EXF_OPTIONS.h decides the path.  With ALLOW_ATM_TEMP EXF forms
    the turbulent and radiative fluxes and the stress from the atmospheric state
    (exf_getforcing.F:260-316): mit_exf feeds era5_bulk_forcing (the same
    Large-Pond kernel, bulk_formulae.exf_bulk_fluxes) and the cell-centre stress
    to era5_wind_forcing.  Without it EXF reads hflux, sflux and the stress
    (exf_mapfields.F:100-284): mit_exf_flux feeds wpflux_surface_forcing and
    era5_wind_forcing.  exf_options carries the four macros the sources refuse;
    the sources check the constants, the periods and the calendar themselves.
    """
    macros = _exf_macros(case_dir)
    bulk = "ALLOW_ATM_TEMP" in macros
    names = _EXF_BULK_FIELDS if bulk else _EXF_FLUX_FIELDS
    source_type = "mit_exf" if bulk else "mit_exf_flux"
    exf = dict(asm.staged("exf", {}) or {})
    staged = dict(asm.staged("exf_field", {}) or {})
    gaps = []
    if bulk:
        missing = [m for m in _EXF_BULK_MACROS if m not in macros]
        read = [m for m in _EXF_READ_FLUX_MACROS if m in macros]
        if missing or read:
            gaps.append(
                "the bulk kernel (bulk_formulae.exf_bulk_fluxes) is EXF with "
                f"{', '.join(_EXF_BULK_MACROS)} defined and {', '.join(_EXF_READ_FLUX_MACROS)} "
                f"undefined; this build has undefined {missing} and defined {read}"
            )
    elif "ALLOW_ATM_WIND" in macros:
        gaps.append(
            "ALLOW_ATM_WIND without ALLOW_ATM_TEMP: EXF forms the stress from the wind "
            "(exf_wind.F); mit_exf_flux reads the stress files"
        )
    if bulk and "ALLOW_RUNOFF" not in macros and dict(staged.get("runoff", {})).get("file"):
        gaps.append("runoff file without ALLOW_RUNOFF: MITgcm stops (exf_check.F:322-329)")
    unread = sorted(f for f, item in staged.items() if item.get("file") and f not in names)
    if unread:
        gaps.append(f"EXF files {unread} that {source_type} does not read (fields {list(names)})")
    for gap in gaps:
        _record_gap(record, {"source": "code/EXF_OPTIONS.h + input/data.exf",
                             "parameter": "<derived:EXF path>", "value": source_type}, gap)
    if gaps:
        return None, None
    use_cal = bool(asm.staged("use_cal", False))
    config: dict[str, Any] = {
        "path": str((case_dir / "input").resolve()),
        "exf_iprec": int(exf.get("exf_iprec", 32)),
        "use_cal": use_cal,
        "start_time": _mit_start_clock(ctx)[0],
        "exf_options": {m: m in macros for m in EXF_OPTIONS_UNDEFINED},
        "fields": {f: _exf_field_config(ctx, asm, f, dict(staged.get(f, {}))) for f in names},
    }
    if bulk and "ALLOW_RUNOFF" not in macros:
        # EXF neither reads nor adds runoff (exf_init_varia.F:355, exf_getforcing.F:312).
        config["fields"]["runoff"].update(file="", const=0.0)
    if not bulk and "SHORTWAVE_HEATING" not in macros:
        # Without ALLOW_ATM_TEMP and SHORTWAVE_HEATING EXF fills no swflux and writes no
        # Qsw (exf_init_varia.F:277, exf_mapfields.F:289).
        config["fields"]["swflux"].update(file="", const=0.0)
    if use_cal:
        config["calendar"] = str(asm.staged("cal_calendar", " ")).strip()
        config["start_date_1"] = int(asm.staged("cal_start_date_1", 0))
        config["start_date_2"] = int(asm.staged("cal_start_date_2", 0))
    if bulk:
        staged_constants = dict(exf.get("constants", {}))
        constants = {item.name: float(staged_constants.get(
            item.name, getattr(EXF_BULK_DEFAULTS, item.name))) for item in fields(EXF_BULK_DEFAULTS)}
        config["exf_offset_atemp"] = float(exf.get("exf_offset_atemp", 0.0))
        config["constants"] = constants
    else:
        outscal = dict(exf.get("outscal", {}))
        for field in ("hflux", "sflux", "swflux", "ustress", "vstress"):
            config[f"exf_outscal_{field}"] = float(outscal.get(field, 1.0))
    record.derived.append({
        "target": f"sources.{_EXF_SOURCE_NAME} ({source_type})",
        "value": config,
        "rationale": (
            "EXF_OPTIONS.h "
            + ("defines ALLOW_ATM_TEMP: bulk formulae on the atmospheric state "
               "(exf_getforcing.F:260-316)" if bulk else
               "leaves ALLOW_ATM_TEMP undefined: hflux, sflux and the stress are read "
               "(exf_mapfields.F:100-284)")
            + "; every field entry is the data.exf value or its exf_readparms.F default, "
            "records on the EXF clock (exf_getffieldrec.F:80-266) sampled at the "
            "step-start myTime = startTime + t (load_fields_driver.F:214)"
        ),
    })
    if _unregistered_source(record, source_type, f"sources.{_EXF_SOURCE_NAME}"):
        return None, None
    yaml_tree.setdefault("sources", {})[_EXF_SOURCE_NAME] = {"type": source_type, **config}
    wind = None
    if mom_stepping:
        wind = _entry(
            "era5_wind_forcing", stress_location="center", reference_density=reference_density,
            windstressmax=float(exf.get("windstressmax", WINDSTRESSMAX_EXF_DEFAULT)),
        )
    surface = None
    if temp_stepping or salt_stepping:
        kwargs = _surface_tracer_kwargs(asm, ctx, reference_density)
        if bulk:
            surface = _entry("era5_bulk_forcing", use_atm_wind=True,
                             no_negative_evap=bool(exf.get("no_negative_evap", False)), **kwargs)
        else:
            surface = _entry("wpflux_surface_forcing", **kwargs)
    return wind, surface


def _wire_core_surface_forcing(
    yaml_tree: dict[str, Any],
    case_dir: Path,
    ctx: Ctx,
    asm: _Assembly,
    record: Record,
    *,
    reference_density: float,
    mom_stepping: bool,
    temp_stepping: bool,
    salt_stepping: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Wire core wind/Qnet/EmPmR files through one forcing source.

    Returns the wind and tracer ``model.dynamics`` entries.  A declared file is
    enabled only when MITgcm steps its corresponding prognostic field.  The
    source still exposes all active fields as one time axis, matching the core
    model's single ``periodicExternalForcing`` cadence.

    * the file is already at the C-grid U/V point (``model/inc/FFIELDS.h:21`` /
      ``:27``, read verbatim by ``model/src/ini_forcing.F:70-75``), which is the
      ``c_grid_face`` stress location -- averaging it from centres, as the ERA5 /
      ``pkg/exf`` A-grid producer needs, would be a discretisation error;
    * the tendency is ``fu*mass2rUnit*recip_drF*recip_hFacW``
      (``model/src/external_forcing_surf.F:222`` + ``apply_forcing.F:143-147``)
      with ``mass2rUnit = 1/rhoConst`` on z-coordinates (``ini_parms.F:1573``),
      which is exactly what the consumer computes from ``reference_density``;
    * ``Qnet`` and ``EmPmR`` are tracer-centred
      (``FFIELDS.h:29-36,44-48,102,104``).
      ``Qnet>0`` is upward/cooling and is converted at ingest to LOCOMgcm
      ``heat_flux>0`` warming; the WPFlux adapter converts it back for the
      existing MIT-form operator.  ``EmPmRFile`` is a volume flux [m/s], which
      ``ini_forcing.F:83-90`` multiplies once by ``rhoConstFresh``; the existing
      ``surface_forcing_from_bulk`` producer owns that multiplication;
    * ``periodicExternalForcing`` decides whether record 1 is held for the whole
      run or the file is cycled; the period/cycle pair rides on the source so the
      record cadence is read from the namelist, never inferred from file size.
    """
    zonal = asm.staged("zonal_wind_file") if mom_stepping else None
    merid = asm.staged("merid_wind_file") if mom_stepping else None
    qnet = asm.staged("surf_qnet_file") if temp_stepping else None
    empmr = asm.staged("empmr_file") if salt_stepping else None
    # pLoad reaches the state only through dPhiHydX (CALC_PHI_HYD is called from
    # dynamics.F:482, DYNAMICS only under momStepping, forward_step.F:786-792).
    pload = asm.staged("pload_file") if mom_stepping else None
    if not any((zonal, merid, qnet, empmr, pload)):
        return None, None
    prec = int(ctx.data("PARM01", "readbinaryprec", 32) or 32)
    periodic = asm.flag("periodic_external_forcing")
    period = asm.number("extern_forcing_period") if periodic else None
    cycle = asm.number("extern_forcing_cycle") if periodic else None
    yaml_tree.setdefault("sources", {})[_FORCING_SOURCE_NAME] = {
        "type": _FORCING_SOURCE_TYPE,
        "mit_verification_dir": str(case_dir.resolve()),
        "read_binary_prec": prec,
        "zonal_wind_file": zonal,
        "merid_wind_file": merid,
        "surf_qnet_file": qnet,
        "empmr_file": empmr,
        "forcing_period_seconds": period,
        "forcing_cycle_seconds": cycle,
    }
    if pload:
        yaml_tree["sources"][_FORCING_SOURCE_NAME]["pload_file"] = pload
    yaml_tree["case"]["extras"]["forcing_source"] = _FORCING_SOURCE_TYPE
    wind: dict[str, Any] | None = None
    if zonal or merid:
        record.derived.append(
            {
                "target": (
                    f"sources.{_FORCING_SOURCE_NAME} + case.extras.forcing_source + "
                    "model.dynamics[era5_wind_forcing]"
                ),
                "value": {
                    "zonal_wind_file": zonal,
                    "merid_wind_file": merid,
                    "read_binary_prec": prec,
                    "forcing_period_seconds": period,
                    "forcing_cycle_seconds": cycle,
                    "stress_location": "c_grid_face",
                    "reference_density": reference_density,
                },
                "rationale": (
                    "MITgcm reads zonalWindFile/meridWindFile straight into fu/fv, "
                    "which are declared at the Southwest C-grid U/V point "
                    "(model/inc/FFIELDS.h:21,:27) by ini_forcing.F:70-75 -- no "
                    "averaging and no land mask -- and applies them as "
                    "fu*mass2rUnit*recip_drF*recip_hFacW "
                    "(external_forcing_surf.F:222, apply_forcing.F:143-147) with "
                    "mass2rUnit = 1/rhoConst on z-coordinates (ini_parms.F:1573).  "
                    "The module name era5_wind_forcing is historical: it is the "
                    "generic gridded-stress consumer, and stress_location declares "
                    "which staggering the producer used, so the MIT-native face "
                    "field is placed on faces instead of being reconstructed from "
                    "centres.  periodicExternalForcing="
                    f"{periodic} carries the record cadence (externForcingPeriod / "
                    "externForcingCycle) on the source rather than letting the "
                    "reader infer it from the file size"
                ),
            }
        )
        wind = _entry(
            "era5_wind_forcing",
            stress_location="c_grid_face",
            reference_density=reference_density,
        )

    surface: dict[str, Any] | None = None
    if qnet or empmr:
        surface_kwargs = _surface_tracer_kwargs(asm, ctx, reference_density)
        convert_fw2salt = surface_kwargs["convert_fw2salt"]
        record.derived.append(
            {
                "target": (
                    f"sources.{_FORCING_SOURCE_NAME} + case.extras.forcing_source + "
                    "model.dynamics[wpflux_surface_forcing]"
                ),
                "value": {
                    "surf_qnet_file": qnet,
                    "empmr_file": empmr,
                    "read_binary_prec": prec,
                    "forcing_period_seconds": period,
                    "forcing_cycle_seconds": cycle,
                    **surface_kwargs,
                },
                "rationale": (
                    "surfQnetFile and EmPmRFile are tracer-centred XY fields "
                    "(model/inc/FFIELDS.h:29-36,44-48,102,104).  The source maps upward-"
                    "positive Qnet to LOCOMgcm heat_flux=-Qnet; the WPFlux adapter "
                    "then feeds the existing MIT-form heat operator.  EmPmRFile "
                    "remains a volume flux [m/s] until surface_forcing_from_bulk "
                    "multiplies once by rhoConstFresh, matching ini_forcing.F:83-90.  "
                    "HeatCapacity_Cp, rhoConstFresh, convertFW2Salt and "
                    "salt_EvPrRn are carried into the existing consumer.  "
                    "tracForcingOutAB selects the existing ab2_eligible flag: "
                    "0 places forcing before tracer AB and 1 after it "
                    "(temp_integrate.F:368-416, salt_integrate.F:366-414).  "
                    + (
                        "convertFW2Salt=-1 reads the current entering-state "
                        "k=0 salinity field.  "
                        if convert_fw2salt == -1.0
                        else ""
                    )
                    + f"periodicExternalForcing={periodic} supplies cadence from "
                    "externForcingPeriod/externForcingCycle; file length never "
                    "defines cadence"
                ),
            }
        )
        surface = _entry("wpflux_surface_forcing", **surface_kwargs)
    return wind, surface
