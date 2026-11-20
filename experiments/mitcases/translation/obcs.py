"""Open boundaries: the shared mapping table of input/data.obcs and the boundary wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from experiments.mitcases.code_inventory import MAP
from defgcm.boundaries.open import OpenBoundaryValueScheme

from .core import (
    Deferred,
    Gap,
    InvalidParameter,
    Record,
    Skipped,
    _ASSEMBLY,
    _Assembly,
    _OBCS_CORE_FILE_KEYS,
    _OBCS_INDEX_KEYS,
    _OBCS_SIDES,
    _OBCS_SIDE_NAMES,
    _OBCS_SOURCE_NAME,
    _OBCS_SOURCE_TYPE,
    _Redirect,
    _record_gap,
)
from .rules import (
    Ctx,
    Rule,
    _defer,
    _false_or_gap,
    _gap,
    _map,
    _skip,
    _stage,
)
from .namelist import (
    _MIT_UNSET,
    _mit_unset,
)
from .convert import (
    _MIT_DEFAULT_SREF,
    _MIT_DEFAULT_TREF,
    _fill_ref,
    _master_dt,
)

# --------------------------------------------------------------------------
# THE SHARED MAPPING TABLE -- ``input/data.obcs``
# --------------------------------------------------------------------------

OBCS_RULES: dict[tuple[str, str], Rule] = {
    ("OBCS_PARM01", "useobcsprescribe"): _stage(
        "use_obcs_prescribe",
        "MIT useOBCSprescribe -> prescribed_3d boundary condition",
    ),
    ("OBCS_PARM01", "useobcsbalance"): _stage(
        "use_obcs_balance",
        "MIT useOBCSbalance -> prescribed_3d.enforce_obcs_balance",
    ),
    ("OBCS_PARM01", "useobcssponge"): _stage(
        "use_obcs_sponge",
        "MIT useOBCSsponge; true is a declared capability gap by the binary ingest",
    ),
    ("OBCS_PARM01", "obcs_u1_adv_t"): _stage(
        "obcs_u1_adv_t",
        "MIT OBCS_u1_adv_T -> horizontal_tracer_advection.obcs_upwind",
    ),
    ("OBCS_PARM01", "obcs_u1_adv_s"): _stage(
        "obcs_u1_adv_s",
        "MIT OBCS_u1_adv_S -> horizontal_tracer_advection.obcs_upwind",
    ),
    ("OBCS_PARM01", "obcsfixtopo"): _map(
        "boundary.obcs_fix_topo",
        "MIT OBCSfixTopo boolean",
        lambda value, _ctx: {"boundary.obcs_fix_topo": bool(value)},
    ),
    ("OBCS_PARM01", "obcs_uvapplyfac"): _defer(
        "non-default OBCS_uvApplyFac is not represented by this binary ingest"
    ),
    ("OBCS_PARM01", "obcs_monitorfreq"): _skip("MIT OBCS monitor output cadence"),
    ("OBCS_PARM01", "obcs_monselect"): _skip("MIT OBCS monitor output selector"),
    ("OBCS_PARM01", "obcsprintdiags"): _skip("MIT OBCS diagnostic output"),
    ("OBCS_PARM01", "obcs_u1_adv_tr"): _gap(
        "passive-tracer OBC upwind selectors require the PTRACERS channel"
    ),
}

for index_key, side in _OBCS_INDEX_KEYS.items():
    OBCS_RULES[("OBCS_PARM01", index_key)] = _stage(
        f"obcs_indices.{side}",
        "MIT OB_J*/OB_I* table -> boundary.ob_index_mode and boundary.sides",
    )


for index_key, side in _OBCS_INDEX_KEYS.items():
    single_key = "ob_single" + index_key.removeprefix("ob_")
    OBCS_RULES[("OBCS_PARM01", single_key)] = _stage(
        f"obcs_single.{side}",
        "MIT OB_single* -> fills the unset entries of its side's OB_J*/OB_I* table "
        "(_obcs_index_tables, obcs_readparms.F:604-665)",
    )


def _obcs_index_tables(asm: _Assembly, ctx: Ctx) -> dict[str, Any]:
    """The OB_J*/OB_I* tables as OBCS_READPARMS leaves them after OB_single*.

    Every entry starts at OB_indexUnset = 0 (obcs_readparms.F:238, :244-255), so
    a namelist list shorter than the side leaves its tail at 0.  A set
    OB_single<side> fills each entry still unset and stops the run on an entry set
    to another value (:604-665); an unset one (0, _MIT_UNSET) is never staged.  A
    table all at 0 is no table.  The merge reads both keys after the whole namelist,
    so their order in data.obcs does not matter.
    """
    tables = dict(asm.staged("obcs_indices", {}) or {})
    index_keys = {side: key for key, side in _OBCS_INDEX_KEYS.items()}
    for side, single in dict(asm.staged("obcs_single", {}) or {}).items():
        key = index_keys[side]
        length = ctx.nx if side in {"N", "S"} else ctx.ny
        raw = tables.get(side, [])
        values = list(raw) if isinstance(raw, list) else [raw]
        values += [_MIT_UNSET[("data.obcs", "OBCS_PARM01", key)].value] * (length - len(values))
        if any(value != single and not _mit_unset("data.obcs", "OBCS_PARM01", key, value)
               for value in values):
            raise InvalidParameter(
                f"OB_single{'J' if side in {'N', 'S'} else 'I'}{_OBCS_SIDE_NAMES[side]}="
                f"{single} conflicts with an entry of its "
                f"OB index table {raw!r}; MITgcm stops (obcs_readparms.F:604-665)"
            )
        tables[side] = [single] * len(values)
    return tables


for file_key, (side, role) in _OBCS_CORE_FILE_KEYS.items():
    OBCS_RULES[("OBCS_PARM01", file_key)] = _stage(
        f"obcs_files.{side}.{role}",
        "MIT per-side OBC file name -> mit_obcs_binary files mapping",
    )

def _use_orlanski_rule(side: str) -> Rule:
    """useOrlanski<side>: OBCS_CALC replaces the side's u, v, theta and salt by ORLANSKI_<side>
    on the step-start state (obcs_calc.F:97,147,197,231); prescribed_3d's ``orlanski`` value
    scheme (defgcm/boundaries/open.py OPEN_BOUNDARY_VALUE_SCHEMES), wired by _wire_obcs_binary."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del ctx
        if value is not True:
            raise Skipped(f"OBCS_CALC calls no ORLANSKI_{_OBCS_SIDE_NAMES[side].upper()} "
                          "(obcs_calc.F:97,147,197,231); the side keeps its prescribed values")
        return {f"{_ASSEMBLY}.obcs_orlanski.{side}": True}

    return Rule(MAP, target=f"model.boundary_condition.kwargs.value_schemes.{_OBCS_SIDE_NAMES[side]}",
                convert=".TRUE. -> 'orlanski'; .FALSE. -> prescribed", fn=fn)


for side in _OBCS_SIDES:
    token = side.lower()
    OBCS_RULES[("OBCS_PARM01", f"useorlanski{_OBCS_SIDE_NAMES[side]}")] = _use_orlanski_rule(side)
    OBCS_RULES[("OBCS_PARM01", f"usestevens{_OBCS_SIDE_NAMES[side]}")] = (
        _false_or_gap("Stevens open-boundary radiation is not mapped by the binary ingest")
    )
    OBCS_RULES[("OBCS_PARM01", f"ob{token}connectfile")] = _gap(
        "OBCS nested-grid connectivity files are not supported"
    )
    OBCS_RULES[("OBCS_PARM01", f"ob{token}ptrfile")] = _gap(
        "passive-tracer OBC files require the PTRACERS channel"
    )
    for suffix in ("a", "h", "sl", "sn", "uice", "vice"):
        OBCS_RULES[("OBCS_PARM01", f"ob{token}{suffix}file")] = _gap(
            "sea-ice OBC files are outside the ocean binary ingest"
        )

for key in (
    "uselinearsponge",
    "obcssponge_n",
    "obcssponge_s",
    "obcssponge_e",
    "obcssponge_w",
    "obcssponge_uatns",
    "obcssponge_uatew",
    "obcssponge_vatns",
    "obcssponge_vatew",
    "obcssponge_theta",
    "obcssponge_salt",
    "obcs_balancefacn",
    "obcs_balancefacs",
    "obcs_balanceface",
    "obcs_balancefacw",
):
    OBCS_RULES[("OBCS_PARM01", key)] = _defer(
        "OBCS sponge/balance detail is outside the binary ingest"
    )

for key in (
    "urelaxobcsinner",
    "urelaxobcsbound",
    "vrelaxobcsinner",
    "vrelaxobcsbound",
    "spongethickness",
):
    OBCS_RULES[("OBCS_PARM03", key)] = _defer(
        "OBCS sponge is a declared capability gap"
    )


def _obcs_parameter_of(
    switches: tuple[str, ...], idle: str, active: type[_Redirect], reason: str
) -> Rule:
    """A data.obcs parameter read only by the code behind ``switches``.

    No OBCS_PARM01 switch of ``switches`` at .TRUE.: a no-op, ``idle`` cites
    the reader and its guard.  Otherwise ``active`` (:class:`Gap` or
    :class:`Deferred`) with ``reason``.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del value
        parms = ctx.namelists.get("data.obcs", {}).get("OBCS_PARM01", {})
        if not any(parms.get(name) is True for name in switches):
            raise Skipped(idle)
        raise active(reason)

    return Rule(
        MAP,
        target=None,
        convert=f"no-op unless one of {', '.join(switches)} is .TRUE.",
        fn=fn,
    )


#: OBCS_PARM02 key -> (OrlanskiParameters field, MITgcm default; obcs_readparms.F:509-513).
_ORLANSKI_PARAMETERS = {
    "cmax": ("cmax", 0.45),
    "cveltimescale": ("cvel_time_scale", 2000.0),
    "cfix": ("cfix", 0.8),
    "usefixedceast": ("use_fixed_c_east", False),
    "usefixedcwest": ("use_fixed_c_west", False),
}


def _orlanski_parameter_rule(key: str) -> Rule:
    """An OBCS_PARM02 key: read only with a useOrlanski* side (obcs_readparms.F:514-518) by
    ORLANSKI_<side> (orlanski_*.F); prescribed_3d.orlanski.<field> (OrlanskiParameters)."""
    target = _ORLANSKI_PARAMETERS[key][0]

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        parms = ctx.namelists.get("data.obcs", {}).get("OBCS_PARM01", {})
        if not any(parms.get(f"useorlanski{name}") is True for name in _OBCS_SIDE_NAMES.values()):
            raise Skipped("MITgcm reads OBCS_PARM02 only with a useOrlanski* side "
                          "(obcs_readparms.F:514-518)")
        return {f"{_ASSEMBLY}.obcs_orlanski_parameters.{target}": value}

    return Rule(MAP, target=f"model.boundary_condition.kwargs.orlanski.{target}",
                convert="identity (MIT units)", fn=fn)


for key in _ORLANSKI_PARAMETERS:
    OBCS_RULES[("OBCS_PARM02", key)] = _orlanski_parameter_rule(key)

for key in (
    "trelaxstevens",
    "srelaxstevens",
    "usestevensphasevel",
    "usestevensadvection",
):
    OBCS_RULES[("OBCS_PARM04", key)] = _obcs_parameter_of(
        tuple(f"usestevens{name}" for name in _OBCS_SIDE_NAMES.values()),
        "read only by OBCS_CALC_STEVENS (obcs_calc_stevens.F:139-145), which "
        "OBCS_CALC calls only when a useStevens* side is .TRUE. (obcs_calc.F:442-448)",
        Gap,
        "Stevens OBC parameter: the Stevens lane is not mapped (useStevens* rows)",
    )

# pkg/seaice alone calls the sea-ice sponge (OBCS_APPLY_SEAICE, seaice_model.F:312;
# obcs_apply_seaice.F:329-332); LOCOMgcm's sea ice has no open-boundary treatment
# (defgcm/seaice/transport/seaice.py:20, thsice.py:29).
OBCS_RULES[("OBCS_PARM01", "useseaicesponge")] = _false_or_gap(
    "sea-ice sponge at open boundaries (pkg/obcs obcs_seaice_sponge.F)"
)
for key in (
    "seaicespongethickness",
    "arelaxobcsinner",
    "arelaxobcsbound",
    "hrelaxobcsinner",
    "hrelaxobcsbound",
    "slrelaxobcsinner",
    "slrelaxobcsbound",
    "snrelaxobcsinner",
    "snrelaxobcsbound",
):
    OBCS_RULES[("OBCS_PARM05", key)] = _obcs_parameter_of(
        ("useseaicesponge",),
        "OBCS_READPARMS reads OBCS_PARM05 only when useSeaiceSponge is .TRUE. "
        "(obcs_readparms.F:557)",
        Gap,
        "sea-ice sponge parameter (useSeaiceSponge row)",
    )

# Read in OBCS_APPLY_SEAICE (obcs_apply_seaice.F:60-141) and OBCS_APPLY_UVICE
# (obcs_apply_uvice.F:65), which pkg/seaice alone calls (seaice_model.F:312,
# seaice_dynsolver.F:327).
OBCS_RULES[("OBCS_PARM01", "useseaiceneumann")] = _false_or_gap(
    "Neumann sea-ice open boundaries (pkg/obcs obcs_apply_seaice.F)"
)

for key in ("useobcstides", "obcs_tidalperiod") + tuple(
    f"ob{side.lower()}_{velocity}tid{part}file"
    for side in _OBCS_SIDES
    for velocity in ("u", "v")
    for part in ("am", "ph")
):
    OBCS_RULES[("OBCS_PARM01", key)] = _obcs_parameter_of(
        ("useobcstides",),
        "MITgcm reads and adds the OB tides only when useOBCStides is .TRUE. "
        "(obcs_init_fixed.F:454-582, obcs_adjust.F:49-51)",
        Deferred,
        "OBCS tides: LOCOMgcm has them (boundaries/tidal.py OBCSTidalConfig, "
        "prescribed_3d tidal_config, from obcs_add_tides.F); the translator "
        "reads no tidal amplitude or phase file",
    )




def _shared_obcs_upwind(asm: _Assembly, record: Record) -> bool | None:
    """Map the two tracer settings in OBCS_U1_ADV_TRACER to the shared switch."""
    temp = int(asm.get("obcs_u1_adv_t"))
    salt = int(asm.get("obcs_u1_adv_s"))
    if temp <= 0 and salt <= 0:
        return False
    if temp > 1 and salt > 1:
        return True
    _record_gap(
        record,
        {
            "source": "input/data.obcs:OBCS_u1_adv_T/OBCS_u1_adv_S",
            "parameter": "<derived:shared obcs_upwind>",
            "value": {"temperature": temp, "salinity": salt},
        },
        "LOCOMgcm requires the same upwind policy for temperature and salinity: "
        "both disabled or both inflow and outflow. Outflow-only (1) is unsupported.",
    )
    return None


def _wire_obcs_binary(
    yaml_tree: dict[str, Any],
    case_dir: Path,
    ctx: Ctx,
    asm: _Assembly,
    record: Record,
) -> bool | None:
    """Wire the prescribed-binary subset of pkg/obcs through the boundary lane.

    A useOrlanski* side takes prescribed_3d's ``orlanski`` value scheme with the OBCS_PARM02
    parameters (unset ones at their obcs_readparms.F:509-513 defaults), abEps and deltaT read
    by orlanski_*.F:91-98; OBCS_CALC computes its u, v, theta and salt (obcs_calc.F:97-103),
    so without useOBCSprescribe the lane is wired when every open side is Orlanski.
    """
    if not bool(asm.staged("use_obcs", False)):
        return None

    prescribe = asm.flag("use_obcs_prescribe")
    balance = asm.flag("use_obcs_balance")
    sponge = asm.flag("use_obcs_sponge")
    orlanski_sides = [side for side in _OBCS_SIDES
                      if dict(asm.staged("obcs_orlanski", {}) or {}).get(side)]
    if not (prescribe or orlanski_sides) or sponge:
        return None

    indices = _obcs_index_tables(asm, ctx)
    active_sides: list[str] = []
    invalid_indices = False
    for side in _OBCS_SIDES:
        if side not in indices:
            continue
        raw = indices[side]
        values = list(raw) if isinstance(raw, list) else [raw]
        expected = ctx.nx if side in {"N", "S"} else ctx.ny
        if len(values) != expected or not values or any(
            value != values[0] for value in values[1:]
        ):
            _record_gap(
                record,
                {
                    "source": f"input/data.obcs:OB index table {side}",
                    "parameter": f"<derived:OB index table {side}>",
                    "value": raw,
                },
                (
                    "mixed or incomplete OB_J*/OB_I* table cannot be represented; "
                    "The binary ingest supports only full_edge"
                ),
            )
            invalid_indices = True
            continue
        active_sides.append(side)

    # OBCS_PRESCRIBE_READ reads the files only under useOBCSprescribe (obcs_calc.F:407-414).
    files_by_side = dict(asm.staged("obcs_files", {}) or {}) if prescribe else {}
    inactive_files = sorted(set(files_by_side) - set(active_sides))
    if inactive_files:
        _record_gap(
            record,
            {
                "source": "input/data.obcs:OB*File",
                "parameter": "<derived:files on inactive OB sides>",
                "value": inactive_files,
            },
            "OB files are declared for sides without a full-edge OB index table",
        )
    if not active_sides and not invalid_indices:
        _record_gap(
            record,
            {
                "source": "input/data.obcs:OB_J*/OB_I*",
                "parameter": "<derived:active OBCS sides>",
                "value": indices,
            },
            "prescribed OB files require at least one non-zero full-edge index table",
        )
    if invalid_indices or inactive_files or not active_sides:
        return None
    orlanski_gaps = []
    if [side for side in orlanski_sides if side not in active_sides]:
        orlanski_gaps.append("useOrlanski* on a side without a full-edge OB index table")
    if [side for side in orlanski_sides if files_by_side.get(side)]:
        orlanski_gaps.append(
            "useOBCSprescribe files on an Orlanski side overwrite its radiated values "
            "(obcs_calc.F:407-414), which prescribed_3d does not do")
    if not prescribe and [side for side in active_sides if side not in orlanski_sides]:
        orlanski_gaps.append(
            "an open side without useOBCSprescribe or Orlanski takes OBCS_CALC's zero velocity, "
            "tRef and sRef (obcs_calc.F:120-137), a lane the translator does not wire")
    for gap in orlanski_gaps:
        _record_gap(
            record,
            {"source": "input/data.obcs:OBCS_PARM01 useOrlanski*",
             "parameter": "<derived:Orlanski open sides>", "value": orlanski_sides},
            gap,
        )
    if orlanski_gaps:
        return None

    periodic = asm.flag("periodic_external_forcing")
    period = asm.number("extern_forcing_period") if periodic else None
    cycle = asm.number("extern_forcing_cycle") if periodic else None
    if periodic and (
        period is None
        or cycle is None
        or period <= 0.0
        or cycle <= 0.0
        or cycle < period
        or cycle != round(cycle / period) * period
    ):
        _record_gap(
            record,
            {
                "source": "input/data:PARM03 periodicExternalForcing cadence",
                "parameter": "<derived:OBC record cadence>",
                "value": {"period": period, "cycle": cycle},
            },
            "periodic OBC files require a positive integral cycle/period cadence",
        )
        return None

    obcs_upwind = _shared_obcs_upwind(asm, record)
    if obcs_upwind is None:
        return None

    prec = int(ctx.data("PARM01", "readbinaryprec", 32) or 32)
    source_files = {
        _OBCS_SIDE_NAMES[side]: dict(files_by_side.get(side, {}))
        for side in active_sides
    }
    source = {
        "type": _OBCS_SOURCE_TYPE,
        "mit_verification_dir": str(case_dir.resolve()),
        "nx": ctx.nx,
        "ny": ctx.ny,
        "nz": ctx.nz,
        "sides": active_sides,
        "files": source_files,
        "t_ref": _fill_ref(
            ctx.data("PARM01", "tref"), ctx.nz, _MIT_DEFAULT_TREF, "tref"
        ),
        "s_ref": _fill_ref(
            ctx.data("PARM01", "sref"), ctx.nz, _MIT_DEFAULT_SREF, "sref"
        ),
        "read_binary_prec": prec,
        "periodic_external_forcing": periodic,
        "forcing_period_seconds": period,
        "forcing_cycle_seconds": cycle,
    }
    yaml_tree.setdefault("sources", {})[_OBCS_SOURCE_NAME] = source
    yaml_tree["boundary"].update(
        {"type": "open", "sides": active_sides, "ob_index_mode": "full_edge"}
    )
    extras = yaml_tree["case"]["extras"]
    extras["obcs_sides"] = active_sides
    extras["west_east_boundary"] = (
        "open" if any(side in active_sides for side in ("E", "W")) else "periodic"
    )
    extras["south_north_boundary"] = (
        "open" if any(side in active_sides for side in ("N", "S")) else "periodic"
    )
    tangential = any(
        (
            side in {"N", "S"} and "u" in files_by_side.get(side, {})
        )
        or (
            side in {"E", "W"} and "v" in files_by_side.get(side, {})
        )
        for side in active_sides
    )
    yaml_tree["model"]["boundary_condition"] = {
        "name": "prescribed_3d",
        "kwargs": {
            "enforce_obcs_balance": balance,
            "apply_tangential_velocity": tangential,
        },
    }
    if orlanski_sides:
        staged = dict(asm.staged("obcs_orlanski_parameters", {}) or {})
        ab_eps = ctx.data("PARM03", "abeps")
        yaml_tree["model"]["boundary_condition"]["kwargs"].update(
            value_schemes={_OBCS_SIDE_NAMES[side]: OpenBoundaryValueScheme.ORLANSKI.value
                           for side in orlanski_sides},
            orlanski={**{name: staged.get(name, default)
                         for name, default in _ORLANSKI_PARAMETERS.values()},
                      # the stepper's abEps (set_defaults.F:317) and MITgcm's deltaT
                      "ab_eps": float(asm.get("ab_eps") if ab_eps is None else ab_eps),
                      "delta_t": float(_master_dt(ctx))},
        )
    record.derived.append(
        {
            "target": (
                f"sources.{_OBCS_SOURCE_NAME} + boundary + "
                "model.boundary_condition[prescribed_3d]"
            ),
            "value": {
                "sides": active_sides,
                "files": source_files,
                "ob_index_mode": "full_edge",
                "periodic_external_forcing": periodic,
                "forcing_period_seconds": period,
                "forcing_cycle_seconds": cycle,
                "enforce_obcs_balance": balance,
                "apply_tangential_velocity": tangential,
                "obcs_upwind": obcs_upwind,
                "value_schemes": yaml_tree["model"]["boundary_condition"]["kwargs"].get(
                    "value_schemes"),
                "orlanski": yaml_tree["model"]["boundary_condition"]["kwargs"].get("orlanski"),
            },
            "rationale": (
                "OB_J*/OB_I* are full constant edge tables "
                "(pkg/obcs/obcs_readparms.F:105-112,139-156); file roles are "
                "read per side and missing T/S/U/V/W/eta use obcs_calc.F:122-137,"
                "170-186 fallbacks.  MIT defaults useOBCSprescribe/useOBCSsponge/"
                "useOBCSbalance are .FALSE. at obcs_readparms.F:267,269,283; "
                "the emitted balance value therefore bypasses LOCOMgcm's True "
                "dataclass default (boundaries/open.py:1519-1527).  BoundaryBundleSource "
                "consumes the source cadence through _canonical_helpers.py:139-146"
            ),
        }
    )
    return obcs_upwind


def _wire_obcs_analytic(
    yaml_tree: dict[str, Any],
    ctx: Ctx,
    asm: _Assembly,
    record: Record,
) -> bool | None:
    """Wire an authorised case-private ``obcs_calc.F`` replacement."""
    spec = asm.staged("analytic_obcs")
    if not isinstance(spec, dict):
        return None

    indices = _obcs_index_tables(asm, ctx)
    active_sides: list[str] = []
    invalid_indices = False
    for side in _OBCS_SIDES:
        if side not in indices:
            continue
        raw = indices[side]
        values = list(raw) if isinstance(raw, list) else [raw]
        expected = ctx.nx if side in {"N", "S"} else ctx.ny
        if len(values) != expected or not values or any(
            value != values[0] for value in values[1:]
        ):
            _record_gap(
                record,
                {
                    "source": f"input/data.obcs:OB index table {side}",
                    "parameter": f"<derived:analytic OB index table {side}>",
                    "value": raw,
                },
                "analytic OBCS requires a complete full_edge OB index table",
            )
            invalid_indices = True
            continue
        active_sides.append(side)

    if not active_sides and not invalid_indices:
        _record_gap(
            record,
            {
                "source": "input/data.obcs:OB_J*/OB_I*",
                "parameter": "<derived:analytic active OBCS sides>",
                "value": indices,
            },
            "analytic OBCS requires at least one non-zero full-edge index table",
        )
    if invalid_indices or not active_sides:
        return None

    if asm.flag("use_obcs_sponge") or asm.flag("use_obcs_balance"):
        _record_gap(
            record,
            {
                "source": "input/data.obcs:analytic OBCS switches",
                "parameter": "<derived:analytic OBCS capability>",
                "value": {
                    "useOBCSsponge": asm.flag("use_obcs_sponge"),
                    "useOBCSbalance": asm.flag("use_obcs_balance"),
                },
            },
            "registered analytic OBCS plugin does not represent sponge or balance",
        )
        return None

    # Staged values only the prescribed-binary lane reads (_wire_obcs_binary); the plugin has
    # neither input, so a true one would leave the yaml while the registry records it mapped.
    # useOrlanski*: the case-private obcs_calc.F keeps the ORLANSKI_<side> calls
    # (internal_wave_open.obcs_calc.F:84-194), and LOCOMgcm's Orlanski scheme exists only on
    # prescribed_3d (its OBCS_PARM02 values are staged only with such a side).
    # useOBCSprescribe: that obcs_calc.F calls no OBCS_PRESCRIBE_READ, but OBCS_INIT_VARIABLES
    # reads the OB files on a pickup start (obcs_init_variables.F:364, :418-426); the OB file
    # names are read only under it (obcs_calc.F:407-414).
    orlanski = dict(asm.staged("obcs_orlanski", {}) or {})
    dropped = {f"useOrlanski{_OBCS_SIDE_NAMES[side].capitalize()}": True
               for side in _OBCS_SIDES if orlanski.get(side)}
    if asm.flag("use_obcs_prescribe"):
        dropped["useOBCSprescribe"] = True
    if dropped:
        _record_gap(
            record,
            {
                "source": "input/data.obcs:OBCS_PARM01",
                "parameter": "<derived:analytic OBCS keys of the prescribed lane>",
                "value": dropped,
            },
            "the registered analytic OBCS plugin does not take "
            + ", ".join(dropped)
            + " (Orlanski radiation is wired only on prescribed_3d; the plugin reads no OB file)",
        )
        return None

    obcs_upwind = _shared_obcs_upwind(asm, record)
    if obcs_upwind is None:
        return None

    plugin = str(spec["plugin"])
    kwargs = dict(spec.get("kwargs", {}))
    tref_kwarg = spec.get("tref_kwarg")
    if tref_kwarg is not None:
        kwargs[str(tref_kwarg)] = _fill_ref(
            ctx.data("PARM01", "tref"), ctx.nz, _MIT_DEFAULT_TREF, "tref"
        )
    salt_values = {
        "sRef": _fill_ref(ctx.data("PARM01", "sref"), ctx.nz, _MIT_DEFAULT_SREF, "sref"),
        "zero": [0.0] * ctx.nz,
    }
    for name, value in spec.get("salt_kwargs", {}).items():
        kwargs[str(name)] = list(salt_values[value])
    yaml_tree["boundary"].update(
        {"type": "open", "sides": active_sides, "ob_index_mode": "full_edge"}
    )
    extras = yaml_tree["case"]["extras"]
    extras["obcs_sides"] = active_sides
    # An axis with no OB side keeps MITgcm's process topology, which is periodic
    # unless the case turns prevention on: EESET_PARMS initialises
    # notUsingXPeriodicity/notUsingYPeriodicity .FALSE.
    # (eesupp/src/eeset_parms.F:66-73) and INI_PROCS walls an axis only when the
    # corresponding flag is true (eesupp/src/ini_procs.F:67-80,123-143); a case
    # that does set either key in &EEPARMS has no EEDATA_RULES entry and so fails
    # closed in _translate_namelist.  The prescribed-binary lane already resolves
    # the untouched axis this way; this lane wrote 'wall', which walled
    # internal_wave's y axis against a y-periodic reference run -- zeroing both
    # outer hFacS faces (geometry/masks.py) and the CG2D south/north edges
    # (steppers/implicit_free_surface_cg2d.py).
    extras["west_east_boundary"] = (
        "open" if any(side in active_sides for side in ("E", "W")) else "periodic"
    )
    extras["south_north_boundary"] = (
        "open" if any(side in active_sides for side in ("N", "S")) else "periodic"
    )
    yaml_tree["model"]["boundary_condition"] = {"name": plugin, "kwargs": kwargs}
    # useOBCSbalance is .FALSE. whenever control reaches here (a true value takes
    # the capability gap above). The analytic specification already disables
    # the boundary plugin's enforce_obcs_balance; the stepper has no balance
    # parameter (see the default_divergence_note on the useOBCSbalance switch,
    # defgcm/io/runner/layer_d/_mit_switches.py).
    record.derived.append(
        {
            "target": f"model.boundary_condition[{plugin}] + boundary",
            "value": {
                "authorized_source": spec["authorized_source"],
                "plugin": plugin,
                "kwargs": kwargs,
                "sides": active_sides,
                "ob_index_mode": "full_edge",
                "obcs_upwind": obcs_upwind,
                "west_east_boundary": extras["west_east_boundary"],
                "south_north_boundary": extras["south_north_boundary"],
            },
            "ruling": spec["ruling"],
            "overrides": spec["overrides"],
            "rationale": (
                "code/obcs_calc.F content matched ANALYTIC_OBCS; the analytic boundary rule "
                "authorises only this exact upstream source identity to select a "
                "registered boundary plugin.  Declared plugin kwargs are resolved "
                "from the MIT namelist; OB_J*/OB_I* alone determine the open sides.  "
                "An axis carrying no OB side keeps MITgcm's default-periodic process "
                "topology (eeset_parms.F:66-73, ini_procs.F:67-80,123-143), and "
                "useOBCSbalance=.FALSE. is the plugin's enforce_obcs_balance"
            ),
        }
    )
    return obcs_upwind
