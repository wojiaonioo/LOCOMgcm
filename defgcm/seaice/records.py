"""Scientific records for offline_exf_seaice (MITgcm 27256b2d).

All horizontal tensors use independent [bj, bi, j, i] storage, float64,
MIT signs and Cartesian components. TICES additionally has one category axis.
The six lifetime groups follow SEAICE.h, THSICE_VARS.h and FFIELDS.h.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

G = (2, 2, 27, 46)
K = (2, 2, 1, 27, 46)
L = (27, 46)
I = (21, 40)
Chain = Literal["T", "L", "H", "B"]


@dataclass(frozen=True)
class Domain:
    """Trim south, north, west, east from the storage edges; the complement retains its old value."""

    south: int = 0
    north: int = 0
    west: int = 0
    east: int = 0

    def __post_init__(self):
        if min(self.south, self.north, self.west, self.east) < 0:
            raise ValueError("domain trims must be nonnegative")

    @property
    def slices(self) -> tuple[slice, slice]:
        return (slice(self.south, -self.north or None), slice(self.west, -self.east or None))


FULL = Domain()
INTERIOR = Domain(3, 3, 3, 3)
D = Domain(1, 1, 1, 1)
LINE = Domain(2, 2, 2, 2)
SOUTH_WEST = Domain(0, 1, 0, 1)
NORTH_EAST = Domain(1, 0, 1, 0)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    location: str
    units: str
    mask: str
    time_level: str
    domain: Domain
    writer: str
    shape: tuple[int, ...] = G
    sign: str = "scalar"
    dtype: torch.dtype = torch.float64


@dataclass(frozen=True)
class SeaiceState:
    UICE: torch.Tensor
    VICE: torch.Tensor
    HEFF: torch.Tensor
    AREA: torch.Tensor
    HSNOW: torch.Tensor
    TICES: torch.Tensor


@dataclass(frozen=True)
class ThsiceState:
    # thsice_write_pickup.F:54-62: native record order.
    iceMask: torch.Tensor
    iceHeight: torch.Tensor
    snowHeight: torch.Tensor
    Tsrf: torch.Tensor
    Tice1: torch.Tensor
    Tice2: torch.Tensor
    Qice1: torch.Tensor
    Qice2: torch.Tensor
    snowAge: torch.Tensor


THSICE_FIELDS = tuple(ThsiceState.__dataclass_fields__)
THSICE_ADVECTED = ("iceMask", "iceHeight", "snowHeight", "Qice1", "Qice2")


@dataclass(frozen=True)
class DynamicsCarry:
    # seaice_init_varia.F:79,435-443; seaice_lsr.F:379-382.
    DWATN: torch.Tensor
    eta: torch.Tensor
    zeta: torch.Tensor


@dataclass(frozen=True)
class Clock:
    myIter: int
    myTime: float
    generation: int = 0


@dataclass(frozen=True)
class OceanState:
    theta: torch.Tensor
    salt: torch.Tensor
    uVel: torch.Tensor
    vVel: torch.Tensor
    etaN: torch.Tensor
    gtNm1: torch.Tensor
    etaH: torch.Tensor
    etaHnm1: torch.Tensor
    dEtaHdt: torch.Tensor
    clock: Clock


@dataclass(frozen=True)
class SeaiceCarry:
    """Cross-step state owned by the sea-ice component; no ocean state."""

    seaice: SeaiceState
    thsice: ThsiceState
    dynamics: DynamicsCarry
    sIceLoad: torch.Tensor
    clock: Clock


@dataclass(frozen=True)
class OfflineCarry:
    """The offline programs' state: the ice component and its single-layer host."""

    ice: SeaiceCarry
    ocean: OceanState


@dataclass(frozen=True)
class Atmosphere:
    uwind: torch.Tensor
    vwind: torch.Tensor
    atemp: torch.Tensor
    aqh: torch.Tensor
    lwdown: torch.Tensor
    swdown: torch.Tensor
    precip: torch.Tensor
    snowPrecip: torch.Tensor
    runoff: torch.Tensor
    wspeed: torch.Tensor
    evap: torch.Tensor
    ustress: torch.Tensor
    vstress: torch.Tensor


@dataclass(frozen=True)
class OpenWaterFluxes:
    Qnet_ow: torch.Tensor
    Qsw_ow: torch.Tensor
    EmPmR_ow: torch.Tensor
    saltFlux_ow: torch.Tensor
    fu_ow: torch.Tensor
    fv_ow: torch.Tensor
    SST: torch.Tensor
    SSS: torch.Tensor
    pLoad: torch.Tensor


@dataclass(frozen=True)
class OceanFluxes:
    Qnet: torch.Tensor
    Qsw: torch.Tensor
    EmPmR: torch.Tensor
    saltFlux: torch.Tensor
    fu: torch.Tensor
    fv: torch.Tensor
    sIceLoad: torch.Tensor
    AREA: torch.Tensor


@dataclass(frozen=True)
class Geometry:
    dxF: torch.Tensor
    dyF: torch.Tensor
    dxC: torch.Tensor
    dyC: torch.Tensor
    dxG: torch.Tensor
    dyG: torch.Tensor
    dxV: torch.Tensor
    dyU: torch.Tensor
    rA: torch.Tensor
    rAw: torch.Tensor
    rAs: torch.Tensor
    rAz: torch.Tensor
    hFacC: torch.Tensor
    maskC: torch.Tensor
    maskInC: torch.Tensor
    HEFFM: torch.Tensor
    recip_hFacC: torch.Tensor
    hFacW: torch.Tensor
    maskW: torch.Tensor
    maskInW: torch.Tensor
    SIMaskU: torch.Tensor
    seaiceMaskU: torch.Tensor
    hFacS: torch.Tensor
    maskS: torch.Tensor
    maskInS: torch.Tensor
    SIMaskV: torch.Tensor
    seaiceMaskV: torch.Tensor
    fCori: torch.Tensor
    fCoriG: torch.Tensor
    Bo_surf: torch.Tensor
    drF: torch.Tensor
    globalArea: torch.Tensor


@dataclass(frozen=True)
class Regularization:
    seaice: SeaiceState
    d_HEFFbyNEG: torch.Tensor
    d_HSNWbyNEG: torch.Tensor


@dataclass(frozen=True)
class ThsiceTemporaryFluxes:
    oldIceMask: torch.Tensor
    sHeating: torch.Tensor
    flxCndBt: torch.Tensor
    icFlxAtm: torch.Tensor
    icFrwAtm: torch.Tensor
    icFlxSW: torch.Tensor
    flx2oc: torch.Tensor
    frw2oc: torch.Tensor
    fsalt: torch.Tensor


@dataclass(frozen=True)
class TransportCapacity:
    iceVol: torch.Tensor
    oldVol: torch.Tensor


@dataclass(frozen=True)
class DynamicsWork:
    uIceNm1: torch.Tensor
    vIceNm1: torch.Tensor
    PRESS0: torch.Tensor
    zMax: torch.Tensor
    zMin: torch.Tensor


@dataclass(frozen=True)
class RecordSchema:
    i4: tuple[str, ...]
    f8: tuple[str, ...] = ()


# seaice_lsr.F:776,985,1070; seaice_dynsolver.F:133;
# thsice_solve4temp.F:394-543. Undefined residuals are gated by the i4 flags.
LSR_CTL = RecordSchema(
    ("seq", "myIter", "ipass", "prepRan", "doNonLinLoop", "doIterate4u", "doIterate4v", "ICOUNT1", "ICOUNT2", "linearIterLoc", "printResidual", "residGate"),
    ("residUini", "residVini", "LSR_ERROR", "WFAU", "WFAV", "S1A", "S2A"),
)
LSR_CTL_FLEX = RecordSchema(LSR_CTL.i4, LSR_CTL.f8 + ("residIni", "residIniNonLin", "LSRflexFac", "LSR_ERRORfu", "LSR_ERRORfv"))
LSR_CHK = RecordSchema(
    ("seq", "myIter", "ipass", "m", "sweptU", "sweptV", "checkedU", "checkedV", "doIterate4u", "doIterate4v", "ICOUNT1", "ICOUNT2"),
    ("S1", "S2", "WFAU", "WFAV"),
)
LSR_END = RecordSchema(("seq", "myIter", "ipass", "ICOUNT1", "ICOUNT2", "doIterate4u", "doIterate4v"))
DYN_GATE = RecordSchema(("seq", "myIter", "gate", "entered"), ("myTime",))
TH_ITER = RecordSchema(("seq", "myIter", "d", "nIter"))
RECORD_SCHEMAS = {"lsr_ctl": LSR_CTL, "lsr_ctl_flex": LSR_CTL_FLEX, "lsr_chk": LSR_CHK, "lsr_end": LSR_END, "dyn_gate": DYN_GATE, "th_iter": TH_ITER}


def _specs() -> dict[str, FieldSpec]:
    result: dict[str, FieldSpec] = {}

    def add(names, units, writer, *, location="C", mask="none", time="step", domain=FULL, shape=G, sign="scalar"):
        for name in names.split():
            result[name] = FieldSpec(name, location, units, mask, time, domain, writer, shape, sign)

    add("HEFF HSNOW", "m", "advection -> regularization -> growth; B mapping", mask="HEFFM on writeback", domain=INTERIOR, time="previous published / B current mapping")
    add("AREA", "1", "advection -> regularization -> growth; B mapping", mask="HEFFM on writeback", domain=INTERIOR)
    add("UICE", "m/s", "LSR; T retains", location="U", mask="seaiceMaskU", time="last dynamics", sign="+x")
    add("VICE", "m/s", "LSR; T retains", location="V", mask="seaiceMaskV", time="last dynamics", sign="+y")
    add("TICES", "K", "regularization -> growth", domain=INTERIOR, shape=K, time="previous thermodynamics")
    for name in THSICE_FIELDS:
        units = "1" if name == "iceMask" else "m" if "Height" in name else "J/kg" if name.startswith("Qice") else "s" if name == "snowAge" else "degC"
        add(name, units, "THSICE thermodynamics; B transport for five fields and snowAge", domain=INTERIOR, time="previous published")
    add("DWATN", "kg/m2/s", "LSR linearization", domain=SOUTH_WEST, time="last dynamics")
    add("eta zeta", "kg/s", "constitutive law", domain=D, time="last constitutive call; initialized outer ring")
    add("etaZ zetaZ", "kg/s", "corner average", location="Z", domain=D)
    add("press", "N/m", "constitutive law", domain=D)
    add("deltaC", "1/s", "constitutive law", domain=D)
    add("uIceNm1 uIceC", "m/s", "LSR", location="U")
    add("vIceNm1 vIceC", "m/s", "LSR", location="V")
    add("PRESS0", "N/m", "ice strength")
    add("zMax zMin SEAICE_zMax SEAICE_zMin", "kg/s", "ice strength")
    add("e11 e22", "1/s", "strain rate", domain=SOUTH_WEST)
    add("e12", "1/s", "strain rate", location="Z", domain=NORTH_EAST)
    add("uwind vwind", "m/s", "atmosphere", domain=INTERIOR, time="step start")
    add("atemp", "K", "atmosphere", domain=INTERIOR)
    add("aqh", "kg/kg", "atmosphere", domain=INTERIOR)
    add("lwdown swdown", "W/m2", "atmosphere", domain=INTERIOR, sign="downward")
    add("precip snowPrecip runoff", "m/s", "atmosphere", domain=INTERIOR, sign="into ocean")
    add("wspeed evap", "m/s", "EXF", domain=INTERIOR, sign="evap out of ocean")
    add("ustress vstress", "N/m2", "EXF", sign="+x,+y")
    for suffix, writer in (("_ow", "EXF"), ("", "T growth; L EXF; H STEP_FWD; B transport rebate")):
        add("Qnet" + suffix, "W/m2", writer, sign="upward")
        add("Qsw" + suffix, "W/m2", "EXF" if suffix else "T growth; L EXF; H/B STEP_TEMP", domain=INTERIOR, sign="upward")
        add("EmPmR" + suffix, "kg/m2/s", writer, sign="upward; increases salinity")
        add("saltFlux" + suffix, "g/m2/s", writer, sign="upward; decreases salinity")
        for name, location in (("fu", "U"), ("fv", "V")):
            add(name + suffix, "N/m2", "EXF" if suffix else "T/L/B ocean stress; H EXF", location=location, mask="face", sign="accelerates")
    add("SST", "degC", "EXF")
    add("SSS", "g/kg", "EXF")
    add("pLoad", "N/m2", "EXF")
    add("sIceLoad", "kg/m2", "T retains zero; L program; H STEP_FWD; B transport", domain=INTERIOR)
    add("theta", "degC", "host", mask="maskC; dry +0.0", time="step start")
    add("salt", "g/kg", "host initialization", mask="maskC; dry +0.0", time="constant")
    add("uVel", "m/s", "host initialization", location="U", mask="maskW", time="constant")
    add("vVel", "m/s", "host initialization", location="V", mask="maskS", time="constant")
    add("etaN etaH etaHnm1", "m", "host initialization", time="constant")
    add("dEtaHdt", "m/s", "host initialization")
    add("gtNm1", "K/s", "host AB; +0.0", time="previous raw tendency")
    add("surfaceForcingT", "K m/s", "host surface forcing")
    add("hOceMxL", "m", "host geometry")
    add("d_HEFFbyNEG d_HSNWbyNEG", "m", "regularization", domain=INTERIOR)
    add("oldIceMask", "1", "THSICE entry snapshot", domain=INTERIOR)
    add("iceVol oldVol", "m2 or m3, according to transport stage", "THSICE transport")
    add("sHeating flxCndBt icFlxAtm icFlxSW flx2oc oceQnet", "W/m2", "THSICE", domain=INTERIOR, sign="downward for icFlxAtm,icFlxSW,flx2oc,oceQnet")
    add("icFrwAtm frw2oc oceFWfx", "kg/m2/s", "THSICE", domain=INTERIOR)
    add("fsalt oceSflx", "g/m2/s", "THSICE", domain=INTERIOR)
    add("dxF dyF dxC dyC dxG dyG dxV dyU", "m", "space", location="MIT metric location", time="static")
    add("rA rAw rAs rAz", "m2", "space", location="C,U,V,Z respectively", time="static")
    add("hFacC maskC maskInC HEFFM recip_hFacC", "1", "space", time="static")
    add("hFacW maskW maskInW SIMaskU seaiceMaskU", "1", "space", location="U", time="static")
    add("hFacS maskS maskInS SIMaskV seaiceMaskV", "1", "space", location="V", time="static")
    add("fCori", "1/s", "space", time="static")
    add("fCoriG", "1/s", "space", location="Z", time="static")
    add("Bo_surf", "m/s2", "space", time="static")
    add("drF", "m", "space", location="layer", shape=(), time="static")
    add("globalArea", "m2", "space", location="scalar", shape=(), time="static")
    add("TAUX", "N/m2", "wind stress", location="U", mask="SIMaskU", domain=NORTH_EAST, sign="+x")
    add("TAUY", "N/m2", "wind stress", location="V", mask="SIMaskV", domain=NORTH_EAST, sign="+y")
    add("seaiceMassC", "kg/m2", "dynamics preparation", domain=NORTH_EAST)
    add("seaiceMassU", "kg/m2", "dynamics preparation", location="U", domain=NORTH_EAST)
    add("seaiceMassV", "kg/m2", "dynamics preparation", location="V", domain=NORTH_EAST)
    add("FORCEX0", "N/m2", "dynamics preparation", location="U", domain=NORTH_EAST)
    add("FORCEY0", "N/m2", "dynamics preparation", location="V", domain=NORTH_EAST)
    add("rhsU", "N/m2", "LSR RHS", location="U", domain=LINE)
    add("rhsV", "N/m2", "LSR RHS", location="V", domain=LINE)
    add("AU BU CU uRt1 uRt2", "kg/m2/s", "LSR matrix", location="U", domain=LINE)
    add("AV BV CV vRt1 vRt2", "kg/m2/s", "LSR matrix", location="V", domain=LINE)
    add("uIce old_u before_exchange_u after_exchange_u", "m/s", "LSR scan", location="U")
    add("vIce old_v before_exchange_v after_exchange_v", "m/s", "LSR scan", location="V")
    add("a_QbyATM_cover a_QbyATM_open a_QSWbyATM_cover a_QSWbyATM_open", "W/m2", "PW79 surface_budget", domain=INTERIOR, sign="upward")
    add("a_FWbySublim", "kg/m2/s", "PW79 surface_budget", domain=INTERIOR, sign="upward")
    add("tOceMxL", "degC", "THSICE get_ocean")
    add("sOceMxL", "g/kg", "THSICE get_ocean")
    add("v2ocMxL", "m2/s2", "THSICE get_ocean", domain=SOUTH_WEST)
    add("prcAtm snowPr", "kg/m2/s", "THSICE map_inputs", domain=INTERIOR, sign="downward")
    add("qPrcRn", "W/m2", "THSICE map_inputs", domain=INTERIOR, sign="downward")
    add("siceAlb", "1", "THSICE albedo", domain=INTERIOR)
    add("dTsrf", "K", "THSICE temperature solve", domain=INTERIOR)
    add("iceFrac", "1", "THSICE calc_thickness", domain=INTERIOR)
    add("iceFrc", "1", "THSICE area transport")
    add("frzmltMxL", "W/m2", "THSICE calc_thickness", domain=INTERIOR)
    add("uTrans vTrans", "m2/s", "THSICE area transport", location="U,V")
    add("uTrIce vTrIce", "m3/s", "THSICE thickness transport", location="U,V")
    add("iceVol_in iceVol_out", "m2 or m3, according to stage", "THSICE capacity transport")
    add("oldVol_in oldVol_out", "m3", "THSICE capacity initialization")
    add("SIatmQnt SItflux", "W/m2", "PW79 diagnostics", domain=INTERIOR)
    add("SIatmFW", "kg/m2/s", "PW79 diagnostics", domain=INTERIOR)
    add("myTime", "s", "program", location="scalar", shape=(), time="component clock")
    result["myIter"] = FieldSpec("myIter", "scalar", "1", "none", "component clock", FULL, "program", (), dtype=torch.int32)
    return result


FIELD_SPECS = _specs()


@dataclass(frozen=True)
class FieldVersion:
    """A particular mathematical write, followed by its named exchange if any."""

    field: FieldSpec
    chain: str
    stage: str
    writer: str
    domain: Domain
    exchange: str | None
    retained_band: str = "previous version of this tile"


def field_version(name: str, *, chain: str, stage: str) -> FieldVersion:
    """Resolve a single writer and valid rectangle, rejecting absent stages.

    The base field catalog describes lifetimes; this function describes one
    actual version, including chain-specific final flux domains.
    """
    spec = FIELD_SPECS[name]
    stages = {
        "EXF": ("TLHB", "forcing.exf_step", FULL if name in ("Qnet_ow", "EmPmR_ow", "fu_ow", "fv_ow", "SST", "SSS", "pLoad", "saltFlux_ow", "ustress", "vstress") else INTERIOR, None),
        "ADVDIFF": ("TL", "transport.seaice_advdiff", INTERIOR, None),
        "REG_RIDGE": ("TL", "transport.seaice_reg_ridge", INTERIOR, None),
        "GROWTH": ("T", "thermo_pw79.growth", INTERIOR, None),
        "STEP_TEMP": ("HB", "thermo_thsice.step_temperature", INTERIOR, None),
        "STEP_FWD": ("HB", "thermo_thsice.step_thermodynamics", INTERIOR, None),
        "MAP_THSICE": ("B", "program.map_thsice", FULL, None),
        "THSICE_TRANSPORT": ("B", "transport.thsice_do_advect", SOUTH_WEST if name in ("Qnet", "EmPmR", "saltFlux", "sIceLoad") else INTERIOR, None),
        "X8": ("TL", "space.exchange", FULL, "X8"),
        "X9": ("B", "space.exchange", FULL, "X9"),
        "HOST": ("TLHB", "host_single_layer_ocean.temperature_step", FULL, "X10"),
        "LSR": ("LB", "dynamics.dynamics_step", spec.domain, "X6" if name in ("UICE", "VICE") else None),
    }
    chains, writer, domain, exchange = stages[stage]
    if chain not in chains or len(chain) != 1:
        raise ValueError(f"{stage} does not execute on chain {chain}")
    if stage in ("X8", "X9"):
        allowed = ("HEFF", "AREA", "HSNOW", "EmPmR", "saltFlux", "Qnet") if stage == "X8" else THSICE_ADVECTED
        if name not in allowed:
            raise ValueError(f"{stage} does not exchange {name}")
    return FieldVersion(spec, chain, stage, writer, domain, exchange)


@dataclass(frozen=True)
class StationField:
    name: str
    shape: str
    domain: Domain
    mask: str = "none"


@dataclass(frozen=True)
class StationSpec:
    name: str
    boundary: str
    fields: tuple[StationField, ...]
    records: tuple[str, ...] = ()
    native_records: int = 0


def _stations() -> dict[str, StationSpec]:
    result: dict[str, StationSpec] = {}

    def add(name, boundary, names="", domain=INTERIOR, shape="G", *, full="", local="", inner="", wet="", records=(), native=0):
        fs = []
        for field in names.split():
            sh = "K" if field == "TICES" and shape == "G" else shape
            if field in local.split():
                sh = "L"
            if field in inner.split():
                sh = "I"
            fs.append(StationField(field, sh, FULL if field in full.split() else domain, "wet" if field in wet.split() else "none"))
        result[name] = StationSpec(name, boundary, tuple(fs), records, native)

    initial = "HEFF AREA HSNOW UICE VICE theta salt uVel vVel etaN HEFFM SIMaskU SIMaskV TICES"
    forcing = "Qnet Qsw EmPmR saltFlux SST fu fv wspeed evap uwind vwind atemp aqh lwdown swdown precip"
    reg = "HEFF AREA HSNOW d_HEFFbyNEG d_HSNWbyNEG TICES"
    exchanged = "HEFF AREA HSNOW EmPmR saltFlux Qnet"
    ice = " ".join(THSICE_FIELDS)
    thermal = ice + " Qnet EmPmR saltFlux sIceLoad flx2oc frw2oc fsalt"
    for chain in "TLHB":
        add(chain + "1", "forcing.exf_step", forcing, full="Qnet EmPmR fu fv SST")
    add("T0", "program.initialize", initial, FULL)
    add("L0", "program.initialize", "HEFF AREA HSNOW UICE VICE uVel vVel etaN HEFFM SIMaskU SIMaskV seaiceMaskU seaiceMaskV TICES", FULL)
    add("H0", "program.initialize", ice + " sIceLoad theta salt uVel vVel", FULL)
    add("B0", "program.initialize", ice + " UICE VICE AREA HEFF HSNOW sIceLoad theta", FULL)
    add("T2", "dynamics.ocean_stress", "uwind vwind TAUX TAUY fu fv DWATN", NORTH_EAST, full="uwind vwind fu fv DWATN")
    add("T3", "transport.seaice_advdiff", "HEFF AREA HSNOW", FULL)
    add("T4", "transport.seaice_reg_ridge", reg)
    add("T5", "thermo_pw79.solve_temperature", "TICES a_QbyATM_cover a_QbyATM_open a_QSWbyATM_cover a_QSWbyATM_open a_FWbySublim", local="TICES", inner="a_QbyATM_cover a_QbyATM_open a_QSWbyATM_cover a_QSWbyATM_open a_FWbySublim")
    add("T6.growth", "thermo_pw79.growth", "HEFF AREA HSNOW Qnet Qsw EmPmR saltFlux TICES")
    add("T6", "program.finish_seaice", "HEFF AREA HSNOW Qnet EmPmR saltFlux Qsw sIceLoad TICES", full=exchanged)
    add("L2", "dynamics.prepare_dynamics", "uwind vwind TAUX TAUY seaiceMassC seaiceMassU seaiceMassV FORCEX0 FORCEY0 PRESS0 SEAICE_zMax SEAICE_zMin", NORTH_EAST, full="uwind vwind PRESS0 SEAICE_zMax SEAICE_zMin")
    add("L3", "dynamics.linearize", "uIceC vIceC e11 e22 e12 zeta eta zetaZ etaZ press deltaC DWATN", D, full="uIceC vIceC")
    # Per-field strain/drag rectangles differ from the constitutive D rectangle.
    result["L3"] = StationSpec("L3", result["L3"].boundary, tuple(
        StationField(f.name, f.shape, SOUTH_WEST if f.name in ("e11", "e22", "DWATN") else NORTH_EAST if f.name == "e12" else f.domain)
        for f in result["L3"].fields))
    add("L4", "dynamics.linear_system", "rhsU rhsV AU BU CU AV BV CV uRt1 uRt2 vRt1 vRt2", LINE)
    add("L5", "dynamics.lsr_control", records=("lsr_ctl_flex",))
    add("L6", "dynamics.lsr_scan and dynamics.check_scan", "old_u old_v before_exchange_u before_exchange_v after_exchange_u after_exchange_v", FULL, "L", records=("lsr_chk",))
    add("L7", "dynamics.lsr_pass_end", "uIce vIce", FULL, records=("lsr_end",))
    add("L8", "dynamics.dynamics_step", "fu fv UICE VICE DWATN", FULL)
    add("L9", "transport.seaice_advdiff", "HEFF AREA HSNOW")
    add("L10.reg", "transport.seaice_reg_ridge", reg)
    add("L10", "program.finish_seaice", reg + " sIceLoad EmPmR saltFlux Qnet", full=exchanged)
    add("H2", "thermo_thsice.map_inputs", "hOceMxL tOceMxL sOceMxL v2ocMxL prcAtm icFlxSW snowPr qPrcRn", INTERIOR, "L", full="hOceMxL tOceMxL sOceMxL")
    h2 = result["H2"]
    result["H2"] = StationSpec(h2.name, h2.boundary, tuple(StationField(f.name, f.shape, SOUTH_WEST if f.name == "v2ocMxL" else f.domain) for f in h2.fields))
    add("H3", "thermo_thsice.step_temperature", "siceAlb Tsrf Tice1 Tice2 Qice1 Qice2 sHeating flxCndBt icFlxAtm icFrwAtm icFlxSW Qsw dTsrf", INTERIOR, "L", inner="dTsrf", records=("th_iter",))
    h3 = result["H3"]
    result["H3"] = StationSpec(h3.name, h3.boundary, tuple(StationField(f.name, f.shape, f.domain, "entry iceMask>0" if f.name != "Qsw" else "none") for f in h3.fields), h3.records)
    add("H4", "thermo_thsice.calc_thickness", "iceFrac iceHeight snowHeight Tsrf Qice1 Qice2 icFrwAtm frzmltMxL flx2oc frw2oc fsalt", INTERIOR, "L")
    add("H5", "thermo_thsice.mix_old_area", "Qnet EmPmR saltFlux icFlxAtm icFrwAtm", INTERIOR, "L")
    add("H6", "thermo_thsice.thermal_step", thermal, INTERIOR, "L")
    add("B2", "thermo_thsice.thermal_step", thermal + " v2ocMxL", INTERIOR, "L")
    add("B3", "program.map_thsice", "AREA HEFF HSNOW " + " ".join(THSICE_ADVECTED), FULL)
    add("B4", "dynamics.dynamics_step", "TAUX TAUY", NORTH_EAST, records=("dyn_gate",))
    add("B5", "dynamics.lsr", records=("lsr_ctl", "lsr_chk", "lsr_end"))
    add("B6", "dynamics.dynamics_step", "fu fv DWATN UICE VICE", FULL)
    add("B7", "transport.thsice_transport_five", "iceFrc uTrans vTrans snowHeight iceHeight uTrIce vTrIce Qice1 Qice2 iceVol_in iceVol_out oldVol_in oldVol_out", FULL, "L")
    add("B8", "transport.thsice_do_advect", "iceMask iceHeight snowHeight Qice1 Qice2 snowAge oceQnet oceFWfx oceSflx Qnet EmPmR saltFlux sIceLoad", INTERIOR, "L")
    b8 = result["B8"]
    result["B8"] = StationSpec(b8.name, b8.boundary, tuple(StationField(f.name, f.shape, SOUTH_WEST if f.name in ("Qnet", "EmPmR", "saltFlux", "sIceLoad", "oceQnet", "oceFWfx", "oceSflx") else f.domain) for f in b8.fields))
    for station in ("T7", "H7", "B9.sf"):
        add(station, "host_single_layer_ocean.surface_forcing", "surfaceForcingT", wet="surfaceForcingT")
    for station in ("T8", "H8", "B9"):
        add(station, "host_single_layer_ocean.temperature_step", "theta gtNm1", full="theta")
    for station, count in (("T9", 14), ("L11", 13), ("H9", 17), ("B10", 19)):
        add(station, "pickup.read_pickup/write_pickup; program.restore", native=count)
    return result


STATIONS = _stations()

# B5 is a composite of the non-flex versions of the L3/L4/L6/L7 events.
STATION_EVENTS = {
    "B5.linearization": STATIONS["L3"], "B5.system": STATIONS["L4"],
    "B5.scan": STATIONS["L6"], "B5.end": STATIONS["L7"],
}
UNDEFINED_RECORD_FIELDS = {
    "B7.area": ("oldVol_in", "oldVol_out"),
    "B7.snow": ("oldVol_in", "oldVol_out"),
}
