"""Large-Pond EXF for the four pinned offline sea-ice configurations.

Fluxes retain MIT signs: heat/salt/freshwater are upward positive; freshwater
is kg/m2/s after mapping, evaporation is m/s, stress is N/m2. The spatial
service owns placement, neighbor access and every halo exchange.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from locomgcm.dynamics.bulk_formulae import EXF_BULK_DEFAULTS, exf_bulk_fluxes
from locomgcm.io.exf_source import FieldSeries, exf_interpolate, exf_record_interval
from ..records import Atmosphere, Geometry, OpenWaterFluxes
from ..space import EXCHANGE_EVENTS, SeaiceSpace

Scalar = float | torch.Tensor


@dataclass(frozen=True)
class ForcingData:
    uwind: FieldSeries
    vwind: FieldSeries
    atemp: FieldSeries
    aqh: FieldSeries
    lwdown: FieldSeries
    swdown: FieldSeries
    precip: FieldSeries
    snowPrecip: FieldSeries
    runoff: FieldSeries
    lwflux: FieldSeries
    swflux: FieldSeries
    saltflx: FieldSeries
    climsst: FieldSeries
    climsss: FieldSeries
    apressure: FieldSeries


@dataclass(frozen=True)
class ForcingSample:
    uwind: torch.Tensor
    vwind: torch.Tensor
    atemp: torch.Tensor
    aqh: torch.Tensor
    lwdown: torch.Tensor
    swdown: torch.Tensor
    precip: torch.Tensor
    snowPrecip: torch.Tensor
    runoff: torch.Tensor
    lwflux: torch.Tensor
    swflux: torch.Tensor
    saltflx: torch.Tensor
    climsst: torch.Tensor
    climsss: torch.Tensor
    apressure: torch.Tensor


@dataclass(frozen=True)
class ExfConfig:
    chain: str
    rho_const_fresh: Scalar
    cen2kel: Scalar
    ocean_emissivity: Scalar
    stefan_boltzmann: Scalar
    albedo: Scalar
    windstressmax: Scalar
    climtempfreeze: Scalar
    surf_p_ref: Scalar
    no_negative_evap: bool
    use_check_range: bool
    debug_level: int
    niter0: int
    start_time: float
    delta_t: float
    atemp_offset: Scalar
    hflux_scale: Scalar
    sflux_scale: Scalar
    ustress_scale: Scalar
    vstress_scale: Scalar
    swflux_scale: Scalar
    sst_scale: Scalar
    sss_scale: Scalar
    pressure_scale: Scalar
    heat_intercept: Scalar
    heat_slope: Scalar
    freshwater_intercept: Scalar
    freshwater_slope: Scalar
    compute_longwave: bool
    compute_shortwave: bool


@dataclass(frozen=True)
class ExfDiagnostics:
    """Pre-map monitor quantities; hflux/sflux have unexchanged halos."""
    hflux: torch.Tensor
    sflux: torch.Tensor
    hs: torch.Tensor
    hl: torch.Tensor
    lwflux: torch.Tensor
    swflux: torch.Tensor
    ustress_before_clip: torch.Tensor
    vstress_before_clip: torch.Tensor


@dataclass(frozen=True)
class ExfResult:
    atmosphere: Atmosphere
    open_water: OpenWaterFluxes
    diagnostics: ExfDiagnostics
    sample: ForcingSample
    exchange_events: tuple[str, ...]

    @property
    def exchange_metadata(self):
        """CPP/runtime gates of the actual X0-X3 event sequence."""
        return tuple(EXCHANGE_EVENTS[name] for name in self.exchange_events)


class ForcingFailure(ValueError):
    """The candidate failed EXF checks; no result was published."""


def _sample(series: FieldSeries, time: float, config: ExfConfig,
            geometry: Geometry, space: SeaiceSpace) -> torch.Tensor:
    # exf_init_fld.F:85-94 seeds all storage with fldConst, even on land.
    seed = torch.ones_like(geometry.maskC) * series.constant
    if not series.has_file:
        return seed
    if series.records.dtype != torch.float64:
        raise ValueError("forcing records must be float64 [record,ny,nx]")
    # The four reference inputs use period=0 and therefore record 1.
    count0, count1, weight = exf_record_interval(
        time, start_time=series.start_time, period=series.period,
        repeat_cycle=series.repeat_cycle, use_cal=False)
    # exf_filter_rl.F:62-70 clears dry input values before scaling/interpolation.
    def record(index):
        if index >= series.records.shape[0]:
            raise ValueError(f"missing forcing record {index}")
        value = space.interior(space.from_global(series.records[index]))
        return torch.where(space.interior(geometry.maskC) != 0, value, 0.) if series.center_mask else value

    if series.period == 0:
        value = series.input_scale * record(count0 - 1)
    else:
        # exf_set_fld.F:303-310. No trend removal in the period=0 init path.
        value = exf_interpolate(record(count0 - 1), record(count1 - 1), weight,
                                in_scale=series.input_scale,
                                remove_intercept=series.remove_intercept,
                                remove_slope=series.remove_slope,
                                elapsed=time - config.start_time)
    return space.replace_domain(seed, value)


def sample_atmosphere(data: ForcingData, config: ExfConfig, geometry: Geometry,
                      space: SeaiceSpace, *, time: float,
                      previous: ForcingSample | None = None) -> ForcingSample:
    """Sample step-start inputs, retaining existing bands when supplied.

    ``previous=None`` is cold initialization. Subsequent calls take the prior
    sample, with uwind/vwind replaced by the program's X4 versions when used.
    exf_set_fld.F:122-124 leaves period-zero fields untouched after init.
    """
    values = []
    for item in fields(data):
        series = getattr(data, item.name)
        if previous is not None and (not series.has_file or series.period == 0):
            value = getattr(previous, item.name).clone()
        else:
            value = _sample(series, time, config, geometry, space)
            if previous is not None:
                value = space.replace_domain(getattr(previous, item.name), space.interior(value))
        values.append(value)
    sample = ForcingSample(*values)
    # exf_getffields.F:203-205,316-321: offset after masking; P >= snow P.
    atemp = space.replace_domain(sample.atemp, space.interior(sample.atemp) + config.atemp_offset)
    precip = space.replace_domain(sample.precip, torch.maximum(space.interior(sample.precip), space.interior(sample.snowPrecip)))
    return ForcingSample(sample.uwind, sample.vwind, atemp, sample.aqh,
                         sample.lwdown, sample.swdown, precip, sample.snowPrecip,
                         sample.runoff, sample.lwflux, sample.swflux, sample.saltflx,
                         sample.climsst, sample.climsss, sample.apressure)


def _check_range(a: ForcingSample, d: ExfDiagnostics, wspeed: torch.Tensor,
                 geometry: Geometry, space: SeaiceSpace) -> None:
    # exf_check_range.F:48-192. Stress checks precede windstressmax clipping.
    checks = (
        ("hflux", d.hflux, geometry.maskC, -500., 1600.),
        ("sflux", d.sflux, geometry.maskC, -1.e-6, 1.e-6),
        ("ustress", d.ustress_before_clip, geometry.maskW, -2.7, 2.7),
        ("vstress", d.vstress_before_clip, geometry.maskS, -2.3, 2.3),
        ("uwind", a.uwind, geometry.maskW, -100., 100.),
        ("vwind", a.vwind, geometry.maskS, -100., 100.),
        ("wspeed", wspeed, geometry.maskS, 0., 100.),
        ("atemp", a.atemp, geometry.maskC, 183., 343.),
        ("aqh", a.aqh, geometry.maskC, 0., 0.1),
        ("precip", a.precip, geometry.maskC, 0., 2.e-6),
        ("snowPrecip", a.snowPrecip, geometry.maskC, 0., 2.e-6),
        ("runoff", a.runoff, geometry.maskC, 0., 1.e-6),
    )
    failures = []
    for name, value, mask, lower, upper in checks:
        value, wet = space.interior(value), space.interior(mask) != 0
        if bool((((value < lower) | (value > upper)) & wet).any()):
            failures.append(name)
    if failures:
        raise ForcingFailure("EXF_CHECK_RANGE: " + ", ".join(failures))


def exf_step(theta: torch.Tensor, sample: ForcingSample, config: ExfConfig,
             geometry: Geometry, space: SeaiceSpace, *, time: float, iteration: int) -> ExfResult:
    """T1/L1/H1/B1: return EXF_MAPFIELDS output, with X0-X3 complete.

    Inputs and returned records are immutable candidates. Qsw's retained band
    comes from sample.swflux; only its interior is overwritten by radiation.
    """
    space.validate(theta)
    for item in fields(sample):
        space.validate(getattr(sample, item.name))
    a = sample
    inside = space.interior
    write = space.replace_domain
    zero = torch.zeros_like(theta)
    # exf_getclim.F:88-113: floor first, then separate climatology exchange.
    climsst = write(a.climsst, torch.clamp_min(inside(a.climsst), config.climtempfreeze))
    climate = space.exchange("X0", {"climsst": climsst, "climsss": a.climsss}, chain=config.chain)
    # exf_radiation.F:60-86,107-150: compute net radiation only behind file gates.
    tsf = inside(theta) + config.cen2kel
    lwflux = (write(a.lwflux, config.ocean_emissivity * config.stefan_boltzmann * tsf**4
                    - inside(a.lwdown) * config.ocean_emissivity)
              if config.compute_longwave else a.lwflux.clone())
    swflux = (write(a.swflux, -inside(a.swdown) * (1.0 - config.albedo))
              if config.compute_shortwave else a.swflux.clone())
    # exf_wind.F:122-150,264-268; exf_bulkformulae.F:281-534. Air rho is 1.200.
    wspeed = write(zero, torch.sqrt(inside(a.uwind)**2 + inside(a.vwind)**2))
    bulk = exf_bulk_fluxes(inside(a.atemp), inside(a.aqh), inside(a.uwind), inside(a.vwind),
                           inside(a.precip), lwflux=inside(lwflux), swflux=inside(swflux),
                           runoff=inside(a.runoff), constants=EXF_BULK_DEFAULTS, sst_kelvin=tsf,
                           rho_const_fresh=config.rho_const_fresh,
                           no_negative_evap=config.no_negative_evap)
    hs, hl, evap = (write(zero, value) for value in (bulk.hs, bulk.hl, bulk.evap))
    # exf_getforcing.F:296-316. Radiation has its own gate, independent of atemp.
    hflux = write(zero, (-bulk.hs - bulk.hl + inside(lwflux) + inside(swflux)) * inside(geometry.maskC))
    sflux = write(zero, (bulk.evap - inside(a.precip) - inside(a.runoff)) * inside(geometry.maskC))
    stress = space.exchange("X1", {"ustress": write(zero, bulk.ustress),
                                   "vstress": write(zero, bulk.vstress)}, chain=config.chain)
    diag = ExfDiagnostics(hflux, sflux, hs, hl, lwflux, swflux, stress["ustress"], stress["vstress"])
    if config.use_check_range and (iteration == config.niter0 or config.debug_level >= 3):
        _check_range(a, diag, wspeed, geometry, space)
    # exf_mapfields.F:102-129: only freshwater gains rhoConstFresh, exactly once.
    qnet = config.hflux_scale * hflux - config.hflux_scale * (
        config.heat_intercept + config.heat_slope * (time - config.start_time))
    empmr = config.sflux_scale * sflux * config.rho_const_fresh - config.rho_const_fresh * (
        config.sflux_scale * (config.freshwater_intercept + config.freshwater_slope * (time - config.start_time)))
    # exf_mapfields.F:218-284: clip center stress, then average onto masked faces.
    ustress = torch.clamp(stress["ustress"], -config.windstressmax, config.windstressmax)
    vstress = torch.clamp(stress["vstress"], -config.windstressmax, config.windstressmax)
    ux, vy = (0, 0, 1, 0), (1, 0, 0, 0)
    fu = space.write(zero, config.ustress_scale * (space.read(ustress, ux) + space.read(ustress, ux, di=-1))
                     * 0.5 * space.read(geometry.maskW, ux), ux)
    fv = space.write(zero, config.vstress_scale * (space.read(vstress, vy) + space.read(vstress, vy, dj=-1))
                     * 0.5 * space.read(geometry.maskS, vy), vy)
    fluxes = space.exchange("X2", {"Qnet_ow": qnet, "EmPmR_ow": empmr,
                                   "fu_ow": fu, "fv_ow": fv}, chain=config.chain)
    climate = space.exchange("X3", {"SST": config.sst_scale * climate["climsst"],
                                    "SSS": config.sss_scale * climate["climsss"],
                                    "pLoad": config.pressure_scale * a.apressure - config.surf_p_ref}, chain=config.chain)
    # exf_mapfields.F:289-295,331-337,360-361: no Qsw or saltFlux exchange.
    output = OpenWaterFluxes(fluxes["Qnet_ow"], config.swflux_scale * swflux,
                            fluxes["EmPmR_ow"], a.saltflx.clone(), fluxes["fu_ow"], fluxes["fv_ow"],
                            climate["SST"], climate["SSS"], climate["pLoad"])
    atmosphere = Atmosphere(a.uwind, a.vwind, a.atemp, a.aqh, a.lwdown, a.swdown,
                            a.precip, a.snowPrecip, a.runoff, wspeed, evap, ustress, vstress)
    for record in (output, atmosphere):
        for item in fields(record):
            if not bool(torch.isfinite(getattr(record, item.name)).all()):
                raise ForcingFailure(f"nonfinite EXF candidate: {item.name}")
    return ExfResult(atmosphere, output, diag, sample, ("X0", "X1", "X2", "X3"))
