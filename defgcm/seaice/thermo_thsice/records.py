"""Explicit THSICE column records, in MIT units and sign conventions.

Horizontal fields retain the caller's native tile storage. ``owned`` is a
boolean write mask supplied by the space owner; no routine exchanges halos.
Temperatures are Celsius except atmospheric ``atemp`` (Kelvin). Heat returned
to the ocean is upward positive; atmospheric and ice-bottom heat is downward
positive. Freshwater and salt returned to the ocean are upward positive.
"""
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class IceState:
    iceMask: Tensor
    iceHeight: Tensor
    snowHeight: Tensor
    Tsrf: Tensor
    Tice1: Tensor
    Tice2: Tensor
    Qice1: Tensor
    Qice2: Tensor
    snowAge: Tensor


@dataclass(frozen=True)
class Atmosphere:
    atemp: Tensor
    aqh: Tensor
    lwdown: Tensor
    swdown: Tensor
    wspeed: Tensor
    precip: Tensor
    snowPrecip: Tensor
    runoff: Tensor


@dataclass(frozen=True)
class OceanFlux:
    Qnet: Tensor
    Qsw: Tensor
    EmPmR: Tensor
    saltFlux: Tensor
    sIceLoad: Tensor


@dataclass(frozen=True)
class OceanSample:
    hOceMxL: Tensor
    tOceMxL: Tensor
    sOceMxL: Tensor
    v2ocMxL: Tensor


@dataclass(frozen=True)
class MappedInputs:
    ocean: OceanSample
    prcAtm: Tensor
    snowPrc: Tensor
    qPrcRn: Tensor
    icFlxSW: Tensor


@dataclass(frozen=True)
class Iteration:
    iceFlag: Tensor
    dTsrf: Tensor
    Tsrf: Tensor
    Tice1: Tensor


@dataclass(frozen=True)
class SolveState:
    iterations: Tensor
    active: Tensor
    dTsrf: Tensor
    history: tuple[Iteration, ...]


@dataclass(frozen=True)
class TemperatureStep:
    ice: IceState
    flux: OceanFlux
    siceAlb: Tensor
    sHeating: Tensor
    flxCndBt: Tensor
    icFlxAtm: Tensor
    icFrwAtm: Tensor
    icFlxSW: Tensor
    solve: SolveState


@dataclass(frozen=True)
class ThicknessStep:
    ice: IceState
    iceFrac: Tensor
    icFrwAtm: Tensor
    frzmltMxL: Tensor
    flx2oc: Tensor
    frw2oc: Tensor
    fsalt: Tensor
    frzSeaWat: Tensor


@dataclass(frozen=True)
class MixedFlux:
    flux: OceanFlux
    icFlxAtm: Tensor
    icFrwAtm: Tensor


@dataclass(frozen=True)
class ThermalStep:
    ice: IceState
    flux: OceanFlux
    flx2oc: Tensor
    frw2oc: Tensor
    fsalt: Tensor
    mapped: MappedInputs
    temperature: TemperatureStep
    thickness: ThicknessStep
    mixed: MixedFlux


@dataclass(frozen=True)
class Parameters:
    """Values for offline_exf_seaice H/B; dt values are required inputs.

    Physical defaults: thsice_readparms.F:99-140, exf_readparms.F:318-372.
    Case overrides: data.ice THSICE_CONST (fraction, snow slope, albedos).
    Scalar tensor parameters are also accepted and retain their derivatives.
    """
    dt: float
    dt_temperature: float
    dt_ocean: float
    rho_ice: float = 900.
    rho_snow: float = 330.
    rho_water: float = 1030.
    rho_fresh: float = 1000.
    cp_ice: float = 2106.
    cp_water: float = 3986.
    k_ice: float = 2.03
    k_snow: float = .30
    latent: float = 334000.
    mu: float = .054
    salinity_internal: float = 1.
    salt_ice: float = 4.
    h_min: float = .01
    h_max: float = 10.
    snow_max: float = 10.
    area_min: float = .001
    area_max: float = 1.
    h_thin: float = .2
    h_thick: float = 2.5
    h_new_max: float = 1.
    snow_slope: float = .1
    melt_fraction: float = .4
    freeze_fraction: float = .4
    basal_melt: float = .006
    shortwave_fraction: float = .3
    solar_attenuation: float = 1.5
    tolerance: float = .5
    max_iterations: int = 20
    albedo_ice_max: float = .6
    albedo_ice_min: float = .6
    albedo_cold_snow: float = .85
    albedo_warm_snow: float = .70
    albedo_old_snow: float = .55
    albedo_snow_temperature: float = -10.
    albedo_ice_height: float = .5
    albedo_snow_height: float = .3
    new_snow_age_height: float = .002
    snow_age_time: float = 4320000.
    kelvin: float = 273.15
    rho_air: float = 1.2
    cp_air: float = 1005.
    ice_ce: float = .00163
    ice_ch: float = .00163
    latent_evap: float = 2500000.
    vapor_factor: float = 11637800.
    vapor_exponent: float = 5897.8
    lapse: float = .010
    temperature_height: float = 2.
    wind_floor: float = .5
    snow_emissivity: float = .95
    ice_emissivity: float = .95
    stefan_boltzmann: float = 5.670e-8
