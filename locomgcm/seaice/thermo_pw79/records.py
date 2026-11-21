"""PW79 records. G fields use native tile storage; TICES uses G plus one category.

Heat and salt ocean fluxes use MIT upward-positive signs. Ice/snow thicknesses
are grid-cell means in metres. No record implies an exchange of its halos.
"""
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class IceState:
    HEFF: Tensor
    HSNOW: Tensor
    AREA: Tensor
    TICES: Tensor


@dataclass(frozen=True)
class Inputs:
    atemp: Tensor
    aqh: Tensor
    lwdown: Tensor
    swdown: Tensor
    wspeed: Tensor
    precip: Tensor
    snowPrecip: Tensor
    evap: Tensor
    runoff: Tensor
    theta: Tensor
    salt: Tensor
    HEFFM: Tensor
    YC: Tensor
    Qnet: Tensor
    Qsw: Tensor
    EmPmR: Tensor
    saltFlux: Tensor
    sIceLoad: Tensor
    d_HEFFbyNEG: Tensor
    d_HSNWbyNEG: Tensor


@dataclass(frozen=True)
class Salinity:
    """SEAICE_VARIABLE_SALINITY: HSALT (g/m2, cell mean) and saltFluxAdjust
    (g/m2/s, upward positive) after REG_RIDGE (seaice_reg_ridge.F:302-307)."""
    HSALT: Tensor
    saltFluxAdjust: Tensor


@dataclass(frozen=True)
class SurfaceBudget:
    TICES: Tensor
    a_QbyATM_cover: Tensor
    a_QbyATM_open: Tensor
    a_QSWbyATM_cover: Tensor
    a_QSWbyATM_open: Tensor
    a_FWbySublim: Tensor
    iterations: Tensor
    residuals: tuple[Tensor, ...]


@dataclass(frozen=True)
class GrowthResult:
    ice: IceState
    Qnet: Tensor
    Qsw: Tensor
    EmPmR: Tensor
    saltFlux: Tensor
    sIceLoad: Tensor
    HSALT: Tensor | None
    SIatmQnt: Tensor
    SItflux: Tensor
    SIatmFW: Tensor
    surface: SurfaceBudget
    d_HEFFbyATMonOCN_cover: Tensor
    d_HEFFbyATMonOCN_open: Tensor
    d_HEFFbySublim: Tensor
    d_HSNWbySublim: Tensor
    d_HSNWbyATMonSNW: Tensor
    d_HSNWbyRAIN: Tensor
    d_HEFFbyFLOODING: Tensor
    r_FWbySublim: Tensor
    d_HEFFbyOCNonICE: Tensor
    d_HSNWbyOCNonSNW: Tensor


@dataclass(frozen=True)
class Parameters:
    """Explicit parameters of zero-layer PW79 without ITD.

    Defaults: seaice_readparms.F:400-552; overrides: input.thermo/data.seaice.
    EXF constants: exf_readparms.F:318-372, EXF_CONSTANTS.h:41-46. The
    switches without defaults are SEAICE_areaLossFormula,
    SEAICE_growMeltByConv, the CPP options SEAICE_VARIABLE_SALINITY,
    SEAICE_DISABLE_SUBLIM and SHORTWAVE_HEATING, useRealFreshWaterFlux,
    IMAX_TICE and SEAICE_saltFrac.
    """
    dt: float
    ocean_depth: float
    area_loss_formula: int
    grow_melt_by_conv: bool
    variable_salinity: bool
    disable_sublimation: bool
    shortwave_heating: bool
    real_freshwater_flux: bool
    solve4temp_iterations: int
    salt_fraction: float
    rho_ice: float = 910.
    rho_snow: float = 330.
    rho_water: float = 1030.
    rho_fresh: float = 1000.
    cp_water: float = 3986.
    latent: float = 334000.
    latent_evap: float = 2500000.
    kelvin: float = 273.15
    freeze_intercept: float = 0.
    freeze_slope: float = -.054
    area_regularization: float = 1.e-5
    ice_regularization: float = .05
    area_max: float = 1.
    category_pdf: float = 1.
    piston_velocity: float = 8.7854425e-5
    frazil_fraction: float = 1.
    mc_phee_taper: float = 0.
    new_ice_height: float = .2
    new_ice_height_south: float = .2
    salt_ice: float = 4.
    wind_floor: float = 1.e-10
    rho_air: float = 1.2
    cp_air: float = 1005.
    dalton: float = .00175
    ice_conductivity: float = 2.1656
    snow_conductivity: float = .31
    snow_cutoff: float = .15
    shortwave_fraction: float = .30
    wet_albedo_temperature: float = -.001
    dry_ice_albedo: float = .6
    wet_ice_albedo: float = .6
    dry_snow_albedo: float = .6
    wet_snow_albedo: float = .6
    dry_ice_albedo_south: float = .6
    wet_ice_albedo_south: float = .6
    dry_snow_albedo_south: float = .6
    wet_snow_albedo_south: float = .6
    ice_emissivity: float = .95
    snow_emissivity: float = .95
    stefan_boltzmann: float = 5.670e-8
    min_air_temperature: float = -50.
    min_longwave: float = 60.
