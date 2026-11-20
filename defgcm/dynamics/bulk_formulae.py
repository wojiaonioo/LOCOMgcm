"""Bulk surface-flux formulae (MITgcm ``pkg/exf``, Large & Pond 1981).

Computes the net
surface heat / shortwave / freshwater fluxes + wind stress from atmospheric
state + model SST, op-level aligned to MITgcm EXF **bulk mode** (default scheme:
``ALLOW_BULKFORMULAE`` defined, ``ALLOW_BULK_LARGEYEAGER04`` undef, no zenith /
atmrho-calc / sea-ice).  Grid-agnostic — pure per-cell ops over any spatial
shape (transfers from cube cs32 op-level to any grid, like KPP).

The functions cite the corresponding MITgcm routines and constant defaults.
Sign convention (docs/NUMERICS.md): all returned
fluxes are **MITgcm upward-positive** (``Qnet>0``/``Qsw>0`` cool the ocean); the
producer converts to LOCOMgcm convention at the data boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from defgcm.constants import RHO_CONST_DEFAULT

# --- EXF defaults (exf_readparms.F / EXF_CONSTANTS.h) ---
_NITER_BULK = 2                       # EXF_CONSTANTS.h:89 (PARAMETER, fixed)
_STEFAN = 5.670e-8                    # EXF_CONSTANTS.h:46
_KARMAN = 0.4                         # EXF_CONSTANTS.h:47
CEN2KEL = 273.15
_GRAVITY_MKS = 9.81
_ATMRHO = 1.200
_ATMCP = 1005.0
_FLAMB = 2.5e6
_CVAPOR_FAC = 640380.0
_CVAPOR_EXP = 5107.4
_HUMID_FAC = 0.606
_GAMMA_BLK = 0.010
_SALTSAT = 0.980
_CDRAG_1, _CDRAG_2, _CDRAG_3 = 0.0027, 0.000142, 0.0000764
_CSTANTON_1, _CSTANTON_2 = 0.0327, 0.0180
_CDALTON = 0.0346
_ZOLMIN = -100.0
_PSIM_FAC = 5.0
_EXF_SCAL_BULKCDN = 1.0
_ZREF, _HU, _HT, _UMIN = 10.0, 10.0, 2.0, 0.5
_OCEAN_EMISSIVITY = 5.50e-8 / 5.670e-8
_EXF_ALBEDO = 0.1
_PI = math.pi


@dataclass(frozen=True)
class ExfBulkConstants:
    """The data.exf EXF_NML_01 constants of :func:`exf_bulk_fluxes` (exf_readparms.F:80-85).

    Field names are the namelist names.  MITgcm sets ``hq = ht``
    (exf_readparms.F:1044), so the moisture height is ``ht``; ``stefanBoltzmann``
    and ``karman`` are EXF_CONSTANTS.h parameters, not namelist entries.
    """

    cen2kel: float
    gravity_mks: float
    atmrho: float
    atmcp: float
    flamb: float
    cvapor_fac: float
    cvapor_exp: float
    humid_fac: float
    gamma_blk: float
    saltsat: float
    cdrag_1: float
    cdrag_2: float
    cdrag_3: float
    cstanton_1: float
    cstanton_2: float
    cdalton: float
    zolmin: float
    psim_fac: float
    exf_scal_BulkCdn: float
    zref: float
    hu: float
    ht: float
    umin: float
    ocean_emissivity: float
    exf_albedo: float


# The MITgcm defaults (exf_readparms.F:326-370, exf_scal_BulkCdn :603); the
# kernel's constants for a source without data.exf (ERA5) and for callers
# that fix them.
EXF_BULK_DEFAULTS = ExfBulkConstants(
    cen2kel=CEN2KEL, gravity_mks=_GRAVITY_MKS, atmrho=_ATMRHO, atmcp=_ATMCP,
    flamb=_FLAMB, cvapor_fac=_CVAPOR_FAC, cvapor_exp=_CVAPOR_EXP,
    humid_fac=_HUMID_FAC, gamma_blk=_GAMMA_BLK, saltsat=_SALTSAT,
    cdrag_1=_CDRAG_1, cdrag_2=_CDRAG_2, cdrag_3=_CDRAG_3,
    cstanton_1=_CSTANTON_1, cstanton_2=_CSTANTON_2, cdalton=_CDALTON,
    zolmin=_ZOLMIN, psim_fac=_PSIM_FAC, exf_scal_BulkCdn=_EXF_SCAL_BULKCDN,
    zref=_ZREF, hu=_HU, ht=_HT, umin=_UMIN,
    ocean_emissivity=_OCEAN_EMISSIVITY, exf_albedo=_EXF_ALBEDO,
)

# ---------------------------------------------------------------------------
# Prep-boundary physics (the IO layer in defgcm/io/era5.py
# reads/regrids/interpolates only; every formula lives here).
# ---------------------------------------------------------------------------

# Legacy wind-only preprocessing keeps 1.225 for its prescribed-stress inputs.
# Coupled ERA5 wind and thermal forcing uses exf_bulk_fluxes instead, where
# _ATMRHO supplies both stress and heat/moisture transfer. The loader selects
# ERA5BulkSource for that path, bypassing this legacy stress conversion.
_RHO_AIR = 1.225  # kg m-3


def _drag_coefficient(wind_speed: torch.Tensor) -> torch.Tensor:
    """Large & Pond (1981) drag coefficient."""
    cd_low = 1.2e-3
    cd = torch.where(
        wind_speed < 11.0,
        torch.full_like(wind_speed, cd_low),
        (0.49 + 0.065 * wind_speed) * 1.0e-3,
    )
    return cd


def wind_speed_to_stress(
    u10: torch.Tensor,
    v10: torch.Tensor,
    *,
    rho_air: float = _RHO_AIR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert 10-m wind (u10, v10) to surface wind stress (tau_x, tau_y) in Pa."""
    speed = torch.sqrt(u10 * u10 + v10 * v10)
    cd = _drag_coefficient(speed)
    tau_x = rho_air * cd * speed * u10
    tau_y = rho_air * cd * speed * v10
    return tau_x, tau_y


# d2m + sp -> specific humidity (ECMWF Teten's formula over water, IFS docs
# Part IV).  Must match the MIT-side prep that builds the exf `aqh` file so the
# two models see the same humidity.
_TETEN_A1 = 611.21      # Pa
_TETEN_A3 = 17.502
_TETEN_A4 = 32.19       # K
_RDRY_OVER_RVAP = 0.621981


def specific_humidity_from_dewpoint(
    dewpoint_k: torch.Tensor, surface_pressure_pa: torch.Tensor
) -> torch.Tensor:
    """2-m specific humidity [kg/kg] from dewpoint [K] + surface pressure [Pa].

    ECMWF Teten's saturation vapour pressure over water::

        e = a1 * exp(a3 * (Td - 273.16) / (Td - a4))         [Pa]
        q = (Rdry/Rvap) * e / (p - (1 - Rdry/Rvap) * e)      [kg/kg]
    """
    e_sat = _TETEN_A1 * torch.exp(
        _TETEN_A3 * (dewpoint_k - 273.16) / (dewpoint_k - _TETEN_A4)
    )
    return _RDRY_OVER_RVAP * e_sat / (
        surface_pressure_pa - (1.0 - _RDRY_OVER_RVAP) * e_sat
    )


@dataclass(frozen=True)
class BulkFluxes:
    """All MITgcm upward-positive (Qnet/Qsw>0 cool); evap/empmr +ve into atmosphere."""
    qnet: torch.Tensor      # net surface heat flux [W/m^2]
    qsw: torch.Tensor       # net shortwave [W/m^2] (<=0 for downward/heating)
    empmr: torch.Tensor     # E - P freshwater flux [m/s]
    ustress: torch.Tensor   # zonal wind stress [N/m^2]
    vstress: torch.Tensor   # meridional wind stress [N/m^2]
    hs: torch.Tensor        # sensible heat [W/m^2], into ocean
    hl: torch.Tensor        # latent heat [W/m^2], into ocean
    evap: torch.Tensor      # evaporation [m/s]


def _signf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fortran ``SIGN(a,b) = |a|*sign(b)``, with ``b>=0 -> +`` (matches ``exf_half+SIGN``)."""
    return torch.where(b >= 0, a.abs(), -a.abs())


def exf_longwave(tsf: torch.Tensor, lwdown: torch.Tensor, *,
                 constants: ExfBulkConstants) -> torch.Tensor:
    """Net longwave ``lwflux`` [W/m^2, upward] from ``lwdown`` (exf_radiation.F:67-79).

    ``tsf`` is ``exf_Tsf`` [K].  EXF_LWDOWN_WITH_EMISSIVITY is ON by default
    (EXF_OPTIONS.h:206) => the downward LW is multiplied by emissivity too:
    lwflux = eps*sig*T^4 - eps*lwdown.  MITgcm forms it only where a lwdown
    file is given and a lwflux file is not (:60); otherwise lwflux keeps its
    EXF field (file or lwfluxconst).
    """
    return constants.ocean_emissivity * _STEFAN * tsf**4 - lwdown * constants.ocean_emissivity


def exf_shortwave(swdown: torch.Tensor, *, constants: ExfBulkConstants) -> torch.Tensor:
    """Net shortwave ``swflux`` [W/m^2, upward, <=0] from ``swdown`` (exf_radiation.F:139-144).

    Formed only where a swdown file is given and a swflux file is not (:107);
    otherwise swflux keeps its EXF field (file or swfluxconst).
    """
    return -(1.0 - constants.exf_albedo) * swdown


def exf_bulk_fluxes(
    atemp: torch.Tensor,
    aqh: torch.Tensor,
    uwind: torch.Tensor,
    vwind: torch.Tensor,
    precip: torch.Tensor,
    sst_degc: torch.Tensor | None = None,
    *,
    lwflux: torch.Tensor,
    swflux: torch.Tensor,
    runoff: torch.Tensor,
    constants: ExfBulkConstants,
    sst_kelvin: torch.Tensor | None = None,
    rho_const_fresh: float = RHO_CONST_DEFAULT,
    no_negative_evap: bool = False,
    use_atm_wind: bool = True,
    stress_x: torch.Tensor | None = None,
    stress_y: torch.Tensor | None = None,
) -> BulkFluxes:
    """Net surface fluxes from atmospheric state + model SST (MITgcm EXF bulk).

    Ports ``exf_bulkformulae.F`` (turbulent) + ``exf_getforcing.F:296-368`` (net
    assembly); the net radiation comes in from :func:`exf_longwave` /
    :func:`exf_shortwave` or the EXF lwflux/swflux fields (exf_radiation.F:60, :107).

    Parameters
    ----------
    atemp : Tensor ``[...]``  2 m air temperature [K] (``=0`` marks land/inactive).
    aqh : Tensor              2 m specific humidity [kg/kg] (ERA5 d2m→q in prep).
    uwind, vwind : Tensor     10 m wind components [m/s].
    lwflux, swflux : Tensor   net long/shortwave [W/m^2], MITgcm upward-positive.
    precip : Tensor           precipitation [m/s] (de-accumulated tp /3600).
    runoff : Tensor           river runoff [m/s], positive into the ocean: the EXF
        runoff file or ``runoffconst`` (``ALLOW_RUNOFF``, exf_getforcing.F:313), zero
        from the ERA5 sources.
    constants : ExfBulkConstants  the data.exf EXF_NML_01 constants of the formulae.
    sst_degc : Tensor         model SST [°C]  (converted internally to K via +cen2kel).
    rho_const_fresh : float   MITgcm ``rhoConstFresh`` [kg/m^3].  Defaults to
        ``RHO_CONST_DEFAULT`` (= rhoConst); MIT likewise defaults ``rhoConstFresh``
        to ``rhoConst`` (ini_parms.F:484) — pass the case ``rhoConst``.
    no_negative_evap : bool   clip ``delq=min(0,delq)`` (default off, matches MIT default).
    use_atm_wind : bool
        Solve momentum transfer from wind, or use prescribed cell-centre stress
        (MIT Large & Pond ``solve4Stress=useAtmWind``). The latter keeps
        ``ustar=sqrt(|stress|/atmrho)`` fixed during the thermal iteration.
    stress_x, stress_y : Tensor
        Required when ``use_atm_wind=False``; un-clipped cell-centre stress
        [N/m^2], before EXF_MAPFIELDS reconstructs the momentum face values.

    Returns
    -------
    BulkFluxes  (all MITgcm upward-positive; inactive cells ``atemp==0``
    have zero turbulent fluxes ``hs``/``hl``/``evap`` and wind-derived stress,
    while radiation, precipitation and runoff still enter ``qnet``/``qsw``/
    ``empmr``; prescribed stress is retained).
    """
    if not use_atm_wind and (stress_x is None or stress_y is None):
        raise ValueError("prescribed-stress bulk formulae require stress_x and stress_y")
    active = atemp != 0.0
    # Safe placeholders at inactive cells (atemp=0 would divide-by-zero in t0); zeroed at end.
    one = torch.ones_like(atemp)
    atemp_u = torch.where(active, atemp, 273.15 * one)
    # SST in K (= exf_Tsf): pass sst_kelvin to match MIT's dumped exf_Tsf bitwise
    # (no degC<->K round-trip), or sst_degc for the +cen2kel convention.
    # ``tsf`` holds a placeholder at inactive cells for the turbulent iteration.
    if sst_kelvin is not None:
        tsf = torch.where(active, sst_kelvin, constants.cen2kel * one)
    elif sst_degc is not None:
        tsf = torch.where(active, sst_degc, 0.0 * one) + constants.cen2kel
    else:
        raise ValueError("provide sst_degc or sst_kelvin")
    recip_rhoConstFresh = 1.0 / rho_const_fresh

    ztln = math.log(constants.ht / constants.zref)            # = log(0.2)
    czol = constants.hu * _KARMAN * constants.gravity_mks

    wspeed = torch.sqrt(uwind * uwind + vwind * vwind)
    sh = wspeed.clamp(min=constants.umin)

    # --- saturated specific humidity over sea + air-sea differences ---
    ssq = constants.saltsat * (constants.cvapor_fac * torch.exp(-constants.cvapor_exp / tsf)) / constants.atmrho
    deltap = atemp_u + constants.gamma_blk * constants.ht - tsf
    delq = aqh - ssq
    if no_negative_evap:
        delq = torch.minimum(torch.zeros_like(delq), delq)

    # --- initial guess (neutral 10 m coefficients) ---
    stable = 0.5 + _signf(torch.full_like(atemp_u, 0.5), deltap)
    if use_atm_wind:
        cdn = constants.exf_scal_BulkCdn * (constants.cdrag_1 / sh + constants.cdrag_2 + constants.cdrag_3 * sh)
        rdn = torch.sqrt(cdn)
        ustar = rdn * sh
        tau = torch.zeros_like(atemp_u)
    else:
        # exf_wind.F:167-171; exf_bulkformulae.F:324-334. Given stress
        # controls heat/evaporation too; do not solve a second stress from wind.
        wind_stress = torch.sqrt(stress_x * stress_x + stress_y * stress_y)
        ustar = torch.sqrt(wind_stress / constants.atmrho)
        ustar = torch.where(active, ustar, torch.ones_like(ustar))
        tau = torch.sqrt(wind_stress * constants.atmrho)
        rdn = torch.zeros_like(atemp_u)
    rhn = (1.0 - stable) * constants.cstanton_1 + stable * constants.cstanton_2
    ren = constants.cdalton
    tstar = rhn * deltap
    qstar = ren * delq
    rd = torch.zeros_like(atemp_u)

    # --- iterate (exf_bulkformulae.F:354-478), exactly niter_bulk=2 ---
    for _ in range(_NITER_BULK):
        t0 = atemp_u * (1.0 + constants.humid_fac * aqh)
        huol = (tstar / t0 + qstar / (1.0 / constants.humid_fac + aqh)) * czol / (ustar * ustar)
        huol = torch.maximum(huol, torch.full_like(huol, constants.zolmin))
        htol = huol * constants.ht / constants.hu
        stable = 0.5 + _signf(torch.full_like(huol, 0.5), huol)

        if use_atm_wind:
            xsq = torch.sqrt(torch.abs(1.0 - 16.0 * huol)).clamp(min=1.0)
            x = torch.sqrt(xsq)
            psimh = -constants.psim_fac * huol * stable + (1.0 - stable) * (
                torch.log((1.0 + 2.0 * x + xsq) * (1.0 + xsq) * 0.125)
                - 2.0 * torch.atan(x) + 0.5 * _PI
            )
        xsq = torch.sqrt(torch.abs(1.0 - 16.0 * htol)).clamp(min=1.0)
        psixh = -constants.psim_fac * htol * stable + (1.0 - stable) * (
            2.0 * torch.log(0.5 * (1.0 + xsq))
        )

        if use_atm_wind:
            usn = sh / (1.0 - rdn / _KARMAN * psimh)
            usm = usn.clamp(min=constants.umin)
            cdn = constants.exf_scal_BulkCdn * (constants.cdrag_1 / usm + constants.cdrag_2 + constants.cdrag_3 * usm)
            rdn = torch.sqrt(cdn)
            rd = rdn / (1.0 - rdn / _KARMAN * psimh)
            ustar = rd * sh
            tau = constants.atmrho * rd * wspeed
        rhn = (1.0 - stable) * constants.cstanton_1 + stable * constants.cstanton_2
        ren = constants.cdalton
        rh = rhn / (1.0 + rhn * (ztln - psixh) / _KARMAN)
        re = ren / (1.0 + ren * (ztln - psixh) / _KARMAN)
        qstar = re * delq
        tstar = rh * deltap

    # --- turbulent fluxes; atemp == 0 zeroes them (exf_bulkformulae.F:481-526) ---
    z = torch.zeros_like(atemp)
    hs = torch.where(active, constants.atmcp * tau * tstar, z)
    hl = torch.where(active, constants.flamb * tau * qstar, z)
    evap = torch.where(active, -recip_rhoConstFresh * tau * qstar, z)
    if use_atm_wind:
        tmpbulk = tau * rd
        ustress = torch.where(active, tmpbulk * uwind, z)
        vstress = torch.where(active, tmpbulk * vwind, z)
    else:
        ustress, vstress = stress_x, stress_y

    # --- net assembly (exf_getforcing.F:296-313): precipitation and runoff are
    # subtracted in every cell, outside the atemp test of the turbulent fluxes ---
    qnet = -hs - hl + lwflux + swflux                           # = hflux
    empmr = evap - precip - runoff                              # = sflux (:304, :313)
    return BulkFluxes(qnet=qnet, qsw=swflux, empmr=empmr, ustress=ustress, vstress=vstress,
                      hs=hs, hl=hl, evap=evap)
