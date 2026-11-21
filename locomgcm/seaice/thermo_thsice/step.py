"""THSICE_MAIN thermodynamics in the GET_OCEAN, TEMP, FWD source order."""
from dataclasses import fields, replace

import torch
from torch import Tensor

from .records import (Atmosphere, IceState, MappedInputs, MixedFlux, OceanFlux,
                      OceanSample, Parameters, TemperatureStep, ThermalStep,
                      ThicknessStep)
from .temperature import step_temperature
from .thickness import calc_thickness, extend_ice
from locomgcm.seaice._arithmetic import divide


def get_ocean(*, hfac: Tensor, depth: Tensor | float, theta: Tensor, salt: Tensor,
              u: Tensor, u_east: Tensor, v: Tensor, v_north: Tensor,
              uice: Tensor, uice_east: Tensor, vice: Tensor, vice_north: Tensor,
              velocity_domain: Tensor, use_seaice: bool) -> OceanSample:
    """GET_OCEAN using space-provided neighbor values and write domain.

    Depth is positive downward in metres. In B the ice velocities must be
    the last published velocities. Space supplies the east/north samples
    from each tile's own storage, and trim[0,1,0,1] as velocity_domain.
    """
    # thsice_get_ocean.F:84-145: different association in H and B is deliberate.
    if use_seaice:
        ur1, ur2, vr1, vr2 = u-uice, u_east-uice_east, v-vice, v_north-vice_north
        speed2 = ((ur1*ur1+ur2*ur2)+(vr1*vr1+vr2*vr2))*.5
    else:
        speed2 = (u*u+u_east*u_east+v_north*v_north+v*v)*.5
    return OceanSample(depth*hfac, theta.clone(), salt.clone(), torch.where(velocity_domain, speed2, 0.))


def map_inputs(ice: IceState, a: Atmosphere, ocean: OceanSample,
               p: Parameters, owned: Tensor) -> MappedInputs:
    """H2 boundary for no snow file and unset rain/runoff temperature."""
    # thsice_map_exf.F:67-117; snowfall is diagnosed from precipitation.
    prc = a.precip*p.rho_fresh
    prc = prc+a.runoff*p.rho_fresh
    snow = torch.where(owned & (ice.iceMask > 0) & (a.atemp <= p.kelvin),
                       a.precip*p.rho_fresh, 0.)
    return MappedInputs(ocean, torch.where(owned, prc, 0.), snow,
                        torch.zeros_like(prc), torch.where(owned, a.swdown, 0.))


def mix_old_area(temp: TemperatureStep, thick: ThicknessStep,
                  mapped: MappedInputs, p: Parameters, owned: Tensor) -> MixedFlux:
    """H5 boundary, before clearing local fluxes and calling EXTEND."""
    # thsice_step_fwd.F:257-300: read the unchanged iceMask, not iceFrac.
    a = temp.ice.iceMask
    icy = owned & (a > 0)
    op = 1-a
    fa = temp.icFlxAtm-p.latent*mapped.snowPrc+mapped.qPrcRn
    wet = owned & (mapped.ocean.hOceMxL > 0)
    atm_heat = torch.where(icy, a*fa-op*temp.flux.Qnet, torch.where(wet, -temp.flux.Qnet, 0.))
    atm_water = torch.where(icy, a*thick.icFrwAtm+op*temp.flux.EmPmR,
                            torch.where(wet, temp.flux.EmPmR, 0.))
    flux = replace(temp.flux,
                   Qnet=torch.where(icy, -a*thick.flx2oc+op*temp.flux.Qnet, temp.flux.Qnet),
                   EmPmR=torch.where(icy, -a*thick.frw2oc+op*temp.flux.EmPmR, temp.flux.EmPmR),
                   saltFlux=torch.where(icy, -a*thick.fsalt, 0.))
    return MixedFlux(flux, atm_heat, atm_water)


def step_thermodynamics(ice: IceState, a: Atmosphere, ocean: OceanSample,
                        flux: OceanFlux, p: Parameters, owned: Tensor, *,
                        advects_ice: bool) -> ThermalStep:
    """H6/B2 boundary, with explicit H2-H5 intermediates for station sampling.

    advects_ice selects the source thSIceAdvScheme>0 loading gate. The caller
    publishes the candidate and performs any X9 exchange outside this routine.
    """
    for record in (ice, a, ocean, flux):
        for field in fields(record):
            value = getattr(record, field.name)
            if value.dtype != torch.float64 or value.shape != owned.shape:
                raise ValueError(f"{field.name} must be float64 with the owned-mask shape")
    if owned.dtype != torch.bool:
        raise TypeError("owned must be a boolean spatial mask")
    mapped = map_inputs(ice, a, ocean, p, owned)
    temp = step_temperature(ice, a, mapped, flux, p, owned)
    # thsice_step_fwd.F:131-202: snow age is advanced after temperature.
    age = p.dt+ice.snowAge*(1-divide(p.dt, p.snow_age_time))
    age = torch.where(mapped.snowPrc > 0,
                       age*torch.exp(divide(-p.dt, p.rho_snow*p.new_snow_age_height)*mapped.snowPrc), age)
    aged_ice = replace(temp.ice, snowAge=torch.where(owned & (ice.iceMask > 0), age, ice.snowAge))
    thick = calc_thickness(replace(temp, ice=aged_ice), mapped, p, owned)
    mixed = mix_old_area(temp, thick, mapped, p, owned)
    # thsice_step_fwd.F:306-342: EXTEND starts with three zero flux buffers.
    ext = extend_ice(thick, mapped, p, owned)
    newflux = replace(mixed.flux,
                      Qnet=torch.where(owned, mixed.flux.Qnet-ext.flx2oc, mixed.flux.Qnet),
                      EmPmR=torch.where(owned, mixed.flux.EmPmR-ext.frw2oc, mixed.flux.EmPmR),
                      saltFlux=torch.where(owned, mixed.flux.saltFlux-ext.fsalt, mixed.flux.saltFlux))
    # thsice_step_fwd.F:360-406: no-ice enthalpy is Lfresh, surface is old SST.
    present = owned & (ext.iceFrac > 0)
    absent = owned & ~present
    end = replace(ext.ice,
                  iceMask=torch.where(owned, torch.where(present, ext.iceFrac, 0.), ice.iceMask),
                  iceHeight=torch.where(absent, 0., ext.ice.iceHeight),
                  snowHeight=torch.where(absent, 0., ext.ice.snowHeight),
                  snowAge=torch.where(absent | (present & (ext.ice.snowHeight == 0)), 0., ext.ice.snowAge),
                  Tsrf=torch.where(absent, ocean.tOceMxL, ext.ice.Tsrf),
                  Tice1=torch.where(absent, 0., ext.ice.Tice1), Tice2=torch.where(absent, 0., ext.ice.Tice2),
                  Qice1=torch.where(absent, p.latent, ext.ice.Qice1), Qice2=torch.where(absent, p.latent, ext.ice.Qice2))
    if not advects_ice:
        load = (end.snowHeight*p.rho_snow+end.iceHeight*p.rho_ice)*end.iceMask
        newflux = replace(newflux, sIceLoad=torch.where(owned, load, flux.sIceLoad))
    return ThermalStep(end, newflux, ext.flx2oc, ext.frw2oc, ext.fsalt, mapped, temp, thick, mixed)
