"""THSICE_ALBEDO, GET_EXF and finite-iteration SOLVE4TEMP."""
from dataclasses import replace

import torch
from torch import Tensor
from defgcm.seaice._arithmetic import divide

from .records import (Atmosphere, IceState, Iteration, MappedInputs, OceanFlux,
                      Parameters, SolveState, TemperatureStep)


def albedo(ice: IceState, p: Parameters, owned: Tensor) -> Tensor:
    """thsice_albedo.F:113-195, no independent near-infrared branch.

    The [0.2, 1] range is checked by ``step_temperature`` with its other guards.
    """
    ai = p.albedo_ice_max + (p.albedo_ice_min-p.albedo_ice_max)*torch.exp(
        -ice.iceHeight/p.albedo_ice_height)
    an = p.albedo_cold_snow + (p.albedo_warm_snow-p.albedo_cold_snow)*(
        1-ice.Tsrf/p.albedo_snow_temperature).clamp(0., 1.)
    ass = p.albedo_old_snow + (an-p.albedo_old_snow)*torch.exp(
        -.2*ice.snowAge/86400.)
    return torch.where(owned & (ice.iceMask > 0),
                       ass+(ai-ass)*torch.exp(-ice.snowHeight/p.albedo_snow_height), 0.)


@torch.compiler.disable(reason="one host read for the three THSICE_SOLVE4TEMP guards")
def _require(thin: Tensor, albedo_range: Tensor, too_cold: Tensor) -> None:
    """Raise in source order: thickness entry, albedo range, iteration limit."""
    flags = torch.stack((thin.any(), albedo_range.any(), too_cold.any())).tolist()
    if flags[0]:
        raise ValueError("THSICE_SOLVE4TEMP entered with ice thinner than hIceMin")
    if flags[1]:
        raise ValueError("THSICE_ALBEDO outside [0.2, 1]")
    if flags[2]:
        raise ValueError("THSICE_SOLVE4TEMP reached iteration limit below -70 C")


def atmospheric_flux(temperature: Tensor, snow: Tensor, a: Atmosphere,
                     p: Parameters) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Downward non-solar heat and upward evaporation, with derivatives."""
    # thsice_get_exf.F:183-259,436-499; fixed-density, no stability branch.
    emiss = torch.where(snow > .3, torch.as_tensor(p.snow_emissivity, dtype=snow.dtype, device=snow.device), p.ice_emissivity)
    tk = temperature+p.kelvin
    tk2 = tk*tk
    # Explicit division preserves the source exponential's argument; Python's
    # scalar / Tensor overload first forms a reciprocal and then multiplies.
    qsat = p.vapor_factor*torch.exp(torch.div(-p.vapor_exponent, tk))/p.rho_air
    dqsat = qsat*p.vapor_exponent/tk2
    w = torch.clamp_min(a.wspeed, p.wind_floor)
    tau = p.rho_air*p.ice_ce*w
    evap = -tau*(a.aqh-qsat)
    devap = tau*dqsat
    sensible = p.cp_air*p.rho_air*p.ice_ch*w*(a.atemp+p.lapse*p.temperature_height-tk)
    latent = -(p.latent_evap+p.latent)*evap
    heat = emiss*a.lwdown-emiss*p.stefan_boltzmann*tk2*tk2
    derivative = -emiss*p.stefan_boltzmann*tk2*tk*4
    derivative = derivative-p.cp_air*p.rho_air*p.ice_ch*w-(p.latent_evap+p.latent)*devap
    heat = heat+sensible+latent
    return tuple(torch.where(a.atemp > 0, v, 0.) for v in (heat, derivative, evap, devap))


def step_temperature(ice: IceState, a: Atmosphere, mapped: MappedInputs,
                     flux: OceanFlux, p: Parameters, owned: Tensor) -> TemperatureStep:
    """Return H3, including H3.k snapshots and per-point iteration counts.

    Tile counts are ``solve.iterations.amax((-2, -1))`` on native storage.
    All stopped points keep their final flux caches for the linear correction.
    Every point runs the masked fixed count of max_iterations sweeps: a stopped
    point is not updated again, so its count and values are those of the
    source loop, and ``history`` holds the state after each of these sweeps.
    """
    icy = owned & (ice.iceMask > 0)
    thin = icy & (ice.iceHeight < p.h_min)
    alb = albedo(ice, p, owned)
    albedo_range = icy & ((alb < .2) | (alb > 1.))
    # thsice_step_temp.F:110-187; safe inactive operands do not alter icy cells.
    H = torch.where(icy, ice.iceHeight, 1.)
    S = torch.where(icy, ice.snowHeight, 0.)
    A = torch.where(icy, ice.iceMask, 1.)
    sw = torch.where(icy, mapped.icFlxSW*(1-alb), 0.)
    tf = -p.mu*mapped.ocean.sOceMxL
    tm = -p.mu*p.salinity_internal
    # thsice_solve4temp.F:251-342.
    fs = S*divide(1., p.snow_slope)/A
    positive_snow = fs > 0
    fs = torch.where(positive_snow, torch.sqrt(torch.where(positive_snow, fs, 1.)), fs)
    fs = torch.where(S > A*p.snow_slope, 1., fs)
    pen = sw*(1-fs)*p.shortwave_fraction
    bottom_sw = pen*torch.exp(-p.solar_attenuation*H)
    internal_sw, surface_sw = pen-bottom_sw, sw-pen
    k12 = torch.div(4*p.k_ice*p.k_snow, p.k_snow*H+4*p.k_ice*S)
    k32 = torch.div(2*p.k_ice, H)
    b = ice.Qice1+(p.cp_water-p.cp_ice)*tm-p.latent
    t1 = .5*(-b-torch.sqrt(b*b-4*p.cp_ice*(p.latent*tm)))/p.cp_ice
    old_t2 = (p.latent-ice.Qice2)/p.cp_ice
    dt = p.dt_temperature
    rcH = p.rho_ice*p.cp_ice*H
    den = 6*dt*k32+rcH
    a10 = rcH/(2*dt)+k32*(4*dt*k32+rcH)/den
    b10 = -H*(p.rho_ice*p.cp_ice*t1+torch.div(p.rho_ice*p.latent*tm, t1))/(2*dt)
    b10 = b10-k32*(4*dt*k32*tf+rcH*old_t2)/den-internal_sw
    c10 = p.rho_ice*p.latent*H*tm/(2*dt)
    ts = ice.Tsrf
    zero = torch.zeros_like(ts)
    f0, _, e0, _ = atmospheric_flux(zero, S, a, p)
    f, df, e, de = zero, zero, zero, zero
    active = icy
    count = torch.zeros_like(ts, dtype=torch.int32)
    dts = torch.where(owned, torch.as_tensor(p.tolerance, dtype=ts.dtype, device=ts.device), 0.)
    history = []
    # thsice_solve4temp.F:392-543: each sweep freezes newly converged points.
    for _ in range(p.max_iterations):
        nf, ndf, ne, nde = atmospheric_flux(ts, S, a, p)
        f, df, e, de = (torch.where(active, new, old)
                       for new, old in zip((nf, ndf, ne, nde), (f, df, e, de)))
        net = surface_sw+f
        aa = a10-k12*df/(k12-df)
        bb = b10-k12*(net-df*ts)/(k12-df)
        nt1 = -(bb+torch.sqrt(bb*bb-4*aa*c10))/(2*aa)
        ndts = (net+k12*(nt1-ts))/(k12-df)
        tmp = ts+ndts
        melting = active & (tmp > 0)
        melt_a = a10+k12
        nt1 = torch.where(melting, (-b10-torch.sqrt(b10*b10-4*melt_a*c10))/(2*melt_a), nt1)
        t1 = torch.where(active, nt1, t1)
        ts = torch.where(active, torch.where(melting, 0., tmp), ts)
        dts = torch.where(active, torch.where(melting, 0., ndts), dts)
        f = torch.where(melting, f0, f)
        e = torch.where(melting, e0, e)
        count = count+active.to(torch.int32)
        active = active & (torch.abs(dts) >= p.tolerance)
        history.append(Iteration(active.to(torch.float64), dts, ts, t1))
    _require(thin, albedo_range, active & (ts < -70.))
    # thsice_solve4temp.F:554-645. Do not refresh F at the final temperature.
    t2 = (2*dt*k32*(t1+2*tf)+rcH*old_t2)/den
    ev = e+de*dts
    fa = sw+f+df*dts+ev*p.latent
    heat = surface_sw+f+df*dts-k12*(ts-t1)
    cb = 4*p.k_ice*(t2-tf)/H
    q1 = -p.cp_water*tm+p.cp_ice*(tm-t1)+p.latent*(1-torch.div(tm, t1))
    q2 = -p.cp_ice*t2+p.latent
    out = replace(ice, Tsrf=torch.where(icy, ts, ice.Tsrf),
                  Tice1=torch.where(icy, t1, ice.Tice1), Tice2=torch.where(icy, t2, ice.Tice2),
                  Qice1=torch.where(icy, q1, ice.Qice1), Qice2=torch.where(icy, q2, ice.Qice2))
    bottom_sw = torch.where(icy, bottom_sw, mapped.icFlxSW)
    # thsice_step_temp.F:194-203, old iceMask weighting.
    qsw = torch.where(icy, (1-ice.iceMask)*flux.Qsw-ice.iceMask*bottom_sw, flux.Qsw)
    return TemperatureStep(out, replace(flux, Qsw=qsw), alb,
                           torch.where(icy, heat, 0.), torch.where(icy, cb, 0.),
                           torch.where(icy, fa, 0.), torch.where(icy, ev, 0.), bottom_sw,
                           SolveState(count, active, torch.where(icy, dts, 0.), tuple(history)))
