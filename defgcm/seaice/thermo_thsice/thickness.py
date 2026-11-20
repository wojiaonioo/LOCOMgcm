"""THSICE two-layer inventory and new-ice formation, without halo exchange."""
from dataclasses import replace

import torch
from torch import Tensor

from .records import IceState, MappedInputs, Parameters, TemperatureStep, ThicknessStep
from defgcm.seaice._arithmetic import divide


def _div(n: Tensor, d: Tensor, active: Tensor) -> Tensor:
    """Evaluate a guarded Fortran division only with valid active operands."""
    return torch.where(active, n, 0.)/torch.where(active, d, 1.)


def calc_thickness(temp: TemperatureStep, mapped: MappedInputs,
                   p: Parameters, owned: Tensor) -> ThicknessStep:
    """H4 boundary; iceMask remains the input area, iceFrac is the new area."""
    ice = temp.ice
    A0, H, S = ice.iceMask, ice.iceHeight, ice.snowHeight
    A, ts, q1, q2 = A0, ice.Tsrf, ice.Qice1, ice.Qice2
    icy = owned & (A0 > 0)
    zero = torch.zeros_like(H)
    dt, ri, rs, lf = p.dt, p.rho_ice, p.rho_snow, p.latent
    tf = -p.mu*mapped.ocean.sOceMxL
    G = (tf-mapped.ocean.tOceMxL)*(p.cp_water*p.rho_water*mapped.ocean.hOceMxL)/p.dt_ocean
    # thsice_calc_thickn.F:287-358: THSICE_FRACEN_POWERLAW, powerLawExp2=2.
    def fraction(f: float) -> Tensor:
        c1, c2 = f**.2, (1-f)**.2
        coef = divide(c1+c2, p.h_thick-p.h_thin)
        href = p.h_thin+divide(c2, coef)
        xx = coef*(H-href)
        tmp = xx*xx
        tmp = tmp*tmp
        return (f-xx*tmp).clamp(0., 1.)
    emax, fmax = fraction(p.melt_fraction), fraction(p.freeze_fraction)
    # thsice_calc_thickn.F:374-500: ocean heat, initial masses, energy split.
    v2 = mapped.ocean.v2ocMxL
    ustar = torch.where(v2 != 0, torch.sqrt(torch.where(v2 != 0, .00536*v2, 1.)), .005)
    ustar = torch.clamp_min(ustar, .005)
    fbm = p.cp_water*p.rho_water*p.basal_melt*(tf-mapped.ocean.tOceMxL)*ustar
    fbm = torch.minimum(torch.maximum(fbm, G), zero)
    fb = torch.where(G >= 0, torch.where(A < p.area_max, fmax*G, G), fbm)
    m0, salt0 = rs*S+ri*H, ri*H*p.salt_ice
    frace = (A-p.area_min*1.01)/(p.area_min*1.10-p.area_min)
    frace = torch.minimum(emax, torch.clamp_min(frace, 0.))
    etop = torch.where(temp.sHeating > 0, (1-frace)*temp.sHeating*dt, 0.)
    etope = torch.where(temp.sHeating > 0, frace*temp.sHeating*dt, 0.)
    surplus = torch.where(temp.sHeating > 0, 0., temp.sHeating*dt)
    eb = (temp.flxCndBt-fb)*dt
    ebote = torch.where(eb > 0, frace*eb, 0.)
    eb = torch.where(eb > 0, eb-ebote, eb)
    h1, h2 = H*.5, H*.5
    evap = temp.icFrwAtm
    heat = temp.icFlxSW+mapped.qPrcRn

    def melt(energy: Tensor, height: Tensor, rq: Tensor | float,
             gate: Tensor) -> tuple[Tensor, Tensor]:
        cap = rq*height
        partial = gate & (energy < cap)
        d = torch.zeros_like(height)+rq
        new_h = torch.where(partial, height-_div(energy, d, partial), 0.)
        new_e = torch.where(partial, 0., energy-cap)
        return torch.where(gate, new_e, energy), torch.where(gate, new_h, height)

    # thsice_calc_thickn.F:518-604: top snow, upper ice, then lower ice.
    etop, S = melt(etop, S, rs*lf, icy & (etop > 0) & (S > 0))
    etop, h1 = melt(etop, h1, ri*q1, icy & (etop > 0))
    etop, h2 = melt(etop, h2, ri*q2, icy & (etop > 0))
    # thsice_calc_thickn.F:608-710: basal growth and reverse-order melting.
    grow = icy & (eb < 0)
    qb = -p.cp_ice*tf+lf
    dh = _div(-eb, qb*ri, grow)
    q2 = torch.where(grow, _div(h2*q2+dh*qb, h2+dh, grow), q2)
    h2 = torch.where(grow, h2+dh, h2)
    formed = torch.where(grow, ri*dh/dt, 0.)
    eb = torch.where(grow, 0., eb)
    eb, h2 = melt(eb, h2, ri*q2, icy & (eb > 0) & (h2 > 0))
    eb, h1 = melt(eb, h1, ri*q1, icy & (eb > 0) & (h1 > 0))
    eb, S = melt(eb, S, rs*lf, icy & (eb > 0) & (S > 0))
    H = torch.where(icy, h1+h2, H)
    thin = icy & (H < p.h_min) & (H+S > 0)
    surplus = torch.where(thin, surplus-rs*lf*S-ri*q1*h1-ri*q2*h2, surplus)
    H, S, ts, A, q1, q2 = (torch.where(thin, 0., x) for x in (H, S, ts, A, q1, q2))
    # thsice_calc_thickn.F:714-853: freshwater before snowfall/sublimation.
    fw = (m0-(rs*S+ri*H))/dt
    gone = icy & (H <= 0)
    fw = torch.where(gone, fw+mapped.snowPrc, fw)
    heat = torch.where(gone, heat-mapped.snowPrc*lf, heat)
    remain = icy & (H > 0)
    S = torch.where(remain, S+dt*mapped.snowPrc/rs, S)
    sn = remain & (S > 0)
    all_snow = sn & (evap/rs*dt > S)
    new_evap = torch.where(all_snow, evap-S*rs/dt, 0.)
    new_snow = torch.where(all_snow, 0., S-evap/rs*dt)
    evap = torch.where(sn, new_evap, evap)
    S = torch.where(sn, new_snow, S)

    def sublime(h: Tensor, q: Tensor, ev: Tensor, es: Tensor):
        gate = remain & (ev > 0)
        dh = ev/ri*dt
        full = gate & (dh >= h)
        part = gate & ~full
        nq = torch.where(part, _div(h*q-dh*lf, h-dh, part), q)
        ne = torch.where(gate, torch.where(full, ev-h*ri/dt, 0.), ev)
        ns = torch.where(full, es-h*ri*(q-lf), es)
        nh = torch.where(gate, torch.where(full, 0., h-dh), h)
        return nh, nq, ne, ns

    h1, q1, evap, surplus = sublime(h1, q1, evap, surplus)
    h2, q2, evap, surplus = sublime(h2, q2, evap, surplus)
    H = torch.where(remain, h1+h2, H)
    thin = remain & (H > 0) & (H < p.h_min)
    fw = torch.where(thin, fw+(rs*S+ri*H)/dt, fw)
    surplus = torch.where(thin, surplus-rs*lf*S-ri*q1*h1-ri*q2*h2, surplus)
    H, S, ts, A, q1, q2 = (torch.where(thin, 0., x) for x in (H, S, ts, A, q1, q2))
    # thsice_calc_thickn.F:858-979: flooding, upper bound, layer reshaping.
    reshape = icy & (H > 0)
    flood_factor = divide(p.rho_water-ri, rs)
    flooded = reshape & ((S > H*flood_factor) | (S > p.snow_max))
    ds = torch.maximum((S-H*flood_factor)*ri/p.rho_water, S-p.snow_max)
    di = ds*rs/ri
    q1 = torch.where(flooded, _div(ri*q1*h1+rs*lf*ds, ri*(h1+di), flooded), q1)
    h1 = torch.where(flooded, h1+di, h1)
    H = torch.where(flooded, H+di, H)
    S = torch.where(flooded, S-ds, S)
    capped = reshape & (H > p.h_max)
    chi = H-p.h_max
    h1 = torch.where(capped, h1-chi/2, h1)
    h2 = torch.where(capped, h2-chi/2, h2)
    fw = torch.where(capped, fw+chi*ri/dt, fw)
    H = torch.where(reshape, h1+h2, H)
    half = H*.5
    top_heavy = reshape & (h1 > h2)
    f1 = _div(h1-half, half, top_heavy)
    q2tmp = f1*q1+(1-f1)*q2
    special = top_heavy & (q2tmp <= lf)
    newq1 = _div(h1*q1+h2*q2-half*q2, half, special)
    light = reshape & ~top_heavy
    f1light = _div(h1, half, light)
    q1 = torch.where(special, newq1, torch.where(light, f1light*q1+(1-f1light)*q2, q1))
    q2 = torch.where(top_heavy & (q2tmp > lf), q2tmp, q2)
    # thsice_calc_thickn.F:994-1052: liquid and remaining vapor exchange.
    A = torch.where(icy & (H <= 0), 0., A)
    heat = heat+(fb+(surplus+etop+eb)/dt)
    fw = fw-evap
    heat = heat+evap*lf
    salt = (salt0-ri*H*p.salt_ice)/dt
    fw = fw+(mapped.prcAtm-mapped.snowPrc)
    # thsice_calc_thickn.F:1057-1138: lateral melt after the vertical budget.
    ext = etope+ebote
    lateral = icy & (A > 0) & (ext > 0)
    capacity = ri*.5*(q1+q2)*H+rs*lf*S
    frac = _div(ext, capacity, lateral & (ext < capacity))
    newa = (1-frac)*A
    partial = lateral & (ext < capacity) & (newa >= p.area_min)
    complete = lateral & ~partial
    fresh_e, salt_e = (rs*S+ri*H)/dt, (ri*H*p.salt_ice)/dt
    fw = fw+torch.where(partial, frac*fresh_e, torch.where(complete, fresh_e, 0.))
    salt = salt+torch.where(partial, frac*salt_e, torch.where(complete, salt_e, 0.))
    heat = heat+torch.where(complete, (ext-capacity)/dt,
                            torch.where(icy & (A <= 0) & (ext > 0), ext/dt, 0.))
    A = torch.where(partial, newa, torch.where(complete, 0., A))
    H, S = (torch.where(complete, 0., x) for x in (H, S))
    out = replace(ice, iceHeight=H, snowHeight=S, Tsrf=ts, Qice1=q1, Qice2=q2)
    return ThicknessStep(out, A, torch.where(icy, temp.icFrwAtm-mapped.prcAtm, temp.icFrwAtm),
                         torch.where(icy, G-fb*A0, G), torch.where(icy, heat, temp.icFlxSW+mapped.qPrcRn),
                         torch.where(icy, fw, 0.), torch.where(icy, salt, 0.), formed)


def extend_ice(thick: ThicknessStep, mapped: MappedInputs,
               p: Parameters, owned: Tensor) -> ThicknessStep:
    """THSICE_EXTEND output; fluxes now refer to the whole grid-cell area."""
    # thsice_extend.F:145-239, including strict nucleation and volume limits.
    ice = thick.ice
    A, H, S = thick.iceFrac, ice.iceHeight, ice.snowHeight
    q1, q2 = ice.Qice1, ice.Qice2
    active = owned & (thick.frzmltMxL > 0)
    tf, tm = -p.mu*mapped.ocean.sOceMxL, -p.mu*p.salinity_internal
    new_column = active & (A <= 0)
    q1 = torch.where(new_column, -p.cp_water*tm+p.cp_ice*(tm-tf)+
                     p.latent*(1-_div(torch.zeros_like(tf)+tm, tf, new_column)), q1)
    q2 = torch.where(new_column, -p.cp_ice*tf+p.latent, q2)
    qavg = p.rho_ice*(q1+q2)*.5
    new = _div(thick.frzmltMxL*p.dt, qavg, active)
    volume = A*H
    formed = torch.zeros_like(A)
    nucleate = active & (A == 0) & (new > p.h_min*p.area_min)
    nh = torch.maximum(torch.clamp_max(new/p.area_min, p.h_thin), new/p.area_max)
    na = _div(new, nh, nucleate)
    extending = active & (A != 0) & (volume < p.h_max*p.area_max)
    hnew = torch.clamp_max(H, p.h_new_max)
    ea = torch.clamp_max(A+_div(new, hnew, extending), p.area_max)
    eh = torch.clamp_max(_div(volume+new, ea, extending), p.h_max)
    S = torch.where(extending, _div(S*A, ea, extending), S)
    formed = torch.where(nucleate, new, torch.where(extending, eh*ea-volume, formed))
    outa = torch.where(nucleate, na, torch.where(extending, ea, A))
    outh = torch.where(nucleate, nh, torch.where(extending, eh, H))
    initialize = active & (outa > 0) & (A == 0)
    out = replace(ice, iceHeight=outh, snowHeight=S,
                  Tsrf=torch.where(initialize, tf, ice.Tsrf),
                  Tice1=torch.where(initialize, tf, ice.Tice1), Tice2=torch.where(initialize, tf, ice.Tice2),
                  Qice1=torch.where(initialize, q1, ice.Qice1), Qice2=torch.where(initialize, q2, ice.Qice2))
    return replace(thick, ice=out, iceFrac=outa,
                   flx2oc=torch.where(active, qavg*formed/p.dt, 0.),
                   frw2oc=torch.where(active, -p.rho_ice*formed/p.dt, 0.),
                   fsalt=torch.where(active, -(p.rho_ice*p.salt_ice)*formed/p.dt, 0.))
