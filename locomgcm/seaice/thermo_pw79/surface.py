"""SEAICE_SOLVE4TEMP and external-flux SEAICE_BUDGET_OCEAN."""
import math

import torch
from torch import Tensor

from .records import IceState, Inputs, Parameters, SurfaceBudget
from locomgcm.seaice._arithmetic import divide


def surface_budget(ice: IceState, x: Inputs, p: Parameters,
                    owned: Tensor) -> SurfaceBudget:
    """Return T5 before the conversion from W/m2 to effective ice metres."""
    # seaice_growth.F:556-585,725-888: regularized single-category columns.
    icy = owned & (ice.HEFF > 0)
    ar = torch.sqrt(ice.AREA*ice.AREA+p.area_regularization*p.area_regularization)
    h = torch.sqrt((ice.HEFF/ar)**2+p.ice_regularization*p.ice_regularization)
    s = ice.HSNOW/ar
    pdf_den = 2*(0.+p.category_pdf)-1.
    h = torch.where(icy, h/pdf_den, 1.)
    s = torch.where(icy, s/pdf_den, 0.)
    t0 = ice.TICES.squeeze(-3)
    t = torch.where(icy, t0, p.kelvin)
    wind = torch.clamp_min(x.wspeed, p.wind_floor)
    # seaice_solve4temp.F:197-339: freeze emissivity/albedo for all ten sweeps.
    d1 = p.dalton*p.cp_air*p.rho_air
    lsub = p.latent_evap+p.latent
    d1i = p.dalton*lsub*p.rho_air
    def choose(condition, left, right):
        return torch.where(condition, torch.as_tensor(left, dtype=h.dtype, device=h.device), right)

    emiss = choose(s > 0., p.snow_emissivity, p.ice_emissivity)
    d3 = emiss*p.stefan_boltzmann
    lw = emiss*torch.clamp_min(x.lwdown, p.min_longwave)
    ta = torch.clamp_min(x.atemp, p.kelvin+p.min_air_temperature)
    tf = (p.freeze_slope*x.salt+p.freeze_intercept)+p.kelvin
    south = x.YC < 0
    wet = t0 >= p.kelvin+p.wet_albedo_temperature
    ai = torch.where(wet, choose(south, p.wet_ice_albedo_south, p.wet_ice_albedo),
                     choose(south, p.dry_ice_albedo_south, p.dry_ice_albedo))
    ass = torch.where(wet, choose(south, p.wet_snow_albedo_south, p.wet_snow_albedo),
                      choose(south, p.dry_snow_albedo_south, p.dry_snow_albedo))
    # seaice_solve4temp.F:306-315: a nonpositive cutoff uses the ice albedo below it.
    if p.snow_cutoff <= 0.:
        alb = torch.where(s > p.snow_cutoff, ass, ai)
    else:
        alb = torch.where(s > p.snow_cutoff, ass,
                          torch.minimum(ai+s*divide(1., p.snow_cutoff)*(ass-ai), ass))
    pen = torch.where(s > 0, 0., p.shortwave_fraction*torch.exp(-1.5*h))
    sw_pen = -(1-alb)*pen*x.swdown
    sw_abs = (1-alb)*(1-pen)*x.swdown
    conduct = torch.div(p.ice_conductivity*p.snow_conductivity,
        p.snow_conductivity*h+p.ice_conductivity*s)
    ln10 = math.log(10.)
    cc0 = math.exp(12.537*ln10)
    cc1, cc2 = cc0*2663.5*.622*100000.*ln10, cc0*(1-.622)

    def evaluate(ts: Tensor):
        # seaice_solve4temp.F:379-425,507-536, exponential vapor closure.
        t2 = ts*ts
        t3, t4 = t2*ts, t2*t2
        # Keep the source division explicit: scalar / Tensor uses reciprocal
        # multiplication, which perturbs the vapor exponential before Newton.
        vapor = torch.exp((torch.div(-2663.5, ts)+12.537)*ln10)
        humidity = .622*vapor/(100000.-(1-.622)*vapor)
        fc = conduct*(tf-ts)
        lh = d1i*wind*(humidity-x.aqh)
        fia = -lw-sw_abs+t4*d3+d1*wind*(ts-ta)+lh
        cc3 = torch.exp(torch.div(2663.5, ts)*ln10)
        dq = cc1*cc3/((cc2-cc3*100000.)**2*t2)
        jac = 4*d3*t3+d1*wind+d1i*wind*dq
        return fc, fia, lh, jac

    residuals = []
    # IMAX_TICE sweeps (seaice_solve4temp.F:367,439-466).
    for _ in range(p.solve4temp_iterations):
        fc, fia, _, jac = evaluate(t)
        residuals.append(torch.where(icy, fc-fia, 0.))
        t = torch.where(icy, torch.clamp_max(t+(fc-fia)/(conduct+jac), p.kelvin), t)
    fc, fia, lh, _ = evaluate(t)
    # seaice_growth.F:820-832 uses F_ia (not the ADX partition F_ia_net).
    tices = torch.where(icy, t, t0).unsqueeze(-3)
    return SurfaceBudget(tices, torch.where(icy, fia*p.category_pdf, 0.),
                         torch.where(owned, x.Qnet, 0.),
                         torch.where(icy, sw_pen*p.category_pdf, 0.),
                         torch.where(owned, x.Qsw, 0.),
                         torch.where(icy, lh/lsub*p.category_pdf, 0.),
                         icy.to(torch.int32)*p.solve4temp_iterations, tuple(residuals))
