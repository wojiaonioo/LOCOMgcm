"""PW79 zero-layer growth, seaice_growth.F without ITD, SITRACER or salt plume."""
from dataclasses import fields

import torch
from torch import Tensor

from .records import GrowthResult, IceState, Inputs, Parameters, Salinity
from .surface import surface_budget
from locomgcm.seaice._arithmetic import divide

#: Zero-layer growth choices implemented below: MITgcm SEAICE_areaLossFormula
#: 1 and 2 (seaice_growth.F:1833-1839), SEAICE_growMeltByConv true and false
#: (:1298-1342, :1528-1568) and SEAICEuseFlooding true (the unconditional
#: flooding step). SEAICE_areaGainFormula is 1 (:1817-1818).
AREA_LOSS_FORMULAS = (1, 2)
AREA_GAIN_FORMULAS = (1,)
GROW_MELT_BY_CONV = (True, False)
FLOODING = (True,)
#: SEAICE_doOpenWaterMelt true: open water melts down to -HEFF
#: (seaice_growth.F:1610-1613, facOpenMelt=1).
OPEN_WATER_MELT = (True,)


def growth(ice: IceState, x: Inputs, p: Parameters, owned: Tensor, *,
           salinity: Salinity | None, swfrac_below_surface: Tensor | None) -> GrowthResult:
    """T6.growth boundary; T5 is ``result.surface``.

    Inputs are REG_RIDGE output, including both already-applied negative
    inventory corrections. The caller alone performs the X8 exchanges.
    ``salinity`` is given exactly when ``p.variable_salinity`` (the REG_RIDGE
    HSALT and saltFluxAdjust); ``swfrac_below_surface`` is SWFrac3D at kSrfS=2
    (ini_forcing.F:148-181), given exactly when ``p.shortwave_heating``.
    """
    if (salinity is not None) != p.variable_salinity:
        raise ValueError("salinity is required exactly with SEAICE_VARIABLE_SALINITY")
    if (swfrac_below_surface is not None) != p.shortwave_heating:
        raise ValueError("swfrac_below_surface is required exactly with SHORTWAVE_HEATING")
    for record in (ice, x):
        for field in fields(record):
            v = getattr(record, field.name)
            shape = (*owned.shape[:-2], 1, *owned.shape[-2:]) if field.name == "TICES" else owned.shape
            if v.dtype != torch.float64 or v.shape != shape:
                raise ValueError(f"{field.name} has invalid dtype or shape")
    if owned.dtype != torch.bool:
        raise TypeError("owned must be a boolean spatial mask")
    surface = surface_budget(ice, x, p, owned)
    H, S, A = ice.HEFF, ice.HSNOW, ice.AREA
    # seaice_growth.F:355-390,938-978: ratios and effective-ice conversion.
    i2s = divide(p.rho_ice, p.rho_snow)
    s2i = divide(1., i2s)
    qi = p.rho_ice*p.latent
    q2h = divide(p.dt, qi)
    h2q = divide(1., q2h)
    pr2h = divide(p.dt*p.rho_fresh, p.rho_ice)
    h2pr = divide(1., pr2h)
    ac = surface.a_QbyATM_cover*q2h*A
    sc = surface.a_QSWbyATM_cover*q2h*A
    ao = surface.a_QbyATM_open*q2h*(1-A)
    so = surface.a_QSWbyATM_open*q2h*(1-A)
    # seaice_growth.F:965-970: SEAICE_DISABLE_SUBLIM zeroes the flux before scaling.
    fw_sublim = torch.zeros_like(surface.a_FWbySublim) if p.disable_sublimation else surface.a_FWbySublim
    sub = p.dt*divide(1., p.rho_ice)*fw_sublim*A
    rc, ro, remaining_sub = ac, ao, sub
    recip_h = torch.where(H > 0, A/torch.sqrt(H*H+p.ice_regularization*p.ice_regularization), 0.)
    # seaice_growth.F:1025-1054: available ocean heat a_QbyOCN.
    tf = p.freeze_intercept+p.freeze_slope*x.salt
    piston = torch.where(x.theta >= tf, torch.as_tensor(p.piston_velocity, dtype=H.dtype, device=H.device),
                          divide(p.frazil_fraction*p.ocean_depth, p.dt))
    taper = torch.where(A > 0, 1-p.mc_phee_taper*A, 1.)
    ocean = piston*(-(p.cp_water*p.rho_water*divide(1., qi))*(x.theta-tf)*p.dt*x.HEFFM)*taper
    # seaice_growth.F:1138-1265: snow sublimates before ice.
    take_snow = torch.clamp_min(torch.minimum(remaining_sub, S*s2i), 0.)
    ds_sub = -take_snow*i2s
    S = S-take_snow*i2s
    remaining_sub = remaining_sub-take_snow
    take_ice = torch.clamp_min(torch.minimum(remaining_sub, H), 0.)
    dh_sub = -take_ice
    H = H-take_ice
    remaining_sub = remaining_sub-take_ice
    ac = ac-remaining_sub
    rc = rc-remaining_sub
    # seaice_growth.F:1298-1342: without growMeltByConv the ocean melts ice first.
    r_ocean = ocean
    dh_ocean = torch.zeros_like(H)
    if not p.grow_melt_by_conv:
        dh_ocean = torch.maximum(r_ocean, -H)
        r_ocean = r_ocean-dh_ocean
        H = H+dh_ocean
    # seaice_growth.F:1373-1453: surface snow melt, covered ice change.
    melt_snow = torch.clamp_max(torch.maximum(rc, -S*s2i), 0.)
    ds_atm = melt_snow*i2s
    S = S+melt_snow*i2s
    rc = rc-melt_snow
    dh_cover = torch.maximum(-H, rc+A*r_ocean)
    dh_atm = dh_cover
    rc = rc-dh_cover
    H = H+dh_cover
    # seaice_growth.F:1469-1521: empty snow-precip file, diagnosed snow/rain.
    rain_water = torch.where(ac >= 0, 0., -pr2h*x.precip*A)
    ds_rain = torch.where(ac >= 0, pr2h*i2s*x.precip*A, 0.)
    S = S+ds_rain
    # seaice_growth.F:1528-1568: then the remaining ocean heat melts snow.
    ds_ocean = torch.zeros_like(S)
    if not p.grow_melt_by_conv:
        ds_ocean = torch.clamp_max(torch.maximum(r_ocean*i2s, -S), 0.)
        r_ocean = r_ocean-ds_ocean*s2i
        S = S+ds_ocean
    # seaice_growth.F:1598-1613: open-water growth and melt enabled; under
    # SHORTWAVE_HEATING the shortwave penetrating below layer 1 is unavailable.
    potential = ro+r_ocean*(1-A)
    below = swfrac_below_surface*so if p.shortwave_heating else 0.
    dh_open = torch.maximum(potential-below, -H)*x.HEFFM
    dh_atm = dh_atm+dh_open
    ro = ro-dh_open
    H = H+dh_open
    # seaice_growth.F:1723-1729: snow flooding, before area evolution.
    waterline = (S*p.rho_snow+H*p.rho_ice)*divide(1., p.rho_water)
    flood = torch.clamp_min(waterline-H, 0.)
    H = H+flood
    S = S-flood*i2s
    # seaice_growth.F:1778-1971: areaGain=1, no ITD.
    r0 = torch.where(x.YC < 0, torch.as_tensor(divide(1., p.new_ice_height_south), dtype=H.dtype, device=H.device),
                     divide(1., p.new_ice_height))
    if p.area_loss_formula == 1:
        loss = torch.clamp_max(dh_cover, 0.)+torch.clamp_max(dh_open, 0.)+torch.clamp_max(dh_ocean, 0.)
    elif p.area_loss_formula == 2:
        loss = torch.clamp_max(dh_cover+dh_open+dh_ocean, 0.)
    area = A+r0*torch.clamp_min(dh_open, 0.)+.5*recip_h*loss*(2*p.category_pdf-1)
    area = torch.where((H > 0) | (S > 0), torch.clamp(area, 0., p.area_max), 0.)
    hsalt = None
    if not p.variable_salinity:
        # seaice_growth.F:1994-2007: fixed salinity with a surface-salt cap.
        salt_delta = x.d_HEFFbyNEG+dh_ocean+dh_atm+flood+dh_sub
        cap = torch.clamp(x.salt, 0., p.salt_ice)
        salt_flux = salt_delta*cap*x.HEFFM*divide(1., p.dt)*p.rho_ice
    else:
        # seaice_growth.F:2056-2144: brine fraction on growth, bulk HSALT on melt.
        made = dh_ocean+dh_atm
        before = H-made-flood
        grown = x.HEFFM*divide(1., p.dt)*p.salt_fraction*x.salt*made*p.rho_ice
        melted = x.HEFFM*divide(1., p.dt)*salinity.HSALT*made/before
        salt_flux = torch.where(made >= 0.0, grown, melted)
        hsalt = torch.where(owned, salinity.HSALT+salt_flux*p.dt, salinity.HSALT)
        salt_flux = salt_flux+salinity.saltFluxAdjust
    # seaice_growth.F:2207-2235: Qnet is not multiplied by HEFFM as a whole.
    qnet = rc+ro+sc-(dh_ocean+ds_ocean*s2i+x.d_HEFFbyNEG+x.d_HSNWbyNEG*s2i-
                     pr2h*x.snowPrecip*(1-A))*x.HEFFM
    qsw = sc+so
    qnet, qsw = qnet*h2q, qsw*h2q
    # seaice_growth.F:2321-2388: virtual freshwater, unset temp_EvPrRn.
    atm_heat = x.HEFFM*h2q*(sc+ac+ao)
    solid = p.rho_fresh*x.HEFFM*h2pr*(-ds_rain*s2i+sub-remaining_sub)
    tflux = atm_heat-(-solid*(-p.latent+p.cp_water*0.))-0.
    water_delta = ds_atm*s2i+rain_water+ds_ocean*s2i+dh_ocean+dh_atm+x.d_HEFFbyNEG+x.d_HSNWbyNEG*s2i+remaining_sub
    empmr = x.HEFFM*((x.evap-x.precip)*(1-A)-x.runoff+water_delta*h2pr)*p.rho_fresh
    atm_fw = x.HEFFM*(x.evap*(1-A)-x.precip-x.runoff)*p.rho_fresh+sub*p.rho_ice*divide(1., p.dt)
    # seaice_growth.F:2424-2438: the ice load is updated only with real freshwater.
    load = torch.where(owned, H*p.rho_ice+S*p.rho_snow, x.sIceLoad) if p.real_freshwater_flux else x.sIceLoad.clone()
    out = IceState(torch.where(owned, H, ice.HEFF), torch.where(owned, S, ice.HSNOW),
                   torch.where(owned, area, ice.AREA), surface.TICES)
    return GrowthResult(out, torch.where(owned, qnet, x.Qnet), torch.where(owned, qsw, x.Qsw),
                        torch.where(owned, empmr, x.EmPmR), torch.where(owned, salt_flux, x.saltFlux),
                        load, hsalt, torch.where(owned, atm_heat, 0.),
                        torch.where(owned, tflux, 0.), torch.where(owned, atm_fw, 0.), surface,
                        *(torch.where(owned, v, 0.) for v in
                          (dh_cover, dh_open, dh_sub, ds_sub, ds_atm, ds_rain, flood, remaining_sub,
                           dh_ocean, ds_ocean)))
