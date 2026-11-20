"""Wind, ocean stress, mass, strength and elliptical viscous rheology."""

from dataclasses import replace
import math

import torch
from torch import Tensor

from .types import (
    CWRITE, D, INNER, ZWRITE, DynamicsCarry, DynamicsForcing,
    DynamicsGeometry, DynamicsParameters, DynamicsSpace, IceState,
    Linearization, PreparedDynamics, Scalar, WindStress, scalar,
)


def signed_magnitude(a: Tensor, b: Tensor) -> Tensor:
    """ifx reference SIGN: a negative zero second argument selects positive."""
    return torch.where(b < 0.0, -torch.abs(a), torch.abs(a))


def turning(angle: Scalar, reference: Tensor) -> tuple[Tensor, Tensor]:
    radians = scalar(angle, reference) * (math.pi / 180.0)
    return torch.sin(radians), torch.cos(radians)


def initialize_carry(heff: Tensor, p: DynamicsParameters) -> DynamicsCarry:
    """Cold start AND native pickup reconstruction, seaice_init_varia.F:79,435-443."""
    zeta = heff * 1.0e11
    eta = zeta / scalar(p.eccentricity, heff) ** 2
    return DynamicsCarry(
        torch.zeros_like(heff), eta, zeta,
        *[torch.zeros_like(heff) for _ in range(7)],
    )


def wind_stress(ice: IceState, forcing: DynamicsForcing, g: DynamicsGeometry,
                p: DynamicsParameters, space: DynamicsSpace) -> WindStress:
    """T2/B4 wind boundary, seaice_get_dynforcing.F:151-205.

    The caller supplies the wind version after X4. South/west bands are zero
    from seaice_dynsolver.F:117-118; no exchange takes place here.
    """
    r, w = space.read, space.write
    u, v = forcing.uwind.clone(), forcing.vwind.clone()
    if p.relative_wind:
        u = w(u, r(u, CWRITE) - 0.5 * (r(ice.UICE, CWRITE) + r(ice.UICE, CWRITE, di=1)), CWRITE)
        v = w(v, r(v, CWRITE) - 0.5 * (r(ice.VICE, CWRITE) + r(ice.VICE, CWRITE, dj=1)), CWRITE)
    speed2 = u * u + v * v
    eps = scalar(p.epsilon, u)
    # Keep the selected sqrt branch finite also at exactly zero wind.
    speed = torch.where(speed2 <= eps * eps, eps, torch.sqrt(torch.where(speed2 <= eps * eps, torch.ones_like(speed2), speed2)))
    drag = torch.where(g.YC < 0.0, scalar(p.air_drag_south, u), scalar(p.air_drag, u))
    cd = p.rho_air * drag * speed
    sine, cosine = turning(p.air_turn_angle, u)
    turn = signed_magnitude(sine, g.fCori)
    tx = cd * (cosine * u - turn * v)
    ty = cd * (turn * u + cosine * v)
    taux = w(torch.zeros_like(u), 0.5 * (r(tx, ZWRITE) + r(tx, ZWRITE, di=-1)) * r(g.SIMaskU, ZWRITE), ZWRITE)
    tauy = w(torch.zeros_like(v), 0.5 * (r(ty, ZWRITE) + r(ty, ZWRITE, dj=-1)) * r(g.SIMaskV, ZWRITE), ZWRITE)
    return WindStress(taux, tauy, forcing.uwind.clone(), forcing.vwind.clone())


def ocean_stress(ice: IceState, forcing: DynamicsForcing, carry: DynamicsCarry,
                 g: DynamicsGeometry, p: DynamicsParameters,
                 space: DynamicsSpace) -> tuple[Tensor, Tensor]:
    """T2/L8/B6 boundary, seaice_ocean_stress.F:99-142 (including X7).

    Uses the last linearization's DWATN even on steps without a momentum solve.
    """
    if not p.update_ocean_stress:
        return forcing.fu.clone(), forcing.fv.clone()
    r, w = space.read, space.write
    q = INNER
    d = carry.DWATN
    du, dv = ice.UICE - forcing.uVel, ice.VICE - forcing.vVel
    sine, cosine = turning(p.water_turn_angle, ice.UICE)
    turn = signed_magnitude(sine, r(g.fCori, q))
    def vpair(di: int = 0) -> Tensor:
        return (r(ice.VICE, q, di=di) - r(forcing.vVel, q, di=di)
                + r(ice.VICE, q, dj=1, di=di) - r(forcing.vVel, q, dj=1, di=di))
    def upair(dj: int = 0) -> Tensor:
        return (r(ice.UICE, q, dj=dj) - r(forcing.uVel, q, dj=dj)
                + r(ice.UICE, q, dj=dj, di=1) - r(forcing.uVel, q, dj=dj, di=1))
    x = (0.5 * (r(d, q) + r(d, q, di=-1)) * cosine * r(du, q)
         - turn * 0.5 * (
             r(d, q) * 0.5 * vpair()
             + r(d, q, di=-1) * 0.5 * vpair(di=-1)))
    y = (0.5 * (r(d, q) + r(d, q, dj=-1)) * cosine * r(dv, q)
         + turn * 0.5 * (
             r(d, q) * 0.5 * upair()
             + r(d, q, dj=-1) * 0.5 * upair(dj=-1)))
    aw = 0.5 * (r(ice.AREA, q) + r(ice.AREA, q, di=-1)) * p.stress_factor
    ass = 0.5 * (r(ice.AREA, q) + r(ice.AREA, q, dj=-1)) * p.stress_factor
    fu = w(forcing.fu, (1.0 - aw) * r(forcing.fu, q) + aw * x, q)
    fv = w(forcing.fv, (1.0 - ass) * r(forcing.fv, q) + ass * y, q)
    return space.exchange_vector(fu, fv, event="X7")


def bgrid_ocean_stress(fu: Tensor, fv: Tensor, area: Tensor,
                       space: DynamicsSpace) -> tuple[Tensor, Tensor]:
    """ostres.F:62-88 (SEAICE_BGRID_DYNAMICS, SEAICE_BICE_STRESS, SEAICE_EXTERNAL_FLUXES).

    Writes the interior; the caller exchanges (ostres.F:89). ``area`` is AREA at
    the dynsolver call, before advection, ridging and growth (seaice_model.F:179-183).
    """
    # ponytail: B-grid builds run without dynamics (BGRID_USE_DYNAMICS), so DWATN keeps
    # its seaice_init_varia.F:79 zero (only lsr.F:221-225 writes it) and fuIce=fvIce=0;
    # port lsr.F with the full fuIce/fvIce once B-grid dynamics is needed.
    r, w, q = space.read, space.write, INNER
    return (w(fu, (1.0 - r(area, q)) * r(fu, q), q),
            w(fv, (1.0 - r(area, q)) * r(fv, q), q))


def prepare_dynamics(ice: IceState, forcing: DynamicsForcing, wind: WindStress,
                     g: DynamicsGeometry, p: DynamicsParameters,
                     space: DynamicsSpace) -> PreparedDynamics:
    """L2: seaice_dynsolver.F:148-302; seaice_calc_ice_strength.F:111-132."""
    r, w = space.read, space.write
    q = ZWRITE
    h, snow = ice.HEFF, ice.HSNOW
    massc = p.rho_ice * r(h, q)
    massu = p.rho_ice * 0.5 * (r(h, q) + r(h, q, di=-1))
    massv = p.rho_ice * 0.5 * (r(h, q) + r(h, q, dj=-1))
    if p.add_snow_mass:
        massc = massc + p.rho_snow * r(snow, q)
        massu = massu + p.rho_snow * 0.5 * (r(snow, q) + r(snow, q, di=-1))
        massv = massv + p.rho_snow * 0.5 * (r(snow, q) + r(snow, q, dj=-1))
    fx, fy = r(wind.TAUX, q), r(wind.TAUY, q)
    if p.scale_surface_stress:
        fx = fx * 0.5 * (r(ice.AREA, q) + r(ice.AREA, q, di=-1))
        fy = fy * 0.5 * (r(ice.AREA, q) + r(ice.AREA, q, dj=-1))
    # ATMOSPHERIC_LOADING, z coordinates, no real freshwater loading.
    phi = g.Bo_surf * forcing.etaN + forcing.pLoad * (scalar(1.0, ice.HEFF) / scalar(p.rho_ocean, ice.HEFF))
    if p.use_tilt:
        fx = fx - massu * (scalar(1.0, ice.HEFF) / (r(g.dxC, q))) * (r(phi, q) - r(phi, q, di=-1))
        fy = fy - massv * (scalar(1.0, ice.HEFF) / (r(g.dyC, q))) * (r(phi, q) - r(phi, q, dj=-1))
    # presPow0=presPow1=1 in all three accepted chains. zMax precedes masking.
    pressure = p.strength * h * torch.exp(-p.c_star * (p.area_max - ice.AREA))
    zmax = p.zeta_max_factor * pressure
    zmin = torch.ones_like(h) * p.zeta_min
    return PreparedDynamics(
        wind, *[w(torch.zeros_like(h), t, q) for t in (massc, massu, massv, fx, fy)],
        pressure * g.HEFFM, zmax, zmin,
    )


def strain_rates(u: Tensor, v: Tensor, carry: DynamicsCarry, g: DynamicsGeometry,
                 space: DynamicsSpace, *, no_slip_factor: float) -> tuple[Tensor, Tensor, Tensor]:
    """seaice_calc_strainrates.F:78-157 without SEAICE_2ndOrderBC; noSlipFac is 1 or 0."""
    r, w = space.read, space.write
    q = CWRITE
    dudx = (scalar(1.0, u) / (r(g.dxF, q))) * (r(u, q, di=1) - r(u, q))
    dvdy = (scalar(1.0, u) / (r(g.dyF, q))) * (r(v, q, dj=1) - r(v, q))
    uave = 0.5 * (r(u, q) + r(u, q, di=1))
    vave = 0.5 * (r(v, q) + r(v, q, dj=1))
    # No gradient beyond an open boundary: OBCS_UVICE_OLD is undefined (seaice_calc_strainrates.F:108-116).
    e11 = w(carry.e11, (dudx + vave * r(g.k2AtC, q)) * r(g.maskInC, q), q)
    e22 = w(carry.e22, (dvdy + uave * r(g.k1AtC, q)) * r(g.maskInC, q), q)
    q = ZWRITE
    dudy = (r(u, q) - r(u, q, dj=-1)) * (scalar(1.0, u) / (r(g.dyU, q)))
    dvdx = (r(v, q) - r(v, q, di=-1)) * (scalar(1.0, u) / (r(g.dxV, q)))
    uave = 0.5 * (r(u, q) + r(u, q, dj=-1))
    vave = 0.5 * (r(v, q) + r(v, q, di=-1))
    hfu = r(g.SIMaskU, q) - r(g.SIMaskU, q, dj=-1)
    hfv = r(g.SIMaskV, q) - r(g.SIMaskV, q, di=-1)
    e12 = (0.5 * (dudy + dvdx - r(g.k1AtZ, q) * vave - r(g.k2AtZ, q) * uave)
           * r(g.HEFFM, q) * r(g.HEFFM, q, di=-1)
           * r(g.HEFFM, q, dj=-1) * r(g.HEFFM, q, dj=-1, di=-1)
           + no_slip_factor * (2.0 * uave * (scalar(1.0, u) / (r(g.dyU, q))) * hfu
                    + 2.0 * vave * (scalar(1.0, u) / (r(g.dxV, q))) * hfv))
    return e11, e22, w(carry.e12, e12, q)


def e12_center_square_mean(e12: Tensor, space: DynamicsSpace) -> Tensor:
    """SEAICEetaZmethod=0, seaice_calc_viscosities.F:117-125, on D."""
    r, q = space.read, D
    e12c = 0.25 * (r(e12, q) + r(e12, q, di=1) + r(e12, q, dj=1) + r(e12, q, dj=1, di=1))
    return e12c * e12c


def e12_center_square_area_weighted(e12: Tensor, g: DynamicsGeometry, space: DynamicsSpace) -> Tensor:
    """SEAICEetaZmethod=3, seaice_calc_viscosities.F:126-138, on D."""
    r, q = space.read, D
    # Area weighting before the square root.
    return 0.25 * (scalar(1.0, e12) / (r(g.rA, q))) * (
        r(g.rAz, q) * r(e12, q) ** 2
        + r(g.rAz, q, di=1) * r(e12, q, di=1) ** 2
        + r(g.rAz, q, dj=1) * r(e12, q, dj=1) ** 2
        + r(g.rAz, q, dj=1, di=1) * r(e12, q, dj=1, di=1) ** 2)


def deformation(e11: Tensor, e22: Tensor, e12sq: Tensor, p: DynamicsParameters,
                space: DynamicsSpace) -> tuple[Tensor, Tensor]:
    """deltaC and deltaCreg on D, seaice_calc_viscosities.F:91-100,337-364."""
    r, q = space.read, D
    ep, em = r(e11, q) + r(e22, q), r(e11, q) - r(e22, q)
    ecc, flow = scalar(p.eccentricity, e11), scalar(p.flow_eccentricity, e11)
    f4 = torch.where(flow != 0.0, flow ** 4, torch.ones_like(flow))
    re4 = torch.where(flow != 0.0, ecc ** 2 / f4, torch.zeros_like(flow))
    delta = torch.sqrt(ep * ep + re4 * (em * em + 4.0 * e12sq))
    # DELTA_SMOOTHREG is undefined in every case build.
    return delta, torch.maximum(delta, scalar(p.delta_min, e11))


def zeta_smooth(delta_reg: Tensor, prepared: PreparedDynamics, g: DynamicsGeometry,
                p: DynamicsParameters, space: DynamicsSpace) -> Tensor:
    """SEAICE_ZETA_SMOOTHREG defined, seaice_calc_viscosities.F:365-372, before zMin."""
    r, q = space.read, D
    arg = torch.exp(scalar(-1.0, delta_reg) / (delta_reg * p.zeta_max_factor))
    return r(prepared.SEAICE_zMax, q) * (1.0 - arg) / (1.0 + arg) * (1.0 + r(g.tensileStrFac, q))


def zeta_capped(delta_reg: Tensor, prepared: PreparedDynamics, g: DynamicsGeometry,
                space: DynamicsSpace) -> Tensor:
    """SEAICE_ZETA_SMOOTHREG undefined, seaice_calc_viscosities.F:373-378, before zMin."""
    r, q = space.read, D
    zeta = 0.5 * (r(prepared.PRESS0, q) * (1.0 + r(g.tensileStrFac, q))) / delta_reg
    return torch.minimum(r(prepared.SEAICE_zMax, q), zeta)


def viscosities(e11: Tensor, e22: Tensor, e12: Tensor, delta: Tensor, zeta_unbounded: Tensor,
                corner_mask: Tensor, old: DynamicsCarry, prepared: PreparedDynamics,
                g: DynamicsGeometry, p: DynamicsParameters, space: DynamicsSpace) -> DynamicsCarry:
    """L3 native D including the old center band, seaice_calc_viscosities.F:380-476.

    ``zeta_unbounded`` is zeta before the zMin bound; ``corner_mask`` multiplies
    etaZ and zetaZ (the free-slip maskZ of :461-472, or ones for no slip).
    """
    r, w = space.read, space.write
    q = D
    flow = scalar(p.flow_eccentricity, e11)
    f2 = torch.where(flow != 0.0, flow ** 2, torch.ones_like(flow))
    re2 = torch.where(flow != 0.0, scalar(1.0, e11) / (f2), torch.zeros_like(flow))
    tension = r(g.tensileStrFac, q)
    zeta_d = torch.maximum(r(prepared.SEAICE_zMin, q), zeta_unbounded) * r(g.HEFFM, q)
    pressure = (r(prepared.PRESS0, q) * (1.0 - p.pressure_replacement)
                + 2.0 * zeta_d * delta * p.pressure_replacement / (1.0 + tension)) * (1.0 - tension)
    zeta, eta = w(old.zeta, zeta_d, q), w(old.eta, zeta_d * re2, q)
    # Read the just-written D and the unchanged outer center band together.
    norm = (r(g.HEFFM, q) + r(g.HEFFM, q, di=-1)
            + r(g.HEFFM, q, dj=-1) + r(g.HEFFM, q, dj=-1, di=-1))
    denom = torch.where(norm > 0.0, norm, torch.ones_like(norm))
    norm = torch.where(norm > 0.0, scalar(1.0, e11) / (denom), norm)
    def average(a: Tensor) -> Tensor:
        return norm * (r(a, q) + r(a, q, di=-1) + r(a, q, dj=-1) + r(a, q, dj=-1, di=-1)) * corner_mask
    return replace(old, eta=eta, zeta=zeta, etaZ=w(old.etaZ, average(eta), q),
                   zetaZ=w(old.zetaZ, average(zeta), q), press=w(old.press, pressure, q),
                   deltaC=w(old.deltaC, delta, q), e11=e11, e22=e22, e12=e12)


def ocean_drag(u: Tensor, v: Tensor, old: Tensor, forcing: DynamicsForcing,
               g: DynamicsGeometry, p: DynamicsParameters, space: DynamicsSpace) -> Tensor:
    """seaice_oceandrag_coeffs.F:63-107; native north/east bands persist."""
    r, w, q = space.read, space.write, CWRITE
    # This file does NOT define OBCS_UVICE_OLD when OBCS is absent.
    du = (u - forcing.uVel) * g.maskInW
    dv = (v - forcing.vVel) * g.maskInS
    speed2 = 0.25 * ((r(du, q) + r(du, q, di=1)) ** 2 + (r(dv, q) + r(dv, q, dj=1)) ** 2)
    cd = torch.where(r(g.YC, q) < 0.0, scalar(p.water_drag_south, u), scalar(p.water_drag, u)) * p.rho_ocean
    minimum = scalar(p.water_drag_min, u)
    active = cd * cd * speed2 > minimum * minimum
    speed = torch.sqrt(torch.where(active, speed2, torch.ones_like(speed2)))
    drag = torch.where(active, cd * speed, minimum) * r(g.HEFFM, q)
    return w(old, drag, q)


def linearize(u: Tensor, v: Tensor, carry: DynamicsCarry, prepared: PreparedDynamics,
              forcing: DynamicsForcing, g: DynamicsGeometry, p: DynamicsParameters,
              space: DynamicsSpace) -> Linearization:
    """L3/B5 coefficient boundary at seaice_lsr.F:382; selects the viscosity formulas."""
    r, q = space.read, D
    if p.no_slip:
        no_slip_factor = 1.0
        corner_mask = torch.ones_like(r(g.HEFFM, q))
    else:
        no_slip_factor = 0.0
        # Free slip: no lateral stress, by masking at Z points (:461-472).
        corner_mask = (r(g.HEFFM, q) * r(g.HEFFM, q, di=-1)
                       * r(g.HEFFM, q, dj=-1) * r(g.HEFFM, q, dj=-1, di=-1))
    e11, e22, e12 = strain_rates(u, v, carry, g, space, no_slip_factor=no_slip_factor)
    if p.eta_z_method == 0:
        e12sq = e12_center_square_mean(e12, space)
    elif p.eta_z_method == 3:
        e12sq = e12_center_square_area_weighted(e12, g, space)
    delta, delta_reg = deformation(e11, e22, e12sq, p, space)
    if p.zeta_smooth_regularization:
        zeta_unbounded = zeta_smooth(delta_reg, prepared, g, p, space)
    else:
        zeta_unbounded = zeta_capped(delta_reg, prepared, g, space)
    new = viscosities(e11, e22, e12, delta, zeta_unbounded, corner_mask, carry, prepared, g, p, space)
    new = replace(new, DWATN=ocean_drag(u, v, carry.DWATN, forcing, g, p, space))
    return Linearization(u, v, new)
