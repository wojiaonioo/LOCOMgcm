"""LSR momentum forcing, cross stresses and the ten matrix coefficients."""

from dataclasses import replace

import torch
from torch import Tensor

from .physics import signed_magnitude, turning
from .types import (
    SOLVE, DynamicsCarry, DynamicsForcing, DynamicsGeometry, DynamicsParameters, DynamicsSpace,
    IceState, Linearization, LinearSystem, PreparedDynamics, scalar,
)


def assemble_system(ice: IceState, old_u: Tensor, old_v: Tensor,
                    lin: Linearization, prepared: PreparedDynamics,
                    forcing: DynamicsForcing, g: DynamicsGeometry,
                    p: DynamicsParameters, space: DynamicsSpace) -> LinearSystem:
    """L4/B5 boundary at seaice_lsr.F:590; adds the metric stress for SEAICEselectMetricTerms=2."""
    system = linear_system_terms(ice, old_u, old_v, lin, prepared, forcing, g, p, space)
    if p.metric_terms == 2:
        rhs_u, rhs_v = metric_stress_rhs(system.rhsU, system.rhsV, lin.carry, g, space)
        system = replace(system, rhsU=rhs_u, rhsV=rhs_v)
    return system


def metric_stress_rhs(rhs_u: Tensor, rhs_v: Tensor, c: DynamicsCarry, g: DynamicsGeometry,
                      space: DynamicsSpace) -> tuple[Tensor, Tensor]:
    """seaice_lsr.F:1740-1760,1896-1915 from the linearization's strain rates and viscosities."""
    r, w, q = space.read, space.write, SOLVE
    sig11 = c.zeta * (c.e11 + c.e22) - c.eta * (c.e11 - c.e22) - 0.5 * c.press
    sig12 = 2.0 * c.e12 * c.etaZ
    sig22 = c.zeta * (c.e11 + c.e22) + c.eta * (c.e11 - c.e22) - 0.5 * c.press
    u = r(rhs_u, q) + r(g.seaiceMaskU, q) * 0.5 * (
        r(g.k2AtU, q) * (r(sig12, q) + r(sig12, q, dj=1))
        - r(g.k1AtU, q) * (r(sig11, q) + r(sig11, q, di=-1)))
    v = r(rhs_v, q) + r(g.seaiceMaskV, q) * 0.5 * (
        r(g.k1AtV, q) * (r(sig12, q) + r(sig12, q, di=1))
        - r(g.k2AtV, q) * (r(sig22, q) + r(sig22, q, dj=-1)))
    return w(rhs_u, u, q), w(rhs_v, v, q)


def linear_system_terms(ice: IceState, old_u: Tensor, old_v: Tensor,
                        lin: Linearization, prepared: PreparedDynamics,
                        forcing: DynamicsForcing, g: DynamicsGeometry,
                        p: DynamicsParameters, space: DynamicsSpace) -> LinearSystem:
    """L4/B5 boundary at seaice_lsr.F:590 without the metric stress, before initial residual evaluation.

    No strong implicit coupling or BDF2. Every metric coefficient is explicit
    even when its current value is zero.
    """
    r, w, q = space.read, space.write, SOLVE
    c = lin.carry
    zero = torch.zeros_like(ice.HEFF)
    ep, em = c.eta + c.zeta, c.zeta - c.eta
    u, v, d = lin.uIceC, lin.vIceC, c.DWATN
    sine, cosine = turning(p.water_turn_angle, u)
    turn = signed_magnitude(sine, r(g.fCori, q))
    drag = d * cosine
    if p.scale_surface_stress:
        aw = 0.5 * (r(ice.AREA, q) + r(ice.AREA, q, di=-1))
        ass = 0.5 * (r(ice.AREA, q) + r(ice.AREA, q, dj=-1))
    else:
        aw, ass = torch.ones_like(r(u, q)), torch.ones_like(r(v, q))

    # Fixed time derivative: seaice_lsr.F:256-267. old_u/v are the CALL entry.
    fx = r(prepared.FORCEX0, q) + r(prepared.seaiceMassU, q) * (scalar(1.0, ice.HEFF) / scalar(p.dt_dyn, ice.HEFF)) * r(old_u, q)
    fy = r(prepared.FORCEY0, q) + r(prepared.seaiceMassV, q) * (scalar(1.0, ice.HEFF) / scalar(p.dt_dyn, ice.HEFF)) * r(old_v, q)
    # seaice_lsr.F:433-460. Preserve the source's left-associated differences.
    def relv(dj: int = 0, di: int = 0) -> Tensor:
        return (r(forcing.vVel, q, dj, di) - r(v, q, dj, di)
                + r(forcing.vVel, q, dj + 1, di) - r(v, q, dj + 1, di))
    def relu(dj: int = 0, di: int = 0) -> Tensor:
        return (r(forcing.uVel, q, dj, di) - r(u, q, dj, di)
                + r(forcing.uVel, q, dj, di + 1) - r(u, q, dj, di + 1))
    fx = fx + (0.5 * (r(d, q) + r(d, q, di=-1)) * cosine * r(forcing.uVel, q)
               - turn * 0.5 * (r(d, q) * 0.5 * relv() + r(d, q, di=-1) * 0.5 * relv(di=-1))) * aw
    fy = fy + (0.5 * (r(d, q) + r(d, q, dj=-1)) * cosine * r(forcing.vVel, q)
               + turn * 0.5 * (r(d, q) * 0.5 * relu() + r(d, q, dj=-1) * 0.5 * relu(dj=-1))) * ass
    # Coriolis is not removed when f=0: seaice_lsr.F:461-482.
    mc = prepared.seaiceMassC
    fx = fx + 0.5 * (
        r(mc, q) * r(g.fCori, q) * 0.5 * (r(v, q) + r(v, q, dj=1))
        + r(mc, q, di=-1) * r(g.fCori, q, di=-1) * 0.5 * (r(v, q, di=-1) + r(v, q, dj=1, di=-1)))
    fy = fy - 0.5 * (
        r(mc, q) * r(g.fCori, q) * 0.5 * (r(u, q) + r(u, q, di=1))
        + r(mc, q, dj=-1) * r(g.fCori, q, dj=-1) * 0.5 * (r(u, q, dj=-1) + r(u, q, dj=-1, di=1)))
    fx, fy = fx * r(g.seaiceMaskU, q), fy * r(g.seaiceMaskV, q)

    # SEAICE_LSR_RHSU:1665-1737, including the boundary difference term.
    ux = (2, 2, 1, 2)
    uy = (2, 1, 2, 2)
    vx = (2, 2, 2, 1)
    vy = (1, 2, 2, 2)
    s11 = (r(em, ux) * (r(v, ux, dj=1) - r(v, ux)) * (scalar(1.0, ice.HEFF) / (r(g.dyF, ux)))
           + r(ep, ux) * r(g.k2AtC, ux) * 0.5 * (r(v, ux, dj=1) + r(v, ux))
           - 0.5 * r(c.press, ux))
    s11 = w(zero, s11, ux)
    def maskz(rect):
        return (r(g.HEFFM, rect) * r(g.HEFFM, rect, di=-1)
                * r(g.HEFFM, rect, dj=-1) * r(g.HEFFM, rect, dj=-1, di=-1))
    s12u = (r(c.etaZ, uy) * (
        (r(v, uy) - r(v, uy, di=-1)) * (scalar(1.0, ice.HEFF) / (r(g.dxV, uy)))
        - r(g.k1AtZ, uy) * 0.5 * (r(v, uy) + r(v, uy, di=-1))) * maskz(uy)
        + r(c.etaZ, uy) * (scalar(1.0, ice.HEFF) / (r(g.dxV, uy))) * (r(v, uy) + r(v, uy, di=-1))
        * (r(g.seaiceMaskV, uy) - r(g.seaiceMaskV, uy, di=-1)) * 2.0)
    s12u = w(zero, s12u, uy)
    rhsu = fx + (scalar(1.0, ice.HEFF) / (r(g.rAw, q))) * r(g.seaiceMaskU, q) * (
        r(g.dyF, q) * r(s11, q) - r(g.dyF, q, di=-1) * r(s11, q, di=-1)
        + r(g.dxV, q, dj=1) * r(s12u, q, dj=1) - r(g.dxV, q) * r(s12u, q))

    # SEAICE_LSR_RHSV:1820-1892.
    s22 = (r(em, vy) * (r(u, vy, di=1) - r(u, vy)) * (scalar(1.0, ice.HEFF) / (r(g.dxF, vy)))
           + r(ep, vy) * r(g.k1AtC, vy) * 0.5 * (r(u, vy, di=1) + r(u, vy))
           - 0.5 * r(c.press, vy))
    s22 = w(zero, s22, vy)
    s12v = (r(c.etaZ, vx) * (
        (r(u, vx) - r(u, vx, dj=-1)) * (scalar(1.0, ice.HEFF) / (r(g.dyU, vx)))
        - r(g.k2AtZ, vx) * 0.5 * (r(u, vx) + r(u, vx, dj=-1))) * maskz(vx)
        + r(c.etaZ, vx) * (scalar(1.0, ice.HEFF) / (r(g.dyU, vx))) * (r(u, vx) + r(u, vx, dj=-1))
        * (r(g.seaiceMaskU, vx) - r(g.seaiceMaskU, vx, dj=-1)) * 2.0)
    s12v = w(zero, s12v, vx)
    rhsv = fy + (scalar(1.0, ice.HEFF) / (r(g.rAs, q))) * r(g.seaiceMaskV, q) * (
        r(g.dyU, q, di=1) * r(s12v, q, di=1) - r(g.dyU, q) * r(s12v, q)
        + r(g.dxF, q) * r(s22, q) - r(g.dxF, q, dj=-1) * r(s22, q, dj=-1))

    # SEAICE_LSR_CALC_COEFFS:1417-1460, metric terms remain explicit.
    uxx = w(zero, r(g.dyF, ux) * r(ep, ux) * (scalar(1.0, ice.HEFF) / (r(g.dxF, ux))), ux)
    uxm = w(zero, r(g.dyF, ux) * r(em, ux) * r(g.k1AtC, ux) * 0.5, ux)
    uyy = w(zero, r(g.dxV, uy) * (r(c.etaZ, uy) + 0.0 * r(c.zetaZ, uy)) * (scalar(1.0, ice.HEFF) / (r(g.dyU, uy))), uy)
    uym = w(zero, r(g.dxV, uy) * r(c.etaZ, uy) * r(g.k2AtZ, uy) * 0.5, uy)
    vxx = w(zero, r(g.dyU, vx) * (r(c.etaZ, vx) + 0.0 * r(c.zetaZ, vx)) * (scalar(1.0, ice.HEFF) / (r(g.dxV, vx))), vx)
    vxm = w(zero, r(g.dyU, vx) * r(c.etaZ, vx) * r(g.k1AtZ, vx) * 0.5, vx)
    vyy = w(zero, r(g.dxF, vy) * r(ep, vy) * (scalar(1.0, ice.HEFF) / (r(g.dyF, vy))), vy)
    vym = w(zero, r(g.dxF, vy) * r(em, vy) * r(g.k2AtC, vy) * 0.5, vy)
    mu, mv = r(g.seaiceMaskU, q), r(g.seaiceMaskV, q)
    au = (-r(uxx, q, di=-1) + r(uxm, q, di=-1)) * mu
    cu = (-r(uxx, q) - r(uxm, q)) * mu
    bu = (1.0 - mu) + (
        r(uxx, q, di=-1) + r(uxx, q) + r(uyy, q, dj=1) + r(uyy, q)
        + r(uxm, q, di=-1) - r(uxm, q) + r(uym, q, dj=1) - r(uym, q)) * mu
    ur1, ur2 = r(uyy, q) + r(uym, q), r(uyy, q, dj=1) - r(uym, q, dj=1)
    hm, hp = r(g.seaiceMaskU, q, dj=-1), r(g.seaiceMaskU, q, dj=1)
    bu = bu + mu * ((1.0 - hm) * (r(uyy, q) + r(uym, q))
                    + (1.0 - hp) * (r(uyy, q, dj=1) - r(uym, q, dj=1)))
    ur1, ur2 = ur1 * hm, ur2 * hp
    raw = scalar(1.0, ice.HEFF) / (r(g.rAw, q))
    au, cu = au * raw, cu * raw
    bu = bu * raw + mu * ((scalar(1.0, ice.HEFF) / scalar(p.dt_dyn, ice.HEFF)) * r(prepared.seaiceMassU, q)
                         + 0.5 * (r(drag, q) + r(drag, q, di=-1)) * aw)
    ur1, ur2 = ur1 * raw, ur2 * raw

    # seaice_lsr.F:1529-1585; transverse neighbors differ from the U chain.
    av = (-r(vyy, q, dj=-1) + r(vym, q, dj=-1)) * mv
    cv = (-r(vyy, q) - r(vym, q)) * mv
    bv = (1.0 - mv) + (
        r(vxx, q) + r(vxx, q, di=1) + r(vyy, q) + r(vyy, q, dj=-1)
        - r(vxm, q) + r(vxm, q, di=1) - r(vym, q) + r(vym, q, dj=-1)) * mv
    vr1, vr2 = r(vxx, q) + r(vxm, q), r(vxx, q, di=1) - r(vxm, q, di=1)
    hm, hp = r(g.seaiceMaskV, q, di=-1), r(g.seaiceMaskV, q, di=1)
    bv = bv + mv * ((1.0 - hm) * (r(vxx, q) + r(vxm, q))
                    + (1.0 - hp) * (r(vxx, q, di=1) - r(vxm, q, di=1)))
    vr1, vr2 = vr1 * hm, vr2 * hp
    ras = scalar(1.0, ice.HEFF) / (r(g.rAs, q))
    av, cv = av * ras, cv * ras
    bv = bv * ras + mv * ((scalar(1.0, ice.HEFF) / scalar(p.dt_dyn, ice.HEFF)) * r(prepared.seaiceMassV, q)
                         + 0.5 * (r(drag, q) + r(drag, q, dj=-1)) * ass)
    vr1, vr2 = vr1 * ras, vr2 * ras
    # Exact-zero diagonal repair is conditional, seaice_lsr.F:1587-1603.
    if p.scale_surface_stress:
        bu = torch.where(bu == 0.0, torch.ones_like(bu), bu)
        bv = torch.where(bv == 0.0, torch.ones_like(bv), bv)
    return LinearSystem(*[w(zero, value, q) for value in (rhsu, rhsv, au, bu, cu, av, bv, cv, ur1, ur2, vr1, vr2)])
