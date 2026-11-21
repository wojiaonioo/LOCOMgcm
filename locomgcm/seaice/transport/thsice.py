"""THSICE's five capacity transports, interior adjustments and ocean return."""

import torch
from torch import Tensor

from ..records import SOUTH_WEST
from ..space import SeaiceSpace
from ._records import (
    CapacityAdvectionResult, CapacityStage, FiveFieldTransport,
    OceanTransportFields, Scalar, ThsiceAdjustment, ThsiceFields,
    ThsiceParameters, ThsiceTransportResult, TransportGeometry,
)
from ._stencils import band, check_field, extensive_sweep, flux77, interior_write, put_band

#: thSIceAdvScheme values of the transport below (Superbee, flux77).
ADVECTION_SCHEMES = (77,)


def thsice_advection(field: Tensor, iceVol: Tensor, uTrans: Tensor,
                     vTrans: Tensor, maskOce: Tensor, rA: Tensor, *,
                     deltaT: Scalar, iceEps: Scalar,
                     useGridArea: bool) -> CapacityAdvectionResult:
    """One native scheme-77 call; return all updated local halo values.

    iceVol is unchanged in the grid-area branch. Otherwise it is updated
    after X and again after Y; the Y CFL uses the X-updated capacity.
    """
    for value in (field, iceVol, uTrans, vTrans, maskOce, rA):
        check_field(value)
    q, capacity = field.clone(), iceVol.clone()
    fluxes = []
    # thsice_advection.F:242-270: ocean/face masks, no OBCS in this path.
    for axis, transport in ((-1, uTrans), (-2, vTrans)):
        b = lambda f, lo, hi: band(f, lo, hi, axis)
        tr, mask, area = transport, maskOce, rA
        face_mask = put_band(torch.zeros_like(mask), torch.minimum(b(mask, 0, -1), b(mask, 1, None)), 1, None, axis)
        # thsice_advection.F:337-358,553-574: upwind capacity-based CFL.
        positive, negative = b(tr, 1, None).clamp(min=0.0), (-b(tr, 1, None)).clamp(min=0.0)
        if useGridArea:
            cfl_value = deltaT * (positive * torch.div(area.new_ones(()), b(area, 0, -1)) + negative * torch.div(area.new_ones(()), b(area, 1, None)))
        else:
            eps = capacity.new_ones(()) * iceEps
            cfl_value = deltaT * (
                positive / torch.maximum(b(capacity, 0, -1), eps)
                + negative / torch.maximum(b(capacity, 1, None), eps)
            )
        cfl = put_band(torch.zeros_like(q), cfl_value, 1, None, axis)
        af = flux77(q, tr, cfl, face_mask, axis=axis)
        if useGridArea:
            q = extensive_sweep(q, af, mask, torch.div(area.new_ones(()), area), deltaT, axis=axis)
        else:
            # thsice_advection.F:496-507,716-727: keep q when capacity <= eps.
            old_capacity = b(capacity, 1, -1)
            new_capacity = old_capacity - deltaT * b(mask, 1, -1) * (b(tr, 2, None) - b(tr, 1, -1))
            valid = new_capacity > iceEps
            denominator = torch.where(valid, new_capacity, torch.ones_like(new_capacity))
            new_q = (b(q, 1, -1) * old_capacity - deltaT * b(mask, 1, -1) * (
                b(af, 2, None) - b(af, 1, -1)
            )) / denominator
            q = put_band(q, torch.where(valid, new_q, b(q, 1, -1)), 1, -1, axis)
            capacity = put_band(capacity, new_capacity, 1, -1, axis)
        fluxes.append(af)
    # thsice_advection.F:509-513,729-733: unwritten exterior flux bands
    # are zero in all parent consumers (thsice_advdiff.F:185-188).
    return CapacityAdvectionResult(q, capacity, fluxes[0], fluxes[1])


def thsice_transport_five(fields: ThsiceFields, UICE: Tensor, VICE: Tensor,
                          hOceMxL: Tensor, geometry: TransportGeometry, *,
                          deltaT: Scalar, cells: int) -> FiveFieldTransport:
    """Return the five B7 events from an X9-exchanged THSICE input.

    iceMask stays at its entry version until the later interior adjustment.
    Every field/capacity is a separate tile-local tensor, with no exchange.
    """
    for field in (fields.iceMask, fields.iceHeight, fields.snowHeight,
                  fields.Qice1, fields.Qice2, fields.snowAge, UICE, VICE, hOceMxL):
        check_field(field)
    g = geometry
    # thsice_advdiff.F:129-133,177-191: ``cells`` is the global Nx*Ny.
    mean_area = g.globalArea / cells
    area_eps, vol_eps = 1.0e-10 * mean_area, 1.0e-15 * mean_area
    mask = (hOceMxL > 0.0).to(dtype=hOceMxL.dtype)
    utrans, vtrans = torch.zeros_like(UICE), torch.zeros_like(VICE)
    # thsice_advdiff.F:208-219: local face transports, first W/S band zero.
    utrans = put_band(utrans, band(UICE, 1, None, -1) * band(g.dyG, 1, None, -1) * band(mask, 0, -1, -1) * band(mask, 1, None, -1), 1, None, -1)
    vtrans = put_band(vtrans, band(VICE, 1, None, -2) * band(g.dxG, 1, None, -2) * band(mask, 0, -1, -2) * band(mask, 1, None, -2), 1, None, -2)
    empty_capacity = torch.zeros_like(fields.iceMask)
    area = thsice_advection(fields.iceMask, empty_capacity, utrans, vtrans, mask, g.rA,
                           deltaT=deltaT, iceEps=area_eps, useGridArea=True)
    events = [CapacityStage("B7.area", area.field, empty_capacity, area.iceVol, None, None, area.afx, area.afy)]

    # thsice_advdiff.F:239-254: snow starts from the old ice-covered area.
    snow_capacity = fields.iceMask * g.rA
    snow = thsice_advection(fields.snowHeight, snow_capacity, area.afx, area.afy, mask, g.rA,
                           deltaT=deltaT, iceEps=area_eps, useGridArea=False)
    events.append(CapacityStage("B7.snow", snow.field, snow_capacity, snow.iceVol, None, None, snow.afx, snow.afy))

    # thsice_advdiff.F:260-272: reset capacity; oldVol uses the OLD height.
    ice_capacity = fields.iceMask * g.rA
    old_volume = ice_capacity * fields.iceHeight
    ice = thsice_advection(fields.iceHeight, ice_capacity, area.afx, area.afy, mask, g.rA,
                          deltaT=deltaT, iceEps=area_eps, useGridArea=False)
    events.append(CapacityStage("B7.ice", ice.field, ice_capacity, ice.iceVol, old_volume, old_volume, ice.afx, ice.afy))

    # thsice_advdiff.F:340-351,377-388: both enthalpies start from oldVol.
    q1_capacity, q2_capacity = old_volume.clone(), old_volume.clone()
    q1 = thsice_advection(fields.Qice1, q1_capacity, ice.afx, ice.afy, mask, g.rA,
                         deltaT=deltaT, iceEps=vol_eps, useGridArea=False)
    events.append(CapacityStage("B7.q1", q1.field, q1_capacity, q1.iceVol, old_volume, old_volume, q1.afx, q1.afy))
    q2 = thsice_advection(fields.Qice2, q2_capacity, ice.afx, ice.afy, mask, g.rA,
                         deltaT=deltaT, iceEps=vol_eps, useGridArea=False)
    events.append(CapacityStage("B7.q2", q2.field, q2_capacity, q2.iceVol, old_volume, old_volume, q2.afx, q2.afy))
    return FiveFieldTransport(
        ThsiceFields(fields.iceMask.clone(), ice.field, snow.field, q1.field, q2.field, fields.snowAge.clone()),
        area.field, old_volume, q2.iceVol, tuple(events),
    )


def thsice_adjust(transported: FiveFieldTransport,
                  parameters: ThsiceParameters) -> ThsiceAdjustment:
    """Apply fraction bounds and thin-ice removal, retaining transported halo."""
    p, f = parameters, transported.fields
    frac, h, s = transported.iceFrc, f.iceHeight, f.snowHeight
    minimum = frac.new_ones(()) * p.iceMaskMin
    r_minimum = torch.where(minimum > 0.0, torch.div(minimum.new_ones(()), torch.where(minimum > 0.0, minimum, torch.ones_like(minimum))), torch.zeros_like(minimum))
    # thsice_advdiff.F:439-459: conserve ice and snow while adjusting area.
    above, below = frac > 1.0, frac < minimum
    a = torch.where(above, torch.ones_like(frac), torch.where(below, minimum, frac))
    h = torch.where(above, h * frac, torch.where(below, h * frac * r_minimum, h))
    s = torch.where(above, s * frac, torch.where(below, s * frac * r_minimum, s))
    # thsice_advdiff.F:472-481: preserve effective ice/snow when thickening.
    thin = h < p.hIceMin
    volume = a * h
    thicken = thin & (volume >= p.hIceMin * p.iceMaskMin)
    melt = thin & ~thicken
    new_a = torch.where(thicken, volume / p.hIceMin, a)
    new_s = torch.where(thicken, s * p.hIceMin / torch.where(thicken, h, torch.ones_like(h)), s)
    new_h = torch.where(thicken, h.new_ones(()) * p.hIceMin, h)
    # thsice_advdiff.F:487-509: melt fluxes use state BEFORE clearing.
    fw = (p.rhos * s + p.rhoi * h) * a / p.deltaT
    salt = p.rhoi * h * p.saltIce * a / p.deltaT
    heat = -(p.rhos * s * p.qsnow + p.rhoi * h * (f.Qice1 + f.Qice2) * 0.5) * a / p.deltaT
    zero = torch.zeros_like(frac)
    adjusted = ThsiceFields(
        interior_write(f.iceMask, torch.where(melt, zero, new_a)),
        interior_write(f.iceHeight, torch.where(melt, zero, new_h)),
        interior_write(f.snowHeight, torch.where(melt, zero, new_s)),
        interior_write(f.Qice1, torch.where(melt, zero, f.Qice1)),
        interior_write(f.Qice2, torch.where(melt, zero, f.Qice2)),
        interior_write(f.snowAge, torch.where(melt, zero, f.snowAge)),
    )
    return ThsiceAdjustment(
        adjusted,
        interior_write(transported.iceVol, torch.where(thin, volume, transported.iceVol)),
        interior_write(zero, torch.where(melt, heat, zero)),
        interior_write(zero, torch.where(melt, fw, zero)),
        interior_write(zero, torch.where(melt, salt, zero)),
    )


def thsice_ocean_return(adjusted: ThsiceAdjustment, ocean: OceanTransportFields,
                        hOceMxL: Tensor, parameters: ThsiceParameters) -> OceanTransportFields:
    """Return B8 ocean fluxes and load on trim[0,1,0,1], MIT signs."""
    f, p = adjusted.fields, parameters
    wet = hOceMxL > 0.0
    # thsice_do_advect.F:65-68,94-104: wet flux return; load at ALL points.
    def write(before: Tensor, value: Tensor) -> Tensor:
        return SeaiceSpace.replace_domain(before, value, SOUTH_WEST)

    return OceanTransportFields(
        write(ocean.Qnet, torch.where(wet, ocean.Qnet - adjusted.oceQnet, ocean.Qnet)),
        write(ocean.EmPmR, torch.where(wet, ocean.EmPmR - adjusted.oceFWfx, ocean.EmPmR)),
        write(ocean.saltFlux, torch.where(wet, ocean.saltFlux - adjusted.oceSflx, ocean.saltFlux)),
        write(ocean.sIceLoad, (f.snowHeight * p.rhos + f.iceHeight * p.rhoi) * f.iceMask),
    )


def thsice_do_advect(fields: ThsiceFields, UICE: Tensor, VICE: Tensor,
                      hOceMxL: Tensor, geometry: TransportGeometry,
                      parameters: ThsiceParameters,
                      ocean: OceanTransportFields, *, cells: int) -> ThsiceTransportResult:
    """B branch transport through B8, retaining all five B7 stage results.

    Tsrf/Tice1/Tice2 and Qsw are not transport fields. The caller preserves
    them. The selected branch has thSIceAdvScheme=77, thSIce_diffK=0,
    stressReduction=0 and useRealFreshWaterFlux=false; no exchange occurs.
    """
    five = thsice_transport_five(fields, UICE, VICE, hOceMxL, geometry, deltaT=parameters.deltaT, cells=cells)
    adjusted = thsice_adjust(five, parameters)
    returned = thsice_ocean_return(adjusted, ocean, hOceMxL, parameters)
    return ThsiceTransportResult(five, adjusted, returned)
