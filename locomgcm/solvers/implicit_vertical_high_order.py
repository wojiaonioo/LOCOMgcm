"""Combined tracer diffusion and high-order implicit vertical advection.

Tracers and row metrics use [..., nz, ny, nx], surface first; w_mit has
nz+1 interfaces and is upward-positive in m/s. Thickness and centre spacing
are positive. The caller selects the matrix hFac time layer. Wet masks are
explicit, and arithmetic retains rhs.dtype. The per-scheme interface
fractions come from ``implicit_vertical_schemes``.
"""
from __future__ import annotations

import torch


def column_scales(*, rhs, dt_lev, recip_dr_f, recip_dr_c):
    """Time step expanded to the rows, and recip_drF/recip_drC as column views."""
    nz = rhs.shape[-3]
    vertical_shape = (1,) * (rhs.ndim - 3) + (nz, 1, 1)
    dt = torch.as_tensor(dt_lev, dtype=rhs.dtype, device=rhs.device)
    if dt.ndim == 1:
        dt = dt.reshape(vertical_shape)
    dt = dt.expand_as(rhs)
    rdf = recip_dr_f.reshape(vertical_shape)
    rdc = recip_dr_c.reshape((1,) * (rhs.ndim - 3) + (nz - 1, 1, 1))
    return dt, rdf, rdc


def high_order_interface_transport(*, w_mit, mask_c, r_a, recip_r_a):
    """MIT rTrans, rCenter and |rCenter| on the internal interfaces."""
    wet = mask_c > 0
    # GAD_IMPLICIT_R masks matrix transport with the cell above the interface.
    # implicit_vertical.tracer_interface_center writes the same rTrans as a
    # plain product with that mask; the two differ only in the sign of zero on
    # dry interfaces.
    w = torch.where(wet[..., :-1, :, :], w_mit[..., 1:-1, :, :], 0.0)
    transport = torch.where(
        wet[..., :-1, :, :], w * r_a.unsqueeze(-3) * mask_c[..., :-1, :, :], 0.0,
    )
    center = 0.5 * transport * recip_r_a.unsqueeze(-3) * -1.0
    absolute = torch.abs(center)
    return transport, center, absolute


def interface_courant_number(*, dt, transport, recip_r_a, rdc):
    """MIT wCFL = deltaTLev(k)*|rTrans|*recip_rA*recip_drC(k) on internal interfaces."""
    return dt[..., 1:, :, :] * torch.abs(transport) * recip_r_a.unsqueeze(-3) * rdc


def assemble_tracer_high_order_vertical_system(
    *, rhs, dt, rdf, rdc, diffusivity, mask_c, recip_hfac,
    center, absolute, high, upwind,
):
    """Assemble MIT GAD's five bands for one high-order scheme and diffusion.

    dt, rdf and rdc come from :func:`column_scales`; center and absolute from
    :func:`high_order_interface_transport`; high and upwind from the scheme's
    kernel in ``implicit_vertical_schemes``.
    """
    nz = rhs.shape[-3]
    kappa = torch.as_tensor(diffusivity, dtype=rhs.dtype, device=rhs.device).expand_as(rhs)
    wet = mask_c > 0
    rh = torch.where(wet, recip_hfac, 0.0)
    interface_wet = wet[..., :-1, :, :] & wet[..., 1:, :, :]
    kappa_i = torch.where(interface_wet, kappa[..., 1:, :, :], 0.0)
    lower = torch.zeros_like(rhs)
    upper = torch.zeros_like(rhs)
    lower[..., 1:, :, :] = torch.where(
        interface_wet,
        -dt[..., 1:, :, :] * mask_c[..., :-1, :, :] * rh[..., 1:, :, :]
        * rdf[..., 1:, :, :] * kappa_i * rdc, 0.0,
    )
    upper[..., :-1, :, :] = torch.where(
        interface_wet,
        -dt[..., :-1, :, :] * mask_c[..., 1:, :, :] * rh[..., :-1, :, :]
        * rdf[..., :-1, :, :] * kappa_i * rdc, 0.0,
    )
    main = 1.0 - (lower + upper)
    lower2 = torch.zeros_like(rhs)
    upper2 = torch.zeros_like(rhs)

    # At the first/last internal interface the unavailable far neighbour is zero.
    mask_m = torch.cat((torch.zeros_like(mask_c[..., :1, :, :]),
                        mask_c[..., :-2, :, :]), dim=-3)[..., :nz - 1, :, :]
    mask_p = torch.cat((mask_c[..., 2:, :, :],
                        torch.zeros_like(mask_c[..., :1, :, :])), dim=-3)[..., :nz - 1, :, :]
    hm = torch.where(mask_m > 0, high * (center + absolute) * mask_m, 0.0)
    hp = torch.where(mask_p > 0, high * (center - absolute) * mask_p, 0.0)
    plus = (center + upwind) + hm
    minus = (center - upwind) + hp

    # Each row receives the deeper interface first (MIT's descending k loop).
    above_dt = dt[..., :-1, :, :]
    above_rh = rh[..., :-1, :, :]
    above_rdf = rdf[..., :-1, :, :]
    below_dt = dt[..., 1:, :, :]
    below_rh = rh[..., 1:, :, :]
    below_rdf = rdf[..., 1:, :, :]
    lower[..., :-1, :, :] = lower[..., :-1, :, :] - hm * above_dt * above_rh * above_rdf
    main[..., :-1, :, :] = main[..., :-1, :, :] + plus * above_dt * above_rh * above_rdf
    upper[..., :-1, :, :] = upper[..., :-1, :, :] + minus * above_dt * above_rh * above_rdf
    upper2[..., :-1, :, :] = -hp * above_dt * above_rh * above_rdf
    lower2[..., 1:, :, :] = hm * below_dt * below_rh * below_rdf
    lower[..., 1:, :, :] = lower[..., 1:, :, :] - plus * below_dt * below_rh * below_rdf
    main[..., 1:, :, :] = main[..., 1:, :, :] - minus * below_dt * below_rh * below_rdf
    upper[..., 1:, :, :] = upper[..., 1:, :, :] + hp * below_dt * below_rh * below_rdf
    return (lower2, lower, main, upper, upper2, torch.where(wet, rhs, 0.0))
