"""Interface coefficients of each implicit vertical tracer advection scheme.

One kernel per scheme of MITgcm's ``GAD_IMPLICIT_R`` dispatch
(gad_implicit_r.F:214-251); the scheme choice is made by the caller,
``dynamics/implicit_vertical_advection.py``.  Inputs and outputs live on the
nz-1 internal interfaces of a column, ``[..., nz-1, ny, nx]``, surface first.
``absolute`` is ``|rCenter|`` with ``rCenter = 0.5*rTrans*recip_rA*rkSign``,
``rTrans`` the MIT upward-positive volume transport (m^3/s) and ``rkSign=-1``;
``transport`` is that ``rTrans``; ``cfl`` is the dimensionless
``deltaT*|rTrans|*recip_rA*recip_drC``.  A five-band kernel returns the
high-order fraction (MIT ``rHigh``/``highFac``) and the upwind term
``rUpwind`` that ``implicit_vertical_high_order`` turns into the five bands.
A three-band scheme with a constant upwind fraction (codes 1, 2, 20) needs no
kernel: ``implicit_vertical.assemble_tracer_implicit_vertical_system`` takes
the fraction.
Wet masks are explicit; arithmetic keeps the input dtype.
"""
from __future__ import annotations

import torch

from defgcm.operators.fluxes.tracer_reconstruction import dst3_limiter_ratio


def upwind3_interface_weights(*, absolute):
    """Third-order upwind, code 3 (gad_u3c4_impl_r.F, the ELSE branch)."""
    high = 1.0 / 6.0
    upwind = 2.0 * (1.0 / 6.0) * absolute
    return high, upwind


def dst3_interface_weights(*, cfl, absolute):
    """DST3, code 30 (gad_u3c4_impl_r.F, the ENUM_DST3 branch)."""
    high = (1.0 - cfl * cfl) * (1.0 / 6.0)
    upwind = (2.0 * high) * absolute
    return high, upwind


def dst3fl_interface_weights(*, cfl, absolute, transport, limiter_tracer, mask_c):
    """DST3 with flux limiter, code 33 (gad_dst3fl_impl_r.F).

    limiter_tracer is the caller-selected time layer ``[..., nz, ny, nx]``:
    full RHS for global multiDimAdvection, or the current tracer otherwise.
    mask_c is the C-cell wet mask of the same shape.
    """
    nz = mask_c.shape[-3]
    wet = mask_c > 0
    tracer = torch.where(wet, limiter_tracer, 0.0)
    tracer_m = torch.cat((tracer[..., :1, :, :], tracer[..., :-2, :, :]),
                         dim=-3)[..., :nz - 1, :, :]
    tracer_p = torch.cat((tracer[..., 2:, :, :], tracer[..., -1:, :, :]),
                         dim=-3)[..., :nz - 1, :, :]
    far_mask_m = torch.cat((mask_c[..., :1, :, :], mask_c[..., :-2, :, :]),
                           dim=-3)[..., :nz - 1, :, :]
    far_mask_p = torch.cat((mask_c[..., 2:, :, :], mask_c[..., -1:, :, :]),
                           dim=-3)[..., :nz - 1, :, :]
    rj = tracer[..., 1:, :, :] - tracer[..., :-1, :, :]
    rjm = torch.where(far_mask_m > 0,
                      (tracer[..., :-1, :, :] - tracer_m) * far_mask_m, 0.0)
    rjp = torch.where(far_mask_p > 0,
                      (tracer_p - tracer[..., 1:, :, :]) * far_mask_p, 0.0)
    rjh = torch.where(transport > 0, rjm, rjp)
    cl = (2.0 - cfl) * (1.0 - cfl) * (1.0 / 6.0)
    ch = (1.0 - cfl * cfl) * (1.0 / 6.0)
    cm = dst3_limiter_ratio(1.0 + cfl, cfl)
    th1 = cl * torch.abs(rj) + ch * torch.abs(rjh)
    th2 = cm * torch.abs(rjh)
    linear = (th1 <= th2) & (th1 <= torch.abs(rj))
    slope = (th2 < th1) & (th2 <= torch.abs(rj))
    low = torch.where(linear, cl, torch.where(slope, 0.0, 1.0))
    high = torch.where(linear, ch, torch.where(slope, cm, 0.0))
    nonpositive = rj * rjh <= 0
    low = torch.where(nonpositive, 0.0, low)
    high = torch.where(nonpositive, 0.0, high)
    upwind = (0.5 - low) * absolute * 2.0
    return high, upwind
