"""Tile-local flux stencils. No exchange or reconstruction from a global field."""

import torch
from torch import Tensor

from ..records import Domain, INTERIOR
from ..space import SeaiceSpace
from ._records import Scalar


def check_field(field: Tensor, *, categories: bool = False) -> None:
    ndim = 5 if categories else 4
    if field.ndim != ndim or field.dtype != torch.float64:
        raise ValueError(f"Expected float64 native field of rank {ndim}, got {field.shape}, {field.dtype}")
    # Reject expand/as_strided layouts in which distinct tile cells alias.
    if not field.is_contiguous():
        raise ValueError("native fields must have disjoint contiguous tile storage")


def interior_write(before: Tensor, value: Tensor) -> Tensor:
    """Write only Fortran i=1:sNx,j=1:sNy, retaining this tile's halo."""
    return SeaiceSpace.replace_domain(before, value, INTERIOR)


def band_domain(start: int, stop: int | None, axis: int) -> Domain:
    """Native directional write/read range, delegated to the spatial owner."""
    end = 0 if stop is None else -stop
    return Domain(0, 0, start, end) if axis == -1 else Domain(start, end, 0, 0)


def band(field: Tensor, start: int, stop: int | None, axis: int) -> Tensor:
    return SeaiceSpace.neighbor(field, 0, 0, domain=band_domain(start, stop, axis))


def put_band(before: Tensor, value: Tensor, start: int, stop: int | None, axis: int) -> Tensor:
    return SeaiceSpace.replace_domain(before, value, band_domain(start, stop, axis))


def sign_positive_zero(value: Tensor) -> Tensor:
    """Fortran SIGN(1,value), ifx without -assume minus0."""
    return torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))


def flux77(q: Tensor, transport: Tensor, cfl: Tensor, face_mask: Tensor, *, axis: int) -> Tensor:
    """Superbee directional flux; CFL is already absolute."""
    # gad_fluxlimit_adv_x.F:72-102 (Y is the transposed stencil).
    b = lambda f, lo, hi: band(f, lo, hi, axis)
    rp = (b(q, 3, None) - b(q, 2, -1)) * b(face_mask, 3, None)
    rc = (b(q, 2, -1) - b(q, 1, -2)) * b(face_mask, 2, -1)
    rm = (b(q, 1, -2) - b(q, 0, -3)) * b(face_mask, 1, -2)
    tr = b(transport, 2, -1)
    numerator = torch.where(tr > 0, rm, rp)
    capped = rc.abs() * 1.0e6 <= numerator.abs()
    ratio = torch.where(
        capped,
        1.0e6 * sign_positive_zero(numerator) * sign_positive_zero(rc),
        numerator / torch.where(capped, torch.ones_like(rc), rc),
    )
    limiter = torch.maximum((2.0 * ratio).clamp(max=1.0), ratio.clamp(max=2.0)).clamp(min=0.0)
    value = tr * (b(q, 2, -1) + b(q, 1, -2)) * 0.5 - tr.abs() * (
        (1.0 - limiter) + b(cfl, 2, -1) * limiter
    ) * rc * 0.5
    return put_band(torch.zeros_like(q), value, 2, -1, axis)


def flux41(q: Tensor, transport: Tensor, velocity: Tensor, mask: Tensor,
           reciprocal_length: Tensor, dt: Scalar, *, axis: int) -> Tensor:
    """Monotone PPM (41), the uniform-grid GAD_PPM_* algorithm."""
    # gad_ppm_p3e_x.F:30-59: masked cubic interpolation at edges 2:-1.
    b = lambda f, lo, hi: band(f, lo, hi, axis)
    left = b(q, 2, -1) + b(mask, 1, -2) * (b(q, 1, -2) - b(q, 2, -1))
    right = b(q, 1, -2) + b(mask, 2, -1) * (b(q, 2, -1) - b(q, 1, -2))
    extrap = 2.0 * left - right
    far_left = extrap + b(mask, 0, -3) * b(mask, 1, -2) * (b(q, 0, -3) - extrap)
    extrap = 2.0 * right - left
    far_right = extrap + b(mask, 3, None) * b(mask, 2, -1) * (b(q, 3, None) - extrap)
    edge = put_band(torch.zeros_like(q), -(1.0 / 12.0) * (far_left + far_right) + (7.0 / 12.0) * (left + right), 2, -1, axis)

    # gad_ppm_hat_x.F:64-93, gad_plm_fun.F:29-58: cells 2:-2.
    center = b(q, 2, -2)
    left = center + b(mask, 1, -3) * (b(q, 1, -3) - center)
    right = center + b(mask, 3, -1) * (b(q, 3, -1) - center)
    dl, dr = center - left, right - center
    slope = 0.5 * (0.5 * (center + right) - 0.5 * (left + center))
    scale = (torch.minimum(dl.abs(), dr.abs()) / slope.abs().clamp(min=1.0e-16)).clamp(max=1.0)
    slope = torch.where(dl * dr > 0.0, scale * slope, torch.zeros_like(slope))

    # gad_ppm_fun.F:80-174: flatten extrema, limit edges, move turning point.
    flat = (right - center) * (center - left) <= 0.0
    low, high = b(edge, 2, -2), b(edge, 3, -1)
    low = torch.where((left - low) * (low - center) <= 0.0, center - slope, low)
    high = torch.where((right - high) * (high - center) <= 0.0, center + slope, high)
    linear = 0.5 * (high - low)
    quadratic = -1.5 * center + 0.75 * (high + low)
    curved = quadratic.abs() > linear.abs() * 0.5
    turn = -0.5 * linear / torch.where(curved, quadratic, torch.ones_like(quadratic))
    high = torch.where(curved & (turn >= -1.0) & (turn <= 0.0), 3.0 * center - 2.0 * low, high)
    low = torch.where(curved & (turn > 0.0) & (turn <= 1.0), 3.0 * center - 2.0 * high, low)
    coeff = (
        torch.where(flat, center, 1.5 * center - 0.25 * (high + low)),
        torch.where(flat, torch.zeros_like(center), 0.5 * (high - low)),
        torch.where(flat, torch.zeros_like(center), -1.5 * center + 0.75 * (high + low)),
    )
    coeff = tuple(put_band(torch.zeros_like(q), c, 2, -2, axis) for c in coeff)

    # gad_ppm_flx_x.F:66-143: integrate upwind polynomial at faces 3:-2.
    vel = b(velocity, 3, -2)
    positive = vel > 0.0
    inverse_dx = torch.where(positive, b(reciprocal_length, 2, -3), b(reciprocal_length, 3, -2))
    cfl = vel * dt * inverse_dx
    end = torch.where(positive, torch.ones_like(vel), -torch.ones_like(vel))
    start = end - 2.0 * cfl
    basis = (end - start, (end**2 - start**2) * 0.5, (end**3 - start**3) * (1.0 / 3.0))
    upwind = tuple(torch.where(positive, b(c, 2, -3), b(c, 3, -2)) for c in coeff)
    integral = basis[0] * upwind[0] + basis[1] * upwind[1] + basis[2] * upwind[2]
    mean = 0.5 * integral / (sign_positive_zero(cfl) * cfl.abs().clamp(min=1.0e-20))
    value = b(transport, 3, -2) * mean
    # gad_ppm_adv_x.F:95-103,139-145; retain the branch as a tensor gate.
    active_line = transport.abs().sum(dim=axis, keepdim=True) > 0.0
    return put_band(torch.zeros_like(q), torch.where(active_line & (vel != 0.0), value, torch.zeros_like(value)), 3, -2, axis)


def extensive_sweep(q: Tensor, flux: Tensor, mask: Tensor,
                    recip_area: Tensor, dt: Scalar, *, axis: int) -> Tensor:
    # seaice_advection.F:508-516 and 722-730, including local halo updates.
    b = lambda f, lo, hi: band(f, lo, hi, axis)
    value = b(q, 1, -1) - dt * b(mask, 1, -1) * b(recip_area, 1, -1) * (
        b(flux, 2, None) - b(flux, 1, -1)
    )
    return put_band(q, value, 1, -1, axis)
