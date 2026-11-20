"""Piecewise-polynomial tracer face fluxes: PPM (MITgcm 40, 41, 42) and PQM (50, 51, 52).

One one-dimensional kernel serves the X, Y and R passes.  Every input is a
tuple of face-aligned windows: entry ``c`` of a cell tuple is cell ``i+c-h`` of
face ``i`` (``h`` the declared ``halo``, face ``i`` between cells ``i-1`` and
``i``), entry ``c`` of a face tuple is face ``i+c-h+1``.  The kernel forms the
edge values (PQM also the edge slopes) and the cell polynomial of the two
cells next to each face, picks
the upwind one by the sign of the face velocity and integrates it over the
Courant number.  Cell masks are the physical wet mask ``maskC`` (a missing
neighbour is extrapolated linearly, ``gad_ppm_p3e_x.F``), cell widths are
``dxF``/``dyF``/``drF`` (not hFac-weighted).  The velocity is positive towards
increasing index (east, north, or DOWNWARD in R, i.e. MITgcm ``-wFld``) and the
returned flux is ``transport * face value`` in the same orientation.  No grid,
no halo exchange.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tracer_reconstruction import _upwind_donor_flux


@dataclass(frozen=True)
class StencilCellInputs:
    """Cell inputs of a scheme that declares ``reads_cell_geometry``.

    ``cell_masks``/``cell_widths`` hold ``maskC`` and the cell width on the
    ``2*halo`` cells of each face; ``recip_face_spacings`` the reciprocal
    centre-to-centre spacing (``recip_dxC``) on the ``2*halo-1`` faces
    ``i-halo+1 .. i+halo-1``; ``velocity`` the face velocity (MITgcm ``uFld``)
    whose sign picks the upwind cell and that forms the Courant number with
    ``dt`` and the upwind cell width.
    """

    cell_masks: tuple[torch.Tensor, ...]
    cell_widths: tuple[torch.Tensor, ...]
    recip_face_spacings: tuple[torch.Tensor, ...]
    velocity: torch.Tensor
    dt: float


_CFL_FLOOR = 1.0e-20     # gad_ppm_flx_x.F: sign(max(abs(uCFL),1.d-20),uCFL)
_PLM_EPS = 1.0e-16       # gad_plm_fun.F: PARAMETER( epsil = 1. _d -16 )
_WENO_ZERO = 1.0e-20     # gad_osc_mul_x.F: zero = 1. _d -20
_WENO_BIG = 1.0e99       # gad_osc_mul_x.F: omin = +1. _d 99, omax = -1. _d 99


def _ppm_coefficients(ff00, fell, ferr):
    """GAD_PPM_FUN_NULL: the three coefficients of the cell parabola."""
    sum_edges = ferr + fell
    return (
        (3.0 / 2.0) * ff00 - (1.0 / 4.0) * sum_edges,
        (1.0 / 2.0) * (ferr - fell),
        -(3.0 / 2.0) * ff00 + (3.0 / 4.0) * sum_edges,
    )


def _plm_slope(ffll, ff00, ffrr):
    """GAD_PLM_FUN_U: ``(dfds(-1), dfds(0), dfds(+1))`` of a uniform cell."""
    d_left = ff00 - ffll
    d_right = ffrr - ff00
    same_sign = d_left * d_right > 0.0
    fell = 0.5 * (ffll + ff00)
    ferr = 0.5 * (ff00 + ffrr)
    centre = 0.5 * (ferr - fell)
    scal = torch.minimum(d_left.abs(), d_right.abs()) / torch.clamp(centre.abs(), min=_PLM_EPS)
    scal = torch.clamp(scal, max=1.0)
    centre = torch.where(same_sign, scal * centre, torch.zeros_like(centre))
    return 0.5 * d_left, centre, 0.5 * d_right


def _ppm_mono(ff00, ffll, ffrr, fell, ferr, dfds0):
    """GAD_PPM_FUN_MONO: limited coefficients and the ``mono`` flag (0, 1 or 2)."""
    zero = torch.zeros_like(ff00)
    one = torch.ones_like(ff00)
    flat = (ffrr - ff00) * (ff00 - ffll) <= 0.0
    left_bad = (ffll - fell) * (fell - ff00) <= 0.0
    fell = torch.where(left_bad, ff00 - dfds0, fell)
    right_bad = (ffrr - ferr) * (ferr - ff00) <= 0.0
    ferr = torch.where(right_bad, ff00 + dfds0, ferr)
    mono = torch.where(left_bad | right_bad, one, zero)
    c1, c2, c3 = _ppm_coefficients(ff00, fell, ferr)
    steep = c3.abs() > c2.abs() * 0.5
    turn = -0.5 * c2 / torch.where(steep, c3, one)
    turn_left = steep & (turn >= -1.0) & (turn <= 0.0)
    turn_right = steep & (turn > 0.0) & (turn <= 1.0)
    ferr = torch.where(turn_left, 3.0 * ff00 - 2.0 * fell, ferr)
    fell = torch.where(turn_right, 3.0 * ff00 - 2.0 * ferr, fell)
    turned = turn_left | turn_right
    mono = torch.where(turned, 2.0 * one, mono)
    t1, t2, t3 = _ppm_coefficients(ff00, fell, ferr)
    c1 = torch.where(turned, t1, c1)
    c2 = torch.where(turned, t2, c2)
    c3 = torch.where(turned, t3, c3)
    mono = torch.where(flat, one, mono)
    return (
        torch.where(flat, ff00, c1),
        torch.where(flat, zero, c2),
        torch.where(flat, zero, c3),
    ), mono


class _Window:
    """Cell and face accessors of one face-aligned window, by offset from face ``i``."""

    def __init__(self, cells, inputs: StencilCellInputs):
        self.h = len(cells) // 2
        self.cells = cells
        self.inputs = inputs

    def T(self, o):
        return self.cells[o + self.h]

    def m(self, o):
        return self.inputs.cell_masks[o + self.h]

    def width(self, o):
        return self.inputs.cell_widths[o + self.h]

    def neighbour(self, o, side):
        """``fbar(o) + mask(o+side)*(fbar(o+side)-fbar(o))`` (``gad_ppm_hat_x.F``)."""
        f = self.T(o)
        return f + self.m(o + side) * (self.T(o + side) - f)


def _ppm_edge(w: _Window, d):
    """GAD_PPM_P3E_X: fourth-order edge value on face ``i+d`` (cells ``d-2 .. d+1``)."""
    m_m1 = w.m(d - 1)
    m_0 = w.m(d)
    f_m1 = w.T(d) + m_m1 * (w.T(d - 1) - w.T(d))
    f_0 = w.T(d - 1) + m_0 * (w.T(d) - w.T(d - 1))
    m_m2 = w.m(d - 2) * m_m1
    ftmp = 2.0 * f_m1 - f_0
    f_m2 = ftmp + m_m2 * (w.T(d - 2) - ftmp)
    m_p1 = w.m(d + 1) * m_0
    ftmp = 2.0 * f_0 - f_m1
    f_p1 = ftmp + m_p1 * (w.T(d + 1) - ftmp)
    return -(1.0 / 12.0) * (f_m2 + f_p1) + (7.0 / 12.0) * (f_m1 + f_0)


def _oscillation(w: _Window, o):
    """GAD_OSC_LOC_X (interior branch): first and second differences of cell ``o``, in float64.

    The centred branch on every cell: MITgcm's one-sided branch is reached
    only on the outermost overlap cell, i.e. never for an interior face when
    the overlap is at least 4 (see ``TracerAdvectionScheme``).
    """
    f0 = w.T(o).to(torch.float64)
    fl = f0 + w.m(o - 1).to(torch.float64) * (w.T(o - 1).to(torch.float64) - f0)
    fr = f0 + w.m(o + 1).to(torch.float64) * (w.T(o + 1).to(torch.float64) - f0)
    return fr * 0.25 - fl * 0.25, fr * 0.25 - f0 * 0.50 + fl * 0.25


def _weno_weights(w: _Window, o):
    """GAD_OSC_MUL_X(ix=o, hh=2): weights of the unlimited and the limited polynomial.

    Computed in float64, MITgcm's ``_RL``: the bounds start at ``+-1e99`` and
    the weights divide by cubes of the indicators, neither of which a float32
    tensor holds.  The weights are returned in the tracer dtype.
    """
    dtype = w.T(o).dtype
    omin = torch.full_like(w.T(o), _WENO_BIG, dtype=torch.float64)
    omax = torch.full_like(w.T(o), -_WENO_BIG, dtype=torch.float64)
    mval = torch.ones_like(w.T(o), dtype=torch.float64)
    for ii in range(o - 2, o + 3):
        dels = (ii - o) * 2.0
        dfs1, dfs2 = _oscillation(w, ii)
        dfs1 = dfs1 + dfs2 * dels
        a = 2.0 * dfs1
        b = 4.0 * dfs2
        oval = a * a + b * b
        omin = torch.where(oval < omin, oval, omin)
        omax = torch.where(oval > omax, oval, omax)
        mval = mval * w.m(ii).to(torch.float64)
    big = omax + _WENO_ZERO
    small = omin + _WENO_ZERO
    s1 = 1.0e5 / (big * big * big)
    s2 = 1.0 / (small * small * small)
    osum = s1 + s2
    wet = mval > 0.0
    return (
        torch.where(wet, s1 / osum, torch.zeros_like(s1)).to(dtype),
        torch.where(wet, s2 / osum, torch.ones_like(s2)).to(dtype),
    )


def _ppm_cell(w: _Window, o, limiter: str):
    """GAD_PPM_HAT_X for cell ``o``: the three polynomial coefficients."""
    ff00 = w.T(o)
    ffll = w.neighbour(o, -1)
    ffrr = w.neighbour(o, +1)
    fell = _ppm_edge(w, o)
    ferr = _ppm_edge(w, o + 1)
    if limiter == "null":
        return _ppm_coefficients(ff00, fell, ferr)
    if limiter == "mono":
        _, dfds0, _ = _plm_slope(ffll, ff00, ffrr)
        return _ppm_mono(ff00, ffll, ffrr, fell, ferr, dfds0)[0]
    if limiter == "weno":
        _, dfds0, _ = _plm_slope(ffll, ff00, ffrr)
        unlimited = _ppm_coefficients(ff00, fell, ferr)
        limited, mono = _ppm_mono(ff00, ffll, ffrr, fell, ferr, dfds0)
        fdel = (ffrr - ff00).abs() + (ff00 - ffll).abs()
        fmag = ffll.abs() + ff00.abs() + ffrr.abs()
        blend = (mono > 0.0) & (fdel > 1.0e-6 * fmag)
        s1, s2 = _weno_weights(w, o)
        return tuple(
            torch.where(blend, s1 * u + s2 * lc, lc) for u, lc in zip(unlimited, limited)
        )
    raise ValueError(f"no PPM limiter {limiter!r}")


def _integrated_face_value(fhat_left, fhat_right, w: _Window, velocity, dt, direction):
    """GAD_PPM_FLX_X / GAD_PQM_FLX_X: ``0.5*intF/sign(max(|uCFL|,1e-20),uCFL)``.

    ``uCFL = velocity*dt*recip_width(upwind cell)``; the upwind cell is the
    lower one where ``direction`` is positive (the face velocity, except the
    PQM R pass, which branches on the transport, ``gad_pqm_flx_r.F``).
    Returns the face value only; the caller multiplies by the transport.
    """
    forward = direction > 0.0
    cfl_left = velocity * dt * (1.0 / w.width(-1))
    cfl_right = velocity * dt * (1.0 / w.width(0))
    cfl = torch.where(forward, cfl_left, cfl_right)
    s22 = torch.where(forward, torch.ones_like(cfl), -torch.ones_like(cfl))
    s11 = s22 - 2.0 * cfl
    p2 = s11 * s11
    powers = (s11, p2, p2 * s11, p2 * p2, p2 * p2 * s11)
    # ss22 is +-1: its even powers are 1, its odd powers itself.
    ivec = [s22 - s11] + [
        ((s22 if n % 2 else 1.0) - powers[n - 1]) * (1.0 / n) for n in range(2, len(fhat_left) + 1)
    ]
    fhat = [torch.where(forward, left, right) for left, right in zip(fhat_left, fhat_right)]
    int_f = ivec[0] * fhat[0]
    for term, coefficient in zip(ivec[1:], fhat[1:]):
        int_f = int_f + term * coefficient
    return 0.5 * int_f / torch.copysign(torch.clamp(cfl.abs(), min=_CFL_FLOOR), cfl)


def ppm_transport_flux(
    limiter: str,
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    cells: tuple[torch.Tensor, ...],
    u_trans: torch.Tensor,
    inputs: StencilCellInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order upwind and PPM (``limiter`` null 40, mono 41, weno 42) transport flux.

    GAD_PPM_ADV_X with GAD_PPM_P3E_X, GAD_PPM_HAT_X and GAD_PPM_FLX_X: the flux
    is ``uTrans * face value`` and zero where the face velocity is zero.
    ``T_left``/``T_right`` are the upwind donors the caller splices onto
    uncoupled edge faces.
    """
    upwind_flux = _upwind_donor_flux(T_left, T_right, u_trans)[3]
    w = _Window(cells, inputs)
    fhat_left = _ppm_cell(w, -1, limiter)
    fhat_right = _ppm_cell(w, 0, limiter)
    value = _integrated_face_value(
        fhat_left, fhat_right, w, inputs.velocity, inputs.dt, inputs.velocity,
    )
    flux = torch.where(inputs.velocity == 0.0, torch.zeros_like(value), u_trans * value)
    return upwind_flux, flux


def vertical_column_windows(
    tracer: torch.Tensor,
    mask: torch.Tensor,
    layer_thickness: torch.Tensor,
    *,
    halo: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Cell windows of the interior interfaces ``1 .. nz-1`` of ``[batch, nz, ny, nx]`` columns.

    GAD_PPM_ADV_R pads each column with ``halo-1`` cells above and below that
    repeat the end value with mask zero (``floc(-2:0) = floc(1)``,
    ``mloc(-2:0) = 0``); returns the tracer, mask and ``drF`` windows of the
    ``2*halo`` cells of every interior interface.
    """
    nz = tracer.shape[-3]
    pad = halo - 1
    values = torch.cat(
        [tracer[..., :1, :, :]] * pad + [tracer] + [tracer[..., -1:, :, :]] * pad, dim=-3,
    )
    mask = mask.expand_as(tracer) if mask.ndim == tracer.ndim else mask.unsqueeze(0).expand_as(tracer)
    zeros = torch.zeros_like(mask[..., :1, :, :])
    masks = torch.cat([zeros] * pad + [mask] + [zeros] * pad, dim=-3)
    drF = layer_thickness.to(dtype=tracer.dtype, device=tracer.device)
    drF = torch.cat([drF[:1]] * pad + [drF] + [drF[-1:]] * pad).view(1, -1, 1, 1)
    # Interface k (1 .. nz-1) sits between cells k-1 and k; its window is
    # cells k-halo .. k+halo-1, i.e. padded rows k-1 .. k+2*halo-2.
    span = nz - 1
    return (
        tuple(values[..., c:c + span, :, :] for c in range(2 * halo)),
        tuple(masks[..., c:c + span, :, :] for c in range(2 * halo)),
        tuple(drF[..., c:c + span, :, :] for c in range(2 * halo)),
    )


def vertical_ppm_flux(
    limiter: str,
    tracer: torch.Tensor,
    transport: torch.Tensor,
    velocity: torch.Tensor,
    *,
    layer_thickness: torch.Tensor,
    dt: float,
    mask: torch.Tensor | None,
    halo: int,
) -> torch.Tensor:
    """PPM vertical advective flux at layer interfaces (DOWNWARD-positive, ``[batch, nz+1, ny, nx]``).

    GAD_PPM_ADV_R with GAD_PPM_FLX_R.  MITgcm integrates over
    ``wCFL = wFld*deltaT*recip_drF(upwind)`` and returns
    ``wfac * (-0.5*intF/sign(max(|wCFL|,1e-20),wCFL))``; with the downward
    velocity ``-wFld`` and transport ``-rTrans`` that is the horizontal formula
    term by term, so the same kernel runs on the padded column windows.  The
    surface and bottom interfaces carry no flux.  ``mask`` is ``maskC`` or
    ``None`` (every cell wet).
    """
    flux = torch.zeros(
        (tracer.shape[0], tracer.shape[1] + 1, tracer.shape[2], tracer.shape[3]),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    if tracer.shape[1] < 2:
        return flux
    wet = torch.ones_like(tracer) if mask is None else mask.to(dtype=tracer.dtype, device=tracer.device)
    cells, masks, widths = vertical_column_windows(tracer, wet, layer_thickness, halo=halo)
    w_vel = velocity[..., 1:-1, :, :]
    inputs = StencilCellInputs(
        cell_masks=masks, cell_widths=widths, recip_face_spacings=(), velocity=w_vel, dt=dt,
    )
    w = _Window(cells, inputs)
    fhat_above = _ppm_cell(w, -1, limiter)
    fhat_below = _ppm_cell(w, 0, limiter)
    value = _integrated_face_value(fhat_above, fhat_below, w, w_vel, dt, w_vel)
    flux[..., 1:-1, :, :] = torch.where(
        w_vel == 0.0, torch.zeros_like(value), transport[..., 1:-1, :, :] * value,
    )
    return flux


# ---------------------------------------------------------------------------
# PQM (gad_pqm_*.F): six-point edge values and slopes, quartic cell polynomial.
# ---------------------------------------------------------------------------


def _pqm_edge(w: _Window, d, recip_spacing):
    """GAD_PQM_P5E_X: edge value and slope on face ``i+d`` (cells ``d-3 .. d+2``)."""
    m_m1 = w.m(d - 1)
    m_0 = w.m(d)
    f_m1 = w.T(d) + m_m1 * (w.T(d - 1) - w.T(d))
    f_0 = w.T(d - 1) + m_0 * (w.T(d) - w.T(d - 1))
    m_m2 = w.m(d - 2) * m_m1
    m_m3 = w.m(d - 3) * m_m2
    ftmp = 2.0 * f_m1 - f_0
    f_m2 = ftmp + m_m2 * (w.T(d - 2) - ftmp)
    ftmp = 2.0 * f_m2 - f_m1
    f_m3 = ftmp + m_m3 * (w.T(d - 3) - ftmp)
    m_p1 = w.m(d + 1) * m_0
    m_p2 = w.m(d + 2) * m_p1
    ftmp = 2.0 * f_0 - f_m1
    f_p1 = ftmp + m_p1 * (w.T(d + 1) - ftmp)
    ftmp = 2.0 * f_p1 - f_0
    f_p2 = ftmp + m_p2 * (w.T(d + 2) - ftmp)
    value = (
        (1.0 / 60.0) * (f_m3 + f_p2) - (8.0 / 60.0) * (f_m2 + f_p1)
        + (37.0 / 60.0) * (f_m1 + f_0)
    )
    slope = (
        -(1.0 / 90.0) * (f_m3 - f_p2) + (5.0 / 36.0) * (f_m2 - f_p1)
        - (49.0 / 36.0) * (f_m1 - f_0)
    ) * recip_spacing
    return value, slope


def _pqm_coefficients(ff00, fell, ferr, dell, derr):
    """GAD_PQM_FUN_NULL: the five coefficients of the cell quartic."""
    return (
        (30.0 / 16.0) * ff00 - (7.0 / 16.0) * (ferr + fell) + (1.0 / 16.0) * (derr - dell),
        (3.0 / 4.0) * (ferr - fell) - (1.0 / 4.0) * (derr + dell),
        -(30.0 / 8.0) * ff00 + (15.0 / 8.0) * (ferr + fell) - (3.0 / 8.0) * (derr - dell),
        -(1.0 / 4.0) * (ferr - fell - derr - dell),
        (30.0 / 16.0) * ff00 - (15.0 / 16.0) * (ferr + fell) + (5.0 / 16.0) * (derr - dell),
    )


def _quadratic_roots(a, b, c):
    """QUADROOT (gad_pqm_fun.F): whether real roots exist, and the two roots."""
    one = torch.ones_like(a)
    sq = b * b - 4.0 * a * c
    quadratic = a.abs() > 0.0
    linear = ~quadratic & (b.abs() > 0.0)
    exists = (quadratic & (sq >= 0.0)) | linear
    root_sq = torch.sqrt(torch.where(quadratic & (sq >= 0.0), sq, torch.zeros_like(sq)))
    scale = 0.5 / torch.where(quadratic, a, one)
    lin = -c / torch.where(linear, b, one)
    return (
        exists,
        torch.where(quadratic, (-b + root_sq) * scale, lin),
        torch.where(quadratic, (-b - root_sq) * scale, lin),
    )


def _pqm_mono(ff00, ffll, ffrr, fell, ferr, dell, derr, dfds):
    """GAD_PQM_FUN_MONO: limited coefficients and the ``mono`` flag (0, 1 or 2)."""
    d_m1, d_0, d_p1 = dfds
    zero = torch.zeros_like(ff00)
    one = torch.ones_like(ff00)
    flat = (ffrr - ff00) * (ff00 - ffll) <= 0.0
    left_bad = (ffll - fell) * (fell - ff00) <= 0.0
    fell = torch.where(left_bad, ff00 - d_0, fell)
    right_bad = (ffrr - ferr) * (ferr - ff00) <= 0.0
    ferr = torch.where(right_bad, ff00 + d_0, ferr)
    dell_bad = dell * d_m1 < 0.0
    dell = torch.where(dell_bad, d_m1, dell)
    derr_bad = derr * d_p1 < 0.0
    derr = torch.where(derr_bad, d_p1, derr)
    mono = torch.where(left_bad | right_bad | dell_bad | derr_bad, one, zero)
    coefficients = _pqm_coefficients(ff00, fell, ferr, dell, derr)
    c2, c3, c4, c5 = coefficients[1:]
    exists, root_1, root_2 = _quadratic_roots(12.0 * c5, 6.0 * c4, 2.0 * c3)
    bind = zero
    for root in (root_1, root_2):
        root_2nd = root * root
        dflx = c2 + root * c3 * 2.0 + root_2nd * c4 * 3.0 + root_2nd * root * c5 * 4.0
        turns = exists & (root > -1.0) & (root < 1.0) & (dflx * d_0 < 0.0)
        bind = torch.where(turns, torch.where(dell.abs() < derr.abs(), -one, one), bind)
    left = bind == -1.0
    derr = torch.where(left, -5.0 * ff00 + 3.0 * ferr + 2.0 * fell, derr)
    dell = torch.where(left, (5.0 / 3.0) * ff00 - (1.0 / 3.0) * ferr - (4.0 / 3.0) * fell, dell)
    fix = left & (dell * d_m1 < 0.0)
    dell = torch.where(fix, zero, dell)
    ferr = torch.where(fix, 5.0 * ff00 - 4.0 * fell, ferr)
    derr = torch.where(fix, 10.0 * ff00 - 10.0 * fell, derr)
    fix = left & (derr * d_p1 < 0.0)
    derr = torch.where(fix, zero, derr)
    fell = torch.where(fix, (5.0 / 2.0) * ff00 - (3.0 / 2.0) * ferr, fell)
    dell = torch.where(fix, -(5.0 / 3.0) * ff00 + (5.0 / 3.0) * ferr, dell)
    right = bind == 1.0
    derr = torch.where(right, -(5.0 / 3.0) * ff00 + (4.0 / 3.0) * ferr + (1.0 / 3.0) * fell, derr)
    dell = torch.where(right, 5.0 * ff00 - 2.0 * ferr - 3.0 * fell, dell)
    fix = right & (dell * d_m1 < 0.0)
    dell = torch.where(fix, zero, dell)
    ferr = torch.where(fix, (5.0 / 2.0) * ff00 - (3.0 / 2.0) * fell, ferr)
    derr = torch.where(fix, (5.0 / 3.0) * ff00 - (5.0 / 3.0) * fell, derr)
    fix = right & (derr * d_p1 < 0.0)
    derr = torch.where(fix, zero, derr)
    fell = torch.where(fix, 5.0 * ff00 - 4.0 * ferr, fell)
    dell = torch.where(fix, -10.0 * ff00 + 10.0 * ferr, dell)
    turned = left | right
    mono = torch.where(turned, 2.0 * one, mono)
    refit = _pqm_coefficients(ff00, fell, ferr, dell, derr)
    coefficients = tuple(torch.where(turned, r, c) for r, c in zip(refit, coefficients))
    mono = torch.where(flat, one, mono)
    return tuple(
        torch.where(flat, ff00 if n == 0 else zero, c) for n, c in enumerate(coefficients)
    ), mono


def _pqm_cell(w: _Window, o, limiter: str, recip_spacing_left, recip_spacing_right):
    """GAD_PQM_HAT_R, and GAD_PQM_HAT_X on a wet cell, for cell ``o``: five coefficients.

    ``recip_spacing_left``/``recip_spacing_right`` are the reciprocal spacings
    of the cell's two faces ``o`` and ``o+1``.  GAD_PQM_HAT_X also sets the
    coefficients of a dry cell to zero; its caller does that.
    """
    half_width = w.width(o) * 0.5
    ff00 = w.T(o)
    ffll = w.neighbour(o, -1)
    ffrr = w.neighbour(o, +1)
    fell, dell = _pqm_edge(w, o, recip_spacing_left)
    ferr, derr = _pqm_edge(w, o + 1, recip_spacing_right)
    dell = dell * half_width
    derr = derr * half_width
    if limiter == "null":
        coefficients = _pqm_coefficients(ff00, fell, ferr, dell, derr)
    elif limiter == "mono":
        dfds = _plm_slope(ffll, ff00, ffrr)
        coefficients = _pqm_mono(ff00, ffll, ffrr, fell, ferr, dell, derr, dfds)[0]
    elif limiter == "weno":
        dfds = _plm_slope(ffll, ff00, ffrr)
        unlimited = _pqm_coefficients(ff00, fell, ferr, dell, derr)
        limited, mono = _pqm_mono(ff00, ffll, ffrr, fell, ferr, dell, derr, dfds)
        fdel = (ffrr - ff00).abs() + (ff00 - ffll).abs()
        fmag = ffll.abs() + ff00.abs() + ffrr.abs()
        blend = (mono > 0.0) & (fdel > 1.0e-6 * fmag)
        s1, s2 = _weno_weights(w, o)
        coefficients = tuple(
            torch.where(blend, s1 * u + s2 * lc, lc) for u, lc in zip(unlimited, limited)
        )
    else:
        raise ValueError(f"no PQM limiter {limiter!r}")
    return coefficients


def pqm_transport_flux(
    limiter: str,
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    cells: tuple[torch.Tensor, ...],
    u_trans: torch.Tensor,
    inputs: StencilCellInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order upwind and PQM (``limiter`` null 50, mono 51, weno 52) transport flux.

    GAD_PQM_ADV_X with GAD_PQM_P5E_X (edge slopes times ``recip_dxC`` of the
    face), GAD_PQM_HAT_X (slopes times half the cell width ``dxF``, dry cells
    zero) and GAD_PQM_FLX_X; otherwise as :func:`ppm_transport_flux`.
    """
    upwind_flux = _upwind_donor_flux(T_left, T_right, u_trans)[3]
    w = _Window(cells, inputs)
    # Faces i-1, i, i+1 (entry c of the face tuple is face i+c-h+1).
    spacing_m1, spacing_0, spacing_p1 = inputs.recip_face_spacings[w.h - 2:w.h + 1]
    # GAD_PQM_HAT_X zeroes the coefficients of a dry cell.
    wet_left = w.m(-1) > 0.0
    wet_right = w.m(0) > 0.0
    fhat_left = tuple(
        torch.where(wet_left, c, torch.zeros_like(c))
        for c in _pqm_cell(w, -1, limiter, spacing_m1, spacing_0)
    )
    fhat_right = tuple(
        torch.where(wet_right, c, torch.zeros_like(c))
        for c in _pqm_cell(w, 0, limiter, spacing_0, spacing_p1)
    )
    value = _integrated_face_value(
        fhat_left, fhat_right, w, inputs.velocity, inputs.dt, inputs.velocity,
    )
    flux = torch.where(inputs.velocity == 0.0, torch.zeros_like(value), u_trans * value)
    return upwind_flux, flux


def vertical_pqm_flux(
    limiter: str,
    tracer: torch.Tensor,
    transport: torch.Tensor,
    velocity: torch.Tensor,
    *,
    recip_center_spacing: torch.Tensor,
    layer_thickness: torch.Tensor,
    dt: float,
    mask: torch.Tensor | None,
    halo: int,
) -> torch.Tensor:
    """PQM vertical advective flux at layer interfaces (DOWNWARD-positive, ``[batch, nz+1, ny, nx]``).

    GAD_PQM_ADV_R with GAD_PQM_P5E_R, GAD_PQM_HAT_R and GAD_PQM_FLX_R, on the
    column windows of :func:`vertical_ppm_flux`.  The edge slopes take
    ``recip_center_spacing``, MITgcm ``recip_drC(1 .. Nr+1)`` on every
    interface, surface and bottom included (``gad_pqm_p5e_r.F``; with a given
    ``delRc`` its end values are not ``drF/2``, ``ini_vertical_grid.F:102-107``).  The
    upwind cell and the zero flux follow the sign of the transport (MITgcm
    ``wfac``, ``gad_pqm_flx_r.F``).
    """
    flux = torch.zeros(
        (tracer.shape[0], tracer.shape[1] + 1, tracer.shape[2], tracer.shape[3]),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    if tracer.shape[1] < 2:
        return flux
    wet = torch.ones_like(tracer) if mask is None else mask.to(dtype=tracer.dtype, device=tracer.device)
    cells, masks, widths = vertical_column_windows(tracer, wet, layer_thickness, halo=halo)
    recip_full = recip_center_spacing.to(dtype=tracer.dtype, device=tracer.device).view(1, -1, 1, 1)
    w_vel = velocity[..., 1:-1, :, :]
    w_trans = transport[..., 1:-1, :, :]
    inputs = StencilCellInputs(
        cell_masks=masks, cell_widths=widths, recip_face_spacings=(), velocity=w_vel, dt=dt,
    )
    w = _Window(cells, inputs)
    nz = tracer.shape[1]
    # Interface k (1 .. nz-1): its faces k-1, k, k+1 in the full array.
    spacing_m1 = recip_full[..., 0:nz - 1, :, :]
    spacing_0 = recip_full[..., 1:nz, :, :]
    spacing_p1 = recip_full[..., 2:nz + 1, :, :]
    fhat_above = _pqm_cell(w, -1, limiter, spacing_m1, spacing_0)
    fhat_below = _pqm_cell(w, 0, limiter, spacing_0, spacing_p1)
    value = _integrated_face_value(fhat_above, fhat_below, w, w_vel, dt, w_trans)
    flux[..., 1:-1, :, :] = torch.where(w_trans == 0.0, torch.zeros_like(value), w_trans * value)
    return flux
