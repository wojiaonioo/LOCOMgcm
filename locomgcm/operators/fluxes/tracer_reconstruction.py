"""Tracer face-flux kernels: the scheme formulas on pre-haloed tensors.

Horizontal: the DST3 / DST3-FL (MITgcm schemes 30 / 33) transport flux on
face-aligned four-point stencils ``T(i-2 .. i+1)`` with the three face masks
``maskLoc(i-1, i, i+1)``, the face transport and the face Courant number; the
caller slices the stencils for its layout (region, six-face or MIT tile) and
splices the edge faces.  Vertical: donor-cell / centered (schemes 1 / 2) and
DST3 / DST3-FL interface fluxes for ``[batch, nz, ny, nx]`` tracers and
``[batch, nz+1, ny, nx]`` DOWNWARD-positive interface velocity or transport;
the optional cell wet mask is ``maskC``.  No kernel takes a grid or exchanges a
halo.
"""

from __future__ import annotations

import torch

from locomgcm.operators.abs_zero_slope import abs_with_zero_slope


# ---------------------------------------------------------------------------
# DST3 / DST3 + Sweby flux limiter helpers
# Direct translation of MITgcm pkg/generic_advdiff/gad_dst3fl_adv_{x,y}.F.
# ---------------------------------------------------------------------------

_THETA_MAX = 1.0e20      # gad_dst3fl_adv_x.F:36  PARAMETER(thetaMax=1.D+20)
_CFL_FLOOR = 1.0e-20     # gad_dst3fl_adv_x.F:87  (uCFL+1.D-20)
_ONE_SIXTH = 1.0 / 6.0


def face_courant_number(
    face_velocity: torch.Tensor, recip_face_spacing: torch.Tensor, dt: float,
) -> torch.Tensor:
    """Face Courant number ``|velocity * dt * recip_spacing|`` (``gad_dst3fl_adv_x.F:59`` uCFL)."""
    return (face_velocity * dt * recip_face_spacing).abs()


def _ratio_safe(num: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """Bounded ratio matching MITgcm theta-guard (gad_dst3fl_adv_x.F:74-83):

        IF |denom|*thetaMax <= |num|  ->  SIGN(thetaMax, num*denom)
        ELSE                          ->  num / denom
    """
    abs_num = num.abs()
    abs_denom = denom.abs()
    # Match Fortran ``SIGN(thetaMax, num*denom)`` — sign(0) = +1.
    # Cast result to num.dtype: ``torch.where`` with bare Python floats
    # silently promotes to float32, losing precision at thetaMax=1e20.
    sign = torch.where(num * denom >= 0, 1, -1).to(dtype=num.dtype)
    capped = sign * _THETA_MAX
    use_capped = abs_denom * _THETA_MAX <= abs_num
    # Double where: the division sees ``denom`` only where its quotient is the
    # selected value.  ``use_capped`` holds wherever denom == 0, and also on a
    # tiny nonzero denom under a large num, where the division's denominator
    # adjoint (num/denom)/denom overflows float32 and the zero cotangent of
    # the discarded quotient turned it into NaN.
    safe_denom = torch.where(use_capped, torch.ones_like(denom), denom)
    raw = num / safe_denom
    return torch.where(use_capped, capped, raw)


def _dst3_coefficients(u_cfl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MITgcm DST3 interpolation coefficients ``d0``/``d1``.

    ``d0=(2-CFL)(1-CFL)/6`` and ``d1=(1-CFL²)/6``, identical in
    ``gad_dst3_adv_x.F:85-86`` (scheme 30, direct form) and
    ``gad_dst3fl_adv_x.F:70-71`` (scheme 33, ratio form).  Both call sites read
    them from here so the evaluation order stays one expression.
    """
    d0 = (2.0 - u_cfl) * (1.0 - u_cfl) * _ONE_SIXTH
    d1 = (1.0 - u_cfl * u_cfl) * _ONE_SIXTH
    return d0, d1


def dst3_sweby_limiter(raw: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    """Apply the DST3-FL bounds with the branch derivatives of GAD_DST3FL_ADV_{X,Y,R}_B."""
    capped = torch.where(raw >= 1.0, torch.ones_like(raw), raw)
    # Tapenade's MIN takes its first argument at equality. Keep minimum's
    # NaN propagation for unordered values and clamp's signed-zero value.
    limited = torch.where(capped == bound, capped, torch.minimum(capped, bound))
    return torch.where(limited == 0.0, limited.detach(), limited.clamp_min(0.0))


def _dst3_psi(theta: torch.Tensor, u_cfl: torch.Tensor, *, with_limiter: bool) -> torch.Tensor:
    """ψ(θ, CFL) for the DST3-FL (scheme 33) face reconstruction.

    DST3-FL (Sweby):    ψ = max(0, min(min(1, d0+d1·θ), θ·(1−CFL)/(CFL+ε)))

    Mirrors gad_dst3fl_adv_x.F:61-90.  ``with_limiter=False`` returns the raw
    ``d0+d1·θ``; it is the ratio-form DST3 that MITgcm compiles only under
    ``#define OLD_DST3_FORMULATION`` (nowhere in the tree), so the scheme-30
    flux in :func:`dst3_transport_flux` does NOT route through here —
    it uses the direct form ``d0·Rj + d1·Rjm`` instead.
    """
    d0, d1 = _dst3_coefficients(u_cfl)
    psi = d0 + d1 * theta
    if not with_limiter:
        return psi
    numerator = theta * (1.0 - u_cfl)
    denominator = u_cfl + _CFL_FLOOR
    # Double where.  Below the bound sits min(1, .) and above it clamp(min=0),
    # so only a bound in [0, 1] can be the selected branch; there its
    # denominator adjoint bound/denominator is at most 1/1e-20.  Elsewhere the
    # bound is discarded, yet in float32 (numerator/denominator)/denominator
    # overflows (theta*1e40 at u_cfl = 0, or a capped theta with u_cfl below
    # about 5e-10) and its zero cotangent gave 0*inf = NaN.  The
    # discarded entries keep the same forward value without a derivative path.
    bound_value = numerator.detach() / denominator.detach()
    live = (bound_value >= 0.0) & (bound_value <= 1.0)
    sweby_bound = torch.where(
        live,
        numerator / torch.where(live, denominator, torch.ones_like(denominator)),
        bound_value,
    )
    return dst3_sweby_limiter(psi, sweby_bound)


def _upwind_donor_flux(
    T_left: torch.Tensor, T_right: torch.Tensor, u_trans: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(|uTrans|, 0.5(uTrans+|uTrans|), 0.5(uTrans-|uTrans|), pos*T_left + neg*T_right)``.

    The first-order upwind flux every face-stencil kernel returns next to its
    own flux, for the caller to splice onto uncoupled edge faces.
    """
    # |u_trans| with the reference ABS(0) adjoint slope: MITgcm-AD records the
    # ``.GE. 0`` branch, i.e. +1 at zero transport.  Forward values unchanged.
    abs_u = abs_with_zero_slope(u_trans, 1.0)
    pos = 0.5 * (u_trans + abs_u)
    neg = 0.5 * (u_trans - abs_u)
    upwind_flux = pos * T_left + neg * T_right
    return abs_u, pos, neg, upwind_flux


def _stencil_upwind_and_differences(
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    Tm2: torch.Tensor,
    Tm1: torch.Tensor,
    T0: torch.Tensor,
    Tp1: torch.Tensor,
    mask_im1: torch.Tensor,
    mask_i: torch.Tensor,
    mask_ip1: torch.Tensor,
    u_trans: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Donor weights, first-order upwind flux and masked differences of a four-point face stencil.

    Returns ``(abs_u, pos, neg, upwind_flux, Rjp, Rj, Rjm)`` on the stencil's
    faces: ``abs_u = |uTrans|``, ``pos/neg = 0.5(uTrans +/- |uTrans|)``,
    ``upwind_flux = pos*T_left +
    neg*T_right`` and ``Rjp/Rj/Rjm`` the face-masked tracer differences at faces
    ``i+1, i, i-1`` (gad_dst3fl_adv_x.F:54-56).
    """
    abs_u, pos, neg, upwind_flux = _upwind_donor_flux(T_left, T_right, u_trans)

    Rjp = (Tp1 - T0) * mask_ip1
    Rj = (T0 - Tm1) * mask_i
    Rjm = (Tm1 - Tm2) * mask_im1
    return abs_u, pos, neg, upwind_flux, Rjp, Rj, Rjm


def dst3_transport_flux(
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    Tm2: torch.Tensor,
    Tm1: torch.Tensor,
    T0: torch.Tensor,
    Tp1: torch.Tensor,
    mask_im1: torch.Tensor,
    mask_i: torch.Tensor,
    mask_ip1: torch.Tensor,
    u_trans: torch.Tensor,
    u_cfl: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order upwind and DST3 (scheme 30) transport flux on the same faces.

    Face-aligned pre-haloed stencils ``[*, nz, ny, nx+1]`` (X) or ``[*, nz, ny+1,
    nx]`` (Y); the math is element-wise and axis-agnostic.  The adapter splices
    the DST3 value into the faces that couple two cells.
    """
    _abs_u, pos, neg, upwind_flux, Rjp, Rj, Rjm = _stencil_upwind_and_differences(
        T_left, T_right, Tm2, Tm1, T0, Tp1, mask_im1, mask_i, mask_ip1, u_trans,
    )
    # Scheme 30 (GAD_DST3_ADV_X, gad_dst3_adv_x.F:109-115, the ``#else
    # OLD_DST3_FORMULATION`` branch -- that macro is nowhere #define'd in the
    # tree, so this is MITgcm's default):
    #   uT = 0.5(uTrans+|uTrans|)(T(i-1) + (d0*Rj+d1*Rjm))
    #      + 0.5(uTrans-|uTrans|)(T( i ) - (d0*Rj+d1*Rjp))
    # This is NOT ``psi*Rj`` with the limiter switched off: the ratio form
    # carries ``(d0 + d1*Rjm/Rj)*Rj``, which collapses to 0 wherever ``Rj=0``
    # while MITgcm still contributes ``d1*Rjm`` there -- a first-order term,
    # not a rounding difference.  The direct form never forms the ratio.
    d0, d1 = _dst3_coefficients(u_cfl)
    flux_dst = (
        pos * (Tm1 + (d0 * Rj + d1 * Rjm))
        + neg * (T0 - (d0 * Rj + d1 * Rjp))
    )
    return upwind_flux, flux_dst


def dst3fl_transport_flux(
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    Tm2: torch.Tensor,
    Tm1: torch.Tensor,
    T0: torch.Tensor,
    Tp1: torch.Tensor,
    mask_im1: torch.Tensor,
    mask_i: torch.Tensor,
    mask_ip1: torch.Tensor,
    u_trans: torch.Tensor,
    u_cfl: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order upwind and DST3-FL (scheme 33) transport flux on the same faces.

    Same stencils as :func:`dst3_transport_flux`; GAD_DST3FL_ADV_X's ratio form
    through the Sweby limiter.
    """
    _abs_u, pos, neg, upwind_flux, Rjp, Rj, Rjm = _stencil_upwind_and_differences(
        T_left, T_right, Tm2, Tm1, T0, Tp1, mask_im1, mask_i, mask_ip1, u_trans,
    )
    thetaP = _ratio_safe(Rjm, Rj)
    thetaM = _ratio_safe(Rjp, Rj)
    psiP = _dst3_psi(thetaP, u_cfl, with_limiter=True)
    psiM = _dst3_psi(thetaM, u_cfl, with_limiter=True)
    flux_dst = pos * (Tm1 + psiP * Rj) + neg * (T0 - psiM * Rj)
    return upwind_flux, flux_dst


_CR_MAX = 1.0e6          # gad_fluxlimit_adv_x.F:63  PARAMETER( CrMax = 1.D+6 )


def _fluxlimit_ratio(cr: torch.Tensor, rj: torch.Tensor) -> torch.Tensor:
    """Bounded slope ratio of GAD_FLUXLIMIT_ADV_{X,Y,R} (gad_fluxlimit_adv_x.F:89-93).

        IF |Rj|*CrMax <= |Cr|  ->  SIGN(CrMax, Cr) * SIGN(1, Rj)
        ELSE                   ->  Cr / Rj

    The two SIGNs are taken separately, not from the product ``Cr*Rj``; a
    signed zero counts as positive.  Double where: the division sees ``Rj``
    only where its quotient is selected.
    """
    capped_sel = rj.abs() * _CR_MAX <= cr.abs()
    sign_cr = torch.where(cr >= 0, 1, -1).to(dtype=cr.dtype)
    sign_rj = torch.where(rj >= 0, 1, -1).to(dtype=cr.dtype)
    capped = sign_cr * _CR_MAX * sign_rj
    safe_rj = torch.where(capped_sel, torch.ones_like(rj), rj)
    return torch.where(capped_sel, capped, cr / safe_rj)


def _superbee(cr: torch.Tensor) -> torch.Tensor:
    """``Limiter(Cr) = max(0, max(min(1, 2Cr), min(2, Cr)))`` (GAD_FLUX_LIMITER.h:41-42)."""
    return torch.clamp(
        torch.maximum(torch.clamp(2.0 * cr, max=1.0), torch.clamp(cr, max=2.0)),
        min=0.0,
    )


def fluxlimit_transport_flux(
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    Tm2: torch.Tensor,
    Tm1: torch.Tensor,
    T0: torch.Tensor,
    Tp1: torch.Tensor,
    mask_im1: torch.Tensor,
    mask_i: torch.Tensor,
    mask_ip1: torch.Tensor,
    u_trans: torch.Tensor,
    u_cfl: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order upwind and flux-limited (scheme 77) transport flux on the same faces.

    GAD_FLUXLIMIT_ADV_X (gad_fluxlimit_adv_x.F:74-103) with ``calcCFL``:
    ``uT = uTrans*(T(i)+T(i-1))*0.5 - |uTrans|*((1-Cr)+uCFL*Cr)*Rj*0.5``, the
    slope ratio ``Cr`` taken upstream (``Rjm`` for ``uTrans>0``, else ``Rjp``)
    and limited by Superbee.  Stencils as :func:`dst3_transport_flux`.
    """
    abs_u, _pos, _neg, upwind_flux, Rjp, Rj, Rjm = _stencil_upwind_and_differences(
        T_left, T_right, Tm2, Tm1, T0, Tp1, mask_im1, mask_i, mask_ip1, u_trans,
    )
    cr = _superbee(_fluxlimit_ratio(torch.where(u_trans > 0.0, Rjm, Rjp), Rj))
    flux = u_trans * (T0 + Tm1) * 0.5 - abs_u * ((1.0 - cr) + u_cfl * cr) * Rj * 0.5
    return upwind_flux, flux


def upwind3_transport_flux(
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    Tm2: torch.Tensor,
    Tm1: torch.Tensor,
    T0: torch.Tensor,
    Tp1: torch.Tensor,
    mask_im1: torch.Tensor,
    mask_i: torch.Tensor,
    mask_ip1: torch.Tensor,
    u_trans: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order upwind and third-order upwind (scheme 3) transport flux on the same faces.

    GAD_U3_ADV_X (gad_u3_adv_x.F:55-72)::

        uT = uTrans*(T(i)+T(i-1) - oneSixth*(Rjjp+Rjjm))*0.5
           + |uTrans|*0.5*oneSixth*(Rjjp-Rjjm)

    with ``Rjjp = Rjp-Rj``, ``Rjjm = Rj-Rjm``.  No Courant number.  Stencils as
    :func:`dst3_transport_flux`.
    """
    abs_u, _pos, _neg, upwind_flux, Rjp, Rj, Rjm = _stencil_upwind_and_differences(
        T_left, T_right, Tm2, Tm1, T0, Tp1, mask_im1, mask_i, mask_ip1, u_trans,
    )
    Rjjp = Rjp - Rj
    Rjjm = Rj - Rjm
    flux = (
        u_trans * (T0 + Tm1 - _ONE_SIXTH * (Rjjp + Rjjm)) * 0.5
        + abs_u * 0.5 * _ONE_SIXTH * (Rjjp - Rjjm)
    )
    return upwind_flux, flux


def centered4_transport_flux(
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    Tm2: torch.Tensor,
    Tm1: torch.Tensor,
    T0: torch.Tensor,
    Tp1: torch.Tensor,
    mask_im1: torch.Tensor,
    mask_i: torch.Tensor,
    mask_ip1: torch.Tensor,
    u_trans: torch.Tensor,
    wet_im1: torch.Tensor,
    wet_ip1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order upwind and fourth-order centered (scheme 4) transport flux on the same faces.

    GAD_C4_ADV_X (gad_c4_adv_x.F:60-75)::

        uT = uTrans*(T(i)+T(i-1) - oneSixth*(Rjjp+Rjjm))*0.5
           + |uTrans|*0.5*oneSixth*(Rjjp-Rjjm)*(1 - maskW(i-1)*maskW(i+1))

    The third-order upwind term of :func:`upwind3_transport_flux` survives
    only where a neighbouring face is dry.  ``wet_im1``/``wet_ip1`` are the
    PHYSICAL face masks ``maskW(i-1)``, ``maskW(i+1)``, without the open
    boundary factor ``maskInW`` that ``mask_im1``/``mask_ip1`` carry.  No
    Courant number.  Stencils as :func:`dst3_transport_flux`.
    """
    abs_u, _pos, _neg, upwind_flux, Rjp, Rj, Rjm = _stencil_upwind_and_differences(
        T_left, T_right, Tm2, Tm1, T0, Tp1, mask_im1, mask_i, mask_ip1, u_trans,
    )
    Rjjp = Rjp - Rj
    Rjjm = Rj - Rjm
    flux = (
        u_trans * (T0 + Tm1 - _ONE_SIXTH * (Rjjp + Rjjm)) * 0.5
        + abs_u * 0.5 * _ONE_SIXTH * (Rjjp - Rjjm) * (1.0 - wet_im1 * wet_ip1)
    )
    return upwind_flux, flux


_OS7MP_EPS = 1.0e-20      # gad_os7mp_adv_x.F:48  Eps = 1.D-20


def _sign_one(x: torch.Tensor) -> torch.Tensor:
    """Fortran ``SIGN(1, x)``; ``x = -0`` takes the positive branch (ifx)."""
    return torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))


def _os7mp_oriented_flux(
    Qippp: torch.Tensor,
    Qipp: torch.Tensor,
    Qip: torch.Tensor,
    Qi: torch.Tensor,
    Qim: torch.Tensor,
    Qimm: torch.Tensor,
    Qimmm: torch.Tensor,
    MskIpp: torch.Tensor,
    MskIp: torch.Tensor,
    MskI: torch.Tensor,
    MskIm: torch.Tensor,
    MskImm: torch.Tensor,
    MskImmm: torch.Tensor,
    transport: torch.Tensor,
    cfl: torch.Tensor,
) -> torch.Tensor:
    """OS7MP face flux from the upwind-oriented stencil, shared by X, Y and R.

    ``gad_os7mp_adv_x.F:109-197`` statement for statement, from the oriented
    values ``Q(i+2 .. i-4)`` and masks of the ``uTrans > 0`` branch (the caller
    mirrors them for the other sign); zero where ``transport`` is zero.
    """
    Fac = 1.0
    DelP = (Qip - Qi) * MskI
    Phi = Fac * DelP
    Fac = Fac * (cfl + 1.0) / 3.0
    DelM = (Qi - Qim) * MskIm
    Del2 = DelP - DelM
    Phi = Phi - Fac * Del2
    Fac = Fac * (cfl - 2.0) / 4.0
    DelPP = (Qipp - Qip) * MskIp * MskI
    Del2P = DelPP - DelP
    Del3P = Del2P - Del2
    Phi = Phi + Fac * Del3P
    Fac = Fac * (cfl - 3.0) / 5.0
    DelMM = (Qim - Qimm) * MskImm * MskIm
    Del2M = DelM - DelMM
    Del3M = Del2 - Del2M
    Del4 = Del3P - Del3M
    Phi = Phi + Fac * Del4
    Fac = Fac * (cfl + 2.0) / 6.0
    DelPPP = (Qippp - Qipp) * MskIpp * MskIp * MskI
    Del2PP = DelPP - DelP
    Del3PP = Del2PP - Del2P
    Del4P = Del3PP - Del3P
    Del5P = Del4P - Del4
    Phi = Phi + Fac * Del5P
    # MITgcm writes (cfl + 2) here as well, not (cfl + 3).
    Fac = Fac * (cfl + 2.0) / 7.0
    DelMMM = (Qimm - Qimmm) * MskImmm * MskImm * MskIm
    Del2MM = DelMM - DelMMM
    Del3MM = Del2M - Del2MM
    Del4M = Del3M - Del3MM
    Del5M = Del4 - Del4M
    Del6 = Del5P - Del5M
    Phi = Phi - Fac * Del6
    DelIp = (Qip - Qi) * MskI
    recip_DelIp = _sign_one(DelIp) / torch.clamp(DelIp.abs(), min=_OS7MP_EPS)
    Phi = Phi * recip_DelIp
    DelI = (Qi - Qim) * MskIm
    recip_DelI = _sign_one(DelI) / torch.clamp(DelI.abs(), min=_OS7MP_EPS)
    rp1h = DelI * recip_DelIp
    rp1h_cfl = rp1h / (cfl + _OS7MP_EPS)
    d2 = Del2
    d2p1 = Del2P
    d2m1 = Del2M
    A = 4.0 * d2 - d2p1
    B = 4.0 * d2p1 - d2
    C = d2
    D = d2p1
    dp1h = (
        torch.clamp(torch.minimum(torch.minimum(A, B), torch.minimum(C, D)), min=0.0)
        + torch.clamp(torch.maximum(torch.maximum(A, B), torch.maximum(C, D)), max=0.0)
    )
    A = 4.0 * d2m1 - d2
    B = 4.0 * d2 - d2m1
    C = d2m1
    D = d2
    dm1h = (
        torch.clamp(torch.minimum(torch.minimum(A, B), torch.minimum(C, D)), min=0.0)
        + torch.clamp(torch.maximum(torch.maximum(A, B), torch.maximum(C, D)), max=0.0)
    )
    PhiMD = 1.0 / (1.0 - cfl) * (DelIp - dp1h) * recip_DelIp
    PhiLC = rp1h_cfl * (1.0 + dm1h * recip_DelI)
    PhiMin = torch.maximum(
        torch.clamp(PhiMD, max=0.0),
        torch.minimum(torch.clamp(2.0 * rp1h_cfl, max=0.0), PhiLC),
    )
    PhiMax = torch.minimum(
        torch.maximum(2.0 / (1.0 - cfl), PhiMD),
        torch.maximum(torch.clamp(2.0 * rp1h_cfl, min=0.0), PhiLC),
    )
    Phi = torch.maximum(PhiMin, torch.minimum(Phi, PhiMax))
    Psi = Phi * 0.5 * (1.0 - cfl)
    flux = transport * (Qi + Psi * DelIp)
    return torch.where(transport != 0.0, flux, torch.zeros_like(flux))



def os7mp_transport_flux(
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    cells: tuple[torch.Tensor, ...],
    face_masks: tuple[torch.Tensor, ...],
    u_trans: torch.Tensor,
    u_cfl: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order upwind and OS7MP (scheme 7) transport flux on the same faces.

    GAD_OS7MP_ADV_X (gad_os7mp_adv_x.F:60-199).  ``cells`` are ``T(i-4) ..
    T(i+3)``, ``face_masks`` ``maskLocW(i-3) .. maskLocW(i+3)``; the stencil is
    oriented upwind by the sign of ``u_trans`` (``uTrans > 0`` reads ``Q(i+2 ..
    i-4)``, ``uTrans < 0`` the mirror ``Q(i-3 .. i+3)``) and passed to
    :func:`_os7mp_oriented_flux`.
    """
    _abs_u, _pos, _neg, upwind_flux = _upwind_donor_flux(T_left, T_right, u_trans)
    c, m = cells, face_masks
    pos = u_trans > 0.0
    flux = _os7mp_oriented_flux(
        torch.where(pos, c[6], c[1]),
        torch.where(pos, c[5], c[2]),
        torch.where(pos, c[4], c[3]),
        torch.where(pos, c[3], c[4]),
        torch.where(pos, c[2], c[5]),
        torch.where(pos, c[1], c[6]),
        torch.where(pos, c[0], c[7]),
        torch.where(pos, m[5], m[1]),
        torch.where(pos, m[4], m[2]),
        m[3],
        torch.where(pos, m[2], m[4]),
        torch.where(pos, m[1], m[5]),
        torch.where(pos, m[0], m[6]),
        u_trans,
        u_cfl,
    )
    return upwind_flux, flux


def upwind_face_flux(
    T_left: torch.Tensor, T_right: torch.Tensor, transport: torch.Tensor, velocity: torch.Tensor
) -> torch.Tensor:
    """First-order upwind face flux ``transport * (T_left if velocity >= 0 else T_right)`` (scheme 1)."""
    return transport * torch.where(velocity >= 0.0, T_left, T_right)


def centered_face_flux(
    T_left: torch.Tensor, T_right: torch.Tensor, transport: torch.Tensor
) -> torch.Tensor:
    """Second-order centered face flux ``transport * 0.5*(T_left + T_right)`` (scheme 2)."""
    return transport * (0.5 * (T_left + T_right))


def dst2_face_flux(
    T_left: torch.Tensor, T_right: torch.Tensor, transport: torch.Tensor, cfl: torch.Tensor
) -> torch.Tensor:
    """Second-order Direct Space-Time (Lax-Wendroff, scheme 20) face flux.

    GAD_DST2U1_ADV_X with ``xLimit = 1`` (gad_dst2u1_adv_x.F:57-81)::

        uAbs = |uTrans| * (1 - (1 - uCFL))
        uT   = (uTrans+uAbs)*0.5*T(i-1) + (uTrans-uAbs)*0.5*T(i)

    ``1 - (1 - uCFL)`` is evaluated as written.  ``|uTrans|`` takes the
    reference ABS(0) adjoint slope +1.
    """
    abs_t = abs_with_zero_slope(transport, 1.0) * (1.0 - 1.0 * (1.0 - cfl))
    return (transport + abs_t) * 0.5 * T_left + (transport - abs_t) * 0.5 * T_right


def first_order_upwind_flux(
    transport: torch.Tensor, T_left: torch.Tensor, T_right: torch.Tensor
) -> torch.Tensor:
    """First-order upwind face flux ``0.5(a+|a|)T_left + 0.5(a-|a|)T_right``.

    ``transport`` is the face velocity or transport ``a``; ``T_left``/``T_right``
    are the cells before and after the face along its axis.  ``|a|`` is the
    plain ``abs`` (zero slope at ``a = 0``), as in ``OBCS_U1_ADV_TRACER``'s
    replacement of the open-boundary faces.
    """
    abs_transport = transport.abs()
    return 0.5 * (transport + abs_transport) * T_left + 0.5 * (transport - abs_transport) * T_right


def _interior_interface_flux(
    tracer: torch.Tensor,
    w_interior: torch.Tensor,
    tracer_interp: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """``w * T_face`` on the interior interfaces, zero at the top and bottom.

    ``mask`` (cell wet mask, 1=wet, 0=dry, ``[nz, ny, nx]`` or ``[batch, nz, ny,
    nx]``) multiplies each interior interface by ``mask[k] * mask[k-1]`` so that
    flux across a wet/dry boundary is exactly zero.  This prevents extrapolated
    tracer values in dry cells from leaking into the advective flux, which
    (after division by small hFacC in partial cells) can trigger NaN
    instability.
    """
    flux = torch.zeros(
        (
            tracer.shape[0],
            tracer.shape[1] + 1,
            tracer.shape[2],
            tracer.shape[3],
        ),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    if tracer.shape[1] < 2:
        return flux

    flux[..., 1:-1, :, :] = w_interior * tracer_interp

    # Zero flux at wet/dry interfaces (matches vertical_diffusive_flux convention)
    if mask is not None:
        m = mask.to(dtype=tracer.dtype, device=tracer.device)
        if m.ndim == 3:
            m = m.unsqueeze(0)  # [1, nz, ny, nx]
        interface_mask = m[..., 1:, :, :] * m[..., :-1, :, :]
        flux[..., 1:-1, :, :] = flux[..., 1:-1, :, :] * interface_mask

    return flux


def vertical_upwind_flux(
    tracer: torch.Tensor,
    vertical_velocity_interface: torch.Tensor,
    *,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """1st-order donor-cell vertical advective flux (scheme 1) at layer interfaces.

    ``tracer`` is ``[batch, nz, ny, nx]``, ``vertical_velocity_interface``
    ``[batch, nz+1, ny, nx]`` DOWNWARD-positive (velocity or volume transport):
    ``w >= 0`` takes the donor above.  ``mask`` as :func:`_interior_interface_flux`.
    """
    w_interior = vertical_velocity_interface[..., 1:-1, :, :]
    tracer_interp = torch.where(
        w_interior >= 0.0, tracer[..., :-1, :, :], tracer[..., 1:, :, :]
    )
    return _interior_interface_flux(tracer, w_interior, tracer_interp, mask)


def vertical_centered_flux(
    tracer: torch.Tensor,
    vertical_velocity_interface: torch.Tensor,
    *,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """2nd-order centered vertical advective flux at layer interfaces.

    Matches MITgcm ``tempAdvScheme = ENUM_CENTERED_2ND``.  Shapes, sign and
    ``mask`` as :func:`vertical_upwind_flux`.
    """
    w_interior = vertical_velocity_interface[..., 1:-1, :, :]
    tracer_interp = 0.5 * (tracer[..., :-1, :, :] + tracer[..., 1:, :, :])
    return _interior_interface_flux(tracer, w_interior, tracer_interp, mask)


def _dst3_vertical_stencil(
    tracer: torch.Tensor,
    vertical_velocity_interface: torch.Tensor,
    *,
    recip_drC: torch.Tensor,
    dt: float,
    mask: torch.Tensor | None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Stencil, masked R-differences, CFL and ``d0``/``d1`` shared by DST3 / DST3-FL.

    ``GAD_DST3_ADV_R`` (scheme 30) and ``GAD_DST3FL_ADV_R`` (scheme 33) build
    exactly the same ``Rjp/Rj/Rjm``, the same ``km2=MAX(1,k-2)`` /
    ``kp1=MIN(Nr,k+1)`` clamping, the same ``maskC`` factors and the same
    ``cfl=ABS(wLoc*dTarg*recip_drC(k))``; only the reconstruction that consumes
    them differs.  Returned in MITgcm's own naming so both callers read against
    the Fortran line-for-line.

    Returns ``(w, t_above, t_below, Rjp, Rj, Rjm, wCFL, d0, d1)`` on the interior
    interfaces ``i = 1..nz-1``; ``t_above`` is MIT ``tracer(km1)``, ``t_below``
    is MIT ``tracer(k)``.
    """
    # interior interfaces i = 1..nz-1 (downward-positive velocity)
    w = vertical_velocity_interface[..., 1:-1, :, :]

    # 4-point stencil about interface i: below=i, above=i-1, below2=i+1, above2=i-2,
    # with MITgcm's km2=MAX(1,k-2)/kp1=MIN(Nr,k+1) clamping at top/bottom.
    t_above = tracer[..., :-1, :, :]                                              # T[i-1]
    t_below = tracer[..., 1:, :, :]                                               # T[i]
    t_below2 = torch.cat((tracer[..., 2:, :, :], tracer[..., -1:, :, :]), dim=-3)  # T[min(i+1, nz-1)]
    t_above2 = torch.cat((tracer[..., :1, :, :], tracer[..., :-2, :, :]), dim=-3)  # T[max(i-2, 0)]

    if mask is not None:
        m = mask.to(dtype=tracer.dtype, device=tracer.device)
        if m.ndim == 3:
            m = m.unsqueeze(0)
        m_above = m[..., :-1, :, :]
        m_below = m[..., 1:, :, :]
        m_below2 = torch.cat((m[..., 2:, :, :], m[..., -1:, :, :]), dim=-3)
    else:
        m_above = m_below = m_below2 = torch.ones_like(t_above)

    Rjp = (t_below - t_below2) * m_below2
    Rj = (t_above - t_below) * m_below * m_above
    Rjm = (t_above2 - t_above) * m_above

    recip = recip_drC.to(dtype=tracer.dtype, device=tracer.device)
    if recip.ndim == 1:
        recip = recip.view(1, -1, 1, 1)
    wCFL = (w * dt * recip).abs()
    d0, d1 = _dst3_coefficients(wCFL)
    return w, t_above, t_below, Rjp, Rj, Rjm, wCFL, d0, d1


def vertical_advective_flux_dst3(
    tracer: torch.Tensor,
    vertical_velocity_interface: torch.Tensor,
    *,
    recip_drC: torch.Tensor,
    dt: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """3rd-order DST vertical advective flux, no limiter (MITgcm scheme 30).

    Translates ``GAD_DST3_ADV_R`` (``pkg/generic_advdiff/gad_dst3_adv_r.F``
    :110-116, the ``#else OLD_DST3_FORMULATION`` branch -- that macro is
    nowhere ``#define``'d in the tree, so it is MITgcm's default) into
    LOCOMgcm's DOWNWARD-positive ``w`` convention. The Fortran reads::

        wT = 0.5(rTrans+|rTrans|) * ( T(k)   + (d0*Rj + d1*Rjp) )
           + 0.5(rTrans-|rTrans|) * ( T(km1) - (d0*Rj + d1*Rjm) )

    with ``rTrans`` up-positive.  LOCOMgcm's ``w`` is ``-rTrans`` and the
    returned flux is ``-wT_MIT`` (the same relation
    :func:`vertical_advective_flux_dst3fl` and the upwind branch use), which
    swaps the two branches::

        flux = 0.5(w+|w|) * ( T(km1) - (d0*Rj + d1*Rjm) )
             + 0.5(w-|w|) * ( T( k ) + (d0*Rj + d1*Rjp) )

    so a downward ``w>0`` takes the donor ABOVE, as it must.  This is NOT the
    limiter kernel with ``psi`` switched off: the ratio form carries
    ``(d0 + d1*Rjm/Rj)*Rj``, which vanishes wherever ``Rj=0`` while MITgcm still
    contributes ``d1*Rjm`` there -- a first-order term, not a rounding
    difference.

    ``rkSign`` note (the MIT vertical-flux sign convention): as in the DST3-FL kernel, rkSign lives in
    MITgcm's flux *divergence*, not here; this kernel uses the natural sign of
    the transport.

    Parameters are identical to :func:`vertical_advective_flux_dst3fl`.
    """
    flux = torch.zeros(
        (tracer.shape[0], tracer.shape[1] + 1, tracer.shape[2], tracer.shape[3]),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    nz = tracer.shape[1]
    if nz < 2:
        return flux

    w, t_above, t_below, Rjp, Rj, Rjm, _wCFL, d0, d1 = _dst3_vertical_stencil(
        tracer, vertical_velocity_interface, recip_drC=recip_drC, dt=dt, mask=mask,
    )

    # |w| with the reference ABS(0) adjoint slope.  MITgcm-AD differentiates
    # ABS(rTrans) through its ``.GE. 0`` branch (+1 at zero); ``w`` here is the
    # downward transport, i.e. -rTrans, so the same convention is -1 at zero.
    abs_w = abs_with_zero_slope(w, -1.0)
    flux[..., 1:-1, :, :] = (
        0.5 * (w + abs_w) * (t_above - (d0 * Rj + d1 * Rjm))
        + 0.5 * (w - abs_w) * (t_below + (d0 * Rj + d1 * Rjp))
    )
    return flux


def dst3_limiter_ratio(numerator: torch.Tensor, cfl: torch.Tensor) -> torch.Tensor:
    """Evaluate a DST3 bound with a finite denominator adjoint.

    The numerator is 1-CFL for the explicit bound and 1+CFL for the implicit
    bound. CFL is nonnegative and dimensionless; the result keeps its dtype.
    """
    # The denominator adjoint value/(cfl+1e-20) exceeds float32 range near
    # zero CFL. A discarded bound then receives zero cotangent and 0*inf
    # produces NaN. Keep the forward value there without a derivative path;
    # every float64 entry retains the original expression and derivative.
    # ponytail: a selected bound at tiny nonzero CFL loses this ratio's
    # derivative where it overflowed; reordering the complete bound could
    # retain that path but would change its forward rounding.
    denominator = cfl + 1.0e-20
    value = numerator.detach() / denominator.detach()
    live = torch.isfinite(value / denominator.detach())
    return torch.where(
        live,
        numerator / torch.where(live, denominator, torch.ones_like(denominator)),
        value,
    )


def vertical_advective_flux_dst3fl(
    tracer: torch.Tensor,
    vertical_velocity_interface: torch.Tensor,
    *,
    recip_drC: torch.Tensor,
    dt: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """3rd-order DST flux-limited vertical advective flux (MITgcm scheme 33).

    Translates ``GAD_DST3FL_ADV_R`` (``pkg/generic_advdiff/gad_dst3fl_adv_r.F``)
    into LOCOMgcm's DOWNWARD-positive ``w`` convention.  In MITgcm the flux is
    up-positive (``rTrans>0`` = upward, donor below); here ``w>0`` = downward
    (donor above), so the two branches swap relative to the Fortran::

        flux = 0.5(w+|w|)(t_above - psiP*Rj) + 0.5(w-|w|)(t_below + psiM*Rj)

    This is the same down-positive relationship (``flux_TO = -wT_MIT``) the upwind
    branch above uses, so it composes with the same ``vertical_flux_divergence``
    and reduces to upwind when ``psiP=psiM=0``. The limiter (Rj/Rjp/Rjm, wCFL,
    psi) is tracer/CFL-based and convention-independent — only which donor each
    branch picks flips.

    rkSign note (the MIT vertical-flux sign convention): rkSign lives in MITgcm's flux *divergence*
    (gad_advection.F:1039), NOT in this flux kernel, which uses the natural sign
    of the transport.  Feeding LOCOMgcm's down-positive ``w`` and keeping LOCOMgcm's existing
    divergence reproduces the same physics — verified at the op level against a
    MIT ``fVerT`` dump, not by self-consistency.

    Parameters
    ----------
    tracer : Tensor [batch, nz, ny, nx]
    vertical_velocity_interface : Tensor [batch, nz+1, ny, nx]
        Interface vertical velocity, DOWNWARD-positive (LOCOMgcm convention).
    recip_drC : Tensor [nz-1] (or broadcastable to the interior interfaces)
        ``1 / (center_depth[i] - center_depth[i-1])`` at interior interfaces
        ``i = 1..nz-1`` (MITgcm ``recip_drC``, center-to-center spacing — NOT
        ``recip_drF`` / layer thickness).  Used only for the Courant number.
    dt : float
        Tracer time step [s] (MITgcm ``deltaTtracer`` / ``dTarg``).
    mask : optional Tensor [nz, ny, nx] or [batch, nz, ny, nx]
        Cell wet mask (1=wet, 0=dry), MITgcm ``maskC``.  Zeroes the tracer
        differences (Rj/Rjp/Rjm) across wet/dry faces exactly as the Fortran.
    """
    flux = torch.zeros(
        (tracer.shape[0], tracer.shape[1] + 1, tracer.shape[2], tracer.shape[3]),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    nz = tracer.shape[1]
    if nz < 2:
        return flux

    w, t_above, t_below, Rjp, Rj, Rjm, wCFL, d0, d1 = _dst3_vertical_stencil(
        tracer, vertical_velocity_interface, recip_drC=recip_drC, dt=dt, mask=mask,
    )

    # Bounded theta = Rjm/Rj (P) and Rjp/Rj (M); MITgcm caps |theta| at thetaMax
    # and (Fortran SIGN) treats theta=0 product as +.  psi*Rj=0 where Rj=0 so the
    # flux is unaffected regardless of the capped theta value.
    thetaP = _ratio_safe(Rjm, Rj)
    thetaM = _ratio_safe(Rjp, Rj)

    lim = dst3_limiter_ratio(1.0 - wCFL, wCFL)
    psiP = dst3_sweby_limiter(d0 + d1 * thetaP, thetaP * lim)
    psiM = dst3_sweby_limiter(d0 + d1 * thetaM, thetaM * lim)

    # |w| with the reference ABS(0) adjoint slope.  MITgcm-AD differentiates
    # ABS(rTrans) through its ``.GE. 0`` branch (+1 at zero); ``w`` here is the
    # downward transport, i.e. -rTrans, so the same convention is -1 at zero.
    abs_w = abs_with_zero_slope(w, -1.0)
    flux[..., 1:-1, :, :] = (
        0.5 * (w + abs_w) * (t_above - psiP * Rj)
        + 0.5 * (w - abs_w) * (t_below + psiM * Rj)
    )
    return flux


def _vertical_four_point_stencil(
    tracer: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, ...]:
    """Column stencil and R-differences shared by the vertical schemes 77, 3 and 4.

    On the interior interfaces ``1 .. nz-1`` (MITgcm ``k = 2 .. Nr``) with
    ``km2 = MAX(1,k-2)`` and ``kp1 = MIN(Nr,k+1)`` clamped to the column, returns
    ``(T(km2), T(km1), T(k), maskC(km2), maskC(km1), maskC(kp1), Rjp, Rj, Rjm)``
    with ``Rjp = (T(kp1)-T(k))*maskC(kp1)``, ``Rj = T(k)-T(km1)`` unmasked and
    ``Rjm = (T(km1)-T(km2))*maskC(km2)`` (gad_fluxlimit_adv_r.F:87-89,
    gad_u3_adv_r.F:68-70; scheme 4 masks its own ``Rjm`` with ``maskC(km1)``).
    ``mask`` is the cell wet mask ``maskC`` or ``None`` (all wet).
    """
    t_above = tracer[..., :-1, :, :]                                              # T(km1)
    t_below = tracer[..., 1:, :, :]                                               # T(k)
    t_below2 = torch.cat((tracer[..., 2:, :, :], tracer[..., -1:, :, :]), dim=-3)  # T(kp1)
    t_above2 = torch.cat((tracer[..., :1, :, :], tracer[..., :-2, :, :]), dim=-3)  # T(km2)
    if mask is not None:
        m = mask.to(dtype=tracer.dtype, device=tracer.device)
        if m.ndim == 3:
            m = m.unsqueeze(0)
        m_above = m[..., :-1, :, :]
        m_below2 = torch.cat((m[..., 2:, :, :], m[..., -1:, :, :]), dim=-3)
        m_above2 = torch.cat((m[..., :1, :, :], m[..., :-2, :, :]), dim=-3)
    else:
        m_above = m_below2 = m_above2 = torch.ones_like(t_above)
    Rjp = (t_below2 - t_below) * m_below2
    Rj = t_below - t_above
    Rjm = (t_above - t_above2) * m_above2
    return t_above2, t_above, t_below, m_above2, m_above, m_below2, Rjp, Rj, Rjm


def vertical_fluxlimit_flux(
    tracer: torch.Tensor,
    transport: torch.Tensor,
    velocity: torch.Tensor,
    *,
    recip_drC: torch.Tensor,
    dt: float,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Flux-limited (scheme 77) vertical advective flux at layer interfaces.

    GAD_FLUXLIMIT_ADV_R (gad_fluxlimit_adv_r.F:71-112) in LOCOMgcm's
    DOWNWARD-positive convention (``w = -rTrans``, ``flux = -wT``)::

        flux = maskC(km1) * ( w*(T(k)+T(km1))*0.5 - |w|*((1-Cr)+wCFL*Cr)*Rj*0.5 )

    with ``w`` the interface ``transport`` (``-rTrans``, or the interface
    velocity in a velocity-space caller), ``wCFL = |velocity*dt*recip_drC(k)|``
    from the interface ``velocity`` (``-wFld``), ``Rj = T(k)-T(km1)`` unmasked,
    ``Rjp`` masked by ``maskC(kp1)``, ``Rjm`` by ``maskC(km2)``, ``km2``/``kp1``
    clamped to the column, and ``Cr`` upstream (``Rjm`` for downward ``w>0``,
    MIT ``rTrans<0``).  ``recip_drC`` and ``mask`` as
    :func:`vertical_advective_flux_dst3fl`.
    """
    flux = torch.zeros(
        (tracer.shape[0], tracer.shape[1] + 1, tracer.shape[2], tracer.shape[3]),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    if tracer.shape[1] < 2:
        return flux

    w = transport[..., 1:-1, :, :]
    w_cfl = velocity[..., 1:-1, :, :]
    _t_above2, t_above, t_below, _m_above2, m_above, _m_below2, Rjp, Rj, Rjm = (
        _vertical_four_point_stencil(tracer, mask)
    )

    recip = recip_drC.to(dtype=tracer.dtype, device=tracer.device)
    if recip.ndim == 1:
        recip = recip.view(1, -1, 1, 1)
    wCFL = (w_cfl * dt * recip).abs()
    cr = _superbee(_fluxlimit_ratio(torch.where(w > 0.0, Rjm, Rjp), Rj))
    # |w| with the reference ABS(0) adjoint slope; ``w`` is -rTrans (see
    # vertical_advective_flux_dst3).
    abs_w = abs_with_zero_slope(w, -1.0)
    flux[..., 1:-1, :, :] = m_above * (
        w * (t_below + t_above) * 0.5 - abs_w * ((1.0 - cr) + wCFL * cr) * Rj * 0.5
    )
    return flux


def vertical_upwind3_flux(
    tracer: torch.Tensor,
    vertical_velocity_interface: torch.Tensor,
    *,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Third-order upwind (scheme 3) vertical advective flux at layer interfaces.

    GAD_U3_ADV_R (gad_u3_adv_r.F:55-81) in LOCOMgcm's DOWNWARD-positive
    convention (``w = -rTrans``, ``flux = -wT``)::

        flux = maskC(km1) * ( w*((T(k)+T(km1))*0.5 - oneSixth*(Rjjm+Rjjp)*0.5)
                              - |w|*oneSixth*(Rjjm-Rjjp)*0.5 )

    with the masks and clamping of :func:`vertical_fluxlimit_flux`.  Linear in
    ``w``: velocity or volume transport.
    """
    flux = torch.zeros(
        (tracer.shape[0], tracer.shape[1] + 1, tracer.shape[2], tracer.shape[3]),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    if tracer.shape[1] < 2:
        return flux

    w = vertical_velocity_interface[..., 1:-1, :, :]
    _t_above2, t_above, t_below, _m_above2, m_above, _m_below2, Rjp, Rj, Rjm = (
        _vertical_four_point_stencil(tracer, mask)
    )
    Rjjp = Rjp - Rj
    Rjjm = Rj - Rjm
    # |w| with the reference ABS(0) adjoint slope; ``w`` is -rTrans.
    abs_w = abs_with_zero_slope(w, -1.0)
    flux[..., 1:-1, :, :] = m_above * (
        w * ((t_below + t_above) * 0.5 - _ONE_SIXTH * (Rjjm + Rjjp) * 0.5)
        - abs_w * _ONE_SIXTH * (Rjjm - Rjjp) * 0.5
    )
    return flux


def vertical_centered4_flux(
    tracer: torch.Tensor,
    vertical_velocity_interface: torch.Tensor,
    *,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Fourth-order centered (scheme 4) vertical advective flux at layer interfaces.

    GAD_C4_ADV_R (gad_c4_adv_r.F:58-89) in LOCOMgcm's DOWNWARD-positive
    convention (``w = -rTrans``, ``flux = -wT``)::

        flux = maskC(km1) * ( w*((T(k)+T(km1))*0.5 - oneSixth*(Rjjm+Rjjp)*0.5)
                              - |w|*oneSixth*(Rjjm-Rjjp)*0.5*(1 - maskBound) )

    with ``Rjm = (T(km1)-T(km2))*maskC(km1)`` (``km1``, where scheme 3 has
    ``km2``) and ``maskBound = maskPM*maskC(km2)*maskC(kp1)``.  ``maskPM`` is 0
    for MITgcm ``k <= 2`` or ``k >= Nr``, i.e. LOCOMgcm interface ``i <= 1`` or
    ``i >= nz-1`` (``k = i+1``), and 1 between, where ``km2`` and ``kp1`` need
    no clamping.  Linear in ``w``: velocity or volume transport.
    """
    flux = torch.zeros(
        (tracer.shape[0], tracer.shape[1] + 1, tracer.shape[2], tracer.shape[3]),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    nz = tracer.shape[1]
    if nz < 2:
        return flux

    w = vertical_velocity_interface[..., 1:-1, :, :]
    t_above2, t_above, t_below, m_above2, m_above, m_below2, Rjp, Rj, _Rjm = (
        _vertical_four_point_stencil(tracer, mask)
    )
    # maskPM on the interior interfaces 1 .. nz-1: 1 on 2 .. nz-2.
    mask_pm = torch.zeros(nz - 1, dtype=tracer.dtype, device=tracer.device)
    mask_pm[1:nz - 2] = 1.0
    mask_bound = mask_pm.view(1, nz - 1, 1, 1) * m_above2 * m_below2

    Rjm = (t_above - t_above2) * m_above
    Rjjp = Rjp - Rj
    Rjjm = Rj - Rjm
    # |w| with the reference ABS(0) adjoint slope; ``w`` is -rTrans.
    abs_w = abs_with_zero_slope(w, -1.0)
    flux[..., 1:-1, :, :] = m_above * (
        w * ((t_below + t_above) * 0.5 - _ONE_SIXTH * (Rjjm + Rjjp) * 0.5)
        - abs_w * _ONE_SIXTH * (Rjjm - Rjjp) * 0.5 * (1.0 - mask_bound)
    )
    return flux


def vertical_os7mp_flux(
    tracer: torch.Tensor,
    transport: torch.Tensor,
    velocity: torch.Tensor,
    *,
    recip_drC: torch.Tensor,
    dt: float,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """OS7MP (scheme 7) vertical advective flux at layer interfaces.

    GAD_OS7MP_ADV_R (gad_os7mp_adv_r.F:45-205) in LOCOMgcm's DOWNWARD-positive
    convention (``w = -wTrans``, ``flux = -wT``), ``w`` the interface
    ``transport``: ``km4 .. kp3`` clamped to the column, masks ``maskC`` times
    ``float`` of the clamped index difference (zero where clamping collapses two
    levels), the stencil oriented upwind (MIT ``wTrans < 0``, i.e. ``w > 0``,
    reads ``Q(kp2 .. km4)``) and passed to :func:`_os7mp_oriented_flux` with
    ``cfl = |velocity*dt*recip_drC(k)|`` from the interface ``velocity``
    (``-wFld``).
    ``recip_drC`` and ``mask`` as :func:`vertical_advective_flux_dst3fl`.
    """
    flux = torch.zeros(
        (tracer.shape[0], tracer.shape[1] + 1, tracer.shape[2], tracer.shape[3]),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    nz = tracer.shape[1]
    if nz < 2:
        return flux

    w = transport[..., 1:-1, :, :]
    k = torch.arange(1, nz, device=tracer.device)            # T(k) is cell k on interface k
    level = {o: torch.clamp(k + o, 0, nz - 1) for o in range(-4, 4)}
    if mask is not None:
        m = mask.to(dtype=tracer.dtype, device=tracer.device)
        if m.ndim == 3:
            m = m.unsqueeze(0)
    else:
        m = torch.ones_like(tracer)

    def Q(o):
        return tracer.index_select(-3, level[o])

    def Msk(o, o_next):
        # maskC(level o) * float(level o - level o_next)
        gap = (level[o] - level[o_next]).to(tracer.dtype).view(1, nz - 1, 1, 1)
        return m.index_select(-3, level[o]) * gap

    down = w > 0.0                                            # MIT wTrans < 0
    recip = recip_drC.to(dtype=tracer.dtype, device=tracer.device)
    if recip.ndim == 1:
        recip = recip.view(1, -1, 1, 1)
    cfl = (velocity[..., 1:-1, :, :] * dt * recip).abs()
    flux[..., 1:-1, :, :] = _os7mp_oriented_flux(
        torch.where(down, Q(2), Q(-3)),
        torch.where(down, Q(1), Q(-2)),
        torch.where(down, Q(0), Q(-1)),
        torch.where(down, Q(-1), Q(0)),
        torch.where(down, Q(-2), Q(1)),
        torch.where(down, Q(-3), Q(2)),
        torch.where(down, Q(-4), Q(3)),
        torch.where(down, Msk(2, 1), Msk(-2, -3)),
        torch.where(down, Msk(1, 0), Msk(-1, -2)),
        Msk(0, -1),
        torch.where(down, Msk(-1, -2), Msk(1, 0)),
        torch.where(down, Msk(-2, -3), Msk(2, 1)),
        torch.where(down, Msk(-3, -4), Msk(3, 2)),
        w,
        cfl,
    )
    return flux


def vertical_dst2_flux(
    tracer: torch.Tensor,
    transport: torch.Tensor,
    velocity: torch.Tensor,
    *,
    recip_drC: torch.Tensor,
    dt: float,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Second-order Direct Space-Time (scheme 20) vertical advective flux at layer interfaces.

    GAD_DST2U1_ADV_R with ``rLimit = 1`` and ``rkSign = -1``
    (gad_dst2u1_adv_r.F:57-93) in LOCOMgcm's DOWNWARD-positive convention
    (``w = -rTrans``, ``flux = -wT``)::

        flux = maskC(km1) * ( (w + |w|c)*0.5*T(km1) + (w - |w|c)*0.5*T(k) ),
        c = 1 - (1 - wCFL),  wCFL = |velocity * dt * recip_drC(k)|

    with ``w`` the interface ``transport`` (``-rTrans``) and ``velocity`` the
    interface velocity (``-wFld``).

    ``recip_drC`` and ``mask`` as :func:`vertical_advective_flux_dst3fl`.
    """
    flux = torch.zeros(
        (tracer.shape[0], tracer.shape[1] + 1, tracer.shape[2], tracer.shape[3]),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    if tracer.shape[1] < 2:
        return flux

    w = transport[..., 1:-1, :, :]
    t_above = tracer[..., :-1, :, :]                                              # T(km1)
    t_below = tracer[..., 1:, :, :]                                               # T(k)
    if mask is not None:
        m = mask.to(dtype=tracer.dtype, device=tracer.device)
        if m.ndim == 3:
            m = m.unsqueeze(0)
        m_above = m[..., :-1, :, :]
    else:
        m_above = torch.ones_like(t_above)
    recip = recip_drC.to(dtype=tracer.dtype, device=tracer.device)
    if recip.ndim == 1:
        recip = recip.view(1, -1, 1, 1)
    wCFL = (velocity[..., 1:-1, :, :] * dt * recip).abs()
    # |w| with the reference ABS(0) adjoint slope; ``w`` is -rTrans.
    abs_w = abs_with_zero_slope(w, -1.0) * (1.0 - 1.0 * (1.0 - wCFL))
    flux[..., 1:-1, :, :] = m_above * (
        (w + abs_w) * 0.5 * t_above + (w - abs_w) * 0.5 * t_below
    )
    return flux
