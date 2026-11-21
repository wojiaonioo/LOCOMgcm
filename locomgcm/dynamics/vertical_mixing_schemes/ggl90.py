"""Gaspar, Gregoris and Lefevre (1990) TKE closure (MITgcm ``pkg/ggl90``).

One time step of GGL90_CALC (``pkg/ggl90/ggl90_calc.F``) with its mixing
length (``ggl90_mixinglength.F``), the initial TKE of ``ggl90_init_varia.F``
and the Langmuir Stokes drift of ``ggl90_add_stokesdrift.F``, z coordinates
only (``coordFac = 1``, ``ggl90_calc.F:209-211``).  The kernels follow
MITgcm's statement order; the caller chains them, selects the branches
named in the option tuples and solves the tridiagonal TKE system between
``ggl90_dirichlet_bottom`` and ``ggl90_tke_floor``.  The merges into KappaRT,
KappaRS, KappaRU and KappaRV are in ``coefficients``.

Interface fields are [B, nz, ny, nx]: index k is the top interface of layer
k (MITgcm level K = k + 1), index 0 the sea surface.  ``tke`` (m^2/s^2) is
the prognostic GGL90TKE on interfaces, carried from step to step.  ``u``
[B, nz, ny, nx + 1], ``v`` [B, nz, ny + 1, nx] and the momentum surface
forcing ``surf_forc_u`` [B, ny, nx + 1], ``surf_forc_v`` [B, ny + 1, nx]
(MITgcm surfaceForcingU/V, m^2/s^2) carry both outer faces.  ``mask_c``
[nz, ny, nx], ``mask_w`` [nz, ny, nx + 1], ``mask_s`` [nz, ny + 1, nx] are
0/1 masks.  Depths are LOCOMgcm's positive metres: ``interface_depth`` and
``center_depth`` [nz] of the top interface and centre of each layer (MITgcm
``rF = -interface_depth``, ``rC = -center_depth``), ``column_depth``
[ny, nx] the water depth (``Ro_surf - R_low`` with ``Ro_surf = 0``).
``drf``, ``recip_drf`` and ``recip_drc`` are positive columns [nz];
``recip_drc[k]`` belongs to the distance between centres k - 1 and k.
``k_low`` [ny, nx] int64 is the number of wet layers (MITgcm kLowC, 0 on
land).  Background columns ``visc_arnr``, ``diffkr_nr_s`` are [nz] in m^2/s.
``sigma_r`` comes from ``density_gradient.vertical_density_gradient``.
dtype follows the inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from locomgcm.dynamics.viscosity_coefficients import _safe_sqrt_zero_grad

from .density_gradient import GRAVITY_SIGN

#: CPP options of ``GGL90_OPTIONS.h`` this port accepts.  ALLOW_GGL90_HORIZDIFF,
#: ALLOW_GGL90_SMOOTH, ALLOW_GGL90_IDEMIX and GGL90_REGULARIZE_MIXINGLENGTH are
#: rejected.
GGL90_CPP_OPTIONS = ("ALLOW_GGL90_LANGMUIR", "GGL90_MISSING_HFAC_BUG")

#: ``mxlMaxFlag`` values (``ggl90_mixinglength.F:166-307``); the caller picks
#: ``ggl90_limit_by_column_depth`` (0), ``ggl90_limit_by_boundary_distance``
#: (1) or ``ggl90_limit_by_sweeps`` (2, 3), then ``ggl90_floor_mixing_length``
#: (0, 1, 2) or ``ggl90_geometric_mixing_length`` (3).
GGL90_MIXING_LENGTH_FLAGS = (0, 1, 2, 3)

#: ``mxlMaxFlag`` values allowed with ``useLANGMUIR`` (``:317-359``).
GGL90_LANGMUIR_MIXING_LENGTH_FLAGS = (1, 2, 3)

#: ``GGL90.h:66-69``.
SQRTTWO = 1.41421356237310
GGL90_EPS = 2.23e-16


@dataclass(frozen=True)
class Ggl90Parameters:
    """Values the kernels read: GGL90_PARM01 (``ggl90_readparms.F:48-56``),
    GGL90_PARM03 (``:75-77``) and whether the build defines
    GGL90_MISSING_HFAC_BUG (``ggl90_calc.F:677-685``).

    ``tke_bottom`` is GGL90TKEbottom after ``:251-253`` (GGL90TKEmin when
    unset); ``mxl_max_flag`` is one of GGL90_MIXING_LENGTH_FLAGS.
    """

    ck: float
    ceps: float
    alpha: float
    m2: float
    tke_min: float
    tke_surf_min: float
    tke_bottom: float
    visc_max: float
    diff_max: float
    mixing_length_min: float
    mxl_max_flag: int
    mxl_surf_flag: bool
    dirichlet: bool
    calc_mean_vert_shear: bool
    use_langmuir: bool
    lc_gamma: float
    lc_num: float
    lc_lambda: float
    missing_hfac_bug: bool


def _with_surface(surface: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
    return torch.cat((surface, inner), dim=1)


def _level_onehot(k_low: torch.Tensor, nz: int, device) -> torch.Tensor:
    # [nz, ny, nx] True on level max(kLowC, 1) - 1 (0-based), MITgcm's kBot
    # in z coordinates (ggl90_calc.F:736, 932).
    levels = torch.arange(nz, device=device).reshape(-1, 1, 1)
    return levels == (torch.clamp(k_low, min=1) - 1)


def ggl90_initial_tke(mask_c: torch.Tensor, *, tke_min: float) -> torch.Tensor:
    """GGL90TKE at the cold start, ``ggl90_init_varia.F:52``: ``GGL90TKEmin*maskC``."""
    return tke_min * mask_c


def ggl90_recip_hfac_interface(hfac_c: torch.Tensor) -> torch.Tensor:
    """``recip_hFacI`` of ``ggl90_calc.F:243-257`` on [nz, ny, nx].

    ``hFac = min(0.5, hFacC(max(k-1,1))) + min(0.5, hFacC(k))``, its
    reciprocal where non-zero and zero elsewhere.  A build with
    GGL90_MISSING_HFAC_BUG replaces levels 2..Nr by one (``:677-685``); the
    caller then passes ones instead of calling this.
    """
    upper = torch.cat((hfac_c[:1], hfac_c[:-1]), dim=0)
    hfac = torch.clamp(upper, max=0.5) + torch.clamp(hfac_c, max=0.5)
    wet = hfac != 0.0
    return torch.where(wet, 1.0 / torch.where(wet, hfac, torch.ones_like(hfac)),
                       torch.zeros_like(hfac))


def ggl90_initial_state(
    tke: torch.Tensor,
    sigma_r: torch.Tensor,
    *,
    mask_c: torch.Tensor,
    gravity: float,
    recip_rho_const: float,
    mixing_length_min: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(SQRTTKE, Nsquare, GGL90mixingLength)`` of ``ggl90_calc.F:270-363``.

    ``SQRTTKE = sqrt(TKE)`` on every level (``:302``, ``:345``);
    ``Nsquare = gravity*gravitySign*recip_rhoConst*sigmaR`` (``:353-354``);
    the first-guess mixing length ``SQRTTWO*SQRTTKE/sqrt(max(Nsquare, eps))``
    times ``maskC(k)*maskC(k-1)`` (``:358-360``).  The surface level keeps
    the initial ``Nsquare = 0`` and ``GGL90mixingLengthMin`` (``:279-284``).
    """
    # double-where: dry or zero TKE takes no gradient through sqrt (ggl90_calc.F:298-306, :341-349)
    sqrt_tke = _safe_sqrt_zero_grad(tke)
    inner_sq = sqrt_tke[:, 1:]
    msk = mask_c[1:] * mask_c[:-1]
    nsq = gravity * GRAVITY_SIGN * recip_rho_const * sigma_r[:, 1:]
    ml = SQRTTWO * inner_sq / torch.sqrt(torch.clamp(nsq, min=GGL90_EPS)) * msk
    nsquare = _with_surface(torch.zeros_like(nsq[:, :1]), nsq)
    mixing_length = _with_surface(torch.full_like(ml[:, :1], mixing_length_min), ml)
    return sqrt_tke, nsquare, mixing_length


def ggl90_surface_mixing_length(mixing_length: torch.Tensor, *, drf: torch.Tensor) -> torch.Tensor:
    """``mxlSurfFlag``: ``GGL90mixingLength(kTop=2) = drF(kSrf=1)``, ``ggl90_mixinglength.F:147-160``."""
    return torch.cat((mixing_length[:, :1], drf[0].expand_as(mixing_length[:, 1:2]),
                      mixing_length[:, 2:]), dim=1)


def ggl90_limit_by_column_depth(mixing_length: torch.Tensor, *, column_depth: torch.Tensor) -> torch.Tensor:
    """``mxlMaxFlag = 0``: ``min(l, Ro_surf - R_low)`` on levels 2..Nr, ``ggl90_mixinglength.F:166-179``."""
    return _with_surface(mixing_length[:, :1], torch.minimum(mixing_length[:, 1:], column_depth))


def ggl90_limit_by_boundary_distance(
    mixing_length: torch.Tensor,
    *,
    interface_depth: torch.Tensor,
    column_depth: torch.Tensor,
) -> torch.Tensor:
    """``mxlMaxFlag = 1``: ``min(l, min(Ro_surf - rF(k), rF(k) - R_low))``, ``ggl90_mixinglength.F:181-193``.

    With ``rF = -interface_depth``, ``Ro_surf = 0`` and ``R_low =
    -column_depth`` the two distances are ``d`` and ``H - d``, the same
    values MITgcm forms.
    """
    d = interface_depth[1:].reshape(-1, 1, 1)
    max_length = torch.minimum(d, column_depth - d)
    return _with_surface(mixing_length[:, :1], torch.minimum(mixing_length[:, 1:], max_length))


def ggl90_limit_by_sweeps(
    mixing_length: torch.Tensor,
    *,
    drf: torch.Tensor,
    mixing_length_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``mxlMaxFlag = 2, 3`` in z coordinates, ``ggl90_mixinglength.F:240-299``.

    Downward sweep ``mxLength_Dn(k) = min(l(k), mxLength_Dn(k-1) + drF(k-1))``
    from ``mxLength_Dn(1) = GGL90mixingLengthMin`` (``:131``, ``:245-258``);
    upward sweep ``l(Nr) = min(l(Nr), lmin + drF(Nr))``, then
    ``l(k) = min(l(k), l(k+1) + drF(k))`` for k = Nr-1..2 (``:262-283``);
    finally ``l = min(l, mxLength_Dn)`` (``:292-299``).  Returns
    ``(GGL90mixingLength, mxLength_Dn)``; level 0 of ``mxLength_Dn`` holds
    the minimum.
    """
    nz = mixing_length.shape[1]
    dn = [torch.full_like(mixing_length[:, 0], mixing_length_min)]
    for k in range(1, nz):
        dn.append(torch.minimum(mixing_length[:, k], dn[k - 1] + drf[k - 1]))
    up = [None] * nz
    up[0] = mixing_length[:, 0]
    up[nz - 1] = torch.minimum(mixing_length[:, nz - 1], mixing_length_min + drf[nz - 1])
    for k in range(nz - 2, 0, -1):
        up[k] = torch.minimum(mixing_length[:, k], up[k + 1] + drf[k])
    mx_length_dn = torch.stack(dn, dim=1)
    limited = torch.stack(up, dim=1)
    limited = _with_surface(limited[:, :1], torch.minimum(limited[:, 1:], mx_length_dn[:, 1:]))
    return limited, mx_length_dn


def ggl90_langmuir_mixing_length(
    mixing_length: torch.Tensor,
    reference: torch.Tensor,
    *,
    lc_gamma: float,
    mixing_length_min: float,
) -> torch.Tensor:
    """``LCmixingLength``, ``ggl90_mixinglength.F:136-144, 317-351``.

    ``LC_Gamma*l`` where the limited length equals ``reference`` and ``l``
    elsewhere, on levels 2..Nr; the surface level keeps the initial
    ``GGL90mixingLengthMin``.  ``reference`` is ``Ro_surf - rF(k)`` (the
    interface depth, broadcast) for mxlMaxFlag 1 and ``mxLength_Dn`` for 2
    and 3.
    """
    inner = mixing_length[:, 1:]
    lc = torch.where(inner == reference[:, 1:], lc_gamma * inner, inner)
    return _with_surface(torch.full_like(lc[:, :1], mixing_length_min), lc)


def ggl90_floor_langmuir_mixing_length(lc_mixing_length: torch.Tensor, *, mixing_length_min: float) -> torch.Tensor:
    """``max(LCmixingLength, GGL90mixingLengthMin)`` for mxlMaxFlag 1, 2, ``ggl90_mixinglength.F:361-374``."""
    return _with_surface(lc_mixing_length[:, :1],
                         torch.clamp(lc_mixing_length[:, 1:], min=mixing_length_min))


def ggl90_floor_mixing_length(
    mixing_length: torch.Tensor,
    *,
    mixing_length_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(GGL90mixingLength, rMixingLength)`` for mxlMaxFlag 0, 1, 2, ``ggl90_mixinglength.F:401-416``.

    ``l = max(l, lmin)``, ``rMixingLength = 1/l`` on levels 2..Nr;
    ``rMixingLength`` is zero at the surface (``:123``).
    """
    floored = torch.clamp(mixing_length[:, 1:], min=mixing_length_min)
    r = 1.0 / floored
    return (_with_surface(mixing_length[:, :1], floored),
            _with_surface(torch.zeros_like(r[:, :1]), r))


def ggl90_geometric_mixing_length(
    mixing_length: torch.Tensor,
    mx_length_dn: torch.Tensor,
    *,
    mixing_length_min: float,
) -> torch.Tensor:
    """``rMixingLength`` for mxlMaxFlag 3, ``ggl90_mixinglength.F:383-400``.

    ``1/max(sqrt(l*mxLength_Dn), lmin)`` on levels 2..Nr, zero at the
    surface; ``GGL90mixingLength`` itself is not floored in this branch.
    """
    ml_tmp = torch.clamp(_safe_sqrt_zero_grad(mixing_length[:, 1:] * mx_length_dn[:, 1:]),
                         min=mixing_length_min)
    r = 1.0 / ml_tmp
    return _with_surface(torch.zeros_like(r[:, :1]), r)


def ggl90_mean_vertical_shear(u: torch.Tensor, v: torch.Tensor, *, recip_drc: torch.Tensor) -> torch.Tensor:
    """``calcMeanVertShear``: the mean of the four face shears, ``ggl90_calc.F:474-488``.

    ``((tU*tU + tUp*tUp) + (tV*tV + tVp*tVp))*0.5*recip_drC(k)*recip_drC(k)``
    with ``tU = uVel(i,k-1) - uVel(i,k)``, ``tUp`` at ``i+1``; shape
    [B, nz - 1, ny, nx].  Without the flag the shear of the centre velocity
    (``:491-503``) is ``richardson.vertical_shear_squared``.
    """
    r = recip_drc[1:].reshape(-1, 1, 1)
    du = u[:, :-1] - u[:, 1:]
    dv = v[:, :-1] - v[:, 1:]
    t_u, t_up = du[..., :-1], du[..., 1:]
    t_v, t_vp = dv[..., :-1, :], dv[..., 1:, :]
    return ((t_u * t_u + t_up * t_up) + (t_v * t_v + t_vp * t_vp)) * 0.5 * r * r


def _stokes_shear_factors(surf_forc_u, surf_forc_v, interface_depth, lc_num, lc_lambda):
    # ggl90_calc.F:320-331 and :513-519: dStokes/dz on U and V points,
    # [B, nz - 1, ny, nx + 1] and [B, nz - 1, ny + 1, nx].
    recip_lasq = 1.0 / lc_num
    recip_lasq = recip_lasq * recip_lasq
    recip_ld = 4.0 * math.pi / lc_lambda
    u_star = torch.copysign(_safe_sqrt_zero_grad(torch.abs(surf_forc_u)), surf_forc_u)
    v_star = torch.copysign(_safe_sqrt_zero_grad(torch.abs(surf_forc_v)), surf_forc_v)
    depth_fac = (recip_lasq * torch.exp(recip_ld * -interface_depth[1:])).reshape(-1, 1, 1)
    return (recip_ld * u_star.unsqueeze(1) * depth_fac,
            recip_ld * v_star.unsqueeze(1) * depth_fac)


def ggl90_langmuir_stokes_shear(
    u: torch.Tensor,
    v: torch.Tensor,
    surf_forc_u: torch.Tensor,
    surf_forc_v: torch.Tensor,
    *,
    interface_depth: torch.Tensor,
    recip_drc: torch.Tensor,
    lc_num: float,
    lc_lambda: float,
) -> torch.Tensor:
    """``stokesterm`` from the centre-velocity shear, ``ggl90_calc.F:540-555``.

    ``0.5*(tempU*(dsU(i) + dsU(i+1)) + tempV*(dsV(j) + dsV(j+1)))`` with
    ``tempU = ((u(i,k-1)+u(i+1,k-1)) - (u(i,k)+u(i+1,k)))*0.5*recip_drC(k)``;
    shape [B, nz - 1, ny, nx].
    """
    ds_u, ds_v = _stokes_shear_factors(surf_forc_u, surf_forc_v, interface_depth, lc_num, lc_lambda)
    r = recip_drc[1:].reshape(-1, 1, 1)
    u_sum = u[..., :-1] + u[..., 1:]
    v_sum = v[..., :-1, :] + v[..., 1:, :]
    temp_u = (u_sum[:, :-1] - u_sum[:, 1:]) * 0.5 * r
    temp_v = (v_sum[:, :-1] - v_sum[:, 1:]) * 0.5 * r
    return 0.5 * (temp_u * (ds_u[..., :-1] + ds_u[..., 1:])
                  + temp_v * (ds_v[..., :-1, :] + ds_v[..., 1:, :]))


def ggl90_langmuir_mean_stokes_shear(
    u: torch.Tensor,
    v: torch.Tensor,
    surf_forc_u: torch.Tensor,
    surf_forc_v: torch.Tensor,
    *,
    interface_depth: torch.Tensor,
    recip_drc: torch.Tensor,
    lc_num: float,
    lc_lambda: float,
) -> torch.Tensor:
    """``stokesterm`` with ``calcMeanVertShear``, ``ggl90_calc.F:521-537``.

    ``((tU*dsU(i) + tUp*dsU(i+1)) + (tV*dsV(j) + tVp*dsV(j+1)))*0.5*recip_drC(k)``
    with the face differences of ``ggl90_mean_vertical_shear``.
    """
    ds_u, ds_v = _stokes_shear_factors(surf_forc_u, surf_forc_v, interface_depth, lc_num, lc_lambda)
    r = recip_drc[1:].reshape(-1, 1, 1)
    du = u[:, :-1] - u[:, 1:]
    dv = v[:, :-1] - v[:, 1:]
    return ((du[..., :-1] * ds_u[..., :-1] + du[..., 1:] * ds_u[..., 1:])
            + (dv[..., :-1, :] * ds_v[..., :-1, :] + dv[..., 1:, :] * ds_v[..., 1:, :])) * 0.5 * r


def ggl90_explicit_tke(
    tke: torch.Tensor,
    sqrt_tke: torch.Tensor,
    nsquare: torch.Tensor,
    mixing_length: torch.Tensor,
    vertical_shear: torch.Tensor,
    *,
    mask_c: torch.Tensor,
    ck: float,
    alpha: float,
    dt: float,
    diffkr_nr_s: torch.Tensor,
    visc_arnr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The explicit part of the TKE step, ``ggl90_calc.F:441-614``, levels 2..Nr.

    ``KappaM = GGL90ck*l*SQRTTKE`` (``:447``, ``:454``; ``l`` is the Langmuir
    length when it is on); ``GGL90visctmp = max(KappaM, diffKrNrS(k))*maskC(k)
    *maskC(k-1)`` and ``KappaM = max(KappaM, viscArNr(k))*maskC(k)*maskC(k-1)``
    (``:462-469``); the Prandtl number ``min(10, 5*Ri if Ri >= 0.2 else 1)``
    with ``Ri = max(Nsquare, 0)/(shear + eps)`` (``:579-587``);
    ``KappaE = GGL90alpha*KappaM*maskC(k)*maskC(k-1)`` and
    ``TKE + dt*(KappaM*shear - KappaH*Nsquare)`` (``:595-614``).  The
    dissipation term there is ``explDissFac = 0`` times a finite product
    (``:218``) and is left out.  ``vertical_shear`` is [B, nz - 1, ny, nx].
    Returns ``(TKE, KappaM, KappaE, GGL90visctmp, TKEPrandtlNumber)`` on all
    levels; the surface level keeps the TKE and the initial values 0, 0, 0, 1
    (``:271-277``, ``:294``).
    """
    mk = mask_c[1:]
    mkm1 = mask_c[:-1]
    kappa_m0 = ck * mixing_length[:, 1:] * sqrt_tke[:, 1:]
    visc_tmp = torch.maximum(kappa_m0, diffkr_nr_s[1:].reshape(-1, 1, 1)) * mk * mkm1
    kappa_m = torch.maximum(kappa_m0, visc_arnr[1:].reshape(-1, 1, 1)) * mk * mkm1
    nsq = nsquare[:, 1:]
    ri = torch.clamp(nsq, min=0.0) / (vertical_shear + GGL90_EPS)
    prandtl = torch.clamp(torch.where(ri >= 0.2, 5.0 * ri, torch.ones_like(ri)), max=10.0)
    kappa_h = kappa_m / prandtl
    kappa_e = alpha * kappa_m * mk * mkm1
    tke_inner = tke[:, 1:] + dt * (kappa_m * vertical_shear - kappa_h * nsq)
    zero = torch.zeros_like(kappa_m[:, :1])
    return (_with_surface(tke[:, :1], tke_inner), _with_surface(zero, kappa_m),
            _with_surface(zero, kappa_e), _with_surface(zero, visc_tmp),
            _with_surface(torch.ones_like(zero), prandtl))


def ggl90_langmuir_tke_source(
    tke: torch.Tensor,
    kappa_m: torch.Tensor,
    stokes_shear: torch.Tensor,
    *,
    dt: float,
) -> torch.Tensor:
    """``TKE + dt*(KappaM*stokesterm)`` on levels 2..Nr, ``ggl90_calc.F:628-638``."""
    return _with_surface(tke[:, :1], tke[:, 1:] + dt * (kappa_m[:, 1:] * stokes_shear))


def ggl90_tke_off_diagonals(
    kappa_e: torch.Tensor,
    *,
    mask_c: torch.Tensor,
    recip_hfac_c: torch.Tensor,
    recip_hfac_i: torch.Tensor,
    recip_drf: torch.Tensor,
    recip_drc: torch.Tensor,
    k_low: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower and upper diagonals of the TKE system, ``ggl90_calc.F:670-722``.

    ``a(k) = -dt*recip_drF(k-1)*recip_hFacC(k-1)*0.5*(KappaE(k) + KappaE(km1))
    *recip_drC(k)*maskC(k)*recip_hFacI(k)`` with ``km1 = max(2, k-1)``;
    ``c(k) = -dt*recip_drF(k)*recip_hFacC(k)*0.5*(KappaE(k) + KappaE(kp1))
    *recip_drC(k)*maskC(k-1)*recip_hFacI(k)`` with
    ``kp1 = max(1, min(kLowC, k+1))``; both zero at the surface level.
    ``recip_hfac_c``, ``recip_hfac_i`` are [nz, ny, nx] at the caller's time
    level.
    """
    nz = kappa_e.shape[1]
    ke = kappa_e[:, 1:]
    ke_lower = torch.cat((kappa_e[:, 1:2], kappa_e[:, 1:nz - 1]), dim=1)
    rdrc = recip_drc[1:].reshape(-1, 1, 1)
    a = (-dt * recip_drf[:-1].reshape(-1, 1, 1) * recip_hfac_c[:-1] * 0.5 * (ke + ke_lower)
         * rdrc * mask_c[1:] * recip_hfac_i[1:])
    # k + 1 for MITgcm levels k = 2..Nr
    levels = torch.arange(3, nz + 2, device=k_low.device).reshape(-1, 1, 1)
    kp1 = torch.clamp(torch.minimum(k_low, levels), min=1) - 1
    ke_upper = torch.gather(kappa_e, 1, kp1.unsqueeze(0).expand(kappa_e.shape[0], -1, -1, -1))
    c = (-dt * recip_drf[1:].reshape(-1, 1, 1) * recip_hfac_c[1:] * 0.5 * (ke + ke_upper)
         * rdrc * mask_c[:-1] * recip_hfac_i[1:])
    zero = torch.zeros_like(a[:, :1])
    return _with_surface(zero, a), _with_surface(zero, c)


def ggl90_neumann_bottom(c: torch.Tensor, *, k_low: torch.Tensor) -> torch.Tensor:
    """``GGL90_dirichlet = .FALSE.``: ``c(max(kLowC,1)) = 0`` before the diagonal, ``ggl90_calc.F:724-741``."""
    return torch.where(_level_onehot(k_low, c.shape[1], c.device), torch.zeros_like(c), c)


def ggl90_tke_diagonal(
    a: torch.Tensor,
    c: torch.Tensor,
    sqrt_tke: torch.Tensor,
    r_mixing_length: torch.Tensor,
    *,
    mask_c: torch.Tensor,
    dt: float,
    ceps: float,
) -> torch.Tensor:
    """Main diagonal, ``ggl90_calc.F:744-754``, all levels.

    ``1 - c - a + implDissFac*dt*GGL90ceps*SQRTTKE*rMixingLength*maskC(k)
    *maskC(max(k-1,1))`` with ``implDissFac = 1`` (``:219``).
    """
    mask_km1 = torch.cat((mask_c[:1], mask_c[:-1]), dim=0)
    return 1.0 - c - a + dt * ceps * sqrt_tke * r_mixing_length * mask_c * mask_km1


def ggl90_surface_friction_squared(surf_forc_u: torch.Tensor, surf_forc_v: torch.Tensor) -> torch.Tensor:
    """``uStarSquare`` before its square root, ``ggl90_calc.F:787-796``.

    ``(0.5*(sfU(i) + sfU(i+1)))**2 + (0.5*(sfV(j) + sfV(j+1)))**2``, [B, ny, nx].
    """
    su = 0.5 * (surf_forc_u[..., :-1] + surf_forc_u[..., 1:])
    sv = 0.5 * (surf_forc_v[..., :-1, :] + surf_forc_v[..., 1:, :])
    return su * su + sv * sv


def ggl90_mean_surface_friction_squared(surf_forc_u: torch.Tensor, surf_forc_v: torch.Tensor) -> torch.Tensor:
    """``uStarSquare`` with ``calcMeanVertShear``, ``ggl90_calc.F:766-785``.

    ``(tU*tU + tUp*tUp + tV*tV + tVp*tVp)*0.5`` summed left to right, as
    MITgcm keeps it (``:778-783``).
    """
    t_u, t_up = surf_forc_u[..., :-1], surf_forc_u[..., 1:]
    t_v, t_vp = surf_forc_v[..., :-1, :], surf_forc_v[..., 1:, :]
    return (t_u * t_u + t_up * t_up + t_v * t_v + t_vp * t_vp) * 0.5


def ggl90_surface_tke(
    tke: torch.Tensor,
    a: torch.Tensor,
    friction_squared: torch.Tensor,
    *,
    mask_c: torch.Tensor,
    m2: float,
    tke_surf_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dirichlet surface condition, ``ggl90_calc.F:866-875, 900-916``.

    ``uStarSquare = sqrt(friction_squared)``; ``TKE(1) = maskC(1)*max(
    GGL90TKEsurfMin, GGL90m2*uStarSquare)``; ``TKE(2) = TKE(2) - a(2)*TKE(1)``
    and ``a(2) = 0``.  Returns ``(TKE, a)``.
    """
    # double-where: zero stress takes no gradient through sqrt (ggl90_calc.F:866-871)
    u_star_square = _safe_sqrt_zero_grad(friction_squared)
    surface = mask_c[0] * torch.clamp(m2 * u_star_square, min=tke_surf_min)
    second = tke[:, 1] - a[:, 1] * surface
    tke = torch.cat((surface.unsqueeze(1), second.unsqueeze(1), tke[:, 2:]), dim=1)
    a = torch.cat((a[:, :1], torch.zeros_like(a[:, 1:2]), a[:, 2:]), dim=1)
    return tke, a


def ggl90_dirichlet_bottom(
    tke: torch.Tensor,
    c: torch.Tensor,
    *,
    k_low: torch.Tensor,
    tke_bottom: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``GGL90_dirichlet``: ``TKE(kBot) -= GGL90TKEbottom*c(kBot)``, ``c(kBot) = 0``, ``ggl90_calc.F:918-939``.

    ``kBot = max(kLowC, 1)``.  Returns ``(TKE, c)``.
    """
    at_bottom = _level_onehot(k_low, c.shape[1], c.device)
    return (torch.where(at_bottom, tke - tke_bottom * c, tke),
            torch.where(at_bottom, torch.zeros_like(c), c))


def ggl90_tke_floor(tke: torch.Tensor, *, mask_c: torch.Tensor, tke_min: float) -> torch.Tensor:
    """After the solve: ``maskC(k)*maskC(k-1)*max(TKE, GGL90TKEmin)`` on levels 2..Nr, ``ggl90_calc.F:955-989``."""
    return _with_surface(tke[:, :1], mask_c[1:] * mask_c[:-1] * torch.clamp(tke[:, 1:], min=tke_min))


def ggl90_coefficients(
    visc_tmp: torch.Tensor,
    prandtl: torch.Tensor,
    *,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
    visc_max: float,
    diff_max: float,
    visc_arnr: torch.Tensor,
    diffkr_nr_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(GGL90viscArU, GGL90viscArV, GGL90diffKr)``, ``ggl90_calc.F:994-1074`` without smoothing.

    ``GGL90diffKr = max(min(visctmp/Prandtl, GGL90diffMax), diffKrNrS(k))``;
    ``GGL90viscArU = max(min(maskW(k-1)*maskW(k)*0.5*(visctmp(i-1) +
    visctmp(i)), GGL90viscMax), viscArNr(k))`` and the same on V faces, on
    levels 2..Nr; level 1 keeps the zero of ``ggl90_init_varia.F:45-47``.
    ``visc_tmp`` [B, nz, ny + 2, nx + 2] carries one halo cell on each side;
    ``prandtl`` is [B, nz, ny, nx].
    """
    vt = visc_tmp[:, 1:]
    centre = vt[..., 1:-1, 1:-1]
    diff = torch.maximum(torch.clamp(centre / prandtl[:, 1:], max=diff_max),
                         diffkr_nr_s[1:].reshape(-1, 1, 1))
    rows = vt[..., 1:-1, :]
    cols = vt[..., :, 1:-1]
    visc_u = mask_w[:-1] * mask_w[1:] * 0.5 * (rows[..., :-1] + rows[..., 1:])
    visc_v = mask_s[:-1] * mask_s[1:] * 0.5 * (cols[..., :-1, :] + cols[..., 1:, :])
    background = visc_arnr[1:].reshape(-1, 1, 1)
    visc_u = torch.maximum(torch.clamp(visc_u, max=visc_max), background)
    visc_v = torch.maximum(torch.clamp(visc_v, max=visc_max), background)
    return (_with_surface(torch.zeros_like(visc_u[:, :1]), visc_u),
            _with_surface(torch.zeros_like(visc_v[:, :1]), visc_v),
            _with_surface(torch.zeros_like(diff[:, :1]), diff))


def ggl90_residual_velocity(
    u: torch.Tensor,
    v: torch.Tensor,
    surf_forc_u: torch.Tensor,
    surf_forc_v: torch.Tensor,
    *,
    center_depth: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
    lc_num: float,
    lc_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Velocity plus Langmuir Stokes drift for the Coriolis term, ``ggl90_add_stokesdrift.F:46-69``.

    ``depthFac = (1/LC_num)**2*exp(4*PI/LC_lambda*rC(k))`` with
    ``rC = -center_depth``; ``uRes = uVel + SIGN(sqrt|sfU|, sfU)*depthFac*maskW``
    and the same for V.  MITgcm feeds ``uRes``, ``vRes`` to the Coriolis term
    of ``mom_vecinv.F:689-694`` and ``mom_fluxform.F:1083-1090``.
    """
    recip_lasq = 1.0 / lc_num
    recip_lasq = recip_lasq * recip_lasq
    depth_fac = (recip_lasq * torch.exp(4.0 * math.pi / lc_lambda * -center_depth)).reshape(-1, 1, 1)
    u_star = torch.copysign(_safe_sqrt_zero_grad(torch.abs(surf_forc_u)), surf_forc_u).unsqueeze(1)
    v_star = torch.copysign(_safe_sqrt_zero_grad(torch.abs(surf_forc_v)), surf_forc_v).unsqueeze(1)
    return u + u_star * depth_fac * mask_w, v + v_star * depth_fac * mask_s
