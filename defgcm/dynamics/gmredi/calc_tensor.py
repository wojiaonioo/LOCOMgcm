"""GM/Redi scheme selection and tensor assembly (MITgcm ``GMREDI_CALC_TENSOR``).

``GM_TAPER_SCHEMES`` is the capability declaration for ``GM_taper_scheme``:
the translator and the case generator read it, :class:`GMRediParameters`
rejects any other value, and each name has its own branch below.  ``""`` is
MITgcm's blank default (no taper; Fortran compares ``' '`` and ``''`` equal).
The covered range is constant ``GM_isopycK``/``GM_background_K`` without
1-D/2-D/3-D factor files, skew-flux or non-separate advective form, z
coordinates (SPEC_gmredi.md §1).  Inputs follow the package docstring; the
outputs are one :class:`GMRediTensor` per step, read by the x, y and r fluxes
of every tracer, by the vertical diffusivity and by the residual flow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .bolus import (
    bolus_streamfunction,
    psi_dm95_taper,
    psi_gkw91_taper,
    psi_ldd97_taper,
    psi_limited_slope,
    psi_linear_taper,
    psi_slope_inputs_x,
    psi_slope_inputs_y,
)
from .slope_limit import (
    ac02_slopes,
    dm95_taper,
    gkw91_taper,
    ldd97_rossby_radii,
    ldd97_taper,
    limited_slopes,
    linear_taper,
    sigma_gradients_at_u,
    sigma_gradients_at_v,
    sigma_gradients_at_w,
)
from .tensor import (
    constant_horizontal_diffusivity,
    horizontal_diagonal_diffusivity,
    vertical_offdiagonal_diffusivity,
    w_tensor,
)
from .visbeck import visbeck_k

GM_TAPER_SCHEMES = ("", "linear", "gkw91", "dm95", "ldd97", "ac02")


@dataclass(frozen=True)
class GMRediParameters:
    """``data.gmredi`` values (after MITgcm's defaults) and the CPP switches that matter in range."""

    taper_scheme: str  # GM_taper_scheme
    adv_form: bool  # GM_AdvForm
    isopyc_k: float  # GM_isopycK, m^2/s (MITgcm default: GM_background_K)
    background_k: float  # GM_background_K, m^2/s
    max_slope: float  # GM_maxSlope
    kmin_horiz: float  # GM_Kmin_horiz, m^2/s
    small_number: float  # GM_Small_Number
    slope_sq_cutoff: float  # GM_slopeSqCutoff
    s_crit: float  # GM_Scrit
    s_d: float  # GM_Sd
    non_unity_diagonal: bool  # CPP GM_NON_UNITY_DIAGONAL
    extra_diagonal: bool  # CPP GM_EXTRA_DIAGONAL
    exclude_tapering: bool  # CPP GM_EXCLUDE_TAPERING
    # GM_Visbeck_* (gmredi_readparms.F:150-157, maxSlope unset -> GM_maxSlope :225-227),
    # read when GM_Visbeck_alpha > 0, which needs CPP GM_VISBECK_VARIABLE_K (gmredi_check.F:166-176).
    visbeck_alpha: float
    visbeck_length: float  # m
    visbeck_depth: float  # m
    visbeck_min_depth: float  # m
    visbeck_max_slope: float
    visbeck_min_val_k: float  # m^2/s
    visbeck_max_val_k: float  # m^2/s
    iso_fac_calc_k: float  # GM_isoFac_calcK, share of the computed K in the Redi K

    def __post_init__(self) -> None:
        if self.taper_scheme not in GM_TAPER_SCHEMES:
            raise ValueError(
                f"GM_taper_scheme={self.taper_scheme!r} is not supported; "
                f"accepted: {GM_TAPER_SCHEMES}"
            )
        if self.exclude_tapering:
            # Every scheme in GM_TAPER_SCHEMES takes the branch that is only a STOP
            # under this macro (gmredi_slope_limit.F:441-445, gmredi_slope_psi.F:185-189).
            raise ValueError(
                f"GM_EXCLUDE_TAPERING is defined, which stops MITgcm for GM_taper_scheme="
                f"{self.taper_scheme!r}"
            )
        if self.visbeck_alpha < 0.0:
            raise ValueError(
                f"GM_Visbeck_alpha={self.visbeck_alpha} is not supported; MITgcm computes the "
                "Visbeck K for alpha > 0 only (gmredi_calc_tensor.F:416,579)"
            )
        if self.visbeck_alpha > 0.0 and self.iso_fac_calc_k != 0.0 and not self.non_unity_diagonal:
            # gmredi_check.F:331-345.
            raise ValueError(
                "GM_Visbeck_alpha with GM_isoFac_calcK != 0 needs GM_NON_UNITY_DIAGONAL"
            )
        if self.extra_diag and not self.extra_diagonal:
            # gmredi_check.F:307-318.
            raise ValueError(
                "GM_ExtraDiag is true (GM_isopycK differs from the skew-flux K) "
                "but GM_EXTRA_DIAGONAL is not defined"
            )

    @property
    def skewflx(self) -> float:
        """``GM_skewflx`` (``gmredi_readparms.F:242-253``)."""
        return 0.0 if self.adv_form else 1.0

    @property
    def extra_diag(self) -> bool:
        """``GM_ExtraDiag`` without K files (``gmredi_readparms.F:242-272``)."""
        if self.iso_fac_calc_k != self.skewflx and self.visbeck_alpha != 0.0:
            return True
        if self.adv_form:
            return self.isopyc_k != 0.0
        return self.isopyc_k != self.background_k


@dataclass(frozen=True)
class GMRediTensor:
    """``Kwx Kwy Kwz`` (W points), ``Kux Kuz`` (U), ``Kvy Kvz`` (V), ``GM_PsiX/Y``, each ``[Nr, NY, NX]``.

    ``kuz``/``kvz`` stay zero unless ``extra_diagonal`` and ``extra_diag``;
    ``psi_x``/``psi_y`` stay zero unless ``adv_form``, as MITgcm's arrays do.
    """

    kwx: torch.Tensor
    kwy: torch.Tensor
    kwz: torch.Tensor
    kux: torch.Tensor
    kvy: torch.Tensor
    kuz: torch.Tensor
    kvz: torch.Tensor
    psi_x: torch.Tensor
    psi_y: torch.Tensor


def _taper(
    params: GMRediParameters,
    slope_sq: torch.Tensor,
    taper: torch.Tensor,
    *,
    r_depth: torch.Tensor,
    l_rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Taper of ``gmredi_slope_limit.F:520-642``; returns ``(taperFct, SlopeSqr)``."""
    scheme = params.taper_scheme
    max_slope_sq = params.max_slope * params.max_slope
    if scheme == "":
        return taper, slope_sq
    elif scheme == "linear":
        return linear_taper(
            slope_sq, taper, max_slope_sq=max_slope_sq, slope_sq_cutoff=params.slope_sq_cutoff
        )
    elif scheme == "gkw91":
        taper = gkw91_taper(
            slope_sq, taper, max_slope_sq=max_slope_sq, slope_sq_cutoff=params.slope_sq_cutoff
        )
        return taper, slope_sq
    elif scheme == "dm95":
        taper = dm95_taper(
            slope_sq,
            taper,
            slope_sq_cutoff=params.slope_sq_cutoff,
            s_crit=params.s_crit,
            s_d=params.s_d,
        )
        return taper, slope_sq
    elif scheme == "ldd97":
        taper = ldd97_taper(
            slope_sq,
            taper,
            slope_sq_cutoff=params.slope_sq_cutoff,
            s_crit=params.s_crit,
            s_d=params.s_d,
            r_depth=r_depth,
            l_rho=l_rho,
        )
        return taper, slope_sq


def _slope_limit(
    params: GMRediParameters,
    dsigma_dx: torch.Tensor,
    dsigma_dy: torch.Tensor,
    dsigma_dr: torch.Tensor,
    *,
    r_depth: torch.Tensor,
    l_rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``GMREDI_SLOPE_LIMIT``: ``(SlopeX, SlopeY, SlopeSqr, taperFct)``.

    ac02 is a branch of its own (``gmredi_slope_limit.F:393-439``); the other
    schemes share the slope and cut-off (``:441-518``) and differ in the taper.
    """
    scheme = params.taper_scheme
    if scheme == "ac02":
        return ac02_slopes(
            dsigma_dx, dsigma_dy, dsigma_dr,
            max_slope_sq=params.max_slope * params.max_slope,
            slope_sq_cutoff=params.slope_sq_cutoff,
        )
    elif scheme in ("", "linear", "gkw91", "dm95", "ldd97"):
        slope_x, slope_y, slope_sq, taper = limited_slopes(
            dsigma_dx, dsigma_dy, dsigma_dr,
            small_number=params.small_number, slope_sq_cutoff=params.slope_sq_cutoff,
        )
        taper, slope_sq = _taper(params, slope_sq, taper, r_depth=r_depth, l_rho=l_rho)
        return slope_x, slope_y, slope_sq, taper


def _psi_taper(
    params: GMRediParameters,
    slope: torch.Tensor,
    taper: torch.Tensor,
    *,
    slope_cutoff: float,
    r_depth: torch.Tensor,
    l_rho: torch.Tensor,
) -> torch.Tensor:
    """Taper of ``gmredi_slope_psi.F:263-419`` for one stream-function component."""
    scheme = params.taper_scheme
    if scheme == "":
        return taper
    elif scheme == "linear":
        return psi_linear_taper(
            slope,
            taper,
            max_slope=params.max_slope,
            slope_cutoff=slope_cutoff,
            small_number=params.small_number,
        )
    elif scheme in ("gkw91", "ac02"):
        # gmredi_slope_psi.F:290-312 is one branch for both.
        return psi_gkw91_taper(
            slope,
            taper,
            max_slope=params.max_slope,
            slope_cutoff=slope_cutoff,
            small_number=params.small_number,
        )
    elif scheme == "dm95":
        return psi_dm95_taper(slope, s_crit=params.s_crit, s_d=params.s_d)
    elif scheme == "ldd97":
        return psi_ldd97_taper(
            slope,
            taper,
            slope_cutoff=slope_cutoff,
            s_crit=params.s_crit,
            s_d=params.s_d,
            r_depth=r_depth,
            l_rho=l_rho,
        )


def gmredi_calc_tensor(
    params: GMRediParameters,
    *,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    sigma_r: torch.Tensor,
    maskC: torch.Tensor,
    maskW: torch.Tensor,
    maskS: torch.Tensor,
    f_cori: torch.Tensor,
    r_depth_f: torch.Tensor,
    r_depth_c: torch.Tensor,
    drc: torch.Tensor,
    gravity: float,
    rho_const: float,
) -> GMRediTensor:
    """``gmredi_calc_tensor.F:163-1118`` in the covered range.

    ``sigma_*`` come from :func:`~defgcm.dynamics.gmredi.grad_sigma.grad_sigma`;
    masks are ``[Nr, NY, NX]``; ``f_cori`` (``fCori`` at centres, ``[NY, NX]``)
    is read only by ldd97; ``r_depth_f[k-1] = rF(1)-rF(k)`` and
    ``r_depth_c[k-1] = rF(1)-rC(k)`` are positive ``[Nr, 1, 1]`` columns
    (``rF(1) = 0``).  ``drc`` (``drC(1..Nr)``, ``[Nr, 1, 1]``), ``gravity`` and
    ``rho_const`` (``rhoConst``) are read only by the Visbeck K.
    """
    small = params.small_number
    if params.taper_scheme == "ldd97":
        lrho_c, lrho_w, lrho_s = ldd97_rossby_radii(f_cori)
    else:
        # gmredi_calc_tensor.F:212-220: set to zero, read only by ldd97 (and the rejected fm07).
        lrho_c = lrho_w = lrho_s = torch.zeros_like(maskC[0])

    # First loop, W points, k = Nr..2 (:382-573), times the diffusivities (:650-685).
    dsdx, dsdy, dsdr = sigma_gradients_at_w(sigma_x, sigma_y, sigma_r, maskC=maskC)
    slope_x, slope_y, slope_sq, taper = _slope_limit(
        params, dsdx, dsdy, dsdr, r_depth=r_depth_f[1:], l_rho=lrho_c
    )
    if params.visbeck_alpha > 0.0:
        visbeck = visbeck_k(
            dsdx, dsdy, sigma_r, maskC=maskC, r_depth_c=r_depth_c, drc=drc,
            alpha=params.visbeck_alpha, length=params.visbeck_length, depth=params.visbeck_depth,
            min_depth=params.visbeck_min_depth, max_slope=params.visbeck_max_slope,
            min_val_k=params.visbeck_min_val_k, max_val_k=params.visbeck_max_val_k,
            gravity=gravity, rho_const=rho_const,
        )
        # :587-597: locK3dRedi, locK3dGM, the same on every level, so the second step's
        # op5*(K(km1)+K(k)) (:660-676) is K itself.
        kredi = params.isopyc_k + visbeck * params.iso_fac_calc_k
        k3d_gm = params.background_k + visbeck
        kgm = kredi + params.skewflx * k3d_gm
        # U and V points: op5*(K(i-1)+K(i)) (:790-845, :987-1042).
        kredi_u = 0.5 * (F.pad(kredi[..., :-1], (1, 0)) + kredi)
        kredi_v = 0.5 * (F.pad(kredi[..., :-1, :], (0, 0, 1, 0)) + kredi)
        kgm_u = 0.5 * (F.pad(k3d_gm[..., :-1], (1, 0)) + k3d_gm)
        kgm_v = 0.5 * (F.pad(k3d_gm[..., :-1, :], (0, 0, 1, 0)) + k3d_gm)
        # gmredi_calc_psi_bolus.F:150-153,171-174: op25*(K(i-1,km1)+K(i,km1)+K(i-1,k)+K(i,k)).
        west, south = F.pad(k3d_gm[..., :-1], (1, 0)), F.pad(k3d_gm[..., :-1, :], (0, 0, 1, 0))
        kgm_psi_x = 0.25 * (west + k3d_gm + west + k3d_gm)
        kgm_psi_y = 0.25 * (south + k3d_gm + south + k3d_gm)
    else:
        kredi = kredi_u = kredi_v = params.isopyc_k
        kgm = kredi + params.skewflx * params.background_k
        kgm_u = kgm_v = kgm_psi_x = kgm_psi_y = params.background_k
    kwx, kwy, kwz = w_tensor(slope_x, slope_y, slope_sq, taper, kgm=kgm, kredi=kredi)

    # Stream functions for the advective form (:609-635, gmredi_calc_psi_bolus.F).
    if params.adv_form:
        slope_cutoff = math.sqrt(params.slope_sq_cutoff)
        psi_slope, psi_dsdr = psi_slope_inputs_x(sigma_x, sigma_r, maskW=maskW)
        psi_slope, psi_taper = psi_limited_slope(
            psi_slope, psi_dsdr, small_number=small, slope_cutoff=slope_cutoff
        )
        psi_taper = _psi_taper(
            params,
            psi_slope,
            psi_taper,
            slope_cutoff=slope_cutoff,
            r_depth=r_depth_f[1:],
            l_rho=lrho_w,
        )
        psi_x = bolus_streamfunction(psi_slope, psi_taper, background_k=kgm_psi_x)
        psi_slope, psi_dsdr = psi_slope_inputs_y(sigma_y, sigma_r, maskS=maskS)
        psi_slope, psi_taper = psi_limited_slope(
            psi_slope, psi_dsdr, small_number=small, slope_cutoff=slope_cutoff
        )
        psi_taper = _psi_taper(
            params,
            psi_slope,
            psi_taper,
            slope_cutoff=slope_cutoff,
            r_depth=r_depth_f[1:],
            l_rho=lrho_s,
        )
        psi_y = bolus_streamfunction(psi_slope, psi_taper, background_k=kgm_psi_y)
    else:
        # :223-232 clears GM_PsiX/Y every call.
        psi_x = psi_y = torch.zeros_like(sigma_x)

    # Second and third loops, U and V points, all levels (:696-1090).
    if params.non_unity_diagonal or params.extra_diagonal:
        dsdx, dsdy, dsdr = sigma_gradients_at_u(sigma_x, sigma_y, sigma_r, maskW=maskW)
        slope_x_u, _, _, taper_u = _slope_limit(
            params, dsdx, dsdy, dsdr, r_depth=r_depth_c, l_rho=lrho_w
        )
        dsdx, dsdy, dsdr = sigma_gradients_at_v(sigma_x, sigma_y, sigma_r, maskS=maskS)
        _, slope_y_v, _, taper_v = _slope_limit(
            params, dsdx, dsdy, dsdr, r_depth=r_depth_c, l_rho=lrho_s
        )
    if params.non_unity_diagonal:
        kux = horizontal_diagonal_diffusivity(
            taper_u, isopyc_k=kredi_u, kmin_horiz=params.kmin_horiz
        )
        kvy = horizontal_diagonal_diffusivity(
            taper_v, isopyc_k=kredi_v, kmin_horiz=params.kmin_horiz
        )
    else:
        # :1093-1118.
        kux = constant_horizontal_diffusivity(sigma_x, isopyc_k=params.isopyc_k)
        kvy = constant_horizontal_diffusivity(sigma_y, isopyc_k=params.isopyc_k)
    if params.extra_diagonal and params.extra_diag:
        kuz = vertical_offdiagonal_diffusivity(
            slope_x_u,
            taper_u,
            isopyc_k=kredi_u,
            background_k=kgm_u,
            skewflx=params.skewflx,
        )
        kvz = vertical_offdiagonal_diffusivity(
            slope_y_v,
            taper_v,
            isopyc_k=kredi_v,
            background_k=kgm_v,
            skewflx=params.skewflx,
        )
    else:
        kuz = kvz = torch.zeros_like(sigma_x)
    return GMRediTensor(
        kwx=kwx, kwy=kwy, kwz=kwz, kux=kux, kvy=kvy, kuz=kuz, kvz=kvz, psi_x=psi_x, psi_y=psi_y
    )
