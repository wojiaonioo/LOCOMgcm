"""Kinetic energy at cell centres and its gradient at U/V faces.

Tensor kernels of the vector-invariant momentum form: MITgcm ``mom_calc_ke.F``
(selectKEscheme 0-3) and ``mom_vi_u_grad_ke.F`` / ``mom_vi_v_grad_ke.F``.

Velocities are ``u`` at U faces ``[..., ny, nx+1]`` and ``v`` at V faces
``[..., ny+1, nx]``, with an optional vertical axis at -3 and, on the cube, a
leading face axis.  KE is returned at cell centres ``[..., ny, nx]`` in the
inputs' dtype.  The weighted schemes take the area metrics ``rA`` (centre),
``rAw`` (U face), ``rAs`` (V face) without a vertical axis and the fractions
``hFacW``/``hFacS``/``hFacC`` with one; ``rA`` and ``hFacC`` enter through a
positive-only reciprocal clamped at 1e-30 (zero where not positive).  The
gradient kernels take KE already haloed by one cell along the derivative and
return ``(KE(i-1) - KE(i)) / dxC`` (``dyC`` for V) unmasked; the caller applies
the face mask.
"""

from __future__ import annotations

import torch


def ke_center_unweighted(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """MIT selectKEscheme=0 center kinetic energy tensor expression."""
    u_sq = u[..., :-1] ** 2 + u[..., 1:] ** 2
    v_sq = v[..., :-1, :] ** 2 + v[..., 1:, :] ** 2
    return 0.25 * (u_sq + v_sq)


def ke_center_area_weighted(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    rA: torch.Tensor,
    rAw: torch.Tensor,
    rAs: torch.Tensor,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """MIT selectKEscheme=1: U/V terms weighted by ``rAw``/``rAs``, times ``recip_rA``."""
    u_w2, u_e2, v_s2, v_n2 = _ke_center_weighted_face_squares(u, v)
    recip_rA_view, rAw_w_view, rAw_e_view, rAs_s_view, rAs_n_view = (
        _ke_center_area_terms(u, rA, rAw, rAs, ny=ny, nx=nx)
    )
    return _ke_center_area_weighted_from_squares(
        u_w2,
        u_e2,
        v_s2,
        v_n2,
        rAw_w_view,
        rAw_e_view,
        rAs_s_view,
        rAs_n_view,
        recip_rA_view,
    )


def ke_center_hfac_weighted(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    hFacC: torch.Tensor,
) -> torch.Tensor:
    """MIT selectKEscheme=2: U/V terms weighted by ``hFacW``/``hFacS``, times ``recip_hFacC``."""
    u_w2, u_e2, v_s2, v_n2 = _ke_center_weighted_face_squares(u, v)
    hFacW_w, hFacW_e, hFacS_s, hFacS_n, recip_hFacC = _ke_center_hfac_terms(
        u, v, hFacW, hFacS, hFacC
    )
    return _ke_center_hfac_weighted_from_squares(
        u_w2,
        u_e2,
        v_s2,
        v_n2,
        hFacW_w,
        hFacW_e,
        hFacS_s,
        hFacS_n,
        recip_hFacC,
    )


def ke_center_area_hfac_weighted(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    rA: torch.Tensor,
    rAw: torch.Tensor,
    rAs: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    hFacC: torch.Tensor,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """MIT selectKEscheme=3: both weights, times ``recip_hFacC * recip_rA``."""
    u_w2, u_e2, v_s2, v_n2 = _ke_center_weighted_face_squares(u, v)
    recip_rA_view, rAw_w_view, rAw_e_view, rAs_s_view, rAs_n_view = (
        _ke_center_area_terms(u, rA, rAw, rAs, ny=ny, nx=nx)
    )
    hFacW_w, hFacW_e, hFacS_s, hFacS_n, recip_hFacC = _ke_center_hfac_terms(
        u, v, hFacW, hFacS, hFacC
    )
    return _ke_center_area_hfac_weighted_from_squares(
        u_w2,
        u_e2,
        v_s2,
        v_n2,
        hFacW_w,
        hFacW_e,
        hFacS_s,
        hFacS_n,
        rAw_w_view,
        rAw_e_view,
        rAs_s_view,
        rAs_n_view,
        recip_hFacC,
        recip_rA_view,
    )


def _ke_center_area_terms(
    u: torch.Tensor,
    rA: torch.Tensor,
    rAw: torch.Tensor,
    rAs: torch.Tensor,
    *,
    ny: int,
    nx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``recip_rA`` and the four adjacent face areas, viewed for the field's axes."""
    recip_rA = _positive_clamped_reciprocal(rA, min_value=1e-30)
    recip_rA_view = _ke_center_view_center_metric(
        recip_rA,
        field_ndim=u.ndim,
        ny=ny,
        nx=nx,
    )
    rAw_w_view = _ke_center_view_center_metric(
        rAw[..., :nx],
        field_ndim=u.ndim,
        ny=ny,
        nx=nx,
    )
    rAw_e_view = _ke_center_view_center_metric(
        rAw[..., 1 : nx + 1], field_ndim=u.ndim, ny=ny, nx=nx,
    )
    rAs_s_view = _ke_center_view_center_metric(
        rAs[..., :ny, :], field_ndim=u.ndim, ny=ny, nx=nx,
    )
    rAs_n_view = _ke_center_view_center_metric(
        rAs[..., 1 : ny + 1, :], field_ndim=u.ndim, ny=ny, nx=nx,
    )
    return recip_rA_view, rAw_w_view, rAw_e_view, rAs_s_view, rAs_n_view


def _ke_center_hfac_terms(
    u: torch.Tensor,
    v: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    hFacC: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four adjacent face hFacs and ``recip_hFacC``, viewed for the field's axes."""
    hFacW = _ke_center_view_hfac(hFacW, u)
    hFacS = _ke_center_view_hfac(hFacS, v)
    hFacC = _ke_center_view_hfac(hFacC, u)
    recip_hFacC = _positive_clamped_reciprocal(hFacC, min_value=1e-30)

    hFacW_w, hFacW_e, hFacS_s, hFacS_n = _ke_center_hfac_face_terms(
        hFacW,
        hFacS,
    )
    return hFacW_w, hFacW_e, hFacS_s, hFacS_n, recip_hFacC


def _positive_clamped_reciprocal(
    value: torch.Tensor,
    *,
    min_value: float,
) -> torch.Tensor:
    """Positive-only reciprocal with MIT-style small-value clamp."""
    return torch.where(
        value > 0,
        1.0 / value.clamp(min=min_value),
        torch.zeros_like(value),
    )


def _ke_center_view_center_metric(
    value: torch.Tensor,
    *,
    field_ndim: int,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """Keep leading face axes and broadcast over the field's vertical axis."""
    return value.view(
        *value.shape[:-2], *([1] * (field_ndim - value.ndim)), ny, nx
    )


def _ke_center_view_hfac(
    value: torch.Tensor,
    field: torch.Tensor,
) -> torch.Tensor:
    """Broadcast hFac's trailing vertical/horizontal axes, retaining cube faces."""
    if field.ndim >= 4 and value.shape[-3] == field.shape[-3]:
        return value.view(*([1] * (field.ndim - value.ndim)), *value.shape)
    surface = value.select(-3, 0)
    return surface.view(*([1] * (field.ndim - surface.ndim)), *surface.shape)


def _ke_center_weighted_face_squares(
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Face-adjacent velocity squares for MIT weighted KE schemes."""
    u_w2 = u[..., :-1] * u[..., :-1]
    u_e2 = u[..., 1:] * u[..., 1:]
    v_s2 = v[..., :-1, :] * v[..., :-1, :]
    v_n2 = v[..., 1:, :] * v[..., 1:, :]
    return u_w2, u_e2, v_s2, v_n2


def _ke_center_hfac_face_terms(
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adjacent hFac terms for MIT hFac-weighted KE schemes."""
    hFacW_w = hFacW[..., :-1]
    hFacW_e = hFacW[..., 1:]
    hFacS_s = hFacS[..., :-1, :]
    hFacS_n = hFacS[..., 1:, :]
    return hFacW_w, hFacW_e, hFacS_s, hFacS_n


def _ke_center_area_weighted_from_squares(
    u_w2: torch.Tensor,
    u_e2: torch.Tensor,
    v_s2: torch.Tensor,
    v_n2: torch.Tensor,
    rAw_w: torch.Tensor,
    rAw_e: torch.Tensor,
    rAs_s: torch.Tensor,
    rAs_n: torch.Tensor,
    recip_rA: torch.Tensor,
) -> torch.Tensor:
    """MIT selectKEscheme=1 center KE from precomputed squares and metrics."""
    return (
        u_w2 * rAw_w
        + u_e2 * rAw_e
        + v_s2 * rAs_s
        + v_n2 * rAs_n
    ) * 0.25 * recip_rA


def _ke_center_hfac_weighted_from_squares(
    u_w2: torch.Tensor,
    u_e2: torch.Tensor,
    v_s2: torch.Tensor,
    v_n2: torch.Tensor,
    hFacW_w: torch.Tensor,
    hFacW_e: torch.Tensor,
    hFacS_s: torch.Tensor,
    hFacS_n: torch.Tensor,
    recip_hFacC: torch.Tensor,
) -> torch.Tensor:
    """MIT selectKEscheme=2 center KE from precomputed squares and hFacs."""
    return (
        u_w2 * hFacW_w
        + u_e2 * hFacW_e
        + v_s2 * hFacS_s
        + v_n2 * hFacS_n
    ) * 0.25 * recip_hFacC


def _ke_center_area_hfac_weighted_from_squares(
    u_w2: torch.Tensor,
    u_e2: torch.Tensor,
    v_s2: torch.Tensor,
    v_n2: torch.Tensor,
    hFacW_w: torch.Tensor,
    hFacW_e: torch.Tensor,
    hFacS_s: torch.Tensor,
    hFacS_n: torch.Tensor,
    rAw_w: torch.Tensor,
    rAw_e: torch.Tensor,
    rAs_s: torch.Tensor,
    rAs_n: torch.Tensor,
    recip_hFacC: torch.Tensor,
    recip_rA: torch.Tensor,
) -> torch.Tensor:
    """MIT selectKEscheme=3 center KE from precomputed squares and metrics."""
    return (
        u_w2 * hFacW_w * rAw_w
        + u_e2 * hFacW_e * rAw_e
        + v_s2 * hFacS_s * rAs_s
        + v_n2 * hFacS_n * rAs_n
    ) * 0.25 * recip_hFacC * recip_rA


def ke_gradient_u_from_haloed_ke(
    ke_ext: torch.Tensor,
    dxC: torch.Tensor,
    *,
    nx: int,
) -> torch.Tensor:
    """U-face KE gradient from already haloed center KE."""
    return (ke_ext[..., : nx + 1] - ke_ext[..., 1 : nx + 2]) / dxC


def ke_gradient_v_from_haloed_ke(
    ke_ext: torch.Tensor,
    dyC: torch.Tensor,
    *,
    ny: int,
) -> torch.Tensor:
    """V-face KE gradient from already haloed center KE."""
    return (ke_ext[..., : ny + 1, :] - ke_ext[..., 1 : ny + 2, :]) / dyC
