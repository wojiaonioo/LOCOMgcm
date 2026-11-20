"""Vertical-shear kernels for vector-invariant momentum.

All public entry points in this module expect ``w_interface`` in MITgcm
upward-positive convention. LOCOMgcm's diagnosed vertical velocity is
normally downward-positive, so callers must convert at the producer-consumer
boundary before storing or passing cached ``w_interface`` here.

The caller owns the halo exchanges: ``w`` (and ``maskC``, ``rA``) arrive with
one halo cell on each side along x for the U faces and along y for the V faces.
The face w-bars come from one of two kernels, area-weighted
(``rAdvAreaWeight``) or simple average; the shear ``-w * du/dz`` is applied by
a per-layer loop or by a whole-z form with the same operator order.  u, v and
their tendencies carry z at dim -3; ``recip_hFacW``/``recip_hFacS`` are
``None`` when the grid carries no ``hFacW``/``hFacS``.
"""
from __future__ import annotations


import torch


def _vertical_shear_face_wbar_pair(
    term_a: torch.Tensor,
    term_b: torch.Tensor,
    *,
    layer_mask: float,
    recip_metric: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average adjacent vertical-shear face terms with optional reciprocal."""
    result = 0.5 * (term_a + term_b)
    if recip_metric is not None:
        result = result * recip_metric
    return result * layer_mask



def _vertical_shear_area_weighted_w(
    w_value: torch.Tensor,
    area: torch.Tensor,
    *,
    center_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Area-weight MIT upward-positive vertical velocity for vertical shear."""
    result = w_value * area
    if center_mask is not None:
        result = result * center_mask
    return result



def _vertical_shear_rksign_deltas(
    field: torch.Tensor,
    *,
    k: int,
    nz: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer deltas for MIT vertical shear with local ``rkSign=-1``."""
    kp1 = min(k + 1, nz - 1)
    km1 = max(k - 1, 0)
    mask_kp1 = 1.0 if k < nz - 1 else 0.0
    mask_km1 = 1.0 if k > 0 else 0.0

    field_k = field[..., k, :, :]
    field_km1 = field[..., km1, :, :]
    field_kp1 = field[..., kp1, :, :]
    delta_m = -(field_k - field_km1 * mask_km1)
    delta_p = -(field_kp1 * mask_kp1 - field_k)
    return delta_m, delta_p



def _vertical_shear_face_combine(
    wbar_m: torch.Tensor,
    wbar_p: torch.Tensor,
    delta_m: torch.Tensor,
    delta_p: torch.Tensor,
) -> torch.Tensor:
    """Combine face w-bars and layer deltas for MIT vertical shear."""
    return -0.5 * (wbar_p * delta_p + wbar_m * delta_m)



def vertical_shear_per_layer(
    u: torch.Tensor,
    v: torch.Tensor,
    wbar_xm: torch.Tensor,
    wbar_xp: torch.Tensor,
    wbar_ym: torch.Tensor,
    wbar_yp: torch.Tensor,
    recip_drF: torch.Tensor,
    *,
    recip_hFacW: torch.Tensor | None,
    recip_hFacS: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor-only MIT vertical-shear kernel, one Python step per layer.

    Inputs ``wbar_*`` are already averaged to U/V faces and use MIT
    upward-positive vertical velocity. The local ``rkSign=-1`` convention is
    represented by the unary minus on the vertical deltas.
    """
    nz = u.shape[-3]
    tend_u = torch.zeros_like(u)
    tend_v = torch.zeros_like(v)
    for k in range(nz):
        uZm, uZp = _vertical_shear_rksign_deltas(u, k=k, nz=nz)
        shear_u = _vertical_shear_face_combine(
            wbar_xm[..., k, :, :],
            wbar_xp[..., k, :, :],
            uZm,
            uZp,
        )
        if recip_hFacW is not None:
            shear_u = shear_u * recip_hFacW[..., k, :, :]
        tend_u[..., k, :, :] = shear_u * recip_drF[k]

        vZm, vZp = _vertical_shear_rksign_deltas(v, k=k, nz=nz)
        shear_v = _vertical_shear_face_combine(
            wbar_ym[..., k, :, :],
            wbar_yp[..., k, :, :],
            vZm,
            vZp,
        )
        if recip_hFacS is not None:
            shear_v = shear_v * recip_hFacS[..., k, :, :]
        tend_v[..., k, :, :] = shear_v * recip_drF[k]

    return tend_u, tend_v


def vertical_shear_all_z(
    u: torch.Tensor,
    v: torch.Tensor,
    wbar_xm: torch.Tensor,
    wbar_xp: torch.Tensor,
    wbar_ym: torch.Tensor,
    wbar_yp: torch.Tensor,
    recip_drF: torch.Tensor,
    *,
    recip_hFacW: torch.Tensor | None,
    recip_hFacS: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor-only MIT vertical-shear kernel over the whole z axis at once.

    Inputs ``wbar_*`` are already averaged to U/V faces and use MIT
    upward-positive vertical velocity. The local ``rkSign=-1`` convention is
    represented by the unary minus on the vertical deltas.
    Same operator order as :func:`vertical_shear_per_layer`, so eager FP64
    rounding is identical.
    """
    nz = u.shape[-3]
    # Whole-z-axis vectorization of the per-layer loop.  z is dim=-3 throughout.
    # Per-layer deltas (_vertical_shear_rksign_deltas) become cat-shifted full
    # tensors plus per-layer 0/1 mask vectors; the combine/recip/scale chain is
    # applied element-wise with the SAME operator order as the loop so eager
    # FP64 rounding is identical (incl. the mask*field x*0.0 boundary terms).
    shear_u = _vertical_shear_combine_all_z(u, wbar_xm, wbar_xp, nz)
    shear_v = _vertical_shear_combine_all_z(v, wbar_ym, wbar_yp, nz)
    if recip_hFacW is not None:
        shear_u = shear_u * recip_hFacW
    if recip_hFacS is not None:
        shear_v = shear_v * recip_hFacS
    recip_drF_b = recip_drF[:, None, None]
    tend_u = shear_u * recip_drF_b
    tend_v = shear_v * recip_drF_b
    return tend_u, tend_v


def _vertical_shear_deltas_all_z(
    field: torch.Tensor,
    nz: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-z-axis form of ``_vertical_shear_rksign_deltas`` (z is dim=-3).

    Layer-k delta_m uses field[k-1] (field[0] at k=0, masked to 0) and
    delta_p uses field[k+1] (field[nz-1] at k=nz-1, masked to 0).  The masked
    boundary terms keep the explicit ``field * 0.0`` multiply (not a drop) so
    NaN / signed-zero semantics match the loop exactly.
    """
    dtype = field.dtype
    device = field.device
    field_km1 = torch.cat([field[..., 0:1, :, :], field[..., :-1, :, :]], dim=-3)
    field_kp1 = torch.cat([field[..., 1:, :, :], field[..., -1:, :, :]], dim=-3)
    mask_km1 = torch.ones(nz, dtype=dtype, device=device)
    mask_km1[0] = 0.0
    mask_kp1 = torch.ones(nz, dtype=dtype, device=device)
    mask_kp1[-1] = 0.0
    mask_km1_b = mask_km1[:, None, None]
    mask_kp1_b = mask_kp1[:, None, None]
    delta_m = -(field - field_km1 * mask_km1_b)
    delta_p = -(field_kp1 * mask_kp1_b - field)
    return delta_m, delta_p



def _vertical_shear_combine_all_z(
    field: torch.Tensor,
    wbar_m: torch.Tensor,
    wbar_p: torch.Tensor,
    nz: int,
) -> torch.Tensor:
    """Whole-z-axis ``_vertical_shear_face_combine`` with rkSign deltas."""
    delta_m, delta_p = _vertical_shear_deltas_all_z(field, nz)
    return -0.5 * (wbar_p * delta_p + wbar_m * delta_m)



def _vertical_shear_layer_masks(
    nz: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer 0/1 masks over z for the k-1 and k+1 interfaces, ``[1, nz, 1, 1]``."""
    mask_km1 = torch.ones(nz, dtype=dtype, device=device)
    mask_km1[0] = 0.0
    mask_kp1 = torch.ones(nz, dtype=dtype, device=device)
    mask_kp1[-1] = 0.0
    mask_km1_b = mask_km1[None, :, None, None]
    mask_kp1_b = mask_kp1[None, :, None, None]
    return mask_km1_b, mask_kp1_b


def _vertical_shear_km1_center_mask(
    maskC_ext: torch.Tensor | None,
    mask_km1_b: torch.Tensor,
    w_top: torch.Tensor,
    *,
    nz: int,
) -> torch.Tensor:
    """k-1 center mask over z, zeros at k=0.

    maskC absent: ones for k>=1 (the legacy "ones for k>=1" branch when hFacC
    is None), zeros at k=0.
    """
    if maskC_ext is not None:
        return torch.cat(
            [torch.zeros_like(maskC_ext[:, 0:1]), maskC_ext[:, 0:nz - 1]], dim=1
        )
    return mask_km1_b.expand(1, nz, w_top.shape[-2], w_top.shape[-1])


def vertical_shear_wbars_area_weighted(
    w_ext_u: torch.Tensor,
    w_ext_v: torch.Tensor,
    maskC_ext_u: torch.Tensor | None,
    maskC_ext_v: torch.Tensor | None,
    rA_ext_u: torch.Tensor,
    rA_ext_v: torch.Tensor,
    u_ref: torch.Tensor,
    v_ref: torch.Tensor,
    *,
    recip_rAw: torch.Tensor,
    recip_rAs: torch.Tensor,
    u_face_keep: torch.Tensor,
    v_face_keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Area-weighted (``rAdvAreaWeight=.TRUE.``) face w-bars from haloed fields.

    The whole-z arithmetic, the ``(w*rA)*maskC`` and ``0.5*(a+b)*recip*mask``
    operator chains, and the k-1 center-mask shift reproduce the legacy cube /
    region vectorized builders exactly (eager FP64 rounding preserved).  z is
    dim=1 of the haloed ``w`` and of the wbar outputs.  The outer-face carve
    multiplies by ``u_face_keep`` / ``v_face_keep``: zero on the closed outer
    faces of a regional grid, all ones on the cube.
    """
    nz = u_ref.shape[-3]
    nx = u_ref.shape[-1] - 1
    ny = v_ref.shape[-2] - 1
    mask_km1_b, mask_kp1_b = _vertical_shear_layer_masks(
        nz, dtype=u_ref.dtype, device=u_ref.device
    )

    w_top_u = w_ext_u[:, 0:nz]
    w_bot_u = w_ext_u[:, 1:nz + 1]
    w_top_v = w_ext_v[:, 0:nz]
    w_bot_v = w_ext_v[:, 1:nz + 1]
    mc_u_all = _vertical_shear_km1_center_mask(maskC_ext_u, mask_km1_b, w_top_u, nz=nz)
    mc_v_all = _vertical_shear_km1_center_mask(maskC_ext_v, mask_km1_b, w_top_v, nz=nz)

    # Insert the z axis into the (z-independent) face metrics / keep masks at
    # dim -3 so 2-D (region [ny,nx+1]) and 3-D (cube [6,ny,nx+1]) both broadcast
    # over (face/batch, z) exactly like the loop's per-k ``result * recip``.
    recip_rAw_b = recip_rAw.unsqueeze(-3)
    recip_rAs_b = recip_rAs.unsqueeze(-3)
    u_keep_b = u_face_keep.unsqueeze(-3)
    v_keep_b = v_face_keep.unsqueeze(-3)

    rA_ext_u_b = rA_ext_u.unsqueeze(-3)
    rA_ext_v_b = rA_ext_v.unsqueeze(-3)

    wrA_top_u = _vertical_shear_area_weighted_w(
        w_top_u, rA_ext_u_b, center_mask=mc_u_all
    )
    wrA_bot_u = _vertical_shear_area_weighted_w(w_bot_u, rA_ext_u_b)
    wrA_top_v = _vertical_shear_area_weighted_w(
        w_top_v, rA_ext_v_b, center_mask=mc_v_all
    )
    wrA_bot_v = _vertical_shear_area_weighted_w(w_bot_v, rA_ext_v_b)

    wbar_xm = _vertical_shear_face_wbar_pair(
        wrA_top_u[..., 1:nx + 2],
        wrA_top_u[..., 0:nx + 1],
        recip_metric=recip_rAw_b,
        layer_mask=mask_km1_b,
    )
    wbar_xp = _vertical_shear_face_wbar_pair(
        wrA_bot_u[..., 1:nx + 2],
        wrA_bot_u[..., 0:nx + 1],
        recip_metric=recip_rAw_b,
        layer_mask=mask_kp1_b,
    )
    wbar_ym = _vertical_shear_face_wbar_pair(
        wrA_top_v[..., 1:ny + 2, :],
        wrA_top_v[..., 0:ny + 1, :],
        recip_metric=recip_rAs_b,
        layer_mask=mask_km1_b,
    )
    wbar_yp = _vertical_shear_face_wbar_pair(
        wrA_bot_v[..., 1:ny + 2, :],
        wrA_bot_v[..., 0:ny + 1, :],
        recip_metric=recip_rAs_b,
        layer_mask=mask_kp1_b,
    )

    wbar_xm = wbar_xm * u_keep_b
    wbar_xp = wbar_xp * u_keep_b
    wbar_ym = wbar_ym * v_keep_b
    wbar_yp = wbar_yp * v_keep_b
    return wbar_xm, wbar_xp, wbar_ym, wbar_yp


def vertical_shear_wbars_simple(
    w_ext_u: torch.Tensor,
    w_ext_v: torch.Tensor,
    maskC_ext_u: torch.Tensor | None,
    maskC_ext_v: torch.Tensor | None,
    u_ref: torch.Tensor,
    v_ref: torch.Tensor,
    *,
    u_face_keep: torch.Tensor,
    v_face_keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple-average (``rAdvAreaWeight=.FALSE.``) face w-bars from haloed fields.

    The whole-z arithmetic, the ``(w*rA)*maskC`` and ``0.5*(a+b)*recip*mask``
    operator chains, and the k-1 center-mask shift reproduce the legacy cube /
    region vectorized builders exactly (eager FP64 rounding preserved).  z is
    dim=1 of the haloed ``w`` and of the wbar outputs.  The outer-face carve
    multiplies by ``u_face_keep`` / ``v_face_keep``: zero on the closed outer
    faces of a regional grid, all ones on the cube.
    """
    nz = u_ref.shape[-3]
    nx = u_ref.shape[-1] - 1
    ny = v_ref.shape[-2] - 1
    mask_km1_b, mask_kp1_b = _vertical_shear_layer_masks(
        nz, dtype=u_ref.dtype, device=u_ref.device
    )

    w_top_u = w_ext_u[:, 0:nz]
    w_bot_u = w_ext_u[:, 1:nz + 1]
    w_top_v = w_ext_v[:, 0:nz]
    w_bot_v = w_ext_v[:, 1:nz + 1]
    mc_u_all = _vertical_shear_km1_center_mask(maskC_ext_u, mask_km1_b, w_top_u, nz=nz)
    mc_v_all = _vertical_shear_km1_center_mask(maskC_ext_v, mask_km1_b, w_top_v, nz=nz)

    u_keep_b = u_face_keep.unsqueeze(-3)
    v_keep_b = v_face_keep.unsqueeze(-3)

    w_top_u_masked = w_top_u * mc_u_all
    w_top_v_masked = w_top_v * mc_v_all
    wbar_xm = _vertical_shear_face_wbar_pair(
        w_top_u_masked[..., 1:nx + 2],
        w_top_u_masked[..., 0:nx + 1],
        layer_mask=mask_km1_b,
    )
    wbar_xp = _vertical_shear_face_wbar_pair(
        w_bot_u[..., 1:nx + 2],
        w_bot_u[..., 0:nx + 1],
        layer_mask=mask_kp1_b,
    )
    wbar_ym = _vertical_shear_face_wbar_pair(
        w_top_v_masked[..., 1:ny + 2, :],
        w_top_v_masked[..., 0:ny + 1, :],
        layer_mask=mask_km1_b,
    )
    wbar_yp = _vertical_shear_face_wbar_pair(
        w_bot_v[..., 1:ny + 2, :],
        w_bot_v[..., 0:ny + 1, :],
        layer_mask=mask_kp1_b,
    )

    wbar_xm = wbar_xm * u_keep_b
    wbar_xp = wbar_xp * u_keep_b
    wbar_ym = wbar_ym * v_keep_b
    wbar_yp = wbar_yp * v_keep_b
    return wbar_xm, wbar_xp, wbar_ym, wbar_yp
