"""Face-flux kernels of a corner vorticity for vector-invariant momentum.

MITgcm ``mom_vi_u_coriolis.F`` / ``mom_vi_v_coriolis.F`` (selectVortScheme=0-4),
``mom_vi_u/v_coriolis_c4.F`` (highOrderVorticity, upwindVorticity) and the planetary term of ``mom_vi_coriolis.F`` (selectCoriScheme=0-3;
scheme 3 is the scheme-4 expression with ``fCoriG``).
The corner value is a relative or absolute vorticity, or ``fCoriG``, at
vorticity points ``[..., ny+1, nx+1]`` in s^-1.  Velocities arrive haloed by
the caller: ``v_ext`` is V-face ``v`` with one x-halo column on each side
``[..., ny+1, nx+2]``, ``u_ext`` is U-face ``u`` with one y-halo row on each
side ``[..., ny+2, nx+1]``.  The U-face result is ``+vort * vBar * recip_dxC``
at ``[..., ny, nx+1]``, the V-face result ``-vort * uBar * recip_dyC`` at
``[..., ny+1, nx]``, in m s^-2 and the inputs' dtype.

Two metric forms are kept apart and chosen by the caller: the per-cell form on
complete 2-D metrics (cube and variable-spacing regional axes) and the factored
row form on uniform regional axes; an IEEE multiply does not distribute over
the add, so one cannot reproduce the other bit for bit.  ``hFacZ`` weights the
scheme-1 velocity average; the planetary average has no hFac.  The caller picks
the average kernel and passes its result to the shared result kernel.
Face masks multiply the result (as ``mask > 0`` on the per-cell form), and an
edge-pair mask zeroes the edge faces of a non-coupling axis.  The
energy-conserving form (scheme 4) has two stages around the caller's halo
exchange: face transports ``u*dyG*hFacW`` and ``v*dxG*hFacS``, then the
corner-weighted average of the haloed transports divided by ``dxC*hFacW``
(``dyC*hFacS``) through positive-only reciprocals.  Schemes 0 and 2 and
their fourth-order and upwind forms take the same haloed transports and the
corner value times ``r_hFacZ`` (zero on dry corners); scheme 3 takes the
velocity, face length and hFac haloed apart, multiplied in MITgcm's order.
The fourth-order form and scheme 3 read corners beyond the face's own two,
from corner fields with one more column or row on each side.
"""

from __future__ import annotations

import torch

from locomgcm.geometry.metric_bundle import _positive_reciprocal


def _apply_face_mask(result: torch.Tensor, face_mask: torch.Tensor) -> torch.Tensor:
    """Apply a caller-provided face mask tensor."""
    return result * face_mask


def _apply_positive_face_mask(
    result: torch.Tensor,
    face_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a positive-valued face mask as a binary tensor mask."""
    return result * (face_mask > 0).to(dtype=result.dtype, device=result.device)


def _mask_corner_value_by_hfac(
    corner_value: torch.Tensor,
    hFacZ: torch.Tensor,
) -> torch.Tensor:
    """Keep corner values only where caller-provided hFacZ is positive."""
    return torch.where(
        hFacZ > 0,
        corner_value,
        torch.zeros_like(corner_value),
    )


def _absolute_vorticity(
    zeta: torch.Tensor,
    f_g: torch.Tensor,
) -> torch.Tensor:
    """Combine relative and planetary vorticity at corners."""
    return zeta + f_g


def _hfac_pair_denominator(
    hFac_a: torch.Tensor,
    hFac_b: torch.Tensor,
    *,
    min_value: float = 1e-9,
) -> torch.Tensor:
    """Clamped denominator for hFacZ-weighted paired-face averages."""
    return torch.clamp(hFac_a + hFac_b, min=min_value)


def _hfac_pair_weighted_average(
    weighted_a: torch.Tensor,
    weighted_b: torch.Tensor,
    hFac_a: torch.Tensor,
    hFac_b: torch.Tensor,
) -> torch.Tensor:
    """Average two caller-preweighted hFacZ-adjacent face terms."""
    return 0.5 * (weighted_a + weighted_b) / _hfac_pair_denominator(
        hFac_a,
        hFac_b,
    )


def _coriolis_quarter_sum2(
    term_a: torch.Tensor,
    term_b: torch.Tensor,
) -> torch.Tensor:
    """Quarter-scale two caller-prepared Coriolis velocity-bar terms."""
    return 0.25 * (term_a + term_b)


def _coriolis_quarter_sum4(
    term_a: torch.Tensor,
    term_b: torch.Tensor,
    term_c: torch.Tensor,
    term_d: torch.Tensor,
) -> torch.Tensor:
    """Quarter-scale four caller-prepared Coriolis velocity-bar terms."""
    return 0.25 * (term_a + term_b + term_c + term_d)


def _coriolis_scaled_quarter_sum4(
    scale: torch.Tensor,
    term_a: torch.Tensor,
    term_b: torch.Tensor,
    term_c: torch.Tensor,
    term_d: torch.Tensor,
) -> torch.Tensor:
    """Scale then quarter-scale four Coriolis velocity-bar terms."""
    return 0.25 * scale * (term_a + term_b + term_c + term_d)


def _coriolis_metric_pair_sum(
    value_a: torch.Tensor,
    metric_a: torch.Tensor,
    value_b: torch.Tensor,
    metric_b: torch.Tensor,
) -> torch.Tensor:
    """Metric-weighted adjacent pair sum used by Coriolis velocity bars."""
    return value_a * metric_a + value_b * metric_b


def _coriolis_pair_sum(
    term_a: torch.Tensor,
    term_b: torch.Tensor,
) -> torch.Tensor:
    """Adjacent pair sum used before Coriolis row-metric scaling."""
    return term_a + term_b


def _coriolis_scaled_pair_sum(
    scale: torch.Tensor,
    term_a: torch.Tensor,
    term_b: torch.Tensor,
) -> torch.Tensor:
    """Scale an adjacent Coriolis row-metric pair without changing order."""
    return scale * _coriolis_pair_sum(term_a, term_b)


def _coriolis_two_scale_pair_sum(
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    term_a: torch.Tensor,
    term_b: torch.Tensor,
) -> torch.Tensor:
    """Apply two scales to an adjacent Coriolis row-metric pair."""
    return scale_a * scale_b * _coriolis_pair_sum(term_a, term_b)


def _u_vbar_no_hfac(
    v_ext: torch.Tensor,
    dxG_ext: torch.Tensor,
    *,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """U-face average of ``v*dxG`` without hFac, per-cell form (planetary term)."""
    v_south_right = v_ext[..., :ny, 1 : nx + 2]
    v_south_left = v_ext[..., :ny, 0 : nx + 1]
    v_north_right = v_ext[..., 1 : ny + 1, 1 : nx + 2]
    v_north_left = v_ext[..., 1 : ny + 1, 0 : nx + 1]

    dxG_s_right = dxG_ext[..., :ny, 1 : nx + 2]
    dxG_s_left = dxG_ext[..., :ny, 0 : nx + 1]
    dxG_n_right = dxG_ext[..., 1 : ny + 1, 1 : nx + 2]
    dxG_n_left = dxG_ext[..., 1 : ny + 1, 0 : nx + 1]

    return _coriolis_quarter_sum4(
        v_south_right * dxG_s_right,
        v_south_left * dxG_s_left,
        v_north_right * dxG_n_right,
        v_north_left * dxG_n_left,
    )


def _u_vbar_hfacz(
    v_ext: torch.Tensor,
    dxG_ext: torch.Tensor,
    hFacZ: torch.Tensor,
    *,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """U-face hFacZ-weighted average of ``v*dxG``, per-cell form (selectVortScheme=1)."""
    v_south_right = v_ext[..., :ny, 1 : nx + 2]
    v_south_left = v_ext[..., :ny, 0 : nx + 1]
    v_north_right = v_ext[..., 1 : ny + 1, 1 : nx + 2]
    v_north_left = v_ext[..., 1 : ny + 1, 0 : nx + 1]

    dxG_s_right = dxG_ext[..., :ny, 1 : nx + 2]
    dxG_s_left = dxG_ext[..., :ny, 0 : nx + 1]
    dxG_n_right = dxG_ext[..., 1 : ny + 1, 1 : nx + 2]
    dxG_n_left = dxG_ext[..., 1 : ny + 1, 0 : nx + 1]

    hFacZ_south = hFacZ[..., :ny, :]
    hFacZ_north = hFacZ[..., 1 : ny + 1, :]
    return _hfac_pair_weighted_average(
        hFacZ_south
        * _coriolis_metric_pair_sum(
            v_south_right,
            dxG_s_right,
            v_south_left,
            dxG_s_left,
        ),
        hFacZ_north
        * _coriolis_metric_pair_sum(
            v_north_right,
            dxG_n_right,
            v_north_left,
            dxG_n_left,
        ),
        hFacZ_south,
        hFacZ_north,
    )


def _u_face_coriolis_from_vbar(
    v_bar: torch.Tensor,
    corner_value: torch.Tensor,
    recip_dxC_u: torch.Tensor,
    *,
    face_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Per-cell U-face result ``corner average * vBar * recip_dxC``, masked as ``mask > 0``.

    ``corner_value`` is either ``omega3`` (relative/absolute vorticity) or
    ``fCoriG``.  All geometry has already been injected as haloed tensors and
    2-D metrics; ``recip_dxC_u`` is the caller's nonzero reciprocal of ``dxC``.
    """
    ny = recip_dxC_u.shape[-2]
    corner_avg = _u_face_corner_average(corner_value, ny=ny)
    result = _u_face_coriolis_result(corner_avg, v_bar, recip_dxC_u)
    return _apply_coriolis_face_mask(result, face_mask, binary=True)


def _v_ubar_no_hfac(
    u_ext: torch.Tensor,
    dyG_ext: torch.Tensor,
    *,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """V-face average of ``u*dyG`` without hFac, per-cell form (planetary term)."""
    u_left_above = u_ext[..., 1 : ny + 2, :nx]
    u_left_below = u_ext[..., 0 : ny + 1, :nx]
    u_right_above = u_ext[..., 1 : ny + 2, 1 : nx + 1]
    u_right_below = u_ext[..., 0 : ny + 1, 1 : nx + 1]

    dyG_l_above = dyG_ext[..., 1 : ny + 2, :nx]
    dyG_l_below = dyG_ext[..., 0 : ny + 1, :nx]
    dyG_r_above = dyG_ext[..., 1 : ny + 2, 1 : nx + 1]
    dyG_r_below = dyG_ext[..., 0 : ny + 1, 1 : nx + 1]

    return _coriolis_quarter_sum4(
        u_left_above * dyG_l_above,
        u_left_below * dyG_l_below,
        u_right_above * dyG_r_above,
        u_right_below * dyG_r_below,
    )


def _v_ubar_hfacz(
    u_ext: torch.Tensor,
    dyG_ext: torch.Tensor,
    hFacZ: torch.Tensor,
    *,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """V-face hFacZ-weighted average of ``u*dyG``, per-cell form (selectVortScheme=1)."""
    u_left_above = u_ext[..., 1 : ny + 2, :nx]
    u_left_below = u_ext[..., 0 : ny + 1, :nx]
    u_right_above = u_ext[..., 1 : ny + 2, 1 : nx + 1]
    u_right_below = u_ext[..., 0 : ny + 1, 1 : nx + 1]

    dyG_l_above = dyG_ext[..., 1 : ny + 2, :nx]
    dyG_l_below = dyG_ext[..., 0 : ny + 1, :nx]
    dyG_r_above = dyG_ext[..., 1 : ny + 2, 1 : nx + 1]
    dyG_r_below = dyG_ext[..., 0 : ny + 1, 1 : nx + 1]

    hFacZ_left = hFacZ[..., :, :nx]
    hFacZ_right = hFacZ[..., :, 1 : nx + 1]
    return _hfac_pair_weighted_average(
        hFacZ_left
        * _coriolis_metric_pair_sum(
            u_left_above,
            dyG_l_above,
            u_left_below,
            dyG_l_below,
        ),
        hFacZ_right
        * _coriolis_metric_pair_sum(
            u_right_above,
            dyG_r_above,
            u_right_below,
            dyG_r_below,
        ),
        hFacZ_left,
        hFacZ_right,
    )


def _v_face_coriolis_from_ubar(
    u_bar: torch.Tensor,
    corner_value: torch.Tensor,
    recip_dyC_v: torch.Tensor,
    *,
    face_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Per-cell V-face result ``-corner average * uBar * recip_dyC``, masked as ``mask > 0``."""
    nx = recip_dyC_v.shape[-1]
    corner_avg = _v_face_corner_average(corner_value, nx=nx)
    result = _v_face_coriolis_result(corner_avg, u_bar, recip_dyC_v)
    return _apply_coriolis_face_mask(result, face_mask, binary=True)


def _u_vbar_no_hfac_row(
    v_ext: torch.Tensor,
    *,
    dxG_south: torch.Tensor,
    dxG_north: torch.Tensor,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """U-face average of ``v*dxG`` without hFac, factored row form (planetary term)."""
    v_south_right = v_ext[..., :ny, 1 : nx + 2]
    v_south_left = v_ext[..., :ny, 0 : nx + 1]
    v_north_right = v_ext[..., 1 : ny + 1, 1 : nx + 2]
    v_north_left = v_ext[..., 1 : ny + 1, 0 : nx + 1]

    return _coriolis_quarter_sum2(
        _coriolis_scaled_pair_sum(
            dxG_south,
            v_south_right,
            v_south_left,
        ),
        _coriolis_scaled_pair_sum(
            dxG_north,
            v_north_right,
            v_north_left,
        ),
    )


def _u_vbar_hfacz_row(
    v_ext: torch.Tensor,
    hFacZ: torch.Tensor,
    *,
    dxG_south: torch.Tensor,
    dxG_north: torch.Tensor,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """U-face hFacZ-weighted average of ``v*dxG``, factored row form (selectVortScheme=1)."""
    v_south_right = v_ext[..., :ny, 1 : nx + 2]
    v_south_left = v_ext[..., :ny, 0 : nx + 1]
    v_north_right = v_ext[..., 1 : ny + 1, 1 : nx + 2]
    v_north_left = v_ext[..., 1 : ny + 1, 0 : nx + 1]

    hFacZ_south = hFacZ[..., :ny, :]
    hFacZ_north = hFacZ[..., 1 : ny + 1, :]
    return _hfac_pair_weighted_average(
        _coriolis_two_scale_pair_sum(
            hFacZ_south,
            dxG_south,
            v_south_right,
            v_south_left,
        ),
        _coriolis_two_scale_pair_sum(
            hFacZ_north,
            dxG_north,
            v_north_right,
            v_north_left,
        ),
        hFacZ_south,
        hFacZ_north,
    )


def _u_face_coriolis_from_vbar_row(
    v_bar: torch.Tensor,
    corner_value: torch.Tensor,
    recip_dxC_u: torch.Tensor,
    *,
    face_mask: torch.Tensor | None,
    edge_face_couples: bool,
    edge_pair_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Regional U-face result from a row-form vBar, with edge-pair and face masks."""
    ny = recip_dxC_u.shape[-2]
    corner_avg = _u_face_corner_average(corner_value, ny=ny)
    result = _u_face_coriolis_result(corner_avg, v_bar, recip_dxC_u)
    if edge_pair_mask is not None:
        result = _apply_u_edge_pair_mask(result, edge_pair_mask)
    elif not edge_face_couples:
        result = _zero_u_edge_faces(result)
    return _apply_coriolis_face_mask(result, face_mask, binary=False)


def _v_ubar_no_hfac_row(
    u_ext: torch.Tensor,
    *,
    dyG: torch.Tensor,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """V-face average of ``u*dyG`` without hFac on a uniform-y axis (planetary term)."""
    u_left_above = u_ext[..., 1 : ny + 2, :nx]
    u_left_below = u_ext[..., 0 : ny + 1, :nx]
    u_right_above = u_ext[..., 1 : ny + 2, 1 : nx + 1]
    u_right_below = u_ext[..., 0 : ny + 1, 1 : nx + 1]

    return _coriolis_scaled_quarter_sum4(
        dyG,
        u_left_above,
        u_left_below,
        u_right_above,
        u_right_below,
    )


def _v_ubar_hfacz_row(
    u_ext: torch.Tensor,
    hFacZ: torch.Tensor,
    *,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """V-face hFacZ-weighted average of ``u`` on a uniform-y axis (selectVortScheme=1).

    ``dyG`` cancels against ``recip_dyC`` on the uniform axis, so neither enters
    (see :func:`_v_face_coriolis_from_ubar_row_unscaled`).
    """
    u_left_above = u_ext[..., 1 : ny + 2, :nx]
    u_left_below = u_ext[..., 0 : ny + 1, :nx]
    u_right_above = u_ext[..., 1 : ny + 2, 1 : nx + 1]
    u_right_below = u_ext[..., 0 : ny + 1, 1 : nx + 1]

    hFacZ_left = hFacZ[..., :, :nx]
    hFacZ_right = hFacZ[..., :, 1 : nx + 1]
    return _hfac_pair_weighted_average(
        _coriolis_scaled_pair_sum(
            hFacZ_left,
            u_left_above,
            u_left_below,
        ),
        _coriolis_scaled_pair_sum(
            hFacZ_right,
            u_right_above,
            u_right_below,
        ),
        hFacZ_left,
        hFacZ_right,
    )


def _v_face_coriolis_from_ubar_row(
    u_bar: torch.Tensor,
    corner_value: torch.Tensor,
    recip_dyC_v: torch.Tensor,
    *,
    face_mask: torch.Tensor | None,
    edge_face_couples: bool,
    edge_pair_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Regional V-face result ``-corner average * uBar * recip_dyC`` with edge and face masks."""
    nx = corner_value.shape[-1] - 1
    corner_avg = _v_face_corner_average(corner_value, nx=nx)
    result = _v_face_coriolis_result(corner_avg, u_bar, recip_dyC_v)
    return _v_face_row_masks(
        result,
        face_mask=face_mask,
        edge_face_couples=edge_face_couples,
        edge_pair_mask=edge_pair_mask,
    )


def _v_face_coriolis_from_ubar_row_unscaled(
    u_bar: torch.Tensor,
    corner_value: torch.Tensor,
    *,
    face_mask: torch.Tensor | None,
    edge_face_couples: bool,
    edge_pair_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Regional V-face result ``-corner average * uBar`` (uBar already per unit dy)."""
    nx = corner_value.shape[-1] - 1
    corner_avg = _v_face_corner_average(corner_value, nx=nx)
    result = -corner_avg * u_bar
    return _v_face_row_masks(
        result,
        face_mask=face_mask,
        edge_face_couples=edge_face_couples,
        edge_pair_mask=edge_pair_mask,
    )


def _v_face_row_masks(
    result: torch.Tensor,
    *,
    face_mask: torch.Tensor | None,
    edge_face_couples: bool,
    edge_pair_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Edge-pair (or edge-zero) and face masks of a regional V-face result."""
    if edge_pair_mask is not None:
        result = _apply_v_edge_pair_mask(result, edge_pair_mask)
    elif not edge_face_couples:
        result = _zero_v_edge_faces(result)
    return _apply_coriolis_face_mask(result, face_mask, binary=False)


def _u_face_corner_average(corner_value: torch.Tensor, *, ny: int) -> torch.Tensor:
    """Average corner values to U faces along the j direction."""
    return 0.5 * (corner_value[..., :ny, :] + corner_value[..., 1 : ny + 1, :])


def _u_face_coriolis_result(
    corner_avg: torch.Tensor,
    velocity_bar: torch.Tensor,
    recip_metric: torch.Tensor,
) -> torch.Tensor:
    """Assemble the positive U-face Coriolis/vorticity result."""
    return corner_avg * velocity_bar * recip_metric


def _v_face_coriolis_result(
    corner_avg: torch.Tensor,
    velocity_bar: torch.Tensor,
    recip_metric: torch.Tensor,
) -> torch.Tensor:
    """Assemble the negative V-face Coriolis/vorticity result."""
    result = -corner_avg * velocity_bar
    result = result * recip_metric
    return result


def _v_face_corner_average(corner_value: torch.Tensor, *, nx: int) -> torch.Tensor:
    """Average corner values to V faces along the i direction."""
    return 0.5 * (corner_value[..., :, :nx] + corner_value[..., :, 1 : nx + 1])


def _apply_coriolis_face_mask(
    result: torch.Tensor,
    face_mask: torch.Tensor | None,
    *,
    binary: bool,
) -> torch.Tensor:
    """Apply a Coriolis/vorticity face mask preserving caller semantics."""
    if face_mask is None:
        return result
    if binary:
        return _apply_positive_face_mask(result, face_mask)
    return _apply_face_mask(result, face_mask)


def _zero_u_edge_faces(result: torch.Tensor) -> torch.Tensor:
    """Zero west/east edge U-face values."""
    result[..., 0] = torch.zeros_like(result[..., 0])
    result[..., -1] = torch.zeros_like(result[..., -1])
    return result


def _zero_v_edge_faces(result: torch.Tensor) -> torch.Tensor:
    """Zero south/north edge V-face values."""
    result[..., 0, :] = torch.zeros_like(result[..., 0, :])
    result[..., -1, :] = torch.zeros_like(result[..., -1, :])
    return result


def _apply_u_edge_pair_mask(
    result: torch.Tensor,
    edge_pair_mask: torch.Tensor,
) -> torch.Tensor:
    view = edge_pair_mask.view(*([1] * (result.ndim - 1)), edge_pair_mask.numel())
    return torch.where(view, result, torch.zeros_like(result))


def _apply_v_edge_pair_mask(
    result: torch.Tensor,
    edge_pair_mask: torch.Tensor,
) -> torch.Tensor:
    view = edge_pair_mask.view(*([1] * (result.ndim - 2)), edge_pair_mask.numel(), 1)
    return torch.where(view, result, torch.zeros_like(result))


def hfac_face_transports(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    dyg: torch.Tensor,
    dxg: torch.Tensor,
    hfac_w: torch.Tensor,
    hfac_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """U/V face transports ``u*dyG*hFacW`` and ``v*dxG*hFacS``, exchanged by the caller."""
    u_transport = u * dyg * hfac_w
    v_transport = v * dxg * hfac_s
    return u_transport, v_transport


def _u_face_transport_pair_sum(v_ext: torch.Tensor) -> torch.Tensor:
    """``(T(i,j)+T(i-1,j)) + (T(i,j+1)+T(i-1,j+1))`` of an x-haloed V-face field."""
    return (v_ext[..., :-1, 1:] + v_ext[..., :-1, :-1]) + (
        v_ext[..., 1:, 1:] + v_ext[..., 1:, :-1]
    )


def _v_face_transport_pair_sum(u_ext: torch.Tensor) -> torch.Tensor:
    """``(T(i,j)+T(i,j-1)) + (T(i+1,j)+T(i+1,j-1))`` of a y-haloed U-face field."""
    return (u_ext[..., 1:, :-1] + u_ext[..., :-1, :-1]) + (
        u_ext[..., 1:, 1:] + u_ext[..., :-1, 1:]
    )


def coriolis_wet_point_from_haloed_transports(
    u_ext: torch.Tensor,
    v_ext: torch.Tensor,
    f_g: torch.Tensor,
    *,
    hfac_w_ext: torch.Tensor,
    hfac_s_ext: torch.Tensor,
    dxc: torch.Tensor,
    dyc: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT selectCoriScheme=1, the partial-cell wet-point (Jamart) average.

    mom_vi_coriolis.F:73-89 and :140-156.  The four neighbouring face
    transports are divided by the sum of their wet fractions,
    ``sum(T) / max(1e-9, sum(hFac))``, so a face next to land averages only its
    wet neighbours.  ``hfac_s_ext`` is ``hFacS`` with the x halo of ``v_ext``,
    ``hfac_w_ext`` is ``hFacW`` with the y halo of ``u_ext``; ``f_g`` is
    ``fCoriG`` at corners.  The result is masked by ``maskW`` / ``maskS``.
    """
    v_bar = _u_face_transport_pair_sum(v_ext) / _hfac_pair_denominator(
        hfac_s_ext[..., :-1, 1:] + hfac_s_ext[..., :-1, :-1],
        hfac_s_ext[..., 1:, 1:] + hfac_s_ext[..., 1:, :-1],
    )
    f_u = _u_face_corner_average(f_g, ny=f_g.shape[-2] - 1)
    tend_u = f_u * v_bar * _positive_reciprocal(dxc) * mask_w

    u_bar = _v_face_transport_pair_sum(u_ext) / _hfac_pair_denominator(
        hfac_w_ext[..., 1:, :-1] + hfac_w_ext[..., :-1, :-1],
        hfac_w_ext[..., 1:, 1:] + hfac_w_ext[..., :-1, 1:],
    )
    f_v = _v_face_corner_average(f_g, nx=f_g.shape[-1] - 1)
    tend_v = -f_v * u_bar * _positive_reciprocal(dyc) * mask_s
    return tend_u, tend_v


def coriolis_hfac_weighted_from_haloed_transports(
    u_ext: torch.Tensor,
    v_ext: torch.Tensor,
    f_g: torch.Tensor,
    *,
    dxc: torch.Tensor,
    dyc: torch.Tensor,
    hfac_w: torch.Tensor,
    hfac_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT selectCoriScheme=2, the hFac-weighted average.

    mom_vi_coriolis.F:90-104 and :157-171: a quarter of the four neighbouring
    face transports, times the face-averaged ``fCoriG``, divided by
    ``dxC*hFacW`` (``dyC*hFacS``) through positive-only reciprocals.
    """
    v_bar = 0.25 * _u_face_transport_pair_sum(v_ext)
    f_u = _u_face_corner_average(f_g, ny=f_g.shape[-2] - 1)
    tend_u = f_u * v_bar * _positive_reciprocal(dxc) * _positive_reciprocal(hfac_w)

    u_bar = 0.25 * _v_face_transport_pair_sum(u_ext)
    f_v = _v_face_corner_average(f_g, nx=f_g.shape[-1] - 1)
    tend_v = -f_v * u_bar * _positive_reciprocal(dyc) * _positive_reciprocal(hfac_s)
    return tend_u, tend_v


def energy_conserving_from_haloed_transports(
    u_ext: torch.Tensor,
    v_ext: torch.Tensor,
    omega3: torch.Tensor,
    *,
    dxc: torch.Tensor,
    dyc: torch.Tensor,
    hfac_w: torch.Tensor,
    hfac_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy-conserving corner flux from haloed face transports.

    Each corner's face transport is multiplied by that corner's value before
    averaging to the output face: MIT selectVortScheme=4 with the vorticity
    (mom_vi_u/v_coriolis.F:151-168) and selectCoriScheme=3 with ``fCoriG``
    (mom_vi_coriolis.F:105-120, :172-187) are the same expression.
    """
    v_bar_south, v_bar_north = _half_pair_sums_u(v_ext)
    vort_u = (
        v_bar_south * omega3[..., :-1, :]
        + v_bar_north * omega3[..., 1:, :]
    ) * 0.5
    tend_u = vort_u * _positive_reciprocal(dxc) * _positive_reciprocal(hfac_w)

    u_bar_west, u_bar_east = _half_pair_sums_v(u_ext)
    vort_v = (
        u_bar_west * omega3[..., :, :-1]
        + u_bar_east * omega3[..., :, 1:]
    ) * 0.5
    tend_v = -vort_v * _positive_reciprocal(dyc) * _positive_reciprocal(hfac_s)
    return tend_u, tend_v


def corner_vorticity_over_hfac(omega3: torch.Tensor, hfac_z: torch.Tensor) -> torch.Tensor:
    """``vort3r = r_hFacZ*omega3`` at corners; ``r_hFacZ`` is zero where ``hFacZ`` is zero."""
    return _positive_reciprocal(hfac_z) * omega3


def _half_pair_sums_u(v_ext: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``vBarXm``, ``vBarXp``: half-sums of the two V transports south and north of a U face."""
    v_bar_south = 0.5 * (v_ext[..., :-1, 1:] + v_ext[..., :-1, :-1])
    v_bar_north = 0.5 * (v_ext[..., 1:, 1:] + v_ext[..., 1:, :-1])
    return v_bar_south, v_bar_north


def _half_pair_sums_v(u_ext: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``uBarYm``, ``uBarYp``: half-sums of the two U transports west and east of a V face."""
    u_bar_west = 0.5 * (u_ext[..., 1:, :-1] + u_ext[..., :-1, :-1])
    u_bar_east = 0.5 * (u_ext[..., 1:, 1:] + u_ext[..., :-1, 1:])
    return u_bar_west, u_bar_east


def enstrophy_from_haloed_transports(
    u_ext: torch.Tensor,
    v_ext: torch.Tensor,
    vort3r: torch.Tensor,
    *,
    dxc: torch.Tensor,
    dyc: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT selectVortScheme=0, the enstrophy-conserving corner flux.

    mom_vi_u_coriolis.F:57-73 and mom_vi_v_coriolis.F:57-73: a quarter of the
    four neighbouring face transports times the face average of ``vort3r``
    (``r_hFacZ*omega3``), times ``recip_dxC*maskW`` (``recip_dyC*maskS``).
    ``u_ext``/``v_ext`` are the haloed transports ``u*dyG*hFacW``/``v*dxG*hFacS``.
    """
    v_bar = 0.25 * _u_face_transport_pair_sum(v_ext)
    vort_u = 0.5 * (vort3r[..., :-1, :] + vort3r[..., 1:, :])
    tend_u = vort_u * v_bar * _positive_reciprocal(dxc) * mask_w

    u_bar = 0.25 * _v_face_transport_pair_sum(u_ext)
    vort_v = 0.5 * (vort3r[..., :, :-1] + vort3r[..., :, 1:])
    tend_v = -vort_v * u_bar * _positive_reciprocal(dyc) * mask_s
    return tend_u, tend_v


def energy_from_haloed_transports(
    u_ext: torch.Tensor,
    v_ext: torch.Tensor,
    omega3: torch.Tensor,
    hfac_z: torch.Tensor,
    *,
    dxc: torch.Tensor,
    dyc: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT selectVortScheme=2, the energy-conserving corner flux with ``r_hFacZ``.

    mom_vi_u_coriolis.F:92-109 and mom_vi_v_coriolis.F:92-109: each half-sum of
    face transports is multiplied by ``r_hFacZ`` and then ``omega3`` of its
    corner, left to right, times ``recip_dxC*maskW`` (``recip_dyC*maskS``).
    """
    r_hfac_z = _positive_reciprocal(hfac_z)
    v_bar_south, v_bar_north = _half_pair_sums_u(v_ext)
    vort_u = (
        v_bar_south * r_hfac_z[..., :-1, :] * omega3[..., :-1, :]
        + v_bar_north * r_hfac_z[..., 1:, :] * omega3[..., 1:, :]
    ) * 0.5
    tend_u = vort_u * _positive_reciprocal(dxc) * mask_w

    u_bar_west, u_bar_east = _half_pair_sums_v(u_ext)
    vort_v = (
        u_bar_west * r_hfac_z[..., :, :-1] * omega3[..., :, :-1]
        + u_bar_east * r_hfac_z[..., :, 1:] * omega3[..., :, 1:]
    ) * 0.5
    tend_v = -vort_v * _positive_reciprocal(dyc) * mask_s
    return tend_u, tend_v


_ONE_THIRD = 1.0 / 3.0
_ONE_SIXTH = 1.0 / 6.0
_ONE_TWELFTH = 1.0 / 12.0


def energy_enstrophy_from_haloed_factors(
    v_x: torch.Tensor,
    dxg_x: torch.Tensor,
    hfac_s_x: torch.Tensor,
    u_y: torch.Tensor,
    dyg_y: torch.Tensor,
    hfac_w_y: torch.Tensor,
    vort3r_x: torch.Tensor,
    vort3r_y: torch.Tensor,
    *,
    dxc: torch.Tensor,
    dyc: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT selectVortScheme=3, the energy- and enstrophy-conserving corner flux.

    mom_vi_u_coriolis.F:111-149 and mom_vi_v_coriolis.F:111-149: each of the
    four neighbouring face transports takes the mean of ``vort3r`` over three
    corners, ``(q0 + (q1 + q2))*oneThird``, multiplied in MITgcm's order by the
    velocity, the face length and the face hFac one at a time.  ``v_x``,
    ``dxg_x``, ``hfac_s_x`` are the V-face factors with one x-halo column on
    each side ``[..., ny+1, nx+2]``; ``u_y``, ``dyg_y``, ``hfac_w_y`` the U-face
    factors with one y-halo row ``[..., ny+2, nx+1]``.  ``vort3r_x`` is
    ``vort3r`` with one x-halo column on each side ``[..., ny+1, nx+3]`` (U
    term), ``vort3r_y`` with one y-halo row ``[..., ny+3, nx+1]`` (V term).
    """
    q = vort3r_x[..., :, 1:-1]
    q_w = vort3r_x[..., :, :-2]
    q_e = vort3r_x[..., :, 2:]
    q_s, q_n = q[..., :-1, :], q[..., 1:, :]

    def v_face(mean, rows, cols):
        return mean * v_x[..., rows, cols] * dxg_x[..., rows, cols] * hfac_s_x[..., rows, cols]

    south, north, west, east = slice(None, -1), slice(1, None), slice(None, -1), slice(1, None)
    vort_mj = v_face((q_s + (q_n + q_w[..., :-1, :])) * _ONE_THIRD, south, west)
    vort_ij = v_face((q_s + (q_n + q_e[..., :-1, :])) * _ONE_THIRD, south, east)
    vort_mp = v_face((q_n + (q_s + q_w[..., 1:, :])) * _ONE_THIRD, north, west)
    vort_ip = v_face((q_n + (q_s + q_e[..., 1:, :])) * _ONE_THIRD, north, east)
    tend_u = (
        ((vort_mj + vort_ij) + (vort_mp + vort_ip)) * 0.25
        * _positive_reciprocal(dxc) * mask_w
    )

    q = vort3r_y[..., 1:-1, :]
    q_s = vort3r_y[..., :-2, :]
    q_n = vort3r_y[..., 2:, :]
    q_w, q_e = q[..., :, :-1], q[..., :, 1:]

    def u_face(mean, rows, cols):
        return mean * u_y[..., rows, cols] * dyg_y[..., rows, cols] * hfac_w_y[..., rows, cols]

    vort_im = u_face((q_w + (q_e + q_s[..., :, :-1])) * _ONE_THIRD, south, west)
    vort_ij = u_face((q_w + (q_e + q_n[..., :, :-1])) * _ONE_THIRD, north, west)
    vort_pm = u_face((q_e + (q_w + q_s[..., :, 1:])) * _ONE_THIRD, south, east)
    vort_pj = u_face((q_e + (q_w + q_n[..., :, 1:])) * _ONE_THIRD, north, east)
    tend_v = (
        -((vort_im + vort_ij) + (vort_pm + vort_pj)) * 0.25
        * _positive_reciprocal(dyc) * mask_s
    )
    return tend_u, tend_v


def _fourth_order_stencils(
    vort3r_y: torch.Tensor,
    vort3r_x: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """The four corners along the advection direction of each face.

    U face: corners ``j-1, j, j+1, j+2`` from the y-haloed ``[..., ny+3, nx+1]``;
    V face: corners ``i-1, i, i+1, i+2`` from the x-haloed ``[..., ny+1, nx+3]``.
    """
    ny = vort3r_y.shape[-2] - 3
    nx = vort3r_x.shape[-1] - 3
    along_u = tuple(vort3r_y[..., o : o + ny, :] for o in range(4))
    along_v = tuple(vort3r_x[..., :, o : o + nx] for o in range(4))
    return along_u, along_v


def enstrophy_fourth_order_from_haloed_transports(
    u_ext: torch.Tensor,
    v_ext: torch.Tensor,
    vort3r_x: torch.Tensor,
    vort3r_y: torch.Tensor,
    *,
    dxc: torch.Tensor,
    dyc: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT highOrderVorticity with selectVortScheme=0 (mom_vi_u/v_coriolis_c4.F:120-159).

    The face vorticity is the fourth-order interpolation
    ``0.5*((q0+q1) - oneTwelve*(Rjp-Rjm))`` of ``vort3r`` along the advection
    direction, ``Rjp = q2-q1``, ``Rjm = q0-q(-1)``.  MITgcm also multiplies
    ``Rjp``/``Rjm`` by ``maskInW``/``maskInS`` under ALLOW_OBCS; without an open
    boundary that factor is zero only next to a dry column, where both corner
    values are already zero, so it is left out; model assembly refuses open
    boundaries (``validate_vorticity_composition``).
    """
    (qm, q0, q1, q2), (pm, p0, p1, p2) = _fourth_order_stencils(vort3r_y, vort3r_x)
    v_bar = 0.25 * _u_face_transport_pair_sum(v_ext)
    vort_u = 0.5 * ((q0 + q1) - _ONE_TWELFTH * ((q2 - q1) - (q0 - qm)))
    tend_u = vort_u * v_bar * _positive_reciprocal(dxc) * mask_w

    u_bar = 0.25 * _v_face_transport_pair_sum(u_ext)
    vort_v = 0.5 * ((p0 + p1) - _ONE_TWELFTH * ((p2 - p1) - (p0 - pm)))
    tend_v = -vort_v * u_bar * _positive_reciprocal(dyc) * mask_s
    return tend_u, tend_v


def energy_fourth_order_from_haloed_transports(
    u_ext: torch.Tensor,
    v_ext: torch.Tensor,
    vort3r_x: torch.Tensor,
    vort3r_y: torch.Tensor,
    *,
    dxc: torch.Tensor,
    dyc: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT highOrderVorticity with selectVortScheme=2 (mom_vi_u/v_coriolis_c4.F:161-203).

    ``Rj = q1-q0``, ``Rjp = q1 - oneSixth*((q2-q1) - Rj)``,
    ``Rjm = q0 - oneSixth*(Rj - (q0-q(-1)))``, face value
    ``0.5*(vBarXm*Rjm + vBarXp*Rjp)``; the ``maskInW``/``maskInS`` factor as in
    the scheme-0 kernel.
    """
    (qm, q0, q1, q2), (pm, p0, p1, p2) = _fourth_order_stencils(vort3r_y, vort3r_x)
    v_bar_south, v_bar_north = _half_pair_sums_u(v_ext)
    rj = q1 - q0
    rjp = q1 - _ONE_SIXTH * ((q2 - q1) - rj)
    rjm = q0 - _ONE_SIXTH * (rj - (q0 - qm))
    tend_u = 0.5 * (v_bar_south * rjm + v_bar_north * rjp) * _positive_reciprocal(dxc) * mask_w

    u_bar_west, u_bar_east = _half_pair_sums_v(u_ext)
    ri = p1 - p0
    rip = p1 - _ONE_SIXTH * ((p2 - p1) - ri)
    rim = p0 - _ONE_SIXTH * (ri - (p0 - pm))
    tend_v = -(0.5 * (u_bar_west * rim + u_bar_east * rip)) * _positive_reciprocal(dyc) * mask_s
    return tend_u, tend_v


def enstrophy_upwind_from_haloed_transports(
    u_ext: torch.Tensor,
    v_ext: torch.Tensor,
    vort3r: torch.Tensor,
    *,
    dxc: torch.Tensor,
    dyc: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT upwindVorticity with selectVortScheme=0 (mom_vi_u/v_coriolis_c4.F:132-138).

    The face takes the corner on the upstream side of its mean transport,
    ``j`` when ``vBarXY > 0`` else ``j+1`` (``i`` / ``i+1`` for a V face).
    """
    v_bar = 0.25 * _u_face_transport_pair_sum(v_ext)
    vort_u = torch.where(v_bar > 0.0, vort3r[..., :-1, :], vort3r[..., 1:, :])
    tend_u = vort_u * v_bar * _positive_reciprocal(dxc) * mask_w

    u_bar = 0.25 * _v_face_transport_pair_sum(u_ext)
    vort_v = torch.where(u_bar > 0.0, vort3r[..., :, :-1], vort3r[..., :, 1:])
    tend_v = -vort_v * u_bar * _positive_reciprocal(dyc) * mask_s
    return tend_u, tend_v


def energy_upwind_from_haloed_transports(
    u_ext: torch.Tensor,
    v_ext: torch.Tensor,
    vort3r: torch.Tensor,
    *,
    dxc: torch.Tensor,
    dyc: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT upwindVorticity with selectVortScheme=2 (mom_vi_u/v_coriolis_c4.F:173-179).

    ``vBarXm*q(j)`` when ``vBarXm+vBarXp > 0`` else ``vBarXp*q(j+1)``; MITgcm
    applies no factor 0.5 on this branch.
    """
    v_bar_south, v_bar_north = _half_pair_sums_u(v_ext)
    vort_u = torch.where(
        (v_bar_south + v_bar_north) > 0.0,
        v_bar_south * vort3r[..., :-1, :],
        v_bar_north * vort3r[..., 1:, :],
    )
    tend_u = vort_u * _positive_reciprocal(dxc) * mask_w

    u_bar_west, u_bar_east = _half_pair_sums_v(u_ext)
    vort_v = torch.where(
        (u_bar_west + u_bar_east) > 0.0,
        u_bar_west * vort3r[..., :, :-1],
        u_bar_east * vort3r[..., :, 1:],
    )
    tend_v = -vort_v * _positive_reciprocal(dyc) * mask_s
    return tend_u, tend_v
