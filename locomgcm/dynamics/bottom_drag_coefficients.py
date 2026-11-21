"""Bottom drag coefficient ``cDrag`` kernels (MITgcm ``mom_u_botdrag_coeff.F``,
``mom_v_botdrag_coeff.F`` with ``selectBotDragQuadr=0``).

``cDrag = bottomDragLinear + no-slip term + bottomDragQuadratic * speed`` in
m s-1, evaluated at the bottom cell of every U or V face column.
``layer_thickness`` is ``drF`` ``[nz]`` in metres; ``k_bottom`` is the bottom
level index ``[..., 1, y, x]`` (``bottom_geometry``).  ``u`` is on U faces
``[..., nz, ny, nx+1]`` (``nx`` faces on a wall-bounded axis) and ``v`` on V
faces; the kinetic energy is at tracer centres ``[..., nz, ny, nx]`` in
m2 s-2.  Arguments named ``*_halo_x`` or ``*_halo_y`` carry the halo the
adapter exchanged.  Results keep the input dtype.
"""

from __future__ import annotations

import torch


def linear_bottom_drag_levels(
    layer_thickness: torch.Tensor,
    *,
    linear: float | torch.Tensor,
) -> torch.Tensor:
    """``bottomDragLinear`` on every level, ``[nz]``."""
    drF = layer_thickness.reshape(-1)
    linear = torch.as_tensor(linear, dtype=drF.dtype, device=drF.device)
    if linear.ndim != 0:
        raise ValueError("bottom drag linear coefficient must be scalar")
    return linear.expand(drF.shape[0])


def no_slip_bottom_drag_levels(
    layer_thickness: torch.Tensor,
    interface_spacing: torch.Tensor,
    *,
    vertical_viscosity: float,
) -> torch.Tensor:
    """No-slip-bottom drag ``viscAr * recip_drC_below * 2`` per level, ``[nz]``.

    Matches the cube-stepper inline form.  ``interface_spacing`` is drC at the
    interior interfaces; the last ``drC_below`` uses ``drF[-1]`` at the grid floor.
    """
    drF = layer_thickness.reshape(-1)
    drC_below = torch.empty_like(drF)
    if drF.shape[0] > 1:
        drC_below[:-1] = interface_spacing
    drC_below[-1] = drF[-1]
    return vertical_viscosity * (1.0 / drC_below) * 2.0


def no_slip_floor_coefficient(
    floor_viscosity: torch.Tensor,
    layer_thickness: torch.Tensor,
    *,
    linear: float | torch.Tensor,
) -> torch.Tensor:
    """``0.0 + (bottomDragLinear + kappaR(Nr+1) * recip_drF(Nr) * 2)`` of a bottom cell on the last level.

    ``mom_u_botdrag_coeff.F:81-85`` takes ``recDrC = recip_drF(Nr)`` at
    ``k = Nr`` and ``:113-128`` adds ``kappaRU(kLowF)*recDrC*viscFac``;
    the operation order of ``linear_bottom_drag_levels`` +
    ``no_slip_bottom_drag_levels`` and ``bottom_level_coefficient``.
    ``floor_viscosity`` is the interface Nr+1 viscosity ``[..., 1, y, x]``.
    """
    drF = layer_thickness.reshape(-1)
    linear = torch.as_tensor(linear, dtype=drF.dtype, device=drF.device)
    return 0.0 + (linear + floor_viscosity * (1.0 / drF[-1]) * 2.0)


def bottom_level_coefficient(c_drag: torch.Tensor, k_bottom: torch.Tensor) -> torch.Tensor:
    """``0.0 + c_drag[k_bottom]`` per column.

    The leading ``0.0 +`` is not decoration: the full-field form built the
    coefficient as ``torch.zeros_like(u) + c_drag``, and ``+0.0 + -0.0`` is
    ``+0.0``, so a ``-0.0`` coefficient must be normalised the same way or the
    sign of the resulting zero flips.
    """
    view = c_drag.view(*([1] * (k_bottom.ndim - 3)), -1, 1, 1)
    return 0.0 + torch.take_along_dim(view, k_bottom, dim=-3)


def non_bottom_zero_sign(level_coefficient: torch.Tensor) -> torch.Tensor:
    """Per-level signed zero reproducing the full-field ``P * 0`` at non-bottom cells.

    The explicit full-field expression wrote ``(-cDrag * u * recip_hFacW *
    recip_drF) * bottom_mask`` everywhere, so an off-bottom cell held a *signed*
    zero whose sign bit is ``signbit(-cDrag) XOR signbit(u)`` (both reciprocals
    are non-negative).  Writing ``u * s[k]`` with ``s[k] = copysign(0, -cDrag[k])``
    reproduces it exactly, including the ``NaN`` a non-finite ``u`` produced.

    ``level_coefficient`` is the per-level linear and no-slip ``cDrag``, or
    zeros when that term is off.  Returns the ``[nz, 1, 1]`` scale.  With
    quadratic drag on, ``cDrag = c_drag[k] + C_d*speed`` and only a
    non-negative ``c_drag[k]`` keeps the sign a level constant;
    :meth:`BottomDrag.require_level_sign_constant` establishes that in the
    eager pre-step, never inside the step.
    """
    base = 0.0 + level_coefficient
    scale = torch.where(
        base >= 0, torch.full_like(base, -0.0), torch.zeros_like(base)
    )
    return scale.view(-1, 1, 1)


def kinetic_energy_center(
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Cell-center KE for ``BottomDrag.ke_scheme == 0`` (MIT KEscheme 0).

    The KE buffer consumed by bot_drag is produced once per (bi,bj,k) by
    ``mom_calc_ke.F`` in the momentum core and passed to
    ``MOM_{U,V}_BOTDRAG_COEFF`` with ``inp_KE=.TRUE.``, so the
    ``IF (.NOT.inp_KE)`` guard at ``mom_u_botdrag_coeff.F:134`` SKIPS the
    hFac-weighted internal recompute at L137-142.  The vector-invariant core
    (``mom_vecinv.F:284,488``) with ``selectKEscheme=0`` makes
    ``mom_calc_ke.F:70-86`` produce the **unweighted** form::

        KE(i,j) = 0.25 * (
              uFld(i, j)²
            + uFld(i+1, j)²
            + vFld(i, j)²
            + vFld(i, j+1)²
          )

    No ``_hFacW``, no ``_hFacS``, no ``_recip_hFacC``. This matches
    ``mom_calc_ke.F:70-86`` line-by-line.

    The hFac-weighted form of ``mom_u_botdrag_coeff.F:137-142`` is not used:
    that block is unreachable on the vec_inv path (``inp_KE=.TRUE.``), so
    the unweighted form matches the runtime path.

    Returns shape ``[B, nz, ny, nx]``.
    """
    # MIT mom_calc_ke.F:70-86 (selectKEscheme=0): unweighted form
    # KE(i,j) = 0.25 * (uFld(i,j)² + uFld(i+1,j)² + vFld(i,j)² + vFld(i,j+1)²)
    u2 = u * u  # [B, nz, ny, nx_u]
    v2 = v * v  # [B, nz, ny_v, nx]

    nx_u = u2.shape[-1]
    # Resolve nx_c from v2 (always center-aligned along x).
    nx_c = v2.shape[-1]
    if nx_u > nx_c:
        # Open boundary: nx+1 u-faces → pair (i, i+1) → nx centers.
        u2_sum = u2[..., :-1] + u2[..., 1:]
    else:
        # Wall boundary: nx u-faces; the (i+1) neighbour is the wall (=0).
        # Doubling the present face keeps the formula symmetric with the
        # face-averaged interior form.
        u2_sum = 2.0 * u2

    ny_v = v2.shape[-2]
    # Resolve ny_c from u2 (always center-aligned along y).
    ny_c = u2.shape[-2]
    if ny_v > ny_c:
        v2_sum = v2[..., :-1, :] + v2[..., 1:, :]
    else:
        v2_sum = 2.0 * v2

    ke = 0.25 * (u2_sum + v2_sum)
    return ke


def u_bottom_speed(
    ke_halo_x: torch.Tensor, k_bottom: torch.Tensor, *, speed_floor: float
) -> torch.Tensor:
    """``sqrt(2 * KE)`` at the bottom U face, KE averaged as ``0.5*(KE(i-1)+KE(i))``."""
    ke_on_u = 0.5 * (
        torch.take_along_dim(ke_halo_x[..., :-1], k_bottom, dim=-3)
        + torch.take_along_dim(ke_halo_x[..., 1:], k_bottom, dim=-3)
    )
    return torch.sqrt((2.0 * ke_on_u).clamp(min=speed_floor))  # sqrt(2*KE) = |v|


def v_bottom_speed(
    ke_halo_y: torch.Tensor, k_bottom: torch.Tensor, *, speed_floor: float
) -> torch.Tensor:
    """``sqrt(2 * KE)`` at the bottom V face, KE averaged as ``0.5*(KE(j-1)+KE(j))``."""
    ke_on_v = 0.5 * (
        torch.take_along_dim(ke_halo_y[..., :-1, :], k_bottom, dim=-3)
        + torch.take_along_dim(ke_halo_y[..., 1:, :], k_bottom, dim=-3)
    )
    return torch.sqrt((2.0 * ke_on_v).clamp(min=speed_floor))


def quadratic_bottom_drag(speed: torch.Tensor, *, quadratic: float) -> torch.Tensor:
    """``bottomDragQuadratic * speed``, the quadratic part of ``cDrag``."""
    return quadratic * speed
