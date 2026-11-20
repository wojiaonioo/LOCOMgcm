"""Bottom drag tendency kernels: apply ``cDrag`` at the bottom cell of every
U or V face column (MITgcm ``mom_fluxform.F`` / ``mom_vecinv.F``:
``gUdiss -= cDrag * uFld * recip_hFacW * recip_drF``).

``velocity`` is ``u`` on U faces or ``v`` on V faces, ``[..., nz, y, x]`` in
m s-1.  ``k_bottom`` is the bottom level index ``[..., 1, y, x]`` and
``has_bottom`` is 1 for a column with a wet cell, 0 for a dry one
(``bottom_geometry``).  ``c_drag`` is the bottom-cell coefficient in m s-1,
``recip_hfac`` and ``recip_dr_f`` the bottom-cell reciprocals.  Each kernel
returns the off-bottom fill ``[..., nz, y, x]`` and the two-dimensional bottom
value ``[..., 1, y, x]`` in m s-2; ``assemble_bottom_tendency`` writes the
bottom value into the fill.  Results keep the velocity dtype.
"""

from __future__ import annotations

import torch


def explicit_bottom_drag(
    velocity: torch.Tensor,
    *,
    k_bottom: torch.Tensor,
    c_drag: torch.Tensor | float,
    recip_hfac: torch.Tensor | float,
    recip_dr_f: torch.Tensor,
    has_bottom: torch.Tensor,
    zero_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``-cDrag * u * recip_hFac * recip_drF`` at the bottom, ``u * zero_scale`` elsewhere."""
    velocity_bottom = torch.take_along_dim(velocity, k_bottom, dim=-3)
    # Explicit path keeps the exact pre-implicit-option expression
    # (operand order included) so default runs stay bitwise identical.
    bottom = -c_drag * velocity_bottom * recip_hfac * recip_dr_f * has_bottom
    return velocity * zero_scale, bottom


def implicit_bottom_drag(
    velocity: torch.Tensor,
    *,
    k_bottom: torch.Tensor,
    c_drag: torch.Tensor | float,
    recip_hfac: torch.Tensor | float,
    recip_dr_f: torch.Tensor,
    has_bottom: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-Euler equivalent tendency ``(u / (1 + dt * rate) - u) / dt`` of the drag rate."""
    velocity_bottom = torch.take_along_dim(velocity, k_bottom, dim=-3)
    drag_rate = c_drag * recip_hfac * recip_dr_f * has_bottom
    bottom = (velocity_bottom / (1.0 + dt * drag_rate) - velocity_bottom) / dt
    # Off-bottom the whole-field form evaluated (u/1 - u)/dt, i.e. +0.0
    # for finite u and NaN for a non-finite one -- exactly ``u - u``.
    return velocity - velocity, bottom


def split_no_slip_increment(
    velocity: torch.Tensor,
    total_vertical_viscosity: torch.Tensor,
    *,
    k_bottom: torch.Tensor,
    k_interface: torch.Tensor,
    bottom_is_internal: torch.Tensor,
    background: float,
    recip_dr_c: torch.Tensor,
    recip_hfac: torch.Tensor | float,
    recip_dr_f: torch.Tensor,
    has_bottom: torch.Tensor,
) -> torch.Tensor:
    """Bottom value of the no-slip drag carried by viscosity above ``background``.

    ``total_vertical_viscosity`` is the background+KPP interface viscosity on
    the same faces; a bottom at ``k < nz-1`` reads it at ``k_interface = k+1``.
    """
    viscosity_below = torch.take_along_dim(
        total_vertical_viscosity, k_interface, dim=-3
    )
    internal_delta = 2.0 * (viscosity_below - background) * recip_dr_c
    # A bottom on the global floor keeps the zeros_like accumulator's +0.0.
    delta_cdrag = torch.where(
        bottom_is_internal, internal_delta, torch.zeros_like(internal_delta)
    )
    return (
        -delta_cdrag
        * torch.take_along_dim(velocity, k_bottom, dim=-3)
        * recip_hfac
        * recip_dr_f
        * has_bottom
    )


def assemble_bottom_tendency(
    fill: torch.Tensor, bottom: torch.Tensor, *, k_bottom: torch.Tensor
) -> torch.Tensor:
    """Write ``bottom`` into ``fill`` at ``k_bottom``, in place (``fill`` is fresh)."""
    return fill.scatter_(-3, k_bottom.expand(bottom.shape), bottom)
