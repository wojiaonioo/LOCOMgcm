from __future__ import annotations

from dataclasses import dataclass

import torch

from defgcm.fields import Field, FieldLocation
from defgcm.geometry import StructuredCGrid
from defgcm.geometry.grid import _compute_interior_mask
from defgcm.operators import (
    divergence_transport_to_center,
    face_volume_transport,
    horizontal_volume_divergence_from_transports,
    vertical_velocity_continuity,
)
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyPhase, TendencyStepDomain
from defgcm.dynamics._registry import register_dynamics_module


def upward_vertical_volume_flux_from_horizontal_divergence(
    horizontal_volume_divergence: torch.Tensor,
) -> torch.Tensor:
    """MIT upward-positive interface volume flux from horizontal divergence.

    ``horizontal_volume_divergence`` is the per-layer horizontal volume
    transport divergence on tracer cells.  The bottom boundary is zero and the
    recurrence is MIT ``INTEGRATE_FOR_W`` / ``INTEGR_CONTINUITY``:
    ``wflux[k] = wflux[k+1] - div_h[k]``.
    """
    if horizontal_volume_divergence.ndim < 3:
        raise ValueError(
            "horizontal_volume_divergence must have at least [nz, ny, nx] dimensions"
        )
    nz = horizontal_volume_divergence.shape[-3]
    ny = horizontal_volume_divergence.shape[-2]
    nx = horizontal_volume_divergence.shape[-1]
    wflux = horizontal_volume_divergence.new_zeros(
        (*horizontal_volume_divergence.shape[:-3], nz + 1, ny, nx)
    )
    for k in range(nz - 1, -1, -1):
        wflux[..., k, :, :] = (
            wflux[..., k + 1, :, :] - horizontal_volume_divergence[..., k, :, :]
        )
    return wflux


def upward_vertical_velocity_from_horizontal_divergence(
    horizontal_volume_divergence: torch.Tensor,
    *,
    cell_area: torch.Tensor,
) -> torch.Tensor:
    """MIT upward-positive interface velocity from horizontal volume divergence."""
    wflux = upward_vertical_volume_flux_from_horizontal_divergence(
        horizontal_volume_divergence
    )
    area = cell_area.to(dtype=wflux.dtype, device=wflux.device)
    if area.ndim == wflux.ndim - 1:
        area = area.unsqueeze(-3)
    elif area.ndim == 2:
        lead = max(wflux.ndim - 3, 0)
        area = area.view(*([1] * lead), 1, area.shape[-2], area.shape[-1])
    return wflux / area


def diagnose_vertical_velocity(
    state: PrognosticState,
    grid: StructuredCGrid,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = state.u.data.dtype
    device = state.u.data.device
    layer_thickness = grid.layer_thickness.to(dtype=dtype, device=device).view(1, grid.nz, 1, 1)

    # Build MIT-style per-layer volume transports:
    # uTrans = u * drF * hFacW * dyG, vTrans = v * drF * hFacS * dxG.
    dyG = grid.metrics.dyG_for(dtype, device).view(1, 1, grid.ny, grid.nx + 1)
    dxG = grid.metrics.dxG_for(dtype, device).view(1, 1, grid.ny + 1, grid.nx)
    hFacW = None
    if grid.masks.hFacW is not None:
        hFacW = grid.masks.hFacW.to(dtype=dtype, device=device)
        if hFacW.ndim == 3:
            hFacW = hFacW.unsqueeze(0)
    hFacS = None
    if grid.masks.hFacS is not None:
        hFacS = grid.masks.hFacS.to(dtype=dtype, device=device)
        if hFacS.ndim == 3:
            hFacS = hFacS.unsqueeze(0)
    u_transport_3d, v_transport_3d = face_volume_transport(
        state.u.data,
        state.v.data,
        layer_thickness,
        dyG,
        dxG,
        hFacW=hFacW,
        hFacS=hFacS,
    )

    # Velocity-based horizontal divergence for diagnostics (unchanged).
    horizontal_divergence = divergence_transport_to_center(state.u.data, state.v.data, grid)

    # Interface w via the geometry-neutral continuity core
    # (operators.vertical_velocity_continuity), shared with the cubed-sphere
    # momentum / tracer w producers.  form='velocity' divides the raw volume
    # divergence by rA per layer (MIT-faithful conv2d*recip_rA ordering);
    # convention='downward' returns LOCOMgcm downward-positive w (positive=down).
    # The rA view and center wet-mask multiply preserve the volume-divergence
    # convention. The OBCS maskInC factor follows MITgcm
    # integr_continuity.F:288-296. bottom_up integrates from
    # the seafloor (w[nz]=0) up, matching free-surface INTEGR_CONTINUITY (surface
    # w carries the free-surface tendency).
    hdiv_volume = horizontal_volume_divergence_from_transports(
        u_transport_3d, v_transport_3d,
    )
    rA = grid.metrics.cell_area_for(dtype, device)
    lead = max(u_transport_3d.ndim - 2, 0)
    cell_area = rA.view(*([1] * lead), grid.ny, grid.nx)
    wet_mask = None
    if grid.masks.has_3d_masks and u_transport_3d.ndim >= 4:
        wet_mask = grid.masks.center_float(dtype=dtype, device=device).unsqueeze(0)
    obcs_mask = None
    if grid.boundary_topology.open_x or grid.boundary_topology.open_y:
        maskInC, _, _ = _compute_interior_mask(grid, dtype, device)  # [ny, nx]
        obcs_mask = maskInC.unsqueeze(0).unsqueeze(0)  # [1, 1, ny, nx]
    if grid.masks.vertical_coordinate == "r_star":
        # INTEGRATE_FOR_W r* branch: the column stretching rStarDhDt of the
        # continuity call these velocities came from; maskInC as
        # integr_continuity.F:288-296 applies it after each level.
        from defgcm.steppers.ifs.r_star import r_star_vertical_velocity

        vertical_velocity_interface = r_star_vertical_velocity(
            hdiv_volume,
            cell_area=cell_area,
            wet_mask=wet_mask,
            thickness_rate=grid.masks.r_star_thickness_rate.to(dtype=dtype, device=device),
            h0_fac_c=grid.masks.resting_hfac[0].to(dtype=dtype, device=device),
            dr_f=grid.layer_thickness_for(dtype, device),
        )
        if obcs_mask is not None:
            vertical_velocity_interface = vertical_velocity_interface * obcs_mask
    else:
        vertical_velocity_interface = vertical_velocity_continuity(
            hdiv_volume,
            cell_area=cell_area,
            wet_mask=wet_mask,
            obcs_mask=obcs_mask,
            form="velocity",
            convention="downward",
        )

    vertical_velocity_center = 0.5 * (
        vertical_velocity_interface[:, :-1, :, :]
        + vertical_velocity_interface[:, 1:, :, :]
    )
    return horizontal_divergence, vertical_velocity_interface, vertical_velocity_center


@dataclass(frozen=True)
@register_dynamics_module("diagnostic_vertical_velocity")
class DiagnosticVerticalVelocity(TendencyModule):
    """Diagnose interface ``w`` from horizontal transport divergence.
    """

    name: str = "diagnostic_vertical_velocity"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER

    def populate_diagnostic_state(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> None:
        horizontal_divergence, vertical_velocity_interface, vertical_velocity_center = diagnose_vertical_velocity(state, grid)
        diagnostics.set_field(
            Field(
                "horizontal_divergence",
                horizontal_divergence,
                FieldLocation.CENTER,
                "s-1",
                mask_ref="center",
            )
        )
        diagnostics.set_field(
            Field(
                "vertical_velocity_interface",
                vertical_velocity_interface,
                FieldLocation.W_FACE,
                "m s-1",
                mask_ref="center",
                metadata={"positive": "down"},
            )
        )
        diagnostics.set_field(
            Field(
                "vertical_velocity_center",
                vertical_velocity_center,
                FieldLocation.CENTER,
                "m s-1",
                mask_ref="center",
                metadata={"positive": "down"},
            )
        )

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        if "vertical_velocity_interface" not in diagnostics.fields:
            self.populate_diagnostic_state(state, diagnostics, grid)
        return StateTendency.zeros_like(state)
