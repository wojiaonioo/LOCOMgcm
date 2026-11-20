from __future__ import annotations

from dataclasses import dataclass

import torch

from defgcm.boundaries.open import BoundarySide
from defgcm.geometry import StructuredCGrid
from defgcm.geometry.grid import _side_is_open
from defgcm.operators.c_grid import synchronize_regional_face_values
from defgcm.operators import (
    depth_integrated_face_transport,
    divergence_volume_transport_to_center_mit,
    gradient_eta_to_u,
    gradient_eta_to_v,
)
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyStepDomain
from defgcm.dynamics._registry import register_dynamics_module


def _compute_maskInC(grid: StructuredCGrid, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    """Compute MITgcm-style maskInC: interior mask excluding OBC boundary strips.

    Returns None when the grid has no open boundaries (no masking needed).
    MITgcm reference: ``integr_continuity.F:196`` — ``dEtaHdt *= maskInC``.
    """
    if not grid.boundary_topology.has_open_boundaries:
        return None
    if grid.masks.has_3d_masks:
        mask = grid.masks.center_float(dtype=dtype, device=device)[0].clone()
    else:
        mask = torch.ones(grid.ny, grid.nx, dtype=dtype, device=device)
    if _side_is_open(BoundarySide.WEST.value, grid):
        mask[:, 0] = 0.0
    if _side_is_open(BoundarySide.EAST.value, grid):
        mask[:, -1] = 0.0
    if _side_is_open(BoundarySide.SOUTH.value, grid):
        mask[0, :] = 0.0
    if _side_is_open(BoundarySide.NORTH.value, grid):
        mask[-1, :] = 0.0
    return mask


@dataclass(frozen=True)
@register_dynamics_module("linear_free_surface_continuity")
class LinearFreeSurfaceContinuity(TendencyModule):
    """Linear free-surface continuity tendency
    ``∂η/∂t = -∇·∫ u dz`` using MIT integrated volume transport.
    """

    name: str = "continuity"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.FAST

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        layer_thickness = grid.layer_thickness.to(
            dtype=state.u.data.dtype,
            device=state.u.data.device,
        ).view(1, grid.nz, 1, 1)

        # MIT-style volume transport:
        # U = u * dyG * dz * hFacW, V = v * dxG * dz * hFacS.
        dyG = grid.metrics.dyG_for(
            state.u.data.dtype, state.u.data.device,
        ).view(1, 1, grid.ny, grid.nx + 1)
        dxG = grid.metrics.dxG_for(
            state.v.data.dtype,
            state.v.data.device,
        ).view(1, 1, grid.ny + 1, grid.nx)
        hFacW = None
        if grid.masks.hFacW is not None:
            hFacW = grid.masks.hFacW.to(
                dtype=state.u.data.dtype, device=state.u.data.device,
            )
            if hFacW.ndim == 3:
                hFacW = hFacW.unsqueeze(0)
        hFacS = None
        if grid.masks.hFacS is not None:
            hFacS = grid.masks.hFacS.to(
                dtype=state.v.data.dtype, device=state.v.data.device,
            )
            if hFacS.ndim == 3:
                hFacS = hFacS.unsqueeze(0)
        u_transport, v_transport = depth_integrated_face_transport(
            state.u.data,
            state.v.data,
            layer_thickness,
            dyG,
            dxG,
            hFacW=hFacW,
            hFacS=hFacS,
        )

        u_transport, v_transport = synchronize_regional_face_values(
            u_transport, v_transport, grid,
        )
        eta_tendency = -divergence_volume_transport_to_center_mit(
            u_transport,
            v_transport,
            grid,
        )

        # MITgcm integr_continuity.F:196 — dEtaHdt *= maskInC
        # Zero η tendency at OBC boundary strips.  Without this, the boundary
        # strip accumulates a divergence that is then discarded when
        # enforce_boundaries clamps η to eta_ext, causing systematic SSH drift.
        maskInC = _compute_maskInC(grid, dtype=eta_tendency.dtype, device=eta_tendency.device)
        if maskInC is not None:
            eta_tendency = eta_tendency * maskInC

        return StateTendency(
            eta=eta_tendency,
            u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data),
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )


@dataclass(frozen=True)
@register_dynamics_module("linear_free_surface_pressure_gradient")
class LinearFreeSurfacePressureGradient(TendencyModule):
    """Barotropic surface-pressure gradient ``-g·∇η`` for linear FS.
    """

    gravity: float = 9.81
    name: str = "pressure_gradient"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.FAST

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        grad_u = gradient_eta_to_u(state.eta.data, grid).unsqueeze(-3).expand_as(state.u.data)
        grad_v = gradient_eta_to_v(state.eta.data, grid).unsqueeze(-3).expand_as(state.v.data)
        gravity = state.eta.data.new_tensor(self.gravity)
        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=-gravity * grad_u,
            v=-gravity * grad_v,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )


@dataclass(frozen=True)
@register_dynamics_module("linear_drag")
class LinearDrag(TendencyModule):
    """Linear (Rayleigh) drag tendency ``-r·u`` on horizontal velocity."""

    rate: float | torch.Tensor
    name: str = "linear_drag"
    # SLOW: an explicit momentum tendency stepped with AB2, like MITgcm's
    # bottomDragLinear (selectImplicitDrag=0, mom_vecinv.F).  The old FAST default
    # was a split-explicit leftover the implicit free-surface stepper never
    # evaluated.
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW

    def __post_init__(self) -> None:
        if not isinstance(self.rate, torch.Tensor):
            # Keep Python float precision until the state dtype/device is known.
            object.__setattr__(self, "rate", torch.as_tensor(self.rate, dtype=torch.float64))
        if self.rate.ndim != 0 or not bool(torch.isfinite(self.rate)):
            raise ValueError("LinearDrag.rate must be a finite scalar")

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        rate = self.rate.to(dtype=state.u.data.dtype, device=state.u.data.device)
        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=-rate * state.u.data,
            v=-rate * state.v.data,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )
