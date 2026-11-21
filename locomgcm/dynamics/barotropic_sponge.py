from __future__ import annotations

from dataclasses import dataclass, field

import torch

from locomgcm.boundaries.open import (
    BoundarySide,
    SideBoundaryTensorSet,
    barotropic_boundary_band_target,
    expand_barotropic_side_boundary_tensor,
)
from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.grid import _side_is_open
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyStepDomain
from locomgcm.dynamics._registry import register_dynamics_module
from .sponge_weights import (
    SPONGE_MODES,
    band_weights as _band_weights,
    mitgcm_boundary_weight as _mitgcm_boundary_weight,
    mitgcm_inverse_relaxation_time as _mitgcm_inverse_relaxation_time,
)


@dataclass
@register_dynamics_module("barotropic_boundary_sponge_relaxation")
class BarotropicBoundarySpongeRelaxation(TendencyModule):
    eta_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    normal_velocity_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    width: int = 6
    timescale_seconds: float = 21_600.0
    timescale_inner_seconds: float | None = None
    sponge_mode: str = "mitgcm"
    use_linear_sponge: bool = False
    name: str = "barotropic_boundary_sponge"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.FAST

    def __post_init__(self) -> None:
        if self.sponge_mode not in SPONGE_MODES:
            raise ValueError(f"sponge_mode must be one of {SPONGE_MODES}, got '{self.sponge_mode}'")
        if self.timescale_inner_seconds is None:
            self.timescale_inner_seconds = 10.0 * self.timescale_seconds

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        del diagnostics
        if self.width < 1:
            raise ValueError("sponge width must be positive")
        if self.timescale_seconds <= 0.0:
            raise ValueError("sponge timescale_seconds must be positive")

        eta_tendency = torch.zeros_like(state.eta.data)
        u_tendency = torch.zeros_like(state.u.data)
        v_tendency = torch.zeros_like(state.v.data)
        scale = state.eta.data.new_tensor(1.0 / self.timescale_seconds)

        for side in BoundarySide:
            if not _side_is_open(side.value, grid):
                continue
            side_width = min(self.width, grid.nx if side in (BoundarySide.WEST, BoundarySide.EAST) else grid.ny)
            if side_width < 1:
                continue
            weights = _band_weights(side_width, dtype=state.eta.data.dtype, device=state.eta.data.device)
            eta_target = self._side_target(
                self.eta_values,
                side,
                grid,
                batch_size=state.batch_size,
                dtype=state.eta.data.dtype,
                device=state.eta.data.device,
            )
            normal_target = self._side_target(
                self.normal_velocity_values,
                side,
                grid,
                batch_size=state.batch_size,
                dtype=state.eta.data.dtype,
                device=state.eta.data.device,
            )
            if eta_target is None and normal_target is None:
                continue
            if self.sponge_mode == "mitgcm":
                eta_tendency = eta_tendency + self._eta_side_tendency_mitgcm(
                    state.eta.data,
                    eta_target,
                    side,
                    grid,
                    width=side_width,
                )
            else:
                eta_tendency = eta_tendency + self._eta_side_tendency(
                    state.eta.data,
                    eta_target,
                    side,
                    grid,
                    width=side_width,
                    scale=scale,
                    weights=weights,
                )
            if side in (BoundarySide.WEST, BoundarySide.EAST):
                if self.sponge_mode == "mitgcm":
                    u_tendency = u_tendency + self._barotropic_normal_side_tendency(
                        state.u.data,
                        normal_target,
                        side,
                        grid,
                        width=side_width,
                    )
                else:
                    u_tendency = u_tendency + self._u_side_tendency(
                        state.u.data,
                        normal_target,
                        side,
                        width=side_width,
                        scale=scale,
                        weights=weights,
                    )
            else:
                if self.sponge_mode == "mitgcm":
                    v_tendency = v_tendency + self._barotropic_normal_side_tendency(
                        state.v.data,
                        normal_target,
                        side,
                        grid,
                        width=side_width,
                    )
                else:
                    v_tendency = v_tendency + self._v_side_tendency(
                        state.v.data,
                        normal_target,
                        side,
                        width=side_width,
                        scale=scale,
                        weights=weights,
                    )

        if grid.masks.has_3d_masks:
            eta_mask = grid.masks.center_float(dtype=state.eta.data.dtype, device=state.eta.data.device)[0]
            u_mask = grid.masks.u_float(dtype=state.u.data.dtype, device=state.u.data.device)
            v_mask = grid.masks.v_float(dtype=state.v.data.dtype, device=state.v.data.device)
            eta_tendency = eta_tendency * eta_mask
            u_tendency = u_tendency * u_mask
            v_tendency = v_tendency * v_mask

        return StateTendency(
            eta=eta_tendency,
            u=u_tendency,
            v=v_tendency,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )

    def _side_target(
        self,
        values: SideBoundaryTensorSet,
        side: BoundarySide,
        grid: StructuredCGrid,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        target = values.get(side)
        if target is None:
            return None
        return expand_barotropic_side_boundary_tensor(
            target,
            side,
            grid,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

    def _eta_side_tendency(
        self,
        eta: torch.Tensor,
        target: torch.Tensor,
        side: BoundarySide,
        grid: StructuredCGrid,
        *,
        width: int,
        scale: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        tendency = torch.zeros_like(eta)
        target_band = barotropic_boundary_band_target(
            target,
            side,
            grid,
            batch_size=eta.shape[0],
            dtype=eta.dtype,
            device=eta.device,
            width=width,
        )
        if side == BoundarySide.WEST:
            tendency[..., :width] = scale * (target_band - eta[..., :width]) * weights.view(1, 1, width)
        elif side == BoundarySide.EAST:
            tendency[..., -width:] = scale * (target_band - eta[..., -width:]) * weights.flip(0).view(1, 1, width)
        elif side == BoundarySide.SOUTH:
            tendency[..., :width, :] = scale * (target_band - eta[..., :width, :]) * weights.view(1, width, 1)
        else:
            tendency[..., -width:, :] = scale * (target_band - eta[..., -width:, :]) * weights.flip(0).view(1, width, 1)
        return tendency

    def _eta_side_tendency_mitgcm(
        self,
        eta: torch.Tensor,
        target: torch.Tensor,
        side: BoundarySide,
        grid: StructuredCGrid,
        *,
        width: int,
    ) -> torch.Tensor:
        tendency = torch.zeros_like(eta)
        target_band = barotropic_boundary_band_target(
            target,
            side,
            grid,
            batch_size=eta.shape[0],
            dtype=eta.dtype,
            device=eta.device,
            width=width,
        )
        boundary_weight = _mitgcm_boundary_weight(width, dtype=eta.dtype, device=eta.device)
        inv_tau = _mitgcm_inverse_relaxation_time(
            width,
            tau_boundary=self.timescale_seconds,
            tau_inner=float(self.timescale_inner_seconds),
            dtype=eta.dtype,
            device=eta.device,
        )
        if side == BoundarySide.WEST:
            band = eta[..., :width]
            if self.use_linear_sponge:
                relax = target_band
            else:
                relax = boundary_weight.view(1, 1, width) * target_band + (1.0 - boundary_weight).view(1, 1, width) * band
            tendency[..., :width] = -(band - relax) * inv_tau.view(1, 1, width)
        elif side == BoundarySide.EAST:
            band = eta[..., -width:]
            boundary_weight_e = boundary_weight.flip(0)
            inv_tau_e = inv_tau.flip(0)
            if self.use_linear_sponge:
                relax = target_band
            else:
                relax = (
                    boundary_weight_e.view(1, 1, width) * target_band
                    + (1.0 - boundary_weight_e).view(1, 1, width) * band
                )
            tendency[..., -width:] = -(band - relax) * inv_tau_e.view(1, 1, width)
        elif side == BoundarySide.SOUTH:
            band = eta[..., :width, :]
            if self.use_linear_sponge:
                relax = target_band
            else:
                relax = boundary_weight.view(1, width, 1) * target_band + (1.0 - boundary_weight).view(1, width, 1) * band
            tendency[..., :width, :] = -(band - relax) * inv_tau.view(1, width, 1)
        else:
            band = eta[..., -width:, :]
            boundary_weight_n = boundary_weight.flip(0)
            inv_tau_n = inv_tau.flip(0)
            if self.use_linear_sponge:
                relax = target_band
            else:
                relax = (
                    boundary_weight_n.view(1, width, 1) * target_band
                    + (1.0 - boundary_weight_n).view(1, width, 1) * band
                )
            tendency[..., -width:, :] = -(band - relax) * inv_tau_n.view(1, width, 1)
        return tendency

    def _u_side_tendency(
        self,
        u: torch.Tensor,
        target: torch.Tensor,
        side: BoundarySide,
        *,
        width: int,
        scale: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        tendency = torch.zeros_like(u)
        target_band = target.unsqueeze(1).unsqueeze(-1).expand(-1, u.shape[1], -1, width)
        if side == BoundarySide.WEST:
            tendency[..., :width] = scale * (target_band - u[..., :width]) * weights.view(1, 1, 1, width)
        else:
            tendency[..., -width:] = scale * (target_band - u[..., -width:]) * weights.flip(0).view(1, 1, 1, width)
        return tendency

    def _v_side_tendency(
        self,
        v: torch.Tensor,
        target: torch.Tensor,
        side: BoundarySide,
        *,
        width: int,
        scale: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        tendency = torch.zeros_like(v)
        target_band = target.unsqueeze(1).unsqueeze(-2).expand(-1, v.shape[1], width, -1)
        if side == BoundarySide.SOUTH:
            tendency[..., :width, :] = scale * (target_band - v[..., :width, :]) * weights.view(1, 1, width, 1)
        else:
            tendency[..., -width:, :] = (
                scale * (target_band - v[..., -width:, :]) * weights.flip(0).view(1, 1, width, 1)
            )
        return tendency

    def _barotropic_normal_side_tendency(
        self,
        field: torch.Tensor,
        target: torch.Tensor,
        side: BoundarySide,
        grid: StructuredCGrid,
        *,
        width: int,
    ) -> torch.Tensor:
        tendency = torch.zeros_like(field)
        inv_tau = _mitgcm_inverse_relaxation_time(
            width,
            tau_boundary=self.timescale_seconds,
            tau_inner=float(self.timescale_inner_seconds),
            dtype=field.dtype,
            device=field.device,
        )
        boundary_weight = _mitgcm_boundary_weight(width, dtype=field.dtype, device=field.device)
        thickness = grid.layer_thickness.to(dtype=field.dtype, device=field.device).view(1, field.shape[1], 1, 1)
        if side in (BoundarySide.WEST, BoundarySide.EAST):
            if grid.masks.hFacW is not None:
                hfac = grid.masks.hFacW.to(dtype=field.dtype, device=field.device)
                if hfac.ndim == 3:
                    hfac = hfac.unsqueeze(0)
            else:
                hfac = torch.ones_like(field)
            if side == BoundarySide.WEST:
                band = field[..., :width]
                weight_band = thickness * hfac[..., :width]
                target_band = target.unsqueeze(-1).expand(-1, target.shape[-1], width)
                band_weight_view = boundary_weight.view(1, 1, width)
                inv_tau_view = inv_tau.view(1, 1, width)
                band_slice = (..., slice(None, width))
            else:
                band = field[..., -width:]
                weight_band = thickness * hfac[..., -width:]
                target_band = target.unsqueeze(-1).expand(-1, target.shape[-1], width)
                band_weight_view = boundary_weight.flip(0).view(1, 1, width)
                inv_tau_view = inv_tau.flip(0).view(1, 1, width)
                band_slice = (..., slice(-width, None))
        else:
            if grid.masks.hFacS is not None:
                hfac = grid.masks.hFacS.to(dtype=field.dtype, device=field.device)
                if hfac.ndim == 3:
                    hfac = hfac.unsqueeze(0)
            else:
                hfac = torch.ones_like(field)
            if side == BoundarySide.SOUTH:
                band = field[..., :width, :]
                weight_band = thickness * hfac[..., :width, :]
                target_band = target.unsqueeze(-2).expand(-1, width, target.shape[-1])
                band_weight_view = boundary_weight.view(1, width, 1)
                inv_tau_view = inv_tau.view(1, width, 1)
                band_slice = (..., slice(None, width), slice(None))
            else:
                band = field[..., -width:, :]
                weight_band = thickness * hfac[..., -width:, :]
                target_band = target.unsqueeze(-2).expand(-1, width, target.shape[-1])
                band_weight_view = boundary_weight.flip(0).view(1, width, 1)
                inv_tau_view = inv_tau.flip(0).view(1, width, 1)
                band_slice = (..., slice(-width, None), slice(None))

        denom = weight_band.sum(dim=1)
        barotropic = torch.where(
            denom > 0.0,
            (band * weight_band).sum(dim=1) / denom,
            torch.zeros_like(denom),
        )
        if self.use_linear_sponge:
            relax = target_band
        else:
            relax = band_weight_view * target_band + (1.0 - band_weight_view) * barotropic
        correction = -(barotropic - relax) * inv_tau_view
        tendency[band_slice] = correction.unsqueeze(1).expand_as(band) * (weight_band > 0.0).to(field.dtype)
        return tendency
