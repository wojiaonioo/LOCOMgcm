from __future__ import annotations

from dataclasses import dataclass

import torch

from defgcm.geometry.metric_bundle import _positive_reciprocal

from defgcm.geometry import StructuredCGrid
from defgcm.operators import (
    explicit_vertical_diffusion_stability_limit,
    vertical_diffusive_flux,
    vertical_flux_divergence,
)
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import MomentumTermClass, TendencyModule, TendencyPhase, TendencyStepDomain
from defgcm.dynamics._registry import register_dynamics_module


@dataclass(frozen=True)
@register_dynamics_module("vertical_mixing")
class VerticalMixing(TendencyModule):
    """Vertical diffusion for tracers and (optionally) momentum.

    When *mix_momentum* is True, vertical viscous diffusion is also applied to
    the u and v velocity components.  The eddy viscosity defaults to
    *diffusivity* but can be overridden with the *viscosity* parameter.

    When *surface_stress_x* or *surface_stress_y* is nonzero **and**
    *mix_momentum* is True, the wind stress is injected as a Neumann flux
    boundary condition at the top interface of the viscous operator
    (following MOM6's stress-BC mode) instead of being applied as a separate
    body force.  In this case do **not** add a :class:`SurfaceWindStress`
    module to the dynamics list, or the forcing will be double-counted.
    """

    tracer_names: tuple[str, ...] = ("temp", "salt")
    diffusivity: float = 0.0
    mix_momentum: bool = False
    viscosity: float | None = None
    surface_stress_x: float = 0.0
    surface_stress_y: float = 0.0
    reference_density: float = 1025.0
    name: str = "vertical_mixing"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER
    momentum_term_class = MomentumTermClass.DISSIPATION

    @property
    def output_phases(self) -> dict[str, TendencyPhase]:
        # MIT temp/salt_integrate put diffusion in the tracer RHS before AB2,
        # independently of the momentum dissipation placement and coefficient.
        phases = {name: TendencyPhase.TRACER for name in self.tracer_names}
        if self.mix_momentum:
            phases.update(u=TendencyPhase.MOMENTUM, v=TendencyPhase.MOMENTUM)
        return phases

    @property
    def effective_viscosity(self) -> float:
        """Eddy viscosity used for momentum mixing."""
        if self.viscosity is not None:
            return self.viscosity
        return self.diffusivity

    def stability_limit_seconds(self, grid: StructuredCGrid) -> torch.Tensor:
        max_coeff = max(abs(self.diffusivity), abs(self.effective_viscosity))
        return explicit_vertical_diffusion_stability_limit(
            grid.layer_thickness,
            diffusivity=max_coeff,
        )

    def tracer_tendency(
        self,
        tracer: torch.Tensor,
        grid: StructuredCGrid,
        *,
        mask: torch.Tensor | None = None,
        hFacC: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flux = vertical_diffusive_flux(
            tracer,
            grid.vertical_coordinate.interface_spacing,
            diffusivity=self.diffusivity,
            mask=mask,
        )
        tendency = vertical_flux_divergence(flux, grid.layer_thickness)

        # Scale by 1/hFacC for partial cells (MITgcm: recip_hFacC * recip_drF
        # in gad_calc_rhs.F:L769; vertical_flux_divergence already divides
        # by drF, so we only need to divide by hFacC here).
        if hFacC is not None:
            hfc = hFacC.to(dtype=tracer.dtype, device=tracer.device)
            if hfc.ndim == 3:
                hfc = hfc.unsqueeze(0)  # [1, nz, ny, nx]
            hfc_wet = hfc > 0
            hfc_safe = torch.where(hfc_wet, hfc, torch.ones_like(hfc))
            recip_hFac = torch.where(
                hfc_wet, 1.0 / hfc_safe, torch.zeros_like(hfc),
            )
            scaled_tendency = tendency * recip_hFac
            tendency = torch.where(
                hfc_wet,
                scaled_tendency,
                torch.zeros_like(scaled_tendency),
            )

        return tendency

    def _momentum_tendency(
        self,
        field: torch.Tensor,
        grid: StructuredCGrid,
        *,
        surface_stress: float = 0.0,
        hFac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply vertical viscous diffusion to a momentum component.

        When *surface_stress* is nonzero, a Neumann flux boundary condition
        ``surface_flux = +tau / rho_0`` is applied at the surface interface
        (see lines 111-115: vertical_flux_divergence inverts so positive
        surface_flux → positive top-layer tendency, matching positive wind
        stress accelerating u positively).

        When *hFac* is provided (e.g. ``hFacW`` for u, ``hFacS`` for v),
        the flux divergence is divided by hFac instead of 1.0.  This matches
        MITgcm ``mom_fluxform.F``: ``guDissip(i,j) = vF(k+1)-vF(k)) /
        (drF(k) * hFacW(i,j,k))`` where the plain ``vertical_flux_divergence``
        divides only by drF.
        """
        sf: torch.Tensor | None = None
        if surface_stress != 0.0:
            # Convention: vertical_flux_divergence = -(F[k+1] - F[k]) / dz
            # For top layer: tendency = -(F[1] - F[0]) / dz ≈ +F[0] / dz
            # So positive surface_flux produces positive tendency.
            # Wind stress τ>0 should accelerate top layer positively,
            # hence surface_flux = +τ/ρ₀.
            sf = field.new_full(
                (field.shape[0], field.shape[2], field.shape[3]),
                surface_stress / self.reference_density,
            )
        flux = vertical_diffusive_flux(
            field,
            grid.vertical_coordinate.interface_spacing,
            diffusivity=self.effective_viscosity,
            surface_flux=sf,
            # Binary wet/dry mask: zero flux at wet/dry interfaces
            # (MITgcm: _maskW(k)*_maskW(k-1) in mom_u_rviscflux.F)
            mask=(hFac > 0).to(dtype=field.dtype) if hFac is not None else None,
        )
        tendency = vertical_flux_divergence(flux, grid.layer_thickness)

        # Scale by 1/hFac for partial cells (MITgcm: drF*hFac normalization)
        if hFac is not None:
            hFac_field = hFac.to(dtype=field.dtype, device=field.device)
            if hFac_field.ndim == 3:
                hFac_field = hFac_field.unsqueeze(0)  # [1, nz, ny, nx_face]
            recip_hFac = _positive_reciprocal(hFac_field)
            tendency = tendency * recip_hFac

        return tendency

    def populate_diagnostic_state(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> None:
        del state
        diagnostics.set_scalar(
            "vertical_diffusion_stability_limit_seconds",
            self.stability_limit_seconds(grid),
        )

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        self.populate_diagnostic_state(state, diagnostics, grid)

        # Tracer diffusion
        no_tracer_mixing = self.diffusivity == 0.0
        if no_tracer_mixing:
            temp_tendency = torch.zeros_like(state.temp.data)
            salt_tendency = torch.zeros_like(state.salt.data)
        else:
            temp_tendency = torch.zeros_like(state.temp.data)
            salt_tendency = torch.zeros_like(state.salt.data)
            # Get center mask for wet/dry interface zeroing
            cmask = (
                grid.masks.center_float(dtype=state.temp.data.dtype, device=state.temp.data.device)
                if grid.masks.has_3d_masks else None
            )
            hFacC = getattr(grid.masks, "hFacC", None)
            if "temp" in self.tracer_names:
                temp_tendency = self.tracer_tendency(state.temp.data, grid, mask=cmask, hFacC=hFacC)
            if "salt" in self.tracer_names:
                salt_tendency = self.tracer_tendency(state.salt.data, grid, mask=cmask, hFacC=hFacC)

        # Momentum diffusion
        if self.mix_momentum and self.effective_viscosity != 0.0:
            # Get hFac for partial-cell scaling (MITgcm: drF*hFac normalization)
            hFacW = getattr(grid.masks, "hFacW", None)
            hFacS = getattr(grid.masks, "hFacS", None)
            u_tendency = self._momentum_tendency(
                state.u.data, grid, surface_stress=self.surface_stress_x,
                hFac=hFacW,
            )
            v_tendency = self._momentum_tendency(
                state.v.data, grid, surface_stress=self.surface_stress_y,
                hFac=hFacS,
            )
        else:
            u_tendency = torch.zeros_like(state.u.data)
            v_tendency = torch.zeros_like(state.v.data)

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=u_tendency,
            v=v_tendency,
            temp=temp_tendency,
            salt=salt_tendency,
        )
