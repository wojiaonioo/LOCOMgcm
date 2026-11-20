from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from defgcm.fields import Field, FieldLocation
from defgcm.geometry import StructuredCGrid
from defgcm.operators import (
    gradient_center_from_haloed_scalar,
    gradient_center_to_u,
    gradient_center_to_v,
)
from defgcm.operators.vertical import hydrostatic_pressure_integral, hydrostatic_pressure_potential
from defgcm.state import DiagnosticState, PrognosticState, StateTendency
from defgcm.state.state import resolved_input

from .base import MomentumTermClass, TendencyModule, TendencyStepDomain
from defgcm.dynamics._registry import register_dynamics_module
from .equation_of_state import ReferenceValue, linear_density_anomaly
from .surface_pressure_load import surface_pressure_load_potential


def pressure_gradient_from_haloed_potential(
    phi_halo: torch.Tensor,
    *,
    dxC_u: torch.Tensor,
    dyC_v: torch.Tensor,
    wet_center_halo: torch.Tensor | None = None,
    halo: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``-grad(phi)`` on U/V faces from a haloed center potential.

    Geometry-specific exchange is done by the caller.  This tensor-only stencil
    consumes a scalar halo, 2-D face metrics, and an optional haloed center wet
    mask.  It is the common HPG narrow waist for cube/seam-aware callers.
    """
    grad_u, grad_v = gradient_center_from_haloed_scalar(
        phi_halo,
        dxC_u=dxC_u,
        dyC_v=dyC_v,
        wet_center_halo=wet_center_halo,
        halo=halo,
    )
    return -grad_u, -grad_v


@dataclass(frozen=True)
@register_dynamics_module("hydrostatic_pressure_gradient")
class HydrostaticPressureGradient(TendencyModule):
    """Hydrostatic pressure gradient ``-∇φ_hyd`` from density anomaly
    cumulative integral, with optional split-AB2 placement.

    The diagnostic ``total_hydrostatic_potential`` is LOCOMgcm's
    MIT-aligned ``totPhiHyd`` hand-off for JMD95P: hydrostatic potential
    anomaly plus the linear-free-surface contribution ``g * eta``.  It uses
    ``m2 s-2`` units and LOCOMgcm's downward-positive z convention.
    """

    gravity: float = 9.81
    reference_density: float = 1_025.0
    # MITgcm distinguishes rhoNil (linear-EOS reference density; find_rho.F:105
    # ``alphaRho = rhoNil*(sBeta*dS - tAlpha*dT)``) from rhoConst (Boussinesq
    # reference; ``recip_rhoConst`` in calc_phi_hyd.F:284).  ``reference_density``
    # here is rhoConst — the ``g/rho`` coefficient — while ``rho_nil`` scales the
    # linear density anomaly.  ``rho_nil=None`` defaults it to ``reference_density``
    # so configs with rhoNil == rhoConst stay byte-identical (the common case;
    # the split only matters for linear-EOS MIT alignment where rhoNil != rhoConst,
    # e.g. MIT default rhoNil=999.8 with rhoConst=1035).
    rho_nil: float | None = None
    thermal_expansion: float = 2.0e-4
    haline_contraction: float = 7.4e-4
    reference_temperature: ReferenceValue = 0.0
    reference_salinity: ReferenceValue = 0.0
    # MITgcm pLoadFile: phi0surf = pLoad/rhoConst joins the potential whose
    # gradient is taken (calc_grad_phi_hyd.F:131).  ``pload`` is [ny, nx] Pa,
    # required exactly when ``surface_pressure_load`` is set.
    surface_pressure_load: bool = False
    pload: torch.Tensor | None = field(default=None, compare=False, repr=False)
    name: str = "hydrostatic_pressure_gradient"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    momentum_term_class = MomentumTermClass.PRESSURE_GRADIENT

    def __post_init__(self) -> None:
        if self.surface_pressure_load and self.pload is None:
            raise ValueError("hydrostatic_pressure_gradient surface_pressure_load requires pload")
        if not self.surface_pressure_load and self.pload is not None:
            raise ValueError("hydrostatic_pressure_gradient pload given without surface_pressure_load")

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "HydrostaticPressureGradient":
        if not self.surface_pressure_load:
            return self
        return replace(self, pload=self.pload.to(device=device, dtype=dtype))

    def density_anomaly(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
    ) -> torch.Tensor:
        """Density anomaly w.r.t. rhoConst (``reference_density``), MITgcm's rhoInSitu.

        An equation-of-state module publishes its anomaly w.r.t. the field
        metadata's ``reference_density``: rhoNil for the linear EOS, rhoConst
        for JMD95 and MDJWF.  Adding the difference to rhoConst is the
        ``dRho = rhoNil - rhoConst`` of find_rho.F:101-108.
        """
        if "density_anomaly" in diagnostics.fields:
            field = diagnostics.fields["density_anomaly"]
            return field.data + (field.metadata["reference_density"] - self.reference_density)
        rho_nil = self.reference_density if self.rho_nil is None else self.rho_nil
        return linear_density_anomaly(
            state.temp.data,
            state.salt.data,
            reference_density=rho_nil,
            rho_const=self.reference_density,
            thermal_expansion=self.thermal_expansion,
            haline_contraction=self.haline_contraction,
            reference_temperature=self.reference_temperature,
            reference_salinity=self.reference_salinity,
        )

    def populate_pressure_diagnostics(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
        *,
        return_potential: bool = False,
    ) -> torch.Tensor:
        """Publish the stored center potentials and return the column mass integral by default."""
        density_anomaly = self.density_anomaly(state, diagnostics)
        vertical = grid.vertical_coordinate
        pressure_potential = hydrostatic_pressure_potential(
            density_anomaly, vertical.center_spacing, vertical.bottom_half_thickness,
            gravity=self.gravity, reference_density=self.reference_density
        )
        diagnostics.set_field(
            Field(
                "baroclinic_pressure_potential",
                pressure_potential,
                FieldLocation.CENTER,
                "m2 s-2",
                mask_ref="center",
            )
        )
        total_pressure_potential = pressure_potential + self.gravity * state.eta.data.unsqueeze(1)
        # CALC_PHI_HYD and DIAGS_PHI_HYD retain the dry-column integral.
        # This total potential is also the next EOS pressure operand.
        diagnostics.set_field(
            Field(
                "total_hydrostatic_potential",
                total_pressure_potential,
                FieldLocation.CENTER,
                "m2 s-2",
                mask_ref="center",
            )
        )
        if return_potential:
            return pressure_potential
        return hydrostatic_pressure_integral(
            density_anomaly, vertical.center_spacing, vertical.bottom_half_thickness)

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        pressure_potential = self.populate_pressure_diagnostics(state, diagnostics, grid, return_potential=True)
        # CALC_PHI_HYD runs with pfFacMom=0; TIMESTEP gates only its gradient.
        if not resolved_input(diagnostics.inputs, "momentum_pressure_forcing", True):
            return StateTendency.zeros_like(state)
        if self.surface_pressure_load:
            phi0surf = surface_pressure_load_potential(self.pload, recip_rho_const=1.0 / self.reference_density)
            pressure_potential = pressure_potential + phi0surf
        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=-gradient_center_to_u(pressure_potential, grid),
            v=-gradient_center_to_v(pressure_potential, grid),
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )
