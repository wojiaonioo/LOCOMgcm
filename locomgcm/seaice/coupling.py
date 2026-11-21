"""The sea-ice component inside the three-dimensional ocean step.

MITgcm calls EXF, THSICE_MAIN and SEAICE_MODEL in DO_OCEANIC_PHYS before
FREEZE_SURFACE (``do_oceanic_phys.F:376-474,557``) and converts the fluxes they
leave into surfaceForcingT/S in EXTERNAL_FORCING_SURF (``:579``).  The stepper
calls :meth:`SeaiceCoupling.advance` at that position with the step-entry ocean
state; the module builds the ice entry record ``locomgcm.seaice.ports.OceanSurface``
from it, runs ``locomgcm.seaice.program.advance_ice`` and maps the published
fluxes to the ocean grid.  Its tracer evaluation applies them
(``external_forcing_surf.F:226-345``), and the stepper hands the published AREA
to surface restoring (``EvaluationInputs.ice_area``).

Ocean tensors are ``[1, nz, ny, nx]`` centres, ``[1, nz, ny, nx+1]`` U faces and
``[1, nz, ny+1, nx]`` V faces, float64, k=0 the surface; eta is upward-positive
in m, theta degC, salt g/kg, u and v positive east and north: the signs of the
ice records, so nothing changes sign here.  The published fluxes keep MITgcm
units and signs (Qnet and Qsw W/m2 positive upward, EmPmR kg/m2/s positive for
net evaporation, saltFlux g/m2/s positive upward); AREA is the ice fraction.
The ice's native storage is the tile layout of ``locomgcm.seaice.space``; this
module only copies between it and the ocean grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from locomgcm.dynamics._registry import register_dynamics_module
from locomgcm.dynamics.base import TendencyModule, TendencyPhase, TendencyStepDomain
from locomgcm.dynamics.surface_flux import (SurfaceForcing, apply_wpflux_surface_tendency,
                                          surface_forcing_from_fluxes)
from locomgcm.geometry import StructuredCGrid
from locomgcm.sampling import SampleClock, SamplingRule
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from .forcing import ExfConfig, ForcingData, ForcingSample, sample_atmosphere
from .ports import OceanSurface
from .program import ProgramParameters, advance_ice
from .records import Geometry, SeaiceCarry
from .space import SeaiceSpace
from .step_state import SeaiceStepState

@dataclass(frozen=True)
class PublishedOceanFluxes:
    """The ice's published fluxes on the ocean grid, read by two consumers.

    ``forcing`` is the surfaceForcing bundle of the tracer apply (and of a
    surface-flux harvest); ``ice_area`` [1, ny, nx] is AREA for surface
    restoring.
    """

    forcing: SurfaceForcing
    ice_area: torch.Tensor


@dataclass
@register_dynamics_module("seaice")
class SeaiceCoupling(TendencyModule):
    """EXF and the sea-ice packages of one ocean step, and their tracer forcing.

    ``program``, ``exf_config``, ``geometry``, ``space`` and ``forcing`` are the
    ice component's inputs (``locomgcm.seaice``); ``initial`` its carry at the
    first step.  ``heat_capacity``, ``rho_const``, ``convert_fw2salt``,
    ``salt_evprn``, ``penetrating_sw`` and ``jwtype`` are the ocean's
    EXTERNAL_FORCING_SURF constants (HeatCapacity_Cp, rhoConst, convertFW2Salt,
    salt_EvPrRn, selectPenetratingSW>0, the Jerlov type); ``ab2_eligible``
    follows tracForcingOutAB.
    """

    sampling_rule = SamplingRule(SampleClock.INDEPENDENT)

    program: ProgramParameters = field(kw_only=True)
    exf_config: ExfConfig = field(kw_only=True)
    geometry: Geometry = field(kw_only=True)
    space: SeaiceSpace = field(kw_only=True)
    forcing: ForcingData = field(kw_only=True)
    initial: SeaiceCarry = field(kw_only=True)
    heat_capacity: float = field(kw_only=True)
    rho_const: float = field(kw_only=True)
    convert_fw2salt: float = field(kw_only=True)
    salt_evprn: float = field(kw_only=True)
    penetrating_sw: bool = field(kw_only=True)
    jwtype: int = field(kw_only=True)
    ab2_eligible: bool = field(kw_only=True)
    tracer_names: tuple[str, ...] = field(kw_only=True)
    name: str = "seaice"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER
    # The fluxes this step published; set by advance, read by evaluate and
    # by the stepper's evaluation inputs.
    published: PublishedOceanFluxes | None = field(default=None, init=False, repr=False)

    def ocean_surface(self, state: PrognosticState) -> OceanSurface:
        """The step-entry ocean surface in native ice storage, halos exchanged.

        uVel/vVel are the west/south faces of each cell: the ocean's U/V faces
        without the repeated east/north face of the periodic grid.
        """
        space, chain = self.space, self.space.chain
        nx = state.temp.data.shape[-1]
        ny = state.temp.data.shape[-2]
        tracers = space.exchange("X10", {
            "theta": space.from_global(state.temp.data[0, 0]),
            "salt": space.from_global(state.salt.data[0, 0]),
        }, chain=chain)
        flow = space.exchange("I1", {
            "uVel": space.from_global(state.u.data[0, 0, :, :nx]),
            "vVel": space.from_global(state.v.data[0, 0, :ny, :]),
            "etaN": space.from_global(state.eta.data[0]),
        }, chain=chain)
        return OceanSurface(tracers["theta"], tracers["salt"], flow["uVel"], flow["vVel"],
                            flow["etaN"])

    def advance(self, step_state: SeaiceStepState | None, state: PrognosticState,
                grid: StructuredCGrid, *, dt: float) -> SeaiceStepState:
        """Run EXF and the ice for this step; publish the fluxes on the ocean grid.

        ``state`` is the step-entry ocean before FREEZE_SURFACE; ``step_state``
        is ``None`` at the first step, which starts from ``initial``.
        """
        if dt != self.program.deltaT:
            raise ValueError(f"the seaice module steps once per ocean step of {self.program.deltaT} s, "
                             f"the ocean steps {dt} s")
        if step_state is None:
            step_state = SeaiceStepState(self.initial, None)
        carry = step_state.carry
        sample = sample_atmosphere(self.forcing, self.exf_config, self.geometry, self.space,
                                   time=carry.clock.myTime, previous=step_state.previous_sample)
        result = advance_ice(carry, self.ocean_surface(state), sample, self.exf_config,
                             self.geometry, self.program, space=self.space)
        values = result.fluxes.values
        to_ocean = lambda native: self.space.to_global(native).unsqueeze(0)
        forcing_t, forcing_s, _, _ = surface_forcing_from_fluxes(
            to_ocean(values.Qnet), to_ocean(values.Qsw), to_ocean(values.EmPmR),
            fu=None, fv=None, salt_flux=to_ocean(values.saltFlux),
            heat_capacity=self.heat_capacity, rho_const=self.rho_const,
            convert_fw2salt=self.convert_fw2salt, salt_evprn=self.salt_evprn,
            penetrating_sw=self.penetrating_sw,
            surface_salinity=state.salt.data.select(dim=-3, index=0), pmep_r=None,
        )
        forcing = SurfaceForcing(sfT=forcing_t, sfS=forcing_s, qsw=to_ocean(values.Qsw),
                                 rho_const=self.rho_const, heat_capacity=self.heat_capacity,
                                 qnet=to_ocean(values.Qnet), convert_fw2salt=self.convert_fw2salt,
                                 salt_evprn=self.salt_evprn, penetrating_sw=self.penetrating_sw)
        self.published = PublishedOceanFluxes(forcing, to_ocean(values.AREA))
        return SeaiceStepState(result.candidate, result.next_forcing_sample)

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        del diagnostics
        tendency = apply_wpflux_surface_tendency(self.published.forcing, state, grid, jwtype=self.jwtype)
        return StateTendency(
            eta=tendency.eta, u=tendency.u, v=tendency.v,
            temp=tendency.temp if "temp" in self.tracer_names else torch.zeros_like(tendency.temp),
            salt=tendency.salt if "salt" in self.tracer_names else torch.zeros_like(tendency.salt),
        )

