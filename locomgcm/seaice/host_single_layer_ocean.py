"""MIT-sign one-layer ocean: full SST restoring and forcing outside AB."""

from dataclasses import dataclass, replace

import torch

from .ports import OceanSurface
from .records import Clock, Geometry, OceanFluxes, OceanState
from .space import SeaiceSpace

#: MITgcm SEAICErestoreUnderIce values implemented: SST restoring also under ice
#: (forcing_surf_relax.F:94-99), see surface_forcing.
RESTORE_UNDER_ICE = (True,)
#: staggerTimeStep values implemented: temperature steps after the clock
#: increment, in the staggered THERMODYNAMICS (forward_step.F:976-1005).
STAGGER_TIME_STEP = (True,)
#: tracForcingOutAB values implemented: surface forcing is added after the AB
#: step (ini_parms.F:1099-1101 derives 1 from forcing_In_AB=.FALSE.). The AB
#: history itself is identically zero, so abEps cannot change the result.
TRACER_FORCING_OUT_AB = (1,)


@dataclass(frozen=True)
class HostParameters:
    deltaT: float
    temp_stepping: bool
    lambda_theta: float | torch.Tensor
    cp: float | torch.Tensor = 3986.0
    rho: float | torch.Tensor = 1030.0


@dataclass(frozen=True)
class SurfaceForcing:
    surfaceForcingT: torch.Tensor
    restoring: torch.Tensor
    fluxes: OceanFluxes


@dataclass(frozen=True)
class TemperatureStep:
    ocean: OceanState
    tendency: torch.Tensor
    surface: SurfaceForcing


def initialize_ocean(geometry: Geometry, uVel: torch.Tensor, vVel: torch.Tensor,
                     etaN: torch.Tensor, *, space: SeaiceSpace,
                     t_ref: float | torch.Tensor, s_ref: float | torch.Tensor) -> OceanState:
    """ini_theta.F; ini_vel.F:44-75: dry tracer cells explicitly get +0.0."""
    zero = torch.zeros_like(geometry.hFacC)
    t = torch.as_tensor(t_ref, dtype=torch.float64, device=zero.device)
    s = torch.as_tensor(s_ref, dtype=torch.float64, device=zero.device)
    theta = torch.where(geometry.maskC != 0, t, zero)
    salt = torch.where(geometry.maskC != 0, s, zero)
    velocity = space.exchange("I1", {"uVel": uVel, "vVel": vVel}, chain=space.chain)
    # ini_psurf.F:62: only L has a surface-height initialization file.
    eta = space.exchange("I1", {"etaN": etaN}, chain=space.chain)["etaN"] if space.chain == "L" else etaN.clone()
    return OceanState(theta, salt, velocity["uVel"] * geometry.maskW,
                      velocity["vVel"] * geometry.maskS, eta, zero.clone(),
                      eta.clone(), eta.clone(), zero.clone(), Clock(0, 0.))


def ocean_surface(ocean: OceanState) -> OceanSurface:
    """The host's side of the ice entry port: its one layer is the surface layer."""
    return OceanSurface(ocean.theta, ocean.salt, ocean.uVel, ocean.vVel, ocean.etaN)


def surface_forcing(ocean: OceanState, SST: torch.Tensor, fluxes: OceanFluxes,
                    geometry: Geometry, parameters: HostParameters) -> SurfaceForcing:
    """T7/H7/B9.sf, after external_forcing_surf.F:226-231.

    All supported chains restore under ice (forcing_surf_relax.F:94-99).
    SHORTWAVE_HEATING is absent: Qnet includes all radiative heat here.
    """
    restoring = -parameters.lambda_theta * (ocean.theta - SST) * geometry.drF * geometry.hFacC
    one = ocean.theta.new_ones(())
    cp = torch.as_tensor(parameters.cp, dtype=one.dtype, device=one.device)
    rho = torch.as_tensor(parameters.rho, dtype=one.dtype, device=one.device)
    sf = restoring - fluxes.Qnet * (one / cp) * (one / rho)
    return SurfaceForcing(sf, restoring, fluxes)


def temperature_step(ocean: OceanState, surface: SurfaceForcing, geometry: Geometry,
                     parameters: HostParameters, *, clock: Clock, space: SeaiceSpace) -> TemperatureStep:
    """T8/H8/B9: apply_forcing.F:618-624; temp_integrate.F:398-404,469,523.

    The program has already advanced the clock (forward_step.F:807-808).
    The zero AB history is positive zero; dry inverse hFac is exactly zero.
    """
    if clock.myIter != ocean.clock.myIter + 1 or clock.myTime != ocean.clock.myTime + parameters.deltaT:
        raise ValueError("host temperature requires the next outer-step clock")
    tendency = surface.surfaceForcingT * (torch.ones_like(geometry.drF) / geometry.drF) * geometry.recip_hFacC
    theta = ocean.theta + parameters.deltaT * tendency if parameters.temp_stepping else ocean.theta.clone()
    exchanged = space.exchange("X10", {"theta": theta, "salt": ocean.salt}, chain=space.chain)
    candidate = replace(ocean, theta=exchanged["theta"], salt=exchanged["salt"],
                        gtNm1=torch.zeros_like(ocean.gtNm1), clock=clock)
    return TemperatureStep(candidate, tendency, surface)
