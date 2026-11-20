"""Surface relaxation of temperature and salinity towards a climatology.

MITgcm ``FORCING_SURF_RELAX`` (``model/src/forcing_surf_relax.F:91-104``) with
the rates of ``ini_forcing.F:46-65``::

    lambda          = 1/tau  where |yC| <= latBandClimRelax, else 0
    surfaceForcingT = -lambdaT*(theta(k=0) - SST)*drF(0)*hFacC(0)    [degC m/s]
    surfaceForcingS = -lambdaS*(salt(k=0)  - SSS)*drF(0)*hFacC(0)    [psu m/s]

applied as ``gT(0) += surfaceForcingT*recip_drF(0)*recip_hFacC(0)``
(``apply_forcing.F:618-624``).  Centre fields are ``[..., ny, nx]``; ``yC`` is
the grid's centre latitude in degrees (``MetricSet.yC_row``); ``drF`` and
``hFacC`` are positive and k=0 is the surface in both conventions, so nothing
changes sign.  SST and SSS come from the module's source at step start: the
core ``thetaClimFile``/``saltClimFile`` records or EXF's climsst/climsss.  The
restoring flux is MITgcm-native ``surfaceForcingT/S``; a surface-forcing
consumer adds it before any other surface flux (``external_forcing_surf.F:173-234``).
Where restoring stops under sea ice (useSEAICE without SEAICErestoreUnderIce,
``forcing_surf_relax.F:73-85``) the flux carries ``1-AREA``, AREA the sea-ice
fraction [0, 1] at centres published by the step's sea-ice component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from defgcm.geometry import StructuredCGrid
from defgcm.geometry.grid import runtime_memo_constant
from defgcm.sampling import SampleClock, SamplingRule
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyPhase, TendencyStepDomain
from .surface_flux import SurfaceForcing, apply_wpflux_surface_tendency
from defgcm.dynamics._registry import register_dynamics_module

if TYPE_CHECKING:
    from defgcm.io.exf_source import ExfClimatologySource
    from defgcm.io.forcing_records import SurfaceClimatologySource


def relaxation_rate(abs_lat: torch.Tensor, *, tau: float, lat_band: float) -> torch.Tensor:
    """``lambdaClimRelax`` of ini_forcing.F:50-61: ``1/tau`` inside the band, 0 outside."""
    return torch.where(abs_lat <= lat_band, torch.full_like(abs_lat, 1.0 / tau),
                       torch.zeros_like(abs_lat))


def surface_relaxation_flux(
    rate: torch.Tensor,
    tracer_surface: torch.Tensor,
    target: torch.Tensor,
    drf_surface: float | torch.Tensor,
    hfac_surface: torch.Tensor,
) -> torch.Tensor:
    """``surfaceForcing`` restoring term, forcing_surf_relax.F:96-104 factor order."""
    return -rate * (tracer_surface - target) * drf_surface * hfac_surface


def surface_relaxation_flux_open_water(
    rate: torch.Tensor,
    ice_area: torch.Tensor,
    tracer_surface: torch.Tensor,
    target: torch.Tensor,
    drf_surface: float | torch.Tensor,
    hfac_surface: torch.Tensor,
) -> torch.Tensor:
    """The restoring term outside sea ice, forcing_surf_relax.F:77-85 factor order."""
    return -rate * (1.0 - ice_area) * (tracer_surface - target) * drf_surface * hfac_surface


def validate_relaxation_composition(*, modules) -> None:
    """Refuse restoring that stops under sea ice without a sea-ice component.

    ``restore_under_ice=False`` reads the AREA the step's sea-ice component
    (dynamics module ``seaice``) publishes (``forcing_surf_relax.F:73-85``).
    """
    stops_under_ice = any(getattr(m, "name", "") == "surface_climate_relaxation"
                          and not m.restore_under_ice for m in modules)
    if stops_under_ice and not any(getattr(m, "name", "") == "seaice" for m in modules):
        raise ValueError("surface_climate_relaxation with restore_under_ice=False needs the sea-ice "
                         "fraction of a `seaice` dynamics module, and the model has none")


@dataclass
@register_dynamics_module("surface_climate_relaxation")
class SurfaceClimateRelaxation(TendencyModule):
    """Restore surface temperature and salinity to ``thetaClimFile``/``saltClimFile``.

    ``tau_theta``/``tau_salt`` are ``tauThetaClimRelax``/``tauSaltClimRelax``
    [s]; a value <= 0 leaves that tracer unrestored (``set_parms.F:250-253``).
    ``lat_band`` is ``latBandClimRelax`` in degrees, MITgcm's actual value or
    its default ``3*sum(delY)`` (``load_grid_spacing.F:166-175``).
    ``ab2_eligible`` follows tracForcingOutAB like every surface tracer forcing.
    ``restore_under_ice`` is False exactly when MITgcm's
    ``useSEAICE .AND. .NOT.SEAICErestoreUnderIce`` holds (``forcing_surf_relax.F:73``):
    the flux then carries ``1-AREA`` and the evaluation reads the sea-ice
    fraction from ``EvaluationInputs.ice_area``.
    """

    @property
    def sampling_rule(self) -> SamplingRule:
        return SamplingRule(SampleClock.START, ("_sst", "_sss"), providers=("source",))

    source: "SurfaceClimatologySource | ExfClimatologySource" = field(kw_only=True)
    tau_theta: float = field(kw_only=True)
    tau_salt: float = field(kw_only=True)
    lat_band: float = field(kw_only=True)
    ab2_eligible: bool = field(kw_only=True)
    restore_under_ice: bool = field(kw_only=True)
    name: str = "surface_climate_relaxation"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER
    current_time_seconds: float = field(default=0.0, init=False, repr=False)
    _sst: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sss: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # ini_parms.F:1529-1548 (exf_check.F:607-640 for EXF): a positive time
        # scale needs its climatology file.
        if self.tau_theta > 0.0 and not self.source.has_sst:
            raise ValueError("tauThetaClimRelax > 0 but the SST climatology has no file")
        if self.tau_salt > 0.0 and not self.source.has_sss:
            raise ValueError("tauSaltClimRelax > 0 but the SSS climatology has no file")

    def update_model_time(self, time_seconds: float) -> None:
        self.current_time_seconds = float(time_seconds)
        self._sst, self._sss = self.source.sample(time_seconds)

    def to_runtime(self, device: torch.device | str,
                   dtype: torch.dtype | None = None) -> "SurfaceClimateRelaxation":
        self.source = self.source.to_runtime(device, dtype)
        self.update_model_time(self.current_time_seconds)
        return self

    def _rate(self, grid: StructuredCGrid, tau: float, dtype: torch.dtype,
              device: torch.device) -> torch.Tensor:
        y_c = grid.metrics.yC_row
        if y_c is None:
            raise ValueError("surface_climate_relaxation needs the grid's centre latitudes "
                             "(MetricSet.yC_row) for latBandClimRelax")

        def build() -> torch.Tensor:
            abs_lat = torch.tensor(y_c, dtype=torch.float64).abs().unsqueeze(-1).expand(
                len(y_c), grid.nx)
            return relaxation_rate(abs_lat, tau=tau, lat_band=self.lat_band).to(
                dtype=dtype, device=device)

        return runtime_memo_constant(
            grid, ("surface_climate_relaxation_rate", tau, self.lat_band, dtype, str(device)),
            build)

    def surface_forcing(self, state: PrognosticState, grid: StructuredCGrid, *,
                        ice_area: torch.Tensor | None) -> SurfaceForcing:
        """The restoring ``surfaceForcingT/S`` of this evaluation.

        ``ice_area`` [batch, ny, nx] is read only when ``restore_under_ice`` is
        False; restoring under ice takes the full flux whatever the ice
        (``forcing_surf_relax.F:73-75,91-104``).
        """
        if not self.restore_under_ice and ice_area is None:
            raise ValueError("surface_climate_relaxation with restore_under_ice=False reads the "
                             "sea-ice fraction (EvaluationInputs.ice_area), which this evaluation "
                             "did not pass")
        temp = state.temp.data
        dtype, device = temp.dtype, temp.device
        drf_surface = grid.layer_thickness.to(dtype=dtype, device=device)[0]
        hfac_surface = grid.masks.hFacC.to(dtype=dtype, device=device).select(dim=-3, index=0)
        zero = torch.zeros_like(temp.select(dim=-3, index=0))

        def flux(tau, tracer_surface, target):
            rate = self._rate(grid, tau, dtype, device)
            target = target.to(dtype=dtype, device=device)
            if self.restore_under_ice:
                return surface_relaxation_flux(rate, tracer_surface, target, drf_surface,
                                               hfac_surface)
            return surface_relaxation_flux_open_water(
                rate, ice_area.to(dtype=dtype, device=device), tracer_surface, target,
                drf_surface, hfac_surface)

        sf_t = flux(self.tau_theta, temp.select(dim=-3, index=0), self._sst) \
            if self.tau_theta > 0.0 else zero
        sf_s = flux(self.tau_salt, state.salt.data.select(dim=-3, index=0), self._sss) \
            if self.tau_salt > 0.0 else zero
        return SurfaceForcing(sfT=sf_t, sfS=sf_s, qsw=zero, penetrating_sw=False)

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        inputs = diagnostics.inputs
        ice_area = None if self.restore_under_ice or inputs is None else inputs.ice_area
        forcing = self.surface_forcing(state, grid, ice_area=ice_area)
        return apply_wpflux_surface_tendency(forcing, state, grid)
