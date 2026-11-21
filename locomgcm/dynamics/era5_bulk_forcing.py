"""Time-varying ERA5 bulk-formula surface forcing as a TendencyModule.

Each step: sample the (model-grid) ERA5 atmospheric state at the current model
time, compute the net surface fluxes from the EXF bulk formulae using the
*model* SST (``state.temp[0]``), and apply them as MITgcm surface tracer forcing
(non-shortwave surface heat + shortwave penetration + virtual salt flux).

Wind-derived momentum stress and thermal/freshwater fluxes are produced by
the same bulk iteration. ``era5_wind_forcing`` maps that stress to faces and
applies momentum; this module applies the tracer part.

The bulk outputs are MITgcm upward-positive (``Qnet``/``Qsw``>0 cool, ``empmr``>0 =
net evaporation; docs/NUMERICS.md), and the apply is the shared single source
:func:`~locomgcm.dynamics.surface_flux.apply_wpflux_surface_tendency` (same
operator as :class:`~locomgcm.dynamics.surface_flux.WPFluxSurfaceForcing` and
the cube stepper), so the fields are never re-derived independently.
"""
from __future__ import annotations

from locomgcm.sampling import SampleClock, SamplingRule

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency
from locomgcm.constants import RHO_CONST_DEFAULT

from .base import TendencyModule, TendencyPhase, TendencyStepDomain
from .bulk_formulae import BulkFluxes, ExfBulkConstants, exf_bulk_fluxes, exf_longwave, exf_shortwave
from .surface_flux import apply_wpflux_surface_tendency, surface_forcing_from_bulk
from locomgcm.dynamics._registry import register_dynamics_module

if TYPE_CHECKING:
    from locomgcm.io.era5 import ERA5BulkInputs, ERA5BulkSource, ERA5WindSource
    from .surface_flux import SurfaceForcing


@dataclass
@register_dynamics_module("era5_bulk_forcing")
class ERA5BulkForcing(TendencyModule):
    """Apply time-varying ERA5 bulk-formula heat/salt/shortwave surface forcing.

    The atmospheric inputs come from an :class:`ERA5BulkSource` whose fields are
    ALREADY regridded to the model grid (so :meth:`ERA5BulkSource.sample` does
    time interpolation only — the MIT-EXF anti-constant-hold guard).  Each step,
    ``exf_bulk_fluxes`` converts (atmo, model SST) → MITgcm upward-positive
    ``Qnet``/``Qsw``/``EmPmR``, then ``apply_wpflux_surface_tendency`` applies them.

    ``rho_const`` is the Boussinesq ``rhoConst`` (mass2rUnit = 1/ρ₀); ``rho_const_fresh``
    is the freshwater density used for the EXF evaporation and the EmPmR
    volume→mass conversion.  Both default to ``RHO_CONST_DEFAULT``; MIT defaults
    ``rhoConstFresh`` to ``rhoConst`` (``ini_parms.F:484``) — keep both at the case ``rhoConst``.

    ``stress_source`` is the same cell-centre prescribed-stress source used by
    ``era5_wind_forcing``. Thermal transfer uses its step-start sample before
    clipping/reconstruction, as MITgcm EXF with ``useAtmWind=False`` does.

    ``constants`` are the data.exf constants of the bulk formulae, the EXF
    defaults for an ERA5 source.
    """

    @property
    def sampling_rule(self) -> SamplingRule:
        # Wind-derived stress is produced inside the step, not sampled here.
        sampled = "_atmo" if self.source.samples_on_host else "_sample_time"
        inputs = (sampled,) if self.use_atm_wind else (
            sampled, "_stress_x", "_stress_y",
        )
        return SamplingRule(
            SampleClock.START, inputs, providers=("source", "stress_source"),
        )

    # Constant discovery must restore derived outputs as well as sampled inputs.
    # Otherwise its no-grad products survive into the first real step.
    prewarm_cache_fields = ("_current_forcing", "_current_flux")

    source: "ERA5BulkSource"
    constants: ExfBulkConstants = field(kw_only=True)
    use_atm_wind: bool = field(default=True, kw_only=True)
    stress_source: "ERA5WindSource | None" = field(default=None, kw_only=True)
    rho_const: float = RHO_CONST_DEFAULT
    rho_const_fresh: float = RHO_CONST_DEFAULT
    heat_capacity: float = 3994.0
    convert_fw2salt: float = 35.0
    salt_evprn: float = 0.0
    penetrating_sw: bool = True
    jwtype: int = 2
    no_negative_evap: bool = False
    name: str = "era5_bulk_forcing"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER
    # MIT tracForcingOutAB=1: surface tracer forcing applied AFTER AB2.
    ab2_eligible: bool = False
    current_time_seconds: float = field(default=0.0, init=False, repr=False)
    _sample_time: torch.Tensor | None = field(default=None, init=False, repr=False)
    _atmo: "ERA5BulkInputs | None" = field(default=None, init=False, repr=False)
    _stress_x: torch.Tensor | None = field(default=None, init=False, repr=False)
    _stress_y: torch.Tensor | None = field(default=None, init=False, repr=False)
    # Per-step SurfaceForcing bundle (sfT/sfS/qsw, MITgcm conventions), exposed
    # so the KPP forcing cache reads the SAME produced fields the tracer apply
    # uses (single produce point; mirrors era5_wind_forcing's _current_tau_x/y).
    # Set by compute_surface_fluxes(); None until the first step.
    _current_forcing: "SurfaceForcing | None" = field(default=None, init=False, repr=False)
    _current_flux: BulkFluxes | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.use_atm_wind and self.stress_source is None:
            raise ValueError("prescribed-stress ERA5 bulk forcing requires stress_source")

    def update_model_time(self, time_seconds: float) -> None:
        self._current_forcing = None
        self._current_flux = None
        self.current_time_seconds = float(time_seconds)
        if self.source.samples_on_host:
            # A source that picks its records on the host samples here,
            # outside the compiled region.
            self._atmo = self.source.sample(time_seconds)
        else:
            # Upload only the clock here. Interpolation of the resident records
            # belongs to the same compiled region as the bulk calculation.
            self._sample_time = self.source.prepare_sample_time(time_seconds)
        if not self.use_atm_wind:
            self._stress_x, self._stress_y = self.stress_source.sample(time_seconds)

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "ERA5BulkForcing":
        if hasattr(self.source, "to_runtime"):
            self.source = self.source.to_runtime(device, dtype)
        if hasattr(self.stress_source, "to_runtime"):
            self.stress_source = self.stress_source.to_runtime(device, dtype)
        self.update_model_time(self.current_time_seconds)
        return self

    def compute_surface_fluxes(self, state: PrognosticState) -> "SurfaceForcing":
        """Bulk fluxes for the current ERA5 atmo + model SST → SurfaceForcing bundle,
        WITHOUT applying it.

        Exposes ``_current_forcing`` (the produced bundle) so the KPP forcing
        cache reads the SAME sfT/sfS/qsw fields the tracer apply uses, at the
        per-step diagnose point — which runs BEFORE the tendency evaluation that
        would otherwise compute them. ``evaluate`` reuses the returned bundle.

        Wind-derived fluxes are retained for all consumers of this step;
        ``update_model_time`` invalidates them when sampling a new step.
        """
        if self.use_atm_wind and self._current_forcing is not None:
            return self._current_forcing
        dtype = state.temp.data.dtype
        device = state.temp.data.device

        if self.source.samples_on_host:
            atmo = self._atmo
        else:
            sample_time = self._sample_time
            atmo = self.source.sample(
                self.current_time_seconds if sample_time is None else sample_time
            )
        stress_x, stress_y = self._stress_x, self._stress_y
        if not self.use_atm_wind and (stress_x is None or stress_y is None):
            stress_x, stress_y = self.stress_source.sample(self.current_time_seconds)

        def _f(x: torch.Tensor) -> torch.Tensor:
            return x.to(dtype=dtype, device=device)

        # Model SST [°C] at the top layer (bulk converts to K internally via +273.15).
        sst_degc = state.temp.data[:, 0]                                  # [B, ny, nx]

        # exf_radiation.F:60, :107: net radiation from the downward fields only
        # where the source says so; otherwise its lwflux/swflux fields.
        if self.source.longwave_from_down:
            lwflux = exf_longwave(sst_degc + self.constants.cen2kel, _f(atmo.lwdown),
                                  constants=self.constants)
        else:
            lwflux = _f(atmo.lwflux)
        if self.source.shortwave_from_down:
            swflux = exf_shortwave(_f(atmo.swdown), constants=self.constants)
        else:
            swflux = _f(atmo.swflux)
        flux = exf_bulk_fluxes(
            _f(atmo.atemp), _f(atmo.aqh), _f(atmo.uwind), _f(atmo.vwind), _f(atmo.precip),
            sst_degc=sst_degc, lwflux=lwflux, swflux=swflux,
            runoff=_f(atmo.runoff),
            constants=self.constants,
            rho_const_fresh=self.rho_const_fresh,
            no_negative_evap=self.no_negative_evap,
            use_atm_wind=self.use_atm_wind,
            stress_x=None if self.use_atm_wind else _f(stress_x),
            stress_y=None if self.use_atm_wind else _f(stress_y),
        )
        # The one produce point: bulk fluxes → surfaceForcing fields.
        # The wind consumer uses this very iteration's ustress/vstress.
        self._current_flux = flux
        forcing = surface_forcing_from_bulk(
            flux.qnet, flux.qsw, flux.empmr,
            heat_capacity=self.heat_capacity, rho_const=self.rho_const,
            rho_const_fresh=self.rho_const_fresh,
            convert_fw2salt=self.convert_fw2salt, salt_evprn=self.salt_evprn,
            penetrating_sw=self.penetrating_sw,
            surface_salinity=state.salt.data.select(dim=-3, index=0),
        )
        self._current_forcing = forcing
        return forcing

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        del diagnostics
        forcing = self.compute_surface_fluxes(state)
        return apply_wpflux_surface_tendency(forcing, state, grid, jwtype=self.jwtype)
