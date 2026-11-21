from __future__ import annotations

from locomgcm.sampling import SampleClock, SamplingRule

from dataclasses import dataclass, field
from typing import Any

import torch

from locomgcm.accel.constant_cache import ConstantCache
from locomgcm.boundaries.open import (
    BOUNDARY_SIDES,
    OB_INDEX_MODES,
    BoundarySide,
    ObcsAlongTables,
    SideBoundaryTensorSet,
    boundary_band_target,
)
from locomgcm.geometry import StructuredCGrid
from locomgcm.diagnostics.reductions import tracer_integrals
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from ._sponge_ob_gate import ObIndexGate
from ._sponge_padding import pad_target_to_field_shape as _pad_target_to_field_shape
from .base import TendencyModule, TendencyPhase, TendencyStepDomain
from locomgcm.dynamics._registry import register_dynamics_module


from locomgcm.dynamics.sponge_weights import (
    SPONGE_MODES,
    band_weights as _band_weights,
    mitgcm_boundary_weight as _mitgcm_boundary_weight,
    mitgcm_inverse_relaxation_time as _mitgcm_inverse_relaxation_time,
)


@dataclass
@register_dynamics_module("tracer_boundary_sponge_relaxation")
class TracerBoundarySpongeRelaxation(ObIndexGate, TendencyModule):
    """Sponge-layer relaxation for tracers near open boundaries.

    Two modes are supported:

    * ``"mitgcm"`` (default) — MITgcm-style dual-timescale sponge with blended
      relaxation target.  Both the timescale and the target value are
      linearly interpolated between the boundary edge and the inner
      edge of the sponge band:

      ``G = -(χ - [(1-l)·χ_BC + l·χ]) / [(1-l)·τ_b + l·τ_i]``

      where *l* = normalised distance from boundary (0 at boundary,
      1 at inner edge), *τ_b* = ``timescale_seconds``, and *τ_i* =
      ``timescale_inner_seconds``.  Reference: MITgcm
      ``pkg/obcs/obcs_sponge.F``.
    * ``"linear"`` — legacy single-timescale linear weight decay.
    """

    sampling_rule = SamplingRule(
        SampleClock.END,
        ("active_temp_values", "active_salt_values"),
        providers=("source",),
    )

    temp_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    salt_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    width: int = 6
    timescale_seconds: float = 21_600.0
    timescale_inner_seconds: float | None = None
    # Per-orientation relaxation timescales (MITgcm obcs_sponge.F rule, see
    # MITgcm pkg/obcs/obcs_sponge.F): OBCS_SPONGE_T/S relax T and S
    # with the SAME orientation timescales as momentum — E/W use Urelaxobcs{bound,
    # inner}, N/S use Vrelaxobcs{bound,inner}. ``timescale_seconds`` /
    # ``timescale_inner_seconds`` are the E/W (Urelax) pair; ``*_ns`` are the N/S
    # (Vrelax) pair and default to the E/W values (existing U==V configs unchanged).
    timescale_seconds_ns: float | None = None
    timescale_inner_seconds_ns: float | None = None
    sponge_mode: str = "mitgcm"
    # MITgcm's OB index table, as for the momentum sponge: OBCS_SPONGE_T and _S relax
    # a column only where it has an OB index (obcs_sponge.F:467-578, :675-786).  Both
    # come from boundary.ob_index_mode and boundary.sides (enrich_manifest);
    # ``full_edge`` leaves every side ungated.
    ob_index_mode: str = "full_edge"
    ob_sides: tuple[str, ...] | None = None
    use_linear_sponge: bool = False
    tracer_names: tuple[str, ...] = ("temp", "salt")
    source: Any | None = None
    time_mode: str = "linear"
    update_interval_seconds: float | None = None
    name: str = "tracer_boundary_sponge"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER
    # MIT writes ``forcing_In_AB=.FALSE.`` so OBCS_SPONGE_T/S is added
    # AFTER Adams-Bashforth via ``apply_forcing.F`` (tracForcingOutAB=1).
    # AssemblyPolicy shares this tracer declaration across evaluation paths.
    ab2_eligible: bool = False
    active_temp_values: SideBoundaryTensorSet = field(init=False, repr=False)
    active_salt_values: SideBoundaryTensorSet = field(init=False, repr=False)
    current_time_seconds: float = field(default=0.0, init=False, repr=False)
    current_sample_metadata: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _ob_tables: ObcsAlongTables | None = field(default=None, init=False, repr=False)
    _ob_tables_grid: Any = field(default=None, init=False, repr=False)
    _ob_gate_cache: ConstantCache = field(
        default_factory=ConstantCache, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.update_interval_seconds is not None and self.update_interval_seconds <= 0.0:
            self.update_interval_seconds = None
        if self.sponge_mode not in SPONGE_MODES:
            raise ValueError(f"sponge_mode must be one of {SPONGE_MODES}, got '{self.sponge_mode}'")
        if self.ob_index_mode not in OB_INDEX_MODES:
            raise ValueError(
                f"ob_index_mode must be one of {list(OB_INDEX_MODES)}, "
                f"got '{self.ob_index_mode}'"
            )
        if self.timescale_inner_seconds is None:
            self.timescale_inner_seconds = 10.0 * self.timescale_seconds
        if self.timescale_seconds_ns is None:
            self.timescale_seconds_ns = self.timescale_seconds
        if self.timescale_inner_seconds_ns is None:
            self.timescale_inner_seconds_ns = self.timescale_inner_seconds
        self.active_temp_values = self.temp_values
        self.active_salt_values = self.salt_values
        if self.source is not None:
            self.update_model_time(0.0)

    def update_model_time(self, model_time_seconds: float) -> None:
        self.current_time_seconds = float(model_time_seconds)
        if self.source is None:
            return
        sample_fields = getattr(self.source, "sample_fields", None)
        if callable(sample_fields) and getattr(self.source, "supports_field_projection", True):
            sample = sample_fields(
                model_time_seconds,
                fields=("temperature", "salinity"),
                mode=self.time_mode,
                update_interval_seconds=self.update_interval_seconds,
            )
            self.active_temp_values = sample.fields["temperature"]
            self.active_salt_values = sample.fields["salinity"]
        else:
            sample = self.source.sample(
                model_time_seconds,
                mode=self.time_mode,
                update_interval_seconds=self.update_interval_seconds,
            )
            self.active_temp_values = sample.temp if hasattr(sample, "temp") else SideBoundaryTensorSet()
            self.active_salt_values = sample.salt if hasattr(sample, "salt") else SideBoundaryTensorSet()
        self.current_sample_metadata = sample.interpolation.as_dict()

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "TracerBoundarySpongeRelaxation":
        self.temp_values = self.temp_values.to(device, dtype)
        self.salt_values = self.salt_values.to(device, dtype)
        self.active_temp_values = self.active_temp_values.to(device, dtype)
        self.active_salt_values = self.active_salt_values.to(device, dtype)
        if self.source is not None:
            if hasattr(self.source, "to_runtime"):
                self.source = self.source.to_runtime(device, dtype)
            elif hasattr(self.source, "to"):
                self.source = self.source.to(device, dtype)
            self.update_model_time(self.current_time_seconds)
        return self

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        if self.width < 1:
            raise ValueError("sponge width must be positive")
        if self.timescale_seconds <= 0.0:
            raise ValueError("sponge timescale_seconds must be positive")

        temp_tendency = torch.zeros_like(state.temp.data)
        salt_tendency = torch.zeros_like(state.salt.data)
        if "temp" in self.tracer_names:
            temp_tendency = self._relax_field(state.temp.data, self.active_temp_values, grid)
        if "salt" in self.tracer_names:
            salt_tendency = self._relax_field(state.salt.data, self.active_salt_values, grid)
        dtype = state.temp.data.dtype
        device = state.temp.data.device
        temp_rate, salt_rate = tracer_integrals(
            temp_tendency,
            salt_tendency,
            grid.layer_thickness.to(dtype=dtype, device=device).view(1, grid.nz, 1, 1),
            grid.metrics.cell_area.to(dtype=dtype, device=device).view(1, 1, grid.ny, grid.nx),
            None if grid.masks.hFacC is None else grid.masks.hFacC.to(dtype=dtype, device=device),
        )
        diagnostics.scalars["temp_sponge_tendency_integral_rate"] = temp_rate
        diagnostics.scalars["salt_sponge_tendency_integral_rate"] = salt_rate
        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data),
            temp=temp_tendency,
            salt=salt_tendency,
        )

    def _relax_field(
        self,
        field: torch.Tensor,
        side_values: SideBoundaryTensorSet,
        grid: StructuredCGrid,
    ) -> torch.Tensor:
        tendency = torch.zeros_like(field)
        # ``for side in BoundarySide`` is EnumType.__iter__,
        # untraceable.  BOUNDARY_SIDES is the same members in the same
        # order, so the accumulation order below is unchanged.
        for side in BOUNDARY_SIDES:
            values = side_values.get(side)
            if values is None:
                continue
            tendency = tendency + self._side_tendency(field, values, side, grid)
        # The centre mask is the only land exclusion (numerics rule 13), for
        # 2-D and 3-D masks alike.  It selects zero on land rather than
        # multiplying by the 0/1 mask, which would keep a NaN target or field
        # value of a land cell in the band (0 * NaN is NaN; rule 10).  The bool
        # mask is read directly: it is already on the compute device, and the
        # memoized float mask would be a cold cache build inside a traced step
        # on grids whose masks are 2-D.
        wet = grid.masks.center.to(device=field.device, dtype=torch.bool)
        return torch.where(wet, tendency, torch.zeros_like(tendency))

    def _side_tendency(
        self,
        field: torch.Tensor,
        values: torch.Tensor,
        side: BoundarySide,
        grid: StructuredCGrid,
    ) -> torch.Tensor:
        width = (
            self._active_width(field, side)
            if self.sponge_mode == "mitgcm"
            else min(self.width, field.shape[self._normal_axis(side)])
        )
        if width < 1:
            return torch.zeros_like(field)
        target = boundary_band_target(
            values,
            side,
            grid,
            batch_size=field.shape[0],
            dtype=field.dtype,
            device=field.device,
            width=width,
        )
        target = self._pad_target_to_field(target, field, grid)

        if self.sponge_mode == "mitgcm":
            tendency = self._mitgcm_side_tendency(field, target, side, width)
        else:
            tendency = self._linear_side_tendency(field, target, side, width)
        return self._gate_along_open_boundary(tendency, side, grid)

    @staticmethod
    def _normal_axis(side: BoundarySide) -> int:
        return -1 if side in (BoundarySide.WEST, BoundarySide.EAST) else -2

    def _active_width(self, field: torch.Tensor, side: BoundarySide) -> int:
        axis = self._normal_axis(side)
        return min(self.width, max(field.shape[axis] - 1, 0))

    def _band_slice(self, field: torch.Tensor, side: BoundarySide, width: int) -> tuple[slice, ...]:
        axis = self._normal_axis(side)
        axis_size = field.shape[axis]
        if side in (BoundarySide.WEST, BoundarySide.SOUTH):
            side_slice = slice(1, 1 + width)
        else:
            side_slice = slice(axis_size - 1 - width, axis_size - 1)
        index = [slice(None)] * field.ndim
        index[axis] = side_slice
        return tuple(index)

    def _legacy_edge_band_slice(
        self,
        field: torch.Tensor,
        side: BoundarySide,
        width: int,
    ) -> tuple[slice, ...]:
        axis = self._normal_axis(side)
        if side in (BoundarySide.WEST, BoundarySide.SOUTH):
            side_slice = slice(None, width)
        else:
            side_slice = slice(field.shape[axis] - width, None)
        index = [slice(None)] * field.ndim
        index[axis] = side_slice
        return tuple(index)

    @staticmethod
    def _pad_target_to_field(
        target: torch.Tensor,
        field: torch.Tensor,
        grid: StructuredCGrid,
    ) -> torch.Tensor:
        del grid
        return _pad_target_to_field_shape(target, field)

    # -- Linear mode (original) ------------------------------------------------

    def _linear_side_tendency(
        self,
        field: torch.Tensor,
        target: torch.Tensor,
        side: BoundarySide,
        width: int,
    ) -> torch.Tensor:
        weights = _band_weights(width, dtype=field.dtype, device=field.device)
        scale = field.new_tensor(1.0 / self.timescale_seconds)
        tendency = torch.zeros_like(field)
        band_slice = self._legacy_edge_band_slice(field, side, width)
        band = field[band_slice]
        target_band = target[band_slice]
        if side == BoundarySide.WEST:
            tendency[band_slice] = scale * (target_band - band) * weights.view(1, 1, 1, width)
        elif side == BoundarySide.EAST:
            tendency[band_slice] = scale * (target_band - band) * weights.flip(0).view(1, 1, 1, width)
        elif side == BoundarySide.SOUTH:
            tendency[band_slice] = scale * (target_band - band) * weights.view(1, 1, width, 1)
        else:
            tendency[band_slice] = scale * (target_band - band) * weights.flip(0).view(1, 1, width, 1)
        return tendency

    # -- MITgcm mode (dual-timescale with blended target) ----------------------

    def _mitgcm_side_tendency(
        self,
        field: torch.Tensor,
        boundary_target: torch.Tensor,
        side: BoundarySide,
        width: int,
    ) -> torch.Tensor:
        """MITgcm obcs_sponge.F formula:

        G = -(χ - [(1-l)·χ_BC + l·χ]) / [(1-l)·τ_b + l·τ_i]
          = -(1-l)·(χ - χ_BC) / [(1-l)·τ_b + l·τ_i]

        l = δL/L ∈ [0,1], 0 at boundary, 1 at inner edge.

        Land is excluded by the centre mask that :meth:`_relax_field` applies
        with ``torch.where`` (numerics rule 10), not by the boundary value: MITgcm dropped its ``IF (OB?t.NE. 0)`` guard in
        ``obcs_sponge.F`` (PR #990, in checkpoint69q), so a wet band
        cell whose prescribed value is exactly zero is relaxed toward zero, the
        same as any other value; a zero target is a valid ocean value.
        """
        boundary_weight = _mitgcm_boundary_weight(width, dtype=field.dtype, device=field.device)
        # Orientation-based relax timescale (MIT obcs_sponge.F): E/W use Urelax
        # (timescale_seconds*), N/S use Vrelax (timescale_seconds_ns*).
        if side in (BoundarySide.WEST, BoundarySide.EAST):
            tau_boundary = self.timescale_seconds
            tau_inner = float(self.timescale_inner_seconds)
        else:  # SOUTH, NORTH
            tau_boundary = float(self.timescale_seconds_ns)
            tau_inner = float(self.timescale_inner_seconds_ns)
        inv_tau = _mitgcm_inverse_relaxation_time(
            width,
            tau_boundary=tau_boundary,
            tau_inner=tau_inner,
            dtype=field.dtype,
            device=field.device,
        )

        tendency = torch.zeros_like(field)
        band_slice = self._band_slice(field, side, width)
        band = field[band_slice]
        target = boundary_target[band_slice]

        if side == BoundarySide.WEST:
            if self.use_linear_sponge:
                blended = target
            else:
                blended = (
                    boundary_weight.view(1, 1, 1, width) * target
                    + (1.0 - boundary_weight).view(1, 1, 1, width) * band
                )
            tendency[band_slice] = -(band - blended) * inv_tau.view(1, 1, 1, width)
        elif side == BoundarySide.EAST:
            boundary_weight_east = boundary_weight.flip(0)
            inv_tau_east = inv_tau.flip(0)
            if self.use_linear_sponge:
                blended = target
            else:
                blended = (
                    boundary_weight_east.view(1, 1, 1, width) * target
                    + (1.0 - boundary_weight_east).view(1, 1, 1, width) * band
                )
            tendency[band_slice] = -(band - blended) * inv_tau_east.view(1, 1, 1, width)
        elif side == BoundarySide.SOUTH:
            if self.use_linear_sponge:
                blended = target
            else:
                blended = (
                    boundary_weight.view(1, 1, width, 1) * target
                    + (1.0 - boundary_weight).view(1, 1, width, 1) * band
                )
            tendency[band_slice] = -(band - blended) * inv_tau.view(1, 1, width, 1)
        else:  # NORTH
            boundary_weight_north = boundary_weight.flip(0)
            inv_tau_north = inv_tau.flip(0)
            if self.use_linear_sponge:
                blended = target
            else:
                blended = (
                    boundary_weight_north.view(1, 1, width, 1) * target
                    + (1.0 - boundary_weight_north).view(1, 1, width, 1) * band
                )
            tendency[band_slice] = -(band - blended) * inv_tau_north.view(1, 1, width, 1)
        return tendency
