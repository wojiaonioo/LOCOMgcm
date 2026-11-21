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
    compute_obcs_balanced_normal_velocities,
)
from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.grid import _side_is_open as _open_side_is_open
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import MomentumTermClass, TendencyModule, TendencyStepDomain
from locomgcm.dynamics._registry import register_dynamics_module


from ._sponge_padding import pad_target_to_field_shape as _pad_target_to_field_shape
from ._sponge_ob_gate import ObIndexGate
from locomgcm.dynamics.sponge_weights import (
    SPONGE_MODES,
    band_weights as _band_weights,
    mitgcm_boundary_weight as _mitgcm_boundary_weight,
    mitgcm_inverse_relaxation_time as _mitgcm_inverse_relaxation_time,
)


@dataclass
@register_dynamics_module("momentum_boundary_sponge_relaxation")
class MomentumBoundarySpongeRelaxation(ObIndexGate, TendencyModule):
    """Sponge-layer relaxation for momentum near open boundaries.

    Supports ``sponge_mode="mitgcm"`` (default) and ``sponge_mode="linear"``
    (legacy single-timescale decay).  See
    :class:`TracerBoundarySpongeRelaxation` for details.
    """

    sampling_rule = SamplingRule(
        SampleClock.END,
        ("active_u_values", "active_v_values", "_balanced_u_values", "_balanced_v_values", "_prepared_balanced_targets"),
        providers=("source", "tidal_config"),
    )

    u_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    v_values: SideBoundaryTensorSet = field(default_factory=SideBoundaryTensorSet)
    width: int = 6
    timescale_seconds: float = 21_600.0  # Default 6 hours
    timescale_inner_seconds: float | None = None
    # Per-orientation relaxation timescales (MITgcm obcs_sponge.F rule, see
    # MITgcm pkg/obcs/obcs_sponge.F): East/West open boundaries
    # relax with Urelaxobcs{bound,inner}, North/South with Vrelaxobcs{bound,inner}
    # — orientation-based, applied to BOTH u and v in that side's band. The
    # plain ``timescale_seconds`` / ``timescale_inner_seconds`` are the E/W
    # (Urelax) pair; ``*_ns`` are the N/S (Vrelax) pair and default to the E/W
    # values so existing U==V configs keep their current behaviour.
    timescale_seconds_ns: float | None = None
    timescale_inner_seconds_ns: float | None = None
    sponge_mode: str = "mitgcm"
    # Which points ALONG an open side actually carry an open boundary — MITgcm's
    # OB_J*/OB_I* index table, carried by the case as ``boundary.ob_index_mode``
    # and injected here by ``io/runner/_canonical_helpers.enrich_manifest``.
    # ``full_edge`` (default) = the whole side is open = the pre-gate behaviour,
    # bit for bit; ``wet_only`` = a column/row is open only where its outermost
    # tracer cell is ocean.  The policy is a property of the case, NOT of the
    # geometry and NOT of ``sponge_mode``: MITgcm can and does declare an entire
    # edge open on a domain with dry boundary cells
    # (the SCS reference configuration's ``data.obcs``).  See
    # :func:`~locomgcm.boundaries.open.resolve_ob_along_masks`.
    ob_index_mode: str = "full_edge"
    # Which sides carry an open boundary at all, i.e. which sides MITgcm writes an
    # OB table for.  Source = the SAME ``boundary.sides`` the data.obcs export
    # reads (injected alongside ob_index_mode by enrich_manifest); deriving it
    # from ``grid.boundary_topology`` here instead would give the two ends two
    # different answers on a case whose topology is open on an axis that
    # boundary.sides does not list (SCS_CLM's western wall is exactly that shape).
    # ``None`` = undeclared (direct construction, no run_cfg) => every side is a
    # candidate.  A side outside this set gets a ``None`` table and therefore the
    # ungated pre-gate behaviour; which sides the sponge is *applied* to at all is
    # a separate, unchanged question answered by ``_relax_field``.
    ob_sides: tuple[str, ...] | None = None
    use_linear_sponge: bool = False
    zero_tangential_targets: bool = True
    source: Any | None = None
    time_mode: str = "linear"
    update_interval_seconds: float | None = None
    name: str = "momentum_boundary_sponge"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    momentum_term_class = MomentumTermClass.FORCING
    # When True, sponge target arrays are balanced via
    # :func:`compute_obcs_balanced_normal_velocities` (MIT
    # OBCS_BALANCE_FLOW analogue) before relaxation, matching the
    # balanced OB arrays consumed by ``OBCS_APPLY_UV``.
    balance_obcs_targets: bool = True
    # OBCS Tides: shared tidal config so sponge targets include tidal overlay,
    # matching MITgcm where OBCS_SPONGE_U/V reads tide-modified OB arrays.
    tidal_config: Any | None = None
    active_u_values: SideBoundaryTensorSet = field(init=False, repr=False)
    active_v_values: SideBoundaryTensorSet = field(init=False, repr=False)
    current_time_seconds: float = field(default=0.0, init=False, repr=False)
    current_sample_metadata: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    # OBCS_BALANCE_FLOW cache for sponge targets — analogous to
    # ``Prescribed3DBoundaryCondition._get_balanced_targets`` so that the
    # sponge and the boundary clamp consume the same balanced arrays.
    _balanced_u_values: SideBoundaryTensorSet | None = field(default=None, init=False, repr=False)
    _balanced_v_values: SideBoundaryTensorSet | None = field(default=None, init=False, repr=False)
    _balance_cache_key: tuple | None = field(default=None, init=False, repr=False)
    # The eager pre-step's answer, as ``(bal_u, bal_v, dtype, device)``.
    # ``None`` = nothing prepared for this clock => the legacy in-step compute.
    _prepared_balanced_targets: tuple | None = field(default=None, init=False, repr=False)
    # Along-boundary tables, resolved once per grid and then reused: MIT's guard
    # is a per-column integer test, so recomputing a depth reduction over
    # ``grid.masks`` on every side / component / step would be pure waste.
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
        self.active_u_values = self.u_values
        self.active_v_values = self.v_values
        if self.source is not None:
            self.update_model_time(0.0)

    def update_model_time(self, model_time_seconds: float) -> None:
        self.current_time_seconds = float(model_time_seconds)
        # Active-source values about to change → invalidate balanced-targets cache.
        self._balanced_u_values = None
        self._balanced_v_values = None
        self._balance_cache_key = None
        self._prepared_balanced_targets = None
        if self.source is None:
            return
        sample_fields = getattr(self.source, "sample_fields", None)
        if callable(sample_fields) and getattr(self.source, "supports_field_projection", True):
            sample = sample_fields(
                model_time_seconds,
                fields=("u", "v"),
                mode=self.time_mode,
                update_interval_seconds=self.update_interval_seconds,
            )
            u_values, v_values = sample.fields["u"], sample.fields["v"]
            self.active_u_values = u_values if u_values is not None else SideBoundaryTensorSet()
            self.active_v_values = v_values if v_values is not None else SideBoundaryTensorSet()
        else:
            sample = self.source.sample(
                model_time_seconds,
                mode=self.time_mode,
                update_interval_seconds=self.update_interval_seconds,
            )
            self.active_u_values = sample.u if hasattr(sample, "u") and sample.u is not None else SideBoundaryTensorSet()
            self.active_v_values = sample.v if hasattr(sample, "v") and sample.v is not None else SideBoundaryTensorSet()
        self.current_sample_metadata = sample.interpolation.as_dict()

    def prepare_balanced_targets(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Eager pre-step: resolve the sponge relaxation targets for this clock.

        Compiled execution observed two graph breaks at
        ``momentum_sponge.py:188:_get_balanced_targets``).  The in-step memo is
        keyed on ``(id(active_u_values), id(active_v_values),
        current_time_seconds, str(dtype), str(device))``: ``id()`` of a Python
        object is untraceable and the float clock bakes a fresh guard every
        step, so ``compute_obcs_balanced_normal_velocities`` was re-traced
        inside the compiled step.  Producing it here -- same sampled side
        values, same grid, same dtype/device, same call -- keeps every bit and
        leaves the step a plain attribute read.  Called by
        ``TorchRegionalOceanModel.prepare_step_times`` for both clocks; that
        method installs the ``future`` sample on the sponge modules and the
        ``current`` sample on everything else before the step is entered, so the
        sponge reads the future targets while the faces stay at current.
        """
        bal_u, bal_v = self._compute_balanced_targets(
            grid, dtype=dtype, device=device
        )
        self._prepared_balanced_targets = (bal_u, bal_v, dtype, device)

    def _get_balanced_targets(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> tuple[SideBoundaryTensorSet, SideBoundaryTensorSet]:
        """Return (balanced_u, balanced_v) target sets for sponge relaxation.

        Serves :meth:`prepare_balanced_targets`' answer when the eager pre-step
        produced one at this dtype/device; otherwise falls back to the legacy
        in-step compute (direct ``evaluate`` callers never run the pre-step).
        """
        prepared = self._prepared_balanced_targets
        if (
            prepared is not None
            and prepared[2] == dtype
            and prepared[3] == device
        ):
            return prepared[0], prepared[1]
        return self._compute_balanced_targets(grid, dtype=dtype, device=device)

    @torch._dynamo.disable
    def _compute_balanced_targets(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> tuple[SideBoundaryTensorSet, SideBoundaryTensorSet]:
        """Balanced (or raw) target sets, computed here and memoised.

        When :attr:`balance_obcs_targets` is False, returns the raw active
        sets unchanged.  Otherwise lazily computes and caches per
        (active values, time, dtype, device); the cache is invalidated by
        :meth:`update_model_time`.  Sharing the same balanced sets with
        :class:`Prescribed3DBoundaryCondition._get_balanced_targets`
        guarantees both sponge and face clamp see identical mass-balanced
        OB targets, matching MIT topology
        (``obcs_adjust.F:42-45 → obcs_balance_flow.F``).

        Opaque to Dynamo: the memo key holds ``id()`` of the sampled sets and
        the float clock, and the tidal branch reads the clock.  The eager
        prepare serves the phase-split steppers (``prepare_balanced_targets``);
        AB2, SSPRK3 and split-explicit reach this from their traced ``evaluate``,
        where it guarded on the identity of every new sample and recompiled.
        """
        if not self.balance_obcs_targets:
            u, v = self.active_u_values, self.active_v_values
            if self.tidal_config is not None:
                from locomgcm.boundaries.tidal import add_tidal_velocities
                u, v = add_tidal_velocities(u, v, grid, self.current_time_seconds, self.tidal_config)
            return u, v
        cache_key = (
            id(self.active_u_values),
            id(self.active_v_values),
            self.current_time_seconds,
            str(dtype),
            str(device),
        )
        if self._balance_cache_key == cache_key and self._balanced_u_values is not None:
            return self._balanced_u_values, self._balanced_v_values
        bal_u, bal_v = compute_obcs_balanced_normal_velocities(
            self.active_u_values,
            self.active_v_values,
            grid,
            dtype=dtype,
            device=device,
        )
        # OBCS_ADD_TIDES: post-balance tidal overlay for sponge targets.
        # MITgcm OBCS_SPONGE_U/V reads tide-modified OB arrays (obcs_sponge.F:72-77).
        if self.tidal_config is not None:
            from locomgcm.boundaries.tidal import add_tidal_velocities
            bal_u, bal_v = add_tidal_velocities(
                bal_u, bal_v, grid, self.current_time_seconds, self.tidal_config,
            )
        self._balanced_u_values = bal_u
        self._balanced_v_values = bal_v
        self._balance_cache_key = cache_key
        return bal_u, bal_v

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "MomentumBoundarySpongeRelaxation":
        self.u_values = self.u_values.to(device, dtype)
        self.v_values = self.v_values.to(device, dtype)
        self.active_u_values = self.active_u_values.to(device, dtype)
        self.active_v_values = self.active_v_values.to(device, dtype)
        if self.tidal_config is not None:
            self.tidal_config = self.tidal_config.to(device, dtype)
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

        balanced_u_values, balanced_v_values = self._get_balanced_targets(
            grid, dtype=state.u.data.dtype, device=state.u.data.device,
        )
        u_central_tendency = self._relax_field(state.u.data, balanced_u_values, grid, is_u_face=True)
        v_central_tendency = self._relax_field(state.v.data, balanced_v_values, grid, is_u_face=False)

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=u_central_tendency,
            v=v_central_tendency,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )

    def _relax_field(
        self,
        field: torch.Tensor,
        side_values: SideBoundaryTensorSet,
        grid: StructuredCGrid,
        is_u_face: bool,
    ) -> torch.Tensor:
        tendency = torch.zeros_like(field)
        # ``for side in BoundarySide`` is EnumType.__iter__,
        # untraceable.  BOUNDARY_SIDES is the same members in the same
        # order, so the accumulation order below is unchanged.
        for side in BOUNDARY_SIDES:
            values = side_values.get(side)
            if values is None and not self._side_is_open(side, grid):
                continue
            if self.zero_tangential_targets and self._is_tangential_side(side, is_u_face):
                # MIT data.obcs in the aligned regional runs omits tangential
                # velocity files, leaving OBNu/OBSu/OBEv/OBWv initialized to 0.
                values = (
                    torch.zeros_like(values)
                    if values is not None
                    else self._zero_side_values(side, grid, dtype=field.dtype, device=field.device)
                )
            elif values is None:
                continue
            tendency = tendency + self._side_tendency(field, values, side, grid, is_u_face)
        if grid.masks.has_3d_masks:
            # Select zero on dry faces rather than multiplying by the 0/1 face mask: a
            # NaN target or velocity on a dry face in the band would survive 0 * NaN
            # (numerics rule 10; the tracer sponge selects the same way,
            # tracer_sponge.py).  Wet faces keep the tendency bit for bit; a dry face
            # gets +0.0 where the product gave -0.0 for a negative tendency.
            wet = (grid.masks.u if is_u_face else grid.masks.v).to(
                device=field.device, dtype=torch.bool
            )
            tendency = torch.where(wet, tendency, torch.zeros_like(tendency))
        return tendency

    def _side_tendency(
        self,
        field: torch.Tensor,
        values: torch.Tensor,
        side: BoundarySide,
        grid: StructuredCGrid,
        is_u_face: bool,
    ) -> torch.Tensor:
        width = (
            self._active_width(field, side, is_u_face)
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

        target = _pad_target_to_field_shape(target, field)

        if self.sponge_mode == "mitgcm":
            tendency = self._mitgcm_side_tendency(field, target, side, width, is_u_face)
        else:
            tendency = self._linear_side_tendency(field, target, side, width)
        return self._gate_along_open_boundary(tendency, side, grid)

    @staticmethod
    def _is_tangential_side(side: BoundarySide, is_u_face: bool) -> bool:
        if is_u_face:
            return side in (BoundarySide.SOUTH, BoundarySide.NORTH)
        return side in (BoundarySide.WEST, BoundarySide.EAST)

    @staticmethod
    def _side_is_open(side: BoundarySide, grid: StructuredCGrid) -> bool:
        return _open_side_is_open(side.value, grid)

    @staticmethod
    def _zero_side_values(
        side: BoundarySide,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        side_len = grid.ny if side in (BoundarySide.WEST, BoundarySide.EAST) else grid.nx
        return torch.zeros((grid.nz, side_len), dtype=dtype, device=device)

    @staticmethod
    def _normal_axis(side: BoundarySide) -> int:
        return -1 if side in (BoundarySide.WEST, BoundarySide.EAST) else -2

    def _mitgcm_outer_skip(self, side: BoundarySide, is_u_face: bool) -> int:
        """Number of outer tensor cells/faces skipped before MIT sponge starts.

        Normal sides skip 2, tangential sides skip 1.

        LOCOMgcm stores an extra outer face at east/north (U: ny by nx+1;
        V: ny+1 by nx). MITgcm’s one-based skip=1 therefore maps to skip=2
        in these zero-based arrays. Using skip=1 would move the sponge band
        one cell toward the open boundary and relax an adjacent face that
        must retain its prescribed value.
        """
        if not self._is_tangential_side(side, is_u_face):
            return 2
        return 1

    def _active_width(self, field: torch.Tensor, side: BoundarySide, is_u_face: bool) -> int:
        axis = self._normal_axis(side)
        skip = self._mitgcm_outer_skip(side, is_u_face)
        return min(self.width, max(field.shape[axis] - skip, 0))

    def _band_slice(
        self,
        field: torch.Tensor,
        side: BoundarySide,
        width: int,
        is_u_face: bool,
    ) -> tuple[slice, ...]:
        axis = self._normal_axis(side)
        skip = self._mitgcm_outer_skip(side, is_u_face)
        axis_size = field.shape[axis]
        if side in (BoundarySide.WEST, BoundarySide.SOUTH):
            side_slice = slice(skip, skip + width)
        else:
            side_slice = slice(axis_size - skip - width, axis_size - skip)
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

    # -- Linear mode (original) ------------------------------------------------

    def _linear_side_tendency(
        self,
        field: torch.Tensor,
        target: torch.Tensor,
        side: BoundarySide,
        width: int,
        is_u_face: bool | None = None,
    ) -> torch.Tensor:
        weights = _band_weights(width, dtype=field.dtype, device=field.device)
        scale = field.new_tensor(1.0 / self.timescale_seconds)
        tendency = torch.zeros_like(field)
        band_slice = (
            self._legacy_edge_band_slice(field, side, width)
            if is_u_face is None
            else self._band_slice(field, side, width, is_u_face)
        )
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
        is_u_face: bool | None = None,
    ) -> torch.Tensor:
        """MITgcm obcs_sponge.F formula for momentum."""
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
        band_slice = (
            self._legacy_edge_band_slice(field, side, width)
            if is_u_face is None
            else self._band_slice(field, side, width, is_u_face)
        )
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
            boundary_weight_e = boundary_weight.flip(0)
            inv_tau_e = inv_tau.flip(0)
            if self.use_linear_sponge:
                blended = target
            else:
                blended = (
                    boundary_weight_e.view(1, 1, 1, width) * target
                    + (1.0 - boundary_weight_e).view(1, 1, 1, width) * band
                )
            tendency[band_slice] = -(band - blended) * inv_tau_e.view(1, 1, 1, width)
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
            boundary_weight_n = boundary_weight.flip(0)
            inv_tau_n = inv_tau.flip(0)
            if self.use_linear_sponge:
                blended = target
            else:
                blended = (
                    boundary_weight_n.view(1, 1, width, 1) * target
                    + (1.0 - boundary_weight_n).view(1, 1, width, 1) * band
                )
            tendency[band_slice] = -(band - blended) * inv_tau_n.view(1, 1, width, 1)
        return tendency
