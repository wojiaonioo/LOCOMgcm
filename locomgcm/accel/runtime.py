"""Per-step gate resolution, provenance and unified ignition accounting."""

from __future__ import annotations

import atexit
import inspect
import json
import os
import sys
from enum import Enum
import warnings
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterator, Literal, Mapping, Sequence

import torch

from .registry import REGISTRY
from .schema import DefaultSource, GateView, IgnitionState

if TYPE_CHECKING:
    from .admission import AdmissionRecord

_SELECTION: AdmissionRecord | None = None
_SELECTION_DEPTH = 0

class Outcome(str, Enum):
    HIT = "hit"
    INAPPLICABLE = "inapplicable"
    ERROR = "error"


class Reason(str, Enum):
    """Stable decline codes plus the existing production vocabulary.

    ERROR_EXCEPTION_CLASS is a schema slot: error events may carry the exact
    ``type(error).__name__`` spelling. It is not a wildcard for declines.
    """
    ERROR_EXCEPTION_CLASS = "error_exception_class"
    INVALID_GRAD_MODE = 'INVALID_GRAD_MODE'
    UNQUALIFIED_CAPABILITY = 'UNQUALIFIED_CAPABILITY'
    UNSUPPORTED_CONTROL_SET = 'UNSUPPORTED_CONTROL_SET'
    UNSUPPORTED_DEVICE = 'UNSUPPORTED_DEVICE'
    UNSUPPORTED_DTYPE = 'UNSUPPORTED_DTYPE'
    UNSUPPORTED_JVP = 'UNSUPPORTED_JVP'
    UNSUPPORTED_LAYOUT = 'UNSUPPORTED_LAYOUT'
    UNSUPPORTED_TOPOLOGY = 'UNSUPPORTED_TOPOLOGY'
    UNSUPPORTED_VJP = 'UNSUPPORTED_VJP'
    ARITY = 'arity'
    AUTOGRAD = 'autograd'
    AXIS = 'axis'
    BACKGROUND_SHAPE = 'background_shape'
    BATCH_BROADCAST = 'batch_broadcast'
    BIHARMONIC = 'biharmonic'
    CACHE_CONSTANTS_DISABLED = 'cache_constants_disabled'
    CAPTURE = 'capture'
    CHECK_INTERVAL = 'check_interval'
    COEFFICIENT_RANK = 'coefficient_rank'
    COEFFICIENT_SHAPE = 'coefficient_shape'
    COLUMN_RANK = 'column_rank'
    COLUMN_SHAPE = 'column_shape'
    COMBINED_CORIOLIS = 'combined_coriolis'
    CUBE = 'cube'
    CUBE_CORNER_OVERRIDE = 'cube_corner_override'
    CUBE_GRID = 'cube_grid'
    # The requested KPP dbloc form (kpp_dbloc_form) is not the one the kernel
    # computes: the STATEKPP and fused KPP kernels evaluate the JMD95 divided
    # difference, so the MITgcm subtraction form runs the reference path.
    DBLOC_FORM = 'dbloc_form'
    DEBUG_STATE = 'debug_state'
    CUSTOM_DOT = 'custom_dot'
    DEGENERATE_EXTENT = 'degenerate_extent'
    DEVICE_MISMATCH = 'device_mismatch'
    DIFFUS_SHAPE = 'diffus_shape'
    DIFFUSIVITY_SHAPE = 'diffusivity_shape'
    DISABLED = 'disabled'
    DT_TYPE = 'dt_type'
    DTYPE = 'dtype'
    DTYPE_MISMATCH = 'dtype_mismatch'
    DTYPE_NOT_FLOAT64 = 'dtype_not_float64'
    DUMP_ENABLED = 'dump_enabled'
    EDGE_FACE_COUPLES = 'edge_face_couples'
    EMIT_TENDENCY = 'emit_tendency'
    EMPTY_SHAPE = 'empty_shape'
    EOS = 'eos'
    ESTIMATE_UREF = 'estimate_uref'
    EXPLICIT_CORIOLIS = 'explicit_coriolis'
    FACE_SLAB_LEADING_DIM_MISMATCH = 'face_slab_leading_dim_mismatch'
    FLOATING_DTYPE = 'floating_dtype'
    FULL_METRIC_CORIOLIS = 'full_metric_coriolis'
    FUSED_THOMAS_RAISED = 'fused_thomas_raised'
    GEOMETRY_GRAD = 'geometry_grad'
    ACTIVE_GEOMETRY_OR_TABLE = 'active_geometry_or_table'
    GRAD = 'grad'
    GRAD_ENABLED = 'grad_enabled'
    GRAPH_CACHE_UNAVAILABLE = 'graph_cache_unavailable'
    GRAPH_CAPTURE_OR_REPLAY_FAILED = 'graph_capture_or_replay_failed'
    GRID_RANK = 'grid_rank'
    GRID_SHAPE = 'grid_shape'
    GRID_VISC_FLOOR_CAP = 'grid_visc_floor_cap'
    GRID_WITHOUT_RUNTIME_CACHE = 'grid_without_runtime_cache'
    HALO_SHAPE = 'halo_shape'
    HAS_MASK_IN_C = 'has_mask_in_c'
    HFAC_TIME_VARYING_NONLIN_FREE_SURF = 'hfac_time_varying_nonlin_free_surf'
    HORIZONTAL_RANK = 'horizontal_rank'
    HORIZONTAL_SHAPE = 'horizontal_shape'
    JWTYPE = 'jwtype'
    KBL_DTYPE = 'kbl_dtype'
    KMTJ_DTYPE = 'kmtj_dtype'
    KMTJ_RANK = 'kmtj_rank'
    KMTJ_SHAPE = 'kmtj_shape'
    LAYER_SHAPE = 'layer_shape'
    LAYOUT = 'layout'
    LEAF_INJECT = 'leaf_inject'
    LEITH_DIV = 'leith_div'
    LIMIT_TYPE = 'limit_type'
    MASK_SHAPE = 'mask_shape'
    METRIC_RANK = 'metric_rank'
    MISSING_RECIP_HFAC = 'missing_recip_hfac'
    NDIM = 'ndim'
    NEGATIVE_STRIDE = 'negative_stride'
    NO_ENCLOSING_COLLECTIVE_SOLVE = 'no_enclosing_collective_solve'
    NO_3D_MASKS = 'no_3d_masks'
    NO_HALO_INJECTION = 'no_halo_injection'
    NO_HFAC = 'no_hfac'
    NO_SLIP_SIDES = 'no_slip_sides'
    NON_CONTIGUOUS = 'non_contiguous'
    NON_CUDA = 'non_cuda'
    NON_TENSOR = 'non_tensor'
    OBCS_UPWIND_X = 'obcs_upwind_x'
    OBCS_UPWIND_Y = 'obcs_upwind_y'
    PRECOND_SHAPE = 'precond_shape'
    PROLOGUE_CONVERGED = 'prologue_converged'
    PRESSURE_NON_TENSOR = 'pressure_non_tensor'
    PRESSURE_DEVICE = 'pressure_device'
    PRESSURE_LAYOUT = 'pressure_layout'
    PRESSURE_NEGATIVE_STRIDE = 'pressure_negative_stride'
    PRESSURE_RANK = 'pressure_rank'
    PRESSURE_SHAPE = 'pressure_shape'
    PROFILE_RANK = 'profile_rank'
    QSW_SHAPE = 'qsw_shape'
    RANK = 'rank'
    RECIP_HFAC_SHAPE = 'recip_hfac_shape'
    SCALAR_DIFFUSIVITY = 'scalar_diffusivity'
    SCALAR_SHAPE = 'scalar_shape'
    SCALAR_TYPE = 'scalar_type'
    SCHEME = 'scheme'
    SELECT_3D_CORI_SCHEME = 'select_3d_cori_scheme'
    SELECT_KE_SCHEME = 'select_ke_scheme'
    SELECT_VORT_SCHEME = 'select_vort_scheme'
    SHAPE = 'shape'
    SHAPE_MISMATCH = 'shape_mismatch'
    SMOOTH_DBLOC = 'smooth_dbloc'
    SMOOTH_SHSQ = 'smooth_shsq'
    SPLICE_BOUNDS = 'splice_bounds'
    STATE_RANK = 'state_rank'
    STATE_SHAPE = 'state_shape'
    STENCIL_WIDTH_X = 'stencil_width_x'
    STENCIL_WIDTH_Y = 'stencil_width_y'
    SUPERSEDED_BY_CG2D_CUDA_GRAPH = 'superseded_by_cg2d_cuda_graph'
    SUPERSEDED_BY_CG3D_FUSED_PCG = 'superseded_by_cg3d_fused_pcg'
    SUPERSEDED_BY_ROOT_ADJOINT = 'superseded_by_root_adjoint'
    TABLE_RANK = 'table_rank'
    TABLE_SHAPE = 'table_shape'
    TILE_EXTENT = 'tile_extent'
    TIMESTEP = 'timestep'
    TORCH_TRANSFORM = 'torch_transform'
    TRAILING_SHAPE = 'trailing_shape'
    UNSUPPORTED_BATCH = 'unsupported_batch'
    UNSUPPORTED_EOS = 'unsupported_eos'
    UNSUPPORTED_PRECONDITIONER = 'unsupported_preconditioner'
    UNSUPPORTED_SHAPE = 'unsupported_shape'
    UNSUPPORTED_TOPOLOGY_CROSS_RANK_SEAM = 'unsupported_topology:cross_rank_seam'
    UNSUPPORTED_TOPOLOGY_INCOMPLETE_RANK_GRID = 'unsupported_topology:incomplete_rank_grid'
    UNSUPPORTED_TOPOLOGY_TILE_BELOW_HALO = 'unsupported_topology:tile_below_halo'
    VELOCITY_DTYPE = 'velocity_dtype'
    VERTICAL_RI_SMOOTHING = 'vertical_ri_smoothing'
    VERTICAL_SHAPE = 'vertical_shape'
    VISCOSITY_SCHEME = 'viscosity_scheme'
    VOLUME_RANK = 'volume_rank'
    VOLUME_SHAPE = 'volume_shape'
    WIDENED_TO_FLOAT64 = 'widened_to_float64'
    WITH_LIMITER = 'with_limiter'
    WSCALE_OVERRIDES = 'wscale_overrides'

_OUTCOMES = frozenset(item.value for item in Outcome)
_REASONS = frozenset(item.value for item in Reason)


def _validate_dispatch(outcome: str, reason: str | None) -> None:
    # Constant frozenset membership is traceable. Enum construction is confined
    # to import time, never the Dynamo path. Successful calls need no reason.
    if outcome not in _OUTCOMES:
        raise ValueError(f"unknown dispatch outcome {outcome!r}")
    if outcome == "hit" and reason is None:
        return
    if not reason:
        raise ValueError(f"{outcome} dispatch requires a reason")
    if reason in _REASONS:
        return
    if outcome == "error" and reason.isidentifier():
        # Exception classes can be defined by extension packages; their names
        # cannot be exhaustively listed by this package. Only this error slot
        # accepts that spelling, consistently with type(error).__name__.
        return
    raise ValueError(f"unknown dispatch reason {reason!r}")



_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))
_IVADV_BACKENDS = frozenset(("cuda",))
_FAST_GATES: dict[str, GateView] = {}
_FAST_RANK: int | None = None
_DISPATCH_DEPTH = 0
_PENDING_EVENTS: list[tuple] = []

# The traced step queues its records through an operation instead of appending
# to ``_PENDING_EVENTS`` from traced code.  An append there made Dynamo guard the
# list -- its length and the entries it read back -- in every frame that queued a
# record, so a frame traced while an earlier record read ``cold-construction``
# retraced on the next step (guards such as
# ``len(...runtime._PENDING_EVENTS) == 40`` at ``enforce`` and
# ``_PENDING_EVENTS[5][2] == 'cold-construction'`` at nine frames).  Dynamo traces a call with constant arguments and
# never reads the list; the real kernel appends the record when the graph runs,
# the fake kernel appends nothing, and ``finalize_step`` drains the queue as
# before.  The call mutates a host order token, the ``distributed/halo_p2p.py``
# mechanism: a mutable schema keeps Inductor from deleting a call whose result
# nothing uses and chains the calls in program order.
_ENQUEUE_ORDER_TOKEN = torch.zeros((), dtype=torch.int64)
_ENQUEUE_KWARGS: dict[str, object] = {
    "mutates_args": ("order_token",),
    "schema": (
        "(Tensor(a!) order_token, str kind, str key, str outcome, str variant, "
        "str site, str backend, int rank, str? reason, int count) -> ()"
    ),
}
_CUDA_GRAPH_UNSAFE = getattr(torch._C.Tag, "cudagraph_unsafe", None)
if (
    _CUDA_GRAPH_UNSAFE is not None
    and "tags" in inspect.signature(torch.library.custom_op).parameters
):
    # A Python list append cannot be replayed by a CUDA graph.
    _ENQUEUE_KWARGS["tags"] = (_CUDA_GRAPH_UNSAFE,)


@torch.library.custom_op("locomgcm::enqueue_accel_record", **_ENQUEUE_KWARGS)
def _enqueue_record(
    order_token: torch.Tensor,
    kind: str,
    key: str,
    outcome: str,
    variant: str,
    site: str,
    backend: str,
    rank: int,
    reason: str | None,
    count: int,
) -> None:
    _PENDING_EVENTS.append((kind, key, outcome, variant, site, backend, rank, reason, count))
    order_token.add_(1)


@_enqueue_record.register_fake
def _enqueue_record_fake(
    order_token: torch.Tensor,
    kind: str,
    key: str,
    outcome: str,
    variant: str,
    site: str,
    backend: str,
    rank: int,
    reason: str | None,
    count: int,
) -> None:
    del order_token, kind, key, outcome, variant, site, backend, rank, reason, count

#: Switches whose first ``error`` outcome has already been announced by
#: :func:`_warn_first_default_on_error`.  Deliberately NOT cleared by
#: :func:`reset_runtime`: the warning is a once-per-process announcement, and a
#: run that resets its accounting has not un-broken the extension.
_DEFAULT_ON_ERROR_ANNOUNCED: set[str] = set()


def _clear_fast_snapshot() -> None:
    global _FAST_RANK
    _FAST_GATES.clear()
    _FAST_RANK = None


@dataclass(frozen=True)
class StepGates:
    cache_constants: bool
    cg2d_cuda_graph: bool
    cg2d_fused_pcg: bool
    cg3d_fused_pcg: bool
    cg3d_fast_thomas: bool
    compile_dynamic: bool
    compile_pin_schedule: bool
    compile_fullgraph: bool
    cube_local_grid_cache: bool
    device_resident_masks: bool
    impldiff_build_batch: bool
    ivadv: bool
    kpp_face_batch: bool
    kpp_static_halo_cache: bool
    kpp_wscale_batch: bool
    vectorize_vertshear: bool
    tridiag_cuda: bool
    tridiag_build_cuda: bool
    kpp_bldepth_cuda: bool
    kpp_blmix_cuda: bool
    kpp_statekpp_cuda: bool
    kpp_fused_cuda: bool
    dst3_cuda: bool
    eos_cuda: bool
    vim_coriolis_u: bool
    vim_fused_cuda: bool
    tracer_adv_fused_cuda: bool
    hvisc_fused_cuda: bool
    seaice_lsr_cuda: bool
    seaice_ordered_sum_cuda: bool
    seaice_lsr_graph: bool

    @classmethod
    def from_views(cls, views: Mapping[str, GateView]) -> "StepGates":
        return cls(**{key: view.effective_value for key, view in views.items()})

#: The event names :func:`record_cache_event` writes into ``counts`` (keep in
#: sync with its ``event`` Literal).  They share the counts surface with
#: dispatch outcomes for summaries/ignition, but are EXCLUDED from the
#: cross-rank uniformity signature: cold/warm cache counts are legitimately
#: rank-local (each rank materialises its own (variant, dtype, device)
#: entries), while the uniformity gate's semantics is backend mixing of
#: dispatches.  Value identity of cached constants is the bitwise leg's job.
_CACHE_EVENTS = frozenset(("cold-construction", "warm-reuse", "use-of-cached"))


def parse_bool(raw: str | None, *, default: bool, name: str) -> bool:
    """Parse the one accepted Boolean grammar; empty means not supplied."""
    if raw is None or raw == "":
        return bool(default)
    normalized = raw.casefold()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(
        f"{name} must be one of 1, true, yes, on, 0, false, no, or off; got {raw!r}"
    )


def _resolve_effective(raw: str | None, *, key: str) -> bool:
    spec = REGISTRY[key]
    if spec.setting_kind == "boolean":
        return parse_bool(raw, default=spec.default, name=spec.env_var)
    if raw is None or raw == "":
        return spec.default
    if raw in _IVADV_BACKENDS:
        return True
    raise ValueError(
        f"{spec.env_var} must be cuda; got {raw!r}"
    )


@dataclass(frozen=True)
class _Resolution:
    raw_spelling: str | None
    effective_value: bool
    default_source: DefaultSource


@dataclass
class _RuntimeRecord:
    resolution: _Resolution | None = None
    requested: bool = False
    hit: bool = False
    inapplicable_or_error: bool = False
    counts: Counter[tuple[str, str, str, str, int, str | None]] = field(
        default_factory=Counter
    )

    def ignition_state(self) -> IgnitionState:
        if self.hit:
            return "requested-hit"
        if self.inapplicable_or_error:
            return "inapplicable-or-error"
        if self.requested:
            return "requested-zero"
        return "disabled"


_RUNTIME = {key: _RuntimeRecord() for key in REGISTRY}


def _source_for(
    key: str,
    env_var: str,
    raw: str | None,
    source_overrides: Mapping[str, DefaultSource] | None,
) -> DefaultSource:
    if raw is None or raw == "":
        return "library"
    if source_overrides is not None:
        source = source_overrides.get(key, source_overrides.get(env_var))
        if source is not None:
            if source not in ("driver", "user"):
                raise ValueError(f"non-library source must be driver or user, got {source!r}")
            return source
    return "user"


def resolve_snapshot(
    *, source_overrides: Mapping[str, DefaultSource] | None = None
) -> Mapping[str, _Resolution]:
    """Resolve all client switches together and return an immutable snapshot."""
    resolved: dict[str, _Resolution] = {}
    for key, spec in REGISTRY.items():
        raw = os.environ.get(spec.env_var)
        resolution = _Resolution(
            raw_spelling=raw,
            effective_value=_resolve_effective(raw, key=key),
            default_source=_source_for(key, spec.env_var, raw, source_overrides),
        )
        resolved[key] = resolution
        runtime = _RUNTIME[key]
        runtime.resolution = resolution
        runtime.requested = runtime.requested or resolution.effective_value
    return MappingProxyType(resolved)


def freeze_step_snapshot(
    *, source_overrides: Mapping[str, DefaultSource] | None = None
) -> dict[str, GateView]:
    """Resolve and install the plain-dict gate view used by traced consumers."""
    _drain_pending_events()
    snapshot = resolve_snapshot(source_overrides=source_overrides)
    return _install_step_snapshot(snapshot)


def step_snapshot_active() -> bool:
    return bool(_FAST_GATES)


def _install_step_snapshot(
    snapshot: Mapping[str, _Resolution],
) -> dict[str, GateView]:
    global _FAST_RANK
    frozen = {
        key: _gate_from_resolution(key, resolution)
        for key, resolution in snapshot.items()
    }
    _FAST_GATES.clear()
    _FAST_GATES.update(frozen)
    _FAST_RANK = _resolve_current_rank()
    return frozen


@contextmanager
def step_snapshot(
    *, source_overrides: Mapping[str, DefaultSource] | None = None
) -> Iterator[Mapping[str, _Resolution]]:
    """Freeze gate resolution for one model-step boundary."""
    if _FAST_GATES:
        yield MappingProxyType(
            {
                key: _RUNTIME[key].resolution
                for key in _FAST_GATES
                if _RUNTIME[key].resolution is not None
            }
        )
        return
    snapshot = resolve_snapshot(source_overrides=source_overrides)
    _install_step_snapshot(snapshot)
    try:
        yield snapshot
    finally:
        _clear_fast_snapshot()


class _ConsumerSnapshot:
    def __enter__(self) -> Mapping[str, GateView]:
        freeze_step_snapshot()
        return _FAST_GATES

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        _clear_fast_snapshot()
        return False


_NULL_CONSUMER_SNAPSHOT = nullcontext()


def consumer_snapshot():
    """Use the active step snapshot, or one call-scoped standalone snapshot."""
    if _FAST_GATES:
        return _NULL_CONSUMER_SNAPSHOT
    return _ConsumerSnapshot()


def _gate_from_resolution(key: str, resolution: _Resolution) -> GateView:
    runtime = _RUNTIME[key]
    # A record owns the selectors its program admitted. The ocean runner admits
    # every key; a sea-ice segment admits its own, and other keys keep theirs.
    owned = _SELECTION is not None and key in _SELECTION.gates
    if owned:
        resolution = _SELECTION.gates[key]
    selected = _SELECTION.selected.get(key) if owned else None
    effective = resolution.effective_value
    if owned:
        from .table import IMPLEMENTATION_TABLE

        # An auto-selected reference disables the native selector at the same
        # freezing boundary used by StepPlan and extension preloading.
        effective = selected is not None and selected == IMPLEMENTATION_TABLE[key][-1]
    return GateView(
        raw_spelling=resolution.raw_spelling,
        effective_value=effective,
        default_source=resolution.default_source,
        ignition_state=runtime.ignition_state(),
        selected=None if selected is None else selected.entry,
    )


def gate(key: str) -> GateView:
    view = _FAST_GATES.get(key)
    if view is not None:
        return view
    if key not in REGISTRY:
        raise KeyError(f"unknown accelerator {key!r}")
    snapshot = resolve_snapshot()
    return _gate_from_resolution(key, snapshot[key])


def requested(keys: tuple[str, ...]) -> tuple[str, ...]:
    if _FAST_GATES:
        return tuple(key for key in keys if _FAST_GATES[key].effective_value)
    snapshot = resolve_snapshot()
    return tuple(key for key in keys if _gate_from_resolution(key, snapshot[key]).effective_value)


def reset_runtime() -> None:
    global _DISPATCH_DEPTH, _FAST_RANK, _SELECTION
    _SELECTION = None
    graph = sys.modules.get("locomgcm.solvers.cg2d")
    if graph is not None:
        graph.reset_cg2d_cuda_graph_fallback_counts()
    _clear_fast_snapshot()
    _PENDING_EVENTS.clear()
    _DISPATCH_DEPTH = 0
    _FAST_RANK = None
    # Re-seat the table on the registry before clearing it.  ``_RUNTIME`` is
    # built once at import, so a key absent here -- a spec added after this
    # module was imported, or one a test injected and then removed -- stays
    # absent for the life of the process, and the next ``resolve_snapshot``
    # raises KeyError from inside the atexit summary, where no caller is left
    # to interpret it.  Reset is the one
    # place entitled to restate 'the table covers the registry'.
    for key in REGISTRY:
        if key not in _RUNTIME:
            _RUNTIME[key] = _RuntimeRecord()
    for runtime in _RUNTIME.values():
        runtime.resolution = None
        runtime.requested = False
        runtime.hit = False
        runtime.inapplicable_or_error = False
        runtime.counts.clear()


def record_dispatch(
    key: str,
    *,
    outcome: Literal["hit", "inapplicable", "error"],
    variant: str,
    site: str,
    backend: str,
    rank: int,
    reason: str | None = None,
    count: int = 1,
) -> None:
    """Record one labelled dispatch outcome from a real consumer boundary."""
    _validate_dispatch(outcome, reason)
    if _is_compiling():
        _enqueue_record(
            _ENQUEUE_ORDER_TOKEN,
            "dispatch", key, outcome, variant, site, backend, rank, reason, count,
        )
        return
    _record_dispatch_eager(
        key, outcome, variant, site, backend, rank, reason, count
    )


def _warn_first_default_on_error(key: str, site: str, reason: str | None) -> None:
    """Announce a default-enabled failure without replacing its exception.

    The caller owns propagation. This diagnostic preserves ledger counts and
    warns once per selector, even when a warning filter is configured to raise.
    """
    spec = REGISTRY[key]
    if not spec.default or key in _DEFAULT_ON_ERROR_ANNOUNCED:
        return
    _DEFAULT_ON_ERROR_ANNOUNCED.add(key)
    message = (
        f"accelerator {key!r} ({spec.env_var}) is ON by library default, and its "
        f"first dispatch at {site} failed with reason {reason!r}; selected "
        f"implementation failures must propagate without execution fallback. "
        f"Further failures of this switch are counted in the LOCOMGCM_ACCEL "
        f"summary and not warned about again; set {spec.env_var}=0 before "
        f"program construction to request a different implementation."
    )
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=4)
    except Exception:  # pragma: no cover - defensive
        # A warning must never mask the original selected exception. The
        # numerical caller re-raises that exception after recording its error.
        pass


def _record_dispatch_eager(
    key: str,
    outcome: str,
    variant: str,
    site: str,
    backend: str,
    rank: int,
    reason: str | None,
    count: int,
) -> None:
    spec = REGISTRY[key]
    view = gate(key)
    if not view.effective_value:
        raise RuntimeError(f"accelerator {key!r} dispatched while disabled")
    if count <= 0:
        raise ValueError("dispatch count must be positive")
    runtime = _RUNTIME[key]
    runtime.counts[(outcome, variant, site, backend, int(rank), reason)] += int(count)
    if outcome == "hit":
        runtime.hit = True
    else:
        runtime.inapplicable_or_error = True
        if outcome == "error":
            _warn_first_default_on_error(key, site, reason)
    if outcome == "inapplicable" and spec.unmet_policy == "raise":
        raise RuntimeError(
            f"accelerator {key!r} is inapplicable and declares unmet_policy='raise': "
            f"{reason}"
        )


def record_cache_event(
    key: str,
    *,
    event: Literal["cold-construction", "warm-reuse", "use-of-cached"],
    variant: str,
    site: str,
    backend: str,
    rank: int,
    count: int = 1,
) -> None:
    """Record cache activity; only warm reuse constitutes ignition.

    Cache events stay out of the cross-rank comparison (``_CACHE_EVENTS``).
    """
    if _is_compiling():
        _enqueue_record(
            _ENQUEUE_ORDER_TOKEN,
            "cache", key, event, variant, site, backend, rank, None, count,
        )
        return
    _record_cache_event_eager(key, event, variant, site, backend, rank, count)


def _record_cache_event_eager(
    key: str,
    event: str,
    variant: str,
    site: str,
    backend: str,
    rank: int,
    count: int,
) -> None:
    view = gate(key)
    if not view.effective_value:
        raise RuntimeError(f"accelerator {key!r} cache activity while disabled")
    if count <= 0:
        raise ValueError("cache event count must be positive")
    runtime = _RUNTIME[key]
    runtime.counts[(event, variant, site, backend, int(rank), None)] += int(count)
    if event == "warm-reuse":
        runtime.hit = True


def _is_compiling() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch._dynamo.is_compiling())


def _drain_pending_events() -> None:
    pending = tuple(_PENDING_EVENTS)
    _PENDING_EVENTS.clear()
    for kind, key, outcome, variant, site, backend, rank, reason, count in pending:
        if kind == "dispatch":
            _record_dispatch_eager(
                key, outcome, variant, site, backend, rank, reason, count
            )
        else:
            _record_cache_event_eager(
                key, outcome, variant, site, backend, rank, count
            )


def drain_pending_dispatch_events() -> None:
    """Record the events the traced step could only queue. **Eager.**

    ``record_dispatch``/``record_cache_event`` append to ``_PENDING_EVENTS``
    while Dynamo is tracing, because the accounting mutates module-level dicts.
    The step's finalize phase owns the drain (``locomgcm/step/finalize.py``),
    so a step's own dispatches are on the books before anything reads them,
    instead of arriving at the next step's gate freeze.
    """
    _drain_pending_events()


def current_rank() -> int:
    if _FAST_RANK is not None:
        return _FAST_RANK
    return _resolve_current_rank()


def _resolve_current_rank() -> int:
    try:
        import torch.distributed as dist
    except (ImportError, RuntimeError):
        return int(os.environ.get("RANK", "0"))
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))


def in_accelerated_dispatch() -> bool:
    return _DISPATCH_DEPTH > 0


class _EagerDispatchScope:
    def __enter__(self) -> None:
        global _DISPATCH_DEPTH
        _DISPATCH_DEPTH += 1

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        global _DISPATCH_DEPTH
        _DISPATCH_DEPTH -= 1
        return False


def dispatch_scope(key: str):
    """Mark the dynamic extent of one accelerator dispatch.

    Inside the scope :func:`in_accelerated_dispatch` is true, so the consumer
    boundary re-entered by the package's own eager fallback runs the frozen
    reference instead of dispatching a second time.  This is the single
    recursion-guard mechanism; a package must not maintain its own.
    """
    if _FAST_GATES:
        if key not in _FAST_GATES:
            raise KeyError(f"unknown accelerator {key!r}")
        return _EagerDispatchScope()
    if key not in REGISTRY:
        raise KeyError(f"unknown accelerator {key!r}")
    return _EagerDispatchScope()



def record_selection(record: "AdmissionRecord | None") -> None:
    """Publish a selected identity, without inventing a dispatch completion."""
    global _SELECTION
    _drain_pending_events()
    _SELECTION = record
    _clear_fast_snapshot()
    if record is not None:
        for key, view in record.gates.items():
            _RUNTIME[key].resolution = _Resolution(view.raw_spelling, view.effective_value, view.default_source)
            _RUNTIME[key].requested |= view.effective_value


@contextmanager
def selection_scope(record: "AdmissionRecord | None", context=None):
    """Bind a runner's immutable decision across eager prepare and execution.

    Nested/interleaved runner calls restore their caller's frozen choices.
    The outermost run leaves its decision observable after completion.
    """
    if os.environ.get("LOCOMGCM_COMPILE_CACHE_ROOT"):
        from .compile_cache import configure_compile_cache

        configure_compile_cache()
    global _SELECTION_DEPTH
    previous = _SELECTION
    previous_fast = dict(_FAST_GATES)
    record_selection(record)
    _SELECTION_DEPTH += 1
    try:
        with nullcontext() if context is None else context:
            yield
    finally:
        _drain_pending_events()
        _SELECTION_DEPTH -= 1
        if _SELECTION_DEPTH:
            record_selection(previous)
            _FAST_GATES.update(previous_fast)


def _count_row_order(row: tuple) -> tuple:
    # Both public views end in (reason, count). A successful dispatch may
    # have no reason or a named reason at the same site and backend.
    *labels, reason, count = row
    return (*labels, reason is not None, reason, count)


def summaries() -> dict[str, dict[str, object]]:
    _drain_pending_events()
    result: dict[str, dict[str, object]] = {}
    unresolved = [key for key, runtime in _RUNTIME.items() if runtime.resolution is None]
    if unresolved:
        resolve_snapshot()
    for key, spec in REGISTRY.items():
        runtime = _RUNTIME[key]
        resolution = runtime.resolution
        assert resolution is not None
        view = _gate_from_resolution(key, resolution)
        counts = [
            {
                "backend": backend,
                "count": count,
                "outcome": outcome,
                "rank": rank,
                "reason": reason,
                "site": site,
                "variant": variant,
            }
            for outcome, variant, site, backend, rank, reason, count in sorted(
                ((*event, count) for event, count in runtime.counts.items()), key=_count_row_order
            )
        ]
        result[key] = {
            "switch": key,
            "selected": view.selected,
            "admission_reason": None if _SELECTION is None else _SELECTION.rejections.get(key),
            "environment": spec.env_var,
            "raw_spelling": view.raw_spelling,
            "effective_value": view.effective_value,
            "default_source": view.default_source,
            "ignition_state": view.ignition_state,
            "library_default": spec.default,
            "gate_class": spec.gate_class,
            "unmet_policy": spec.unmet_policy,
            "grid_families": list(spec.grid_families),
            "distributed": list(spec.distributed),
            "hydrostatic": spec.hydrostatic,
            "gate_satisfiable": spec.gate_satisfiable,
            "production_reachable": spec.production_reachable,
            "setting_kind": spec.setting_kind,
            "counts": counts,
        }
    return result


def format_summary(key: str) -> str:
    payload = summaries()[key]
    return "LOCOMGCM_ACCEL " + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def emit_summaries() -> None:
    """Emit diagnostics without corrupting a caller's machine-readable stdout.

    LOCOMGCM_ACCEL_REPORT=0 silences the report; the default stays on because the
    lines are the only evidence that a requested kernel actually ran.
    """
    if os.environ.get("LOCOMGCM_ACCEL_REPORT", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    for key in REGISTRY:
        print(format_summary(key), file=sys.stderr, flush=True)
    from .launch import format_launch_decisions

    for line in format_launch_decisions():
        print(line, file=sys.stderr, flush=True)


atexit.register(emit_summaries)


def dispatch_signature() -> dict[str, tuple]:
    """This rank's dispatch rows per selector, rank-stripped and sorted:
    ``{key: ((outcome, variant, site, backend, reason, count), ...)}``.

    The rows are what the ranks exchange; :func:`dispatch_divergence` compares
    only the backends they chose (:func:`selected_backends`) and prints the rows,
    counts included, as the evidence of a divergence.

    Cache events (``_CACHE_EVENTS``) are excluded: cold-construction /
    warm-reuse counts are legitimately rank-local (e.g. ``MaskSet._float_for``
    materialises a different number of (attr, dtype, device) entries per
    rank).  Cached-value identity across ranks is the bitwise correctness
    leg's job, not this gate's.
    """
    signature: dict[str, tuple] = {}
    for key, record in _RUNTIME.items():
        dispatches = (
            (outcome, variant, site, backend, reason, int(count))
            for (outcome, variant, site, backend, _rank, reason), count in record.counts.items()
            if outcome not in _CACHE_EVENTS
        )
        rows = tuple(sorted(dispatches, key=_count_row_order))
        if rows:
            signature[key] = rows
    return signature


def selected_backends(rows: Sequence[tuple]) -> frozenset[tuple[str, str, str, str]]:
    """The (outcome, variant, site, backend) choices in one rank's rows of a selector.

    ``outcome`` says where the result came from (``hit``: the backend ran;
    ``inapplicable``: the reference ran instead), so a rank that ran the
    kernel and a rank that ran the reference at the same site differ here.
    Counts and reasons are left out: how often a site dispatches can follow the
    rank's own data (the Newton and line-search loops of the convective
    implicit KPP stop per tile, so the KPP batch records 155 calls on one rank
    and 152 on the other in one two-rank case), and two ranks that both
    declined for different reasons both ran the reference.
    """
    return frozenset(row[:4] for row in rows)


def dispatch_divergence(signatures: Sequence[Mapping[str, tuple] | None]) -> list[str]:
    """One line per selector whose :func:`selected_backends` differ across ranks;
    ``signatures[r]`` is rank r's :func:`dispatch_signature`.  Empty when every rank
    chose the same backends.
    """
    # ponytail: a selector that took both the kernel and the reference on every
    # rank compares equal even when the ranks interleaved them differently (the
    # order digest that caught this folded counts in, so it also raised on
    # data-dependent loop lengths).  Such a selector shows both outcomes in every
    # rank's LOCOMGCM_ACCEL summary; a per-call check would record each call's
    # choice per rank and compare them call by call.
    sigs = [sig or {} for sig in signatures]
    detail = []
    for key in sorted({k for sig in sigs for k in sig}):
        chosen = [selected_backends(sig.get(key, ())) for sig in sigs]
        if any(c != chosen[0] for c in chosen):
            detail.append(f"{key}: " + "; ".join(f"r{r}={sig.get(key)}" for r, sig in enumerate(sigs)))
    return detail


def verify_dispatch_uniformity(group=None) -> None:
    """Raise if the ranks chose different backends for the same selector.

    Backend-mixing guard: dispatcher
    predicates are rank-local by design; on heterogeneous setups one rank can
    silently fall back to the eager reference while peers run the CUDA kernel,
    quietly mixing numerics.  This check runs at an END-OF-RUN synchronous
    point (never inside dispatch paths - an in-path collective could itself
    wedge, the failure class this check exists to catch).  All ranks gather the
    rank-stripped signatures and all raise together on a divergence
    (SPMD-uniform outcome by construction: identical data, identical
    comparison).  The comparison is :func:`dispatch_divergence`.
    """
    try:
        import torch.distributed as dist
    except (ImportError, RuntimeError):
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    world = dist.get_world_size(group)
    if world <= 1:
        return
    local = dispatch_signature()
    gathered: list = [None] * world
    dist.all_gather_object(gathered, local, group=group)
    detail = dispatch_divergence(gathered)
    if detail:
        raise RuntimeError(
            "accelerator dispatch diverged across ranks (silent backend mixing): "
            + " | ".join(detail[:6])
        )
