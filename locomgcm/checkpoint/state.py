"""Ownership and codec for IFS cross-step restart state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from locomgcm.steppers.ifs.cube_inputs import CubeSurfaceForcing



RESTART_STATE_KEYS_V2 = frozenset({
    "_step_index",
    "_ab2_prev_tendency",
    "_ab2_prev_tracer_tendency",
    "_etaH",
    "_eta_prev_step",  # Legacy checkpoint field; current steppers never produce it.
    "_w_interface_prev",
    "_momentum_w_nm_half_mit",
    "_cached_phi_nh",
    "_cached_gwNm1",
    "_cached_w_vel_mit_for_pre_cg3d",
    "_obc_eta",
    "__cached_total_phi_hyd",
    # MITgcm PmEpR of the last continuity (nonlinear free surface, real freshwater).
    "_pmep_r",
    # MITgcm CD scheme D-grid velocity and lagged fields (locomgcm.dynamics.cd_scheme.state).
    "_cd_scheme",
    # MITgcm GGL90TKE (locomgcm.steppers.ifs.columns.GGL90_RESTART_KEY).
    "_ggl90_tke",
    # The sea-ice component's carry (locomgcm.seaice.step_state.SEAICE_RESTART_KEY).
    "_seaice",
})

RESTART_STATE_KEYS_V3 = RESTART_STATE_KEYS_V2 | {"boundary_history"}
RESTART_STATE_KEYS_V5 = RESTART_STATE_KEYS_V3 | {"nonlinear_geometry"}

_MOMENTUM_KEY = "_momentum_w_nm_half_mit"
_MOMENTUM_KIND = "momentum_w_nm_half_mit/v1"
#: The legacy momentum codec stores kind, data and an integer step_index.
#: Checkpoints no longer write it: the lagged w is stored once, as
#: ``_w_interface_prev``, and ``decoded_step_state`` merges a legacy wrapper
#: into that key.
_MOMENTUM_FIELDS = frozenset({"kind", "data", "step_index"})

def require_exact_int(value: Any, *, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be an exact int, got {type(value).__name__}")
    return value


def live_step_index(value: Any) -> torch.Tensor:
    """The completed-step field as a 0-d int64 CPU tensor.

    One spelling for the two places that seed or restore the counter
    (``initialize_step_state``, ``apply_decoded_step_state``), matching the
    container the one that *advances* it already writes
    (``ImplicitFreeSurfaceStepper.advance_step_index``:
    ``torch.tensor(n + 1, dtype=torch.int64)``, unchanged by this package), so
    the container cannot drift between a fresh run, a resumed one, and a
    stepping one. A Python int is guarded on its value; a 0-d tensor is
    guarded on dtype and shape only.

    CPU regardless of the model's device, matching ``advance_step_index``: this
    is a host counter that the kernel receives through
    ``StepPlan.step_index``, never a field the step body computes with, so
    putting it on the accelerator would only add a transfer.

    The restart *file* keeps an exact int -- ``require_exact_int`` and
    ``encode_step_state_from_stepper`` are unchanged -- so this conversion
    happens when the completed-step field is installed.
    """
    if isinstance(value, torch.Tensor):
        if value.dtype is not torch.int64 or value.ndim != 0:
            raise TypeError(
                "live step index tensor must be 0-d int64, got "
                f"{value.dtype} with shape {tuple(value.shape)}"
            )
        return value.detach().to(device="cpu")
    return torch.tensor(
        require_exact_int(value, path="step_state['_step_index']"),
        dtype=torch.int64,
    )


def momentum_implicit_enabled(stepper: Any) -> bool:
    """Return the canonical momentum flag; absence means phase-2 capability off."""
    value = getattr(stepper, "momentum_implicit_vertical_advection", False)
    if type(value) is not bool:
        raise TypeError(
            "stepper.momentum_implicit_vertical_advection must be an exact bool, "
            f"got {type(value).__name__}"
        )
    return value


def initialize_step_state(model: Any, bounded_initial_state: Any) -> None:
    """Initialize fresh-run cross-step state after boundary/sync processing.

    With momentum implicit vertical advection the lagged w is seeded here: the
    complete MITgcm-convention vertical velocity, surface-first and
    upward-positive, diagnosed from the already bounded/synchronised U/V state
    with the LOCOMgcm downward-positive to MIT upward-positive conversion
    done exactly once.  It is the one stored copy of w: the momentum solve
    wraps the same tensor the tendency modules read.
    """
    stepper = model.stepper
    if getattr(stepper, "name", None) != "implicit_free_surface":
        return
    # A 0-d int64 tensor, the same container ``advance_step_index`` writes: the
    # step after this one must not be the one that changes the counter's type
    # under the tracer (``live_step_index``).
    object.__setattr__(stepper, "_step_index", live_step_index(0))
    from locomgcm.step.boundary_history import BoundaryHistory, boundary_adapter

    adapter = boundary_adapter(model)
    if adapter is not None and adapter.history is None:
        adapter.history = BoundaryHistory.initial(bounded_initial_state)
        adapter.initialization_reason = "explicit_new_run"
    if momentum_implicit_enabled(stepper):
        grid = getattr(model, "grid", None)
        if grid is None:
            raise ValueError(
                "momentum implicit vertical advection fresh initialization "
                "requires model.grid"
            )
        if bounded_initial_state is None:
            raise ValueError(
                "momentum implicit vertical advection fresh initialization "
                "requires a bounded initial state"
            )
        from locomgcm.dynamics.implicit_vertical_advection import (
            make_momentum_w_nm_half_mit,
        )

        from locomgcm.steppers.ifs.freshwater import ContinuityHistory

        w_interface_prev = make_momentum_w_nm_half_mit(
            bounded_initial_state,
            grid,
            cube_ctx=getattr(model, "cube_ctx", None),
        ).data
        object.__setattr__(stepper, "_continuity", ContinuityHistory(
            w_interface_prev=w_interface_prev,
        ))


def encode_step_state_from_stepper(
    stepper: Any,
    *,
    detach: bool = True,
    device_resident: bool = False,
) -> dict[str, Any]:
    """Export only present v2 allowlist keys.

    ``device_resident`` keeps each value as a copy on its own device (a
    window's first segment head); otherwise values are copied to the host.
    """
    if type(detach) is not bool:
        raise TypeError(f"detach must be an exact bool, got {type(detach).__name__}")
    live = _live_restart_state(stepper)
    copy = _snapshot_copy(detach=detach, device_resident=device_resident)
    out = _restart_payload(live, RESTART_STATE_KEYS_V2, copy=copy)
    validate_encoded_step_state(out)
    _validate_momentum_presence(out, stepper)
    return out


def head_step_state(stepper: Any, *, snapshot: bool = True) -> dict[str, Any]:
    """The restart allowlist as device clones that keep their graph, unchecked.

    A segment head carries what ``encode_step_state_from_stepper`` exports, in
    the same mapping, but stays on the model device and skips the per-value
    restart checks; the segment head spec runs those once per head layout.
    """
    return _restart_payload(_live_restart_state(stepper), RESTART_STATE_KEYS_V2,
                            copy=torch.Tensor.clone if snapshot else lambda value: value)


@dataclass(frozen=True)
class BorrowedStepInputs:
    obc_eta: float | torch.Tensor | None
    cube_surface_forcing: CubeSurfaceForcing | None
    cube_w_interface_readout: torch.Tensor | None


def borrow_step_inputs(stepper: Any, *, obc_eta) -> BorrowedStepInputs:
    """Capture enclosing-window inputs without the segment's derived bulk flux."""
    forcing = vars(stepper).get("_cube_surface_forcing")
    return BorrowedStepInputs(
        obc_eta=obc_eta,
        cube_surface_forcing=(replace(forcing, bulk=None) if forcing is not None else None),
        cube_w_interface_readout=vars(stepper).get("cube_w_interface_readout"),
    )


def _install_step_state(stepper, state):
    for name, (_, decode) in stepper.OWNED_STEP_FIELDS.items():
        if decode is not None:
            object.__setattr__(stepper, name, decode(state))


def release_working_step_state(model: Any, *, borrowed_inputs=None) -> None:
    """End a checkpoint segment's mutable history after its head was exported.

    Restart state, current-step values and diagnostics share this lifetime.
    Geometry constants and solver coefficients survive it. Inputs borrowed
    from the enclosing window return unchanged; a segment's new expressions
    never replace their owner. Nonlinear geometry belongs to the whole-model
    checkpoint codec, which also carries it between gradient window segments.
    """
    stepper = model.stepper
    borrowed_state = {}
    if borrowed_inputs is not None and borrowed_inputs.obc_eta is not None:
        borrowed_state["_obc_eta"] = borrowed_inputs.obc_eta
    _install_step_state(stepper, borrowed_state)
    for name in ("_cached_total_phi_hyd",
                 "_prepared_boundary", "_prepared_step_key", "_prepared_pnh_phi",
                 "_prepared_cg3d_coefficients"):
        object.__setattr__(stepper, name, None)
    from locomgcm.steppers.ifs.cube_inputs import EMPTY_CUBE_SURFACE_FORCING

    object.__setattr__(stepper, "_cube_surface_forcing", EMPTY_CUBE_SURFACE_FORCING)
    if borrowed_inputs is not None and borrowed_inputs.cube_surface_forcing is not None:
        object.__setattr__(stepper, "_cube_surface_forcing", borrowed_inputs.cube_surface_forcing)
    object.__setattr__(stepper, "cube_w_interface_readout", None)
    if borrowed_inputs is not None and borrowed_inputs.cube_w_interface_readout is not None:
        object.__setattr__(stepper, "cube_w_interface_readout", borrowed_inputs.cube_w_interface_readout)
    # Absent, not None: the traced step guards an existing attribute's value.
    for name in ("_evaluated_w_interface", "_evaluated_total_phi_hyd", "_evaluated_pmep_r"):
        vars(stepper).pop(name, None)
    if getattr(model, "_ocean_carry", None) is not None:
        model._ocean_carry = None
        object.__setattr__(stepper, "_history_model", None)
    from locomgcm.step.boundary_history import boundary_adapter

    adapter = boundary_adapter(model)
    if adapter is not None:
        adapter.reset()


def _live_restart_state(stepper: Any) -> dict[str, Any]:
    """The live explicit histories, completed-step counter and pressure."""
    live: dict[str, Any] = {}
    for name, (encode, _) in stepper.OWNED_STEP_FIELDS.items():
        if encode is not None:
            live.update(encode(getattr(stepper, name)))
    owner = getattr(stepper, "_history_model", None)
    carry = getattr(owner, "_ocean_carry", None)
    if carry is not None:
        from locomgcm.step.carry import validate_history

        validate_history(carry.state, carry.ab_history, stepper)
        if "_ab2_prev_tendency" in live or "_ab2_prev_tracer_tendency" in live:
            raise ValueError("explicit AB history must not have a second cache owner")
        # The hydro group records validity even at a reinitialized nonzero
        # window. Legacy all-valid groups remain readable by checkpoint_carry.
        if carry.hydro_history is not None:
            from locomgcm.step.carry import validate_hydro_history

            validate_hydro_history(carry.state, carry.hydro_history)
            live["_ab2_prev_tendency"] = {"u": carry.ab_history["mom_u"],
                                        "v": carry.ab_history["mom_v"],
                                        **{name: value for name, value in carry.ab_history.items()
                                           if name not in {"mom_u", "mom_v"}}}
            live["_etaH"] = carry.hydro_history.eta_h
            live["_w_interface_prev"] = carry.hydro_history.w_interface
        elif bool(carry.ab_history["mom_valid"]):
            live["_ab2_prev_tendency"] = {
                "u": carry.ab_history["mom_u"], "v": carry.ab_history["mom_v"]}
        live["_step_index"] = carry.clock["step_index"]
    counter = live.get("_step_index")
    if isinstance(counter, torch.Tensor):
        # The stepper keeps the counter as a 0-d int64 tensor so the compiled
        # step guards on its shape, not its value (an
        # int is EQUALS_MATCH-guarded whenever it is traced, and it changes
        # every step). The restart contract stays an
        # exact int.
        live["_step_index"] = int(counter.item())
    phi = getattr(stepper, "_cached_total_phi_hyd", None)
    if phi is not None:
        live["__cached_total_phi_hyd"] = phi
    return live


def _restart_payload(live: dict[str, Any], keys: frozenset[str], *,
                     copy) -> dict[str, Any]:
    """The present ``keys`` of ``live``, each tensor passed through ``copy``."""
    out: dict[str, Any] = {}
    for key in sorted(keys):
        if key not in live:
            continue
        out[key] = _copy_tree(live[key], copy=copy, path=f"step_state[{key!r}]")
    return out


def validate_encoded_step_state(state: Any, *, legacy_v3: bool = False) -> None:
    """Validate a serialized v2 step-state mapping without mutating a stepper."""
    if type(state) is not dict:
        raise TypeError(f"checkpoint step_state must be a dict, got {type(state).__name__}")
    if any(type(key) is not str for key in state):
        raise TypeError("checkpoint step_state keys must all be exact strings")
    allowed = RESTART_STATE_KEYS_V2 if legacy_v3 else RESTART_STATE_KEYS_V5
    unknown = sorted(set(state) - allowed)
    if unknown:
        raise ValueError(
            f"checkpoint step_state contains keys outside the restart schema: {unknown}"
        )
    if "_step_index" not in state:
        raise ValueError("checkpoint step_state is missing required key '_step_index'")
    step_index = require_exact_int(state["_step_index"], path="step_state['_step_index']")
    for key, value in state.items():
        if key == _MOMENTUM_KEY:
            _, wrapper_index = _validate_momentum_payload(value)
            if wrapper_index != step_index:
                raise ValueError(
                    "checkpoint momentum wrapper clock mismatch: "
                    f"wrapper step_index={wrapper_index}, _step_index={step_index}"
                )
        else:
            _validate_restart_value(value, path=f"step_state[{key!r}]")


def validate_checkpoint_structure(ckpt: Any, *, legacy_v3: bool = False) -> None:
    """Validate all model-independent v2 invariants of a checkpoint."""
    from .types import (
        PROGNOSTIC_FIELDS,
        ModelCheckpoint,
        validate_checkpoint_meta,
    )

    if not isinstance(ckpt, ModelCheckpoint):
        raise TypeError(f"checkpoint must be ModelCheckpoint, got {type(ckpt).__name__}")
    validate_checkpoint_meta(ckpt.meta, legacy_v3=legacy_v3)
    if type(ckpt.prognostic) is not dict:
        raise TypeError(
            f"checkpoint prognostic must be a dict, got {type(ckpt.prognostic).__name__}"
        )
    prognostic_keys = frozenset(ckpt.prognostic)
    expected_prognostic = frozenset(PROGNOSTIC_FIELDS)
    if prognostic_keys != expected_prognostic:
        missing = sorted(expected_prognostic - prognostic_keys)
        extra = sorted(prognostic_keys - expected_prognostic)
        raise ValueError(
            "checkpoint prognostic fields must match exactly; "
            f"missing={missing}, extra={extra}"
        )
    for key, value in ckpt.prognostic.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"checkpoint prognostic[{key!r}] must be a torch.Tensor, "
                f"got {type(value).__name__}"
            )

    validate_encoded_step_state(ckpt.step_state, legacy_v3=legacy_v3)
    if not legacy_v3:
        from .nonlinear import validate_nonlinear_geometry

        validate_nonlinear_geometry(ckpt)
    if not legacy_v3 and "boundary_history" in ckpt.step_state:
        from .boundary import validate_boundary_payload

        validate_boundary_payload(ckpt.step_state["boundary_history"], ckpt.prognostic,
                                  time_seconds=ckpt.meta.model_time_seconds)
        if ckpt.meta.global_step > 0 and not bool(ckpt.step_state["boundary_history"]["valid"]):
            raise ValueError("continuous checkpoint requires valid boundary_history")
    actual_state_keys = tuple(sorted(ckpt.step_state))
    if frozenset(ckpt.meta.state_keys) != frozenset(actual_state_keys):
        raise ValueError(
            "checkpoint meta.state_keys mismatch: "
            f"meta={ckpt.meta.state_keys}, payload={actual_state_keys}"
        )
    step_index = require_exact_int(
        ckpt.step_state["_step_index"], path="step_state['_step_index']"
    )
    if step_index != ckpt.meta.global_step:
        raise ValueError(
            "checkpoint clock mismatch: "
            f"step_state['_step_index']={step_index}, "
            f"meta.global_step={ckpt.meta.global_step}"
        )
    momentum_flag_name = "momentum_implicit_vertical_advection"
    if momentum_flag_name in ckpt.meta.physics_flags:
        enabled = ckpt.meta.physics_flags[momentum_flag_name]
        if type(enabled) is not bool:
            raise TypeError(
                f"checkpoint meta.physics_flags[{momentum_flag_name!r}] "
                f"must be an exact bool, got {type(enabled).__name__}"
            )
        # The lagged w is the one stored copy; a legacy checkpoint may carry
        # the momentum wrapper instead (``decoded_step_state`` merges it).
        # The lagged w also exists without the flag (tracer advection reads
        # it), so only a legacy wrapper can be stale.
        wrapper = _MOMENTUM_KEY in ckpt.step_state
        present = wrapper or "_w_interface_prev" in ckpt.step_state
        if (enabled and not present) or (not enabled and wrapper):
            state = "missing" if enabled else "stale"
            raise ValueError(
                f"checkpoint momentum wrapper is {state} for metadata flag "
                f"{momentum_flag_name}={enabled}"
            )


def validate_step_state_for_stepper(state: dict[str, Any], stepper: Any) -> None:
    validate_encoded_step_state(state)
    _validate_momentum_presence(state, stepper)
    _validate_freshwater_history_presence(state, stepper)


def _validate_freshwater_history_presence(state: dict[str, Any], stepper: Any) -> None:
    """A continued synchronous nonlinear-free-surface run with real freshwater
    flux carries its PmEpR, which the next step's surface salt forcing reads.

    Staggered stepping sets PmEpR from the step's own EmPmR before the surface
    forcing (``ImplicitFreeSurfaceStepper._pmep_r_for_step``), so its restart
    does not read the history and is not refused for lacking it.
    """
    uses = getattr(stepper, "_uses_freshwater_history", None)
    if uses is None or not uses() or stepper.stagger_tracer_stepping:
        return
    if "_etaH" in state and "_pmep_r" not in state:
        raise ValueError(
            "checkpoint step_state of a nonlinear-free-surface run with real freshwater "
            "flux continues a run ('_etaH') without its PmEpR history ('_pmep_r')")


def validate_momentum_wrapper_for_model(state: dict[str, Any], model: Any) -> None:
    """Validate reserved tensor dtype and trailing physical shape."""
    validate_step_state_for_stepper(state, model.stepper)
    if _MOMENTUM_KEY not in state:
        return
    data, _ = _validate_momentum_payload(state[_MOMENTUM_KEY])
    expected_dtype = model.config.dtype
    if data.dtype != expected_dtype:
        raise ValueError(
            "checkpoint momentum wrapper dtype mismatch: "
            f"checkpoint={data.dtype}, model={expected_dtype}"
        )
    expected_trailing = (model.grid.nz + 1, model.grid.ny, model.grid.nx)
    if data.ndim < 3 or tuple(data.shape[-3:]) != expected_trailing:
        raise ValueError(
            "checkpoint momentum wrapper shape mismatch: "
            f"got {tuple(data.shape)}, expected trailing dimensions {expected_trailing}"
        )


def decode_step_state(
    state: dict[str, Any],
    *,
    stepper: Any,
    device: Any,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Fully validate and decode a payload before any live-state mutation.

    A legacy momentum wrapper's ``step_index`` is validated against
    ``_step_index`` (``validate_encoded_step_state``) and its data merged into
    ``_w_interface_prev`` (``decoded_step_state``).
    """
    validate_step_state_for_stepper(state, stepper)
    if _MOMENTUM_KEY in state:
        data, _ = _validate_momentum_payload(state[_MOMENTUM_KEY])
        if data.dtype != dtype:
            raise ValueError(
                "checkpoint momentum wrapper dtype mismatch: "
                f"checkpoint={data.dtype}, model={dtype}"
            )
    return decoded_step_state(state, device=device, dtype=dtype)


def decoded_step_state(state: dict[str, Any], *, device: Any,
                       dtype: torch.dtype) -> dict[str, Any]:
    """``decode_step_state`` without its checks, for a payload checked earlier.

    Segment heads come from the model the rollout steps, in the layout the
    segment head spec checked once per window.

    A checkpoint written before the lagged w was stored once carries the
    momentum wrapper, a second copy of the same w.  It is merged here: with
    ``_w_interface_prev`` present the two must be equal byte for byte and the
    wrapper is dropped; alone (``pickup_to_ckpt.py`` writes only the wrapper)
    its data becomes ``_w_interface_prev``.
    """
    out = {key: (value if key == "nonlinear_geometry"
                 else _to_device(value, device=device, dtype=dtype))
           for key, value in state.items() if key != _MOMENTUM_KEY}
    if _MOMENTUM_KEY in state:
        legacy = state[_MOMENTUM_KEY]["data"].to(device=device)
        current = out.get("_w_interface_prev")
        if current is None:
            out["_w_interface_prev"] = legacy
        elif not torch.equal(current, legacy):
            raise ValueError(
                "legacy checkpoint holds two different lagged w: "
                "'_momentum_w_nm_half_mit' data differs from '_w_interface_prev'"
            )
    return out


def apply_decoded_step_state(stepper: Any, state: dict[str, Any], *,
                             history_model=None, history_carry=None,
                             boundary_history_validated=False,
                             model_geometry_validated=False) -> None:
    """Commit a previously decoded mapping to a fresh stepper.

    The counter field uses a 0-d int64 CPU tensor, ``live_step_index``, so a restored
    stepper and a fresh one that has taken a step are indistinguishable to the
    tracer.  The decoded *mapping* keeps the exact int the restart contract
    requires (``require_exact_int`` below, and the arithmetic that sets the
    carry's clock). The owner codec converts it when installing the field.
    """
    if "boundary_history" in state and not boundary_history_validated:
        raise ValueError("boundary_history requires whole-model restore for boundary identity validation")
    if "nonlinear_geometry" in state and not model_geometry_validated:
        raise ValueError("nonlinear_geometry requires whole-model restore")
    if history_model is None:
        history_model = getattr(stepper, "_history_model", None)
    if history_model is not None and history_carry is None:
        from locomgcm.step.carry import checkpoint_carry

        previous = history_model._ocean_carry
        history_carry = checkpoint_carry(
            history_model, previous.state, state,
            time_seconds=state["_step_index"] * history_model.config.default_dt,
        )
    working = {}
    phi = None
    for key, value in state.items():
        if key in {"boundary_history", "nonlinear_geometry"}:
            continue  # Published through the whole-model adapters.
        if history_carry is not None and key == "_ab2_prev_tendency":
            continue
        if (history_carry is not None and history_carry.hydro_history is not None
                and key in {"_etaH", "_w_interface_prev", "__cached_total_phi_hyd"}):
            continue
        if key == "__cached_total_phi_hyd":
            phi = value
        else:
            working[key] = value
    _install_step_state(stepper, working)
    object.__setattr__(stepper, "_cached_total_phi_hyd", phi)
    if history_carry is not None:
        from locomgcm.step.carry import _publish

        _publish(history_model, history_carry)


def _validate_momentum_presence(state: dict[str, Any], stepper: Any) -> None:
    """The lagged w (or a legacy wrapper) is present when the solve reads it."""
    enabled = momentum_implicit_enabled(stepper)
    wrapper = _MOMENTUM_KEY in state
    if enabled and not (wrapper or "_w_interface_prev" in state):
        raise ValueError(
            "momentum implicit vertical advection requires restart state "
            "'_w_interface_prev'"
        )
    if not enabled and wrapper:
        raise ValueError(
            "checkpoint contains stale '_momentum_w_nm_half_mit' while "
            "momentum implicit vertical advection is disabled"
        )


def _validate_momentum_payload(value: Any) -> tuple[torch.Tensor, int]:
    if type(value) is not dict:
        raise TypeError(
            "checkpoint '_momentum_w_nm_half_mit' must use reserved codec "
            f"{_MOMENTUM_KIND!r}, got {type(value).__name__}"
        )
    fields = frozenset(value)
    if fields != _MOMENTUM_FIELDS:
        missing = sorted(_MOMENTUM_FIELDS - fields)
        extra = sorted(fields - _MOMENTUM_FIELDS)
        raise ValueError(
            "checkpoint momentum wrapper fields must be exactly "
            f"{sorted(_MOMENTUM_FIELDS)}; missing={missing}, extra={extra}"
        )
    if value["kind"] != _MOMENTUM_KIND:
        raise ValueError(
            "unknown checkpoint momentum wrapper kind: "
            f"{value['kind']!r}; expected {_MOMENTUM_KIND!r}"
        )
    data = value["data"]
    if not isinstance(data, torch.Tensor):
        raise TypeError(
            "checkpoint momentum wrapper data must be a torch.Tensor, "
            f"got {type(data).__name__}"
        )
    if not data.is_floating_point():
        raise TypeError(
            "checkpoint momentum wrapper data must have a floating dtype, "
            f"got {data.dtype}"
        )
    if data.ndim < 3:
        raise ValueError(
            "checkpoint momentum wrapper data must have at least three dimensions, "
            f"got shape {tuple(data.shape)}"
        )
    if not bool(torch.isfinite(data).all().item()):
        from locomgcm.deferred_checks import DeferredCheckError, report_numerical_failure

        # A non-finite state is a numerical failure of the run, not a malformed
        # payload: an owned window records it and keeps its communication
        # schedule; any other caller gets the channel's raise.
        report_numerical_failure(DeferredCheckError(
            "checkpoint momentum wrapper data contains non-finite values"))
    step_index = require_exact_int(
        value["step_index"], path="checkpoint momentum wrapper step_index"
    )
    return data, step_index


def _snapshot_copy(*, detach: bool, device_resident: bool):
    """The per-tensor copy of a snapshot: to the host, or on the tensor's device."""
    def copy(tensor):
        value = tensor.detach() if detach else tensor
        return value.clone() if device_resident else value.to(device="cpu", copy=True)
    return copy


def _host_clone(value: Any, *, detach: bool, path: str, device_resident: bool = False) -> Any:
    return _copy_tree(value, copy=_snapshot_copy(detach=detach, device_resident=device_resident),
                      path=path)


def _copy_tree(value: Any, *, copy, path: str) -> Any:
    """``value`` with every tensor replaced by ``copy(tensor)``; containers rebuilt."""
    if isinstance(value, torch.Tensor):
        return copy(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    if type(value) is dict:
        return {
            key: _copy_tree(item, copy=copy, path=f"{path}[{key!r}]")
            for key, item in value.items()
        }
    if type(value) in (list, tuple):
        items = [
            _copy_tree(item, copy=copy, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return type(value)(items)
    raise TypeError(f"{path} has unsupported restart-state type {type(value).__name__}")


def _validate_restart_value(value: Any, *, path: str) -> None:
    if isinstance(value, torch.Tensor) or value is None or type(value) in (
        str,
        int,
        float,
        bool,
    ):
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) not in (str, int, float, bool):
                raise TypeError(
                    f"{path} has unsupported mapping-key type {type(key).__name__}"
                )
            _validate_restart_value(item, path=f"{path}[{key!r}]")
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _validate_restart_value(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} has unsupported restart-state type {type(value).__name__}")


def _to_device(value: Any, *, device: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if type(value) is dict:
        return {key: _to_device(item, device=device, dtype=dtype) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_to_device(item, device=device, dtype=dtype) for item in value)
    return value
