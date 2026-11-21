"""Checkpoint capture & restore — *what* is captured, and how it goes back.

This is the CAPTURE layer.  It is the only place that reaches into the
live model/stepper, and it does so through a narrow, duck-typed contract
so the checkpoint package stays decoupled from stepper internals:

  * Prognostic state: the five fields off ``PrognosticState``.
  * Cross-step state: the closed allowlist and reserved codecs in
    ``checkpoint.state``.  Cache object type never determines ownership.

The capture (snapshot) phase moves tensors GPU->host so the persist
phase can run off the hot path (CheckFreq / PyTorch-DCP two-phase model).

``detach=True`` severs the autograd graph for restart; ``False`` retains
state and history derivatives for in-memory continuation and segment replay.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import fields, is_dataclass, replace
from datetime import date, datetime
from enum import Enum
import math
from pathlib import Path
import re
import reprlib
import subprocess
from collections.abc import Mapping
from typing import Any
import warnings

from locomgcm.fields import Field, FieldLocation
from locomgcm.state import PrognosticState

from .types import (
    CHECKPOINT_SCHEMA_VERSION,
    PROGNOSTIC_FIELDS,
    CheckpointMeta,
    ModelCheckpoint,
    require_nonempty_fingerprint,
)
from .state import (
    apply_decoded_step_state,
    decode_step_state,
    decoded_step_state,
    encode_step_state_from_stepper,
    momentum_implicit_enabled,
    require_exact_int,
    validate_checkpoint_structure,
    validate_momentum_wrapper_for_model,
)


# ---------------------------------------------------------------------------
# Schema-v2 cross-step state (de)serialisation.
# ---------------------------------------------------------------------------
def export_step_state(stepper: Any, *, detach: bool = True,
                      device_resident: bool = False) -> dict[str, Any]:
    """Export only present keys from the closed restart allowlist."""
    return encode_step_state_from_stepper(stepper, detach=detach, device_resident=device_resident)


def import_step_state(stepper: Any, state: dict[str, Any], *, device, dtype) -> None:
    """Validate/decode completely, then commit state to a fresh stepper."""
    decoded = decode_step_state(
        state,
        stepper=stepper,
        device=device,
        dtype=dtype,
    )
    apply_decoded_step_state(stepper, decoded)


# ---------------------------------------------------------------------------
# Whole-model capture / restore
# ---------------------------------------------------------------------------
def capture(
    model: Any,
    state: PrognosticState,
    *,
    global_step: int,
    model_time_seconds: float,
    dt: float,
    config_fingerprint: str | dict[str, Any],
    permanent: bool = False,
    detach: bool = True,
    device_resident: bool = False,
) -> ModelCheckpoint:
    """Snapshot ``model``+``state`` into a host-resident ``ModelCheckpoint``.

    This is the main-thread (snapshot) phase: GPU->host copies happen here;
    the returned object is fully decoupled from live device memory so a
    background writer can persist it without racing the integration loop.
    ``device_resident`` copies on each tensor's own device instead: a
    window's first segment head, used on the device at once.
    """
    global_step = require_exact_int(global_step, path="capture global_step")
    if type(model_time_seconds) not in (int, float) or isinstance(
        model_time_seconds, bool
    ) or not math.isfinite(model_time_seconds):
        raise TypeError("capture model_time_seconds must be a finite number")
    if type(dt) not in (int, float) or isinstance(dt, bool) or not math.isfinite(dt):
        raise TypeError("capture dt must be a finite number")
    require_nonempty_fingerprint(
        config_fingerprint, path="capture config_fingerprint"
    )
    prognostic = {}
    for name in PROGNOSTIC_FIELDS:
        t = getattr(state, name).data
        if detach:
            t = t.detach()
        prognostic[name] = t.clone() if device_resident else t.to(device="cpu", copy=True)

    step_state = export_step_state(model.stepper, detach=detach, device_resident=device_resident)
    from .boundary import encode_boundary

    boundary_history = encode_boundary(model, state, detach=detach, device_resident=device_resident)
    if boundary_history is not None:
        step_state["boundary_history"] = boundary_history
    from .nonlinear import encode_nonlinear_geometry

    geometry = encode_nonlinear_geometry(model, detach=detach, device_resident=device_resident)
    if geometry is not None:
        step_state["nonlinear_geometry"] = geometry
    validate_momentum_wrapper_for_model(step_state, model)
    if step_state["_step_index"] != global_step:
        raise ValueError(
            "checkpoint capture clock mismatch: "
            f"step_state['_step_index']={step_state['_step_index']}, "
            f"global_step={global_step}"
        )

    meta = checkpoint_meta(model, step_state, global_step=global_step,
                           model_time_seconds=model_time_seconds, dt=dt,
                           config_fingerprint=config_fingerprint,
                           permanent=permanent, requires_grad=not detach,
                           compute_device=state.u.data.device)
    ckpt = ModelCheckpoint(prognostic=prognostic, step_state=step_state, meta=meta)
    validate_checkpoint_structure(ckpt)
    return ckpt


def checkpoint_meta(model: Any, step_state: dict[str, Any], *, global_step: int,
                    model_time_seconds: float, dt: float,
                    config_fingerprint: str | dict[str, Any], permanent: bool,
                    requires_grad: bool, compute_device: torch.device) -> CheckpointMeta:
    """The header ``capture`` writes: clock, physics identity and provenance."""
    from locomgcm.accel.compile_resources import observation

    return CheckpointMeta(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        global_step=global_step,
        model_time_seconds=float(model_time_seconds),
        dt=float(dt),
        mom_forcing_out_ab=_mom_forcing_out_ab(model.stepper),
        physics_flags=physics_flags(model),
        config_fingerprint=config_fingerprint,
        conventions={
            "w_interface_prev": "MITgcm upward-positive",
            "depth": "LOCOMgcm downward-positive",
            **observation(model, global_step, compute_device),
        },
        git_head=_git_head(),
        created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        permanent=permanent,
        requires_grad=requires_grad,
        state_keys=tuple(sorted(step_state.keys())),
    )


def restore(
    model: Any,
    ckpt: ModelCheckpoint,
    *,
    config_fingerprint: str | dict[str, Any],
    new_run: bool = False,
    detach: bool = True,
) -> PrognosticState:
    """Inject ``ckpt`` into a freshly built ``model``; return the seed state.

    ``detach=False`` restores an in-memory differentiable continuation;
    both prognostic and history tensors retain their graph, with fresh storage.
    Restores the already bounded prognostic tensors and cross-step state,
    including both nonlinear geometry layers, and sets the model clock.
    """
    ckpt = checked_for_resume(model, ckpt, config_fingerprint)
    state = install_checkpoint(model, ckpt, new_run=new_run, detach=detach, checked=True)
    from locomgcm.accel.compile_resources import CONVENTION, PREVIOUS_CONVENTION

    stored = ckpt.meta.conventions
    convention = CONVENTION if CONVENTION in stored else PREVIOUS_CONVENTION
    model._restored_compiled_geometry = (
        ckpt.meta.global_step, stored.get(convention), convention,
    )
    return state


def checked_for_resume(model: Any, ckpt: ModelCheckpoint,
                       config_fingerprint: str | dict[str, Any]) -> ModelCheckpoint:
    """``ckpt`` after the resume checks, migrated to the current schema."""
    if type(ckpt.meta.schema_version) is int and ckpt.meta.schema_version == 3:
        from .migration import migrate_checkpoint_v3_to_current

        assert_resume_compatible(ckpt, model, config_fingerprint, legacy_v3=True)
        ckpt = migrate_checkpoint_v3_to_current(ckpt)
    assert_resume_compatible(ckpt, model, config_fingerprint)
    return ckpt


def install_checkpoint(model: Any, ckpt: ModelCheckpoint, *, new_run: bool,
                       detach: bool, checked: bool, transfer: bool = False,
                       prepare_next_step: bool = False) -> PrognosticState:
    """The model writes of ``restore``: owned copies of ``ckpt`` become the live state.

    ``checked=False`` skips the per-value payload checks for a checkpoint whose
    layout was checked earlier (a segment head); every write is the same.
    ``prepare_next_step`` gives an immediately following plan the START sample
    produced after all checkpoint state is installed, through the same
    one-consumption handoff as the runner. A boundary-history adapter keeps
    plan resampling because its begin operation changes the sampling context.
    """
    from torch.utils._pytree import tree_map_only
    import torch

    def owned(value):
        value = value.detach() if detach else value
        return value if transfer else value.clone()

    ckpt = replace(ckpt, step_state=tree_map_only(torch.Tensor, owned, ckpt.step_state))
    from .boundary import boundary_candidate as unchecked_boundary, decode_boundary
    from locomgcm.step.boundary_history import boundary_adapter

    if type(new_run) is not bool:
        raise TypeError("new_run must be an exact bool")
    if new_run and (ckpt.meta.global_step != 0 or ckpt.meta.model_time_seconds != 0.):
        raise ValueError("explicit new_run initialization requires step zero and time zero")
    boundary_candidate = (decode_boundary(model, ckpt, new_run=new_run) if checked
                          else unchecked_boundary(model, ckpt))
    device = model.config.device
    dtype = model.config.dtype

    tensors = {
        name: owned(ckpt.prognostic[name]).to(device=device, dtype=dtype)
        for name in PROGNOSTIC_FIELDS
    }
    seed = PrognosticState(
        eta=Field("eta", tensors["eta"], FieldLocation.CENTER, "m", mask_ref="center"),
        u=Field("u", tensors["u"], FieldLocation.U_FACE, "m s-1", mask_ref="u"),
        v=Field("v", tensors["v"], FieldLocation.V_FACE, "m s-1", mask_ref="v"),
        temp=Field("temp", tensors["temp"], FieldLocation.CENTER, "degC", mask_ref="center"),
        salt=Field("salt", tensors["salt"], FieldLocation.CENTER, "psu", mask_ref="center"),
    )
    if checked:
        seed.validate_against(model.grid)

    model_time = float(ckpt.meta.model_time_seconds)
    from locomgcm.step.carry import supports_explicit_history, checkpoint_carry

    decoded = (decode_step_state(ckpt.step_state, stepper=model.stepper,
                                 device=device, dtype=dtype) if checked
               else decoded_step_state(ckpt.step_state, device=device, dtype=dtype))
    if supports_explicit_history(model):
        # Validate the original precision and group structure before publication.
        candidate = checkpoint_carry(model, seed, ckpt.step_state, time_seconds=model_time,
                                     detach=detach, transfer=transfer)
        apply_decoded_step_state(model.stepper, decoded,
                                 history_model=model, history_carry=candidate,
                                 boundary_history_validated=True, model_geometry_validated=True)
    else:
        apply_decoded_step_state(model.stepper, decoded, boundary_history_validated=True,
                                model_geometry_validated=True)
    from .nonlinear import install_nonlinear_geometry

    install_nonlinear_geometry(model, decoded.get("nonlinear_geometry"))
    model.current_time_seconds = model_time
    if not prepare_next_step:
        prepare = getattr(model, "prepare_step_time", None)
        if callable(prepare):
            prepare(model_time)
    adapter = boundary_adapter(model)
    if adapter is not None:
        adapter.history = boundary_candidate
        adapter.candidate = None
        adapter.active_stamp = None
        adapter.initialization_reason = "explicit_new_run" if new_run else None
    if prepare_next_step:
        # Restore the head clock after boundary state. An adapter's begin
        # runs later in build_step_plan, so that path resamples there.
        prepare = (getattr(model, "_prepare_runner_step_time", None)
                   if adapter is None else None)
        if not callable(prepare):
            prepare = getattr(model, "prepare_step_time", None)
        if callable(prepare):
            prepare(model_time)
    return seed


# ---------------------------------------------------------------------------
# Identity / validation helpers
# ---------------------------------------------------------------------------
def physics_flags(model: Any) -> dict[str, Any]:
    """A small, human-readable fingerprint of physics-relevant switches."""
    stepper = model.stepper
    nh = getattr(stepper, "nonhydrostatic", False)
    flags = {
        "stepper": getattr(stepper, "name", type(stepper).__name__),
        "nonhydrostatic": bool(getattr(nh, "nonhydrostatic", nh)),
        "nonlin_free_surf": int(getattr(stepper, "nonlin_free_surf", 0) or 0),
        "select_r_star": int(getattr(stepper, "select_r_star", 0) or 0),
        "stagger_tracer_stepping": bool(
            getattr(stepper, "stagger_tracer_stepping", False)
        ),
        "momentum_implicit_vertical_advection": momentum_implicit_enabled(stepper),
        "nx": int(model.grid.nx),
        "ny": int(model.grid.ny),
        "nz": int(model.grid.nz),
        "assembly_policy": _assembly_policy_semantics(model),
    }
    layout = model.grid.parallel_layout
    flags["parallel_layout"] = (None if layout is None else {
        name: getattr(layout, name) for name in (
            "world_size", "rank", "nPx", "nPy", "global_nx", "global_ny",
            "x_start", "x_stop", "y_start", "y_stop")})
    if getattr(model.grid, "face_count", 1) == 6:
        flags["cube_advection_topology"] = getattr(model.grid, "advection_topology", None)
        ctx = getattr(model, "cube_ctx", None)
        flags["cube_layout"] = None if ctx is None else {
            "n": ctx.decomposition.n,
            "world_size": ctx.decomposition.n_ranks,
            "rank": ctx.rank,
            "face_to_rank": list(ctx.face_to_rank),
            "local_face_ids": list(ctx.local_face_ids),
        }
    from locomgcm.step.boundary_history import boundary_adapter
    from .boundary import boundary_identity

    if boundary_adapter(model) is not None:
        flags["boundary_schemes"] = boundary_identity(model.boundary_condition)
    return flags


def config_fingerprint(run_cfg: Any) -> dict[str, Any]:
    """Return the physics-determining config as plain data for resume checks.

    A checkpoint stores this mapping as-is and a resume compares it key by
    key (``assert_resume_compatible``); no digest is taken.
    Deliberately *excludes* output cadence and total run length — those
    may legitimately change across a resume (modify config -> continue).
    Includes grid, dt, physics, advection, boundary, and the stepper /
    dynamics manifest.
    """

    def _plain(obj: Any, *, path: str) -> Any:
        if hasattr(obj, "to_dict"):
            try:
                plain = obj.to_dict()
            except Exception as exc:
                raise TypeError(f"config fingerprint serialization failed at {path}.to_dict()") from exc
            return _plain(plain, path=path)
        if is_dataclass(obj) and not isinstance(obj, type):
            return {
                field.name: _plain(
                    getattr(obj, field.name), path=f"{path}.{field.name}"
                )
                for field in fields(obj)
            }
        if isinstance(obj, Mapping):
            out = {}
            for key, value in obj.items():
                if type(key) is not str:
                    raise TypeError(
                        f"config fingerprint mapping key at {path} must be an exact str, "
                        f"got {type(key).__name__}"
                    )
                out[key] = _plain(value, path=f"{path}.{key}")
            return {key: out[key] for key in sorted(out)}
        if type(obj) in (list, tuple):
            return [
                _plain(value, path=f"{path}[{index}]")
                for index, value in enumerate(obj)
            ]
        if isinstance(obj, Enum):
            return _plain(obj.value, path=f"{path}.value")
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if type(obj) is float:
            if not math.isfinite(obj):
                raise ValueError(
                    f"config fingerprint value at {path} must be finite, got {obj!r}"
                )
            return obj
        if type(obj) in (str, int, bool) or obj is None:
            return obj
        raise TypeError(
            f"config fingerprint value at {path} has unsupported type "
            f"{type(obj).__name__}"
        )

    grid = run_cfg.grid
    payload = {
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz},
        "dt": float(run_cfg.time.dt),
        "physics": _plain(run_cfg.physics, path="physics"),
        "advection": _plain(run_cfg.advection, path="advection"),
        "boundary": _plain(run_cfg.boundary, path="boundary"),
        "stepper": {
            "name": run_cfg.model.stepper.name,
            "kwargs": _plain(dict(run_cfg.model.stepper.kwargs), path="stepper.kwargs"),
        },
        "dynamics": [
            {
                "name": entry.name,
                "kwargs": _plain(
                    dict(entry.kwargs), path=f"dynamics[{index}].kwargs"
                ),
            }
            for index, entry in enumerate(run_cfg.model.dynamics)
        ],
    }
    # A second pass puts the raw grid sizes, dt and names under the same
    # plain-data check (finite floats, no foreign scalar types).
    return _plain(payload, path="config")


def assert_resume_compatible(ckpt: ModelCheckpoint, model: Any,
                             fingerprint: str | dict[str, Any],
                             *, legacy_v3: bool = False) -> None:
    """Fail loudly if a checkpoint cannot be safely resumed into ``model``."""
    validate_checkpoint_structure(ckpt, legacy_v3=legacy_v3)
    require_nonempty_fingerprint(fingerprint, path="current config_fingerprint")
    validate_momentum_wrapper_for_model(ckpt.step_state, model)
    current_mom_forcing_out_ab = _mom_forcing_out_ab(model.stepper)
    if ckpt.meta.mom_forcing_out_ab != current_mom_forcing_out_ab:
        raise ValueError(
            "checkpoint AB2 history semantics mismatch — refusing to resume.\n"
            f"  checkpoint mom_forcing_out_ab: {ckpt.meta.mom_forcing_out_ab}\n"
            f"  current mom_forcing_out_ab:    {current_mom_forcing_out_ab}\n"
            "Re-create or re-convert the checkpoint for the current setting, "
            "or first use an explicit legacy compatibility conversion mode. "
            "Direct resume never rewrites _ab2_prev_tendency semantics."
        )
    flags = physics_flags(model)
    if ckpt.meta.physics_flags.get("parallel_layout") != flags["parallel_layout"]:
        raise ValueError("checkpoint decomposition / ownership does not match model layout")
    if ckpt.meta.physics_flags.get("cube_layout") != flags.get("cube_layout"):
        raise ValueError("checkpoint cube ownership does not match model layout")
    # Existing single-rank files retain their fingerprint-based compatibility.
    # Newly captured files additionally bind the actual selected topology.
    if "cube_advection_topology" in ckpt.meta.physics_flags and (
        ckpt.meta.physics_flags["cube_advection_topology"]
        != flags.get("cube_advection_topology")
    ):
        raise ValueError("checkpoint cube advection topology does not match model")
    checkpoint_policy = ckpt.meta.physics_flags.get("assembly_policy")
    is_cube = getattr(model.grid, "face_count", 1) == 6
    if checkpoint_policy is None:
        legacy_policy = ckpt.meta.physics_flags.get("assembly_policy_fingerprint")
        if legacy_policy is not None:
            warnings.warn(
                "checkpoint records its resolved AssemblyPolicy only as a legacy "
                f"digest ({legacy_policy}); the policy inputs cannot be compared, "
                "so this resume continues without the assembly policy check."
            )
        elif is_cube:
            raise ValueError(
                "checkpoint is missing the resolved cube assembly policy "
                "— refusing to resume. Re-create or re-convert the checkpoint with "
                "the current policy resolver; direct resume never guesses cube AB2 "
                "history semantics."
            )
    else:
        differences = _plain_differences(
            checkpoint_policy, flags["assembly_policy"], "assembly_policy"
        )
        if differences:
            raise ValueError(
                "checkpoint resolved assembly policy mismatch — "
                "refusing to resume.\n"
                f"{_describe_differences(differences)}\n"
                "Re-create or re-convert the checkpoint for the current resolved "
                "AssemblyPolicy; direct resume never reinterprets AB2 history."
            )
    if ckpt.meta.physics_flags.get("nx") != flags["nx"] or (
        ckpt.meta.physics_flags.get("ny") != flags["ny"]
    ) or ckpt.meta.physics_flags.get("nz") != flags["nz"]:
        raise ValueError(
            "checkpoint grid does not match model grid: "
            f"{ckpt.meta.physics_flags} vs {flags}"
        )
    _check_config(ckpt.meta.config_fingerprint, fingerprint)


#: Form of the SHA-256 config digest that older checkpoints recorded.  Such a
#: checkpoint holds no config to compare against.
_LEGACY_CONFIG_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _check_config(recorded: Any, current: Any) -> None:
    """Compare the recorded config identity with the current one.

    Two config mappings are compared key by key and every differing key is
    reported.  A legacy digest resumed under a config mapping warns and
    continues.  Label strings from callers without a run config must be equal.
    """
    if type(recorded) is dict and type(current) is dict:
        differences = _plain_differences(recorded, current, "")
        if differences:
            raise ValueError(
                "checkpoint physics config mismatch — refusing to resume.\n"
                f"{_describe_differences(differences)}\n"
                "Output cadence / run length may change across a resume, but "
                "physics / grid / dt / stepper may not."
            )
        return
    if (type(recorded) is str and type(current) is dict
            and _LEGACY_CONFIG_DIGEST.fullmatch(recorded)):
        warnings.warn(
            f"checkpoint records its config only as a legacy digest ({recorded}); "
            "the physics config cannot be compared, so this resume continues "
            "without the config check."
        )
        return
    if recorded != current:
        raise ValueError(
            "checkpoint physics fingerprint mismatch — refusing to resume.\n"
            f"  checkpoint: {reprlib.repr(recorded)}\n"
            f"  current:    {reprlib.repr(current)}\n"
            "Output cadence / run length may change across a resume, but "
            "physics / grid / dt / stepper may not."
        )


_ABSENT = object()


def _plain_differences(recorded: Any, current: Any, path: str) -> list[tuple[str, Any, Any]]:
    """Return ``(path, recorded, current)`` wherever two plain-data trees differ.

    Leaves compare type-strictly, so ``1``, ``1.0`` and ``True`` count as
    different values although Python's ``==`` equates them; the earlier
    comparison through the config's JSON text told them apart as well.
    """
    if type(recorded) is dict and type(current) is dict:
        out = []
        for key in sorted(recorded.keys() | current.keys()):
            key_path = f"{path}.{key}" if path else key
            if key in recorded and key in current:
                out.extend(_plain_differences(recorded[key], current[key], key_path))
            else:
                out.append((key_path, recorded.get(key, _ABSENT),
                            current.get(key, _ABSENT)))
        return out
    if type(recorded) is list and type(current) is list and len(recorded) == len(current):
        out = []
        for index, (left, right) in enumerate(zip(recorded, current)):
            out.extend(_plain_differences(left, right, f"{path}[{index}]"))
        return out
    if type(recorded) is type(current) and recorded == current:
        return []
    return [(path, recorded, current)]


def _describe_differences(differences: list[tuple[str, Any, Any]]) -> str:
    def show(value: Any) -> str:
        return "<absent>" if value is _ABSENT else reprlib.repr(value)

    return "\n".join(
        f"  {path}: checkpoint {show(recorded)}, current {show(current)}"
        for path, recorded, current in differences
    )


def _mom_forcing_out_ab(stepper: Any) -> bool:
    if not hasattr(stepper, "mom_forcing_out_ab"):
        raise ValueError(
            f"stepper {type(stepper).__name__} does not expose the required "
            "mom_forcing_out_ab checkpoint semantic"
        )
    value = stepper.mom_forcing_out_ab
    if type(value) is not bool:
        raise TypeError(
            "stepper.mom_forcing_out_ab must be an exact bool, "
            f"got {type(value).__name__}"
        )
    return value


def _assembly_policy_semantics(model: Any) -> dict[str, bool]:
    from locomgcm.dynamics import AssemblyPolicy

    policy = getattr(model, "assembly_policy", None)
    if policy is None:
        resolved = getattr(model.stepper, "_resolved_assembly_policy", None)
        if resolved is None and getattr(model.grid, "face_count", 1) == 6:
            raise ValueError(
                "cube checkpoint model does not expose a resolved AssemblyPolicy. "
                "Construct the cube through TorchRegionalOceanModel so bottom-drag "
                "compatibility inputs are resolved before capture/resume."
            )
        policy = resolved or AssemblyPolicy.regional(model.stepper)
    if not isinstance(policy, AssemblyPolicy):
        raise TypeError(
            "model assembly policy checkpoint semantic must resolve to an "
            f"AssemblyPolicy, got {type(policy).__name__}"
        )
    return policy.semantics()


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"
