"""Checkpoint codec for the step-owned Stevens boundary history."""

import torch

from defgcm.step.boundary_history import BoundaryHistory, boundary_adapter

from .state import _copy_tree, _host_clone


def boundary_identity(boundary):
    from defgcm.boundaries.open import BoundarySide

    if getattr(boundary, "uses_orlanski", False):
        # The payload below carries the Stevens tracers only; MITgcm keeps the
        # Orlanski storage in its own pickup fields, which this codec does not write.
        raise NotImplementedError("checkpoints of an Orlanski open boundary are not supported")
    return {side.value: boundary.schemes.get(side).value for side in BoundarySide}


def encode_boundary(model, state, *, detach, device_resident=False):
    payload = _boundary_payload(model, state)
    return None if payload is None else _host_clone(payload, detach=detach, path="boundary_history",
                                                    device_resident=device_resident)


def head_boundary(model, state, *, snapshot=True):
    """``encode_boundary`` as device clones that keep their graph, for a segment head."""
    payload = _boundary_payload(model, state)
    return None if payload is None else _copy_tree(payload, copy=(torch.Tensor.clone
                                                   if snapshot else lambda value: value),
                                                   path="boundary_history")


def _boundary_payload(model, state):
    adapter = boundary_adapter(model)
    if adapter is None:
        return None
    history = adapter.history
    if history is None:
        history = BoundaryHistory.initial(state, model.current_time_seconds)
    return dict(prev_temp=history.prev_temp, prev_salt=history.prev_salt,
                valid=history.valid, stamp=history.stamp,
                schemes=boundary_identity(model.boundary_condition),
                units={"prev_temp": "degC", "prev_salt": "psu"},
                sampling="last_pre_tracer_bounded", shape=tuple(history.prev_temp.shape))


def validate_boundary_payload(payload, prognostic, *, time_seconds):
    if type(payload) is not dict:
        raise TypeError("boundary_history must be a mapping")
    required = {"prev_temp", "prev_salt", "valid", "stamp", "schemes", "units", "sampling", "shape"}
    if set(payload) != required:
        raise ValueError(f"boundary_history fields mismatch: missing={sorted(required - set(payload))}")
    for name in ("temp", "salt"):
        tensor, reference = payload["prev_" + name], prognostic[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != reference.shape
                or tensor.dtype != reference.dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f"boundary_history prev_{name} shape/dtype/finite mismatch")
    valid, stamp = payload["valid"], payload["stamp"]
    if not isinstance(valid, torch.Tensor) or valid.shape != () or valid.dtype != torch.bool:
        raise ValueError("boundary_history valid must be a scalar bool tensor")
    if (not isinstance(stamp, torch.Tensor) or stamp.shape != ()
            or stamp.dtype != torch.float64 or not torch.isfinite(stamp)
            or float(stamp) != time_seconds):
        raise ValueError("boundary_history stamp does not match the next-step clock")
    if (payload["units"] != {"prev_temp": "degC", "prev_salt": "psu"}
            or payload["sampling"] != "last_pre_tracer_bounded"
            or payload["shape"] != tuple(prognostic["temp"].shape)):
        raise ValueError("boundary_history sampling/shape/units mismatch")
    from defgcm.boundaries.open import BoundarySide, TracerBoundaryScheme

    schemes = payload["schemes"]
    if (type(schemes) is not dict or set(schemes) != {side.value for side in BoundarySide}
            or any(value not in {scheme.value for scheme in TracerBoundaryScheme}
                   for value in schemes.values())):
        raise ValueError("boundary_history schemes are invalid")


def decode_boundary(model, ckpt, *, new_run=False):
    adapter = boundary_adapter(model)
    payload = ckpt.step_state.get("boundary_history")
    saved_schemes = ckpt.meta.physics_flags.get("boundary_schemes")
    if saved_schemes is not None and (
        adapter is None or saved_schemes != boundary_identity(model.boundary_condition)
    ):
        raise ValueError("boundary_history configuration differs from checkpoint boundary schemes")
    if adapter is None:
        if payload is not None:
            raise ValueError("boundary_history conflicts with non-Stevens configuration")
        return None
    if payload is None:
        if not new_run:
            raise ValueError("Stevens continuation requires boundary_history")
    else:
        validate_boundary_payload(payload, ckpt.prognostic, time_seconds=ckpt.meta.model_time_seconds)
        if payload["schemes"] != boundary_identity(model.boundary_condition):
            raise ValueError("boundary_history schemes do not match active Stevens configuration")
        if not bool(payload["valid"]) and not new_run:
            raise ValueError("Stevens continuation requires valid boundary_history")
    return boundary_candidate(model, ckpt)


def boundary_candidate(model, ckpt):
    """The Stevens history ``ckpt`` installs, on the model device; ``decode_boundary`` unchecked."""
    if boundary_adapter(model) is None:
        return None
    boundary_identity(model.boundary_condition)  # refuses an Orlanski boundary
    payload = ckpt.step_state.get("boundary_history")
    if payload is None:
        temp = ckpt.prognostic["temp"].to(device=model.config.device, dtype=model.config.dtype)
        salt = ckpt.prognostic["salt"].to(device=model.config.device, dtype=model.config.dtype)
        return BoundaryHistory(torch.zeros_like(temp), torch.zeros_like(salt),
                               torch.zeros((), dtype=torch.bool, device=temp.device),
                               torch.tensor(ckpt.meta.model_time_seconds, dtype=torch.float64,
                                            device=temp.device), {})
    return BoundaryHistory(**{key: payload[key] for key in ("prev_temp", "prev_salt", "valid", "stamp")},
                           orlanski={}).to(
        model.config.device, model.config.dtype)
