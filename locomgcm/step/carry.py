"""Explicit regional integrator histories and their publication boundary."""

from dataclasses import dataclass, replace
from copy import copy
from enum import Enum
from math import isclose

import torch
from torch.utils import _pytree

from locomgcm.accel.tensor_context import ordinary_tensor_mode

from locomgcm.fields import Field
from locomgcm.state import PrognosticState
from .boundary_history import boundary_adapter


@dataclass(frozen=True)
class HydroHistory:
    """Committed etaH and corrected interface velocity (MIT upward-positive)."""

    eta_h: torch.Tensor
    w_interface: torch.Tensor


_pytree.register_pytree_node(
    HydroHistory,
    lambda history: ([history.eta_h, history.w_interface], None),
    lambda leaves, context: HydroHistory(*leaves),
)


@dataclass(frozen=True)
class OceanCarry:
    state: PrognosticState
    ab_history: dict[str, torch.Tensor]
    clock: dict[str, torch.Tensor]
    hydro_history: HydroHistory | None = None


class WindowMode(Enum):
    REINITIALIZE_HISTORY = "reinitialize_history"
    FREEZE_PREWINDOW_HISTORY = "freeze_prewindow_history"
    CONTINUE_DIFFERENTIABLE = "continue_differentiable"


_FIELDS = ("eta", "u", "v", "temp", "salt")
_HISTORY = ("mom_u", "mom_v", "mom_valid")


def _flatten(carry):
    fields = [getattr(carry.state, name) for name in _FIELDS]
    metadata = [(f.name, f.location, f.units, f.halo_shape, f.mask_ref, f.metadata)
                for f in fields]
    names = tuple(sorted(carry.ab_history))
    return ([f.data for f in fields]
            + [carry.ab_history[name] for name in names]
            + [carry.clock["time_seconds"], carry.clock["step_index"]]
            + ([carry.hydro_history] if carry.hydro_history is not None else []),
            (metadata, names, carry.hydro_history is not None))


def _unflatten(leaves, metadata):
    metadata, names, hydro_enabled = metadata
    state = PrognosticState(**{
        name: Field(meta[0], tensor, *meta[1:])
        for name, tensor, meta in zip(_FIELDS, leaves[:5], metadata)
    })
    offset = 5 + len(names)
    return OceanCarry(state, dict(zip(names, leaves[5:offset])),
                      dict(zip(("time_seconds", "step_index"), leaves[offset:offset + 2])),
                      leaves[offset + 2] if hydro_enabled else None)


_pytree.register_pytree_node(OceanCarry, _flatten, _unflatten)


def _supports_hydro_history(model):
    from locomgcm.boundaries import StructuredBoundaryCondition
    from locomgcm.dynamics import (
        DiagnosticVerticalVelocity, FCoriolis, HorizontalTracerAdvection,
        HydrostaticPressureGradient, LinearDrag, LinearEquationOfState,
        LinearFreeSurfaceContinuity, LinearFreeSurfacePressureGradient,
        MomentumAdvection, VerticalMixing, VerticalTracerTransport,
    )
    from locomgcm.steppers.implicit_free_surface import ImplicitFreeSurfaceStepper

    stepper = model.stepper
    if type(stepper) is not ImplicitFreeSurfaceStepper:
        return False
    modules = {type(module) for module in model.dynamics_modules}
    return (
        {LinearEquationOfState, HydrostaticPressureGradient,
             DiagnosticVerticalVelocity, MomentumAdvection} <= modules
        and modules <= {LinearFreeSurfaceContinuity, LinearFreeSurfacePressureGradient,
                        LinearEquationOfState, HydrostaticPressureGradient,
                        DiagnosticVerticalVelocity, MomentumAdvection,
                        HorizontalTracerAdvection, VerticalTracerTransport,
                        VerticalMixing, FCoriolis, LinearDrag}
        and type(model.boundary_condition) is StructuredBoundaryCondition
        and model.grid.face_count == 1
        and not model.grid.distributed_neighbors
        and not model.grid.boundary_topology.has_open_boundaries
        and model.config.dtype == torch.float64
        and torch.device(model.config.device).type == "cpu"
        and stepper.ab_eps > 0.
        and not stepper.nonhydrostatic.nonhydrostatic
        and stepper.nonlin_free_surf == 0 and stepper.select_r_star == 0
        and not stepper.stagger_tracer_stepping
        and not stepper.kpp_enabled
        and not stepper.temperature_implicit_vertical_advection
        and not stepper.salinity_implicit_vertical_advection
        and not stepper.momentum_implicit_vertical_advection
        and stepper.external_pnh_provider is None
    )


def supports_explicit_history(model):
    """Admission is based on the installed modules and static science options."""
    from locomgcm.boundaries import StructuredBoundaryCondition
    from locomgcm.dynamics import LinearDrag, TendencyStepDomain
    from locomgcm.steppers.implicit_free_surface import ImplicitFreeSurfaceStepper

    stepper = model.stepper
    modules = getattr(model, "dynamics_modules", ())
    return _supports_hydro_history(model) or (
        type(stepper) is ImplicitFreeSurfaceStepper
        and len(modules) == 1 and type(modules[0]) is LinearDrag
        and modules[0].stepper_domain == TendencyStepDomain.SLOW
        and modules[0].rate == .3
        and type(model.boundary_condition) is StructuredBoundaryCondition
        and getattr(model.grid, "face_count", 1) == 1
        and not getattr(model.grid, "distributed_neighbors", {})
        and model.config.dtype == torch.float64
        and torch.device(model.config.device).type == "cpu"
        and model.config.default_dt == .2
        and stepper.ab_eps == .1
        and not stepper.mom_forcing_out_ab
        and not stepper.ab2_on_temperature_tendency
        and not stepper.ab2_on_salinity_tendency
        and not stepper.stagger_tracer_stepping
        and not stepper.implicit_int_grav_wave
        and stepper.mom_dissip_in_ab and stepper.bottom_drag_in_ab
        and stepper.implicit_viscosity == 0.
        and not stepper.implicit_tracer_diffusion
        and stepper.tracer_vertical_diffusivity_kr == 0.
        and stepper.tracer_diffusivity_kh == 0.
        and not stepper.use_real_freshwater_flux
        and stepper.nonlin_free_surf == 0 and stepper.select_r_star == 0
        and not stepper.nonhydrostatic.nonhydrostatic
        and not stepper.temperature_implicit_vertical_advection
        and not stepper.salinity_implicit_vertical_advection
        and not stepper.momentum_implicit_vertical_advection
        and not stepper.kpp_enabled
        and stepper.external_pnh_provider is None
        and getattr(stepper, "_cached_total_phi_hyd", None) is None
    )


def validate_history(state, history, stepper):
    names = set(_HISTORY)
    for tracer, enabled in (("temp", stepper.ab2_on_temperature_tendency),
                            ("salt", stepper.ab2_on_salinity_tendency)):
        if enabled:
            names.update((tracer, tracer + "_valid"))
    if not isinstance(history, dict) or set(history) != names:
        raise ValueError("AB history requires momentum and complete enabled tracer pairs")
    references = {"mom_u": state.u.data, "mom_v": state.v.data,
                  **{name: getattr(state, name).data for name in ("temp", "salt")
                     if name in history}}
    for name, reference in references.items():
        value = history[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if (value.shape != reference.shape or value.dtype != reference.dtype
                or value.device != reference.device or value.stride() != reference.stride()):
            raise ValueError(f"{name} shape/dtype/device/stride mismatch")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite, including invalid payloads")
    for name in names - references.keys():
        valid = history[name]
        if (not isinstance(valid, torch.Tensor) or valid.shape != ()
                or valid.dtype != torch.bool or valid.device != state.u.data.device):
            raise ValueError(f"{name} must be a same-device scalar bool tensor")


def validate_hydro_history(state, history):
    if not isinstance(history, HydroHistory):
        raise ValueError("hydro history requires eta_h and w_interface")
    for name, shape in (("eta_h", state.eta.data.shape),
                        ("w_interface", (*state.temp.data.shape[:-3],
                                         state.temp.data.shape[-3] + 1,
                                         *state.temp.data.shape[-2:]))):
        value = getattr(history, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if (value.shape != shape or value.dtype != state.eta.data.dtype
                or value.device != state.eta.data.device or not value.is_contiguous()):
            raise ValueError(f"{name} shape/dtype/device/stride mismatch")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")


def _zero_history(state, stepper):
    history = {"mom_u": torch.zeros_like(state.u.data),
               "mom_v": torch.zeros_like(state.v.data),
               "mom_valid": torch.zeros((), dtype=torch.bool, device=state.u.data.device)}
    for name, enabled in (("temp", stepper.ab2_on_temperature_tendency),
                          ("salt", stepper.ab2_on_salinity_tendency)):
        if enabled:
            history[name] = torch.zeros_like(getattr(state, name).data)
            history[name + "_valid"] = torch.zeros_like(history["mom_valid"])
    return history


def _initial_hydro_history(model, state):
    from locomgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

    _, w_interface, _ = diagnose_vertical_velocity(state, model.grid)
    eta_h = model.stepper._initial_eta_h(
        state, model.grid, model.config.time_step_bundle().dt_freesurf,
        w_interface_to_down=w_interface,
    )
    return HydroHistory(eta_h.clone(), (-w_interface).clone())


def _publish(model, carry):
    model._ocean_carry = carry
    # This is a link to the adapter, never a second history owner.
    object.__setattr__(model.stepper, "_history_model", model)


def initialize(model, state, *, time_seconds, window_mode):
    if not supports_explicit_history(model):
        raise NotImplementedError("unsupported explicit history configuration")
    if not isinstance(window_mode, WindowMode):
        raise ValueError("window_mode must be a WindowMode")
    state.validate_against(model.grid)
    state = model.enforce_boundaries(state, record_budget=False)
    previous = getattr(model, "_ocean_carry", None)
    legacy_history = (model.stepper._momentum_ab is not None
                      or model.stepper._tracer_ab is not None)
    if _supports_hydro_history(model):
        legacy_history = (legacy_history or model.stepper._continuity.eta_h is not None
                          or model.stepper._continuity.w_interface_prev is not None)
    if legacy_history:
        raise ValueError("legacy cache history requires checkpoint migration before explicit initialization")
    if window_mode is WindowMode.REINITIALIZE_HISTORY or previous is None:
        history = _zero_history(state, model.stepper)
    else:
        history = {name: (value.detach().clone()
                         if window_mode is WindowMode.FREEZE_PREWINDOW_HISTORY else value.clone())
                   for name, value in previous.ab_history.items()}
    validate_history(state, history, model.stepper)
    hydro = None
    if _supports_hydro_history(model):
        if window_mode is WindowMode.REINITIALIZE_HISTORY or previous is None:
            hydro = _initial_hydro_history(model, state)
        else:
            validate_hydro_history(state, previous.hydro_history)
            hydro = _pytree.tree_map(
                lambda value: (value.detach().clone()
                               if window_mode is WindowMode.FREEZE_PREWINDOW_HISTORY
                               else value.clone()), previous.hydro_history)
    carry = OceanCarry(
        state.with_tensors(**{name: getattr(state, name).data.clone() for name in _FIELDS}),
        history,
        {"time_seconds": torch.tensor(time_seconds, dtype=torch.float64, device=state.u.data.device),
         "step_index": (previous.clock["step_index"].clone() if previous is not None
                        else torch.as_tensor(0 if model.stepper._step_index is None
                                             else model.stepper._step_index,
                                             dtype=torch.int64, device=state.u.data.device).clone())},
        hydro,
    )
    _publish(model, carry)
    model.current_time_seconds = time_seconds
    return carry


def advance(model, carry, *, dt, time_seconds):
    """Evaluate from the supplied carry, including during checkpoint replay."""
    from . import build_step_plan, finalize_step

    if not supports_explicit_history(model):
        raise NotImplementedError("unsupported explicit history configuration")
    validate_history(carry.state, carry.ab_history, model.stepper)
    if _supports_hydro_history(model):
        validate_hydro_history(carry.state, carry.hydro_history)
    if not isinstance(carry.clock, dict) or set(carry.clock) != {"time_seconds", "step_index"}:
        raise ValueError("carry clock requires time_seconds and step_index")
    for name, dtype in (("time_seconds", torch.float64), ("step_index", torch.int64)):
        value = carry.clock[name]
        if (not isinstance(value, torch.Tensor) or value.shape != ()
                or value.dtype != dtype or value.device != carry.state.u.data.device):
            raise ValueError(f"carry clock {name} must be a same-device scalar {dtype} tensor")
    if not isclose(float(carry.clock["time_seconds"]), time_seconds, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("carry clock does not match the requested step time")
    host = build_step_plan(model, carry.state, dt, time_seconds=time_seconds)
    host = replace(host, plan=replace(host.plan, step_index=carry.clock["step_index"]))
    outcome = model.step_kernel(carry.state, host.plan, carry=carry)
    finalize_step(model, outcome, host)
    return model._ocean_carry, outcome.diagnostics


@ordinary_tensor_mode()
def finalize_step(model, outcome, host):
    """Publish after the legacy boundary's persist_step_outputs copy succeeds."""
    from locomgcm.deferred_checks import inspect_step
    from .output_ownership import publish_value, shared_storages
    from .finalize import finalize_step as finalize_legacy
    from .plan import _step_index_tensor

    inspect_step(outcome.diagnostics, outcome.state)
    history = outcome.retained.get("ab_history")
    if getattr(model, "_ocean_carry", None) is not None and history is None:
        raise ValueError("explicit history candidate is missing AB history")
    candidate = None
    if history is not None:
        if outcome.retained["publication_source"] is not model._ocean_carry:
            raise ValueError("history candidate has a stale publication source")
        validate_history(outcome.state, history, model.stepper)
        if _supports_hydro_history(model):
            validate_hydro_history(outcome.state, outcome.retained.get("hydro_history"))
        source_clock = outcome.retained["source_clock"]
        next_clock = outcome.retained["clock"]
        if (not isclose(float(source_clock["time_seconds"]), host.times.current_seconds,
                        abs_tol=1e-12, rel_tol=1e-12)
                or float(next_clock["time_seconds"]) != host.times.future_seconds
                or int(source_clock["step_index"]) != int(host.plan.step_index)
                or int(next_clock["step_index"]) != int(source_clock["step_index"]) + 1):
            raise ValueError("history candidate has a mismatched clock")
        shared = shared_storages(history, outcome.retained["clock"],
                                 tuple(_pytree.tree_leaves(outcome.retained["hydro_history"])))
        candidate = publish_value(model, history, shared)
        clock = publish_value(model, outcome.retained["clock"], shared)
        hydro = _pytree.tree_map_only(
            torch.Tensor, lambda value: publish_value(model, value, shared),
            outcome.retained["hydro_history"])
    adapter = boundary_adapter(model)
    if adapter is not None:
        boundary_candidate = outcome.retained["boundary_history"]
        source_stamp = float(outcome.retained["boundary_source_stamp"])
        if (source_stamp != host.times.current_seconds
                or float(adapter.history.stamp) != source_stamp
                or float(boundary_candidate.stamp) != host.times.future_seconds
                or int(outcome.retained["boundary_step_index"]) != int(_step_index_tensor(model.stepper))):
            raise ValueError("boundary history candidate has a stale step or mismatched stamp")
    publication_model = model
    if "owned_step_state" in outcome.retained:
        # Counter promotion and retained-output copies may fail; finish them
        # on the candidate before replacing any committed stepper attributes.
        publication_model = copy(model)
        publication_model.stepper = copy(model.stepper)
        for name, value in outcome.retained["owned_step_state"].items():
            object.__setattr__(publication_model.stepper, name, value)
        object.__setattr__(publication_model.stepper, "_cached_total_phi_hyd", outcome.retained["total_phi_hyd"])
    state = finalize_legacy(publication_model, outcome, host)
    if candidate is not None:
        object.__setattr__(publication_model.stepper, "_step_index", clock["step_index"].clone())
    if publication_model is not model:
        for name in (*model.stepper.OWNED_STEP_FIELDS, "_cached_total_phi_hyd",
                     "cube_w_interface_readout"):
            object.__setattr__(model.stepper, name, getattr(publication_model.stepper, name, None))
        # The inputs the candidate step evaluated, which diagnostics read
        # (``evaluated_inputs``).  Absent stays absent, not None: the traced
        # step guards an existing attribute.
        evaluated = outcome.retained["evaluated_inputs"]
        for name in ("_evaluated_w_interface", "_evaluated_total_phi_hyd"):
            if name in evaluated:
                object.__setattr__(model.stepper, name, evaluated[name])
            else:
                vars(model.stepper).pop(name, None)
    if adapter is not None:
        adapter.history = boundary_candidate
        adapter.candidate = None
        adapter.active_stamp = None
    if candidate is not None:
        _publish(model, OceanCarry(state, candidate, clock, hydro))
    model.current_time_seconds = host.times.future_seconds
    return state


def checkpoint_carry(model, state, payload, *, time_seconds, detach=True, transfer=False):
    """Decode the active history group before any restore publication."""
    hydro_enabled = _supports_hydro_history(model)
    if (("__cached_total_phi_hyd" in payload and not hydro_enabled)
            or "_ab2_prev_tracer_tendency" in payload):
        raise ValueError("unsupported pressure or tracer history in drag-only checkpoint")
    group = payload.get("_ab2_prev_tendency")
    if "_ab2_prev_tendency" not in payload:
        if payload["_step_index"] > 0:
            raise ValueError("checkpoint is missing required momentum history")
        history = _zero_history(state, model.stepper)
    else:
        history = _zero_history(state, model.stepper)
        tracers = tuple(name for name in ("temp", "salt") if name in history)
        validity = {"mom_valid", *(name + "_valid" for name in tracers)}
        # Old checkpoints omit validity because every stored AB group was active.
        encoded_validity = isinstance(group, dict) and bool(validity & group.keys())
        names = {"u", "v", *tracers} | (validity if encoded_validity else set())
        if not isinstance(group, dict) or set(group) != names:
            raise ValueError("checkpoint AB history requires u, v and complete enabled tracers")
        history = {"mom_u": group["u"], "mom_v": group["v"],
                   "mom_valid": torch.ones((), dtype=torch.bool, device=state.u.data.device)}
        for name in tracers:
            history[name] = group[name]
            history[name + "_valid"] = torch.ones_like(history["mom_valid"])
        if encoded_validity:
            history.update({name: group[name] for name in validity})
        validate_history(state, history, model.stepper)
        history = {name: ((value.detach() if detach else value) if transfer
                         else (value.detach() if detach else value).clone())
                   for name, value in history.items()}
    hydro = None
    if hydro_enabled:
        if "_etaH" not in payload and "_w_interface_prev" not in payload and payload["_step_index"] == 0:
            hydro = _initial_hydro_history(model, model.enforce_boundaries(state, record_budget=False))
        else:
            if not {"_etaH", "_w_interface_prev"} <= payload.keys():
                raise ValueError("checkpoint requires etaH and interface velocity history")
            hydro = HydroHistory(payload["_etaH"], payload["_w_interface_prev"])
            validate_hydro_history(state, hydro)
            hydro = _pytree.tree_map_only(
                torch.Tensor, lambda value: ((value.detach() if detach else value) if transfer
                              else (value.detach() if detach else value).clone()), hydro)
    return OceanCarry(state, history,
                      {"time_seconds": torch.tensor(time_seconds, dtype=torch.float64,
                                                    device=state.u.data.device),
                       "step_index": torch.tensor(payload["_step_index"], dtype=torch.int64,
                                                  device=state.u.data.device)}, hydro)
