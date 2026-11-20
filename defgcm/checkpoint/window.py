"""A gradient window owns heads, replay, numerical judgment and publication."""

from __future__ import annotations

import contextlib
import itertools
import traceback
from dataclasses import replace
from functools import wraps

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map_only

from defgcm.deferred_checks import (
    DeferredCheckError, bind_window_owner, release_window_owner, report_numerical_failure,
    window_agreement, window_numerical_error,
)
from defgcm.distributed.comm_adjoint import _progress
from defgcm.step.runtime import segment_workspace
from defgcm.step.output_ownership import preserve_backward_buffers

from .capture import capture as snapshot
from defgcm.gradient_boundary import GradientBoundary
from .head import SegmentHeadSpec
from .window_schedule import WindowScheduler
from .store import CheckpointStore
from .types import ModelCheckpoint


_WINDOW_SERIALS = itertools.count()


def require_differentiable_window(model):
    """Reject at the window entry a configuration whose reverse is not built. **Eager.**

    A step compiled with fully dynamic shapes (``DEFGCM_COMPILE_DYNAMIC=1``) is
    refused: the window differentiates whatever step the caller configured,
    even one configured as forward-only.  Raised before the owner, the store or
    a session takes part, so every rank raises this error itself.
    """
    from defgcm.runner import refuse_dynamic_derivative

    refuse_dynamic_derivative(getattr(model, "_active_compile_config", None), "vjp",
                              site="defgcm.checkpoint.window.require_differentiable_window")


def _owned_operation(function):
    @wraps(function)
    def execute(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except Exception as error:
            self.terminate(error)
            raise
    return execute


class WindowOwner:
    """One terminal verdict, including aborted and never-admitted windows.

    ``capture`` consumes a one-element state box. The box is empty on return;
    callers keep only their declared controls. ``run`` returns observations;
    ``final`` exists only when requested. ``backward`` is the common reverse
    entry for local and distributed windows.
    """

    def __init__(self, *, session=None):
        self.session = session
        #: Process-local window number, for the communication round log.
        self.serial = next(_WINDOW_SERIALS)
        self.phase = "forward"
        self.cause = None
        self.numerical_error = None
        self.store = None
        self._running = None
        self._segments = []
        self.final = None
        self.attributed_error = None
        self.code_error = None
        self.failure_tracebacks = []
        self._failure_ids = set()
        self._configured = False
        self._terminal = False
        self._external_reverse = False
        self._objective_collecting = False
        self._window_reverse_active = False
        self._window_outputs = ()
        self._objective_cotangents = None
        self._objective_entry_link = None
        self._raw_boundaries = []
        self._nonowner_rejected = False
        self.bindings, self._control_versions = (), ()
        self.adjoint_bytes = 0
        self.state_finite = None
        self._objective_done = self._objective_failed = False
        bind_window_owner(self)

    @classmethod
    def for_model(cls, model):
        from defgcm.distributed.comm_adjoint import session_for_grid
        session = session_for_grid(model.grid)
        return cls() if session is None else session.window

    @classmethod
    def capture(cls, model, state_box, *, global_step, model_time_seconds, dt,
                config_fingerprint, need_final=False, final_device="cpu", store=None,
                scheduler=None):
        if not isinstance(state_box, list) or len(state_box) != 1:
            raise TypeError("window capture consumes a one-element state list")
        require_differentiable_window(model)
        window = cls.for_model(model)
        try:
            # The first head is used on the device at once: copy it there,
            # not through the host and back.
            initial = snapshot(model, state_box.pop(), global_step=global_step,
                               model_time_seconds=model_time_seconds, dt=dt,
                               config_fingerprint=config_fingerprint, detach=False,
                               device_resident=True)
            window.configure(model, initial, need_final=need_final, final_device=final_device,
                             store=store, scheduler=scheduler)
        except Exception as error:
            window.terminate(error)
            raise
        return window

    @_owned_operation
    def configure(self, model, initial, *, need_final=False, final_device="cpu",
                  store=None, scheduler=None, codec=False):
        if self._configured or self._terminal:
            raise RuntimeError("a WindowOwner can capture only one window")
        self.model, self.spec = model, SegmentHeadSpec(model, initial)
        self.store = CheckpointStore() if store is None else store
        self.scheduler = (WindowScheduler(device_bytes=self.store.device_bytes)
                          if scheduler is None else scheduler)
        if codec:
            self.scheduler = WindowScheduler(device_bytes=0, prefetch_distance=0)
        self.need_final, self.final_device, self.codec = need_final, final_device, codec
        self._running = self.spec.to_device(self.spec.initial)
        # Controls are chosen by requires_grad below; a forward-mode-only dual
        # would not be one and its tangent would vanish in _WindowGraph.
        _refuse_forward_mode("initial state")
        flat, _ = tree_flatten((self.spec.initial.prognostic, self.spec.initial.step_state))
        tensors = [v for v in flat if isinstance(v, torch.Tensor)]
        self._initial_positions = [i for i, v in enumerate(tensors) if v.requires_grad]
        self._initial_controls = tuple(tensors[i] for i in self._initial_positions)
        self._initial_devices = tuple(v.device for v in self._initial_controls)
        # Only metadata belongs to a schema. The window holds the sole live head.
        self.spec.initial = None
        self._configured = True

    @_owned_operation
    def run(self, *, steps, dt, observe, segment_steps, parameter_bindings=()):
        from defgcm.steppers.base import coerce_time_step_bundle
        if not self._configured or self.phase != "forward" or self._segments:
            raise RuntimeError("run requires a captured, unused window")
        if steps <= 0 or segment_steps <= 0:
            raise ValueError("steps and segment_steps must be positive")
        self.bundle = coerce_time_step_bundle(dt)
        self.bindings, self.observe = parameter_bindings, observe
        self.parameter_count = len(parameter_bindings)
        self._control_versions = tuple(value._version for _, _, value in parameter_bindings)
        self.counts = [min(segment_steps, steps - i) for i in range(0, steps, segment_steps)]
        # The window's forward runs inside _WindowGraph (a Function, where
        # PyTorch turns forward grads off) and the objective sees detached
        # outputs (_ObjectiveGuard), so a tangent would vanish; refuse it first.
        _refuse_forward_mode("objective")
        self.scheduler.open(self.store, len(self.counts), self.model.config.device,
                            steps=self.counts)
        parameters = tuple(value for _, _, value in parameter_bindings)
        # The scalar anchor admits constant objectives on ranks without controls.
        anchor = (torch.zeros((), device=self.model.config.device, requires_grad=True)
                  if self.session is None else self.session._link)
        try:
            outputs = _WindowGraph.apply(self, anchor, *parameters, *self._initial_controls)
            # The objective graph has no edge back into the window or its
            # initial communication link. Its guard records seeds only; the
            # owning entry drives the original graph after the whole objective.
            self._window_outputs = outputs
            if self.session is not None:
                # Objective communication must not traverse the initial link
                # already owned by the original window graph. Both phases keep
                # their own root, including ranks with a constant objective.
                self.session._link = self.session._link.detach().requires_grad_()
                self._objective_entry_link = self.session._link
            trigger = torch.zeros((), dtype=anchor.dtype, device=anchor.device, requires_grad=True)
            outputs = _ObjectiveGuard.apply(self, trigger, *[value.detach() for value in outputs])
            samples, *final, self._anchor_output = outputs
            if self.need_final:
                leaves = list(self._final_constants)
                for index, value in zip(self._final_positions, final):
                    leaves[index] = value
                prognostic, step_state = tree_unflatten(leaves, self._final_structure)
                self.final = ModelCheckpoint(prognostic, step_state, self._final_meta)
            self._initial_controls = ()
            return samples
        except Exception as error:
            self.terminate(error)
            raise

    def _forward(self, parameters):
        from .recompute import _ReplayState
        observations = []
        for slot, count in enumerate(self.counts):
            segment = _Segment(self, slot, count)
            segment.rng = _ReplayState((*parameters, *self._running.prognostic.values()))
            self._segments.append(segment)
            if self.session is None:
                outputs = segment(*parameters)
            else:
                outputs, segment.record = self.session.record_checkpoint(segment, parameters)
            observations.append(outputs[-1])
            del outputs
        result = [torch.cat(observations)]
        # Inspect the complete endpoint while its fields are still owned here.
        # Retain only a scalar, never the device endpoint for telemetry.  A
        # non-finite endpoint is a numerical failure of the window: it enters
        # the channel, so the admission verdict refuses the window on every rank.
        finite = {name: torch.isfinite(value).all()
                  for name, value in self._running.prognostic.items()}
        self.state_finite = torch.stack(tuple(finite.values())).all().detach()
        if not bool(self.state_finite):
            report_numerical_failure(DeferredCheckError(
                "non-finite prognostic field(s) at the window endpoint: "
                + ", ".join(name for name, ok in finite.items() if not bool(ok))))
        if self.need_final:
            head = self.spec.final(self.model, self._running, dt=self.bundle.dt_clock)
            self._final_meta = head.meta
            head = tree_map_only(torch.Tensor, lambda v: v.to(self.final_device),
                                 (head.prognostic, head.step_state))
            flat, self._final_structure = tree_flatten(head)
            self._final_positions = [i for i, v in enumerate(flat) if isinstance(v, torch.Tensor)]
            self._final_constants = [None if isinstance(v, torch.Tensor) else v for v in flat]
            result.extend(flat[i] for i in self._final_positions)
        self._running = None
        # The first reverse head can copy while the caller builds its objective.
        self.scheduler.begin_reverse()
        return tuple(result)

    def record_numerical_failure(self, error):
        self.numerical_error = window_numerical_error(self.numerical_error, error) or error
        if self.session is not None:
            self.session._numerical_error = self.numerical_error

    def _qualify(self, cleanup=None, *, verdict="admission"):
        if self.session is not None:
            return self.session._qualify(cleanup=cleanup, verdict=verdict)
        agreement = window_agreement()
        row = agreement.row(int(self.cause is not None or self.numerical_error is not None), "cpu")
        agreement.seal()
        if cleanup is not None:
            try:
                cleanup()
            except Exception as error:
                self.cause = self.cause or error
                row[0].fill_(1)
        failed, _, coded = agreement.verdict([row.tolist()], 0)
        self.code_error = coded
        cause = self._window_error(coded)
        message = "" if cause is None else str(cause)
        return failed, message, cause

    def _window_error(self, coded=None):
        """The window's one error: its numerical failures merged, else the local cause.

        Recorded failures, a numerical cause and the verdict's codes become one
        ``DeferredCheckError`` (each solver entry once); a non-numerical local
        cause becomes its ``__cause__``.  Without a numerical failure the
        local cause is the error, with its own type.
        """
        numerical = window_numerical_error(self.numerical_error, self.cause, coded)
        if numerical is not None:
            self.numerical_error = numerical
            local = None if isinstance(self.cause, DeferredCheckError) else self.cause
            if local is not None and local is not numerical and numerical.__cause__ is None:
                numerical.__cause__ = local
            return numerical
        return self.cause

    def _admission(self):
        """Consume the forward verdict once, irrespective of the local exit entry."""
        if self._terminal:
            raise RuntimeError("the gradient window has ended")
        if self.phase != "forward":
            return False, "", None
        if self._configured and tuple(value._version for _, _, value in self.bindings) != self._control_versions:
            self.cause = self.cause or RuntimeError("window controls changed before reverse")
            if self.session is not None:
                self.session.fail_transport(self.cause)
                self.terminate(self.cause)
        failed, message, cause = self._qualify()
        self.phase = "reverse"
        return failed, message, cause

    def admit(self):
        failed, message, cause = self._admission()
        if failed:
            self.terminate(cause or RuntimeError(message))

    def _release(self):
        errors = [self.cause, self.numerical_error]
        if self.session is not None:
            errors += [self.session._local_error, self.session._numerical_error]
        seen = set()
        while errors:
            error = errors.pop()
            if error is None or id(error) in seen:
                continue
            seen.add(id(error))
            if error.__traceback__ is not None and id(error) not in self._failure_ids:
                self.failure_tracebacks.append("".join(traceback.format_exception(
                    type(error), error, error.__traceback__, chain=False)))
                self._failure_ids.add(id(error))
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
            errors.extend((error.__cause__, error.__context__))
        self._running = None
        self._initial_controls = ()
        self._window_outputs = ()
        self._objective_cotangents = None
        self._objective_entry_link = None
        for boundary in self._raw_boundaries:
            boundary.outputs, boundary.cotangents = (), None
        self._raw_boundaries.clear()
        for segment in self._segments:
            segment.boundary = None
            segment.adjoint = None
            segment.solutions.clear()
            segment.kpp_roots.clear()
            segment._retained_order.clear()
        self._anchor_output = None
        self._segments.clear()
        cleanup_error = None
        if self.store is not None:
            try:
                self.store.release_all()
            except Exception as error:
                cleanup_error = error
        if self._configured and self.model is not None:
            from .state import release_working_step_state
            try:
                release_working_step_state(self.model)
            except Exception as error:
                cleanup_error = cleanup_error or error
            finally:
                self.model, self.spec = None, None
                self.observe, self.bindings = None, ()
        if cleanup_error is not None:
            self.cause = self.cause or cleanup_error
        if self.phase == "aborted" or self.cause is not None or self.numerical_error is not None:
            self.final = None
        if self.session is not None:
            self.session._tape.clear()
            self.session._link = self.session._entry
        if cleanup_error is not None:
            raise cleanup_error

    def terminate(self, error=None):
        """Seal causes, attribute codes, release resources, agree, then allow publication."""
        if self._terminal:
            return
        if self.cause is None:
            self.cause = error
        try:
            if self.session is not None and self.cause is not None:
                self.session.record_failure(self.cause)
                if not isinstance(self.cause, DeferredCheckError) and not self.session.transport_broken:
                    # A local exception cannot safely reconstruct the peers'
                    # outstanding work. Abort before any terminal collective.
                    self.session.fail_transport(self.cause)
            # Every forward exit consumes admission and completion in that
            # order, even when another participant enters through backward.
            if self.session is None or not self.session.transport_broken:
                failed, message, cause = self._admission()
                if failed and self.cause is None:
                    self.cause = cause or RuntimeError(message)
                    if self.session is not None:
                        self.session.record_failure(self.cause)
            if self.session is not None and self.session.transport_broken:
                self.phase = self.session._phase = "broken"
                raise self.session.transport_error() from self.cause
            if self.session is not None:
                try:
                    self.session._transport.quiesce()
                except Exception as transport_error:
                    self.session.fail_transport(transport_error)
                    raise self.session.transport_error() from transport_error
            failed, message, cause = self._qualify(cleanup=self._release)
            self.attributed_error = cause
            self.phase = "aborted" if failed else "complete"
            if self.session is not None:
                self.session._phase = self.phase
            if failed:
                if self.session is None:
                    raise cause if cause is not None else RuntimeError("gradient window rejected")
                from defgcm.distributed.comm_adjoint import CommSessionError
                raise CommSessionError(
                    "gradient window rejected; optimizer commit is invalid\n" + message) from cause
        finally:
            self._terminal = True
            if self.session is not None and self.session.transport_broken:
                self.phase = self.session._phase = "broken"
            try:
                self._release()
            except Exception as cleanup_error:
                if self.session is not None:
                    self.session.fail_transport(cleanup_error)
                    raise self.session.transport_error() from cleanup_error
                raise
            finally:
                try:
                    if self.session is not None:
                        try:
                            self.session._transport.close()
                        except Exception as transport_error:
                            self.session.fail_transport(transport_error)
                            self.phase = self.session._phase = "broken"
                            raise self.session.transport_error() from (self.cause or transport_error)
                finally:
                    release_window_owner(self)

    @_owned_operation
    def backward(self, loss):
        if self._terminal:
            raise RuntimeError("the gradient window has ended")
        if self._nonowner_rejected:
            raise RuntimeError("the objective graph was traversed outside its owning entry; discard this window")
        # Before admission, the objective reverse and any session round: the
        # segment heads would only refuse it during recomputation.
        _refuse_forward_mode("reverse")
        if self._configured and not self._segments:
            self.terminate(RuntimeError("reverse admission requires a completed forward window"))
        if not self._configured:
            roots = (loss, *(value for boundary in self._raw_boundaries for value in boundary.outputs))
            with _GradientPublication(*roots, internal=(() if self.session is None
                                                       else (self.session._entry,))) as publication:
                self._external_reverse = True
                try:
                    if self.session is None:
                        self.admit()
                        result = torch.autograd.backward(loss)
                        self._objective_verdict(False)
                        publication.prepare()
                        self.terminate()
                    else:
                        self.session._prepare_reverse()
                        objective_roots = [self.session._link]
                        seeds = [torch.zeros_like(self.session._link)]
                        if loss.requires_grad:
                            objective_roots.insert(0, loss)
                            seeds.insert(0, torch.ones_like(loss))
                        self._objective_collecting = True
                        try:
                            retained = bool(self._raw_boundaries)
                            buffers = (preserve_backward_buffers() if retained
                                       else contextlib.nullcontext())
                            with buffers:
                                torch.autograd.backward(objective_roots, seeds,
                                                        retain_graph=retained)
                        finally:
                            self._objective_collecting = False
                        self._objective_verdict(False)
                        self._window_reverse_active = True
                        try:
                            result = self.session._reverse(
                                None, reverse=self._reverse_raw, before_terminate=publication.prepare)
                        finally:
                            self._window_reverse_active = False
                except Exception as error:
                    self.terminate(error)
                    raise
                finally:
                    self._external_reverse = False
                publication.publish()
                return result
        objective_link = (None if self._objective_entry_link is None else self.session._link)
        roots_to_stage = (loss, *self._window_outputs,
                          *(value for boundary in self._raw_boundaries for value in boundary.outputs),
                          *(() if objective_link is None else (objective_link,)))
        with _GradientPublication(*roots_to_stage) as publication:
            self._external_reverse = True
            try:
                self.admit()
                progress_start = (None if self.session is None else _progress(self.session.group_name))
                # The anchor ensures the guard records even a constant local
                # objective. No window edge is reachable in this traversal.
                roots = [self._anchor_output]
                seeds = [torch.zeros_like(self._anchor_output)]
                if loss.requires_grad:
                    roots.insert(0, loss)
                    seeds.insert(0, torch.ones_like(loss))
                if objective_link is not None:
                    roots.append(objective_link)
                    seeds.append(torch.zeros_like(objective_link))
                self._objective_collecting = True
                try:
                    retained = bool(self._raw_boundaries)
                    buffers = (preserve_backward_buffers()
                               if retained and self.session is not None
                               else contextlib.nullcontext())
                    with buffers:
                        torch.autograd.backward(roots, seeds, retain_graph=retained)
                finally:
                    self._objective_collecting = False
                if self._objective_cotangents is None:
                    raise RuntimeError("the objective did not reach this window's guard")
                self._objective_verdict(False)
                self._window_reverse_active = True
                try:
                    raw_outputs, raw_seeds = self._raw_reverse_roots()
                    GradientBoundary.backward((*self._window_outputs, *raw_outputs),
                                              (*self._objective_cotangents, *raw_seeds))
                finally:
                    self._window_reverse_active = False
                if self.session is not None:
                    executed = _progress(self.session.group_name) - progress_start
                    schedule = self.session._schedule()
                    if executed != len(schedule):
                        self.session.record_failure(RuntimeError(
                            f"reverse executed {executed} rounds, planned {len(schedule)}"))
                    replayed = sum(len(getattr(record, "rounds", ())) for record in self.session._tape)
                    self.session._reverse_token = torch.full_like(self.session.token, executed - replayed)
                if not self._terminal:
                    publication.prepare()
                    self.terminate()
            except Exception as error:
                if self._terminal:
                    raise
                self.terminate(error)
                raise
            finally:
                self._external_reverse = False
            publication.publish()
        return None if self.session is None else self.session._reverse_token

    @_owned_operation
    def guard_raw(self, *outputs):
        """Separate each raw session boundary from the complete objective."""
        from defgcm.accel.graph_exit import graph_exit
        # Resource registration and an owner-bound autograd context are an
        # explicit eager exit, as are the runner's solver/diagnostic boundaries.
        with graph_exit(), torch.enable_grad():
            if self._terminal:
                raise RuntimeError("the gradient window has ended")
            boundary = _RawBoundary(outputs)
            self._raw_boundaries.append(boundary)
            return _RawObjectiveGuard.apply(self, boundary, self.session._entry, *outputs)

    def _raw_reverse_roots(self):
        outputs, cotangents = [], []
        for boundary in self._raw_boundaries:
            outputs.extend(boundary.outputs)
            cotangents.extend(boundary.cotangents or (None,) * len(boundary.outputs))
        return outputs, cotangents

    def _reverse_raw(self):
        # One original traversal preserves shared local subgraphs and the
        # link-ordered aggregation of all checkpoint/communication seeds.
        GradientBoundary.backward(*self._raw_reverse_roots())

    def abort_all(self, reason):
        self.terminate(RuntimeError(reason))

    def _objective_verdict(self, failed):
        """Judge the complete objective before driving any window edge."""
        if self._objective_done:
            return
        self._objective_done = True
        if failed and self.cause is None:
            self.cause = RuntimeError("objective reverse failed")
        if self.cause is not None and not isinstance(self.cause, DeferredCheckError):
            self.terminate(self.cause)
        rejected, message, cause = self._qualify(verdict="objective")
        self._objective_failed = rejected
        if rejected:
            self.terminate(cause or RuntimeError(message))

    def _reverse(self, sample_seed, final_seeds, parameters):
        if not self._external_reverse:
            self.admit()
        if self.session is not None:
            from defgcm.distributed.comm_adjoint import _progress
            progress_start = _progress(self.session.group_name)
            expected = sum(2 * len(segment.record.rounds) for segment in self._segments)
        total = [None] * len(parameters)
        adjoint = tuple(final_seeds) if self.need_final else None
        offset = sum(self.counts)
        for segment in reversed(self._segments):
            offset -= segment.count
            observation_seed = None if sample_seed is None else sample_seed[offset:offset + segment.count]
            segment.reverse = True
            segment.adjoint = adjoint
            gradients, adjoint = segment.vjp(parameters, observation_seed)
            if segment.slot > 0:
                storage = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
                           for v in adjoint or () if v is not None
                           and v.device.type == self.store._device.type}
                self.adjoint_bytes = max(self.adjoint_bytes, sum(storage.values()))
            segment.adjoint = None
            if self.codec and adjoint is not None:
                adjoint = tuple(None if v is None else v.to("cpu") for v in adjoint)
            segment.boundary = None
            for index, gradient in enumerate(gradients):
                if gradient is not None:
                    total[index] = gradient if total[index] is None else total[index] + gradient
        initial = [None if adjoint is None or adjoint[i] is None else adjoint[i].to(device)
                   for i, device in zip(self._initial_positions, self._initial_devices)]
        if self.session is not None:
            executed = _progress(self.session.group_name) - progress_start
            if executed != expected:
                self.session._local_error = self.session._local_error or RuntimeError(
                    f"window reverse executed {executed} rounds, planned {expected}")
            self.session._reverse_token = torch.full_like(self.session.token, executed // 2)
        if not self._external_reverse:
            self.terminate()
        return (*total, *initial)


def backward(loss, *, controls=(), diagnostics=None):
    """Own a library reverse, optionally materializing unused control gradients.

    Declared controls must be gradient-requiring leaves. Supplying them also
    requires a diagnostics dict: an absent autograd path alone cannot distinguish
    a constant objective from detach, no_grad or a stale constant cache. The dict
    records each missing VJP as ``unverified_disconnected``; callers must check
    those dependencies independently before accepting the derivative. In
    particular, a flat Taylor experiment cannot certify an absent dependency.

    Existing gradients accumulate as usual. Zeros are published only after the
    owning window completes its checks, including for a constant objective.
    """
    from defgcm.deferred_checks import _bound_owner
    controls = tuple(controls)
    if controls:
        if not isinstance(diagnostics, dict):
            raise ValueError("declared controls require a gradient diagnostics dict")
        if any(not isinstance(value, torch.Tensor) or not value.is_leaf
               or not value.requires_grad for value in controls):
            raise ValueError("controls must be gradient-requiring leaf tensors")
        if len({id(value) for value in controls}) != len(controls):
            raise ValueError("controls must not contain duplicate leaves")
        if loss.numel() != 1:
            raise ValueError("the control objective must be scalar")
    present = [False] * len(controls)
    hooks = []
    owner = _bound_owner()
    if owner is None:
        owner = WindowOwner()
    try:
        for index, value in enumerate(controls):
            def observed(gradient, index=index):
                present[index] = present[index] or gradient is not None
            hooks.append(value.register_hook(observed))
        # Preserve the objective's bits and keep its absence of dependencies
        # explicit. The isolated leaf only lets the owner finish its transaction.
        reverse_loss = (loss.detach().requires_grad_(True)
                        if controls and not loss.requires_grad else loss)
        owner.backward(reverse_loss)
    finally:
        for hook in hooks:
            hook.remove()
    if controls:
        materialized = [value.grad is None for value in controls]
        for value in controls:
            if value.grad is None:
                value.grad = torch.zeros_like(value)
        diagnostics.update(
            objective_requires_grad=loss.requires_grad,
            control_vjp_present=present,
            materialized_zero=materialized,
            dependency_status=("autograd_connected" if all(present)
                               else "unverified_disconnected"),
            requires_independent_check=not all(present),
        )


class _ObjectiveGuard(torch.autograd.Function):
    """Detached output proxies whose owned backward records absent seeds as None."""

    @staticmethod
    def forward(ctx, owner, trigger, *outputs):
        GradientBoundary.configure(ctx)
        ctx.owner = owner
        return tuple(value.view_as(value) for value in outputs)

    @staticmethod
    def backward(ctx, *cotangents):
        owner = ctx.owner
        if not owner._external_reverse or not owner._objective_collecting:
            from defgcm.distributed.comm_adjoint import CommSessionError
            error_type = RuntimeError if owner.session is None else CommSessionError
            error = error_type("a checkpoint window is differentiated through WindowOwner.backward "
                               "or CommSession.backward")
            owner._nonowner_rejected = True
            owner.cause = owner.cause or error
            raise error
        if torch.is_grad_enabled() or not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("checkpoint windows support first-order WindowOwner.backward, not autograd.grad")
        owner._objective_cotangents = cotangents
        # Detached data inputs and an independent trigger stop this traversal
        # here, including any initial communication link of the window.
        return (None, None, *[None] * len(cotangents))


class _RawBoundary:
    def __init__(self, outputs):
        self.outputs, self.cotangents = outputs, None


class _RawObjectiveGuard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, boundary, trigger, *outputs):
        GradientBoundary.configure(ctx)
        ctx.owner, ctx.boundary = owner, boundary
        return tuple(value.view_as(value) for value in outputs)

    @staticmethod
    def backward(ctx, *cotangents):
        owner = ctx.owner
        if not owner._external_reverse or not (owner._objective_collecting or owner._window_reverse_active):
            from defgcm.distributed.comm_adjoint import CommSessionError
            error = CommSessionError("a communication window is differentiated through CommSession.backward")
            owner._nonowner_rejected = True
            owner.cause = owner.cause or error
            raise error
        if torch.is_grad_enabled() or not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("communication windows support first-order CommSession.backward, not autograd.grad")
        if owner._objective_collecting:
            ctx.boundary.cotangents = cotangents
            return (None, None, None, *[None] * len(cotangents))
        return (None, None, None, *cotangents)


class _WindowGraph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, anchor, *controls):
        GradientBoundary.configure(ctx)
        ctx.owner = owner
        ctx.anchor_like = (anchor.shape, anchor.dtype, anchor.device)
        ctx.save_for_backward(*controls[:owner.parameter_count])
        outputs = owner._forward(controls[:owner.parameter_count])
        return (*outputs, torch.zeros_like(anchor))

    @staticmethod
    def backward(ctx, sample_seed, *tail):
        owner = ctx.owner
        if not owner._external_reverse or not owner._window_reverse_active:
            from defgcm.distributed.comm_adjoint import CommSessionError
            error_type = RuntimeError if owner.session is None else CommSessionError
            raise error_type("a checkpoint window is differentiated through WindowOwner.backward "
                             "or CommSession.backward")
        if torch.is_grad_enabled() or not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("checkpoint windows support first-order WindowOwner.backward, not autograd.grad")
        gradients = owner._reverse(sample_seed, tail[:-1], ctx.saved_tensors)
        shape, dtype, device = ctx.anchor_like
        link = (None if owner.session is None else torch.zeros(shape, dtype=dtype, device=device))
        return (None, link, *gradients)


class _Segment:
    """Metadata and a slot handle; the executor consumes each installed head."""

    def __init__(self, owner, slot, count):
        self.owner, self.slot, self.count = owner, slot, count
        self.reverse = False
        self.boundary = None
        self.adjoint = None
        # Accepted roots travel with the segment head, then enter recompute.
        self.solutions = []
        self.kpp_roots = []
        self._retained_order = []

    def seeds(self, observation):
        return (*(self.adjoint if self.adjoint is not None else (None,) * self.output_count), observation)

    def vjp(self, parameters, observation_seed):
        """The recompute graph ends in this scope; only its cotangents escape."""
        session = self.owner.session
        with self.rng.restored():
            if session is None:
                controls = [p.detach().requires_grad_(p.requires_grad) for p in parameters]
                with torch.enable_grad():
                    outputs = self(*controls)
                GradientBoundary.backward(outputs, self.seeds(observation_seed))
                gradients = [p.grad for p in controls]
            else:
                _, gradients = session._replay(self.record, self.seeds(observation_seed),
                                               torch.zeros_like(session._entry))
        from defgcm.step.output_ownership import publish_value
        adjoint = (publish_value(self.owner.model, self.boundary.cotangents)
                   if self.boundary is not None else None)
        return gradients, adjoint

    def __call__(self, *parameters):
        from .recompute import _bind
        from defgcm.solvers.cg2d import segment_solutions
        from defgcm.dynamics._kpp_convective_replay import segment_roots
        owner = self.owner
        model = owner.model
        groups = (self.solutions, self.kpp_roots)
        if self.reverse:
            tensors = owner.scheduler.take(self.slot)
            # Kept roots follow the head, in the forward's per-step order.
            tensors, kept = tensors[:len(self.positions)], tensors[len(self.positions):]
            for value, (kind, index) in zip(kept, self._retained_order, strict=True):
                _, *metadata = groups[kind][index]
                groups[kind][index] = (value, *metadata)
            del kept
        else:
            head, owner._running = owner._running, None
            values, self.structure = tree_flatten((head.prognostic, head.step_state))
            self.positions = [i for i, v in enumerate(values) if isinstance(v, torch.Tensor)]
            self.constants = [None if isinstance(v, torch.Tensor) else v for v in values]
            self.meta = head.meta
            tensors = tuple(values[i] for i in self.positions)
            owner.scheduler.save(self.slot, tensors)
            del values, head
        with segment_workspace(model), torch.enable_grad(), contextlib.ExitStack() as bound, \
                segment_solutions(self.solutions, replay=self.reverse), \
                segment_roots(self.kpp_roots, replay=self.reverse):
            # A leaf's AccumulateGrad owns its data until backward completes.
            # Receive head cotangents through a handle instead: neither this
            # executor nor that autograd edge retains a consumed head tensor.
            self.boundary = _HeadBoundary(tensors)
            del tensors
            head = self.boundary.head(self, model.config.device)
            for (target, name, _), value in zip(owner.bindings, parameters):
                value = value if self.reverse else value.detach().requires_grad_(value.requires_grad)
                _bind(target, name, value, bound)
            # A failed install/prepare must not lend its sample to a later
            # call at the same clock, possibly with different bound controls.
            if hasattr(model, "_runner_step_time"):
                bound.callback(setattr, model, "_runner_step_time", None)
            state = owner.spec.inject(model, head, new_run=self.meta.global_step == 0
                                      and self.meta.model_time_seconds == 0.)
            del head
            samples = []
            for step in range(self.count):
                before = tuple(len(entries) for entries in groups)
                state = model.step(state, dt=owner.bundle,
                    time_seconds=self.meta.model_time_seconds + step * owner.bundle.dt_clock)
                samples.append(owner.observe(model, state))
                if not self.reverse:
                    # The scheduler predicts later attachments from the first
                    # step's sizes, so keep solver families grouped per step.
                    for kind, entries in enumerate(groups):
                        self._retained_order.extend(
                            (kind, index) for index in range(before[kind], len(entries)))
            result = owner.spec.readout(model, state, global_step=self.meta.global_step + self.count,
                model_time_seconds=self.meta.model_time_seconds + self.count * owner.bundle.dt_clock)
            leaves, _ = tree_flatten((result.prognostic, result.step_state))
            outputs = [v for v in leaves if isinstance(v, torch.Tensor)]
            self.output_count = len(outputs)
            samples = torch.stack(samples)
            if not self.reverse:
                # Root copies use the head's tier and the store's stream.
                owner.scheduler.attach(
                    self.slot, [groups[kind][index][0] for kind, index in self._retained_order])
                for entries in groups:
                    entries[:] = [(None, *metadata) for _, *metadata in entries]
                prognostic, step_state = tree_map_only(torch.Tensor, torch.Tensor.detach,
                                                     (result.prognostic, result.step_state))
                owner._running = replace(result, prognostic=prognostic, step_state=step_state)
                return (*[v.detach() for v in outputs], samples.detach())
            return (*outputs, samples)


def _refuse_forward_mode(where: str) -> None:
    """Refuse a live forward mode at a window boundary that would drop its tangent.

    The window carries first-order reverse derivatives only.  The reading is
    program state (``forward_mode_active``): ranks that all open the same dual
    level refuse together.
    """
    from defgcm.accel.tensor_context import forward_mode_active

    if forward_mode_active():
        raise NotImplementedError(
            f"checkpoint gradient window ({where}): forward-mode derivatives are not "
            f"supported; the window provides first-order reverse derivatives only. "
            f"Take forward-mode derivatives without a checkpoint window "
            f"(ModelRunner.run with track_gradients=True)")


class _HeadBoundary:
    """Transfer input storage immediately and receive only its eventual cotangents."""

    def __init__(self, tensors):
        self.values = tensors
        self.cotangents = (None,) * len(tensors)

    def head(self, segment, device):
        # _HeadEdge detaches the transferred head, tangent included.
        _refuse_forward_mode("segment head")
        anchor = torch.zeros((), device=device, requires_grad=True)
        tensors = _HeadEdge.apply(self, anchor)
        values = list(segment.constants)
        for index, tensor in zip(segment.positions, tensors):
            values[index] = tensor
        prognostic, step_state = tree_unflatten(values, segment.structure)
        return ModelCheckpoint(prognostic, step_state, segment.meta)


class _HeadEdge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boundary, anchor):
        GradientBoundary.configure(ctx)
        ctx.boundary = boundary
        values, boundary.values = boundary.values, ()
        return tuple(value.detach() for value in values)

    @staticmethod
    def backward(ctx, *cotangents):
        ctx.boundary.cotangents = cotangents
        return None, None


class _GradientPublication:
    """Stage leaf accumulation until the complete reverse transaction agrees."""

    def __init__(self, *roots, internal=()):
        self.leaves, seen, pending = {}, set(), [v.grad_fn for v in roots if v.requires_grad]
        for value in roots:
            if value.is_leaf and value.requires_grad:
                self.leaves[id(value)] = value
        while pending:
            node = pending.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            value = getattr(node, "variable", None)
            if value is not None:
                self.leaves[id(value)] = value
            pending.extend(n for n, _ in node.next_functions)
        for value in internal:
            self.leaves.pop(id(value), None)
        self.before = {key: value.grad for key, value in self.leaves.items()}
        self.pending, self.hooks = {}, []
        self.candidates = None

    def __enter__(self):
        try:
            for key, value in self.leaves.items():
                value.grad = None
                def stage(leaf, key=key):
                    gradient = leaf.grad
                    if gradient is not None:
                        previous = self.pending.get(key)
                        self.pending[key] = gradient if previous is None else previous + gradient
                    leaf.grad = None
                self.hooks.append(value.register_post_accumulate_grad_hook(stage))
        except Exception:
            self.__exit__()
            raise
        return self

    def prepare(self):
        """Allocate all accumulated leaf results before the terminal agreement."""
        candidates = {}
        for key, value in self.leaves.items():
            gradient, before = self.pending.get(key), self.before[key]
            candidates[key] = (before if gradient is None else gradient if before is None
                               else before + gradient)
            candidate = candidates[key]
            if candidate is not None and not bool(torch.isfinite(candidate).all()):
                report_numerical_failure(DeferredCheckError(
                    "non-finite gradient at window publication: "
                    f"leaf shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
                ))
        self.candidates = candidates

    def publish(self):
        if self.candidates is None:
            raise RuntimeError("gradient publication requires a prepared, accepted window")
        self.before, self.candidates = self.candidates, None

    def __exit__(self, *_):
        for hook in self.hooks:
            hook.remove()
        for key, value in self.leaves.items():
            value.grad = self.before[key]
        self.pending.clear()
        self.before.clear()
        self.leaves.clear()
        self.hooks.clear()
        self.candidates = None
