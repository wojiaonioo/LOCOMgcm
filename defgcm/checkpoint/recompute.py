"""Checkpoint rollout compatibility entry and replay context helpers."""
import contextlib
import dataclasses
import torch


def checkpointed_rollout(model, initial, *, steps, dt, observe, segment_steps,
                         parameter_bindings=(), store=None, restart_codec_heads=False):
    """Run one owned window and request its differentiable final checkpoint.

    New callers consume their state through WindowOwner.capture and declare
    final-state demand. Raw backward enters the same owned reverse and verdict.
    """
    from .window import WindowOwner, require_differentiable_window
    require_differentiable_window(model)
    owner = WindowOwner.for_model(model)
    try:
        owner.configure(model, initial, need_final=True, final_device=("cpu" if restart_codec_heads
                         else model.config.device), store=store, codec=restart_codec_heads)
        samples = owner.run(steps=steps, dt=dt, observe=observe, segment_steps=segment_steps,
                            parameter_bindings=parameter_bindings)
        return owner.final, samples
    except Exception as error:
        owner.terminate(error)
        raise


class _ReplayState:
    """The RNG and autocast state of a forward pass, re-entered for its recomputation.

    As ``torch.utils.checkpoint`` keeps them: the CPU generator, the generators
    of the CUDA devices among the inputs, and autocast for the inputs' device
    type and the CPU.
    """

    def __init__(self, values):
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        self.device_type = next((tensor.device.type for tensor in tensors
                                 if tensor.device.type != "cpu"), "cuda")
        self.cpu_rng = torch.get_rng_state()
        self.devices = sorted({tensor.get_device() for tensor in tensors if tensor.is_cuda})
        self.device_rng = [torch.cuda.get_rng_state(device) for device in self.devices]
        self.autocast = [(kind, dict(enabled=torch.is_autocast_enabled(kind),
                                     dtype=torch.get_autocast_dtype(kind),
                                     cache_enabled=torch.is_autocast_cache_enabled()))
                         for kind in (self.device_type, "cpu")
                         if torch.amp.is_autocast_available(kind)]

    @contextlib.contextmanager
    def restored(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.random.fork_rng(devices=self.devices,
                                                      device_type=self.device_type))
            torch.set_rng_state(self.cpu_rng)
            for device, state in zip(self.devices, self.device_rng):
                torch.cuda.set_rng_state(state, device)
            for kind, settings in self.autocast:
                stack.enter_context(torch.amp.autocast(kind, **settings))
            yield


def _bind(owner, name, value, bound: contextlib.ExitStack):
    """Set ``owner.name = value`` in place for one segment.

    A registered ``nn.Module`` parameter or buffer is set in its slot, the way
    ``torch.func.functional_call`` sets it, and the registered tensor returns
    when the segment ends (``bound`` closes): the module's optimizer,
    ``state_dict`` and ``.to()`` hold that tensor.  ``object.__setattr__``
    would put the value in the instance ``__dict__`` instead, which eager
    attribute reads find and a compiled frame traced on the slot does not
    (it keeps reading the registered tensor).

    Other owners keep the bound value after the rollout, set as
    ``dataclasses.replace`` would set it.  A frozen dataclass derives its
    ``init=False`` fields in ``__post_init__``; ``BottomDrag.active``
    summarizes the drag coefficients once, on the host, so the step reads a
    Python bool instead of the possibly-tensor ``linear``.
    ``object.__setattr__`` alone keeps such a field at its construction value.
    The owner cannot be replaced because the model holds it, so the fields are
    re-derived from a ``replace`` copy here, at the segment head, outside the
    step.
    """
    if isinstance(owner, torch.nn.Module):
        for slots in (owner._parameters, owner._buffers):
            if name in slots:
                bound.callback(slots.__setitem__, name, slots[name])
                slots[name] = value
                return
    object.__setattr__(owner, name, value)
    if dataclasses.is_dataclass(owner) and type(owner).__dataclass_params__.frozen:
        derived = dataclasses.replace(owner)
        for field in dataclasses.fields(owner):
            if not field.init:
                object.__setattr__(owner, field.name, getattr(derived, field.name))
