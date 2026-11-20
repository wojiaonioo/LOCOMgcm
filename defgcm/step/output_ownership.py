"""The compile binding declares whether its pool may overwrite published values."""

from contextlib import contextmanager, nullcontext
from functools import wraps
from dataclasses import fields, is_dataclass, replace

import torch


@contextmanager
def preserve_backward_buffers():
    """Keep AOT saved buffers alive for a window's retained traversals."""
    from torch._functorch import config as autograd_config

    with autograd_config.patch(donated_buffer=False):
        yield


def bind_output_ownership(configure):
    @wraps(configure)
    @contextmanager
    def binding(model, compile_config, **configure_kwargs):
        previous = getattr(model, "_step_output_pool_reused", False)
        reused = False
        if compile_config is not None and compile_config.target != "disabled":
            from torch._inductor import config, list_mode_options
            options = list_mode_options(compile_config.mode or "default")
            reused = bool(options.get("triton.cudagraphs", config.triton.cudagraphs))
        model._step_output_pool_reused = reused
        try:
            from defgcm.distributed.comm_adjoint import session_for_grid

            # Only communication windows retain the objective traversal. Include
            # internal fused compilations even when the outer target is disabled.
            retained = torch.is_grad_enabled() and session_for_grid(model.grid) is not None
            buffers = preserve_backward_buffers() if retained else nullcontext()
            with buffers, configure(model, compile_config, **configure_kwargs) as (step, metadata):
                model._step_output_pool_reused = bool(metadata["enabled"] and reused)
                metadata["output_pool_reused"] = model._step_output_pool_reused
                yield step, metadata
        finally:
            model._step_output_pool_reused = previous
    return binding


def shared_storages(*values):
    """Storages that more than one published name holds, across ``values``.

    Walks tensors in dicts, tuples and init fields of dataclasses, as
    :func:`publish_value` does; the same tensor under two names counts twice.
    """
    counts = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            key = value.untyped_storage().data_ptr()
            counts[key] = counts.get(key, 0) + 1
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, tuple):
            for item in value:
                visit(item)
        elif is_dataclass(value):
            for item in fields(value):
                if item.init:
                    visit(getattr(value, item.name))

    for value in values:
        visit(value)
    return frozenset(key for key, count in counts.items() if count > 1)


def publish_value(model, value, shared=frozenset()):
    """Transfer ordinary eager outputs; copy graph-pool and inference outputs.

    A tensor whose storage is in ``shared`` (from :func:`shared_storages`)
    is copied under every name that holds it, so the compiled step never
    meets two names for one storage and its no-alias guards hold on every
    step.  Copying every name, not only the later ones, keeps each name's
    reverse contributions summed apart before they meet at the producer,
    the same grouping publish-time clones would give.
    """
    if isinstance(value, torch.Tensor):
        if (value.untyped_storage().data_ptr() in shared
                or getattr(model, "_step_output_pool_reused", False) or value.is_inference()):
            return value.clone()
        return value
    if isinstance(value, dict):
        return {key: publish_value(model, item, shared) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(publish_value(model, item, shared) for item in value)
    if is_dataclass(value):
        return replace(value, **{item.name: publish_value(model, getattr(value, item.name), shared)
                                 for item in fields(value) if item.init})
    return value
