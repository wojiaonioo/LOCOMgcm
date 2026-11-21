"""Creation context for tensors that outlive one numerical evaluation."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.autograd import forward_ad


def forward_mode_active() -> bool:
    """True while forward-mode AD is live: a dual level is open and forward grads are on.

    A native kernel, a custom operator without a forward rule, a CUDA graph
    replay, a communication operator or an Inductor-compiled region returns its
    output without the tangent and without an error, and ``no_grad`` does not
    stop forward mode, so a choice that reads only the grad mode misses it.
    This is the program's state, not a tensor's: dual tensors exist only inside
    an open level, so hidden tangents (closures, cached operands) are covered,
    and every rank that opened the level gives the same answer.  Inside a custom
    ``autograd.Function``'s forward PyTorch turns forward grads off, so the
    primal solves of a ROOT derivative read False here and keep their kernels.
    Dynamo guards ``forward_ad._current_level``, so a compiled caller retraces
    when a level opens; the C flag is read only while a level is open.
    """
    return forward_ad._current_level >= 0 and torch._C._is_fwd_grad_enabled()


def grad_mode_requires_reference() -> bool:
    """Uniform admission policy for kernels without a usable native backward.

    Local inputs can lose their control dependency on only some spatial ranks.
    Use the shared program's grad mode to choose the reference, never a local
    requires_grad flag. Native backward bodies still run under no_grad. KPP
    BLDEPTH/BLMIX instead check their replicated geometry/table control scope.
    A live forward mode (:func:`forward_mode_active`) requires the reference
    too: the kernels carry no forward rule and would drop the tangent.
    """
    return torch.is_grad_enabled() or forward_mode_active()


def extent_mismatch(actual, expected) -> bool:
    """True when extent ``actual`` neither broadcasts (is 1) nor equals ``expected``.

    Shared by the shape rules that ask whether one tensor's extent can serve
    another's. Written as two comparisons, not ``actual not in (1, expected)``:
    under ``torch.compile(dynamic=True)``, PyTorch 2.9's Dynamo answers a
    membership test whose left operand is a Python int by comparing it with the
    Python-int items only (``torch/_dynamo/utils.py`` ``iter_contains``). Which
    extents stay Python ints in a trace depends on where they came from: literal
    sizes and sizes specialized by a Python loop do, while sizes of input tensors
    and ints read from argument objects may remain symbolic. A Python-int extent then
    never matches a symbolic one of the same value, and the rule declines a shape
    it accepts in eager execution. Explicit comparisons give the eager answer for
    every mix.
    """
    return actual != 1 and actual != expected

#: Whether the code that entered the innermost ``ordinary_tensor_mode`` was in
#: inference mode; None outside every such scope.  Set and read eagerly only.
_CALLER_INFERENCE: ContextVar[bool | None] = ContextVar("_CALLER_INFERENCE", default=None)


@contextmanager
def ordinary_tensor_mode():
    """Create ordinary tensors while preserving the caller's gradient policy.

    Persistent constants and published history can be read by a later gradient
    window. Inference tensors cannot serve that lifetime. Disabling inference
    alone also enables gradients, so restore the original grad policy inside
    the context. Differentiable geometry/control construction keeps its graph.
    The caller's inference mode is recorded for :func:`caller_tensor_mode`.
    """
    enabled = torch.is_grad_enabled()
    token = _CALLER_INFERENCE.set(torch.is_inference_mode_enabled())
    try:
        with torch.inference_mode(False), torch.set_grad_enabled(enabled):
            yield
    finally:
        _CALLER_INFERENCE.reset(token)


@contextmanager
def caller_tensor_mode():
    """Create tensors in the inference mode of the innermost ``ordinary_tensor_mode``'s caller.

    Only for working values that do not outlive the step: a placeholder the
    step kernel overwrites in place before any read.  ``build_step_plan``
    prepares in ordinary mode and the kernel runs in its caller's mode, so a
    placeholder made here is the kind of tensor the kernel then writes and
    the compiled step's guards on it hold from the first step.
    """
    if _CALLER_INFERENCE.get():
        with torch.inference_mode():
            yield
    else:
        yield


def ordinary_tensor(value):
    """Transfer an inference result into ordinary storage at a window boundary."""
    return value.clone() if value.is_inference() else value
