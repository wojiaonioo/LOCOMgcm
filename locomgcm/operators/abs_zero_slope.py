"""Absolute value with an explicit reverse-mode slope at zero.

``torch.abs`` back-propagates ``sgn(x)``, which is 0 at ``x == 0``.  The
MITgcm-AD reference differentiates ``ABS(x)`` through the branch Tapenade
records for it (``x .GE. 0``), so its adjoint carries slope ``+1`` at zero.
Cold-start transports are exactly zero on the first step and those zeros
receive non-zero adjoints from the DST3 upwind split, so the two conventions
give different velocity gradients.  The forward value is ``x.abs()`` itself;
only the derivative differs.

Forward mode uses the same slope, so a JVP and the VJP are transposes of one
linear map.  The tangent is formed at the entry rather than by a ``jvp`` on
the Function: PyTorch 2.9's Dynamo refuses a traced ``autograd.Function``
whose class declares one (as for the tridiagonal solves, ``tridiag.py``).
That entry split is eager only.  A traced region does not see the tangent
(Dynamo traces the branch once and reuses the graph for dual inputs), so the
split reads the program's forward-mode state (``forward_mode_active``, which
Dynamo guards) and a compiled caller never runs it: with a dual level open
the trace breaks there and the call runs eagerly, or, under
``fullgraph=True``, compilation fails.
"""
from __future__ import annotations

import torch

from locomgcm.accel.tensor_context import forward_mode_active


def _slope(value: torch.Tensor, zero_slope: float) -> torch.Tensor:
    return torch.where(value == 0, torch.full_like(value, zero_slope), torch.sign(value))


class _AbsWithZeroSlope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, zero_slope: float) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.zero_slope = zero_slope
        return value.abs()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (value,) = ctx.saved_tensors
        return grad_output * _slope(value, ctx.zero_slope), None


def abs_with_zero_slope(value: torch.Tensor, zero_slope: float) -> torch.Tensor:
    """``|value|`` whose derivative at ``value == 0`` is ``zero_slope`` instead of 0.

    ``zero_slope`` is the reference model's ``ABS(0)`` slope expressed in the
    sign convention of ``value``: ``+1`` when ``value`` and the reference
    argument point the same way, ``-1`` when one is the negative of the other.
    """
    if forward_mode_active():
        if torch.compiler.is_compiling():
            raise NotImplementedError(
                "abs_with_zero_slope: forward-mode derivatives inside a compiled region "
                "are not supported (the trace does not see the tangent); call it eagerly"
            )
        primal, tangent = torch.autograd.forward_ad.unpack_dual(value)
        if tangent is not None:
            return torch.autograd.forward_ad.make_dual(
                _AbsWithZeroSlope.apply(primal, zero_slope), _slope(primal, zero_slope) * tangent
            )
    return _AbsWithZeroSlope.apply(value, zero_slope)
