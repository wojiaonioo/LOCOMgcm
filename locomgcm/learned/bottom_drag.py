"""Learned bottom-drag coefficient for the hybrid model.

``BottomDrag`` (locomgcm/dynamics/bottom_drag.py) applies MITgcm's
``mom_u_botdrag_coeff.F`` drag at the deepest wet face of every column,
``-cDrag * u_bot / (hFac * drF)``, with
``cDrag = bottomDragLinear + no-slip term + bottomDragQuadratic * |u|``.
``|u| = sqrt(KE(i-1) + KE(i))`` is the KE speed of the bottom face.

A coefficient law replaces ``bottomDragLinear + bottomDragQuadratic * |u|``
by ``constant + law(|u|, h)``, ``h = hFac * drF`` of the bottom cell; the
no-slip term, the face selection, the KE speed and the tendency are the
module's own.  ``LearnedDragInjection`` sets a law on the model's
``BottomDrag`` instances (``BottomDrag.coefficient_law``) for the duration of
a ``with`` block.

* ``TruthDragLaw(linear, quadratic)``: the configured law itself; with it the
  module reproduces ``BottomDrag`` bit for bit (same operands, same order).
* ``MLPDragLaw``: ``c = c_scale * softplus(MLP(|u| / u_scale, h / h_scale))``,
  non-negative, constant part 0, float64, a few hundred parameters.

``MLPDragLaw.theta`` is a registered parameter, so ``parameters()``,
``state_dict()`` and ``.to()`` see it.  A training loop binds it per
checkpoint segment with
``checkpointed_rollout(parameter_bindings=((law, "theta", law.theta),))``:
each segment reads its own copy and the parameter is back on the law when the
segment ends.  ``constant`` stays a plain CPU 0-d tensor like
``BottomDrag.linear`` (``.to()`` does not move it); ``theta`` must sit on the
model's device.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from locomgcm.dynamics.bottom_drag import BottomDrag


class TruthDragLaw(nn.Module):
    """``linear + quadratic * |u|``: the configured MITgcm law as a coefficient law."""

    def __init__(self, linear: float | torch.Tensor, quadratic: float | torch.Tensor):
        super().__init__()
        self.constant = torch.as_tensor(linear, dtype=torch.float64)
        self.quadratic = quadratic

    def forward(self, speed: torch.Tensor, thickness: torch.Tensor) -> torch.Tensor:
        del thickness
        return self.quadratic * speed


class MLPDragLaw(nn.Module):
    """Non-negative drag coefficient ``c(|u|, h)`` from a small SiLU network.

    ``c = c_scale * softplus(net([|u| / speed_scale, h / thickness_scale]))`` in
    m/s, added to the no-slip term; the constant part is 0, so ``c`` stands for
    the whole linear plus quadratic coefficient.  ``theta`` holds every weight
    and bias of ``depth`` hidden ``Linear + SiLU`` layers and the output layer,
    in that order; ``hidden=16, depth=2`` gives 337 values.  They start from
    ``N(0, init_std)`` drawn from ``seed``: small, not zero, so the initial law
    is close to the constant ``c_scale * log 2`` without being degenerate.
    Speed and thickness broadcast against each other.
    """

    def __init__(
        self,
        *,
        hidden: int = 16,
        depth: int = 2,
        speed_scale: float = 0.1,
        thickness_scale: float = 50.0,
        c_scale: float = 1.0e-3,
        init_std: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"MLPDragLaw needs depth >= 1, got {depth}")
        shapes: list[tuple[int, ...]] = []
        width = 2
        for _ in range(depth):
            shapes += [(hidden, width), (hidden,)]
            width = hidden
        shapes += [(1, width), (1,)]
        self.shapes = tuple(shapes)
        size = sum(math.prod(shape) for shape in shapes)
        gen = torch.Generator().manual_seed(int(seed))
        self.theta = nn.Parameter(torch.randn(size, generator=gen, dtype=torch.float64) * init_std)
        self.constant = torch.zeros((), dtype=torch.float64)
        self.speed_scale = float(speed_scale)
        self.thickness_scale = float(thickness_scale)
        self.c_scale = float(c_scale)

    def layers(self, theta: torch.Tensor) -> list[torch.Tensor]:
        """Weights and biases as views of ``theta``."""
        out, start = [], 0
        for shape in self.shapes:
            size = math.prod(shape)
            out.append(theta[start:start + size].view(shape))
            start += size
        return out

    def forward(self, speed: torch.Tensor, thickness: torch.Tensor) -> torch.Tensor:
        speed, thickness = torch.broadcast_tensors(speed, thickness)
        h = torch.stack((speed / self.speed_scale, thickness / self.thickness_scale), dim=-1)
        tensors = self.layers(self.theta)
        for k in range(0, len(tensors) - 2, 2):
            h = F.silu(F.linear(h, tensors[k], tensors[k + 1]))
        raw = F.linear(h, tensors[-2], tensors[-1]).squeeze(-1)
        return self.c_scale * F.softplus(raw)


def _bottom_drag_modules(model) -> tuple[BottomDrag, ...]:
    """The ``BottomDrag`` objects the model's step evaluates, each once.

    A lat-lon step evaluates ``model.dynamics_modules``; the cube step evaluates
    the template in ``stepper.cube_static_state.bottom_drag``, which may be
    another object.
    """
    found = [module for module in model.dynamics_modules if isinstance(module, BottomDrag)]
    static = getattr(model.stepper, "cube_static_state", None)
    cube = None if static is None else static.bottom_drag
    if isinstance(cube, BottomDrag) and all(cube is not module for module in found):
        found.append(cube)
    return tuple(found)


class LearnedDragInjection:
    """Route every ``BottomDrag`` of ``model`` through ``law`` inside a ``with`` block.

    The law is set on the module objects the step evaluates, the lat-lon
    modules and the cube template (``BottomDrag.set_coefficient_law``), and
    cleared on exit.  Being instance state, it is part of what a compiled step
    guards on: a frame traced outside the block is not reused inside it, and
    each block evaluates its own law, with the law's tensors as graph inputs.
    Every module needs ``quadratic > 0``; injections into one model do not nest.
    """

    def __init__(self, law: nn.Module, model) -> None:
        self.law = law
        self.model = model
        self._modules: tuple[BottomDrag, ...] = ()

    def __enter__(self) -> "LearnedDragInjection":
        modules = _bottom_drag_modules(self.model)
        if not modules:
            raise ValueError("the model evaluates no BottomDrag module to route through a law")
        if any(module.coefficient_law is not None for module in modules):
            raise RuntimeError("a coefficient law is already set on this model's bottom drag")
        self._modules = modules
        try:
            for module in modules:
                module.set_coefficient_law(self.law)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for module in self._modules:
            module.set_coefficient_law(None)
        self._modules = ()
