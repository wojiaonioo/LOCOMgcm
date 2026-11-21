"""One topology-free preconditioned-conjugate-gradient core.

Serves the 2-D free-surface solve (cg2d, five-point) and the 3-D
non-hydrostatic solve (cg3d, seven-point) from one host loop, one dispatcher
and one operand predicate.  Nothing under this package reads a grid, tests a
grid family, or mentions a rank count: topology reaches it exclusively as
injected callables and pre-assembled tensors,
which is why the same code serves lat-lon and cubed-sphere, one rank and N.

* :mod:`.loop` -- ``pcg_iterate``: the generic host loop.
* :mod:`.dispatch` -- gate read and ledger write, once per solve.
* :mod:`.layout` -- the operand predicate shared by every gate and kernel.
* :mod:`.adjoint` -- a differentiable entry whose tape does not grow with the
  iteration count.
"""

from .adjoint import (
    PcgAdjointSpec,
    five_point_coefficient_grads,
    pcg_solve_adjoint,
    seven_point_coefficient_grads,
)
from .dispatch import FusedIterationLedger, open_iteration
from .layout import (
    autograd_reason,
    halo_shape_reason,
    layout_reason,
    same_shape_reason,
    scalar_reason,
)
from .loop import (
    PcgControl,
    PcgKernelSet,
    PcgLoopResult,
    PcgTopology,
    never,
    pcg_iterate,
)

__all__ = [
    "FusedIterationLedger",
    "PcgAdjointSpec",
    "PcgControl",
    "PcgKernelSet",
    "PcgLoopResult",
    "PcgTopology",
    "autograd_reason",
    "five_point_coefficient_grads",
    "halo_shape_reason",
    "layout_reason",
    "never",
    "open_iteration",
    "pcg_iterate",
    "pcg_solve_adjoint",
    "same_shape_reason",
    "scalar_reason",
    "seven_point_coefficient_grads",
]
