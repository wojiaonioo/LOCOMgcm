"""The topology-free preconditioned-conjugate-gradient host loop.

One loop serves the 2-D free-surface solve (cg2d, five-point) and the 3-D
non-hydrostatic solve (cg3d, seven-point).  It reads no grid, tests no grid
family, and never mentions a rank count: every topological fact reaches it as a
closure the CALLER layer built.

Three injections, and nothing else:

``PcgTopology.halo``
    ``x -> x`` padded on its last two axes.  Whatever produced the padding --
    a zero fill, a periodic wrap, a peer-to-peer receive from N ranks, a cube
    face rotated into this face's frame -- is invisible here.  A consumer that
    reads ``phi_halo[..., j+-1, i+-1]`` cannot tell which ran.
``PcgTopology.dot``
    the global inner product.  Lat-lon injects ``all_reduce(sum(a*b))``; the
    cube injects the face-ordered reducer that makes single-rank and N-rank
    bitwise equal; a single-rank run injects the same expression without a
    collective.  It is ``dot(a, b)`` rather than ``reduce(scalar)`` because the
    face-ordered reducer needs the elementwise product -- it sums each face in
    isolation and adds the faces in a fixed order -- and could not be expressed
    as a scalar-to-scalar map without special-casing the cube again.
``PcgKernelSet``
    the four arithmetic steps, pre-assembled coefficients bound inside.  The
    kernels see padded interior blocks and coefficient tensors and nothing else.

The two solvers differ in loop ROTATION, not in structure.  Eager
``CG3DSolver.solve`` opens each pass with the preconditioner and closes with the
operator apply; eager ``pcg_solve`` opens with the operator apply.  Lifting
cg3d's first head into its pre-loop makes them the same loop statement for
statement -- including where both breakdown tests sit -- leaving exactly one
difference: on the final permitted iteration eager cg3d stops without running
the next head, and eager cg2d runs it.  That is
:attr:`PcgControl.emit_tail_on_final_iteration`, and it is a statement about an
eager reference, not about a grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch


def never(_value: Any) -> bool:
    """A breakdown predicate that never fires (and never syncs the host).

    ``pcg_solve`` has no ``rz`` breakdown test; injecting this keeps the loop
    single-bodied without adding a device comparison the eager reference does
    not make.
    """
    return False


@dataclass(frozen=True)
class PcgTopology:
    """Everything the loop is allowed to know about where its data lives."""

    #: Width-1 (or wider) padded block for the operator apply.
    halo: Callable[[torch.Tensor], torch.Tensor]
    #: Global inner product ``<a, b>`` as a 0-d device tensor.
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PcgControl:
    """The stopping and diagnostic decisions, in the caller's own rounding.

    Every predicate here is the eager reference's own expression, handed in
    rather than re-implemented, so the loop cannot round a stopping test
    differently from the solver it is standing in for.
    """

    max_iters: int
    #: ``(r, err_sq) -> truthy``.  The eager convergence test, including its
    #: comparison operator (cg2d uses ``<``, cg3d absolute uses ``<=``) and any
    #: renormalisation it performs.  Evaluating it is the second host sync.
    converged: Callable[[torch.Tensor, torch.Tensor], Any]
    #: ``(p.Ap) -> truthy``.  The alpha-denominator breakdown guard; the first
    #: host sync.
    alpha_breakdown: Callable[[torch.Tensor], Any]
    #: ``(rz) -> truthy``.  The beta-denominator breakdown guard, evaluated
    #: where eager evaluates it: after the NEW ``rz`` exists and before the
    #: division.  ``never`` for cg2d, which has no such test.
    beta_breakdown: Callable[[torch.Tensor], Any] = never
    #: Called once per completed iteration, after ``err_sq`` and before the
    #: convergence test -- eager's position for the iteration trace, the
    #: residual history and cg3d's ``last_residual`` readback.
    on_iteration: Callable[..., None] | None = None
    #: Called with each new ``beta``.  cg2d's iteration trace carries the
    #: PREVIOUS iteration's beta, so it needs the tail value separately.
    on_beta: Callable[[torch.Tensor], None] | None = None
    #: ``False`` when the eager reference stops after the convergence test on
    #: its last permitted pass instead of computing the next search direction.
    emit_tail_on_final_iteration: bool = True


class PcgKernelSet(Protocol):
    """The four arithmetic steps a fused package supplies to the loop."""

    def spmv(self, phi_halo: torch.Tensor) -> torch.Tensor:
        """``A p`` from an already-padded block."""

    def precond(self, r: torch.Tensor) -> torch.Tensor:
        """``M^-1 r``."""

    def update(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        ap: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``(x + alpha p, r - alpha Ap, z)``; ``z`` is ``None`` unless the
        preconditioner is pointwise in the new residual and rode along."""

    def axpy(
        self, z: torch.Tensor, p: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """``z + beta p``."""


@dataclass
class PcgLoopResult:
    """The loop's terminal state.  The caller owns scaling and packaging."""

    x: torch.Tensor
    p: torch.Tensor
    r: torch.Tensor
    z: torch.Tensor | None
    rz: torch.Tensor
    err_sq: torch.Tensor
    num_iters: int
    #: Which exit was taken: ``"converged"``, ``"alpha_breakdown"``,
    #: ``"beta_breakdown"`` or ``"max_iters"``.  Diagnostic only.
    exit_reason: str = field(default="max_iters")


def pcg_iterate(
    *,
    kernels: PcgKernelSet,
    topology: PcgTopology,
    control: PcgControl,
    x: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
    z: torch.Tensor | None,
    rz: torch.Tensor,
    err_sq: torch.Tensor,
) -> PcgLoopResult:
    """Run the PCG iteration from a caller-prepared initial state.

    The caller has already formed ``r = b - A x``, its ``err_sq``, the initial
    convergence test, the first preconditioned residual ``z`` and the first
    search direction ``p``; this drives the loop from there.  Splitting the
    solve at that seam is what lets one loop serve two solvers whose pre-loops
    differ (cg2d clones ``z`` into ``p``; cg3d forms ``s = (q + beta*0)*mask``
    through its own axpy so the ``-0.0`` entries of a masked field round the way
    eager rounds them).

    Not touched, deliberately -- this is what keeps "bitwise vs eager" a
    checkable claim rather than a hopeful one:

    * both host synchronisations stay where eager put them -- per ITERATION.
      (cg3d's rotation drops eager's first ``|eta_qr_prev| == 0`` test, which is
      unreachable there because ``eta_qr_prev`` starts at 1, so its per-SOLVE
      count is ``4N - 1`` against eager's ``4N``: value-neutral, and named here
      so a sync-count claim must include it);
    * ``alpha`` and ``beta`` stay host-side 0-d tensor divisions and reach the
      kernels as device pointers, so no kernel divides and the reciprocal-multiply
      trap that has cost this repository two 1-ULP bugs never arises;
    * the three reductions stay ``topology.dot`` -- same reduction, same order,
      same collective, so ``err_sq`` and therefore ``num_iters`` cannot drift;
    * the exchange stays outside every kernel, so multi-rank keeps today's
      point-to-point contract at ANY rank count.
    """
    num_iters = 0
    exit_reason = "max_iters"
    final_index = control.max_iters - 1
    for it in range(control.max_iters):
        ap = kernels.spmv(topology.halo(p))
        p_ap = topology.dot(p, ap)
        if control.alpha_breakdown(p_ap):
            exit_reason = "alpha_breakdown"
            break
        alpha = rz / p_ap

        # ``z`` is about to be rebound; the iteration trace's station is the
        # PREVIOUS preconditioned residual (what eager passes), so the old
        # binding is kept for the hook.  Writing the new one under the same
        # name breaks the trace manifest's closure ("same-named station,
        # different sampling point").
        z_prev = z
        x, r, z = kernels.update(x, p, r, ap, alpha)

        err_sq = topology.dot(r, r)
        num_iters = it + 1

        if control.on_iteration is not None:
            control.on_iteration(
                num_iters,
                rz=rz,
                p_ap=p_ap,
                alpha=alpha,
                err_sq=err_sq,
                residual=r,
                preconditioned=z_prev,
            )

        if control.converged(r, err_sq):
            exit_reason = "converged"
            break
        if it == final_index and not control.emit_tail_on_final_iteration:
            break

        if z is None:
            z = kernels.precond(r)
        rz_new = topology.dot(r, z)
        if control.beta_breakdown(rz):
            exit_reason = "beta_breakdown"
            break
        beta = rz_new / rz
        rz = rz_new
        if control.on_beta is not None:
            control.on_beta(beta)

        p = kernels.axpy(z, p, beta)

    return PcgLoopResult(
        x=x,
        p=p,
        r=r,
        z=z,
        rz=rz,
        err_sq=err_sq,
        num_iters=num_iters,
        exit_reason=exit_reason,
    )


__all__ = [
    "PcgControl",
    "PcgKernelSet",
    "PcgLoopResult",
    "PcgTopology",
    "never",
    "pcg_iterate",
]
