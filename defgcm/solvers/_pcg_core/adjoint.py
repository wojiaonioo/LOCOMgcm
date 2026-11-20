"""A differentiable PCG entry that does not tape the iterations.

``x = A(theta)^-1 b`` is an implicit function of the coefficients and the right
hand side, so its derivative is available in closed form and does NOT require
the iteration history:

    d_bar   = A^-T x_bar = A^-1 x_bar        <- ONE more solve, the SAME core,
                                                the SAME injected halo/reducer
                                                (A is SPD, so no transpose)
    b_bar   = d_bar
    theta_bar[i][j] = -d_bar[i] * x[j]       <- pointwise, per coupled pair

Tape footprint: ``x`` plus the coefficient tensors -- independent of
``num_iters``.  Taping the loop instead records roughly ``5 * num_iters``
intermediates per solve (~56 iterations in the production cg2d posture, ~100 in
cg3d), which is where the differentiable trajectory's memory goes today.

**The plain path is untouched.**  Nothing here runs unless grad is enabled AND
an input requires grad; :func:`pcg_solve_adjoint` is an added entry, not a
replacement, so the no-grad forward keeps exactly the bytes it had.  Forward
runs under ``torch.no_grad()``, which is why it is free.

Second-order derivatives are out of scope: ``backward`` is itself written under
``no_grad``, so ``gradgradcheck`` will not pass and is not claimed.

**Forward mode** is the same implicit function differentiated the other way:

    x_dot = A^-1 (b_dot - A_dot x)            <- ONE more solve, the SAME
                                                 ``spec.solve`` as the VJP

:func:`pcg_solve_adjoint` splits the dual tensors at the entry, as the
tridiagonal solves do (``tridiag.py`` ``_tridiag_solve_tangent``), instead of
declaring ``jvp`` on the Function: PyTorch 2.9's Dynamo refuses a traced
``autograd.Function`` whose class declares one (``Unsupported custom jvp``).
``A_dot x`` comes from :attr:`PcgAdjointSpec.coefficient_tangent`, one
application of the caller's operator to the fixed ``x`` with the dual
coefficients.  Mixed forward-over-reverse use is not qualified.

Topology: this file names no grid family and no rank count.  The re-solve is
performed by the caller-supplied :attr:`PcgAdjointSpec.solve`, which closes over
the same ``PcgTopology`` the forward used, so the adjoint's halo exchange and
its reducer are whatever the caller injected -- at any rank count.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from defgcm.accel.tensor_context import forward_mode_active

from ..tridiag import forward_mode_requested


@dataclass(frozen=True)
class PcgAdjointSpec:
    """How to re-solve, and how to turn ``(d_bar, x)`` into coefficient grads.

    ``solve``
        ``rhs -> A^-1 rhs``.  Must close over the SAME injections (halo,
        reducer, coefficients, preconditioner) as the forward solve and must
        not itself build a tape; the machinery below already wraps it in
        ``no_grad``.
    ``coefficient_grads``
        ``(d_bar, x, needs) -> tuple`` -- one entry per coefficient tensor
        passed to :func:`pcg_solve_adjoint`, in the same order, ``None`` where
        ``needs`` says the gradient is not wanted.  Stencil-specific, hence
        injected: the five-point and seven-point forms are
        :func:`five_point_coefficient_grads` and
        :func:`seven_point_coefficient_grads`, both of which read the padded
        blocks and therefore carry no topology either.
    ``coefficient_tangent``
        ``(x, coefficients) -> A_dot x`` -- the forward-mode counterpart of
        ``coefficient_grads``: the operator applied once to the plain ``x``
        with the caller's DUAL coefficients (same order), returning the
        tangent of that application, i.e. ``sum_j dA/dtheta_j theta_dot_j x``.
        It must apply the operator the solve inverts (its masks, seams and
        halo), and it runs only when some coefficient carries a tangent.
    """

    solve: Callable[[torch.Tensor], torch.Tensor]
    coefficient_grads: Callable[
        [torch.Tensor, torch.Tensor, Sequence[bool]],
        Sequence[torch.Tensor | None],
    ]
    coefficient_tangent: Callable[
        [torch.Tensor, Sequence[torch.Tensor]], torch.Tensor
    ]
    # A warm-started primal may use a different numerical initial guess from
    # the transpose solve, without making that guess a physical control.
    primal_solve: Callable[[torch.Tensor], torch.Tensor] | None = None


class _PcgSolveAdjoint(torch.autograd.Function):
    """``x = A^-1 b`` with an analytic adjoint instead of a taped loop."""

    @staticmethod
    def forward(ctx, spec: PcgAdjointSpec, rhs: torch.Tensor, *coefficients):
        with torch.no_grad():
            x = (spec.primal_solve or spec.solve)(rhs)
        ctx.spec = spec
        ctx.n_coefficients = len(coefficients)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        if torch.is_grad_enabled():
            # Grad mode on inside backward means ``create_graph=True``.  Both
            # solves below run under ``no_grad``, so every gradient returned
            # here is graph-free: a second differentiation would silently come
            # back empty rather than wrong-but-visible.  Say so instead.
            raise RuntimeError(
                "pcg_solve_adjoint provides a first-order VJP only; "
                "create_graph=True (second-order) is not qualified for the "
                "analytic PCG adjoint, whose re-solve carries no tape"
            )
        (x,) = ctx.saved_tensors
        spec: PcgAdjointSpec = ctx.spec
        with torch.no_grad():
            d_bar = spec.solve(grad_x.contiguous())
        needs_rhs = ctx.needs_input_grad[1]
        needs_coefficients = tuple(ctx.needs_input_grad[2:])
        if any(needs_coefficients):
            with torch.no_grad():
                grads = tuple(
                    spec.coefficient_grads(d_bar, x, needs_coefficients)
                )
        else:
            grads = (None,) * ctx.n_coefficients
        if len(grads) != ctx.n_coefficients:
            raise RuntimeError(
                "coefficient_grads returned "
                f"{len(grads)} gradients for {ctx.n_coefficients} coefficients"
            )
        return (None, d_bar if needs_rhs else None, *grads)


def pcg_solve_adjoint(
    spec: PcgAdjointSpec,
    rhs: torch.Tensor,
    *coefficients: torch.Tensor,
    session=None,
) -> torch.Tensor:
    """``A^-1 rhs``, differentiable in ``rhs`` and ``coefficients``, untaped.

    ``session`` is the caller's bound ``CommSession`` or None.  A bound session
    takes the solve and its transpose as one ordered round
    (``CommSession.linear_solve``).  A forward-mode tangent on ``rhs`` or a
    coefficient takes :func:`_pcg_solve_tangent`.  A live forward mode inside a
    session is refused before anything runs, because the round and the
    communication operators carry first-order reverse derivatives only; the
    reading (``forward_mode_active``) is program state, so every rank raises
    when all of them open the same dual level.
    """
    if session is not None and forward_mode_active():
        raise NotImplementedError(
            "pcg_solve_adjoint: forward-mode derivatives of a solve inside a "
            "CommSession are not supported; the session's solve round and its "
            "communication operators provide first-order reverse derivatives only"
            "; take forward-mode derivatives on a single rank"
        )
    if forward_mode_requested(rhs, *coefficients):
        return _pcg_solve_tangent(spec, rhs, coefficients)
    if session is not None:
        return session.linear_solve(spec, rhs, *coefficients)
    return _PcgSolveAdjoint.apply(spec, rhs, *coefficients)


def _pcg_solve_tangent(
    spec: PcgAdjointSpec,
    rhs: torch.Tensor,
    coefficients: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Implicit JVP: ``x = A^-1 b`` and ``x_dot = A^-1 (b_dot - A_dot x)``.

    Both solves run through :class:`_PcgSolveAdjoint` on the primal tensors, so
    the solution is the one the plain call returns, and the tangent solve is the
    VJP's own ``spec.solve`` (zero first guess, relative tolerance, failure
    channel) with ``A`` symmetric.  Inside a Function's forward PyTorch turns
    forward gradients off, so operator closures that captured the dual
    coefficients iterate on their primal values.  The first guess and the
    preconditioner are passive here as in the VJP: a tangent on either does not
    reach ``x_dot``.  Unpacking keeps any reverse dependence of the primals.
    """
    unpack = torch.autograd.forward_ad.unpack_dual
    rhs_primal, rhs_tangent = unpack(rhs)
    duals = tuple(unpack(value) for value in coefficients)
    primals = tuple(value.primal for value in duals)
    x = _PcgSolveAdjoint.apply(spec, rhs_primal, *primals)
    tangent_rhs = torch.zeros_like(x) if rhs_tangent is None else rhs_tangent
    if any(value.tangent is not None for value in duals):
        applied = spec.coefficient_tangent(x, coefficients)
        if applied is None:
            raise RuntimeError(
                "coefficient_tangent returned no tangent although a coefficient "
                "carries one; the operator application dropped it"
            )
        tangent_rhs = tangent_rhs - applied
    x_dot = _PcgSolveAdjoint.apply(
        replace(spec, primal_solve=None), tangent_rhs.contiguous(), *primals
    )
    return torch.autograd.forward_ad.make_dual(x, x_dot)


# ---------------------------------------------------------------------------
# Stencil gradient forms.  Both read PADDED blocks, so the COLUMN a boundary
# coefficient couples to is whatever the injected exchange put in the ghost --
# zero for a closed wall, the wrapped cell for a periodic domain, the peer's
# cell for one rank of N.  No branch decides which.
#
# The ROW is a different matter, and getting it wrong was a real defect
# .  An edge coefficient sits in at most two
# rows of the operator: the cell on either side of the face.  At a boundary slot
# one of those cells is not in this array -- it is the cell behind the ghost,
# which on a periodic domain is owned by the OTHER slot of the same physical
# face and under a decomposition is owned by the peer rank.  These functions
# return the gradient of the array they are handed, entry by entry, so each slot
# receives ONLY the partial of the row it actually occupies here.  A caller that
# ties two slots to one parameter (the periodic seam, or a shared coefficient
# across ranks) gets the sum for free -- autograd adds the two slot gradients
# when they flow from one source, and a decomposition adds them with the
# all-reduce it already performs on parameter gradients.  Forming the sum HERE
# instead handed every boundary slot twice the tied derivative and, on a
# periodic seam, the wrong sign.
# ---------------------------------------------------------------------------


def _sum_leading(value: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reduce a per-cell product to the coefficient's own rank.

    Coefficients broadcast over the leading (batch or face) axes of the state,
    so their gradient is the sum of the per-leading-index contributions.
    """
    extra = value.ndim - ndim
    if extra <= 0:
        return value
    return value.sum(dim=tuple(range(extra)))


def _west_edge_grad(d_in: torch.Tensor, x_halo: torch.Tensor, ny: int, nx: int):
    """Per-slot gradient of a ``[..., ny, nx+1]`` west-edge coefficient array.

    ``aW[j,i]`` multiplies ``p[j,i-1]`` in row ``(j,i)`` and ``p[j,i]`` in row
    ``(j,i-1)``.  Row ``(j,i)`` exists for ``i <= nx-1`` and row ``(j,i-1)`` for
    ``i >= 1``, so the two contributions occupy overlapping but different slot
    ranges and are placed by padding rather than summed on top of each other.
    """
    west = -(d_in * x_halo[..., 1 : ny + 1, 0:nx])       # slots 0 .. nx-1
    east = -(d_in * x_halo[..., 1 : ny + 1, 2 : nx + 2])  # slots 1 .. nx
    return F.pad(west, (0, 1)) + F.pad(east, (1, 0))


def _south_edge_grad(d_in: torch.Tensor, x_halo: torch.Tensor, ny: int, nx: int):
    """Per-slot gradient of a ``[..., ny+1, nx]`` south-edge coefficient array."""
    south = -(d_in * x_halo[..., 0:ny, 1 : nx + 1])       # slots 0 .. ny-1
    north = -(d_in * x_halo[..., 2 : ny + 2, 1 : nx + 1])  # slots 1 .. ny
    return F.pad(south, (0, 0, 0, 1)) + F.pad(north, (0, 0, 1, 0))


def five_point_coefficient_grads(
    d_bar_halo: torch.Tensor,
    x_halo: torch.Tensor,
    needs: Sequence[bool],
    *,
    aW_ndim: int = 2,
    aS_ndim: int = 2,
    aC_ndim: int = 2,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Gradients of ``(aW_edge, aS_edge, aC)`` for the five-point apply.

    The apply is ``out[j,i] = aC[j,i]*p[j,i] + aW[j,i]*p[j,i-1]
    + aW[j,i+1]*p[j,i+1] + aS[j,i]*p[j-1,i] + aS[j+1,i]*p[j+1,i]``
    (``operators/c_grid.py:288-290``), so each west/south EDGE coefficient
    appears in at most two rows -- the cell on either side of the face -- and
    the centre coefficient in one.  At a boundary slot only one of those cells
    is in this array; see the module comment above for why the other one's
    partial belongs to its owner and is deliberately not added here.
    """
    ny = int(d_bar_halo.shape[-2]) - 2
    nx = int(d_bar_halo.shape[-1]) - 2
    d_in = d_bar_halo[..., 1 : ny + 1, 1 : nx + 1]
    x_in = x_halo[..., 1 : ny + 1, 1 : nx + 1]

    grad_aW = grad_aS = grad_aC = None
    if needs[0]:
        grad_aW = _sum_leading(_west_edge_grad(d_in, x_halo, ny, nx), aW_ndim)
    if needs[1]:
        grad_aS = _sum_leading(_south_edge_grad(d_in, x_halo, ny, nx), aS_ndim)
    if needs[2]:
        grad_aC = _sum_leading(-(d_in * x_in), aC_ndim)
    return grad_aW, grad_aS, grad_aC


def seven_point_coefficient_grads(
    d_bar_halo: torch.Tensor,
    x_halo: torch.Tensor,
    needs: Sequence[bool],
    *,
    mask: torch.Tensor | None = None,
    coefficient_ndim: int = 3,
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    """Gradients of ``(aW3d, aS3d, aV3d, aC3d)`` for the seven-point apply.

    Horizontal terms as in :func:`five_point_coefficient_grads`. The vertical
    coefficient ``aV[k]`` couples level ``k`` to ``k-1`` in both directions
    (``cg3d_operator.py:357-359``), and ``aV[0]`` couples nothing, so its
    gradient is exactly zero.

    ``mask``
        cg3d applies its mask at the operator's STORE, so the operator that is
        actually inverted is ``M A`` and the coefficient gradient is
        ``-(M d_bar)[i] * x[j]``.  Pass the same ``mask4`` the kernels use and
        that factor is applied here; omit it only when the caller's ``d_bar``
        is already masked -- which it is when ``d_bar`` came out of this core's
        cg3d kernels, since every one of them masks its own store.  Stating it
        as an argument rather than a docstring assumption is the point: the
        function otherwise depends on a property of an injection it never sees
        .
    """
    ny = int(d_bar_halo.shape[-2]) - 2
    nx = int(d_bar_halo.shape[-1]) - 2
    d_in = d_bar_halo[..., 1 : ny + 1, 1 : nx + 1]
    x_in = x_halo[..., 1 : ny + 1, 1 : nx + 1]
    if mask is not None:
        # Every term is linear in d_bar, and the helpers below read only the
        # INTERIOR of d_bar, so masking it once here masks all four gradients.
        d_in = d_in * mask

    grad_aW = grad_aS = grad_aV = grad_aC = None
    if needs[0]:
        grad_aW = _sum_leading(
            _west_edge_grad(d_in, x_halo, ny, nx), coefficient_ndim
        )
    if needs[1]:
        grad_aS = _sum_leading(
            _south_edge_grad(d_in, x_halo, ny, nx), coefficient_ndim
        )
    if needs[2]:
        below = d_in[..., 1:, :, :] * x_in[..., :-1, :, :]
        above = d_in[..., :-1, :, :] * x_in[..., 1:, :, :]
        pair = -(below + above)
        grad_aV = torch.cat([torch.zeros_like(pair[..., :1, :, :]), pair], dim=-3)
        grad_aV = _sum_leading(grad_aV, coefficient_ndim)
    if needs[3]:
        grad_aC = _sum_leading(-(d_in * x_in), coefficient_ndim)
    return grad_aW, grad_aS, grad_aV, grad_aC


__all__ = [
    "PcgAdjointSpec",
    "five_point_coefficient_grads",
    "pcg_solve_adjoint",
    "seven_point_coefficient_grads",
]
