"""Checked eager, first-order ROOT VJP in real Euclidean coordinates.

The caller supplies a linear, uniquely solvable PCG-compatible operator and
lists every operator tensor in ``coefficients``. Callbacks must be pure and
bind those exact tensors. Omitting the transpose declares symmetry. A tensor
of any shape is one vector (all entries participate in the Euclidean dot).
Initial guesses and the left inverse preconditioner are algorithmic controls;
the adjoint uses no preconditioner and solves the original transpose equation.
FIXED, singular/gauge systems, compilation and higher derivatives are not
qualified by this entry. No physical sign or coordinate conversion is made.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum
import math

import torch

from locomgcm.accel.tensor_context import forward_mode_active

from ._pcg_core.loop import PcgControl, PcgTopology, pcg_iterate


class SolveReason(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    PARAMETER_VERSION_CHANGED = "PARAMETER_VERSION_CHANGED"
    PRIMAL_NOT_CONVERGED = "PRIMAL_NOT_CONVERGED"
    ADJOINT_NOT_CONVERGED = "ADJOINT_NOT_CONVERGED"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    BREAKDOWN = "BREAKDOWN"
    NONFINITE = "NONFINITE"
    INCOMPATIBLE_RHS = "INCOMPATIBLE_RHS"


@dataclass(frozen=True)
class SolveStatus:
    """Immutable solve snapshot; backward publishes a new snapshot to evaluation.

    The directly returned snapshot describes the forward only. A failed
    backward exposes its completed snapshot on ``SolveRejected.status``;
    ``evaluate_loss_and_grad`` returns the completed success snapshot.
    """

    mode: str
    operator_identity: str
    coordinate_system: str
    control_set: tuple
    primal_status: str = "NOT_EVALUATED"
    derivative_status: str = "NOT_EVALUATED"
    status: str = "NOT_EVALUATED"
    reason: SolveReason | None = None
    iterations: int = 0
    stop_reason: str = "NOT_STARTED"
    true_residual_abs: float | None = None
    true_residual_rel: float | None = None
    normalizer: float | None = None
    tolerances: tuple = ()
    recursive_residual_abs: float | None = None
    tolerance_mode: str = "ABSOLUTE_OR_RELATIVE"
    adjoint: SolveStatus | None = None


class SolveRejected(RuntimeError):
    def __init__(self, reason, message="", *, status=None):
        self.reason = SolveReason(reason)
        self.status = status
        super().__init__(f"{self.reason.value}: {message}")


_evaluations = ContextVar("checked_root_evaluations", default=None)


def _finite(value):
    if not bool(torch.isfinite(value).all()):
        raise SolveRejected(SolveReason.NONFINITE)
    return value


def _apply(operator, vector):
    result = operator(vector)
    if (not isinstance(result, torch.Tensor) or result.shape != vector.shape
            or result.dtype != vector.dtype or result.device != vector.device):
        raise SolveRejected(SolveReason.INVALID_INPUT, "operator output must match its input")
    return _finite(result)


class _Kernels:
    """Plain tensor kernel adapter; iteration and stopping live in pcg_iterate."""

    def __init__(self, operator, preconditioner):
        self.operator = operator
        self.preconditioner = preconditioner

    def spmv(self, vector):
        return _apply(self.operator, vector)

    def precond(self, residual):
        if self.preconditioner is None:
            return residual.clone()
        return _apply(self.preconditioner, residual)

    def update(self, x, p, r, ap, alpha):
        return _finite(x + alpha * p), _finite(r - alpha * ap), None

    def axpy(self, z, p, beta):
        return _finite(z + beta * p)


def _solve(operator, rhs, x0, preconditioner, max_iters, atol, rtol, status):
    kernels = _Kernels(operator, preconditioner)
    dot = lambda a, b: _finite((a * b).sum())
    normalizer = _finite(rhs.norm()).item()
    threshold = max(atol, rtol * normalizer)
    if atol:
        mode = "ZERO_RHS_ABSOLUTE" if normalizer == 0 else "ABSOLUTE_OR_RELATIVE"
    else:
        mode = "ZERO_RHS_EXACT" if normalizer == 0 else "RELATIVE"
    status = replace(status, normalizer=normalizer, tolerance_mode=mode)
    try:
        x = torch.zeros_like(rhs) if x0 is None else x0.detach().clone()
        r = _finite(rhs - kernels.spmv(x))
        err_sq = dot(r, r)
        iterations, stop = 0, "INITIAL_RESIDUAL"
        if err_sq.sqrt().item() > threshold:
            z = kernels.precond(r)
            result = pcg_iterate(
                kernels=kernels,
                topology=PcgTopology(halo=lambda v: v, dot=dot),
                control=PcgControl(
                    max_iters=max_iters,
                    converged=lambda r, e: e.sqrt().item() <= threshold,
                    alpha_breakdown=lambda v: v.item() <= 0,
                    beta_breakdown=lambda v: v.item() <= 0,
                    emit_tail_on_final_iteration=False,
                ),
                x=x, p=z.clone(), r=r, z=z, rz=dot(r, z), err_sq=err_sq,
            )
            x, err_sq = result.x, result.err_sq
            iterations, stop = result.num_iters, result.exit_reason
        # This fresh operator call is independent of the recursive residual.
        absolute = _finite((rhs - kernels.spmv(x)).norm()).item()
        relative = absolute / normalizer if normalizer else None
        status = replace(status, iterations=iterations, stop_reason=stop,
                         true_residual_abs=absolute, true_residual_rel=relative,
                         recursive_residual_abs=err_sq.sqrt().item())
        if absolute > threshold:
            reason = (SolveReason.BREAKDOWN if "breakdown" in stop else
                      SolveReason.MAX_ITERATIONS if stop == "max_iters" else
                      SolveReason.PRIMAL_NOT_CONVERGED)
            raise SolveRejected(reason)
        return x, replace(status, primal_status="PASS", status="PASS")
    except SolveRejected as error:
        error.status = replace(status, primal_status="FAIL", status="FAIL", reason=error.reason)
        raise


def _check_versions(coefficients, versions):
    if any(c._version != version for c, version in zip(coefficients, versions)):
        raise SolveRejected(SolveReason.PARAMETER_VERSION_CHANGED)


class _CheckedRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, request, rhs, *coefficients):
        matvec, transpose, x0, preconditioner, limit, atol, rtol, record = request
        versions = tuple(c._version for c in coefficients)
        x, status = _solve(matvec, rhs, x0, preconditioner, limit, atol, rtol, record[0])
        _check_versions(coefficients, versions)
        record[0] = status
        ctx.request = request
        # Keep explicit references: check versions before accessing saved tensors
        # so mutation raises the contract reason, not PyTorch's generic error.
        ctx.coefficients, ctx.versions = coefficients, versions
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, bar_x):
        matvec, transpose, _, _, limit, atol, rtol, record = ctx.request
        try:
            if torch.is_grad_enabled():
                raise SolveRejected(SolveReason.UNSUPPORTED_CAPABILITY, "first-order VJP only")
            _check_versions(ctx.coefficients, ctx.versions)
            (x,) = ctx.saved_tensors
            # The adjoint right-hand side carries the loss scale, so tol_abs
            # would stop a small cotangent at its zero first guess.  The
            # re-solve keeps only a relative test: the relative residual the
            # forward threshold stands for, max(tol_rel, tol_abs / ||rhs||),
            # or tol_rel alone when the forward right-hand side is at or below
            # tol_abs; a zero cotangent therefore returns exactly zero.
            forward = record[0].normalizer
            adjoint_rtol = max(rtol, atol / forward) if forward > atol else rtol
            adjoint_status = replace(record[0], operator_identity=repr(transpose),
                                     primal_status="NOT_EVALUATED", status="NOT_EVALUATED",
                                     iterations=0, stop_reason="NOT_STARTED",
                                     true_residual_abs=None, true_residual_rel=None,
                                     recursive_residual_abs=None, adjoint=None,
                                     tolerances=(("tol_abs", 0.0), ("tol_rel", adjoint_rtol)))
            try:
                lam, adjoint_status = _solve(
                    transpose, bar_x, None, None, limit, 0.0, adjoint_rtol, adjoint_status)
            except SolveRejected as error:
                if error.reason == SolveReason.PRIMAL_NOT_CONVERGED:
                    error = SolveRejected(
                        SolveReason.ADJOINT_NOT_CONVERGED,
                        status=replace(error.status, reason=SolveReason.ADJOINT_NOT_CONVERGED))
                record[0] = replace(record[0], adjoint=error.status)
                raise error
            grads = [None] * len(ctx.coefficients)
            active = [i for i, c in enumerate(ctx.coefficients) if c.requires_grad]
            if active:
                with torch.enable_grad():
                    ax = _apply(matvec, x.detach())
                    if ax.requires_grad:
                        values = torch.autograd.grad(
                            ax, tuple(ctx.coefficients[i] for i in active),
                            grad_outputs=-lam, allow_unused=True)
                    else:
                        values = (None,) * len(active)
                for i, value in zip(active, values):
                    grads[i] = (torch.zeros_like(ctx.coefficients[i]) if value is None
                                else _finite(value))
            _check_versions(ctx.coefficients, ctx.versions)
            record[0] = replace(record[0], derivative_status="PASS", status="PASS",
                                adjoint=adjoint_status)
            return (None, lam, *grads)
        except SolveRejected as error:
            record[0] = replace(record[0], derivative_status="FAIL", status="FAIL",
                                reason=error.reason)
            error.status = record[0]
            raise


@torch._dynamo.disable
def _reject_non_eager():
    # Raised outside the traced frame: Dynamo (2.9) cannot construct an
    # exception whose __init__ coerces its arguments and fails with an
    # InternalTorchDynamoError instead of surfacing the rejection.
    raise SolveRejected(SolveReason.UNSUPPORTED_CAPABILITY, "eager VJP only")


def _eager_only():
    # A live forward mode is refused too: the solve runs in a Function with a
    # backward only, where a tangent hidden in the callbacks would be dropped.
    if (torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active()
            or forward_mode_active()):
        _reject_non_eager()


def checked_root_solve(matvec, rhs, *, coefficients, adjoint_matvec=None,
                       x0=None, preconditioner=None, max_iters, tol_abs, tol_rel,
                       mode="ROOT_IMPLICIT"):
    """Return a checked root and an immutable forward-status snapshot.

    ``coefficients`` must be a nonempty tuple, including constant operator
    tensors. All callbacks act in the input's real Euclidean coordinates.
    PCG requires positive definite operators and positive preconditioners;
    no nonsymmetric or singular-system qualification is implied.
    """
    _eager_only()
    if mode != "ROOT_IMPLICIT":
        raise SolveRejected(SolveReason.UNSUPPORTED_CAPABILITY, "ROOT_IMPLICIT only")
    if (not callable(matvec) or (adjoint_matvec is not None and not callable(adjoint_matvec))
            or (preconditioner is not None and not callable(preconditioner))
            or not isinstance(coefficients, tuple) or not coefficients
            or not isinstance(rhs, torch.Tensor) or rhs.numel() == 0
            or rhs.dtype not in (torch.float32, torch.float64)
            or type(max_iters) is not int or max_iters < 0
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0
                   for t in (tol_abs, tol_rel))):
        raise SolveRejected(SolveReason.INVALID_INPUT)
    for coefficient in coefficients:
        if (not isinstance(coefficient, torch.Tensor)
                or coefficient.dtype != rhs.dtype or coefficient.device != rhs.device):
            raise SolveRejected(SolveReason.INVALID_INPUT, "coefficient dtype/device mismatch")
        _finite(coefficient)
    coefficients = tuple({id(c): c for c in coefficients}.values())
    _finite(rhs)
    if x0 is not None:
        if (not isinstance(x0, torch.Tensor) or x0.shape != rhs.shape
                or x0.dtype != rhs.dtype or x0.device != rhs.device):
            raise SolveRejected(SolveReason.INVALID_INPUT, "invalid initial guess")
        _finite(x0)
    transpose = matvec if adjoint_matvec is None else adjoint_matvec
    record = [SolveStatus(
        mode=mode, operator_identity=repr(matvec), coordinate_system="REAL_EUCLIDEAN",
        control_set=("rhs", "coefficients"),
        tolerances=(("tol_abs", tol_abs), ("tol_rel", tol_rel)),
    )]
    request = (matvec, transpose, x0, preconditioner, max_iters, tol_abs, tol_rel, record)
    x = _CheckedRoot.apply(request, rhs, *coefficients)
    evaluations = _evaluations.get()
    if evaluations is not None:
        evaluations.append(record)
    return x, record[0]


def evaluate_loss_and_grad(loss_fn, params):
    """Compute without publishing .grad or performing optimizer updates.

    ``loss_fn(*params)`` must be pure and return a scalar tensor. Every checked
    solve reached during evaluation must also complete its VJP. No gradient
    tuple is returned on failure; existing parameter/optimizer state is unused.
    Multiple solves are represented by a tuple of completed status snapshots.
    """
    _eager_only()
    params = tuple(params)
    if not params or any(not isinstance(p, torch.Tensor) or not p.requires_grad for p in params):
        raise SolveRejected(SolveReason.INVALID_INPUT, "parameters must require gradients")
    records = []
    token = _evaluations.set(records)
    try:
        loss = loss_fn(*params)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise SolveRejected(SolveReason.INVALID_INPUT, "loss must be scalar")
        _finite(loss)
        if not records:
            raise SolveRejected(SolveReason.INVALID_INPUT, "loss contains no checked ROOT solve")
        grads = torch.autograd.grad(loss, params)
        for grad in grads:
            _finite(grad)
        if any(record[0].derivative_status != "PASS" for record in records):
            raise SolveRejected(SolveReason.ADJOINT_NOT_CONVERGED, "VJP was not evaluated")
        statuses = tuple(record[0] for record in records)
        return loss, grads, statuses[0] if len(statuses) == 1 else statuses
    finally:
        _evaluations.reset(token)


__all__ = ["SolveStatus", "SolveRejected", "checked_root_solve", "evaluate_loss_and_grad"]
