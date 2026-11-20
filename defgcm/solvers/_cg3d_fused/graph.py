"""CG3D fused iteration replay with graph-owned operands and exact stop points.

Each inner product is formed inside the kernel that holds its operands,
changing summation order from the eager multiply and reduction sequence.
The shared host core uses the standard operators when a caller supplies its
own dot product, enables gradients, or supplies a topology the graph cannot
capture. Collective reductions remain at the same points in the iteration."""

from __future__ import annotations

import torch
from defgcm.distributed.comm_adjoint import active_data_group

from defgcm.accel import cuda_graphs
from defgcm.accel.tensor_context import ordinary_tensor_mode
from defgcm.geometry.padded_field import PaddedField
from defgcm.solvers.cg3d_operator import _global_sum_scalar

from .dispatcher import Cg3dFusedIteration
from ._fast import cg3d_update_xr_inplace_dot


def _reduce(partials):
    """Finish a folding kernel's per-block partial sums into the global scalar.

    The rank-local half of the reduction moved into the kernel; the collective
    did NOT move.  It is the same ``all_reduce`` on the same 0-d tensor at the
    same point of the iteration, so every rank still executes exactly one per
    inner product per replay -- which is what keeps the captured graph
    rank-uniform.

    """
    return _global_sum_scalar(torch.sum(partials), group=active_data_group())


@ordinary_tensor_mode()
def _capture(operands, halo, refresh, x, r, p, err_sq, tolerance_sq, normalized):
    # The cached workspace is reseeded by later solves, including gradient
    # windows following an inference-only forward on this grid.
    static = tuple(value.detach().clone() for value in operands)
    native = Cg3dFusedIteration(
        aW3d=static[0], aS3d=static[1], aV3d=static[2], aC3d=static[3],
        mask=static[4], thomas=static[5:],
    )
    sx, sr = x.clone(), r.clone()
    # The search direction lives in a RESIDENT padded buffer: the axpy writes
    # its interior, ``refresh`` brings the border up to date in place, and the
    # SPMV reads the whole buffer without concatenating or copying a new one.
    sp = PaddedField.like(p, halo=1)
    prev = torch.ones((), dtype=x.dtype, device=x.device)
    serr, tolerance = err_sq.clone(), tolerance_sq.clone()
    status = torch.empty(4, dtype=x.dtype, device=x.device)

    def iteration():
        # This is the original CG3D loop rotation: precondition first. All
        # ranks execute the same collectives, including a terminating replay.
        #
        # Each inner product is produced by the kernel that already holds both
        # its operands, and ``_reduce`` finishes the per-block partial sums.
        # See the module docstring for why that makes this path class B.
        q, eta_partials = native.precond_dot(sr)
        eta = _reduce(eta_partials)
        beta_bad = prev.abs() == 0
        beta = eta / torch.where(beta_bad, torch.ones_like(prev), prev)
        # Reads the previous direction from the interior and writes the new one
        # back into it: the operand and destination are the same cells.
        native.axpy_into(q, sp, beta)
        refresh(sp)
        # ``p`` for <p, Ap> is the centre of the padded block the SPMV already
        # reads, so this product costs no additional load at all.
        ap, denom_partials = native.spmv_dot(sp.buffer)
        denom = _reduce(denom_partials)
        alpha_bad = denom.abs() == 0
        alpha = eta / torch.where(alpha_bad, torch.ones_like(denom), denom)
        valid = ~(beta_bad | alpha_bad)
        # A graph cannot branch at the two eager breakdown sites. Select the
        # pre-iteration state on those exits, including its residual square.
        # A rejected iteration writes neither x/r nor a non-zero contribution,
        # so the partials sum to 0 and the ``where`` below discards it anyway.
        err_partials = cg3d_update_xr_inplace_dot(
            sx, sp.buffer, sr, ap, alpha, static[4], valid, sp.halo
        )
        next_err = _reduce(err_partials)
        serr.copy_(torch.where(valid, next_err, serr))
        prev.copy_(eta)
        converged = serr < tolerance if normalized else serr <= tolerance
        status.copy_(torch.stack((beta_bad, alpha_bad, converged, serr.sqrt())))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            iteration()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode=cuda_graphs.CAPTURE_ERROR_MODE):
        iteration()
    cuda_graphs.register(graph)
    return dict(graph=graph, operands=static, x=sx, r=sr, p=sp, prev=prev,
                err=serr, tolerance=tolerance, status=status)


def cg3d_graphed_iteration(
    kernels, halo, cache, *, x, r, p, err_sq, tolerance_sq, tolerance_mode,
    max_iters, first_residual, err_sq_history,
):
    """Replay one iteration per host stop decision; return independent state.

    Coefficients and Thomas factors are refreshed on every solve. The cache is
    owned by the grid, whose decomposition also owns the injected halo.
    """
    operands = (kernels._aW3d, kernels._aS3d, kernels._aV3d, kernels._aC3d,
                kernels._mask, *kernels._thomas)
    key = ("cg3d_fused_graph", x.shape, x.dtype, x.device, tolerance_mode)
    ctx = cache.get(key)
    if ctx is None:
        # The in-place refresh rides on the halo injection, resolved against
        # the same grid (``dispatcher.make_center_halo``).  Its ABSENCE is an
        # error, not a default: a graph that skipped the refresh would leave
        # the border at whatever the allocation held, and on a periodic domain
        # that is NOT what a ZERO-filled exchange produces -- ``_ghost_cells``
        # tests PERIODIC before it tests the fill, so the west ghost is the
        # east column, not zero.  Falling back to "no refresh" would therefore
        # solve a different system in silence.
        refresh = getattr(halo, "refresh", None)
        if refresh is None:
            raise ValueError(
                "the injected halo carries no in-place refresh; the graph "
                "cannot keep a resident border up to date without one. Build "
                "the injection with dispatcher.make_center_halo, which "
                "attaches it."
            )
        ctx = _capture(operands, halo, refresh,
                       x, r, p, err_sq, tolerance_sq,
                       tolerance_mode == "mit_normalized")
        cache[key] = ctx
    for target, source in zip(ctx["operands"], operands):
        target.copy_(source)
    ctx["x"].copy_(x)
    ctx["r"].copy_(r)
    ctx["p"].interior.copy_(p)
    ctx["prev"].fill_(1)
    ctx["err"].copy_(err_sq)
    ctx["tolerance"].copy_(tolerance_sq)
    completed = replays = 0
    last_residual = first_residual
    for _ in range(max_iters):
        ctx["graph"].replay()
        replays += 1
        beta_bad, alpha_bad, converged, residual = ctx["status"].cpu().tolist()
        if beta_bad or alpha_bad:
            break
        completed += 1
        last_residual = residual
        if err_sq_history is not None:
            err_sq_history.append(ctx["err"].detach().clone())
        if converged:
            break
    kernels._launches.update({name: replays for name in (
        "thomas_mask", "axpy_s", "spmv_halo", "update_xr", "graph_replay",
    )})
    kernels.iterations = completed
    return ctx["x"].clone(), ctx["r"].clone(), completed, last_residual
