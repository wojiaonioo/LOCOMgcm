"""Five-band column solve and equation derivatives.

All six inputs have shape [..., nz, ny, nx] and retain their dtype. Bands
a/b/c/d/e multiply levels k-2/k-1/k/k+1/k+2; out-of-column bands are unused.
The caller supplies wet-row equations and nonsingular elimination pivots.
No velocity or depth convention enters this algebraic solve.
"""
from __future__ import annotations

import torch

from .tridiag import _shift_down, _shift_up, forward_mode_requested


def _pentadiag_mitgcm(a, b, c, d, e, rhs):
    """SOLVE_PENTADIAGONAL's ordinary top-down normalized elimination."""
    nz = rhs.shape[-3]
    dp = torch.empty_like(rhs)
    ep = torch.empty_like(rhs)
    yp = torch.empty_like(rhs)
    for k in range(nz):
        ck = c[..., k, :, :]
        dk = d[..., k, :, :]
        yk = rhs[..., k, :, :]
        if k == 1:
            beta = b[..., k, :, :]
            ck = ck - beta * dp[..., k - 1, :, :]
            dk = dk - beta * ep[..., k - 1, :, :]
            yk = yk - beta * yp[..., k - 1, :, :]
        elif k > 1:
            ak = a[..., k, :, :]
            beta = b[..., k, :, :] - ak * dp[..., k - 2, :, :]
            ck = ck - beta * dp[..., k - 1, :, :] - ak * ep[..., k - 2, :, :]
            dk = dk - beta * ep[..., k - 1, :, :]
            yk = yk - beta * yp[..., k - 1, :, :] - ak * yp[..., k - 2, :, :]
        reciprocal = 1.0 / ck
        dp[..., k, :, :] = dk * reciprocal
        ep[..., k, :, :] = e[..., k, :, :] * reciprocal
        yp[..., k, :, :] = yk * reciprocal
    x = torch.empty_like(rhs)
    for k in range(nz - 1, -1, -1):
        value = yp[..., k, :, :]
        if k < nz - 1:
            value = value - x[..., k + 1, :, :] * dp[..., k, :, :]
        if k < nz - 2:
            value = value - x[..., k + 2, :, :] * ep[..., k, :, :]
        x[..., k, :, :] = value
    return x


class _PentadiagSolveAdjoint(torch.autograd.Function):
    """Implicit VJP with all five matrix bands and the RHS active."""

    generate_vmap_rule = True

    @staticmethod
    def forward(a, b, c, d, e, rhs):
        return _pentadiag_mitgcm(a, b, c, d, e, rhs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:5], output)

    @staticmethod
    def backward(ctx, grad_x):
        a, b, c, d, e, x = ctx.saved_tensors
        rhs_bar = _PentadiagSolveAdjoint.apply(
            _shift_down(_shift_down(e)), _shift_down(d), c,
            _shift_up(b), _shift_up(_shift_up(a)), grad_x.contiguous(),
        )
        neighbours = (_shift_down(_shift_down(x)), _shift_down(x), x,
                      _shift_up(x), _shift_up(_shift_up(x)))
        bands_bar = tuple(-rhs_bar * value if needed else None
                          for value, needed in zip(neighbours, ctx.needs_input_grad[:5]))
        return (*bands_bar, rhs_bar if ctx.needs_input_grad[5] else None)


def pentadiag_solve(a, b, c, d, e, rhs):
    """Solve A x = rhs, with implicit forward- and reverse-mode derivatives.

    The forward tangent solves A dx = d(rhs) - dA x using the same bands.
    The transpose solve in the VJP retains coefficient dependencies for
    reverse derivatives, as in the tridiagonal solver.
    """
    operands = (a, b, c, d, e, rhs)
    if forward_mode_requested(*operands):
        duals = tuple(torch.autograd.forward_ad.unpack_dual(value) for value in operands)
        primals = tuple(value.primal for value in duals)
        x = pentadiag_solve(*primals)
        tangent_rhs = duals[5].tangent
        if tangent_rhs is None:
            tangent_rhs = torch.zeros_like(x)
        neighbours = (_shift_down(_shift_down(x)), _shift_down(x), x,
                      _shift_up(x), _shift_up(_shift_up(x)))
        for dual, value in zip(duals[:5], neighbours):
            if dual.tangent is not None:
                tangent_rhs = tangent_rhs - dual.tangent * value
        tangent = pentadiag_solve(*primals[:5], tangent_rhs)
        return torch.autograd.forward_ad.make_dual(x, tangent)
    if torch.is_grad_enabled() and any(value.requires_grad for value in operands):
        return _PentadiagSolveAdjoint.apply(*operands)
    return _pentadiag_mitgcm(*operands)
