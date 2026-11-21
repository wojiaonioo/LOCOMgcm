"""Nonhydrostatic CG3D pressure solver.

``cg3d_b`` is assembled by PRE_CG3D.  Any LOCOMgcm-downward-positive diagnostic
vertical velocity has already been converted to MIT-upward-positive ``wVel``
before the RHS reaches this solver, so no w/wVel sign conversion occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import torch
from locomgcm.accel.tensor_context import forward_mode_active
from locomgcm.distributed.comm_adjoint import active_data_group
import torch.distributed as dist

from locomgcm.dycore_parameters import NonhydrostaticParams
from locomgcm.geometry import StructuredCGrid

from .cg3d_operator import (
    CG3DOperator,
    _global_max_scalar,
    _global_sum_scalar,
    apply_cg3d_operator,
)
from .cg2d import ToleranceMode, _validate_tolerance_mode
from .cg3d_preconditioner import CG3DPreconditioner, apply_cg3d_preconditioner
from ._pcg_core.adjoint import PcgAdjointSpec, pcg_solve_adjoint, seven_point_coefficient_grads
from .tridiag import forward_mode_requested

# Fused CUDA kernels for the PCG body run on the shared topology-free core
# (``solvers/_pcg_core``), which also serves cg2d.  The gate read and the ledger
# write live in that core's dispatcher, and the kernels see padded interior
# blocks plus pre-assembled coefficients -- no grid, no grid family, no rank
# count.  Default off; enable with LOCOMGCM_CG3D_FUSED_PCG=1.
# Declining is always safe: the eager loop below is the bit-identical reference.




def _mit_rhs_normalization(rhs: torch.Tensor) -> torch.Tensor:
    rhs_max = _global_max_scalar(rhs.abs().max(), group=active_data_group())
    rhs_normalization = torch.ones((), dtype=rhs.dtype, device=rhs.device)
    rhs_nonzero = rhs_max != 0
    rhs_safe = torch.where(rhs_nonzero, rhs_max, rhs_normalization)
    normalized = rhs_normalization / rhs_safe
    return torch.where(rhs_nonzero, normalized, rhs_normalization)


class CG3DResult(NamedTuple):
    phi_nh: torch.Tensor
    first_residual: float
    last_residual: float
    num_iters: int


@dataclass(frozen=True)
class CG3DSolver:
    """MITgcm-style plain PCG solver for the 3-D NH pressure equation."""

    # Below this iteration budget the caller is exercising a deliberately
    # truncated map rather than asking for the root, so the derivative of the
    # ITERATES is the one it wants.  The contract suite pins k=1 and k=2 for
    # exactly that purpose; production cg3d budgets are in the dozens-to-
    # hundreds.
    _ROOT_MIN_ITERS = 8

    params: NonhydrostaticParams
    time_scales: Any = None

    def solve(
        self,
        cg3d_b: torch.Tensor,
        cg3d_x_init: torch.Tensor,
        operator: CG3DOperator,
        preconditioner: CG3DPreconditioner,
        mask: torch.Tensor | None = None,
        grid: StructuredCGrid | None = None,
        *,
        tolerance_mode: ToleranceMode = "absolute",
        err_sq_history: list[torch.Tensor] | None = None,
        dot: Any = None,
        root_adjoint: bool | None = None,
    ) -> CG3DResult:
        """Solve ``A phi_nh = cg3d_b`` with the selected residual criterion.

        ``absolute`` preserves the historical operator-resolved path byte for
        byte. ``mit_normalized`` forces MITgcm's dimensionless
        ``||r|| / max(abs(rhs)) < target`` path, including RHS/initial-solution
        scaling before the PCG loop. ``err_sq_history`` records per-iteration
        residual squares for verification and does not affect solver state.

        ``dot(a, b)`` is the global inner product -- the topology injection the
        accelerated path reduces with.  ``None`` means the default lat-lon
        reduction (a single ``torch.sum`` then an all-reduce); a face-ordered
        reducer is passed in by a caller whose decomposition needs single-rank
        and N-rank to agree bitwise.  It is a CALLER decision, never a test this
        solver makes. The eager loop below reduces with the same
        expression written inline, unchanged.
        """
        tolerance_mode = _validate_tolerance_mode(tolerance_mode)
        # A x = b is an implicit function of (b, coefficients), so its VJP is
        # one more solve of the SAME system -- it does not need the iteration
        # history.  Taping the loop instead differentiates the ITERATES, and a
        # PCG iterate's derivative converges far slower than the iterate does:
        # on the closed zero-mean gauge system the taped VJP sits at 2.03e-4
        # relative while the forward residual is already 8.9e-8, and tightening
        # the tolerance makes it worse, not better.  The analytic
        # adjoint reaches 1.16e-11 on that same system.
        #
        # Only the differentiable call takes this path; with grad disabled or
        # no input requiring grad the plain loop below runs exactly as before,
        # byte for byte, with the tape footprint it always had (none).
        if forward_mode_active():
            self._refuse_peers_in_forward_mode(grid)
        if self._wants_adjoint(cg3d_b, cg3d_x_init, preconditioner, operator,
                               grid=grid, root_adjoint=root_adjoint):
            return self._solve_with_adjoint(
                cg3d_b, cg3d_x_init, operator, preconditioner, mask,
                grid=grid, tolerance_mode=tolerance_mode,
                err_sq_history=err_sq_history, dot=dot,
            )
        if cg3d_b.ndim != 4:
            raise ValueError(
                f"cg3d_b must be [batch,nz,ny,nx], got {tuple(cg3d_b.shape)}"
            )
        if cg3d_x_init.shape != cg3d_b.shape:
            raise ValueError("cg3d_x_init must match cg3d_b")
        if operator.aC3d.shape != cg3d_b.shape[1:]:
            raise ValueError("operator shape must match cg3d_b")
        if preconditioner.zMC.shape != cg3d_b.shape[1:]:
            raise ValueError("preconditioner shape must match cg3d_b")
        if grid is None and dist.is_available() and dist.is_initialized():
            raise ValueError(
                "distributed CG3D solve requires grid for pressure halo exchange"
            )

        dtype = cg3d_b.dtype
        device = cg3d_b.device
        if mask is None:
            mask4 = torch.ones_like(cg3d_b)
        else:
            mask_t = mask.to(dtype=dtype, device=device)
            if mask_t.ndim == 2:
                mask4 = mask_t.unsqueeze(0).unsqueeze(0)
            elif mask_t.ndim == 3:
                mask4 = mask_t.unsqueeze(0)
            elif mask_t.ndim == 4:
                mask4 = mask_t
            else:
                raise ValueError(
                    "mask must be [ny,nx], [nz,ny,nx], or [batch,nz,ny,nx], "
                    f"got {tuple(mask_t.shape)}"
                )
            mask4 = mask4.expand_as(cg3d_b)

        b = cg3d_b * operator.cg3dNorm.to(dtype=dtype, device=device) * mask4
        x = cg3d_x_init.clone() * mask4
        rhs_norm = torch.ones((), dtype=dtype, device=device)
        normalise_rhs = (
            operator.cg3d_normalise_rhs or tolerance_mode == "mit_normalized"
        )
        if normalise_rhs:
            if tolerance_mode == "mit_normalized":
                rhs_norm = _mit_rhs_normalization(b)
            else:
                rhs_max = _global_max_scalar(b.abs().max(), group=active_data_group())
                # Double-where: in the rhs_max==0 dead branch, feed reciprocal()
                # a sentinel 1.0 so backward never differentiates 1/0.
                _rhs_pos = rhs_max > 0
                _rhs_safe = torch.where(
                    _rhs_pos,
                    rhs_max,
                    torch.ones((), dtype=dtype, device=device),
                )
                rhs_norm = torch.where(
                    _rhs_pos,
                    _rhs_safe.reciprocal(),
                    torch.ones((), dtype=dtype, device=device),
                )
            b = b * rhs_norm
            x = x * rhs_norm

        r = (b - apply_cg3d_operator(x, operator, grid=grid)) * mask4
        err_sq = _global_sum_scalar(torch.sum(r * r), group=active_data_group())
        first_residual = float(torch.sqrt(err_sq).detach().cpu().item())
        if tolerance_mode == "mit_normalized":
            tolerance = torch.as_tensor(
                self.params.cg3d_target_residual,
                dtype=dtype,
                device=device,
            )
            tolerance_sq = tolerance * tolerance
        else:
            tolerance_sq = operator.cg3dTolerance_sq.to(dtype=dtype, device=device)
        converged = (
            err_sq < tolerance_sq
            if tolerance_mode == "mit_normalized"
            else err_sq <= tolerance_sq
        )
        if converged:
            if normalise_rhs:
                x = x / rhs_norm
            return CG3DResult(
                phi_nh=x * mask4,
                first_residual=first_residual,
                last_residual=first_residual,
                num_iters=0,
            )

        s = torch.zeros_like(b)
        eta_qr_prev = torch.ones((), dtype=dtype, device=device)
        actual_its = 0
        last_residual = first_residual

        fused_state = self._fused_iteration(
            x=x,
            r=r,
            s=s,
            err_sq=err_sq,
            eta_qr_prev=eta_qr_prev,
            mask4=mask4,
            operator=operator,
            preconditioner=preconditioner,
            grid=grid,
            dot=dot,
            tolerance_sq=tolerance_sq,
            tolerance_mode=tolerance_mode,
            err_sq_history=err_sq_history,
            first_residual=first_residual,
            states=(cg3d_b, cg3d_x_init),
        )
        if fused_state is not None:
            x, r, actual_its, last_residual = fused_state
        else:
            for _ in range(int(self.params.cg3d_max_iters)):
                q_prec = apply_cg3d_preconditioner(r, preconditioner) * mask4
                eta_qr = _global_sum_scalar(torch.sum(q_prec * r), group=active_data_group())
                if torch.abs(eta_qr_prev) == 0:
                    break
                cg_beta = eta_qr / eta_qr_prev
                s = (q_prec + cg_beta * s) * mask4
                q_mat = apply_cg3d_operator(s, operator, grid=grid) * mask4
                alpha_denom = _global_sum_scalar(torch.sum(s * q_mat), group=active_data_group())
                if torch.abs(alpha_denom) == 0:
                    break
                alpha = eta_qr / alpha_denom
                x = (x + alpha * s) * mask4
                r = (r - alpha * q_mat) * mask4
                err_sq = _global_sum_scalar(torch.sum(r * r), group=active_data_group())
                actual_its += 1
                last_residual = float(torch.sqrt(err_sq).detach().cpu().item())
                if err_sq_history is not None:
                    err_sq_history.append(err_sq.detach().clone())
                converged = (
                    err_sq < tolerance_sq
                    if tolerance_mode == "mit_normalized"
                    else err_sq <= tolerance_sq
                )
                if converged:
                    break
                eta_qr_prev = eta_qr

        if normalise_rhs:
            x = x / rhs_norm
        return CG3DResult(
            phi_nh=x * mask4,
            first_residual=first_residual,
            last_residual=last_residual,
            num_iters=actual_its,
        )

    def _wants_adjoint(self, cg3d_b, cg3d_x_init, preconditioner, operator,
                       *, grid=None, root_adjoint=None) -> bool:
        """True when the implicit-function VJP is the right derivative here.

        Three conditions, all necessary:

        1. Grad is enabled and some mathematical input requires it.  Without
           this the plain loop runs and nothing is taped, as before.
        2. The iteration is allowed to run to convergence.  ``ROOT`` semantics
           say ``x = A^-1 b``, and only then is the derivative one more solve.
           A caller that pins ``cg3d_max_iters`` to a small ``k`` is asking for
           a FIXED truncated map -- ``k`` steps of PCG, a different function
           whose derivative genuinely does run through the iterates -- so the
           taped loop is the CORRECT answer there, not a fallback.  That is
           also the boundary ``checked_root.py`` draws in its first paragraph.
        3. Neither the initial guess nor the preconditioner carries gradient.
           Both are algorithmic controls that the root does not depend on, so
           under ROOT semantics their derivative is zero -- but a caller that
           made one of them a leaf is measuring the truncated map, same as (2).
        Second derivatives are NOT available on this path -- the adjoint's
        re-solve runs under ``no_grad``, so there is no graph to differentiate
        again.  ``create_graph`` is only known once backward runs, far too late
        to choose here, so the rejection is raised there instead of silently
        returning a gradient that carries no ``grad_fn``.

        Getting (2) wrong is not a performance question: it silently replaces
        one function's derivative with another's.

        A live forward mode (``forward_mode_active``) counts as an enabled
        grad in (1) whatever the grad mode, since ``no_grad`` does not stop it;
        a tangent on an input counts as that input requiring a derivative in
        (1) and (3).  Reading ``requires_grad`` alone sent a tangent through
        the plain loop, the iterates' derivative with the warm start's tangent
        included.

        ``root_adjoint=True`` declares ROOT semantics, so (2) and (3) do not
        apply: a derivative requested through any input, the warm start and the
        preconditioner included, takes the ROOT path, where those two are
        passive and their derivative is zero.  Before, a request through them
        alone ran the plain loop and returned the iterates' derivative.
        """
        if root_adjoint is False:
            return False
        tensors = tuple(t for t in (cg3d_b, operator.aW3d, operator.aS3d, operator.aV3d,
                                    operator.aC3d, operator.cg3dNorm)
                        if isinstance(t, torch.Tensor))
        passive = tuple(t for t in (cg3d_x_init, preconditioner.zMC,
                                    preconditioner.zML, preconditioner.zMU)
                        if isinstance(t, torch.Tensor))
        if root_adjoint is True:
            from locomgcm.distributed.comm_adjoint import session_for_grid

            # Checkpoint recording runs under no_grad, but it must record the
            # same atomic solve as the gradient-enabled replay, not its inner
            # iteration halos. Session membership fixes that choice per window.
            if session_for_grid(grid) is not None:
                return True
            # The physical lagged pressure enters RHS; its numerical warm
            # start is not a control of the converged pressure equation.
            return forward_mode_active() or (torch.is_grad_enabled() and any(
                t.requires_grad for t in (*tensors, *passive)
            ))
        if not torch.is_grad_enabled() and not forward_mode_active():
            return False
        if int(self.params.cg3d_max_iters) < self._ROOT_MIN_ITERS:
            return False
        if any(t.requires_grad for t in passive) or forward_mode_requested(*passive):
            return False
        return forward_mode_requested(*tensors) or any(
            isinstance(t, torch.Tensor) and t.requires_grad
            for t in (cg3d_b, operator.aW3d, operator.aS3d,
                      operator.aV3d, operator.aC3d)
        )

    @staticmethod
    def _refuse_peers_in_forward_mode(grid) -> None:
        """Refuse a live forward mode with peer ranks, before the first exchange.

        Every path (ROOT, plain loop, any iteration budget) exchanges halos and
        reduces through operators and collectives without a forward-mode rule.
        The reading (open level, session, data group size) is the same on every
        rank that opens the same dual level, so those ranks raise here together.
        """
        from locomgcm.distributed.comm_adjoint import session_for_grid

        if session_for_grid(grid) is not None or (
                dist.is_available() and dist.is_initialized()
                and dist.get_world_size(active_data_group()) > 1):
            raise NotImplementedError(
                "CG3D: forward-mode derivatives of a solve with peer ranks are not "
                "supported; its communication provides first-order reverse "
                "derivatives only; take forward-mode derivatives on a single rank"
            )

    def _solve_with_adjoint(
        self,
        cg3d_b: torch.Tensor,
        cg3d_x_init: torch.Tensor,
        operator: CG3DOperator,
        preconditioner: CG3DPreconditioner,
        mask: torch.Tensor | None,
        *,
        grid: StructuredCGrid | None,
        tolerance_mode: ToleranceMode,
        err_sq_history: list[torch.Tensor] | None,
        dot: Any,
    ) -> CG3DResult:
        """Differentiate the normalized equation, with one collective VJP.

        A_norm x = norm*b. The normalization belongs on the explicit RHS,
        before the autograd boundary. The inverse used in backward is then
        exactly A_norm^-T, not norm*A_norm^-T. The operator is symmetric on
        the active wet rows. Coefficient partials use the actual halo topology.

        A bound session schedules the whole solve and its transpose as one
        ordered operation. Krylov-internal communication uses the same grid
        and native kernels but no model session; it is neither taped nor
        replayed as model-level halo rounds.
        """
        from locomgcm.distributed.comm_adjoint import constant_transport, session_for_grid
        from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange

        session = session_for_grid(grid)
        # The numerical solve needs topology, not a snapshot of the grid's
        # per-step physical bindings. A shallow copy retains old pressure and
        # velocity fields in every checkpointed solve recipe. Keep one grid;
        # its transport is unbound only while the untaped solve/exchange runs.
        # All numerical coefficients and masks remain captured below.
        detached = CG3DOperator(
            aW3d=operator.aW3d.detach(), aS3d=operator.aS3d.detach(),
            aV3d=operator.aV3d.detach(), aC3d=operator.aC3d.detach(),
            cg3dNorm=torch.ones_like(operator.cg3dNorm),
            cg3dTolerance_sq=operator.cg3dTolerance_sq.detach(),
            cg3d_normalise_rhs=operator.cg3d_normalise_rhs,
        )
        numerical_preconditioner = CG3DPreconditioner(
            *(value.detach() for value in preconditioner)
        )
        stats: dict[str, Any] = {}
        # The transpose re-solve's right-hand side carries the loss scale, not
        # the forward's.  An absolute floor (cg3d_target_res_w_unit > 0, no
        # RHS normalization) would stop a cotangent nine orders of magnitude
        # smaller at its zero first guess, so the re-solve scales the floor by
        # ||rhs_adjoint|| / ||rhs_forward||: it stops at the relative residual
        # the floor stands for in the forward, and exactly at zero for a zero
        # cotangent.  A forward right-hand side at or below the floor stands
        # for no relative accuracy; the re-solve then takes MITgcm's normalized
        # test (cg3d_target_residual).  Normalizing operators and
        # ``mit_normalized`` already test relatively and are left as they are.
        absolute_floor = tolerance_mode == "absolute" and not operator.cg3d_normalise_rhs
        forward_rhs_sq: list[torch.Tensor] = []

        def rhs_norm_sq(rhs):
            masked = rhs * mask4
            return _global_sum_scalar(torch.sum(masked * masked), group=active_data_group())

        def numerical_solve(rhs, initial, *, primal=False, mode=tolerance_mode,
                            solve_operator=detached):
            with constant_transport():
                result = self.solve(
                    rhs, initial, solve_operator, numerical_preconditioner, mask,
                    grid=grid, tolerance_mode=mode,
                    err_sq_history=err_sq_history if primal else None,
                    dot=dot, root_adjoint=False,
                )
            tolerance = (float(self.params.cg3d_target_residual)
                         if mode == "mit_normalized"
                         else float(solve_operator.cg3dTolerance_sq.sqrt()))
            if not math.isfinite(result.last_residual) or result.last_residual > tolerance:
                raise RuntimeError(
                    f"CG3D ROOT solve did not converge: residual={result.last_residual}, "
                    f"tolerance={tolerance}, iterations={result.num_iters}"
                )
            if primal:
                stats.update(first_residual=result.first_residual,
                             last_residual=result.last_residual, num_iters=result.num_iters)
            return result.phi_nh

        def invert(rhs):
            zeros = torch.zeros_like(rhs)
            if not absolute_floor:
                return numerical_solve(rhs, zeros)
            forward_sq = forward_rhs_sq[-1]
            if forward_sq <= detached.cg3dTolerance_sq:
                return numerical_solve(rhs, zeros, mode="mit_normalized")
            relative = detached._replace(
                cg3dTolerance_sq=detached.cg3dTolerance_sq * (rhs_norm_sq(rhs) / forward_sq))
            return numerical_solve(rhs, zeros, solve_operator=relative)

        # A checkpoint retains the solve recipe. Capture only numerical
        # storage here, not the initial guess's upstream autograd graph.
        numerical_initial = cg3d_x_init.detach()

        def primal(rhs):
            if absolute_floor:
                forward_rhs_sq.append(rhs_norm_sq(rhs))
            return numerical_solve(rhs, numerical_initial, primal=True)

        mask4 = self._expand_mask(mask, cg3d_b)

        def coefficient_grads(d_bar, x, needs):
            # This exchange is unconditional: every rank must participate,
            # including a rank whose own coefficient slots are passive.
            if grid is None:
                x_halo = torch.nn.functional.pad(x, (1, 1, 1, 1))
            else:
                with constant_transport():
                    x_halo = exchange(x, grid=grid, location="T",
                                      extent=HaloExtentV1.uniform(1), fill=FillSpec.ZERO)
            d_halo = torch.nn.functional.pad(d_bar, (1, 1, 1, 1))
            return seven_point_coefficient_grads(
                d_halo, x_halo, needs, mask=mask4, coefficient_ndim=3)

        def coefficient_tangent(x, coefficients):
            # The operator the solve inverts, M A (mask at the store, as in
            # ``seven_point_coefficient_grads``), applied once to the plain
            # ``x`` with the dual coefficients.
            aW3d, aS3d, aV3d, aC3d = coefficients
            applied_operator = detached._replace(aW3d=aW3d, aS3d=aS3d, aV3d=aV3d, aC3d=aC3d)
            with constant_transport():
                applied = apply_cg3d_operator(x, applied_operator, grid=grid) * mask4
            return torch.autograd.forward_ad.unpack_dual(applied).tangent

        spec = PcgAdjointSpec(solve=invert, coefficient_grads=coefficient_grads,
                              coefficient_tangent=coefficient_tangent, primal_solve=primal)
        rhs = cg3d_b * operator.cg3dNorm.to(cg3d_b) * mask4
        coefficients = (operator.aW3d, operator.aS3d, operator.aV3d, operator.aC3d)
        phi = pcg_solve_adjoint(spec, rhs, *coefficients, session=session)
        return CG3DResult(phi_nh=phi * mask4, **stats)

    @staticmethod
    def _expand_mask(mask, cg3d_b) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(cg3d_b)
        mask_t = mask.to(dtype=cg3d_b.dtype, device=cg3d_b.device)
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
        elif mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(0)
        elif mask_t.ndim != 4:
            raise ValueError(
                "mask must be [ny,nx], [nz,ny,nx], or [batch,nz,ny,nx], "
                f"got {tuple(mask_t.shape)}"
            )
        return mask_t.expand_as(cg3d_b)

    def _fused_iteration(
        self,
        *,
        x: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        err_sq: torch.Tensor,
        eta_qr_prev: torch.Tensor,
        mask4: torch.Tensor,
        operator: CG3DOperator,
        preconditioner: CG3DPreconditioner,
        grid: StructuredCGrid | None,
        dot: Any,
        tolerance_sq: torch.Tensor,
        tolerance_mode: ToleranceMode,
        err_sq_history: list[torch.Tensor] | None,
        first_residual: float,
        states: tuple[torch.Tensor, ...],
    ):
        """Replay fused PCG on CUDA, use the shared host core, or return ``None``.

       The graph keeps eager's unrotated order and reads a packed stop status
        once per iteration. The loop rotation below applies to the host core
        used for gradients, custom dot products and uncapturable topologies.

        ``None`` means the caller runs the eager loop above, which is the
        bit-identical reference; the reason is already in the ledger.  Nothing
        is mutated in place, so a mid-loop refusal leaves the caller's ``x`` and
        ``r`` exactly as they were and the fallback is a clean restart.

        **The loop is rotated by half a turn.**  Eager opens each pass with the
        preconditioner and closes with the operator apply; the core opens with
        the apply.  Lifting the first head into this pre-loop makes the two
        identical statement for statement, including where both breakdown tests
        sit -- ``|eta_qr_prev| == 0`` is evaluated after the new ``eta_qr``
        exists and before the division, which is exactly the core's
        ``beta_breakdown``. The one residue is the final pass: eager stops
        after the convergence test without computing the next search direction,
        hence ``emit_tail_on_final_iteration=False``.

       The pre-loop's ``|eta_qr_prev| == 0`` test is absent because eager's is
        unreachable there: it runs with the initial ``eta_qr_prev = 1``.  It is
        the one host synchronisation this path does not reproduce, so the
        per-SOLVE count is ``4N - 1`` where eager's is ``4N`` -- value-neutral,
        but any future sync-count claim has to say ``4N - 1``.
        Per ITERATION the four are all in eager's
        places.
        """
        from ._cg3d_fused.dispatcher import make_center_halo, pcg_iteration_kernels
        from ._pcg_core import PcgControl, PcgTopology, pcg_iterate

        halo = make_center_halo(grid)
        # ``mask4`` reaches here as ``expand_as``, i.e. a stride-0 view whenever
        # the caller passed a 2-D/3-D mask or a batch > 1.  The kernels index
        # raw pointers, so the shared predicate refuses a non-contiguous
        # operand -- correctly, but it would name a broadcast a layout fault and
        # decline a solve it could serve.  Production
        # (batch 1, 3-D mask) is already contiguous, so this is a no-op there
        # and value-neutral everywhere: it is the same numbers in a dense
        # buffer, and the eager loop keeps the view it had.
        mask4 = mask4.contiguous()
        thomas = (preconditioner.zMC, preconditioner.zML, preconditioner.zMU)
        kernels = pcg_iteration_kernels(
            states=states,
            aW3d=operator.aW3d,
            aS3d=operator.aS3d,
            aV3d=operator.aV3d,
            aC3d=operator.aC3d,
            mask=mask4,
            thomas=thomas,
            halo=halo,
        )
        if kernels is None:
            return None

        from locomgcm.accel import current_rank, gate, record_dispatch

        if gate("cg3d_fast_thomas").effective_value:
            record_dispatch(
                "cg3d_fast_thomas", outcome="inapplicable",
                variant="vertical_line_thomas", site="cg3d.CG3DSolver.solve",
                backend="cuda", rank=current_rank(),
                reason="superseded_by_cg3d_fused_pcg",
            )

        # Reuse the CG2D topology contract. The native kernels remain unaware
        # of rank layout; only their caller decides whether halo capture is legal.
        from .cg2d import Cg2dGraphPartition, cg2d_graph_topology

        layout = getattr(grid, "parallel_layout", None)
        partition = Cg2dGraphPartition(
            faces=int(getattr(grid, "face_count", 1)),
            tiles_x=int(getattr(layout, "nPx", 1)),
            tiles_y=int(getattr(layout, "nPy", 1)),
            ranks=dist.get_world_size() if dist.is_initialized() else 1,
            cells_x=int(getattr(layout, "global_nx", grid.nx)),
            cells_y=int(getattr(layout, "global_ny", grid.ny)),
            halo_width=1,
        )
        graph_reason = cg2d_graph_topology(partition)
        if torch.is_grad_enabled():
            graph_reason = "grad_enabled"
        elif dot is not None:
            graph_reason = "custom_dot"
        cache = getattr(grid, "_runtime_cache", None)
        if cache is None:
            graph_reason = "graph_cache_unavailable"
        if graph_reason is None:
            from ._cg3d_fused.graph import cg3d_graphed_iteration

            with kernels:
                return cg3d_graphed_iteration(
                    kernels, halo, cache, x=x, r=r, p=s, err_sq=err_sq,
                    tolerance_sq=tolerance_sq, tolerance_mode=tolerance_mode,
                    max_iters=int(self.params.cg3d_max_iters),
                    first_residual=first_residual, err_sq_history=err_sq_history,
                )
        kernels._record("inapplicable", variant="pcg_cuda_graph", reason=graph_reason)

        if dot is None:
            def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return _global_sum_scalar(torch.sum(a * b), group=active_data_group())

        residual_state = {"last": first_residual}

        def _on_iteration(num_iters, *, err_sq, **_ignored) -> None:
            # Eager's order exactly: the readback first, then the history.
            residual_state["last"] = float(
                torch.sqrt(err_sq).detach().cpu().item()
            )
            if err_sq_history is not None:
                err_sq_history.append(err_sq.detach().clone())

        def _converged(residual: torch.Tensor, err_sq_value: torch.Tensor):
            if tolerance_mode == "mit_normalized":
                return err_sq_value < tolerance_sq
            return err_sq_value <= tolerance_sq

        # A fallback re-runs the eager loop, which appends to
        # ``err_sq_history`` from scratch; without this the caller would see the
        # abandoned attempt's entries in front of the real ones -- a verification
        # instrument silently corrupted by an accelerator refusal
        # . The solve itself was already correct; the
        # history was not.
        history_mark = None if err_sq_history is None else len(err_sq_history)
        try:
            with kernels:
                # The rotated head of eager's first pass.  ``dot(q_prec, r)``
                # rather than eager's ``sum(q_prec * r)`` operand order is the
                # same bits: IEEE multiplication is commutative.
                q_prec = kernels.precond(r)
                eta_qr = dot(q_prec, r)
                cg_beta = eta_qr / eta_qr_prev
                s = kernels.axpy(q_prec, s, cg_beta)
                outcome = pcg_iterate(
                    kernels=kernels,
                    topology=PcgTopology(halo=halo, dot=dot),
                    control=PcgControl(
                        max_iters=int(self.params.cg3d_max_iters),
                        converged=_converged,
                        alpha_breakdown=lambda denom: torch.abs(denom) == 0,
                        beta_breakdown=lambda rz: torch.abs(rz) == 0,
                        on_iteration=_on_iteration,
                        emit_tail_on_final_iteration=False,
                    ),
                    x=x,
                    p=s,
                    r=r,
                    z=q_prec,
                    rz=eta_qr,
                    err_sq=err_sq,
                )
        except Exception as exc:  # noqa: BLE001
            # Preserve the original selected failure, including its cause.
            # __exit__ may have recorded it already; record_error is idempotent.
            kernels.record_error(exc)
            if history_mark is not None:
                del err_sq_history[history_mark:]
            raise
        return outcome.x, outcome.r, outcome.num_iters, residual_state["last"]
