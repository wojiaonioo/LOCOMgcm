"""Regional pressure potentials, Picard initialization, and NH corrections.

Potentials are cell-centered [batch, nz, ny, nx] fields in m^2/s^2.
The initial-pressure RMS is summed in the potential's dtype over MITgcm's
maskC (a boolean wet mask) and returned as a Python float.
Layer thickness and depth are in meters, positive downward. Picard iterates
keep the EOS/pressure-diagnostic dtype and retain active gradient tapes.
U/V corrections are in m/s on staggered faces; nonhydrostatic potentials are
converted to the velocity dtype/device before wet and interior face masking.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.distributed as dist

from defgcm.accel.graph_exit import graph_exit
from defgcm.geometry import StructuredCGrid
from defgcm.state import DiagnosticState, EvaluationInputs, PrognosticState
from defgcm.geometry.grid import _compute_interior_mask

from .carry import _carry
from .trace import _DIST_TRACE_ENABLED, _dist_trace


#: Iteration count of MITgcm's INI_PRESSURE loop (``ini_pressure.F:91``).
INI_PRESSURE_ITERATIONS = 15


def _initial_pressure_wet_cells(grid: StructuredCGrid) -> torch.Tensor:
    """MITgcm's maskC as a boolean mask: ``hFacC > 0``, or, on a grid without hFacC, the
    centre mask (``MaskSet``: every wet cell is then fully wet), 2D or 3D."""
    hfac = grid.masks.hFacC
    return (grid.masks.center if hfac is None else hfac).detach() > 0


def _picard_update_rms(
    *,
    new_phi: torch.Tensor, prev_phi: Optional[torch.Tensor], wet: torch.Tensor
) -> float:
    """RMS change of one initial-pressure iteration, formed as ``ini_pressure.F:118-140`` does.

    The squares of ``new_phi - prev_phi`` summed over the wet cells (``wet`` the boolean maskC
    of :func:`_initial_pressure_wet_cells`; a dry cell's value, finite or not, does not enter),
    divided by the number of wet cells, then the square root; zero when no cell is wet
    (:137-138).  ``prev_phi`` None is the zero potential the loop starts from (:62-72).
    Both sums are global (``_GLOBAL_SUM_RL``, :135-136): every rank reads the same value and
    stops at the same iteration, which the loop needs because its body exchanges halos.
    Single rank: no all_reduce.

    The loop reads only two properties of the value: whether it is exactly zero (continue
    while positive, :92) and, after the last iteration, whether it equals the previous one
    (:155-156).  The summation order, which differs from MIT's tile order, can move only
    the second, and only when 15 iterations end without an exact zero.
    """
    update = new_phi if prev_phi is None else new_phi - prev_phi
    wet = wet.expand_as(update)
    sums = torch.stack((torch.where(wet, update ** 2, 0.0).sum(), wet.sum().to(update.dtype)))
    with graph_exit():  # The initial-pressure convergence read:
        # a host read that decides how many iterations run, the PCG host
        # loop's twin, with the same treatment: a declared exit, not a silent
        # break.  It runs where the hydrostatic seed runs, before the first
        # traced step.
        if dist.is_available() and dist.is_initialized():
            from defgcm.distributed.transport_failure import data_all_reduce
            data_all_reduce(sums, op=dist.ReduceOp.SUM)
        squares, count = (float(value) for value in sums.tolist())
    return 0.0 if count == 0.0 else math.sqrt(squares / count)


def clone_total_phi_hyd(*, phi: torch.Tensor) -> torch.Tensor:
    """Clone a hydrostatic potential while preserving tracked gradients."""
    return _carry(phi).clone()


def seed_total_phi_hyd_picard(
    *,
    state: PrognosticState,
    grid: StructuredCGrid,
    eos,
    hpg,
    tensors: tuple[torch.Tensor, ...],
    standard_hpg: bool,
    rematerialize: bool,
    momentum_pressure_forcing: bool,
) -> Optional[torch.Tensor]:
    """Iterate EOS and HPG diagnostics to produce the initial potential.

    MITgcm's INI_PRESSURE rule (``ini_pressure.F:86-176``): at most
    ``INI_PRESSURE_ITERATIONS`` iterations, each run only while the previous
    iteration's RMS change is positive, so the loop ends at the first exact
    zero.  If 15 iterations end with a nonzero change, it is accepted only
    when it equals the change of the iteration before (converged to machine
    precision; an INFO record says so, as MITgcm prints a note); otherwise the
    seed raises, as MITgcm stops.
    """
    from defgcm.accel.tensor_context import forward_mode_active
    from defgcm.dynamics.equation_of_state import JMD95EquationOfState
    from defgcm.solvers.tridiag import forward_mode_requested
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    # The seed consumes EOS and column pressure diagnostics. Its unused
    # horizontal pressure gradients exchange halos; rematerialization must
    # omit those exchanges so backward adds no communication schedule.
    from defgcm.distributed.comm_adjoint import constant_transport

    def iteration(previous, *inputs):
        current = state.with_tensors(temp=inputs[0], salt=inputs[1], eta=inputs[2]) if inputs else state
        if inputs:
            saved_thickness = grid.layer_thickness
            saved_hfac = grid.masks.hFacC
        try:
            if inputs:
                object.__setattr__(grid.vertical_coordinate, "layer_thickness", inputs[3])
                object.__setattr__(grid.masks, "hFacC", inputs[4] if len(inputs) > 4 else None)
            with constant_transport():
                diagnostics = DiagnosticState(inputs=EvaluationInputs(
                    total_phi_hyd=previous,
                    momentum_pressure_forcing=momentum_pressure_forcing,
                ))
                eos.populate_diagnostic_state(current, diagnostics, grid)
                if standard_hpg:
                    hpg.populate_pressure_diagnostics(current, diagnostics, grid, return_potential=True)
                else:
                    hpg.evaluate(current, diagnostics, grid)
                phi_field = diagnostics.fields.get("total_hydrostatic_potential")
                return None if phi_field is None else clone_total_phi_hyd(phi=phi_field.data)
        finally:
            if inputs:
                object.__setattr__(grid.vertical_coordinate, "layer_thickness", saved_thickness)
                object.__setattr__(grid.masks, "hFacC", saved_hfac)

    changes: list[float] = []
    wet = _initial_pressure_wet_cells(grid)

    def record_change(phi, previous):
        rms = _picard_update_rms(
            new_phi=phi.detach(),
            prev_phi=None if previous is None else previous.detach(),
            wet=wet,
        )
        changes.append(rms)
        if _DIST_TRACE_ENABLED:
            _dist_trace(f"picard iter={len(changes)} rms={rms!r}")
        return rms

    # Subclasses may implement a different EOS or have side effects.
    group_columns = (
        rematerialize and state.temp.data.is_cuda and standard_hpg
        and type(eos) is JMD95EquationOfState and not forward_mode_active()
    )
    # One midpoint stays live through the rollout. During seed backward each
    # half rebuilds its own column tape after the rollout heads are released.
    group_size = (INI_PRESSURE_ITERATIONS + 1) // 2 if group_columns else 1

    def group(previous, *inputs, start):
        # start is an argument of this checkpoint. On replay the original
        # global decisions are already recorded, including a short last group.
        replay = start < len(changes)
        stop = min(start + group_size, len(changes) if replay else INI_PRESSURE_ITERATIONS)
        for _ in range(start, stop):
            argument = torch.zeros_like(inputs[0]) if previous is None else previous
            phi = iteration(argument, *inputs)
            if phi is None:
                return None
            if not replay:
                rms = record_change(phi, previous)
                if not rms > 0.0:
                    return phi
            previous = phi
        return previous

    new_phi = None
    with constant_transport(), set_checkpoint_early_stop(False):
        for start in range(0, INI_PRESSURE_ITERATIONS, group_size):
            if group_columns:
                new_phi = checkpoint(group, new_phi, *tensors, start=start, use_reentrant=False)
            else:
                previous = new_phi
                argument = torch.zeros_like(state.temp.data) if previous is None else previous
                if rematerialize and (previous is None or not forward_mode_requested(previous)):
                    new_phi = checkpoint(iteration, argument, *tensors, use_reentrant=False)
                else:
                    new_phi = iteration(argument, *tensors)
                if new_phi is not None:
                    record_change(new_phi, previous)
            if new_phi is None:
                return None
            if not changes[-1] > 0.0:  # ini_pressure.F:92
                break
    wet = None  # Only the primal reads maskC; the group closures outlive this call.
    rms = changes[-1]
    rms_previous = changes[-2] if len(changes) > 1 else 1.0
    if rms != 0.0 and rms != rms_previous:  # :155-174
        raise RuntimeError(
            "initial hydrostatic pressure did not converge after "
            f"{INI_PRESSURE_ITERATIONS} iterations (RMS change {rms!r}, previous "
            f"{rms_previous!r}); MITgcm stops here (ini_pressure.F:170-174)"
        )
    if rms != 0.0:
        # MITgcm prints the same note on this branch (ini_pressure.F:155-168).
        logging.getLogger("defgcm.steppers.implicit_free_surface").info(
            "initial hydrostatic pressure: after %d iterations the RMS change %r equals the "
            "previous one; the last iterate is accepted as converged to machine precision",
            INI_PRESSURE_ITERATIONS, rms,
        )

    return new_phi


def apply_pnh_correction_from_phi(
    *,
    u_new: torch.Tensor,
    v_new: torch.Tensor,
    grid: StructuredCGrid,
    dt_mom: float,
    phi: torch.Tensor,
    surface_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a MIT pressure-potential ``phi_nh`` correction to U/V."""
    if phi.ndim == 3:
        phi = phi.unsqueeze(0)
    if phi.ndim != 4 or phi.shape[1:] != (grid.nz, grid.ny, grid.nx):
        raise ValueError(
            "phi_nh must be [nz, ny, nx] or [batch, nz, ny, nx], "
            f"got {tuple(phi.shape)}"
        )
    phi = phi.to(dtype=u_new.dtype, device=u_new.device)

    from defgcm.operators.c_grid import gradient_center_to_u, gradient_center_to_v

    grad_phi_u = gradient_center_to_u(phi, grid)
    grad_phi_v = gradient_center_to_v(phi, grid)

    if grid.masks.has_3d_masks and surface_mask is not None:
        surf = surface_mask[0]
        uf = torch.ones(grid.ny, grid.nx + 1, dtype=surf.dtype, device=surf.device)
        uf[:, 1:-1] = surf[:, :-1] * surf[:, 1:]
        uf[:, 0] = surf[:, 0]
        uf[:, -1] = surf[:, -1]
        vf = torch.ones(grid.ny + 1, grid.nx, dtype=surf.dtype, device=surf.device)
        vf[1:-1, :] = surf[:-1, :] * surf[1:, :]
        vf[0, :] = surf[0, :]
        vf[-1, :] = surf[-1, :]
        grad_phi_u = grad_phi_u * uf.unsqueeze(0).unsqueeze(0)
        grad_phi_v = grad_phi_v * vf.unsqueeze(0).unsqueeze(0)

    if grid.boundary_topology.open_x or grid.boundary_topology.open_y:
        _, maskInW, maskInS = _compute_interior_mask(
            grid, grad_phi_u.dtype, grad_phi_u.device,
        )
        grad_phi_u = grad_phi_u * maskInW.unsqueeze(0).unsqueeze(0)
        grad_phi_v = grad_phi_v * maskInS.unsqueeze(0).unsqueeze(0)

    dt_t = torch.tensor(dt_mom, dtype=u_new.dtype, device=u_new.device)
    u_new = u_new - dt_t * grad_phi_u.expand_as(u_new)
    v_new = v_new - dt_t * grad_phi_v.expand_as(v_new)
    return u_new, v_new
