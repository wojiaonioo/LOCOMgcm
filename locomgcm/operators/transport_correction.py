"""Divergence correction for frozen velocity fields via pressure-projection.

Implements the standard approach used by MITgcm (pressure correction) and
ROMS (continuity-consistent mass flux): ensure the transport field fed to
tracer advection satisfies the discrete continuity equation.

Two-step algorithm for wall-boundary domains:

    Step 1 — Boundary flux balance:
        Compute total domain divergence.  Redistribute the imbalance
        uniformly across all boundary faces so that the net divergence
        integrates to zero.

    Step 2 — Poisson pressure-projection:
        Solve  DG(φ) = div(u_balanced, v_balanced)  where DG = div ∘ grad
        is the *exact* discrete composition of the C-grid operators.
        Then  u_corr = u_balanced − ∂φ/∂x,  v_corr = v_balanced − ∂φ/∂y.

The corrected ``(u_corr, v_corr)`` satisfies ``div(u_corr, v_corr) ≈ 0``
to solver tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.geometry import BoundaryType, StructuredCGrid
from locomgcm.operators.c_grid import (
    divergence_transport_to_center,
    gradient_eta_to_u,
    gradient_eta_to_v,
)


@dataclass(frozen=True)
class CorrectionInfo:
    """Diagnostic information from a divergence correction solve."""

    div_before_rms: float
    div_after_rms: float
    div_before_max: float
    div_after_max: float
    iterations: int
    converged: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "div_before_rms": self.div_before_rms,
            "div_after_rms": self.div_after_rms,
            "div_before_max": self.div_before_max,
            "div_after_max": self.div_after_max,
            "iterations": self.iterations,
            "converged": self.converged,
        }


def _rms(t: torch.Tensor) -> float:
    return float(t.square().mean().sqrt().item())


def _amax(t: torch.Tensor) -> float:
    return float(t.abs().max().item())


def _div_grad(phi: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    """Compute DG(φ) = div(grad(φ)), the exact discrete Laplacian operator
    consistent with the C-grid gradient and divergence stencils."""
    return divergence_transport_to_center(
        gradient_eta_to_u(phi, grid),
        gradient_eta_to_v(phi, grid),
        grid,
    )


def _balance_boundary_flux(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust wall-boundary face velocities so that total domain divergence
    integrates to zero.

    For periodic boundaries no adjustment is needed (periodic divergence
    already integrates to zero by construction).  For wall boundaries,
    the net volume flux through wall faces is redistributed uniformly
    across all active (wall) boundary faces.

    Returns adjusted (u, v) with the same shapes.
    """
    periodic_x = grid.boundary_topology.periodic_x
    periodic_y = grid.boundary_topology.periodic_y
    wall_x = grid.boundary_topology.west_east == BoundaryType.WALL
    wall_y = grid.boundary_topology.south_north == BoundaryType.WALL

    if periodic_x and periodic_y:
        return u, v  # nothing to do

    # Total divergence integral = sum(div * cell_area) which, for this
    # uniform-grid operator, is the discrete net outward boundary flux.
    div = divergence_transport_to_center(u, v, grid)
    total_div = div.sum(dim=(-2, -1)) * grid.dx * grid.dy

    # Count boundary face lengths and redistribute
    u_adj = u.clone()
    v_adj = v.clone()

    total_boundary_length = 0.0
    if wall_x:
        # West face (index 0) and East face (index nx)
        total_boundary_length += 2.0 * grid.ny * grid.dy
    if wall_y:
        # South face (index 0) and North face (index ny)
        total_boundary_length += 2.0 * grid.nx * grid.dx

    if total_boundary_length < 1e-30:
        return u, v

    # Desired outward-flux change per unit boundary length.  The face
    # variables are velocities [m/s], so dividing the integrated flux
    # imbalance [m²/s] by a boundary length [m] gives the per-face
    # velocity correction directly.
    correction_per_length = -total_div / total_boundary_length

    if wall_x:
        # West outward flux is -u_w; East outward flux is +u_e.
        u_adj[..., 0] = u_adj[..., 0] - correction_per_length.unsqueeze(-1)
        u_adj[..., -1] = u_adj[..., -1] + correction_per_length.unsqueeze(-1)
    if wall_y:
        # South outward flux is -v_s; North outward flux is +v_n.
        v_adj[..., 0, :] = v_adj[..., 0, :] - correction_per_length.unsqueeze(-1)
        v_adj[..., -1, :] = v_adj[..., -1, :] + correction_per_length.unsqueeze(-1)

    return u_adj, v_adj


def _poisson_solve(
    rhs: torch.Tensor,
    grid: StructuredCGrid,
    *,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, int, bool]:
    """Solve DG(φ) = rhs using weighted Jacobi iteration.

    The operator DG = div ∘ grad uses the exact C-grid stencils, so the
    correction is automatically consistent with the transport divergence.

    Prerequisites: rhs must have near-zero mean (Neumann compatibility).

    Returns (phi, iterations, converged).
    """
    dx = grid.dx
    dy = grid.dy
    dx2 = dx * dx
    dy2 = dy * dy
    periodic_x = grid.boundary_topology.periodic_x
    periodic_y = grid.boundary_topology.periodic_y

    # Build per-cell diagonal of the DG operator
    diag = torch.zeros_like(rhs)
    if periodic_x:
        diag -= 2.0 / dx2
    else:
        diag[..., 1:] -= 1.0 / dx2
        diag[..., :-1] -= 1.0 / dx2
    if periodic_y:
        diag -= 2.0 / dy2
    else:
        diag[..., 1:, :] -= 1.0 / dy2
        diag[..., :-1, :] -= 1.0 / dy2

    # Safety for degenerate grids
    diag = torch.where(diag.abs() < 1e-30, diag.new_tensor(-1.0), diag)

    # Damped Jacobi: omega < 1 for guaranteed convergence
    # (omega = 2/3 is the standard safe choice for 2D Laplacian)
    omega = 2.0 / 3.0

    phi = torch.zeros_like(rhs)
    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        dg_phi = _div_grad(phi, grid)
        residual = rhs - dg_phi
        phi = phi + omega * residual / diag
        phi = phi - phi.mean()
        iterations = iteration

        if iteration % 100 == 0:
            res_rms = _rms(residual)
            if res_rms < tol:
                converged = True
                break

    # Final residual check
    if not converged:
        final_res = _rms(rhs - _div_grad(phi, grid))
        if final_res < tol:
            converged = True

    return phi, iterations, converged


def divergence_correction(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, CorrectionInfo]:
    """Correct a velocity field to be divergence-free via pressure projection.

    Parameters
    ----------
    u : Tensor
        Velocity on u-faces, shape ``(..., ny, nx+1)``.
    v : Tensor
        Velocity on v-faces, shape ``(..., ny+1, nx)``.
    grid : StructuredCGrid
        Grid geometry.
    max_iter : int
        Maximum Jacobi iterations.
    tol : float
        Convergence tolerance on the RMS Poisson residual.

    Returns
    -------
    u_corr, v_corr : Tensor
        Corrected velocities with the same shapes as the inputs.
    info : CorrectionInfo
        Diagnostic information about the correction.
    """
    if not grid.metrics.is_uniform_x or not grid.metrics.is_uniform_y:
        raise NotImplementedError(
            "divergence correction supports only uniform horizontal spacing"
        )
    if grid.boundary_topology.has_open_boundaries:
        raise NotImplementedError("divergence correction is only supported for periodic or wall boundary topologies")

    # 0. Record raw divergence
    div_raw = divergence_transport_to_center(u, v, grid)
    div_before_rms = _rms(div_raw)
    div_before_max = _amax(div_raw)

    # 1. Balance boundary flux so total divergence integrates to zero
    u_bal, v_bal = _balance_boundary_flux(u, v, grid)

    # 2. Poisson solve: DG(φ) = div(u_bal, v_bal)
    div_bal = divergence_transport_to_center(u_bal, v_bal, grid)
    div_bal = div_bal - div_bal.mean(dim=(-2, -1), keepdim=True)
    phi, iterations, converged = _poisson_solve(
        div_bal,
        grid,
        max_iter=max_iter,
        tol=tol,
    )

    # 3. Correct: u_corr = u_bal - ∂φ/∂x, v_corr = v_bal - ∂φ/∂y
    u_corr = u_bal - gradient_eta_to_u(phi, grid)
    v_corr = v_bal - gradient_eta_to_v(phi, grid)

    # 4. Diagnostics on corrected divergence
    div_corr = divergence_transport_to_center(u_corr, v_corr, grid)
    div_after_rms = _rms(div_corr)
    div_after_max = _amax(div_corr)

    info = CorrectionInfo(
        div_before_rms=div_before_rms,
        div_after_rms=div_after_rms,
        div_before_max=div_before_max,
        div_after_max=div_after_max,
        iterations=iterations,
        converged=converged,
    )
    return u_corr, v_corr, info
