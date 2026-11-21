"""Velocity, transport and free-surface stages shared by regional and cube IFS.

Regional fields use [batch, nz, ny, nx] centres and C-grid U/V faces; cube
fields use face-first layouts. Elevation is upward-positive in m and velocity
in m s-1. Tensor casts, pressure-gradient masks and result masks stay explicit.
Reciprocal wet fractions use double-where so dry cells never divide by zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from locomgcm.operators.c_grid import (
    depth_integrated_face_transport,
    divergence_volume_transport_to_center_mit,
    gradient_eta_to_u,
    gradient_eta_to_v,
    synchronize_regional_face_values,
)

from .implicit_free_surface_cg2d import (
    _cube_eta_gradient,
    _reconcile_seam_transport,
    cg2d_solve,
)
from .ifs.trace import StreamTrace, _dist_trace


def make_momentum_predictor_masks(u, v, *, hfw_for_mask, hfs_for_mask) -> tuple:
    if hfw_for_mask is None:
        predictor_mask_u = torch.ones_like(u)
    else:
        predictor_mask_u = (hfw_for_mask > 0).to(dtype=u.dtype, device=u.device)
        if predictor_mask_u.ndim == u.ndim - 1:
            predictor_mask_u = predictor_mask_u.unsqueeze(0)
    if hfs_for_mask is None:
        predictor_mask_v = torch.ones_like(v)
    else:
        predictor_mask_v = (hfs_for_mask > 0).to(dtype=v.dtype, device=v.device)
        if predictor_mask_v.ndim == v.ndim - 1:
            predictor_mask_v = predictor_mask_v.unsqueeze(0)
    return predictor_mask_u, predictor_mask_v


def compute_velocity_predictor(
    u, v, tend_u, tend_v, *, dt_mom, predictor_mask_u, predictor_mask_v,
) -> tuple:
    # TIMESTEP leaves the old velocity outside the wet tendency mask.
    u_star = u + dt_mom * tend_u * predictor_mask_u
    v_star = v + dt_mom * tend_v * predictor_mask_v
    return u_star, v_star


def compute_unmasked_velocity_predictor(*, u, v, tend_u, tend_v, dt_mom) -> tuple:
    u_star = u + dt_mom * tend_u
    v_star = v + dt_mom * tend_v
    return u_star, v_star


def surface_pressure_gradients(eta, grid, *, dtype, device, ctx) -> tuple:
    if grid.face_count == 6:
        geu, gev = _cube_eta_gradient(eta, grid, dtype, device, ctx=ctx)
    else:
        geu = gradient_eta_to_u(eta, grid)  # [batch, ny, nx+1]
        gev = gradient_eta_to_v(eta, grid)  # [batch, ny+1, nx]
    return geu, gev


def apply_explicit_surface_pressure(
    u_star, v_star, *, explicit_frac, gravity, dt_mom, geu, gev,
    predictor_mask_u, predictor_mask_v,
) -> tuple:
    u_star = (
        u_star
        - explicit_frac
        * gravity
        * dt_mom
        * geu.unsqueeze(1)
        * predictor_mask_u
    )
    v_star = (
        v_star
        - explicit_frac
        * gravity
        * dt_mom
        * gev.unsqueeze(1)
        * predictor_mask_v
    )
    return u_star, v_star


def mask_velocity_pair(u, v, *, u_mask, v_mask) -> tuple:
    return u * u_mask, v * v_mask


def apply_unmasked_explicit_surface_pressure(
    *, u_star, v_star, explicit_frac, gravity, dt_mom, geu, gev,
) -> tuple:
    u_star = u_star - explicit_frac * gravity * dt_mom * geu.unsqueeze(1)
    v_star = v_star - explicit_frac * gravity * dt_mom * gev.unsqueeze(1)
    return u_star, v_star


def reciprocal_velocity_hfacs(u_star, v_star, *, hfw_in, hfs_in) -> tuple:
    recip_hFacW = None
    recip_hFacS = None
    if hfw_in is not None:
        hfw = hfw_in.to(dtype=u_star.dtype, device=u_star.device)
        _hfw_mask = hfw > 0
        _hfw_safe = torch.where(_hfw_mask, hfw, torch.ones_like(hfw))
        recip_hFacW = torch.where(_hfw_mask, 1.0 / _hfw_safe, torch.zeros_like(hfw))
    if hfs_in is not None:
        hfs = hfs_in.to(dtype=v_star.dtype, device=v_star.device)
        _hfs_mask = hfs > 0
        _hfs_safe = torch.where(_hfs_mask, hfs, torch.ones_like(hfs))
        recip_hFacS = torch.where(_hfs_mask, 1.0 / _hfs_safe, torch.zeros_like(hfs))
    return recip_hFacW, recip_hFacS


def surface_pressure_face_masks(surf, *, ny, nx) -> tuple:
    # u-face mask: 1 only if BOTH adjacent center cells are wet
    uf = torch.ones(ny, nx + 1, dtype=surf.dtype, device=surf.device)
    uf[:, 1:-1] = surf[:, :-1] * surf[:, 1:]
    uf[:, 0] = surf[:, 0]      # west domain edge
    uf[:, -1] = surf[:, -1]     # east domain edge
    # v-face mask
    vf = torch.ones(ny + 1, nx, dtype=surf.dtype, device=surf.device)
    vf[1:-1, :] = surf[:-1, :] * surf[1:, :]
    vf[0, :] = surf[0, :]
    vf[-1, :] = surf[-1, :]

    return uf, vf


def mask_surface_pressure_gradients(
    grad_u, grad_v, *, wet_face_mask_u, wet_face_mask_v, mask_in_w, mask_in_s,
) -> tuple:
    if wet_face_mask_u is not None:
        grad_u = grad_u * wet_face_mask_u.unsqueeze(0)
        grad_v = grad_v * wet_face_mask_v.unsqueeze(0)
    if mask_in_w is not None:
        grad_u = grad_u * mask_in_w.unsqueeze(0)
        grad_v = grad_v * mask_in_s.unsqueeze(0)

    return grad_u, grad_v


def compute_velocity_correction(
    u_star, v_star, *, alpha, gravity, dt_mom, grad_u, grad_v,
) -> tuple:
    u_new = u_star - alpha * gravity * dt_mom * grad_u.unsqueeze(1)
    v_new = v_star - alpha * gravity * dt_mom * grad_v.unsqueeze(1)
    return u_new, v_new


@dataclass(frozen=True)
class PredictorDivergence:
    u_transport: torch.Tensor
    v_transport: torch.Tensor
    divergence: torch.Tensor


def compute_predictor_divergence(
    u_star: torch.Tensor,
    v_star: torch.Tensor,
    grid: Any,
    *,
    stream_trace: StreamTrace | None,
    layer_thickness: torch.Tensor,
    dyG: torch.Tensor,
    dxG: torch.Tensor,
    hFacW: torch.Tensor | None = None,
    hFacS: torch.Tensor | None = None,
    rA: torch.Tensor | None = None,
    wet_mask: torch.Tensor | None = None,
    ctx: Any = None,
) -> PredictorDivergence:
    """Depth-integrated face transports -> transport divergence (cg2d RHS input).

    MITgcm ``calc_div_ghat.F``: build the face volume transports
    ``xA = dyG*drF*hFacW`` / ``yA = dxG*drF*hFacS``, sum over k, then take the
    centred divergence divided by ``rA``.  ``layer_thickness`` is the regional
    name and ``drF`` the cube name for the same metric; the caller supplies it in
    a broadcast-ready shape (regional ``[1,nz,1,1]``, cube per-face).

    No ``w`` / depth sign convention is in play (docs/NUMERICS.md): the inputs are
    horizontal face velocities and positive-definite metrics, and the divergence
    sign convention lives in the leaf operator, shared by both callers.

    Faithfulness notes:

    * **The two call shapes are load-bearing — do not unify**:
      the cube branch keeps ``keepdim=True`` plus the trailing ``[:, 0]``, and
      the regional branch keeps the bare 2-D reduction.  Dropping ``keepdim`` on
      the cube path silently mis-broadcasts ``rA.unsqueeze(1)`` and shifts the
      divergence by ~2e-5 (signal order, measured); giving the regional path the
      cube form would allocate nz× more and buy nothing.  Likewise the cube
      branch passes ``rA=`` / ``wet_mask=`` explicitly (the multi-rank path hands
      in locally-scattered tensors) while the regional branch lets the operator
      read them off ``grid`` — single-rank measurement: bitwise equal.

    * Each shared face has one transport after depth integration and
      before divergence. Regional tiles publish the east/north owner values.
      The cube rotates the vector across the seam (the depth-summed east/north
      ghost transports must come from the cross-seam neighbour, as MITgcm's
      ``EXCH_UV`` of gU/gV before ``CALC_DIV_GHAT`` does).

    * Traces: the two regional ``_dist_trace`` strings stay on the
      regional branch rather than cross-emitting. The cube branch
      is *shared* with the exactConserv consumer through
      ``implicit_free_surface_cube._compute_cube_depth_integrated_divergence``'s
      shim, and printing "predicted transport divergence" for the *corrected*
      divergence would be a false trace.  For the same reason the
      ``cube.predictor_transport_divergence`` stream trace fires only when a
      ``stream_trace`` is given: the predictor call sites pass one, the stepper-less
      shim passes ``None`` (that trace belongs to the predictor, never to the
      exactConserv path).
    """
    if grid.face_count == 6:
        # Keep a singleton nz axis on the depth-summed transports so the cube
        # divergence operator's ``rA.unsqueeze(1)=[face,1,ny,nx]`` broadcast aligns
        # (a bare 3-D ``[face,ny,nx]`` would mis-broadcast to ``[face,face,ny,nx]``).
        u_transport, v_transport = depth_integrated_face_transport(
            u_star,
            v_star,
            layer_thickness,
            dyG,
            dxG,
            hFacW=hFacW,
            hFacS=hFacS,
            keepdim=True,
        )
        u_transport, v_transport = _reconcile_seam_transport(
            u_transport,
            v_transport,
            grid,
            ctx=ctx,
        )
        divergence = divergence_volume_transport_to_center_mit(
            u_transport,
            v_transport,
            grid,
            rA=rA,
            wet_mask=wet_mask,
        )[:, 0]
    else:
        if hFacW is not None:
            u_transport, v_transport = depth_integrated_face_transport(
                u_star,
                v_star,
                layer_thickness,
                dyG,
                dxG,
                hFacW=hFacW,
                hFacS=hFacS,
            )  # [batch, ny, nx+1] / [batch, ny+1, nx]
        else:
            u_transport, v_transport = depth_integrated_face_transport(
                u_star,
                v_star,
                layer_thickness,
                dyG,
                dxG,
            )  # [batch, ny, nx+1] / [batch, ny+1, nx]
        u_transport, v_transport = synchronize_regional_face_values(
            u_transport, v_transport, grid,
        )
        _dist_trace("predicted transport divergence begin")
        divergence = divergence_volume_transport_to_center_mit(
            u_transport, v_transport, grid,
        )  # [batch, ny, nx]
        _dist_trace("predicted transport divergence done")

    if stream_trace is not None:
        stream_trace.emit_stream_trace(
            "cube.predictor_transport_divergence",
            u_transport_2d=u_transport,
            v_transport_2d=v_transport,
            div_star=divergence,
        )
    return PredictorDivergence(
        u_transport=u_transport,
        v_transport=v_transport,
        divergence=divergence,
    )


@torch._dynamo.disable
def _solve_cg2d_and_stage(
    stream_trace: StreamTrace,
    rhs: torch.Tensor,
    x0: torch.Tensor,
    coeffs: tuple[torch.Tensor, ...],
    grid: Any,
    *,
    is_cube: bool,
    max_iters: int,
    tolerance: float,
    tolerance_mode: str | None,
    use_min_res_sol: bool,
    ctx: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The free-surface solve, whole, as the traced step's one declared exit.

    ``cg2d_solve`` reports eager scalars or asynchronous graph diagnostics.
    Resolve the latter's completion event before publishing CPU tensors.
    Without this declared exit Dynamo resumes
    *inside* the PCG loop's break, so those two scalars land on the resumed
    frame's stack and are specialised there:
    ``L['___stack0'][2] == 1.166e-12`` failed on every step of the static-shape
    arm (the compile trace showed the guard
    surviving a fix applied only to the tail).  Fencing the tail was not enough
    because the crossing is the *return*, so the whole call is the exit: inside
    this function Dynamo does not look, and what comes back out is three
    tensors.

    The returned 0-d tensors reach StepDiagnostics with the solved elevation.
    """
    _dist_trace("cg2d_solve begin")
    eta_new, num_iters, residual = cg2d_solve(
        rhs,
        x0,  # initial guess (masked per the caller's x0 policy)
        *coeffs,
        grid,
        max_iters=max_iters,
        tolerance=tolerance,
        tolerance_mode=tolerance_mode,
        use_min_res_sol=use_min_res_sol,
        ctx=ctx,
    )
    _dist_trace("cg2d_solve done")
    from locomgcm.solvers._pcg_core.readback import read_pcg_stats

    # This event wait also precedes every downstream StepDiagnostics reader,
    # including deferred finiteness checks and cube observers.
    num_iters, residual = read_pcg_stats(num_iters, residual)
    if is_cube:
        eta_new = eta_new[0]
    stream_trace.emit_stream_trace(
        "cube.cg2d_solution",
        eta_new=eta_new,
        num_iters=int(num_iters),
        residual=float(residual),
    )
    stats = (
        torch.tensor(int(num_iters), dtype=torch.int64, device="cpu"),
        torch.tensor(float(residual), dtype=torch.float64, device="cpu"),
    )
    return eta_new, stats[0], stats[1]


def subtract_real_freshwater_rhs(
    *,
    rhs: torch.Tensor,
    empmr_mass_flux: torch.Tensor,
    rA: torch.Tensor,
    free_surf_fac: float,
    mass_to_r_unit: float,
    implic_div_2d_flow: float,
    dt_freesurf: float,
) -> torch.Tensor:
    """Map MITgcm's real-freshwater ``cg2d_b`` term to integrated eta.

    ``empmr_mass_flux`` is MIT ``EmPmR`` in kg m-2 s-1, positive for net
    evaporation.  Preserve the factor order from ``solve_for_pressure.F:146-153``
    before applying the ``-deltaTMom*deltaTFreeSurf`` pressure-to-eta mapping.
    """
    empmr = empmr_mass_flux.to(dtype=rA.dtype, device=rA.device)
    tmp_fac = (
        torch.as_tensor(free_surf_fac, dtype=rA.dtype, device=rA.device)
        * torch.as_tensor(mass_to_r_unit, dtype=rA.dtype, device=rA.device)
        * torch.as_tensor(implic_div_2d_flow, dtype=rA.dtype, device=rA.device)
    )
    freshwater_rhs = tmp_fac * rA * empmr
    freshwater_rhs = freshwater_rhs * torch.as_tensor(
        dt_freesurf, dtype=rA.dtype, device=rA.device
    )
    return rhs - freshwater_rhs


def compute_free_surface_rhs(*, rA, eta_source, beta, dt_freesurf, div_star):
    """Integrated eta Helmholtz RHS before freshwater and boundary masks."""
    rhs = rA * (eta_source - beta * dt_freesurf * div_star)
    return rhs


def prepare_free_surface_solve(
    *, rhs, eta, surface_mask, mask_in_c, x0_mask, is_cube, stream_trace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask the integrated RHS and retain each layout's initial-guess policy."""
    # Mask RHS at land cells (force η=0 there)
    if surface_mask is not None:
        rhs = rhs * surface_mask
    # Mask RHS at OB cells (MITgcm: cg2d_b *= maskInC in solve_for_pressure.F);
    # OB cells have aC=1, i.e. the trivial η=0 equation.
    if mask_in_c is not None:
        rhs = rhs * mask_in_c
    stream_trace.emit_stream_trace(
        "cube.cg2d_rhs",
        rhs=rhs,
    )

    x0 = eta if x0_mask is None else eta * x0_mask
    if is_cube:
        rhs = rhs.unsqueeze(0)
        x0 = x0.unsqueeze(0)

    return rhs, x0
