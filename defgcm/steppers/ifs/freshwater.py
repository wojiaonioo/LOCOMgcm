"""Surface thermodynamics, continuity history and free-surface sources.

Elevation is upward-positive in m; continuity subtracts outward transport
divergence. PmEpR is -EmPmR*maskInC in kg m-2 s-1; the carried value is the
preceding continuity's, which synchronous stepping reads, while staggered
stepping reads the current step's. Regional tensors use C-grid centres and U/V
faces; cube continuity history keeps its face-first layout. Dtype casts are
explicit.
ContinuityHistory.w_interface_prev is MIT upward-positive interface w in
m s-1 with shape [B,nz+1,ny,nx].
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from defgcm.state import PrognosticState
from defgcm.operators import depth_integrated_face_transport, divergence_volume_transport_to_center_mit
from defgcm.operators.c_grid import synchronize_regional_face_values
from defgcm.geometry.grid import _compute_interior_mask
from defgcm.steppers.step_segments import compute_predictor_divergence

from .carry import _carry
from .surface import surface_level_slice


@dataclass(frozen=True)
class ContinuityHistory:
    eta_h: torch.Tensor | None = None
    w_interface_prev: torch.Tensor | None = None

    def with_eta_h(self, eta_h: torch.Tensor) -> ContinuityHistory:
        return ContinuityHistory(eta_h, self.w_interface_prev)


def encode_continuity_history(history: ContinuityHistory):
    payload = {}
    if history.eta_h is not None:
        payload["_etaH"] = history.eta_h
    if history.w_interface_prev is not None:
        payload["_w_interface_prev"] = history.w_interface_prev
    return payload


def decode_continuity_history(payload):
    return ContinuityHistory(payload.get("_etaH"), payload.get("_w_interface_prev"))


def encode_freshwater_history(value):
    return {} if value is None else {"_pmep_r": value}


def decode_freshwater_history(payload):
    return payload.get("_pmep_r")


def freeze_surface(
    temp: torch.Tensor, *, freezing_point: float,
) -> tuple:
    # MITgcm DO_OCEANIC_PHYS calls FREEZE_SURFACE before either
    # THERMODYNAMICS branch. Clamp the state entering all step computation;
    # the completed tracer is deliberately allowed to fall below the floor.
    # The vertical axis is dim=-3 so optional leading dimensions do not
    # alter which level is the surface.
    previous_surface = temp[..., :1, :, :]
    surface = torch.clamp(previous_surface, min=freezing_point)
    return previous_surface, surface


def surface_freezing_flux(
    previous_surface, *, freezing_point, dt_tracer, layer_thickness, hfac_c,
) -> torch.Tensor:
    sst = previous_surface.squeeze(-3)
    cold = sst < freezing_point
    warming = torch.where(cold, freezing_point - sst, 0.0)
    return torch.where(
        cold,
        warming * layer_thickness[0] * hfac_c[0] / dt_tracer,
        0.0,
    )


def install_frozen_surface(bounded: PrognosticState, *, surface) -> PrognosticState:
    return bounded.with_tensors(
        temp=torch.cat((surface, bounded.temp.data[..., 1:, :, :]), dim=-3)
    )


def regional_continuity_rate(
    *, u_final, v_final, layer_thickness, hFacW, hFacS, grid,
) -> torch.Tensor:
    # 9. Recompute η from corrected velocity divergence (exactConserv)
    #    MITgcm: integr_continuity.F with exactConserv=.TRUE.
    #    dEtaHdt = -div(H·u^{n+1}) / A   (sign: positive dEtaHdt = rising η)
    #    etaN = etaH_prev + β·dEtaHdt·dt_freesurf  (integr_continuity.F:220)
    #    etaH = etaN + (1-β)·dEtaHdt·dt_freesurf   (update_etah.F:64, only when β≠1)
    #         = etaH_prev + dEtaHdt·dt_freesurf     (full forward-Euler update)
    #    Uses POST-boundary velocities so OB transport is correct.
    dyG = grid.metrics.dyG_for(u_final.dtype, u_final.device).view(
        1, 1, grid.ny, grid.nx + 1,
    )
    # full v-face field (bit-equal on row grids).
    dxG = grid.metrics.dxG_for(v_final.dtype, v_final.device).view(
        1, 1, grid.ny + 1, grid.nx,
    )
    if grid.masks.hFacW is not None:
        (
            u_corrected_transport,
            v_corrected_transport,
        ) = depth_integrated_face_transport(
            u_final,
            v_final,
            layer_thickness,
            dyG,
            dxG,
            hFacW=hFacW,
            hFacS=hFacS,
        )
    else:
        (
            u_corrected_transport,
            v_corrected_transport,
        ) = depth_integrated_face_transport(
            u_final,
            v_final,
            layer_thickness,
            dyG,
            dxG,
        )

    # Zero transport at faces belonging to OB-strip cells before
    # computing divergence.  MITgcm integr_continuity.F L127 uses
    # maskC(i,j,k) inside the divergence loop, which implicitly
    # zeros the contribution from OB-strip cells.  Without this
    # masking, the divergence telescope over interior cells
    # (maskInC=1) picks up transport through shared faces with
    # OB-strip cells (maskInC=0), producing a non-zero residual
    # that drives spurious η drift (~+9e-5 m/step).
    if grid.boundary_topology.has_open_boundaries:
        maskInC, maskInW, maskInS = _compute_interior_mask(
            grid, u_corrected_transport.dtype,
            u_corrected_transport.device,
        )
        u_corrected_transport = u_corrected_transport * maskInW.unsqueeze(0)
        v_corrected_transport = v_corrected_transport * maskInS.unsqueeze(0)

    u_corrected_transport, v_corrected_transport = synchronize_regional_face_values(
        u_corrected_transport, v_corrected_transport, grid,
    )
    div_corrected = divergence_volume_transport_to_center_mit(
        u_corrected_transport, v_corrected_transport, grid,
    )
    # Mask divergence at OB cells to prevent η accumulation
    # MITgcm integr_continuity.F L196: dEtaHdt *= maskInC
    if grid.boundary_topology.has_open_boundaries:
        div_corrected = div_corrected * maskInC.unsqueeze(0)
    # dEtaHdt = -hDivFlow*recip_rA (positive = surface rises)
    return -div_corrected


def subtract_continuity_freshwater(
    d_eta_dt_times_dt, *, mass_to_r_unit, empmr_mass_flux, dt_freesurf,
) -> torch.Tensor:
    mass_to_r_unit_tensor = torch.as_tensor(
        mass_to_r_unit,
        dtype=d_eta_dt_times_dt.dtype,
        device=d_eta_dt_times_dt.device,
    )
    freshwater_d_eta_dt = mass_to_r_unit_tensor * empmr_mass_flux
    freshwater_d_eta_dt_times_dt = freshwater_d_eta_dt * torch.as_tensor(
        dt_freesurf,
        dtype=d_eta_dt_times_dt.dtype,
        device=d_eta_dt_times_dt.device,
    )
    d_eta_dt_times_dt = (
        d_eta_dt_times_dt - freshwater_d_eta_dt_times_dt
    )
    return d_eta_dt_times_dt


def advance_continuity_surface(etaH_prev, *, beta, d_eta_dt_times_dt) -> tuple:
    # etaN = etaH_prev + β·dEtaHdt·dt  (state.eta for next step)
    eta_final = etaH_prev + beta * d_eta_dt_times_dt
    # etaH = etaN + (1-β)·dEtaHdt·dt = etaH_prev + dEtaHdt·dt
    # (MITgcm: update_etah.F:54-68)
    if beta == 1.0:
        eta_H_new = eta_final
    else:
        eta_H_new = eta_final + (1.0 - beta) * d_eta_dt_times_dt
    return eta_final, eta_H_new


def continuity_freshwater_history(empmr_mass_flux, *, grid) -> torch.Tensor:
    # PmEpR = -EmPmR*maskInC over the whole tile (integr_continuity.F:
    # 173-178; EmPmR is masked by maskInC in external_forcing_surf.F:
    # 149-155).  Every forward step takes this branch: forward_step.F:807
    # advances myIter before INTEGR_CONTINUITY, so only the INITIALISE_VARIA
    # call sees myIter == nIter0.  The continuity value feeds the next
    # step's surface forcing under synchronous stepping; staggered stepping
    # applies the same expression to the current step's EmPmR
    # (external_forcing_surf.F:158-166).
    mask_in_c, _, _ = _compute_interior_mask(
        grid, empmr_mass_flux.dtype, empmr_mass_flux.device)
    return _carry(-(empmr_mass_flux * mask_in_c))


def regional_predictor_transport(u_star, v_star, grid, *, stream_trace) -> tuple:
    # 4. Compute predicted transport divergence
    #    MITgcm calc_div_ghat.F builds face volume transports with
    #    xA=dyG*drF*hFacW and yA=dxG*drF*hFacS, then divides by rA.
    #    Unified with the cube step body in
    #    step_segments.compute_predictor_divergence.  The metric
    #    views and the hFac cast stay here: the exactConserv block below
    #    reuses layer_thickness / hFacW / hFacS.
    layer_thickness = grid.layer_thickness.to(
        dtype=u_star.dtype, device=u_star.device,
    ).view(1, grid.nz, 1, 1)
    dyG = grid.metrics.dyG_for(u_star.dtype, u_star.device).view(
        1, 1, grid.ny, grid.nx + 1,
    )
    # full v-face field (bit-equal broadcast on row grids).
    dxG = grid.metrics.dxG_for(v_star.dtype, v_star.device).view(
        1, 1, grid.ny + 1, grid.nx,
    )

    if grid.masks.hFacW is not None:
        hFacW = grid.masks.hFacW.to(dtype=u_star.dtype, device=u_star.device)  # [nz, ny, nx+1]
        hFacS = grid.masks.hFacS.to(dtype=v_star.dtype, device=v_star.device)  # [nz, ny+1, nx]
    else:
        hFacW = None
        hFacS = None

    predicted = compute_predictor_divergence(
        u_star,
        v_star,
        grid,
        stream_trace=stream_trace,
        layer_thickness=layer_thickness,
        dyG=dyG,
        dxG=dxG,
        hFacW=hFacW,
        hFacS=hFacS,
    )
    div_transport_star = predicted.divergence  # [batch, ny, nx]

    return div_transport_star, layer_thickness, hFacW, hFacS


def prepare_surface_rhs(
    etaH_prev, empmr_mass_flux, *, grid,
) -> tuple:
    # 5. Assemble RHS of Helmholtz equation (MIT integrated form).
    # Conceptually [-α·β·g·dt_mom·dt_freesurf·∇·(H·∇) + I] η^{n+1} = etaH - β·dt_freesurf·div(U*).
    # We assemble the matrix in MIT integrated form (rA on diagonal, no /rA
    # on off-diagonal); the RHS therefore must also be multiplied by rA so
    # both sides scale consistently.  MITgcm: solve_for_pressure.F uses
    # etaH (not etaN) for the mass term when exactConserv=.TRUE.; β=1 ⇒
    # etaH=etaN so they collapse.
    # The pressure segment receives explicit etaH, masks and initial guess.
    # Match mask dtype/device to etaH_prev, the RHS's dtype/device in the
    # aligned single-dtype step.
    # Mask RHS at land cells (force η=0 there)
    surface_mask = None
    if grid.masks.has_3d_masks:
        surface_mask = grid.masks.center_float(
            dtype=etaH_prev.dtype, device=etaH_prev.device,
        )[0:1]  # [1, ny, nx]

    # Mask RHS and initial guess at OB cells
    # (MITgcm: cg2d_b *= maskInC in solve_for_pressure.F)
    # OB cells have aC=1 (trivial η=0 equation)
    maskInC_3d = None
    if grid.boundary_topology.open_x or grid.boundary_topology.open_y:
        maskInC, _, _ = _compute_interior_mask(
            grid, etaH_prev.dtype, etaH_prev.device,
        )
        maskInC_3d = maskInC.unsqueeze(0)  # [1, ny, nx]

    if empmr_mass_flux is not None:
        if surface_mask is not None:
            empmr_mass_flux = torch.where(
                surface_mask != 0,
                empmr_mass_flux,
                torch.zeros_like(empmr_mass_flux),
            )
        if maskInC_3d is not None:
            empmr_mass_flux = torch.where(
                maskInC_3d != 0,
                empmr_mass_flux,
                torch.zeros_like(empmr_mass_flux),
            )

    return surface_mask, maskInC_3d, empmr_mass_flux


def nh_free_surface_source(phi_nh_prev, *, eta_h, gravity, masks, ny, nx):
    # MITgcm multiplies by a precomputed reciprocal
    # (``recip_Bo = 1/gBaro``, ini_linear_phisurf.F:85), so form the
    # reciprocal first: ``x*(1/g)`` and ``x/g`` differ in the last bit.
    recip_bo = 1.0 / gravity
    nh_surface_source = surface_level_slice(phi_nh_prev, masks, ny=ny, nx=nx) * recip_bo
    # MITgcm ``solve_for_pressure.F:194-205``: under ``oldFreeSurfTerm``
    # (``= use3Dsolver .AND. .NOT.exactConserv``, ``:86``) the surface source
    # term carries a non-hydrostatic pressure half beside the η half,
    #
    #   -freeSurfFac·rA·deepFac2F(ks)/(ΔtMom·ΔtFS)·( etaN + phi_nh(ks)·recip_Bo )
    #
    # — both halves grouped inside one parenthesis before the rA scaling.
    # CG2D assembles that source term times ``-ΔtMom·ΔtFS``, so the pair
    # becomes ``rA·(eta_h + nh_surface_source)``: the same grouping, both
    # halves added before either meets rA, as in the Fortran.
    eta_source = eta_h + nh_surface_source
    return nh_surface_source, eta_source


def initial_continuity_surface(*, eta, beta, w_interface_to_down, dt_freesurf):
    """Initial etaH, including MITgcm's explicit barotropic divergence.

    INITIALISE_VARIA calls INTEGR_CONTINUITY before the first pressure solve.
    UPDATE_ETAH then adds (1-implicDiv2DFlow)*dEtaHdt*deltaTFreeSurf.
    The continuity diagnostic returns downward-positive w; its surface value
    is minus dEtaHdt, with the wet-cell and OBC interior masks already applied.
    """
    return eta + (1.0 - beta) * (-w_interface_to_down[:, 0]) * dt_freesurf
