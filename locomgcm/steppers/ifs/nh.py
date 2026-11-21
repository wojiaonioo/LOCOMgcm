"""Internal nonhydrostatic momentum, pressure sources and CG3D solve stages.

Interface diagnosis is LOCOMgcm downward-positive; the producer negates and
slices it to MIT upward-positive Nr-level wVel in m s-1. phi_nh is in
m2 s-2 and gW in m s-2. C/U/V tensors retain their existing dtype casts.
Presence flags preserve absent and present-None restart fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd import forward_ad

from locomgcm.accel.graph_exit import graph_exit
from locomgcm.accel.tensor_context import forward_mode_active
from locomgcm.dynamics.horizontal_viscosity import VARIABLE_VISCOSITY_SCHEMES
from locomgcm.dynamics.hydrostatic_pressure import HydrostaticPressureGradient
from locomgcm.geometry import StructuredCGrid
from locomgcm.solvers import CG3DSolver, assemble_cg3d_rhs, post_cg3d_separate_dphinh
from locomgcm.geometry.grid import _compute_interior_mask
from locomgcm.steppers.implicit_free_surface_cg2d import _calc_div_ghat_cg3d_seed

from . import surface as ifs_surface

from .carry import _carry


@dataclass(frozen=True)
class NHHistory:
    phi_nh: torch.Tensor | None = None
    gw_nm1: torch.Tensor | None = None
    w_vel: torch.Tensor | None = None
    phi_nh_present: bool = False
    gw_nm1_present: bool = False
    w_vel_present: bool = False


def encode_nh_history(history: NHHistory):
    payload = {}
    if history.phi_nh_present:
        payload["_cached_phi_nh"] = history.phi_nh
    if history.gw_nm1_present:
        payload["_cached_gwNm1"] = history.gw_nm1
    if history.w_vel_present:
        payload["_cached_w_vel_mit_for_pre_cg3d"] = history.w_vel
    return payload


def decode_nh_history(payload):
    return NHHistory(
        payload.get("_cached_phi_nh"), payload.get("_cached_gwNm1"),
        payload.get("_cached_w_vel_mit_for_pre_cg3d"),
        "_cached_phi_nh" in payload, "_cached_gwNm1" in payload,
        "_cached_w_vel_mit_for_pre_cg3d" in payload,
    )


def mit_w_velocity_at_levels(interface_w_down, *, nz):
    return -interface_w_down[:, :nz]


def prepare_nh_w_velocity(bounded, *, previous_w_vel, u_star, grid):
    cached_w_vel = previous_w_vel
    if cached_w_vel is None:
        from locomgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

        # Producer returns LOCOMgcm downward-positive interface w [B,nz+1,ny,nx].
        # NH kernels consume MIT upward-positive Nr-sized wVel, so negate
        # and slice off the bottom no-flux interface.
        _, w_iface_to_down, _ = diagnose_vertical_velocity(bounded, grid)
        w_vel_mit = _carry(mit_w_velocity_at_levels(w_iface_to_down, nz=grid.nz))
    else:
        # History stores MIT upward-positive Nr-sized wVel for CALC_GW,
        # TIMESTEP_WVEL, and PRE_CG3D.
        w_vel_mit = cached_w_vel.to(dtype=u_star.dtype, device=u_star.device)
    return w_vel_mit


def diagnostic_tensor(diagnostics, *names: str) -> torch.Tensor | None:
    for name in names:
        value = diagnostics.tensors.get(name)
        if value is not None:
            return value
    return None


def _module_scalar(modules, module_name: str, attr: str) -> float | None:
    for module in modules:
        if getattr(module, "name", None) != module_name:
            continue
        value = getattr(module, attr, None)
        if value is None and attr == "effective_viscosity":
            value = getattr(module, "viscosity", None)
        if value is not None:
            return float(value)
    return None


def _module_by_name(modules, module_name: str) -> object | None:
    for module in modules:
        if getattr(module, "name", None) == module_name:
            return module
    return None


def vertical_momentum_viscosity_inputs(
    *, modules, diagnostics, bounded, w_vel_mit, grid, kpp_fields, implicit_viscosity,
):
    horizontal_viscosity_module = _module_by_name(modules, "horizontal_viscosity")
    no_slip_sides_for_gw = True
    side_drag_factor_for_gw = 2.0
    if horizontal_viscosity_module is not None:
        no_slip_sides_for_gw = bool(
            getattr(horizontal_viscosity_module, "no_slip_sides", True)
        )
        side_drag_factor_for_gw = horizontal_viscosity_module.side_drag_factor

    viscAh_W_for_gw = diagnostic_tensor(diagnostics, "viscAh3d_00", "viscAh_W")
    if viscAh_W_for_gw is None:
        scheme = getattr(horizontal_viscosity_module, "viscosity_scheme", None)
        if scheme in VARIABLE_VISCOSITY_SCHEMES:
            if not hasattr(horizontal_viscosity_module, "viscah_w_for_nonhydrostatic"):
                raise RuntimeError(f"{scheme} HorizontalViscosity must provide viscAh_W for CALC_GW")
            # MIT `viscAh_W` is a viscosity field, not a wVel field;
            # it is derived from U/V strain and then consumed by
            # CALC_GW with MIT upward-positive w_vel_mit unchanged.
            viscAh_W_for_gw = horizontal_viscosity_module.viscah_w_for_nonhydrostatic(
                bounded,
                grid,
            ).to(dtype=w_vel_mit.dtype, device=w_vel_mit.device)
            diagnostics.set_tensor("viscAh_W", viscAh_W_for_gw)
            diagnostics.set_tensor("viscAh3d_00", viscAh_W_for_gw)
        else:
            viscosity = _module_scalar(modules, "horizontal_viscosity", "viscosity")
            # No horizontal viscosity module is a zero viscAh_W, not a
            # missing one.
            viscAh_W_for_gw = (
                torch.zeros_like(w_vel_mit)
                if viscosity is None
                else torch.full_like(w_vel_mit, float(viscosity))
            )
    # MITgcm hands CALC_GW the kappaRU/kappaRV of CALC_VISCOSITY
    # (dynamics.F:385-390, :645-649): viscArNr, raised by KPP_CALC_VISC
    # where KPP runs (kpp_calc_visc.F:35-49), at the Nr+1 interfaces,
    # the last one viscArNr(Nr) (calc_viscosity.F:66-72).  The KPP
    # fields are the ones the predictor's implicit viscosity uses.
    n_batch = w_vel_mit.shape[0]
    u_faces = (n_batch, grid.nz + 1, grid.ny, grid.nx + 1)
    v_faces = (n_batch, grid.nz + 1, grid.ny + 1, grid.nx)
    if kpp_fields is not None:
        background = float(implicit_viscosity)  # KPP's visc_arnr
        kappa_ru_for_gw = extend_vertical_viscosity(
            kpp_fields.kappa_ru,
            bottom_shape=(n_batch, 1, grid.ny, grid.nx + 1),
            background=background,
        )
        kappa_rv_for_gw = extend_vertical_viscosity(
            kpp_fields.kappa_rv,
            bottom_shape=(n_batch, 1, grid.ny + 1, grid.nx),
            background=background,
        )
    else:
        kappa_value = (
            implicit_viscosity
            if implicit_viscosity > 0.0
            else _module_scalar(modules, "vertical_mixing", "effective_viscosity")
        )
        # No vertical viscosity anywhere is a zero kappaRU, not a
        # missing one.
        kappa_ru_for_gw, kappa_rv_for_gw = uniform_vertical_viscosities(
            w_vel_mit,
            value=0.0 if kappa_value is None else float(kappa_value),
            u_faces=u_faces, v_faces=v_faces,
        )

    # MIT viscA4_W for the CALC_GW biharmonic w-dissipation (None when
    # the horizontal_viscosity module has no biharmonic active => del^4 of
    # w skipped, byte-identical to before).
    viscA4_W_for_gw = None
    if horizontal_viscosity_module is not None and hasattr(
        horizontal_viscosity_module, "visca4_w_for_nonhydrostatic"
    ):
        viscA4_W_for_gw = horizontal_viscosity_module.visca4_w_for_nonhydrostatic(
            bounded, grid
        )
        if viscA4_W_for_gw is not None:
            viscA4_W_for_gw = viscA4_W_for_gw.to(
                dtype=w_vel_mit.dtype, device=w_vel_mit.device
            )
    return (
        viscAh_W_for_gw, kappa_ru_for_gw, kappa_rv_for_gw, viscA4_W_for_gw,
        no_slip_sides_for_gw, side_drag_factor_for_gw,
    )


def extend_vertical_viscosity(viscosity, *, bottom_shape, background):
    return torch.cat(
        (viscosity, viscosity.new_full(bottom_shape, background)),
        dim=1,
    )


def uniform_vertical_viscosities(w_vel_mit, *, value, u_faces, v_faces):
    uniform = w_vel_mit.new_full((1, 1, 1, 1), value)
    return uniform.expand(u_faces), uniform.expand(v_faces)


def pressure_mass_to_r_unit(modules):
    """MITgcm's mass-to-r-unit factor as a host float (the nonhydrostatic
    pressure solve, the vertical mixing schemes' N^2, KPP and GGL90 surfaceForcingU/V).

    MITgcm sets ``mass2rUnit = recip_rhoConst`` in z coordinates
    (``ini_parms.F``); rhoConst is the pressure-gradient module's
    ``reference_density``, the density of its g/rho coefficient.  Other modules
    carry a ``reference_density`` of their own (vertical mixing, the equation of
    state) that need not equal rhoConst, so the factor is taken from the
    pressure-gradient module only; a model without one has no rhoConst and is
    refused.

    Called in the stepper's eager prepare phase (``prepare_step_host``), never in
    the traced step, so that the value is a host float before tracing starts.
    """
    pressure_gradient = next(
        (module for module in modules if isinstance(module, HydrostaticPressureGradient)),
        None,
    )
    if pressure_gradient is None:
        raise ValueError(
            "mass2rUnit = 1/rhoConst (nonhydrostatic solve, vertical mixing N^2, "
            "KPP/GGL90 surface stress) needs rhoConst, "
            "the reference_density of a hydrostatic_pressure_gradient module, and the "
            "model has none"
        )
    return 1.0 / float(pressure_gradient.reference_density)


def refuse_real_freshwater_pressure_term(modules, grid, *, dtype, device):
    """Refuse a step whose EmPmR would reach the nonhydrostatic pressure solve.

    Under ``useRealFreshWaterFlux`` MITgcm's PRE_CG3D adds
    ``EmPmR*mass2rUnit*maskInC`` to ``wSurfP2d``, or to the top level of
    ``cg3d_b`` (``pre_cg3d.F:88-107``).  ``assemble_internal_pressure_rhs``
    does not implement that term.  It and its derivative vanish only when
    EmPmR is zero on every wet surface cell and is not differentiated, so a
    step is refused when this step's EmPmR is nonzero on a wet cell, requires
    grad, or carries a forward-mode tangent (a zero value with a derivative
    still drops the term's derivative).  ``requires_grad`` is read with grad
    mode on, so a no-grad caller (a checkpoint window's primal sweep) sees the
    same activity as the differentiated step.

    The EmPmR read is the one the step consumes: ``real_freshwater_mass_flux``
    after prepare installed this step's forcing sample, i.e. the volume flux of
    a ``wpflux_surface_forcing`` module.  The ERA5 bulk and sea-ice freshwater
    fluxes do not reach this entry; with them alone the free-surface solve
    raises for the missing EmPmR (``_solve_free_surface``).  Called in the
    stepper's eager prepare phase, once per step, on each rank's own tile.
    """
    with torch.enable_grad():
        empmr, _ = ifs_surface.real_freshwater_mass_flux(modules, dtype=dtype, device=device)
    if empmr is None:
        return
    wet = grid.masks.maskInC
    if wet is None:
        center = grid.masks.center
        wet = center.any(dim=0) if center.ndim == 3 else center
    wet = wet.to(device=empmr.device) != 0
    tangent = forward_ad.unpack_dual(empmr).tangent if forward_mode_active() else None
    if empmr.requires_grad or tangent is not None:
        reason = "a differentiated EmPmR (requires_grad or a forward-mode tangent)"
    elif bool(((empmr.detach() != 0) & wet).any()):
        reason = "a nonzero EmPmR on wet cells"
    else:
        return
    raise NotImplementedError(
        f"nonhydrostatic pressure solve with use_real_freshwater_flux=True and {reason}: "
        "the EmPmR*mass2rUnit term of MITgcm's PRE_CG3D (pre_cg3d.F:88-107) is not "
        "implemented, and running without it would drop the term and its derivative "
        "silently. Turn off the real freshwater flux (use_real_freshwater_flux=False) "
        "or run hydrostatic."
    )


def old_free_surface_increment(area, eta_h_prev, nh_surface_source, *, dt_mom, dt_freesurf):
    return -(area / dt_mom / dt_freesurf) * (eta_h_prev + nh_surface_source)


def prepare_internal_pressure(
    *, u_star, v_star, grid, dt_mom, implicit_div_2d_flow, template,
    phi_nh_history, mass_to_r_unit, gravity, eta_new,
):
    maskInC_for_nh = None
    if grid.boundary_topology.open_x or grid.boundary_topology.open_y:
        maskInC_for_nh, _, _ = _compute_interior_mask(
            grid,
            template.dtype,
            template.device,
        )
    cg3d_b_init = _calc_div_ghat_cg3d_seed(
        u_star,
        v_star,
        grid,
        dt_mom=dt_mom,
        implic_div_2d_flow=implicit_div_2d_flow,
        mask_in_c=maskInC_for_nh,
    )
    phi_nh_init = phi_nh_history
    if phi_nh_init is None:
        phi_nh_init = torch.zeros_like(template)
    else:
        phi_nh_init = phi_nh_init.to(dtype=template.dtype, device=template.device)
    cg2d_x_solution = gravity * eta_new
    return cg3d_b_init, phi_nh_init, mass_to_r_unit, cg2d_x_solution, maskInC_for_nh


def add_old_free_surface_source(
    *, cg3d_b_init, grid, eta_h_prev, nh_surface_source, dt_mom, dt_freesurf,
):
    # MITgcm ``solve_for_pressure.F:205-209`` -- the twin of the cg2d
    # source term at ``:200-204``, added to ``cg3d_b(ks)`` in the same
    # DO loop, under the same ``oldFreeSurfTerm`` gate (``:85``):
    #
    #   -freeSurfFac*rA*deepFac2F(ks)/deltaTMom/deltaTFreeSurf
    #     *( etaN + phi_nh(ks)*recip_Bo )
    #
    # MITgcm adds it before CG2D, so its ``etaN`` is the *previous*
    # step's.  LOCOMgcm assembles cg3d_b_init after the cg2d solve, so it
    # must use the saved value -- and ``oldFreeSurfTerm`` implies
    # ``.NOT.exactConserv``, under which etaH is kept equal to etaN
    # (see the exactConserv branch below), so ``etaH_prev`` *is* that
    # previous etaN.  ``nh_surface_source`` is the same
    # ``phi_nh(ks)*recip_Bo`` slice the cg2d RHS used, so the two
    # halves stay grouped as one parenthesis exactly as in the Fortran.
    rA_nh = grid.metrics.cell_area_for(
        eta_h_prev.dtype, eta_h_prev.device,
    ).unsqueeze(0)
    cg3d_b_init = ifs_surface.add_at_surface_level(
        cg3d_b_init,
        old_free_surface_increment(
            rA_nh, eta_h_prev, nh_surface_source,
            dt_mom=dt_mom, dt_freesurf=dt_freesurf,
        ),
        grid.masks,
        nz=grid.nz,
    )
    return cg3d_b_init


def assemble_internal_pressure_rhs(
    *, cg3d_b_init, cg2d_x_solution, eta_new, etaH_prev,
    w_vel_new_mit_for_cg3d, phi_nh_init, grid, bundle, params,
    old_free_surf_term, mass_to_r_unit_value, implicit_surf_press,
    implicit_div_2d_flow, template, maskInC_for_nh,
):
    # PRE_CG3D consumes MIT upward-positive transient wVel; no sign flip.
    cg3d_b, phi_nh_init_masked = assemble_cg3d_rhs(
        cg3d_b_init=cg3d_b_init,
        cg2d_x=cg2d_x_solution,
        eta_n=eta_new,
        eta_h=etaH_prev,
        w_vel_mit_from_timestep_wvel=w_vel_new_mit_for_cg3d,
        phi_nh_init=phi_nh_init,
        # No EmPmR*mass2rUnit term (pre_cg3d.F:88-107): a step whose EmPmR is
        # nonzero on a wet cell or differentiated is refused in prepare
        # (refuse_real_freshwater_pressure_term), so here the wpflux EmPmR is
        # None or a constant zero.
        empmr=None,
        grid=grid,
        time_scales=bundle,
        params=params,
        old_free_surf_term=old_free_surf_term,
        using_p_coords=False,
        use_real_freshwater_flux=False,
        fluid_is_water=True,
        mass_to_r_unit=mass_to_r_unit_value,
        # The same factors the CG2D solve and the CG3D seed use; the
        # constructor keeps them at 1 for the nonhydrostatic chain.
        implicit_surf_press=implicit_surf_press,
        implic_div_2d_flow=implicit_div_2d_flow,
    )
    nh_mask = (
        grid.masks.center_float(dtype=template.dtype, device=template.device)
        if grid.masks.has_3d_masks
        else torch.ones(
            (grid.nz, grid.ny, grid.nx),
            dtype=template.dtype,
            device=template.device,
        )
    )
    if maskInC_for_nh is not None:
        nh_mask = nh_mask * maskInC_for_nh.unsqueeze(0)
    return cg3d_b, phi_nh_init_masked, nh_mask


def solve_internal_pressure(
    *, cg3d_b, phi_nh_init_masked, operator, preconditioner, nh_mask, grid,
    bundle, params, tolerance_mode, exact_conserv,
):
    with graph_exit():  # The cg3d host loop
        (
            phi_nh_new,
            nh_iters,
            nh_first_residual,
            nh_last_residual,
        ) = _solve_cg3d_and_stage(
            CG3DSolver(
                params=params,
                time_scales=bundle,
            ),
            cg3d_b,
            phi_nh_init_masked,
            operator,
            preconditioner,
            nh_mask,
            grid=grid,
            tolerance_mode=tolerance_mode,
        )
    phi_nh_history = _carry(phi_nh_new)
    dphi_nh = post_cg3d_separate_dphinh(
        phi_nh_new,
        grid,
        params,
        exact_conserv=exact_conserv,
    ).detach()
    return phi_nh_new, phi_nh_history, dphi_nh, nh_iters, nh_first_residual, nh_last_residual


@torch._dynamo.disable
def _solve_cg3d_and_stage(
    solver: Any,
    cg3d_b: torch.Tensor,
    x0: torch.Tensor,
    operator: Any,
    preconditioner: Any,
    mask: torch.Tensor,
    *,
    grid: StructuredCGrid,
    tolerance_mode: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The internal non-hydrostatic solve, whole, as a declared exit.

    ``_solve_cg2d_and_stage``'s twin.  The cg3d PCG
    loop tests convergence on the host, so its iteration count and its two
    residuals are Python scalars, and Dynamo resumes *inside* the loop's break
    with them on the resumed frame stack, where changing scalar values can
    invalidate guards. The whole call is the exit; what comes back is four
    tensors, and the three diagnostics leave in ``StepOutcome`` as 0-d tensors
    so the traced step stores no per-step Python scalar.

    Argument order and values are those of the inline call; only the frame moved.
    """
    outcome = solver.solve(
        cg3d_b,
        x0,
        operator,
        preconditioner,
        mask,
        grid=grid,
        tolerance_mode=tolerance_mode,
        root_adjoint=True,
    )
    return (
        outcome.phi_nh,
        torch.tensor(int(outcome.num_iters), dtype=torch.int64, device="cpu"),
        torch.tensor(float(outcome.first_residual), dtype=torch.float64, device="cpu"),
        torch.tensor(float(outcome.last_residual), dtype=torch.float64, device="cpu"),
    )
