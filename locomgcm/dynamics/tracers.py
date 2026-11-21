from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from locomgcm.geometry.metric_bundle import _positive_reciprocal

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.grid import _compute_interior_mask
from locomgcm.geometry.metric_bundle import build_tracer_metric_bundle
from locomgcm.operators import (
    TracerFluxScheme,
    horizontal_tracer_flux_divergence,
    vertical_flux_divergence,
)
from locomgcm.operators.c_grid import (
    tracer_flux_divergence_to_center,
    divergence_volume_transport_to_center_mit,
    horizontal_volume_divergence_from_transports,
    meridional_face_difference,
    zonal_face_difference,
)
from locomgcm.operators.fluxes.tracer_flux import (
    HORIZONTAL_STENCIL_SCHEMES,
    HORIZONTAL_TRACER_SCHEMES,
    VERTICAL_CFL_SCHEMES,
    VERTICAL_VELOCITY_FLUX_SCHEMES,
    VERTICAL_TRACER_SCHEMES,
    _stencil_face_flux_x,
    _stencil_face_flux_y,
    _two_point_face_flux_x,
    _two_point_face_flux_y,
    interface_recip_drC,
    tracer_scheme_spec,
    vertical_tracer_flux,
)
from locomgcm.operators.fluxes.tracer_reconstruction import first_order_upwind_flux
from locomgcm.operators.reconstructions import (
    center_to_u_upwind,
    center_to_v_upwind,
)
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyPhase, TendencyStepDomain
from locomgcm.dynamics._registry import register_dynamics_module


def horizontal_tracer_advection_from_volume_transports(
    tracer: torch.Tensor,
    velocity_u: torch.Tensor,
    velocity_v: torch.Tensor,
    u_volume_transport: torch.Tensor,
    v_volume_transport: torch.Tensor,
    grid: StructuredCGrid,
    *,
    recip_hFacC: torch.Tensor,
    layer_thickness: torch.Tensor,
) -> torch.Tensor:
    """MIT volume-transport upwind horizontal tracer tendency.

    The caller supplies face velocities for donor selection and pre-weighted
    face volume transports. The kernel reconstructs tracer values at U/V faces
    through the grid's exchange backend, computes the MIT-style volume transport
    divergence, then divides by hFac and layer thickness.
    """
    f_u = center_to_u_upwind(tracer, velocity_u, grid)
    f_v = center_to_v_upwind(tracer, velocity_v, grid)
    div = divergence_volume_transport_to_center_mit(
        u_volume_transport * f_u,
        v_volume_transport * f_v,
        grid,
    )
    return -div * recip_hFacC / layer_thickness


def _calc_adv_flow_transports_mit_order(
    u: torch.Tensor,
    v: torch.Tensor,
    dyG: torch.Tensor,
    dxG: torch.Tensor,
    drF: torch.Tensor,
    *,
    hFacW: torch.Tensor | None = None,
    hFacS: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build tracer face transports in MITgcm ``calc_adv_flow`` order."""
    u_trans = u * dyG * drF
    if hFacW is not None:
        u_trans = u_trans * hFacW

    v_trans = v * dxG * drF
    if hFacS is not None:
        v_trans = v_trans * hFacS

    return u_trans, v_trans


def _combined_advective_tendency_from_fluxes(
    tracer: torch.Tensor,
    h_flux_div: torch.Tensor,
    h_transport_div: torch.Tensor,
    v_flux: torch.Tensor,
    r_trans: torch.Tensor,
    mc: torch.Tensor,
    drF: torch.Tensor,
    rA: torch.Tensor,
) -> torch.Tensor:
    """Combined H+V advective tendency of ``gad_calc_rhs.F:769-781`` from face fluxes.

    ``h_flux_div``/``h_transport_div`` are the horizontal divergences of the
    tracer and volume transports, ``v_flux``/``r_trans`` the interface tracer
    flux and volume transport (LOCOMgcm downward-positive), ``mc`` the interior
    mask ``maskInC`` broadcastable to the cells; ``drF``/``rA`` the cell
    thickness and area.  ``recip_hFacC`` is left to the caller.
    """
    v_flux = v_flux * mc

    # MITgcm: (fVerT_kDown - fVerT_kUp)*rkSign, upward-positive
    # LOCOMgcm vertical_flux_divergence: (F[k]-F[k+1])/dz, downward-positive
    # These have opposite signs, so negate to match MITgcm's convention.
    dz_1d = drF.squeeze()
    v_flux_div_raw = -(vertical_flux_divergence(v_flux, dz_1d) * drF)

    # MITgcm: (rTransKp1 - rTrans)*rAdvFac = (bot-top)*(-1) in upward-pos
    # LOCOMgcm: (top-bot)/dz in downward-pos — opposite sign, so negate.
    # MITgcm calc_adv_flow.F:112-120: rTrans=0 at the surface interface.
    rTrans_div = _surface_zeroed_vertical_transport(r_trans)
    v_transport_div_raw = -(vertical_flux_divergence(rTrans_div, dz_1d) * drF)

    # --- Combined gad_calc_rhs.F:769-781 ---
    # All quantities now in "raw" units (tracer * m³/s or m³/s)
    # maskInC applied to H terms and correction, NOT to V flux div
    h_part = h_flux_div * mc
    correction = tracer * (
        h_transport_div * mc
        + v_transport_div_raw * mc
    )

    raw_total = h_part + v_flux_div_raw - correction

    # Divide by volume: -recip_hFacC * recip_drF * recip_rA
    # (recip_hFacC applied in evaluate(), so just /drF/rA here)
    return -raw_total / (drF * rA)


def _advective_form_tendency_from_divergences(
    tracer: torch.Tensor,
    flux_div: torch.Tensor,
    div_transport: torch.Tensor,
    maskInC: torch.Tensor | None = None,
) -> torch.Tensor:
    """MIT ``gad_calc_rhs`` advective-form correction from divergence terms."""
    if maskInC is not None:
        if flux_div.ndim >= 4:
            mc = maskInC.unsqueeze(0).unsqueeze(0)
        else:
            mc = maskInC.unsqueeze(0)
        flux_div = flux_div * mc
        div_transport = div_transport * mc
    return -(flux_div - tracer * div_transport)


def _multidim_advective_pass_update(
    tracer_orig: torch.Tensor,
    div_flux: torch.Tensor,
    div_transport: torch.Tensor,
    recip_local_volume: torch.Tensor,
    maskInC: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT ``gad_advection`` pass update and advective correction."""
    correction = tracer_orig * div_transport
    update = (div_flux - correction) * recip_local_volume
    if maskInC is not None:
        update = update * maskInC
    return update, correction


def _surface_zeroed_vertical_transport(w_interface: torch.Tensor) -> torch.Tensor:
    """Zero the surface interface; expects LOCOMgcm downward-positive w transport."""
    return torch.cat(
        (
            torch.zeros_like(w_interface[..., :1, :, :]),
            w_interface[..., 1:, :, :],
        ),
        dim=-3,
    )


def _multidim_vertical_pass_update(
    tracer_orig: torch.Tensor,
    div_flux: torch.Tensor,
    div_transport: torch.Tensor,
    *,
    maskInC: torch.Tensor | None = None,
    recip_hFacC: torch.Tensor | None = None,
    squeeze_leading_batch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT ``gad_advection`` R-pass update and advective correction."""
    correction = tracer_orig * div_transport
    update = div_flux - correction
    if squeeze_leading_batch:
        update = update.squeeze(0)
        correction = correction.squeeze(0)
    if maskInC is not None:
        update = update * maskInC
    if recip_hFacC is not None:
        update = update * recip_hFacC
    return update, correction


def _obcs_upwind_x(
    flux_u: torch.Tensor,
    transport_u: torch.Tensor,
    tracer: torch.Tensor,
    grid: StructuredCGrid,
    maskInC: torch.Tensor,
    hFacW: torch.Tensor | None,
) -> torch.Tensor:
    """Replace X-direction flux at OBC faces with 1st-order upwind."""
    nx = grid.nx
    mL = maskInC[:, :-1]  # [ny, nx-1]
    mR = maskInC[:, 1:]
    ob_face_mask = (mL != mR)
    # No early return on ``ob_face_mask.any()`` : the masked
    # ``torch.where`` below leaves every non-OBC face unchanged, so the
    # unconditional spelling is bit-neutral and has no host branch.

    T_left = tracer[..., :-1]
    T_right = tracer[..., 1:]
    u_face = transport_u[..., 1:nx]
    upwind_flux = first_order_upwind_flux(u_face, T_left, T_right)

    if flux_u.ndim >= 4:
        mask_expand = ob_face_mask.unsqueeze(0).unsqueeze(0)
    else:
        mask_expand = ob_face_mask.unsqueeze(0)
    if hFacW is not None:
        wet_face = hFacW[:, :, 1:nx] > 0
        if flux_u.ndim >= 4:
            wet_face = wet_face.unsqueeze(0)
        mask_expand = mask_expand & wet_face

    flux_u = flux_u.clone()
    flux_u[..., 1:nx] = torch.where(mask_expand, upwind_flux, flux_u[..., 1:nx])
    return flux_u


def _obcs_upwind_y(
    flux_v: torch.Tensor,
    transport_v: torch.Tensor,
    tracer: torch.Tensor,
    grid: StructuredCGrid,
    maskInC: torch.Tensor,
    hFacS: torch.Tensor | None,
) -> torch.Tensor:
    """Replace Y-direction flux at OBC faces with 1st-order upwind."""
    ny = grid.ny
    mB = maskInC[:-1, :]
    mT = maskInC[1:, :]
    ob_face_mask = (mB != mT)
    # No early return on ``ob_face_mask.any()`` : the masked
    # ``torch.where`` below leaves every non-OBC face unchanged, so the
    # unconditional spelling is bit-neutral and has no host branch.

    T_bot = tracer[..., :-1, :]
    T_top = tracer[..., 1:, :]
    v_face = transport_v[..., 1:ny, :]
    upwind_flux = first_order_upwind_flux(v_face, T_bot, T_top)

    if flux_v.ndim >= 4:
        mask_expand = ob_face_mask.unsqueeze(0).unsqueeze(0)
    else:
        mask_expand = ob_face_mask.unsqueeze(0)
    if hFacS is not None:
        wet_face = hFacS[:, 1:ny, :] > 0
        if flux_v.ndim >= 4:
            wet_face = wet_face.unsqueeze(0)
        mask_expand = mask_expand & wet_face

    flux_v = flux_v.clone()
    flux_v[..., 1:ny, :] = torch.where(mask_expand, upwind_flux, flux_v[..., 1:ny, :])
    return flux_v


def _diagnostic_obc_tracer(
    diagnostics: DiagnosticState,
    *,
    tracer_name: str,
    side: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    value = diagnostics.tensors.get(f"{tracer_name}_obc_{side}")
    if value is None:
        return None
    return value.to(dtype=dtype, device=device)


def _obc_tracer_kwargs(
    diagnostics: DiagnosticState,
    *,
    tracer_name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    return {
        "obc_west_tracer": _diagnostic_obc_tracer(
            diagnostics,
            tracer_name=tracer_name,
            side="west",
            dtype=dtype,
            device=device,
        ),
        "obc_east_tracer": _diagnostic_obc_tracer(
            diagnostics,
            tracer_name=tracer_name,
            side="east",
            dtype=dtype,
            device=device,
        ),
        "obc_south_tracer": _diagnostic_obc_tracer(
            diagnostics,
            tracer_name=tracer_name,
            side="south",
            dtype=dtype,
            device=device,
        ),
        "obc_north_tracer": _diagnostic_obc_tracer(
            diagnostics,
            tracer_name=tracer_name,
            side="north",
            dtype=dtype,
            device=device,
        ),
    }


_MULTIDIM_DEBUG_STATE = {
    "step": 0,
    "enabled": False,
    "log": [],
}

# Per-tensor capture hook for bitwise compare vs MITgcm GAD_ADVECTION.
# When enabled, _multidim_advective_tendency writes full intermediate
# tensors (localT after X-pass, div_flux_x, update_x, final tendency)
# into _MULTIDIM_CAPTURE["tensors"].  Enable only for the ctypes
# per-operator compare driver; production runs leave it off.
_MULTIDIM_CAPTURE: dict = {
    "enabled": False,
    "tensors": {},
}

_MULTIDIM_LEAF_INJECT: dict = {
    "enabled": False,
    "leaves": {},
}


@dataclass(frozen=True)
@register_dynamics_module("horizontal_tracer_advection")
class HorizontalTracerAdvection(TendencyModule):
    tracer_names: tuple[str, ...] = ("temp", "salt")
    scheme: TracerFluxScheme = TracerFluxScheme.UPWIND
    # Vertical R-pass scheme, separate from the horizontal ``scheme``.  Default
    # UPWIND preserves the historical hardcoded R-pass. CENTERED reuses the
    # existing second-order vertical flux; DST3_FL selects the limiter kernel.
    vertical_scheme: TracerFluxScheme = TracerFluxScheme.UPWIND
    name: str = "horizontal_tracer_advection"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER
    velocity_u_override: torch.Tensor | None = None
    velocity_v_override: torch.Tensor | None = None
    use_advective_form: bool = False
    obcs_upwind: bool = True
    multi_dim: bool = False
    dt: float = 0.0
    combined_with_vtt: bool = False
    # GM_AdvForm: advect with uFld vFld wFld, the Eulerian velocity plus the GM
    # bolus velocity (thermodynamics.F:258-276), which the gmredi module publishes
    # as gm_bolus_u/v/w; set from the gmredi entry (_canonical_helpers.enrich_manifest).
    gm_residual_flow: bool = False

    def __post_init__(self) -> None:
        if self.scheme not in HORIZONTAL_TRACER_SCHEMES:
            raise ValueError(
                "HorizontalTracerAdvection supports scheme in "
                f"{tuple(s.value for s in HORIZONTAL_TRACER_SCHEMES)}, got {self.scheme!r}"
            )
        if self.vertical_scheme not in VERTICAL_TRACER_SCHEMES:
            raise ValueError(
                "HorizontalTracerAdvection supports vertical_scheme in "
                f"{tuple(s.value for s in VERTICAL_TRACER_SCHEMES)}, got {self.vertical_scheme!r}"
            )
        # PPM and PQM exist only in MITgcm's multi-dimensional sweep; its
        # single-RHS path stops on them (gad_calc_rhs.F:295-296, :424-425).
        for scheme in (self.scheme, self.vertical_scheme):
            if not tracer_scheme_spec(scheme).single_rhs and not self._runs_multi_dim():
                raise ValueError(
                    f"HorizontalTracerAdvection scheme {scheme.value!r} runs only in the "
                    "multi-dimensional sweep: it needs multi_dim=True, dt>0 and a "
                    f"horizontal scheme without Adams-Bashforth, got scheme={self.scheme.value!r}, "
                    f"multi_dim={self.multi_dim}, dt={self.dt}"
                )

    def _runs_multi_dim(self) -> bool:
        """Whether this module takes the multi-dimensional path (MITgcm ``tempMultiDimAdvec``).

        ``multi_dim`` with a positive tracer step, forced off for a horizontal
        scheme that declares ``adams_bashforth`` (``gad_init_fixed.F:126-137``).
        Such a tracer runs X, Y and R on the single-RHS path instead
        (``temp_integrate.F:289``, ``GAD_CALC_RHS``), so the module stays its R
        consumer and takes the combined path.
        """
        return (
            self.multi_dim
            and self.dt > 0
            and not tracer_scheme_spec(self.scheme).adams_bashforth
        )

    def effective_velocity(
        self,
        state: PrognosticState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.velocity_u_override if self.velocity_u_override is not None else state.u.data
        v = self.velocity_v_override if self.velocity_v_override is not None else state.v.data
        return u, v

    def _tracer_velocity(
        self, state: PrognosticState, diagnostics: DiagnosticState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(uFld, vFld)``: ``effective_velocity`` plus the bolus velocity under GM_AdvForm."""
        u, v = self.effective_velocity(state)
        if self.gm_residual_flow:
            u = u + diagnostics.tensors["gm_bolus_u"]
            v = v + diagnostics.tensors["gm_bolus_v"]
        return u, v

    def _tracer_w_interface(
        self, state: PrognosticState, diagnostics: DiagnosticState, grid: StructuredCGrid,
    ) -> torch.Tensor:
        """Downward-positive interface ``w`` of ``wFld``: ``-(wVel + w_bolus)`` under GM_AdvForm
        (the bolus ``w`` is MIT upward-positive, ``docs/NUMERICS.md``)."""
        from locomgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

        _, w_interface, _ = diagnose_vertical_velocity(state, grid)
        if self.gm_residual_flow:
            w_interface = w_interface - diagnostics.tensors["gm_bolus_w"]
        return w_interface

    def _advective_tendency(
        self,
        tracer: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
        w_interface: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute advective tracer tendency matching MITgcm ``gad_calc_rhs.F``.

        MITgcm computes (per layer, for one tracer):

          gTracer -= recip_hFacC * recip_drF * recip_rA * (
              div(tracer_flux) * maskInC
              - T * div(uTrans) * maskInC  # <-- flux-to-advective correction
          )

        The ``-T * div(uTrans)`` term converts the flux-form divergence
        ``div(u*T)`` to the advective form ``u·∇T``.  For an incompressible
        interior where ``div(u)≈0`` both forms are equivalent, but near open
        boundaries the local ``div(uTrans) ≠ 0`` so the correction matters.

        Additionally, at OBC faces MITgcm replaces the interior-scheme flux
        with 1st-order upwind (``OBCS_U1_ADV_TRACER``).
        """
        has_obcs = grid.boundary_topology.open_x or grid.boundary_topology.open_y

        # 1. Compute flux divergence with OBC upwind override
        flux_div = horizontal_tracer_flux_divergence(
            tracer, u, v, grid,
            scheme=self.scheme,
            obcs_upwind=(has_obcs and self.obcs_upwind),
            # DST3 / DST3-FL need deltaTLev for their Courant number; the
            # donor-cell schemes ignore it.  MITgcm runs these schemes on this
            # single-RHS path whenever multiDimAdvection=.FALSE.
            # (gad_init_fixed.F:126-137 never forces multi-dim for 30/33).
            dt=self.dt,
        )

        # 2. Flux-to-advective form correction: subtract T * div(uTrans)
        #    MITgcm gad_calc_rhs.F:775-778 ALWAYS includes this term (advFac=1
        #    when calcAdvection=.TRUE.), not just for advective-form mode.
        #    Without it, non-zero div(transport) at OBC faces creates fictitious
        #    tracer sources that grow exponentially.
        if has_obcs:
            maskInC, _, _ = _compute_interior_mask(
                grid, tracer.dtype, tracer.device,
            )
        else:
            maskInC = None

        # Build per-layer volume transport = u * hFacW, v * hFacS
        u_trans = u
        v_trans = v
        if grid.masks.hFacW is not None and u.ndim >= 4:
            hFacW = grid.masks.hFacW.to(dtype=u.dtype, device=u.device)
            u_trans = u * hFacW
        if grid.masks.hFacS is not None and v.ndim >= 4:
            hFacS = grid.masks.hFacS.to(dtype=v.dtype, device=v.device)
            v_trans = v * hFacS

        # geometry-aware form (matches the w-diagnosis
        # transport/rA chain on spherical grids; quotient kept bitwise on
        # row-uniform grids).
        div_transport = tracer_flux_divergence_to_center(u_trans, v_trans, grid)
        if (
            self.use_advective_form
            and w_interface is not None
            and tracer.ndim >= 4
            and tracer.shape[1] > 1
        ):
            w_interface_transport = w_interface.to(dtype=tracer.dtype, device=tracer.device)
            if w_interface_transport.ndim == 3:
                w_interface_transport = w_interface_transport.unsqueeze(0)

            # Use w_interface directly (no mask_above). w is diagnosed from
            # div(u*dz*hFacW, v*dz*hFacS) via cumsum and is zero at dry
            # interfaces by construction. Masking breaks the exact
            # cancellation div_x+div_y+div_z=0 at partial cells.
            div_transport = div_transport - vertical_flux_divergence(
                w_interface_transport, grid.layer_thickness,
            )

        # MITgcm gad_calc_rhs.F:769-778 always applies:
        #   gTracer -= recip_vol * (div(flux) - T * div(transport))
        # The -T*div(transport) term converts flux-form to advective-form,
        # critical for stability when div(transport) != 0 (e.g. at OBC faces).
        return _advective_form_tendency_from_divergences(
            tracer,
            flux_div,
            div_transport,
            maskInC,
        )

    def _combined_advective_tendency(
        self,
        tracer: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
        w_interface: torch.Tensor,
    ) -> torch.Tensor:
        """Combined H+V advective tendency matching gad_calc_rhs.F:769-781.

        Computes horizontal and vertical advection in a single expression,
        summing all transport divergences inside one pair of parentheses
        before multiplying by T.  This eliminates the ~1.6e-9/step FP
        rounding difference that arises when VTT and HTA compute the
        correction term separately (catastrophic cancellation when
        div_h + div_v ≈ 0 due to incompressibility).

        The transport correction is scheme-independent.  Reconstruct the
        tracer fluxes with the selected horizontal and vertical schemes on
        the same tracer state, as in GAD_CALC_RHS.  ``w_interface`` uses LOCOMgcm's
        downward-positive velocity convention.
        """
        has_obcs = grid.boundary_topology.open_x or grid.boundary_topology.open_y

        if has_obcs:
            maskInC_2d, _, _ = _compute_interior_mask(
                grid, tracer.dtype, tracer.device,
            )
            mc = maskInC_2d.unsqueeze(0).unsqueeze(0) if tracer.ndim >= 4 else maskInC_2d.unsqueeze(0)
        else:
            # maskInC is 1 on every cell without open boundaries; MITgcm still
            # multiplies by it (gad_calc_rhs.F:583, :775-781), and a product with
            # 1 leaves every value unchanged.
            mc = torch.ones((), dtype=tracer.dtype, device=tracer.device)

        wet_mask = None
        if grid.masks.hFacC is not None:
            wet_mask = (grid.masks.hFacC > 0).to(dtype=tracer.dtype)

        _lead = max(tracer.ndim - 2, 0)
        if grid.metrics.del_x is None:
            dxC = grid.metrics.dxC_row_for(tracer.dtype, tracer.device).view(
                *([1] * _lead), grid.ny, 1
            )
            dxG_ext = grid.metrics.dxG_row_ext_for(
                tracer.dtype, tracer.device
            ).view(*([1] * _lead), grid.ny + 1, 1)
        else:
            dxC = grid.metrics.dxC_for(tracer.dtype, tracer.device).view(
                *([1] * _lead), grid.ny, grid.nx + 1
            )
            dxG_ext = grid.metrics.dxG_for(tracer.dtype, tracer.device).view(
                *([1] * _lead), grid.ny + 1, grid.nx
            )
        # rA is the true per-cell area on BOTH branches. `spherical_polar` leaves
        # del_x unset, so the row branch is the one every spherical case takes,
        # and the scalar grid.dx*grid.dy represents the
        # ny//2 row's area -- off by 1/cos(lat), a factor 4.8 by +-78 deg. That
        # is exactly what metrics.py warns against just above its `return cls(`:
        # "Flux-form operators should NOT rely on these for bitwise alignment;
        # they should consume dxG_row / cell_area instead." On uniform Cartesian
        # grids rA is bitwise equal to dx*dy, so this is a no-op there.
        rA = grid.metrics.rA_for(tracer.dtype, tracer.device).view(
            *([1] * _lead), grid.ny, grid.nx
        )
        dy = grid.metrics.dyG_for(tracer.dtype, tracer.device).view(
            *([1] * _lead), grid.ny, grid.nx + 1)
        dyG = dy
        drF = grid.layer_thickness_for(tracer.dtype, tracer.device).view(
            *([1] * max(tracer.ndim - 3, 0)), grid.nz, 1, 1)

        hFacW = grid.masks.hFacW.to(dtype=tracer.dtype, device=tracer.device) if grid.masks.hFacW is not None else None
        hFacS = grid.masks.hFacS.to(dtype=tracer.dtype, device=tracer.device) if grid.masks.hFacS is not None else None

        # MITgcm calc_adv_flow: face volume transports for tracer advection.
        uTrans, vTrans = _calc_adv_flow_transports_mit_order(
            u,
            v,
            dyG,
            dxG_ext,
            drF,
            hFacW=hFacW,
            hFacS=hFacS,
        )

        # Every scheme consumes these same preweighted volume transports, so
        # hFac and face area are applied exactly once.  The two-point schemes
        # take the region two-point adapter, whose one-cell exchange (OBC_VALUE
        # without open-boundary values) equals a topology WRAP/REPLICATE fill:
        # a periodic axis wraps, or reaches its wrap partner rank, under either
        # fill; any other edge replicates.
        if tracer_scheme_spec(self.scheme).needs_dt and not self.dt > 0:
            raise ValueError(f"scheme={self.scheme} requires dt>0 for its Courant number")
        if self.scheme in HORIZONTAL_STENCIL_SCHEMES:
            fZon = _stencil_face_flux_x(
                tracer, u, uTrans, grid, dt=self.dt,
                scheme=self.scheme,
            )
            fMer = _stencil_face_flux_y(
                tracer, v, vTrans, grid, dt=self.dt,
                scheme=self.scheme,
            )
        else:
            fZon = _two_point_face_flux_x(
                tracer, uTrans, u, grid,
                scheme=self.scheme, dt=self.dt, obc_west_tracer=None, obc_east_tracer=None,
            )
            fMer = _two_point_face_flux_y(
                tracer, vTrans, v, grid,
                scheme=self.scheme, dt=self.dt, obc_south_tracer=None, obc_north_tracer=None,
            )

        if has_obcs and self.obcs_upwind:
            if grid.boundary_topology.open_x:
                fZon = _obcs_upwind_x(fZon, uTrans, tracer, grid, maskInC_2d, hFacW)
            if grid.boundary_topology.open_y:
                fMer = _obcs_upwind_y(fMer, vTrans, tracer, grid, maskInC_2d, hFacS)

        h_flux_div = horizontal_volume_divergence_from_transports(fZon, fMer)
        h_transport_div = horizontal_volume_divergence_from_transports(uTrans, vTrans)

        # --- Vertical flux (transport-space) ---
        rTrans = w_interface * rA
        if self.vertical_scheme in VERTICAL_CFL_SCHEMES:
            if not self.dt > 0:
                raise ValueError(
                    f"vertical_scheme={self.vertical_scheme} requires dt>0 for its Courant number"
                )
            recip_drC = interface_recip_drC(grid, tracer.dtype, tracer.device)
            if self.vertical_scheme in VERTICAL_VELOCITY_FLUX_SCHEMES:
                # The DST3 kernels weight the flux by the velocity.  Convert
                # their result to volume flux before the combined
                # divergence/correction.
                v_flux = vertical_tracer_flux(
                    self.vertical_scheme, tracer, None, velocity=w_interface,
                    recip_drC=recip_drC, dt=self.dt, mask=wet_mask, layer_thickness=None,
                    recip_center_spacing=None,
                ) * rA
            else:
                # MITgcm multiplies the bracket by rTrans and reads wFld only
                # for the Courant number (gad_calc_rhs.F:522-542).
                v_flux = vertical_tracer_flux(
                    self.vertical_scheme, tracer, rTrans, velocity=w_interface,
                    recip_drC=recip_drC, dt=self.dt, mask=wet_mask, layer_thickness=None,
                    recip_center_spacing=None,
                )
        else:
            v_flux = vertical_tracer_flux(
                self.vertical_scheme, tracer, rTrans, velocity=None,
                recip_drC=None, dt=None, mask=wet_mask, layer_thickness=None,
                recip_center_spacing=None,
            )
        return _combined_advective_tendency_from_fluxes(
            tracer, h_flux_div, h_transport_div, v_flux, rTrans, mc, drF, rA,
        )

    def _multidim_advective_tendency(
        self,
        tracer: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
        dt: float,
        *,
        horizontal_only: bool,
        w_interface: torch.Tensor | None = None,
        obc_west_tracer: torch.Tensor | None = None,
        obc_east_tracer: torch.Tensor | None = None,
        obc_south_tracer: torch.Tensor | None = None,
        obc_north_tracer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-dimensional advection matching MITgcm ``GAD_ADVECTION``.

        Sequential X→Y→R dimensional splitting in advective form:
        1. X-pass: compute X-fluxes from current tracer, update tracer
        2. Y-pass: compute Y-fluxes from X-updated tracer, update again
        3. R-pass: compute vertical fluxes from XY-updated tracer, update again
        4. Return (tracer_final - tracer_original) / dt as tendency

        Each pass uses:
          localT -= dt * recip_hFacC * recip_drF * recip_rA *
                    (flux_diff - T_orig * transport_diff) * maskInC

        The correction ``-T_orig * div(uTrans)`` converts flux-form to
        advective-form, matching MITgcm ``gad_advection.F`` L498-506.
        ``T_orig`` (not the intermediate ``localT``) is used in the correction,
        consistent with MITgcm's ``tracer(i,j,k,bi,bj)``.

        ``horizontal_only=True`` is the explicit implicit-R hand-off: execution
        stops after Y and never consumes a vertical velocity.  The full path
        requires a LOCOMgcm downward-positive ``w_interface``.  A missing
        ``w_interface`` is never interpreted as a horizontal-only mode.
        """
        if type(horizontal_only) is not bool:
            raise TypeError("horizontal_only must be an exact bool")
        if not horizontal_only and w_interface is None:
            raise ValueError(
                "full multi-dimensional tracer advection requires w_interface; "
                "use the explicit horizontal-only API to stop after Y"
            )
        if horizontal_only:
            # Fused X→Y CUDA pass (gate ``tracer_adv_fused_cuda``, default OFF).
            # Returns None for every configuration it cannot reproduce bitwise,
            # and the untouched eager body below then runs unchanged.  Gate
            # first, package import second (the repository's wiring idiom, cf.
            # kpp_mixing.py): with the gate off this call must not pay for
            # importing the package or registering its custom op.
            from locomgcm.accel import (
                consumer_snapshot,
                in_accelerated_dispatch,
                requested,
            )

            if not in_accelerated_dispatch():
                with consumer_snapshot():
                    if requested(("tracer_adv_fused_cuda",)):
                        from locomgcm.dynamics._tracer_adv_fused import (
                            dispatcher,
                        )

                        fused = dispatcher.fused_horizontal_multidim_tendency(
                            self, tracer, u, v, grid, dt
                        )
                        if fused is not None:
                            return fused
        has_obcs = grid.boundary_topology.open_x or grid.boundary_topology.open_y
        # MITgcm horizontal multi-dim advection is formulated in terms of
        # face volume transports and cell volume normalisation:
        #   xA = dyG * drF * hFacW
        #   yA = dxG * drF * hFacS
        #   uTrans = u * xA
        #   vTrans = v * yA
        #   localT -= dt * recip_hFacC * recip_drF * recip_rA
        #               * (af diff - T_orig * transport diff)
        # See gad_advection.F:297-308,499-505,567-573,720-726,790-796.
        # Narrow-waist 2-D metric injection: read the geometry-neutral
        # TracerMetricBundle instead of the 1-D *_row leak.  Byte-identical to
        # the per-row fetches (verified broadcast equivalence): bundle.dxG == dxG_row_ext.view(ny+1,1) in the transport
        # multiply; bundle.recip_volume values == 1/(cell_area·drF·hFacC).
        bundle = build_tracer_metric_bundle(grid, tracer.dtype, tracer.device)
        drF = bundle.layer_thickness
        dyG = bundle.dyG
        dxG = bundle.dxG

        # hFacW/hFacS face-weight only 4-D fields (preserve the prior guard).
        hFacW = bundle.hFacW if (bundle.hFacW is not None and tracer.ndim >= 4) else None
        hFacS = bundle.hFacS if (bundle.hFacS is not None and tracer.ndim >= 4) else None
        recip_hFacC = bundle.recip_hFacC if grid.masks.hFacC is not None else None
        recip_local_volume = bundle.recip_volume

        # maskInC for OBCS: prevents updating tracer in OBC ghost cells
        maskInC = None
        maskInC_2d = None
        if has_obcs:
            maskInC_2d, _, _ = _compute_interior_mask(grid, tracer.dtype, tracer.device)
            if tracer.ndim >= 4:
                maskInC = maskInC_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, ny, nx]
            else:
                maskInC = maskInC_2d.unsqueeze(0)  # [1, ny, nx]

        tracer_orig = tracer  # keep reference to original
        localT = tracer.clone()  # working copy, will be modified sequentially

        u_trans, v_trans = _calc_adv_flow_transports_mit_order(
            u,
            v,
            dyG,
            dxG,
            drF,
            hFacW=hFacW,
            hFacS=hFacS,
        )

        # ---- Pass 1: X-direction ----
        if self.scheme in HORIZONTAL_STENCIL_SCHEMES:
            # MITgcm gad_dst3fl_adv_x.F directly outputs transport-form flux
            # (af = uT in the kernel signature) — see gad_advection.F:421-423
            # which calls GAD_DST3FL_ADV_X(uTrans, uFld, maskLocW, localTij → af).
            # OBC cells (where ``obc_west/east_tracer`` would be applied) fall
            # back to the upwind-flux branch inside _stencil_face_flux_x via the
            # boundary-near 2-face skirt; the existing _obcs_upwind_x override
            # below additionally clamps the OB face when present.
            flux_u = _stencil_face_flux_x(
                localT, u, u_trans, grid,
                dt=dt,
                scheme=self.scheme,
            )
        else:
            flux_u = _two_point_face_flux_x(
                localT, u_trans, u, grid,
                scheme=self.scheme,
                dt=dt,
                obc_west_tracer=obc_west_tracer,
                obc_east_tracer=obc_east_tracer,
            )

        if has_obcs and self.obcs_upwind and grid.boundary_topology.open_x:
            flux_u = _obcs_upwind_x(flux_u, u_trans, localT, grid, maskInC_2d, hFacW)

        div_flux_x = zonal_face_difference(flux_u)
        div_utrans_x = zonal_face_difference(u_trans)

        # Advective form: div(u*T) - T_orig * div(u)
        update_x, correction_x = _multidim_advective_pass_update(
            tracer_orig,
            div_flux_x,
            div_utrans_x,
            recip_local_volume,
            maskInC,
        )

        if _MULTIDIM_DEBUG_STATE["enabled"]:
            with torch.no_grad():
                _MULTIDIM_DEBUG_STATE["log"].append({
                    "step": _MULTIDIM_DEBUG_STATE["step"],
                    "pass": "X",
                    "max_div_flux": float(div_flux_x.abs().max()),
                    "max_correction": float(correction_x.abs().max()),
                    "max_div_utrans": float(div_utrans_x.abs().max()),
                    "max_update": float(update_x.abs().max()),
                    "max_update_x_dt": float((update_x * dt).abs().max()),
                    "max_localT_before": float(localT.abs().max()),
                })

        if _MULTIDIM_LEAF_INJECT["enabled"]:
            leaf = update_x.detach().clone().requires_grad_(True)
            _MULTIDIM_LEAF_INJECT["leaves"]["update_x"] = leaf
            update_x = leaf
        localT = localT - dt * update_x

        if _MULTIDIM_CAPTURE["enabled"]:
            with torch.no_grad():
                _MULTIDIM_CAPTURE["tensors"]["tracer_orig"] = tracer_orig.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["localT_after_X"] = localT.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["update_x"] = update_x.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["div_flux_x"] = div_flux_x.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["flux_u"] = flux_u.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["u_trans_x"] = u_trans.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["correction_x"] = correction_x.detach().clone()

        # ---- Pass 2: Y-direction ----
        if self.scheme in HORIZONTAL_STENCIL_SCHEMES:
            flux_v = _stencil_face_flux_y(
                localT, v, v_trans, grid,
                dt=dt,
                scheme=self.scheme,
            )
        else:
            flux_v = _two_point_face_flux_y(
                localT, v_trans, v, grid,
                scheme=self.scheme,
                dt=dt,
                obc_south_tracer=obc_south_tracer,
                obc_north_tracer=obc_north_tracer,
            )

        if has_obcs and self.obcs_upwind and grid.boundary_topology.open_y:
            flux_v = _obcs_upwind_y(flux_v, v_trans, localT, grid, maskInC_2d, hFacS)

        div_flux_y = meridional_face_difference(flux_v)
        div_vtrans_y = meridional_face_difference(v_trans)

        update_y, correction_y = _multidim_advective_pass_update(
            tracer_orig,
            div_flux_y,
            div_vtrans_y,
            recip_local_volume,
            maskInC,
        )

        if _MULTIDIM_DEBUG_STATE["enabled"]:
            with torch.no_grad():
                _MULTIDIM_DEBUG_STATE["log"].append({
                    "step": _MULTIDIM_DEBUG_STATE["step"],
                    "pass": "Y",
                    "max_div_flux": float(div_flux_y.abs().max()),
                    "max_correction": float(correction_y.abs().max()),
                    "max_div_vtrans": float(div_vtrans_y.abs().max()),
                    "max_update": float(update_y.abs().max()),
                    "max_update_y_dt": float((update_y * dt).abs().max()),
                    "max_localT_before": float(localT.abs().max()),
                })

        if _MULTIDIM_LEAF_INJECT["enabled"]:
            leaf = update_y.detach().clone().requires_grad_(True)
            _MULTIDIM_LEAF_INJECT["leaves"]["update_y"] = leaf
            update_y = leaf
        localT = localT - dt * update_y

        if _MULTIDIM_CAPTURE["enabled"]:
            with torch.no_grad():
                _MULTIDIM_CAPTURE["tensors"]["localT_after_Y"] = localT.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["update_y"] = update_y.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["div_flux_y"] = div_flux_y.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["flux_v"] = flux_v.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["v_trans_y"] = v_trans.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["correction_y"] = correction_y.detach().clone()

        if horizontal_only:
            tendency = (localT - tracer_orig) / dt
            if _MULTIDIM_CAPTURE["enabled"]:
                with torch.no_grad():
                    _MULTIDIM_CAPTURE["tensors"]["tendency"] = tendency.detach().clone()
            return tendency

        # ---- Pass 3: R-direction ----
        tracer_r = localT if localT.ndim >= 4 else localT.unsqueeze(0)
        tracer_orig_r = tracer_orig if tracer_orig.ndim >= 4 else tracer_orig.unsqueeze(0)
        w_interface_r = w_interface.to(dtype=tracer.dtype, device=tracer.device)
        if w_interface_r.ndim == 3:
            w_interface_r = w_interface_r.unsqueeze(0)

        # Use w_interface as vertical transport.  w_interface is
        # diagnosed from div(u*dz*hFacW, v*dz*hFacS) via cumsum and
        # is already zero at dry interior interfaces by construction.
        # Masking interior interfaces with hFacC[:-1] breaks the
        # exact cancellation of X+Y+Z corrections at partial cells
        # near bathymetry.
        #
        # HOWEVER: the surface interface (k=0) must still be zeroed
        # to match MITgcm GAD_DST2U1_ADV_R:87 —
        #   rTransV = rTrans * maskC(km1)
        # where km1=0 for the surface layer implies maskC=0 (no
        # cell above the free surface).  Without this, the R-pass
        # correction term `tracer * div_rtrans` at k=0 picks up a
        # spurious `tracer × w_surface` contribution, which under a
        # linear-drift free surface accumulates into a persistent
        # warm bias at the surface (+0.35 K over 7 days,
        # +6.7e-5 K/step steady state).
        # expects LOCOMgcm downward-positive w_interface; only zero surface.
        r_trans = _surface_zeroed_vertical_transport(w_interface_r)

        # MITgcm hands GAD_DST2U1_ADV_R the HORIZONTAL scheme in this pass
        # (gad_advection.F:987-991), and that routine takes rLimit=1 only for
        # scheme 20 (gad_dst2u1_adv_r.F:57-58): a vertical 1 or 20 runs DST2
        # exactly when the horizontal scheme is 20.  The single-RHS path passes
        # the vertical scheme itself (gad_calc_rhs.F:519-521).
        r_scheme = self.vertical_scheme
        if r_scheme in (TracerFluxScheme.UPWIND, TracerFluxScheme.DST2):
            r_scheme = (
                TracerFluxScheme.DST2 if self.scheme == TracerFluxScheme.DST2
                else TracerFluxScheme.UPWIND
            )
        if r_scheme in VERTICAL_CFL_SCHEMES:
            # MITgcm scheme 30 / 33: high-order DST3 vertical advection, without
            # and with the Sweby flux limiter.  recip_drC = center-to-center
            # spacing at interior interfaces (1/diff(center_depth)); dt is the
            # tracer step; the wet mask (maskC) is hFacC>0 (== recip_hFacC>0).
            recip_drC = interface_recip_drC(grid, tracer.dtype, tracer.device)
            v_mask = (recip_hFacC > 0).to(dtype=tracer.dtype) if recip_hFacC is not None else None
            if r_scheme in VERTICAL_VELOCITY_FLUX_SCHEMES:
                flux_r = vertical_tracer_flux(
                    r_scheme, tracer_r, None, velocity=r_trans,
                    recip_drC=recip_drC, dt=dt, mask=v_mask, layer_thickness=None,
                    recip_center_spacing=None,
                )
            else:
                # Velocity space: r_trans is the flux weight (MITgcm rTrans
                # without rA) and w_interface_r the Courant velocity (wFld).
                # PPM reads drF for its Courant number (gad_ppm_flx_r.F); PQM
                # also reads recip_drC(1..Nr+1), surface and bottom included
                # (gad_pqm_p5e_r.F).
                recip_center_spacing = None
                if tracer_scheme_spec(r_scheme).reads_cell_geometry:
                    recip_center_spacing = 1.0 / grid.vertical_coordinate.center_spacing.to(
                        dtype=tracer.dtype, device=tracer.device,
                    )
                flux_r = vertical_tracer_flux(
                    r_scheme, tracer_r, r_trans, velocity=w_interface_r,
                    recip_drC=recip_drC, dt=dt, mask=v_mask,
                    layer_thickness=grid.layer_thickness,
                    recip_center_spacing=recip_center_spacing,
                )
        else:
            flux_r = vertical_tracer_flux(
                r_scheme, tracer_r, r_trans, velocity=None,
                recip_drC=None, dt=None, mask=None, layer_thickness=None,
                recip_center_spacing=None,
            )
        div_flux_r = -vertical_flux_divergence(flux_r, grid.layer_thickness)
        div_rtrans = -vertical_flux_divergence(r_trans, grid.layer_thickness)

        if localT.ndim >= 4:
            div_flux_r_store = div_flux_r
            flux_r_store = flux_r
            r_trans_store = r_trans
        else:
            div_flux_r_store = div_flux_r.squeeze(0)
            flux_r_store = flux_r.squeeze(0)
            r_trans_store = r_trans.squeeze(0)

        update_r_apply, correction_r_store = _multidim_vertical_pass_update(
            tracer_orig_r,
            div_flux_r,
            div_rtrans,
            maskInC=maskInC,
            recip_hFacC=recip_hFacC,
            squeeze_leading_batch=(localT.ndim < 4),
        )

        if _MULTIDIM_DEBUG_STATE["enabled"]:
            with torch.no_grad():
                _MULTIDIM_DEBUG_STATE["log"].append({
                    "step": _MULTIDIM_DEBUG_STATE["step"],
                    "pass": "R",
                    "max_div_flux": float(div_flux_r_store.abs().max()),
                    "max_correction": float(correction_r_store.abs().max()),
                    "max_div_rtrans": float(div_rtrans.abs().max()),
                    "max_update": float(update_r_apply.abs().max()),
                    "max_update_r_dt": float((update_r_apply * dt).abs().max()),
                    "max_localT_before": float(localT.abs().max()),
                })

        if _MULTIDIM_LEAF_INJECT["enabled"]:
            leaf = update_r_apply.detach().clone().requires_grad_(True)
            _MULTIDIM_LEAF_INJECT["leaves"]["update_r"] = leaf
            update_r_apply = leaf
        localT = localT - dt * update_r_apply

        if _MULTIDIM_CAPTURE["enabled"]:
            with torch.no_grad():
                _MULTIDIM_CAPTURE["tensors"]["localT_after_R"] = localT.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["update_r"] = update_r_apply.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["div_flux_r"] = div_flux_r_store.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["flux_r"] = flux_r_store.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["r_trans_r"] = r_trans_store.detach().clone()
                _MULTIDIM_CAPTURE["tensors"]["correction_r"] = correction_r_store.detach().clone()

        if _MULTIDIM_CAPTURE["enabled"]:
            with torch.no_grad():
                _MULTIDIM_CAPTURE["tensors"]["tendency"] = ((localT - tracer_orig) / dt).detach().clone()

        # Return tendency = (localT - tracer_orig) / dt
        # Already includes recip_hFacC (applied inside the passes)
        return (localT - tracer_orig) / dt

    def evaluate_horizontal_only(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
        *,
        tracer_names: tuple[str, ...],
    ) -> StateTendency:
        """Evaluate X/Y tracer advection without a direct vertical R pass.

        This API deliberately has no vertical-velocity parameter.  It is the
        producer side of the implicit vertical-advection hand-off; the stepper
        supplies the distinct MIT-upward tracer-time wrapper to the combined
        matrix and continuity-correction operators instead.
        """
        requested = tuple(tracer_names)
        unknown = tuple(name for name in requested if name not in {"temp", "salt"})
        if unknown:
            raise ValueError(f"unsupported tracer_names for horizontal-only advection: {unknown}")
        selected = tuple(name for name in self.tracer_names if name in requested)
        if self.combined_with_vtt and selected:
            raise ValueError(
                "combined_with_vtt cannot provide a horizontal-only implicit-R hand-off"
            )

        u, v = self._tracer_velocity(state, diagnostics)
        temp_tendency = torch.zeros_like(state.temp.data)
        salt_tendency = torch.zeros_like(state.salt.data)
        if self._runs_multi_dim():
            if "temp" in selected:
                temp_tendency = self._multidim_advective_tendency(
                    state.temp.data,
                    u,
                    v,
                    grid,
                    self.dt,
                    horizontal_only=True,
                    **_obc_tracer_kwargs(
                        diagnostics,
                        tracer_name="temp",
                        dtype=state.temp.data.dtype,
                        device=state.temp.data.device,
                    ),
                )
            if "salt" in selected:
                salt_tendency = self._multidim_advective_tendency(
                    state.salt.data,
                    u,
                    v,
                    grid,
                    self.dt,
                    horizontal_only=True,
                    **_obc_tracer_kwargs(
                        diagnostics,
                        tracer_name="salt",
                        dtype=state.salt.data.dtype,
                        device=state.salt.data.device,
                    ),
                )
        else:
            if "temp" in selected:
                temp_tendency = self._advective_tendency(
                    state.temp.data,
                    u,
                    v,
                    grid,
                )
            if "salt" in selected:
                salt_tendency = self._advective_tendency(
                    state.salt.data,
                    u,
                    v,
                    grid,
                )
            if grid.masks.hFacC is not None:
                hFacC = grid.masks.hFacC.to(
                    dtype=temp_tendency.dtype,
                    device=temp_tendency.device,
                )
                recip_hFacC = _positive_reciprocal(hFacC)
                temp_tendency = temp_tendency * recip_hFacC
                salt_tendency = salt_tendency * recip_hFacC

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data),
            temp=temp_tendency,
            salt=salt_tendency,
        )

    def evaluate_for_implicit_vertical_advection(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
        *,
        implicit_tracer_names: tuple[str, ...],
    ) -> StateTendency:
        """Split this module by tracer name at the explicit/implicit R seam."""
        enabled = tuple(
            name for name in self.tracer_names if name in set(implicit_tracer_names)
        )
        disabled = tuple(name for name in self.tracer_names if name not in set(enabled))
        accumulator = StateTendency.accumulator(state)
        if enabled:
            accumulator.add(
                self.evaluate_horizontal_only(
                    state,
                    diagnostics,
                    grid,
                    tracer_names=enabled,
                )
            )
        if disabled:
            accumulator.add(
                replace(self, tracer_names=disabled).evaluate(
                    state,
                    diagnostics,
                    grid,
                )
            )
        return accumulator.finish()

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        u, v = self._tracer_velocity(state, diagnostics)
        temp_tendency = torch.zeros_like(state.temp.data)
        salt_tendency = torch.zeros_like(state.salt.data)

        if self._runs_multi_dim():
            w_interface = self._tracer_w_interface(state, diagnostics, grid)
            # Multi-dimensional advection (MITgcm GAD_ADVECTION):
            # sequential X→Y→R splitting, advective form.
            # recip_hFacC is already applied inside the method.
            if "temp" in self.tracer_names:
                temp_obc_kwargs = _obc_tracer_kwargs(
                    diagnostics,
                    tracer_name="temp",
                    dtype=state.temp.data.dtype,
                    device=state.temp.data.device,
                )
                temp_tendency = self._multidim_advective_tendency(
                    state.temp.data,
                    u,
                    v,
                    grid,
                    self.dt,
                    horizontal_only=False,
                    w_interface=w_interface,
                    **temp_obc_kwargs,
                )
            if "salt" in self.tracer_names:
                salt_obc_kwargs = _obc_tracer_kwargs(
                    diagnostics,
                    tracer_name="salt",
                    dtype=state.salt.data.dtype,
                    device=state.salt.data.device,
                )
                salt_tendency = self._multidim_advective_tendency(
                    state.salt.data,
                    u,
                    v,
                    grid,
                    self.dt,
                    horizontal_only=False,
                    w_interface=w_interface,
                    **salt_obc_kwargs,
                )
        elif self.combined_with_vtt or (self.multi_dim and self.dt > 0):
            # The second case is multi_dim forced off by an Adams-Bashforth
            # horizontal scheme (_runs_multi_dim): the module still owns R.
            w_interface = self._tracer_w_interface(state, diagnostics, grid)

            if "temp" in self.tracer_names:
                temp_tendency = self._combined_advective_tendency(
                    state.temp.data, u, v, grid, w_interface,
                )
            if "salt" in self.tracer_names:
                salt_tendency = self._combined_advective_tendency(
                    state.salt.data, u, v, grid, w_interface,
                )

            if grid.masks.hFacC is not None:
                hFacC = grid.masks.hFacC.to(
                    dtype=temp_tendency.dtype, device=temp_tendency.device,
                )
                recip_hFacC = _positive_reciprocal(hFacC)
                temp_tendency = temp_tendency * recip_hFacC
                salt_tendency = salt_tendency * recip_hFacC
        else:
            w_interface = None
            if self.use_advective_form:
                w_interface = self._tracer_w_interface(state, diagnostics, grid)

            if "temp" in self.tracer_names:
                temp_tendency = self._advective_tendency(
                    state.temp.data, u, v, grid, w_interface=w_interface,
                )
            if "salt" in self.tracer_names:
                salt_tendency = self._advective_tendency(
                    state.salt.data, u, v, grid, w_interface=w_interface,
                )

            if grid.masks.hFacC is not None:
                hFacC = grid.masks.hFacC.to(
                    dtype=temp_tendency.dtype, device=temp_tendency.device,
                )
                recip_hFacC = _positive_reciprocal(hFacC)
                temp_tendency = temp_tendency * recip_hFacC
                salt_tendency = salt_tendency * recip_hFacC

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data),
            temp=temp_tendency,
            salt=salt_tendency,
        )

    def populate_diagnostic_state(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> None:
        del grid
        u, v = self.effective_velocity(state)
        diagnostics.tensors["transport_velocity_u"] = u
        diagnostics.tensors["transport_velocity_v"] = v
