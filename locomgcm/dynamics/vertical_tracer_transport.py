from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.geometry.metric_bundle import _positive_reciprocal

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.grid import _compute_interior_mask
from locomgcm.operators import (
    vertical_flux_divergence,
    vertical_tracer_advection_transport_space,
)
from locomgcm.operators.fluxes.tracer_flux import (
    TRACER_SCHEME_BY_NAME,
    VERTICAL_CFL_SCHEMES,
    VERTICAL_TRACER_SCHEMES,
    interface_recip_drC,
    vertical_tracer_flux,
)
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyPhase, TendencyStepDomain
from locomgcm.dynamics._registry import register_dynamics_module
from .vertical_velocity import (
    diagnose_vertical_velocity,
    upward_vertical_volume_flux_from_horizontal_divergence,
)


def _broadcast_layer_thickness_like(
    layer_thickness: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    dz = layer_thickness.to(dtype=reference.dtype, device=reference.device)
    if dz.ndim == 1:
        lead = max(reference.ndim - 3, 0)
        return dz.view(*([1] * lead), dz.shape[0], 1, 1)
    return dz


def _broadcast_cell_area_like(
    cell_area: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    area = cell_area.to(dtype=reference.dtype, device=reference.device)
    if area.ndim == 0:
        return area
    if area.ndim == 2:
        lead = max(reference.ndim - 3, 0)
        return area.view(*([1] * lead), 1, area.shape[-2], area.shape[-1])
    if area.ndim == reference.ndim - 1:
        return area.unsqueeze(-3)
    return area


def vertical_tracer_advection_from_horizontal_divergence(
    tracer: torch.Tensor,
    horizontal_volume_divergence: torch.Tensor,
    *,
    cell_area: torch.Tensor,
    layer_thickness: torch.Tensor,
    hFacC: torch.Tensor | None = None,
    wet_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MIT-aligned vertical tracer tendency from horizontal volume divergence.

    ``horizontal_volume_divergence`` is the per-layer horizontal volume
    transport divergence in m3/s on tracer cells.  The kernel reconstructs the
    MITgcm upward-positive vertical volume flux internally via bottom-up
    continuity, sets top and bottom tracer fluxes to zero, uses donor-cell
    vertical interpolation (positive flux takes the cell below), and applies
    the surface advective-form correction needed when the free-surface
    kinematic flux at the top interface is not used as a tracer flux.

    This function is geometry-neutral: leading dimensions may be batch,
    cubed-sphere face, or any other independent column index.  It consumes only
    tensors and 2-D metrics; horizontal exchange must already have been applied
    by the caller when constructing ``horizontal_volume_divergence``.
    """
    if tracer.ndim < 3:
        raise ValueError("tracer must have at least [nz, ny, nx] dimensions")
    if horizontal_volume_divergence.shape != tracer.shape:
        raise ValueError(
            "horizontal_volume_divergence must have the same shape as tracer"
        )

    nz = tracer.shape[-3]
    ny = tracer.shape[-2]
    nx = tracer.shape[-1]
    # MIT upward-positive volume flux: wflux[k] = wflux[k+1] - div_h[k].
    wflux = upward_vertical_volume_flux_from_horizontal_divergence(
        horizontal_volume_divergence
    )

    wT = torch.zeros_like(wflux)
    if nz > 1:
        wf_int = wflux[..., 1:nz, :, :]
        t_above = tracer[..., 0:nz - 1, :, :]
        t_below = tracer[..., 1:nz, :, :]
        # wflux is MIT upward-positive: positive flow takes the donor below.
        t_iface = torch.where(wf_int >= 0.0, t_below, t_above)
        wT[..., 1:nz, :, :] = wf_int * t_iface

    dz = _broadcast_layer_thickness_like(layer_thickness, tracer)
    area = _broadcast_cell_area_like(cell_area, tracer)
    if hFacC is None:
        hfac = torch.ones_like(tracer)
    else:
        hfac = hFacC.to(dtype=tracer.dtype, device=tracer.device)
    c_mask = (
        wet_mask.to(dtype=tracer.dtype, device=tracer.device)
        if wet_mask is not None
        else (hfac > 0).to(dtype=tracer.dtype)
    )

    dz_vol = area * dz * hfac
    recip_dz_vol = _positive_reciprocal(dz_vol)
    tend = -(wT[..., 0:nz, :, :] - wT[..., 1:nz + 1, :, :]) * recip_dz_vol * c_mask

    surf_dilute = (
        -tracer[..., 0, :, :]
        * wflux[..., 0, :, :]
        * recip_dz_vol[..., 0, :, :]
        * c_mask[..., 0, :, :]
    )
    tend = torch.cat(
        ((tend[..., 0, :, :] + surf_dilute).unsqueeze(-3), tend[..., 1:, :, :]),
        dim=-3,
    )
    return tend * c_mask


@dataclass(frozen=True)
@register_dynamics_module("vertical_tracer_transport")
class VerticalTracerTransport(TendencyModule):
    tracer_names: tuple[str, ...] = ("temp", "salt")
    name: str = "vertical_tracer_transport"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER
    vertical_advection_scheme: str = "centered"
    transport_space: bool = True
    #: Tracer time step [s] (MITgcm ``deltaTLev``).  Only the DST3 family reads
    #: it — for its Courant number — and for those schemes it is mandatory; the
    #: donor-cell schemes ignore it.  Mirrors
    #: ``HorizontalTracerAdvection.dt``.
    dt: float = 0.0
    #: GM_AdvForm: the vertical transport is wFld = wVel + the GM bolus w
    #: (thermodynamics.F:258-276), as ``HorizontalTracerAdvection.gm_residual_flow``.
    gm_residual_flow: bool = False

    #: Vertical advection kernels this module can dispatch.  MITgcm never forces
    #: multi-dim for DST3 / DST3-FL (``gad_init_fixed.F:126-137`` clears the flag
    #: only for ENUM_CENTERED_2ND / ENUM_UPWIND_3RD / ENUM_CENTERED_4TH), so with
    #: ``multiDimAdvection=.FALSE.`` the vertical DST3 kernels run on the
    #: single-RHS path this module implements — GAD_CALC_RHS dispatches
    #: GAD_DST3_ADV_R / GAD_DST3FL_ADV_R at ``gad_calc_rhs.F:533-536`` /
    #: ``:567-570``.  PPM and PQM have no single-RHS branch (``single_rhs``).
    _SUPPORTED_SCHEMES = tuple(
        scheme.value for scheme in VERTICAL_TRACER_SCHEMES
        if TRACER_SCHEME_BY_NAME[scheme].single_rhs
    )
    #: The subset whose CFL/limiter needs ``dt`` and ``recip_drC``.
    _DT_REQUIRED_SCHEMES = VERTICAL_CFL_SCHEMES

    def __post_init__(self) -> None:
        if self.vertical_advection_scheme not in self._SUPPORTED_SCHEMES:
            raise ValueError(
                "VerticalTracerTransport supports vertical_advection_scheme in "
                f"{self._SUPPORTED_SCHEMES}, got "
                f"'{self.vertical_advection_scheme}'."
            )

        # The scheme's requirements are checked here, not left to a silent
        # fallback.  A DST3 scheme without a time step cannot
        # form its Courant number, so it fails at construction rather than
        # quietly degrading to donor-cell upwind (which would be MITgcm scheme 1,
        # a different discretisation) or raising from deep inside the shared core
        # after the vertical velocity has already been diagnosed.
        if self.vertical_advection_scheme in self._DT_REQUIRED_SCHEMES and not self.dt > 0:
            raise ValueError(
                "VerticalTracerTransport(vertical_advection_scheme="
                f"'{self.vertical_advection_scheme}') requires dt>0 (the tracer "
                f"time step, MITgcm deltaTLev) for its Courant number; got dt={self.dt}"
            )

    def vertical_velocity_interface(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> torch.Tensor:
        if "vertical_velocity_interface" in diagnostics.fields:
            w = diagnostics.fields["vertical_velocity_interface"].data
        else:
            _, w, _ = diagnose_vertical_velocity(state, grid)
        if self.gm_residual_flow:
            # Downward-positive -(wVel + w_bolus); gm_bolus_w is MIT upward-positive.
            w = w - diagnostics.tensors["gm_bolus_w"]
        # MITgcm `calc_adv_flow.F:113-119` forces rTrans[k=1] = 0 (the
        # surface interface) for tracer RHS in linear free-surface mode:
        # the rigid lid carries no tracer transport even when the
        # kinematic ∂η/∂t produces a non-zero diagnostic wVel there.
        # Without this zeroing, `_tracer_tendency_transport_space` leaves
        # an algebraic residual `-S·w_surface/dz_surface` on the surface
        # layer that injects ~8 PSU per step on cells adjacent to OBC
        # u-injection.  Verified against MIT exp4 step-1: torch i=1
        # surface salt 43.51 → 35.05 (= MIT 35.04) when w[k=0]=0.
        w = w.clone()
        w[..., 0, :, :] = 0.0
        return w

    def tracer_tendency(
        self,
        tracer: torch.Tensor,
        vertical_velocity_interface: torch.Tensor,
        grid: StructuredCGrid,
    ) -> torch.Tensor:
        mask = None
        if grid.masks.hFacC is not None:
            mask = (grid.masks.hFacC > 0).to(dtype=tracer.dtype)

        has_obcs = grid.boundary_topology.open_x or grid.boundary_topology.open_y
        if has_obcs:
            maskInC, _, _ = _compute_interior_mask(grid, tracer.dtype, tracer.device)
            mc = maskInC.unsqueeze(0).unsqueeze(0) if tracer.ndim >= 4 else maskInC.unsqueeze(0)
        else:
            mc = None

        if self.transport_space:
            return self._tracer_tendency_transport_space(
                tracer, vertical_velocity_interface, grid, mask, mc,
            )
        return self._tracer_tendency_velocity_space(
            tracer, vertical_velocity_interface, grid, mask, mc,
        )

    def _tracer_tendency_transport_space(
        self,
        tracer: torch.Tensor,
        w_iface: torch.Tensor,
        grid: StructuredCGrid,
        mask: torch.Tensor | None,
        mc: torch.Tensor | None,
    ) -> torch.Tensor:
        """Transport-space vertical tracer tendency (MITgcm gad_calc_rhs.F form).

        Builds the interface volume transport ``rTrans = wVel * rA``
        (``calc_adv_flow.F:112``) and delegates to the geometry-neutral
        :func:`vertical_tracer_advection_transport_space` core that the cubed
        sphere ``gad_advection_first_order_cube`` vertical pass also uses, so the
        two geometries share one vertical-advection kernel.  The multiply-by-rA / divide-by-rA is a
        mathematical identity that fixes MITgcm's transport-space FP ordering
        (vs the velocity-space path, ~1.6e-9/step difference).
        """
        rA_t = grid.metrics.rA_for(tracer.dtype, tracer.device)
        lead = max(tracer.ndim - 2, 0)
        rA_t = rA_t.view(*([1] * lead), grid.ny, grid.nx)

        # rTrans = w * rA (volume transport at interfaces, positive-down)
        rTrans = w_iface * rA_t

        return vertical_tracer_advection_transport_space(
            tracer,
            rTrans,
            velocity=w_iface,
            cell_area=rA_t,
            layer_thickness=grid.layer_thickness,
            scheme=self.vertical_advection_scheme,
            mask=mask,
            interior_mask=mc,
            **self._dst3_courant_kwargs(tracer, grid),
        )

    def _dst3_courant_kwargs(
        self,
        tracer: torch.Tensor,
        grid: StructuredCGrid,
    ) -> dict[str, object]:
        """``recip_drC``/``dt`` for the DST3 kernels; empty for donor-cell ones.

        ``recip_drC`` is MITgcm's center-to-center spacing at interior
        interfaces, built from ``grid.center_depth`` exactly as the multi-dim
        R-pass does (``tracers.py`` DST3 branch).
        """
        if self.vertical_advection_scheme not in self._DT_REQUIRED_SCHEMES:
            return {}
        return {
            "recip_drC": interface_recip_drC(grid, tracer.dtype, tracer.device),
            "dt": self.dt,
        }

    def _tracer_tendency_velocity_space(
        self,
        tracer: torch.Tensor,
        w_iface: torch.Tensor,
        grid: StructuredCGrid,
        mask: torch.Tensor | None,
        mc: torch.Tensor | None,
    ) -> torch.Tensor:
        """Original velocity-space path (no rA multiply-divide)."""
        if self.vertical_advection_scheme in self._DT_REQUIRED_SCHEMES:
            courant = self._dst3_courant_kwargs(tracer, grid)
            flux = vertical_tracer_flux(
                self.vertical_advection_scheme, tracer, w_iface, velocity=w_iface,
                mask=mask, layer_thickness=None,
                recip_center_spacing=None, **courant,
            )
        else:
            flux = vertical_tracer_flux(
                self.vertical_advection_scheme, tracer, w_iface, velocity=None,
                recip_drC=None, dt=None, mask=mask, layer_thickness=None,
                recip_center_spacing=None,
            )
        if mc is not None:
            flux = flux * mc

        flux_div = vertical_flux_divergence(flux, grid.layer_thickness)

        div_r_transport = vertical_flux_divergence(w_iface, grid.layer_thickness)
        if mc is not None:
            div_r_transport = div_r_transport * mc

        return flux_div - tracer * div_r_transport

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        vertical_velocity_interface = self.vertical_velocity_interface(state, diagnostics, grid)
        temp_tendency = torch.zeros_like(state.temp.data)
        salt_tendency = torch.zeros_like(state.salt.data)

        if "temp" in self.tracer_names:
            temp_tendency = self.tracer_tendency(state.temp.data, vertical_velocity_interface, grid)
        if "salt" in self.tracer_names:
            salt_tendency = self.tracer_tendency(state.salt.data, vertical_velocity_interface, grid)

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
