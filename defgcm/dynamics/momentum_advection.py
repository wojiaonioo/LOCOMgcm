"""3D momentum advection tendency — volume-flux form on C-grid.

Implements the complete 3D flux-form momentum advection:

    ∂u/∂t = −∇·(𝐯 u)   (horizontal: Fuu, Fvu; vertical: Fwu)
    ∂v/∂t = −∇·(𝐯 v)   (horizontal: Fuv, Fvv; vertical: Fwv)

Horizontal fluxes use volume-flux weighted form following MITgcm
mom_fluxform.F:
    uTrans = u · dy · dz · hFacW
    Fuu(i) = 0.25 · (uTrans(i) + uTrans(i+1)) · (u(i) + u(i+1))

Vertical fluxes follow MITgcm mom_u_adv_wu.F / mom_v_adv_wv.F:
    RU = 0.5 · (w(i-1)·A(i-1) + w(i)·A(i))  (on uniform grid: avg w)
    u_if = 0.5 · (u(k-1) + u(k))
    Fwu = RU · u_if

Upstream references
-------------------
- MITgcm ``mom_fluxform.F``, ``mom_u_adv_wu.F``, ``mom_v_adv_wv.F``
- Veros ``veros/core/momentum.py``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from defgcm.geometry.metric_bundle import _positive_reciprocal

from defgcm.geometry import StructuredCGrid
from defgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyStepDomain
from ._momentum_advection_topology import apply_corner_flux3_override
from .momentum_fluxes import (
    corner_cross_flux,
    corner_mean_x,
    corner_mean_y,
    face_transport,
    u_flux_divergence,
    u_self_flux,
    v_flux_divergence,
    v_self_flux,
)
from .momentum_vertical_fluxes import (
    interface_flux,
    surface_correction_source,
    u_face_vertical_velocity,
    u_surface_correction,
    v_face_vertical_velocity,
    v_surface_correction,
    vertical_flux_divergence,
    vertical_transport,
)
from defgcm.dynamics._registry import register_dynamics_module


def _topology_edge_pair_mask(
    length: int,
    *,
    grid: StructuredCGrid,
    extent: HaloExtentV1,
    device: torch.device,
) -> torch.Tensor:
    core = torch.ones(length, dtype=torch.bool, device=device)
    core_ext = exchange(
        core,
        grid=grid,
        location="F",
        extent=extent,
        fill=FillSpec.ZERO,
    )
    return core_ext[:-1] & core_ext[1:]


@dataclass(frozen=True)
class _FluxFormGeometry:
    """Face geometry the horizontal and vertical flux paths read, built once per evaluation."""

    layer_thickness: torch.Tensor  # drF [1, nz, 1, 1]
    hfac_w: torch.Tensor  # [1, nz, ny, nx+1]; ones(1, 1, 1, 1) without hFacW
    hfac_s: torch.Tensor  # [1, nz, ny+1, nx]; ones(1, 1, 1, 1) without hFacS
    dyG: torch.Tensor  # U-face width: scalar dy (row metric) or [1, 1, ny, nx+1]
    dxG: torch.Tensor  # V-face width: [1, 1, ny+1, 1] (row metric) or [1, 1, ny+1, nx]
    rAw: torch.Tensor  # native U-cell area [1, 1, ny, nx+1]
    rAs: torch.Tensor  # native V-cell area [1, 1, ny+1, nx]


def _flux_form_geometry(
    grid: StructuredCGrid,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    hFacW: torch.Tensor | None,
    hFacS: torch.Tensor | None,
) -> _FluxFormGeometry:
    nz = u.shape[-3]
    nx = grid.nx
    ny = grid.ny
    # Layer thickness [1, nz, 1, 1]
    dz = grid.layer_thickness.to(dtype=u.dtype, device=u.device).view(1, nz, 1, 1)

    # hFacW for u-faces [1, nz, ny, nx+1]; fall back to 1.0 for flat grids
    if hFacW is not None:
        hfw = hFacW.to(dtype=u.dtype, device=u.device)
        if hfw.ndim == 3:
            hfw = hfw.unsqueeze(0)  # [1, nz, ny, nx+1]
    else:
        hfw = u.new_ones(1, 1, 1, 1)  # broadcast to all

    # hFacS for v-faces [1, nz, ny+1, nx]
    if hFacS is not None:
        hfs = hFacS.to(dtype=v.dtype, device=v.device)
        if hfs.ndim == 3:
            hfs = hfs.unsqueeze(0)
    else:
        hfs = v.new_ones(1, 1, 1, 1)

    # dxG = latitude-varying zonal face width at the v-face; a scalar dx biases
    # rows off-centre on a spherical-polar grid.
    if grid.metrics.del_x is None:
        dyG = u.new_tensor(grid.dy)
        dxG = grid.metrics.dxG_row_ext_for(v.dtype, v.device).view(1, 1, ny + 1, 1)
    else:
        dyG = grid.metrics.dyG_for(u.dtype, u.device).view(1, 1, ny, nx + 1)
        dxG = grid.metrics.dxG_for(v.dtype, v.device).view(1, 1, ny + 1, nx)
    rAw = grid.metrics.rAw_for(u.dtype, u.device).view(1, 1, ny, nx + 1)
    rAs = grid.metrics.rAs_for(v.dtype, v.device).view(1, 1, ny + 1, nx)
    return _FluxFormGeometry(
        layer_thickness=dz, hfac_w=hfw, hfac_s=hfs, dyG=dyG, dxG=dxG, rAw=rAw, rAs=rAs,
    )


@dataclass(frozen=True)
@register_dynamics_module("momentum_advection")
class MomentumAdvection(TendencyModule):
    """3D momentum advection in volume-flux form.

    Computes horizontal (uu, vu, uv, vv) and vertical (wu, wv) advection
    tendencies. The interface vertical velocity comes from
    ``EvaluationInputs.w_interface`` (MIT upward-positive, negated here) or from
    the ``vertical_velocity_interface`` diagnostics field; a caller that
    supplies neither raises (the explicit-input requirement).

    On a uniform-metric Arakawa C-grid with partial cells (hFac), the
    fluxes use volume-flux weighting: transport = velocity × face area,
    and the tendency is normalized by cell volume.
    """

    name: str = "momentum_advection"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    # MITgcm select_rStar of the run's stepper, written by the canonical assembly.
    select_r_star: int = field(kw_only=True)


    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        u = state.u.data  # [B, nz, ny, nx+1]
        v = state.v.data  # [B, nz, ny+1, nx]

        # Get partial-cell fractions (MITgcm: hFacW, hFacS)
        hFacW = getattr(grid.masks, "hFacW", None)
        hFacS = getattr(grid.masks, "hFacS", None)

        geometry = _flux_form_geometry(grid, u, v, hFacW=hFacW, hFacS=hFacS)
        tend_u = self._u_tendency_horizontal(u, v, grid, geometry=geometry)
        tend_v = self._v_tendency_horizontal(u, v, grid, geometry=geometry)

        # Vertical advection needs the interface vertical velocity in LOCOMgcm downward-positive
        # convention.  The production stepper passes EvaluationInputs.w_interface (MIT
        # upward-positive, the same source the vector-invariant module consumes) so
        # prefer it and negate to LOCOMgcm-downward.  Fall back to the
        # diagnostics field (already LOCOMgcm-downward, metadata positive=down) for standalone /
        # op-level use where the cache is not populated.
        cached_w = None if diagnostics.inputs is None else diagnostics.inputs.w_interface
        if cached_w is not None:
            w = -cached_w
        elif hasattr(diagnostics, 'fields') and "vertical_velocity_interface" in diagnostics.fields:
            w = diagnostics.fields["vertical_velocity_interface"].data
        else:
            # the explicit-input requirement: MITgcm evaluates MOM_U_ADV_WU / MOM_V_ADV_WV
            # whenever momAdvection is on (mom_fluxform.F:384,476,789), so a
            # missing w is an unwired input, not an absent term.
            raise RuntimeError(
                "MomentumAdvection needs the interface vertical velocity: set "
                "EvaluationInputs.w_interface (MIT upward-positive) or put "
                "'vertical_velocity_interface' in the diagnostics"
            )

        # No level-count gate here.  MITgcm keeps the vertical advective flux at
        # Nr=1: mom_fluxform.F:383-416 fills the surface flux fVerUkm/fVerVkm on the
        # first call (k=1) from MOM_U_ADV_WU / MOM_V_ADV_WV, and :539 / :836 add
        # (fVerUkp - fVerUkm); with a single layer the bottom flux fVerUkp is zero,
        # so the surface flux survives ALONE and enters gU/gV.  The kernels below
        # already implement exactly that at nz == 1 (one-sided top Fw[0] = W̄·u[0],
        # Fw[nz] = 0). Keep the vertical contribution for single-layer columns.
        tend_u = tend_u + self._u_tendency_vertical(u, w, grid, geometry=geometry)
        tend_v = tend_v + self._v_tendency_vertical(v, w, grid, geometry=geometry)

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=tend_u,
            v=tend_v,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )

    # ------------------------------------------------------------------
    # Horizontal u-momentum tendency (volume-flux form)
    # ------------------------------------------------------------------

    def _linear_free_surface_correction_active(self) -> bool:
        """``select_rStar.EQ.0 .AND. .NOT.rigidLid`` of ``mom_u_adv_wu.F:93``,
        ``mom_v_adv_wv.F:94``; every LOCOMgcm stepper has a free surface."""
        return self.select_r_star == 0

    @staticmethod
    def _center_wet_mask(
        grid: StructuredCGrid,
        *,
        nz: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        hFacC = getattr(grid.masks, "hFacC", None)
        if hFacC is not None:
            wet = (hFacC.to(dtype=dtype, device=device) > 0).to(dtype=dtype)
        else:
            wet = grid.masks.center.to(dtype=dtype, device=device)
            if wet.ndim == 2:
                wet = wet.unsqueeze(0).expand(nz, -1, -1)
        if wet.ndim != 3:
            raise ValueError(f"expected 3D center wet mask, got shape {tuple(wet.shape)}")
        return wet.unsqueeze(0)

    @staticmethod
    def _u_free_surface_flux_correction(
        u: torch.Tensor,
        r_trans: torch.Tensor,
        grid: StructuredCGrid,
        *,
        center_wet: torch.Tensor,
        face_area: torch.Tensor,
    ) -> torch.Tensor:
        """MITgcm-style one-sided interface correction for wu fluxes.

        ``mom_u_adv_wu.F`` adds a correction on interior interfaces where the
        cell above is dry and the cell below is wet, so the centered
        ``0.5*(u[k-1] + u[k])`` stencil collapses to the one-sided
        ``u[k]`` flux used at the first wet interface.
        """
        batch, nz, ny, nxp1 = u.shape
        nx = grid.nx
        if nz <= 1:
            return u.new_zeros(batch, nz + 1, ny, nxp1)

        lower_u = u[:, 1:, :, :]
        x_halo = HaloExtentV1(
            west_left_extent=1,
            east_right_extent=1,
            south_bottom_extent=0,
            north_top_extent=0,
        )
        q = surface_correction_source(r_trans, center_wet, nz=nz)
        q_x = exchange(
            q,
            grid=grid,
            location="W",
            extent=x_halo,
            fill=FillSpec.REPLICATE,
        )
        x_pair_mask = _topology_edge_pair_mask(
            nx,
            grid=grid,
            extent=x_halo,
            device=u.device,
        )
        # The east edge u-face value depends on topology.  Route the
        # coupled-edge neighbour through exchange() instead of hard-indexing
        # column 0: on a periodic axis the east ghost is the wrap (column 0);
        # on a future cubed-sphere seam it is the neighbouring face's u-value.
        # When the edge does not couple (wall / open) keep the stored
        # boundary u-face lower_u[..., nx].
        lower_u_inner = lower_u[:, :, :, :nx]
        lower_u_east_ext = exchange(
            lower_u_inner,
            grid=grid,
            location="U",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=1,
                south_bottom_extent=0,
                north_top_extent=0,
            ),
            fill=FillSpec.REPLICATE,
        )
        east_lower_u = torch.where(
            x_pair_mask[-1],
            lower_u_east_ext[:, :, :, nx : nx + 1],
            lower_u[:, :, :, nx : nx + 1],
        )
        lower_u_faces = torch.cat((lower_u_inner, east_lower_u), dim=-1)
        return u_surface_correction(u, q_x, lower_u_faces, face_area=face_area, nx=nx)

    @staticmethod
    def _v_free_surface_flux_correction(
        v: torch.Tensor,
        r_trans: torch.Tensor,
        grid: StructuredCGrid,
        *,
        center_wet: torch.Tensor,
        face_area: torch.Tensor,
    ) -> torch.Tensor:
        """MITgcm-style one-sided interface correction for wv fluxes."""
        batch, nz, nyp1, nx = v.shape
        ny = grid.ny
        if nz <= 1:
            return v.new_zeros(batch, nz + 1, nyp1, nx)

        lower_v = v[:, 1:, :, :]
        y_halo = HaloExtentV1(
            west_left_extent=0,
            east_right_extent=0,
            south_bottom_extent=1,
            north_top_extent=1,
        )
        q = surface_correction_source(r_trans, center_wet, nz=nz)
        q_y = exchange(
            q,
            grid=grid,
            location="W",
            extent=y_halo,
            fill=FillSpec.REPLICATE,
        )
        y_pair_mask = _topology_edge_pair_mask(
            ny,
            grid=grid,
            extent=y_halo,
            device=v.device,
        )
        # The north edge v-face value depends on topology.  Route the
        # coupled-edge neighbour through exchange() instead of hard-indexing
        # row 0: on a periodic axis the north ghost is the wrap (row 0); on a
        # future cubed-sphere seam it is the neighbouring face's v-value.
        # When the edge does not couple (wall / open) keep the stored
        # boundary v-face lower_v[..., ny, :].
        lower_v_inner = lower_v[:, :, :ny, :]
        lower_v_north_ext = exchange(
            lower_v_inner,
            grid=grid,
            location="V",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=0,
                south_bottom_extent=0,
                north_top_extent=1,
            ),
            fill=FillSpec.REPLICATE,
        )
        north_lower_v = torch.where(
            y_pair_mask[-1],
            lower_v_north_ext[:, :, ny : ny + 1, :],
            lower_v[:, :, ny : ny + 1, :],
        )
        lower_v_faces = torch.cat((lower_v_inner, north_lower_v), dim=-2)
        return v_surface_correction(v, q_y, lower_v_faces, face_area=face_area, ny=ny)

    @staticmethod
    def _u_tendency_horizontal(
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
        *,
        geometry: _FluxFormGeometry,
    ) -> torch.Tensor:
        """Compute horizontal advection tendency for u.

        Uses volume-flux weighted form following MITgcm ``mom_fluxform.F``:
            uTrans = u · dy · dz · hFacW
            Fuu(i) = 0.25 · (uTrans(i) + uTrans(i+1)) · (u(i) + u(i+1))

        The raw flux divergence is then normalized by the u-cell volume
        ``dx · dy · dz · hFacW`` (MITgcm: ``recip_rAw * recip_drF *
        recip_hFacW``).
        """
        nx = grid.nx
        ny = grid.ny
        x_halo = HaloExtentV1(
            west_left_extent=1,
            east_right_extent=1,
            south_bottom_extent=0,
            north_top_extent=0,
        )
        y_halo = HaloExtentV1(
            west_left_extent=0,
            east_right_extent=0,
            south_bottom_extent=1,
            north_top_extent=1,
        )
        dz = geometry.layer_thickness
        hfw = geometry.hfac_w

        # ----- Transport on u-faces: uTrans = u * dy * dz * hFacW -----
        uTrans = face_transport(
            u, face_width=geometry.dyG, layer_thickness=dz, face_hfac=hfw,
        )  # [B, nz, ny, nx+1]

        # ----- Fuu at cell centres [*, nz, ny, nx] -----
        Fuu = u_self_flux(uTrans, u)  # [B, nz, ny, nx]
        Fuu_x = exchange(
            Fuu,
            grid=grid,
            location="U",
            extent=x_halo,
            fill=FillSpec.REPLICATE,
        )

        # ----- Fvu at corner points [*, ny+1, nx+1] -----
        #
        # MITgcm mom_u_adv_vu.F uses vTrans for the transport factor.
        # On uniform Cartesian: vTrans = v * dx * dz * hFacS.
        # For simplicity on uniform dx/dy grids, the cross-term flux is:
        #   Fvu = 0.5*(vTrans_j + vTrans_{j+1}) * 0.5*(u_j + u_{j+1})
        # which simplifies when hFacS is uniform to v_corner * u_corner * dx*dz.
        #
        # Here we compute Fvu in "raw" units (area-flux * velocity) and
        # normalize at the end by cell volume.

        # vTrans = v * dxG[j] * dz * hFacS  (dxG = latitude-varying zonal face width at
        # the v-face; a scalar dx biases rows off-centre on a spherical-polar grid)
        vTrans = face_transport(
            v, face_width=geometry.dxG, layer_thickness=dz, face_hfac=geometry.hfac_s,
        )  # [B, nz, ny+1, nx]
        vTrans_f_x = exchange(
            vTrans,
            grid=grid,
            location="F",
            extent=x_halo,
            fill=FillSpec.REPLICATE,
        )
        u_f_y = exchange(
            u,
            grid=grid,
            location="F",
            extent=y_halo,
            fill=FillSpec.REPLICATE,
        )

        # Corners: j in [0..ny], i in [0..nx].  The support mask is routed
        # through exchange so non-periodic edge/corner fluxes stay exact +0.0.
        # The north corner row of vT and the east corner column of u depend
        # on topology: route them through exchange() instead of hard-indexing
        # row 0 / column 0.  On a periodic axis the ghost is the wrap (row 0 /
        # column 0); on a future cubed-sphere seam it is the neighbouring
        # face's value.  When the edge does not couple (wall / open) the
        # corner_mask below zeroes that row/column regardless of the ghost.
        vT_at_corner = corner_mean_x(vTrans_f_x, ny=ny, nx=nx)
        vT_at_corner = exchange(
            vT_at_corner[..., :ny, :],
            grid=grid,
            location="Z",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=0,
                south_bottom_extent=0,
                north_top_extent=1,
            ),
            fill=FillSpec.REPLICATE,
        )
        u_at_corner = corner_mean_y(u_f_y, ny=ny, nx=nx)
        u_at_corner = exchange(
            u_at_corner[..., :nx],
            grid=grid,
            location="Z",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=1,
                south_bottom_extent=0,
                north_top_extent=0,
            ),
            fill=FillSpec.REPLICATE,
        )
        x_corner_mask = _topology_edge_pair_mask(
            nx,
            grid=grid,
            extent=x_halo,
            device=u.device,
        )
        y_corner_mask = _topology_edge_pair_mask(
            ny,
            grid=grid,
            extent=y_halo,
            device=u.device,
        )
        leading = (1,) * (u_at_corner.ndim - 2)
        corner_mask = (
            x_corner_mask.view(*leading, 1, nx + 1)
            & y_corner_mask.view(*leading, ny + 1, 1)
        )
        Fvu = corner_cross_flux(vT_at_corner, u_at_corner, corner_mask=corner_mask)
        Fvu = apply_corner_flux3_override(
            Fvu,
            vTrans,
            u,
            grid,
            flux_kind="vu",
        )

        # Normalize by u-cell volume: rAw * drF * hFacW  (MITgcm: recip_rAw*recip_drF*recip_hFacW).
        # rAw is the native U-point horizontal area (latitude-varying); a scalar
        # dx*dy biases rows off-centre on a spherical-polar grid.
        vol_u = geometry.rAw * dz * hfw  # [1, nz, ny, nx+1]
        recip_vol_u = _positive_reciprocal(vol_u)

        return u_flux_divergence(Fuu_x, Fvu, recip_volume=recip_vol_u, nx=nx)

    # ------------------------------------------------------------------
    # Horizontal v-momentum tendency (volume-flux form)
    # ------------------------------------------------------------------

    @staticmethod
    def _v_tendency_horizontal(
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
        *,
        geometry: _FluxFormGeometry,
    ) -> torch.Tensor:
        """Compute horizontal advection tendency for v.

        Uses volume-flux weighted form following MITgcm ``mom_fluxform.F``:
            vTrans = v · dx · dz · hFacS
            Fvv(j) = 0.25 · (vTrans(j) + vTrans(j+1)) · (v(j) + v(j+1))

        Raw flux divergence is normalized by v-cell volume
        ``dx · dy · dz · hFacS``.
        """
        nx = grid.nx
        ny = grid.ny
        x_halo = HaloExtentV1(
            west_left_extent=1,
            east_right_extent=1,
            south_bottom_extent=0,
            north_top_extent=0,
        )
        y_halo = HaloExtentV1(
            west_left_extent=0,
            east_right_extent=0,
            south_bottom_extent=1,
            north_top_extent=1,
        )
        dz = geometry.layer_thickness
        hfs = geometry.hfac_s

        # ----- Transport on v-faces: vTrans = v * dxG[j] * dz * hFacS -----
        vTrans = face_transport(
            v, face_width=geometry.dxG, layer_thickness=dz, face_hfac=hfs,
        )  # [B, nz, ny+1, nx]

        # ----- Fvv at cell centres [*, nz, ny, nx] -----
        Fvv = v_self_flux(vTrans, v)
        Fvv_y = exchange(
            Fvv,
            grid=grid,
            location="V",
            extent=y_halo,
            fill=FillSpec.REPLICATE,
        )

        # ----- Fuv at corner points [*, ny+1, nx+1] -----
        # MITgcm: uses uTrans for transport in the cross-term
        uTrans = face_transport(
            u, face_width=geometry.dyG, layer_thickness=dz, face_hfac=geometry.hfac_w,
        )  # [B, nz, ny, nx+1]
        uTrans_f_y = exchange(
            uTrans,
            grid=grid,
            location="F",
            extent=y_halo,
            fill=FillSpec.REPLICATE,
        )
        v_f_x = exchange(
            v,
            grid=grid,
            location="F",
            extent=x_halo,
            fill=FillSpec.REPLICATE,
        )

        # Corners: j in [0..ny], i in [0..nx].  The support mask is routed
        # through exchange so non-periodic edge/corner fluxes stay exact +0.0.
        # The east corner column of uT and the north corner row of v depend
        # on topology: route them through exchange() instead of hard-indexing
        # row 0 / column 0.  On a periodic axis the ghost is the wrap (row 0 /
        # column 0); on a future cubed-sphere seam it is the neighbouring
        # face's value.  When the edge does not couple (wall / open) the
        # corner_mask below zeroes that row/column regardless of the ghost.
        uT_at_corner = corner_mean_y(uTrans_f_y, ny=ny, nx=nx)
        uT_at_corner = exchange(
            uT_at_corner[..., :nx],
            grid=grid,
            location="Z",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=1,
                south_bottom_extent=0,
                north_top_extent=0,
            ),
            fill=FillSpec.REPLICATE,
        )
        v_at_corner = corner_mean_x(v_f_x, ny=ny, nx=nx)
        v_at_corner = exchange(
            v_at_corner[..., :ny, :],
            grid=grid,
            location="Z",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=0,
                south_bottom_extent=0,
                north_top_extent=1,
            ),
            fill=FillSpec.REPLICATE,
        )
        x_corner_mask = _topology_edge_pair_mask(
            nx,
            grid=grid,
            extent=x_halo,
            device=v.device,
        )
        y_corner_mask = _topology_edge_pair_mask(
            ny,
            grid=grid,
            extent=y_halo,
            device=v.device,
        )
        leading = (1,) * (v_at_corner.ndim - 2)
        corner_mask = (
            x_corner_mask.view(*leading, 1, nx + 1)
            & y_corner_mask.view(*leading, ny + 1, 1)
        )
        Fuv = corner_cross_flux(uT_at_corner, v_at_corner, corner_mask=corner_mask)
        Fuv = apply_corner_flux3_override(
            Fuv,
            uTrans,
            v,
            grid,
            flux_kind="uv",
        )

        # Normalize by v-cell volume: rAs * drF * hFacS  (MITgcm: recip_rAs*recip_drF*recip_hFacS).
        # rAs is the NATIVE V-cell horizontal area (MITgcm ini_spherical_polar_grid.F:212-225,
        # exact sine band centred on the V latitude) — NOT the adjacent tracer-area average,
        # which understates the denominator by cos(dlat/2) on a spherical grid and inflated
        # V advection by sec(dlat/2)-1 (tgol 4°: 6.0954e-4).
        vol_v = geometry.rAs * dz * hfs  # [1, nz, ny+1, nx]
        recip_vol_v = _positive_reciprocal(vol_v)

        return v_flux_divergence(Fvv_y, Fuv, recip_volume=recip_vol_v, ny=ny)

    # ------------------------------------------------------------------
    # Vertical u-momentum tendency: δ_k(W̄ᵢ · ū_k)
    # ------------------------------------------------------------------

    def _u_tendency_vertical(
        self,
        u: torch.Tensor,
        w: torch.Tensor,
        grid: StructuredCGrid,
        *,
        geometry: _FluxFormGeometry,
    ) -> torch.Tensor:
        """Compute vertical advection tendency for u.

        Following MITgcm ``mom_u_adv_wu.F`` + ``mom_fluxform.F``:
        - W at u-faces: RU = 0.5 * (w[i-1]*rA[i-1] + w[i]*rA[i]) / rAw
          (rTransU of mom_calc_rtrans.F:95-98, recip_rAw of mom_fluxform.F:535)
        - u at interfaces: u_if = 0.5 * (u[k-1] + u[k])
        - Flux: Fwu = RU * u_if
        - Top: Fwu[0] = RU[0] * u[0]  (one-sided)
        - Bottom: Fwu[nz] = 0
        - Tendency: -(Fwu[k+1] - Fwu[k]) / (dz * hFacW)
        """
        nz = u.shape[-3]
        nx = grid.nx
        # w: [B, nz+1, ny, nx]
        x_halo = HaloExtentV1(
            west_left_extent=1,
            east_right_extent=1,
            south_bottom_extent=0,
            north_top_extent=0,
        )
        rA = grid.metrics.cell_area_for(u.dtype, u.device)            # [ny, nx]
        rTrans = vertical_transport(w, cell_area=rA)                   # [B, nz+1, ny, nx]
        rTrans_x = exchange(
            rTrans,
            grid=grid,
            location="W",
            extent=x_halo,
            fill=FillSpec.REPLICATE,
        )
        w_at_u = u_face_vertical_velocity(
            rTrans_x, face_area=geometry.rAw, nx=nx, dtype=u.dtype,
        )
        Fwu = interface_flux(w_at_u, u)

        if self._linear_free_surface_correction_active():
            center_wet = self._center_wet_mask(
                grid,
                nz=nz,
                dtype=u.dtype,
                device=u.device,
            )
            Fwu = Fwu + self._u_free_surface_flux_correction(
                u,
                rTrans,
                grid,
                center_wet=center_wet,
                face_area=geometry.rAw,
            )

        # Tendency: -(Fwu[k+1] - Fwu[k]) / (dz * hFacW)
        # (MITgcm: recip_drF * recip_hFacW applied in mom_fluxform.F)
        denom = geometry.layer_thickness * geometry.hfac_w
        recip_denom = torch.where(denom > 0, 1.0 / denom, torch.zeros_like(denom))
        return vertical_flux_divergence(Fwu, recip_thickness=recip_denom)

    # ------------------------------------------------------------------
    # Vertical v-momentum tendency: δ_k(W̄ⱼ · v̄_k)
    # ------------------------------------------------------------------

    def _v_tendency_vertical(
        self,
        v: torch.Tensor,
        w: torch.Tensor,
        grid: StructuredCGrid,
        *,
        geometry: _FluxFormGeometry,
    ) -> torch.Tensor:
        """Compute vertical advection tendency for v.

        Following MITgcm ``mom_v_adv_wv.F`` + ``mom_fluxform.F``:
        - W at v-faces: RV = 0.5 * (w[j-1]*rA[j-1] + w[j]*rA[j]) / rAs
        - v at interfaces: v_if = 0.5 * (v[k-1] + v[k])
        - Flux: Fwv = RV * v_if
        - Tendency: -(Fwv[k+1] - Fwv[k]) / (dz * hFacS)
        """
        nz = v.shape[-3]
        ny = grid.ny
        y_halo = HaloExtentV1(
            west_left_extent=0,
            east_right_extent=0,
            south_bottom_extent=1,
            north_top_extent=1,
        )
        # MITgcm mom_v_adv_wv uses the vertical VOLUME TRANSPORT rTrans = wVel*rA (area-
        # weighted), averaged to the v-point and normalised by the V-cell area rAs — NOT a
        # plain w-average.  On a spherical-polar grid rA varies with latitude, so the plain
        # 0.5*(w[j-1]+w[j]) average biases the v vertical advection near steep topography.
        rA = grid.metrics.cell_area_for(v.dtype, v.device)            # [ny, nx]
        # Denominator: native V-cell area rAs (MITgcm recip_rAs, mom_fluxform.F:819-837;
        # exact spherical band ini_spherical_polar_grid.F:212-225) — NOT the adjacent
        # tracer-area average.
        rTrans = vertical_transport(w, cell_area=rA)                   # [B, nz+1, ny, nx]
        rTrans_y = exchange(
            rTrans,
            grid=grid,
            location="W",
            extent=y_halo,
            fill=FillSpec.REPLICATE,
        )
        w_at_v = v_face_vertical_velocity(
            rTrans_y, face_area=geometry.rAs, ny=ny, dtype=v.dtype,
        )
        Fwv = interface_flux(w_at_v, v)

        # MITgcm default: no wet/dry interface masking (see u-tendency comment).

        if self._linear_free_surface_correction_active():
            center_wet = self._center_wet_mask(
                grid,
                nz=nz,
                dtype=v.dtype,
                device=v.device,
            )
            Fwv = Fwv + self._v_free_surface_flux_correction(
                v,
                rTrans,
                grid,
                center_wet=center_wet,
                face_area=geometry.rAs,
            )

        # Tendency: -(Fwv[k+1] - Fwv[k]) / (dz * hFacS)
        denom = geometry.layer_thickness * geometry.hfac_s
        recip_denom = torch.where(denom > 0, 1.0 / denom, torch.zeros_like(denom))
        return vertical_flux_divergence(Fwv, recip_thickness=recip_denom)
