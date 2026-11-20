"""Spherical-polar grid metric momentum tendency (flux-form).

Mirrors MITgcm ``mom_u_metric_sphere.F`` / ``mom_v_metric_sphere.F``
(``selectMetricTerms=1`` analytical ``tanPhi`` branch), the curvature terms that
the vector-invariant formulation absorbs into the KE-gradient + vorticity flux
but the flux-form formulation must add explicitly:

    uMetric(i,j) = + u(i,j) · (1/a) · ¼(v(i,j)+v(i-1,j)+v(i,j+1)+v(i-1,j+1)) · tanφ|_U
    vMetric(i,j) = - (1/a) · [¼(u(i,j)+u(i+1,j)+u(i,j-1)+u(i+1,j-1))]² · tanφ|_V

where ``a`` is the Earth radius (``recip_rSphere = 1/a``), ``tanφ|_U = tan(yC)``
at U-points and ``tanφ|_V = tan(yG)`` at V-points.  The 4-point velocity averages
use MITgcm's **exact flat-sum order** and ``u·recip_rSphere`` is computed by
multiplication (not division) to match FP-floor.  Verified bitwise vs the
compiled-MITgcm ``MOM_U/V_METRIC_SPHERE`` dump on scs-768-5d.

Halo padding mirrors ``momentum_advection`` (``location="F"`` + REPLICATE for
v-in-x and u-in-y).  ``recip_deepFacC = 1`` for the ocean (non-deep-atmosphere).

Upstream reference: MITgcm ``pkg/mom_fluxform/mom_u_metric_sphere.F`` /
``mom_v_metric_sphere.F``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from defgcm.geometry import StructuredCGrid
from defgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from defgcm.accel.constant_cache import constant_cache
from ._host_constants import host_profile_to_tensor
from .base import TendencyModule, TendencyStepDomain
from defgcm.dynamics._registry import register_dynamics_module

# MITgcm default Earth radius (rSphere = 6_370_000 m); recip_rSphere = 1/a.
_RECIP_RSPHERE_DEFAULT = 1.0 / 6_370_000.0


@dataclass(frozen=True)
@register_dynamics_module("spherical_metric")
class SphericalMetric(TendencyModule):
    """Flux-form spherical-polar metric (curvature) momentum tendency.

    Parameters
    ----------
    tan_phi_u : tuple[float, ...]
        ``tan(yC(j))`` at U-points, length ``ny`` (cell-centre latitudes).
    tan_phi_v : tuple[float, ...]
        ``tan(yG(j))`` at V-points, length ``ny+1`` (corner/interface latitudes).
    recip_rsphere : float
        ``1/a`` reciprocal Earth radius; default ``1/6_370_000`` (MITgcm).
    """

    tan_phi_u: tuple[float, ...] | None = None
    tan_phi_v: tuple[float, ...] | None = None
    recip_rsphere: float = _RECIP_RSPHERE_DEFAULT
    name: str = "spherical_metric"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW

    def prepare_host_constants(self, grid, *, dtype, device) -> None:
        """Resolve host profiles before tracing, rebuilding each step if uncached."""
        if self.tan_phi_u is not None and self.tan_phi_v is not None:
            object.__setattr__(self, "_prepared_profiles", (
                grid, dtype, torch.device(device), self._profiles_for(grid, dtype, device),
            ))

    def _profiles_for(self, grid, dtype, device):
        ny = grid.ny
        # Grid-constant profiles: built eagerly once per (dtype, device) and
        # reused from the traced body (the disabled builder was a graph
        # break plus a host->device copy every step).
        tanU = constant_cache.get_or_build(
            ("tan_phi_u", dtype, str(device)),
            lambda: host_profile_to_tensor(self.tan_phi_u, dtype=dtype, device=device),
            variant="host_profile", site="spherical_metric.tan_phi_u", owner=self,
        )
        tanV = constant_cache.get_or_build(
            ("tan_phi_v", dtype, str(device)),
            lambda: host_profile_to_tensor(self.tan_phi_v, dtype=dtype, device=device),
            variant="host_profile", site="spherical_metric.tan_phi_v", owner=self,
        )
        if tanU.shape[0] != ny:
            raise ValueError(f"tan_phi_u length {tanU.shape[0]} != ny={ny}")
        if tanV.shape[0] != ny + 1:
            raise ValueError(f"tan_phi_v length {tanV.shape[0]} != ny+1={ny + 1}")
        return tanU, tanV

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        del diagnostics
        if self.tan_phi_u is None or self.tan_phi_v is None:
            return StateTendency.zeros_like(state)

        u = state.u.data  # [B, nz, ny, nx+1]
        v = state.v.data  # [B, nz, ny+1, nx]
        nx, ny = grid.nx, grid.ny

        prepared = getattr(self, "_prepared_profiles", None)
        if (torch.compiler.is_compiling() and prepared is not None
                and prepared[0] is grid and prepared[1] == u.dtype
                and prepared[2] == u.device):
            tanU, tanV = prepared[3]
        else:
            tanU, tanV = self._profiles_for(grid, u.dtype, u.device)
        rrS = self.recip_rsphere

        x_halo = HaloExtentV1(west_left_extent=1, east_right_extent=1,
                              south_bottom_extent=0, north_top_extent=0)
        y_halo = HaloExtentV1(west_left_extent=0, east_right_extent=0,
                              south_bottom_extent=1, north_top_extent=1)

        # ── u-metric: u·rrS · ¼(v(i,j)+v(i-1,j)+v(i,j+1)+v(i-1,j+1)) · tanφ|_U ──
        v_xpad = exchange(v, grid=grid, location="F", extent=x_halo,
                          fill=FillSpec.REPLICATE)          # [B,nz,ny+1,nx+2]
        v_i = v_xpad[..., 1:nx + 2]
        v_im1 = v_xpad[..., 0:nx + 1]
        v_i_j = v_i[..., 0:ny, :]
        v_i_jp1 = v_i[..., 1:ny + 1, :]
        v_im1_j = v_im1[..., 0:ny, :]
        v_im1_jp1 = v_im1[..., 1:ny + 1, :]
        # MIT metric order: (i,j)+(i-1,j)+(i,j+1)+(i-1,j+1)
        sumv_m = ((v_i_j + v_im1_j) + v_i_jp1) + v_im1_jp1
        vavg_m = 0.25 * sumv_m                               # [B,nz,ny,nx+1]
        tanU_b = tanU.view(1, 1, ny, 1)
        u_tend = ((u * rrS) * vavg_m) * tanU_b

        # ── v-metric: -rrS · [¼(u(i,j)+u(i+1,j)+u(i,j-1)+u(i+1,j-1))]² · tanφ|_V ──
        u_ypad = exchange(u, grid=grid, location="F", extent=y_halo,
                          fill=FillSpec.REPLICATE)           # [B,nz,ny+2,nx+1]
        u_j = u_ypad[..., 1:ny + 2, :]
        u_jm1 = u_ypad[..., 0:ny + 1, :]
        u_j_i = u_j[..., 0:nx]
        u_j_ip1 = u_j[..., 1:nx + 1]
        u_jm1_i = u_jm1[..., 0:nx]
        u_jm1_ip1 = u_jm1[..., 1:nx + 1]
        # MIT order: (i,j)+(i+1,j)+(i,j-1)+(i+1,j-1)
        sumu_m = ((u_j_i + u_j_ip1) + u_jm1_i) + u_jm1_ip1
        uavg_m = 0.25 * sumu_m                               # [B,nz,ny+1,nx]
        tanV_b = tanV.view(1, 1, ny + 1, 1)
        v_tend = (((-rrS) * uavg_m) * uavg_m) * tanV_b

        if grid.masks.has_3d_masks:
            u_mask = grid.masks.u_float(dtype=u.dtype, device=u.device)
            v_mask = grid.masks.v_float(dtype=v.dtype, device=v.device)
            u_tend = u_tend * u_mask
            v_tend = v_tend * v_mask

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=u_tend,
            v=v_tend,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )
