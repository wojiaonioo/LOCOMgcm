from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch

from defgcm.geometry.metric_bundle import _positive_reciprocal

from defgcm.geometry import StructuredCGrid
from defgcm.geometry.grid import top_layer_thickness_value
from defgcm.state import DiagnosticState, PrognosticState, StateTendency
from defgcm.constants import RHO_CONST_DEFAULT

from .base import MomentumTermClass, TendencyModule, TendencyStepDomain
from defgcm.dynamics._registry import register_dynamics_module


def local_surface_wind_tendency_from_face_stress(
    tau_x: torch.Tensor,
    tau_y: torch.Tensor,
    *,
    hfac_w: torch.Tensor,
    hfac_s: torch.Tensor,
    layer_thickness: torch.Tensor,
    reference_density: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place face stress at each column's first wet W/S level.

    ``tau_x/y`` have shape ``[..., ny, nx+1]`` / ``[..., ny+1, nx]``;
    ``hfac_w/s`` add the surface-first vertical axis at ``dim=-3``.  This is
    MITgcm's ``kSurface=-1`` ownership rule: each face uses its own ``kSurfW``
    or ``kSurfS``, together with that level's ``drF`` and reciprocal hFac.
    """
    for name, stress, hfac in (
        ("tau_x", tau_x, hfac_w),
        ("tau_y", tau_y, hfac_s),
    ):
        if not isinstance(stress, torch.Tensor) or not isinstance(hfac, torch.Tensor):
            raise TypeError(f"{name} and its hFac must be torch.Tensor objects")
        if hfac.ndim != stress.ndim + 1:
            raise ValueError(
                f"{name} hFac must add one vertical axis; got stress "
                f"{tuple(stress.shape)} and hFac {tuple(hfac.shape)}"
            )
        if tuple(hfac.shape[:-3]) != tuple(stress.shape[:-2]) or tuple(
            hfac.shape[-2:]
        ) != tuple(stress.shape[-2:]):
            raise ValueError(f"{name} and its hFac horizontal/leading shapes disagree")
        if stress.dtype != hfac.dtype or stress.device != hfac.device:
            raise TypeError(f"{name} and its hFac must share dtype and device")
    nz = hfac_w.shape[-3]
    if hfac_s.shape[-3] != nz:
        raise ValueError("hfac_w and hfac_s must have the same vertical length")
    if tuple(layer_thickness.shape) != (nz,):
        raise ValueError(f"layer_thickness must have shape ({nz},)")
    if layer_thickness.dtype != tau_x.dtype or layer_thickness.device != tau_x.device:
        raise TypeError("layer_thickness must share stress dtype and device")
    rho0 = float(reference_density)
    if not math.isfinite(rho0) or rho0 <= 0.0:
        raise ValueError("reference_density must be finite and positive")

    def apply(stress: torch.Tensor, hfac: torch.Tensor) -> torch.Tensor:
        wet = hfac > 0
        first_wet = wet & (torch.cumsum(wet.to(torch.int64), dim=-3) == 1)
        reciprocal_hfac = torch.where(
            wet,
            1.0 / torch.where(wet, hfac, torch.ones_like(hfac)),
            torch.zeros_like(hfac),
        )
        drf = layer_thickness.view(
            *([1] * (hfac.ndim - 3)), nz, 1, 1
        )
        return (
            stress.unsqueeze(-3)
            / (rho0 * drf)
            * reciprocal_hfac
            * first_wet.to(dtype=stress.dtype)
        )

    return apply(tau_x, hfac_w), apply(tau_y, hfac_s)


def surface_wind_tendency_from_face_stress(
    tau_x: torch.Tensor,
    tau_y: torch.Tensor,
    *,
    hFacW0: torch.Tensor,
    hFacS0: torch.Tensor,
    rho0: float,
    drF0: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-layer momentum tendencies from compact C-grid face stresses.

    ``tau_x`` and ``tau_y`` are compact ``[face, ny, nx]`` face-point stresses.
    They are placed directly on U/V faces without center-to-face averaging; the
    east/north halo faces remain zero for the caller's exchange step to fill.
    ``hFacW0`` and ``hFacS0`` are the surface-layer wet fractions on the target
    U/V faces.
    """
    if tau_x.ndim != 3 or tau_y.ndim != 3:
        raise ValueError("tau_x/tau_y must be [face,ny,nx]")
    if tau_x.shape != tau_y.shape:
        raise ValueError("tau_x and tau_y must have matching shapes")

    face_count, ny, nx = tau_x.shape
    if hFacW0.shape != (face_count, ny, nx + 1):
        raise ValueError("hFacW0 must be [face,ny,nx+1]")
    if hFacS0.shape != (face_count, ny + 1, nx):
        raise ValueError("hFacS0 must be [face,ny+1,nx]")

    tx_u = torch.zeros(face_count, ny, nx + 1, dtype=tau_x.dtype, device=tau_x.device)
    tx_u[:, :, 0:nx] = tau_x
    ty_v = torch.zeros(face_count, ny + 1, nx, dtype=tau_y.dtype, device=tau_y.device)
    ty_v[:, 0:ny, :] = tau_y

    recip_hFacW0 = _positive_reciprocal(hFacW0)
    recip_hFacS0 = _positive_reciprocal(hFacS0)

    wind_u = (tx_u / (float(rho0) * float(drF0))) * recip_hFacW0
    wind_v = (ty_v / (float(rho0) * float(drF0))) * recip_hFacS0
    return wind_u, wind_v


class WindStressProfile(StrEnum):
    UNIFORM = "uniform"
    ZONAL_COSINE = "zonal_cosine"


@dataclass(frozen=True)
@register_dynamics_module("surface_wind_stress")
class SurfaceWindStress(TendencyModule):
    """Surface wind stress applied as a momentum tendency to the top model layer.

    The tendency is computed as (MITgcm ``apply_forcing.F:139-145``):
        du/dt = τˣ(x,y) / (ρ₀ · Δz_top · hFacW_top)
        dv/dt = τʸ(x,y) / (ρ₀ · Δz_top · hFacS_top)
    with the current surface hFac, which a nonlinear free surface rewrites
    every step (zero at a dry face).

    Supported profiles:
        UNIFORM:       τˣ = tau_x,  τʸ = tau_y  (constants)
        ZONAL_COSINE:  τˣ(y) = −tau_x · cos(π·(y − y₀) / Ly),  τʸ = tau_y
                       This is the classic Stommel/Munk gyre forcing.
    """

    tau_x: float = 0.0
    tau_y: float = 0.0
    reference_density: float = RHO_CONST_DEFAULT
    profile: WindStressProfile = WindStressProfile.UNIFORM
    name: str = "surface_wind_stress"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    momentum_term_class = MomentumTermClass.FORCING

    def _stress_fields(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (τˣ, τʸ) as 2-D tensors on cell center grid [ny, nx]."""
        if self.profile == WindStressProfile.UNIFORM:
            tau_x_2d = torch.full((grid.ny, grid.nx), self.tau_x, dtype=dtype, device=device)
            tau_y_2d = torch.full((grid.ny, grid.nx), self.tau_y, dtype=dtype, device=device)
        elif self.profile == WindStressProfile.ZONAL_COSINE:
            if grid.metrics.del_y is None or grid.metrics.is_uniform_y:
                layout = grid.parallel_layout
                y_start = 0 if layout is None else layout.tile_offset_j
                global_ny = grid.ny if layout is None else layout.global_ny
                y_center = (
                    torch.arange(y_start, y_start + grid.ny, dtype=dtype, device=device) + 0.5
                ) * float(grid.dy)
                ly = float(global_ny) * float(grid.dy)
            else:
                del_y = grid.metrics.del_y.to(dtype=dtype, device=device)
                y_center = torch.cumsum(del_y, dim=0) - 0.5 * del_y
                ly = grid.metrics.meridional_length_value(dtype=del_y.dtype, device=del_y.device)
            tau_x_2d = -self.tau_x * torch.cos(math.pi * y_center / ly).view(grid.ny, 1).expand(grid.ny, grid.nx)
            tau_y_2d = torch.full((grid.ny, grid.nx), self.tau_y, dtype=dtype, device=device)
        else:
            raise ValueError(f"unsupported wind stress profile: {self.profile}")
        return tau_x_2d, tau_y_2d

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        del diagnostics
        dtype = state.u.data.dtype
        device = state.u.data.device
        tau_x_2d, tau_y_2d = self._stress_fields(grid, dtype=dtype, device=device)

        top_thickness = top_layer_thickness_value(grid)
        scale = 1.0 / (self.reference_density * top_thickness)

        # Apply momentum tendency only to the top layer.
        # u tendency: on u-face, interpolate tau_x from center to u-face via simple average.
        u_tendency = torch.zeros_like(state.u.data)
        tau_x_on_u = 0.5 * (tau_x_2d[:, :-1] + tau_x_2d[:, 1:])
        if grid.nx + 1 == state.u.data.shape[-1]:
            # Periodic or padded: also handle first/last face
            tau_x_full = torch.zeros(grid.ny, grid.nx + 1, dtype=dtype, device=device)
            tau_x_full[:, 1:-1] = tau_x_on_u
            tau_x_full[:, 0] = 0.5 * (tau_x_2d[:, -1] + tau_x_2d[:, 0])
            tau_x_full[:, -1] = tau_x_full[:, 0]
        else:
            tau_x_full = tau_x_on_u
        recip_w = (1.0 if grid.masks.hFacW is None else
                   _positive_reciprocal(grid.masks.hFacW[0].to(dtype=dtype, device=device)))
        u_tendency[:, 0, :, :] = scale * tau_x_full.unsqueeze(0) * recip_w

        # v tendency: on v-face, interpolate tau_y from center to v-face.
        v_tendency = torch.zeros_like(state.v.data)
        tau_y_on_v = 0.5 * (tau_y_2d[:-1, :] + tau_y_2d[1:, :])
        if grid.ny + 1 == state.v.data.shape[-2]:
            tau_y_full = torch.zeros(grid.ny + 1, grid.nx, dtype=dtype, device=device)
            tau_y_full[1:-1, :] = tau_y_on_v
            tau_y_full[0, :] = 0.5 * (tau_y_2d[-1, :] + tau_y_2d[0, :])
            tau_y_full[-1, :] = tau_y_full[0, :]
        else:
            tau_y_full = tau_y_on_v
        recip_s = (1.0 if grid.masks.hFacS is None else
                   _positive_reciprocal(grid.masks.hFacS[0].to(dtype=dtype, device=device)))
        v_tendency[:, 0, :, :] = scale * tau_y_full.unsqueeze(0) * recip_s

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=u_tendency,
            v=v_tendency,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )

    def evaluate_local_surface(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        """Enabled-path wind forcing using per-face first-wet surface levels."""
        del diagnostics
        dtype = state.u.data.dtype
        device = state.u.data.device
        tau_x_2d, tau_y_2d = self._stress_fields(grid, dtype=dtype, device=device)
        tau_x_on_u = 0.5 * (tau_x_2d[:, :-1] + tau_x_2d[:, 1:])
        if grid.nx + 1 == state.u.data.shape[-1]:
            tau_x_full = torch.zeros(grid.ny, grid.nx + 1, dtype=dtype, device=device)
            tau_x_full[:, 1:-1] = tau_x_on_u
            tau_x_full[:, 0] = 0.5 * (tau_x_2d[:, -1] + tau_x_2d[:, 0])
            tau_x_full[:, -1] = tau_x_full[:, 0]
        else:
            tau_x_full = tau_x_on_u
        tau_y_on_v = 0.5 * (tau_y_2d[:-1, :] + tau_y_2d[1:, :])
        if grid.ny + 1 == state.v.data.shape[-2]:
            tau_y_full = torch.zeros(grid.ny + 1, grid.nx, dtype=dtype, device=device)
            tau_y_full[1:-1, :] = tau_y_on_v
            tau_y_full[0, :] = 0.5 * (tau_y_2d[-1, :] + tau_y_2d[0, :])
            tau_y_full[-1, :] = tau_y_full[0, :]
        else:
            tau_y_full = tau_y_on_v

        hfac_w = grid.masks.hFacW
        hfac_s = grid.masks.hFacS
        hfac_w = (
            torch.ones_like(state.u.data)
            if hfac_w is None
            else hfac_w.to(dtype=dtype, device=device)
        )
        hfac_s = (
            torch.ones_like(state.v.data)
            if hfac_s is None
            else hfac_s.to(dtype=dtype, device=device)
        )
        if hfac_w.ndim == state.u.data.ndim - 1:
            hfac_w = hfac_w.unsqueeze(0)
        if hfac_s.ndim == state.v.data.ndim - 1:
            hfac_s = hfac_s.unsqueeze(0)
        u_tendency, v_tendency = local_surface_wind_tendency_from_face_stress(
            tau_x_full.unsqueeze(0),
            tau_y_full.unsqueeze(0),
            hfac_w=hfac_w,
            hfac_s=hfac_s,
            layer_thickness=grid.layer_thickness.to(dtype=dtype, device=device),
            reference_density=self.reference_density,
        )
        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=u_tendency.expand_as(state.u.data),
            v=v_tendency.expand_as(state.v.data),
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )
