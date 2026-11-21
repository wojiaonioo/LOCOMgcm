"""PRE_CG3D RHS assembly.

``w_vel_mit_from_timestep_wvel`` is MIT upward-positive transient ``wVel``.
The stepper converts LOCOMgcm downward-positive diagnostic interface ``w`` by
negating and slicing before this function is called.
"""
from __future__ import annotations

from typing import Any

import torch

from locomgcm.dycore_parameters import NonhydrostaticParams
from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange


def _time_attr(time_scales: Any, *names: str, default: float = 1.0) -> float:
    for name in names:
        value = getattr(time_scales, name, None)
        if value is not None:
            return float(value)
    return float(default)


def _mask_in_c(grid: StructuredCGrid, ref: torch.Tensor) -> torch.Tensor:
    if grid.masks.maskInC is not None:
        return grid.masks.maskInC.to(dtype=ref.dtype, device=ref.device)
    return torch.ones((grid.ny, grid.nx), dtype=ref.dtype, device=ref.device)


def _mask_c_3d(grid: StructuredCGrid, ref: torch.Tensor) -> torch.Tensor:
    mask = grid.masks.center_float(dtype=ref.dtype, device=ref.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(grid.nz, -1, -1)
    return mask


def _first_wet_index(mask_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    has_wet = mask_c > 0
    wet_col = has_wet.any(dim=0)
    ks = has_wet.to(torch.int64).argmax(dim=0)
    return wet_col, ks


def _has_distributed_neighbor(grid: StructuredCGrid, side: str) -> bool:
    neighbours = getattr(grid, "distributed_neighbors", None) or {}
    entry = neighbours.get(side) if hasattr(neighbours, "get") else None
    return entry is not None


def assemble_cg3d_rhs(
    cg3d_b_init: torch.Tensor,
    cg2d_x: torch.Tensor,
    eta_n: torch.Tensor,
    eta_h: torch.Tensor,
    w_vel_mit_from_timestep_wvel: torch.Tensor,
    phi_nh_init: torch.Tensor,
    empmr: torch.Tensor | None,
    grid: StructuredCGrid,
    time_scales: Any,
    params: NonhydrostaticParams,
    *,
    old_free_surf_term: bool = False,
    using_p_coords: bool = False,
    use_real_freshwater_flux: bool = False,
    fluid_is_water: bool = True,
    free_surf_fac: float = 1.0,
    mass_to_r_unit: float,
    implicit_surf_press: float = 1.0,
    implic_div_2d_flow: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cg3d_b, phi_nh_init_masked)`` for CG3D.

    ``cg2d_x`` must be in MIT surface-potential units.  Under z-coord
    integration the stepper passes ``gravity * eta_new``, not raw ``eta_new``.
    """
    del params
    if cg3d_b_init.ndim != 4:
        raise ValueError(
            f"cg3d_b_init must be [batch,nz,ny,nx], got {tuple(cg3d_b_init.shape)}"
        )
    batch, nz, ny, nx = cg3d_b_init.shape
    if (nz, ny, nx) != (grid.nz, grid.ny, grid.nx):
        raise ValueError("cg3d_b_init shape does not match grid")
    if phi_nh_init.shape != cg3d_b_init.shape:
        raise ValueError("phi_nh_init must match cg3d_b_init")
    if w_vel_mit_from_timestep_wvel.shape != cg3d_b_init.shape:
        raise ValueError("w_vel_mit_from_timestep_wvel must match cg3d_b_init")
    for name, value in {"cg2d_x": cg2d_x, "eta_n": eta_n, "eta_h": eta_h}.items():
        if value.shape != (batch, ny, nx):
            raise ValueError(f"{name} must be [batch,ny,nx], got {tuple(value.shape)}")
    if empmr is not None and empmr.shape != (batch, ny, nx):
        raise ValueError(f"empmr must be [batch,ny,nx], got {tuple(empmr.shape)}")

    dtype = cg3d_b_init.dtype
    device = cg3d_b_init.device
    dt_mom = _time_attr(time_scales, "dt_mom", "delta_t_mom", default=1.0)
    dt_fs = _time_attr(
        time_scales,
        "dt_freesurf",
        "delta_t_freesurf",
        default=dt_mom,
    )
    r_a = grid.metrics.cell_area_for(dtype, device)
    dr_f = grid.layer_thickness_for(dtype, device)
    hfac_w = grid.hfac_w_for(dtype, device)
    hfac_s = grid.hfac_s_for(dtype, device)
    dy_g = grid.metrics.dyG_for(dtype, device)
    dx_g_ext = grid.metrics.dxG_for(dtype, device)
    dx_c = grid.metrics.dxC_for(dtype, device)
    dy_c = grid.metrics.dyC_for(dtype, device)
    mask_in = _mask_in_c(grid, cg3d_b_init)
    mask_c = _mask_c_3d(grid, cg3d_b_init)

    cg3d_b = cg3d_b_init.clone()
    w_surf_p2d = torch.zeros_like(eta_n)
    if (not old_free_surf_term) and implic_div_2d_flow != 0:
        # eta uses the existing positive-up free-surface convention; no sign flip.
        w_surf_p2d = (eta_n - eta_h) / (float(implic_div_2d_flow) * dt_fs)

    if use_real_freshwater_flux and fluid_is_water:
        emp = torch.zeros_like(eta_n) if empmr is None else empmr.to(dtype=dtype, device=device)
        if old_free_surf_term or using_p_coords:
            ks = grid.nz - 1 if using_p_coords else 0
            tmp_fac = (
                float(free_surf_fac)
                * float(mass_to_r_unit)
                * float(implic_div_2d_flow)
                / dt_mom
            )
            cg3d_b[:, ks] = cg3d_b[:, ks] + tmp_fac * r_a.unsqueeze(0) * emp * mask_in.unsqueeze(0)
        else:
            w_surf_p2d = w_surf_p2d + emp * float(mass_to_r_unit) * mask_in.unsqueeze(0)

    wet_col, ks = _first_wet_index(mask_c)
    if old_free_surf_term:
        # ``pre_cg3d.F:147-149`` sets ``surfFac = freeSurfFac*deepFac2F(1)``
        # for the uniform surface level (z-coordinate) and the k=1 row at
        # ``:203`` then carries ``surfFac*etaN/deltaTFreeSurf*rA/deltaTMom``;
        # ``wSurfP2d`` stays zero (``:73-78``), so the ``.NOT.oldFreeSurfTerm``
        # form below contributes nothing on this arm.  ``:152-159``, the
        # non-uniform surface-level arm, is unreachable in z coordinates.
        # MITgcm writes the k=1 row unconditionally while this adds at ks; the
        # two agree because a uniform surface level means ks=0 in every wet
        # column, and dry columns are zeroed by the mask at the end either way.
        surf_add = (
            float(free_surf_fac) * eta_n / dt_fs * r_a.unsqueeze(0) / dt_mom
        )
    else:
        surf_add = (
            float(free_surf_fac)
            * float(implic_div_2d_flow)
            / dt_mom
            * w_surf_p2d
            * r_a.unsqueeze(0)
        )
    levels = torch.arange(nz, device=device).view(nz, 1, 1)
    at_surface = (wet_col & (ks == levels)).unsqueeze(0)
    cg3d_b = torch.where(at_surface, cg3d_b + surf_add.unsqueeze(1), cg3d_b)

    uf = torch.zeros((batch, ny, nx + 1), dtype=dtype, device=device)
    vf = torch.zeros((batch, ny + 1, nx), dtype=dtype, device=device)
    if nx > 1:
        face_mask = mask_in[:, 1:nx] * mask_in[:, : nx - 1]
        face_flux = (
            -float(implicit_surf_press)
            * float(implic_div_2d_flow)
            * (cg2d_x[:, :, 1:] - cg2d_x[:, :, : nx - 1])
            / dx_c[:, 1:nx].unsqueeze(0)
            * face_mask.unsqueeze(0)
        )
        uf[:, :, 1:nx] = torch.where(
            face_mask.unsqueeze(0) != 0,
            face_flux,
            torch.zeros_like(face_flux),
        )
    x_edge_couples = bool(grid.edge_face_couples("x"))
    west_couples = x_edge_couples or _has_distributed_neighbor(grid, "west")
    east_couples = x_edge_couples or _has_distributed_neighbor(grid, "east")
    if west_couples or east_couples:
        x_halo = exchange(
            cg2d_x,
            grid=grid,
            location="T",
            extent=HaloExtentV1(
                west_left_extent=1,
                east_right_extent=1,
                south_bottom_extent=0,
                north_top_extent=0,
            ),
            fill=FillSpec.ZERO,
        )
        mask_halo_x = exchange(
            mask_in,
            grid=grid,
            location="T",
            extent=HaloExtentV1(
                west_left_extent=1,
                east_right_extent=1,
                south_bottom_extent=0,
                north_top_extent=0,
            ),
            fill=FillSpec.ZERO,
        )
    else:
        x_halo = None
        mask_halo_x = None
    if x_halo is not None and mask_halo_x is not None and west_couples:
        face_mask = mask_halo_x[:, 0] * mask_halo_x[:, 1]
        face_flux = (
            -float(implicit_surf_press)
            * float(implic_div_2d_flow)
            * (x_halo[:, :, 1] - x_halo[:, :, 0])
            / dx_c[:, 0].unsqueeze(0)
            * face_mask.unsqueeze(0)
        )
        uf[:, :, 0] = torch.where(
            face_mask.unsqueeze(0) != 0,
            face_flux,
            torch.zeros_like(face_flux),
        )
    if x_halo is not None and mask_halo_x is not None and east_couples:
        face_mask = mask_halo_x[:, nx] * mask_halo_x[:, nx + 1]
        face_flux = (
            -float(implicit_surf_press)
            * float(implic_div_2d_flow)
            * (x_halo[:, :, nx + 1] - x_halo[:, :, nx])
            / dx_c[:, nx].unsqueeze(0)
            * face_mask.unsqueeze(0)
        )
        uf[:, :, nx] = torch.where(
            face_mask.unsqueeze(0) != 0,
            face_flux,
            torch.zeros_like(face_flux),
        )
    if ny > 1:
        face_mask = mask_in[1:ny, :] * mask_in[: ny - 1, :]
        face_flux = (
            -float(implicit_surf_press)
            * float(implic_div_2d_flow)
            * (cg2d_x[:, 1:ny, :] - cg2d_x[:, : ny - 1, :])
            / dy_c[1:ny, :]
            * face_mask.unsqueeze(0)
        )
        vf[:, 1:ny, :] = torch.where(
            face_mask.unsqueeze(0) != 0,
            face_flux,
            torch.zeros_like(face_flux),
        )
    y_edge_couples = bool(grid.edge_face_couples("y"))
    south_couples = y_edge_couples or _has_distributed_neighbor(grid, "south")
    north_couples = y_edge_couples or _has_distributed_neighbor(grid, "north")
    if south_couples or north_couples:
        y_halo = exchange(
            cg2d_x,
            grid=grid,
            location="T",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=0,
                south_bottom_extent=1,
                north_top_extent=1,
            ),
            fill=FillSpec.ZERO,
        )
        mask_halo_y = exchange(
            mask_in,
            grid=grid,
            location="T",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=0,
                south_bottom_extent=1,
                north_top_extent=1,
            ),
            fill=FillSpec.ZERO,
        )
    else:
        y_halo = None
        mask_halo_y = None
    if y_halo is not None and mask_halo_y is not None and south_couples:
        face_mask = mask_halo_y[0, :] * mask_halo_y[1, :]
        face_flux = (
            -float(implicit_surf_press)
            * float(implic_div_2d_flow)
            * (y_halo[:, 1, :] - y_halo[:, 0, :])
            / dy_c[0, :]
            * face_mask.unsqueeze(0)
        )
        vf[:, 0, :] = torch.where(
            face_mask.unsqueeze(0) != 0,
            face_flux,
            torch.zeros_like(face_flux),
        )
    if y_halo is not None and mask_halo_y is not None and north_couples:
        face_mask = mask_halo_y[ny, :] * mask_halo_y[ny + 1, :]
        face_flux = (
            -float(implicit_surf_press)
            * float(implic_div_2d_flow)
            * (y_halo[:, ny + 1, :] - y_halo[:, ny, :])
            / dy_c[ny, :]
            * face_mask.unsqueeze(0)
        )
        vf[:, ny, :] = torch.where(
            face_mask.unsqueeze(0) != 0,
            face_flux,
            torch.zeros_like(face_flux),
        )

    u_flux = uf.unsqueeze(1) * dr_f.view(1, nz, 1, 1) * hfac_w.unsqueeze(0) * dy_g
    v_flux = (
        vf.unsqueeze(1)
        * dr_f.view(1, nz, 1, 1)
        * hfac_s.unsqueeze(0)
        * dx_g_ext.view(1, 1, ny + 1, nx)
    )
    hdiv_integrated = (u_flux[..., 1:] - u_flux[..., :-1]) + (
        v_flux[..., 1:, :] - v_flux[..., :-1, :]
    )
    cg3d_b = cg3d_b + hdiv_integrated

    w = w_vel_mit_from_timestep_wvel
    vertical = torch.zeros_like(cg3d_b)
    if nz > 1:
        vertical[:, 0] = -w[:, 1]
        # Independent levels must remain one slice operation under compile;
        # unrolled writes otherwise load the entire column for each output.
        vertical[:, 1:-1] = w[:, 1:-1] * mask_c[:-2].unsqueeze(0) - w[:, 2:]
        vertical[:, nz - 1] = w[:, nz - 1] * mask_c[nz - 2].unsqueeze(0)
    # w is MIT upward-positive; PRE_CG3D consumes it without negation.
    cg3d_b = cg3d_b + vertical * r_a.view(1, 1, ny, nx) / dt_mom

    mask4 = (mask_c * mask_in.unsqueeze(0)).unsqueeze(0)
    phi_nh_init_masked = phi_nh_init * mask4
    return cg3d_b * mask4, phi_nh_init_masked
