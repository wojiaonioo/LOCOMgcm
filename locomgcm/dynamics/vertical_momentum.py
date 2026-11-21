"""Vertical momentum tendency for nonhydrostatic stepping.

Inputs and outputs use MITgcm's upward-positive ``wVel`` convention.
LOCOMgcm diagnostic vertical velocity is downward-positive; the stepper must
negate and slice ``diagnose_vertical_velocity`` output before calling here.
"""
from __future__ import annotations

import torch

from locomgcm.dycore_parameters import NonhydrostaticParams
from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from locomgcm.state import PrognosticState

from .nh_coriolis import w_coriolis_scheme1
from .vector_invariant_momentum import (
    CORI_3D_SCHEMES,
    _scheme_name,
    _u_coriolis_nh_runtime_f_cori_cos,
)
from .w_advection import horizontal_w_advection
from .w_viscosity import (
    add_biharmonic_w_fluxes,
    biharmonic_wall_stress,
    calc_gw_vertical_viscosity,
    harmonic_w_fluxes,
    harmonic_wall_stress,
    viscous_w_tendency,
    w_laplacian,
    w_side_drag,
)
from .w_control_volume import (
    w_cell_thickness_center,
    w_column_bounds,
    w_face_thickness_x,
    w_face_thickness_y,
    w_recip_volume,
)
from locomgcm.constants import omega_from_rotation_period


def _has_distributed_neighbor(grid: StructuredCGrid, side: str) -> bool:
    neighbours = getattr(grid, "distributed_neighbors", None) or {}
    entry = neighbours.get(side) if hasattr(neighbours, "get") else None
    return entry is not None


def _exchange_center_x(field: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    return exchange(
        field,
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


def _exchange_center_y(field: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    return exchange(
        field,
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


def _center_mask(grid: StructuredCGrid, ref: torch.Tensor) -> torch.Tensor:
    mask = grid.masks.center_float(dtype=ref.dtype, device=ref.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(grid.nz, -1, -1)
    return mask.unsqueeze(0)


def _batch_like(
    value: torch.Tensor,
    ref: torch.Tensor,
) -> torch.Tensor:
    out = value.to(dtype=ref.dtype, device=ref.device)
    if out.shape == ref.shape[1:]:
        out = out.unsqueeze(0).expand_as(ref)
    if out.shape != ref.shape:
        raise ValueError(f"optional tensor must match {tuple(ref.shape)} or {tuple(ref.shape[1:])}")
    return out


def _w_cell_thickness_center(
    grid: StructuredCGrid,
    ref: torch.Tensor,
    *,
    wet_only: bool = True,
) -> torch.Tensor:
    """Return MIT W-cell thickness at center W levels.

    ``wet_only=False`` returns CALC_GW's unmasked ``rThickC_C`` geometry
    (``w_cell_thickness_center``); the default multiplies it by the centre
    wet mask.
    """
    dtype = ref.dtype
    device = ref.device
    nz = grid.nz
    dr_f = grid.layer_thickness_for(dtype, device)
    r_c = -grid.center_depth.to(dtype=dtype, device=device)
    if grid.masks.hFacC is not None:
        hfac_c = grid.masks.hFacC.to(dtype=dtype, device=device)
    else:
        hfac_c = torch.ones((nz, grid.ny, grid.nx), dtype=dtype, device=device)
    r_low, ro_surf = w_column_bounds(dr_f, hfac_c, nz=nz)
    thickness = w_cell_thickness_center(r_c, r_low, ro_surf)
    if wet_only:
        return thickness * _center_mask(grid, ref)[0]
    return thickness


def _w_cell_thickness_w_face(grid: StructuredCGrid, ref: torch.Tensor) -> torch.Tensor:
    dtype = ref.dtype
    device = ref.device
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    dr_f = grid.layer_thickness_for(dtype, device)
    r_c = -grid.center_depth.to(dtype=dtype, device=device)
    if grid.masks.hFacC is not None:
        hfac_c = grid.masks.hFacC.to(dtype=dtype, device=device)
    else:
        hfac_c = torch.ones((nz, ny, nx), dtype=dtype, device=device)
    # hFac-resynced bottom (mirror _w_cell_thickness_center): derive r_low from
    # the truncated hFacC column depth, not raw depth_center, so ro_surf==0 and
    # the W-face area matches MIT rSurfW/rLowW in CALC_GW. Raw depth leaks the
    # bottom partial-cell truncation into upper-layer face geometry.
    r_low_c, ro_surf_c = w_column_bounds(dr_f, hfac_c, nz=nz)
    edge_couples = grid.w_face_thickness_edge_couples("x")
    west_neighbor = _has_distributed_neighbor(grid, "west")
    east_neighbor = _has_distributed_neighbor(grid, "east")
    r_low_halo = None
    if west_neighbor or east_neighbor:
        r_low_halo = _exchange_center_x(r_low_c, grid)
    return w_face_thickness_x(
        dr_f=dr_f,
        r_c=r_c,
        r_low_c=r_low_c,
        ro_surf_c=ro_surf_c,
        hfac_w=grid.hfac_w_for(dtype, device),
        edge_couples=edge_couples,
        r_low_halo=r_low_halo,
        west_neighbor=west_neighbor,
        east_neighbor=east_neighbor,
    )


def _w_cell_thickness_s_face(grid: StructuredCGrid, ref: torch.Tensor) -> torch.Tensor:
    dtype = ref.dtype
    device = ref.device
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    dr_f = grid.layer_thickness_for(dtype, device)
    r_c = -grid.center_depth.to(dtype=dtype, device=device)
    if grid.masks.hFacC is not None:
        hfac_c = grid.masks.hFacC.to(dtype=dtype, device=device)
    else:
        hfac_c = torch.ones((nz, ny, nx), dtype=dtype, device=device)
    # hFac-resynced bottom (mirror _w_cell_thickness_center): derive r_low from
    # the truncated hFacC column depth, not raw depth_center, so ro_surf==0 and
    # the W-face area matches MIT rSurfW/rLowW in CALC_GW. Raw depth leaks the
    # bottom partial-cell truncation into upper-layer face geometry.
    r_low_c, ro_surf_c = w_column_bounds(dr_f, hfac_c, nz=nz)
    edge_couples = grid.w_face_thickness_edge_couples("y")
    south_neighbor = _has_distributed_neighbor(grid, "south")
    north_neighbor = _has_distributed_neighbor(grid, "north")
    r_low_halo = None
    if south_neighbor or north_neighbor:
        r_low_halo = _exchange_center_y(r_low_c, grid)
    return w_face_thickness_y(
        dr_f=dr_f,
        r_c=r_c,
        r_low_c=r_low_c,
        ro_surf_c=ro_surf_c,
        hfac_s=grid.hfac_s_for(dtype, device),
        edge_couples=edge_couples,
        r_low_halo=r_low_halo,
        south_neighbor=south_neighbor,
        north_neighbor=north_neighbor,
    )


def _w_coriolis_scheme1(
    state: PrognosticState,
    grid: StructuredCGrid,
    ref: torch.Tensor,
    *,
    f_cori_cos: float | None,
    omega: float,
) -> torch.Tensor:
    dtype = ref.dtype
    device = ref.device
    batch, nz, ny, nx = ref.shape
    u = state.u.data.to(dtype=dtype, device=device)
    v = state.v.data.to(dtype=dtype, device=device)
    if u.shape != (batch, nz, ny, nx + 1):
        raise ValueError(f"state.u must be [batch,nz,ny,nx+1], got {tuple(u.shape)}")
    if v.shape != (batch, nz, ny + 1, nx):
        raise ValueError(f"state.v must be [batch,nz,ny+1,nx], got {tuple(v.shape)}")

    f_cori_cos = _u_coriolis_nh_runtime_f_cori_cos(
        grid,
        f_cori_cos,
        dtype=dtype,
        device=device,
        ny=ny,
        omega=omega,
    ).view(1, 1, ny, 1)
    return w_coriolis_scheme1(u, ref, f_cori_cos=f_cori_cos)


def _edge_couplings(grid: StructuredCGrid) -> tuple[bool, bool, bool, bool]:
    """``(west, east, south, north)``: the domain-edge face couples two cells.

    Numerics rule 11: a periodic or seam edge (``edge_face_couples``) or a
    distributed rank neighbour on that side.
    """
    x_edge_couples = bool(grid.edge_face_couples("x"))
    west_couples = x_edge_couples or _has_distributed_neighbor(grid, "west")
    east_couples = x_edge_couples or _has_distributed_neighbor(grid, "east")
    y_edge_couples = bool(grid.edge_face_couples("y"))
    south_couples = y_edge_couples or _has_distributed_neighbor(grid, "south")
    north_couples = y_edge_couples or _has_distributed_neighbor(grid, "north")
    return west_couples, east_couples, south_couples, north_couples


def _horizontal_w_advection(
    state: PrognosticState,
    w_vel_mit: torch.Tensor,
    grid: StructuredCGrid,
) -> torch.Tensor:
    dtype = w_vel_mit.dtype
    device = w_vel_mit.device
    batch, nz, ny, nx = w_vel_mit.shape
    u = state.u.data.to(dtype=dtype, device=device)
    v = state.v.data.to(dtype=dtype, device=device)
    if u.shape != (batch, nz, ny, nx + 1):
        raise ValueError(f"state.u must be [batch,nz,ny,nx+1], got {tuple(u.shape)}")
    if v.shape != (batch, nz, ny + 1, nx):
        raise ValueError(f"state.v must be [batch,nz,ny+1,nx], got {tuple(v.shape)}")
    if nz == 1:
        return torch.zeros_like(w_vel_mit)
    west_couples, east_couples, south_couples, north_couples = _edge_couplings(grid)

    dr_f = grid.layer_thickness_for(dtype, device).view(1, nz, 1, 1)
    dy_g = grid.metrics.dyG_for(dtype, device).view(1, 1, ny, nx + 1)
    dx_g = grid.metrics.dxG_for(dtype, device).view(1, 1, ny + 1, nx)
    hfac_w = grid.hfac_w_for(dtype, device).unsqueeze(0)
    hfac_s = grid.hfac_s_for(dtype, device).unsqueeze(0)
    r_a = grid.metrics.cell_area_for(dtype, device).view(1, 1, ny, nx)
    r_thick = _w_cell_thickness_center(grid, w_vel_mit).view(1, nz, ny, nx)
    recip_volume = w_recip_volume(r_thick, cell_area=r_a)

    w_halo_x = None
    if west_couples or east_couples:
        w_halo_x = _exchange_center_x(w_vel_mit[:, 1:], grid)
    w_halo_y = None
    if south_couples or north_couples:
        w_halo_y = _exchange_center_y(w_vel_mit[:, 1:], grid)
    return horizontal_w_advection(
        w_vel_mit,
        u,
        v,
        dr_f=dr_f,
        dy_g=dy_g,
        dx_g=dx_g,
        hfac_w=hfac_w,
        hfac_s=hfac_s,
        r_a=r_a,
        recip_volume=recip_volume,
        w_halo_x=w_halo_x,
        w_halo_y=w_halo_y,
        west_couples=west_couples,
        east_couples=east_couples,
        south_couples=south_couples,
        north_couples=north_couples,
    )


def _interface_viscosity(
    name: str, value: torch.Tensor | None, ref: torch.Tensor, *, u_face: bool
) -> torch.Tensor:
    """MITgcm ``kappaRU`` / ``kappaRV``: face viscosity at the Nr+1 interfaces."""
    batch, nz, ny, nx = ref.shape
    shape = (nz + 1, ny, nx + 1) if u_face else (nz + 1, ny + 1, nx)
    if value is None:
        # the explicit-input requirement: a missing viscosity must not turn into "no term".
        raise ValueError(f"{name} is required when CALC_GW's viscous terms are on")
    out = value.to(dtype=ref.dtype, device=ref.device)
    if tuple(out.shape) == shape:
        out = out.unsqueeze(0).expand((batch, *shape))
    if tuple(out.shape) != (batch, *shape):
        raise ValueError(
            f"{name} must be MITgcm's face viscosity at the nz+1 interfaces, "
            f"{(batch, *shape)} or {shape}, got {tuple(out.shape)}"
        )
    return out


def _viscous_w_dissipation(
    w_vel_mit: torch.Tensor,
    grid: StructuredCGrid,
    *,
    horizontal_viscosity: torch.Tensor,
    kappa_ru: torch.Tensor,
    kappa_rv: torch.Tensor,
    no_slip_sides: bool,
    side_drag_factor: float = 2.0,
    horizontal_biharmonic_viscosity: torch.Tensor | None,
) -> torch.Tensor:
    dtype = w_vel_mit.dtype
    device = w_vel_mit.device
    batch, nz, ny, nx = w_vel_mit.shape
    if nz == 1:
        return torch.zeros_like(w_vel_mit)
    west_couples, east_couples, south_couples, north_couples = _edge_couplings(grid)
    r_a = grid.metrics.cell_area_for(dtype, device).view(1, ny, nx)
    dx_c = grid.metrics.dxC_for(dtype, device)
    dy_g = grid.metrics.dyG_for(dtype, device)
    dy_c = grid.metrics.dyC_for(dtype, device)
    raw_r_thick = _w_cell_thickness_center(grid, w_vel_mit, wet_only=False)
    r_thick = raw_r_thick * _center_mask(grid, w_vel_mit)[0]
    recip_volume = w_recip_volume(r_thick, cell_area=r_a).unsqueeze(0)
    x_area = _w_cell_thickness_w_face(grid, w_vel_mit) * dy_g
    y_area = _w_cell_thickness_s_face(grid, w_vel_mit) * grid.metrics.dxG_for(
        dtype, device,
    ).view(1, ny + 1, nx)

    visc_h = horizontal_viscosity
    visc4 = horizontal_biharmonic_viscosity  # MIT viscA4_W; None => del^4(w) skipped
    visc_loc, visc_top = calc_gw_vertical_viscosity(kappa_ru, kappa_rv)
    dr_f = grid.layer_thickness_for(dtype, device)

    w_halo_x = visc_halo_x = None
    if west_couples or east_couples:
        w_halo_x = _exchange_center_x(w_vel_mit[:, 1:], grid)
        visc_halo_x = _exchange_center_x(visc_h[:, 1:], grid)
    w_halo_y = visc_halo_y = None
    if south_couples or north_couples:
        w_halo_y = _exchange_center_y(w_vel_mit[:, 1:], grid)
        visc_halo_y = _exchange_center_y(visc_h[:, 1:], grid)
    flx_ew, flx_ns = harmonic_w_fluxes(
        w_vel_mit,
        visc_h,
        dx_c=dx_c,
        dy_c=dy_c,
        x_area=x_area,
        y_area=y_area,
        w_halo_x=w_halo_x,
        visc_halo_x=visc_halo_x,
        w_halo_y=w_halo_y,
        visc_halo_y=visc_halo_y,
        west_couples=west_couples,
        east_couples=east_couples,
        south_couples=south_couples,
        north_couples=north_couples,
    )

    del2w = None
    if visc4 is not None:
        del2w = w_laplacian(
            w_vel_mit,
            dx_c=dx_c,
            dy_c=dy_c,
            x_area=x_area,
            y_area=y_area,
            recip_volume=recip_volume,
            w_halo_x=w_halo_x,
            w_halo_y=w_halo_y,
            west_couples=west_couples,
            east_couples=east_couples,
            south_couples=south_couples,
            north_couples=north_couples,
        )
        del2w_halo_x = visc4_halo_x = None
        if west_couples or east_couples:
            del2w_halo_x = _exchange_center_x(del2w, grid)
            visc4_halo_x = _exchange_center_x(visc4[:, 1:], grid)
        del2w_halo_y = visc4_halo_y = None
        if south_couples or north_couples:
            del2w_halo_y = _exchange_center_y(del2w, grid)
            visc4_halo_y = _exchange_center_y(visc4[:, 1:], grid)
        flx_ew, flx_ns = add_biharmonic_w_fluxes(
            flx_ew,
            flx_ns,
            del2w,
            visc4,
            dx_c=dx_c,
            dy_c=dy_c,
            x_area=x_area,
            y_area=y_area,
            del2w_halo_x=del2w_halo_x,
            visc4_halo_x=visc4_halo_x,
            del2w_halo_y=del2w_halo_y,
            visc4_halo_y=visc4_halo_y,
            west_couples=west_couples,
            east_couples=east_couples,
            south_couples=south_couples,
            north_couples=north_couples,
        )

    gw_diss = viscous_w_tendency(
        w_vel_mit,
        flx_ew,
        flx_ns,
        visc_loc=visc_loc,
        visc_top=visc_top,
        dr_f=dr_f,
        r_a=r_a,
        recip_volume=recip_volume,
    )

    if no_slip_sides and nx > 0 and ny > 0:
        thickness = raw_r_thick[1:]
        thickness_x = _exchange_center_x(thickness, grid)
        thickness_y = _exchange_center_y(thickness, grid)
        # ZERO fills describe a solid exterior only on WALL axes. OBCS owns
        # open exteriors; internal coasts and coupled rank/periodic edges
        # retain their actual geometric height differences.
        x_wall = grid.physical_wall("x")
        y_wall = grid.physical_wall("y")
        if visc4 is None:
            wall_stress = harmonic_wall_stress(w_vel_mit, visc_h)
        else:
            wall_stress = biharmonic_wall_stress(
                w_vel_mit, visc_h, visc4=visc4, del2w=del2w,
            )
        gw_diss = w_side_drag(
            gw_diss,
            wall_stress=wall_stress,
            thickness=thickness,
            thickness_x=thickness_x,
            thickness_y=thickness_y,
            dx_c=dx_c,
            dy_c=dy_c,
            dy_g=dy_g,
            dx_g=grid.metrics.dxG_for(dtype, device).view(ny + 1, nx),
            recip_volume=recip_volume,
            side_drag_factor=side_drag_factor,
            zero_west=not x_wall and not west_couples,
            zero_east=not x_wall and not east_couples,
            zero_south=not y_wall and not south_couples,
            zero_north=not y_wall and not north_couples,
        )
    return gw_diss


def compute_gw(
    state: PrognosticState,
    *,
    w_vel_mit: torch.Tensor,
    grid: StructuredCGrid,
    params: NonhydrostaticParams,
    mom_advection: bool = True,
    mom_viscosity: bool = True,
    no_slip_sides: bool = True,
    side_drag_factor: float = 2.0,
    use_smag_3d: bool = False,
    use_nh_m_terms: bool = False,
    select_3d_cori_scheme: int = 0,
    # Resolved MIT fCoriCos (None: latitude-based); always stated by the caller.
    f_cori_cos: float | None,
    kappa_ru: torch.Tensor | None = None,
    kappa_rv: torch.Tensor | None = None,
    viscAh3d_00: torch.Tensor,
    viscA4_W: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the masked current ``gW`` (advection and Coriolis) and ``gW_diss``.

    The stepper combines them with the ``gwNm1`` history (Adams-Bashforth).

    ``kappa_ru`` / ``kappa_rv`` are MITgcm's ``kappaRU`` / ``kappaRV`` as CALC_GW
    receives them (dynamics.F:645-649): the total vertical viscosity, KPP
    included, on U faces ``[batch, nz+1, ny, nx+1]`` and V faces
    ``[batch, nz+1, ny+1, nx]`` at every interface down to the bottom one
    (calc_viscosity.F:66-72 sets it to viscArNr(Nr)).  Required when
    ``mom_viscosity`` is on; the centre averages are taken here, as
    calc_gw.F:407-411 and :427-429 do.

    ``viscAh3d_00`` is MIT ``viscAh_W`` (zero where there is no horizontal
    viscosity).  ``viscA4_W`` is MIT's biharmonic ``viscA4_W``; the caller
    passes None to state that CALC_GW's biharmonic term is off.
    """
    if w_vel_mit.ndim != 4:
        raise ValueError(f"w_vel_mit must be [batch,nz,ny,nx], got {tuple(w_vel_mit.shape)}")
    if w_vel_mit.shape[1:] != (grid.nz, grid.ny, grid.nx):
        raise ValueError("w_vel_mit shape must match grid [nz,ny,nx]")
    omega = omega_from_rotation_period(params.rotation_period)
    resolved_3d_cori_scheme = int(select_3d_cori_scheme)
    if use_nh_m_terms:
        raise NotImplementedError("useNHMTerms is not implemented")
    cori_3d_scheme = _scheme_name(
        CORI_3D_SCHEMES, resolved_3d_cori_scheme, "select_3d_cori_scheme")
    if use_smag_3d:
        # CALC_GW's 3-D Smagorinsky terms (mom_w_smag_3d.F) are not implemented.
        raise NotImplementedError("useSmag3D is not implemented in compute_gw")
    if mom_viscosity:
        kappa_ru = _interface_viscosity("kappa_ru", kappa_ru, w_vel_mit, u_face=True)
        kappa_rv = _interface_viscosity("kappa_rv", kappa_rv, w_vel_mit, u_face=False)
        viscAh3d_00 = _batch_like(viscAh3d_00, w_vel_mit)

    mask = _center_mask(grid, w_vel_mit)
    gW_current = torch.zeros_like(w_vel_mit)
    gw_diss = torch.zeros_like(w_vel_mit)
    if mom_advection:
        # _horizontal_w_advection consumes MIT upward-positive wVel directly.
        gW_current = gW_current + _horizontal_w_advection(state, w_vel_mit, grid)
    if mom_viscosity:
        # Viscous W fluxes also consume MIT upward-positive wVel directly.
        gw_diss = _viscous_w_dissipation(
            w_vel_mit,
            grid,
            horizontal_viscosity=viscAh3d_00,
            kappa_ru=kappa_ru,
            kappa_rv=kappa_rv,
            no_slip_sides=no_slip_sides,
            side_drag_factor=side_drag_factor,
            horizontal_biharmonic_viscosity=viscA4_W,
        )
    if cori_3d_scheme == "none":
        pass
    elif cori_3d_scheme == "original":
        # MOM_W_CORIOLIS_NH scheme 1, non-rotated z-coordinate grid:
        # -gravitySign*fCoriCos*uMean = +fCoriCos*uMean below the surface.
        gW_current = gW_current + _w_coriolis_scheme1(
            state, grid, w_vel_mit, f_cori_cos=f_cori_cos, omega=omega)

    return gW_current * mask, gw_diss * mask
