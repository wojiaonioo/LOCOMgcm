"""Topology adapters for horizontal viscosity assembly."""

from __future__ import annotations

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import (
    FillSpec,
    HaloExtentV1,
    _exchange_cubed_sphere_scalar,
    exchange,
)
from locomgcm.operators.c_grid import _curl_cube
from locomgcm.state import PrognosticState, StateTendency

from ._hfacz_topology import build_hfacz_for_grid
from ._vector_mask_topology import mask_vector_tendency_for_grid
from .divvort_viscosity import _divvort_dissipation_from_haloed_hdiv
from .lateral_drag import _side_drag_u_2d, _side_drag_v_2d
from .momentum_invariants import _hdiv_from_face_velocities
from .viscosity_geometry import _recip, _side_drag_closed_fractions


def _has_distributed_neighbor(grid: "StructuredCGrid", side: str) -> bool:
    neighbours = getattr(grid, "distributed_neighbors", None)
    if neighbours is None:
        neighbours = getattr(grid, "distributed_neighbours", None)
    if not neighbours:
        return False
    return neighbours.get(side) is not None if hasattr(neighbours, "get") else False


def _has_any_distributed_neighbor(grid: "StructuredCGrid") -> bool:
    return (
        _has_distributed_neighbor(grid, "west")
        or _has_distributed_neighbor(grid, "east")
        or _has_distributed_neighbor(grid, "south")
        or _has_distributed_neighbor(grid, "north")
    )


def _edge_couples(grid: "StructuredCGrid", side: str) -> bool:
    """Whether the domain-edge face on ``side`` couples two cells (numerics rule 11).

    ``grid.edge_face_couples(axis)`` answers for the tile's own topology: on a
    periodic axis of a single tile ``exchange`` wraps the halo, so the edge face
    is an interior face of the domain.  A distributed neighbour answers for a
    rank seam.  Only a wall or open edge without a neighbour rank leaves the
    face uncoupled.  Asking for the distributed neighbour alone turns a
    single-tile periodic seam into a wall.
    """
    axis = "x" if side in ("west", "east") else "y"
    return bool(grid.edge_face_couples(axis)) or _has_distributed_neighbor(grid, side)


def _any_edge_couples(grid: "StructuredCGrid") -> bool:
    """True when any domain-edge face couples, so seam-aware stencils must run."""
    return (
        _edge_couples(grid, "west")
        or _edge_couples(grid, "east")
        or _edge_couples(grid, "south")
        or _edge_couples(grid, "north")
    )


def _has_internal_distributed_neighbor(grid: "StructuredCGrid", side: str) -> bool:
    if not _has_distributed_neighbor(grid, side):
        return False
    layout = getattr(grid, "parallel_layout", None)
    if layout is None:
        return False
    tile_i = int(getattr(layout, "tile_i", 0))
    tile_j = int(getattr(layout, "tile_j", 0))
    nPx = int(getattr(layout, "nPx", 1))
    nPy = int(getattr(layout, "nPy", 1))
    if side == "west":
        return tile_i > 0
    if side == "east":
        return tile_i < nPx - 1
    if side == "south":
        return tile_j > 0
    if side == "north":
        return tile_j < nPy - 1
    return False


def _u_with_east_global_wrap(u: torch.Tensor, grid: "StructuredCGrid") -> torch.Tensor:
    """Replace the physical east U face with the global west U face."""
    nx = grid.nx
    base = u[..., :nx]
    wrapped = exchange(
        base,
        grid=grid,
        location="T",
        extent=HaloExtentV1(
            west_left_extent=0,
            east_right_extent=1,
            south_bottom_extent=0,
            north_top_extent=0,
        ),
        fill=FillSpec.WRAP,
    )
    if _has_distributed_neighbor(grid, "east"):
        return u
    return torch.cat((base, wrapped[..., nx:nx + 1]), dim=-1)


def _v_with_north_global_wrap(v: torch.Tensor, grid: "StructuredCGrid") -> torch.Tensor:
    """Replace the physical north V face with the global south V face."""
    ny = grid.ny
    base = v[..., :ny, :]
    wrapped = exchange(
        base,
        grid=grid,
        location="T",
        extent=HaloExtentV1(
            west_left_extent=0,
            east_right_extent=0,
            south_bottom_extent=0,
            north_top_extent=1,
        ),
        fill=FillSpec.WRAP,
    )
    if _has_distributed_neighbor(grid, "north"):
        return v
    return torch.cat((base, wrapped[..., ny:ny + 1, :]), dim=-2)



def _divvort_curl_halos(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
    *,
    dxC: torch.Tensor,
    dyC: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``dxC`` and ``u`` with one j-halo row each side, ``v*dyC`` with one i-halo
    column each side: the inputs of the corner vorticity and strain.

    The three exchanges run in this order: ``dxC``, ``u``, ``v*dyC``.
    """
    nx = grid.nx
    ny = grid.ny
    # MITgcm computes vort3 at ALL corners using halo data filled by
    # EXCH (periodic self-wrap on single tile).  Replicate by padding
    # u (j-direction) and v (i-direction) with opposite-side values.
    # dxC_row also needs wrapping for the boundary j indices.
    y_halo = HaloExtentV1(
        west_left_extent=0,
        east_right_extent=0,
        south_bottom_extent=1,
        north_top_extent=1,
    )
    x_halo = HaloExtentV1(
        west_left_extent=1,
        east_right_extent=1,
        south_bottom_extent=0,
        north_top_extent=0,
    )
    dxC_ext = exchange(
        dxC,
        grid=grid,
        location="U",
        extent=y_halo,
        fill=FillSpec.WRAP,
    )  # [ny+2]
    # MITgcm default-periodic EXCH: at the East-OB vorticity corner (i=nx) the
    # strain/vort u-stencil reads the wrapped halo uFld(sNx+1)=uFld(1)=West
    # U-face, NOT the OB normal face u[...,nx]. LOCOMgcm's u[...,nx] carries the
    # OBCS-prescribed East normal velocity, so without this i-wrap the East-OB
    # corner strain feeds a wrong Smag viscAh_D->viscAhW at the OB cell (d0),
    # which the d0/d1 W-viscous flux propagates into gW_visc(d1)->w->cg3d->vel
    # (~6e-5 m/s OB-ring residual). Replicate MIT by wrapping the East-most
    # U-face to the West-edge U-face for the corner stencil. (Checked against
    # the MIT input; the W/S boundaries are already handled by the
    # j-WRAP exchange / v_dyC x-WRAP.)
    #
    # On a distributed internal east seam, the extra local U column is a real
    # shared face and must not be replaced.  On a distributed physical east
    # edge, the MIT-style corner wrap must come from the global west edge,
    # not from the west edge of this local tile.
    u_iwrap = _u_with_east_global_wrap(u, grid)
    u_ext = exchange(
        u_iwrap,
        grid=grid,
        location="U",
        extent=y_halo,
        fill=FillSpec.WRAP,
    )

    # Symmetric to the East u-wrap above: at the North-OB vorticity corner
    # (j=ny) MIT reads the periodic halo vFld(sNy+1)=vFld(1)=South V-face, not
    # the North OB normal face v[...,ny,:]. Wrap it so the North-OB corner
    # strain matches MIT. (East/West handled by u-wrap + this; West/South are
    # the boundary faces themselves, used directly = MIT's uFld(1)/vFld(1).)
    #
    # On a distributed internal north seam, the extra local V row is a real
    # shared face and must not be replaced.  On a distributed physical north
    # edge, the MIT-style corner wrap must come from the global south edge,
    # not from the south edge of this local tile.
    v_jwrap = _v_with_north_global_wrap(v, grid)
    v_dyC = v_jwrap * dyC.view(1, 1, ny + 1, nx)
    v_dyC_ext = exchange(
        v_dyC,
        grid=grid,
        location="V",
        extent=x_halo,
        fill=FillSpec.WRAP,
    )  # [B, nz, ny+1, nx+2]
    return dxC_ext, u_ext, v_dyC_ext


def raw_divvort_for_grid(
    u: torch.Tensor,
    v: torch.Tensor,
    viscosity: torch.Tensor,
    grid: StructuredCGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cube topology inputs and apply the shared MIT div-vort stencil."""
    dtype, device = u.dtype, u.device
    n = grid.nx
    ny = nx = n
    fg = grid.metrics.face_geometry

    hFacC = (
        grid.masks.hFacC.to(dtype=dtype, device=device)
        if grid.masks.hFacC is not None
        else torch.ones(grid.nz, ny, nx, dtype=dtype, device=device)
    )
    hFacW = (
        grid.masks.hFacW.to(dtype=dtype, device=device)
        if grid.masks.hFacW is not None
        else torch.ones(grid.nz, ny, nx + 1, dtype=dtype, device=device)
    )
    hFacS = (
        grid.masks.hFacS.to(dtype=dtype, device=device)
        if grid.masks.hFacS is not None
        else torch.ones(grid.nz, ny + 1, nx, dtype=dtype, device=device)
    )

    dyG_2d = fg["dyG"].to(dtype=dtype, device=device)
    dxG_2d = fg["dxG"].to(dtype=dtype, device=device)
    rA_2d = fg["rA"].to(dtype=dtype, device=device)
    dxC_2d = grid.metrics.dxC_for(dtype, device)
    dyC_2d = grid.metrics.dyC_for(dtype, device)
    dyG_b = dyG_2d.unsqueeze(1) if u.ndim == 4 else dyG_2d
    dxG_b = dxG_2d.unsqueeze(1) if v.ndim == 4 else dxG_2d
    rA_b = _recip(rA_2d).unsqueeze(1) if u.ndim == 4 else _recip(rA_2d)
    dxC_b = dxC_2d.unsqueeze(1) if u.ndim == 4 else dxC_2d
    dyC_b = dyC_2d.unsqueeze(1) if v.ndim == 4 else dyC_2d

    hDiv = _hdiv_from_face_velocities(
        u,
        v,
        dyG=dyG_b,
        dxG=dxG_b,
        hFacW=hFacW,
        hFacS=hFacS,
        recip_rA=rA_b,
        recip_hFacC=_recip(hFacC),
    )

    # Face-local routing: distributed scalar exchange when ctx attached (local
    # grid), else the byte-identical single-rank rot90 backend.
    ctx = getattr(grid, "_cube_step_ctx", None)
    if ctx is not None:
        from locomgcm.geometry.exchange import exchange_cubed_sphere_scalar_distributed
        hDiv_ext = exchange_cubed_sphere_scalar_distributed(
            hDiv, grid=grid, location="T", extent=HaloExtentV1.uniform(1), ctx=ctx,
        )
    else:
        hDiv_ext = _exchange_cubed_sphere_scalar(
            hDiv,
            grid=grid,
            location="T",
            extent=HaloExtentV1.uniform(1),
        )
    vort3 = _curl_cube(u, v, grid)
    hFacZ = build_hfacz_for_grid(grid, dtype, device)

    return _divvort_dissipation_from_haloed_hdiv(
        hDiv_ext=hDiv_ext,
        vort3=vort3,
        hFacZ=hFacZ,
        hFacW=hFacW,
        hFacS=hFacS,
        dxC=dxC_b,
        dyG=dyG_b,
        dyC=dyC_b,
        dxG=dxG_b,
        viscosity=viscosity,
        halo=1,
    )


def divvort_tendency_for_grid(
    state: PrognosticState,
    grid: StructuredCGrid,
    viscosity: torch.Tensor,
    *,
    no_slip_sides: bool,
    side_drag_factor: float,
) -> StateTendency | None:
    """Return the cubed-sphere div-vort viscosity tendency, or ``None``.

    This owns only the topology-specific assembly: paired cube div-vort route,
    optional cube side-drag metrics, final cube mask, and tendency packaging.
    The MIT-aligned raw div-vort stencil is the shared kernel of
    ``divvort_viscosity``.
    """
    if getattr(grid, "face_count", 1) != 6:
        return None

    u = state.u.data
    v = state.v.data
    u_result, v_result = raw_divvort_for_grid(
        u,
        v,
        viscosity,
        grid,
    )

    if no_slip_sides:
        dtype, device = u.dtype, u.device
        fg = grid.metrics.face_geometry
        hFacW = grid.masks.hFacW.to(dtype=dtype, device=device)
        hFacS = grid.masks.hFacS.to(dtype=dtype, device=device)
        hFacZ = build_hfacz_for_grid(grid, dtype, device)
        h0FacW, h0FacS, h0FacZ = _side_drag_closed_fractions(grid, hFacW, hFacS, hFacZ)
        dxV = fg["dxV"].to(dtype=dtype, device=device).unsqueeze(1)
        dyU = fg["dyU"].to(dtype=dtype, device=device).unsqueeze(1)
        rAw = fg["rAw"].to(dtype=dtype, device=device).unsqueeze(1)
        rAs = fg["rAs"].to(dtype=dtype, device=device).unsqueeze(1)
        u_result = u_result + _side_drag_u_2d(
            u,
            hFacW,
            h0FacW,
            h0FacZ,
            rAw,
            dxV,
            dyU,
            viscosity,
            side_drag_factor,
        )
        v_result = v_result + _side_drag_v_2d(
            v,
            hFacS,
            h0FacS,
            h0FacZ,
            rAs,
            dyU,
            dxV,
            viscosity,
            side_drag_factor,
        )

    u_result, v_result = mask_vector_tendency_for_grid(u_result, v_result, grid)
    return StateTendency(
        eta=torch.zeros_like(state.eta.data),
        u=u_result,
        v=v_result,
        temp=torch.zeros_like(state.temp.data),
        salt=torch.zeros_like(state.salt.data),
    )
