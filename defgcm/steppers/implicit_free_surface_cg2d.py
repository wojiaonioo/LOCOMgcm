"""CG2D, Helmholtz, and cube-seam helpers for implicit free-surface stepping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import functools
import os as _os_cg2d_pc
import torch
from defgcm.distributed.comm_adjoint import session_for_grid
import torch.distributed as dist

from defgcm.accel.graph_exit import graph_exit
from defgcm.diagnostics import ckpt_dump

if TYPE_CHECKING:
    from defgcm.geometry.cube_decomposition import CubeRankContext

from defgcm.distributed.layout import minimum_tile_extents
from defgcm.geometry import StructuredCGrid
from defgcm.geometry.grid import runtime_memo_constant
from defgcm.geometry.metrics import _cache_device_key
from defgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from defgcm.operators import (
    depth_integrated_face_thickness,
    edge_coefficient_helmholtz_from_haloed_center,
    face_volume_transport,
    gradient_center_from_haloed_scalar,
    helmholtz_edge_coefficients_from_depth,
    horizontal_volume_divergence_from_transports,
    incident_edge_coefficient_sum,
    mask_edge_coefficients_from_center_mask,
)
from defgcm.solvers import (
    apply_cg2d_jacobi_preconditioner,
    apply_cg2d_ssor_preconditioner,
    apply_cg2d_ssor_preconditioner_edge,
    build_cg2d_jacobi_inverse,
    build_cg2d_ssor_preconditioner,
    build_cg2d_ssor_preconditioner_edge,
    pcg_solve,
)
from defgcm.solvers._cg2d_fused import pcg_iteration_kernels
from defgcm.solvers.cg2d_preconditioner import (
    apply_cg2d_ssor_preconditioner_wrapped,
    build_cg2d_ssor_preconditioner_from_halo,
)
from defgcm.solvers.cg2d import (
    CG2DGraphOperands,
    Cg2dGraphPartition,
    Cg2dRootGraph,
    cg2d_cuda_graph_decline_reason,
    cg2d_cuda_graph_enabled,
    cg2d_cuda_graph_should_run,
    exception_summary,
    pcg_solve_fused,
    pcg_solve_graphed,
    record_cg2d_cuda_graph_fallback,
    root_path_selected,
)
from defgcm.solvers._pcg_core.readback import PcgReadback
from defgcm.accel.tensor_context import forward_mode_active


#: Stopping tests of the stepper's CG2D solve (``cg_tolerance_mode``).
#: ``absolute`` and ``mit_normalized`` reach the solver unchanged; ``mit_w_unit``
#: is MITgcm's ``cg2dTargetResWunit > 0`` test (``cg_tolerance`` in m/s), which
#: :func:`cg2d_w_unit_tolerance` converts to the ``absolute`` test.
CG2D_STOP_CRITERIA = ("absolute", "mit_normalized", "mit_w_unit")


def cg2d_w_unit_tolerance(
    grid: StructuredCGrid,
    *,
    target_res_w_unit: float,
    implicit_div_2d_flow: float,
    dt_freesurf: float,
    rA: torch.Tensor,
    surface_mask: torch.Tensor | None,
    mask_in_c: torch.Tensor | None,
    distributed: bool,
) -> float:
    """MITgcm's W-unit CG2D tolerance in the units of this solve's residual.

    With ``cg2dTargetResWunit > 0`` MITgcm leaves the RHS unnormalised
    (ini_cg2d.F:148) and stops on ``err_sq < cg2dTolerance**2`` (cg2d.F:204,
    :337) with ``cg2dTolerance = cg2dNorm * W * implicDiv2DFlow / deltaTMom *
    globalArea / SQRT(n2dWetPts)``, or ``cg2dNorm * W * globalArea /
    deltaTMom`` when implicDiv2DFlow is 0 (ini_cg2d.F:152-161).  globalArea
    and n2dWetPts are the sums of ``rA * maskInC`` and ``maskInC``
    (ini_global_domain.F:73-116), computed once at initialisation.

    MITgcm's operator, scaled by ``cg2dNorm`` (ini_cg2d.F:117-129, :198-206),
    acts on ``Bo_surf * eta`` (solve_for_pressure.F:129); this solve's operator (``_assemble_helmholtz_coefficients``)
    is ``rA`` plus ``alpha*beta*g*dt_mom*dt_freesurf`` times the Laplacian,
    which is ``-dt_mom*dt_freesurf/cg2dNorm`` times MITgcm's.  Residuals scale
    the same way, so MITgcm's test is ``||r|| < W * implicDiv2DFlow *
    dt_freesurf * globalArea / SQRT(n2dWetPts)`` here (``W * dt_freesurf *
    globalArea`` when implicDiv2DFlow is 0): the absolute test with that
    tolerance.  ``maskInC`` is the product of the solve's surface and OB
    interior masks.
    """

    def build() -> float:
        wet = torch.ones_like(rA, dtype=torch.float64)
        if surface_mask is not None:
            wet = wet * surface_mask.to(torch.float64)
        if mask_in_c is not None:
            wet = wet * mask_in_c.to(torch.float64)
        n_wet = wet.sum()
        area = (rA.detach().to(torch.float64) * wet).sum()
        if distributed:
            from defgcm.distributed.comm_adjoint import constant_transport
            from defgcm.distributed.reductions import global_sum

            with constant_transport():
                n_wet = global_sum(n_wet, session=session_for_grid(grid))
                area = global_sum(area, session=session_for_grid(grid))
        if implicit_div_2d_flow > 0.0:
            return (target_res_w_unit * implicit_div_2d_flow * dt_freesurf
                    * area.item() / n_wet.sqrt().item())
        return target_res_w_unit * dt_freesurf * area.item()

    return runtime_memo_constant(  # type: ignore[return-value]
        grid,
        ("cg2d_w_unit_tolerance", target_res_w_unit, implicit_div_2d_flow,
         dt_freesurf, _cache_device_key(rA.device)),
        build,
    )


# ---------------------------------------------------------------------------
# Preconditioned Conjugate Gradient (CG2D) Solver
# ---------------------------------------------------------------------------

def _compute_depth_at_faces(grid: StructuredCGrid, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute total water column depth at u- and v-faces.

    Uses continuous hFac fractions when available (partial cells),
    falling back to binary masks otherwise.

    Returns
    -------
    H_u : Tensor [ny, nx+1]
        Depth-integrated thickness at u-faces.
    H_v : Tensor [ny+1, nx]
        Depth-integrated thickness at v-faces.
    """
    layer_thickness = grid.layer_thickness.to(dtype=dtype, device=device)  # [nz]
    dz = layer_thickness.view(grid.nz, 1, 1)  # [nz, 1, 1]

    if grid.masks.hFacW is not None:
        # Use continuous hFac fractions (MITgcm partial cells)
        hFacW = grid.masks.hFacW.to(dtype=dtype, device=device)  # [nz, ny, nx+1]
        hFacS = grid.masks.hFacS.to(dtype=dtype, device=device)  # [nz, ny+1, nx]
        H_u, H_v = depth_integrated_face_thickness(
            dz,
            hFacW=hFacW,
            hFacS=hFacS,
            vertical_dim=0,
        )  # [ny, nx+1] / [ny+1, nx]
    elif grid.masks.has_3d_masks:
        u_mask = grid.masks.u_float(dtype=dtype, device=device)  # [nz, ny, nx+1]
        v_mask = grid.masks.v_float(dtype=dtype, device=device)  # [nz, ny+1, nx]
        H_u, H_v = depth_integrated_face_thickness(
            dz,
            hFacW=u_mask,
            hFacS=v_mask,
            vertical_dim=0,
        )  # [ny, nx+1] / [ny+1, nx]
    else:
        H_u, H_v = depth_integrated_face_thickness(
            dz,
            u_shape=(grid.ny, grid.nx + 1),
            v_shape=(grid.ny + 1, grid.nx),
            vertical_dim=0,
        )

    return H_u, H_v


def _has_distributed_neighbor(grid: StructuredCGrid, side: str) -> bool:
    neighbours = getattr(grid, "distributed_neighbors", None) or {}
    entry = neighbours.get(side) if hasattr(neighbours, "get") else None
    return entry is not None


def _axis_is_decomposed(grid: StructuredCGrid, axis: str) -> bool:
    layout = getattr(grid, "parallel_layout", None)
    if layout is None:
        return False
    parts_attr = "nPx" if axis == "x" else "nPy"
    return int(getattr(layout, parts_attr, 1)) > 1


def _mask_distributed_edge_coefficients_from_center_mask(
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    cell_mask: torch.Tensor,
    grid: StructuredCGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask regional distributed seam edge coefficients by both center cells.

    ``cell_mask`` (the wet mask or the open-boundary interior mask) depends on no
    control on any rank, and both exchanges declare it ``constant``.  Under
    ``nonlinFreeSurf > 2`` the assembly runs inside the step on every step, so in a
    differentiation window these exchanges go through the bound session; an
    undeclared mask would come back requiring grad, each product below would then
    save its edge-column view of the coefficient for backward, and the in-place
    store into that column would invalidate it.
    """
    west_dist = _has_distributed_neighbor(grid, "west")
    east_dist = _has_distributed_neighbor(grid, "east")
    south_dist = _has_distributed_neighbor(grid, "south")
    north_dist = _has_distributed_neighbor(grid, "north")
    if not (west_dist or east_dist or south_dist or north_dist):
        return aW_edge, aS_edge

    aW_out = aW_edge
    aS_out = aS_edge
    ny, nx = grid.ny, grid.nx
    if west_dist or east_dist:
        mask_halo_x = exchange(
            cell_mask,
            grid=grid,
            location="T",
            extent=HaloExtentV1(
                west_left_extent=1,
                east_right_extent=1,
                south_bottom_extent=0,
                north_top_extent=0,
            ),
            fill=FillSpec.ZERO,
            constant=True,
        )
        if west_dist:
            aW_out = aW_out.clone()
            aW_out[..., 0] = aW_out[..., 0] * mask_halo_x[..., :, 0] * mask_halo_x[..., :, 1]
        if east_dist:
            if aW_out is aW_edge:
                aW_out = aW_out.clone()
            aW_out[..., nx] = aW_out[..., nx] * mask_halo_x[..., :, nx] * mask_halo_x[..., :, nx + 1]

    if south_dist or north_dist:
        mask_halo_y = exchange(
            cell_mask,
            grid=grid,
            location="T",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=0,
                south_bottom_extent=1,
                north_top_extent=1,
            ),
            fill=FillSpec.ZERO,
            constant=True,
        )
        if south_dist:
            aS_out = aS_out.clone()
            aS_out[..., 0, :] = aS_out[..., 0, :] * mask_halo_y[..., 0, :] * mask_halo_y[..., 1, :]
        if north_dist:
            if aS_out is aS_edge:
                aS_out = aS_out.clone()
            aS_out[..., ny, :] = aS_out[..., ny, :] * mask_halo_y[..., ny, :] * mask_halo_y[..., ny + 1, :]
    return aW_out, aS_out


def _calc_div_ghat_cg3d_seed(
    u_star: torch.Tensor,
    v_star: torch.Tensor,
    grid: StructuredCGrid,
    *,
    dt_mom: float,
    implic_div_2d_flow: float,
    mask_in_c: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build MIT ``CALC_DIV_GHAT`` per-level CG3D seed from predicted flow."""
    if u_star.ndim != 4 or v_star.ndim != 4:
        raise ValueError("u_star/v_star must be [batch,nz,ny,nx+1] and [batch,nz,ny+1,nx]")
    batch, nz, ny, nxp1 = u_star.shape
    if (nz, ny, nxp1) != (grid.nz, grid.ny, grid.nx + 1):
        raise ValueError("u_star shape does not match grid")
    if v_star.shape != (batch, grid.nz, grid.ny + 1, grid.nx):
        raise ValueError("v_star shape does not match grid")
    dtype = u_star.dtype
    device = u_star.device
    dr_f = grid.layer_thickness_for(dtype, device).view(1, grid.nz, 1, 1)
    dy_g = grid.metrics.dyG_for(dtype, device).view(1, 1, grid.ny, grid.nx + 1)
    # full v-face field (bit-equal broadcast on row grids).
    dx_g = grid.metrics.dxG_for(dtype, device).view(1, 1, grid.ny + 1, grid.nx)
    hfac_w = grid.hfac_w_for(dtype, device).unsqueeze(0)
    hfac_s = grid.hfac_s_for(dtype, device).unsqueeze(0)
    scale = float(implic_div_2d_flow) / float(dt_mom)
    u_pf, v_pf = face_volume_transport(
        scale * u_star,
        scale * v_star,
        dr_f,
        dy_g,
        dx_g,
        hFacW=hfac_w,
        hFacS=hfac_s,
    )
    cg3d_b = horizontal_volume_divergence_from_transports(
        u_pf,
        v_pf,
    )
    if mask_in_c is not None:
        cg3d_b = cg3d_b * mask_in_c.to(dtype=dtype, device=device).view(1, 1, grid.ny, grid.nx)
    return cg3d_b



def _compute_depth_at_faces_cube(
    grid: StructuredCGrid, dtype: torch.dtype, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cube analogue of :func:`_compute_depth_at_faces`.

    On the cubed sphere the hFac masks carry a leading ``[6, …]`` face axis
    (``hFacW`` is ``[6, nz, ny, nx+1]``), so the ``nz`` reduction must be over
    ``dim=1`` (not ``dim=0`` as the regional helper assumes).  Returns
    ``H_u[6, ny, nx+1]`` / ``H_v[6, ny+1, nx]``.
    """
    layer_thickness = grid.layer_thickness.to(dtype=dtype, device=device)  # [nz]
    dz = layer_thickness.view(1, grid.nz, 1, 1)  # [1, nz, 1, 1]
    if grid.masks.hFacW is not None:
        hFacW = grid.masks.hFacW.to(dtype=dtype, device=device)  # [6, nz, ny, nx+1]
        hFacS = grid.masks.hFacS.to(dtype=dtype, device=device)  # [6, nz, ny+1, nx]
        H_u, H_v = depth_integrated_face_thickness(
            dz,
            hFacW=hFacW,
            hFacS=hFacS,
            vertical_dim=1,
        )  # [6, ny, nx+1] / [6, ny+1, nx]
    elif grid.masks.has_3d_masks:
        u_mask = grid.masks.u_float(dtype=dtype, device=device)
        v_mask = grid.masks.v_float(dtype=dtype, device=device)
        H_u, H_v = depth_integrated_face_thickness(
            dz,
            hFacW=u_mask,
            hFacS=v_mask,
            vertical_dim=1,
        )
    else:
        H_u, H_v = depth_integrated_face_thickness(
            dz,
            u_shape=(grid.face_count, grid.ny, grid.nx + 1),
            v_shape=(grid.face_count, grid.ny + 1, grid.nx),
            vertical_dim=1,
        )
    return H_u, H_v


def _assemble_helmholtz_coefficients(
    grid: StructuredCGrid,
    gravity: float,
    dt_mom: float,
    dt_freesurf: float,
    dtype: torch.dtype,
    device: torch.device,
    alpha_beta: float = 1.0,
    interior_mask: torch.Tensor | None = None,
    *,
    ctx: "CubeRankContext | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble the 2D Helmholtz operator in MIT edge-coefficient form.

    The off-diagonal Laplacian terms are scaled by ``alpha_beta`` which equals
    ``implicSurfPress * implicDiv2DFlow`` (MITgcm convention). The diagonal
    identity (free-surface ``rA``) term is NOT scaled — only the Laplacian part.

    The operator is:  L[η] = -α·β·g·dt_mom·dt_freesurf·∇·(H·∇η) + rA·η

    **Edge-coefficient storage (MIT ``aW2d``/``aS2d`` form).**  Instead of four
    per-cell off-diagonal arrays (aW/aE/aS/aN), the operator is stored as ONE
    west-edge array ``aW_edge[…, ny, nx+1]`` and ONE south-edge array
    ``aS_edge[…, ny+1, nx]``.  ``aW_edge[…, i]`` is the single coefficient on the
    u-face between cells ``i-1`` and ``i`` — shared by both, so the matvec adds it
    to BOTH (``cg3d_operator.apply_cg3d_operator``), making the operator Euclidean
    self-adjoint *by construction* (the old separate-array form was symmetric only
    as a numerical coincidence enforced by matched masking).  The legacy mapping
    is ``aW_edge[…, i]==aW_legacy[…, i]`` (interior i) and
    ``aW_edge[…, i+1]==aE_legacy[…, i]``; likewise for the south edge.

    On a ``face_count==6`` cubed-sphere grid the assembly uses the per-face 2-D
    metric accessors (``dyG_for``/``dyC_for``/``dxC_row_for``/``dxG_row_ext_for``
    all carry the ``[6, …]`` face axis) and then vector-exchanges the
    ``(aW_edge, aS_edge)`` pair once across seams (the analogue of MITgcm
    ``ini_cg2d.F:138`` ``EXCH_UV_XY_RS(aW2d,aS2d,.FALSE.)``) so every seam edge
    carries one agreed coefficient → the global operator is Euclidean-symmetric
    across seams.

    Parameters
    ----------
    ctx : CubeRankContext | None
        When *None* (default): single-rank path, byte-identical to the original
        implementation.  All tensors are global ``[6, …]``.

        When given (the distributed path): the global ``[6, …]`` geometry
        tensors are sliced to local faces via ``scatter_faces`` before the
        coefficient math; the seam exchange is routed to
        ``exchange_cubed_sphere_vector_distributed``/
        ``exchange_cubed_sphere_scalar_distributed`` so east/north seam slots are
        filled from the neighbour's locally-computed west/south values (same-source
        property, gives SPD by construction).  Only valid for ``face_count==6``
        (cube) grids; ignored on regional grids.

    Returns ``(aW_edge, aS_edge, aC)``.
    """
    from defgcm.geometry.cube_decomposition import scatter_faces

    is_cube = grid.face_count == 6
    # Distributed cube path: scatter global face tensors to local owned faces.
    # Regional grids ignore ctx (no face axis to scatter).
    use_dist = is_cube and ctx is not None

    ny, nx = grid.ny, grid.nx
    # n_local: number of faces owned by this rank (== 6 in single-rank / ctx=None).
    n_local = len(ctx.local_face_ids) if use_dist else grid.face_count

    g_dt2_ab = torch.tensor(
        gravity * dt_mom * dt_freesurf * alpha_beta, dtype=dtype, device=device,
    )
    # --- metrics (2-D-capable accessors; bit-equal to the old scalar reads on
    # uniform regional grids, genuinely per-face on the cube) ----------------
    rA = grid.metrics.cell_area_for(dtype, device)         # [ny,nx] / [6,ny,nx]
    H_u, H_v = (
        _compute_depth_at_faces_cube(grid, dtype, device)
        if is_cube else _compute_depth_at_faces(grid, dtype, device)
    )                                                       # u:[*,ny,nx+1] v:[*,ny+1,nx]
    dyG = grid.metrics.dyG_for(dtype, device)               # u-face [*,ny,nx+1]
    dyC = grid.metrics.dyC_for(dtype, device)               # v-face [*,ny+1,nx]
    dxG_ext = grid.metrics.dxG_row_ext_for(dtype, device)   # v-face [ny+1] / [6,ny+1,nx]
    dxC = grid.metrics.dxC_row_for(dtype, device)           # u-face [ny] / [6,ny,nx]

    if is_cube:
        if use_dist:
            # --- scatter all face-leading tensors to owned faces ---
            # All cube metrics carry face axis 0: [6, ny, nx*].
            # scatter_faces(global[6,...], ctx) → local[n_local,...].
            # This is a pure local slice (no communication).
            rA = scatter_faces(rA, ctx)                      # [n_local,ny,nx]
            H_u = scatter_faces(H_u, ctx)                    # [n_local,ny,nx+1]
            H_v = scatter_faces(H_v, ctx)                    # [n_local,ny+1,nx]
            dyG = scatter_faces(dyG, ctx)                    # [n_local,ny,nx+1]
            dyC = scatter_faces(dyC, ctx)                    # [n_local,ny+1,nx]
            dxG_ext = scatter_faces(dxG_ext, ctx)            # [n_local,ny+1,nx]
            dxC = scatter_faces(dxC, ctx)                    # [n_local,ny,nx]

        # Per-face 2-D metrics carry the face axis and u-/v-face stagger.  On the
        # cube ``dxC[:,:,i]`` is the centre-to-centre distance across the WEST
        # u-edge of cell ``i`` for *every* i in 0..nx-1, INCLUDING the west-seam
        # edge i=0 (MIT ini_cg2d.F:96 reads recip_dxC(1,j) locally there).  So
        # the west edges 0..nx-1 are computed locally; only the east-seam slot
        # i=nx has no local metric and is filled by the vector exchange below
        # (the analogue of MIT filling aW2d(sNx+1,j) via EXCH_UV_XY_RS).
        dxC_edge = dxG_ext.new_zeros(n_local, ny, nx + 1)   # [n_local,ny,nx+1]
        dxC_edge[:, :, 0:nx] = dxC[:, :, 0:nx]
        dyC_edge = dyC                                       # [n_local,ny+1,nx]
        dxG_edge = dxG_ext                                   # [n_local,ny+1,nx]
    else:
        # Uniform regional grid: dxC is per-row [ny]; broadcast to edges, dyG /
        # dyC / dxG are constant tensors bit-equal to the old scalar reads.
        # Full 2-D accessors: bit-equal to the old row broadcasts on
        # row-uniform grids (the accessor fallback IS that broadcast), and the
        # true per-column fields on vector-Cartesian metrics.
        dxC_edge = grid.metrics.dxC_for(dtype, device)       # [ny,nx+1]
        dyC_edge = dyC                                       # [ny+1,nx]
        dxG_edge = grid.metrics.dxG_for(dtype, device)       # [ny+1,nx]

    # --- west-edge (u-face) and south-edge (v-face) coefficients -------------
    # aW_edge[..., i]  : coefficient on the u-edge between cells (i-1, i).
    #                    = -α·β·g·dt² · dyG·H_u / dxC   (shared face area / dist).
    # aS_edge[..., j]  : coefficient on the v-edge between cells (j-1, j).
    aW_edge, aS_edge = helmholtz_edge_coefficients_from_depth(
        H_u,
        H_v,
        dyG,
        dxC_edge,
        dxG_edge,
        dyC_edge,
        g_dt2_ab,
    )                                                        # [*,ny,nx+1] / [*,ny+1,nx]

    # On a closed (non-periodic, non-seam) regional grid the physical far
    # boundary edges have no cell across them; zero so the diagonal omits the
    # phantom coupling.  A localized distributed tile may have a neighbour on
    # only one side of the axis, so keep internal rank seam edges coupled and
    # zero only true physical global edges.
    if not is_cube:
        if not grid.boundary_topology.periodic_x:
            if not _has_distributed_neighbor(grid, "west"):
                aW_edge[..., 0] = 0.0
            if not _has_distributed_neighbor(grid, "east"):
                aW_edge[..., nx] = 0.0
        if not grid.boundary_topology.periodic_y:
            if not _has_distributed_neighbor(grid, "south"):
                aS_edge[..., 0, :] = 0.0
            if not _has_distributed_neighbor(grid, "north"):
                aS_edge[..., ny, :] = 0.0

    # --- land masking: an edge coefficient is live only if BOTH cells it
    # separates are wet.  Mirrors the legacy aW*surface*aW_mask product. -------
    surface_mask: torch.Tensor | None = None
    if grid.masks.has_3d_masks:
        surface_mask = grid.masks.center_float(dtype=dtype, device=device)
        if is_cube:
            surface_mask = surface_mask[:, 0]                # [6,ny,nx]
            if use_dist:
                surface_mask = scatter_faces(surface_mask, ctx)  # [n_local,ny,nx]
        else:
            surface_mask = surface_mask[0]                   # [ny,nx]
        local_periodic_x = (
            not is_cube
            and grid.boundary_topology.periodic_x
            and not _axis_is_decomposed(grid, "x")
        )
        local_periodic_y = (
            not is_cube
            and grid.boundary_topology.periodic_y
            and not _axis_is_decomposed(grid, "y")
        )
        aW_edge, aS_edge = mask_edge_coefficients_from_center_mask(
            aW_edge,
            aS_edge,
            surface_mask,
            periodic_x=local_periodic_x,
            periodic_y=local_periodic_y,
            seam_x=is_cube,
            seam_y=is_cube,
        )
        if not is_cube:
            aW_edge, aS_edge = _mask_distributed_edge_coefficients_from_center_mask(
                aW_edge,
                aS_edge,
                surface_mask,
                grid,
            )

    if interior_mask is not None:
        local_periodic_x = (
            not is_cube
            and grid.boundary_topology.periodic_x
            and not _axis_is_decomposed(grid, "x")
        )
        local_periodic_y = (
            not is_cube
            and grid.boundary_topology.periodic_y
            and not _axis_is_decomposed(grid, "y")
        )
        aW_edge, aS_edge = mask_edge_coefficients_from_center_mask(
            aW_edge,
            aS_edge,
            interior_mask,
            periodic_x=local_periodic_x,
            periodic_y=local_periodic_y,
            seam_x=is_cube,
            seam_y=is_cube,
        )
        if not is_cube:
            aW_edge, aS_edge = _mask_distributed_edge_coefficients_from_center_mask(
                aW_edge,
                aS_edge,
                interior_mask,
                grid,
            )

    # --- seam coefficient reconciliation (cube only) -------------------------
    # MIT ini_cg2d.F:138 EXCH_UV_XY_RS(aW2d,aS2d,.FALSE.): a single vector halo
    # exchange of the coefficient pair so each seam edge carries ONE agreed
    # number → the operator is Euclidean-symmetric across seams.  with_signs=False
    # because the coefficients are non-negative face-area magnitudes.
    #
    # Mechanism (mirrors MIT): each face computes its WEST/SOUTH edges locally
    # (incl. the west-seam i=0 and south-seam j=0); the exchange fills the
    # EAST-seam slot aW_edge[...,nx] and NORTH-seam slot aS_edge[...,ny] with the
    # neighbour face's locally-computed west/south coefficient (rotated), so the
    # east coupling of cell nx-1 equals the neighbour's west coupling of its cell
    # 0 → one number per seam edge → Euclidean-symmetric (gate (i) max|L-Lᵀ|=0).
    if is_cube:
        seam_mask = surface_mask if grid.masks.has_3d_masks else None
        if interior_mask is not None:
            seam_mask = interior_mask if seam_mask is None else seam_mask * interior_mask
        aW_edge, aS_edge = _exchange_cube_edge_coeffs(
            aW_edge, aS_edge, grid, surface_mask=seam_mask, ctx=ctx,
        )

    # --- diagonal: aC = -(incident edge coeffs) + rA  (integrated SPD form) ---
    aC = -(incident_edge_coefficient_sum(aW_edge, aS_edge)) + rA

    # Mask the diagonal back to rA on land / OB cells (trivial rA·η=0 row).
    if grid.masks.has_3d_masks and surface_mask is not None:
        aC = -(incident_edge_coefficient_sum(aW_edge, aS_edge)) * surface_mask + rA
    if interior_mask is not None:
        aC = -(incident_edge_coefficient_sum(aW_edge, aS_edge)) * interior_mask + rA

    return aW_edge, aS_edge, aC


def _exchange_cube_edge_coeffs(
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    grid: StructuredCGrid,
    *,
    surface_mask: torch.Tensor | None = None,
    ctx: "CubeRankContext | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vector-exchange the edge-coefficient pair across cube seams.

    Analogue of MITgcm ``ini_cg2d.F:138`` ``EXCH_UV_XY_RS(aW2d,aS2d,.FALSE.)``.
    ``exchange_cubed_sphere_vector`` consumes ``u[6,ny,nx+1]``/``v[6,ny+1,nx]``
    (exactly the ``aW_edge``/``aS_edge`` layout) and returns symmetric-haloed
    same-size frames ``[6,n+2OL,n+2OL]``.

    The west/south edges (incl. the west-seam i=0 and south-seam j=0) are
    LOCAL (computed by the assembly).  This routine fills ONLY the east-seam
    slot ``aW_edge[...,nx]`` (from the east halo of ``uc``) and the north-seam
    slot ``aS_edge[...,ny]`` (from the north halo of ``vc``) with the neighbour
    face's locally-computed coefficient.  The matvec's east coupling of cell
    ``nx-1`` then equals the neighbour's west coupling of its cell 0 → one agreed
    number per seam edge → the operator is Euclidean-symmetric (verified
    ``max|L-Lᵀ|=0``).  The slot reads ``uc[...,1:n+1,n+1]`` /
    ``vc[...,n+1,1:n+1]`` are the FP-floor-validated correct mapping.

    ``surface_mask`` (``[6,ny,nx]`` or ``[n_local,ny,nx]`` wet/interior mask)
    zeros the LOCAL west-seam and south-seam coefficients where the cross-seam
    neighbour cell is land, so a wet→land seam coupling vanishes on both sides
    (the east/north slots inherit the neighbour's local masking through the
    coefficient exchange).

    Parameters
    ----------
    ctx : CubeRankContext | None
        When *None* (default): single-rank path, byte-identical to the original
        implementation using ``exchange_cubed_sphere_vector``.

        When given (the distributed path): routes the vector exchange to
        ``exchange_cubed_sphere_vector_distributed`` and the scalar mask exchange
        to ``exchange_cubed_sphere_scalar_distributed``.  Inputs are local
        ``[n_local, ...]`` tensors; outputs are also ``[n_local, ...]``.  The
        same-source property is inherited: each rank's east/north seam slot is
        filled from the neighbour rank's locally-computed west/south value, so
        every seam edge carries one agreed number (SPD by construction).
    """
    from defgcm.geometry.exchange import (
        HaloExtentV1,
        _exchange_cubed_sphere_scalar,
        exchange_cubed_sphere_scalar_distributed,
        exchange_cubed_sphere_vector,
        exchange_cubed_sphere_vector_distributed,
    )

    n = grid.nx
    _extent = HaloExtentV1.uniform(1)

    if ctx is None:
        # --- ctx=None: single-rank path, BYTE-IDENTICAL to the original ------
        uc, vc = exchange_cubed_sphere_vector(
            aW_edge, aS_edge, grid=grid, extent=_extent, with_signs=False,
        )
    else:
        # --- ctx given: distributed path ------------------------------------
        # aW_edge[n_local,ny,nx+1] / aS_edge[n_local,ny+1,nx] are already local.
        # The distributed exchange routes cross-seam strips via P2P so each rank's
        # east/north halo carries the neighbour's locally-computed west/south
        # coefficient: same-source property inherited bitwise.
        uc, vc = exchange_cubed_sphere_vector_distributed(
            aW_edge, aS_edge, grid=grid, extent=_extent, with_signs=False, ctx=ctx,
        )

    aW_out = aW_edge.clone()
    aS_out = aS_edge.clone()
    # East seam edge (edge index nx) <- neighbour's west coefficient (east halo).
    aW_out[..., n] = uc[..., 1:n + 1, n + 1]
    # North seam edge (edge index ny) <- neighbour's south coefficient (north halo).
    aS_out[..., n, :] = vc[..., n + 1, 1:n + 1]

    if surface_mask is not None:
        # An edge coupling is live only if BOTH cells it separates are wet.
        # Local cell wetness for the four seam slots:
        #   west-seam i=0   : cell (·,0)        east-seam i=nx : cell (·,nx-1)
        #   south-seam j=0  : cell (0,·)        north-seam j=ny: cell (ny-1,·)
        aW_out[..., n] = uc[..., 1:n + 1, n + 1] * surface_mask[..., n - 1]
        aS_out[..., n, :] = vc[..., n + 1, 1:n + 1] * surface_mask[..., n - 1, :]
        # Cross-seam NEIGHBOUR wetness (from the exchanged mask halo) for the
        # local west-/south-seam slots; the east/north slots inherit the
        # neighbour's masking through the coefficient exchange itself.
        if ctx is None:
            mh = _exchange_cubed_sphere_scalar(
                surface_mask, grid=grid, location="T", extent=_extent,
            )                                                   # [6,n+2,n+2]
        else:
            # The wet/interior mask is a constant of the run on every rank; see
            # _mask_distributed_edge_coefficients_from_center_mask for why a mask
            # exchanged by a rebuilt assembly must declare it.
            mh = exchange_cubed_sphere_scalar_distributed(
                surface_mask, grid=grid, location="T", extent=_extent, ctx=ctx,
                constant=True,
            )                                               # [n_local,n+2,n+2]
        west_nbr = mh[..., 1:n + 1, 0]                      # neighbour across west seam
        south_nbr = mh[..., 0, 1:n + 1]                     # neighbour across south seam
        aW_out[..., 0] = aW_edge[..., 0] * west_nbr
        aS_out[..., 0, :] = aS_edge[..., 0, :] * south_nbr
    return aW_out, aS_out


def _cube_eta_gradient(
    eta: torch.Tensor,
    grid: StructuredCGrid,
    dtype: torch.dtype,
    device: torch.device,
    ctx: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seam-aware ∂η/∂x at u-faces and ∂η/∂y at v-faces on the cube.

    ``eta`` is ``[n_local_or_6, ny, nx]``.  Returns
    ``(grad_u[n_local_or_6,ny,nx+1], grad_v[n_local_or_6,ny+1,nx])``.

    The face-i u-edge (between cells i-1,i) difference is ``η(i)-η(i-1)`` divided
    by the centre-to-centre distance ``dxC`` at that edge; the v-edge analogue
    uses ``dyC``.  The cube ``dxC_row_for``/``dyC_for`` metrics give the per-face
    2-D distances (``dxC[...,i]`` = the WEST edge of cell i for i=0..nx-1; the
    east-seam edge nx uses the neighbour's local edge via the width-1 scalar
    halo).  This is the gradient counterpart of the Helmholtz ``aW_edge`` metric
    so the velocity correction is consistent with the cg2d operator.

    The regional ``gradient_center_to_u`` cannot run here: its cube branch
    divides the ``nx+1``-wide difference by the ``nx``-wide centre ``dxC`` (shape
    mismatch); this helper builds the proper ``nx+1``-wide edge metric.

    When *ctx* is provided (multi-rank), the scalar halo exchange is
    routed to ``exchange_cubed_sphere_scalar_distributed`` and the dxC/dyC
    metrics are scattered to local face layout.  ctx=None: unchanged single-rank
    path (byte-identical to the single-rank behaviour).
    """
    from defgcm.geometry.exchange import (
        HaloExtentV1, _exchange_cubed_sphere_scalar,
    )
    n = grid.nx
    n_local_or_6 = len(ctx.local_face_ids) if ctx is not None else grid.face_count

    if ctx is not None:
        from defgcm.geometry.exchange import exchange_cubed_sphere_scalar_distributed
        from defgcm.geometry.cube_decomposition import scatter_faces
        # Distributed exchange: eta is [n_local,ny,nx].
        eh = exchange_cubed_sphere_scalar_distributed(
            eta.unsqueeze(1), grid=grid, location="T",
            extent=HaloExtentV1.uniform(1), ctx=ctx,
        )[:, 0]                                                   # [n_local,n+2,n+2]
        dxC = scatter_faces(grid.metrics.dxC_row_for(dtype, device), ctx)
        dyC = scatter_faces(grid.metrics.dyC_for(dtype, device), ctx)
    else:
        eh = _exchange_cubed_sphere_scalar(
            eta.unsqueeze(1), grid=grid, location="T", extent=HaloExtentV1.uniform(1),
        )[:, 0]                                                   # [6,n+2,n+2]
        dxC = grid.metrics.dxC_row_for(dtype, device)            # [6,ny,nx]
        dyC = grid.metrics.dyC_for(dtype, device)                # [6,ny+1,nx]

    if dxC.ndim == 2:
        dxC = dxC.unsqueeze(-1).expand(n_local_or_6, n, n)
    # Edge distance: dxC at the WEST edge of cell i covers faces 0..nx-1; reuse
    # the last interior dxC for the east-seam edge nx (sufficient for the smoke;
    # the cg2d operator's east-seam coeff is reconciled exactly via the vector
    # exchange — flagged minor inconsistency only at the seam face nx).
    dxC_edge = dxC.new_zeros(n_local_or_6, n, n + 1)
    dxC_edge[:, :, 0:n] = dxC
    dxC_edge[:, :, n] = dxC[:, :, n - 1]
    return gradient_center_from_haloed_scalar(
        eh,
        dxC_u=dxC_edge,
        dyC_v=dyC,
    )


def _reconcile_seam_transport(
    u_tr: torch.Tensor,
    v_tr: torch.Tensor,
    grid,
    ctx=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill the east/north seam ghost barotropic transports from the neighbour.

    The depth-summed barotropic transports ``u_tr[6,1,ny,nx+1]`` /
    ``v_tr[6,1,ny+1,nx]`` feed the cg2d RHS ``div(H*u*)``.  LOCOMgcm computes the
    east-ghost slot ``u_tr[...,nx]`` (and north ``v_tr[...,ny,:]``) from its own
    local ghost-column predictor, which is NON-zero — the momentum predictor runs
    on the full ``[...,nx+1]`` tensor and ``hFacW[...,nx]`` is wet there — whereas
    MITgcm fills that halo from the cross-seam NEIGHBOUR via ``EXCH_UV`` of gU/gV
    before ``CALC_DIV_GHAT``.  Leaving the stale local value corrupts the
    divergence (hence the cg2d RHS and the solved eta) at the easternmost /
    northernmost cells — the step-1 seam-corner eta mismatch vs MIT.

    We reproduce MIT by vector-exchanging the transports and copying the
    neighbour's value into the east/north seam slots (the west-seam i=0 /
    south-seam j=0 edges are interior and already correct).  ``with_signs=True``
    because the transport is a C-grid VECTOR (it flips under the cube-face
    rotation), unlike the scalar cg2d coefficients reconciled by
    ``_exchange_cube_edge_coeffs``.  The slot mapping ``uc[...,1:n+1,n+1]`` /
    ``vc[...,n+1,1:n+1]`` is the same FP-floor-validated layout that routine uses.

    When *ctx* is provided the vector exchange is routed to the distributed path
    so cross-rank seam transports are fetched via P2P communication.
    ctx=None retains the single-rank path (byte-identical).
    """
    from defgcm.dynamics._vector_seam_topology import (
        reconcile_vector_redundant_edges_for_grid,
    )

    return reconcile_vector_redundant_edges_for_grid(u_tr, v_tr, grid, ctx=ctx)


#: Halo width the five-point Helmholtz apply reads on every side.  A property
#: of the stencil, so it is also the width the CUDA-Graph admission compares
#: against the narrowest tile a decomposition produces
#: (``Cg2dGraphPartition.halo_width``): one number, both consumers.
CG2D_CENTER_HALO_WIDTH = 1


def _regional_center_halo(
    field: torch.Tensor, grid: StructuredCGrid, *, constant: bool = False
) -> torch.Tensor:
    """The width-1 zero-filled centre halo the regional five-point apply reads.

    One expression, two consumers: the eager matvec below and the fused SPMV's
    ``_fused_halo`` at the CG2D call site.  They MUST be the same exchange --
    the fused path's bitwise claim is against the eager operator, so a second
    literal copy is a place for the two to drift apart silently.
    """
    return exchange(
        field,
        grid=grid,
        location="T",
        extent=HaloExtentV1.uniform(CG2D_CENTER_HALO_WIDTH),
        fill=FillSpec.ZERO,
        constant=constant,
    )


#: Halo width of the residual exchange in the one-exchange SSOR iteration
#: (``_regional_ssor_seam_mode -> "ring"``): the preconditioner is evaluated on
#: a one-cell ring outside the tile, and the five-point apply at a ring cell
#: reads one cell further.
CG2D_RING_RESIDUAL_HALO_WIDTH = 2


def _regional_residual_ring_halo(
    field: torch.Tensor, grid: StructuredCGrid, *, constant: bool = False
) -> torch.Tensor:
    """The width-2 zero-filled centre halo of the ``"ring"`` mode.

    The same ``exchange`` as :func:`_regional_center_halo`, two cells wide.
    ``exchange`` pads x first and then y with the x-padded rows, so the four
    corner cells of the inner ring come from the diagonal tile: the apply at
    ring cell ``(-1, 0)`` reads ``r(-1, -1)``.  Walls and open edges fill +0.
    """
    return exchange(
        field,
        grid=grid,
        location="T",
        extent=HaloExtentV1.uniform(CG2D_RING_RESIDUAL_HALO_WIDTH),
        fill=FillSpec.ZERO,
        constant=constant,
    )


def _apply_helmholtz_operator(
    eta: torch.Tensor,
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
    grid: StructuredCGrid,
) -> torch.Tensor:
    """Apply the five-point Helmholtz operator in edge-coefficient form.

    ``aW_edge[..., i]`` is the shared u-edge coefficient between cells ``i-1``
    and ``i`` and is added to BOTH adjacent cells (mirrors
    ``cg3d_operator.apply_cg3d_operator``), so the operator is Euclidean
    self-adjoint by construction.

    Regional  : ``eta``/``aC`` are ``[batch,ny,nx]`` / ``[ny,nx]``; a regular
                scalar exchange supplies the one-cell halo.
    Cube      : ``eta`` is ``[batch,6,ny,nx]``; the state halo is exchanged once
                (corner-ignoring width-1 scalar exchange) and N/S/E/W neighbours
                are read from the haloed strips.  Seam coupling rides the
                pre-exchanged boundary edge coefficients (``aW_edge[...,0/nx]``,
                ``aS_edge[...,0/ny]``).
    """
    if grid.face_count == 6:
        return _apply_helmholtz_operator_cube(eta, aW_edge, aS_edge, aC, grid)

    eta_halo = _regional_center_halo(eta, grid)
    result = edge_coefficient_helmholtz_from_haloed_center(
        eta_halo,
        aW_edge=aW_edge,
        aS_edge=aS_edge,
        aC=aC,
        halo=1,
    )
    return result.unsqueeze(0) if eta.ndim == 2 else result


def _apply_helmholtz_operator_ringed(
    eta_ring: torch.Tensor,
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
) -> torch.Tensor:
    """:func:`_apply_helmholtz_operator` on a field that carries its halo.

    ``eta_ring`` is ``[batch, ny+2, nx+2]``: the tile plus the one-cell ring
    the ``"ring"`` mode keeps on the search direction, holding what the
    exchange would have delivered.  The stencil is the same call, so the
    result equals the exchanged apply bit for bit when the ring does.
    """
    result = edge_coefficient_helmholtz_from_haloed_center(
        eta_ring,
        aW_edge=aW_edge,
        aS_edge=aS_edge,
        aC=aC,
        halo=1,
    )
    return result.unsqueeze(0) if eta_ring.ndim == 2 else result


def _apply_helmholtz_operator_cube(
    eta: torch.Tensor,
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
    grid: StructuredCGrid,
    *,
    ctx: "CubeRankContext | None" = None,
) -> torch.Tensor:
    """Cube matvec: exchange the state halo once, then read N/S/E/W from it.

    ``eta`` is ``[batch, 6, ny, nx]`` (or ``[6, ny, nx]``) in single-rank
    mode, or ``[batch, n_local, ny, nx]`` (or ``[n_local, ny, nx]``) in
    distributed mode.  The width-1, corner-ignoring scalar halo exchange
    (``_exchange_cubed_sphere_scalar``) fills the four side strips from the
    correct neighbour faces; the 5-point stencil never reads diagonal corners
    so no corner fill is needed.

    Parameters
    ----------
    ctx : CubeRankContext | None
        When *None* (default): single-rank path, byte-identical to the
        original implementation.  ``eta``/``aW_edge``/``aS_edge``/``aC``
        must be global ``[*, 6, ny, nx*]`` tensors.

        When given: distributed path.  ``eta``/coefficients are
        LOCAL tensors for the faces owned by this rank (face-axis length
        ``n_local = len(ctx.local_face_ids)``).  The scalar halo exchange is
        routed to ``exchange_cubed_sphere_scalar_distributed``, which handles
        cross-rank P2P via ``ctx``.  The 5-point stencil is identical to the
        single-rank path (only the exchange is different).
    """
    from defgcm.geometry.exchange import (
        HaloExtentV1, _exchange_cubed_sphere_scalar,
        exchange_cubed_sphere_scalar_distributed,
    )

    squeeze = eta.ndim == 3
    e = eta.unsqueeze(0) if squeeze else eta            # [batch,n_local_or_6,ny,nx]
    # The scalar cube exchange requires the FACE axis to lead (it indexes
    # ``field[neighbour_face]``), so move batch behind the face axis.
    ef = e.transpose(0, 1).contiguous()                 # [n_local_or_6,batch,ny,nx]

    _extent = HaloExtentV1.uniform(1)
    if ctx is None:
        # --- ctx=None: single-rank path, BYTE-IDENTICAL to the original ---
        eh = _exchange_cubed_sphere_scalar(
            ef, grid=grid, location="T", extent=_extent,
        )                                                   # [6,batch,n+2,n+2]
    else:
        # --- ctx given: distributed path ------------------------------------
        # ef is [n_local,batch,ny,nx]; distributed exchange returns same-shape
        # haloed tensor [n_local,batch,n+2,n+2].  Math is identical to the
        # single-rank oracle (same rotation + slice per seam, same corner fill).
        eh = exchange_cubed_sphere_scalar_distributed(
            ef, grid=grid, location="T", extent=_extent, ctx=ctx,
        )                                                   # [n_local,batch,n+2,n+2]

    # Coefficients are [n_local_or_6,ny,nx*]; broadcast over the batch axis
    # (dim 1).  Same stencil math for both paths.
    result = edge_coefficient_helmholtz_from_haloed_center(
        eh,
        aW_edge=aW_edge.unsqueeze(1),
        aS_edge=aS_edge.unsqueeze(1),
        aC=aC.unsqueeze(1),
        halo=1,
    )                                                   # [n_local_or_6,batch,ny,nx]
    result = result.transpose(0, 1).contiguous()        # [batch,n_local_or_6,ny,nx]
    return result[0] if squeeze else result


def _cube_scalar_halo_one(
    field: torch.Tensor,
    grid: StructuredCGrid,
    ctx: "CubeRankContext | None",
) -> torch.Tensor:
    """Width-1 cross-seam center-scalar halo for a cube field ``[*, ny, nx]``.

    Single-rank (``ctx=None``) routes to the oracle ``_exchange_cubed_sphere_scalar``;
    multi-rank routes to ``exchange_cubed_sphere_scalar_distributed`` (P2P).  The
    field's leading axis must be the FACE axis (``[n_local_or_6, ..., ny, nx]``);
    returns ``[n_local_or_6, ..., ny+2, nx+2]`` (symmetric width-1 halo, corner
    fill identical to the matvec's state exchange).
    """
    from defgcm.geometry.exchange import (
        HaloExtentV1, _exchange_cubed_sphere_scalar,
        exchange_cubed_sphere_scalar_distributed,
    )
    _extent = HaloExtentV1.uniform(1)
    if ctx is None:
        return _exchange_cubed_sphere_scalar(
            field, grid=grid, location="T", extent=_extent,
        )
    return exchange_cubed_sphere_scalar_distributed(
        field, grid=grid, location="T", extent=_extent, ctx=ctx,
    )


def _build_cube_ssor_precond(
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
    grid: StructuredCGrid,
    ctx: "CubeRankContext | None",
    *,
    offdiag_factor: float = 0.51,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the seam-aware edge-form SSOR coefficients for the cube cg2d.

    The neighbour-diagonal halos of ``aC`` (the only seam-sensitive input; the
    edge coefficients are already seam-reconciled by ``_assemble_helmholtz_*``)
    are filled by a single width-1 cross-seam scalar exchange, then
    ``build_cg2d_ssor_preconditioner_edge`` produces ``(pC, pW_edge, pS_edge)``.
    """
    aC_eh = _cube_scalar_halo_one(aC, grid, ctx)             # [*,ny+2,nx+2]
    ny, nx = aC.shape[-2:]
    # West cell of u-edge i (i=0..nx)  -> haloed columns ol-1 .. ol-1+nx (ol=1).
    aC_west_halo = aC_eh[..., 1:1 + ny, 0:nx + 1]            # [*,ny,nx+1]
    # South cell of v-edge j (j=0..ny) -> haloed rows ol-1 .. ol-1+ny.
    aC_south_halo = aC_eh[..., 0:ny + 1, 1:1 + nx]           # [*,ny+1,nx]
    return build_cg2d_ssor_preconditioner_edge(
        aW_edge, aS_edge, aC, aC_west_halo, aC_south_halo,
        offdiag_factor=offdiag_factor,
    )


def _regional_ssor_seam_mode(grid: StructuredCGrid) -> str:
    """Where the regional SSOR apply finds its neighbours across the tile edges.

    Numerics rule 11: a tile edge couples iff ``grid.edge_face_couples(axis)``
    or that side has a distributed neighbour.  MIT's preconditioner reads the
    residual after the exchange at ``cg2d.F:181``/``:353`` and builds its
    coefficients from exchanged diagonals (``ini_cg2d.F:199-241``), so a
    coupled edge carries the same five-point terms as an interior one.

    ``"local"``
        No edge couples (single process, no periodic axis): the tile-local
        apply, unchanged since before rule 11 reached the preconditioner.
    ``"wrap_x"`` / ``"wrap_y"`` / ``"wrap_xy"``
        Single process with periodic axes: the neighbours are the opposite end
        of the same array (:func:`apply_cg2d_ssor_preconditioner_wrapped`), so
        no exchange runs inside the iteration.
    ``"ring"``
        Some side has a distributed neighbour and every split axis is at
        least two cells wide on every tile (:func:`_residual_ring_fits`): one
        width-2 residual exchange per iteration
        (:func:`_regional_residual_ring_halo`), the apply evaluated on the tile
        plus a one-cell ring with the neighbour tiles' own coefficients, and
        the search direction carried with that ring, so the matvec reads it
        instead of exchanging ``p``.  The ring holds the values the neighbour
        tiles compute for their own edge cells, bit for bit, so the solve is
        the ``"halo"`` solve with one exchange per iteration instead of two.
    ``"halo"``
        Some side has a distributed neighbour and some split axis has a tile
        of one cell, which cannot send a two-cell strip: the residual goes
        through the same width-1 ``exchange`` as the matvec
        (:func:`_regional_center_halo`, which also wraps a periodic axis the
        process owns whole), and the edge-shaped coefficients carry the
        neighbour tile's east/north values; ``p`` and ``r`` are both exchanged
        every iteration, as ``cg2d.F:264``/``:353`` do.

    This is the one decision the eager, CUDA-graph and fused paths all read:
    every rank reaches it from the global decomposition, so no two neighbours
    post different exchanges.
    """
    if any(
        _has_distributed_neighbor(grid, side)
        for side in ("west", "east", "south", "north")
    ):
        return "ring" if _residual_ring_fits(grid) else "halo"
    wrap = "".join(
        axis for axis in ("x", "y") if grid.edge_face_couples(axis)
    )
    return f"wrap_{wrap}" if wrap else "local"


def _residual_ring_fits(grid: StructuredCGrid) -> bool:
    """Whether every tile along every split axis is at least two cells wide.

    The width-2 residual exchange sends two columns (rows) of the sender's own
    cells, which a one-cell tile does not have (``exchange`` refuses it).  The
    verdict comes from the global decomposition (``minimum_tile_extents``), not
    from this tile, so all ranks agree without a collective.  A periodic axis
    that one process owns whole wraps inside its array and sets no minimum.  A
    grid without a ``parallel_layout`` has no global decomposition to read and
    keeps the two-exchange form.
    """
    layout = getattr(grid, "parallel_layout", None)
    if layout is None:
        return False
    narrowest = minimum_tile_extents(
        global_nx=int(layout.global_nx),
        global_ny=int(layout.global_ny),
        nPx=int(layout.nPx),
        nPy=int(layout.nPy),
    )
    return all(
        extent >= CG2D_RING_RESIDUAL_HALO_WIDTH
        for parts, extent in zip((int(layout.nPx), int(layout.nPy)), narrowest)
        if parts > 1
    )


def _build_regional_ssor_precond(
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
    grid: StructuredCGrid,
    mode: str,
    *,
    offdiag_factor: float = 0.51,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Regional SSOR operands for ``mode`` plus the cell-shaped ``(pC, pW, pS)``.

    The second tuple is what every mode records (``cg_coef``): the per-tile
    coefficients MIT stores, ``pW``/``pS`` on each cell's west/south edge.

    ``"local"`` builds with zero neighbour diagonals (the unchanged tile-local
    path).  Every other mode reads the neighbour diagonal through the width-1
    centre exchange -- a wrap on a periodic axis, the neighbour rank across a
    rank seam -- which is ``ini_cg2d.F:199-236`` evaluating ``aC2d(0,j)`` and
    ``aC2d(i,0)`` in the halo.  ``"halo"`` then exchanges ``(pW, pS)`` once more
    so the east/north edges carry the neighbour tile's own coefficient
    (``ini_cg2d.F:241`` ``EXCH_UV_XY_RS(pW, pS)``): ``pW_edge[..., nx]`` is the
    east tile's ``pW[..., 0]``, never a local recomputation.  ``"ring"``
    exchanges ``(pC, pW, pS)`` two cells wide instead, so the apply can also run
    on the one-cell ring with the neighbour tiles' own coefficients:
    ``pC [ny+2, nx+2]`` on cells ``-1..ny`` x ``-1..nx``, ``pW [ny+2, nx+3]``
    on the west edges of cells ``-1..nx+1`` and ``pS [ny+3, nx+2]`` on the
    south edges of rows ``-1..ny+1`` (the ``"halo"`` operands are their inner
    part).  The exchanges run once per solve, outside the iteration.
    """
    if mode == "local":
        cell = build_cg2d_ssor_preconditioner(
            aW_edge, aS_edge, aC, offdiag_factor=offdiag_factor,
        )
        return cell, cell
    if mode not in ("wrap_x", "wrap_y", "wrap_xy", "halo", "ring"):
        raise ValueError(f"unknown regional SSOR seam mode {mode!r}")
    cell = build_cg2d_ssor_preconditioner_from_halo(
        aW_edge,
        aS_edge,
        _regional_center_halo(aC, grid, constant=not aC.requires_grad),
        offdiag_factor=offdiag_factor,
    )
    if mode not in ("halo", "ring"):
        return cell, cell
    pC, pW, pS = cell
    ny, nx = pC.shape[-2:]
    if mode == "ring":
        ring = _regional_residual_ring_halo(torch.stack(cell), grid,
                                            constant=not any(c.requires_grad for c in cell))
        return (
            ring[0, 1:ny + 3, 1:nx + 3].contiguous(),
            ring[1, 1:ny + 3, 1:nx + 4].contiguous(),
            ring[2, 1:ny + 4, 1:nx + 3].contiguous(),
        ), cell
    neighbour = _regional_center_halo(torch.stack((pW, pS)), grid,
                                      constant=not (pW.requires_grad or pS.requires_grad))
    pW_edge = torch.cat((pW, neighbour[0, 1:ny + 1, nx + 1:nx + 2]), dim=-1)
    pS_edge = torch.cat((pS, neighbour[1, ny + 1:ny + 2, 1:nx + 1]), dim=-2)
    return (pC, pW_edge, pS_edge), cell


def _regional_ssor_apply(
    mode: str, grid: StructuredCGrid
) -> Callable[..., torch.Tensor]:
    """The regional SSOR apply ``(residual, pC, pW, pS) -> z`` for ``mode``.

    Resolved once per solve, so the iteration pays no mode dispatch.
    ``"halo"`` takes the edge-shaped operands of
    :func:`_build_regional_ssor_precond` and reads the residual through the
    matvec's own width-1 exchange (MIT exchanges ``cg2d_r`` at ``cg2d.F:181`` /
    ``:353`` before the apply); ``"ring"`` takes the ring-shaped operands, reads
    the residual through the width-2 exchange and returns ``z`` on the tile
    plus its one-cell ring, ``[batch, ny+2, nx+2]`` (the same stencil call on
    the larger block, so every cell sees the five products in MIT's order);
    the other modes take the cell-shaped operands and need no exchange.
    ``"local"`` returns :func:`apply_cg2d_ssor_preconditioner` itself, the
    pre-change apply, looked up by name at solve time.
    """
    if mode == "local":
        return apply_cg2d_ssor_preconditioner
    if mode == "halo":
        def halo_apply(
            residual: torch.Tensor,
            pC: torch.Tensor,
            pW_edge: torch.Tensor,
            pS_edge: torch.Tensor,
        ) -> torch.Tensor:
            return apply_cg2d_ssor_preconditioner_edge(
                _regional_center_halo(residual, grid), pC, pW_edge, pS_edge, halo=1,
            )

        return halo_apply
    if mode == "ring":
        def ring_apply(
            residual: torch.Tensor,
            pC_ring: torch.Tensor,
            pW_ring: torch.Tensor,
            pS_ring: torch.Tensor,
        ) -> torch.Tensor:
            return apply_cg2d_ssor_preconditioner_edge(
                _regional_residual_ring_halo(residual, grid),
                pC_ring, pW_ring, pS_ring, halo=1,
            )

        return ring_apply
    if mode not in ("wrap_x", "wrap_y", "wrap_xy"):
        raise ValueError(f"unknown regional SSOR seam mode {mode!r}")
    return functools.partial(
        apply_cg2d_ssor_preconditioner_wrapped,
        wrap_x=mode in ("wrap_x", "wrap_xy"),
        wrap_y=mode in ("wrap_y", "wrap_xy"),
    )


def _cast_solution_back(
    result: tuple[torch.Tensor, int, float | PcgReadback],
    solve_dtype: torch.dtype,
) -> tuple[torch.Tensor, int, float | PcgReadback]:
    """Cast the PCG solution tensor back to the caller's dtype (f64-island exit).

    ``result`` is ``(x, num_iters, residual)`` from :func:`pcg_solve` /
    :func:`pcg_solve_graphed`; diagnostics are Python scalars or an asynchronous
    readback passed through to the publisher.  When ``solve_dtype`` is float64
    (the default state precision) this is the identity: the f64 path never allocates.
    """
    if solve_dtype is torch.float64:
        return result
    x, num_iters, residual = result
    return x.to(solve_dtype), num_iters, residual


def _cg2d_graph_partition(
    grid: StructuredCGrid, *, world_size: int
) -> Cg2dGraphPartition:
    """Describe this run's decomposition for the CUDA-Graph admission.

    Reading the decomposition is the CALLER's job (a
    low-level operator is not handed a topology label and asked to judge it),
    so the translation from "what this grid is" to "what the partition looks
    like" happens here, once, and the solver sees only numbers.

    A grid with no ``parallel_layout`` owns the whole domain: a one-by-one tile
    grid.  That is the honest description of a single-process run and also of
    the distributed cube, which partitions by FACE through its own context and
    publishes no tile grid -- so the admission sees ``ranks`` disagreeing with
    the tile count and declines it as ``incomplete_rank_grid`` rather than
    guessing a layout that does not exist.
    """
    layout = getattr(grid, "parallel_layout", None)
    return Cg2dGraphPartition(
        faces=int(getattr(grid, "face_count", 1)),
        tiles_x=int(getattr(layout, "nPx", 1)),
        tiles_y=int(getattr(layout, "nPy", 1)),
        ranks=int(world_size),
        cells_x=int(getattr(layout, "global_nx", grid.nx)),
        cells_y=int(getattr(layout, "global_ny", grid.ny)),
        halo_width=CG2D_CENTER_HALO_WIDTH,
    )


def _record_cg2d_graph(
    outcome: str,
    *,
    reason: str | None = None,
    variant: str = "cube_pcg_graph_replay",
) -> None:
    """Dispatch record for DEFGCM_CG2D_CUDA_GRAPH.

    Replay hits and selected failures are recorded here.  Admission declines
    and solves converging before replay are recorded by the shared solver's
    ``record_cg2d_cuda_graph_fallback`` for both ordinary and ROOT solves.
    Selected failures still propagate after their error record.
    """
    from defgcm.accel import current_rank, record_dispatch

    record_dispatch(
        "cg2d_cuda_graph",
        outcome=outcome,
        variant=variant,
        site="implicit_free_surface_cg2d.cg2d_solve",
        backend="cuda",
        rank=current_rank(),
        reason=reason,
    )


_CG2D_PRECOND_CHOICE: dict[bool, str] = {}


def _cg2d_precond_choice(is_cube: bool) -> str:
    """``DEFGCM_CUBE_CG2D_PRECOND`` / ``DEFGCM_CG2D_PRECOND`` read once per process.

    Call ``refresh_cg2d_precond_choice`` after changing either variable.
    """
    choice = _CG2D_PRECOND_CHOICE.get(is_cube)
    if choice is None:
        choice = refresh_cg2d_precond_choice(is_cube)
    return choice


def refresh_cg2d_precond_choice(is_cube: bool | None = None) -> str:
    if is_cube is None:
        refresh_cg2d_precond_choice(True)
        return refresh_cg2d_precond_choice(False)
    name, default = (
        ("DEFGCM_CUBE_CG2D_PRECOND", "jacobi") if is_cube else ("DEFGCM_CG2D_PRECOND", "ssor")
    )
    choice = _os_cg2d_pc.environ.get(name, default).strip().lower()
    _CG2D_PRECOND_CHOICE[is_cube] = choice
    return choice


def cg2d_solve(
    rhs: torch.Tensor,
    x0: torch.Tensor,
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
    grid: StructuredCGrid,
    *,
    max_iters: int = 1000,
    tolerance: float = 1e-13,
    tolerance_mode: str | None = None,
    use_min_res_sol: bool,
    ctx: "CubeRankContext | None" = None,
    err_sq_history: list | None = None,
    root_adjoint: bool = True,
) -> tuple[torch.Tensor, int, float | PcgReadback]:
    """Preconditioned conjugate gradient solver for Ax = b (edge-coeff form).

    ``root_adjoint`` selects the derivative semantics and defaults to True,
    MITgcm's default adjoint (TAF ``cg2d_mad``, Tapenade ``CG2D_B0``): the
    derivative of ``A^-1 b`` is one more solve of the same system from a zero
    first guess, and the first guess and the preconditioner carry no
    derivative.  Every differentiable solve of a rollout takes it, the regional
    and the cube one, on one rank or inside a ``CommSession``, where the solve
    and its transpose are one ordered round.  The solution is the eager
    ``pcg_solve``'s bit for bit.  False tapes the PCG loop.  Rationale and the
    CS32 measurements are on ``cg2d.pcg_solve``.

    ``tolerance_mode`` (``"absolute"`` / ``"mit_normalized"``) is passed
    straight to the solver; ``None`` keeps the solver default and the
    process-level override scope (``cg2d_tolerance_mode_scope``).  The
    explicit argument exists so the regional step body carries no ``with``
    block around the solve (a context manager whose body graph-breaks
    at the PCG loop cannot be resumed by Dynamo).

    Parameters
    ----------
    rhs : [batch, ny, nx] (regional) or [batch, 6, ny, nx] (cube, single-rank)
          or [batch, n_local, ny, nx] (cube, distributed with ctx given)
    x0 : same shape as ``rhs`` initial guess
    aW_edge : west-edge coefficient [*, ny, nx+1]
    aS_edge : south-edge coefficient [*, ny+1, nx]
    aC : main diagonal [*, ny, nx]
    grid : StructuredCGrid
    max_iters : maximum CG iterations
    tolerance : convergence tolerance on ||r||²
    use_min_res_sol : MITgcm ``cg2dUseMinResSol``, passed to every solver path
          (``pcg_solve``); the fused and CUDA-graph paths raise when it is set
          and the solve ends unconverged
    ctx : CubeRankContext | None
        When *None* (default): existing single-rank logic is BYTE-IDENTICAL.
        When given: cube distributed path — matvec routes to the ctx-aware
        ``_apply_helmholtz_operator_cube(..., ctx=ctx)``; the face-ordered dot
        passes ``ctx`` so it uses the multi-rank all_gather path;
        Jacobi preconditioner is unchanged (no halo needed).
        Inputs rhs/x0/aW_edge/aS_edge/aC must already be LOCAL tensors
        ``[batch, n_local, ny, nx]`` / ``[n_local, ny, nx*]`` for this rank.
        The CG math and stopping criterion are IDENTICAL to the single-rank path;
        only the matvec and dot are routed through ctx.  Because the dot is a
        global all_gather reduce, every rank sees the identical scalar err_sq at
        every iteration → num_iters and residual are naturally identical across
        ranks without additional synchronization.
    err_sq_history : list | None
        If given, the per-iteration err_sq tensor is appended each CG step.
        Forwarded to ``pcg_solve``; used only by tests for residual-history
        bitwise verification.

    Returns
    -------
    x : solution (same shape as ``rhs``)
    num_iters : actual iteration count
    residual : final ||r||, or a PcgReadback to resolve with read_pcg_stats
    """
    # --- float64 solver island (precision-configurable state) --------------
    # The convergence test (``tolerance``**2 on ||r||², default 1e-26) is
    # UNREACHABLE in float32 (eps~1.2e-7 → ||r||² rounding floor ~1e-14), so a
    # reduced-precision state (eta) must still assemble the coefficients and run
    # the whole PCG iteration in float64.  We up-cast the rhs / initial guess /
    # coefficients here so every closure built below (matvec, preconditioner,
    # face-ordered dot) operates in float64, then cast the solution back to the
    # caller's dtype on the way out (:func:`_cast_solution_back`).  float64
    # inputs skip the cast entirely — the historical path runs byte-for-byte
    # with zero extra allocation.  Covers both the cube and regional steppers
    # (they share this entry).
    _solve_dtype = rhs.dtype
    if _solve_dtype is not torch.float64:
        rhs = rhs.to(torch.float64)
        x0 = x0.to(torch.float64)
        aW_edge = aW_edge.to(torch.float64)
        aS_edge = aS_edge.to(torch.float64)
        aC = aC.to(torch.float64)
    is_cube = grid.face_count == 6
    is_dist_cube = is_cube and ctx is not None
    # Preconditioner selection (env-gated).
    #
    # Regional path: default = SSOR.  The 100-step d4 ablation proved it
    # alignment-neutral (u/v/T/S/eta trajectories agree with Jacobi to FP floor,
    # max|d| ~1e-14) while cutting cg2d iterations ~1.95x (jacobi mean 348 -> ssor
    # mean 177).  Jacobi remains opt-in via DEFGCM_CG2D_PRECOND=jacobi.
    #
    # Cube path: SEPARATE env knob ``DEFGCM_CUBE_CG2D_PRECOND`` (default = JACOBI).
    # It is intentionally NOT the global DEFGCM_CG2D_PRECOND: a global default of
    # "ssor" must NOT flip the cube default, because the cs32x15 epoch-0 baseline
    # is frozen on Jacobi and any change there bumps the epoch.  High-res cube
    # cases (cs192/cs384) opt in by exporting DEFGCM_CUBE_CG2D_PRECOND=ssor.  The
    # cube SSOR path is seam-aware (build reads a cross-seam aC halo; apply reads
    # a cross-seam r halo) so it is alignment-neutral vs cube Jacobi to FP floor
    # and bitwise R==1 across ranks.
    # Resolved once per process: ``os.environ.get`` inside the
    # compiled step is a Dynamo constant-fold failure (one graph break).
    _precond = _cg2d_precond_choice(is_cube)
    _solver_mode_kwargs = (
        {} if tolerance_mode is None else {"tolerance_mode": tolerance_mode}
    )

    # Jacobi preconditioner: M^{-1} = diag(1/aC).  Use double-where so the
    # backward graph never evaluates 1/aC at aC=0 (land cells) -- the dead
    # branch sees a sentinel 1.0 instead of 0, producing finite gradients.
    precond_inv = build_cg2d_jacobi_inverse(aC)
    _ssor_mode = (
        _regional_ssor_seam_mode(grid)
        if _precond == "ssor" and not is_cube
        else None
    )

    if _precond == "ssor" and is_cube:
        # Seam-aware cube SSOR (MIT ini_cg2d.F:212-238 build / cg2d.F:250-255
        # apply, cg2dpcOffDFac=0.51).  The neighbour-diagonal halo of aC is the
        # only seam-sensitive build input (the edge coeffs aW/aS are already
        # reconciled across seams by _assemble_helmholtz_coefficients); it is
        # filled by a width-1 cross-seam scalar exchange.  Edge-form coefficients
        # (pW_edge[ny,nx+1] / pS_edge[ny+1,nx]) represent the reconciled
        # east/north seam edges too.  The apply re-exchanges r each iteration
        # (a parallel 5-point stencil, no serial sweep -> order-independent),
        # mirroring the matvec's haloed-state read.
        _pC, _pW_e, _pS_e = _build_cube_ssor_precond(
            aW_edge, aS_edge, aC, grid, ctx, offdiag_factor=0.51,
        )
        precond_operands = (_pC, _pW_e, _pS_e)
    elif _precond == "ssor":
        # Regional MIT off-diagonal (SSOR-like) preconditioner (ini_cg2d.F:210-241
        # build, cg2d.F:216-222 apply), cg2dpcOffDFac=0.51 (MIT compiled default).
        #   pC = 1/aC                                          (guard aC==0 -> 0)
        #   pW(i,j) = -aW(i,j) / (offd*(aC(i-1,j)+aC(i,j)))**2 (guard den==0 -> 0)
        #   pS(i,j) = -aS(i,j) / (offd*(aC(i,j-1)+aC(i,j)))**2 (guard den==0 -> 0)
        # Same double-where backward-safety pattern as the Jacobi branch so the
        # dead branch never differentiates 1/0.  aW/aS are the operator's own
        # west/south face coefficients; aC(i-1)/aC(j-1) at a coupled tile edge
        # is the neighbour's diagonal (numerics rule 11, _regional_ssor_seam_mode).
        precond_operands, (_pC, _pW, _pS) = _build_regional_ssor_precond(
            aW_edge, aS_edge, aC, grid, _ssor_mode, offdiag_factor=0.51,
        )
    else:
        precond_operands = (precond_inv,)

    graph_native_ring = False

    def _make_graph_precond(
        operands: tuple[torch.Tensor, ...],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if _precond == "ssor" and is_cube:
            _pC, _pW_e, _pS_e = operands

            def graph_precond(r_in: torch.Tensor) -> torch.Tensor:
                # r_in is [batch, n_local_or_6, ny, nx].  Move the face axis ahead of
                # batch (the cube scalar exchange indexes the face axis first), halo,
                # run the edge stencil, then restore the batch-leading layout.
                squeeze = r_in.ndim == 3
                r = r_in.unsqueeze(0) if squeeze else r_in       # [batch,nf,ny,nx]
                rf = r.transpose(0, 1).contiguous()              # [nf,batch,ny,nx]
                rh = _cube_scalar_halo_one(rf, grid, ctx)        # [nf,batch,ny+2,nx+2]
                z = apply_cg2d_ssor_preconditioner_edge(
                    rh,
                    _pC.unsqueeze(1),
                    _pW_e.unsqueeze(1),
                    _pS_e.unsqueeze(1),
                    halo=1,
                )                                                # [nf,batch,ny,nx]
                z = z.transpose(0, 1).contiguous()               # [batch,nf,ny,nx]
                return z[0] if squeeze else z
        elif _precond == "ssor":
            # Cell-shaped (pC, pW, pS), or edge-shaped for the "halo" mode.
            _pC, _pW, _pS = operands
            if _ssor_mode == "ring" and graph_native_ring:
                from defgcm.solvers._cg2d_fused._fast import cg2d_spmv

                def graph_precond(r_in: torch.Tensor) -> torch.Tensor:
                    return cg2d_spmv(
                        _regional_residual_ring_halo(r_in, grid),
                        _pW, _pS, _pC,
                    )
            else:
                _ssor_apply = _regional_ssor_apply(_ssor_mode, grid)

                def graph_precond(r_in: torch.Tensor) -> torch.Tensor:
                    # 5-point SSOR apply (cg2d.F:216-222). r_in [batch,ny,nx].
                    return _ssor_apply(r_in, _pC, _pW, _pS)
        else:
            (precond_inv,) = operands

            def graph_precond(r_in: torch.Tensor) -> torch.Tensor:
                return apply_cg2d_jacobi_preconditioner(precond_inv, r_in)

        return graph_precond

    # ---- matvec routing -------------------------------------------------------
    # ctx=None: existing single-rank logic, byte-identical.
    # ctx given + is_cube: distributed path — route to ctx-aware matvec.
    # ring=True ("ring" SSOR mode): the operator on a search direction that
    # carries its one-cell ring, which is its halo -- no exchange.
    def _make_graph_matvec(
        operands: tuple[torch.Tensor, ...],
        *,
        ring: bool = False,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        aW_edge, aS_edge, aC = operands
        if ring:
            def graph_matvec(p_ring: torch.Tensor) -> torch.Tensor:
                return _apply_helmholtz_operator_ringed(p_ring, aW_edge, aS_edge, aC)
        elif is_dist_cube:
            # Distributed cube matvec.  eta is [batch,n_local,ny,nx];
            # coefficients are [n_local,ny,nx*].  The call dispatches directly to
            # _apply_helmholtz_operator_cube(..., ctx=ctx) which internally routes
            # the state halo exchange to exchange_cubed_sphere_scalar_distributed.
            def graph_matvec(x_in: torch.Tensor) -> torch.Tensor:
                return _apply_helmholtz_operator_cube(x_in, aW_edge, aS_edge, aC, grid, ctx=ctx)
        else:
            def graph_matvec(x_in: torch.Tensor) -> torch.Tensor:
                return _apply_helmholtz_operator(x_in, aW_edge, aS_edge, aC, grid)

        return graph_matvec

    _matvec = _make_graph_matvec((aW_edge, aS_edge, aC))
    _apply_precond = _make_graph_precond(precond_operands)
    # "ring": the preconditioner returns z with its ring and every path carries
    # p with it; ``_matvec`` still forms b - A x0 and the diagnostics.
    _ring = _ssor_mode == "ring"
    _ring_matvec = (
        _make_graph_matvec((aW_edge, aS_edge, aC), ring=True) if _ring else None
    )

    # ---- dot routing ----------------------------------------------------------
    # The cube path reduces with a face-ordered dot so single- and
    # multi-rank are bitwise identical. ``rhs``/``x0`` are [batch, 6, ny, nx] in
    # single-rank mode, or [batch, n_local, ny, nx] in distributed mode.
    # face_dim=1 in both cases.  The regional path passes dot=None -> byte-
    # identical legacy reduction (the lat-lon path is untouched).
    #
    # Distributed cube path: face_ordered_dot receives ctx so it uses the
    # all_gather multi-rank path.  The dot is a global synchronising
    # reduce; every rank sees identical err_sq → num_iters/residual identical
    # across ranks without extra synchronisation.
    #
    # Single-rank cube path: when ctx=None but dist is already initialized (e.g.
    # a distributed test running the oracle on the global tensor in-process),
    # we pass group=False to face_ordered_dot to FORCE the single-rank reduction
    # path.  The group=False sentinel bypasses the "dist initialized but ctx=None"
    # guard in face_ordered_dot, making the oracle byte-identical to the
    # pre-dist single-rank path.
    _dot = None
    if is_cube:
        from defgcm.distributed.reductions import face_ordered_dot
        _ctx_for_dot = ctx  # None for single-rank, ctx for distributed
        _group_for_dot = None if is_dist_cube else False  # False -> force single-rank

        def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return face_ordered_dot(
                a, b, face_dim=1, n_faces=6,
                ctx=_ctx_for_dot, group=_group_for_dot,
                session=session_for_grid(grid),
            )

    # ---- fused-kernel PCG body ---------------------------------
    # Tried before the CUDA-graph path because it addresses the same cost from
    # the other end: the graph makes the unfused body cheaper to LAUNCH, this
    # makes the body smaller (41 -> 21 kernel launches per SSOR iteration).
    # They are alternatives, never composed; retiring the graph requires
    # the fused readings stand.  Declining is always safe: the eager
    # ``pcg_solve`` below is the bit-identical reference, and the dispatcher
    # has already recorded WHY it declined.
    def _fused_halo(field: torch.Tensor) -> torch.Tensor:
        return _regional_center_halo(field, grid)

    def _fused_ring_halo(field: torch.Tensor) -> torch.Tensor:
        return _regional_residual_ring_halo(field, grid)

    def _record_cg_coefficients() -> None:
        """Dump the assembled coefficients once, whichever gated path runs.

        Lives outside both branches because a checkpoint dump that only fires
        on one accelerator path is a dump that changes with a switch.
        """
        if not ckpt_dump.ENABLED:
            return
        coefficient_tensors = {
            "aW_edge": aW_edge,
            "aS_edge": aS_edge,
            "aC": aC,
        }
        if _precond == "ssor":
            coefficient_tensors["pC"] = _pC
            if is_cube:
                coefficient_tensors.update(pW=_pW_e, pS=_pS_e)
            else:
                coefficient_tensors.update(pW=_pW, pS=_pS)
        ckpt_dump.record("cg_coef", **coefficient_tensors)

    # Every externally supplied state tensor, not a sample: ``rhs`` and ``x0``
    # can differ in grad state, and admitting on ``rhs`` alone let a
    # ``x0.requires_grad`` solve reach kernels that have no backward formula.
    # The solver is told which preconditioner APPLY it would otherwise run
    # eagerly, never which grid produced it (a low-level operator does
    # not receive a topology label and judge on it).  The cube's seam-aware
    # ``apply_cg2d_ssor_preconditioner_edge`` is a different expression from the
    # regional five-point apply, so it gets its own name here -- decided in the
    # caller layer, where the topology already lives -- and the dispatcher
    # declines it as an unserved algorithm.  The cube state and coefficients
    # additionally carry a face axis, which the shared operand predicate
    # declines on shape; the two are independent and either alone suffices.
    # The regional apply is named by where its neighbours come from
    # (``_regional_ssor_seam_mode``): ``ssor`` tile-local, ``ssor_wrap_*`` the
    # in-array periodic wrap, ``ssor_halo`` the exchanged residual -- it takes
    # the same ``_fused_halo`` the SPMV takes -- and ``ssor_ring`` the width-2
    # exchanged residual with the ringed search direction.
    if _precond == "ssor" and is_cube:
        _fused_precond_kind = "ssor_edge"
    elif _ssor_mode is not None and _ssor_mode != "local":
        _fused_precond_kind = f"ssor_{_ssor_mode}"
    else:
        _fused_precond_kind = _precond
    # ---- path precedence: an admissible CUDA-Graph replay beats the fused body --
    # Measured with 2xA800, N=960, 2 ranks, same tree
    # and switch set, clean CUDA-event timing 3x100 steps): fused body 472.9 ms
    # per step, graphed replay 336.5 ms per step.  The fused kernels are faster
    # than the eager ones, but the fused loop keeps the eager control flow --
    # two host reads per iteration, ~115 blocking calls per step -- and the GPU
    # idles for ~64% of the PCG segment, whereas the replay checks
    # convergence per block.  Until the fused core carries that block control
    # flow, the graph wins whenever its gate admits the
    # solve; the fused switch is then recorded as superseded, never silently
    # zero (the accelerator reachability contract: a requested switch that does
    # not run must say why).  Both paths are byte-identical to ``pcg_solve``.
    graph_cache = getattr(grid, "_runtime_cache", None)
    graph_family = "cube" if is_cube else "latlon"
    graph_variant = f"{graph_family}_pcg_graph_replay"
    graph_world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    graph_requested = cg2d_cuda_graph_enabled()
    graph_partition = _cg2d_graph_partition(grid, world_size=graph_world_size)
    # A live forward mode is a differentiable request too, whatever the grad
    # mode: the graph replay stages the operands into captured buffers and the
    # fused body runs custom operators without a forward-mode rule, so either
    # would drop the tangent without an error.  It declines both and selects
    # the ROOT path below.  Program state, so the routing agrees across ranks
    # that all open the same dual level.
    forward_mode = forward_mode_active()
    graph_grad = forward_mode or (torch.is_grad_enabled() and (
        rhs.requires_grad or x0.requires_grad or aC.requires_grad
    ))
    # ROOT call: ``pcg_solve`` below takes the ROOT path (``root_operands``).
    # With a bound session that path is one atomic round
    # (``CommSession.linear_solve``) on every rank whatever each rank's local
    # operands require, so the call is routed there from facts that are the
    # same on every rank (``root_path_selected``) and never from local
    # ``requires_grad``: a rank that took the graph path or the fused body
    # instead would register no round, and its collectives would not pair with
    # the peers' round (a graph path would also capture outside the round's
    # owner).  Neither runs on a ROOT call.  Without a session only this
    # process takes part, and a solve with no differentiable operand keeps the
    # graph and fused paths.  The ROOT path's first pass
    # and zero-start re-solve run under ``no_grad`` on detached operands and
    # the ROOT adjoint supplies the derivative (the re-solve), so the
    # ``grad_enabled`` decline does not apply to them; the switch, partition,
    # device and cache decide as for the graph path.
    session = session_for_grid(grid)
    root_call = (
        root_adjoint
        and root_path_selected(
            max_iters=max_iters, session=session, forward_mode=forward_mode
        )
        and (session is not None or graph_grad)
    )
    root_graph = None
    # Both graph paths capture their dot-product reductions on the window's
    # data group; the transport layer refuses a captured collective there
    # unless an enclosing collective solve owns the capture.  A ROOT solve
    # captures inside the solve round the session opens for it
    # (comm_adjoint._SolveRound.primal and reverse, capture_owner=True), which
    # is not open yet here, so only the graph path reads the predicate: a
    # no-grad step taken beside a session (the compile pass's warm-up) has no
    # round and is declined; a ROOT solve is not (declining it would lose every
    # root graph).
    from defgcm.distributed.comm_adjoint import active_data_group
    from defgcm.distributed.halo_p2p import _group_name
    from defgcm.distributed.transport_failure import captured_collectives_unowned

    graph_capture_unowned = (
        graph_requested
        and not root_call
        and captured_collectives_unowned(_group_name(active_data_group()))
    )
    root_decline = (
        cg2d_cuda_graph_decline_reason(
            requested=graph_requested,
            partition=graph_partition,
            device_type=rhs.device.type,
            grad_enabled=False,
            graph_cache_available=graph_cache is not None,
        )
        if root_call
        else None
    )
    if root_decline not in (None, "disabled"):
        record_cg2d_cuda_graph_fallback(root_decline)
    if root_call and root_decline is None:

        def _record_root_graph(outcome, solve, reason=None):
            _record_cg2d_graph(
                outcome, reason=reason, variant=f"{graph_variant}_root_{solve}"
            )

        root_graph = Cg2dRootGraph(
            cache=graph_cache,
            # Own entry, apart from the graph path's: the captured collectives
            # run on the window's data group, named here, and the body applies
            # the eager loop's preconditioner (``graph_native_ring`` is only
            # set when ``use_graph``, which a ROOT graph excludes).
            key=(
                "cg2d_graph",
                graph_family,
                graph_world_size,
                tuple(rhs.shape),
                _precond,
                rhs.dtype,
                "root",
                _group_name(active_data_group()),
            ),
            make_matvec=(
                functools.partial(_make_graph_matvec, ring=True)
                if _ring
                else _make_graph_matvec
            ),
            record=_record_root_graph,
        )
    use_graph = not root_call and cg2d_cuda_graph_should_run(
        requested=graph_requested,
        partition=graph_partition,
        device_type=rhs.device.type,
        grad_enabled=graph_grad,
        graph_cache_available=graph_cache is not None,
        capture_unowned=graph_capture_unowned,
    )
    if use_graph and _ring and not any(
        operand.requires_grad for operand in (rhs, x0, aW_edge, aS_edge, aC, *precond_operands)
    ):
        from defgcm.solvers._cg2d_fused._fast import fused_pcg_available

        graph_native_ring = fused_pcg_available()
    if (use_graph and graph_cache is not None) or root_call:
        # The fused body would return before the ROOT path below, so on a ROOT
        # call it would skip the session's atomic round on this rank alone
        # whenever its local operands declined differently from a peer's.
        fused_kernels = None
        from defgcm.accel import current_rank, gate, record_dispatch

        if gate("cg2d_fused_pcg").effective_value:
            record_dispatch(
                "cg2d_fused_pcg",
                outcome="inapplicable",
                variant=f"superseded:{_fused_precond_kind}",
                site="cg2d.pcg_solve_fused",
                backend="cuda",
                rank=current_rank(),
                reason=(
                    "superseded_by_root_adjoint" if root_call
                    else "superseded_by_cg2d_cuda_graph"
                ),
            )
    else:
        fused_kernels = pcg_iteration_kernels(
            states=(rhs, x0),
            aW_edge=aW_edge,
            aS_edge=aS_edge,
            aC=aC,
            precond_kind=_fused_precond_kind,
            precond_operands=precond_operands,
            precond_halo=(
                _fused_ring_halo if _ring
                else _fused_halo if _ssor_mode == "halo"
                else None
            ),
        )
    if fused_kernels is not None:
        _record_cg_coefficients()
        # Every path below this point that falls through re-runs a solve which
        # appends to ``err_sq_history`` from scratch, so an abandoned attempt
        # must not leave its entries in front of the real ones
        # . The solve stays correct either way; the
        # history is a verification instrument and a polluted one is worse than
        # an absent one.
        _history_mark = None if err_sq_history is None else len(err_sq_history)
        try:
            with graph_exit():  # PCG host loop (fused body)
                with fused_kernels:
                    fused_solution = pcg_solve_fused(
                        _matvec,
                        rhs,
                        x0,
                        kernels=fused_kernels,
                        halo=_fused_halo,
                        max_iters=max_iters,
                        tolerance=tolerance,
                        dot=_dot,
                        **_solver_mode_kwargs,
                        use_min_res_sol=use_min_res_sol,
                        err_sq_history=err_sq_history,
                    )
        except Exception as exc:
            # A selected implementation failure is not an admission decline.
            # The ledger is idempotent when __exit__ already recorded the error.
            fused_kernels.record_error(exc)
            if _history_mark is not None:
                del err_sq_history[_history_mark:]
            raise
        else:
            return _cast_solution_back(fused_solution, _solve_dtype)

    # ---- gated CUDA-Graph fast path ------------------------------------------
    # The cg2d PCG body is shape-static and control-flow-static (only the scalar
    # err_sq/pAp early-exit varies), making it a textbook CUDA-Graph target.  We
    # record the iteration body once and replay it every step (the captured
    # graph is cached on the grid while current coefficients are staged into
    # graph-owned buffers; see ``pcg_solve_graphed``).  Which decompositions are
    # supported is ``cg2d_graph_topology``'s answer, read off the partition:
    # any Cartesian tile grid that accounts for every process and whose
    # narrowest tile carries the halo, at any process count (NCCL halo P2P +
    # all-reduce dots are captured in-graph; the gate decision below consumes
    # only rank-uniform launch/configuration facts, so all ranks agree before
    # the first captured collective).  Every other case -- grad, non-CUDA,
    # a seam that would cross a process boundary, other decompositions, a
    # session beside the call with no solve round open (the transport layer
    # would refuse the captured reductions, ``no_enclosing_collective_solve``),
    # capture failure -- falls back to the byte-identical eager ``pcg_solve``
    # with a named fallback counter (same toggle discipline as the fused CG3D
    # kernel).  The math/stopping criterion are identical; replay runs the SAME
    # kernels at the SAME addresses (byte-exact).  ``use_graph`` was decided
    # above, before the fused attempt (path precedence).
    if use_graph:
        if graph_cache is not None:
            graph_operands = CG2DGraphOperands(
                matvec_operands=(aW_edge, aS_edge, aC),
                precond_operands=precond_operands,
                # The captured body applies the operator to p: with its ring
                # in the "ring" mode, the same factory as ``_ring_matvec``.
                make_matvec=(
                    functools.partial(_make_graph_matvec, ring=True)
                    if _ring
                    else _make_graph_matvec
                ),
                make_precond=_make_graph_precond,
            )
            graph_key = (
                "cg2d_graph",
                graph_family,
                graph_world_size,
                tuple(rhs.shape),
                _precond,
                rhs.dtype,
                # ``graph_cache`` is this grid's runtime cache: the grid owns
                # the entry, so the key holds no object identity.
                graph_native_ring,
            )
            _record_cg_coefficients()
            graph_stats: dict[str, int] = {}
            try:
                with graph_exit():  # PCG host loop
                    graphed = pcg_solve_graphed(
                        _matvec,
                        rhs,
                        x0,
                        _apply_precond,
                        max_iters=max_iters,
                        tolerance=tolerance,
                        dot=_dot,
                        **_solver_mode_kwargs,
                        use_min_res_sol=use_min_res_sol,
                        err_sq_history=err_sq_history,
                        graph_cache=graph_cache,
                        graph_key=graph_key,
                        graph_operands=graph_operands,
                        graph_stats=graph_stats,
                        ring_matvec=_ring_matvec,
                    )
                graphed_solution = _cast_solution_back(graphed, _solve_dtype)
            except Exception as exc:
                # Invalidate the resource, record the selected failure, and
                # propagate it instead of committing an eager substitute. The
                # legacy diagnostic counter retains its stable reason token.
                record_cg2d_cuda_graph_fallback(
                    "graph_capture_or_replay_failed",
                    detail=exception_summary(exc),
                )
                _record_cg2d_graph(
                    "error",
                    reason="graph_capture_or_replay_failed",
                    variant=graph_variant,
                )
                graph_cache.pop(graph_key, None)
                raise
            else:
                # The captured graph really replayed.  Recorded after the
                # replay returned, never at the `use_graph` predicate.  The
                # solver records prologue convergence separately (no hit).
                if graph_stats.get("replays"):
                    _record_cg2d_graph("hit", variant=graph_variant)
                return graphed_solution

    with graph_exit():  # PCG host loop (eager solver)
        eager_solution = pcg_solve(
            _matvec,
            rhs,
            x0,
            _apply_precond,
            max_iters=max_iters,
            tolerance=tolerance,
            dot=_dot,
            **_solver_mode_kwargs,
            use_min_res_sol=use_min_res_sol,
            err_sq_history=err_sq_history,
            session=session,
            root_adjoint=root_adjoint,
            ring_matvec=_ring_matvec,
            root_operands=CG2DGraphOperands(
                matvec_operands=(aW_edge, aS_edge, aC),
                precond_operands=precond_operands,
                make_matvec=_make_graph_matvec,
                make_precond=_make_graph_precond,
            ),
            root_graph=root_graph,
        )
    return _cast_solution_back(
        eager_solution,
        _solve_dtype,
    )
