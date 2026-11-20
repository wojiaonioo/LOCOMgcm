"""MITgcm GAD_ADVECTION-style tracer update kernels."""

from __future__ import annotations

from typing import Any

import torch

from defgcm.geometry.metric_bundle import _positive_reciprocal

from ._gad_advection_topology import (
    assert_mit_12tile_runtime_shapes,
    horizontal_gad_first_order_cube_tiles,
)
from .horizontal_tracer_diffusion import horizontal_tracer_diffusion_flux_form
from defgcm.geometry.cubed_sphere_corner import fill_cs_corner_tr_rl
from defgcm.geometry.exchange import (
    FillSpec,
    HaloExtentV1,
    exchange,
    exchange_hfac_pair,
)
from defgcm.operators import vertical_tracer_advection_transport_space
from defgcm.operators.fluxes.tracer_flux import (
    HORIZONTAL_STENCIL_SCHEMES,
    TRACER_SCHEME_BY_NAME,
    VERTICAL_CFL_SCHEMES,
    VERTICAL_VELOCITY_FLUX_SCHEMES,
    TracerFluxScheme,
    horizontal_stencil_flux,
    horizontal_two_point_flux,
    interface_recip_drC,
    vertical_tracer_flux,
)

#: Horizontal schemes the cube passes accept (MITgcm codes in
#: ``TRACER_ADVECTION_SCHEMES``).  The cube has only the multi-dimensional
#: path, which MITgcm forces off for the Adams-Bashforth schemes.
#: ponytail: both cube stencil adapters build a two-cell halo (OL=2 corner
#: fill, fixed tile slices), so a scheme declaring a wider halo is not
#: accepted; widening them means an OL=halo corner fill and halo-sized slices.
CUBE_HORIZONTAL_SCHEMES = tuple(
    scheme for scheme in (TracerFluxScheme.UPWIND, *HORIZONTAL_STENCIL_SCHEMES)
    if not TRACER_SCHEME_BY_NAME[scheme].adams_bashforth
    and TRACER_SCHEME_BY_NAME[scheme].halo <= 2
)
#: Vertical schemes the cube R pass accepts (MITgcm codes in
#: ``TRACER_ADVECTION_SCHEMES``); scheme 30 is not wired on the cube.
CUBE_VERTICAL_SCHEMES = (TracerFluxScheme.UPWIND, TracerFluxScheme.DST3_FL)


def _ol2_corner_halo(local: torch.Tensor, *, mode: int, grid: Any) -> torch.Tensor:
    """OL=2 corner-filled center halo for the DST3 4-point stencil.

    The 1st-order upwind path needs only OL=1 (``fill_cs_corner_tr_rl`` builds
    its own one-cell side halo).  DST3's ``T(i-2..i+1)`` stencil needs OL=2, so
    we pre-pad to a symmetric 2-wide halo with a cross-face seam exchange
    (``location='T'`` -> ``exchange`` routes to the cube scalar backend, single-
    rank or distributed per ``_cube_step_ctx``), then run the corner fill on the
    pre-padded tensor.  ``fill_cs_corner_tr_rl`` infers ``olx=oly=2`` from the
    pre-padded shape (single-rank ``_prepare_existing_halo`` / distributed
    ``extra_x==2`` branch) and fills the 2-wide cube corners.
    """
    padded = exchange(
        local,
        grid=grid,
        location="T",
        extent=HaloExtentV1.uniform(2),
        fill=FillSpec.ZERO,
    )
    return fill_cs_corner_tr_rl(padded, mode=mode, grid=grid)


def _stencil_face_flux_6face_x(
    local: torch.Tensor,
    u_trans: torch.Tensor,
    u_vel: torch.Tensor,
    grid: Any,
    *,
    dt: float,
    scheme: TracerFluxScheme,
) -> torch.Tensor:
    """DST3 / DST3-FL transport flux ``af`` on the six_face X (u) faces.

    Full-face approximation of ``gad_dst3fl_adv_x.F`` on the cube.  Unlike the
    MIT tile path (which zeroes the tile-halo edge faces MIT zeroes), the
    six_face path supplies cross-seam neighbours for EVERY face 0..nx via the
    OL=2 corner-filled halo, so DST3 is computed on the full ``[*, nz, ny, nx+1]``
    face array -- exactly the faces the UPWIND six_face fills.  This is the
    DECLARED seam approximation (no MIT-bitwise reference): the corner-fill +
    OL=2 exchange give the face-edge stencils, where the MIT-faithful 12-tile
    instead does an explicit tile split.  The reconstruction math is the shared
    stencil-flux selection ``horizontal_stencil_flux``.
    """
    nx, ny = grid.nx, grid.ny
    h2 = _ol2_corner_halo(local, mode=1, grid=grid)  # [*, ny+4, nx+4]
    # Interior cell c -> halo index c+2.  Face i (0..nx): cells i-2,i-1,i,i+1
    # = h2[i], h2[i+1], h2[i+2], h2[i+3]; interior rows = h2[2:ny+2].
    Tm2 = h2[..., 2:ny + 2, 0:nx + 1]
    Tm1 = h2[..., 2:ny + 2, 1:nx + 2]
    T0 = h2[..., 2:ny + 2, 2:nx + 3]
    Tp1 = h2[..., 2:ny + 2, 3:nx + 4]

    # maskLocW = (hFacW > 0) on the U-face stagger; the cube seam rotates u
    # stagger into v stagger, so the pair must swap together without a sign flip
    # (``exchange_hfac_pair`` = ``exchange_cubed_sphere_vector(with_signs=False)``,
    # the same routing the validated MIT tile path uses for maskW/maskS).
    hFacW = grid.masks.hFacW.to(dtype=local.dtype, device=local.device)
    hFacS = grid.masks.hFacS.to(dtype=local.dtype, device=local.device)
    maskW = (hFacW > 0).to(local.dtype)
    maskS = (hFacS > 0).to(local.dtype)
    extent = HaloExtentV1.uniform(2)
    maskW_h, _ = exchange_hfac_pair(maskW, maskS, grid=grid, extent=extent)
    maskW_h = (maskW_h > 0).to(local.dtype)  # square cube frame [*, n+4, n+4]
    # The cube vector exchange carries the U-face mask in a SQUARE (n+2*OL) halo
    # frame: interior cell c -> halo col c+2, so U-face f -> halo col f+2 (the
    # shared east seam face f=nx lands at col nx+2, supplied by the neighbour).
    # Face i needs faces i-1,i,i+1 = cols i+1, i+2, i+3 (interior rows 2:ny+2).
    mask_im1 = maskW_h[..., 2:ny + 2, 1:nx + 2]
    mask_i = maskW_h[..., 2:ny + 2, 2:nx + 3]
    mask_ip1 = maskW_h[..., 2:ny + 2, 3:nx + 4]

    # uCFL = |uFld(i) * dt * recip_dxC(i)| on every face (local; no halo).
    dxC = grid.metrics.dxC_for(local.dtype, local.device)  # [6, ny, nx+1]
    recip_dxC = torch.where(dxC > 0, 1.0 / dxC, torch.zeros_like(dxC)).unsqueeze(-3)
    u_cfl = (u_vel * float(dt) * recip_dxC).abs()

    # T_left = cell i-1 = Tm1, T_right = cell i = T0 (upwind donors; the six_face
    # path uses flux_dst on all faces, matching the all-face UPWIND fill).
    face_masks = (mask_im1, mask_i, mask_ip1)
    _, flux_dst = horizontal_stencil_flux(
        scheme, Tm1, T0, (Tm2, Tm1, T0, Tp1), face_masks, u_trans, u_cfl,
        # The cube masks carry no maskInW: they are the physical masks too.
        physical_face_masks=face_masks, cell_inputs=None,
    )
    return flux_dst


def _stencil_face_flux_6face_y(
    local: torch.Tensor,
    v_trans: torch.Tensor,
    v_vel: torch.Tensor,
    grid: Any,
    *,
    dt: float,
    scheme: TracerFluxScheme,
) -> torch.Tensor:
    """Y mirror of :func:`_stencil_face_flux_6face_x` (``gad_dst3fl_adv_y.F``)."""
    nx, ny = grid.nx, grid.ny
    h2 = _ol2_corner_halo(local, mode=2, grid=grid)  # [*, ny+4, nx+4]
    # Face j (0..ny): cells j-2,j-1,j,j+1 along axis -2; interior cols h2[2:nx+2].
    Tm2 = h2[..., 0:ny + 1, 2:nx + 2]
    Tm1 = h2[..., 1:ny + 2, 2:nx + 2]
    T0 = h2[..., 2:ny + 3, 2:nx + 2]
    Tp1 = h2[..., 3:ny + 4, 2:nx + 2]

    hFacW = grid.masks.hFacW.to(dtype=local.dtype, device=local.device)
    hFacS = grid.masks.hFacS.to(dtype=local.dtype, device=local.device)
    maskW = (hFacW > 0).to(local.dtype)
    maskS = (hFacS > 0).to(local.dtype)
    extent = HaloExtentV1.uniform(2)
    _, maskS_h = exchange_hfac_pair(maskW, maskS, grid=grid, extent=extent)
    maskS_h = (maskS_h > 0).to(local.dtype)  # square cube frame [*, n+4, n+4]
    # Square frame (see X): V-face f -> halo row f+2 (the shared north seam face
    # f=ny lands at row ny+2, supplied by the neighbour).
    # Face j needs faces j-1,j,j+1 = rows j+1, j+2, j+3 (interior cols 2:nx+2).
    mask_jm1 = maskS_h[..., 1:ny + 2, 2:nx + 2]
    mask_j = maskS_h[..., 2:ny + 3, 2:nx + 2]
    mask_jp1 = maskS_h[..., 3:ny + 4, 2:nx + 2]

    dyC = grid.metrics.dyC_for(local.dtype, local.device)  # [6, ny+1, nx]
    recip_dyC = torch.where(dyC > 0, 1.0 / dyC, torch.zeros_like(dyC)).unsqueeze(-3)
    v_cfl = (v_vel * float(dt) * recip_dyC).abs()

    face_masks = (mask_jm1, mask_j, mask_jp1)
    _, flux_dst = horizontal_stencil_flux(
        scheme, Tm1, T0, (Tm2, Tm1, T0, Tp1), face_masks, v_trans, v_cfl,
        physical_face_masks=face_masks, cell_inputs=None,
    )
    return flux_dst


def _x_flux_first_order(
    local: torch.Tensor,
    u_trans: torch.Tensor,
    grid: Any,
    *,
    h_scheme: TracerFluxScheme = TracerFluxScheme.UPWIND,
    u_vel: torch.Tensor | None = None,
    dt: float = 0.0,
) -> torch.Tensor:
    if h_scheme in HORIZONTAL_STENCIL_SCHEMES:
        if u_vel is None:
            raise ValueError("six_face DST3/DST3_FL X flux requires u_vel")
        return _stencil_face_flux_6face_x(
            local, u_trans, u_vel, grid,
            dt=dt, scheme=h_scheme,
        )
    halo = fill_cs_corner_tr_rl(local, mode=1, grid=grid)
    left = halo[..., 1 : grid.ny + 1, 0 : grid.nx + 1]
    right = halo[..., 1 : grid.ny + 1, 1 : grid.nx + 2]
    return horizontal_two_point_flux(h_scheme, left, right, u_trans, u_trans, None)


def _y_flux_first_order(
    local: torch.Tensor,
    v_trans: torch.Tensor,
    grid: Any,
    *,
    h_scheme: TracerFluxScheme = TracerFluxScheme.UPWIND,
    v_vel: torch.Tensor | None = None,
    dt: float = 0.0,
) -> torch.Tensor:
    if h_scheme in HORIZONTAL_STENCIL_SCHEMES:
        if v_vel is None:
            raise ValueError("six_face DST3/DST3_FL Y flux requires v_vel")
        return _stencil_face_flux_6face_y(
            local, v_trans, v_vel, grid,
            dt=dt, scheme=h_scheme,
        )
    halo = fill_cs_corner_tr_rl(local, mode=2, grid=grid)
    south = halo[..., 0 : grid.ny + 1, 1 : grid.nx + 1]
    north = halo[..., 1 : grid.ny + 2, 1 : grid.nx + 1]
    return horizontal_two_point_flux(h_scheme, south, north, v_trans, v_trans, None)


def _horizontal_gad_first_order_6face_approx(
    *,
    tracer: torch.Tensor,
    u_transport: torch.Tensor,
    v_transport: torch.Tensor,
    grid: Any,
    dt: float,
    recip_volume: torch.Tensor,
    c_mask: torch.Tensor,
    h_scheme: TracerFluxScheme = TracerFluxScheme.UPWIND,
    u_vel: torch.Tensor | None = None,
    v_vel: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local = tracer.clone()
    afx = torch.zeros_like(u_transport)
    afy = torch.zeros_like(v_transport)
    x_pass_faces = ((0, 1), (2, 3), (4, 5))
    y_pass_faces = ((3, 4), (0, 5), (1, 2))

    # Face-local: the pass schedule above carries GLOBAL face ids.  On a
    # rank-local grid (``_cube_step_ctx`` present) each rank commits only the
    # faces it owns, at their LOCAL leading-dim indices; the flux kernels still
    # run on the full local tensor and re-exchange halos via the now-distributed
    # ``fill_cs_corner_tr_rl`` between sub-commits, so cross-rank coupling stays
    # bitwise.  A rank owning none of a group commits nothing (empty index) but
    # still runs the flux + exchange in lockstep.  ctx=None -> global ids
    # unchanged (single-rank path byte-identical).
    ctx = getattr(grid, "_cube_step_ctx", None)

    def _commit_idx(global_group: tuple[int, ...]) -> torch.Tensor:
        if ctx is None:
            return torch.as_tensor(global_group, device=u_transport.device)
        owned = [ctx.global_to_local[f] for f in global_group if f in ctx.global_to_local]
        return torch.as_tensor(owned, device=u_transport.device, dtype=torch.long)

    for x_faces, y_faces in zip(x_pass_faces, y_pass_faces, strict=True):
        fx = _x_flux_first_order(
            local, u_transport, grid, h_scheme=h_scheme, u_vel=u_vel, dt=dt,
        )
        face_idx = _commit_idx(x_faces)
        afx.index_copy_(0, face_idx, fx.index_select(0, face_idx))
        x_delta = (
            fx[..., 1:] - fx[..., :-1]
            - tracer * (u_transport[..., 1:] - u_transport[..., :-1])
        )
        local.index_copy_(
            0,
            face_idx,
            (
                local.index_select(0, face_idx)
                - float(dt)
                * x_delta.index_select(0, face_idx)
                * recip_volume.index_select(0, face_idx)
                * c_mask.index_select(0, face_idx)
            ),
        )

        fy = _y_flux_first_order(
            local, v_transport, grid, h_scheme=h_scheme, v_vel=v_vel, dt=dt,
        )
        face_idx = _commit_idx(y_faces)
        afy.index_copy_(0, face_idx, fy.index_select(0, face_idx))
        y_delta = (
            fy[..., 1:, :] - fy[..., :-1, :]
            - tracer * (v_transport[..., 1:, :] - v_transport[..., :-1, :])
        )
        local.index_copy_(
            0,
            face_idx,
            (
                local.index_select(0, face_idx)
                - float(dt)
                * y_delta.index_select(0, face_idx)
                * recip_volume.index_select(0, face_idx)
                * c_mask.index_select(0, face_idx)
            ),
        )
    return local, afx, afy


def cube_horizontal_tracer_diffusion(
    tracer: torch.Tensor,
    grid: Any,
    diffKh: float,
) -> torch.Tensor:
    """Full-flux-form horizontal tracer diffusion on the six_face cube path.

    The cube counterpart of the regional ``HorizontalTracerDiffusion._diffuse``,
    but using the MIT-faithful FULL flux form
    (:func:`horizontal_tracer_diffusion_flux_form`) because the cube's 2-D-varying
    rotated metrics make the regional ``1/dxC^2`` reduction invalid (it is only
    bitwise-MIT when ``rA = dyG*dxC`` and ``hFacC == 1``).

    Adapter responsibilities (geometry lives HERE, not in the core):

    1. Exchange the tracer with an OL=1 corner-filled cube scalar halo — the SAME
       ``fill_cs_corner_tr_rl`` path the cube UPWIND tracer flux uses
       (``_x_flux_first_order`` / ``_y_flux_first_order``): ``mode=1`` for the
       X-stencil, ``mode=2`` for the Y-stencil.
    2. Cross-face-exchange the face metrics so the seam U/V-faces receive the
       NEIGHBOUR cell's ``dyG``/``recip_dxC``/``hFacW`` (resp. ``dxG``/
       ``recip_dyC``/``hFacS``).  Each is a U/V-stagger pair swapped without a
       sign flip via ``exchange_hfac_pair`` (= ``exchange_cubed_sphere_vector(
       with_signs=False)``) — exactly how the validated MIT tile DST3 path
       cross-face-exchanges ``(dxC, dyG)`` and ``(maskW, maskS)``
       (``_gad_advection_topology.horizontal_gad_first_order_cube_tiles``).

    ``exchange_hfac_pair`` returns each cube field in the same-size square haloed
    frame ``[F, nz, n+2, n+2]`` (OL=1, ``F=6`` single-rank / ``F=n_local`` per
    rank): interior cell ``c`` -> halo index ``c+1``, so a U-face ``f`` (0..nx)
    lands at halo column ``f+1`` (the shared east seam face ``nx`` at column
    ``nx+1``, supplied by the neighbour) and interior rows are ``[1:ny+1]``.

    Single-rank AND multi-card: ``fill_cs_corner_tr_rl`` and ``exchange_hfac_pair``
    both self-dispatch on ``grid._cube_step_ctx`` (single-rank cube exchange when
    ``None``, the distributed P2P seam when a :class:`CubeRankContext` is attached
    — the SAME routing the six_face DST3 path uses).  So pass a rank-LOCAL grid
    (``localize_cube_grid``, which carries ctx) on the face-local multi-card path
    and the GLOBAL grid single-rank; the body is identical, only the leading face
    dim (``6`` vs ``n_local``) and the dispatch differ.  All non-exchanged metrics
    (``recip_rA``/``recip_hFacC``/``rA``) are fetched from the SAME grid, so they
    stay shape-consistent with ``tracer``.

    Returns the diffusion tendency ``[F, nz, ny, nx]`` (tracer/s).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dtype, device = tracer.dtype, tracer.device
    # Leading face count of the data this call operates on: 6 single-rank, the
    # rank's owned-face count (``len(ctx.local_face_ids)``) on a localized grid.
    # The metric ``*_for`` accessors + masks below are scattered to the SAME local
    # faces by ``localize_cube_grid``, so they all carry ``nf`` leading.
    nf = tracer.shape[0]

    # (1) OL=1 corner-filled tracer halo (UPWIND tracer path).  Interior rows
    # [1:ny+1] -> [*, nz, ny, nx+2]; col i holds cell i-1 (matches the core's
    # tracer_x[..., 1:nx+2]-tracer_x[..., 0:nx+1] = T(i)-T(i-1) U-face stencil).
    halo_x = fill_cs_corner_tr_rl(tracer, mode=1, grid=grid)  # [nf, nz, ny+2, nx+2]
    halo_y = fill_cs_corner_tr_rl(tracer, mode=2, grid=grid)  # [nf, nz, ny+2, nx+2]
    tracer_x = halo_x[..., 1 : ny + 1, :]
    tracer_y = halo_y[..., 0 : ny + 2, 1 : nx + 1]

    # (2) Cross-face-exchange the static face metrics (OL=1, square frame).
    extent = HaloExtentV1.uniform(1)
    dyG = grid.metrics.dyG_for(dtype, device)  # [nf, ny, nx+1] (U-face)
    dxG = grid.metrics.dxG_for(dtype, device)  # [nf, ny+1, nx] (V-face)
    dyG3d = dyG.unsqueeze(1).expand(nf, nz, dyG.shape[-2], dyG.shape[-1]).contiguous()
    dxG3d = dxG.unsqueeze(1).expand(nf, nz, dxG.shape[-2], dxG.shape[-1]).contiguous()
    dyG_h, dxG_h = exchange_hfac_pair(dyG3d, dxG3d, grid=grid, extent=extent)

    dxC = grid.metrics.dxC_for(dtype, device)  # [nf, ny, nx+1] (U-face)
    dyC = grid.metrics.dyC_for(dtype, device)  # [nf, ny+1, nx] (V-face)
    recip_dxC = torch.where(dxC > 0, 1.0 / dxC, torch.zeros_like(dxC))
    recip_dyC = torch.where(dyC > 0, 1.0 / dyC, torch.zeros_like(dyC))
    rdxC3d = recip_dxC.unsqueeze(1).expand(nf, nz, *recip_dxC.shape[-2:]).contiguous()
    rdyC3d = recip_dyC.unsqueeze(1).expand(nf, nz, *recip_dyC.shape[-2:]).contiguous()
    rdxC_h, rdyC_h = exchange_hfac_pair(rdxC3d, rdyC3d, grid=grid, extent=extent)

    hFacW = grid.masks.hFacW.to(dtype=dtype, device=device)  # [nf, nz, ny, nx+1]
    hFacS = grid.masks.hFacS.to(dtype=dtype, device=device)  # [nf, nz, ny+1, nx]
    hFacW_h, hFacS_h = exchange_hfac_pair(hFacW, hFacS, grid=grid, extent=extent)

    # Slice each square frame back to U-faces 0..nx (cols 1:nx+2, rows 1:ny+1)
    # and V-faces 0..ny (rows 1:ny+2, cols 1:nx+1).
    dyG_x = dyG_h[..., 1 : ny + 1, 1 : nx + 2]
    rdxC_x = rdxC_h[..., 1 : ny + 1, 1 : nx + 2]
    hFacW_x = hFacW_h[..., 1 : ny + 1, 1 : nx + 2]
    dxG_y = dxG_h[..., 1 : ny + 2, 1 : nx + 1]
    rdyC_y = rdyC_h[..., 1 : ny + 2, 1 : nx + 1]
    hFacS_y = hFacS_h[..., 1 : ny + 2, 1 : nx + 1]

    hFacC = grid.masks.hFacC.to(dtype=dtype, device=device)  # [nf, nz, ny, nx]
    recip_hFacC = _positive_reciprocal(hFacC)
    rA = grid.metrics.cell_area_for(dtype, device)  # [nf, ny, nx]
    recip_rA = torch.where(rA > 0, 1.0 / rA, torch.zeros_like(rA)).unsqueeze(1)

    return horizontal_tracer_diffusion_flux_form(
        tracer_x,
        tracer_y,
        dyG_x=dyG_x,
        dxG_y=dxG_y,
        recip_dxC_x=rdxC_x,
        recip_dyC_y=rdyC_y,
        hFacW_x=hFacW_x,
        hFacS_y=hFacS_y,
        recip_rA=recip_rA,
        recip_hFacC=recip_hFacC,
        diffKh=tracer.new_tensor(float(diffKh)),
        nx=nx,
        ny=ny,
        maskInC=None,
    )


def gad_advection_first_order_cube(
    *,
    tracer: torch.Tensor,
    u_transport: torch.Tensor,
    v_transport: torch.Tensor,
    vertical_volume_flux: torch.Tensor,
    grid: Any,
    dt: float,
    h_scheme: TracerFluxScheme = TracerFluxScheme.UPWIND,
    v_scheme: TracerFluxScheme = TracerFluxScheme.UPWIND,
    u_vel: torch.Tensor | None = None,
    v_vel: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Replay non-compressible MIT GAD_ADVECTION on cube faces.

    ``h_scheme`` selects the horizontal flux reconstruction.  UPWIND (default)
    is the audited MIT 1st-order upwind path (bitwise no-op vs the pre-scheme
    code).  DST3 / DST3_FL run the shared geometry-neutral DST3 core on the
    MIT tile faces (``gad_dst3fl_adv_{x,y}.F``); they require the face
    VELOCITIES ``u_vel`` / ``v_vel`` (= ``u_transport / (dyG·drF·hFacW)`` etc.,
    computed at the call site) for the CFL number.

    ``v_scheme`` selects the vertical R-pass reconstruction: UPWIND (default,
    byte-identical baseline) or DST3_FL (MITgcm scheme 33) via the SAME shared
    kernel the region R-pass uses (``vertical_advective_flux_dst3fl``).  The shared
    vertical core derives the interface velocity from ``r_trans/cell_area`` for the
    DST3-FL Courant number, so no extra ``w_vel`` argument is needed.

    Each directional update uses ``flux_div - tracer_n * transport_div`` so the
    locally updated tracer follows MIT's advective-form multidimensional path.
    The horizontal path is chosen by the grid's ``advection_topology``
    DECLARATION: ``mit_12tile`` runs MIT's 12-tile, OL=4 cube pass schedule (and
    each rank carries two tiles per owned face, and the runtime state must
    match that ownership, else ``RuntimeError``); ``six_face_approx``
    runs the six-face approximation.  Setting both ``h_scheme`` and
    ``v_scheme`` to DST3_FL reproduces MIT production (tempAdvScheme=33 +
    tempVertAdvScheme=33).
    """
    if h_scheme not in CUBE_HORIZONTAL_SCHEMES:
        raise ValueError(
            f"cube horizontal tracer advection supports {CUBE_HORIZONTAL_SCHEMES}, "
            f"not {h_scheme!r}"
        )
    nz = grid.nz
    drf = grid.layer_thickness_for(u_transport.dtype, u_transport.device).view(1, nz, 1, 1)
    hfac = grid.masks.hFacC.to(dtype=u_transport.dtype, device=u_transport.device)
    c_mask = (hfac > 0).to(u_transport.dtype)
    r_a = grid.metrics.cell_area_for(u_transport.dtype, u_transport.device).unsqueeze(1)
    volume = r_a * drf * hfac
    recip_volume = _positive_reciprocal(volume)

    # Horizontal path selection reads the grid's DECLARATION — the single source
    # of truth, validated at construction (``CubedSphereGrid.__post_init__``).
    # There is no runtime probe and no env override: a run that wants the
    # six-face path on a cs32 fixture declares it at construction (e.g.
    # ``from_cs32_grid(..., advection_topology="six_face_approx")``), which is
    # visible in the grid rather than in ambient process state.
    if grid.advection_topology == "mit_12tile":
        # Declaration is authoritative; the state must match it or we raise.
        assert_mit_12tile_runtime_shapes(
            grid=grid,
            tracer=tracer,
            u_transport=u_transport,
            v_transport=v_transport,
            vertical_volume_flux=vertical_volume_flux,
        )
        # afx/afy sidecar is the 1st-order upwind six_face flux (non-gating on
        # the tile path); the gating local_t3d is the MIT tile schedule on
        # ``grid.mit_tiling``, scheme-dispatched.
        _, afx, afy = _horizontal_gad_first_order_6face_approx(
            tracer=tracer,
            u_transport=u_transport,
            v_transport=v_transport,
            grid=grid,
            dt=dt,
            recip_volume=recip_volume,
            c_mask=c_mask,
        )
        local_t3d = horizontal_gad_first_order_cube_tiles(
            tracer=tracer,
            u_transport=u_transport,
            v_transport=v_transport,
            grid=grid,
            dt=dt,
            h_scheme=h_scheme,
            u_vel=u_vel,
            v_vel=v_vel,
        )
        horizontal_scheme = "mit_12tile"
        flux_sidecar_kind = "six_face_approx_non_gating"
    elif grid.advection_topology == "six_face_approx":
        if h_scheme in HORIZONTAL_STENCIL_SCHEMES and (
            u_vel is None or v_vel is None
        ):
            raise ValueError(
                "cube six_face_approx DST3/DST3_FL tracer advection requires "
                "u_vel and v_vel (face velocities) for the CFL number"
            )
        local_t3d, afx, afy = _horizontal_gad_first_order_6face_approx(
            tracer=tracer,
            u_transport=u_transport,
            v_transport=v_transport,
            grid=grid,
            dt=dt,
            recip_volume=recip_volume,
            c_mask=c_mask,
            h_scheme=h_scheme,
            u_vel=u_vel,
            v_vel=v_vel,
        )
        horizontal_scheme = "six_face_approx"
        flux_sidecar_kind = "production_6face_approx"
    else:
        # The declaration can be absent on __post_init__-bypassing views
        # (object.__new__); absent or unknown values must raise here rather
        # than silently take a branch.
        raise ValueError(
            "cube GAD requires grid.advection_topology in "
            f"{{'mit_12tile', 'six_face_approx'}}; got {grid.advection_topology!r}"
        )
    horizontal_tendency = (local_t3d - tracer) / float(dt)

    # Unified vertical tracer advection via the shared geometry-neutral core
    # (also used by the regional VerticalTracerTransport -> one kernel, not two).

    # LOCOMgcm downward-positive interface transport by NEGATING at this
    # injection boundary; surface (k=0) and seafloor (k=nz) interface transports
    # are zeroed.  The flux donor is the post-horizontal running tracer
    # ``local_t3d``; the ``-T*div(rTrans)`` correction uses the ORIGINAL
    # ``tracer`` (MIT gad_advection.F:572).  recip_hFacC is applied here (the core
    # returns the /rA /drF tendency), matching MIT's separate recip factors
    # (gad_advection.F:500-502) and the regional evaluate() FP ordering.
    r_trans = (-vertical_volume_flux).clone()
    r_trans[:, 0] = 0.0
    r_trans[:, nz] = 0.0
    drf_1d = grid.layer_thickness_for(u_transport.dtype, u_transport.device)
    recip_hfac = _positive_reciprocal(hfac)
    # Vertical scheme: UPWIND (default, byte-id baseline) or DST3_FL (MITgcm
    # scheme 33, the same shared kernel the region R-pass uses).  ONE knob turns on
    # both horizontal AND vertical DST3-FL (matches MIT production tempAdvScheme=33
    # + tempVertAdvScheme=33).  The shared core derives the interface VELOCITY from
    # r_trans/cell_area for the DST3-FL CFL (consistent by construction).
    if v_scheme not in CUBE_VERTICAL_SCHEMES:
        raise ValueError(
            f"cube vertical tracer advection supports UPWIND or DST3_FL, not {v_scheme!r}"
        )
    recip_drC = None
    if v_scheme in VERTICAL_CFL_SCHEMES:
        recip_drC = interface_recip_drC(grid, u_transport.dtype, u_transport.device)
    vertical_tendency_raw = vertical_tracer_advection_transport_space(
        local_t3d,
        r_trans,
        velocity=None,
        cell_area=r_a,
        layer_thickness=drf_1d,
        scheme=v_scheme,
        mask=c_mask,
        interior_mask=None,
        tracer_for_correction=tracer,
        recip_drC=recip_drC,
        dt=dt,
    )
    final = (local_t3d + float(dt) * vertical_tendency_raw * recip_hfac) * c_mask
    vertical_tendency = (final - local_t3d) / float(dt)
    # MIT upward-positive vertical advective flux for op-level diagnostics / Gate A
    # (negate the downward-positive core flux); matches MITgcm ``fVerT``.  DST3-FL:
    # the kernel is velocity-weighted, so multiply by rA to recover the MIT volume
    # flux units before negating.
    if v_scheme in VERTICAL_VELOCITY_FLUX_SCHEMES:
        fver = -vertical_tracer_flux(
            v_scheme, local_t3d, None, velocity=r_trans / r_a, recip_drC=recip_drC, dt=dt, mask=c_mask,
            layer_thickness=None, recip_center_spacing=None,
        ) * r_a
    else:
        fver = -vertical_tracer_flux(
            v_scheme, local_t3d, r_trans, velocity=None, recip_drC=None, dt=None, mask=c_mask,
            layer_thickness=None, recip_center_spacing=None,
        )

    return {
        "horizontal_tendency": horizontal_tendency,
        "vertical_tendency": vertical_tendency,
        "total_tendency": (final - tracer) / float(dt),
        "local_t3d_after_horizontal": local_t3d,
        "horizontal_scheme": horizontal_scheme,
        "flux_sidecar_kind": flux_sidecar_kind,
        "afx": afx,
        "afy": afy,
        "fver": fver,
        "final": final * c_mask,
    }
