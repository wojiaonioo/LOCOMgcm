from __future__ import annotations

import os
from typing import Literal

import torch
import torch.distributed as dist

from defgcm.distributed.comm_adjoint import CommSession, session_for_grid, transfer
from defgcm.geometry import StructuredCGrid
from defgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from defgcm.geometry.grid import (
    device_resident_masks_enabled,
    record_device_resident_use,
)


_HALO_X = HaloExtentV1(
    west_left_extent=1,
    east_right_extent=1,
    south_bottom_extent=0,
    north_top_extent=0,
)
_HALO_Y = HaloExtentV1(
    west_left_extent=0,
    east_right_extent=0,
    south_bottom_extent=1,
    north_top_extent=1,
)
_HALO_XY = HaloExtentV1.uniform(1)


def _cgrid_trace(message: str) -> None:
    if os.environ.get("DEFGCM_DIST_TRACE") != "1":
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[rank{rank}] c_grid: {message}", flush=True)


def _axis_is_decomposed(grid: "StructuredCGrid", axis: str) -> bool:
    """True if the grid is domain-decomposed (multi-rank) along ``axis``.

    Mirror of ``implicit_free_surface_cg2d._axis_is_decomposed`` kept local to
    avoid an operators -> steppers import cycle.
    """
    layout = getattr(grid, "parallel_layout", None)
    if layout is None:
        return False
    parts_attr = "nPx" if axis == "x" else "nPy"
    return int(getattr(layout, parts_attr, 1)) > 1


def _edge_pair_mask(
    length: int,
    *,
    grid: StructuredCGrid,
    axis: Literal["x", "y"],
    extent: HaloExtentV1,
    device: torch.device,
) -> torch.Tensor | None:
    if grid.edge_face_couples(axis):  # periodic/cube paths keep all edge pairs.
        return None
    core = torch.ones(length, dtype=torch.bool, device=device)
    core_ext = exchange(
        core,
        grid=grid,
        location="F",
        extent=extent,
        fill=FillSpec.ZERO,
    )
    return core_ext[:-1] & core_ext[1:]


def center_to_u(field: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    field_x = exchange(
        field,
        grid=grid,
        location="T",
        extent=_HALO_X,
        fill=FillSpec.ZERO,
    )
    interpolated = torch.zeros(
        (*field.shape[:-2], grid.ny, grid.nx + 1),
        dtype=field.dtype,
        device=field.device,
    )
    interpolated[...] = 0.5 * (
        field_x[..., : grid.nx + 1] + field_x[..., 1 : grid.nx + 2]
    )
    # For wall/open edges the ZERO ghost makes the edge interpolation
    # meaningless.  A distributed tile may have only one physical edge and one
    # internal rank seam, so build a per-face support mask through exchange().
    mask = _edge_pair_mask(
        grid.nx,
        grid=grid,
        axis="x",
        extent=_HALO_X,
        device=field.device,
    )
    if mask is not None:
        view = mask.view(*([1] * (interpolated.ndim - 1)), grid.nx + 1)
        interpolated = torch.where(view, interpolated, torch.zeros_like(interpolated))
    return interpolated


def center_to_v(field: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    field_y = exchange(
        field,
        grid=grid,
        location="T",
        extent=_HALO_Y,
        fill=FillSpec.ZERO,
    )
    interpolated = torch.zeros(
        (*field.shape[:-2], grid.ny + 1, grid.nx),
        dtype=field.dtype,
        device=field.device,
    )
    interpolated[...] = 0.5 * (
        field_y[..., : grid.ny + 1, :] + field_y[..., 1 : grid.ny + 2, :]
    )
    # For wall/open edges the ZERO ghost makes the edge interpolation
    # meaningless.  A distributed tile may have only one physical edge and one
    # internal rank seam, so build a per-face support mask through exchange().
    mask = _edge_pair_mask(
        grid.ny,
        grid=grid,
        axis="y",
        extent=_HALO_Y,
        device=field.device,
    )
    if mask is not None:
        view = mask.view(*([1] * (interpolated.ndim - 2)), grid.ny + 1, 1)
        interpolated = torch.where(view, interpolated, torch.zeros_like(interpolated))
    return interpolated


def _dxC_bcast(grid: StructuredCGrid, tensor: torch.Tensor) -> torch.Tensor:
    """Return full ``dxC`` broadcast-reshaped for ``(..., ny, nx+1)`` fields.

    The accessor retains latitude variation on spherical-polar grids and both
    axes on vector Cartesian grids.  Cubed-sphere metrics also retain the face
    dimension and gain only the vertical broadcast dimension.
    """
    metric = grid.metrics.dxC_for(tensor.dtype, tensor.device)
    if metric.ndim == 3:
        return metric.unsqueeze(1)
    lead = max(tensor.ndim - 2, 0)
    return metric.view(*([1] * lead), grid.ny, grid.nx + 1)


def gradient_center_to_u(field: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    field_x = exchange(
        field,
        grid=grid,
        location="T",
        extent=_HALO_X,
        fill=FillSpec.REPLICATE,
    )
    grad = torch.zeros(
        (*field.shape[:-2], grid.ny, grid.nx + 1),
        dtype=field.dtype,
        device=field.device,
    )
    dx_metric = _dxC_bcast(grid, field)                                  # [..., ny, nx+1]
    grad[...] = (
        field_x[..., 1 : grid.nx + 2] - field_x[..., : grid.nx + 1]
    ) / dx_metric
    if grid.masks.has_3d_masks and field.ndim >= 4:
        grad = grad * grid.masks.u_float(dtype=field.dtype, device=field.device)
    return grad


def gradient_center_to_v(field: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    field_y = exchange(
        field,
        grid=grid,
        location="T",
        extent=_HALO_Y,
        fill=FillSpec.REPLICATE,
    )
    grad = torch.zeros(
        (*field.shape[:-2], grid.ny + 1, grid.nx),
        dtype=field.dtype,
        device=field.device,
    )
    # dy = dyC (meridional centre-to-centre distance) at the v-face: the proper
    # denominator of this center→v gradient.  2-D on cubed-sphere / variable-
    # spacing grids, bit-equal to the scalar on uniform grids.
    dy = grid.metrics.dyC_for(field.dtype, field.device)
    grad[...] = (
        field_y[..., 1 : grid.ny + 2, :] - field_y[..., : grid.ny + 1, :]
    ) / dy
    if grid.masks.has_3d_masks and field.ndim >= 4:
        grad = grad * grid.masks.v_float(dtype=field.dtype, device=field.device)
    return grad


def gradient_center_from_haloed_scalar(
    field_halo: torch.Tensor,
    *,
    dxC_u: torch.Tensor,
    dyC_v: torch.Tensor,
    wet_center_halo: torch.Tensor | None = None,
    halo: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center-scalar gradient on U/V faces from an already exchanged halo.

    The caller owns geometry-specific exchange and metric construction.  This
    tensor-only stencil consumes a scalar center halo, U/V face metrics, and an
    optional center wet mask halo.  It returns the positive gradients
    ``(dphi/dx, dphi/dy)``.
    """
    ny = dxC_u.shape[-2]
    nx = dxC_u.shape[-1] - 1
    ol = halo

    d_u = (
        field_halo[..., ol:ol + ny, ol:ol + nx + 1]
        - field_halo[..., ol:ol + ny, ol - 1:ol + nx]
    )
    d_v = (
        field_halo[..., ol:ol + ny + 1, ol:ol + nx]
        - field_halo[..., ol - 1:ol + ny, ol:ol + nx]
    )

    grad_u = torch.where(dxC_u > 0, d_u / dxC_u, torch.zeros_like(d_u))
    grad_v = torch.where(dyC_v > 0, d_v / dyC_v, torch.zeros_like(d_v))

    if wet_center_halo is not None:
        mask_w = (
            wet_center_halo[..., ol:ol + ny, ol:ol + nx + 1]
            * wet_center_halo[..., ol:ol + ny, ol - 1:ol + nx]
        )
        mask_s = (
            wet_center_halo[..., ol:ol + ny + 1, ol:ol + nx]
            * wet_center_halo[..., ol - 1:ol + ny, ol:ol + nx]
        )
        grad_u = grad_u * mask_w
        grad_v = grad_v * mask_s

    return grad_u, grad_v


def gradient_eta_to_u(eta: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    return gradient_center_to_u(eta, grid)


def gradient_eta_to_v(eta: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
    return gradient_center_to_v(eta, grid)


def zonal_face_difference(field: torch.Tensor) -> torch.Tensor:
    """Difference adjacent C-grid face values along x."""
    return field[..., 1:] - field[..., :-1]


def meridional_face_difference(field: torch.Tensor) -> torch.Tensor:
    """Difference adjacent C-grid face values along y."""
    return field[..., 1:, :] - field[..., :-1, :]


def edge_coefficient_helmholtz_from_haloed_center(
    field_halo: torch.Tensor,
    *,
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
    halo: int = 1,
) -> torch.Tensor:
    """Five-point edge-coefficient Helmholtz stencil from a center halo.

    The caller owns geometry-specific scalar exchange.  Coefficients are
    expected to be broadcast-ready against ``field_halo``'s leading dimensions.
    """
    ny, nx = aC.shape[-2:]
    ol = halo
    inner = field_halo[..., ol:ol + ny, ol:ol + nx]
    west = field_halo[..., ol:ol + ny, ol - 1:ol + nx - 1]
    east = field_halo[..., ol:ol + ny, ol + 1:ol + nx + 1]
    south = field_halo[..., ol - 1:ol + ny - 1, ol:ol + nx]
    north = field_halo[..., ol + 1:ol + ny + 1, ol:ol + nx]

    aW_w = aW_edge[..., 0:nx]
    aW_e = aW_edge[..., 1:nx + 1]
    aS_s = aS_edge[..., 0:ny, :]
    aS_n = aS_edge[..., 1:ny + 1, :]

    result = aC * inner
    result = result + aW_w * west + aW_e * east
    result = result + aS_s * south + aS_n * north
    return result


def helmholtz_edge_coefficients_from_depth(
    H_u: torch.Tensor,
    H_v: torch.Tensor,
    dyG: torch.Tensor,
    dxC_edge: torch.Tensor,
    dxG_edge: torch.Tensor,
    dyC_edge: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT-style free-surface Helmholtz edge coefficients from wet depth."""
    den_w = torch.where(dxC_edge != 0, dxC_edge, torch.ones_like(dxC_edge))
    aW_edge = torch.where(
        dxC_edge != 0,
        -scale * dyG * H_u / den_w,
        torch.zeros_like(dxC_edge),
    )
    aS_edge = -scale * dxG_edge * H_v / dyC_edge
    return aW_edge, aS_edge


def incident_edge_coefficient_sum(
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
) -> torch.Tensor:
    """Sum the west/east/south/north edge coefficients incident on each cell."""
    return (
        aW_edge[..., :-1] + aW_edge[..., 1:]
        + aS_edge[..., :-1, :] + aS_edge[..., 1:, :]
    )


def mask_edge_coefficients_from_center_mask(
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    cell_mask: torch.Tensor,
    *,
    periodic_x: bool = False,
    periodic_y: bool = False,
    seam_x: bool = False,
    seam_y: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask U/V edge coefficients by adjacent center-cell wetness."""
    nx = cell_mask.shape[-1]
    ny = cell_mask.shape[-2]
    aW_out = aW_edge.clone()
    aS_out = aS_edge.clone()

    aW_out[..., 1:nx] = (
        aW_out[..., 1:nx] * cell_mask[..., :-1] * cell_mask[..., 1:]
    )
    if seam_x:
        aW_out[..., 0] = aW_out[..., 0] * cell_mask[..., 0]
    elif periodic_x:
        wrap_x = cell_mask[..., 0] * cell_mask[..., -1]
        aW_out[..., 0] = aW_out[..., 0] * wrap_x
        aW_out[..., nx] = aW_out[..., nx] * wrap_x

    aS_out[..., 1:ny, :] = (
        aS_out[..., 1:ny, :] * cell_mask[..., :-1, :] * cell_mask[..., 1:, :]
    )
    if seam_y:
        aS_out[..., 0, :] = aS_out[..., 0, :] * cell_mask[..., 0, :]
    elif periodic_y:
        wrap_y = cell_mask[..., 0, :] * cell_mask[..., -1, :]
        aS_out[..., 0, :] = aS_out[..., 0, :] * wrap_y
        aS_out[..., ny, :] = aS_out[..., ny, :] * wrap_y

    return aW_out, aS_out


def face_volume_transport(
    u: torch.Tensor,
    v: torch.Tensor,
    layer_thickness: torch.Tensor,
    dyG: torch.Tensor,
    dxG: torch.Tensor,
    *,
    hFacW: torch.Tensor | None = None,
    hFacS: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer C-grid face volume transports.

    This tensor-only kernel computes ``u * drF * hFacW * dyG`` and
    ``v * drF * hFacS * dxG``.  Missing hFac tensors fall back to the all-wet
    expression for that component, preserving callers that have only binary or
    no partial-cell masks.
    """
    if hFacW is not None:
        u_transport = u * layer_thickness * hFacW * dyG
    else:
        u_transport = u * layer_thickness * dyG
    if hFacS is not None:
        v_transport = v * layer_thickness * hFacS * dxG
    else:
        v_transport = v * layer_thickness * dxG
    return u_transport, v_transport


def active_face_velocity_from_transport_area(
    u: torch.Tensor,
    v: torch.Tensor,
    u_face_area: torch.Tensor,
    v_face_area: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero C-grid face velocities where the transport area is inactive."""
    return (
        torch.where(u_face_area > 0, u, torch.zeros_like(u)),
        torch.where(v_face_area > 0, v, torch.zeros_like(v)),
    )


def depth_integrated_face_thickness(
    layer_thickness: torch.Tensor,
    *,
    hFacW: torch.Tensor | None = None,
    hFacS: torch.Tensor | None = None,
    u_shape: tuple[int, ...] | torch.Size | None = None,
    v_shape: tuple[int, ...] | torch.Size | None = None,
    vertical_dim: int = -3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth-integrated wet thickness on U/V faces.

    ``layer_thickness`` must be broadcast-ready against the supplied hFac
    tensors.  If an hFac tensor is omitted, the corresponding output is an
    all-wet column of total thickness expanded to ``u_shape`` or ``v_shape``.
    """
    if hFacW is not None:
        H_u = (layer_thickness * hFacW).sum(dim=vertical_dim)
    else:
        if u_shape is None:
            raise ValueError("u_shape is required when hFacW is omitted")
        H_u = layer_thickness.sum().expand(tuple(u_shape))

    if hFacS is not None:
        H_v = (layer_thickness * hFacS).sum(dim=vertical_dim)
    else:
        if v_shape is None:
            raise ValueError("v_shape is required when hFacS is omitted")
        H_v = layer_thickness.sum().expand(tuple(v_shape))

    return H_u, H_v


def _distributed_neighbor_rank(grid: StructuredCGrid, side: str) -> int | None:
    neighbours = getattr(grid, "distributed_neighbors", None)
    if neighbours is None:
        neighbours = getattr(grid, "distributed_neighbours", None)
    if not neighbours:
        return None
    entry = neighbours.get(side) if hasattr(neighbours, "get") else None
    if entry is None:
        return None
    if isinstance(entry, int):
        return entry
    if isinstance(entry, dict):
        rank = entry.get("rank")
        return None if rank is None else int(rank)
    rank = getattr(entry, "rank", None)
    return None if rank is None else int(rank)


def _sync_owned_edge_from_neighbor(
    tensor: torch.Tensor,
    *,
    grid: StructuredCGrid,
    axis: str,
    receive_side: str,
    send_side: str,
    recv_slice: tuple[object, ...],
    send_slice: tuple[object, ...],
    session: CommSession | None = None,
) -> torch.Tensor:
    neighbours = getattr(grid, "distributed_neighbors", None)
    if neighbours is None:
        neighbours = getattr(grid, "distributed_neighbours", None)
    if not neighbours:
        # One tile spans the axis.  Its upper face is shared only when the axis
        # couples its edge faces, and then the owner is this tile.
        if not grid.edge_face_couples(axis):
            return tensor
        out = tensor.clone()
        out[recv_slice] = tensor[send_slice]
        return out

    recv_peer = _distributed_neighbor_rank(grid, receive_side)
    send_peer = _distributed_neighbor_rank(grid, send_side)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = getattr(getattr(grid, "parallel_layout", None), "rank", None)
    if recv_peer is not None and recv_peer == rank:
        # A periodic axis that is not split wraps onto this tile.
        out = tensor.clone()
        out[recv_slice] = tensor[send_slice]
        return out
    send_peer = None if send_peer == rank else send_peer
    if recv_peer is None and send_peer is None:
        return tensor
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    recv = torch.empty_like(tensor[recv_slice]) if recv_peer is not None else None
    send = tensor[send_slice].contiguous() if send_peer is not None else None
    # fg_fix : one opaque custom op for the NCCL point-to-point round
    # (receive posted first, then send, then wait) -- same pairing as before.
    received = transfer(
        [send] if send is not None else [],
        [recv] if recv is not None else [],
        [send_peer] if send_peer is not None else [],
        [recv_peer] if recv_peer is not None else [],
        [0] if send_peer is not None else [],
        [0] if recv_peer is not None else [],
        session=session,
    )
    recv = received[0] if received else None
    if recv is None:
        if session is None:
            from defgcm.distributed.halo_p2p import inference_ghost_after_send

            ordered = inference_ghost_after_send(send)
            if ordered is not None:
                out = tensor.clone()
                out[send_slice] = ordered
                return out
        return tensor
    out = tensor.clone()
    out[recv_slice] = recv
    return out


def synchronize_regional_face_values(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
    *,
    session: CommSession | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill duplicate regional east/north faces from the west/south face of their owner.

    Each shared C-grid face has one physical value. This applies both to the
    velocities and to their depth-integrated metric transports: independent
    reductions of identical columns can have different rounding on two ranks.
    Physical boundaries have no peer. Cube faces use their rotating vector exchange.
    """
    session = session_for_grid(grid, session)
    if getattr(grid, "face_count", 1) != 1:
        return u, v
    u_out = _sync_owned_edge_from_neighbor(
        u,
        grid=grid,
        axis="x",
        receive_side="east",
        send_side="west",
        recv_slice=(..., slice(-1, None)),
        send_slice=(..., slice(0, 1)),
        session=session,
    )
    v_out = _sync_owned_edge_from_neighbor(
        v,
        grid=grid,
        axis="y",
        receive_side="north",
        send_side="south",
        recv_slice=(..., slice(-1, None), slice(None)),
        send_slice=(..., slice(0, 1), slice(None)),
        session=session,
    )
    return u_out, v_out


def depth_integrated_face_transport(
    u: torch.Tensor,
    v: torch.Tensor,
    layer_thickness: torch.Tensor,
    dyG: torch.Tensor,
    dxG: torch.Tensor,
    *,
    hFacW: torch.Tensor | None = None,
    hFacS: torch.Tensor | None = None,
    vertical_dim: int = 1,
    keepdim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth-integrate C-grid face volume transports.

    This tensor-only kernel computes the MIT-style face transports
    ``sum_k(u * drF * hFacW * dyG)`` and
    ``sum_k(v * drF * hFacS * dxG)``.  The caller supplies geometry-specific
    metrics in broadcast-ready shapes.
    """
    u_transport, v_transport = face_volume_transport(
        u,
        v,
        layer_thickness,
        dyG,
        dxG,
        hFacW=hFacW,
        hFacS=hFacS,
    )
    return (
        u_transport.sum(dim=vertical_dim, keepdim=keepdim),
        v_transport.sum(dim=vertical_dim, keepdim=keepdim),
    )


def horizontal_volume_divergence_from_transports(
    u_volume_transport: torch.Tensor,
    v_volume_transport: torch.Tensor,
    *,
    wet_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unnormalized C-grid horizontal volume divergence from face transports.

    The inputs are already metric-weighted face transports.  The returned tensor
    is ``dU + dV`` before division by cell area, matching the volume-flux form
    consumed by MIT-style vertical velocity / tracer-continuity kernels.
    """
    div_u = zonal_face_difference(u_volume_transport)
    div_v = meridional_face_difference(v_volume_transport)
    result = div_u + div_v
    if wet_mask is not None:
        result = result * wet_mask
    return result


def tracer_flux_divergence_to_center(
    flux_u: torch.Tensor,
    flux_v: torch.Tensor,
    grid: StructuredCGrid,
) -> torch.Tensor:
    """Geometry-aware center divergence for TRACER point-fluxes.

    Row-uniform grids keep the legacy Cartesian quotient
    :func:`divergence_transport_to_center` (bitwise no-op).  J-varying
    geometry (spherical rows / vector spacing) uses the MIT form
    (``calc_adv_flow.F`` + ``gad_calc_rhs.F:764-779``): weight faces by
    ``dyG``/``dxG``, difference, and normalize by exact ``rA`` — the quotient
    ``diff/dxF + diff/dyF`` is not the spherical divergence.

    Only tracer advection consumes this; the shared quotient operator keeps
    its other consumers (w diagnosis, conservation, transport correction)
    untouched.
    """
    if (
        grid.metrics.is_uniform_x
        and grid.metrics.is_uniform_y
        and grid.metrics.has_uniform_row_geometry
    ):
        return divergence_transport_to_center(flux_u, flux_v, grid)
    dtype, device = flux_u.dtype, flux_u.device
    ny, nx = grid.ny, grid.nx
    lead = max(flux_u.ndim - 2, 0)
    dyG = grid.metrics.dyG_for(dtype, device).view(*([1] * lead), ny, nx + 1)
    dxG = grid.metrics.dxG_for(dtype, device).view(*([1] * lead), ny + 1, nx)
    rA = grid.metrics.rA_for(dtype, device).view(*([1] * lead), ny, nx)
    recip_rA = torch.where(rA != 0, 1.0 / rA, torch.zeros_like(rA))
    result = (
        zonal_face_difference(flux_u * dyG)
        + meridional_face_difference(flux_v * dxG)
    ) * recip_rA
    if grid.masks.has_3d_masks and flux_u.ndim >= 4:
        result = result * grid.masks.center_float(dtype=dtype, device=device)
    return result


def divergence_transport_to_center(
    u_transport: torch.Tensor,
    v_transport: torch.Tensor,
    grid: StructuredCGrid,
) -> torch.Tensor:
    dx = grid.metrics.dxF_for(u_transport.dtype, u_transport.device)
    dy = grid.metrics.dyF_for(v_transport.dtype, v_transport.device)
    if dx.ndim == 3:
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)
    else:
        lead = max(u_transport.ndim - 2, 0)
        dx = dx.view(*([1] * lead), grid.ny, grid.nx)
        dy = dy.view(*([1] * lead), grid.ny, grid.nx)
    result = (
        zonal_face_difference(u_transport) / dx
        + meridional_face_difference(v_transport) / dy
    )
    if grid.masks.has_3d_masks and u_transport.ndim >= 4:
        result = result * grid.masks.center_float(dtype=u_transport.dtype, device=u_transport.device)
    return result


def divergence_volume_transport_to_center_mit(
    u_volume_transport: torch.Tensor,
    v_volume_transport: torch.Tensor,
    grid: StructuredCGrid,
    *,
    rA: "torch.Tensor | None" = None,
    wet_mask: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """MIT-aligned divergence of face volume transports at cell centers.

    Parameters
    ----------
    u_volume_transport : Tensor [..., ny, nx+1]
        U-face volume transport, already weighted as
        ``u * drF * hFacW * dyG``.
    v_volume_transport : Tensor [..., ny+1, nx]
        V-face volume transport, already weighted as
        ``v * drF * hFacS * dxG``.
    rA : Tensor or None
        Pre-computed cell area.  When provided, overrides the value read from
        ``grid.metrics.cell_area_for``.  Supply a locally-sliced area when the
        transport tensors hold fewer faces than the global grid (the multi-rank
        path, where the transport has ``[n_local, ...]`` shape).
        Shape must broadcast against the transport batch dimensions.
    wet_mask : Tensor or None
        Pre-computed centre wet mask.  When provided, overrides the mask read
        from ``grid.masks.center_float``.  Same multi-rank motivation as ``rA``.

    Returns
    -------
    Tensor [..., ny, nx]
        Layer contribution to horizontal transport divergence in ``m s-1``.
        The caller-provided transports already include ``drF``; dividing
        ``Δtransport`` by MITgcm ``rA`` gives the layer-integrated vertical
        velocity contribution used by ``INTEGRATE_FOR_W``/``INTEGR_CONTINUITY``.
    """
    if rA is None:
        rA = grid.metrics.cell_area_for(u_volume_transport.dtype, u_volume_transport.device)
    is_cube = getattr(grid, "face_count", 1) == 6
    if is_cube:
        # Cube: rA is the per-face 2-D area [face, ny, nx]; keep the face dim
        # and view as [face, 1, ny, nx] so it broadcasts over nz against the
        # face-first transports [face, nz, ny, nx±1].
        rA_bcast = rA.unsqueeze(1)
    else:
        lead = max(u_volume_transport.ndim - 2, 0)
        rA_bcast = rA.view(*([1] * lead), grid.ny, grid.nx)

    result = horizontal_volume_divergence_from_transports(
        u_volume_transport,
        v_volume_transport,
    ) / rA_bcast

    if is_cube and grid.masks.has_3d_masks and u_volume_transport.ndim >= 4:
        # Cube center mask is full per-face 3-D [face, nz, ny, nx] — identical
        # shape to ``result``, so multiply directly (no batch unsqueeze, which
        # would inject a spurious axis).  (Regional [nz,ny,nx] keeps unsqueeze(0).)
        if wet_mask is None:
            wet_mask = grid.masks.center_float(
                dtype=u_volume_transport.dtype,
                device=u_volume_transport.device,
            )
        result = result * wet_mask
    elif grid.masks.has_3d_masks and u_volume_transport.ndim >= 4:
        if wet_mask is None:
            wet_mask = grid.masks.center_float(
                dtype=u_volume_transport.dtype,
                device=u_volume_transport.device,
            )
        result = result * wet_mask.unsqueeze(0)
    return result


def average_u_to_center(u: torch.Tensor) -> torch.Tensor:
    return 0.5 * (u[..., 1:] + u[..., :-1])


def average_v_to_center(v: torch.Tensor) -> torch.Tensor:
    return 0.5 * (v[..., 1:, :] + v[..., :-1, :])


def _curl_cube(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: "StructuredCGrid",
) -> torch.Tensor:
    """Cube-sphere relative vorticity at corners ``[6,(nz,)ny+1,nx+1]``.

    Uses pre-haloed (uc, vc) via ``exchange_cubed_sphere_vector`` and
    2-D per-face metrics from ``face_geometry`` for the MITgcm
    circulation/area form with distributed multiply (``dxC(j)*u(j)``
    not ``dxC*(u(j)-u(j-1))``).  Cube-corner 3-term override applied.
    """
    from defgcm.geometry.exchange import (
        HaloExtentV1,
        exchange_cubed_sphere_vector,
        exchange_cubed_sphere_vector_distributed,
    )

    dtype, device = u.dtype, u.device
    n = grid.nx
    ol = 1
    i0 = j0 = ol
    # Face-local routing: on a localized grid (ctx attached) u/v are rank-local
    # [n_local,...] and face_geometry is already scattered, so route the
    # cross-face vorticity halo to the distributed P2P backend.  ctx=None
    # (single-rank, and every current caller on a global grid) is byte-identical
    # to the original single-rank exchange.
    ctx = getattr(grid, "_cube_step_ctx", None)
    if ctx is not None:
        uc, vc = exchange_cubed_sphere_vector_distributed(
            u, v, grid=grid, extent=HaloExtentV1.uniform(ol), ctx=ctx, with_signs=True,
        )
    else:
        uc, vc = exchange_cubed_sphere_vector(
            u, v, grid=grid, extent=HaloExtentV1.uniform(ol), with_signs=True,
        )
    if device_resident_masks_enabled() and hasattr(grid, "face_geometry_for"):
        # Memoized device-resident metric copies (removes per-step H2D).
        record_device_resident_use("c_grid._curl_cube", "face_geometry")
        dxC_raw = grid.face_geometry_for("dxC", dtype, device)
        dyC_raw = grid.face_geometry_for("dyC", dtype, device)
        rAz_raw = grid.face_geometry_for("rAz", dtype, device)
    else:
        fg = grid.metrics.face_geometry
        dxC_raw = fg["dxC"].to(dtype=dtype, device=device)
        dyC_raw = fg["dyC"].to(dtype=dtype, device=device)
        rAz_raw = fg["rAz"].to(dtype=dtype, device=device)
    if ctx is not None:
        dxCh, dyCh = exchange_cubed_sphere_vector_distributed(
            dxC_raw, dyC_raw, grid=grid,
            extent=HaloExtentV1.uniform(ol), ctx=ctx, with_signs=False,
        )
    else:
        dxCh, dyCh = exchange_cubed_sphere_vector(
            dxC_raw, dyC_raw, grid=grid,
            extent=HaloExtentV1.uniform(ol), with_signs=False,
        )
    has_nz = u.ndim == 4
    if has_nz:
        dxCh = dxCh.unsqueeze(1)
        dyCh = dyCh.unsqueeze(1)
    cj = slice(j0, j0 + n + 1)
    ci = slice(i0, i0 + n + 1)
    vdyc = vc * dyCh
    udxc = uc * dxCh
    v_here = vdyc[..., cj, ci]
    v_left = vdyc[..., cj, i0 - 1:i0 + n]
    u_here = udxc[..., cj, ci]
    u_below = udxc[..., j0 - 1:j0 + n, ci]
    # rAz at the [ny+1, nx+1] vorticity points: use the genuine per-face
    # corner-point area array DIRECTLY, NOT a halo-exchanged copy.  The scalar
    # exchange of rAz into the halo is misaligned by one cell along the j=ny /
    # i=nx edges (verified vs MIT cs32 dump: 32/33 north-edge points wrong,
    # corner 2.08x too large); rAz_raw matches MITgcm rAz/recip_rAz bitwise at
    # all 24 cube corners + edges.  Interior (j,i < ny,nx) is bitwise-unchanged
    # because the haloed interior equalled rAz_raw there anyway.
    rAz_corner = rAz_raw.unsqueeze(1) if has_nz else rAz_raw
    recip_rAz = torch.where(
        rAz_corner != 0, 1.0 / rAz_corner, torch.zeros_like(rAz_corner),
    )
    zeta = recip_rAz * ((v_here - v_left) - (u_here - u_below))

    # Valence-3 cube-vertex override (MITgcm mom_calc_relvort3.F SW/SE/NW/NE
    # corner blocks): at a cube corner only 3 faces meet, so the 4-term
    # circulation picks up a geometrically-invalid v-ghost which MITgcm drops.
    # WEST corners (i=0) drop v_left; EAST corners (i=nx) drop v_here.  The
    # dropped term and genuine rAz_raw together reproduce MIT vort3 to FP-floor
    # (abs <= 3.3e-24) at all 24 corners across odd & even faces.  (MIT's
    # per-face summation order is not replicated: the 3-term math is identical,
    # only the last FP bit differs — already below the FP-floor bar.)
    zeta[..., 0, 0] = recip_rAz[..., 0, 0] * (
        v_here[..., 0, 0] - u_here[..., 0, 0] + u_below[..., 0, 0]
    )
    zeta[..., n, 0] = recip_rAz[..., n, 0] * (
        v_here[..., n, 0] - u_here[..., n, 0] + u_below[..., n, 0]
    )
    zeta[..., 0, n] = recip_rAz[..., 0, n] * (
        -v_left[..., 0, n] - u_here[..., 0, n] + u_below[..., 0, n]
    )
    zeta[..., n, n] = recip_rAz[..., n, n] * (
        -v_left[..., n, n] - u_here[..., n, n] + u_below[..., n, n]
    )
    return zeta


def curl_uv_to_corner(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: StructuredCGrid,
) -> torch.Tensor:
    """Relative vorticity ζ at corner points (MITgcm mom_calc_relvort3.F).

    Computes the MITgcm-native **circulation / area** form:

        ζ(i, j) = recip_rAz(i, j) · (
                    (vFld(i, j)·dyC(i, j)  -  vFld(i-1, j)·dyC(i-1, j))
                  - (uFld(i, j)·dxC(i, j)  -  uFld(i, j-1)·dxC(i, j-1))
                  )

    matching ``pkg/mom_common/mom_calc_relvort3.F:50-60`` (non-cubed-sphere
    main branch).  For a non-rotated uniform spherical polar grid the per-row
    factors reduce to:

        * ``dyC`` is a constant ``r·dlat`` → the δ_i(dyC·v) term factors as
          ``dy_const · (v[j, i] - v[j, i-1])``.
        * ``dxC`` varies along j → δ_j(dxC·u) requires the per-row values
          at cell-row ``j`` AND ``j-1`` (half-cell north and south of the
          corner respectively).
        * ``rAz`` is the corner cell area, also per-row — stored as
          :attr:`MetricSet.rAz_row` of length ``ny+1``.

    On a spherical polar grid the per-row metrics must remain distinct.
    With constant metrics and rAz = dx*dy, this reduces algebraically to
    the Cartesian curl.

    Parameters
    ----------
    u : Tensor [*, ny, nx+1]
        Velocity on u-faces.
    v : Tensor [*, ny+1, nx]
        Velocity on v-faces.
    grid : StructuredCGrid

    Returns
    -------
    vorticity : Tensor [*, ny+1, nx+1]
        Relative vorticity at corner points.  Boundary corners (j=0, j=ny,
        i=0, i=nx) stay at zero for non-periodic boundaries (matching
        MITgcm's zero-initialised halo); periodic axes are filled by the
        mask-selected wrapping candidates below.
    """
    if getattr(grid, "face_count", 1) == 6:
        return _curl_cube(u, v, grid)

    ny = grid.ny
    nx = grid.nx

    dxC = grid.metrics.dxC_for(u.dtype, u.device)                  # [ny,nx+1]
    rAz = grid.metrics.rAz_for(u.dtype, u.device)                  # [ny+1,nx+1]
    dyC = grid.metrics.dyC_for(v.dtype, v.device)                  # [ny+1,nx]

    result = torch.zeros(
        (*u.shape[:-2], ny + 1, nx + 1),
        dtype=u.dtype,
        device=u.device,
    )

    # --- Interior corners: j ∈ [1, ny-1], i ∈ [1, nx-1] ---
    # At corner (j, i):
    #   Δ_v  = dyC · (v[j, i] - v[j, i-1])
    #   Δ_u  = u[j, i]·dxC[j] - u[j-1, i]·dxC[j-1]
    #   ζ[j, i] = (Δ_v - Δ_u) / rAz[j]
    #
    # ``u[j, i]`` is the u-face NORTH of corner (j, i) at latitude yC[j],
    # so its metric is ``dxC_row[j]``.  ``u[j-1, i]`` is the u-face SOUTH
    # of the corner at yC[j-1] with metric ``dxC_row[j-1]``.
    dxC_north = dxC[1:ny, 1:nx]
    dxC_south = dxC[: ny - 1, 1:nx]
    recip_rAz_interior = 1.0 / rAz[1:ny, 1:nx]

    v_right = v[..., 1:ny, 1:]                                     # v[j, i]   for j∈[1,ny-1], i∈[1,nx-1]
    v_left = v[..., 1:ny, :-1]                                     # v[j, i-1]
    delta_v = dyC[1:ny, 1:nx] * (v_right - v_left)

    u_north = u[..., 1:, 1:nx]                                     # u[j,   i] for j∈[1,ny-1], i∈[1,nx-1]
    u_south = u[..., :-1, 1:nx]                                    # u[j-1, i]
    delta_u = u_north * dxC_north - u_south * dxC_south

    result[..., 1:ny, 1:nx] = recip_rAz_interior * (delta_v - delta_u)

    _cgrid_trace("curl u_corner exchange begin")
    u_corner = exchange(
        u,
        grid=grid,
        location="F",
        extent=_HALO_XY,
        fill=FillSpec.ZERO,
    )
    _cgrid_trace("curl u_corner exchange done")
    _cgrid_trace("curl v_corner exchange begin")
    v_corner = exchange(
        v,
        grid=grid,
        location="F",
        extent=_HALO_XY,
        fill=FillSpec.ZERO,
    )
    _cgrid_trace("curl v_corner exchange done")

    x_core = torch.ones(nx, dtype=torch.bool, device=u.device)
    _cgrid_trace("curl x_core exchange begin")
    x_core_ext = exchange(
        x_core,
        grid=grid,
        location="F",
        extent=_HALO_X,
        fill=FillSpec.ZERO,
        constant=True,
    )
    _cgrid_trace("curl x_core exchange done")
    x_corner_mask = x_core_ext[:-1] & x_core_ext[1:]
    y_core = torch.ones(ny, dtype=torch.bool, device=u.device)
    _cgrid_trace("curl y_core exchange begin")
    y_core_ext = exchange(
        y_core,
        grid=grid,
        location="F",
        extent=_HALO_Y,
        fill=FillSpec.ZERO,
        constant=True,
    )
    _cgrid_trace("curl y_core exchange done")
    y_corner_mask = y_core_ext[:-1] & y_core_ext[1:]
    _cgrid_trace("curl dxC exchange begin")
    dxC_ext = exchange(
        dxC,
        grid=grid,
        location="F",
        extent=_HALO_Y,
        fill=FillSpec.ZERO,
        constant=True,
    )
    _cgrid_trace("curl dxC exchange done")

    # --- Wrapped candidates selected by topology support masks ---
    # For a rectangular grid the per-row metrics are constant, so these reduce
    # to the same scalar-dx/dy expression the old code used.

    # West/east edge corners.  Periodic single-process grids select identical
    # wrap candidates on both sides; distributed tiles can have one physical
    # edge and one internal seam, so compute each side separately.
    dxC_j_west = dxC[1:ny, 0]
    dxC_jm1_west = dxC[: ny - 1, 0]
    recip_rAz_west = 1.0 / rAz[1:ny, 0]
    delta_v_west = dyC[1:ny, 0] * (
        v_corner[..., 2 : ny + 1, 1] - v_corner[..., 2 : ny + 1, 0]
    )
    delta_u_west = (
        u_corner[..., 2 : ny + 1, 1] * dxC_j_west
        - u_corner[..., 1:ny, 1] * dxC_jm1_west
    )
    x_west_candidate = recip_rAz_west * (delta_v_west - delta_u_west)
    x_edge_leading = (1,) * (x_west_candidate.ndim - 1)
    x_west_support = (
        x_corner_mask[0]
        & y_corner_mask[1:ny].view(*x_edge_leading, ny - 1)
    )
    x_west = torch.where(
        x_west_support,
        x_west_candidate,
        torch.zeros_like(x_west_candidate),
    )
    result[..., 1:ny, 0] = x_west

    dxC_j_east = dxC[1:ny, nx]
    dxC_jm1_east = dxC[: ny - 1, nx]
    recip_rAz_east = 1.0 / rAz[1:ny, nx]
    delta_v_east = dyC[1:ny, nx - 1] * (
        v_corner[..., 2 : ny + 1, nx + 1] - v_corner[..., 2 : ny + 1, nx]
    )
    delta_u_east = (
        u_corner[..., 2 : ny + 1, nx + 1] * dxC_j_east
        - u_corner[..., 1:ny, nx + 1] * dxC_jm1_east
    )
    x_east_candidate = recip_rAz_east * (delta_v_east - delta_u_east)
    x_east_support = (
        x_corner_mask[-1]
        & y_corner_mask[1:ny].view(*x_edge_leading, ny - 1)
    )
    x_east = torch.where(
        x_east_support,
        x_east_candidate,
        torch.zeros_like(x_east_candidate),
    )
    result[..., 1:ny, nx] = x_east

    # South/north edge corners.  The haloed metric row supplies the neighbour
    # row metric on periodic/distributed y edges; physical edges are masked off.
    # Include true corners in these rows: separate scalar corner loads can
    # produce invalid scalar-pointer/block-mask loads in Inductor/Triton.
    dyC_south = torch.cat((dyC[0], dyC[0, -1:]))
    dyC_north = torch.cat((dyC[ny], dyC[ny, -1:]))
    dxC_south_ghost = dxC_ext[0, : nx + 1]
    dxC_south_inner = dxC_ext[1, : nx + 1]
    dxC_north_inner = dxC_ext[ny, : nx + 1]
    dxC_north_ghost = dxC_ext[ny + 1, : nx + 1]
    recip_rAz_south = 1.0 / rAz[0, : nx + 1]
    recip_rAz_north = 1.0 / rAz[ny, : nx + 1]
    delta_v_south = dyC_south * (
        v_corner[..., 1, 1 : nx + 2] - v_corner[..., 1, : nx + 1]
    )
    delta_u_south = (
        u_corner[..., 1, 1 : nx + 2] * dxC_south_inner
        - u_corner[..., 0, 1 : nx + 2] * dxC_south_ghost
    )
    y_south_candidate = recip_rAz_south * (delta_v_south - delta_u_south)
    y_edge_leading = (1,) * (y_south_candidate.ndim - 1)
    y_south_support = (
        y_corner_mask[0]
        & x_corner_mask.view(*y_edge_leading, nx + 1)
    )
    y_south = torch.where(
        y_south_support,
        y_south_candidate,
        torch.zeros_like(y_south_candidate),
    )
    result[..., 0, :] = y_south

    delta_v_north = dyC_north * (
        v_corner[..., ny + 1, 1 : nx + 2] - v_corner[..., ny + 1, : nx + 1]
    )
    delta_u_north = (
        u_corner[..., ny + 1, 1 : nx + 2] * dxC_north_ghost
        - u_corner[..., ny, 1 : nx + 2] * dxC_north_inner
    )
    y_north_candidate = recip_rAz_north * (delta_v_north - delta_u_north)
    y_north_support = (
        y_corner_mask[-1]
        & x_corner_mask.view(*y_edge_leading, nx + 1)
    )
    y_north = torch.where(
        y_north_support,
        y_north_candidate,
        torch.zeros_like(y_north_candidate),
    )
    result[..., ny, :] = y_north

    # Periodic-seam consistency (non-decomposed axes only).  On a single
    # process the far edge ``i=nx`` / ``j=ny`` is the periodic IMAGE of the
    # near edge ``i=0`` / ``j=0`` (the same physical seam), so it must equal it.
    # The per-side candidates above compute the far edge from the redundant
    # duplicate face column/row (``u[..., nx]`` / ``v[..., ny, :]``); that
    # equals the near edge only when the duplicate faces are synced — always
    # true in production (the runner syncs duplicated C-grid faces at init) but
    # not for arbitrary input.  Copying near->far restores the invariant
    # and is byte-identical for synced input.  A
    # *decomposed* axis keeps the per-rank support-masked compute: the global
    # periodic seam spans two ranks (consistency via synced faces), while a
    # rank's local ``i=0`` / ``i=nx`` are distinct interior corners.
    if grid.boundary_topology.periodic_x and not _axis_is_decomposed(grid, "x"):
        result[..., :, nx] = result[..., :, 0]
    if grid.boundary_topology.periodic_y and not _axis_is_decomposed(grid, "y"):
        result[..., ny, :] = result[..., 0, :]

    return result


def laplacian_center(
    field: torch.Tensor,
    grid: StructuredCGrid,
) -> torch.Tensor:
    """Compute horizontal Laplacian ∇²φ at cell centers using a 5-point stencil.

    Parameters
    ----------
    field : Tensor [*, ny, nx]
        Scalar field at cell centers.
    grid : StructuredCGrid

    Returns
    -------
    lap : Tensor [*, ny, nx]
        Laplacian at cell centers.
    """
    if not grid.metrics.is_uniform_x or not grid.metrics.is_uniform_y:
        raise NotImplementedError(
            "laplacian_center supports only uniform horizontal spacing"
        )
    dx = field.new_tensor(grid.dx)
    dy = field.new_tensor(grid.dy)
    dx2 = dx * dx
    dy2 = dy * dy

    field_x = exchange(
        field,
        grid=grid,
        location="T",
        extent=_HALO_X,
        fill=FillSpec.REPLICATE,
    )
    field_xy = exchange(
        field_x,
        grid=grid,
        location="T",
        extent=_HALO_Y,
        fill=FillSpec.REPLICATE,
    )

    # 5-point stencil on the padded field
    # Center is at [1:-1, 1:-1]
    d2fdx2 = (field_xy[..., 1:-1, 2:] + field_xy[..., 1:-1, :-2] - 2.0 * field_xy[..., 1:-1, 1:-1]) / dx2
    d2fdy2 = (field_xy[..., 2:, 1:-1] + field_xy[..., :-2, 1:-1] - 2.0 * field_xy[..., 1:-1, 1:-1]) / dy2

    return d2fdx2 + d2fdy2
