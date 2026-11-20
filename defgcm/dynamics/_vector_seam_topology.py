"""Topology helpers for C-grid vector seams."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from defgcm.geometry.exchange import HaloExtentV1, exchange_cubed_sphere_vector

if TYPE_CHECKING:
    from defgcm.geometry.cube_decomposition import CubeRankContext


def exchange_vector_velocities_for_grid(
    u: torch.Tensor,
    v: torch.Tensor,
    grid,
    ol: int,
    *,
    with_signs: bool = True,
    ctx: "CubeRankContext | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same-size haloed frame ``(uc, vc)`` for cube ``(u, v)`` vectors.

    When *ctx* is provided, routes to the distributed exchange so each rank
    exchanges only its local faces.  ctx=None (default) retains the single-rank
    ``exchange_cubed_sphere_vector`` path (byte-identical to the original).
    """
    if ctx is not None:
        from defgcm.geometry.exchange import exchange_cubed_sphere_vector_distributed
        return exchange_cubed_sphere_vector_distributed(
            u,
            v,
            grid=grid,
            extent=HaloExtentV1.uniform(ol),
            ctx=ctx,
            with_signs=with_signs,
        )
    return exchange_cubed_sphere_vector(
        u,
        v,
        grid=grid,
        extent=HaloExtentV1.uniform(ol),
        with_signs=with_signs,
    )


def reconcile_vector_redundant_edges_for_grid(
    u: torch.Tensor,
    v: torch.Tensor,
    grid,
    *,
    u_mask: torch.Tensor | None = None,
    v_mask: torch.Tensor | None = None,
    with_signs: bool = True,
    ctx: "CubeRankContext | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overwrite redundant east/north C-grid vector slots from seam exchange.

    When *ctx* is provided the exchange is routed to the distributed path so
    cross-rank seam strips are fetched via point-to-point communication.
    ctx=None (default) retains the single-rank path (byte-identical).
    """
    nx, ny = grid.nx, grid.ny
    if tuple(u.shape[-2:]) != (ny, nx + 1):
        raise ValueError(f"u must be [...,ny,nx+1]; got {tuple(u.shape)}")
    if tuple(v.shape[-2:]) != (ny + 1, nx):
        raise ValueError(f"v must be [...,ny+1,nx]; got {tuple(v.shape)}")
    if u_mask is not None and u_mask.shape != u.shape:
        raise ValueError("u_mask must match u shape")
    if v_mask is not None and v_mask.shape != v.shape:
        raise ValueError("v_mask must match v shape")

    uc, vc = exchange_vector_velocities_for_grid(
        u, v, grid, 1, with_signs=with_signs, ctx=ctx,
    )
    u_edge = uc[..., 1 : ny + 1, 1 + nx]
    v_edge = vc[..., 1 + ny, 1 : nx + 1]
    if u_mask is not None:
        u_edge = u_edge * u_mask[..., :, nx].to(dtype=u_edge.dtype, device=u_edge.device)
    if v_mask is not None:
        v_edge = v_edge * v_mask[..., ny, :].to(dtype=v_edge.dtype, device=v_edge.device)

    u_out = u.clone()
    v_out = v.clone()
    u_out[..., :, nx] = u_edge
    v_out[..., ny, :] = v_edge
    return u_out, v_out
