"""MITgcm-style CG3D operator assembly.

The coefficients are scalar geometry/time-step terms.  This module does not
consume ``w``/``wVel``; the LOCOMgcm-downward to MIT-upward sign conversion belongs
at the stepper boundary before PRE_CG3D.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import torch
from locomgcm.distributed.comm_adjoint import active_data_group
import torch.distributed as dist

from locomgcm.dycore_parameters import NonhydrostaticParams
from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange


class CG3DOperator(NamedTuple):
    """Normalized 7-diagonal CG3D operator coefficients."""

    aW3d: torch.Tensor
    aS3d: torch.Tensor
    aV3d: torch.Tensor
    aC3d: torch.Tensor
    cg3dNorm: torch.Tensor
    cg3dTolerance_sq: torch.Tensor
    cg3d_normalise_rhs: bool


def _global_sum_scalar(value: torch.Tensor, *, group=None) -> torch.Tensor:
    result = value.clone()
    if dist.is_available() and dist.is_initialized():
        from locomgcm.distributed.transport_failure import data_all_reduce
        data_all_reduce(result, op=dist.ReduceOp.SUM, group=group)
    return result


def _global_max_scalar(value: torch.Tensor, *, group=None) -> torch.Tensor:
    result = value.clone()
    if dist.is_available() and dist.is_initialized():
        from locomgcm.distributed.transport_failure import data_all_reduce
        data_all_reduce(result, op=dist.ReduceOp.MAX, group=group)
    return result


def _time_attr(time_scales: Any, *names: str, default: float = 1.0) -> float:
    for name in names:
        value = getattr(time_scales, name, None)
        if value is not None:
            return float(value)
    return float(default)


def _center_mask_3d(
    grid: StructuredCGrid,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    mask = grid.masks.center_float(dtype=dtype, device=device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(grid.nz, -1, -1)
    return mask


def _mask_in_c_2d(
    grid: StructuredCGrid,
    mask_in_c: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    if mask_in_c is not None:
        return mask_in_c.to(dtype=dtype, device=device)
    if grid.masks.maskInC is not None:
        return grid.masks.maskInC.to(dtype=dtype, device=device)
    return torch.ones((grid.ny, grid.nx), dtype=dtype, device=device)


def _first_wet_index(mask_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return wet-column flag and 0-indexed kSurfC equivalent."""
    has_wet = mask_c > 0
    wet_col = has_wet.any(dim=0)
    ks = has_wet.to(torch.int64).argmax(dim=0)
    return wet_col, ks


def _has_distributed_neighbor(grid: StructuredCGrid, side: str) -> bool:
    neighbours = getattr(grid, "distributed_neighbors", None) or {}
    entry = neighbours.get(side) if hasattr(neighbours, "get") else None
    return entry is not None


def build_cg3d_operator(
    grid: StructuredCGrid,
    params: NonhydrostaticParams,
    time_scales: Any,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    recip_bo: torch.Tensor | float = 1.0,
    mask_in_c: torch.Tensor | None = None,
) -> CG3DOperator:
    """Build MITgcm ``INI_CG3D``-style normalized CG3D coefficients.

    Off-diagonal coefficients are non-negative; ``aC3d`` is non-positive.
    ``cg3d_target_res_w_unit <= 0`` selects MIT's RHS-normalized residual
    branch.  Direct callers must pass a positive ``nh_am2``.
    """
    if params.nh_am2 <= 0:
        raise ValueError(f"nh_am2 must be positive, got {params.nh_am2}")

    device = torch.device(device)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    mask_c = _center_mask_3d(grid, dtype=dtype, device=device)
    mask_in_2d = _mask_in_c_2d(grid, mask_in_c, dtype=dtype, device=device)
    mask_in_3d = mask_in_2d.unsqueeze(0).expand(nz, -1, -1)
    wet3d = mask_c * mask_in_3d

    dr_f = grid.layer_thickness_for(dtype, device)
    hfac_w = grid.hfac_w_for(dtype, device)
    hfac_s = grid.hfac_s_for(dtype, device)
    r_a = grid.metrics.cell_area_for(dtype, device)
    dx_c = grid.metrics.dxC_for(dtype, device)
    dy_g = grid.metrics.dyG_for(dtype, device)
    dx_g_ext = grid.metrics.dxG_for(dtype, device)
    dy_c = grid.metrics.dyC_for(dtype, device)

    aW = torch.zeros((nz, ny, nx + 1), dtype=dtype, device=device)
    aS = torch.zeros((nz, ny + 1, nx), dtype=dtype, device=device)
    aV = torch.zeros((nz, ny, nx), dtype=dtype, device=device)

    if nx > 1:
        face_mask_w = mask_in_2d[:, 1:nx] * mask_in_2d[:, : nx - 1]
        face_area_w = dr_f.view(nz, 1, 1) * dy_g[:, 1:nx] * hfac_w[:, :, 1:nx]
        aW[:, :, 1:nx] = (
            face_area_w
            * face_mask_w.unsqueeze(0)
            / dx_c[:, 1:nx].unsqueeze(0)
        )
    x_edge_couples = bool(grid.edge_face_couples("x"))
    west_couples = x_edge_couples or _has_distributed_neighbor(grid, "west")
    east_couples = x_edge_couples or _has_distributed_neighbor(grid, "east")
    if west_couples or east_couples:
        # The interior mask is a constant of the run on every rank.  Active
        # geometry rebuilds this operator inside the step, where the exchange
        # goes through a bound differentiation session; undeclared, the mask
        # would come back requiring grad.
        mask_halo_x = exchange(
            mask_in_2d,
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
    else:
        mask_halo_x = None
    if mask_halo_x is not None and west_couples:
        face_mask_w = mask_halo_x[:, 0] * mask_halo_x[:, 1]
        face_area_w = dr_f.view(nz, 1) * dy_g[:, 0] * hfac_w[:, :, 0]
        aW[:, :, 0] = face_area_w * face_mask_w.unsqueeze(0) / dx_c[:, 0].unsqueeze(0)
    if mask_halo_x is not None and east_couples:
        face_mask_w = mask_halo_x[:, nx] * mask_halo_x[:, nx + 1]
        face_area_w = dr_f.view(nz, 1) * dy_g[:, nx] * hfac_w[:, :, nx]
        aW[:, :, nx] = face_area_w * face_mask_w.unsqueeze(0) / dx_c[:, nx].unsqueeze(0)
    if ny > 1:
        face_mask_s = mask_in_2d[1:ny, :] * mask_in_2d[: ny - 1, :]
        face_area_s = (
            dr_f.view(nz, 1, 1)
            * dx_g_ext[1:ny, :].unsqueeze(0)
            * hfac_s[:, 1:ny, :]
        )
        aS[:, 1:ny, :] = face_area_s * face_mask_s.unsqueeze(0) / dy_c[1:ny, :].unsqueeze(0)
    y_edge_couples = bool(grid.edge_face_couples("y"))
    south_couples = y_edge_couples or _has_distributed_neighbor(grid, "south")
    north_couples = y_edge_couples or _has_distributed_neighbor(grid, "north")
    if south_couples or north_couples:
        mask_halo_y = exchange(
            mask_in_2d,
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
    else:
        mask_halo_y = None
    if mask_halo_y is not None and south_couples:
        face_mask_s = mask_halo_y[0, :] * mask_halo_y[1, :]
        face_area_s = (
            dr_f.view(nz, 1)
            * dx_g_ext[0, :]
            * hfac_s[:, 0, :]
        )
        aS[:, 0, :] = face_area_s * face_mask_s.unsqueeze(0) / dy_c[0, :]
    if mask_halo_y is not None and north_couples:
        face_mask_s = mask_halo_y[ny, :] * mask_halo_y[ny + 1, :]
        face_area_s = (
            dr_f.view(nz, 1)
            * dx_g_ext[ny, :]
            * hfac_s[:, ny, :]
        )
        aS[:, ny, :] = face_area_s * face_mask_s.unsqueeze(0) / dy_c[ny, :]

    if nz > 1:
        dr_c = grid.vertical_coordinate.interface_spacing.to(dtype=dtype, device=device)
        vertical_mask = wet3d[1:] * wet3d[:-1]
        aV[1:] = (
            r_a.view(1, ny, nx)
            * vertical_mask
            * dr_c.reciprocal().view(nz - 1, 1, 1)
            * torch.as_tensor(params.nh_am2, dtype=dtype, device=device)
        )

    max_offdiag = torch.stack((aW.abs().max(), aS.abs().max(), aV.abs().max())).max()
    max_offdiag = _global_max_scalar(max_offdiag, group=active_data_group())
    # Double-where so backward never differentiates 1/0 when max_offdiag==0.
    _mo_pos = max_offdiag > 0
    _mo_safe = torch.where(_mo_pos, max_offdiag, torch.ones((), dtype=dtype, device=device))
    cg3d_norm = torch.where(
        _mo_pos,
        _mo_safe.reciprocal(),
        torch.ones((), dtype=dtype, device=device),
    )

    aC = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    aC = aC - aW[:, :, :nx] - aW[:, :, 1:]
    aC = aC - aS[:, :ny, :] - aS[:, 1:, :]
    aC = aC - aV
    if nz > 1:
        aC[:-1] = aC[:-1] - aV[1:]

    dt_mom = _time_attr(time_scales, "dt_mom", "delta_t_mom", default=1.0)
    dt_fs = _time_attr(
        time_scales,
        "dt_freesurf",
        "delta_t_freesurf",
        default=dt_mom,
    )
    recip_bo_t = torch.as_tensor(recip_bo, dtype=dtype, device=device)
    if recip_bo_t.ndim == 0:
        recip_bo_t = recip_bo_t.expand(ny, nx)

    wet_col, ks = _first_wet_index(mask_c)
    # Unconditional masked update : ``torch.where`` leaves every
    # level with no surface cell unchanged, so dropping the ``at_surface.any()``
    # host branch (one D2H sync + one graph break per level) is bit-neutral.
    delta = recip_bo_t * r_a / (dt_mom * dt_fs)
    for k in range(nz):
        at_surface = wet_col & (ks == k)
        aC[k] = torch.where(at_surface, aC[k] - delta, aC[k])

    # MIT INI_CG3D (ini_cg3d.F:237-244) adds surface diagonal without maskInC, so wet
    # OBC-strip cells (maskInC=0 but maskC=1) keep -recip_Bo*rA/(dt_mom*dt_fs). Using
    # wet3d here would zero those rows and break MIT bitwise parity (CONV-cg3d-001).
    aC = torch.where(mask_c > 0, aC, torch.zeros_like(aC))
    aW = aW * cg3d_norm
    aS = aS * cg3d_norm
    aV = aV * cg3d_norm
    aC = aC * cg3d_norm

    normalise_rhs = params.cg3d_target_res_w_unit <= 0.0
    if normalise_rhs:
        tolerance = torch.as_tensor(
            params.cg3d_target_residual,
            dtype=dtype,
            device=device,
        )
    else:
        n3d_wet = _global_sum_scalar(wet3d.sum(), group=active_data_group())
        deep_fac_2c = torch.ones((nz,), dtype=dtype, device=device)
        loc3d_area = _global_sum_scalar(
            (deep_fac_2c.view(nz, 1, 1) * r_a.unsqueeze(0) * wet3d).sum()
        , group=active_data_group())
        # Double-where so backward never differentiates loc3d_area/0
        # when no wet cells (all-land subdomain edge case).
        _wet_pos = n3d_wet > 0
        _wet_safe = torch.where(_wet_pos, n3d_wet, torch.ones((), dtype=n3d_wet.dtype, device=device))
        r_ac_3d_mean = torch.where(
            _wet_pos,
            loc3d_area / _wet_safe,
            torch.zeros((), dtype=dtype, device=device),
        )
        tolerance = (
            cg3d_norm
            * torch.as_tensor(
                params.cg3d_target_res_w_unit,
                dtype=dtype,
                device=device,
            )
            / dt_mom
            * r_ac_3d_mean
            * torch.sqrt(n3d_wet)
        )

    return CG3DOperator(
        aW3d=aW,
        aS3d=aS,
        aV3d=aV,
        aC3d=aC,
        cg3dNorm=cg3d_norm,
        cg3dTolerance_sq=tolerance * tolerance,
        cg3d_normalise_rhs=normalise_rhs,
    )


def apply_cg3d_operator(
    phi: torch.Tensor,
    operator: CG3DOperator,
    *,
    grid: StructuredCGrid | None = None,
) -> torch.Tensor:
    """Apply the normalized 7-point CG3D stencil to ``phi``."""
    if phi.ndim != 4:
        raise ValueError(f"phi must be [batch,nz,ny,nx], got {tuple(phi.shape)}")
    _, nz, ny, nx = phi.shape
    if operator.aC3d.shape != (nz, ny, nx):
        raise ValueError(
            f"operator shape {tuple(operator.aC3d.shape)} does not match phi {tuple(phi.shape)}"
        )

    if grid is not None:
        if (grid.nz, grid.ny, grid.nx) != (nz, ny, nx):
            raise ValueError("grid shape must match phi")
        phi_halo = exchange(
            phi,
            grid=grid,
            location="T",
            extent=HaloExtentV1.uniform(1),
            fill=FillSpec.ZERO,
        )
        inner = phi_halo[:, :, 1:ny + 1, 1:nx + 1]
        west = phi_halo[:, :, 1:ny + 1, 0:nx]
        east = phi_halo[:, :, 1:ny + 1, 2:nx + 2]
        south = phi_halo[:, :, 0:ny, 1:nx + 1]
        north = phi_halo[:, :, 2:ny + 2, 1:nx + 1]
        out = operator.aC3d.unsqueeze(0) * inner
        out = out + operator.aW3d[:, :, :nx].unsqueeze(0) * west
        out = out + operator.aW3d[:, :, 1:nx + 1].unsqueeze(0) * east
        out = out + operator.aS3d[:, :ny, :].unsqueeze(0) * south
        out = out + operator.aS3d[:, 1:ny + 1, :].unsqueeze(0) * north
    else:
        out = operator.aC3d.unsqueeze(0) * phi
        if nx > 1:
            coeff = operator.aW3d[:, :, 1:nx].unsqueeze(0)
            out[:, :, :, 1:] += coeff * phi[:, :, :, :-1]
            out[:, :, :, :-1] += coeff * phi[:, :, :, 1:]
        if ny > 1:
            coeff = operator.aS3d[:, 1:ny, :].unsqueeze(0)
            out[:, :, 1:, :] += coeff * phi[:, :, :-1, :]
            out[:, :, :-1, :] += coeff * phi[:, :, 1:, :]
    if nz > 1:
        coeff = operator.aV3d[1:].unsqueeze(0)
        out[:, 1:] += coeff * phi[:, :-1]
        out[:, :-1] += coeff * phi[:, 1:]
    return out
