"""Nonhydrostatic Coriolis term of the U momentum equation, MITgcm scheme 1.

MITgcm ``mom_u_coriolis_nh.F`` (select3dCoriScheme=1): the U-face tendency
``gravitySign * fCoriCos * wMean`` with ``gravitySign = -1``, where ``wMean``
is the four-point mean of the interface vertical velocity around the U face.
``w_ext`` is MIT upward-positive ``w`` at the ``nz+1`` interfaces with one
x-halo column on each side, ``[batch, nz+1, ny, nx+2]``; ``f_cori_cos``
broadcasts to ``[batch, ny, nx]``.  The result has the shape and dtype of
``u_ref`` ``[batch, nz, ny, nx+1]``, in m s^-2.  The edge faces ``i=0`` and
``i=nx`` are filled on a periodic x axis and on a side with a distributed
neighbour, and stay zero otherwise.

``w_coriolis_scheme1`` is the W-equation term of the same scheme
(``mom_w_coriolis_nh.F``), consumed by ``vertical_momentum.compute_gw``.
"""

from __future__ import annotations

import torch


def u_coriolis_nh_from_haloed_w(
    w_ext: torch.Tensor,
    u_ref: torch.Tensor,
    *,
    f_cori_cos: torch.Tensor,
    periodic_x: bool,
    west_distributed: bool = False,
    east_distributed: bool = False,
) -> torch.Tensor:
    """MOM_U_CORIOLIS_NH scheme-1 U-face tensor kernel.

    ``w_ext`` is MIT upward-positive ``w`` with one x-halo on both sides,
    shaped ``[batch,nz+1,ny,nx+2]``.  The caller owns geometry-specific halo
    exchange and supplies ``f_cori_cos`` broadcastable to ``[batch,ny,nx]``.
    """
    batch, nz, ny, nx_plus_1 = u_ref.shape
    nx = nx_plus_1 - 1
    expected_shape = (batch, nz + 1, ny, nx + 2)
    if w_ext.shape != expected_shape:
        raise ValueError(
            "w_ext must be MIT upward-positive [batch,nz+1,ny,nx+2], "
            f"got {tuple(w_ext.shape)} for u_ref {tuple(u_ref.shape)}"
        )
    term = torch.zeros_like(u_ref)
    for k in range(nz):
        if nx > 1:
            w_mean = _u_coriolis_nh_interior_w_mean(w_ext, k=k, nx=nx)
            term[:, k, :, 1:nx] = _u_coriolis_nh_interior_tendency(
                f_cori_cos,
                w_mean,
            )
        if periodic_x and nx > 0:
            w_mean_edge = _u_coriolis_nh_periodic_edge_w_mean(w_ext, k=k)
            edge_term = _u_coriolis_nh_periodic_edge_tendency(
                f_cori_cos,
                w_mean_edge,
            )
            term[:, k, :, 0] = edge_term
            term[:, k, :, nx] = edge_term
        elif nx > 0:
            if west_distributed:
                w_mean_west = _u_coriolis_nh_periodic_edge_w_mean(w_ext, k=k)
                term[:, k, :, 0] = _u_coriolis_nh_periodic_edge_tendency(
                    f_cori_cos,
                    w_mean_west,
                )
            if east_distributed:
                w_mean_east = _u_coriolis_nh_east_seam_w_mean(w_ext, k=k, nx=nx)
                term[:, k, :, nx] = _u_coriolis_nh_periodic_edge_tendency(
                    f_cori_cos,
                    w_mean_east,
                )
    return term


def _u_coriolis_nh_interior_tendency(
    f_cori_cos: torch.Tensor,
    w_mean: torch.Tensor,
) -> torch.Tensor:
    """Interior U-face nonhydrostatic Coriolis term with MIT gravitySign=-1."""
    return -f_cori_cos * w_mean


def _u_coriolis_nh_periodic_edge_tendency(
    f_cori_cos: torch.Tensor,
    w_mean_edge: torch.Tensor,
) -> torch.Tensor:
    """Periodic-edge U-face nonhydrostatic Coriolis term."""
    return -f_cori_cos.squeeze(-1) * w_mean_edge


def _u_coriolis_nh_interior_w_mean(
    w_ext: torch.Tensor,
    *,
    k: int,
    nx: int,
) -> torch.Tensor:
    """Interior U-face mean of MIT upward-positive haloed ``w``."""
    kp1 = k + 1
    return 0.25 * (
        w_ext[:, k, :, 2 : nx + 1]
        + w_ext[:, k, :, 1:nx]
        + w_ext[:, kp1, :, 2 : nx + 1]
        + w_ext[:, kp1, :, 1:nx]
    )


def _u_coriolis_nh_periodic_edge_w_mean(
    w_ext: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """West-edge U-face mean of MIT upward-positive haloed ``w`` (periodic edge or distributed west seam)."""
    kp1 = k + 1
    return 0.25 * (
        w_ext[:, k, :, 1]
        + w_ext[:, k, :, 0]
        + w_ext[:, kp1, :, 1]
        + w_ext[:, kp1, :, 0]
    )


def _u_coriolis_nh_east_seam_w_mean(
    w_ext: torch.Tensor,
    *,
    k: int,
    nx: int,
) -> torch.Tensor:
    """Distributed east-seam U-face mean of MIT upward-positive haloed ``w``."""
    kp1 = k + 1
    return 0.25 * (
        w_ext[:, k, :, nx + 1]
        + w_ext[:, k, :, nx]
        + w_ext[:, kp1, :, nx + 1]
        + w_ext[:, kp1, :, nx]
    )


def w_coriolis_scheme1(
    u: torch.Tensor, w: torch.Tensor, *, f_cori_cos: torch.Tensor
) -> torch.Tensor:
    """MOM_W_CORIOLIS_NH scheme 1 on a non-rotated grid: ``fCoriCos * uMean`` below the surface.

    ``u`` is on U faces ``[B, nz, ny, nx+1]``, ``w`` gives the W-point shape
    ``[B, nz, ny, nx]`` and ``f_cori_cos`` is ``fCoriCos`` per row
    ``[1, 1, ny, 1]`` in s-1.  The term is in MIT's upward-positive ``gW``
    convention (m s-2) and is zero at level 0.
    """
    term = torch.zeros_like(w)
    u_mean = 0.25 * (
        u[:, :-1, :, :-1]
        + u[:, :-1, :, 1:]
        + u[:, 1:, :, :-1]
        + u[:, 1:, :, 1:]
    )
    # Non-rotated SCS has angleSinC=0, so vMean is intentionally unused.
    term[:, 1:] = f_cori_cos * u_mean
    return term
