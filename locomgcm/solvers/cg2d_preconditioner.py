from __future__ import annotations

import torch
import torch.nn.functional as F


def build_cg2d_jacobi_inverse(
    aC: torch.Tensor,
    *,
    threshold: float = 1e-30,
) -> torch.Tensor:
    """Build the double-where-safe Jacobi inverse for CG2D."""
    mask = aC.abs() > threshold
    safe = torch.where(mask, aC, torch.ones_like(aC))
    return torch.where(mask, 1.0 / safe, torch.zeros_like(aC))


def build_cg2d_ssor_preconditioner(
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
    *,
    offdiag_factor: float = 0.51,
    threshold: float = 1e-30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tile-local MIT CG2D SSOR coefficients: no neighbour across any tile edge.

    :func:`build_cg2d_ssor_preconditioner_from_halo` with a zero halo, which is
    what the former body computed with its zero-filled shifts.  Only a tile whose
    four edges couple nothing may use it (numerics rule 11); the regional caller
    decides that from the grid topology.
    """
    return build_cg2d_ssor_preconditioner_from_halo(
        aW_edge,
        aS_edge,
        F.pad(aC, (1, 1, 1, 1)),
        offdiag_factor=offdiag_factor,
        threshold=threshold,
    )


def build_cg2d_ssor_preconditioner_from_halo(
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC_halo: torch.Tensor,
    *,
    offdiag_factor: float = 0.51,
    threshold: float = 1e-30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MIT CG2D SSOR coefficients with the neighbour diagonals read from a halo.

    MITgcm ``ini_cg2d.F:210-236`` builds, for every cell ``i = 1..sNx`` of a
    tile, ``pW(i) = -aW2d(i) / (cg2dpcOffDFac*(aC2d(i-1) + aC2d(i)))**2`` (and
    ``pS`` likewise), where ``aC2d(0)`` is the neighbour tile's diagonal in the
    halo (``ini_cg2d.F:199-209`` evaluates it from the exchanged ``aW2d/aS2d``).
    ``aC_halo`` is ``aC`` padded by one cell on every side, ``[..., ny+2,
    nx+2]``; its west column and south row carry that neighbour diagonal: the
    wrap on a periodic axis, the neighbour rank's value across a rank seam, zero
    across a wall or open edge (where ``aW_edge``/``aS_edge`` are zero anyway).

    Returns cell-shaped ``(pC, pW, pS)``, ``[..., ny, nx]``: ``pW[..., j, i]``
    sits on the WEST edge of cell ``i`` and ``pS[..., j, i]`` on the SOUTH edge
    of cell ``j``, as MIT stores them per tile.  The east edge of the last
    column and the north edge of the last row belong to the neighbour tile;
    MIT fills them by ``EXCH_UV_XY_RS(pW, pS)`` (``ini_cg2d.F:241``).
    """
    ny = aC_halo.shape[-2] - 2
    nx = aC_halo.shape[-1] - 2
    aC = aC_halo[..., 1:ny + 1, 1:nx + 1]
    aC_w = aC_halo[..., 1:ny + 1, 0:nx]
    aC_s = aC_halo[..., 0:ny, 1:nx + 1]
    aW = aW_edge[..., 0:nx]
    aS = aS_edge[..., 0:ny, :]
    pC = build_cg2d_jacobi_inverse(aC, threshold=threshold)

    denW = (offdiag_factor * (aC_w + aC)) ** 2
    denS = (offdiag_factor * (aC_s + aC)) ** 2
    maskW = denW > threshold
    maskS = denS > threshold
    pW = torch.where(
        maskW,
        -aW / torch.where(maskW, denW, torch.ones_like(denW)),
        torch.zeros_like(aW),
    )
    pS = torch.where(
        maskS,
        -aS / torch.where(maskS, denS, torch.ones_like(denS)),
        torch.zeros_like(aS),
    )
    return pC, pW, pS


def apply_cg2d_jacobi_preconditioner(
    precond_inv: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Apply diagonal CG2D Jacobi preconditioning."""
    scale = (
        precond_inv.unsqueeze(0)
        if residual.ndim == precond_inv.ndim + 1
        else precond_inv
    )
    return scale * residual


def apply_cg2d_ssor_preconditioner(
    residual: torch.Tensor,
    pC: torch.Tensor,
    pW: torch.Tensor,
    pS: torch.Tensor,
) -> torch.Tensor:
    """Tile-local five-point SSOR apply: no neighbour across any tile edge.

    The same statements as :func:`apply_cg2d_ssor_preconditioner_wrapped`
    with neither axis wrapped, kept as their own body: this is the eager
    apply of every single-process regional run without a periodic axis, and a
    call through the wrapped function costs one more Python frame per
    iteration (about 1 us in the measured CUDA kernel).
    """
    has_batch = residual.ndim == pC.ndim + 1
    r = residual[0] if has_batch else residual
    z = pC * r
    z[..., :, 1:] += pW[..., :, 1:] * r[..., :, :-1]
    z[..., :, :-1] += pW[..., :, 1:] * r[..., :, 1:]
    z[..., 1:, :] += pS[..., 1:, :] * r[..., :-1, :]
    z[..., :-1, :] += pS[..., 1:, :] * r[..., 1:, :]
    return z.unsqueeze(0) if has_batch else z


def apply_cg2d_ssor_preconditioner_wrapped(
    residual: torch.Tensor,
    pC: torch.Tensor,
    pW: torch.Tensor,
    pS: torch.Tensor,
    *,
    wrap_x: bool,
    wrap_y: bool,
) -> torch.Tensor:
    """Five-point SSOR apply of ``cg2d.F:216-222`` on one process's array.

    ``z(i,j) = pC r + pW(i) r(i-1) + pW(i+1) r(i+1) + pS(j) r(j-1) + pS(j+1) r(j+1)``
    with the cell-shaped coefficients of
    :func:`build_cg2d_ssor_preconditioner_from_halo`.  A wrapped axis is a
    periodic axis this process owns whole: its neighbour across the domain edge
    is the opposite end of the same array, and the edge between cell ``n-1`` and
    cell ``0`` is cell ``0``'s west (south) edge, which is what MIT's exchange
    gives on a single tile (``r(0) = r(sNx)``, ``pW(sNx+1) = pW(1)``).  On an
    unwrapped axis the edge terms are absent, as across a wall or open edge.

    Each cell accumulates its five terms in MIT's order (centre, west, east,
    south, north).  On a wrapped axis each direction is one whole-array add:
    the west term is ``pW * roll(r, +1)`` and the east term is
    ``roll(pW * r, -1)``, the product ``pW(i+1) r(i+1)`` formed where it sits
    and moved one cell, so the wrap column needs no separate statement and
    every cell sees the same two rounded operations as an interior one.  (An
    edge-column form costs four more small launches and views per call; the
    eager apply is host-bound at production sizes.)
    The halo form :func:`apply_cg2d_ssor_preconditioner_edge` and the fused
    kernel K3 compute the same expression; all three agree bit for bit.
    """
    has_batch = residual.ndim == pC.ndim + 1
    r = residual[0] if has_batch else residual
    z = pC * r
    if wrap_x:
        z += pW * torch.roll(r, 1, -1)
        z += torch.roll(pW * r, -1, -1)
    else:
        z[..., :, 1:] += pW[..., :, 1:] * r[..., :, :-1]
        z[..., :, :-1] += pW[..., :, 1:] * r[..., :, 1:]
    if wrap_y:
        z += pS * torch.roll(r, 1, -2)
        z += torch.roll(pS * r, -1, -2)
    else:
        z[..., 1:, :] += pS[..., 1:, :] * r[..., :-1, :]
        z[..., :-1, :] += pS[..., 1:, :] * r[..., 1:, :]
    return z.unsqueeze(0) if has_batch else z


def build_cg2d_ssor_preconditioner_edge(
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
    aC_west_halo: torch.Tensor,
    aC_south_halo: torch.Tensor,
    *,
    offdiag_factor: float = 0.51,
    threshold: float = 1e-30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seam-aware edge-coefficient CG2D SSOR preconditioner (cube path).

    Unlike :func:`build_cg2d_ssor_preconditioner` (which reads neighbour
    diagonals via plain array shifts that read 0 across a cube seam), this
    builder consumes EXPLICIT west/south neighbour-diagonal halos so the seam
    edges ``i=0`` / ``j=0`` use the cross-face neighbour ``aC`` (the analogue of
    the matvec reading the haloed state).  It returns EDGE-shaped preconditioner
    coefficients so the reconciled east-seam edge ``aW_edge[...,nx]`` and
    north-seam edge ``aS_edge[...,ny]`` are represented too:

      ``pW_edge[..., i]`` is the SSOR coefficient on the u-edge between cells
      ``i-1`` and ``i``  =  ``-aW_edge[...,i] / (offd·(aC(i-1)+aC(i)))²``.
      ``pS_edge[..., j]`` is the analogue on the v-edge between cells ``j-1,j``.

    Parameters
    ----------
    aW_edge : west-edge coefficient ``[..., ny, nx+1]`` (already seam-reconciled).
    aS_edge : south-edge coefficient ``[..., ny+1, nx]`` (already seam-reconciled).
    aC : main diagonal ``[..., ny, nx]``.
    aC_west_halo : ``aC`` with a width-1 WEST halo, shape ``[..., ny, nx+1]``;
        ``aC_west_halo[..., i]`` is the diagonal of the cell WEST of edge ``i``
        (``aC(i-1)`` for the local cell ``i``; at ``i=0`` the cross-seam
        neighbour, at ``i=nx`` the local cell ``nx-1``).
    aC_south_halo : ``aC`` with a width-1 SOUTH halo, shape ``[..., ny+1, nx]``;
        ``aC_south_halo[..., j, :]`` is the diagonal of the cell SOUTH of
        edge ``j``.

    The denominator at edge ``i`` is ``offd·(aC_west(i) + aC_east(i))`` where
    ``aC_east(i)`` is the cell EAST of the edge: for edges ``1..nx`` that is the
    local ``aC`` shifted, and the WEST halo already carries the seam value at
    ``i=0``.  Double-where guards keep the backward graph finite at land cells.
    """
    ny = aC.shape[-2]
    nx = aC.shape[-1]
    pC = build_cg2d_jacobi_inverse(aC, threshold=threshold)

    # aC on both sides of every u-edge (length nx+1):
    #   aC_w_e[..., i] = aC of cell WEST of edge i  (from the west-halo array)
    #   aC_e_e[..., i] = aC of cell EAST of edge i
    aC_w_e = aC_west_halo                                   # [..., ny, nx+1]
    aC_e_e = aC.new_zeros(aC_west_halo.shape)               # [..., ny, nx+1]
    aC_e_e[..., 0:nx] = aC                                  # edges 0..nx-1: east cell = local i
    aC_e_e[..., nx] = aC[..., nx - 1]                       # edge nx: east cell = local nx-1 (its own east edge)
    denW = (offdiag_factor * (aC_w_e + aC_e_e)) ** 2
    maskW = denW > threshold
    pW_edge = torch.where(
        maskW,
        -aW_edge / torch.where(maskW, denW, torch.ones_like(denW)),
        torch.zeros_like(aW_edge),
    )

    # aC on both sides of every v-edge (length ny+1).
    aC_s_e = aC_south_halo                                  # [..., ny+1, nx]
    aC_n_e = aC.new_zeros(aC_south_halo.shape)              # [..., ny+1, nx]
    aC_n_e[..., 0:ny, :] = aC
    aC_n_e[..., ny, :] = aC[..., ny - 1, :]
    denS = (offdiag_factor * (aC_s_e + aC_n_e)) ** 2
    maskS = denS > threshold
    pS_edge = torch.where(
        maskS,
        -aS_edge / torch.where(maskS, denS, torch.ones_like(denS)),
        torch.zeros_like(aS_edge),
    )
    return pC, pW_edge, pS_edge


def apply_cg2d_ssor_preconditioner_edge(
    residual_halo: torch.Tensor,
    pC: torch.Tensor,
    pW_edge: torch.Tensor,
    pS_edge: torch.Tensor,
    *,
    halo: int = 1,
) -> torch.Tensor:
    """Five-point symmetric SSOR apply from an exchanged center halo.

    Mirrors :func:`apply_cg2d_ssor_preconditioner` but reads the four neighbours
    from a pre-exchanged width-``halo`` center halo (the caller supplies the
    exchange: the cube's cross-seam scalar exchange, or the regional
    ``exchange`` across rank seams and periodic edges), using EDGE-shaped
    coefficients so the east/north tile edges are honoured.  It is the SSOR
    analogue of :func:`edge_coefficient_helmholtz_from_haloed_center` (same
    neighbour-slice structure, ``pC/pW/pS`` in place of ``aC/aW/aS``), i.e.
    MIT ``cg2d.F:216-222`` reading ``cg2d_r`` after the EXCH at ``cg2d.F:181``
    / ``:353``:

      ``z(i) = pC·r(i)
               + pW_edge(i)·r(i-1) + pW_edge(i+1)·r(i+1)
               + pS_edge(j)·r(j-1) + pS_edge(j+1)·r(j+1)``

    The apply is a single parallel stencil (no serial sweep), so the halo read
    has no ordering dependence — identical to the matvec's treatment.
    ``residual_halo`` carries the leading batch/face axes broadcast-ready against
    the coefficients.  Returns the un-haloed result.
    """
    ny, nx = pC.shape[-2:]
    ol = halo
    inner = residual_halo[..., ol:ol + ny, ol:ol + nx]
    west = residual_halo[..., ol:ol + ny, ol - 1:ol + nx - 1]
    east = residual_halo[..., ol:ol + ny, ol + 1:ol + nx + 1]
    south = residual_halo[..., ol - 1:ol + ny - 1, ol:ol + nx]
    north = residual_halo[..., ol + 1:ol + ny + 1, ol:ol + nx]

    pW_w = pW_edge[..., 0:nx]
    pW_e = pW_edge[..., 1:nx + 1]
    pS_s = pS_edge[..., 0:ny, :]
    pS_n = pS_edge[..., 1:ny + 1, :]

    z = pC * inner
    z = z + pW_w * west + pW_e * east
    z = z + pS_s * south + pS_n * north
    return z
