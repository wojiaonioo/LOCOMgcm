"""W control-volume geometry of MITgcm ``CALC_GW`` (``calc_gw.F``).

The W cell of level ``k >= 1`` reaches from the centre of tracer level ``k-1``
to the centre of level ``k``.  Depths use MIT's upward ``r``: ``r_c`` is
``rC = -center_depth`` ``[nz]`` (negative below the surface), ``r_low`` the
column bottom ``R_low < 0`` resynced from the truncated ``hFacC`` and
``ro_surf`` the resting surface (0).  ``dr_f`` is ``drF`` ``[nz]`` and
``hfac_c`` is ``hFacC`` ``[nz, ny, nx]`` (ones on a full-cell grid).
Thicknesses are in metres: ``[nz, ny, nx]`` at tracer columns, ``[nz, ny, nx+1]``
on W (U-point) faces and ``[nz, ny+1, nx]`` on S (V-point) faces.  No wet mask
is applied here; the caller multiplies the centre thickness by the centre
mask where CALC_GW does.  Results keep the dtype of ``dr_f``.
"""

from __future__ import annotations

import torch


def w_column_bounds(
    dr_f: torch.Tensor, hfac_c: torch.Tensor, *, nz: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(r_low, ro_surf)`` of every tracer column from the truncated ``hFacC``.

    Implements MIT ``ini_masks_etc.F:141`` R_low resync: after hFacC truncation
    (hfacMin/hfacMinDz floor + halfRL close-to-zero cutoff), MIT recomputes
    R_low as ``rF(1) - Σ_k drF(k)·hFacC(k)`` so that R_low and Ro_surf use the
    SAME truncated hFacC source.  Without this resync LOCOMgcm would mix raw
    ``MetricSet.depth_center`` (raw bathy) with truncated ``hFacC``, producing
    sub-epsilon FP residue at coastal partial cells whose bathymetry collides
    with a layer interface.  That residue propagates through
    ``_horizontal_w_advection`` (recip_volume = 1/(rA·thin) → O(1e7)) and
    can amplify the vertical momentum tendency, pressure correction and
    free-surface response near coastal partial cells.
    """
    # MIT-aligned R_low resync: derive r_low from truncated hFacC, NOT from
    # raw ``MetricSet.depth_center``.  This keeps r_low and ro_surf
    # self-consistent (ro_surf == 0 at all wet columns) and matches MIT
    # ini_masks_etc.F:141.
    column_sum = (dr_f.view(nz, 1, 1) * hfac_c).sum(dim=0)
    r_low = -column_sum
    ro_surf = r_low + column_sum  # ≡ 0 by construction; kept for clarity
    return r_low, ro_surf


def w_cell_thickness_center(
    r_c: torch.Tensor, r_low: torch.Tensor, ro_surf: torch.Tensor
) -> torch.Tensor:
    """CALC_GW's unmasked ``rThickC_C`` at tracer columns.

    A W cell can contain part of the wet layer above even when its lower
    tracer cell is dry. MOM_W_SIDEDRAG needs that thickness in its neighbour
    height differences; the reciprocal volume still uses the wet mask.
    """
    thickness = r_low.new_zeros((r_c.shape[0], *r_low.shape))
    thickness[0] = torch.clamp(ro_surf - torch.maximum(r_low, r_c[0]), min=0.0)
    thickness[1:] = torch.clamp(
        torch.minimum(ro_surf, r_c[:-1, None, None])
        - torch.maximum(r_low, r_c[1:, None, None]),
        min=0.0,
    )
    return thickness


def w_recip_volume(r_thick: torch.Tensor, *, cell_area: torch.Tensor) -> torch.Tensor:
    """``1 / (rA * thickness)`` of wet W cells, 0 where the masked thickness is 0."""
    wet = r_thick > 0
    volume = cell_area * r_thick
    denominator = torch.where(wet, volume, torch.ones_like(volume))
    return torch.where(
        wet, 1.0 / denominator, torch.zeros_like(r_thick),
    )


def w_face_thickness_x(
    *,
    dr_f: torch.Tensor,
    r_c: torch.Tensor,
    r_low_c: torch.Tensor,
    ro_surf_c: torch.Tensor,
    hfac_w: torch.Tensor,
    edge_couples: bool,
    r_low_halo: torch.Tensor | None,
    west_neighbor: bool,
    east_neighbor: bool,
) -> torch.Tensor:
    """W-cell thickness on W faces, MIT ``rSurfW``/``rLowW`` in CALC_GW.

    ``edge_couples`` is ``grid.w_face_thickness_edge_couples("x")``.
    ``r_low_halo`` is ``r_low_c`` with a one-column ZERO-filled halo, exchanged
    only when ``west_neighbor`` or ``east_neighbor`` (a distributed rank
    neighbour on that side) holds; otherwise it is None and unread.
    """
    nz = dr_f.shape[0]
    ny, nx = r_low_c.shape
    r_surf_w = r_low_c.new_zeros((ny, nx + 1))
    r_low_w = r_low_c.new_zeros((ny, nx + 1))
    if nx > 1:
        r_surf_w[:, 1:nx] = torch.minimum(ro_surf_c[:, :-1], ro_surf_c[:, 1:])
        r_low_w[:, 1:nx] = torch.maximum(r_low_c[:, :-1], r_low_c[:, 1:])
    # Topology-agnostic W-cell edge-face pairing (numerics rule 11): the Grid
    # is the single authority on whether the domain-edge W-cell faces couple
    # the two domain-end cells (wrap pairing -- periodic, open, cube seam) or
    # replicate a single interior cell (wall).  The distributed per-side halo
    # resolution below deliberately runs after this branch, never inside the
    # predicate; see ``StructuredCGrid.w_face_thickness_edge_couples``.
    if edge_couples:
        r_low_edge = torch.maximum(r_low_c[:, 0], r_low_c[:, -1])
        r_surf_edge = torch.maximum(
            torch.minimum(ro_surf_c[:, 0], ro_surf_c[:, -1]),
            r_low_edge,
        )
        r_low_w[:, 0] = r_low_edge
        r_low_w[:, -1] = r_low_edge
        r_surf_w[:, 0] = r_surf_edge
        r_surf_w[:, -1] = r_surf_edge
    else:
        r_low_w[:, 0] = r_low_c[:, 0]
        r_low_w[:, -1] = r_low_c[:, -1]
        r_surf_w[:, 0] = ro_surf_c[:, 0]
        r_surf_w[:, -1] = ro_surf_c[:, -1]
    if west_neighbor:
        r_low_w[:, 0] = torch.maximum(r_low_halo[:, 0], r_low_halo[:, 1])
    if east_neighbor:
        r_low_w[:, -1] = torch.maximum(r_low_halo[:, nx], r_low_halo[:, nx + 1])
    r_surf_w = r_low_w + (dr_f.view(nz, 1, 1) * hfac_w).sum(dim=0)
    thickness = r_low_c.new_zeros((nz, ny, nx + 1))
    thickness[0] = torch.clamp(r_surf_w - torch.maximum(r_low_w, r_c[0]), min=0.0)
    thickness[1:] = torch.clamp(
        torch.minimum(r_surf_w, r_c[:-1, None, None])
        - torch.maximum(r_low_w, r_c[1:, None, None]),
        min=0.0,
    )
    # No per-level hFacW mask: the W-cell straddles tracer levels k-1/k, so a
    # face dry at level k can still carry wet geometric overlap from level k-1
    # (staircase coast). MIT calc_gw.f:3346-3351 keeps that overlap; the clamp
    # above already yields 0 for fully-dry columns. A per-level mask here zeroed
    # valid dry-adjacent flx_EW and broke gW alignment (~6.15e-5).
    return thickness


def w_face_thickness_y(
    *,
    dr_f: torch.Tensor,
    r_c: torch.Tensor,
    r_low_c: torch.Tensor,
    ro_surf_c: torch.Tensor,
    hfac_s: torch.Tensor,
    edge_couples: bool,
    r_low_halo: torch.Tensor | None,
    south_neighbor: bool,
    north_neighbor: bool,
) -> torch.Tensor:
    """W-cell thickness on S faces; the y-axis twin of :func:`w_face_thickness_x`."""
    nz = dr_f.shape[0]
    ny, nx = r_low_c.shape
    r_surf_s = r_low_c.new_zeros((ny + 1, nx))
    r_low_s = r_low_c.new_zeros((ny + 1, nx))
    if ny > 1:
        r_surf_s[1:ny, :] = torch.minimum(ro_surf_c[:-1, :], ro_surf_c[1:, :])
        r_low_s[1:ny, :] = torch.maximum(r_low_c[:-1, :], r_low_c[1:, :])
    # Topology-agnostic W-cell edge-face pairing: see w_face_thickness_x.
    if edge_couples:
        r_low_edge = torch.maximum(r_low_c[0, :], r_low_c[-1, :])
        r_surf_edge = torch.maximum(
            torch.minimum(ro_surf_c[0, :], ro_surf_c[-1, :]),
            r_low_edge,
        )
        r_low_s[0, :] = r_low_edge
        r_low_s[-1, :] = r_low_edge
        r_surf_s[0, :] = r_surf_edge
        r_surf_s[-1, :] = r_surf_edge
    else:
        r_low_s[0, :] = r_low_c[0, :]
        r_low_s[-1, :] = r_low_c[-1, :]
        r_surf_s[0, :] = ro_surf_c[0, :]
        r_surf_s[-1, :] = ro_surf_c[-1, :]
    if south_neighbor:
        r_low_s[0, :] = torch.maximum(r_low_halo[0, :], r_low_halo[1, :])
    if north_neighbor:
        r_low_s[-1, :] = torch.maximum(r_low_halo[ny, :], r_low_halo[ny + 1, :])
    r_surf_s = r_low_s + (dr_f.view(nz, 1, 1) * hfac_s).sum(dim=0)
    thickness = r_low_c.new_zeros((nz, ny + 1, nx))
    thickness[0] = torch.clamp(r_surf_s - torch.maximum(r_low_s, r_c[0]), min=0.0)
    thickness[1:] = torch.clamp(
        torch.minimum(r_surf_s, r_c[:-1, None, None])
        - torch.maximum(r_low_s, r_c[1:, None, None]),
        min=0.0,
    )
    # No per-level hFacS mask: see w_face_thickness_x (matches MIT
    # calc_gw.f:3357-3362; the clamp already zeros fully-dry columns).
    return thickness
