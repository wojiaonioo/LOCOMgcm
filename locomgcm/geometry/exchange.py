"""Operational halo exchange for StructuredCGrid tensors.

``HaloExtentV1`` is the local immutable schema for the per-side widths used
by this implementation. There is no separate registry or pending identity
layer behind this contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import os
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist

from locomgcm.accel.tensor_context import extent_mismatch
from locomgcm.distributed.halo_p2p import halo_p2p, inference_ghost_after_send
from locomgcm.distributed.comm_adjoint import CommSession, session_for_grid, transfer
from locomgcm.geometry.grid import BoundaryType, Grid


class GridLocation(StrEnum):
    """C-grid location of an exchanged field: the one table of location names.

    A member is the ``str`` of its name, so a caller passes either.  ``exchange``
    hands the traced path plain strings: a member tested against a ``frozenset``
    of strings breaks the Dynamo trace.  Every location literal passed to
    ``exchange`` names a member of this table.
    """

    T = "T"
    U = "U"
    V = "V"
    W = "W"
    F = "F"
    Z = "Z"


CGridLocation = GridLocation | str
BoundarySide = Literal["west", "east", "south", "north"]
ObcEdges = Mapping[BoundarySide, torch.Tensor]

_VALID_LOCATIONS = frozenset(item.value for item in GridLocation)
_CUBED_SPHERE_FACE_COUNT = 6

# Read once, at import, so the switch is a host constant: a per-call
# ``os.environ`` lookup inside ``_pad_axis_distributed`` would be traced as a
# guard (and, under a compiled posture, would silently freeze whichever value
# the first trace saw).  The packed transport is opt-in until the CUDA/NCCL
# comparison lands; the default (unset or "0") runs the historical
# per-(wrap, position) rounds byte for byte.
EXCHANGE_PACKED = os.environ.get("LOCOMGCM_EXCHANGE_PACKED", "0") == "1"


@dataclass(frozen=True)
class HaloExtentV1:
    """Per-side halo widths consumed by the exchange implementation."""

    west_left_extent: int
    east_right_extent: int
    south_bottom_extent: int
    north_top_extent: int
    schema_version: str = "v1"

    def __post_init__(self) -> None:
        for name in (
            "west_left_extent",
            "east_right_extent",
            "south_bottom_extent",
            "north_top_extent",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.schema_version != "v1":
            raise ValueError("HaloExtentV1.schema_version must be 'v1'")

    @classmethod
    def uniform(cls, h: int) -> "HaloExtentV1":
        """Construct an extent with the same halo width on all four sides."""

        return cls(
            west_left_extent=h,
            east_right_extent=h,
            south_bottom_extent=h,
            north_top_extent=h,
        )


class FillSpec(StrEnum):
    """Ghost-fill policy for non-periodic axes."""

    ZERO = "zero"
    REPLICATE = "replicate"
    OBC_VALUE = "obc_value"
    WRAP = "wrap"


def exchange(
    field: torch.Tensor,
    *,
    grid: Grid,
    location: CGridLocation,
    extent: HaloExtentV1,
    fill: FillSpec,
    obc: ObcEdges | None = None,
    session: CommSession | None = None,
    constant: bool = False,
) -> torch.Tensor:
    """Return a new tensor with horizontal ghost cells added.

    Only the last two dimensions are padded: ``j`` on axis ``-2`` and ``i``
    on axis ``-1``. Leading batch and vertical dimensions pass through
    unchanged. A 1-D metric row is also accepted; its single dimension is
    padded along whichever horizontal axis has non-zero extent.

    ``obc`` is used only with ``fill=FillSpec.OBC_VALUE`` and may contain
    ``"west"``, ``"east"``, ``"south"``, and/or ``"north"`` edge-value
    tensors. Supplied edge values are copied directly into ghost cells on
    OPEN axes. Missing sides, and non-OPEN axes, fall back to REPLICATE.
    Edge tensors may either match the ghost strip shape, carry a singleton
    ghost-width dimension, or omit the padded axis so it can be inserted.
    For south/north edges, values with the unpadded x-width are x-padded
    before insertion so x-then-y corner order is preserved.

    On a PERIODIC axis a U array along x (``nx + 1`` faces) or a V array
    along y (``ny + 1``) stores the seam face twice; the wrap then runs over
    the unique faces (west ghosts are faces ``nx-w .. nx-1``, the ghosts
    after the stored face ``nx`` are faces ``1 .. w``), as MITgcm EXCH does
    with ``u(0) = u(sNx)``.  The one predicate is
    ``_distributed_send_skips_duplicate_face``, shared with the distributed
    send strips.

    ``constant=True`` declares a field that depends on no control on any rank
    (masks, metrics, static geometry, sampled forcing that is not a control):
    inside a differentiation window its round stays on the session (recorded,
    replayed and ordered on the session link like every other round) but the
    received halo is detached, so the result stays free of ``requires_grad``.
    A field received through a session round requires grad because the link
    does, and a constant that enters the session undetached turns every
    expression consuming it into a differentiable one (two ranks saved 181
    tile-sized fields per step where one rank saved 78 domain-sized ones).
    Routing constants to the inference transport instead deadlocked
    the compiled two-rank step (see ``comm_adjoint.transfer``).  The
    declaration must hold on every rank, or a rank would drop the cotangent
    its neighbour's field needs; a field that requires grad is rejected here
    rather than exchanged silently.
    """

    if not isinstance(field, torch.Tensor):
        raise TypeError("field must be a torch.Tensor")
    if field.ndim == 0:
        raise ValueError("exchange requires at least one tensor dimension")
    if isinstance(location, GridLocation):
        location = location.value
    if location not in _VALID_LOCATIONS:
        raise ValueError(f"unsupported C-grid location: {location!r}")
    if constant and field.requires_grad:
        raise ValueError("a constant exchange must not carry a gradient: the field requires grad")

    # Enum construction from an enum instance is a traced-Python detour
    # (``EnumType.__iter__`` polyfill); the identity path is the common case.
    fill_spec = fill if isinstance(fill, FillSpec) else FillSpec(fill)

    # Transport dispatch: a cubed-sphere (SEAM) grid needs a whole-tensor
    # cross-face gather (face f's halo comes from a *different* face, with a
    # geometric rotation), which the single-patch per-axis flow below cannot
    # express.  Route it to the connectivity-driven cube backend.  Regional
    # PERIODIC/WALL/OPEN grids never carry SEAM and fall through unchanged.
    boundary = grid.boundary_topology
    if BoundaryType.SEAM in (boundary.west_east, boundary.south_north):
        # Face-local (multi-rank) routing: when a CubeRankContext is attached
        # the field is rank-local [n_local,...] and the cross-face halo crosses
        # ranks, so route to the distributed P2P backend (proven bitwise
        # == single-rank rot90).  ctx=None (every caller today) is byte-identical
        # to the original single-rank path below.
        ctx = getattr(grid, "_cube_step_ctx", None)
        if ctx is not None:
            return exchange_cubed_sphere_scalar_distributed(
                field, grid=grid, location=location, extent=extent, ctx=ctx,
                session=session, constant=constant,
            )
        return _exchange_cubed_sphere_scalar(
            field, grid=grid, location=location, extent=extent,
        )

    if field.ndim == 1:
        return _exchange_metric_row(field, grid=grid, extent=extent, fill=fill_spec, obc=obc, session=session,
                                    constant=constant)

    out = field
    padded = False

    west = extent.west_left_extent
    east = extent.east_right_extent
    if west or east:
        if _has_distributed_transport(grid, ("west", "east")):
            out = _pad_axis_distributed(
                out,
                grid=grid,
                location=location,
                axis=-1,
                before=west,
                after=east,
                before_side="west",
                after_side="east",
                boundary_type=grid.boundary_topology.west_east,
                fill=fill_spec,
                obc=obc,
                session=session,
                constant=constant,
            )
        else:
            out = _pad_axis(
                out,
                grid=grid,
                location=location,
                axis=-1,
                before=west,
                after=east,
                before_side="west",
                after_side="east",
                boundary_type=grid.boundary_topology.west_east,
                fill=fill_spec,
                obc=obc,
            )
        padded = True

    south = extent.south_bottom_extent
    north = extent.north_top_extent
    if south or north:
        if _has_distributed_transport(grid, ("south", "north")):
            out = _pad_axis_distributed(
                out,
                grid=grid,
                location=location,
                axis=-2,
                before=south,
                after=north,
                before_side="south",
                after_side="north",
                boundary_type=grid.boundary_topology.south_north,
                fill=fill_spec,
                obc=obc,
                x_extent=(west, east),
                x_boundary_type=grid.boundary_topology.west_east,
                session=session,
                constant=constant,
            )
        else:
            out = _pad_axis(
                out,
                grid=grid,
                location=location,
                axis=-2,
                before=south,
                after=north,
                before_side="south",
                after_side="north",
                boundary_type=grid.boundary_topology.south_north,
                fill=fill_spec,
                obc=obc,
                x_extent=(west, east),
                x_boundary_type=grid.boundary_topology.west_east,
            )
        padded = True

    if not padded:
        return field.clone()
    return out


def exchange_velocity_pair(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    grid: Grid,
    extent: HaloExtentV1,
    with_signs: bool = True,
    session: CommSession | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paired (u, v) halo exchange — cube-safe.

    On a cubed-sphere grid (SEAM topology), u and v must be exchanged
    together because inter-face seams rotate the velocity components.
    This routes to ``exchange_cubed_sphere_vector``.

    On a regional grid, u and v are exchanged independently via the
    standard ``exchange()`` function (no coupling needed).

    Returns ``(u_ext, v_ext)`` with halos added.  On the cube the
    returned tensors have shape ``[6,...,ny+2*OL,nx+2*OL]`` (same-size
    haloed frame).  On a regional grid the shapes match what
    ``exchange()`` would return for each field individually.
    """
    boundary = grid.boundary_topology
    is_cube = BoundaryType.SEAM in (boundary.west_east, boundary.south_north)

    if is_cube:
        # Face-local routing (see exchange()): distributed when ctx attached,
        # else the byte-identical single-rank vector exchange.  ctx present ⟺
        # (u, v) are rank-local [n_local,...] and extent is uniform OL>=1.
        ctx = getattr(grid, "_cube_step_ctx", None)
        if ctx is not None:
            return exchange_cubed_sphere_vector_distributed(
                u, v, grid=grid, extent=extent, ctx=ctx, with_signs=with_signs,
                session=session,
            )
        return exchange_cubed_sphere_vector(
            u, v, grid=grid, extent=extent, with_signs=with_signs,
        )

    u_ext = exchange(
        u, grid=grid, location="U", extent=extent, fill=FillSpec.ZERO,
        session=session,
    )
    v_ext = exchange(
        v, grid=grid, location="V", extent=extent, fill=FillSpec.ZERO,
        session=session,
    )
    return u_ext, v_ext


def exchange_hfac_pair(
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    *,
    grid: Grid,
    extent: HaloExtentV1,
    session: CommSession | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange (hFacW, hFacS) as a stagger-paired metric field.

    hFacW lives at u-face stagger ``[..., ny, nx+1]`` and hFacS at v-face
    stagger ``[..., ny+1, nx]``.  On a cubed-sphere grid the inter-face
    seam rotates u-stagger into v-stagger, so the two fields must be
    exchanged together through ``exchange_cubed_sphere_vector`` with
    ``with_signs=False`` (swap without sign flip).

    On a regional grid the two fields are exchanged independently via the
    standard ``exchange()`` function with ``fill=WRAP`` (matching the
    periodic self-wrap used in MITgcm's ``mom_calc_hfacz``).

    Returns ``(hW_ext, hS_ext)`` with halos added.  On the cube the
    returned tensors are in the same-size haloed frame
    ``[6,...,n+2*OL,n+2*OL]``.
    """
    boundary = grid.boundary_topology
    is_cube = BoundaryType.SEAM in (boundary.west_east, boundary.south_north)

    if is_cube:
        # Face-local routing (see exchange()): distributed when ctx attached,
        # else the byte-identical single-rank exchange.  with_signs=False = the
        # stagger swap without the velocity sign flip.
        ctx = getattr(grid, "_cube_step_ctx", None)
        if ctx is not None:
            return exchange_cubed_sphere_vector_distributed(
                hFacW, hFacS, grid=grid, extent=extent, ctx=ctx, with_signs=False,
                session=session,
            )
        return exchange_cubed_sphere_vector(
            hFacW, hFacS, grid=grid, extent=extent, with_signs=False,
        )

    hW_ext = exchange(
        hFacW, grid=grid, location="U", extent=extent, fill=FillSpec.WRAP,
        session=session,
    )
    hS_ext = exchange(
        hFacS, grid=grid, location="V", extent=extent, fill=FillSpec.WRAP,
        session=session,
    )
    return hW_ext, hS_ext


def _validate_cube_halo_extent(grid: Grid, extent: HaloExtentV1) -> None:
    """Cube exchanges take each side strip from one neighbour's interior."""
    widths = (
        extent.west_left_extent, extent.east_right_extent,
        extent.south_bottom_extent, extent.north_top_extent,
    )
    if max(widths[:2]) > grid.nx or max(widths[2:]) > grid.ny:
        raise ValueError(
            f"cubed-sphere halo widths {widths} exceed face shape "
            f"(ny={grid.ny}, nx={grid.nx}); exchange requires each halo "
            "to fit inside one neighbour face"
        )


def _exchange_cubed_sphere_scalar(
    field: torch.Tensor,
    *,
    grid: Grid,
    location: CGridLocation,
    extent: HaloExtentV1,
) -> torch.Tensor:
    """Cross-face halo gather for a centre (T-point) cubed-sphere scalar.

    The 6 cube faces are stored as one tensor ``field[6, ..., ny, nx]``.  Each
    face's side halo is filled from the *interior* cells of the neighbour face
    given by ``grid.tile_connectivity[(face, side)] = (nf, ns, rotation)``: the
    whole neighbour face is rotated into this face's frame (``torch.rot90``) and
    the adjacent ``width``-deep strip is copied in (a verbatim copy — scalars
    carry no sign, matching MITgcm ``exch2_rx1_cube`` ``signOption=.FALSE.``).

    After the edge strips are populated, a corner-fill post-pass (below;
    ``FILL_CS_CORNER_TR_RL`` ``fill4dir=1`` equivalent) transposes each adjacent
    edge strip into the four ``OLx×OLy`` halo *corner* blocks, matching MITgcm
    where ``FILL_CS_CORNER`` runs after the edge exchange.  Staggered (U/V/Z) and
    the C-grid vector u<->v rotation are separate; this routine is centre-scalar only.
    """
    _validate_cube_halo_extent(grid, extent)
    if field.ndim < 3 or field.shape[0] != _CUBED_SPHERE_FACE_COUNT:
        raise ValueError(
            "cubed-sphere exchange needs field shape "
            f"({_CUBED_SPHERE_FACE_COUNT}, ..., ny, nx)"
        )
    n = grid.nx
    if grid.ny != n:
        raise NotImplementedError(
            "cubed-sphere exchange requires square faces (ny == nx)"
        )
    if tuple(field.shape[-2:]) != (n, n):
        raise NotImplementedError(
            "cubed-sphere scalar exchange supports only centre (T) fields "
            f"[{_CUBED_SPHERE_FACE_COUNT}, ..., ny, nx]; staggered U/V/Z "
            "(last dim nx+1 / ny+1) uses the C-grid vector exchange"
        )
    connectivity = getattr(grid, "tile_connectivity", None)
    if not connectivity or len(connectivity) != _CUBED_SPHERE_FACE_COUNT * 4:
        raise ValueError(
            "cubed-sphere exchange requires a 24-entry tile_connectivity table"
        )

    west, east = extent.west_left_extent, extent.east_right_extent
    south, north = extent.south_bottom_extent, extent.north_top_extent
    out = field.new_zeros(
        (*field.shape[:-2], n + south + north, n + west + east)
    )
    out[(slice(None), Ellipsis, slice(south, south + n), slice(west, west + n))] = field

    for face in range(_CUBED_SPHERE_FACE_COUNT):
        for side, width in (
            ("west", west),
            ("east", east),
            ("south", south),
            ("north", north),
        ):
            if width == 0:
                continue
            neighbour_face, _neighbour_side, rotation = connectivity[(face, side)]
            # Rotate the whole neighbour face into THIS face's frame, then the
            # neighbour cells adjacent to the shared edge sit on the opposite
            # side of the rotated tensor (a west halo pulls the rotated
            # neighbour's east strip, etc.).  ``tile_connectivity`` stores the
            # MITgcm pij rotation (this->neighbour index map); bringing the
            # neighbour INTO this frame applies its INVERSE, hence ``-k``.
            k = (4 - rotation // 90) % 4
            neighbour = torch.rot90(field[neighbour_face], k, dims=(-2, -1))
            if side == "west":
                out[(face, Ellipsis, slice(south, south + n), slice(0, width))] = (
                    neighbour[..., :, n - width : n]
                )
            elif side == "east":
                out[(
                    face, Ellipsis, slice(south, south + n),
                    slice(west + n, west + n + width),
                )] = neighbour[..., :, 0:width]
            elif side == "south":
                out[(face, Ellipsis, slice(0, width), slice(west, west + n))] = (
                    neighbour[..., n - width : n, :]
                )
            else:  # north
                out[(
                    face, Ellipsis, slice(south + n, south + n + width),
                    slice(west, west + n),
                )] = neighbour[..., 0:width, :]

    # ---- Corner fill (FILL_CS_CORNER_TR_RL fill4dir=1 equivalent) ----
    # After edge strips are populated, fill the OL×OL corner blocks by
    # transposing from the adjacent edge strip (MIT eesupp/src/
    # fill_cs_corner_tr_rl.F fill4dir=1: SW trFld(1-i,1-j)=trFld(1-j,i)).
    # In LOCOMgcm's 0-based haloed frame with (south, west) offsets:
    #   SW: out[face,..., south-i, west-j] = out[face,..., south-j, west+i-1]
    # Equivalent to: corner block = transpose of adjacent south strip portion.
    if west > 0 and south > 0:
        for face in range(_CUBED_SPHERE_FACE_COUNT):
            # SW corner: rows [0:south], cols [0:west]
            # from south strip: rows [0:south], cols [west:west+west]
            sw_src = out[(face, Ellipsis, slice(0, south), slice(west, west + west))]
            out[(face, Ellipsis, slice(0, south), slice(0, west))] = (
                sw_src.flip(-2).transpose(-2, -1) if south == west
                else sw_src[..., :west, :south].transpose(-2, -1)
            )
            # SE corner: rows [0:south], cols [west+n:west+n+east]
            se_src = out[(face, Ellipsis, slice(0, south), slice(west + n - east, west + n))]
            out[(face, Ellipsis, slice(0, south), slice(west + n, west + n + east))] = (
                se_src.flip(-1).transpose(-2, -1) if south == east
                else se_src[..., :east, :south].transpose(-2, -1)
            )
            # NW corner: rows [south+n:south+n+north], cols [0:west]
            nw_src = out[(face, Ellipsis, slice(south + n - north, south + n), slice(0, west))]
            out[(face, Ellipsis, slice(south + n, south + n + north), slice(0, west))] = (
                nw_src.flip(-2).transpose(-2, -1) if north == west
                else nw_src[..., :west, :north].transpose(-2, -1)
            )
            # NE corner: rows [south+n:south+n+north], cols [west+n:west+n+east]
            # Source: north strip's rightmost columns
            ne_src = out[(face, Ellipsis, slice(south + n, south + n + north), slice(west + n - east, west + n))]
            out[(face, Ellipsis, slice(south + n, south + n + north), slice(west + n, west + n + east))] = (
                ne_src.flip(-1).transpose(-2, -1) if north == east
                else ne_src[..., :east, :north].transpose(-2, -1)
            )
    return out


# ---------------------------------------------------------------------------
# Distributed scalar halo exchange for cubed-sphere
# ---------------------------------------------------------------------------

def _cube_p2p_tag(global_face: int, side: str, halo_width: int) -> int:
    """Deterministic tag for a cube scalar P2P message.

    Tag namespace: 7000 + face*40 + side_idx*10 + width.  The width term
    allows up to 9 distinct halo widths (which covers all current uses).
    Keeps us well clear of the lat-lon range (6300-6311).
    """
    side_idx = {"west": 0, "east": 1, "south": 2, "north": 3}[side]
    return 7000 + global_face * 40 + side_idx * 10 + min(halo_width, 9)


def _cube_scalar_p2p_rounds(
    recv_specs: list,
    send_specs: list,
) -> list[tuple[str, int, str, int]]:
    """Return deterministic rounds for cube scalar P2P halo exchange.

    A *round* is a ``(field_family, global_face, side, halo_width)`` tuple.
    Within each round each peer sends at most one message and receives at
    most one message — this satisfies the NCCL tagless-safe constraint.

    ``recv_specs`` items: ``(round_key, recv_buf, tag, peer_rank)``
    ``send_specs`` items: ``(round_key, send_buf, tag, peer_rank)``
    where ``round_key = ('scalar', global_face, side, halo_width)``.

    Returns the deduplicated, deterministically sorted list of active rounds.
    """
    active: set[tuple] = set()
    for rk, _buf, _tag, _peer in recv_specs:
        active.add(rk)
    for rk, _buf, _tag, _peer in send_specs:
        active.add(rk)
    return sorted(active)


def _sender_strip(
    nf_field: "torch.Tensor", side: str, rotation: int, n: int, width: int
) -> "torch.Tensor":
    """Rotate my face into the receiver's frame and slice the exact ``width``-deep
    border strip the receiver's ``side`` halo needs.

    Byte-identical to the receiver doing rot90+slice on the *full* interior:
    ``rot90`` is a deterministic index permutation and the border slice is a
    deterministic sub-index, so ``slice(rotate(face))`` == ``rotate(face)[slice]``
    bitwise (pure gather, no FP arithmetic).  Sending only this strip therefore
    transmits the same bytes the halo consumes (n*width vs n*n) with no numerical
    change — verified vs the full-face send + single-rank oracle (max|d|=0, incl.
    CUDA NCCL at production cs768)."""
    k = (4 - rotation // 90) % 4
    rot = torch.rot90(nf_field, k, dims=(-2, -1))
    if side == "west":
        return rot[..., :, n - width : n].contiguous()      # [*inner, n, width]
    if side == "east":
        return rot[..., :, 0:width].contiguous()            # [*inner, n, width]
    if side == "south":
        return rot[..., n - width : n, :].contiguous()      # [*inner, width, n]
    return rot[..., 0:width, :].contiguous()                # north [*inner, width, n]


def exchange_cubed_sphere_scalar_distributed(
    local: "torch.Tensor",
    *,
    grid: Grid,
    location: CGridLocation,
    extent: HaloExtentV1,
    ctx: "CubeRankContext",
    group: "dist.ProcessGroup | None" = None,
    session: CommSession | None = None,
    constant: bool = False,
) -> "torch.Tensor":
    """Distributed cubed-sphere scalar halo exchange.

    ``local`` has shape ``[n_local, ..., ny, nx]`` where ``n_local =
    len(ctx.local_face_ids)``.  Returns the haloed tensor of shape
    ``[n_local, ..., ny+south+north, nx+west+east]``.

    For each owned face and each side:
    - Same-rank neighbour: applies the identical rotation + slice as the
      single-rank oracle ``_exchange_cubed_sphere_scalar`` using the local
      tensor — guaranteed bitwise equal.
    - Cross-rank neighbour: P2P exchange of the full interior neighbour face
      (unrotated, sender-frame), then receiver applies the same rotation +
      slice.  Math is identical to the oracle path.

    The P2P pattern: post all recvs, post all sends,
    wait — recv-first/send-after/wait-last via ``dist.batch_isend_irecv``
    with deterministic rounds (field_family, global_face, side, halo_width).
    """
    _validate_cube_halo_extent(grid, extent)
    session = session_for_grid(grid, session)
    n_local = len(ctx.local_face_ids)
    if local.shape[0] != n_local:
        raise ValueError(
            f"local face dim {local.shape[0]} != n_local {n_local} for rank {ctx.rank}"
        )

    n = grid.nx
    if grid.ny != n:
        raise NotImplementedError(
            "distributed cube exchange requires square faces (ny == nx)"
        )
    if tuple(local.shape[-2:]) != (n, n):
        raise ValueError(
            f"local must have trailing shape (ny={n}, nx={n}); got {tuple(local.shape[-2:])}"
        )

    connectivity = getattr(grid, "tile_connectivity", None)
    if not connectivity or len(connectivity) != _CUBED_SPHERE_FACE_COUNT * 4:
        raise ValueError(
            "distributed cube exchange requires a 24-entry tile_connectivity table"
        )

    west, east = extent.west_left_extent, extent.east_right_extent
    south, north = extent.south_bottom_extent, extent.north_top_extent
    inner = local.shape[1:-2]   # batch / z dims

    # Allocate haloed output; copy interiors.
    out_shape = (n_local, *inner, n + south + north, n + west + east)
    out = local.new_zeros(out_shape)
    for lf in range(n_local):
        out[(lf, Ellipsis, slice(south, south + n), slice(west, west + n))] = local[lf]

    # ----- Identify cross-rank seams and build P2P specs -----
    # recv_specs: (round_key, recv_buf, tag, peer_rank)
    # send_specs: (round_key, send_buf, tag, peer_rank)
    recv_specs: list = []
    send_specs: list = []

    # recv_buffers[(global_face, side)] = tensor holding the pre-rotated border
    # strip the sender sliced for this side (shape [*inner, n, width] for
    # west/east, [*inner, width, n] for south/north); dropped straight into the
    # halo with NO receiver-side rotation (the sender already rotated).
    recv_buffers: dict[tuple, "torch.Tensor"] = {}

    for lf, gf in enumerate(ctx.local_face_ids):
        for side, width in (
            ("west", west),
            ("east", east),
            ("south", south),
            ("north", north),
        ):
            if width == 0:
                continue
            nf, _ns, rotation = connectivity[(gf, side)]
            peer_rank = ctx.face_to_rank[nf]
            if peer_rank == ctx.rank:
                continue  # same-rank: handled locally below
            # Cross-rank recv: the sender pre-rotates+slices, so we receive only
            # the border strip the halo needs (n*width, not n*n) -> 768x less.
            rk = ("scalar", gf, side, width)
            tag = _cube_p2p_tag(gf, side, width)
            strip_shape = (
                (*inner, n, width) if side in ("west", "east") else (*inner, width, n)
            )
            recv_buf = local.new_zeros(*strip_shape)
            recv_buffers[(gf, side)] = recv_buf
            recv_specs.append((rk, recv_buf, tag, peer_rank))

    # Build sends: for each of my faces, check which remote faces need it.
    # We iterate over all 24 seams in connectivity; if the neighbour (nf) is
    # mine and the owner (gf) is remote, I need to send local[lf] to remote.
    for (receiver_gf, receiver_side), (nf, _ns, _rot) in connectivity.items():
        if nf not in ctx.global_to_local:
            continue  # nf is not mine
        owner_rank = ctx.face_to_rank[receiver_gf]
        if owner_rank == ctx.rank:
            continue  # same-rank: no send needed
        # I own nf; the receiver_gf owner needs my face rotated into its frame and
        # sliced to the border strip -> send ONLY that strip (n*width), not n*n.
        lf_send = ctx.global_to_local[nf]
        rk = ("scalar", receiver_gf, receiver_side, _width_for_side(receiver_side, extent))
        width_s = _width_for_side(receiver_side, extent)
        if width_s == 0:
            continue
        tag = _cube_p2p_tag(receiver_gf, receiver_side, width_s)
        send_buf = _sender_strip(local[lf_send], receiver_side, _rot, n, width_s)
        send_specs.append((rk, send_buf, tag, owner_rank))

    # Keep the existing single coalesced batch.  The deterministic round keys
    # define the relative receive and send ordering inside that batch.
    if recv_specs or send_specs:
        rounds = _cube_scalar_p2p_rounds(recv_specs, send_specs)
        ordered_recvs = []
        send_buffers, recv_templates = [], []
        send_peers, recv_peers = [], []
        send_tags, recv_tags = [], []
        for active_rk in rounds:
            for spec in recv_specs:
                if spec[0] == active_rk:
                    ordered_recvs.append(spec)
                    recv_templates.append(spec[1])
                    recv_tags.append(spec[2])
                    recv_peers.append(spec[3])
            for rk, buf, tag, peer in send_specs:
                if rk == active_rk:
                    send_buffers.append(buf)
                    send_tags.append(tag)
                    send_peers.append(peer)
        received = transfer(
            send_buffers, recv_templates, send_peers, recv_peers, send_tags, recv_tags,
            group,
            session=session, constant=constant,
        )
        for (rk, _buf, _tag, _peer), recv_buf in zip(ordered_recvs, received):
            recv_buffers[(rk[1], rk[2])] = recv_buf

    # ----- Fill halo strips (same math as single-rank oracle) -----
    for lf, gf in enumerate(ctx.local_face_ids):
        for side, width in (
            ("west", west),
            ("east", east),
            ("south", south),
            ("north", north),
        ):
            if width == 0:
                continue
            nf, _ns, rotation = connectivity[(gf, side)]
            peer_rank = ctx.face_to_rank[nf]

            if peer_rank == ctx.rank:
                # Same-rank: rotate+slice the local neighbour (identical to oracle).
                k = (4 - rotation // 90) % 4
                neighbour = torch.rot90(local[ctx.global_to_local[nf]], k, dims=(-2, -1))
                if side == "west":
                    strip = neighbour[..., :, n - width : n]
                elif side == "east":
                    strip = neighbour[..., :, 0:width]
                elif side == "south":
                    strip = neighbour[..., n - width : n, :]
                else:  # north
                    strip = neighbour[..., 0:width, :]
            else:
                # Cross-rank: the sender already rotated+sliced -> drop strip directly.
                strip = recv_buffers[(gf, side)]

            # Same-rank and cross-rank both yield the receiver-frame border strip;
            # place it into the halo (identical math to _exchange_cubed_sphere_scalar).
            if side == "west":
                out[(lf, Ellipsis, slice(south, south + n), slice(0, width))] = strip
            elif side == "east":
                out[(
                    lf, Ellipsis, slice(south, south + n),
                    slice(west + n, west + n + width),
                )] = strip
            elif side == "south":
                out[(lf, Ellipsis, slice(0, width), slice(west, west + n))] = strip
            else:  # north
                out[(
                    lf, Ellipsis, slice(south + n, south + n + width),
                    slice(west, west + n),
                )] = strip

    # ----- Corner fill (same as oracle :363-397, per owned face) -----
    if west > 0 and south > 0:
        for lf in range(n_local):
            sw_src = out[(lf, Ellipsis, slice(0, south), slice(west, west + west))]
            out[(lf, Ellipsis, slice(0, south), slice(0, west))] = (
                sw_src.flip(-2).transpose(-2, -1) if south == west
                else sw_src[..., :west, :south].transpose(-2, -1)
            )
            se_src = out[(lf, Ellipsis, slice(0, south), slice(west + n - east, west + n))]
            out[(lf, Ellipsis, slice(0, south), slice(west + n, west + n + east))] = (
                se_src.flip(-1).transpose(-2, -1) if south == east
                else se_src[..., :east, :south].transpose(-2, -1)
            )
            nw_src = out[(lf, Ellipsis, slice(south + n - north, south + n), slice(0, west))]
            out[(lf, Ellipsis, slice(south + n, south + n + north), slice(0, west))] = (
                nw_src.flip(-2).transpose(-2, -1) if north == west
                else nw_src[..., :west, :north].transpose(-2, -1)
            )
            ne_src = out[(lf, Ellipsis, slice(south + n, south + n + north), slice(west + n - east, west + n))]
            out[(lf, Ellipsis, slice(south + n, south + n + north), slice(west + n, west + n + east))] = (
                ne_src.flip(-1).transpose(-2, -1) if north == east
                else ne_src[..., :east, :north].transpose(-2, -1)
            )

    return out


def _width_for_side(side: str, extent: "HaloExtentV1") -> int:
    """Return the halo width for a given side from an extent."""
    return {
        "west": extent.west_left_extent,
        "east": extent.east_right_extent,
        "south": extent.south_bottom_extent,
        "north": extent.north_top_extent,
    }[side]


# ---------------------------------------------------------------------------
# Distributed C-grid vector halo exchange
# ---------------------------------------------------------------------------


def _cube_corner_slices(n: int):
    """The 4 corner 2x2 (row, col) slice pairs of an n*n face interior:
    TL, TR, BL, BR."""
    return (
        (slice(0, 2), slice(0, 2)),
        (slice(0, 2), slice(n - 2, n)),
        (slice(n - 2, n), slice(0, 2)),
        (slice(n - 2, n), slice(n - 2, n)),
    )


def _pack_cube_corners(interior: "torch.Tensor", n: int) -> "torch.Tensor":
    """``[*inner, n, n]`` -> ``[*inner, 4, 2, 2]``: the 4 corner 2x2 blocks.

    Pure gather (stack of deterministic slices), no FP arithmetic — carries
    exactly the cells reads (all reads land in {0,1,n-1}^2), so it is
    bitwise-equal to sending the full interior."""
    blocks = [interior[..., rs, cs] for (rs, cs) in _cube_corner_slices(n)]
    return torch.stack(blocks, dim=-3).contiguous()


def _scatter_cube_corners(
    packed: "torch.Tensor", n: int, like: "torch.Tensor"
) -> "torch.Tensor":
    """``[*inner, 4, 2, 2]`` -> ``[*inner, n, n]`` zero buffer with the 4 corner
    blocks placed at their interior positions (non-corner cells stay 0 — never
    read by , so a stray read would surface as a bitwise failure)."""
    buf = like.new_zeros(*packed.shape[:-3], n, n)
    for ci, (rs, cs) in enumerate(_cube_corner_slices(n)):
        buf[..., rs, cs] = packed[..., ci, :, :]
    return buf


def exchange_cubed_sphere_vector_distributed(
    u_local: "torch.Tensor",
    v_local: "torch.Tensor",
    *,
    grid: Grid,
    extent: HaloExtentV1,
    ctx: "CubeRankContext",
    with_signs: bool = True,
    group: "dist.ProcessGroup | None" = None,
    session: CommSession | None = None,
    constant: bool = False,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Distributed cubed-sphere C-grid vector halo exchange.

    Reproduces the single-rank oracle ``exchange_cubed_sphere_vector`` in
    face-local layout, bitwise: the scalar prefill and the seam swap (side
    strips), then the valence-3 corner fix.  The corner fix is per-owned-face
    local except for the ``OL==1`` diagonal, which reads up to three source
    faces' interior cells — same-rank read locally, cross-rank
    via direct corner P2P — and applies the oracle's own formulas, so the
    full output (including the 4 corner blocks) matches the oracle bitwise for
    both ``with_signs`` values at ``OL==1``.  (``OL>=2`` is local; not yet
    multi-rank gated.)

    Inputs ``u_local[n_local, ..., ny, nx+1]`` and ``v_local[n_local, ...,
    ny+1, nx]`` carry the redundant C-grid edge (matching the single-rank
    persistent state).  Returns ``(uc, vc)`` each ``[n_local, ..., ny+2OL,
    nx+2OL]``.

    The seam swap is purely local: the oracle reads only ``u_loc[face]`` /
    ``v_loc[face]`` (the frozen prefill snapshot of THIS face's halo), never
    a cross-face ``[other_face]`` index.  The scalar prefill is the
    only cross-rank communication and is delegated to
    ``exchange_cubed_sphere_scalar_distributed``.
    """
    session = session_for_grid(grid, session)
    n_local = len(ctx.local_face_ids)
    if u_local.shape[0] != n_local or v_local.shape[0] != n_local:
        raise ValueError(
            f"u/v local face dim must be n_local={n_local} for rank {ctx.rank}; "
            f"got u={u_local.shape[0]}, v={v_local.shape[0]}"
        )
    nx, ny = grid.nx, grid.ny
    if nx != ny:
        raise NotImplementedError("cubed-sphere vector exchange requires square faces")
    if tuple(u_local.shape[-2:]) != (ny, nx + 1):
        raise ValueError(f"u_local must be [n_local,...,ny,nx+1]; got {tuple(u_local.shape)}")
    if tuple(v_local.shape[-2:]) != (ny + 1, nx):
        raise ValueError(f"v_local must be [n_local,...,ny+1,nx]; got {tuple(v_local.shape)}")
    ol = extent.west_left_extent
    if extent != HaloExtentV1.uniform(ol) or ol < 1:
        raise NotImplementedError("vector exchange requires a uniform halo OL>=1")

    sign = -1.0 if with_signs else 1.0

    # collapse to same-size square base, scalar-prefill each component
    # via the distributed scalar exchange.  Identical math to the
    # oracle's `_exchange_cubed_sphere_scalar` per component.
    u_base = u_local[..., :, 0:nx].contiguous()
    v_base = v_local[..., 0:ny, :].contiguous()
    uc = exchange_cubed_sphere_scalar_distributed(
        u_base, grid=grid, location="T", extent=extent, ctx=ctx, group=group,
        session=session, constant=constant,
    )
    vc = exchange_cubed_sphere_scalar_distributed(
        v_base, grid=grid, location="T", extent=extent, ctx=ctx, group=group,
        session=session, constant=constant,
    )

    # Frozen snapshot of the prefilled halo -> order-independent stage 1.
    u_loc = uc.clone()
    v_loc = vc.clone()
    n = nx
    h = n + 2 * ol  # haloed width/height
    i0 = j0 = ol    # first interior index

    # per-owned-face parity-branched component swap + sign + half-cell
    # shift.  Verbatim from oracle :712-729, but the parity branch is keyed by
    # the GLOBAL face id (gf) so a rank owning a subset of faces still applies
    # the correct branch.  All reads are local (lf), no cross-face index.
    for lf, gf in enumerate(ctx.local_face_ids):
        if gf % 2 == 0:
            jn = slice(j0 + n, j0 + n + ol)  # north halo rows
            uc[lf, ..., jn, 1 : h - 1] = sign * v_loc[lf, ..., jn, 0 : h - 2]
            vc[lf, ..., jn, 0:h] = u_loc[lf, ..., jn, 0:h]
            iw = slice(0, ol)  # west halo cols
            uc[lf, ..., 0:h, iw] = v_loc[lf, ..., 0:h, iw]
            vc[lf, ..., 1:h, iw] = sign * u_loc[lf, ..., 0 : h - 1, iw]
        else:
            ie = slice(j0 + n, j0 + n + ol)  # east halo cols
            uc[lf, ..., 0:h, ie] = v_loc[lf, ..., 0:h, ie]
            vc[lf, ..., 1:h, ie] = sign * u_loc[lf, ..., 0 : h - 1, ie]
            js = slice(0, ol)  # south halo rows
            uc[lf, ..., js, 1 : h - 1] = sign * v_loc[lf, ..., js, 0 : h - 2]
            vc[lf, ..., js, 0:h] = u_loc[lf, ..., js, 0:h]

    # -----------------------------------------------------------------------
    # Valence-3 near-corner fix.
    #
    # Oracle reference: exchange.py:829-896 (all 6 faces, both parity branches,
    # ol==1 and ol>=2).  The same-face parts read only u2/v2 -> purely
    # local, no P2P.  The diagonal (ol==1 only) reads *interior* cells of up to 3
    # other (source) faces; those may live on remote ranks -> P2P required.
    #
    # Strategy (correct-by-construction):
    #   1. Clone uc/vc after to obtain u2/v2 (same as oracle).
    #   2. For ol==1: determine the set of source faces each owned receiver face
    #      needs (odd: fp,f1,f2; even: fp,f1,f2,f3).  Receive their interiors
    #      via P2P when they are on remote ranks.
    #   3. Build a 6-slot "u2_all / v2_all" index for each owned face that maps
    #      global source face id -> interior tensor [*inner, n, n].  Same-rank
    #      source faces read directly from u2/v2 interior; remote faces from P2P.
    #   4. Run , 2B, 2C verbatim from the oracle, substituting
    #      u2[src_face, ..., j, i]  ->  u2_inner[src_face][..., j-ol, i-ol]
    #      (interior offset cancels; inner layout is stage-0-post interior = u_base).
    #
    # Note: runs for BOTH with_signs values (oracle has no guard); only
    # the ``sign`` factor differs.  The oracle docstring comment claiming metric
    # fields use Stage0+1 only is inaccurate — see exchange.py:829-833.
    # -----------------------------------------------------------------------

    # +B+C frozen snapshot (mirrors oracle u2 = uc.clone()).
    u2 = uc.clone()
    v2 = vc.clone()
    n1 = j0 + n  # first east/north halo index (= ol + n)

    # ------------------------------------------------------------------
    # (ol==1): collect source-face interiors via P2P when needed.
    # ------------------------------------------------------------------
    # For each owned receiver face gf, reads source faces:
    #   odd  (gf%2==0): fp=(gf+4)%6, f1=(gf+1)%6, f2=(gf+2)%6
    #   even (gf%2==1): fp=(gf+5)%6, f1=(gf+4)%6, f2=(gf+2)%6, f3=(gf+1)%6
    #
    # "Source-face interior" in the haloed u2 frame means u2[src, ..., ol:ol+n, ol:ol+n],
    # which equals u_base[lf_src] (Stage-0 scalar fill preserves interior exactly).
    # We store received/local interiors as dict[gf_src -> tensor[*inner, n, n]].
    # When assembling oracle-verbatim index expressions:
    #   u2[src_face, ..., j, i]  (j,i in [ol..ol+n-1])
    # becomes
    #   _src_interior(src_face)[..., j - ol, i - ol]
    # which is equivalent because ol is subtracted from both j and i.

    if ol == 1:
        # Gather the set of all source faces needed by ANY owned receiver.
        needed_src_faces: set[int] = set()
        for gf in ctx.local_face_ids:
            if gf % 2 == 0:  # odd facet
                needed_src_faces.add((gf + 4) % 6)
                needed_src_faces.add((gf + 1) % 6)
                needed_src_faces.add((gf + 2) % 6)
            else:  # even facet
                needed_src_faces.add((gf + 5) % 6)
                needed_src_faces.add((gf + 4) % 6)
                needed_src_faces.add((gf + 2) % 6)
                needed_src_faces.add((gf + 1) % 6)

        # Separate into local (already owned) and remote (need P2P).
        remote_src_faces = sorted(
            gfs for gfs in needed_src_faces
            if ctx.face_to_rank[gfs] != ctx.rank
        )

        # P2P: for each remote source face, the owner sends its Stage-0-post
        # interior (= u_base[lf]) to all ranks that need it.
        # Tag namespace: 8000 + face*10 + component (0=u, 1=v).
        # We broadcast by having the owner send to each needing rank separately.
        # Round key: ('corner', src_global_face, 'u'|'v').
        corner_recv_u: dict[int, "torch.Tensor"] = {}
        corner_recv_v: dict[int, "torch.Tensor"] = {}

        # Build recv specs: only the 4 corner 2x2 blocks each source face needs
        # (all reads land in {0,1,n-1}^2), packed [*inner, 4, 2, 2].
        corner_recv_packed_u: dict[int, "torch.Tensor"] = {}
        corner_recv_packed_v: dict[int, "torch.Tensor"] = {}
        corner_recv_specs: list = []
        for gfs in remote_src_faces:
            peer = ctx.face_to_rank[gfs]
            buf_u = u_base.new_zeros(*u_base.shape[1:-2], 4, 2, 2)
            buf_v = v_base.new_zeros(*v_base.shape[1:-2], 4, 2, 2)
            corner_recv_packed_u[gfs] = buf_u
            corner_recv_packed_v[gfs] = buf_v
            tag_u = 8000 + gfs * 10 + 0
            tag_v = 8000 + gfs * 10 + 1
            rk_u = ("corner", gfs, "u")
            rk_v = ("corner", gfs, "v")
            corner_recv_specs.append((rk_u, buf_u, tag_u, peer))
            corner_recv_specs.append((rk_v, buf_v, tag_v, peer))

        # Build send specs: for each of my faces, determine which remote ranks
        # need my interior.  Iterate all 6 faces; if face gf_dst is not mine
        # and needs my face gfs as a source, I send u_base[lf_me] to gf_dst's owner.
        #
        # IMPORTANT: deduplicate by (gfs, dst_rank) to avoid sending the same
        # face data to the same peer multiple times (which would cause tag
        # collisions / deadlock when the receiver has only one recv buffer per
        # source face).  Multiple gf_dst on the same dst_rank may all need the
        # same gfs, but only ONE send is needed per (gfs, dst_rank) pair.
        corner_send_specs: list = []
        _sent: set[tuple[int, int]] = set()  # (gfs, dst_rank) already scheduled
        for gf_dst in range(_CUBED_SPHERE_FACE_COUNT):
            dst_rank = ctx.face_to_rank[gf_dst]
            if dst_rank == ctx.rank:
                continue  # dst is local; no send needed
            # Which source faces does gf_dst need?
            if gf_dst % 2 == 0:
                dst_needs = {(gf_dst + 4) % 6, (gf_dst + 1) % 6, (gf_dst + 2) % 6}
            else:
                dst_needs = {
                    (gf_dst + 5) % 6, (gf_dst + 4) % 6,
                    (gf_dst + 2) % 6, (gf_dst + 1) % 6,
                }
            for gfs in dst_needs:
                if gfs not in ctx.global_to_local:
                    continue  # not my face -> won't send
                if (gfs, dst_rank) in _sent:
                    continue  # already scheduled this (src_face, dst_rank) pair
                _sent.add((gfs, dst_rank))
                lf_me = ctx.global_to_local[gfs]
                tag_u = 8000 + gfs * 10 + 0
                tag_v = 8000 + gfs * 10 + 1
                rk_u = ("corner", gfs, "u")
                rk_v = ("corner", gfs, "v")
                send_u = _pack_cube_corners(u_base[lf_me], n)
                send_v = _pack_cube_corners(v_base[lf_me], n)
                corner_send_specs.append((rk_u, send_u, tag_u, dst_rank))
                corner_send_specs.append((rk_v, send_v, tag_v, dst_rank))

        # Deduplicate sends (same face may be needed by multiple dst ranks ->
        # multiple sends per round key; each gets a unique (tag, peer) pair).
        if corner_recv_specs or corner_send_specs:
            # Collect all active round keys.
            active_rks: set[tuple] = set()
            for rk, _b, _t, _p in corner_recv_specs:
                active_rks.add(rk)
            for rk, _b, _t, _p in corner_send_specs:
                active_rks.add(rk)
            for active_rk in sorted(active_rks):
                active_recvs = []
                send_buffers, recv_templates = [], []
                send_peers, recv_peers = [], []
                send_tags, recv_tags = [], []
                for spec in corner_recv_specs:
                    if spec[0] == active_rk:
                        active_recvs.append(spec)
                        recv_templates.append(spec[1])
                        recv_tags.append(spec[2])
                        recv_peers.append(spec[3])
                for rk, buf, tag, peer in corner_send_specs:
                    if rk == active_rk:
                        send_buffers.append(buf)
                        send_tags.append(tag)
                        send_peers.append(peer)
                received = transfer(
                    send_buffers, recv_templates, send_peers, recv_peers, send_tags, recv_tags,
                    group,
                    session=session, constant=constant,
                )
                for (rk, _buf, _tag, _peer), recv_buf in zip(active_recvs, received):
                    if rk[2] == "u":
                        corner_recv_packed_u[rk[1]] = recv_buf
                    else:
                        corner_recv_packed_v[rk[1]] = recv_buf

        # Scatter received corner blocks into full [*inner, n, n] zero buffers
        # so the read code below is byte-for-byte identical (non-corner cells
        # stay zero and are never read).
        for gfs in remote_src_faces:
            corner_recv_u[gfs] = _scatter_cube_corners(
                corner_recv_packed_u[gfs], n, u_base
            )
            corner_recv_v[gfs] = _scatter_cube_corners(
                corner_recv_packed_v[gfs], n, v_base
            )

    # ------------------------------------------------------------------
    # Helper: retrieve source-face interior tensor [*inner, n, n].
    # For oracle index u2[gf_src, ..., j, i] with j,i in [ol..ol+n-1]:
    #   -> _u2_int(gf_src)[..., j - ol, i - ol]
    # ------------------------------------------------------------------
    def _u2_int(gf_src: int) -> "torch.Tensor":
        """Stage-0-post interior of u-component for global face gf_src."""
        if gf_src in ctx.global_to_local:
            lf = ctx.global_to_local[gf_src]
            return u2[lf, ..., ol : ol + n, ol : ol + n]
        return corner_recv_u[gf_src]  # type: ignore[return-value]  # always set for ol==1 path

    def _v2_int(gf_src: int) -> "torch.Tensor":
        """Stage-0-post interior of v-component for global face gf_src."""
        if gf_src in ctx.global_to_local:
            lf = ctx.global_to_local[gf_src]
            return v2[lf, ..., ol : ol + n, ol : ol + n]
        return corner_recv_v[gf_src]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # + 2B + 2C — verbatim from oracle :839-895, substituting
    #   face  -> lf (local index into uc/vc)
    #   gf    -> global face id (parity and source-face arithmetic)
    #   u2[src_face, ..., j, i] -> _u2_int(src_face)[..., j - ol, i - ol]
    #   v2[src_face, ..., j, i] -> _v2_int(src_face)[..., j - ol, i - ol]
    #   u2[lf, ..., j, i]       -> u2[lf, ..., j, i]   (same-face: direct index)
    # ------------------------------------------------------------------
    for lf, gf in enumerate(ctx.local_face_ids):
        odd = gf % 2 == 0
        # four-corner g-loop (oracle :839-849).
        for g in range(1, ol + 1):
            if odd:
                uc[lf, ..., j0 - g, n1] = v2[lf, ..., j0, n1 + g - 1]
                vc[lf, ..., j0, i0 - g] = sign * u2[lf, ..., j0 - g, i0]
                vc[lf, ..., n1, n1 + g - 1] = sign * u2[lf, ..., n1 + g - 1, n1]
                uc[lf, ..., n1 + g - 1, i0] = v2[lf, ..., n1, i0 - g]
            else:
                uc[lf, ..., j0 - g, i0] = sign * v2[lf, ..., j0, i0 - g]
                vc[lf, ..., j0, n1 + g - 1] = u2[lf, ..., j0 - g, n1]
                uc[lf, ..., n1 + g - 1, n1] = sign * v2[lf, ..., n1, n1 + g - 1]
                vc[lf, ..., n1, i0 - g] = u2[lf, ..., n1 + g - 1, i0]

        if ol >= 2:
            # wide-vorticity stencil (oracle :850-861), parity-independent.
            uc[lf, ..., j0 - 1, i0 - 1] = v2[lf, ..., j0 - 1, i0]
            vc[lf, ..., j0 - 1, i0 - 1] = u2[lf, ..., j0, i0 - 1]
            uc[lf, ..., n1, i0 - 1] = sign * v2[lf, ..., n1 + 1, i0]
            vc[lf, ..., n1 + 1, i0 - 1] = sign * u2[lf, ..., n1 - 1, i0 - 1]
            uc[lf, ..., j0 - 1, n1 + 1] = sign * v2[lf, ..., j0 - 1, n1 - 1]
            vc[lf, ..., j0 - 1, n1] = sign * u2[lf, ..., j0, n1 + 1]
            uc[lf, ..., n1, n1 + 1] = v2[lf, ..., n1 + 1, n1 - 1]
            vc[lf, ..., n1 + 1, n1] = u2[lf, ..., n1 - 1, n1 + 1]

        elif ol == 1:
            # diagonal-corner cross-face fill (oracle :862-895).
            # Source-face reads via _u2_int/_v2_int (local or P2P-received interior).
            # Index mapping: oracle j,i in haloed frame -> interior offset j-ol, i-ol.
            # j0=ol=1, n1=ol+n=n+1; so:
            #   n1-1 = n  -> interior row/col n-1  (last)
            #   j0   = 1  -> interior row/col 0   (first)
            #   i0   = 1  -> interior col   0   (first)
            #   i0+1 = 2  -> interior col   1
            #   n1-1-ol = n-1, i0+1-ol = 1, etc.
            if odd:
                fp = (gf + 4) % 6
                f1 = (gf + 1) % 6
                f2 = (gf + 2) % 6
                # SW v: vc[face,...,j0-1,i0-1] = v2[fp,...,n1-1,n1-1]
                #   n1-1 in haloed = ol+(n-1) -> interior index n-1
                vc[lf, ..., j0 - 1, i0 - 1] = _v2_int(fp)[..., n - 1, n - 1]
                # SE v: vc[face,...,j0-1,n1]  = sign * u2[f1,...,j0,i0+1]
                #   j0->interior 0, i0+1->interior 1
                vc[lf, ..., j0 - 1, n1] = sign * _u2_int(f1)[..., 0, 1]
                # NW u: uc[face,...,n1,i0-1]  = sign * u2[f2,...,n1-1,i0+1]
                #   n1-1->interior n-1, i0+1->interior 1
                uc[lf, ..., n1, i0 - 1] = sign * _u2_int(f2)[..., n - 1, 1]
                # NE u: uc[face,...,n1,n1]    = sign * v2[f2,...,j0,i0]
                #   j0->interior 0, i0->interior 0
                uc[lf, ..., n1, n1] = sign * _v2_int(f2)[..., 0, 0]
                # NE v: vc[face,...,n1,n1]    = v2[f2,...,j0,i0]
                vc[lf, ..., n1, n1] = _v2_int(f2)[..., 0, 0]
            else:  # even facet
                fp = (gf + 5) % 6
                f1 = (gf + 4) % 6
                f2 = (gf + 2) % 6
                f3 = (gf + 1) % 6
                # SW v: vc[face,...,j0-1,i0-1] = u2[fp,...,j0,n1-1]
                #   j0->interior 0, n1-1->interior n-1
                vc[lf, ..., j0 - 1, i0 - 1] = _u2_int(fp)[..., 0, n - 1]
                # SE u: uc[face,...,j0-1,n1]   = sign * v2[f1,...,j0,n1-1]
                uc[lf, ..., j0 - 1, n1] = sign * _v2_int(f1)[..., 0, n - 1]
                # SE v: vc[face,...,j0-1,n1]   = sign * v2[f2,...,j0+1,n1-1]
                #   j0+1->interior 1, n1-1->interior n-1
                vc[lf, ..., j0 - 1, n1] = sign * _v2_int(f2)[..., 1, n - 1]
                # NW u: uc[face,...,n1,i0-1]   = sign * v2[f3,...,j0+1,i0]
                #   j0+1->interior 1, i0->interior 0
                uc[lf, ..., n1, i0 - 1] = sign * _v2_int(f3)[..., 1, 0]
                # E&S row-1: vc[face,...,j0,n1] = sign * v2[f1,...,j0,n1-1]
                vc[lf, ..., j0, n1] = sign * _v2_int(f1)[..., 0, n - 1]

    return uc, vc


def exchange_cubed_sphere_vector(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    grid: Grid,
    extent: HaloExtentV1,
    with_signs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-face C-grid (u, v) halo exchange on the cubed sphere (edge stage).

    Mirrors MITgcm ``EXCH2_UV_CGRID_3D_RX`` (``exch2_uv_cgrid_3d_rx.template``)
    stages 0-1.  Inputs are LOCOMgcm's staggered ``u[6,...,ny,nx+1]`` (u on the
    west cell edge, extra LAST column = east edge) and ``v[6,...,ny+1,nx]`` (v on
    the south edge, extra LAST row = north edge).  The redundant shared edge is
    dropped to a MITgcm-style same-size square base (``uC=u[...,:,0:nx]``,
    ``vC=v[...,0:ny,:]``) — losslessly, since the exchange reconstructs it as a
    halo column — and the result is returned as the symmetric-haloed same-size
    frame ``[6,...,ny+2OL,nx+2OL]`` (the frame ``fill_cs_corner_uv_rl`` consumes).

    It scalar-prefills ``uC`` and ``vC`` each as a tracer, then
    at every orientation-flipping seam applies the parity-branched component swap
    + sign flip + half-cell index shift, reading a frozen snapshot of the
    prefilled halo so the writes are order-independent (MITgcm's "order matters"
    constraint, template:104-106, dissolves).  ``with_signs=False`` (metric
    fields) sets the sign factor to +1.

    It then applies the valence-3 near-cube-corner fix (MITgcm
    template:236-395 mechanism 1 — distinct from the operator-time
    ``fill_cs_corner_uv_rl``) plus the single extra valid (u,v) next to each
    corner for the wide vorticity stencil.  The exact i+1/j+1 shift off-by-one
    and the corner cells are below the analytic gate's floor and are
    FP-floor-validated separately against a MITgcm cs6 dump (pending).
    """
    if u.shape[0] != _CUBED_SPHERE_FACE_COUNT or v.shape[0] != _CUBED_SPHERE_FACE_COUNT:
        raise ValueError(
            f"cubed-sphere u/v need leading face dim {_CUBED_SPHERE_FACE_COUNT}"
        )
    nx, ny = grid.nx, grid.ny
    if nx != ny:
        raise NotImplementedError("cubed-sphere vector exchange requires square faces")
    if tuple(u.shape[-2:]) != (ny, nx + 1):
        raise ValueError(f"u must be [6,...,ny,nx+1]; got {tuple(u.shape)}")
    if tuple(v.shape[-2:]) != (ny + 1, nx):
        raise ValueError(f"v must be [6,...,ny+1,nx]; got {tuple(v.shape)}")
    ol = extent.west_left_extent
    if extent != HaloExtentV1.uniform(ol) or ol < 1:
        raise NotImplementedError("vector exchange requires a uniform halo OL>=1")

    sign = -1.0 if with_signs else 1.0

    # collapse to same-size square base, scalar-prefill each component.
    u_base = u[..., :, 0:nx].contiguous()
    v_base = v[..., 0:ny, :].contiguous()
    uc = _exchange_cubed_sphere_scalar(u_base, grid=grid, location="T", extent=extent)
    vc = _exchange_cubed_sphere_scalar(v_base, grid=grid, location="T", extent=extent)

    # Frozen snapshot of the prefilled halo -> order-independent stage 1.
    u_loc = uc.clone()
    v_loc = vc.clone()
    n = nx
    h = n + 2 * ol  # haloed width/height
    i0 = j0 = ol    # first interior index

    for face in range(_CUBED_SPHERE_FACE_COUNT):
        # LOCOMgcm face % 2 == 0  <=>  MIT odd facet {1,3,5}: fix NORTH + WEST.
        # LOCOMgcm face % 2 == 1  <=>  MIT even facet {2,4,6}: fix EAST + SOUTH.
        if face % 2 == 0:
            jn = slice(j0 + n, j0 + n + ol)  # north halo rows
            # u <- -v with +1 i-shift; v <- +u (verbatim).  [template:130-142]
            uc[face, ..., jn, 1 : h - 1] = sign * v_loc[face, ..., jn, 0 : h - 2]
            vc[face, ..., jn, 0:h] = u_loc[face, ..., jn, 0:h]
            iw = slice(0, ol)  # west halo cols
            # u <- +v (verbatim); v <- -u with +1 j-shift.  [template:163-175]
            uc[face, ..., 0:h, iw] = v_loc[face, ..., 0:h, iw]
            vc[face, ..., 1:h, iw] = sign * u_loc[face, ..., 0 : h - 1, iw]
        else:
            ie = slice(j0 + n, j0 + n + ol)  # east halo cols  (i0+n .. i0+n+ol-1)
            # u <- +v (verbatim); v <- -u with +1 j-shift.  [template:182-194]
            uc[face, ..., 0:h, ie] = v_loc[face, ..., 0:h, ie]
            vc[face, ..., 1:h, ie] = sign * u_loc[face, ..., 0 : h - 1, ie]
            js = slice(0, ol)  # south halo rows
            # u <- -v with +1 i-shift; v <- +u (verbatim).  [template:215-227]
            uc[face, ..., js, 1 : h - 1] = sign * v_loc[face, ..., js, 0 : h - 2]
            vc[face, ..., js, 0:h] = u_loc[face, ..., js, 0:h]

    # valence-3 near-corner fix (MITgcm template:236-395, mechanism 1).
    # Runs for BOTH with_signs=True (velocity) AND with_signs=False (metric/coeff
    # fields); there is NO ``if with_signs`` guard.  The ``sign`` factor (-1 for
    # velocity, +1 for metrics) is the only difference between the two paths.
    u2 = uc.clone()
    v2 = vc.clone()
    n1 = j0 + n  # MIT sNy+1 / sNx+1 -> first east/north halo index
    for face in range(_CUBED_SPHERE_FACE_COUNT):
        odd = face % 2 == 0
        for g in range(1, ol + 1):
            if odd:  # MIT odd facet
                vc[face, ..., j0, i0 - g] = sign * u2[face, ..., j0 - g, i0]            # W&S [239-246]
                uc[face, ..., j0 - g, n1] = v2[face, ..., j0, n1 + g - 1]              # E&S [256-263]
                vc[face, ..., n1, n1 + g - 1] = sign * u2[face, ..., n1 + g - 1, n1]   # E&N [273-280]
                uc[face, ..., n1 + g - 1, i0] = v2[face, ..., n1, i0 - g]              # W&N [290-297]
            else:  # MIT even facet
                uc[face, ..., j0 - g, i0] = sign * v2[face, ..., j0, i0 - g]            # W&S [247-252]
                vc[face, ..., j0, n1 + g - 1] = u2[face, ..., j0 - g, n1]              # E&S [264-269]
                uc[face, ..., n1 + g - 1, n1] = sign * v2[face, ..., n1, n1 + g - 1]   # E&N [281-286]
                vc[face, ..., n1, i0 - g] = u2[face, ..., n1 + g - 1, i0]              # W&N [298-303]
        if ol >= 2:
            # one extra valid (u,v) next to each corner for the wide vorticity
            # stencil (template:329-393); parity-INDEPENDENT.  SW/NE no sign,
            # NW/SE carry the sign factor.  (needs OL>=2: writes reach i0+n+1.)
            uc[face, ..., j0 - 1, i0 - 1] = v2[face, ..., j0 - 1, i0]               # SW [329-330]
            vc[face, ..., j0 - 1, i0 - 1] = u2[face, ..., j0, i0 - 1]
            uc[face, ..., n1, i0 - 1] = sign * v2[face, ..., n1 + 1, i0]            # NW [350-351]
            vc[face, ..., n1 + 1, i0 - 1] = sign * u2[face, ..., n1 - 1, i0 - 1]
            uc[face, ..., j0 - 1, n1 + 1] = sign * v2[face, ..., j0 - 1, n1 - 1]    # SE [371-372]
            vc[face, ..., j0 - 1, n1] = sign * u2[face, ..., j0, n1 + 1]
            uc[face, ..., n1, n1 + 1] = v2[face, ..., n1 + 1, n1 - 1]               # NE [392-393]
            vc[face, ..., n1 + 1, n1] = u2[face, ..., n1 - 1, n1 + 1]
        elif ol == 1:
            # OL=1 has no room for the wide-overlap edge EXCH that, at MIT's OLx=4,
            # carries the *diagonal* (third-face) neighbour into the 4 extreme
            # corner cells (jrow/icol in {j0-1, n1}).  fill_cs_corner_uv_rl.F skips
            # them on purpose (DO i=2,OLx / DO j=2,OLy -> empty loops at OL=1) and
            # the per-face transpose above cannot reach the diagonal neighbour, so
            # 23/48 diagonal-corner velocity components are left wrong at OL=1.
            # Each source is a neighbour interior cell frozen in u2/v2, so
            # the fill is
            # order-independent and cross-face.  Overwrites the odd-face vc(n1,n1)
            # that the E&N transpose above sets wrong for faces 2/4.
            if odd:  # MIT odd facet {1,3,5}
                fp = (face + 4) % 6   # pair  (SW v)
                f1 = (face + 1) % 6   #       (SE v)
                f2 = (face + 2) % 6   #       (NW u, NE u, NE v)
                vc[face, ..., j0 - 1, i0 - 1] = v2[fp, ..., n1 - 1, n1 - 1]        # SW v
                vc[face, ..., j0 - 1, n1] = sign * u2[f1, ..., j0, i0 + 1]          # SE v
                uc[face, ..., n1, i0 - 1] = sign * u2[f2, ..., n1 - 1, i0 + 1]      # NW u
                uc[face, ..., n1, n1] = sign * v2[f2, ..., j0, i0]                  # NE u
                vc[face, ..., n1, n1] = v2[f2, ..., j0, i0]                         # NE v
            else:  # MIT even facet {2,4,6}
                fp = (face + 5) % 6   # pair  (SW v)
                f1 = (face + 4) % 6   #       (SE u)
                f2 = (face + 2) % 6   #       (SE v)
                f3 = (face + 1) % 6   #       (NW u)
                vc[face, ..., j0 - 1, i0 - 1] = u2[fp, ..., j0, n1 - 1]            # SW v
                uc[face, ..., j0 - 1, n1] = sign * v2[f1, ..., j0, n1 - 1]          # SE u
                vc[face, ..., j0 - 1, n1] = sign * v2[f2, ..., j0 + 1, n1 - 1]      # SE v
                uc[face, ..., n1, i0 - 1] = sign * v2[f3, ..., j0 + 1, i0]          # NW u
                # row-1 E&S east-halo: MIT has vc(j0,n1) == the SE diagonal u-ghost
                # uc(j0-1,n1), but the four-corner E&S above set it from the *stale*
                # pre-fix u2(j0-1,n1) -> overwrite with the same cross-face RHS as SE u.
                vc[face, ..., j0, n1] = sign * v2[f1, ..., j0, n1 - 1]              # E&S row-1

    return uc, vc


def _exchange_metric_row(
    field: torch.Tensor,
    *,
    grid: Grid,
    extent: HaloExtentV1,
    fill: FillSpec,
    obc: ObcEdges | None,
    session: CommSession | None = None,
    constant: bool = False,
) -> torch.Tensor:
    x_pad = extent.west_left_extent or extent.east_right_extent
    y_pad = extent.south_bottom_extent or extent.north_top_extent
    if x_pad and y_pad:
        raise ValueError("1-D metric rows can be padded along only one horizontal axis")
    if x_pad:
        kwargs = dict(
            axis=0,
            before=extent.west_left_extent,
            after=extent.east_right_extent,
            before_side="west",
            after_side="east",
            boundary_type=grid.boundary_topology.west_east,
            fill=fill,
            obc=obc,
        )
        if _has_distributed_transport(grid, ("west", "east")):
            return _pad_axis_distributed(field, grid=grid, location="F", **kwargs, session=session,
                                         constant=constant)
        return _pad_axis(field, grid=grid, location="F", **kwargs)
    if y_pad:
        kwargs = dict(
            axis=0,
            before=extent.south_bottom_extent,
            after=extent.north_top_extent,
            before_side="south",
            after_side="north",
            boundary_type=grid.boundary_topology.south_north,
            fill=fill,
            obc=obc,
        )
        if _has_distributed_transport(grid, ("south", "north")):
            return _pad_axis_distributed(field, grid=grid, location="F", **kwargs, session=session,
                                         constant=constant)
        return _pad_axis(field, grid=grid, location="F", **kwargs)
    return field.clone()


def _pad_axis(
    tensor: torch.Tensor,
    *,
    grid: Grid,
    location: CGridLocation,
    axis: int,
    before: int,
    after: int,
    before_side: BoundarySide,
    after_side: BoundarySide,
    boundary_type: BoundaryType,
    fill: FillSpec,
    obc: ObcEdges | None,
    x_extent: tuple[int, int] = (0, 0),
    x_boundary_type: BoundaryType | None = None,
) -> torch.Tensor:
    axis = _canonical_axis(axis, tensor.ndim)
    parts: list[torch.Tensor] = []
    if before:
        parts.append(
            _ghost_cells(
                tensor,
                grid=grid,
                location=location,
                axis=axis,
                width=before,
                side="before",
                boundary_side=before_side,
                boundary_type=boundary_type,
                fill=fill,
                obc=obc,
                x_extent=x_extent,
                x_boundary_type=x_boundary_type,
            )
        )
    parts.append(tensor)
    if after:
        parts.append(
            _ghost_cells(
                tensor,
                grid=grid,
                location=location,
                axis=axis,
                width=after,
                side="after",
                boundary_side=after_side,
                boundary_type=boundary_type,
                fill=fill,
                obc=obc,
                x_extent=x_extent,
                x_boundary_type=x_boundary_type,
            )
        )
    if len(parts) == 1:
        return tensor.clone()
    return torch.cat(parts, dim=axis)


def _pad_axis_distributed(
    tensor: torch.Tensor,
    *,
    grid: Grid,
    location: CGridLocation,
    axis: int,
    before: int,
    after: int,
    before_side: BoundarySide,
    after_side: BoundarySide,
    boundary_type: BoundaryType,
    fill: FillSpec,
    obc: ObcEdges | None,
    x_extent: tuple[int, int] = (0, 0),
    x_boundary_type: BoundaryType | None = None,
    session: CommSession | None = None,
    constant: bool = False,
) -> torch.Tensor:
    """Pad one axis, taking internal halos from neighbour ranks.

    Physical global edges still use the ordinary fill policy.  Internal
    distributed sides exchange contiguous edge strips.  For each axis, all
    receives are posted before sends to avoid deadlock across >2 ranks.
    """

    session = session_for_grid(grid, session)
    # Packed byte views erase the floating-point graph at the sender.
    if session is not None and EXCHANGE_PACKED:
        raise NotImplementedError("differentiable exchange requires the typed transport")
    if EXCHANGE_PACKED:
        return _pad_axis_distributed_packed(
            tensor,
            grid=grid,
            location=location,
            axis=axis,
            before=before,
            after=after,
            before_side=before_side,
            after_side=after_side,
            boundary_type=boundary_type,
            fill=fill,
            obc=obc,
            x_extent=x_extent,
            x_boundary_type=x_boundary_type,
        )

    axis = _canonical_axis(axis, tensor.ndim)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    marshal_bool = tensor.dtype == torch.bool
    p2p_dtype = torch.uint8 if marshal_bool else tensor.dtype
    recv_specs: list[tuple[str, torch.Tensor, int, int, bool]] = []
    send_specs: list[tuple[int, torch.Tensor, int, bool, str]] = []
    local_parts: dict[str, torch.Tensor] = {}

    before_peer = _distributed_neighbour_rank(grid, before_side)
    after_peer = _distributed_neighbour_rank(grid, after_side)
    before_wrap_peer = (
        _distributed_global_wrap_rank(grid, before_side)
        if fill is FillSpec.WRAP and before_peer is None
        else None
    )
    after_wrap_peer = (
        _distributed_global_wrap_rank(grid, after_side)
        if fill is FillSpec.WRAP and after_peer is None
        else None
    )

    for position, width, boundary_side, peer in (
        ("before", before, before_side, before_peer if before_peer is not None else before_wrap_peer),
        ("after", after, after_side, after_peer if after_peer is not None else after_wrap_peer),
    ):
        if width == 0:
            continue
        if peer is None:
            local_parts[position] = _ghost_cells(
                tensor,
                grid=grid,
                location=location,
                axis=axis,
                width=width,
                side=position,  # type: ignore[arg-type]
                boundary_side=boundary_side,
                boundary_type=boundary_type,
                fill=fill,
                obc=obc,
                x_extent=x_extent,
                x_boundary_type=x_boundary_type,
            )
            continue
        if peer == rank:
            # This rank is its own neighbour (a periodic axis it owns alone):
            # skip the duplicated seam face exactly as a send to another rank
            # does.  A global WRAP peer wraps the whole array, like the WRAP
            # sends below.
            neighbour = before_peer if position == "before" else after_peer
            local_parts[position] = _wrap_ghost(
                tensor,
                axis=axis,
                width=width,
                side=position,  # type: ignore[arg-type]
                skip_duplicate_face=neighbour is not None
                and _distributed_send_skips_duplicate_face(
                    tensor, grid=grid, location=location, axis=axis
                ),
            )
            continue
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                f"distributed exchange for side {boundary_side!r} requires "
                "torch.distributed to be initialized"
            )
        if width > tensor.shape[axis]:
            raise ValueError(
                f"distributed halo width {width} exceeds axis length {tensor.shape[axis]}"
            )
        shape = list(tensor.shape)
        shape[axis] = width
        recv = torch.empty(shape, dtype=p2p_dtype, device=tensor.device)
        source_is_wrap = (
            (position == "before" and before_peer is None and before_wrap_peer is not None)
            or (position == "after" and after_peer is None and after_wrap_peer is not None)
        )
        recv_specs.append((position, recv, _distributed_halo_tag(position, wrap=source_is_wrap), peer, source_is_wrap))

    # Send the width requested by the peer's receiving side.  With asymmetric
    # extents, our east send satisfies the peer's west halo (``before``), while
    # our west send satisfies the peer's east halo (``after``).
    if before_peer is not None and before_peer != rank and after:
        if after > tensor.shape[axis]:
            raise ValueError(
                f"distributed halo width {after} exceeds axis length {tensor.shape[axis]}"
            )
        start = _distributed_send_start(
            tensor,
            grid=grid,
            location=location,
            axis=axis,
            width=after,
            side="before",
        )
        send = tensor.narrow(axis, start, after).contiguous()
        if marshal_bool:
            send = send.to(dtype=torch.uint8)
        send_specs.append((before_peer, send, _distributed_halo_tag("after", wrap=False), False, "after"))
    if after_peer is not None and after_peer != rank and before:
        if before > tensor.shape[axis]:
            raise ValueError(
                f"distributed halo width {before} exceeds axis length {tensor.shape[axis]}"
            )
        start = _distributed_send_start(
            tensor,
            grid=grid,
            location=location,
            axis=axis,
            width=before,
            side="after",
        )
        send = tensor.narrow(axis, start, before).contiguous()
        if marshal_bool:
            send = send.to(dtype=torch.uint8)
        send_specs.append((after_peer, send, _distributed_halo_tag("before", wrap=False), False, "before"))

    if fill is FillSpec.WRAP and after and before_peer is None:
        wrap_target = _distributed_global_wrap_rank(grid, before_side)
        if (
            wrap_target is not None
            and wrap_target != rank
            and _distributed_rank_owns_global_side(grid, before_side)
        ):
            if after > tensor.shape[axis]:
                raise ValueError(
                    f"distributed halo width {after} exceeds axis length {tensor.shape[axis]}"
                )
            start = _distributed_send_start(
                tensor,
                axis=axis,
                width=after,
                side="before",
            )
            send = tensor.narrow(axis, start, after).contiguous()
            if marshal_bool:
                send = send.to(dtype=torch.uint8)
            send_specs.append((wrap_target, send, _distributed_halo_tag("after", wrap=True), True, "after"))
    if fill is FillSpec.WRAP and before and after_peer is None:
        wrap_target = _distributed_global_wrap_rank(grid, after_side)
        if (
            wrap_target is not None
            and wrap_target != rank
            and _distributed_rank_owns_global_side(grid, after_side)
        ):
            if before > tensor.shape[axis]:
                raise ValueError(
                    f"distributed halo width {before} exceeds axis length {tensor.shape[axis]}"
                )
            start = _distributed_send_start(
                tensor,
                axis=axis,
                width=before,
                side="after",
            )
            send = tensor.narrow(axis, start, before).contiguous()
            if marshal_bool:
                send = send.to(dtype=torch.uint8)
            send_specs.append((wrap_target, send, _distributed_halo_tag("before", wrap=True), True, "before"))

    if recv_specs or send_specs:
        # NCCL point-to-point does not provide MPI-style tag matching.  When a
        # two-rank periodic split uses the same peer for both before/after halos,
        # or an edge rank needs both an ordinary seam halo and a global WRAP halo
        # from the same peer, batching those messages together can pair receives
        # with the wrong send on CUDA.  ``_distributed_p2p_batches`` therefore
        # keeps ordinary and WRAP halos in separate rounds, and the two semantic
        # sides of one kind share a round only when each peer has at most one
        # matching message per direction in it; Gloo tags remain harmless.
        received_buffers = []
        for _position, recv, _tag, _peer, _is_wrap in recv_specs:
            received_buffers.append(recv)
        send_only_positions: list[str] = []
        for wrap_round, positions in _distributed_p2p_batches(recv_specs, send_specs):
            # CommSession can exit the graph during transfer. On Python 3.11
            # list comprehensions become separate frames whose iterator cannot
            # be resumed by Dynamo; keep argument construction in this frame.
            recv_indices, active_sends = [], []
            send_buffers, recv_templates = [], []
            send_peers, recv_peers = [], []
            send_tags, recv_tags = [], []
            for index, (position, recv, tag, peer, is_wrap) in enumerate(recv_specs):
                if is_wrap == wrap_round and position in positions:
                    recv_indices.append(index)
                    recv_templates.append(recv)
                    recv_tags.append(tag)
                    recv_peers.append(peer)
            for spec in send_specs:
                peer, send, tag, is_wrap, position = spec
                if is_wrap == wrap_round and position in positions:
                    active_sends.append(spec)
                    send_buffers.append(send)
                    send_tags.append(tag)
                    send_peers.append(peer)
            received = transfer(
                send_buffers, recv_templates, send_peers, recv_peers, send_tags, recv_tags,
                session=session, constant=constant,
            )
            for index, recv in zip(recv_indices, received):
                received_buffers[index] = recv
            if active_sends and not recv_indices:
                for _peer, _send, _tag, _is_wrap, position in active_sends:
                    if position not in send_only_positions:
                        send_only_positions.append(position)
        for (position, _old_recv, _tag, _peer, _is_wrap), recv in zip(
            recv_specs, received_buffers
        ):
            if marshal_bool:
                recv = recv.to(dtype=torch.bool)
            local_parts[position] = recv

        # A send-only round has no receive output to order this rank's local
        # ghost consumer. Carry the edge through the actual ghost value.
        # CommSession owns the differentiation path and keeps its native data.
        if session is None:
            for position in send_only_positions:
                ordered = inference_ghost_after_send(local_parts[position])
                if ordered is not None:
                    local_parts[position] = ordered

    parts: list[torch.Tensor] = []
    if before:
        parts.append(local_parts["before"])
    parts.append(tensor)
    if after:
        parts.append(local_parts["after"])
    return torch.cat(parts, dim=axis)


# ---------------------------------------------------------------------------
# Packed halo transport (LOCOMGCM_EXCHANGE_PACKED=1)
#
# The default path above runs one ``halo_p2p`` per kind (ordinary, WRAP), split
# into its (wrap, position) rounds when one peer serves both semantic halo
# sides, so a peer never has two in-flight messages in one direction that a
# tagless transport could pair the wrong way round.  The packed transport
# removes the ambiguity at its source instead of scheduling around it: every
# strip bound for one peer is concatenated into a *single* message, so there is
# exactly one message per peer per direction and a mispairing is not
# expressible.  Sender and receiver order the chunks inside that message by the
# same key -- the peer-independent slot ``(request, wrap, position)``, where
# ``position`` is named in the *receiver's* frame on both sides, exactly as the
# existing tags are -- so the layout is agreed without any handshake.
#
# The transported bytes are unchanged, so the unpacked strips are bit-identical
# to the ones the per-round path receives; only the number of rounds changes.
# ---------------------------------------------------------------------------

#: Chunk order inside one packed message.  ``position`` is always the receiving
#: side's own name for the halo it is filling; a sender labels its strip with
#: the position it satisfies on the peer (the ``remote_position`` it already
#: computes for the tag), so both ends sort the same sequence.
_HALO_SLOT_POSITIONS: dict[str, int] = {"before": 0, "after": 1}

#: One message per peer per direction carries no information in its tag.
_PACKED_HALO_TAG = 6320

#: Byte alignment of a per-peer message.  ``halo_p2p`` returns every receive as
#: a view into one flat buffer, so a message that did not end on an 8-byte
#: boundary would leave the *next* peer's message at an offset that cannot be
#: re-viewed as float64.  Padding each message up costs at most 7 bytes.
#:
#: Within a message, ordering the chunks by descending element size is what
#: keeps each one's own offset legal: every torch dtype's element size is a
#: power of two, so a wider element's size is a multiple of every narrower one,
#: and the running offset stays a multiple of the current chunk's element size.
#: A violation is loud rather than silent -- ``Tensor.view(dtype)`` refuses a
#: storage offset it cannot divide.
_PACK_ALIGNMENT = 8


def _halo_slot(
    request: int, axis_role: int, *, is_wrap: bool, position: str
) -> tuple[int, int, int, int]:
    """Sort key that both ends of one message compute independently.

    ``request`` indexes the epoch (0 for a lone ``exchange()``) and ``axis_role``
    separates a request's x pass from its y pass.  Neither depends on which rank
    is asking, so a sender and its receiver sort the same message the same way
    without exchanging a layout.
    """
    return (request, axis_role, int(is_wrap), _HALO_SLOT_POSITIONS[position])


def _pack_message(chunks: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate strips into one byte message, padded to ``_PACK_ALIGNMENT``."""
    raw = [chunk.reshape(-1).view(torch.uint8) for chunk in chunks]
    total = sum(part.numel() for part in raw)
    padding = -total % _PACK_ALIGNMENT
    if padding:
        raw.append(
            torch.zeros(padding, dtype=torch.uint8, device=chunks[0].device)
        )
    if len(raw) == 1:
        return raw[0]
    return torch.cat(raw)


def _message_bytes(templates: list[torch.Tensor]) -> int:
    total = sum(template.numel() * template.element_size() for template in templates)
    return total + (-total % _PACK_ALIGNMENT)


def _packed_halo_round(
    sends: list[tuple[int, tuple[int, ...], torch.Tensor]],
    recvs: list[tuple[int, tuple[int, ...], torch.Tensor]],
) -> list[torch.Tensor]:
    """Run one round in which each peer gets at most one message per direction.

    ``sends`` and ``recvs`` are ``(peer, slot, tensor)``; a send tensor is the
    strip to transmit, a receive tensor is only a template for dtype/shape.  The
    returned list is aligned with ``recvs``.

    Chunks inside a message are ordered by ``(-element_size, slot)``: the slot
    makes the order agree across ranks, and taking wider elements first keeps
    every chunk at an offset that is a multiple of its own element size, which
    is what lets the receiver view its slice back as the original dtype.

    Only integers are sorted -- the entries carry a position, and the tensors
    are fetched by that position afterwards.  Sorting tuples that *contain*
    tensors is not traceable in torch 2.4 (``builtin: sorted ListVariable``),
    and the ordering must be identical inside and outside a compiled region.
    """
    order_send: dict[int, list[tuple[int, tuple[int, ...], int]]] = {}
    for index, (peer, slot, strip) in enumerate(sends):
        entries = order_send.get(peer)
        if entries is None:
            entries = []
            order_send[peer] = entries
        entries.append((-strip.element_size(), slot, index))
    order_recv: dict[int, list[tuple[int, tuple[int, ...], int]]] = {}
    for index, (peer, slot, template) in enumerate(recvs):
        entries = order_recv.get(peer)
        if entries is None:
            entries = []
            order_recv[peer] = entries
        entries.append((-template.element_size(), slot, index))

    send_messages: list[torch.Tensor] = []
    peers_send: list[int] = []
    for peer in sorted(order_send):
        ordered = sorted(order_send[peer])
        send_messages.append(
            _pack_message([sends[index][2] for _size, _slot, index in ordered])
        )
        peers_send.append(peer)

    device = sends[0][2].device if sends else recvs[0][2].device
    recv_messages: list[torch.Tensor] = []
    peers_recv: list[int] = []
    layouts: list[list[int]] = []
    for peer in sorted(order_recv):
        ordered = sorted(order_recv[peer])
        positions = [index for _size, _slot, index in ordered]
        recv_messages.append(
            torch.empty(
                _message_bytes([recvs[index][2] for index in positions]),
                dtype=torch.uint8,
                device=device,
            )
        )
        peers_recv.append(peer)
        layouts.append(positions)

    messages = halo_p2p(
        send_messages,
        recv_messages,
        peers_send,
        peers_recv,
        [_PACKED_HALO_TAG] * len(peers_send),
        [_PACKED_HALO_TAG] * len(peers_recv),
    )

    # Start from the templates so the list is well typed, then overwrite every
    # entry.  ``filled`` is not defensive bookkeeping: an unwritten entry would
    # be an *uninitialised* ``torch.empty`` strip that reads as plausible
    # numbers, so the failure has to be made loud where it happens.
    received: list[torch.Tensor] = [template for _peer, _slot, template in recvs]
    filled = 0
    for message, positions in zip(messages, layouts):
        offset = 0
        for index in positions:
            template = recvs[index][2]
            width = template.numel() * template.element_size()
            chunk = message.narrow(0, offset, width)
            received[index] = chunk.view(template.dtype).view(template.shape)
            offset += width
            filled += 1
    if filled != len(recvs):
        raise AssertionError(
            f"packed round unpacked {filled} of {len(recvs)} receives; the "
            "remainder would have been uninitialised memory"
        )
    return received


class _HaloAxisSpecs(NamedTuple):
    """What one axis of one request needs from the network, before it runs."""

    recv_specs: list[tuple[str, torch.Tensor, int, int, bool]]
    send_specs: list[tuple[int, torch.Tensor, int, bool, str]]
    local_parts: dict[str, torch.Tensor]
    marshal_bool: bool
    axis: int


def _distributed_halo_specs(
    tensor: torch.Tensor,
    *,
    grid: Grid,
    location: CGridLocation,
    axis: int,
    before: int,
    after: int,
    before_side: BoundarySide,
    after_side: BoundarySide,
    boundary_type: BoundaryType,
    fill: FillSpec,
    obc: ObcEdges | None,
    x_extent: tuple[int, int] = (0, 0),
    x_boundary_type: BoundaryType | None = None,
    peers: tuple[int | None, int | None] | None = None,
) -> _HaloAxisSpecs:
    """Plan one axis of one exchange: which strips go where, no communication.

    Every rule here mirrors :func:`_pad_axis_distributed` statement for
    statement -- peer resolution, the physical-edge fills, the bool marshalling,
    the width checks, the C-grid seam skip, and the four send cases -- because
    the packed transport must reproduce its bytes exactly.

    ``peers`` supplies the ``(before, after)`` neighbour ranks when the caller
    has already resolved them; it exists because ``_distributed_neighbour_rank``
    probes the neighbour table with ``hasattr``, which torch 2.4's Dynamo
    refuses to trace on a dict (measured; it breaks the *default* exchange path
    under ``fullgraph=True`` too).  An epoch resolves the peers outside the
    traced region and passes them in, so nothing in the traced body inspects a
    Python container -- with the *same* resolver, so the peers cannot drift:
    ``_epoch_topology`` calls :func:`_distributed_neighbour_rank`.
    """

    axis = _canonical_axis(axis, tensor.ndim)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    marshal_bool = tensor.dtype == torch.bool
    p2p_dtype = torch.uint8 if marshal_bool else tensor.dtype
    recv_specs: list[tuple[str, torch.Tensor, int, int, bool]] = []
    send_specs: list[tuple[int, torch.Tensor, int, bool, str]] = []
    local_parts: dict[str, torch.Tensor] = {}

    if peers is None:
        before_peer = _distributed_neighbour_rank(grid, before_side)
        after_peer = _distributed_neighbour_rank(grid, after_side)
    else:
        before_peer, after_peer = peers
    before_wrap_peer = (
        _distributed_global_wrap_rank(grid, before_side)
        if fill is FillSpec.WRAP and before_peer is None
        else None
    )
    after_wrap_peer = (
        _distributed_global_wrap_rank(grid, after_side)
        if fill is FillSpec.WRAP and after_peer is None
        else None
    )

    for position, width, boundary_side, peer in (
        ("before", before, before_side, before_peer if before_peer is not None else before_wrap_peer),
        ("after", after, after_side, after_peer if after_peer is not None else after_wrap_peer),
    ):
        if width == 0:
            continue
        if peer is None:
            local_parts[position] = _ghost_cells(
                tensor,
                grid=grid,
                location=location,
                axis=axis,
                width=width,
                side=position,  # type: ignore[arg-type]
                boundary_side=boundary_side,
                boundary_type=boundary_type,
                fill=fill,
                obc=obc,
                x_extent=x_extent,
                x_boundary_type=x_boundary_type,
            )
            continue
        if peer == rank:
            # This rank is its own neighbour (a periodic axis it owns alone):
            # skip the duplicated seam face exactly as a send to another rank
            # does.  A global WRAP peer wraps the whole array, like the WRAP
            # sends below.
            neighbour = before_peer if position == "before" else after_peer
            local_parts[position] = _wrap_ghost(
                tensor,
                axis=axis,
                width=width,
                side=position,  # type: ignore[arg-type]
                skip_duplicate_face=neighbour is not None
                and _distributed_send_skips_duplicate_face(
                    tensor, grid=grid, location=location, axis=axis
                ),
            )
            continue
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                f"distributed exchange for side {boundary_side!r} requires "
                "torch.distributed to be initialized"
            )
        if width > tensor.shape[axis]:
            raise ValueError(
                f"distributed halo width {width} exceeds axis length {tensor.shape[axis]}"
            )
        shape = list(tensor.shape)
        shape[axis] = width
        recv = torch.empty(shape, dtype=p2p_dtype, device=tensor.device)
        source_is_wrap = (
            (position == "before" and before_peer is None and before_wrap_peer is not None)
            or (position == "after" and after_peer is None and after_wrap_peer is not None)
        )
        recv_specs.append((position, recv, _distributed_halo_tag(position, wrap=source_is_wrap), peer, source_is_wrap))

    if before_peer is not None and before_peer != rank and after:
        if after > tensor.shape[axis]:
            raise ValueError(
                f"distributed halo width {after} exceeds axis length {tensor.shape[axis]}"
            )
        start = _distributed_send_start(
            tensor,
            grid=grid,
            location=location,
            axis=axis,
            width=after,
            side="before",
        )
        send = tensor.narrow(axis, start, after).contiguous()
        if marshal_bool:
            send = send.to(dtype=torch.uint8)
        send_specs.append((before_peer, send, _distributed_halo_tag("after", wrap=False), False, "after"))
    if after_peer is not None and after_peer != rank and before:
        if before > tensor.shape[axis]:
            raise ValueError(
                f"distributed halo width {before} exceeds axis length {tensor.shape[axis]}"
            )
        start = _distributed_send_start(
            tensor,
            grid=grid,
            location=location,
            axis=axis,
            width=before,
            side="after",
        )
        send = tensor.narrow(axis, start, before).contiguous()
        if marshal_bool:
            send = send.to(dtype=torch.uint8)
        send_specs.append((after_peer, send, _distributed_halo_tag("before", wrap=False), False, "before"))

    if fill is FillSpec.WRAP and after and before_peer is None:
        wrap_target = _distributed_global_wrap_rank(grid, before_side)
        if (
            wrap_target is not None
            and wrap_target != rank
            and _distributed_rank_owns_global_side(grid, before_side)
        ):
            if after > tensor.shape[axis]:
                raise ValueError(
                    f"distributed halo width {after} exceeds axis length {tensor.shape[axis]}"
                )
            start = _distributed_send_start(
                tensor,
                axis=axis,
                width=after,
                side="before",
            )
            send = tensor.narrow(axis, start, after).contiguous()
            if marshal_bool:
                send = send.to(dtype=torch.uint8)
            send_specs.append((wrap_target, send, _distributed_halo_tag("after", wrap=True), True, "after"))
    if fill is FillSpec.WRAP and before and after_peer is None:
        wrap_target = _distributed_global_wrap_rank(grid, after_side)
        if (
            wrap_target is not None
            and wrap_target != rank
            and _distributed_rank_owns_global_side(grid, after_side)
        ):
            if before > tensor.shape[axis]:
                raise ValueError(
                    f"distributed halo width {before} exceeds axis length {tensor.shape[axis]}"
                )
            start = _distributed_send_start(
                tensor,
                axis=axis,
                width=before,
                side="after",
            )
            send = tensor.narrow(axis, start, before).contiguous()
            if marshal_bool:
                send = send.to(dtype=torch.uint8)
            send_specs.append((wrap_target, send, _distributed_halo_tag("before", wrap=True), True, "before"))

    return _HaloAxisSpecs(
        recv_specs=recv_specs,
        send_specs=send_specs,
        local_parts=local_parts,
        marshal_bool=marshal_bool,
        axis=axis,
    )


def _halo_slot_sends(
    specs: _HaloAxisSpecs, request: int, axis_role: int = 0
) -> list[tuple[int, tuple[int, ...], torch.Tensor]]:
    """Send entries for :func:`_packed_halo_round`, keyed by the peer's frame."""
    return [
        (
            peer,
            _halo_slot(request, axis_role, is_wrap=is_wrap, position=remote_position),
            send,
        )
        for peer, send, _tag, is_wrap, remote_position in specs.send_specs
    ]


def _halo_slot_recvs(
    specs: _HaloAxisSpecs, request: int, axis_role: int = 0
) -> list[tuple[int, tuple[int, ...], torch.Tensor]]:
    """Receive entries for :func:`_packed_halo_round`, keyed by our own frame."""
    return [
        (peer, _halo_slot(request, axis_role, is_wrap=is_wrap, position=position), recv)
        for position, recv, _tag, peer, is_wrap in specs.recv_specs
    ]


def _store_received_halos(specs: _HaloAxisSpecs, received: list[torch.Tensor]) -> None:
    """Fold received strips into ``local_parts``, undoing the bool marshalling."""
    for (position, _template, _tag, _peer, _is_wrap), recv in zip(
        specs.recv_specs, received
    ):
        if specs.marshal_bool:
            recv = recv.to(dtype=torch.bool)
        specs.local_parts[position] = recv


def _assemble_padded_axis(
    tensor: torch.Tensor, *, specs: _HaloAxisSpecs, before: int, after: int
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    if before:
        parts.append(specs.local_parts["before"])
    parts.append(tensor)
    if after:
        parts.append(specs.local_parts["after"])
    return torch.cat(parts, dim=specs.axis)


def _pad_axis_distributed_packed(
    tensor: torch.Tensor,
    *,
    grid: Grid,
    location: CGridLocation,
    axis: int,
    before: int,
    after: int,
    before_side: BoundarySide,
    after_side: BoundarySide,
    boundary_type: BoundaryType,
    fill: FillSpec,
    obc: ObcEdges | None,
    x_extent: tuple[int, int] = (0, 0),
    x_boundary_type: BoundaryType | None = None,
) -> torch.Tensor:
    """:func:`_pad_axis_distributed` with one message per peer per direction.

    Invariant: this issues at most one ``halo_p2p`` round, and inside that round
    a peer appears at most once as a sender and at most once as a receiver.  A
    tagless transport therefore has exactly one candidate for every message, so
    the mispairing the per-round path schedules around cannot occur -- with the
    same peer serving before/after, ordinary and WRAP halos, or all four at
    once.
    """
    specs = _distributed_halo_specs(
        tensor,
        grid=grid,
        location=location,
        axis=axis,
        before=before,
        after=after,
        before_side=before_side,
        after_side=after_side,
        boundary_type=boundary_type,
        fill=fill,
        obc=obc,
        x_extent=x_extent,
        x_boundary_type=x_boundary_type,
    )
    if specs.recv_specs or specs.send_specs:
        received = _packed_halo_round(
            _halo_slot_sends(specs, 0), _halo_slot_recvs(specs, 0)
        )
        _store_received_halos(specs, received)
    return _assemble_padded_axis(tensor, specs=specs, before=before, after=after)


def _distributed_p2p_rounds(
    recv_specs: list[tuple[str, torch.Tensor, int, int, bool]],
    send_specs: list[tuple[int, torch.Tensor, int, bool, str]],
) -> tuple[tuple[bool, str], ...]:
    """Return deterministic P2P rounds for tagless-safe halo exchange."""

    rounds: list[tuple[bool, str]] = []
    for wrap_round in (False, True):
        for position_round in ("before", "after"):
            has_recv = any(
                is_wrap == wrap_round and position == position_round
                for position, _recv, _tag, _peer, is_wrap in recv_specs
            )
            has_send = any(
                is_wrap == wrap_round and remote_position == position_round
                for _peer, _send, _tag, is_wrap, remote_position in send_specs
            )
            if has_recv or has_send:
                rounds.append((wrap_round, position_round))
    return tuple(rounds)


def _distributed_p2p_batches(
    recv_specs: list[tuple[str, torch.Tensor, int, int, bool]],
    send_specs: list[tuple[int, torch.Tensor, int, bool, str]],
) -> tuple[tuple[bool, tuple[str, ...]], ...]:
    """Group the rounds of :func:`_distributed_p2p_rounds` into ``halo_p2p`` calls.

    Each entry is ``(wrap, positions)``: one ``batch_isend_irecv`` carrying the
    receives and sends of those rounds.  The ``before`` and ``after`` rounds of
    one kind (ordinary or WRAP) share a call when, within it, this rank sends at
    most one message to each peer and receives at most one from each peer.  A
    tagless transport pairs the messages between two ranks in one group call by
    peer and direction, so each message then has exactly one counterpart, and
    the received bytes are those of the separate rounds.  With npx=2 and open
    edges this halves the rounds: the edge rank's ``before`` round only sent and
    its ``after`` round only received.  On a two-rank periodic axis both sides
    are the same peer, two messages would share a direction, and the rounds
    stay separate.

    The two ends of every message decide alike, so their calls pair one to one.
    Both sends of one kind go to the peers serving this rank's two sides, which
    also serve its two receives of that kind (the same neighbour, or the same
    global WRAP rank).  Two sends to one peer therefore mean every message of
    that kind is with that peer, which receives both and keeps them separate
    too; two receives mirror this.  A rank merging a kind exchanges that kind
    only with ranks that merge it.
    """

    rounds = _distributed_p2p_rounds(recv_specs, send_specs)
    batches: list[tuple[bool, tuple[str, ...]]] = []
    for wrap_round in (False, True):
        positions = tuple(position for is_wrap, position in rounds if is_wrap == wrap_round)
        if not positions:
            continue
        send_peers = [peer for peer, _send, _tag, is_wrap, _position in send_specs
                      if is_wrap == wrap_round]
        recv_peers = [peer for _position, _recv, _tag, peer, is_wrap in recv_specs
                      if is_wrap == wrap_round]
        if len(set(send_peers)) == len(send_peers) and len(set(recv_peers)) == len(recv_peers):
            batches.append((wrap_round, positions))
        else:
            batches.extend((wrap_round, (position,)) for position in positions)
    return tuple(batches)


def _distributed_send_start(
    tensor: torch.Tensor,
    *,
    axis: int,
    width: int,
    side: Literal["before", "after"],
    grid: Grid | None = None,
    location: CGridLocation | None = None,
) -> int:
    """Return send-strip start, skipping duplicated C-grid seam faces when needed."""
    n = tensor.shape[axis]
    if (
        grid is not None
        and location is not None
        and _distributed_send_skips_duplicate_face(tensor, grid=grid, location=location, axis=axis)
    ):
        start = 1 if side == "before" else n - width - 1
    else:
        start = 0 if side == "before" else n - width
    if start < 0 or start + width > n:
        raise ValueError(
            f"distributed halo width {width} cannot be sent from {location}-location "
            f"axis length {n} on side {side!r}"
        )
    return start


def _distributed_send_skips_duplicate_face(
    tensor: torch.Tensor,
    *,
    grid: Grid,
    location: CGridLocation,
    axis: int,
) -> bool:
    """Whether ``tensor`` stores the seam face twice along ``axis``.

    A U array along x (``nx + 1`` faces) or a V array along y (``ny + 1``)
    ends with the face that the next tile, or on a periodic axis this tile,
    starts with.  Send strips and the single-process periodic wrap both skip
    it; this is the one place that says when.
    """
    axis = axis % tensor.ndim          # -1 and ndim-1 are the same axis
    if location == "U" and axis == tensor.ndim - 1:
        nx = getattr(grid, "nx", None)
        if nx is None:
            raise ValueError("U-location duplicate-face test needs grid.nx")
        return tensor.shape[axis] == int(nx) + 1
    if location == "V" and axis == tensor.ndim - 2:
        ny = getattr(grid, "ny", None)
        if ny is None:
            raise ValueError("V-location duplicate-face test needs grid.ny")
        return tensor.shape[axis] == int(ny) + 1
    return False


def _has_distributed_transport(grid: Grid, sides: tuple[BoundarySide, BoundarySide]) -> bool:
    if sides == ("west", "east") and grid.boundary_topology.west_east == BoundaryType.DISTRIBUTED:
        return True
    if sides == ("south", "north") and grid.boundary_topology.south_north == BoundaryType.DISTRIBUTED:
        return True
    neighbours = getattr(grid, "distributed_neighbors", None)
    if neighbours is None:
        neighbours = getattr(grid, "distributed_neighbours", None)
    if not neighbours:
        return False
    return any(side in neighbours for side in sides)


def _distributed_halo_tag(position: str, *, wrap: bool) -> int:
    """Separate opposite halos when the same peer serves both sides."""
    base = 6310 if wrap else 6300
    return base + (1 if position == "before" else 2)


def _distributed_neighbour_rank(grid: Grid, side: BoundarySide) -> int | None:
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
    if isinstance(entry, Mapping):
        rank = entry.get("rank")
        return None if rank is None else int(rank)
    rank = getattr(entry, "rank", None)
    return None if rank is None else int(rank)


def _distributed_global_wrap_rank(grid: Grid, side: BoundarySide) -> int | None:
    """Rank owning the opposite global edge for explicit WRAP fills.

    Some MITgcm-aligned operators request ``FillSpec.WRAP`` even on regional
    open/wall topologies because MIT's default EXCH halo supplies a periodic
    value to corner-only coefficient stencils.  In a decomposed domain, a
    physical edge rank must therefore wrap to the opposite *global* edge, not
    to the opposite side of its local tile.
    """

    layout = getattr(grid, "parallel_layout", None)
    if layout is None:
        return None
    nPx = int(getattr(layout, "nPx", 1))
    nPy = int(getattr(layout, "nPy", 1))
    tile_i = int(getattr(layout, "tile_i", 0))
    tile_j = int(getattr(layout, "tile_j", 0))
    rank = int(getattr(layout, "rank", 0))

    if side == "west":
        if nPx <= 1:
            return rank
        return tile_j * nPx + (nPx - 1)
    if side == "east":
        if nPx <= 1:
            return rank
        return tile_j * nPx
    if side == "south":
        if nPy <= 1:
            return rank
        return (nPy - 1) * nPx + tile_i
    if side == "north":
        if nPy <= 1:
            return rank
        return tile_i
    return None


def _distributed_rank_owns_global_side(grid: Grid, side: BoundarySide) -> bool:
    layout = getattr(grid, "parallel_layout", None)
    if layout is None:
        return False
    nPx = int(getattr(layout, "nPx", 1))
    nPy = int(getattr(layout, "nPy", 1))
    tile_i = int(getattr(layout, "tile_i", 0))
    tile_j = int(getattr(layout, "tile_j", 0))
    if side == "west":
        return tile_i == 0
    if side == "east":
        return tile_i == nPx - 1
    if side == "south":
        return tile_j == 0
    if side == "north":
        return tile_j == nPy - 1
    return False


def _ghost_cells(
    tensor: torch.Tensor,
    *,
    grid: Grid,
    location: CGridLocation,
    axis: int,
    width: int,
    side: Literal["before", "after"],
    boundary_side: BoundarySide,
    boundary_type: BoundaryType,
    fill: FillSpec,
    obc: ObcEdges | None,
    x_extent: tuple[int, int],
    x_boundary_type: BoundaryType | None,
) -> torch.Tensor:
    if boundary_type == BoundaryType.SEAM:
        raise NotImplementedError(
            "BoundaryType.SEAM has no per-axis exchange backend; "
            "cubed-sphere grids use the CUBED_SPHERE_FACE dispatch"
        )
    if boundary_type == BoundaryType.DISTRIBUTED:
        boundary_type = BoundaryType.WALL
    if fill is FillSpec.WRAP or boundary_type == BoundaryType.PERIODIC:
        # On a periodic axis the stored seam face is face 0 again, so the halo
        # wraps the unique faces (MITgcm EXCH: u(0) = u(sNx)).  A WRAP fill on
        # a non-periodic axis keeps wrapping the whole stored array, as the
        # global WRAP sends in _pad_axis_distributed do.
        return _wrap_ghost(
            tensor,
            axis=axis,
            width=width,
            side=side,
            skip_duplicate_face=boundary_type == BoundaryType.PERIODIC
            and _distributed_send_skips_duplicate_face(
                tensor, grid=grid, location=location, axis=axis
            ),
        )
    if fill is FillSpec.ZERO:
        return _zero_ghost(tensor, axis=axis, width=width)
    if fill is FillSpec.REPLICATE:
        return _replicate_ghost(tensor, axis=axis, width=width, side=side)
    if fill is FillSpec.OBC_VALUE:
        is_open = boundary_type == BoundaryType.OPEN
        if is_open and obc is not None and boundary_side in obc:
            return _obc_ghost(
                obc[boundary_side],
                tensor=tensor,
                grid=grid,
                location=location,
                axis=axis,
                width=width,
                boundary_side=boundary_side,
                x_extent=x_extent,
                x_boundary_type=x_boundary_type,
            )
        return _replicate_ghost(tensor, axis=axis, width=width, side=side)
    raise ValueError(f"unsupported fill specification: {fill!r}")


def _wrap_ghost(
    tensor: torch.Tensor,
    *,
    axis: int,
    width: int,
    side: Literal["before", "after"],
    skip_duplicate_face: bool,
) -> torch.Tensor:
    n = tensor.shape[axis]
    if n == 0:
        raise ValueError("cannot wrap-pad an empty dimension")
    if skip_duplicate_face:
        # Faces 0 .. n-2 are unique and face n-1 repeats face 0: wrap with
        # period n-1.  Before ghosts are faces n-1-width .. n-2; after ghosts
        # follow the stored face n-1 and are faces 1 .. width.
        period = n - 1
        start = period - width if side == "before" else 1
        index = torch.arange(start, start + width, device=tensor.device).remainder(period)
        return tensor.index_select(axis, index)
    if side == "before":
        index = torch.arange(n - width, n, device=tensor.device).remainder(n)
    else:
        index = torch.arange(width, device=tensor.device).remainder(n)
    return tensor.index_select(axis, index)


def _zero_ghost(tensor: torch.Tensor, *, axis: int, width: int) -> torch.Tensor:
    shape = list(tensor.shape)
    shape[axis] = width
    return tensor.new_zeros(shape)


def _replicate_ghost(
    tensor: torch.Tensor,
    *,
    axis: int,
    width: int,
    side: Literal["before", "after"],
) -> torch.Tensor:
    source_index = 0 if side == "before" else tensor.shape[axis] - 1
    edge = tensor.narrow(axis, source_index, 1)
    shape = list(tensor.shape)
    shape[axis] = width
    return edge.expand(shape)


def _obc_ghost(
    edge_value: torch.Tensor,
    *,
    tensor: torch.Tensor,
    grid: Grid,
    location: CGridLocation,
    axis: int,
    width: int,
    boundary_side: BoundarySide,
    x_extent: tuple[int, int],
    x_boundary_type: BoundaryType | None,
) -> torch.Tensor:
    if not isinstance(edge_value, torch.Tensor):
        raise TypeError(f"obc[{boundary_side!r}] must be a torch.Tensor")
    value = edge_value.to(device=tensor.device, dtype=tensor.dtype)
    target_shape = list(tensor.shape)
    target_shape[axis] = width

    if axis == tensor.ndim - 2 and x_extent != (0, 0):
        maybe = _try_expand_edge(value, target_shape, axis)
        if maybe is not None:
            return maybe
        unpadded_x = tensor.shape[-1] - x_extent[0] - x_extent[1]
        if unpadded_x > 0 and value.ndim >= 1 and value.shape[-1] == unpadded_x:
            y_axis = value.ndim - 1 if value.ndim == tensor.ndim - 1 else value.ndim - 2
            if value.ndim == tensor.ndim - 1:
                value = value.unsqueeze(y_axis)
            value = _expand_axis(value, axis=y_axis, width=width)
            value = _pad_obc_cross_x(
                value,
                grid=grid,
                location=location,
                west=x_extent[0],
                east=x_extent[1],
                x_boundary_type=x_boundary_type,
            )
            maybe = _try_expand_edge(value, target_shape, axis)
            if maybe is not None:
                return maybe

    maybe = _try_expand_edge(value, target_shape, axis)
    if maybe is not None:
        return maybe

    raise ValueError(
        f"obc[{boundary_side!r}] with shape {tuple(edge_value.shape)} cannot fill "
        f"ghost strip shape {tuple(target_shape)}"
    )


def _pad_obc_cross_x(
    value: torch.Tensor,
    *,
    grid: Grid,
    location: CGridLocation,
    west: int,
    east: int,
    x_boundary_type: BoundaryType | None,
) -> torch.Tensor:
    if not west and not east:
        return value
    fill = FillSpec.WRAP if x_boundary_type == BoundaryType.PERIODIC else FillSpec.REPLICATE
    return _pad_axis(
        value,
        grid=grid,
        location=location,
        axis=-1,
        before=west,
        after=east,
        before_side="west",
        after_side="east",
        boundary_type=x_boundary_type or BoundaryType.WALL,
        fill=fill,
        obc=None,
    )


def _try_expand_edge(
    value: torch.Tensor,
    target_shape: Sequence[int],
    axis: int,
) -> torch.Tensor | None:
    if value.ndim == 0:
        return value.expand(target_shape)

    axis = _canonical_axis(axis, len(target_shape))
    if value.ndim == len(target_shape) - 1:
        value = value.unsqueeze(axis)
    elif value.ndim != len(target_shape):
        return None

    if extent_mismatch(value.shape[axis], target_shape[axis]):
        return None

    expand_shape = list(target_shape)
    for dim, (actual, expected) in enumerate(zip(value.shape, target_shape, strict=True)):
        if dim == axis:
            continue
        if extent_mismatch(actual, expected):
            return None
    return value.expand(expand_shape)


def _expand_axis(value: torch.Tensor, *, axis: int, width: int) -> torch.Tensor:
    axis = _canonical_axis(axis, value.ndim)
    if extent_mismatch(value.shape[axis], width):
        raise ValueError(f"cannot expand axis {axis} of shape {tuple(value.shape)} to {width}")
    shape = list(value.shape)
    shape[axis] = width
    return value.expand(shape)


def _canonical_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} out of range for ndim={ndim}")
    return axis
