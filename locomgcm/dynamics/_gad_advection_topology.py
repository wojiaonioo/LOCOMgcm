"""Topology predicates and the MIT cube tile schedule for GAD advection paths.

The ``mit_12tile`` horizontal advection runs MITgcm's three-pass cube schedule
(``pkg/generic_advdiff/gad_advection.F``, cube branch) on the rank's owned
faces, each cut into the SIZE.h tiles the grid declares (``grid.mit_tiling``).
Every tile index -- the halo-included window of a tile in its face, the update
bands of each pass, the corner-fill blocks -- is derived from
:class:`~locomgcm.geometry.tiling.CubeTileLayout`; the audited layout is MIT
cs32x15 (``sNx=32, sNy=16, OLx=OLy=4`` on the 32x32 face, two tiles per face),
and only that layout has been checked bitwise against MIT dumps.  Any other
tiling the layout admits runs the same transcription unaudited.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from locomgcm.geometry.exchange import (
    FillSpec,
    HaloExtentV1,
    exchange,
    exchange_hfac_pair,
    exchange_velocity_pair,
)
from locomgcm.geometry.tiling import CubeTileLayout, MitTiling, TileEdges
from locomgcm.operators.fluxes.tracer_flux import (
    HORIZONTAL_STENCIL_SCHEMES,
    TRACER_SCHEME_BY_NAME,
    TracerFluxScheme,
    horizontal_stencil_flux,
    horizontal_two_point_flux,
)


#: MIT's per-scheme overlap minimum ``GAD_advScheme_OlMin`` of the schemes the
#: tile schedule runs (the declared ``halo``).  The schedule is multi-dimensional
#: only, which MITgcm forces off for the Adams-Bashforth schemes.
_MIT_SCHEME_MIN_OVERLAP = {
    scheme: TRACER_SCHEME_BY_NAME[scheme].halo
    for scheme in (TracerFluxScheme.UPWIND, *HORIZONTAL_STENCIL_SCHEMES)
    if not TRACER_SCHEME_BY_NAME[scheme].adams_bashforth
    and TRACER_SCHEME_BY_NAME[scheme].halo <= 2  # ponytail: tile slices are two cells wide
}


def cube_tile_min_halo(h_scheme: TracerFluxScheme) -> int:
    """The smallest tile halo MIT accepts for ``h_scheme`` on the cube.

    ``gad_check.F:116-118``: ``minOlSize = MAX(m1+m2, m2+m3, m1*m3)`` with
    ``m1`` the scheme minimum, ``m2 = 0`` (no GM/Redi contribution here) and
    ``m3 = 2`` for cube multi-dimensional advection
    (``gad_init_fixed.F:178-180``).  Upwind needs 2, DST3 / DST3-FL need 4 --
    which is why cs32x15's SIZE.h carries ``OLx=OLy=4``.
    """
    try:
        m1 = _MIT_SCHEME_MIN_OVERLAP[h_scheme]
    except KeyError:
        raise ValueError(f"unsupported cube h_scheme={h_scheme!r}") from None
    m2, m3 = 0, 2
    return max(m1 + m2, m2 + m3, m1 * m3)


def assert_mit_12tile_runtime_shapes(
    *,
    grid: Any,
    tracer: torch.Tensor,
    u_transport: torch.Tensor,
    v_transport: torch.Tensor,
    vertical_volume_flux: torch.Tensor,
) -> None:
    """Assert the runtime state matches a declared ``mit_12tile`` grid.

    The grid-side half of the layout check (seam topology,
    24-entry connectivity, 8 cube corners, per-face rA/dxG/dyG, tiling
    derivation) lives in ``CubedSphereGrid.__post_init__``, validated once at
    construction.  What remains — and can only be checked when the state
    arrives — is that the four field TENSORS carry the face shapes the grid
    declares, so the tile windows derived from ``grid.mit_tiling`` address the
    cells they were derived for.

    A mismatch raises instead of returning ``False``.  The declaration is the authority, so a grid
    that says ``mit_12tile`` while the state says otherwise is a contradiction to
    surface, never a reason to quietly run ``six_face_approx`` (that silent
    downgrade produced a numerically different answer with no signal at all).
    """
    nz = int(getattr(grid, "nz", 0))
    nx = int(getattr(grid, "nx", 0))
    ny = int(getattr(grid, "ny", 0))
    ctx = getattr(grid, "_cube_step_ctx", None)
    n_faces = len(ctx.local_face_ids) if ctx is not None else 6
    expected = {
        "grid face_count": (int(getattr(grid, "face_count", 1)), 6),
        "grid mit_tiling declared": (getattr(grid, "mit_tiling", None) is not None, True),
        "tracer": (tuple(tracer.shape), (n_faces, nz, ny, nx)),
        "u_transport": (tuple(u_transport.shape), (n_faces, nz, ny, nx + 1)),
        "v_transport": (tuple(v_transport.shape), (n_faces, nz, ny + 1, nx)),
        "vertical_volume_flux": (
            tuple(vertical_volume_flux.shape),
            (n_faces, nz + 1, ny, nx),
        ),
    }
    mismatches = [
        f"{name}: declared mit_12tile expects {want}, got {got}"
        for name, (got, want) in expected.items()
        if got != want
    ]
    if mismatches:
        raise RuntimeError(
            "grid declares advection_topology='mit_12tile' but the runtime state "
            "does not match the declared face size / SIZE.h tiling — "
            + "; ".join(mismatches)
            + ". Fix the declaration or the state; the six_face_approx path is "
            "NOT a silent fallback."
        )


def cube_tile_edge_flags(
    tile_count: int, device: torch.device, *, layout: CubeTileLayout
) -> dict[str, torch.Tensor]:
    """Return the four face-edge masks for ``tile_count`` rank-local tiles.

    The tile layout is fixed at construction, so the schedule the step runs
    reads :class:`CubeTileTables` instead (host integers, no data-dependent
    branch).  This tensor form survives as the statement of the layout.
    """
    if tile_count % layout.tiles_per_face:
        raise ValueError(
            f"tile_count={tile_count} is not a whole number of "
            f"{layout.tiles_per_face}-tile faces"
        )
    flags = {side: [] for side in ("west", "east", "south", "north")}
    for tile in range(tile_count):
        edges = layout.tile_edges(tile % layout.tiles_per_face)
        for side in flags:
            flags[side].append(getattr(edges, side))
    return {
        side: torch.tensor(values, dtype=torch.bool, device=device)
        for side, values in flags.items()
    }


#: One write band of a schedule entry: the rank-local tiles it addresses and
#: the ``[row0, row1) x [col0, col1)`` tile-local window they receive.
WriteBand = tuple[tuple[int, ...], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class CubeTileStep:
    """One ``(ipass, direction, mode)`` entry of the MIT cube GAD schedule.

    ``tiles`` is the ascending list of rank-local tile indices this entry
    writes -- what ``cube_tile_selector`` returns as a boolean mask.  ``bands``
    are the tile-local windows the entry updates, one per distinct edge-flag
    class among its tiles, transcribed from the ``iMinUpd``/``jMinUpd`` loops
    of ``gad_advection.F``; ``corners`` are the ``(sw, se, nw, ne)`` tile tables
    the corner fill addresses (``fill_cs_corner_tr_rl.F:77-84``: a corner is
    filled only on a tile that lies on both of its face edges).

    Host tuples make schedule traversal a loop over constants rather than
    tensor-dependent branches.
    """

    ipass: int
    direction: str
    mode: str
    tiles: tuple[int, ...]
    bands: tuple[WriteBand, ...]
    corners: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    pre_corner_fill: int | None
    post_corner_fill: int | None


@dataclass(frozen=True)
class CubeTileTables:
    """The whole three-pass schedule for one rank, as host index tables.

    A pure function of ``(local_face_ids, layout)`` (:func:`cube_tile_tables`),
    both fixed for the run: the rank's face ownership and the grid's tiling do
    not change between steps.  ``steps`` is in the order the loop visits the
    schedule, so the tile slices the step writes and the order it writes them
    in are unchanged.
    """

    layout: CubeTileLayout
    tile_count: int
    local_face_ids: tuple[int, ...]
    steps: tuple[CubeTileStep, ...]


def _cube_selected_tiles(
    local_face_ids: tuple[int, ...], faces: tuple[int, ...], layout: CubeTileLayout
) -> tuple[int, ...]:
    """Every tile of every owned face whose global ID is in *faces*.

    The host form of :func:`cube_tile_selector`: same loop, same tiles per
    face, and ascending because ``enumerate`` is.  Boolean advanced indexing
    selects in ascending index order too, so a tuple built this way addresses
    the same rows in the same order as the mask it replaces.
    """
    per_face = layout.tiles_per_face
    selected: list[int] = []
    for local_face, global_face in enumerate(local_face_ids):
        if global_face in faces:
            selected.extend(range(per_face * local_face, per_face * (local_face + 1)))
    return tuple(selected)


def _x_windows(
    layout: CubeTileLayout, mode: str, edges: TileEdges
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Tile-local ``(rows, cols)`` windows one X entry updates on a tile.

    ``gad_advection.F`` ``calc_fluxes_X`` update block, in tensor indices
    (Fortran ``i`` maps to column ``i + OLx - 1``, ``j`` to row ``j + OLy - 1``):

    * overlap-only: ``iMinUpd = 1-OLx+1`` (``W_edge``: ``1``), ``iMaxUpd =
      sNx+OLx-1`` (``E_edge``: ``sNx``); rows ``1-OLy..0`` if ``S_edge`` and
      ``sNy+1..sNy+OLy`` if ``N_edge``;
    * otherwise ``i = 2-OLx..sNx+OLx-1`` on rows ``jMinUpd..jMaxUpd`` =
      ``1-OLy..sNy+OLy`` narrowed to ``1..`` / ``..sNy`` on ``interiorOnly``
      edge tiles.
    """
    ol, snx, sny = layout.halo, layout.tile_nx, layout.tile_ny
    width, height = layout.local_nx, layout.local_ny
    if mode == "overlap":
        c0 = ol if edges.west else 1
        c1 = snx + ol if edges.east else width - 1
        windows = []
        if edges.south:
            windows.append(((0, ol), (c0, c1)))
        if edges.north:
            windows.append(((sny + ol, height), (c0, c1)))
        return windows
    if mode not in ("full", "interior"):
        raise ValueError(f"unsupported x mode={mode!r}")
    r0, r1 = 0, height
    if mode == "interior":
        if edges.south:
            r0 = ol
        if edges.north:
            r1 = sny + ol
    return [((r0, r1), (1, width - 1))]


def _y_windows(
    layout: CubeTileLayout, mode: str, edges: TileEdges
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """The Y mirror of :func:`_x_windows` (``calc_fluxes_Y`` update block)."""
    ol, snx, sny = layout.halo, layout.tile_nx, layout.tile_ny
    width, height = layout.local_nx, layout.local_ny
    if mode == "overlap":
        r0 = ol if edges.south else 1
        r1 = sny + ol if edges.north else height - 1
        windows = []
        if edges.west:
            windows.append(((r0, r1), (0, ol)))
        if edges.east:
            windows.append(((r0, r1), (snx + ol, width)))
        return windows
    if mode not in ("full", "interior"):
        raise ValueError(f"unsupported y mode={mode!r}")
    c0, c1 = 0, width
    if mode == "interior":
        if edges.west:
            c0 = ol
        if edges.east:
            c1 = snx + ol
    return [((1, height - 1), (c0, c1))]


def _write_bands(
    layout: CubeTileLayout, direction: str, mode: str, tiles: tuple[int, ...]
) -> tuple[WriteBand, ...]:
    """Group *tiles* by the windows they receive; one band per (group, window)."""
    windows_of = _x_windows if direction == "x" else _y_windows
    groups: dict[tuple, list[int]] = {}
    for tile in tiles:
        edges = layout.tile_edges(tile % layout.tiles_per_face)
        key = tuple(windows_of(layout, mode, edges))
        groups.setdefault(key, []).append(tile)
    return tuple(
        (tuple(group), rows, cols)
        for windows, group in groups.items()
        for rows, cols in windows
    )


def _corner_tables(
    layout: CubeTileLayout, tiles: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """``(sw, se, nw, ne)``: the tiles of *tiles* lying on both edges of each corner."""
    sw: list[int] = []
    se: list[int] = []
    nw: list[int] = []
    ne: list[int] = []
    for tile in tiles:
        edges = layout.tile_edges(tile % layout.tiles_per_face)
        if edges.south and edges.west:
            sw.append(tile)
        if edges.south and edges.east:
            se.append(tile)
        if edges.north and edges.west:
            nw.append(tile)
        if edges.north and edges.east:
            ne.append(tile)
    return tuple(sw), tuple(se), tuple(nw), tuple(ne)


def cube_tile_tables(
    local_face_ids: tuple[int, ...], layout: CubeTileLayout
) -> CubeTileTables:
    """Resolve the MIT cube GAD schedule into host index tables for this rank.

    Pure in ``(local_face_ids, layout)``; the pass schedule is
    :func:`cube_face_modes_for_pass`. Resolve per-entry selectors and write
    bands once into Python tuples for reuse by each tracer step.

    The corner-fill modes are part of the entry because they are decided by
    ``(direction, mode, ipass)`` alone, which this function already holds.
    """
    tile_count = layout.tiles_per_face * len(local_face_ids)
    steps: list[CubeTileStep] = []
    for ipass in (1, 2, 3):
        schedule = cube_face_modes_for_pass(ipass)
        for direction in ("x", "y"):
            modes: dict[str, list[int]] = {}
            for face, spec in schedule.items():
                if direction in spec:
                    modes.setdefault(spec[direction], []).append(face)
            for mode, faces in modes.items():
                tiles = _cube_selected_tiles(local_face_ids, tuple(faces), layout)
                if direction == "x":
                    pre_fill = 1 if mode == "overlap" else None
                    post_fill = 2 if mode == "overlap" and ipass == 1 else None
                else:
                    pre_fill = 2 if mode == "overlap" else None
                    post_fill = 1 if mode == "overlap" and ipass == 1 else None
                steps.append(
                    CubeTileStep(
                        ipass=ipass,
                        direction=direction,
                        mode=mode,
                        tiles=tiles,
                        bands=_write_bands(layout, direction, mode, tiles),
                        corners=_corner_tables(layout, tiles),
                        pre_corner_fill=pre_fill,
                        post_corner_fill=post_fill,
                    )
                )
    return CubeTileTables(
        layout=layout,
        tile_count=tile_count,
        local_face_ids=tuple(local_face_ids),
        steps=tuple(steps),
    )


#: The tables memoised on their arguments, keyed by the face size, the SIZE.h
#: tiling and ``local_face_ids`` -- plain integers, so the traced step's lookup
#: is a constant dictionary read.
#:
#: A module dictionary, not ``functools.lru_cache``: Dynamo inlines *through* the
#: ``lru_cache`` wrapper and traces the builder on every step (measured -- the
#: wrapper reported ``hits=0, misses=0`` while the step ran, and the builder's
#: ``dict.setdefault`` calls showed up as graph breaks).  A plain dictionary
#: read is a traceable constant lookup.
#:
#: The memo lives here rather than on the stepper's ``CubeStaticContext`` because
#: the only route from the stepper to :func:`horizontal_gad_first_order_cube_tiles`
#: runs through ``dynamics/gad_advection.py``.  ``cube_tile_tables`` is the pure
#: function either route would call, and ``warm_cube_tile_tables`` is how the
#: eager phase fills this before the step is traced.
_CUBE_TILE_TABLES: dict[tuple[int, ...], CubeTileTables] = {}


def _tables_key(
    local_face_ids: tuple[int, ...], *, nx: int, ny: int, tiling: MitTiling
) -> tuple[int, ...]:
    return (int(nx), int(ny), int(tiling.tile_nx), int(tiling.tile_ny), int(tiling.halo),
            *(int(face) for face in local_face_ids))


def warm_cube_tile_tables(
    local_face_ids: tuple[int, ...], *, nx: int, ny: int, tiling: MitTiling
) -> CubeTileTables:
    """Build this rank's tables if absent and return them. **Eager.**

    The cube stepper calls this once per run (``build_cube_static_context``), so
    the traced step's :func:`cube_tile_tables_for` is a warm dictionary hit.
    Calling it from the traced region still works -- it is Python on host
    integers -- it just pays a cold build inside the trace, which is the
    performance regression this exists to avoid, not a wrong number.
    """
    key = _tables_key(tuple(local_face_ids), nx=nx, ny=ny, tiling=tiling)
    tables = _CUBE_TILE_TABLES.get(key)
    if tables is None:
        layout = CubeTileLayout.from_face(int(nx), int(ny), tiling)
        tables = cube_tile_tables(tuple(local_face_ids), layout)
        _CUBE_TILE_TABLES[key] = tables
    return tables


def cube_tile_tables_for(
    local_face_ids: tuple[int, ...], *, nx: int, ny: int, tiling: MitTiling
) -> CubeTileTables:
    """This rank's tables, from the memo.  Read by the traced step."""
    return warm_cube_tile_tables(local_face_ids, nx=nx, ny=ny, tiling=tiling)


def cube_to_tile_local(face_halo: torch.Tensor, layout: CubeTileLayout) -> torch.Tensor:
    """Cut each owned halo-padded face into its MIT tiles, halo included.

    ``face_halo`` is ``[n_faces, nz, ny + 2*halo, nx + 2*halo]``; the result is
    ``[n_faces * tiles_per_face, nz, tile_ny + 2*halo, tile_nx + 2*halo]``, tile
    ``t`` of local face ``f`` at row ``tiles_per_face * f + t``
    (:meth:`CubeTileLayout.face_window`).
    """
    tiles = []
    for face in range(face_halo.shape[0]):
        for tile in range(layout.tiles_per_face):
            rows, cols = layout.face_window(tile)
            tiles.append(face_halo[face, :, rows, cols])
    return torch.stack(tiles, dim=0)


def cube_from_tile_local(tile_local: torch.Tensor, layout: CubeTileLayout) -> torch.Tensor:
    """Pack each owned face's tiles back to its interior (halo dropped)."""
    per_face = layout.tiles_per_face
    if tile_local.shape[0] % per_face:
        raise ValueError(
            f"{tile_local.shape[0]} tiles is not a whole number of {per_face}-tile faces"
        )
    n_faces = tile_local.shape[0] // per_face
    ol = layout.halo
    out = tile_local.new_empty(n_faces, tile_local.shape[1], layout.ny, layout.nx)
    for face in range(n_faces):
        for tile in range(per_face):
            rows, cols = layout.interior_window(tile)
            out[face, :, rows, cols] = tile_local[
                per_face * face + tile, :, ol : ol + layout.tile_ny, ol : ol + layout.tile_nx
            ]
    return out


def fill_cube_tile_corners_tr(
    local: torch.Tensor,
    *,
    mode: int,
    corners: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    layout: CubeTileLayout,
) -> torch.Tensor:
    """Apply MIT ``FILL_CS_CORNER_TR_RL`` (``fill4dir`` 1 or 2) on local tiles.

    *corners* are the ``(sw, se, nw, ne)`` host tile tables of the entry being
    applied (:class:`CubeTileStep`): a tile appears in a corner table only when
    it lies on both face edges of that corner (``fill_cs_corner_tr_rl.F:77-84``).
    Tensor index ``[j + OLy - 1, i + OLx - 1]`` holds MIT local ``(i, j)``, so
    with ``ol = OLx = OLy``, ``sNx``, ``sNy``:

    ``fill4dir=1`` (``fill_cs_corner_tr_rl.F:160-185``, prepares X gradients)::

        SW  T(1-i,   1-j  ) = T(1-j,   i      )
        SE  T(sNx+i, 1-j  ) = T(sNx+j, i      )
        NW  T(1-i,   sNy+j) = T(1-j,   sNy+1-i)
        NE  T(sNx+i, sNy+j) = T(sNx+j, sNy+1-i)

    ``fill4dir=2`` (``:191-216``, prepares Y gradients)::

        SW  T(1-i,   1-j  ) = T(j,       1-i  )
        SE  T(sNx+i, 1-j  ) = T(sNx+1-j, 1-i  )
        NW  T(1-i,   sNy+j) = T(j,       sNy+i)
        NE  T(sNx+i, sNy+j) = T(sNx+1-j, sNy+i)

    The ``if`` is a host test on a tuple instead of ``torch.any`` on a tensor,
    which is what the schedule's 32 per-step graph breaks were.  An
    empty table writes nothing, as an all-false mask did.
    """

    if mode not in (1, 2):
        raise ValueError(f"unsupported corner-fill mode={mode!r}")
    out = local.clone()
    # Lists, not the tuples they arrive as: a tuple inside an index tuple is the
    # multi-dimensional spelling, so ``x[(0, 2), :, j, i]`` only happens to mean
    # "rows 0 and 2" here.  A list says it unambiguously.
    sw_tiles, se_tiles, nw_tiles, ne_tiles = (list(table) for table in corners)
    ol, snx, sny = layout.halo, layout.tile_nx, layout.tile_ny

    for jj in range(1, ol + 1):
        for ii in range(1, ol + 1):
            # Tensor rows/cols of the four corner cells being written.
            s_row, n_row = ol - jj, sny + ol - 1 + jj
            w_col, e_col = ol - ii, snx + ol - 1 + ii
            if mode == 1:
                if sw_tiles:
                    out[sw_tiles, :, s_row, w_col] = local[sw_tiles, :, ol - 1 + ii, ol - jj]
                if se_tiles:
                    out[se_tiles, :, s_row, e_col] = local[se_tiles, :, ol - 1 + ii, snx + ol - 1 + jj]
                if nw_tiles:
                    out[nw_tiles, :, n_row, w_col] = local[nw_tiles, :, sny + ol - ii, ol - jj]
                if ne_tiles:
                    out[ne_tiles, :, n_row, e_col] = local[ne_tiles, :, sny + ol - ii, snx + ol - 1 + jj]
            else:
                if sw_tiles:
                    out[sw_tiles, :, s_row, w_col] = local[sw_tiles, :, ol - ii, ol - 1 + jj]
                if se_tiles:
                    out[se_tiles, :, s_row, e_col] = local[se_tiles, :, ol - ii, snx + ol - jj]
                if nw_tiles:
                    out[nw_tiles, :, n_row, w_col] = local[nw_tiles, :, sny + ol - 1 + ii, ol - 1 + jj]
                if ne_tiles:
                    out[ne_tiles, :, n_row, e_col] = local[ne_tiles, :, sny + ol - 1 + ii, snx + ol - jj]
    return out


def cube_face_modes_for_pass(ipass: int) -> dict[int, dict[str, str]]:
    """MIT GAD cube pass schedule, zero-based faces.

    Source: MITgcm ``pkg/generic_advdiff/gad_advection.F`` cube branch
    (``overlapOnly``, ``interiorOnly``, ``calc_fluxes_X/Y``).
    """
    if ipass == 1:
        return {
            0: {"x": "interior"},
            1: {"x": "interior"},
            2: {"y": "overlap"},
            3: {"y": "interior"},
            4: {"y": "interior"},
            5: {"x": "overlap"},
        }
    if ipass == 2:
        return {
            0: {"y": "interior"},
            1: {"x": "overlap"},
            2: {"x": "full"},
            3: {"x": "interior"},
            4: {"y": "overlap"},
            5: {"y": "full"},
        }
    if ipass == 3:
        return {
            1: {"y": "interior"},
            2: {"y": "interior"},
            4: {"x": "interior"},
            5: {"x": "interior"},
        }
    raise ValueError(f"unsupported ipass={ipass}")


def cube_tile_selector(
    tile_count: int,
    faces: tuple[int, ...],
    device: torch.device,
    *,
    layout: CubeTileLayout,
    local_face_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Select every owned tile of the schedule's global face IDs.

    The step does not call this: the schedule reads :class:`CubeTileTables`,
    whose ``tiles`` table is the host form of this mask
    (``_cube_selected_tiles``).  It remains as the tensor statement of that
    selection.
    """
    per_face = layout.tiles_per_face
    mask = torch.zeros(tile_count, dtype=torch.bool, device=device)
    if local_face_ids is None:
        local_face_ids = tuple(range(tile_count // per_face))
    for local_face, global_face in enumerate(local_face_ids):
        if global_face in faces:
            mask[per_face * local_face : per_face * (local_face + 1)] = True
    return mask


def _stencil_af_x_cube_tile(
    flux_local: torch.Tensor,
    u_trans: torch.Tensor,
    u_vel: torch.Tensor,
    recip_dxC: torch.Tensor,
    maskW: torch.Tensor,
    *,
    layout: CubeTileLayout,
    dt: float,
    scheme: TracerFluxScheme,
) -> torch.Tensor:
    """DST3 / DST3-FL transport flux ``af`` on the tile X faces.

    Direct transcription of ``gad_dst3fl_adv_x.F`` onto the tile-local layout
    ``[n_tiles, nz, local_ny, local_nx]``.  Faces are co-located with cells: face
    ``j`` has left cell ``flux_local[...,j-1]`` and right cell
    ``flux_local[...,j]``.  The DST3 4-point stencil ``T(j-2..j+1)`` is computed
    for the interior tile faces ``j=2..local_nx-2``; the three faces ``j in {0, 1,
    local_nx-1}`` are ZEROED to mirror MIT zeroing ``uT`` at ``i in {1-OLx, 2-OLx,
    sNx+OLx}`` (gad_dst3fl_adv_x.F:42-46).  The reconstruction math itself is
    delegated to the shared stencil-flux selection ``horizontal_stencil_flux``
    (only its ``flux_dst`` return is used here; the boundary faces fall back to
    ZERO, not upwind, per the MIT zeroing).
    """
    width = layout.local_nx
    af = torch.zeros_like(flux_local)
    # j = 2..width-2 (width-3 active faces).  Slices for the 4-point stencil at
    # face j:
    #   Tm2=flux_local[j-2], Tm1=flux_local[j-1], T0=flux_local[j], Tp1=flux_local[j+1]
    Tm2 = flux_local[:, :, :, 0 : width - 3]
    Tm1 = flux_local[:, :, :, 1 : width - 2]
    T0 = flux_local[:, :, :, 2 : width - 1]
    Tp1 = flux_local[:, :, :, 3:width]
    # maskLocW(j-1), maskLocW(j), maskLocW(j+1) — U-face stagger, co-located.
    mask_im1 = maskW[:, :, :, 1 : width - 2]
    mask_i = maskW[:, :, :, 2 : width - 1]
    mask_ip1 = maskW[:, :, :, 3:width]
    u_trans_f = u_trans[:, :, :, 2 : width - 1]
    # uCFL = |uFld(j) * dt * recip_dxC(j)|  (gad_dst3fl_adv_x.F:51).
    u_cfl = (
        u_vel[:, :, :, 2 : width - 1] * float(dt) * recip_dxC[:, :, :, 2 : width - 1]
    ).abs()
    # T_left=flux_local[j-1]=Tm1, T_right=flux_local[j]=T0 (upwind donors; unused
    # here because boundary faces are zeroed, but the core requires them).
    face_masks = (mask_im1, mask_i, mask_ip1)
    _, flux_dst = horizontal_stencil_flux(
        scheme, Tm1, T0, (Tm2, Tm1, T0, Tp1), face_masks, u_trans_f, u_cfl,
        # The cube masks carry no maskInW: they are the physical masks too.
        physical_face_masks=face_masks, cell_inputs=None,
    )
    af[:, :, :, 2 : width - 1] = flux_dst
    return af


def _update_x_cube_tiles(
    local: torch.Tensor,
    tracer0: torch.Tensor,
    u_trans: torch.Tensor,
    recip_volume: torch.Tensor,
    schedule: CubeTileStep,
    *,
    layout: CubeTileLayout,
    dt: float,
    h_scheme: TracerFluxScheme = TracerFluxScheme.UPWIND,
    c_mask: torch.Tensor,
    u_vel: torch.Tensor | None = None,
    recip_dxC: torch.Tensor | None = None,
    maskW: torch.Tensor | None = None,
) -> torch.Tensor:
    """One X entry of the schedule, on the tile windows *schedule* names.

    *schedule* replaces the former ``(tile_mask, mode, edge_flags,
    pre_corner_fill, post_corner_fill)`` argument group: the same decisions,
    resolved once by :func:`cube_tile_tables` instead of per call.
    """
    width = layout.local_nx
    pre_corner_fill = schedule.pre_corner_fill
    post_corner_fill = schedule.post_corner_fill
    flux_local = (
        fill_cube_tile_corners_tr(
            local, mode=pre_corner_fill, corners=schedule.corners, layout=layout,
        )
        if pre_corner_fill is not None
        else local
    )
    af = torch.zeros_like(local)
    if h_scheme in HORIZONTAL_STENCIL_SCHEMES:
        af = _stencil_af_x_cube_tile(
            flux_local,
            u_trans,
            u_vel,
            recip_dxC,
            maskW,
            layout=layout,
            dt=dt,
            scheme=h_scheme,
        )
    else:
        u = u_trans[:, :, :, 1:width]
        af[:, :, :, 1:width] = horizontal_two_point_flux(
            h_scheme, flux_local[:, :, :, 0 : width - 1], flux_local[:, :, :, 1:width], u, u, None,
        )
    delta = (
        af[:, :, :, 1:width]
        - af[:, :, :, 0 : width - 1]
        - tracer0[:, :, :, 0 : width - 1]
        * (u_trans[:, :, :, 1:width] - u_trans[:, :, :, 0 : width - 1])
    )
    update = (
        -float(dt) * delta * recip_volume[:, :, :, 0 : width - 1] * c_mask[:, :, :, 0 : width - 1]
    )

    target = torch.zeros_like(local, dtype=torch.bool)
    for tiles, (r0, r1), (c0, c1) in schedule.bands:
        target[list(tiles), :, r0:r1, c0:c1] = True

    base_local = (
        fill_cube_tile_corners_tr(
            flux_local, mode=post_corner_fill, corners=schedule.corners, layout=layout,
        )
        if post_corner_fill is not None
        else flux_local
    )
    out = base_local.clone()
    out[:, :, :, 0 : width - 1] = torch.where(
        target[:, :, :, 0 : width - 1],
        base_local[:, :, :, 0 : width - 1] + update,
        base_local[:, :, :, 0 : width - 1],
    )
    return out


def _stencil_af_y_cube_tile(
    flux_local: torch.Tensor,
    v_trans: torch.Tensor,
    v_vel: torch.Tensor,
    recip_dyC: torch.Tensor,
    maskS: torch.Tensor,
    *,
    layout: CubeTileLayout,
    dt: float,
    scheme: TracerFluxScheme,
) -> torch.Tensor:
    """DST3 / DST3-FL transport flux ``af`` on the tile Y faces.

    Y mirror of :func:`_stencil_af_x_cube_tile` (``gad_dst3fl_adv_y.F``).  Faces
    lie along axis ``-2``: face ``j`` has south cell ``flux_local[...,j-1,:]``
    and north cell ``flux_local[...,j,:]``.  The DST3 stencil is computed for
    the interior tile faces ``j=2..local_ny-2``; faces ``j in {0, 1, local_ny-1}``
    are ZEROED (MIT zeroes ``vT`` at ``j in {1-OLy, 2-OLy, sNy+OLy}``,
    gad_dst3fl_adv_y.F:38-42).  Reconstruction delegated to the shared core.
    """
    height = layout.local_ny
    af = torch.zeros_like(flux_local)
    Tm2 = flux_local[:, :, 0 : height - 3, :]
    Tm1 = flux_local[:, :, 1 : height - 2, :]
    T0 = flux_local[:, :, 2 : height - 1, :]
    Tp1 = flux_local[:, :, 3:height, :]
    mask_jm1 = maskS[:, :, 1 : height - 2, :]
    mask_j = maskS[:, :, 2 : height - 1, :]
    mask_jp1 = maskS[:, :, 3:height, :]
    v_trans_f = v_trans[:, :, 2 : height - 1, :]
    v_cfl = (
        v_vel[:, :, 2 : height - 1, :] * float(dt) * recip_dyC[:, :, 2 : height - 1, :]
    ).abs()
    face_masks = (mask_jm1, mask_j, mask_jp1)
    _, flux_dst = horizontal_stencil_flux(
        scheme, Tm1, T0, (Tm2, Tm1, T0, Tp1), face_masks, v_trans_f, v_cfl,
        physical_face_masks=face_masks, cell_inputs=None,
    )
    af[:, :, 2 : height - 1, :] = flux_dst
    return af


def _update_y_cube_tiles(
    local: torch.Tensor,
    tracer0: torch.Tensor,
    v_trans: torch.Tensor,
    recip_volume: torch.Tensor,
    schedule: CubeTileStep,
    *,
    layout: CubeTileLayout,
    dt: float,
    h_scheme: TracerFluxScheme = TracerFluxScheme.UPWIND,
    c_mask: torch.Tensor,
    v_vel: torch.Tensor | None = None,
    recip_dyC: torch.Tensor | None = None,
    maskS: torch.Tensor | None = None,
) -> torch.Tensor:
    """The Y mirror of :func:`_update_x_cube_tiles`, on the same schedule entry."""
    height = layout.local_ny
    pre_corner_fill = schedule.pre_corner_fill
    post_corner_fill = schedule.post_corner_fill
    flux_local = (
        fill_cube_tile_corners_tr(
            local, mode=pre_corner_fill, corners=schedule.corners, layout=layout,
        )
        if pre_corner_fill is not None
        else local
    )
    af = torch.zeros_like(local)
    if h_scheme in HORIZONTAL_STENCIL_SCHEMES:
        af = _stencil_af_y_cube_tile(
            flux_local,
            v_trans,
            v_vel,
            recip_dyC,
            maskS,
            layout=layout,
            dt=dt,
            scheme=h_scheme,
        )
    else:
        v = v_trans[:, :, 1:height, :]
        af[:, :, 1:height, :] = horizontal_two_point_flux(
            h_scheme, flux_local[:, :, 0 : height - 1, :], flux_local[:, :, 1:height, :], v, v, None,
        )
    delta = (
        af[:, :, 1:height, :]
        - af[:, :, 0 : height - 1, :]
        - tracer0[:, :, 0 : height - 1, :]
        * (v_trans[:, :, 1:height, :] - v_trans[:, :, 0 : height - 1, :])
    )
    update = (
        -float(dt) * delta * recip_volume[:, :, 0 : height - 1, :] * c_mask[:, :, 0 : height - 1, :]
    )

    target = torch.zeros_like(local, dtype=torch.bool)
    for tiles, (r0, r1), (c0, c1) in schedule.bands:
        target[list(tiles), :, r0:r1, c0:c1] = True

    base_local = (
        fill_cube_tile_corners_tr(
            flux_local, mode=post_corner_fill, corners=schedule.corners, layout=layout,
        )
        if post_corner_fill is not None
        else flux_local
    )
    out = base_local.clone()
    out[:, :, 0 : height - 1, :] = torch.where(
        target[:, :, 0 : height - 1, :],
        base_local[:, :, 0 : height - 1, :] + update,
        base_local[:, :, 0 : height - 1, :],
    )
    return out


def horizontal_gad_first_order_cube_tiles(
    *,
    tracer: torch.Tensor,
    u_transport: torch.Tensor,
    v_transport: torch.Tensor,
    grid: object,
    dt: float,
    h_scheme: TracerFluxScheme = TracerFluxScheme.UPWIND,
    u_vel: torch.Tensor | None = None,
    v_vel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run MIT's cube tile horizontal GAD schedule on owned faces.

    The tile layout comes from ``grid.mit_tiling`` and the face size
    (:class:`CubeTileLayout`); the audited instance is cs32x15's 12-tile /
    OL=4 layout.  ``h_scheme`` selects the per-face flux reconstruction.
    UPWIND (default) is the audited 1st-order MIT path (bitwise no-op vs the
    pre-scheme code).  DST3 / DST3_FL run the shared geometry-neutral DST3 core
    on the tile faces (``gad_dst3fl_adv_{x,y}.F``); these require the face
    VELOCITIES (``u_vel``, ``v_vel``) for the CFL number, and exchange the face
    metrics ``recip_dxC`` / ``recip_dyC`` and the U/V-face masks ``maskW`` /
    ``maskS`` onto the same tile layout.  Existing cube exchanges supply the
    tiling's halo across ranks.  Each rank keeps every tile of each owned face;
    global face IDs select the original three-pass schedule and all tile
    arithmetic remains local.

    The schedule itself is :func:`cube_tile_tables_for`: host index tables,
    resolved once per rank from ``local_face_ids`` and the layout. Reusing
    these entries avoids per-step selector allocation and tensor-dependent
    branches while preserving schedule order and slices.
    """
    tiling = getattr(grid, "mit_tiling", None)
    if tiling is None:
        raise ValueError(
            "the mit_12tile schedule needs grid.mit_tiling (the MIT SIZE.h "
            "sNx/sNy/OLx block declaration); the grid carries none"
        )
    min_halo = cube_tile_min_halo(h_scheme)
    if tiling.halo < min_halo:
        raise ValueError(
            f"mit_12tile with h_scheme={h_scheme!r} needs a tile halo of at "
            f"least {min_halo} (MIT gad_check.F minOlSize); grid.mit_tiling is "
            f"{tiling!r}"
        )
    ctx = getattr(grid, "_cube_step_ctx", None)
    local_face_ids = ctx.local_face_ids if ctx is not None else tuple(range(6))
    n_faces = len(local_face_ids)
    tables = cube_tile_tables_for(
        tuple(local_face_ids), nx=grid.nx, ny=grid.ny, tiling=tiling
    )
    layout = tables.layout
    extent = HaloExtentV1.uniform(layout.halo)
    nz = grid.nz
    drf = grid.layer_thickness_for(u_transport.dtype, u_transport.device).view(1, nz, 1, 1)
    hfac = grid.masks.hFacC.to(dtype=u_transport.dtype, device=u_transport.device)
    c_mask = (hfac > 0).to(u_transport.dtype)
    r_a = grid.metrics.cell_area_for(u_transport.dtype, u_transport.device).unsqueeze(1)
    volume = r_a * drf * hfac

    tracer_halo = exchange(tracer, grid=grid, location="T", extent=extent, fill=FillSpec.ZERO)
    volume_halo = exchange(volume, grid=grid, location="T", extent=extent, fill=FillSpec.ZERO)
    cmask_halo = (exchange(c_mask, grid=grid, location="T", extent=extent, fill=FillSpec.ZERO) > 0).to(u_transport.dtype)
    u_halo, v_halo = exchange_velocity_pair(
        u_transport,
        v_transport,
        grid=grid,
        extent=extent,
        with_signs=True,
    )

    local = cube_to_tile_local(tracer_halo, layout)
    tracer_tile0 = cube_to_tile_local(tracer_halo, layout)
    volume_tile = cube_to_tile_local(volume_halo, layout)
    wet_volume = volume_tile > 0
    safe_volume = torch.where(wet_volume, volume_tile, torch.ones_like(volume_tile))
    recip_volume = torch.where(wet_volume, 1.0 / safe_volume, torch.zeros_like(volume_tile))
    c_mask_tile = cube_to_tile_local(cmask_halo, layout)
    u_tile = cube_to_tile_local(u_halo, layout)
    v_tile = cube_to_tile_local(v_halo, layout)

    # DST3 tile-layout metrics/masks (threaded only when the scheme needs them).
    # The face VELOCITIES rotate across seams exactly like the transports, so they
    # use the same signed vector exchange.  The U/V-face metrics dxC/dyC and masks
    # maskW/maskS swap stagger across seams without a sign flip -> the hfac-paired
    # exchange (``exchange_cubed_sphere_vector`` with ``with_signs=False``).  The
    # reciprocal of dxC/dyC is taken AFTER exchange (the cube halo carries true
    # positive distances; the valence-3 corner blocks are filled by the vector
    # swap, so no zeros leak into the interior DST3 stencil faces).
    u_vel_tile = v_vel_tile = None
    recip_dxC_tile = recip_dyC_tile = None
    maskW_tile = maskS_tile = None
    if h_scheme in HORIZONTAL_STENCIL_SCHEMES:
        if u_vel is None or v_vel is None:
            raise ValueError(
                "cube DST3/DST3_FL tracer advection requires u_vel and v_vel "
                "(face velocities) for the CFL number"
            )
        u_vel_halo, v_vel_halo = exchange_velocity_pair(
            u_vel, v_vel, grid=grid, extent=extent, with_signs=True,
        )
        u_vel_tile = cube_to_tile_local(u_vel_halo, layout)
        v_vel_tile = cube_to_tile_local(v_vel_halo, layout)

        dxC = grid.metrics.dxC_for(u_transport.dtype, u_transport.device)  # [6, ny, nx+1]
        dyC = grid.metrics.dyC_for(u_transport.dtype, u_transport.device)  # [6, ny+1, nx]
        dxC3d = dxC.unsqueeze(1).expand(n_faces, nz, dxC.shape[-2], dxC.shape[-1]).contiguous()
        dyC3d = dyC.unsqueeze(1).expand(n_faces, nz, dyC.shape[-2], dyC.shape[-1]).contiguous()
        dxC_halo, dyC_halo = exchange_hfac_pair(dxC3d, dyC3d, grid=grid, extent=extent)
        dxC_tile = cube_to_tile_local(dxC_halo, layout)
        dyC_tile = cube_to_tile_local(dyC_halo, layout)
        recip_dxC_tile = torch.where(
            dxC_tile > 0, 1.0 / dxC_tile, torch.zeros_like(dxC_tile)
        )
        recip_dyC_tile = torch.where(
            dyC_tile > 0, 1.0 / dyC_tile, torch.zeros_like(dyC_tile)
        )

        hFacW = grid.masks.hFacW.to(dtype=u_transport.dtype, device=u_transport.device)
        hFacS = grid.masks.hFacS.to(dtype=u_transport.dtype, device=u_transport.device)
        maskW = (hFacW > 0).to(u_transport.dtype)
        maskS = (hFacS > 0).to(u_transport.dtype)
        maskW_halo, maskS_halo = exchange_hfac_pair(maskW, maskS, grid=grid, extent=extent)
        maskW_tile = (cube_to_tile_local(maskW_halo, layout) > 0).to(u_transport.dtype)
        maskS_tile = (cube_to_tile_local(maskS_halo, layout) > 0).to(u_transport.dtype)

    for entry in tables.steps:
        if entry.direction == "x":
            local = _update_x_cube_tiles(
                local,
                tracer_tile0,
                u_tile,
                recip_volume,
                entry,
                layout=layout,
                dt=dt,
                h_scheme=h_scheme,
                c_mask=c_mask_tile,
                u_vel=u_vel_tile,
                recip_dxC=recip_dxC_tile,
                maskW=maskW_tile,
            )
        else:
            local = _update_y_cube_tiles(
                local,
                tracer_tile0,
                v_tile,
                recip_volume,
                entry,
                layout=layout,
                dt=dt,
                h_scheme=h_scheme,
                c_mask=c_mask_tile,
                v_vel=v_vel_tile,
                recip_dyC=recip_dyC_tile,
                maskS=maskS_tile,
            )
    return cube_from_tile_local(local, layout)
