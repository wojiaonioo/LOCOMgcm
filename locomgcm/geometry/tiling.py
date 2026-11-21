"""MIT SIZE.h numerical tiling declaration, and the face layout derived from it.

This module carries the *numerical-layer* block declaration only.  It is
deliberately separate from the execution-layer rank decomposition
(``cube_decomposition.CubeDecomposition``): the two carry invariants that point
in opposite directions (a tiling subdivides one face into MIT ``sNx``/``sNy``
blocks with a halo; a decomposition assigns whole faces to ranks), so they are
never merged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MitTiling:
    """MIT SIZE.h numerical tiling declaration (sNx/sNy/OLx).

    The execution-level rank decomposition is
    cube_decomposition.CubeDecomposition; the two have opposite invariants and
    are never merged."""

    tile_nx: int
    tile_ny: int
    halo: int

    def __post_init__(self) -> None:
        if self.tile_nx <= 0 or self.tile_ny <= 0 or self.halo <= 0:
            raise ValueError(
                "MitTiling requires tile_nx>0, tile_ny>0, halo>0; got "
                f"tile_nx={self.tile_nx}, tile_ny={self.tile_ny}, halo={self.halo}"
            )


#: The overlap the MIT cube multi-dimensional advection needs regardless of the
#: tracer scheme: ``gad_init_fixed.F:178-180`` sets ``GAD_OlMinSize(3)=2`` when
#: ``useCubedSphereExchange .AND. useMultiDimAdvec``, and ``gad_check.F:116-118``
#: rejects ``OLx``/``OLy`` below ``MAX(m1+m2, m2+m3, m1*m3)``, which is ``2`` for
#: the least demanding scheme (``m1=1``, ``m2=0``, ``m3=2``).  The scheme-specific
#: bound is checked where the scheme is known
#: (``dynamics._gad_advection_topology.cube_tile_min_halo``).
MIT_CUBE_MULTIDIM_MIN_HALO = 2


@dataclass(frozen=True)
class TileEdges:
    """Which sides of a tile lie on the edge of its cube face.

    The MIT cube GAD schedule decides every per-tile range from these four
    flags (``gad_advection.F`` ``W_edge``/``E_edge``/``S_edge``/``N_edge``, from
    exch2's ``isWedge`` etc.), and the corner fill from their pairwise
    conjunctions (``fill_cs_corner_tr_rl.F:77-84``).
    """

    west: bool
    east: bool
    south: bool
    north: bool


@dataclass(frozen=True)
class CubeTileLayout:
    """One cube face cut into ``tile_nx x tile_ny`` MIT tiles with a ``halo``.

    Host integers only.  Tile ``t`` of a face has origin
    ``(tx, ty) = (t % tiles_x, t // tiles_x)`` -- x fastest, the order exch2
    numbers the tiles of one facet -- and its rank-local index is
    ``tiles_per_face * local_face + t``.  Every index the tile schedule needs is
    derived here from the face size and the SIZE.h declaration; the audited MIT
    cs32x15 layout (``sNx=32, sNy=16, OLx=OLy=4`` on the 32x32 face) is the
    ``tiles_x=1, tiles_y=2`` instance.
    """

    nx: int
    ny: int
    tile_nx: int
    tile_ny: int
    halo: int

    @classmethod
    def from_face(cls, nx: int, ny: int, tiling: MitTiling) -> "CubeTileLayout":
        """Derive the layout, rejecting a declaration the face cannot carry.

        Every rejection raises ``ValueError`` naming the reason: exact tiling of
        the face (MIT ``SIZE.h`` ``nSx*sNx`` must equal the facet extent), the
        MIT multi-dim minimum overlap, and a halo no wider than the tile.  The
        last is MIT's own rule -- ``config_check.F:608-617`` stops on
        ``sNx.LT.OLx`` ("not allowed with ALLOW_EXCH2 defined"; likewise
        ``sNy``) because a tile's halo is served by its edge-adjacent tiles
        and no further -- and the transcription inherits it: the update bands
        reach ``OLx-1`` cells into the halo with the DST3 boundary fluxes
        zeroed at ``2-OLx``, one tile away at most.  Beyond it tiling
        invariance breaks (measured: ``MitTiling(2, 2, 4)`` under DST3-FL
        differs from the one-tile layout at one cell).  The bound
        also keeps the halo within the face, which the face-level exchange
        needs (``exchange.py`` ``_exchange_cubed_sphere_scalar`` fills a halo
        from one neighbour face).
        """
        if nx % tiling.tile_nx != 0 or ny % tiling.tile_ny != 0:
            raise ValueError(
                f"MitTiling {tiling!r} does not tile the {nx}x{ny} face exactly"
            )
        if tiling.halo < MIT_CUBE_MULTIDIM_MIN_HALO:
            raise ValueError(
                f"MitTiling {tiling!r} halo is below the MIT cube multi-dim "
                f"advection minimum OLx=OLy={MIT_CUBE_MULTIDIM_MIN_HALO} "
                "(gad_init_fixed.F:178-180 GAD_OlMinSize(3)=2; gad_check.F:116-118)"
            )
        if tiling.halo > min(tiling.tile_nx, tiling.tile_ny):
            raise ValueError(
                f"MitTiling {tiling!r} halo exceeds the tile (MIT config_check.F: "
                "sNx<OLx not allowed with exch2); a tile's halo is served by its "
                "edge-adjacent tiles only"
            )
        return cls(
            nx=nx, ny=ny, tile_nx=tiling.tile_nx, tile_ny=tiling.tile_ny, halo=tiling.halo
        )

    @property
    def tiling(self) -> MitTiling:
        return MitTiling(self.tile_nx, self.tile_ny, self.halo)

    @property
    def tiles_x(self) -> int:
        return self.nx // self.tile_nx

    @property
    def tiles_y(self) -> int:
        return self.ny // self.tile_ny

    @property
    def tiles_per_face(self) -> int:
        return self.tiles_x * self.tiles_y

    @property
    def local_nx(self) -> int:
        """Tile-local extent in x, halo included (MIT ``1-OLx:sNx+OLx``)."""
        return self.tile_nx + 2 * self.halo

    @property
    def local_ny(self) -> int:
        """Tile-local extent in y, halo included (MIT ``1-OLy:sNy+OLy``)."""
        return self.tile_ny + 2 * self.halo

    def tile_origin(self, tile: int) -> tuple[int, int]:
        """``(tx, ty)`` block position of tile ``tile`` within its face."""
        if not 0 <= tile < self.tiles_per_face:
            raise ValueError(
                f"tile {tile} is outside the {self.tiles_per_face} tiles of a face"
            )
        return tile % self.tiles_x, tile // self.tiles_x

    def tile_edges(self, tile: int) -> TileEdges:
        """The four face-edge flags of tile ``tile`` (face-local index)."""
        tx, ty = self.tile_origin(tile)
        return TileEdges(
            west=tx == 0,
            east=tx == self.tiles_x - 1,
            south=ty == 0,
            north=ty == self.tiles_y - 1,
        )

    def face_window(self, tile: int) -> tuple[slice, slice]:
        """``(rows, cols)`` of the tile's halo-included block in the halo-padded face.

        The padded face is ``[ny + 2*halo, nx + 2*halo]`` with the interior at
        offset ``halo``; the tile's own interior starts at ``(ty*tile_ny,
        tx*tile_nx)`` there, so its halo-included window starts ``halo``
        earlier -- exactly at ``(ty*tile_ny, tx*tile_nx)`` in padded indices.
        """
        tx, ty = self.tile_origin(tile)
        r0, c0 = ty * self.tile_ny, tx * self.tile_nx
        return slice(r0, r0 + self.local_ny), slice(c0, c0 + self.local_nx)

    def interior_window(self, tile: int) -> tuple[slice, slice]:
        """``(rows, cols)`` of the tile's interior in the unpadded face."""
        tx, ty = self.tile_origin(tile)
        r0, c0 = ty * self.tile_ny, tx * self.tile_nx
        return slice(r0, r0 + self.tile_ny), slice(c0, c0 + self.tile_nx)


__all__ = ["CubeTileLayout", "MIT_CUBE_MULTIDIM_MIN_HALO", "MitTiling", "TileEdges"]
