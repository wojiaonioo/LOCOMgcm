"""Cube face -> rank decomposition schema.

The connectivity table (``build_tile_connectivity``) is pure, N-independent
topology: ``(face, side) -> (neighbour_face, neighbour_side, rotation)``. This
module adds the *decomposition* half of the "schema v2" the multi-GPU work
needs, WITHOUT touching that topology table or its 24 consumers:

  1. N parameterization — carried as ``n`` (faces are faces for any CSN; the
     topology is N-free, so decomposition is too).
  2. neighbour_rank + the canonical face->rank table for R in {1,2,3,6}
     (fixed assignment, face-aligned: whole faces never split).
  3. global face id <-> local face index remap (used by the P2P exchange).

This is metadata only — it changes NO numerics. The single-rank case
(``n_ranks=1``) is the degenerate identity; the multi-rank tables drive the
distributed P2P exchange.

Structure is fixed ONCE here: downstream code reads these helpers; it does not
redefine the assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch
import torch.distributed as dist

FACE_COUNT = 6
_SIDES = ("north", "south", "east", "west")

# Canonical face->rank assignment (face-aligned; whole faces only).
#   R=1: all faces on rank 0 (degenerate identity used today).
#   R=2: {0,1,2}->0, {3,4,5}->1.
#   R=3: each rank 2 faces -> {0,1}->0, {2,3}->1, {4,5}->2.
#   R=6: each face its own rank.
_D4_FACE_TO_RANK: Mapping[int, tuple[int, ...]] = {
    1: (0, 0, 0, 0, 0, 0),
    2: (0, 0, 0, 1, 1, 1),
    3: (0, 0, 1, 1, 2, 2),
    6: (0, 1, 2, 3, 4, 5),
}

SUPPORTED_RANKS = tuple(sorted(_D4_FACE_TO_RANK))


@dataclass(frozen=True)
class CubeDecomposition:
    """Face->rank decomposition for an N-cube over ``n_ranks`` ranks."""

    n: int                       # CSN face size (faces per side)
    n_ranks: int                 # in SUPPORTED_RANKS
    face_to_rank: tuple[int, ...]  # len 6: owning rank of each global face

    @classmethod
    def for_ranks(cls, n_ranks: int, *, n: int = 32) -> "CubeDecomposition":
        if n_ranks not in _D4_FACE_TO_RANK:
            raise ValueError(
                f"n_ranks={n_ranks} not in supported {SUPPORTED_RANKS} (face-aligned only)"
            )
        if n <= 0:
            raise ValueError(f"n (face size) must be positive, got {n}")
        dec = cls(n=n, n_ranks=n_ranks, face_to_rank=_D4_FACE_TO_RANK[n_ranks])
        dec.validate()
        return dec

    def local_faces(self, rank: int) -> tuple[int, ...]:
        """Global face ids owned by ``rank``, in ascending global order."""
        return tuple(f for f in range(FACE_COUNT) if self.face_to_rank[f] == rank)

    def n_local_faces(self, rank: int) -> int:
        return len(self.local_faces(rank))

    def global_to_local(self, rank: int, global_face: int) -> int:
        """Local index of ``global_face`` within ``rank``'s owned faces."""
        locals_ = self.local_faces(rank)
        if global_face not in locals_:
            raise ValueError(f"global face {global_face} not owned by rank {rank}")
        return locals_.index(global_face)

    def local_to_global(self, rank: int, local_face: int) -> int:
        """Global face id of ``rank``'s ``local_face``-th owned face."""
        locals_ = self.local_faces(rank)
        if not 0 <= local_face < len(locals_):
            raise ValueError(f"local face {local_face} out of range for rank {rank}")
        return locals_[local_face]

    def rank_of_face(self, global_face: int) -> int:
        return self.face_to_rank[global_face]

    def neighbour_rank(
        self,
        connectivity: Mapping[tuple[int, str], tuple[int, str, int]],
        face: int,
        side: Literal["north", "south", "east", "west"],
    ) -> int:
        """Rank owning the neighbour across ``(face, side)`` per ``connectivity``."""
        neighbour_face = connectivity[(face, side)][0]
        return self.face_to_rank[neighbour_face]

    def is_cross_rank_seam(
        self,
        connectivity: Mapping[tuple[int, str], tuple[int, str, int]],
        face: int,
        side: Literal["north", "south", "east", "west"],
    ) -> bool:
        """True if ``(face, side)`` crosses a rank boundary (needs P2P)."""
        return self.neighbour_rank(connectivity, face, side) != self.face_to_rank[face]

    def validate(self) -> None:
        """Self-check: full coverage, balanced rank sizes, contiguous rank ids."""
        if len(self.face_to_rank) != FACE_COUNT:
            raise ValueError(f"face_to_rank must have {FACE_COUNT} entries")
        ranks = set(self.face_to_rank)
        if ranks != set(range(self.n_ranks)):
            raise ValueError(
                f"ranks {sorted(ranks)} must be exactly 0..{self.n_ranks - 1}"
            )
        # Face-aligned + balanced: every rank owns the same number of faces.
        expected = FACE_COUNT // self.n_ranks
        if FACE_COUNT % self.n_ranks != 0:
            raise ValueError(f"{FACE_COUNT} faces not divisible by {self.n_ranks} ranks")
        for r in range(self.n_ranks):
            got = self.n_local_faces(r)
            if got != expected:
                raise ValueError(f"rank {r} owns {got} faces, expected {expected}")


def validate_seam_rank_consistency(
    connectivity: Mapping[tuple[int, str], tuple[int, str, int]],
    decomposition: CubeDecomposition,
) -> None:
    """Every seam's two sides must agree on the (rank, rank') pair (reciprocity).

    For each ``(face, side) -> (nf, ns, rot)``, the reverse link ``(nf, ns)``
    must point back to ``face`` and the neighbour-rank seen from each side must
    be the other side's owning rank. This proves the P2P send/recv pairing
    is well-defined: no seam where the two ranks disagree on who owns
    what.
    """
    for (face, side), (nf, ns, _rot) in connectivity.items():
        back = connectivity.get((nf, ns))
        if back is None or back[0] != face:
            raise ValueError(f"seam ({face},{side})->({nf},{ns}) has no reciprocal link")
        # rank seen from this side == owner of neighbour face; and vice versa.
        if decomposition.neighbour_rank(connectivity, face, side) != decomposition.rank_of_face(nf):
            raise ValueError(f"seam ({face},{side}) neighbour_rank mismatch")
        if decomposition.neighbour_rank(connectivity, nf, ns) != decomposition.rank_of_face(face):
            raise ValueError(f"reverse seam ({nf},{ns}) neighbour_rank mismatch")


# ---------------------------------------------------------------------------
# Per-rank face-local layout context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CubeRankContext:
    """Per-rank metadata for the face-local multi-rank cube layout.

    Fields:
      - decomposition: CubeDecomposition  — global face->rank table
      - rank: int                          — this rank's id
      - local_face_ids: tuple[int, ...]    — global face ids owned by this rank
                                             (ascending global order)
      - global_to_local: dict[int, int]    — owned global face -> local index
      - face_to_rank: tuple[int, ...]      — length-6 owning rank per global face

    Construct with ``CubeRankContext.build(decomposition, rank)``; the
    ``for_ranks`` class method is a convenience shortcut.
    """

    decomposition: CubeDecomposition
    rank: int
    local_face_ids: tuple[int, ...]
    global_to_local: dict[int, int]
    face_to_rank: tuple[int, ...]

    @classmethod
    def build(cls, decomposition: CubeDecomposition, rank: int) -> "CubeRankContext":
        """Build a context for ``rank`` from an existing ``CubeDecomposition``."""
        if not 0 <= rank < decomposition.n_ranks:
            raise ValueError(
                f"rank {rank} out of range [0, {decomposition.n_ranks})"
            )
        local_face_ids = decomposition.local_faces(rank)
        g2l = {gf: lf for lf, gf in enumerate(local_face_ids)}
        return cls(
            decomposition=decomposition,
            rank=rank,
            local_face_ids=local_face_ids,
            global_to_local=g2l,
            face_to_rank=decomposition.face_to_rank,
        )

    @classmethod
    def for_ranks(cls, n_ranks: int, rank: int, *, n: int = 32) -> "CubeRankContext":
        """Shortcut: build decomposition + context for ``(n_ranks, rank)``."""
        dec = CubeDecomposition.for_ranks(n_ranks, n=n)
        return cls.build(dec, rank)

    def is_owned(self, global_face: int) -> bool:
        return global_face in self.global_to_local

    def local_index(self, global_face: int) -> int:
        """Local index of owned ``global_face``; raises if not owned."""
        try:
            return self.global_to_local[global_face]
        except KeyError:
            raise ValueError(
                f"global face {global_face} not owned by rank {self.rank}"
            ) from None


def gather_faces(
    local: "torch.Tensor",
    ctx: CubeRankContext,
    group: "dist.ProcessGroup | None" = None,
) -> "torch.Tensor":
    """Gather face-local tensors from all ranks into global face order.

    ``local`` has shape ``[n_local, ...]`` where ``n_local =
    len(ctx.local_face_ids)``.  The result has shape ``[6, ...]`` with
    global face ordering (face 0 .. 5).

    Primary uses: OUTPUT (harness checkpoint dumps) and TESTING (bitwise
    comparison against the single-rank baseline).

    Exception — gather-run-scatter: ``_compute_cube_tracer_update``
    gathers local velocity/tracer tensors to the global layout, runs the
    existing single-rank GAD pass schedule (which operates on ``[6, ...]``
    tensors), then scatters the result back to local layout.  This is an
    explicit, documented temporary aggregation kept for correctness-first GAD
    support; the returned global tensor IS fed back via scatter in that path.
    True per-rank GAD localization is deferred as a tracked follow-up.

    All other paths: the returned global tensor must not be fed back into
    a numerical step.

    Uses ``dist.all_gather`` so all ranks receive the complete global tensor.
    """
    n_global = FACE_COUNT
    inner = local.shape[1:]  # shape after face axis

    # Each rank builds a 6-slot tensor; non-owned slots stay zero.
    six_slot = local.new_zeros(n_global, *inner)
    for lf, gf in enumerate(ctx.local_face_ids):
        six_slot[gf] = local[lf]

    # all_gather: collect one 6-slot tensor from every rank.
    world_size = ctx.decomposition.n_ranks
    if world_size == 1 or not dist.is_initialized():
        return six_slot

    gathered = [local.new_zeros(n_global, *inner) for _ in range(world_size)]
    from locomgcm.distributed.transport_failure import data_all_gather
    data_all_gather(gathered, six_slot, group=group)

    # Merge: exactly one rank owns each global face, so we sum (or pick max).
    # Sum is safe because non-owned slots are zero.
    result = local.new_zeros(n_global, *inner)
    for slab in gathered:
        result += slab
    return result


def scatter_faces(
    global_tensor: "torch.Tensor",
    ctx: CubeRankContext,
) -> "torch.Tensor":
    """Slice the global ``[6, ...]`` tensor to owned faces for this rank.

    Returns a local tensor of shape ``[n_local, ...]``.  This is a pure
    local slice (no communication) — the caller must have the complete
    global tensor already (e.g. during test round-trips or startup).
    """
    n_local = len(ctx.local_face_ids)
    inner = global_tensor.shape[1:]
    result = global_tensor.new_empty(n_local, *inner)
    for lf, gf in enumerate(ctx.local_face_ids):
        result[lf] = global_tensor[gf]
    return result
