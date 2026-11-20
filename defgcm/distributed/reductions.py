"""Distributed reductions with owned-cell accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import torch.distributed as dist

from .comm_adjoint import CommSession
from .transport_failure import data_all_reduce

if TYPE_CHECKING:
    from defgcm.geometry.cube_decomposition import CubeRankContext


@dataclass(frozen=True)
class HaloTrim:
    """Halo widths around a local tensor, ordered by compass side."""

    west: int = 0
    east: int = 0
    south: int = 0
    north: int = 0

    def __post_init__(self) -> None:
        for name in ("west", "east", "south", "north"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} halo width must be a non-negative integer")


def owned_interior(
    tensor: torch.Tensor,
    halo: int | HaloTrim | Sequence[int] = 0,
) -> torch.Tensor:
    """Return the owned interior view, excluding horizontal halo cells."""

    trim = _as_halo_trim(halo)
    if tensor.ndim < 2:
        raise ValueError("tensor must have at least horizontal [y, x] dimensions")
    y_stop = tensor.shape[-2] - trim.north
    x_stop = tensor.shape[-1] - trim.east
    if trim.south > y_stop or trim.west > x_stop:
        raise ValueError(
            f"halo {trim} exceeds tensor horizontal shape {tuple(tensor.shape[-2:])}"
        )
    return tensor[..., trim.south:y_stop, trim.west:x_stop]


def global_sum(value: torch.Tensor, *, group: dist.ProcessGroup | None = None,
               session: CommSession | None = None) -> torch.Tensor:
    """Sum a scalar tensor across ranks when distributed is initialized.

    Reduced-precision guardrail: when ``value`` is not float64 the cross-rank
    sum is accumulated in float64 (the all_reduce runs on the up-cast tensor so
    every rank reduces the identical dtype) and the result is cast back to the
    input dtype.  A float64 ``value`` takes the original path unchanged —
    byte-for-byte identical to the pre-guardrail behaviour, zero extra
    allocation on the default (float64) precision path.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError("global_sum expects a torch.Tensor")
    # The caller selects differentiable communication by binding a session.
    if session is not None:
        return session.sum(value.to(torch.float64)).to(value.dtype)
    if value.dtype is not torch.float64:
        acc = value.to(torch.float64)
        if dist.is_available() and dist.is_initialized():
            data_all_reduce(acc, op=dist.ReduceOp.SUM, group=group)
        return acc.to(value.dtype)
    result = value.clone()
    if dist.is_available() and dist.is_initialized():
        data_all_reduce(result, op=dist.ReduceOp.SUM, group=group)
    return result


def global_dot(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    halo: int | HaloTrim | Sequence[int] = 0,
    group: dist.ProcessGroup | None = None,
    session: CommSession | None = None,
) -> torch.Tensor:
    """Dot product over owned cells only, with optional wet mask.

    ``halo`` excludes local overlap before the local dot is formed.  This is
    the CG-safe reduction path: halo cells must not be counted twice.
    """

    if lhs.shape != rhs.shape:
        raise ValueError(f"lhs/rhs shapes differ: {tuple(lhs.shape)} != {tuple(rhs.shape)}")
    lhs_owned = owned_interior(lhs, halo)
    rhs_owned = owned_interior(rhs, halo)
    product = lhs_owned * rhs_owned
    if mask is not None:
        mask_t = mask.to(dtype=product.dtype, device=product.device)
        if mask_t.shape == lhs.shape:
            mask_t = owned_interior(mask_t, halo)
        elif mask_t.shape != lhs_owned.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match full shape "
                f"{tuple(lhs.shape)} or owned shape {tuple(lhs_owned.shape)}"
            )
        product = product * mask_t
    # Reduced-precision guardrail: accumulate the LOCAL cell sum in float64 too
    # (not just the cross-rank all_reduce) so a float32 dot does not shed bits
    # summing many cells, then cast the scalar back to the input dtype.  float64
    # input takes the original path unchanged (byte-identical).
    if product.dtype is not torch.float64:
        local_sum = product.to(torch.float64).sum()
        return global_sum(local_sum, group=group, session=session).to(product.dtype)
    return global_sum(product.sum(), group=group, session=session)


def face_ordered_dot(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    face_dim: int = 1,
    n_faces: int = 6,
    local_face_ids: Sequence[int] | None = None,
    ctx: "CubeRankContext | None" = None,
    group: dist.ProcessGroup | None = None,
    session: CommSession | None = None,
) -> torch.Tensor:
    """Face-ordered global dot product for cube reductions (the root of the
    program's bitwise n-rank-vs-1-rank judgement).

    Computes ``sum_f (lhs[..f..] * rhs[..f..]).sum()`` where the OUTER face sum
    runs in a FIXED order ``0..n_faces-1`` by sequential add (NOT torch's
    internal tree reduction of a stacked tensor), and each face's INNER
    reduction is taken in ISOLATION (``select(face_dim, f).sum()``).

    Both choices exist for one reason: make the result identical whether the 6
    faces live on a single rank or are split across ranks (face-aligned
    decomposition).

    * Inner: a face summed alone (multi-rank) is bitwise-equal to that same
      face summed inside the ``[*, 6, *, *]`` tensor (single-rank), because
      both reduce exactly that face's elements with the same per-face kernel.
    * Outer: float ``+`` is non-associative, so a defined face order makes the
      total independent of how faces are grouped onto ranks.

    Single-rank path (ctx=None / dist not initialized / ``group is False``):
    this rank holds all ``n_faces`` faces -> per-face isolate + ordered add.
    The added ``.contiguous()`` is a no-op (returns
    self) whenever ``select(face_dim, f)`` is already contiguous, which is the
    case for the sole caller (cube cg2d with ``[1, 6, ny, nx]``, ``face_dim=1`` ->
    each face is a contiguous ``[1, ny, nx]`` block); single-rank cube T0 stays
    bitwise vs epoch-0.  (For a hypothetical non-contiguous face view it would
    change the FP reduction order; no such caller exists.)

    Multi-rank path (dist initialized + ctx given, ``group is not False``):
    Each rank contributes only its owned faces into a 6-slot accumulator,
    all_gather collects one slot-tensor per rank, merge by summing (exactly
    one owner per face → no double-count), then face-ordered add 0..5.

    Routing summary:
    - ``group is False``                       → force single-rank (tests).
    - dist not initialized OR ctx is None      → single-rank path.
    - dist initialized AND ctx given           → multi-rank path.
    - dist initialized AND ctx is None         → raise (ambiguous; the cube
      multi-rank solve always passes ctx).

    The lat-lon ``global_sum``/``global_dot`` path is untouched (this is a
    NEW cube-only entry, never a global switch).
    """
    if lhs.shape != rhs.shape:
        raise ValueError(f"lhs/rhs shapes differ: {tuple(lhs.shape)} != {tuple(rhs.shape)}")
    prod = lhs * rhs
    # Reduced-precision guardrail: accumulate every face's inner sum, the ordered
    # outer add, and the multi-rank all_reduce in float64, then cast the scalar
    # back to the input dtype.  A float64 product is NOT re-cast — the original
    # single- and multi-rank paths run byte-for-byte, so the n-rank-vs-1-rank
    # bitwise invariant is fully preserved on the default precision path.
    _out_dtype = prod.dtype
    if _out_dtype is not torch.float64:
        prod = prod.to(torch.float64)
    n_local = prod.shape[face_dim]

    # ---- routing ----
    is_dist = dist.is_available() and dist.is_initialized()
    force_single = group is False  # explicit sentinel used in tests

    if force_single or not is_dist:
        # Single-rank path.
        # NOTE: .contiguous() before .sum() ensures the same memory-traversal
        # order as the multi-rank path (where local tensors are always
        # contiguous).  The bitwise judgement requires both paths to
        # agree at float64 precision, which they do only when the reduction
        # kernel sees the same stride pattern.
        per_face = [prod.select(face_dim, f).contiguous().sum() for f in range(n_local)]
        if n_local != n_faces:
            raise ValueError(
                f"single-rank face_ordered_dot expects {n_faces} faces on "
                f"face_dim={face_dim}, got {n_local}"
            )
        total = per_face[0]
        for f in range(1, n_faces):
            total = total + per_face[f]
        return total if _out_dtype is torch.float64 else total.to(_out_dtype)

    # dist is initialized; ctx must be given for the multi-rank cube path.
    if ctx is None:
        raise RuntimeError(
            "face_ordered_dot: dist is initialized but ctx=None. "
            "Pass a CubeRankContext or use group=False to force "
            "the single-rank path."
        )

    # ---- multi-rank path ----
    # Build a 6-slot scalar tensor; owned faces fill their contribution,
    # non-owned stay 0.  Each global face has exactly ONE non-zero owner, so the
    # cross-rank sum of the 6-vector is 0+x+0 = x exactly (no rounding) — a single
    # all_reduce(SUM) reproduces the old all_gather+merge bitwise and is ~2.2x
    # cheaper than a 6-slot all_gather (verified 2-rank, R={2,3} x cs{192,768} x
    # NCCL/gloo, max|d|=0).  The face-ordered add 0..5 below (the n-rank-vs-1-rank
    # bitwise invariant) is unchanged.
    # Create six-slot accumulator on the same device/dtype as the product.
    six = prod.new_zeros(n_faces)
    for lf, gf in enumerate(ctx.local_face_ids):
        # .contiguous() ensures the per-face reduction uses the same kernel
        # as the single-rank path (both reduce a contiguous [batch,ny,nx] view).
        six[gf] = prod.select(face_dim, lf).contiguous().sum()

    # None retains automatic process-group inference.
    if session is None:
        per_face_global = six
        data_all_reduce(per_face_global, op=dist.ReduceOp.SUM, group=group)
    else:
        per_face_global = session.sum(six)

    total = per_face_global[0]
    for f in range(1, n_faces):
        total = total + per_face_global[f]
    return total if _out_dtype is torch.float64 else total.to(_out_dtype)


def _as_halo_trim(halo: int | HaloTrim | Sequence[int]) -> HaloTrim:
    if isinstance(halo, HaloTrim):
        return halo
    if isinstance(halo, int):
        return HaloTrim(west=halo, east=halo, south=halo, north=halo)
    values = tuple(halo)
    if len(values) != 4:
        raise ValueError("halo sequence must be (west, east, south, north)")
    return HaloTrim(west=int(values[0]), east=int(values[1]), south=int(values[2]), north=int(values[3]))
