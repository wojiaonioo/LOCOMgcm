from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np
import torch
import torch.distributed as dist

from locomgcm.accel.tensor_context import forward_mode_active, ordinary_tensor_mode
from locomgcm.distributed.comm_adjoint import CommSession, active_data_group
from locomgcm.distributed.transport_failure import data_all_reduce

from locomgcm.diagnostics import ckpt_dump
from locomgcm.solvers._pcg_core import PcgControl, PcgTopology, pcg_iterate
from locomgcm.solvers._pcg_core.readback import (
    PcgReadback, copy_pcg_stats, read_pcg_stats,
)
from locomgcm.solvers.tridiag import forward_mode_requested

if TYPE_CHECKING:
    from locomgcm.solvers._cg2d_fused import Cg2dFusedIteration


ToleranceMode = Literal["absolute", "mit_normalized"]
#: Residual target modes of the CG2D and CG3D solves: ``absolute`` compares the raw
#: residual with the target, ``mit_normalized`` MITgcm's normalised residual.
CG_TOLERANCE_MODES = ("absolute", "mit_normalized")
_CG2D_CONTINUE = 0
_CG2D_PAP_ZERO = 1
_CG2D_CONVERGED = 2
_CG2D_NONFINITE = 3
# Plain module state: a ``ContextVar`` would be read
# inside the compiled step (``_resolve_tolerance_mode``), and Dynamo cannot
# trace ``ContextVar.get`` -- one graph break per solve.  The stepper sets the
# override for the dynamic extent of one cg2d call on one thread, which a module
# global expresses exactly (no concurrent steps share a process).
_TOLERANCE_MODE_OVERRIDE: ToleranceMode | None = None
_CG2D_GRAPH_FALLBACK_COUNTS: Counter[str] = Counter()
_CG2D_GRAPH_FALLBACK_DETAILS: dict[str, str] = {}
#: Why a decomposition cannot host a recorded replay.  Each name is a distinct
#: structural cause read off the partition, never a rank count: a single
#: ``unsupported_topology`` token cannot distinguish "this run was launched with
#: more processes than the tile grid has tiles" from "this grid's halo rounds
#: cross a face seam", and those have different owners and different fixes.
_CG2D_GRAPH_TOPOLOGY_REASONS = (
    "incomplete_rank_grid",
    "cross_rank_seam",
    "tile_below_halo",
)
#: ``no_enclosing_collective_solve``: a transport lifetime (a CommSession) is
#: active for the data group but no collective solve owns the capture, so the
#: captured dot-product reductions could never be retired and the transport
#: layer would refuse them (``transport_failure.captured_collectives_unowned``).
#: A no-grad step taken while a session exists but before its window binds the
#: grid is the case.  Only the graph path reads it: a ROOT solve opens its
#: owning round after the stepper's gate and captures inside it, so the
#: stepper does not read the predicate for ROOT calls.
_CG2D_GRAPH_DECLINE_REASONS = (
    "disabled",
    *(f"unsupported_topology:{cause}" for cause in _CG2D_GRAPH_TOPOLOGY_REASONS),
    "non_cuda",
    "grad_enabled",
    "no_enclosing_collective_solve",
    "graph_cache_unavailable",
)


def owned_cells(field: torch.Tensor) -> torch.Tensor:
    """The tile's own cells of a field carried with a one-cell ring (a view).

    In the one-exchange iteration (``ring_matvec`` on :func:`pcg_solve`,
    :func:`pcg_solve_graphed`, and a ringed fused kernel set) the
    preconditioned residual ``z`` and the search direction ``p`` carry one
    cell beyond the tile on every side; ``x``, ``r`` and ``A p`` do not.  The
    inner products and the ``x`` update read this view, so their operands
    hold the same values in the same order as without the ring.
    """
    return field[..., 1:-1, 1:-1]


def _as_is(field: torch.Tensor) -> torch.Tensor:
    """Identity: the halo of a field that already carries its ring, and the
    owned cells of a field that carries none."""
    return field


@dataclass(frozen=True)
class CG2DGraphOperands:
    """Current operands and the factories that rebuild a solve's closures from them.

    The CUDA-graph capture rebuilds the operator and the preconditioner on
    graph-owned static copies.  The ROOT solve of :func:`pcg_solve`
    (``root_operands``) rebuilds the preconditioner on detached operands and
    the operator on leaf coefficients, to form the coefficient part of the
    implicit-function VJP.
    """

    matvec_operands: tuple[torch.Tensor, ...]
    precond_operands: tuple[torch.Tensor, ...]
    make_matvec: Callable[
        [tuple[torch.Tensor, ...]], Callable[[torch.Tensor], torch.Tensor]
    ]
    make_precond: Callable[
        [tuple[torch.Tensor, ...]], Callable[[torch.Tensor], torch.Tensor]
    ]


@dataclass(frozen=True)
class Cg2dRootGraph:
    """The CUDA-graph replay admitted for a ROOT solve's two inner solves.

    ``cg2d_solve`` decides it once per ROOT call (:func:`root_path_selected`)
    from the facts the graph path's admission reads (the switch, the
    partition, the device, a persistent cache), all identical on every rank,
    and passes it to :func:`pcg_solve` with ``root_operands``; the inner
    solves never consult a rank-local quantity.  The ``grad_enabled`` decline
    does not apply here: the first pass and the zero-start re-solve run under
    ``no_grad`` on detached operands, and the ROOT adjoint supplies the
    derivative (the re-solve), so no tape passes through the replayed
    iterations.

    ``make_matvec`` builds the operator the captured body applies to the
    search direction: the ringed one in the "ring" preconditioner mode, as on
    the graph path, while ``root_operands.make_matvec`` stays the plain one
    for ``coefficient_grads``.  ``record(outcome, solve, reason=None)`` writes
    the ``LOCOMGCM_CG2D_CUDA_GRAPH`` dispatch record of one inner solve.
    """

    cache: dict
    key: tuple
    make_matvec: Callable[
        [tuple[torch.Tensor, ...]], Callable[[torch.Tensor], torch.Tensor]
    ]
    record: Callable[..., None]


class _Cg2dIterationTrace:
    """Passive writer for one MIT-equivalent normalized PCG trajectory."""

    def __init__(
        self,
        root: Path,
        tolerance_mode: ToleranceMode,
        state_scale: torch.Tensor,
        rhs: torch.Tensor,
        x0: torch.Tensor,
    ) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() != 1:
            raise RuntimeError("LOCOMGCM_CG2D_TRACE_DIR requires a single-rank solve")
        root.mkdir(parents=True, exist_ok=True)
        for solve_index in range(1_000_000_000):
            solve_dir = root / f"solve_{solve_index:010d}"
            try:
                solve_dir.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(f"no free CG2D trace solve tag under {root}")
        self.solve_dir = solve_dir
        self.state_scale = state_scale.detach().clone()
        self.quadratic_scale = self.state_scale * self.state_scale
        self.iterations_path = solve_dir / "iterations.jsonl"
        self.iterations_path.write_text("", encoding="utf-8")
        self._save("normalized_b.npy", rhs * self.state_scale)
        self._save("normalized_x0.npy", x0 * self.state_scale)
        self._manifest = {
            "format_version": 1,
            "producer": "locomgcm.solvers.cg2d",
            "status": "recording",
            "tolerance_mode": tolerance_mode,
            "dtype": str(rhs.dtype),
            "shape": list(rhs.shape),
            "state_scale": float(self.state_scale.cpu().item()),
            "scalar_fields": [
                "iteration",
                "eta_qrN",
                "pAp",
                "alpha",
                "beta",
                "err_sq",
            ],
            "state_files": {
                "normalized_b": "normalized_b.npy",
                "normalized_x0": "normalized_x0.npy",
                "initial_residual": "initial_residual.npy",
                "residual_pattern": "residual_{iteration:06d}.npy",
                "preconditioned_pattern": "preconditioned_{iteration:06d}.npy",
            },
            "closure": (
                "err_sq[k] is independently recomputed from residual_k; "
                "eta_qrN[k] from residual_(k-1) dot preconditioned_k; "
                "beta[k] = eta_qrN[k] / eta_qrN[k-1], eta_qrN[0]=1"
            ),
        }
        self._write_manifest()

    def _save(self, name: str, value: torch.Tensor) -> None:
        array = value.detach().cpu().numpy()
        np.save(self.solve_dir / name, array, allow_pickle=False)

    def _write_manifest(self) -> None:
        (self.solve_dir / "manifest.json").write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record_initial_residual(self, residual: torch.Tensor) -> None:
        self._save("initial_residual.npy", residual * self.state_scale)

    def normalized_quadratic(self, value: torch.Tensor) -> torch.Tensor:
        return value.detach() * self.quadratic_scale

    def record_iteration(
        self,
        iteration: int,
        *,
        eta_qrN: torch.Tensor,
        p_ap: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        err_sq: torch.Tensor,
        residual: torch.Tensor,
        preconditioned: torch.Tensor,
    ) -> None:
        self._save(
            f"residual_{iteration:06d}.npy", residual * self.state_scale
        )
        self._save(
            f"preconditioned_{iteration:06d}.npy",
            preconditioned * self.state_scale,
        )
        row = {
            "iteration": iteration,
            "eta_qrN": float(self.normalized_quadratic(eta_qrN).cpu().item()),
            "pAp": float(self.normalized_quadratic(p_ap).cpu().item()),
            "alpha": float(alpha.detach().cpu().item()),
            "beta": float(beta.detach().cpu().item()),
            "err_sq": float(self.normalized_quadratic(err_sq).cpu().item()),
        }
        with self.iterations_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def finish(self, num_iters: int) -> None:
        self._manifest["status"] = "complete"
        self._manifest["num_iters"] = num_iters
        self._write_manifest()


_CG2D_TRACE_DIR: str | None = os.environ.get("LOCOMGCM_CG2D_TRACE_DIR")


def _cg2d_trace_dir() -> str | None:
    """``LOCOMGCM_CG2D_TRACE_DIR`` resolved once per process.

    A per-call ``os.environ.get`` inside the compiled step is a Dynamo
    constant-fold failure. Call ``refresh_cg2d_trace_dir`` after changing
    the variable.
    """
    return _CG2D_TRACE_DIR


def refresh_cg2d_trace_dir() -> str | None:
    global _CG2D_TRACE_DIR
    _CG2D_TRACE_DIR = os.environ.get("LOCOMGCM_CG2D_TRACE_DIR")
    return _CG2D_TRACE_DIR


def _start_iteration_trace(
    tolerance_mode: ToleranceMode,
    rhs_normalization: torch.Tensor | None,
    rhs: torch.Tensor,
    x0: torch.Tensor,
) -> _Cg2dIterationTrace | None:
    raw = _cg2d_trace_dir()
    if not raw:
        return None
    state_scale = (
        rhs_normalization
        if rhs_normalization is not None
        else _mit_rhs_normalization(rhs)
    )
    trace = _Cg2dIterationTrace(
        Path(raw).expanduser(), tolerance_mode, state_scale, rhs, x0
    )
    if tolerance_mode == "mit_normalized":
        # The manifest and inputs retain the original scaling provenance, but
        # the loop now supplies already-normalized states and inner products.
        trace.state_scale = torch.ones_like(state_scale)
        trace.quadratic_scale = trace.state_scale
    return trace


def _global_sum_scalar(value: torch.Tensor, *, session: CommSession | None = None) -> torch.Tensor:
    # The reference solver binds the session before its iteration loop.
    if session is not None:
        return session.sum(value)
    result = value.clone()
    if dist.is_available() and dist.is_initialized():
        data_all_reduce(result, op=dist.ReduceOp.SUM, group=active_data_group())
    return result


def _global_max_scalar(value: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    if dist.is_available() and dist.is_initialized():
        data_all_reduce(result, op=dist.ReduceOp.MAX, group=active_data_group())
    return result


def _validate_tolerance_mode(mode: str) -> ToleranceMode:
    if mode not in CG_TOLERANCE_MODES:
        raise ValueError(
            "tolerance_mode must be 'absolute' or 'mit_normalized', "
            f"got {mode!r}"
        )
    return cast(ToleranceMode, mode)


class cg2d_tolerance_mode_scope:
    """Scope the mode used by a stepper call through the shared CG2D segment.

    A class-based context manager (not ``contextlib``): Dynamo inlines
    ``__enter__``/``__exit__`` of a user class but skips generator-based
    managers ("SKIPPED INLINING contextlib.__enter__").
    """

    __slots__ = ("_mode", "_previous")

    def __init__(self, mode: str) -> None:
        self._mode = _validate_tolerance_mode(mode)
        self._previous: ToleranceMode | None = None

    def __enter__(self) -> None:
        global _TOLERANCE_MODE_OVERRIDE
        self._previous = _TOLERANCE_MODE_OVERRIDE
        _TOLERANCE_MODE_OVERRIDE = self._mode

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        global _TOLERANCE_MODE_OVERRIDE
        _TOLERANCE_MODE_OVERRIDE = self._previous
        return False


def _resolve_tolerance_mode(mode: str) -> ToleranceMode:
    override = _TOLERANCE_MODE_OVERRIDE
    return override if override is not None else _validate_tolerance_mode(mode)


def _mit_rhs_normalization(rhs: torch.Tensor) -> torch.Tensor:
    rhs_max = _global_max_scalar(rhs.abs().max())
    rhs_normalization = torch.ones((), dtype=rhs.dtype, device=rhs.device)
    rhs_nonzero = rhs_max != 0
    rhs_safe = torch.where(rhs_nonzero, rhs_max, rhs_normalization)
    normalized = rhs_normalization / rhs_safe
    return torch.where(rhs_nonzero, normalized, rhs_normalization)


def _mit_normalized_err_sq(
    residual: torch.Tensor,
    rhs_normalization: torch.Tensor,
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    normalized_residual = residual * rhs_normalization
    return dot(normalized_residual, normalized_residual)


def _cg2d_residual_done(
    err_sq: torch.Tensor, tolerance_sq, iteration: int,
    *, flags: tuple[bool, bool] | None = None,
):
    """Stop on convergence or report the first nonfinite global residual."""
    # Keep the comparison in its original dtype and read both flags once.
    # Graph replay supplies these same flags in its single host packet.
    finite, converged = flags if flags is not None else torch.stack(
        (torch.isfinite(err_sq), err_sq < tolerance_sq)
    ).tolist()
    if not finite:
        from locomgcm.deferred_checks import DeferredCheckError, report_numerical_failure

        report_numerical_failure(DeferredCheckError(
            f"cg2d non-finite residual: residual_sq={err_sq.item()}, iteration={iteration}"
        ))
        # A bound window defers rejection until its communication terminates.
        return True
    return converged


# --- FP64 true-residual sidecar (default OFF) --------------------------------
#
# An acceptance check may require an EXPLICIT true residual
# ``||rhs - A x||`` recomputed in FP64 from the RETURNED solution; the harness
# refuses to accept the loop's recurrence residual as a substitute.  The switch
# is a single environment lookup per solve when unset, and the recomputation
# happens strictly at the solver's return points, so the solution, the iteration
# count and the reported recurrence residual are bitwise unchanged either way.
_TRUE_RESIDUAL_TRACE_FILENAME = "cg2d_solver_true_residual.jsonl"
_TRUE_RESIDUAL_TRACE_STATE: dict[str, Any] = {"origin": None, "solves": 0}


_TRUE_RESIDUAL_TRACE_RAW: str | None = os.environ.get("LOCOMGCM_CG2D_TRUE_RESIDUAL_TRACE")


def refresh_true_residual_trace_dir() -> str | None:
    """Re-read ``LOCOMGCM_CG2D_TRUE_RESIDUAL_TRACE`` (test seam; use ``cg2d_switches``)."""
    global _TRUE_RESIDUAL_TRACE_RAW
    _TRUE_RESIDUAL_TRACE_RAW = os.environ.get("LOCOMGCM_CG2D_TRUE_RESIDUAL_TRACE")
    return _TRUE_RESIDUAL_TRACE_RAW


def _true_residual_trace_dir() -> Path | None:
    """Directory requested by ``LOCOMGCM_CG2D_TRUE_RESIDUAL_TRACE`` (default OFF).

    Resolved once per process: a per-call ``os.environ.get`` inside
    the compiled step is a Dynamo constant-fold failure.
    """
    raw = _TRUE_RESIDUAL_TRACE_RAW
    if not raw:
        return None
    return Path(raw).expanduser()


def _true_residual_writer_rank() -> bool:
    """Whether this rank owns the artifact file (rank 0, or a serial run)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _fp64_true_residual(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    solution: torch.Tensor,
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tolerance_mode: ToleranceMode,
) -> tuple[float, str, str | None]:
    """Independently recompute the FP64 true residual of a finished solve.

    Returns ``(residual, dtype_label, degraded_reason)``.  Nothing from the
    iteration recurrence is consumed: the operator is applied once more to the
    RETURNED solution and both the subtraction and the reduction are formed from
    that application alone.  The norm follows the solve's own ``tolerance_mode``
    -- ``absolute`` reports ``||r||``, ``mit_normalized`` reports
    ``||r / max|rhs| ||`` using a freshly recomputed normalization -- so the
    number is directly comparable to ``tolerance``.  ``dot`` is the solve's own
    reducer, so a distributed solve reduces over the same all_reduce path; every
    rank executes this, only rank 0 writes the artifact.

    A solve whose operator cannot accept FP64 input is reported as degraded
    rather than silently downgraded: the artifact then carries a non-FP64 dtype
    label and the acceptance check fails closed instead of accepting a
    lower-precision number.
    """
    rhs64 = rhs.detach().to(torch.float64)
    solution64 = solution.detach().to(torch.float64)
    dtype_label = "float64"
    degraded: str | None = None
    try:
        operator = matvec(solution64)
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        if dist.is_available() and dist.is_initialized():
            # A partially executed collective has already desynchronised the
            # group; retrying in another precision would hang instead of fail.
            raise
        operator = matvec(solution.detach())
        dtype_label = (
            str(solution.dtype).replace("torch.", "") + "_operator_degraded"
        )
        degraded = exception_summary(exc)
    residual64 = rhs64 - operator.to(torch.float64)
    if tolerance_mode == "absolute":
        err_sq = dot(residual64, residual64)
    else:
        err_sq = _mit_normalized_err_sq(
            residual64, _mit_rhs_normalization(rhs64), dot
        )
    return float(err_sq.sqrt().item()), dtype_label, degraded


def _passthrough_result(
    result: tuple[torch.Tensor, int, float],
) -> tuple[torch.Tensor, int, float]:
    """Solver return hook when the sidecar is off: the identity."""
    return result


def _true_residual_recorder(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tolerance: float,
    tolerance_mode: ToleranceMode,
) -> Callable[
    [tuple[torch.Tensor, int, float]], tuple[torch.Tensor, int, float]
]:
    """Build the per-solve return hook once, from the resolved solve config.

    With the switch unset this is the module-level identity, so every return
    point of the solver costs one call and nothing else.
    """
    trace_dir = _true_residual_trace_dir()
    if trace_dir is None:
        return _passthrough_result

    def record(
        result: tuple[torch.Tensor, int, float],
    ) -> tuple[torch.Tensor, int, float]:
        return _traced_result(
            trace_dir,
            result,
            matvec=matvec,
            rhs=rhs,
            dot=dot,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
        )

    return record


def _traced_result(
    trace_dir: Path,
    result: tuple[torch.Tensor, int, float],
    *,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tolerance: float,
    tolerance_mode: ToleranceMode,
) -> tuple[torch.Tensor, int, float]:
    """Return ``result`` unchanged, appending one leg-10 sidecar row first.

    Emitted fields are exactly the leg-10 minimum contract plus provenance:
    ``time_seconds`` (wall seconds since this process' first traced solve),
    ``fp64_true_residual``, ``iterations``, ``target_residual`` (the tolerance
    actually enforced, in the same norm as the residual), ``true_residual_dtype``
    and ``tolerance_mode``.
    """
    true_residual, dtype_label, degraded = _fp64_true_residual(
        matvec, rhs, result[0], dot, tolerance_mode
    )
    state = _TRUE_RESIDUAL_TRACE_STATE
    now = time.monotonic()
    if state["origin"] is None:
        state["origin"] = now
    solve_index = int(state["solves"])
    state["solves"] = solve_index + 1
    if not _true_residual_writer_rank():
        return result
    row: dict[str, Any] = {
        "solve_index": solve_index,
        "time_seconds": now - float(state["origin"]),
        "fp64_true_residual": true_residual,
        "iterations": int(result[1]),
        "target_residual": float(tolerance),
        "true_residual_dtype": dtype_label,
        "tolerance_mode": tolerance_mode,
    }
    if degraded is not None:
        row["degraded_reason"] = degraded
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / _TRUE_RESIDUAL_TRACE_FILENAME).open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    return result


def cg2d_cuda_graph_enabled() -> bool:
    """Whether the gated cg2d CUDA-Graph fast path is requested.

    Default OFF.  Set ``LOCOMGCM_CG2D_CUDA_GRAPH=1`` to enable.  Admission is decided
    by :func:`cg2d_graph_topology` from the decomposition itself; the path
    additionally requires CUDA, no grad, and a persistent graph cache, and
    otherwise falls back to :func:`pcg_solve`.  A differentiated solve on the
    ROOT path replays the same body in its two untaped inner solves
    (:class:`Cg2dRootGraph`).  The request decision is owned by the unified
    control plane (``locomgcm.accel``), not by this module.
    """
    from locomgcm.accel import gate

    return gate("cg2d_cuda_graph").effective_value


@dataclass(frozen=True)
class Cg2dGraphPartition:
    """The decomposition a recorded replay would have to hold still.

    Every field is a launch-time configuration number, identical on every rank,
    so :func:`cg2d_graph_topology` reaches the same verdict everywhere without
    a collective -- which it must, because the first thing an admitted graph
    does is capture one.

    ``faces`` is a SHAPE, not a grid label: it is the horizontal face axis the
    coefficient tensors carry.  A tile with more than one face has seams inside
    its halo rounds (``geometry/exchange.py`` keys a cube round by
    ``(field_family, global_face, side, halo_width)``, not by a Cartesian tile
    edge), and that is what the admission below reads -- never a family name.
    """

    #: Horizontal faces carried per tile.  One = a plain Cartesian tile.
    faces: int
    #: Rank-grid extent, ``nPx``/``nPy`` in ``distributed/layout.py``.
    tiles_x: int
    tiles_y: int
    #: Processes the run actually launched with (``dist.get_world_size()``).
    ranks: int
    #: Global cells the tile grid divides (``ParallelLayout.global_nx/ny``).
    cells_x: int
    cells_y: int
    #: Halo the operator apply reads, uniform on all four sides.
    halo_width: int

    def __post_init__(self) -> None:
        for name in (
            "faces",
            "tiles_x",
            "tiles_y",
            "ranks",
            "cells_x",
            "cells_y",
            "halo_width",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def cg2d_graph_topology(partition: Cg2dGraphPartition) -> str | None:
    """Name the structural reason ``partition`` cannot host a replay, or None.

    A hand-recorded graph fixes one sequence of kernels and collectives for the
    life of a run.  Three properties of the decomposition decide whether that
    sequence exists, and all three are read off the partition -- there is no
    rank count and no grid-family table here, which is the point: the old
    admission hard-coded ``(cube, 1)`` and ``(latlon, 2)`` and therefore fired
    in exactly one posture, silently declining the single-card runs everyone
    believed were using it (a requested path with zero executions) and every
    partition wider than the production benchmark's.

    ``incomplete_rank_grid``
        The tile grid does not account for every process.
        ``distributed/layout.py:184-190`` refuses exactly this combination when
        it builds a layout, so the neighbour table
        ``build_cartesian_neighbors`` describes is not the one the run would
        use, and a captured point-to-point round can address a rank that owns
        no tile.  This is also how a decomposition that lives somewhere other
        than the Cartesian tile grid (the distributed cube, which partitions by
        face through its own context) declines: it presents no tile grid.
    ``cross_rank_seam``
        Some halo round would cross a process boundary AND the tile carries
        more than one face, i.e. some rounds are seam rounds with their own
        rotation and corner handling (``geometry/exchange.py:451-457``).  Their
        pairing is not determined by the Cartesian adjacency alone, so the
        recorded sequence is not the same on every rank.  Today no production
        partition reaches this branch -- ``localize_structured_grid`` refuses
        ``face_count != 1`` (``distributed/localize.py:359-360``), so a
        multi-face grid never carries a tile grid and declines one line above.
        It is kept because that is the accident being relied on: the day a
        cube is tiled Cartesian-wise, ``incomplete_rank_grid`` stops firing and
        without this the seamed partition would be admitted and deadlock inside
        a captured graph, which no test can observe.
    ``tile_below_halo``
        The narrowest tile the partition produces is thinner than the halo the
        operator reads.  ``distributed/layout.py:120-145`` records what happens
        then: the undersized rank takes an early return that skips exchanges
        its neighbours are still posting, and the run deadlocks with no error.
        Inside a captured graph that would be a permanent deadlock rather than
        a first-step one.

    Whether a process boundary exists at all is asked of the layout, not of a
    rank count: a partition whose tiles have no Cartesian neighbour exchanges
    nothing across processes, so a seam or a thin tile cannot be observed by a
    peer and neither restriction applies.  Periodicity is deliberately not
    supplied to the neighbour builder -- the question here is whether a HALO
    ROUND CROSSES A PROCESS, and a periodic wrap on a single tile is rank-local
    whatever the physical boundary says.
    """
    from locomgcm.distributed.layout import (
        build_cartesian_neighbors,
        minimum_tile_extents,
    )

    if partition.ranks != partition.tiles_x * partition.tiles_y:
        return "incomplete_rank_grid"

    neighbours = build_cartesian_neighbors(
        nPx=partition.tiles_x, nPy=partition.tiles_y, rank=0
    )
    if not any(neighbour is not None for neighbour in neighbours.values()):
        return None

    if partition.faces > 1:
        return "cross_rank_seam"
    narrowest = min(
        minimum_tile_extents(
            global_nx=partition.cells_x,
            global_ny=partition.cells_y,
            nPx=partition.tiles_x,
            nPy=partition.tiles_y,
        )
    )
    if narrowest < partition.halo_width:
        return "tile_below_halo"
    return None


def cg2d_cuda_graph_decline_reason(
    *,
    requested: bool,
    partition: Cg2dGraphPartition,
    device_type: str,
    grad_enabled: bool,
    graph_cache_available: bool,
    capture_unowned: bool = False,
) -> str | None:
    """Return the first rank-uniform reason the CUDA-Graph path cannot run.

    The caller supplies only launch/configuration facts, never local field
    values.  Consequently identical model configuration on every rank yields
    an identical decision before the first captured collective.  The topology
    half of the question belongs to :func:`cg2d_graph_topology`, which answers
    it from the decomposition; this function adds the per-launch conditions the
    partition cannot see.  ``capture_unowned`` is the transport layer's answer
    (``captured_collectives_unowned``) for the data group: the same predicate
    that refuses a captured collective at issue time, read here before capture.
    """
    if not requested:
        return "disabled"
    topology = cg2d_graph_topology(partition)
    if topology is not None:
        return f"unsupported_topology:{topology}"
    if device_type != "cuda":
        return "non_cuda"
    if grad_enabled:
        return "grad_enabled"
    if capture_unowned:
        return "no_enclosing_collective_solve"
    if not graph_cache_available:
        return "graph_cache_unavailable"
    return None


def exception_summary(exc: BaseException) -> str:
    """``Type: first non-empty lines of the message``, joined, length-bounded.

    Up to four non-empty lines are kept, joined with `` | `` so that a
    swallowed capture failure stays one greppable token instead of a
    multi-page traceback in the run log.  Four rather than one because a
    Dynamo-augmented assertion reads ``from user code:`` on its first line and
    only names the offending ``File "...", line N`` plus the source line
    afterwards (a first-line-only summary loses the location).
    """
    lines = [ln.strip() for ln in str(exc).strip().splitlines() if ln.strip()]
    head = " | ".join(lines[:4])
    summary = f"{type(exc).__name__}: {head}" if head else type(exc).__name__
    return summary[:400]


def record_cg2d_cuda_graph_fallback(
    reason: str, *, detail: str | None = None
) -> None:
    """Record a declined replay and retain the legacy diagnostic counter.

    Admission declines and prologue convergence use the same dispatch ledger
    as replay hits.  Capture/replay failures are recorded by their callers as
    errors before propagating the original exception, so do not duplicate them.

    ``detail`` carries the caller's :func:`exception_summary` for the failure
    modes that would otherwise lack a cause. Only the first detail for each
    reason is retained and announced to avoid per-step log flooding. The reason
    feeds the rank-uniformity dispatch signature, which must stay rank-uniform.
    """
    known_reasons = _CG2D_GRAPH_DECLINE_REASONS[1:] + (
        "graph_capture_or_replay_failed",
        "prologue_converged",
    )
    if reason not in known_reasons:
        raise ValueError(f"unknown cg2d CUDA-Graph fallback reason {reason!r}")
    _CG2D_GRAPH_FALLBACK_COUNTS[reason] += 1
    if reason != "graph_capture_or_replay_failed" and cg2d_cuda_graph_enabled():
        from locomgcm.accel import current_rank, record_dispatch

        record_dispatch(
            "cg2d_cuda_graph",
            outcome="inapplicable",
            variant="cuda_graph",
            site="cg2d.record_cg2d_cuda_graph_fallback",
            backend="cuda",
            rank=current_rank(),
            reason=reason,
        )
    if detail is not None and reason not in _CG2D_GRAPH_FALLBACK_DETAILS:
        _CG2D_GRAPH_FALLBACK_DETAILS[reason] = detail
        print(
            "LOCOMGCM_CG2D_GRAPH_FALLBACK "
            + json.dumps({"reason": reason, "detail": detail}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )


def cg2d_cuda_graph_fallback_counts() -> dict[str, int]:
    """Return a snapshot of named eager-fallback counts."""
    return dict(_CG2D_GRAPH_FALLBACK_COUNTS)


def cg2d_cuda_graph_fallback_details() -> dict[str, str]:
    """First-seen exception summary per fallback reason (empty when none)."""
    return dict(_CG2D_GRAPH_FALLBACK_DETAILS)


def reset_cg2d_cuda_graph_fallback_counts() -> None:
    """Reset fallback accounting (test and run-boundary helper)."""
    _CG2D_GRAPH_FALLBACK_COUNTS.clear()
    _CG2D_GRAPH_FALLBACK_DETAILS.clear()


def cg2d_cuda_graph_should_run(
    *,
    requested: bool,
    partition: Cg2dGraphPartition,
    device_type: str,
    grad_enabled: bool,
    graph_cache_available: bool,
    capture_unowned: bool = False,
) -> bool:
    """Resolve the graph gate and account for every requested fallback."""
    reason = cg2d_cuda_graph_decline_reason(
        requested=requested,
        partition=partition,
        device_type=device_type,
        grad_enabled=grad_enabled,
        graph_cache_available=graph_cache_available,
        capture_unowned=capture_unowned,
    )
    if reason is None:
        return True
    if reason != "disabled":
        record_cg2d_cuda_graph_fallback(reason)
    return False


def _callable_bound_values(function: Callable[..., Any]) -> list[Any]:
    """Values directly bound by a callable, in binding order."""
    target = getattr(function, "__func__", function)
    bound: list[Any] = []
    closure = getattr(target, "__closure__", None) or ()
    for cell in closure:
        try:
            bound.append(cell.cell_contents)
        except ValueError:
            continue
    bound.extend(getattr(target, "__defaults__", None) or ())
    bound.extend((getattr(target, "__kwdefaults__", None) or {}).values())
    owner = getattr(function, "__self__", None)
    if owner is not None:
        bound.extend(getattr(owner, "__dict__", {}).values())
    return bound


def _callable_bound_tensors(function: Callable[..., Any]) -> tuple[torch.Tensor, ...]:
    """Tensor values directly bound by a callable, in binding order."""
    return tuple(
        value
        for value in _callable_bound_values(function)
        if isinstance(value, torch.Tensor)
    )


def _assert_closed_capture(ctx: dict[str, Any]) -> None:
    """Reject capture closures that bind tensors outside graph-owned operands."""
    allowed_ptrs = {
        tensor.data_ptr()
        for tensor in (
            *ctx["static_matvec_ops"],
            *ctx["static_precond_ops"],
        )
    }
    for name in ("matvec_s", "precond_s"):
        for tensor in _callable_bound_tensors(ctx[name]):
            if tensor.data_ptr() not in allowed_ptrs:
                raise RuntimeError(
                    "cg2d CUDA graph capture is not closed: "
                    f"{name} binds tensor id={id(tensor)} "
                    f"data_ptr={tensor.data_ptr()} outside static operands"
                )


def _refresh_cg2d_static_operands(
    ctx: dict[str, Any], operands: CG2DGraphOperands
) -> bool:
    """Refresh graph-owned operands, returning false when recapture is required."""
    pairs = (
        (ctx["static_matvec_ops"], operands.matvec_operands),
        (ctx["static_precond_ops"], operands.precond_operands),
    )
    for static_ops, current_ops in pairs:
        if len(static_ops) != len(current_ops):
            return False
        if any(
            static.shape != current.shape
            or static.dtype != current.dtype
            or static.device != current.device
            for static, current in zip(static_ops, current_ops)
        ):
            return False
    with torch.no_grad():
        for static_ops, current_ops in pairs:
            for static, current in zip(static_ops, current_ops):
                static.copy_(current)
    return True


def _cg2d_ckpt_fact(
    gctx: dict[str, Any],
    rhs: torch.Tensor,
    x0: torch.Tensor,
    *,
    rank: int,
    step: int | None,
    guard_hit: bool,
    num_iters: int,
    cuda_stream: int,
) -> dict[str, Any]:
    """Build one JSON-compatible graph-buffer and callable-binding fact."""
    tensors = {
        name: gctx[name]
        for name in ("sx", "sr", "sp", "srz", "s_err", "s_pAp")
    }
    tensors.update(rhs=rhs, x0=x0)
    static_ops = (
        *gctx["static_matvec_ops"],
        *gctx["static_precond_ops"],
    )
    tensors.update(
        {f"static_ops[{index}]": tensor for index, tensor in enumerate(static_ops)}
    )
    tensors.update(
        {
            f"matvec[{index}]": tensor
            for index, tensor in enumerate(
                _callable_bound_tensors(gctx["matvec_s"])
            )
        }
    )
    tensors.update(
        {
            f"precond[{index}]": tensor
            for index, tensor in enumerate(
                _callable_bound_tensors(gctx["precond_s"])
            )
        }
    )
    return {
        "rank": int(rank),
        "step": step,
        "guard_hit": bool(guard_hit),
        "capture_count": int(gctx["capture_count"]),
        "num_iters": int(num_iters),
        "cuda_stream": int(cuda_stream),
        "tensors": {
            name: {"id": id(tensor), "data_ptr": int(tensor.data_ptr())}
            for name, tensor in tensors.items()
        },
    }


#: Below this iteration cap the caller is asking for a FIXED truncated map --
#: ``k`` steps of PCG, a different function whose derivative really does run
#: through the iterates -- not for the root ``A^-1 b``.  Same number and same
#: reason as ``CG3DPcgSolver._ROOT_MIN_ITERS`` (``cg3d.py:78``).
_ROOT_MIN_ITERS = 8

#: Why a differentiable solve kept the taped loop, by reason.  Read through
#: :func:`cg2d_root_adjoint_counts`; a reason that never appears is a path that
#: never ran, which is what distinguishes "the adjoint declined" from "the
#: adjoint was never reached".
_CG2D_ROOT_ADJOINT_COUNTS: Counter[str] = Counter()


def cg2d_root_adjoint_counts() -> dict[str, int]:
    """Snapshot of ROOT-adjoint admissions (``root``) and declines, by reason."""
    return dict(_CG2D_ROOT_ADJOINT_COUNTS)


def reset_cg2d_root_adjoint_counts() -> None:
    """Reset ROOT-adjoint accounting (test and run-boundary helper)."""
    _CG2D_ROOT_ADJOINT_COUNTS.clear()


class _SegmentSolutions:
    """The ROOT solutions one checkpoint segment keeps, and the replay cursor."""

    def __init__(self, entries: list, replay: bool) -> None:
        self.entries, self.replay, self.taken = entries, replay, 0

    def take(self) -> tuple[torch.Tensor, int, float, float | None]:
        if self.taken == len(self.entries):
            raise RuntimeError(
                "the recompute solves more cg2d systems than its forward kept"
            )
        entry, self.entries[self.taken] = self.entries[self.taken], None
        self.taken += 1
        return entry


#: The checkpoint segment being run, or None (:func:`segment_solutions`).  A
#: plain module global: the solve is an eager graph exit and the autograd
#: thread must see the same value.
_SEGMENT_SOLUTIONS: _SegmentSolutions | None = None


class segment_solutions:
    """Keep one checkpoint segment's ROOT solutions for its recompute.

    A checkpoint window runs a segment twice: forward, and again in the reverse
    to rebuild the graph for its VJP.  The second pass solves the same systems
    with the same right-hand sides, so it would reproduce the forward's
    solutions bit for bit.  In the forward (``replay=False``) every ROOT solve
    appends its solution, iteration count, residual and re-solve tolerance to
    ``entries``; in the recompute (``replay=True``) the solves take them back
    in order instead of iterating, as MITgcm-JAX keeps ``cg2d_x`` in its
    rematerialized reverse pass.  One copy of the solution per step waits from
    the segment's forward to its recompute in the segment head's store slot
    (the window owner attaches it there; the recompute receives it on the
    solve's device).  A recompute that does not take exactly what the forward
    kept raises.

    A class-based context manager, as :class:`cg2d_tolerance_mode_scope`.
    """

    __slots__ = ("_scope", "_previous")

    def __init__(self, entries: list, *, replay: bool) -> None:
        self._scope = _SegmentSolutions(entries, replay)
        self._previous: _SegmentSolutions | None = None

    def __enter__(self) -> None:
        global _SEGMENT_SOLUTIONS
        self._previous = _SEGMENT_SOLUTIONS
        _SEGMENT_SOLUTIONS = self._scope

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        global _SEGMENT_SOLUTIONS
        _SEGMENT_SOLUTIONS = self._previous
        scope = self._scope
        if scope.replay:
            kept = len(scope.entries)
            scope.entries.clear()
            if exc_type is None and scope.taken != kept:
                raise RuntimeError(
                    f"the recompute took {scope.taken} of the {kept} cg2d "
                    "solutions its forward kept"
                )
        return False


def _has_peers() -> bool:
    """Whether this process's data group has other members."""
    return (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(active_data_group()) > 1
    )


def _taped_loop_cause(
    x0: torch.Tensor,
    apply_preconditioner: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_iters: int,
    session: CommSession | None,
    root_adjoint: bool,
    passive_controls: bool = False,
    forward_mode: bool = False,
) -> str | None:
    """Why this solve must keep the taped loop, or ``None`` for the adjoint.

    These tokens describe derivative semantics, not accelerator declines.
    They produce no record_dispatch event or accelerator ledger row;
    :func:`cg2d_root_adjoint_counts` counts them separately.

    The conditions are ``CG3DPcgSolver._wants_adjoint``'s (``cg3d.py:281-320``),
    plus one that is specific to cg2d's transport:

    ``no_grad``
        Nothing is taped anyway; the plain loop is already the whole story.
        Not with ``forward_mode`` (a live forward mode, ``forward_mode_active``):
        ``no_grad`` does not stop forward mode, whose tangent would otherwise
        run through the iterations, so the request is judged as a
        differentiable one.
    ``root_adjoint_not_requested``
        The caller did not ask for ROOT derivative semantics, so the taped loop
        is what it gets.  This is the DEFAULT, and it is a declared choice, not
        a fallback -- see the switch's rationale on :func:`pcg_solve`.
    ``fixed_iteration_cap``
        ROOT semantics say ``x = A^-1 b``, and only then is the derivative one
        more solve.  A caller that pins ``max_iters`` to a small ``k`` is asking
        for the truncated map, whose derivative genuinely runs through the
        iterates, so the taped loop is the CORRECT answer there.
    ``initial_guess_requires_grad`` / ``preconditioner_requires_grad``
        Both are algorithmic controls the root does not depend on, so under ROOT
        semantics their derivative is zero -- but a caller that put a gradient on
        one of them is measuring the truncated map, same as above.  The
        preconditioner arrives as a closure, so the question is asked by
        APPLYING it to a zero field and reading ``requires_grad`` off the answer:
        that is exact (it sees every tensor the apply touches, however it is
        bound), where reading the closure's cells is a guess about binding order.
        One extra apply on the differentiable path only, against the ~2*iters
        applies the solve is about to run.
    ``communication_window``
        The adjoint re-solves during ``backward``, and a :class:`CommSession` is
        a one-shot forward window -- ``session.sum`` in the reverse pass raises
        "a new forward window requires a new CommSession" by construction.  A
        bare ``dist`` group is declined for the same reason at one remove: the
        re-solve's collectives are not part of any plan the peers agreed on.
        Single-rank solves (including the single-rank cube, whose seam exchange
        is local) are unaffected.

    ``passive_controls`` is the shared stepper entry's declaration (it passes
    ``root_operands``): the first guess and the preconditioner carry no
    derivative, as in MITgcm's ``CG2D_B0``, so their two causes do not apply,
    and a bound session takes the solve as one ordered round
    (:meth:`CommSession.linear_solve`) instead of declining.  That round must
    be the same in the no-grad recording of a checkpoint and in its replay, so
    a bound session takes it whatever the grad mode.  Only peers without a
    session still decline: no plan exists for the re-solve's collectives.
    """
    atomic = passive_controls and root_adjoint and session is not None
    if not torch.is_grad_enabled() and not atomic and not forward_mode:
        return "no_grad"
    if not root_adjoint:
        return "root_adjoint_not_requested"
    if int(max_iters) < _ROOT_MIN_ITERS:
        return "fixed_iteration_cap"
    if passive_controls:
        if session is None and _has_peers():
            return "communication_window"
        return None
    if x0.requires_grad:
        return "initial_guess_requires_grad"
    if session is not None or (dist.is_available() and dist.is_initialized()):
        return "communication_window"
    if apply_preconditioner(torch.zeros_like(x0)).requires_grad:
        return "preconditioner_requires_grad"
    return None


def root_path_selected(
    *, max_iters: int, session: CommSession | None, forward_mode: bool = False
) -> bool:
    """Whether ``pcg_solve(root_adjoint=True, root_operands=...)`` takes the ROOT path.

    With passive controls :func:`_taped_loop_cause` reads only the grad mode,
    the iteration cap, the bound session and whether the data group has peers.
    With a bound session the answer is therefore the same on every rank,
    whatever each rank's local operands require.  ``forward_mode`` is the
    caller's ``forward_mode_active()`` reading, program state as well.
    """
    return _taped_loop_cause(
        None, None, max_iters=max_iters, session=session,
        root_adjoint=True, passive_controls=True, forward_mode=forward_mode,
    ) is None


class _Cg2dRootSolve(torch.autograd.Function):
    """``x = A^-1 b`` for cg2d: the converged iterate, one more solve as its VJP.

    ``x_star`` is returned unchanged, so the SOLUTION is bit-for-bit the plain
    loop's -- only the derivative changes.  ``defect`` is ``b - A(theta) x_star``
    evaluated ONCE with the tape on, which is where every upstream dependence
    lives: differentiating it gives ``b_bar = d_bar`` and
    ``theta_bar = -(d(A x_star)/d theta)^T d_bar``, exactly the implicit-function
    VJP that ``_pcg_core.adjoint`` writes out by hand for cg3d.

    Why not ``pcg_solve_adjoint`` + ``five_point_coefficient_grads`` here: that
    pair wants the coefficient TENSORS and their exchanged halos.  ``pcg_solve``
    is handed a ``matvec`` closure and no grid, and cg2d's halo is a cubed-sphere
    seam exchange (a peer exchange when decomposed), which this layer cannot
    reproduce -- a zero halo would be silently wrong on exactly the cs32 geometry
    the acceptance gate measures.  Differentiating one application of the
    caller's own operator forms the same ``-d_bar[i] * x[j]`` contraction and
    takes the seam coupling from the operator instead of guessing it.
    """

    @staticmethod
    def forward(ctx, resolve, x_star, defect):  # noqa: D102 - see class docstring
        ctx.resolve = resolve
        return x_star

    @staticmethod
    def backward(ctx, grad_x):  # noqa: D102 - see class docstring
        if torch.is_grad_enabled():
            # Grad mode on inside backward means ``create_graph=True``.  The
            # re-solve below runs under ``no_grad``, so the gradient returned
            # here carries no ``grad_fn``: a second differentiation would come
            # back empty rather than wrong-but-visible.  Say so instead.
            raise RuntimeError(
                "cg2d's ROOT adjoint provides a first-order VJP only; "
                "create_graph=True (second-order) is not qualified for the "
                "analytic PCG adjoint, whose re-solve carries no tape"
            )
        if not ctx.needs_input_grad[2]:
            return None, None, None
        with torch.no_grad():
            d_bar = ctx.resolve(grad_x.contiguous())
        return None, None, d_bar


def _solve_as_root(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    x0: torch.Tensor,
    apply_preconditioner: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_iters: int,
    tolerance: float,
    tolerance_mode: ToleranceMode,
    use_min_res_sol: bool,
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    err_sq_history: list | None,
    ring_matvec: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[torch.Tensor, int, float]:
    """Differentiable solve: the plain loop for the value, the adjoint for the VJP.

    The forward and the adjoint re-solve go through the SAME :func:`pcg_solve`,
    so the reducer, the tolerance, the preconditioner and the ring are whatever
    this call already selected -- there is no second configuration to keep in
    step.  Both run under ``no_grad``, where :func:`_taped_loop_cause` answers
    ``no_grad`` and the plain body runs, so there is no recursion beyond one
    level and nothing is taped by the iterations.

    The re-solve passes ``err_sq_history=None``: the history is a verification
    instrument, and a backward pass appending a second solve's residuals to the
    forward's list makes it unreadable.  ``LOCOMGCM_CG2D_TRACE_DIR`` /
    ``LOCOMGCM_CG2D_TRUE_RESIDUAL_TRACE`` are not suppressed the same way (they have no
    per-call seam); with either set, a differentiable run records the adjoint
    re-solves as additional solves.
    """

    def resolve(vector: torch.Tensor) -> torch.Tensor:
        return pcg_solve(
            matvec,
            vector,
            torch.zeros_like(vector),
            apply_preconditioner,
            max_iters=max_iters,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            use_min_res_sol=use_min_res_sol,
            dot=dot,
            err_sq_history=None,
            session=None,
            ring_matvec=ring_matvec,
        )[0]

    with torch.no_grad():
        x_star, num_iters, residual = pcg_solve(
            matvec,
            rhs,
            x0,
            apply_preconditioner,
            max_iters=max_iters,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            use_min_res_sol=use_min_res_sol,
            dot=dot,
            err_sq_history=err_sq_history,
            session=None,
            ring_matvec=ring_matvec,
        )

    # ONE taped application of the operator.  ``x_star`` is detached by
    # construction (it came out of the no_grad solve), so the only paths into
    # ``defect`` are ``rhs`` and whatever the matvec closure binds -- which is
    # precisely the set the implicit-function VJP differentiates.
    defect = rhs - matvec(x_star)
    if forward_mode_requested(defect, x_star):
        # The coefficients sit inside ``matvec``, where the no_grad solve above
        # ran their tangents through the iterations; the implicit tangent needs
        # them as tensors (``root_operands``, as ``cg2d_solve`` passes).
        raise NotImplementedError(
            "pcg_solve(root_adjoint=True) without root_operands has no forward-mode "
            "derivative; pass root_operands (the operator's coefficient tensors) "
            "or root_adjoint=False for the derivative of the executed iterations"
        )
    if not defect.requires_grad:
        # Grad was enabled but nothing upstream actually wants a gradient.
        _CG2D_ROOT_ADJOINT_COUNTS["root_no_dependence"] += 1
        return x_star, num_iters, residual
    _CG2D_ROOT_ADJOINT_COUNTS["root"] += 1
    return _Cg2dRootSolve.apply(resolve, x_star, defect), num_iters, residual


def _solve_with_root_operands(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    x0: torch.Tensor,
    operands: CG2DGraphOperands,
    *,
    max_iters: int,
    tolerance: float,
    tolerance_mode: ToleranceMode,
    use_min_res_sol: bool,
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    err_sq_history: list | None,
    ring_matvec: Callable[[torch.Tensor], torch.Tensor] | None,
    session: CommSession | None,
    graph: Cg2dRootGraph | None = None,
) -> tuple[torch.Tensor, int, float]:
    """MITgcm's cg2d adjoint structure, single rank or inside a session.

    ``CG2D_B0`` (``pkg/tapenade/stubs_tap_adj.F:47-74``) re-solves the same
    system from a zero first guess with the incoming adjoint as right-hand
    side, and gives the first guess no derivative; the preconditioner is not
    differentiated either.  Here the first guess and the preconditioner
    operands are detached, the right-hand side receives ``A^-1 x_bar`` and the
    coefficients ``-(d(A x)/d theta)^T A^-1 x_bar`` (``PcgAdjointSpec``), the
    construction cg3d uses (``cg3d.py:441-468``).

    The forward and the re-solve are the plain :func:`pcg_solve` under
    ``no_grad`` with this call's reducer, ring and tolerance, so the solution
    is the eager loop's bit for bit.  Their halo exchanges and reductions run
    under ``constant_transport()``: untaped, on the session's data group when
    one is bound.  A bound session records the solve and its transpose as one
    ordered round (:meth:`CommSession.linear_solve`); its iteration count
    comes from global inner products, so every rank runs the same count, and
    a numerical failure (a NaN residual, no convergence) completes the round
    on every rank and is reported by the step's residual check.  A re-solve
    that ends non-finite or above its tolerance goes to the numerical-failure
    channel (:func:`locomgcm.deferred_checks.report_numerical_failure`).
    ``coefficient_grads`` exchanges ``x`` on every rank, whether or not that
    rank's coefficients need a gradient, because the exchange is collective.

    In ``absolute`` mode the forward stops at ``||r|| < tolerance``, an
    accuracy of ``tolerance / max|b|`` relative to its right-hand side ``b``.
    The re-solve's right-hand side has the loss's scale, so it takes MITgcm's
    normalized test with that relative accuracy, ``tolerance / max|b|``.

    Inside a checkpoint segment (:func:`segment_solutions`) the forward pass
    keeps the solution and the recompute takes it back instead of iterating.

    With ``graph`` both inner solves run :func:`pcg_solve_graphed` with the
    same closures, reducer, tolerance and initial guess, and share one
    captured body in ``graph.cache``; the replay runs the eager loop's
    operations, and on one CUDA rank the solutions, iteration counts,
    residuals and gradients were measured bitwise equal to the eager loop's
    (the ringed body of a decomposed tile is not yet measured).
    A capture or replay failure is recorded and raised, as on the graph path;
    there is no eager substitute.  Inside a session the capture happens in
    the first pass, within the round's capture-owning transport operation.
    """
    from locomgcm.distributed.comm_adjoint import constant_transport
    from locomgcm.solvers._pcg_core.adjoint import PcgAdjointSpec, pcg_solve_adjoint

    initial = x0.detach()
    preconditioner = operands.make_precond(
        tuple(value.detach() for value in operands.precond_operands)
    )
    stats: dict[str, Any] = {}
    graph_operands = (
        None if graph is None
        else CG2DGraphOperands(
            matvec_operands=operands.matvec_operands,
            precond_operands=operands.precond_operands,
            make_matvec=graph.make_matvec,
            make_precond=operands.make_precond,
        )
    )

    def plain(b, guess, history, tol, mode, solve):
        with torch.no_grad(), constant_transport():
            if graph is None:
                return pcg_solve(
                    matvec, b, guess, preconditioner,
                    max_iters=max_iters, tolerance=tol, tolerance_mode=mode,
                    use_min_res_sol=use_min_res_sol, dot=dot,
                    err_sq_history=history, ring_matvec=ring_matvec,
                )
            graph_stats: dict[str, int] = {}
            try:
                result = pcg_solve_graphed(
                    matvec, b, guess, preconditioner,
                    max_iters=max_iters, tolerance=tol, tolerance_mode=mode,
                    use_min_res_sol=use_min_res_sol, dot=dot,
                    err_sq_history=history, ring_matvec=ring_matvec,
                    graph_cache=graph.cache, graph_key=graph.key,
                    graph_operands=graph_operands, graph_stats=graph_stats,
                )
            except Exception as exc:
                record_cg2d_cuda_graph_fallback(
                    "graph_capture_or_replay_failed", detail=exception_summary(exc)
                )
                graph.record("error", solve, reason="graph_capture_or_replay_failed")
                graph.cache.pop(graph.key, None)
                raise
            if graph_stats.get("replays"):
                # A solve that converged in the prologue ran no replay.
                graph.record("hit", solve)
            return result

    kept = _SEGMENT_SOLUTIONS

    def primal(b):
        if kept is not None and kept.replay:
            x, num_iters, residual, adjoint_tolerance = kept.take()
            if x.shape != b.shape or x.dtype != b.dtype or x.device != b.device:
                raise RuntimeError(
                    "the recompute's cg2d system differs from the one its forward kept"
                )
            stats.update(num_iters=num_iters, residual=residual,
                         adjoint_tolerance=adjoint_tolerance)
            return x
        if tolerance_mode == "absolute":
            stats["adjoint_tolerance"] = tolerance * float(_mit_rhs_normalization(b))
        x, num_iters, residual = plain(
            b, initial, err_sq_history, tolerance, tolerance_mode, "primal"
        )
        num_iters, residual = read_pcg_stats(num_iters, residual)
        stats.update(num_iters=num_iters, residual=residual)
        if kept is not None:
            kept.entries.append(
                (x.clone(), num_iters, residual, stats.get("adjoint_tolerance"))
            )
        return x

    def invert(cotangent):
        zeros = torch.zeros_like(cotangent)
        if tolerance_mode == "absolute":
            target, mode = stats["adjoint_tolerance"], "mit_normalized"
        else:
            target, mode = tolerance, tolerance_mode
        x, num_iters, residual = plain(cotangent, zeros, None, target, mode, "resolve")
        num_iters, residual = read_pcg_stats(num_iters, residual)
        if not math.isfinite(residual) or residual > target:
            # A truncated re-solve is a truncated derivative, a numerical failure
            # of the window.  A bound owner judges it on every rank after the
            # reverse while the communication schedule continues; without an
            # owner the channel raises here, as for any unowned window.
            from locomgcm.deferred_checks import (
                DeferredCheckError, report_numerical_failure,
            )
            report_numerical_failure(DeferredCheckError(
                f"cg2d ROOT re-solve did not converge: residual={residual}, "
                f"tolerance={target}, iterations={num_iters} of {max_iters}"
            ))
        return x

    def coefficient_grads(d_bar, x, needs):
        leaves = tuple(
            value.detach().requires_grad_(bool(need))
            for value, need in zip(operands.matvec_operands, needs)
        )
        with torch.enable_grad(), constant_transport():
            applied = operands.make_matvec(leaves)(x)
        wanted = [leaf for leaf in leaves if leaf.requires_grad]
        grads = iter(torch.autograd.grad(applied, wanted, -d_bar, allow_unused=True)
                     if wanted else ())
        return tuple(next(grads) if leaf.requires_grad else None for leaf in leaves)

    def coefficient_tangent(x, coefficients):
        # The forward-mode counterpart of ``coefficient_grads``: the same
        # operator factory, applied once to the plain ``x`` with the dual
        # coefficients, so seams and halo come from the operator.
        with constant_transport():
            applied = operands.make_matvec(tuple(coefficients))(x)
        return torch.autograd.forward_ad.unpack_dual(applied).tangent

    spec = PcgAdjointSpec(
        solve=invert, coefficient_grads=coefficient_grads,
        coefficient_tangent=coefficient_tangent, primal_solve=primal,
    )
    coefficients = operands.matvec_operands
    dependence = (rhs.requires_grad or any(value.requires_grad for value in coefficients)
                  or forward_mode_requested(rhs, *coefficients))
    _CG2D_ROOT_ADJOINT_COUNTS["root" if dependence else "root_no_dependence"] += 1
    x = pcg_solve_adjoint(spec, rhs, *coefficients, session=session)
    return x, stats["num_iters"], stats["residual"]


def pcg_solve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    x0: torch.Tensor,
    apply_preconditioner: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_iters: int = 1000,
    tolerance: float = 1e-13,
    tolerance_mode: ToleranceMode = "absolute",
    use_min_res_sol: bool,
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    err_sq_history: list | None = None,
    session: CommSession | None = None,
    root_adjoint: bool = False,
    ring_matvec: Callable[[torch.Tensor], torch.Tensor] | None = None,
    root_operands: CG2DGraphOperands | None = None,
    root_graph: Cg2dRootGraph | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Preconditioned conjugate-gradient loop for CG2D tensors.

    ``dot(a, b)`` is the global inner product ``<a, b>``. The default reproduces
    the legacy lat-lon behaviour byte-for-byte (single ``torch.sum`` of the full
    tensor, then all_reduce). The cube path injects a face-ordered
    reducer so single- and multi-rank give bitwise-identical results; only the
    cube call site selects it (never a global switch -> lat-lon stays identical).
    Only the summation ORDER changes; the CG math is untouched.

    Parameters
    ----------
    tolerance_mode : {"absolute", "mit_normalized"}
        ``absolute`` preserves the historical ``||r|| < tolerance`` test.
        ``mit_normalized`` applies MITgcm's dimensionless test using the current
        solve's global RHS maximum: ``||r|| / max(abs(rhs)) < tolerance``.  A
        zero RHS uses normalization one, matching ``cg2d.F``.
    use_min_res_sol : bool
        MITgcm ``cg2dUseMinResSol`` (``nIterMin >= 0``, solve_for_pressure.F:278).
        True keeps the iterate of lowest residual, the first guess included
        (cg2d.F:148-155, :190-193, :338-351), and returns it when the last
        residual is larger (cg2d.F:358-369).  A converged solve returns its
        last iterate either way.  The residual reported is the last one, as
        MITgcm's ``lastResidual``.
    err_sq_history : list | None
        If given, each iteration's ``err_sq`` tensor is appended in order,
        in the loop's units (RHS-normalized for ``mit_normalized``).
        Used only by tests to verify bitwise residual history; has no effect on
        the solve itself.  The list must be pre-allocated (e.g. ``[]``) by the
        caller.  The initial ``err_sq`` (before the first iteration) is NOT
        appended (that would require callers to handle the early-exit path
        specially); only per-iteration values are recorded.
    ring_matvec : callable | None
        ``None`` (default): ``matvec`` applies the operator to the search
        direction, and every field has the tile's shape.  Given: the
        one-exchange iteration of a decomposed tile.  ``apply_preconditioner``
        then returns ``z`` on the tile plus a one-cell ring -- the values the
        neighbour tiles compute for their own edge cells -- the search
        direction ``p = z + beta p`` is carried with that ring, and
        ``ring_matvec`` applies the operator to it with the ring as its halo,
        so ``p`` is never exchanged.  ``matvec`` still forms ``b - A x`` and
        the diagnostics.  The inner products and the ``x`` update read
        :func:`owned_cells`, so the solution, the iteration count and the
        residual history are those of the loop with ``ring_matvec=None`` and
        the same preconditioner on the tile.
    root_operands : CG2DGraphOperands | None
        The coefficient tensors of ``matvec`` and the preconditioner's
        operands, with the factories that rebuild both closures.  Given with
        ``root_adjoint=True``, it declares MITgcm's adjoint structure: the
        first guess and the preconditioner carry no derivative, and a bound
        ``session`` takes the solve as one ordered round
        (:func:`_solve_with_root_operands`).
    root_graph : Cg2dRootGraph | None
        The caller's CUDA-graph admission for the ROOT solve's two inner
        solves (see :class:`Cg2dRootGraph`), given only when
        :func:`root_path_selected` holds; a call that keeps the taped loop
        with it raises.

    Differentiability
    -----------------
    ``root_adjoint`` picks WHICH FUNCTION is differentiated: the executed
    iterations or the root ``A^-1 b``.  The two coincide as the solve converges
    and differ at a finite tolerance through truncation and the initial-guess
    dependence.  The VALUE is untouched either way: the returned solution is
    the plain loop's, bit for bit.

    ``root_adjoint=False`` (default) tapes the executed PCG iterations, initial
    guess included.  MITgcm's default adjoint does NOT tape cg2d: Tapenade
    treats it as external
    (``tools/TAP_support/flow_tap:86-92``) and ``CG2D_B0``
    (``pkg/tapenade/stubs_tap_adj.F:47-74``) re-solves the same symmetric
    system from a zero first guess, with the incoming adjoint as right-hand
    side, the matrix fixed and no derivative for the initial guess; TAF
    re-solves the same way through ``cg2d_mad`` (``pkg/autodiff/cg2d.flow``).
    Only the NSA solver (``ALLOW_CG2D_NSA`` + ``useNSACGSolver``, e.g.
    ``verification/bottom_ctrl_5x5``) is differentiated through its iterations.
    With the matrix independent of the controls both approximate the
    implicit-function derivative: against the CS32 two-step Tapenade adjoint the
    taped loop gives U/V relative errors ``3.69e-14 / 4.51e-14``, a re-solve
    built like MITgcm's (both steps, no initial-guess derivative, MITgcm's
    preconditioner) ``1.4e-15 / 5.3e-16``.

    ``root_adjoint=True`` returns the implicit-function VJP -- one more solve of
    the same system -- i.e. the derivative of the ROOT, which is the structure
    of MITgcm's re-solve (cg3d has run on it with the same root-adjoint construction,
    ``cg3d.py:118-130``).  A direct comparison measured ``1.53e-12 / 1.61e-12`` with it on
    the CS32 step-1 solve.  That came from this loop's former absolute exit
    ``|<p, Ap>| < 1e-30``, not from the derivative: the adjoint right-hand side
    (about 10.6, against 2.3e10 in the forward) crossed it after 18 iterations
    at normalized residual 2.2e-11.  Without it the ROOT result is
    ``3.85e-14 / 4.65e-14`` ; the loop now breaks only on an exact
    zero.  That run re-solved step 1 only; step 2 kept the taped loop because
    its first guess, the previous ``eta``, carries grad.  Re-solving both steps
    with no first-guess derivative gives ``3.5e-15 / 4.0e-15`` (exact re-solve)
    and ``1.4e-15 / 5.3e-16`` (MITgcm's preconditioner at 1e-13) on CPU.

    The default stays False for a direct caller, whose closures are all this
    loop sees: a first guess or preconditioner carrying grad asks for the
    truncated map, and a session-bound solve has no atomic form here.  The
    stepper's shared entry (``cg2d_solve``) passes ``root_operands`` with
    ``root_adjoint=True``; there the first guess and the preconditioner are
    passive, as in ``CG2D_B0``, so every step of a rollout takes the ROOT path,
    on one rank or inside a :class:`CommSession`.
    """
    tolerance_mode = _resolve_tolerance_mode(tolerance_mode)
    forward_mode = forward_mode_active()
    if forward_mode and (session is not None or _has_peers()):
        # Every path with peers (ROOT, taped loop, any iteration budget)
        # exchanges halos and reduces through operators and collectives that
        # carry no forward-mode rule.  Program state only, so ranks that all
        # open the same dual level raise here, before the first exchange.
        raise NotImplementedError(
            "cg2d pcg_solve: forward-mode derivatives of a solve with peer ranks are "
            "not supported; its communication provides first-order reverse "
            "derivatives only; take forward-mode derivatives on a single rank"
        )
    cause = _taped_loop_cause(
        x0, apply_preconditioner, max_iters=max_iters, session=session,
        root_adjoint=root_adjoint, passive_controls=root_operands is not None,
        forward_mode=forward_mode,
    )
    if cause is None and root_operands is not None:
        return _solve_with_root_operands(
            matvec,
            rhs,
            x0,
            root_operands,
            max_iters=max_iters,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            use_min_res_sol=use_min_res_sol,
            dot=dot,
            err_sq_history=err_sq_history,
            ring_matvec=ring_matvec,
            session=session,
            graph=root_graph,
        )
    if root_graph is not None:
        raise ValueError(
            "root_graph is admitted only for a call that takes the ROOT path "
            f"(root_path_selected); this one keeps the taped loop: {cause}"
        )
    if cause is None:
        return _solve_as_root(
            matvec,
            rhs,
            x0,
            apply_preconditioner,
            max_iters=max_iters,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            use_min_res_sol=use_min_res_sol,
            dot=dot,
            err_sq_history=err_sq_history,
            ring_matvec=ring_matvec,
        )
    if cause != "no_grad":
        _CG2D_ROOT_ADJOINT_COUNTS[cause] += 1
    rhs_normalization = None
    tolerance_sq = tolerance * tolerance
    if tolerance_mode == "mit_normalized":
        rhs_normalization = _mit_rhs_normalization(rhs)
        tolerance_tensor = torch.as_tensor(
            tolerance, dtype=rhs.dtype, device=rhs.device
        )
        tolerance_sq = tolerance_tensor * tolerance_tensor

    if dot is None:
        def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return _global_sum_scalar((a * b).sum(), session=session)

    iteration_trace = _start_iteration_trace(
        tolerance_mode, rhs_normalization, rhs, x0
    )
    trace_true_residual = _true_residual_recorder(
        matvec, rhs, dot, tolerance, tolerance_mode
    )
    x = x0.clone()
    loop_rhs = rhs
    if rhs_normalization is not None:
        # MIT cg2d.F scales b and x before forming any residual or dot product.
        loop_rhs = rhs * rhs_normalization
        x = x * rhs_normalization
    if use_min_res_sol:
        # cg2d.F:148-155: the saved solution starts as the (scaled) first guess.
        # ``x`` is rebound, never written in place, so no copy is needed.
        x_min = x
    r = loop_rhs - matvec(x)
    if iteration_trace is not None:
        iteration_trace.record_initial_residual(r)

    err_sq = dot(r, r)
    if use_min_res_sol:
        # cg2d.F:190-193.
        min_err_sq = err_sq
    if _cg2d_residual_done(err_sq, tolerance_sq, 0):
        if iteration_trace is not None:
            iteration_trace.finish(0)
        if rhs_normalization is not None:
            x = x / rhs_normalization
        return trace_true_residual((x, 0, err_sq.sqrt().item()))

    z = apply_preconditioner(r)
    rz = dot(r, z if ring_matvec is None else owned_cells(z))
    p = z.clone()
    trace_beta = (
        iteration_trace.normalized_quadratic(rz)
        if iteration_trace is not None
        else None
    )

    num_iters = 0
    for it in range(max_iters):
        if ring_matvec is None:
            p_owned = p
            Ap = matvec(p)
        else:
            p_owned = owned_cells(p)
            Ap = ring_matvec(p)
        pAp = dot(p_owned, Ap)
        # Exact zero only: <p, Ap> scales with the RHS squared, so an absolute
        # threshold would not scale with the problem.
        if pAp == 0:
            break
        alpha = rz / pAp

        x = x + alpha * p_owned
        r = r - alpha * Ap

        err_sq = dot(r, r)
        num_iters = it + 1

        if iteration_trace is not None:
            iteration_trace.record_iteration(
                num_iters,
                eta_qrN=rz,
                p_ap=pAp,
                alpha=alpha,
                beta=trace_beta,
                err_sq=err_sq,
                residual=r,
                preconditioned=z if ring_matvec is None else owned_cells(z),
            )

        if err_sq_history is not None:
            err_sq_history.append(err_sq.detach().clone())

        if _cg2d_residual_done(err_sq, tolerance_sq, num_iters):
            break
        if use_min_res_sol and err_sq < min_err_sq:
            # cg2d.F:338-351, after the convergence test as in MITgcm.
            min_err_sq = err_sq
            x_min = x

        z = apply_preconditioner(r)
        rz_new = dot(r, z if ring_matvec is None else owned_cells(z))
        beta = rz_new / rz
        rz = rz_new
        if iteration_trace is not None:
            trace_beta = beta.detach().clone()

        p = z + beta * p

    if use_min_res_sol and err_sq > min_err_sq:
        # cg2d.F:358-369, before the solution is unscaled (cg2d.F:371-384).
        x = x_min
    if iteration_trace is not None:
        iteration_trace.finish(num_iters)
    if rhs_normalization is not None:
        x = x / rhs_normalization
    return trace_true_residual((x, num_iters, err_sq.sqrt().item()))


def pcg_solve_fused(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    x0: torch.Tensor,
    *,
    kernels: "Cg2dFusedIteration",
    halo: Callable[[torch.Tensor], torch.Tensor],
    max_iters: int = 1000,
    tolerance: float = 1e-13,
    tolerance_mode: ToleranceMode = "absolute",
    use_min_res_sol: bool,
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    err_sq_history: list | None = None,
) -> tuple[torch.Tensor, int, float]:
    """:func:`pcg_solve` with the body's pointwise and stencil work fused.

    Reproduces :func:`pcg_solve` bitwise in eager execution. Under
    ``torch.compile``, different Inductor transformations may change rounding.

    It reaches that by a narrower route than
    :func:`pcg_solve_graphed`: this does not replay recorded launches, it
    replaces groups of them with single kernels that reproduce eager rounding
    (``locomgcm/solvers/_cg2d_fused``).  What is NOT touched, and is what
    makes the claim checkable rather than hopeful:

    * the three inner products stay ``dot(...)`` -- same reduction, same order,
      same all-reduce, so ``err_sq`` and therefore ``num_iters`` cannot drift;
    * both stopping tests, and both host synchronisations they imply, are the
      eager ones in the eager places;
    * ``alpha = rz / pAp`` and ``beta = rz_new / rz`` stay host-side 0-d tensor
      divisions and reach the kernels as device pointers;
    * the halo exchange stays outside the kernels (``halo``), so multi-rank
      keeps today's P2P contract.

    A ringed kernel set (``kernels.ringed``, the fused counterpart of
    ``ring_matvec`` on :func:`pcg_solve`) returns ``z`` with a one-cell ring
    from ``precond`` and reads ``p``'s ring as the SPMV halo, so the shared
    loop gets the ring itself as its halo and a ``dot`` that reads
    :func:`owned_cells` of a ringed operand; ``_pcg_core/loop.py`` is the same
    loop either way.

    Parameters
    ----------
    matvec :
       The eager operator apply. The loop does not use it -- ``kernels.spmv``
        does -- but the true-residual sidecar recomputes ``||b - A x||`` with
        it, so the diagnostic keeps measuring the eager operator.
    kernels :
       The bundle from ``_cg2d_fused.pcg_iteration_kernels``, entered as a
        context manager by the caller (it owns the ignition ledger entry).
    halo :
       The width-1 centre halo exchange the fused SPMV consumes.  It must be
        the same exchange ``matvec`` performs internally; the call site builds
        both from one expression.  With a ringed kernel set it serves only the
        initial residual ``b - A x0``.
    use_min_res_sol :
       As on :func:`pcg_solve`.  The fused loop keeps only the last iterate,
        so a solve that ends unconverged with it set raises.
    """
    tolerance_mode = _resolve_tolerance_mode(tolerance_mode)
    rhs_normalization = None
    tolerance_sq = tolerance * tolerance
    if tolerance_mode == "mit_normalized":
        rhs_normalization = _mit_rhs_normalization(rhs)
        tolerance_tensor = torch.as_tensor(
            tolerance, dtype=rhs.dtype, device=rhs.device
        )
        tolerance_sq = tolerance_tensor * tolerance_tensor

    if dot is None:
        def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return _global_sum_scalar((a * b).sum())

    iteration_trace = _start_iteration_trace(
        tolerance_mode, rhs_normalization, rhs, x0
    )
    trace_true_residual = _true_residual_recorder(
        matvec, rhs, dot, tolerance, tolerance_mode
    )
    x = x0.clone()
    loop_rhs = rhs
    if rhs_normalization is not None:
        loop_rhs = rhs * rhs_normalization
        x = x * rhs_normalization
    r = loop_rhs - kernels.spmv(halo(x))
    if iteration_trace is not None:
        iteration_trace.record_initial_residual(r)

    err_sq = dot(r, r)
    if _cg2d_residual_done(err_sq, tolerance_sq, 0):
        if iteration_trace is not None:
            iteration_trace.finish(0)
        if rhs_normalization is not None:
            x = x / rhs_normalization
        return trace_true_residual((x, 0, err_sq.sqrt().item()))

    ringed = kernels.ringed
    z = kernels.precond(r)
    rz = dot(r, owned_cells(z) if ringed else z)
    p = z.clone()
    trace_beta = (
        iteration_trace.normalized_quadratic(rz)
        if iteration_trace is not None
        else None
    )
    topology = PcgTopology(halo=halo, dot=dot)
    if ringed:
        owned_width = r.shape[-1]

        def ring_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # The loop hands over ``p`` and ``z`` with their ring and ``r``,
            # ``A p`` without one; the tile's cells are what ``dot`` reduces.
            return dot(
                a if a.shape[-1] == owned_width else owned_cells(a),
                b if b.shape[-1] == owned_width else owned_cells(b),
            )

        topology = PcgTopology(halo=_as_is, dot=ring_dot)

    # ``trace_beta`` is the PREVIOUS iteration's beta (what eager's trace
    # station carries), so it lives in a cell the two hooks share rather than
    # inside the loop.
    trace_state = {"beta": trace_beta}

    def _converged(residual: torch.Tensor, err_sq_value: torch.Tensor):
        """Eager's stopping test, handed to the core rather than restated."""
        del residual
        return _cg2d_residual_done(err_sq_value, tolerance_sq, trace_state["iteration"])

    def _on_iteration(
        num_iters, *, rz, p_ap, alpha, err_sq, residual, preconditioned
    ) -> None:
        trace_state["iteration"] = num_iters
        if iteration_trace is not None:
            iteration_trace.record_iteration(
                num_iters,
                eta_qrN=rz,
                p_ap=p_ap,
                alpha=alpha,
                beta=trace_state["beta"],
                err_sq=err_sq,
                residual=residual,
                preconditioned=(
                    owned_cells(preconditioned) if ringed else preconditioned
                ),
            )
        if err_sq_history is not None:
            err_sq_history.append(err_sq.detach().clone())

    def _on_beta(beta: torch.Tensor) -> None:
        if iteration_trace is not None:
            trace_state["beta"] = beta.detach().clone()

    outcome = pcg_iterate(
        kernels=kernels,
        topology=topology,
        control=PcgControl(
            max_iters=max_iters,
            converged=_converged,
            alpha_breakdown=lambda p_ap: p_ap == 0,
            on_iteration=_on_iteration,
            on_beta=_on_beta,
            # ``pcg_solve`` runs the search-direction tail on its last permitted
            # pass (``cg2d.py:837``); the core must too or the fused path would
            # emit one collective fewer on an unconverged solve.
            emit_tail_on_final_iteration=True,
        ),
        x=x,
        p=p,
        r=r,
        z=z,
        rz=rz,
        err_sq=err_sq,
    )
    x, r, err_sq, num_iters = (
        outcome.x, outcome.r, outcome.err_sq, outcome.num_iters
    )
    if use_min_res_sol and outcome.exit_reason != "converged":
        # The fused loop keeps no lowest-residual iterate (cg2d.F:338-364).
        raise NotImplementedError(
            "cg2d fused PCG: cg_use_min_res_sol requires the lowest-residual "
            f"solution, and the solve ended unconverged ({outcome.exit_reason}, "
            f"{num_iters} of {max_iters} iterations); the fused path keeps only "
            "the last iterate"
        )

    if iteration_trace is not None:
        iteration_trace.finish(num_iters)
    if rhs_normalization is not None:
        x = x / rhs_normalization
    return trace_true_residual((x, num_iters, err_sq.sqrt().item()))


# Kept out of Dynamo on purpose: the body captures a CUDA graph, which cannot
# be traced, and letting Dynamo try costs real time -- in the compiled
# production posture every rank raised an AssertionError from the resumed
# frame on each of its first 8 calls before the frame was abandoned, so the first 8
# solves per rank fell back to the eager PCG instead of replaying the graph.
# The call already graph-breaks at the data-dependent early exit above, so
# this adds no break; the graphed path is byte-identical to the eager one.
@torch.compiler.disable(recursive=True)
def pcg_solve_graphed(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    x0: torch.Tensor,
    apply_preconditioner: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_iters: int = 1000,
    tolerance: float = 1e-13,
    tolerance_mode: ToleranceMode = "absolute",
    use_min_res_sol: bool,
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    err_sq_history: list | None = None,
    graph_cache: dict | None = None,
    graph_key: Any = None,
    graph_operands: CG2DGraphOperands,
    graph_stats: dict[str, int] | None = None,
    ring_matvec: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, int, float | PcgReadback]:
    """:func:`pcg_solve` with the per-iteration body captured in a CUDA graph.

    BYTE-IDENTICAL to :func:`pcg_solve` (same op order, same dot/precond/matvec,
    same two break points) but the iteration body (matvec + 3 dots + precond +
    axpy) is recorded once into a ``torch.cuda.CUDAGraph`` and *replayed* every
    iteration, collapsing the ~50-iteration launch flood into ~50 graph replays.
    The graph encodes breakdown before convergence in one integer status.
    Four replays share one host wait, with snapshots restoring the first
    stopping state; diagnostic observers retain one wait per replay. The
    stopping criterion and iteration count are identical to the eager solve.

    Two-break replication (what bitwise agreement hinges on).  Eager :func:`pcg_solve` has exactly
    two exits inside the loop:

      (top)  ``if pAp == 0: break``            — x is NOT updated this iteration.
      (mid)  ``if err_sq < tol*tol: break``    — x IS updated, then stop.

    The captured body always runs both the x-update and the residual update.  The
    graph therefore snapshots ``x`` BEFORE each iteration; if the pAp-break fires
    the host restores that snapshot (eager would not have advanced x) — giving a
    byte-identical solution and iteration count.

    Cross-step graph reuse.  ``graph_cache``/``graph_key`` let the caller cache
    the captured graph and graph-owned static buffers.  Current coefficient and
    preconditioner operands are copied into those buffers before every solve;
    shape, dtype, or device changes rebuild the context.

    Any failure to capture/replay (allocator quirk, unsupported op, etc.) is the
    caller's signal to fall back to the eager :func:`pcg_solve`; this function
    raises rather than silently differing.

    ``graph_stats`` counts ``captures`` and ``replays``.  A solve that meets
    its tolerance in the prologue returns before the graph and replays nothing;
    one that stops on ``pAp == 0`` has replayed once with zero iterations.

    ``tolerance_mode`` has the same semantics as :func:`pcg_solve`; RHS and
    initial guess normalization precedes capture, and the returned solution
    is unscaled after replay. Tolerance occupies a static device slot
    refreshed for each solve. The final residual is a :class:`PcgReadback`;
    consumers call :func:`read_pcg_stats` before reading its Python numbers.
    Initial convergence keeps the eager return path.

    ``use_min_res_sol`` has the meaning of :func:`pcg_solve`.  The captured
    body keeps only the last iterate, so a solve that ends unconverged with it
    set raises.

    ``ring_matvec`` has the same meaning as on :func:`pcg_solve` (``p`` and
    ``z`` carry a one-cell ring, the preconditioner's residual exchange is the
    iteration's only one); ``graph_operands.make_matvec`` must then build that
    same ring operator from the static operands, since the captured body
    applies it to the ringed ``p``.  The checkpoint records keep the tile's
    shape (:func:`owned_cells`).
    """
    tolerance_mode = _resolve_tolerance_mode(tolerance_mode)
    rhs_normalization = None
    tolerance_sq = tolerance * tolerance
    if tolerance_mode == "mit_normalized":
        rhs_normalization = _mit_rhs_normalization(rhs)
        tolerance_tensor = torch.empty((), dtype=rhs.dtype, device=rhs.device)
        tolerance_tensor.fill_(tolerance)
        tolerance_sq = tolerance_tensor * tolerance_tensor

    if dot is None:
        def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return _global_sum_scalar((a * b).sum())

    iteration_trace = _start_iteration_trace(
        tolerance_mode, rhs_normalization, rhs, x0
    )
    trace_true_residual = _true_residual_recorder(
        matvec, rhs, dot, tolerance, tolerance_mode
    )
    # --- prologue (eager, identical to pcg_solve) ---
    x = x0.clone()
    loop_rhs = rhs
    if rhs_normalization is not None:
        loop_rhs = rhs * rhs_normalization
        x = x * rhs_normalization
    r = loop_rhs - matvec(x)
    if iteration_trace is not None:
        iteration_trace.record_initial_residual(r)

    err_sq0 = dot(r, r)
    if _cg2d_residual_done(err_sq0, tolerance_sq, 0):
        if iteration_trace is not None:
            iteration_trace.finish(0)
        record_cg2d_cuda_graph_fallback("prologue_converged")
        if rhs_normalization is not None:
            x = x / rhs_normalization
        return trace_true_residual((x, 0, err_sq0.sqrt().item()))
    if tolerance_mode == "absolute":
        # Python squares first in absolute mode; MIT mode squares in rhs.dtype.
        tolerance_sq = torch.empty_like(err_sq0).fill_(tolerance * tolerance)

    ringed = ring_matvec is not None
    owned = owned_cells if ringed else _as_is
    z = apply_preconditioner(r)
    rz = dot(r, owned(z))
    p = z.clone()
    if ckpt_dump.ENABLED:
        ckpt_dump.record(
            "cg_pro", x=x, r=r, z=owned(z), p=owned(p), rz=rz, err_sq0=err_sq0
        )
    trace_beta = (
        iteration_trace.normalized_quadratic(rz)
        if iteration_trace is not None
        else None
    )

    # --- resolve / build the captured iteration body + static buffers ---
    gctx = None
    guard_hit = False
    operands_refreshed = False
    cached_capture_count = 0
    if graph_cache is not None and graph_key is not None:
        cached = graph_cache.get(graph_key)
        if cached is not None:
            cached_capture_count = int(cached.get("capture_count", 0))
        # Tolerance is refreshed below; both modes share the captured body.
        if (
            cached is not None
            and cached.get("trace_enabled") == (iteration_trace is not None)
            and _refresh_cg2d_static_operands(cached, graph_operands)
        ):
            gctx = cached
            guard_hit = True
            operands_refreshed = True

    if gctx is None:
        gctx = _build_cg2d_graph_ctx(
            graph_operands,
            dot,
            x,
            r,
            p,
            rz,
            err_sq0,
            tolerance_mode=tolerance_mode,
            tolerance_sq=tolerance_sq,
            trace_values=iteration_trace is not None,
            ringed=ringed,
        )
        gctx["trace_enabled"] = iteration_trace is not None
        gctx["capture_count"] = cached_capture_count + 1
        if graph_stats is not None:
            graph_stats["captures"] = graph_stats.get("captures", 0) + 1
        if graph_cache is not None and graph_key is not None:
            graph_cache[graph_key] = gctx

    if (
        not operands_refreshed
        and not _refresh_cg2d_static_operands(gctx, graph_operands)
    ):
        raise RuntimeError("new cg2d graph context rejected its source operands")

    sx, sr, sp, srz = gctx["sx"], gctx["sr"], gctx["sp"], gctx["srz"]
    s_err, s_pAp, gph = gctx["s_err"], gctx["s_pAp"], gctx["g"]
    snapshot = gctx["snapshots"][0]
    x_prev, err_prev = snapshot["x"], snapshot["err"]
    replay_host = snapshot["host"]
    replay_status = replay_host[1:].view(torch.int64)[0]
    replay_event = snapshot["event"]
    gctx["tolerance_sq"].copy_(tolerance_sq)
    trace_z = gctx.get("trace_z")
    trace_alpha = gctx.get("trace_alpha")
    trace_beta_next = gctx.get("trace_beta")

    # Re-seed the post-prologue state into the persistent static buffers.  The
    # captured body mutates these in place; every step must start from the real
    # post-prologue (x,r,p,rz) of THIS solve.
    sx.copy_(x)
    sr.copy_(r)
    sp.copy_(p)
    srz.copy_(rz)
    if trace_z is not None:
        trace_z.copy_(z)
    # Seed s_err with the prologue err (so a 0-iteration pAp-break would report
    # the pre-loop residual, matching eager which keeps the prologue err_sq).
    s_err.copy_(err_sq0)

    # --- host replay loop reproducing pcg_solve's two breaks exactly ---
    num_iters = 0
    status = _CG2D_CONTINUE
    batched = iteration_trace is None and err_sq_history is None and not ckpt_dump.ENABLED
    if batched:
        slots = gctx["replay_slots"]
        for first in range(0, max_iters, len(slots)):
            count = min(len(slots), max_iters - first)
            for slot in slots[:count]:
                gph.replay()
                # The captured body saved these before its speculative update.
                # Keep each pair until every earlier status has been inspected.
                slot["x"].copy_(x_prev)
                slot["err"].copy_(err_prev)
                slot["host"].copy_(snapshot["packet"], non_blocking=True)
            replay_event.record(torch.cuda.current_stream(sx.device))
            replay_event.synchronize()
            if graph_stats is not None:
                graph_stats["replays"] = graph_stats.get("replays", 0) + count
            restore = None
            for offset, slot in enumerate(slots[:count]):
                status = int(slot["host"][1:].view(torch.int64)[0])
                if status == _CG2D_PAP_ZERO:
                    restore = slot
                    break
                num_iters = first + offset + 1
                if _cg2d_residual_done(
                    slot["host"][0], tolerance_sq, num_iters,
                    flags=(status != _CG2D_NONFINITE, status == _CG2D_CONVERGED),
                ):
                    # The next replay's input is this replay's exact output.
                    # The final replay already left that output in sx/s_err.
                    if offset + 1 < count:
                        restore = slots[offset + 1]
                    break
            else:
                continue
            if restore is not None:
                sx.copy_(restore["x"])
                s_err.copy_(restore["err"])
            break

    # Per-iteration histories and dump observers keep their original replay
    # sequence and storage. The batched path has already completed its solve.
    for it in range(0 if batched else max_iters):
        if iteration_trace is not None:
            trace_eta = srz.detach().clone()
            trace_preconditioned = trace_z.detach().clone()
        gph.replay()
        replay_host.copy_(snapshot["packet"], non_blocking=True)
        replay_event.record(torch.cuda.current_stream(sx.device))
        replay_event.synchronize()
        status = int(replay_status)
        if graph_stats is not None:
            graph_stats["replays"] = graph_stats.get("replays", 0) + 1
        if it == 0 and ckpt_dump.is_dump_step():
            ckpt_dump.record(
                "cg_it1",
                sx=sx,
                sr=sr,
                sp=owned(sp),
                srz=srz,
                s_err=s_err,
                s_pAp=s_pAp,
            )
            torch.cuda.synchronize()
            eager_x = x.clone()
            eager_r = r.clone()
            eager_p = p.clone()
            eager_rz = rz.clone()
            eager_Ap = (matvec if ring_matvec is None else ring_matvec)(eager_p)
            eager_pAp = dot(owned(eager_p), eager_Ap)
            eager_alpha = eager_rz / eager_pAp
            eager_x.add_(eager_alpha * owned(eager_p))
            eager_r.sub_(eager_alpha * eager_Ap)
            eager_err = dot(eager_r, eager_r)
            eager_z = apply_preconditioner(eager_r)
            eager_rz_new = dot(eager_r, owned(eager_z))
            eager_beta = eager_rz_new / eager_rz
            eager_rz.copy_(eager_rz_new)
            eager_p.copy_(eager_z + eager_beta * eager_p)
            ckpt_dump.record(
                "cg_it1_eager",
                sx=eager_x,
                sr=eager_r,
                sp=owned(eager_p),
                srz=eager_rz,
                s_err=eager_err,
                s_pAp=eager_pAp,
            )
        if status == _CG2D_PAP_ZERO:
            # eager breaks at the TOP, before the x-update -> roll x AND the
            # reported residual back (eager returns the previous iter's err_sq).
            sx.copy_(x_prev)
            s_err.copy_(err_prev)
            break
        num_iters = it + 1
        if iteration_trace is not None:
            iteration_trace.record_iteration(
                num_iters,
                eta_qrN=trace_eta,
                p_ap=s_pAp,
                alpha=trace_alpha,
                beta=trace_beta,
                err_sq=s_err,
                residual=sr,
                preconditioned=owned(trace_preconditioned),
            )
            trace_beta = trace_beta_next.detach().clone()
        if err_sq_history is not None:
            err_sq_history.append(s_err.detach().clone())
        if _cg2d_residual_done(
            replay_host[0], tolerance_sq, num_iters,
            flags=(status != _CG2D_NONFINITE, status == _CG2D_CONVERGED),
        ):
            break

    if use_min_res_sol and status not in (_CG2D_CONVERGED, _CG2D_NONFINITE):
        # The captured body keeps no lowest-residual iterate (cg2d.F:338-364).
        raise NotImplementedError(
            "cg2d CUDA-graph PCG: cg_use_min_res_sol requires the lowest-residual "
            f"solution, and the solve ended unconverged (status {status}, "
            f"{num_iters} of {max_iters} iterations); the replay keeps only the "
            "last iterate"
        )
    if tolerance_mode == "absolute":
        if iteration_trace is not None:
            iteration_trace.finish(num_iters)
        if ckpt_dump.ENABLED:
            ckpt_dump.record(
                "cg_end",
                x=sx,
                num_iters=torch.tensor(
                    num_iters, dtype=torch.int64, device=sx.device
                ),
                err=s_err,
            )
            fact = _cg2d_ckpt_fact(
                gctx,
                rhs,
                x0,
                rank=(
                    dist.get_rank()
                    if dist.is_available() and dist.is_initialized()
                    else 0
                ),
                step=ckpt_dump.current_step(),
                guard_hit=guard_hit,
                num_iters=num_iters,
                cuda_stream=torch.cuda.current_stream().cuda_stream,
            )
            print(f"CG2D_CKPT {json.dumps(fact, sort_keys=True)}", flush=True)
        residual = copy_pcg_stats(s_err.sqrt(), num_iters, status)
        if trace_true_residual is not _passthrough_result:
            num_iters, residual = read_pcg_stats(num_iters, residual)
        return trace_true_residual((sx.clone(), num_iters, residual))
    if iteration_trace is not None:
        iteration_trace.finish(num_iters)
    if ckpt_dump.ENABLED:
        ckpt_dump.record(
            "cg_end",
            x=sx / rhs_normalization,
            num_iters=torch.tensor(
                num_iters, dtype=torch.int64, device=sx.device
            ),
            err=s_err,
        )
        fact = _cg2d_ckpt_fact(
            gctx,
            rhs,
            x0,
            rank=(
                dist.get_rank()
                if dist.is_available() and dist.is_initialized()
                else 0
            ),
            step=ckpt_dump.current_step(),
            guard_hit=guard_hit,
            num_iters=num_iters,
            cuda_stream=torch.cuda.current_stream().cuda_stream,
        )
        print(f"CG2D_CKPT {json.dumps(fact, sort_keys=True)}", flush=True)
    # Residuals already use normalized loop units; unscale only the solution.
    residual = copy_pcg_stats(s_err.sqrt(), num_iters, status)
    if trace_true_residual is not _passthrough_result:
        num_iters, residual = read_pcg_stats(num_iters, residual)
    return trace_true_residual((sx / rhs_normalization, num_iters, residual))


@ordinary_tensor_mode()
def _build_cg2d_graph_ctx(
    graph_operands: CG2DGraphOperands,
    dot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    r: torch.Tensor,
    p: torch.Tensor,
    rz: torch.Tensor,
    err_sq0: torch.Tensor,
    *,
    tolerance_mode: ToleranceMode,
    tolerance_sq: torch.Tensor,
    trace_values: bool = False,
    ringed: bool = False,
) -> dict:
    """Side-stream warmup + capture ONE pcg iteration body into a CUDA graph.

    The captured body mirrors :func:`pcg_solve`'s loop op order EXACTLY:

        Ap   = matvec(p)
        pAp  = dot(p, Ap)              # exposed to host for the top break
        alpha= rz / pAp
        x   += alpha * p
        r   -= alpha * Ap
        err  = dot(r, r)              # exposed to host for the mid break
        z    = precond(r)
        rzn  = dot(r, z)
        beta = rzn / rz
        rz   = rzn
        p    = z + beta * p

    ``ringed``: ``p`` and ``z`` carry a one-cell ring (``ring_matvec`` on
    :func:`pcg_solve_graphed`), ``matvec`` is the ring operator, and the two
    inner products and the ``x`` update read :func:`owned_cells` -- the same
    statements as :func:`pcg_solve` with its ``ring_matvec``.

    The supplied ``dot`` and ``matvec`` may contain NCCL collectives (including
    lat-lon halo P2P and global all-reduce dots).  The eager prologue has already
    initialized their communicator before this function enters side-stream
    warmup.  Capture never inspects a Work object or performs host-side control
    flow; the host reads the replay statuses in fixed groups, or after each
    replay when a diagnostic observer requires its intermediate state.

    Persistent static buffers (sx, sr, sp, scalar srz/s_err/s_pAp) are written
    with ``copy_`` between replays — never reallocated — so replay runs the same
    kernels at the same addresses (byte-exact by construction).
    They outlive inference-only forwards and may be reseeded by a ROOT solve
    in a later gradient window, so their storage must be ordinary tensors.
    """
    sx = x.clone()
    sr = r.clone()
    sp = p.clone()
    srz = rz.clone()                  # scalar rz carried across iters
    s_err = err_sq0.clone()           # scalar err_sq output of the body
    s_pAp = err_sq0.clone()           # scalar pAp for the host top-break check
    snapshot = {
        "x": x.clone(),
        "err": err_sq0.clone(),
        "packet": torch.empty(2, dtype=torch.float64, device=x.device),
        "host": torch.empty(2, dtype=torch.float64, device="cpu", pin_memory=True),
        "event": torch.cuda.Event(),
    }
    # Four private input snapshots let the host select the original first stop
    # after one wait. The graph itself and its arithmetic remain unchanged.
    replay_slots = tuple({
        "x": torch.empty_like(x),
        "err": torch.empty_like(err_sq0),
        "host": torch.empty(2, dtype=torch.float64, device="cpu", pin_memory=True),
    } for _ in range(4))
    x_prev, err_prev = snapshot["x"], snapshot["err"]
    packet_err = snapshot["packet"][0]
    packet_status = snapshot["packet"][1:].view(torch.int64)[0]
    s_tolerance_sq = torch.empty_like(tolerance_sq)
    trace_z = sp.clone() if trace_values else None
    trace_alpha = err_sq0.clone() if trace_values else None
    trace_beta = err_sq0.clone() if trace_values else None
    static_matvec_ops = tuple(
        tensor.detach().clone() for tensor in graph_operands.matvec_operands
    )
    static_precond_ops = tuple(
        tensor.detach().clone() for tensor in graph_operands.precond_operands
    )
    matvec_s = graph_operands.make_matvec(static_matvec_ops)
    precond_s = graph_operands.make_precond(static_precond_ops)
    context = {
        "sx": sx, "sr": sr, "sp": sp, "srz": srz,
        "s_err": s_err, "s_pAp": s_pAp,
        "snapshots": (snapshot,),
        "replay_slots": replay_slots,
        "tolerance_mode": tolerance_mode,
        "tolerance_sq": s_tolerance_sq,
        "static_matvec_ops": static_matvec_ops,
        "static_precond_ops": static_precond_ops,
        "matvec_s": matvec_s,
        "precond_s": precond_s,
    }
    _assert_closed_capture(context)
    owned = owned_cells if ringed else _as_is

    # --- side-stream warmup (3 iters) to stabilise allocator/addresses ---
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            Ap_w = matvec_s(sp)
            pAp_w = dot(owned(sp), Ap_w)
            alpha_w = srz / pAp_w
            sx.add_(alpha_w * owned(sp))
            sr.sub_(alpha_w * Ap_w)
            s_err_w = dot(sr, sr)
            z_w = precond_s(sr)
            rz_new_w = dot(sr, owned(z_w))
            beta_w = rz_new_w / srz
            srz.copy_(rz_new_w)
            sp.copy_(z_w + beta_w * sp)
            s_err.copy_(s_err_w)
    torch.cuda.current_stream().wait_stream(s)

    # --- capture ONE iteration body (mirrors pcg_solve op order EXACTLY) ---
    from locomgcm.accel import cuda_graphs

    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph, capture_error_mode=cuda_graphs.CAPTURE_ERROR_MODE):
        x_prev.copy_(sx)
        err_prev.copy_(s_err)
        Ap_c = matvec_s(sp)
        pAp_c = dot(owned(sp), Ap_c)
        s_pAp.copy_(pAp_c)
        alpha_c = srz / pAp_c
        if trace_values:
            trace_alpha.copy_(alpha_c)
        sx.add_(alpha_c * owned(sp))
        sr.sub_(alpha_c * Ap_c)
        err_c = dot(sr, sr)
        z_c = precond_s(sr)
        if trace_values:
            trace_z.copy_(z_c)
        rz_new_c = dot(sr, owned(z_c))
        beta_c = rz_new_c / srz
        if trace_values:
            trace_beta.copy_(beta_c)
        srz.copy_(rz_new_c)
        sp.copy_(z_c + beta_c * sp)
        s_err.copy_(err_c)
        # The zero denominator wins even if the speculative updates made NaNs.
        packet_status.copy_(torch.where(
            s_pAp == 0,
            _CG2D_PAP_ZERO,
            torch.where(
                torch.isfinite(s_err),
                torch.where(s_err < s_tolerance_sq, _CG2D_CONVERGED, _CG2D_CONTINUE),
                _CG2D_NONFINITE,
            ),
        ))
        packet_err.copy_(s_err)

    context["g"] = gph
    # The captured body holds NCCL collectives; shutdown_distributed resets every
    # registered graph before the process group is destroyed.
    cuda_graphs.register(gph)
    if trace_values:
        context.update(
            trace_z=trace_z,
            trace_alpha=trace_alpha,
            trace_beta=trace_beta,
        )
    return context
