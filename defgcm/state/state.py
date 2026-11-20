from __future__ import annotations

import atexit
import contextlib
import dataclasses
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from defgcm.accel.tensor_context import forward_mode_active
from defgcm.fields import Field
from defgcm.geometry import StructuredCGrid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from typing import Sequence


@dataclass(frozen=True)
class EvaluationInputs:
    """Per-evaluation inputs of the tendency modules that are not prognostic.

    One object per tendency evaluation, created by its caller and discarded
    with the evaluation; nothing here is stored on the grid or a module.

    ``w_interface`` is the lagged interface vertical velocity in the MIT
    upward-positive convention ``[batch, nz+1, ny, nx]`` (MOM_VI vertical
    shear, NH Coriolis, flux-form vertical advection).  ``total_phi_hyd`` is
    the previous step's total hydrostatic potential for the dynamic-pressure
    EOS.  ``select_3d_cori_scheme`` and ``f_cori_cos`` are the 3-D Coriolis
    controls resolved once from configuration and fixed geometry; ``None``
    keeps each module's configured value. ``momentum_pressure_forcing`` is
    the stepper's resolved pressure switch and must be passed explicitly:
    disabling it suppresses momentum gradients while retaining hydrostatic
    diagnostics and pressure history.  ``pmep_r`` is MITgcm's ``PmEpR``
    [kg m-2 s-1], the freshwater volume flux of a nonlinear free surface with
    real freshwater flux that the surface tracer forcing reads
    (``external_forcing_surf.F:261-288``); the stepper states it exactly for
    such a run and the forcing module that is resolved for one requires it.
    ``kappa_ru_floor`` [batch, 1, ny, nx+1] and ``kappa_rv_floor``
    [batch, 1, ny+1, nx] are MITgcm's kappaRU/kappaRV at interface Nr+1,
    which the no-slip drag of a bottom cell on the last level reads
    (``mom_u_botdrag_coeff.F:72-85,113-128``); the stepper states them
    exactly when a vertical mixing scheme copies interface Nr there
    (``calc_viscosity.F:121-130``).  Otherwise interface Nr+1 is
    ``viscArNr(Nr)``, the drag module's own ``vertical_viscosity``.
    ``coriolis_u`` [batch, nz, ny, nx+1] and ``coriolis_v`` [batch, nz, ny+1, nx]
    are the velocities the flux-form Coriolis term reads in place of the
    state's: GGL90's residual velocity uRes/vRes under useLANGMUIR
    (``mom_fluxform.F:1083-1090``); the stepper states them exactly then.
    ``ice_area`` [batch, ny, nx] is the sea-ice fraction AREA (0 to 1, centres)
    the step's sea-ice component published, which surface restoring that stops
    under ice reads (``forcing_surf_relax.F:73-85``); the stepper states it
    exactly when a sea-ice component runs.
    """

    w_interface: torch.Tensor | None = None
    total_phi_hyd: torch.Tensor | None = None
    select_3d_cori_scheme: int | None = None
    f_cori_cos: float | None = None
    pmep_r: torch.Tensor | None = None
    kappa_ru_floor: torch.Tensor | None = None
    kappa_rv_floor: torch.Tensor | None = None
    coriolis_u: torch.Tensor | None = None
    coriolis_v: torch.Tensor | None = None
    ice_area: torch.Tensor | None = None
    momentum_pressure_forcing: bool = field(kw_only=True)


def resolved_input(inputs: "EvaluationInputs | None", name: str, configured):
    """``inputs.<name>`` when the evaluation states it, else ``configured``."""
    value = None if inputs is None else getattr(inputs, name)
    return configured if value is None else value


@dataclass(frozen=True)
class DiagnosticState:
    scalars: dict[str, float | torch.Tensor] = field(default_factory=dict)
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    fields: dict[str, Field] = field(default_factory=dict)
    #: Inputs of the evaluation this context belongs to (see EvaluationInputs).
    inputs: EvaluationInputs | None = None

    def set_scalar(self, name: str, value: float | torch.Tensor) -> None:
        self.scalars[name] = value

    def set_tensor(self, name: str, value: torch.Tensor) -> None:
        self.tensors[name] = value

    def set_field(self, field: Field) -> None:
        self.fields[field.name] = field
        self.tensors[field.name] = field.data


@dataclass(frozen=True)
class AuxiliaryState:
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class ForcingState:
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class PrognosticState:
    eta: Field
    u: Field
    v: Field
    temp: Field
    salt: Field

    @property
    def batch_size(self) -> int:
        """Size of the leading tensor dimension.

        Regional-only semantics: this is the ensemble/batch dimension. Under the
        cube FACE-FIRST layout the leading dim is the face-subset count (6
        global, or the rank-local face count 1/2/3), NOT an ensemble batch —
        see ``validate_against``, which special-cases that layout.
        """
        return int(self.eta.data.shape[0])

    def validate_against(self, grid: StructuredCGrid) -> None:
        if getattr(grid, "face_count", 1) == 6:
            # Cube layout is FACE-FIRST: the face dim occupies the leading
            # "batch" slot, so the prognostic tensors are the grid shape
            # properties with no extra batch dim:
            #   eta=[d,ny,nx]  u=[d,nz,ny,nx+1]  v=[d,nz,ny+1,nx]  T/S=[d,nz,ny,nx]
            # where d is a face SUBSET count: 6 for the global state, or the
            # rank-local face count of the face-aligned decomposition
            # (SUPPORTED_RANKS {2,3,6} -> d in {3,2,1}).  A rank-local state is
            # validated against the same face_count=6 grid (e.g. checkpoint
            # restore in defgcm/checkpoint/capture.py), so the leading dim
            # is checked as "a divisor of 6", with the trailing dims exact.
            # The regional formula below would double-count the face dim
            # (tracer_shape[:-2] already carries the 6) and reject every state.
            d = int(self.eta.data.shape[0])
            if d not in (1, 2, 3, 6):
                raise ValueError(
                    f"cube state leading dim {d} is not a face subset of 6 "
                    "(expected 1, 2, 3, or 6 faces)"
                )
            nz = grid.nz
            expected_eta = (d, grid.ny, grid.nx)
            expected_u = (d, nz, grid.ny, grid.nx + 1)
            expected_v = (d, nz, grid.ny + 1, grid.nx)
            expected_scalar = (d, nz, grid.ny, grid.nx)
        else:
            expected_eta = (self.batch_size, *grid.center_shape)
            expected_u = (self.batch_size, *grid.tracer_shape[:-2], *grid.u_shape)
            expected_v = (self.batch_size, *grid.tracer_shape[:-2], *grid.v_shape)
            expected_scalar = (self.batch_size, *grid.tracer_shape)
        if tuple(self.eta.data.shape) != expected_eta:
            raise ValueError(f"eta shape {tuple(self.eta.data.shape)} != {expected_eta}")
        if tuple(self.u.data.shape) != expected_u:
            raise ValueError(f"u shape {tuple(self.u.data.shape)} != {expected_u}")
        if tuple(self.v.data.shape) != expected_v:
            raise ValueError(f"v shape {tuple(self.v.data.shape)} != {expected_v}")
        if tuple(self.temp.data.shape) != expected_scalar:
            raise ValueError(f"temp shape {tuple(self.temp.data.shape)} != {expected_scalar}")
        if tuple(self.salt.data.shape) != expected_scalar:
            raise ValueError(f"salt shape {tuple(self.salt.data.shape)} != {expected_scalar}")

    def with_tensors(
        self,
        *,
        eta: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        temp: torch.Tensor | None = None,
        salt: torch.Tensor | None = None,
    ) -> "PrognosticState":
        return PrognosticState(
            eta=self.eta.with_data(eta if eta is not None else self.eta.data),
            u=self.u.with_data(u if u is not None else self.u.data),
            v=self.v.with_data(v if v is not None else self.v.data),
            temp=self.temp.with_data(temp if temp is not None else self.temp.data),
            salt=self.salt.with_data(salt if salt is not None else self.salt.data),
        )

    def apply_tendency(self, tendency: "StateTendency", scale: float) -> "PrognosticState":
        return self.with_tensors(
            eta=self.eta.data + scale * tendency.eta,
            u=self.u.data + scale * tendency.u,
            v=self.v.data + scale * tendency.v,
            temp=self.temp.data + scale * tendency.temp,
            salt=self.salt.data + scale * tendency.salt,
        )


@dataclass(frozen=True)
class StateTendency:
    eta: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    temp: torch.Tensor
    salt: torch.Tensor

    @classmethod
    def zeros_like(cls, state: PrognosticState) -> "StateTendency":
        return cls(
            eta=torch.zeros_like(state.eta.data),
            u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data),
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )

    def __add__(self, other: "StateTendency") -> "StateTendency":
        return StateTendency(
            eta=self.eta + other.eta,
            u=self.u + other.u,
            v=self.v + other.v,
            temp=self.temp + other.temp,
            salt=self.salt + other.salt,
        )

    def scaled(self, factor: float) -> "StateTendency":
        return StateTendency(
            eta=self.eta * factor,
            u=self.u * factor,
            v=self.v * factor,
            temp=self.temp * factor,
            salt=self.salt * factor,
        )

    def apply(self, state: PrognosticState, scale: float) -> PrognosticState:
        return state.apply_tendency(self, scale)

    @classmethod
    def accumulator(cls, state: PrognosticState) -> "TendencyAccumulator":
        """Open an assembly over ``state``'s shapes. See ``TendencyAccumulator``."""
        return TendencyAccumulator(state)


def blend_states(
    state_a: PrognosticState,
    state_b: PrognosticState,
    weight_a: float,
    weight_b: float,
) -> PrognosticState:
    return state_a.with_tensors(
        eta=weight_a * state_a.eta.data + weight_b * state_b.eta.data,
        u=weight_a * state_a.u.data + weight_b * state_b.u.data,
        v=weight_a * state_a.v.data + weight_b * state_b.v.data,
        temp=weight_a * state_a.temp.data + weight_b * state_b.temp.data,
        salt=weight_a * state_a.salt.data + weight_b * state_b.salt.data,
    )


# ---------------------------------------------------------------------------
# Tendency assembly seam
# ---------------------------------------------------------------------------
#
# Every slow-tendency assembly in this repository has the same shape:
#
#     tendency = StateTendency.zeros_like(state)
#     for module in modules:
#         tendency = tendency + module.evaluate(...)
#
# Two units are in play and mixing them misreads the seam by 5x: an **assembly** is one accumulator, opened once and folded once,
# while a ``StateTendency`` carries five fields -- so one assembly issues five
# ``torch.zeros_like`` calls and one module contribution issues five adds. Let
# ``n`` be the bytes of one whole tendency set: 2.48 GiB at N=960 (eta + u + v
# + T + S, float64). Per assembly of width M the pairwise chain moves
# ``n + 3Mn`` (each add reads 2n, writes n); a single n-ary add moves
# ``(M+1)n``, so the ceiling is ``2Mn``.
#
# Assembly widths depend on configuration. ASSEMBLY_WIDTHS records the widths
# actually folded in each run. Module-local zeros_like returns, including
# equation_of_state and vertical_velocity, are outside this accumulator.
#
# That is the honest size of this particular prize; the seam is retained mainly
# because it is the cheapest place to calibrate whether the byte model cashes
# out in wall clock at production shapes.
#
# ``TendencyAccumulator`` is the one place the arms differ, so a call site
# cannot disagree about which one it got. Three arms, all summing in the same
# left-to-right order:
#
#   pairwise  the shipped chain, reproduced exactly -- the A/B reference
#   inplace   ``parts[0].clone()`` then ``add_`` -- same traffic per add but no
#             per-add allocation, which separates allocator cost from bandwidth
#   fused     one compiled n-ary add per field -- reads M*n, writes n
#
# Bit-for-bit: ``inplace`` and ``fused`` drop the leading ``0 +``. For every
# finite value ``0.0 + x == x``, and the summation order is unchanged, so the
# only representable difference is the sign of zero: ``0.0 + (-0.0)`` is
# ``+0.0`` while the fused arms keep ``-0.0``. Additions cannot contract into
# an FMA, so the compiled arm has no fp-contraction exposure either
# (contrast the multiply-then-add cores, where ``TRITON_DEFAULT_FP_FUSION=0``
# was required). Both claims are checked, not assumed.
#
# Cost paid: the fused arm must hold all M contributions live at once instead
# of folding each into the accumulator as it arrives, so peak memory rises by
# (M-1) tendency sets, about 0.5 GiB each at N=960. Runs near the capacity
# wall must read the peak-memory column, not just the timing column.

_TENDENCY_FIELDS = tuple(f.name for f in dataclasses.fields(StateTendency))

_ASSEMBLY_ARMS = ("pairwise", "inplace", "fused")

#: Ignition ledger. A speed-up with no proof its kernel ran is not evidence, so
#: every ``finish()`` records which arm executed and how wide the assembly was.
ASSEMBLY_IGNITION: dict[str, int] = dict.fromkeys(_ASSEMBLY_ARMS, 0)
#: Pre-seeded so the ledger always states the fallback count. A key that
#: only appears when it fires reads as "absent" and "zero" alike.
ASSEMBLY_IGNITION["fused-fallback"] = 0
ASSEMBLY_WIDTHS: dict[int, int] = {}


@lru_cache(maxsize=1)
def assembly_arm() -> str:
    """The selected assembly arm. Single source for gate and kernel alike.

    Read once and cached: an arm that could change mid-run would make the
    ignition ledger unreadable. Fail-closed on an unknown value rather than
    silently falling back to the default, because a typo that silently selects
    the baseline is exactly how an accelerated arm gets measured as slow.
    """
    arm = os.environ.get("DEFGCM_ASSEMBLY_ARM", "pairwise").strip().lower()
    if arm not in _ASSEMBLY_ARMS:
        raise ValueError(
            f"DEFGCM_ASSEMBLY_ARM={arm!r} is not one of {_ASSEMBLY_ARMS}"
        )
    return arm


def _add2(a, b):
    return a + b


def _add3(a, b, c):
    return a + b + c


def _add4(a, b, c, d):
    return a + b + c + d


def _add5(a, b, c, d, e):
    return a + b + c + d + e


def _add6(a, b, c, d, e, f):
    return a + b + c + d + e + f


def _add7(a, b, c, d, e, f, g):
    return a + b + c + d + e + f + g


def _add8(a, b, c, d, e, f, g, h):
    return a + b + c + d + e + f + g + h


#: One real, named function per arity, spelled out rather than generated.
#: ``exec``-built functions carry a synthetic filename, Dynamo treats them as
#: skipped, and routes every call through the single shared ``wrap_inline.inner``
#: frame in ``torch/_dynamo/external_utils.py`` -- one code object for the whole
#: process, guarded on ``len(args)``, against a default ``cache_size_limit`` of
#: 8. Three arities were enough to exhaust it, and exhaustion **raises**
#: (``torch._dynamo.exc.Unsupported: cache_size_limit reached``) rather than
#: falling back to eager. Real defs in a real file each get their own cache.
_NARY_ADD = {2: _add2, 3: _add3, 4: _add4, 5: _add5, 6: _add6, 7: _add7, 8: _add8}
_MAX_NARY = max(_NARY_ADD)


@lru_cache(maxsize=None)
def _compiled_nary_add(width: int):
    """The compiled left-to-right sum of ``width`` tensors.

    ``dynamic`` is left at its default rather than pinned to ``False``. Dynamo
    compiles the first shape statically and only generalises if a second shape
    for the same arity turns up, which is the behaviour this seam wants: a
    production run assembles two shape classes (2-D eta, 3-D interior fields)
    and stays static, while a process that sweeps many grids -- a test session,
    a resolution ladder -- generalises instead of recompiling until it hits
    ``cache_size_limit`` and **raises**.
    """
    return torch.compile(_NARY_ADD[width], fullgraph=True)


class TendencyAccumulator:
    """Collects module tendencies and folds them once, under the chosen arm."""

    __slots__ = ("_state", "_parts")

    def __init__(self, state: "PrognosticState") -> None:
        self._state = state
        self._parts: list[StateTendency] = []

    def add(self, tendency: "StateTendency") -> None:
        self._parts.append(tendency)

    def finish(self) -> "StateTendency":
        parts = self._parts
        arm = assembly_arm()
        # Ignition bookkeeping only outside the compiled step: a module-level
        # dict mutated inside the traced region becomes a Dynamo guard on its
        # contents (``ASSEMBLY_WIDTHS[4] == 6``) that fails on every step and
        # recompiles the whole step frame until the limit sends it eager.  The
        # runner takes its warm-up steps eagerly, so the arm still shows as ignited.
        if not torch._dynamo.is_compiling():
            ASSEMBLY_IGNITION[arm] += 1
            ASSEMBLY_WIDTHS[len(parts)] = ASSEMBLY_WIDTHS.get(len(parts), 0) + 1
        if not parts:
            return StateTendency.zeros_like(self._state)
        if arm == "pairwise":
            total = StateTendency.zeros_like(self._state)
            for part in parts:
                total = total + part
            return total
        if arm == "inplace":
            return StateTendency(
                **{
                    name: _inplace_sum([getattr(part, name) for part in parts])
                    for name in _TENDENCY_FIELDS
                }
            )
        return StateTendency(
            **{
                name: _fused_sum([getattr(part, name) for part in parts])
                for name in _TENDENCY_FIELDS
            }
        )


def _report_assembly_ignition() -> None:
    """Emit the ignition ledger, but only when an arm was explicitly requested.

    A timing number whose arm cannot be proven is not a reading. ``accel_sweep``
    records wall clock and peak memory but knows nothing about this seam, so the
    proof has to come from the library. It stays silent when nobody asked for a
    non-default arm, so production logs are unchanged.
    """
    if "DEFGCM_ASSEMBLY_ARM" not in os.environ:
        return
    import json

    print("DEFGCM_ASSEMBLY " + json.dumps({
        "arm": assembly_arm(),
        "ignition": ASSEMBLY_IGNITION,
        "widths": {str(k): v for k, v in sorted(ASSEMBLY_WIDTHS.items())},
    }, sort_keys=True), flush=True)


atexit.register(_report_assembly_ignition)


def _inplace_sum(tensors: "Sequence[torch.Tensor]") -> torch.Tensor:
    # ``clone`` rather than accumulating into ``tensors[0]``: a module may hand
    # back a cached or aliased tensor, and folding into it would corrupt the
    # producer's own state -- a silent, step-delayed corruption.
    total = tensors[0].clone()
    for tensor in tensors[1:]:
        total.add_(tensor)
    return total


def _fused_sum(tensors: "Sequence[torch.Tensor]") -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0].clone()
    if len(tensors) <= _MAX_NARY:
        return _guarded_nary_add(tensors)
    # Wider than the widest compiled kernel: fold in chunks. Chaining
    # ``nary(t0..t7)`` then ``nary(that, t8, ...)`` still evaluates strictly
    # left to right, so the summation order is the same one the pairwise arm
    # produces -- the chunking changes how many kernels run, not the arithmetic.
    total = _guarded_nary_add(tensors[:_MAX_NARY])
    rest = list(tensors[_MAX_NARY:])
    while rest:
        chunk, rest = rest[: _MAX_NARY - 1], rest[_MAX_NARY - 1 :]
        total = _guarded_nary_add([total, *chunk])
    return total


def _guarded_nary_add(tensors: "Sequence[torch.Tensor]") -> torch.Tensor:
    """Compiled sum, degrading to the eager chain instead of killing the run.

    ``torch.compile`` does not always fall back on its own: ``fullgraph=True``
    raises on a graph break, and an exhausted Dynamo cache raises
    ``Unsupported`` no matter how the call is configured. Neither belongs in a
    long integration, so both are caught.

    The fallback is **counted, not silent**. An accelerator that quietly stops
    accelerating produces a slow arm labelled fast, which is the failure mode
    this repository has already paid for once; ``ASSEMBLY_IGNITION`` therefore
    carries a ``fused-fallback`` tally, and a reading whose tally is non-zero is
    not a reading of the fused arm.

    A live forward mode (``forward_mode_active``) takes the same counted
    fallback before the compiled sum: Inductor's kernel returns the total
    without the tangent and without an error, and the eager chain is the same
    left-to-right sum (bitwise the same total) carrying it.  The forward-mode
    run is refused compilation at the runner, so this branch runs eagerly.

    Outside a compiled step the compiled sum is called only with autograd out of
    the picture: Dynamo guards each cache entry on the grad mode and on every
    argument's ``requires_grad``, so a reverse-mode run (a window's primal pass,
    a parameter control whose tendencies require grad while the state's do not)
    adds one entry per pattern on top of the two shape classes, and a function
    compiled with ``fullgraph=True`` raises ``FailOnRecompileLimitHit`` (not an
    ``Unsupported``) at the default limit of 8 that a non-compiling run keeps.
    An argument that requires grad therefore takes the counted eager chain, and
    the rest run under ``no_grad`` (no argument requires grad, so the total is
    the same), which leaves the two shape classes as the only entries.  Inside
    a compiled step the call is inlined into the step's graph as before.
    """
    if forward_mode_active():
        return _fused_fallback(tensors, "forward mode: the compiled sum drops tangents")
    context = contextlib.nullcontext()
    if not torch._dynamo.is_compiling():
        if any(tensor.requires_grad for tensor in tensors):
            return _fused_fallback(tensors, "reverse mode: the compiled sum would recompile per requires_grad pattern")
        context = torch.no_grad()
    try:
        with context:
            return _compiled_nary_add(len(tensors))(*tensors)
    except torch._dynamo.exc.Unsupported as exc:  # pragma: no cover - env-dependent
        return _fused_fallback(tensors, exc)


def _fused_fallback(tensors: "Sequence[torch.Tensor]", cause) -> torch.Tensor:
    """The eager chain in place of the compiled sum, counted in ``fused-fallback``."""
    ASSEMBLY_IGNITION["fused-fallback"] = (
        ASSEMBLY_IGNITION.get("fused-fallback", 0) + 1
    )
    if ASSEMBLY_IGNITION["fused-fallback"] == 1:
        print(f"DEFGCM_ASSEMBLY fused arm fell back to eager: {cause}",
              flush=True)
    return _eager_chain(tensors)


def _eager_chain(tensors: "Sequence[torch.Tensor]") -> torch.Tensor:
    """``_addN`` evaluated eagerly: out-of-place ``+`` left to right, so every
    partial sum promotes like the compiled kernel's (``_inplace_sum`` keeps the
    first term's dtype).  Two terms at least, as for ``_addN``; the result is a
    new tensor, never an alias of a term."""
    total = tensors[0] + tensors[1]
    for tensor in tensors[2:]:
        total = total + tensor
    return total
