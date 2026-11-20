"""Frozen payload types at the eager/traced boundaries of one time step.

The execution sequence is prepare (eager), kernel (traced), then finalize
(eager). StepPlan carries tensors and run-invariant choices into the kernel;
StepOutcome carries state updates and tensor diagnostics back to finalize.
HostPlan retains changing host timestamps and other values that must stay
outside the traced region. A Python scalar reachable through a frozen
dataclass can acquire a Dynamo value guard even when the kernel never reads it.

BoundaryPayload holds the sampled component inputs for the step. Solver
residuals and iteration counts remain scalar tensors, while deferred_checks
owns its run-scoped error buffer. Cross-step outputs are published explicitly
at finalization rather than changing the traced plan's Python structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from defgcm.accel import StepGates
    from defgcm.state import PrognosticState
    from defgcm.steppers.base import TimeStepBundle
    from defgcm.steppers.ifs.kpp_inputs import KppStepForcing
    from defgcm.steppers.implicit_free_surface_nonlin import SurfaceEtaBoundary


#: One shared frozen empty mapping, for a plan whose prepare resolved no
#: constants and for an outcome that staged no retained values.
EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})


#: The declared exits of the traced step, as
#: ``(module, attribute)`` pairs.  This is their single source.
#: Adding an exit means adding a line here.
DECLARED_EXITS: tuple[tuple[str, str], ...] = (
    ("defgcm.steppers.step_segments", "_solve_cg2d_and_stage"),
    ("defgcm.steppers.ifs.nh", "_solve_cg3d_and_stage"),
    ("defgcm.steppers.ifs.pressure", "_picard_update_rms"),
)


def declared_exit_names() -> tuple[str, ...]:
    """The declared exits' names, each checked against the function it names.

    Resolving them is what keeps the list honest: rename an exit and this raises
    instead of leaving the gate silently allowing a name nothing answers to.
    The imports are deferred because the steppers import this module.
    """
    import importlib

    names: list[str] = []
    for module_name, attribute in DECLARED_EXITS:
        module = importlib.import_module(module_name)
        target = getattr(module, attribute, None)
        if not callable(target):
            raise RuntimeError(
                f"declared exit {module_name}.{attribute} is not a callable; "
                "DECLARED_EXITS is out of step with the code"
            )
        names.append(attribute)
    return tuple(names)


@dataclass(frozen=True)
class ShapeContract:
    """The shapes the traced step is allowed to assume are run-invariant.

    ``halo`` is the grid-level halo extent when the grid declares one.  No grid
    does today -- every ``HaloExtentV1`` in the tree is built at its own call
    site (``geometry/exchange.py``) -- so it is ``None``, which reads "this grid
    declares no halo extent", not "the halo is zero".  Package C resolves the
    per-call-site extents into this field.
    """

    nx: int
    ny: int
    nz: int
    face_count: int
    halo: Any
    batch: int | None

    @classmethod
    def from_grid(cls, grid: Any, state: Any = None) -> "ShapeContract":
        batch = None
        if state is not None:
            data = getattr(getattr(state, "eta", None), "data", None)
            if isinstance(data, torch.Tensor) and data.ndim >= 1:
                batch = int(data.shape[0])
        return cls(
            nx=int(grid.nx),
            ny=int(grid.ny),
            nz=int(grid.nz),
            face_count=int(getattr(grid, "face_count", 1)),
            halo=getattr(grid, "halo_extent", None),
            batch=batch,
        )


@dataclass(frozen=True)
class StepTimes:
    """The clocks of one step as the traced region may see them.

    0-d float64 tensors, plus the ``bundle`` whose ``dt`` values are constant
    for the whole run and therefore guard once and hold.  The host halves of
    the same two samples are :class:`HostTimes`, on the other side of the
    boundary: a Python float here is a per-step guard on this frame even with
    nothing reading it ().
    """

    bundle: "TimeStepBundle"
    current: torch.Tensor
    future: torch.Tensor

    @classmethod
    def sample(
        cls, *, current_seconds: float, bundle: "TimeStepBundle"
    ) -> "StepTimes":
        current = float(current_seconds)
        future = current + float(bundle.dt_clock)
        return cls(
            bundle=bundle,
            current=torch.tensor(current, dtype=torch.float64),
            future=torch.tensor(future, dtype=torch.float64),
        )


@dataclass(frozen=True)
class HostTimes:
    """The same two clocks as host floats, for prepare, finalize and the
    eager legacy paths.

    The pair with :class:`StepTimes` exists because the two consumers have
    opposite requirements: a tensor cannot index a Python-side bundle table or
    reach an external provider declared ``f(time_seconds: float)``, and a
    per-step Python float on an object the tracer walks is what Dynamo
    specialises on.  Splitting them by *object*, not by field, is what makes
    the second requirement structural.
    """

    current_seconds: float
    future_seconds: float
    bundle: "TimeStepBundle"

    @classmethod
    def sample(
        cls, *, current_seconds: float, bundle: "TimeStepBundle"
    ) -> "HostTimes":
        current = float(current_seconds)
        return cls(
            current_seconds=current,
            future_seconds=current + float(bundle.dt_clock),
            bundle=bundle,
        )


@dataclass(frozen=True)
class ComponentSample:
    """One time-dependent object's device payload at one clock.

    ``attributes`` contains the complete attribute set named by the component's
    SamplingRule, including constant/empty channels. The common eager preparer
    resolves START/END for the entry and later application events. Host clocks
    and cache keys are kept outside this payload.
    """

    component: Any
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class ClockSample:
    """Everything prepare sampled at one clock.

    ``sampled`` is the positive declaration "this model context ran the eager
    pre-step". It is ``False`` on the cube path, which never reaches install.
    Regional execution requires declared samples and rejects an absent payload.
    It is a Python bool on the kernel's argument, in the same category as the
    frozen gate booleans: fixed for the whole run, so it guards once and holds.

    ``obc_eta`` holds four tensor samples for nonlinear free-surface geometry.
    Initialization consumes the current sample; the step-end surface refresh
    consumes the application sample. Linear/cube paths do not need this field.
    """

    sampled: bool
    components: tuple[ComponentSample, ...]
    obc_eta: SurfaceEtaBoundary | None = None

    def install(self) -> None:
        """Install this clock's device payload on its components.

        Called at entry and at subsequent boundary applications. The eager
        prepare also installs the entry sample before the kernel is entered.
        The values are graph inputs reached through ``StepPlan.boundary``; none
        of them is a Python scalar (``prepare_step_times`` routes host state to
        its own eager install), so this installs tensors under identity guards
        and reads nothing from the model.
        """
        for entry in self.components:
            for name, value in entry.attributes.items():
                if getattr(entry.component, "uses_stevens", False):
                    from defgcm.steppers.ifs.carry import _clone_retained_value
                    value = _clone_retained_value(value)
                object.__setattr__(entry.component, name, value)

    @classmethod
    def absent(cls) -> "ClockSample":
        """No eager pre-step ran for this clock."""
        return cls(sampled=False, components=())


@dataclass(frozen=True)
class BoundaryPayload:
    """The device half of the two time samples prepare took.

    The step body reads its boundary and sponge payload from here -- from the
    plan, on the kernel's own signature -- instead of from a dict the model
    carried and the traced step reached into.
    """

    current: ClockSample
    future: ClockSample

    @classmethod
    def empty(cls) -> "BoundaryPayload":
        return cls(current=ClockSample.absent(), future=ClockSample.absent())


@dataclass(frozen=True)
class StepPlan:
    """Everything the traced step needs and nothing it must decide.

    This object is the kernel's second argument, so everything reachable from
    it is guarded.  Tensors and run-invariant constants only: the gates are
    booleans frozen for the run, ``shapes`` are integers the run does not
    change, ``times.bundle`` holds the run's fixed ``dt`` values, and the two
    clocks and the step counter are 0-d tensors.  What changes per step and is
    a Python number belongs on :class:`HostPlan`.
    """

    gates: "StepGates"
    times: StepTimes
    step_index: torch.Tensor
    shapes: ShapeContract
    constants: Mapping[str, torch.Tensor]
    boundary: BoundaryPayload
    kpp_forcing_inputs: KppStepForcing | None
    boundary_history: Any = None


@dataclass(frozen=True)
class HostPlan:
    """What prepare resolved, in its two halves.

    ``plan`` is the half the kernel receives; ``times`` is the half it must not
    -- the host clocks prepare sampled, which finalize and the eager paths
    read.  ``build_step_plan`` returns this, so a caller has to name ``.plan``
    to reach the traced side, and adding a host value to the step is a choice
    between the two halves rather than a field appended to whichever dataclass
    was nearest.
    """

    plan: StepPlan
    times: HostTimes


@dataclass(frozen=True)
class StepDiagnostics:
    """The traced step's observable outputs, all tensors.

    ``cg2d_residual``/``cg2d_iters`` are 0-d (float64 / int64), and
    ``cg3d_residual`` is the internal non-hydrostatic solve's last residual.
    ``None`` is the positive declaration "this configuration ran no such solve"
    -- the non-implicit steppers, and every hydrostatic run for the cg3d slot --
    not "the value was unavailable". ``cg3d_iters`` and
    ``cg3d_first_residual`` carry the other NH solve readings.
    ``eta_n`` is the detached CG2D elevation in metres before exact continuity;
    ``dphi_nh`` is the detached surface-separated NH pressure in m2 s-2.

    The two residuals are the ``production`` level's whole probe: the PCG loops
    test convergence on the host, so both numbers are already there when the
    step boundary reads them (defgcm/deferred_checks.py).

    There is no ``solver_codes`` slot here.  Solver error
    codes are still a kernel output, but the buffer they are written into is
    owned by ``deferred_checks`` and read back asynchronously, one step behind,
    so that the boundary never synchronises ; routing it
    through this object would have been a second handle on the same run-scoped
    tensor and would have forced the read to happen the moment the step ended.
    """

    cg2d_residual: torch.Tensor | None
    cg2d_iters: torch.Tensor | None
    cg3d_residual: torch.Tensor | None = None
    cg3d_iters: torch.Tensor | None = None
    cg3d_first_residual: torch.Tensor | None = None
    eta_n: torch.Tensor | None = None
    dphi_nh: torch.Tensor | None = None

    @classmethod
    def absent(cls) -> "StepDiagnostics":
        """No solve ran in this step; every slot is explicitly empty."""
        return cls(
            cg2d_residual=None,
            cg2d_iters=None,
            cg3d_residual=None,
        )


@dataclass(frozen=True)
class StepOutcome:
    """What the traced step hands back to finalize.

    ``retained`` names the cross-step values the kernel staged and finalize must
    take ownership of. Solver and pressure readouts leave through
    ``diagnostics``; they are not restart history.
    """

    state: "PrognosticState"
    diagnostics: StepDiagnostics
    retained: Mapping[str, Any]


__all__ = [
    "BoundaryPayload",
    "ClockSample",
    "ComponentSample",
    "DECLARED_EXITS",
    "EMPTY_MAPPING",
    "HostPlan",
    "HostTimes",
    "ShapeContract",
    "StepDiagnostics",
    "StepOutcome",
    "StepPlan",
    "StepTimes",
    "declared_exit_names",
]
