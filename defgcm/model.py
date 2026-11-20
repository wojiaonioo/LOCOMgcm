from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, fields, replace
import os

import torch
import torch.distributed as dist

from defgcm.accel.tensor_context import ordinary_tensor_mode
from defgcm.boundaries import BoundaryCondition, StructuredBoundaryCondition
from defgcm.boundaries.open import SideBoundaryTensorSet
from defgcm.capabilities import ModelCapabilities
from defgcm.distributed.comm_adjoint import session_for_grid
from defgcm.diagnostics import (
    BoundaryFluxBudgetDiagnostic,
    CFLEstimateDiagnostic,
    DiagnosticSuite,
    FreeSurfaceEnergyDiagnostic,
    KineticEnergyDiagnostic,
    SurfaceMassDiagnostic,
    ThreeDContinuityDiagnostic,
    TracerMassDiagnostic,
)
from defgcm.dynamics import (
    AssemblyBucket,
    AssemblyPolicy,
    BetaPlaneCoriolis,
    DiagnosticVerticalVelocity,
    FCoriolis,
    HorizontalTracerAdvection,
    HydrostaticPressureGradient,
    LinearDrag,
    LinearEquationOfState,
    LinearFreeSurfaceContinuity,
    LinearFreeSurfacePressureGradient,
    MomentumAdvection,
    MomentumTermClass,
    TendencyModule,
    TendencyPhase,
    TendencyStepDomain,
    VerticalMixing,
    VerticalTracerTransport,
)
from defgcm.dynamics.bottom_drag import bind_momentum_core_ke_scheme
from defgcm.dynamics.convective_adjustment import convective_adjustment_of, validate_convective_adjustment_composition
from defgcm.dynamics.equation_of_state import validate_dynamic_pressure_composition
from defgcm.dynamics.vector_invariant_momentum import validate_vorticity_composition
from defgcm.dynamics.surface_pressure_load import validate_surface_pressure_load_composition
from defgcm.dynamics.surface_relaxation import validate_relaxation_composition
from defgcm.seaice.step_state import validate_seaice_composition
from defgcm.fields import Field, FieldLocation
from defgcm.geometry import BoundaryTopology, BoundaryType, CubedSphereGrid, MaskSet, MetricSet, StructuredCGrid, ZLevelCoordinate
from defgcm.operators import TracerFluxScheme
from defgcm.registries import LedgerBundle
from defgcm.sampling import SampleClock, sampled_host_paths, sampling_components, sampling_rule
from defgcm.state import DiagnosticState, EvaluationInputs, PrognosticState, StateTendency
from defgcm.steppers import SSPRK3Stepper, SplitExplicitStepper, TimeStepper
from defgcm.steppers.base import TimeStepBundle
from defgcm.steppers.ifs.carry import _clone_retained_value
from defgcm.steppers.ifs import faces
from defgcm.steppers.ifs.columns import validate_convection
from defgcm.steppers.ifs.kpp_inputs import KppStepForcing
from defgcm.steppers.implicit_free_surface import (
    ImplicitFreeSurfaceStepper,
)
from defgcm.steppers.implicit_free_surface_cube import refuse_unimplemented_cube_kwargs
from defgcm.steppers.ifs.admission import (
    _composition_tracer_names,
    _validate_runtime_implicit_vertical_advection,
    resolve_package_modules,
    resolve_vector_invariant_momentum,
    resolve_volume_freshwater,
    validate_negative_salinity_check,
    validate_rotation_period,
    validate_tracer_ab_scheme,
    validate_tracer_advection_scheme,
)
from defgcm.step import (
    EMPTY_MAPPING,
    BoundaryPayload,
    ClockSample,
    ComponentSample,
    StepDiagnostics,
    StepOutcome,
    StepPlan,
    build_step_plan,
    finalize_step,
)


# Guard before building trace messages. Even a disabled Python callback can
# specialize a Dynamo frame for each module name during checkpoint replay.
_MODEL_TRACE_ENABLED = os.environ.get("DEFGCM_DIST_TRACE") == "1"


def _model_trace(message: str) -> None:
    if not _MODEL_TRACE_ENABLED:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(f"[rank{rank}] model: {message}", flush=True)


#: Values that are host state even when they sit inside a container.
_HOST_SCALARS = (bool, int, float, complex, str, bytes, torch.dtype, torch.device)


def _carries_tensor(value: object) -> bool:
    """Check undeclared metadata/cache placement, never scientific clock choice.

    Inspect every reachable value, even after finding a tensor, so an unknown
    record cannot hide behind another field. Object identity bounds aliases and
    cycles. Mixed host/tensor records remain payload; the boundary purity gate
    separately reports any changing host scalars inside their attributes.
    """
    # ``defgcm.io`` imports this module (``io/hycom.py``), so the ERA5
    # record type cannot be imported at module scope without a cycle.
    from defgcm.io.era5 import ERA5BulkInputs

    seen: set[int] = set()

    def visit(item: object, path: str) -> bool:
        if isinstance(item, torch.Tensor):
            return True
        if item is None or isinstance(item, _HOST_SCALARS):
            return False
        if isinstance(item, Mapping):
            children = ((f"[{key!r}]", child) for key, child in item.items())
        elif isinstance(item, (tuple, list, set, frozenset)):
            children = ((f"[{index}]", child) for index, child in enumerate(item))
        elif type(item) in (SideBoundaryTensorSet, ERA5BulkInputs):
            children = ((f".{field.name}", getattr(item, field.name))
                        for field in fields(item))
        else:
            raise TypeError(f"unregistered boundary payload type {type(item).__name__} at {path}")
        if id(item) in seen:
            return False
        seen.add(id(item))
        contains_tensor = False
        for suffix, child in children:
            child_contains_tensor = visit(child, path + suffix)
            contains_tensor = contains_tensor or child_contains_tensor
        return contains_tensor

    return visit(value, "payload")


@dataclass
class BoundaryBudgetAccumulator:
    temp_boundary_reset_integral: float = 0.0
    salt_boundary_reset_integral: float = 0.0
    temp_sponge_integral: float = 0.0
    salt_sponge_integral: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "temp_boundary_reset_integral": self.temp_boundary_reset_integral,
            "salt_boundary_reset_integral": self.salt_boundary_reset_integral,
            "temp_sponge_integral": self.temp_sponge_integral,
            "salt_sponge_integral": self.salt_sponge_integral,
        }


@dataclass(frozen=True)
class RegionalOceanConfig:
    nx: int
    ny: int
    nz: int = 1
    lx: float = 100_000.0
    ly: float = 100_000.0
    depth: float = 100.0
    gravity: float = 9.81
    damping_rate: float | torch.Tensor = 0.0
    reference_density: float = 1_025.0
    # MITgcm rhoNil (linear-EOS reference density) vs rhoConst (Boussinesq
    # reference = ``reference_density`` here).  ``rho_nil=None`` -> rhoNil == rhoConst
    # (byte-identical to before); set it only for linear-EOS MIT alignment where MIT
    # uses its default rhoNil=999.8 with rhoConst=1035.  No effect on JMD95 (which
    # references rhoConst).  See HydrostaticPressureGradient.rho_nil.
    rho_nil: float | None = None
    thermal_expansion: float = 2.0e-4
    haline_contraction: float = 7.4e-4
    reference_temperature: float = 0.0
    reference_salinity: float = 0.0
    vertical_diffusivity: float = 0.0
    coriolis_parameter: float = 0.0
    default_dt: float = 10.0
    dt_mom: float | None = None
    dt_tracer: float | None = None
    dt_freesurf: float | None = None
    dt_clock: float | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    tracer_advection_scheme: TracerFluxScheme = TracerFluxScheme.UPWIND
    west_east_boundary: BoundaryType = BoundaryType.PERIODIC
    south_north_boundary: BoundaryType = BoundaryType.PERIODIC

    def time_step_bundle(self) -> TimeStepBundle:
        return TimeStepBundle(
            dt=self.default_dt,
            dt_mom=self.default_dt if self.dt_mom is None else self.dt_mom,
            dt_tracer=self.default_dt if self.dt_tracer is None else self.dt_tracer,
            dt_freesurf=self.default_dt if self.dt_freesurf is None else self.dt_freesurf,
            dt_clock=self.default_dt if self.dt_clock is None else self.dt_clock,
        )


def default_dynamics(config: RegionalOceanConfig, *, select_r_star: int) -> list[TendencyModule]:
    """The modules a model without ``dynamics_modules`` runs.

    The MITgcm data export reads the same list (``layer_d/_mit_switches.py``),
    so it sees what such a model runs.
    """
    # rhoNil (linear-EOS anomaly reference) defaults to rhoConst when unset.
    rho_nil = config.reference_density if config.rho_nil is None else config.rho_nil
    modules: list[TendencyModule] = [
        LinearFreeSurfaceContinuity(),
        LinearFreeSurfacePressureGradient(gravity=config.gravity),
        LinearEquationOfState(
            reference_density=rho_nil,
            thermal_expansion=config.thermal_expansion,
            haline_contraction=config.haline_contraction,
            reference_temperature=config.reference_temperature,
            reference_salinity=config.reference_salinity,
        ),
        HydrostaticPressureGradient(
            gravity=config.gravity,
            reference_density=config.reference_density,
            rho_nil=rho_nil,
            thermal_expansion=config.thermal_expansion,
            haline_contraction=config.haline_contraction,
            reference_temperature=config.reference_temperature,
            reference_salinity=config.reference_salinity,
        ),
        DiagnosticVerticalVelocity(),
        MomentumAdvection(select_r_star=select_r_star),
        HorizontalTracerAdvection(scheme=config.tracer_advection_scheme),
        # MITgcm tempVertAdvScheme defaults to tempAdvScheme (ini_parms.F:509-510),
        # i.e. vertical tracer advection follows the horizontal scheme.  Mirror that
        # here instead of the standalone "centered" default (which mismatched MIT's
        # upwind for tempAdvScheme=1 alignment cases).
        VerticalTracerTransport(
            vertical_advection_scheme=str(config.tracer_advection_scheme),
            # The DST3 family needs the tracer step for its Courant number
            # and rejects dt<=0 at construction, so the default assembly must
            # pass it or ModelConfig(tracer_advection_scheme=DST3) could not
            # build a model at all.  Same source as every other tracer-time
            # consumer (MIT deltaTLev); the donor-cell schemes ignore it.
            dt=float(config.time_step_bundle().dt_tracer),
        ),
        VerticalMixing(diffusivity=config.vertical_diffusivity),
        FCoriolis(coriolis_parameter=config.coriolis_parameter),
    ]
    if isinstance(config.damping_rate, torch.Tensor) or config.damping_rate > 0.0:
        modules.append(LinearDrag(rate=config.damping_rate))
    return modules


class TorchRegionalOceanModel:
    def __init__(
        self,
        config: RegionalOceanConfig,
        *,
        grid: StructuredCGrid | None = None,
        dynamics_modules: list[TendencyModule] | None = None,
        stepper: TimeStepper | None = None,
        diagnostics: DiagnosticSuite | None = None,
        boundary_condition: BoundaryCondition | None = None,
        cube_ctx: object | None = None,
        tracer_multi_dim_advection: bool | None = None,
    ) -> None:
        self.config = config
        self.tracer_multi_dim_advection = tracer_multi_dim_advection
        # Cube multi-rank routing context (CubeRankContext); always None on the
        # regional line.  The cube step body reads it via
        # ``getattr(model, "cube_ctx", None)`` (implicit_free_surface_cube.py) —
        # a pure carrier attribute, no behavioural branch in the model itself.
        self.cube_ctx = cube_ctx
        self.grid = grid if grid is not None else StructuredCGrid.rectangular(
            nx=config.nx,
            ny=config.ny,
            nz=config.nz,
            lx=config.lx,
            ly=config.ly,
            depth=config.depth,
            boundary_topology=BoundaryTopology(
                west_east=config.west_east_boundary,
                south_north=config.south_north_boundary,
            ),
            device=config.device,
            dtype=config.dtype,
        )
        self.boundary_condition = boundary_condition or StructuredBoundaryCondition()
        self.stepper = stepper or SSPRK3Stepper()
        if getattr(self.stepper, "implicit_int_grav_wave", False):
            raise NotImplementedError(
                "implicit_int_grav_wave (MITgcm implicitIntGravWave) is not "
                "supported on any LOCOMgcm grid: the implicit internal "
                "gravity-wave 3D pressure solve is not implemented."
            )
        self.dynamics_modules = tuple(dynamics_modules or self._default_dynamics())
        device = torch.device(config.device)
        dtype = config.dtype

        # Validate before runtime migration can sample any component.
        sampling_components(self.boundary_condition, self.dynamics_modules)

        # Runtime Migration
        self.grid = self._grid_to_runtime(self.grid, device=device, dtype=dtype)
        self.boundary_condition = self._dispatch_runtime_component(
            self.boundary_condition,
            device=device,
            dtype=dtype,
        )
        self.dynamics_modules = tuple(
            self._dispatch_runtime_component(module, device=device, dtype=dtype)
            for module in self.dynamics_modules
        )
        if getattr(self.grid, "face_count", 1) == 1:
            # Bottom drag reads the momentum core's KE (MIT inp_KE=.TRUE.).  The
            # cube step evaluates the module injected into
            # ``stepper.cube_static_state.bottom_drag``, which a rebinding
            # here would not reach.
            self.dynamics_modules = bind_momentum_core_ke_scheme(self.dynamics_modules)
        self.stepper = self._dispatch_runtime_component(self.stepper, device=device, dtype=dtype)
        self._validate_tracer_vertical_consumers()
        if isinstance(self.stepper, ImplicitFreeSurfaceStepper):
            adjustment = convective_adjustment_of(self.dynamics_modules)
            validate_convection(
                c_adj_freq=0.0 if adjustment is None else adjustment.c_adj_freq,
                ivdc_kappa=self.stepper.ivdc_kappa, scheme=self.stepper.vertical_mixing_scheme,
                kpp_enabled=self.stepper.kpp_enabled,
            )
            _validate_runtime_implicit_vertical_advection(
                self.stepper._implicit_vertical_advection_composition(self),
                source="TorchRegionalOceanModel.__init__",
            )
            validate_rotation_period(self.stepper, self.dynamics_modules)
            validate_negative_salinity_check(
                self.stepper, self.dynamics_modules, face_count=getattr(self.grid, "face_count", 1),
            )
            if (getattr(self.boundary_condition, "uses_orlanski", False)
                    and getattr(self.stepper.nonhydrostatic, "nonhydrostatic", False)):
                # ORLANSKI_<side> also radiates OBEw, applied by OBCS_APPLY_W
                # (orlanski_east.F:201-226, integr_continuity.F:287-299).
                raise NotImplementedError(
                    "an Orlanski open boundary with a nonhydrostatic stepper is not supported: "
                    "the OBEw radiation and OBCS_APPLY_W are not ported")
            object.__setattr__(
                self.stepper,
                "_resolved_vector_invariant_momentum",
                resolve_vector_invariant_momentum(
                    self.stepper.vector_invariant_momentum,
                    self.stepper.nonlin_free_surf,
                    self.dynamics_modules,
                ),
            )
            object.__setattr__(
                self.stepper,
                "_resolved_volume_freshwater",
                resolve_volume_freshwater(
                    self.stepper.nonlin_free_surf,
                    self.stepper.use_real_freshwater_flux,
                    self.dynamics_modules,
                ),
            )
            object.__setattr__(self.stepper, "_package_modules", resolve_package_modules(self))
        validate_surface_pressure_load_composition(
            modules=self.dynamics_modules,
            face_count=getattr(self.grid, "face_count", 1),
            use_real_freshwater_flux=getattr(self.stepper, "use_real_freshwater_flux", False),
        )
        validate_convective_adjustment_composition(
            modules=self.dynamics_modules, face_count=getattr(self.grid, "face_count", 1),
        )
        validate_vorticity_composition(modules=self.dynamics_modules, grid=self.grid)
        validate_relaxation_composition(modules=self.dynamics_modules)
        validate_dynamic_pressure_composition(
            modules=self.dynamics_modules, stepper=self.stepper,
            face_count=getattr(self.grid, "face_count", 1),
        )
        validate_seaice_composition(modules=self.dynamics_modules, stepper=self.stepper, grid=self.grid)
        # Cube stepper guard (reads the capability declaration): only the implicit free-surface stepper has the
        # ``face_count == 6`` cube dispatch with seam-aware exchange;
        # SSPRK/AB2/split-explicit would broadcast across the 6-face axis
        # silently and produce wrong results.  The ``grid_support`` class
        # attribute (defgcm/grid_support.py) is inherited by IFS
        # subclasses, preserving the original isinstance semantics.
        if getattr(self.grid, "face_count", 1) == 6 and "cube" not in getattr(
            type(self.stepper), "grid_support", ("latlon",)
        ):
            raise ValueError(
                "cubed-sphere grid requires the implicit free-surface stepper "
                f"(got {type(self.stepper).__name__}): SSPRK/AB2/split-explicit have no "
                "seam-aware exchange and would broadcast across the 6-face axis silently."
            )
        self.fast_dynamics_modules, self.slow_dynamics_modules = self._partition_dynamics_modules(self.dynamics_modules)
        self.slow_momentum_modules, self.slow_tracer_modules = self._partition_by_phase(self.slow_dynamics_modules)
        # A module the stepper never evaluates is a physics term that vanishes
        # with zero signal (the explicit-input requirement); refuse it here.
        evaluated = getattr(self.stepper, "evaluated_domains", frozenset({"fast", "slow"}))
        superseded = getattr(self.stepper, "superseded_module_names", frozenset())
        unplaced = [
            m for m in self.dynamics_modules
            if str(m.stepper_domain) not in evaluated and getattr(m, "name", None) not in superseded
        ]
        if unplaced:
            raise ValueError(
                f"{type(self.stepper).__name__} evaluates only the {sorted(evaluated)} tendency "
                f"domain(s); {[type(m).__name__ for m in unplaced]} declare(s) a domain it never "
                "runs and would be assembled but silently never applied. Give the module the "
                "stepper's domain or pick a stepper that evaluates it."
            )
        self.assembly_policy: AssemblyPolicy | None = None
        if isinstance(self.stepper, ImplicitFreeSurfaceStepper):
            is_cube = getattr(self.grid, "face_count", 1) == 6
            if is_cube:
                refuse_unimplemented_cube_kwargs(self.stepper)
                bottom_drag = next(
                    (
                        module
                        for module in self.slow_momentum_modules
                        if getattr(module, "momentum_term_class", None)
                        == MomentumTermClass.BOTTOM_DRAG
                    ),
                    None,
                )
                self.assembly_policy = AssemblyPolicy.cube(
                    self.stepper,
                    bottom_drag_implicit=bool(
                        getattr(bottom_drag, "implicit", False)
                    ),
                )
                policy_source = "TorchRegionalOceanModel.__init__ cube"
                modules_to_validate = self.slow_momentum_modules
            else:
                self.assembly_policy = AssemblyPolicy.regional(self.stepper)
                policy_source = "TorchRegionalOceanModel.__init__ regional"
                modules_to_validate = tuple(
                    module
                    for module in self.slow_dynamics_modules
                    if module.tendency_phase != TendencyPhase.TRACER
                    or module.momentum_term_class != MomentumTermClass.DYNAMICS
                )
            self.assembly_policy.validate_modules(
                modules_to_validate,
                source=policy_source,
            )
            object.__setattr__(
                self.stepper,
                "_resolved_assembly_policy",
                self.assembly_policy,
            )
        self.diagnostics = diagnostics or DiagnosticSuite(
            modules=[
                SurfaceMassDiagnostic(),
                TracerMassDiagnostic(field_name="temp", name="temp_mass"),
                TracerMassDiagnostic(field_name="salt", name="salt_mass"),
                BoundaryFluxBudgetDiagnostic(),
                KineticEnergyDiagnostic(),
                FreeSurfaceEnergyDiagnostic(),
                CFLEstimateDiagnostic(),
                ThreeDContinuityDiagnostic(),
            ],
        )
        self.capabilities = self._build_capabilities()
        # Named `ledgers`, not `registries`: it is a CapabilityLedger, not a
        # plugin dispatcher.
        # `self.registries` retained as deprecated property below.
        self.ledgers = LedgerBundle()
        self.current_time_seconds = 0.0
        self._runner_step_time: float | None = None
        self._runner_sample_serial = 0
        self.boundary_budget = BoundaryBudgetAccumulator()
        participants = sampling_components(self.boundary_condition, self.dynamics_modules)
        for component in participants:
            bind_grid = getattr(component, "bind_sampling_grid", None)
            if bind_grid is not None:
                bind_grid(self.grid)
        self._time_dependent_components = tuple(
            component for component in participants
            if callable(getattr(component, "update_model_time", None))
        )
        self._register_runtime_components()
        self.prepare_step_time(0.0)

    @staticmethod
    def _dispatch_runtime_component(
        component: object,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> object:
        if hasattr(component, "to_runtime"):
            return component.to_runtime(device, dtype)
        if not hasattr(component, "to"):
            return component
        try:
            return component.to(device=device, dtype=dtype)
        except TypeError:
            try:
                return component.to(device, dtype)
            except TypeError:
                return component.to(device)

    @staticmethod
    def _grid_to_runtime(
        grid: StructuredCGrid,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> StructuredCGrid:
        # A CubedSphereGrid must pass through with its cube identity
        # (face_count=6, tile_connectivity, cube_corner_indices, per-face
        # CubedSphereMetricSet) intact — the StructuredCGrid rebuild below
        # would silently collapse it to a single broken patch.  It carries its
        # own device/dtype migration via ``to_runtime``.
        if isinstance(grid, CubedSphereGrid):
            return grid.to_runtime(device, dtype)
        if hasattr(grid, "to_runtime"):
            return grid.to_runtime(device, dtype)
        if hasattr(grid, "to"):
            try:
                return grid.to(device=device, dtype=dtype)
            except TypeError:
                try:
                    return grid.to(device, dtype)
                except TypeError:
                    pass
        metrics = MetricSet(
            dx_scalar=grid.metrics.dx_scalar,
            dy_scalar=grid.metrics.dy_scalar,
            dx_center=grid.metrics.dx_center.to(device=device, dtype=dtype),
            dy_center=grid.metrics.dy_center.to(device=device, dtype=dtype),
            cell_area=grid.metrics.cell_area.to(device=device, dtype=dtype),
            cell_volume=grid.metrics.cell_volume.to(device=device, dtype=dtype),
            depth_center=grid.metrics.depth_center.to(device=device, dtype=dtype),
            face_area_u=grid.metrics.face_area_u.to(device=device, dtype=dtype),
            face_area_v=grid.metrics.face_area_v.to(device=device, dtype=dtype),
            dxG_row=grid.metrics.dxG_row.to(device=device, dtype=dtype),
            dxC_row=grid.metrics.dxC_row.to(device=device, dtype=dtype),
            dyG_scalar_1d=grid.metrics.dyG_scalar_1d,
            dxG_row_ext=grid.metrics.dxG_row_ext.to(device=device, dtype=dtype),
            rAz_row=grid.metrics.rAz_row.to(device=device, dtype=dtype),
        )
        vertical_coordinate = ZLevelCoordinate(
            layer_thickness=grid.vertical_coordinate.layer_thickness.to(device=device, dtype=dtype),
        )
        masks = MaskSet(
            center=grid.masks.center.to(device=device),
            u=grid.masks.u.to(device=device),
            v=grid.masks.v.to(device=device),
            hFacC=None if grid.masks.hFacC is None else grid.masks.hFacC.to(device=device, dtype=dtype),
            hFacW=None if grid.masks.hFacW is None else grid.masks.hFacW.to(device=device, dtype=dtype),
            hFacS=None if grid.masks.hFacS is None else grid.masks.hFacS.to(device=device, dtype=dtype),
            maskInC=None if grid.masks.maskInC is None else grid.masks.maskInC.to(device=device, dtype=dtype),
            maskInW=None if grid.masks.maskInW is None else grid.masks.maskInW.to(device=device, dtype=dtype),
            maskInS=None if grid.masks.maskInS is None else grid.masks.maskInS.to(device=device, dtype=dtype),
            # Geometry mode flag (depth-control corner hFacZ); see
            # StructuredCGrid.to_runtime for why it must be carried.
            smooth_abs_func_range=grid.masks.smooth_abs_func_range,
        )
        return StructuredCGrid(
            nx=grid.nx,
            ny=grid.ny,
            metrics=metrics,
            vertical_coordinate=vertical_coordinate,
            masks=masks,
            boundary_topology=grid.boundary_topology,
            face_count=getattr(grid, "face_count", 1),
            tile_connectivity=getattr(grid, "tile_connectivity", {}),
            cube_corner_indices=getattr(grid, "cube_corner_indices", ()),
            distributed_neighbors=getattr(grid, "distributed_neighbors", {}),
            parallel_layout=getattr(grid, "parallel_layout", None),
        )

    def _check_open_boundary_runtime_support(self) -> None:
        if not self.grid.boundary_topology.has_open_boundaries:
            return
        if not self.fast_dynamics_modules:
            return
        if bool(getattr(self.boundary_condition, "supports_barotropic_open_boundary", False)):
            return
        raise NotImplementedError(
            "open boundary topology with fast/barotropic dynamics requires a boundary condition "
            "that explicitly supports barotropic open boundaries"
        )

    def _default_dynamics(self) -> list[TendencyModule]:
        # A stepper without select_r_star runs z coordinates (MITgcm select_rStar 0).
        return default_dynamics(self.config, select_r_star=int(getattr(self.stepper, "select_r_star", 0)))

    def _has_module_type(self, module_type: type[TendencyModule]) -> bool:
        return any(isinstance(module, module_type) for module in self.dynamics_modules)

    def _build_capabilities(self) -> ModelCapabilities:
        has_baroclinic_pg = self._has_module_type(HydrostaticPressureGradient)
        has_vertical_velocity = self._has_module_type(DiagnosticVerticalVelocity)
        has_vertical_transport = self._has_module_type(VerticalTracerTransport)
        has_vertical_mixing = self._has_module_type(VerticalMixing)
        has_coriolis = self._has_module_type(FCoriolis) or self._has_module_type(BetaPlaneCoriolis)
        supports_hydrostatic_3d = (
            self.grid.nz > 1
            and has_baroclinic_pg
            and has_vertical_velocity
            and has_vertical_transport
            and has_vertical_mixing
            and has_coriolis
        )
        supports_mitgcm_strict_alignment = (
            isinstance(self.stepper, ImplicitFreeSurfaceStepper)
            and getattr(self.stepper, "stagger_tracer_stepping", False)
        )
        supports_kpp = (
            isinstance(self.stepper, ImplicitFreeSurfaceStepper)
            and bool(getattr(self.stepper, "kpp_enabled", False))
        )
        return ModelCapabilities(
            supports_open_boundary=bool(getattr(self.boundary_condition, "supports_open_boundary", False)),
            supports_split_explicit=isinstance(self.stepper, SplitExplicitStepper),
            supports_fullgraph_model_step=self.stepper.supports_fullgraph_model_step,
            supports_finite_volume_tracers=True,
            supports_hydrostatic_3d=supports_hydrostatic_3d,
            supports_baroclinic_pressure_gradient=has_baroclinic_pg,
            supports_diagnostic_vertical_velocity=has_vertical_velocity,
            supports_vertical_tracer_transport=has_vertical_transport,
            supports_vertical_mixing=has_vertical_mixing,
            supports_kpp=supports_kpp,
            kpp_dbloc_form=self.stepper.kpp_dbloc_form if supports_kpp else "n/a",
            supports_coriolis=has_coriolis,
            supports_mitgcm_strict_alignment=supports_mitgcm_strict_alignment,
        )

    def _partition_dynamics_modules(
        self,
        modules: tuple[TendencyModule, ...] | list[TendencyModule],
    ) -> tuple[tuple[TendencyModule, ...], tuple[TendencyModule, ...]]:
        fast_modules: list[TendencyModule] = []
        slow_modules: list[TendencyModule] = []
        for module in modules:
            if module.stepper_domain == TendencyStepDomain.SLOW:
                slow_modules.append(module)
            elif module.stepper_domain == TendencyStepDomain.FAST:
                fast_modules.append(module)
            elif module.stepper_domain == TendencyStepDomain.PREDICTOR:
                # The stepper calls these itself inside its predictor; they
                # enter neither tendency assembly.
                pass
        return tuple(fast_modules), tuple(slow_modules)

    @staticmethod
    def _partition_by_phase(
        modules: tuple[TendencyModule, ...],
    ) -> tuple[tuple[TendencyModule, ...], tuple[TendencyModule, ...]]:
        """Place each output in its staggered phase, retaining module order."""
        momentum: list[TendencyModule] = []
        tracer: list[TendencyModule] = []
        for module in modules:
            phases = module.output_phases.values()
            if TendencyPhase.TRACER in phases:
                tracer.append(module)
            if TendencyPhase.MOMENTUM in phases:
                momentum.append(module)
        return tuple(momentum), tuple(tracer)

    def _validate_tracer_vertical_consumers(self) -> None:
        """Reject multiple R-advection consumers for a declared tracer."""
        implicit_names = set()
        convective_implicit = False
        tendency_ab: dict[str, bool] = {}
        if isinstance(self.stepper, ImplicitFreeSurfaceStepper):
            convective_implicit = self.stepper.kpp_convective_implicit is not None
            tendency_ab = {
                "temp": self.stepper.ab2_on_temperature_tendency,
                "salt": self.stepper.ab2_on_salinity_tendency,
            }
            if self.stepper.temperature_implicit_vertical_advection:
                implicit_names.add("temp")
            if self.stepper.salinity_implicit_vertical_advection:
                implicit_names.add("salt")
        consumers: dict[str, list[str]] = {
            name: ["implicit vertical advection"] for name in implicit_names
        }
        for index, module in enumerate(self.dynamics_modules):
            if isinstance(module, HorizontalTracerAdvection):
                for tracer in _composition_tracer_names(module, module.name):
                    validate_tracer_advection_scheme(
                        tracer=tracer, horizontal=module.scheme,
                        vertical=module.vertical_scheme, implicit=tracer in implicit_names,
                        convective_implicit=convective_implicit,
                    )
                    if tracer in tendency_ab:
                        validate_tracer_ab_scheme(
                            tracer=tracer, horizontal=module.scheme,
                            tendency_ab=tendency_ab[tracer],
                        )
                    if (tracer in implicit_names
                            and module.vertical_scheme == TracerFluxScheme.DST3_FL
                            and self.tracer_multi_dim_advection is None):
                        raise ValueError(
                            "implicit vertical DST3_FL requires the global "
                            "tracer_multi_dim_advection declaration"
                        )
                if not ((module.multi_dim and module.dt > 0) or module.combined_with_vtt):
                    continue
                # The implicit-R hand-off evaluates HTA in horizontal-only
                # mode for these names; it is not a second R consumer.
                names = tuple(
                    name for name in module.tracer_names if name not in implicit_names
                )
            elif isinstance(module, VerticalTracerTransport):
                names = module.tracer_names
            else:
                continue
            for name in dict.fromkeys(names):
                consumers.setdefault(name, []).append(f"{module.name}[{index}]")
        conflicts = {
            name: paths for name, paths in consumers.items() if len(paths) > 1
        }
        if conflicts:
            detail = "; ".join(
                f"{name}: {', '.join(paths)}" for name, paths in conflicts.items()
            )
            raise ValueError(
                "duplicate vertical tracer advection (R) consumers: " + detail
                + ". Configure exactly one R consumer per tracer."
            )

    def evaluate_momentum_tendency_with_diagnostics(
        self,
        state: PrognosticState,
        *,
        inputs: EvaluationInputs | None = None,
    ) -> tuple[StateTendency, DiagnosticState]:
        """Evaluate only MOMENTUM-phase slow outputs (for staggered stepping)."""
        state.validate_against(self.grid)
        diagnostics = DiagnosticState(inputs=inputs)
        tendency = self._evaluate_modules(
            self.slow_momentum_modules, state, diagnostics, phase=TendencyPhase.MOMENTUM,
        )
        tendency = self.boundary_condition.enforce_tendency(tendency, self.grid)
        return tendency, diagnostics

    def evaluate_momentum_tendency_split_ab2(
        self,
        state: PrognosticState,
        *,
        assembly_policy: AssemblyPolicy,
        inputs: EvaluationInputs | None = None,
    ) -> tuple[StateTendency, StateTendency | None, DiagnosticState]:
        """Evaluate MOMENTUM-phase modules through an assembly policy.

        MITgcm ``timestep.F`` controls AB2 placement via several independent
        flags. HPG specifically follows the regime test
        ``.NOT.staggerTimeStep .AND. .NOT.implicitIntGravWave`` (``timestep.F:116-126``):
        in that branch HPG enters ``gU/gV`` BEFORE AB2; otherwise HPG is added
        AFTER AB2 (``timestep.F:320-325``). The policy derives that placement
        from ``stagger_tracer_stepping`` and ``implicit_int_grav_wave``.

        ``AssemblyPolicy.mom_dissip_in_ab`` independently follows ``momDissip_In_AB``
        (``timestep.F:128-136`` vs ``:218-226``).  The two axes are orthogonal:
        exp4 needs HPG-in-AB2 with dissipation-post-AB.

        ``AssemblyPolicy.mom_forcing_out_ab`` follows ``momForcingOutAB``.

        Returns
        -------
        ab2_tendency : StateTendency
            Tendencies that go through AB2 per the unified policy.
        post_ab2_tendency : StateTendency | None
            Tendencies added after AB2.  None if no post-AB2 modules exist.
        diagnostics : DiagnosticState
        """
        state.validate_against(self.grid)
        diagnostics = DiagnosticState(inputs=inputs)

        ab2_modules, post_modules = assembly_policy.split(self.slow_momentum_modules)

        # Populate diagnostics from ALL modules first (some modules depend on
        # diagnostics set by earlier modules, e.g. density_anomaly for HPG).
        if _MODEL_TRACE_ENABLED:
            _model_trace(
                "momentum split_ab2 modules "
                f"ab2={[getattr(m, 'name', type(m).__name__) for m in ab2_modules]} "
                f"post={[getattr(m, 'name', type(m).__name__) for m in post_modules]}"
            )
        self._populate_module_diagnostics(self.slow_momentum_modules, state, diagnostics)

        ab2_accumulator = StateTendency.accumulator(state)
        for module in ab2_modules:
            if _MODEL_TRACE_ENABLED:
                module_name = getattr(module, 'name', type(module).__name__)
                _model_trace(f"evaluate ab2 module begin {module_name}")
            ab2_accumulator.add(module.select_tendency_phase(
                module.evaluate(state, diagnostics, self.grid), TendencyPhase.MOMENTUM,
            ))
            if _MODEL_TRACE_ENABLED:
                _model_trace(f"evaluate ab2 module done {module_name}")
        ab2_tendency = ab2_accumulator.finish()
        if _MODEL_TRACE_ENABLED:
            _model_trace("enforce ab2 tendency begin")
        ab2_tendency = self.boundary_condition.enforce_tendency(ab2_tendency, self.grid)
        if _MODEL_TRACE_ENABLED:
            _model_trace("enforce ab2 tendency done")

        post_ab2_tendency: StateTendency | None = None
        if post_modules:
            post_accumulator = StateTendency.accumulator(state)
            for module in post_modules:
                if _MODEL_TRACE_ENABLED:
                    module_name = getattr(module, 'name', type(module).__name__)
                    _model_trace(f"evaluate post module begin {module_name}")
                post_accumulator.add(module.select_tendency_phase(
                    module.evaluate(state, diagnostics, self.grid), TendencyPhase.MOMENTUM,
                ))
                if _MODEL_TRACE_ENABLED:
                    _model_trace(f"evaluate post module done {module_name}")
            post_ab2_tendency = post_accumulator.finish()
            if _MODEL_TRACE_ENABLED:
                _model_trace("enforce post tendency begin")
            post_ab2_tendency = self.boundary_condition.enforce_tendency(post_ab2_tendency, self.grid)
            if _MODEL_TRACE_ENABLED:
                _model_trace("enforce post tendency done")

        return ab2_tendency, post_ab2_tendency, diagnostics

    def evaluate_slow_tendency_split(
        self,
        state: PrognosticState,
        *,
        assembly_policy: AssemblyPolicy,
        tracer_ab2_names: tuple[str, ...] = ("temp", "salt"),
        inputs: EvaluationInputs | None = None,
    ) -> tuple[StateTendency, StateTendency | None, DiagnosticState]:
        """Evaluate ALL slow modules using the unified AB2 split policy.

        Non-staggered counterpart of :meth:`evaluate_momentum_tendency_split_ab2`.

        Honours all momentum split axes owned by ``assembly_policy``. HPG follows
        ``timestep.F:116-126`` (synchronous + non-IIGW: HPG-in-AB2) versus
        ``timestep.F:320-325`` (staggered or IIGW: HPG-post-AB).  Sponge /
        forcing post-AB matches ``forcing_In_AB=.FALSE.`` ⇒
        ``momForcingOutAB=1`` (``timestep.F:160-185``).

        Tracer outputs use ``AssemblyPolicy.tracer_bucket_of`` independently of
        momentum. Fields without AB extrapolation retain their original
        summation order; ``tracer_ab2_names`` names the extrapolated fields.

        Returns
        -------
        ab2_tendency : StateTendency
            Sum of all slow modules that pass the AB2 policy.
        post_ab2_tendency : StateTendency | None
            Sum of all slow modules deferred to post-AB2.  ``None`` when no
            post-AB2 modules exist (preserves the historical "no split"
            invariant for policies that place every module inside AB2).
        diagnostics : DiagnosticState
        """
        state.validate_against(self.grid)
        diagnostics = DiagnosticState(inputs=inputs)

        ab2_modules, post_modules = assembly_policy.split(self.slow_dynamics_modules)
        tracer_ab2_accumulator = StateTendency.accumulator(state)
        tracer_post_accumulator = StateTendency.accumulator(state)
        has_post_tracers = False

        def collect_tracers(module, tendency):
            nonlocal has_post_tracers
            if tracer_ab2_names:
                if assembly_policy.tracer_bucket_of(module) == AssemblyBucket.POST_AB2:
                    tracer_post_accumulator.add(tendency)
                    has_post_tracers = True
                else:
                    tracer_ab2_accumulator.add(tendency)

        if _MODEL_TRACE_ENABLED:
            _model_trace(
                "slow split modules "
                f"ab2={[getattr(m, 'name', type(m).__name__) for m in ab2_modules]} "
                f"post={[getattr(m, 'name', type(m).__name__) for m in post_modules]}"
            )
        self._populate_module_diagnostics(self.slow_dynamics_modules, state, diagnostics)

        ab2_accumulator = StateTendency.accumulator(state)
        for module in ab2_modules:
            if _MODEL_TRACE_ENABLED:
                _model_trace(f"evaluate ab2 module begin {getattr(module, 'name', type(module).__name__)}")
            tendency = module.evaluate(state, diagnostics, self.grid)
            ab2_accumulator.add(tendency)
            collect_tracers(module, tendency)
            if _MODEL_TRACE_ENABLED:
                _model_trace(f"evaluate ab2 module done {getattr(module, 'name', type(module).__name__)}")
        ab2_tendency = ab2_accumulator.finish()
        if _MODEL_TRACE_ENABLED:
            _model_trace("enforce ab2 tendency begin")
        ab2_tendency = self.boundary_condition.enforce_tendency(ab2_tendency, self.grid)
        if _MODEL_TRACE_ENABLED:
            _model_trace("enforce ab2 tendency done")

        post_ab2_tendency: StateTendency | None = None
        if post_modules:
            post_accumulator = StateTendency.accumulator(state)
            for module in post_modules:
                if _MODEL_TRACE_ENABLED:
                    _model_trace(f"evaluate post module begin {getattr(module, 'name', type(module).__name__)}")
                tendency = module.evaluate(state, diagnostics, self.grid)
                post_accumulator.add(tendency)
                collect_tracers(module, tendency)
                if _MODEL_TRACE_ENABLED:
                    _model_trace(f"evaluate post module done {getattr(module, 'name', type(module).__name__)}")
            post_ab2_tendency = post_accumulator.finish()
            if _MODEL_TRACE_ENABLED:
                _model_trace("enforce post tendency begin")
            post_ab2_tendency = self.boundary_condition.enforce_tendency(post_ab2_tendency, self.grid)
            if _MODEL_TRACE_ENABLED:
                _model_trace("enforce post tendency done")

        if tracer_ab2_names:
            tracer_ab2 = self.boundary_condition.enforce_tendency(
                tracer_ab2_accumulator.finish(), self.grid,
            )
            tracer_post = self.boundary_condition.enforce_tendency(
                tracer_post_accumulator.finish(), self.grid,
            )

            def tracer_fields(momentum, tracers):
                return StateTendency(
                    eta=momentum.eta, u=momentum.u, v=momentum.v,
                    temp=tracers.temp if "temp" in tracer_ab2_names else momentum.temp,
                    salt=tracers.salt if "salt" in tracer_ab2_names else momentum.salt,
                )

            ab2_tendency = tracer_fields(ab2_tendency, tracer_ab2)
            if post_ab2_tendency is not None or has_post_tracers:
                if post_ab2_tendency is None:
                    post_ab2_tendency = StateTendency.zeros_like(state)
                post_ab2_tendency = tracer_fields(post_ab2_tendency, tracer_post)

        return ab2_tendency, post_ab2_tendency, diagnostics

    def evaluate_tracer_tendency_with_diagnostics(
        self,
        state: PrognosticState,
        *,
        inputs: EvaluationInputs | None = None,
    ) -> tuple[StateTendency, DiagnosticState]:
        """Evaluate only TRACER-phase slow outputs (for staggered stepping).

        Called with a staggered state containing u^{n+1} so that tracer
        advection and vertical velocity diagnostics see pressure-corrected
        velocities, matching MITgcm staggerTimeStep behaviour.

        SurfaceHeatSaltFlux computes TsurfCor/SsurfCor from the evaluated
        state's velocity and surface tracers.
        """
        state.validate_against(self.grid)
        diagnostics = DiagnosticState(inputs=inputs)
        tendency = self._evaluate_modules(
            self.slow_tracer_modules, state, diagnostics, phase=TendencyPhase.TRACER,
        )
        tendency = self.boundary_condition.enforce_tendency(tendency, self.grid)
        return tendency, diagnostics

    def evaluate_tracer_tendency_split_ab2(
        self,
        state: PrognosticState,
        *,
        inputs: EvaluationInputs | None = None,
    ) -> tuple[StateTendency, StateTendency | None, DiagnosticState]:
        """Evaluate TRACER-phase modules, split by AB2 eligibility.

        Mirrors MITgcm ``temp_integrate.F`` /  ``salt_integrate.F`` with
        ``doAB_onGtGs=T`` + ``tracForcingOutAB=1`` (production runtime config):

            G_combined = AB2(G_adv + G_diff)        ← ab2_eligible=True bucket
            gT_loc     = G_combined + gtForc^n_raw  ← ab2_eligible=False bucket
            T^{n+1}    = T^n + dt * gT_loc

        See ``temp_integrate.F:283`` (GAD_ADVECTION → gT_loc), ``:328``
        (APPLY_FORCING_T → gtForc), ``:357`` (GAD_CALC_RHS → gT_loc),
        ``:390`` (ADAMS_BASHFORTH2 over gT_loc), ``:407`` (gT_loc += gtForc
        AFTER AB2 when tracForcingOutAB=1).

        Returns
        -------
        ab2_tendency
            Tracer tendencies that go through AB2 (``ab2_eligible=True``
            modules: advection, horizontal diffusion, explicit VerticalMixing).
        post_ab2_tendency : StateTendency | None
            Tracer tendencies added after AB2 (``ab2_eligible=False``
            modules: SurfaceHeatSaltFlux, TracerBoundarySpongeRelaxation).
            ``None`` when no post-AB2 modules.
        diagnostics : DiagnosticState
        """
        state.validate_against(self.grid)
        diagnostics = DiagnosticState(inputs=inputs)
        ab2_modules, post_modules = AssemblyPolicy.split_tracers(
            self.slow_tracer_modules,
        )
        self._populate_module_diagnostics(self.slow_tracer_modules, state, diagnostics)

        ab2_accumulator = StateTendency.accumulator(state)
        for module in ab2_modules:
            ab2_accumulator.add(module.select_tendency_phase(
                module.evaluate(state, diagnostics, self.grid), TendencyPhase.TRACER,
            ))
        ab2_tendency = self.boundary_condition.enforce_tendency(
            ab2_accumulator.finish(), self.grid
        )

        post_ab2_tendency: StateTendency | None = None
        if post_modules:
            post_accumulator = StateTendency.accumulator(state)
            for module in post_modules:
                post_accumulator.add(module.select_tendency_phase(
                    module.evaluate(state, diagnostics, self.grid), TendencyPhase.TRACER,
                ))
            post_ab2_tendency = self.boundary_condition.enforce_tendency(
                post_accumulator.finish(), self.grid
            )

        return ab2_tendency, post_ab2_tendency, diagnostics

    def _register_runtime_components(self) -> None:
        self.ledgers.operators.register("center_to_u", "defgcm.operators.center_to_u")
        self.ledgers.operators.register("center_to_v", "defgcm.operators.center_to_v")
        self.ledgers.operators.register("gradient_center_to_u", "defgcm.operators.gradient_center_to_u")
        self.ledgers.operators.register("gradient_center_to_v", "defgcm.operators.gradient_center_to_v")
        self.ledgers.operators.register("gradient_eta_to_u", "defgcm.operators.gradient_eta_to_u")
        self.ledgers.operators.register("gradient_eta_to_v", "defgcm.operators.gradient_eta_to_v")
        self.ledgers.operators.register(
            "divergence_transport_to_center",
            "defgcm.operators.divergence_transport_to_center",
        )
        self.ledgers.operators.register(
            "horizontal_tracer_flux_divergence",
            "defgcm.operators.horizontal_tracer_flux_divergence",
        )
        self.ledgers.operators.register(
            "hydrostatic_pressure_potential",
            "defgcm.operators.hydrostatic_pressure_potential",
        )
        self.ledgers.operators.register(
            "vertical_velocity_from_divergence",
            "defgcm.operators.vertical_velocity_from_divergence",
        )
        self.ledgers.operators.register(
            "vertical_flux_divergence",
            "defgcm.operators.vertical_flux_divergence",
        )
        self.ledgers.steppers.register(self.stepper.name, self.stepper)
        self.ledgers.boundaries.register(self.boundary_condition.name, self.boundary_condition)

    @property
    def registries(self) -> "LedgerBundle":
        """Deprecated alias for `self.ledgers`.

        Named `ledgers` because it is a CapabilityLedger, not a plugin
        dispatcher.
        """
        import warnings

        warnings.warn(
            "`model.registries` is deprecated; use `model.ledgers` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.ledgers

    def create_state(
        self,
        *,
        eta: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        temp: torch.Tensor | None = None,
        salt: torch.Tensor | None = None,
        batch_size: int = 1,
    ) -> PrognosticState:
        """The run's initial state as given; no boundary values are applied.

        ``ModelRunner.run`` and ``runner.initialize_fresh_step_state`` apply the
        base-time open-boundary values before the first step; a direct
        ``step`` does so itself only as its docstring says.
        """
        eta_data = self._coerce_tensor(eta, (batch_size, self.grid.ny, self.grid.nx))
        u_data = self._coerce_tensor(u, (batch_size, self.grid.nz, self.grid.ny, self.grid.nx + 1))
        v_data = self._coerce_tensor(v, (batch_size, self.grid.nz, self.grid.ny + 1, self.grid.nx))
        scalar_shape = (batch_size, self.grid.nz, self.grid.ny, self.grid.nx)
        temp_data = self._coerce_tensor(temp, scalar_shape)
        salt_data = self._coerce_tensor(salt, scalar_shape)
        state = PrognosticState(
            eta=Field("eta", eta_data, FieldLocation.CENTER, "m", mask_ref="center"),
            u=Field("u", u_data, FieldLocation.U_FACE, "m s-1", mask_ref="u"),
            v=Field("v", v_data, FieldLocation.V_FACE, "m s-1", mask_ref="v"),
            temp=Field("temp", temp_data, FieldLocation.CENTER, "degC", mask_ref="center"),
            salt=Field("salt", salt_data, FieldLocation.CENTER, "psu", mask_ref="center"),
        )
        state.validate_against(self.grid)
        # The run's initial field as given (INI_THETA, INI_SALT, INI_VEL and
        # INI_PSURF, ini_fields.F:29-32).  The base-time open-boundary values
        # and the seam exchange that follows them (obcs_init_variables.F:386-412)
        # are applied once, by runner.initialize_fresh_step_state after the
        # INI_PRESSURE seed (ini_fields.F:33 precedes them).
        return state

    def _coerce_tensor(self, tensor: torch.Tensor | None, shape: tuple[int, ...]) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(shape, dtype=self.config.dtype, device=self.config.device)
        result = tensor.to(dtype=self.config.dtype, device=self.config.device)
        if result.shape == shape[1:]:
            result = result.unsqueeze(0)
        if tuple(result.shape) != shape:
            raise ValueError(f"expected tensor shape {shape}, got {tuple(result.shape)}")
        return result

    def _tracer_cell_volume(self, tensor: torch.Tensor) -> torch.Tensor:
        # Negative-index/broadcast form so the leading dim can be either the
        # regional batch ([B,nz,ny,nx], cell_area [ny,nx]) or the cube face dim
        # ([6,nz,ny,nx], cell_area [6,ny,nx] -> unsqueeze(-3) = [6,1,ny,nx]).
        # Element-wise identical to the old view(1,nz,1,1)/view(1,1,ny,nx) on
        # the regional path.
        thickness = self.grid.layer_thickness.to(dtype=tensor.dtype, device=tensor.device).view(self.grid.nz, 1, 1)
        area = self.grid.metrics.cell_area.to(dtype=tensor.dtype, device=tensor.device)
        # A multi-rank cube holds its owned faces only; the stepper scatters the
        # same global metrics (implicit_free_surface_cube._prepare_cube_step_entry).
        ctx = getattr(self, "cube_ctx", None)
        if ctx is not None and area.shape[0] != tensor.shape[0]:
            from defgcm.geometry.cube_decomposition import scatter_faces
            area = scatter_faces(area, ctx)
        return thickness * area.unsqueeze(-3)

    @torch._dynamo.disable
    def prepare_step_time(self, time_seconds: float) -> None:
        """Install every time-dependent component's sample at ``time_seconds``. Host work.

        Opaque to Dynamo.  The phase-split steppers call this in their eager
        prepare (``prepare_step_times``); AB2, SSPRK3 and split-explicit call it
        inside their own ``step``, which is the traced kernel under
        ``compile=model_step``.  Traced there, every sampler below it was
        compiled as its own frame and guarded on the per-call Python clock
        (``BoundaryBundleSource._time_bracket``, ``ERA5WindSource.sample``), so
        each new clock recompiled those frames until the recompile limit.
        """
        self._runner_step_time = None
        # Explicit histories publish their clock only with an accepted state.
        # Sampling a candidate's clocks must leave that committed clock alone.
        if getattr(self, "_ocean_carry", None) is None:
            self.current_time_seconds = float(time_seconds)
        for component in self._time_dependent_components:
            component.update_model_time(float(time_seconds))
        # Sync stepper OBC η from Flather's time-varying eta_values.
        # MITgcm equivalent: OBCS_CALC(myTime+deltaT) called every step
        # in do_oceanic_phys.F:320 updates OBWeta/OBEeta before they are
        # used by OBCS_APPLY_ETA and the pressure solver.
        self._sync_stepper_obc_eta()

    @ordinary_tensor_mode()
    def _prepare_runner_step_time(self, time_seconds: float) -> None:
        # The plan samples in ordinary mode too.  Keep the single START
        # sample's tensor kind and gradient policy identical to that path.
        self.prepare_step_time(time_seconds)
        self._runner_step_time = float(time_seconds)
        self._runner_sample_serial += 1

    def _time_state_objects(self) -> tuple[object, ...]:
        return sampling_components(
            self.boundary_condition,
            getattr(self, "dynamics_modules", self._time_dependent_components),
        )

    def _prepare_balanced_targets(
        self,
        objects: tuple[object, ...],
        reference: torch.Tensor | None,
    ) -> None:
        """Eagerly build the OBCS-balanced targets of the installed clock.

        Compiled execution constraint: ``_get_balanced_targets`` keys its memo on
        ``id()`` of the active side sets and on the float clock, so the whole
        ``compute_obcs_balanced_normal_velocities`` body was re-traced inside
        the compiled step.  ``reference`` fixes the dtype/device the step will
        ask for -- ``state.u.data`` -- so the prepared answer is served rather
        than recomputed; without one (tests calling ``prepare_step_times``
        directly) nothing is prepared and every consumer keeps its legacy path.
        """
        if reference is None:
            return
        for component in objects:
            prepare = getattr(component, "prepare_balanced_targets", None)
            if prepare is None:
                continue
            prepare(self.grid, dtype=reference.dtype, device=reference.device)

    def prepare_step_times(
        self,
        current: float,
        future: float,
        *,
        reference: torch.Tensor | None = None,
    ) -> "BoundaryPayload":
        """Resolve declared clocks into entry/application tensor attributes.

        Scientific attributes are explicitly named by SamplingRule, including
        unchanged or empty channels. Mutation snapshots remain only for host
        metadata/cache restoration; tensor detection cannot select input clocks
        or silently discard an END scalar. The full typed payload API is a
        separate migration.
        """
        objects = self._time_state_objects()
        # Consume only the runner's explicit hand-off.  Constructor samples,
        # direct steps and checkpoint replays still sample their own clocks.
        if self._runner_step_time != float(current):
            self.prepare_step_time(current)
        self._runner_step_time = None

        def geometry_eta_sample():
            if getattr(self.stepper, "nonlin_free_surf", 0) < 1:
                return None
            from defgcm.boundaries.open import BoundarySide, expand_barotropic_side_boundary_tensor
            from defgcm.steppers.implicit_free_surface_nonlin import SurfaceEtaBoundary

            # Preserve each side before a 2-D compatibility field merges its
            # corners. Initial geometry consumes START; step-end geometry END.
            flather = self._find_flather_bc()
            if flather is None:
                field = torch.as_tensor(
                    self.stepper._get_obc_eta(), dtype=self.config.dtype,
                    device=self.config.device,
                ).expand(self.grid.ny, self.grid.nx)
                return SurfaceEtaBoundary(
                    field[:, 0].clone(), field[:, -1].clone(),
                    field[0, :].clone(), field[-1, :].clone(),
                )
            sides = {}
            for side in BoundarySide:
                value = flather._active_eta_values.get(side)
                value = torch.as_tensor(
                    0.0 if value is None else value,
                    dtype=self.config.dtype, device=self.config.device,
                )
                sides[side.value] = expand_barotropic_side_boundary_tensor(
                    value, side, self.grid,
                    batch_size=1, dtype=self.config.dtype, device=self.config.device,
                )[0].clone()
            return SurfaceEtaBoundary(**sides)

        current_obc_eta = geometry_eta_sample()
        # An END-only component uses its future attributes at both entry and
        # application. Build START products only when either payload consumes
        # them, using the same clock rules as the frame selection below.
        start_objects = tuple(
            obj for obj in objects
            if sampling_rule(obj).clock != SampleClock.END
            or sampling_rule(obj).at_entry() != SampleClock.END
        )
        self._prepare_balanced_targets(start_objects, reference)
        current_states = tuple((obj, vars(obj).copy()) for obj in objects)

        # START providers retain their samples for every consumer in this step.
        # Boundary entry is handled separately when constructing the payload.
        for obj in objects:
            rule = sampling_rule(obj)
            update = getattr(obj, "update_model_time", None)
            if update is not None and rule.clock == SampleClock.END:
                update(float(future))
        self._sync_stepper_obc_eta()
        future_obc_eta = geometry_eta_sample()
        self._prepare_balanced_targets(objects, reference)
        future_states = tuple((obj, vars(obj).copy()) for obj in objects)

        prepared_current: list[ComponentSample] = []
        prepared_future: list[ComponentSample] = []
        host_current: list[tuple[object, dict[str, object]]] = []
        for (obj, start), (_, end) in zip(current_states, future_states, strict=True):
            rule = sampling_rule(obj)
            # Stateless providers return their samples to their owner; they do
            # not install attributes of their own inside the step.
            if not callable(getattr(obj, "update_model_time", None)):
                continue
            name = getattr(obj, "name", type(obj).__name__)
            changed = {key for key in start.keys() | end.keys()
                       if start.get(key) is not end.get(key)}
            missing = set(rule.inputs) - (start.keys() & end.keys())
            if missing:
                raise ValueError(f"{name}: missing declared sampled inputs {sorted(missing)}")
            scalar_inputs = sorted({
                path for key in rule.inputs for frame in (start, end)
                for path in sampled_host_paths(frame[key], key)
            })
            if scalar_inputs:
                raise ValueError(
                    f"{name}: {rule.clock.name} sampled inputs would be dropped by the "
                    f"host/device partition: {sorted(scalar_inputs)}; produce tensors"
                )
            # Only non-scientific attributes pass through payload placement.
            # A changed undeclared tensor is an incomplete contract, not a
            # discovered science input. Host timestamps/cache keys stay eager.
            host = {}
            for key in changed - set(rule.inputs):
                if _carries_tensor(start.get(key)) or _carries_tensor(end.get(key)):
                    raise ValueError(f"{name}: tensor attribute {key} missing from sampling_rule.inputs")
                host[key] = start.get(key)
            if rule.clock == SampleClock.END and not rule.inputs:
                raise ValueError(
                    f"{name}: END sampled inputs would be dropped by the host/device "
                    f"partition: {sorted(changed)}; declare tensor inputs"
                )
            host_current.append((obj, host))

            def attributes(frame):
                return {key: (_clone_retained_value(frame[key])
                              if getattr(obj, "uses_stevens", False) else frame[key])
                        for key in rule.inputs}

            entry = end if rule.at_entry() == SampleClock.END else start
            application = end if rule.clock == SampleClock.END else start
            prepared_current.append(ComponentSample(component=obj, attributes=attributes(entry)))
            prepared_future.append(ComponentSample(component=obj, attributes=attributes(application)))
        payload = BoundaryPayload(
            current=ClockSample(sampled=True, components=tuple(prepared_current),
                                obc_eta=current_obc_eta),
            future=ClockSample(sampled=True, components=tuple(prepared_future),
                               obc_eta=future_obc_eta),
        )
        for component, attributes in host_current:
            for name, value in attributes.items():
                object.__setattr__(component, name, value)
        # Geometry consumers use the explicit samples; the compatibility cache
        # retains END for existing pressure/boundary callers.
        payload.current.install()
        return payload

    def prepare_step_host(
        self,
        state: PrognosticState,
        bundle: TimeStepBundle,
        *,
        time_seconds: float,
    ) -> tuple[BoundaryPayload, KppStepForcing | None]:
        """Return the prepared boundary payload and KPP forcing inputs.

        A stepper without the hook prepares no time sample, and the empty
        payload says exactly that: the step body then takes its legacy route
        (``ClockSample.sampled`` is ``False``), it does not silently proceed
        with half a sample.

        A stepper that *has* the hook returns its ``BoundaryPayload`` and KPP
        inputs as a tuple, with ``BoundaryPayload.empty()`` if it sampled no clock.
        A hook that falls off its end returns ``None``, and accepting that here would translate
        "the hook forgot to return" into "this step took no sample": the step
        body would silently drop back to the legacy in-step clock advance and
        run different physics with no error and no diagnostic
        (the explicit-input requirement). The plan carries both outputs explicitly.
        """
        from defgcm.step.carry import (
            supports_explicit_history, initialize, WindowMode, OceanCarry, _publish,
        )

        supported = supports_explicit_history(self)
        previous = getattr(self, "_ocean_carry", None)
        if previous is not None and not supported:
            raise NotImplementedError("unsupported configuration change during explicit history")
        if supported:
            if bundle != self.config.time_step_bundle():
                raise ValueError("explicit history requires the configured fixed time steps")
            if previous is None:
                initial = initialize(self, state, time_seconds=time_seconds,
                                     window_mode=WindowMode.REINITIALIZE_HISTORY)
                _publish(self, OceanCarry(state, initial.ab_history, initial.clock,
                                         initial.hydro_history))
        prepare = getattr(self.stepper, "prepare_step_host", None)
        if prepare is None:
            return BoundaryPayload.empty(), None
        payload, kpp_forcing_inputs = prepare(self, state, bundle, time_seconds=time_seconds)
        if not isinstance(payload, BoundaryPayload):
            raise TypeError(
                f"{type(self.stepper).__name__}.prepare_step_host returned "
                f"{type(payload).__name__}, not a BoundaryPayload; a prepare "
                "hook that sampled no clock must return "
                "BoundaryPayload.empty()"
            )
        return payload, kpp_forcing_inputs

    def _sync_stepper_obc_eta(self) -> None:
        """Push Flather's current eta_values to the stepper's OBC η cache.

        The ImplicitFreeSurfaceStepper pins η at open-boundary cells to
        a cached ``_obc_eta`` tensor via ``set_obc_eta()``.  Without
        per-step synchronization this cache stays frozen at the initial
        value, causing multi-meter η drift in realistic simulations.

        This method reads the **already-updated** active eta values from
        the ``FlatherBarotropicBoundaryCondition`` (updated by
        ``update_model_time`` above) and pushes them to the stepper.
        """
        # Guard: stepper must support set_obc_eta
        if not hasattr(self.stepper, 'set_obc_eta'):
            return
        # Guard: need open boundaries to have anything to sync
        if not self.grid.boundary_topology.has_open_boundaries:
            return

        # Walk boundary condition chain to find FlatherBarotropicBoundaryCondition
        flather = self._find_flather_bc()
        if flather is None:
            return

        # Build a 2D [ny, nx] tensor with boundary η placed at edges
        from defgcm.boundaries.open import (
            BoundarySide,
            expand_barotropic_side_boundary_tensor,
        )
        dtype = self.config.dtype
        device = self.config.device
        eta_2d = torch.zeros(self.grid.ny, self.grid.nx, dtype=dtype, device=device)

        active = flather._active_eta_values
        for side in BoundarySide:
            vals = active.get(side)
            if vals is None:
                continue
            # Expand to [1, length] then squeeze → [length]
            expanded = expand_barotropic_side_boundary_tensor(
                vals, side, self.grid,
                batch_size=1, dtype=dtype, device=device,
            )[0]  # [length]
            if side == BoundarySide.WEST:
                eta_2d[:, 0] = expanded
            elif side == BoundarySide.EAST:
                eta_2d[:, -1] = expanded
            elif side == BoundarySide.SOUTH:
                eta_2d[0, :] = expanded
            elif side == BoundarySide.NORTH:
                eta_2d[-1, :] = expanded

        self.stepper.set_obc_eta(eta_2d)

    def _find_flather_bc(self) -> object | None:
        """Walk the boundary condition chain to find a FlatherBarotropicBoundaryCondition."""
        from defgcm.boundaries.open import FlatherBarotropicBoundaryCondition
        bc = self.boundary_condition
        # Check the top-level BC
        if isinstance(bc, FlatherBarotropicBoundaryCondition):
            return bc
        # Check velocity_boundary (TracerOpenBoundaryCondition wraps Flather)
        vb = getattr(bc, 'velocity_boundary', None)
        if isinstance(vb, FlatherBarotropicBoundaryCondition):
            return vb
        return None

    def reset_runtime_budget(self) -> None:
        self.boundary_budget = BoundaryBudgetAccumulator()

    def evaluate_tendency(self, state: PrognosticState) -> StateTendency:
        state.validate_against(self.grid)
        tendency, _ = self.evaluate_tendency_with_diagnostics(state)
        return tendency

    def evaluate_tendency_with_diagnostics(self, state: PrognosticState) -> tuple[StateTendency, DiagnosticState]:
        state.validate_against(self.grid)
        diagnostics = DiagnosticState()
        tendency = self._evaluate_modules(self.dynamics_modules, state, diagnostics)
        tendency = self.boundary_condition.enforce_tendency(tendency, self.grid)
        return tendency, diagnostics

    def evaluate_fast_tendency(self, state: PrognosticState) -> StateTendency:
        self._check_open_boundary_runtime_support()
        state.validate_against(self.grid)
        tendency, _ = self.evaluate_fast_tendency_with_diagnostics(state)
        return tendency

    def evaluate_fast_tendency_with_diagnostics(self, state: PrognosticState) -> tuple[StateTendency, DiagnosticState]:
        self._check_open_boundary_runtime_support()
        state.validate_against(self.grid)
        diagnostics = DiagnosticState()
        tendency = self._evaluate_modules(self.fast_dynamics_modules, state, diagnostics)
        tendency = self.boundary_condition.enforce_tendency(tendency, self.grid)
        return tendency, diagnostics

    def evaluate_slow_tendency(self, state: PrognosticState) -> StateTendency:
        state.validate_against(self.grid)
        tendency, _ = self.evaluate_slow_tendency_with_diagnostics(state)
        return tendency

    def evaluate_slow_tendency_with_diagnostics(
        self, state: PrognosticState, *, inputs: EvaluationInputs | None = None,
    ) -> tuple[StateTendency, DiagnosticState]:
        state.validate_against(self.grid)
        diagnostics = DiagnosticState(inputs=inputs)
        tendency = self._evaluate_modules(self.slow_dynamics_modules, state, diagnostics)
        tendency = self.boundary_condition.enforce_tendency(tendency, self.grid)
        return tendency, diagnostics

    def _evaluate_modules(
        self,
        modules: tuple[TendencyModule, ...],
        state: PrognosticState,
        diagnostics: DiagnosticState,
        *,
        phase: TendencyPhase | None = None,
    ) -> StateTendency:
        self._populate_module_diagnostics(modules, state, diagnostics)
        accumulator = StateTendency.accumulator(state)
        for module in modules:
            tendency = module.evaluate(state, diagnostics, self.grid)
            if phase is not None:
                tendency = module.select_tendency_phase(tendency, phase)
            accumulator.add(tendency)
        return accumulator.finish()

    def _active_tracer_boundary_values(self, tracer_name: str) -> object | None:
        if tracer_name == "temp":
            for attr in ("active_temp_values", "_active_temp_values"):
                values = getattr(self.boundary_condition, attr, None)
                if values is not None:
                    return values
            return None
        if tracer_name == "salt":
            for attr in ("active_salt_values", "_active_salt_values"):
                values = getattr(self.boundary_condition, attr, None)
                if values is not None:
                    return values
            return None
        raise ValueError(f"unsupported tracer boundary field: {tracer_name}")

    def _populate_boundary_diagnostics(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
    ) -> None:
        from defgcm.boundaries.open import (
            BOUNDARY_SIDES,
            expand_side_boundary_tensor,
        )

        for tracer_name, field in (("temp", state.temp.data), ("salt", state.salt.data)):
            boundary_values = self._active_tracer_boundary_values(tracer_name)
            if boundary_values is None:
                continue
            # ``for side in BoundarySide`` is
            # EnumType.__iter__, untraceable; BOUNDARY_SIDES is the
            # same members in the same order.
            for side in BOUNDARY_SIDES:
                side_values = boundary_values.get(side)
                if side_values is None:
                    continue
                diagnostics.set_tensor(
                    f"{tracer_name}_obc_{side.value}",
                    expand_side_boundary_tensor(
                        side_values,
                        side,
                        self.grid,
                        batch_size=field.shape[0],
                        dtype=field.dtype,
                        device=field.device,
                    ),
                )

    def _populate_module_diagnostics(
        self,
        modules: tuple[TendencyModule, ...],
        state: PrognosticState,
        diagnostics: DiagnosticState,
    ) -> None:
        if _MODEL_TRACE_ENABLED:
            _model_trace("populate boundary diagnostics begin")
        self._populate_boundary_diagnostics(state, diagnostics)
        if _MODEL_TRACE_ENABLED:
            _model_trace("populate boundary diagnostics done")
        for module in modules:
            if _MODEL_TRACE_ENABLED:
                _model_trace(f"populate diagnostics begin {getattr(module, 'name', type(module).__name__)}")
            module.populate_diagnostic_state(state, diagnostics, self.grid)
            if _MODEL_TRACE_ENABLED:
                _model_trace(f"populate diagnostics done {getattr(module, 'name', type(module).__name__)}")

    @torch.no_grad()
    def accumulate_tendency_diagnostics(self, diagnostics: DiagnosticState, scale: float) -> None:
        """Accumulate monitoring values without retaining past step tapes."""
        self.boundary_budget.temp_sponge_integral += scale * diagnostics.scalars.get("temp_sponge_tendency_integral_rate", 0.0)
        self.boundary_budget.salt_sponge_integral += scale * diagnostics.scalars.get("salt_sponge_tendency_integral_rate", 0.0)

    def enforce_boundaries(self, state: PrognosticState, *, record_budget: bool = True,
                           boundary_evaluation=None, initial: bool = False,
                           advance_stevens: bool = True, stevens_source=None,
                           flather_targets=None, orlanski_source=None,
                           step_entry: bool = False) -> PrognosticState:
        """Apply boundary fields, retaining initial etaN for nonlinear geometry.

        A Stevens source pairs the state at diagnosis with its tracer time step.
        Velocity projections skip the tracer increment until thermodynamics.
        MIT's initial OBCS_APPLY_UV/TS leaves etaN intact: the first
        CALC_SURF_DR uses its boundary values with myIter=-1. Prescribed ETA
        enters etaH at the subsequent initial UPDATE_ETAH event.
        ``flather_targets`` come from ``freeze_boundary_targets`` of the
        calling step; without them a Flather boundary evaluates its own.
        ``orlanski_source`` is the state an Orlanski boundary's OBCS_CALC reads
        (``tracer_boundary_phases`` binds the step-start state).  The step-entry
        application (``step_entry``) has no MITgcm counterpart, so an Orlanski
        boundary computes nothing there, and no boundary writes tracers there.
        """
        stevens = (dict(advance_stevens=advance_stevens, stevens_source=stevens_source)
                   if getattr(self.boundary_condition, "uses_stevens", False) else {})
        if getattr(self.boundary_condition, "uses_orlanski", False):
            stevens["orlanski_source"] = None if step_entry else orlanski_source
        if flather_targets is not None:
            stevens["flather_targets"] = flather_targets
        bounded = (self.boundary_condition.enforce(state, self.grid, **stevens)
                   if boundary_evaluation is None else
                   boundary_evaluation.enforce(self.boundary_condition, state, self.grid, **stevens))
        if step_entry:
            # MITgcm applies tracer boundary values in THERMODYNAMICS only
            # (OBCS_APPLY_TS, thermodynamics.F:359); what TRACERS_CORRECTION_STEP
            # leaves on the open boundary (forward_step.F:1025) is what the next
            # step's tendencies read, so the step-entry application keeps it.
            bounded = bounded.with_tensors(temp=state.temp.data, salt=state.salt.data)
        if (initial and self.grid.face_count == 1
                and getattr(self.stepper, "nonlin_free_surf", 0) > 0):
            bounded = bounded.with_tensors(eta=state.eta.data)
        if record_budget:
            cell_volume = self._tracer_cell_volume(state.temp.data)
            self.boundary_budget.temp_boundary_reset_integral += float(
                ((bounded.temp.data - state.temp.data) * cell_volume).sum().item()
            )
            self.boundary_budget.salt_boundary_reset_integral += float(
                ((bounded.salt.data - state.salt.data) * cell_volume).sum().item()
            )
        # Each boundary condition gates its writes with the interior-facing
        # face mask, as in MITgcm obcs_apply_uv.F. Re-masking with the boundary
        # face mask here could erase a valid prescribed boundary value.
        #
        # On a wall or open axis, the last velocity face is an interior
        # boundary face and retains the boundary condition value. On a periodic
        # axis it duplicates face 0; ``faces.sync_distributed_shared_velocity_faces``
        # restores that seam value after applying the boundary conditions.

        # Publish the owner's seam values as part of every open-boundary
        # application. Tangential OBCS writes cover only nx/ny owned faces;
        # MIT applies them through the halo extent (obcs_apply_uv.F:84-141).
        # The predicate uses global topology so all ranks participate. Other
        # domains retain their existing sync and observation points.
        if faces.open_boundary_crosses_rank_seam(
            face_count=getattr(self.grid, "face_count", 1),
            parallel_layout=getattr(self.grid, "parallel_layout", None),
            boundary_topology=self.grid.boundary_topology,
        ):
            u_sync, v_sync = faces.sync_distributed_shared_velocity_faces(
                bounded.u.data, bounded.v.data, self.grid
            )
            if u_sync is not bounded.u.data or v_sync is not bounded.v.data:
                bounded = bounded.with_tensors(u=u_sync, v=v_sync)

        bounded.validate_against(self.grid)
        return bounded

    def freeze_boundary_targets(self, state: PrognosticState):
        """This step's Flather velocity targets from its start state, or None.

        ``tracer_boundary_phases`` calls this when a step creates its boundary
        phases and binds the result into them; the model keeps no copy.
        """
        bc = self._find_flather_bc()
        return None if bc is None else bc.freeze_targets(state, self.grid)

    def with_comm_session(self, session):
        """Bind a differentiation window to this model's grid transport.

        The returned model owns the binding; numerical tensors and the
        stepper remain shared. Advance one model at a time for each stepper.
        Every participating rank finishes the window with session.backward.
        """
        result = copy(self)
        result._model_runner_compile_cache = {}
        result.grid = copy(self.grid)
        object.__setattr__(result.grid, "comm_session", session)
        return result

    def step_kernel(
        self,
        state: PrognosticState,
        plan: "StepPlan",
        *,
        carry=None,
    ) -> "StepOutcome":
        """The traced phase of one step: ``(state, plan)`` in, tensors out.

        This is what the runner hands to ``torch.compile``.  It reads no host
        state of its own -- the plan carries every decision -- and it returns
        tensors only: the new prognostic state plus the step's diagnostics as
        0-d tensors.  A per-step Python scalar crossing either seam is what
        Dynamo specialises on, so neither seam carries a changing Python scalar.

        ``plan`` is the ``StepPlan``, not the ``HostPlan`` prepare returned:
        everything reachable from this argument is guarded whether the body
        reads it or not, which is how ``times.future_seconds`` recompiled this
        frame every step, even without a reader.
        """
        stepper = self.stepper
        kernel = getattr(stepper, "step_kernel", None)
        # The binding is prepare-owned and run-scoped -- ``runner.run`` resolves
        # it once and installs it with ``model.with_comm_session`` before the
        # step loop starts -- so this is a folded read on the grid.
        comm_session = session_for_grid(self.grid)
        if kernel is None:
            context = self
            retained = {}
            if plan.boundary_history is not None:
                from defgcm.step.boundary_history import BoundaryHistoryAdapter
                context = copy(self)
                context.boundary_condition = copy(self.boundary_condition)
                context.boundary_condition._history_adapter = BoundaryHistoryAdapter(
                    history=plan.boundary_history, active_stamp=plan.times.future)
                context._time_dependent_components = tuple(
                    context.boundary_condition if component is self.boundary_condition else component
                    for component in self._time_dependent_components)
            # Steppers without a phase split (AB2, SSPRK, split-explicit): their
            # ``step`` *is* the kernel -- it holds no step boundary of its own --
            # and the eager prepare has already run.  They consume the clock as
            # a host value (``model.prepare_step_time(time_seconds)`` inside
            # their own bodies), so the tensor is read back here rather than
            # carried as a float on the plan: the read costs nothing eagerly,
            # which is where these steppers run, and a compiled one breaks the
            # graph at the read instead of specialising the whole frame
            # silently. The runner rejects fullgraph model_step for these
            # steppers through supports_fullgraph_model_step. This branch is
            # dead at trace time whenever the installed stepper has a kernel.
            result = stepper.step(
                state,
                context,
                plan.times.bundle.dt,
                time_seconds=float(plan.times.current),
            )
            if plan.boundary_history is not None:
                retained = {"boundary_history": context.boundary_condition._history_adapter.candidate,
                            "boundary_source_stamp": plan.boundary_history.stamp,
                            "boundary_step_index": plan.step_index.clone()}
            if comm_session is not None:
                retained["communication_token"] = comm_session.terminal_token()
            return StepOutcome(
                state=result,
                diagnostics=StepDiagnostics.absent(),
                retained=retained,
            )
        history_carry = carry if carry is not None else getattr(self, "_ocean_carry", None)
        if history_carry is not None:
            from defgcm.step.carry import (
                _supports_hydro_history, validate_history, validate_hydro_history,
            )

            validate_history(state, history_carry.ab_history, stepper)
            if _supports_hydro_history(self):
                validate_hydro_history(state, history_carry.hydro_history)
        retained = EMPTY_MAPPING
        boundary_kwargs = {}
        hydro_active = history_carry is not None and history_carry.hydro_history is not None
        if plan.boundary_history is not None or hydro_active:
            stepper = copy(stepper)
            kernel = stepper.step_kernel
        kernel_model = self
        if hydro_active:
            kernel_model = copy(self)
            kernel_model.stepper = stepper
        if plan.boundary_history is not None:
            from defgcm.step.boundary_history import BoundaryEvaluation
            evaluation = BoundaryEvaluation(plan.boundary_history, plan.times.future)
            boundary_kwargs = {"boundary_evaluation": evaluation}
        if history_carry is not None:
            if carry is None and state is not history_carry.state:
                raise ValueError("external state replacement requires initialize with a WindowMode")
            kernel_outcome = kernel(
                state, kernel_model, plan, ab_history=history_carry.ab_history,
                hydro_history=history_carry.hydro_history, **boundary_kwargs)
            retained = {"ab_history": kernel_outcome.retained["ab_history"],
                        "hydro_history": kernel_outcome.retained["hydro_history"],
                        "publication_source": self._ocean_carry,
                        "source_clock": history_carry.clock,
                        "clock": {"time_seconds": plan.times.future,
                                  "step_index": history_carry.clock["step_index"] + 1}}
        else:
            kernel_outcome = kernel(state, self, plan, **boundary_kwargs)
        if plan.boundary_history is not None:
            retained = {**retained, "boundary_history": evaluation.candidate,
                        "boundary_source_stamp": plan.boundary_history.stamp,
                        "boundary_step_index": plan.step_index.clone()}
        if plan.boundary_history is not None or hydro_active:
            # The kernel ran on a candidate copy of the stepper; the inputs it
            # evaluated (``evaluated_inputs``, read by diagnostics) leave with
            # the candidate's other outputs, only the names the step wrote.
            retained = {**retained, "owned_step_state": {name: getattr(stepper, name)
                                             for name in stepper.OWNED_STEP_FIELDS},
                        "total_phi_hyd": stepper._cached_total_phi_hyd,
                        "evaluated_inputs": {
                            name: vars(stepper)[name]
                            for name in ("_evaluated_w_interface", "_evaluated_total_phi_hyd")
                            if name in vars(stepper)}}
        if comm_session is not None:
            retained = {**retained, "communication_token": comm_session.terminal_token()}
        return StepOutcome(
            state=kernel_outcome.state,
            diagnostics=kernel_outcome.diagnostics,
            retained=retained,
        )

    def observe_state(self, state: PrognosticState) -> PrognosticState:
        """The state exposed to an objective, callback or direct step caller."""
        session = session_for_grid(self.grid)
        if session is None:
            return state
        names = ("eta", "u", "v", "temp", "salt")
        fields = tuple(getattr(state, name).data for name in names)
        observed = session.observe(fields)
        return state if observed is fields else state.with_tensors(**dict(zip(names, observed, strict=True)))

    def step(
        self,
        state: PrognosticState,
        dt: float | TimeStepBundle | None = None,
        *,
        time_seconds: float | None = None,
    ) -> PrognosticState:
        """Advance one step and return its observed state. **Eager.**

        The first call of a new run applies the base-time open-boundary values
        and seeds the step state (``runner.initialize_fresh_step_state``) when
        the stepper is the implicit free surface or the boundary keeps a
        Stevens or Orlanski history (``_fresh_start_pending``); with AB2, SSPRK3
        or split-explicit and an open boundary that assigns values, call
        ``initialize_fresh_step_state`` before the first step.
        """
        return self.step_outcome(state, dt, time_seconds=time_seconds).state

    def _fresh_start_pending(self) -> bool:
        """Whether a direct step starts a new run that has not been started.

        A boundary that assigns sampled open-boundary values
        (``sampling_rule(...).inputs``) owes the run its base-time application
        before the first step (obcs_init_variables.F:386-412).  A new carry is
        known by the Stevens or Orlanski boundary history, else by the
        implicit free-surface step counter; the initializer and a restore
        install both.  The other steppers keep no such counter.
        """
        boundary = self.boundary_condition
        if not sampling_rule(boundary).inputs:
            return False
        from defgcm.step.boundary_history import boundary_adapter

        if boundary_adapter(self) is not None:
            return boundary.boundary_history is None
        return (getattr(self.stepper, "name", None) == "implicit_free_surface"
                and self.stepper._step_index is None)

    def step_outcome(
        self,
        state: PrognosticState,
        dt: float | TimeStepBundle | None = None,
        *,
        time_seconds: float | None = None,
    ) -> StepOutcome:
        """Run all three phases of one step. **Eager.**

       The runner drives the same three calls itself so it can compile the
        middle one alone; this method is the direct-API equivalent and the
        reference both postures agree with. Converting time_seconds to float
        is safe here because this frame runs eagerly.
        """
        actual_time = (
            self.current_time_seconds if time_seconds is None else float(time_seconds)
        )
        if self._fresh_start_pending():
            from .runner import initialize_fresh_step_state

            self.prepare_step_time(actual_time)
            state = initialize_fresh_step_state(self, state, dt=dt)
        host = build_step_plan(self, state, dt, time_seconds=actual_time)
        outcome = self.step_kernel(state, host.plan)
        return replace(outcome, state=self.observe_state(finalize_step(self, outcome, host)))

    def compute_diagnostics(self, state: PrognosticState, *, dt: float | None = None) -> DiagnosticState:
        # Diagnostics see the inputs of the last evaluated step, as the
        # tendency modules did; the stepper owns them until its segment ends.
        evaluated = getattr(self.stepper, "evaluated_inputs", None)
        inputs = None if evaluated is None else evaluated()
        if ((inputs is None or inputs.total_phi_hyd is None)
                and any(getattr(m, "dynamic_pressure", False) for m in self.dynamics_modules)):
            # Before the first step nothing was evaluated: the dynamic-pressure
            # EOS reads the seeded potential when prepared, else an explicit zero
            # (the potential INI_PRESSURE starts from).
            # (validate_dynamic_pressure_composition: the stepper is the IFS.)
            phi = getattr(self.stepper, "_cached_total_phi_hyd", None)
            phi = torch.zeros_like(state.temp.data) if phi is None else phi
            inputs = (EvaluationInputs(total_phi_hyd=phi,
                                       momentum_pressure_forcing=self.stepper.momentum_pressure_forcing)
                      if inputs is None else replace(inputs, total_phi_hyd=phi))
        context = DiagnosticState(inputs=inputs)
        self._populate_module_diagnostics(self.dynamics_modules, state, context)
        scalars = dict(context.scalars)
        scalars.update(self.diagnostics.compute_all(
            state,
            self.grid,
            gravity=self.config.gravity,
            dt=dt,
            context=context,
        ))
        scalars.update(self.boundary_budget.as_dict())
        current_sample_metadata = getattr(self.boundary_condition, "current_sample_metadata", {})
        if current_sample_metadata:
            scalars.update(
                {
                    "boundary_time_lower_index": float(current_sample_metadata["lower_index"]),
                    "boundary_time_upper_index": float(current_sample_metadata["upper_index"]),
                    "boundary_time_lower_seconds": float(current_sample_metadata["lower_time_seconds"]),
                    "boundary_time_upper_seconds": float(current_sample_metadata["upper_time_seconds"]),
                    "boundary_time_lower_weight": float(current_sample_metadata["lower_weight"]),
                    "boundary_time_upper_weight": float(current_sample_metadata["upper_weight"]),
                }
            )
        return DiagnosticState(scalars=scalars)
