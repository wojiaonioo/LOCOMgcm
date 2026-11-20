"""RunConfig dataclass schema.

Schema versions:
  - ``0.1.0``: pre-case-kind configs; missing ``case`` defaults to regional.
  - ``0.2.0``: adds ``case.kind`` for regional vs MIT verification dispatch.
    The ``grid.kind`` cubed-sphere extension (``case.kind='cube_global'`` plus
    the optional ``cube_*`` grid fields) is a backward-compatible addition
    *within* 0.2.0 — every new field carries a default, so existing configs
    load byte-identically and no new version number is opened."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Mapping

from defgcm.constants import ROTATION_PERIOD_EARTH

#: Legal ``grid.cube_advection_topology`` values.  Kept as a literal tuple (not
#: imported from ``defgcm.geometry``) so the config layer stays importable
#: without pulling in torch; ``geometry/cubed_sphere_grid.ADVECTION_TOPOLOGIES``
#: is the consuming authority and re-validates whatever the builder passes on.
_CUBE_ADVECTION_TOPOLOGIES = ("mit_12tile", "six_face_approx")


@dataclass(frozen=True)
class RunIdentity:
    experiment_name: str
    start_date: date
    run_id: str | None = None
    device: Literal["cuda", "cpu", "auto"] = "auto"
    preprocess: bool = True
    # Compute precision for the state / grid / forcing tensors.  Default
    # ``float64`` reproduces the historical (and MIT-aligned) behaviour exactly;
    # ``float32`` is the reduced-precision production path (cube f32 groundwork).
    # The runtime maps this string to a ``torch.dtype`` at the single dtype
    # source (experiments/cases/_runtime.py).
    precision: Literal["float64", "float32"] = "float64"

    def __post_init__(self) -> None:
        if self.run_id is not None:
            if not isinstance(self.run_id, str):
                raise TypeError(
                    f"run.run_id must be str or None, got {type(self.run_id).__name__}"
                )
            if not self.run_id or self.run_id.strip() != self.run_id:
                raise ValueError("run.run_id must be a nonempty path component")
            if self.run_id in {".", ".."} or "/" in self.run_id or "\\" in self.run_id:
                raise ValueError(
                    f"run.run_id must be a single path component, got {self.run_id!r}"
                )
        # Config fail-fast: an unknown precision string (typo) must raise rather
        # than silently fall through to float64.
        if self.precision not in {"float64", "float32"}:
            raise ValueError(
                f"run.precision must be 'float64' or 'float32', got {self.precision!r}"
            )


@dataclass(frozen=True)
class GridConfig:
    nx: int
    ny: int
    nz: int
    center_lat: float
    center_lon: float
    resolution_deg: float
    depth_m: float
    vertical_layer_thicknesses: tuple[float, ...]
    # MITgcm delRc: nz+1 centre spacings (m), the grid's drC
    # (geometry/vertical_coordinates.py ZLevelCoordinate); empty puts the
    # centres midway between interfaces.
    vertical_center_spacing: tuple[float, ...] = ()
    # Grid family discriminator. ``latlon`` (default) is the historical
    # StructuredCGrid path — unchanged. ``cubed_sphere`` selects the global
    # cube path; the ``cube_*`` fields below then locate the MITgcm face
    # geometry / bathymetry (see io/runner/builders/grid_factory.py). All new
    # fields carry defaults so existing lat-lon configs load byte-identically.
    kind: str = "latlon"
    cube_face_n: int | None = None
    cube_grid_dir: str | None = None
    cube_grid_stem: str | None = None
    cube_bathy_path: str | None = None
    # Cube horizontal-advection topology declaration, carried from yaml into
    # ``CubedSphereGrid.advection_topology`` (io/runner/builders/grid_factory.py).
    # Mandatory on the cube path: which advection algorithm a case runs is a
    # property of the case, not something a runtime probe should guess.
    cube_advection_topology: str | None = None
    # MIT SIZE.h block declaration (sNx / sNy / OLx); required when
    # ``cube_advection_topology='mit_12tile'``, absent otherwise.
    cube_tile_nx: int | None = None
    cube_tile_ny: int | None = None
    cube_tile_halo: int | None = None

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError(
                f"grid nx/ny/nz must be positive, got nx={self.nx}, "
                f"ny={self.ny}, nz={self.nz}"
            )
        if len(self.vertical_layer_thicknesses) != self.nz:
            raise ValueError(
                f"vertical_layer_thicknesses len {len(self.vertical_layer_thicknesses)} "
                f"!= nz {self.nz}"
            )
        if not all(
            math.isfinite(float(v)) and float(v) > 0.0
            for v in self.vertical_layer_thicknesses
        ):
            raise ValueError(
                "vertical_layer_thicknesses must contain finite positive values"
            )
        if not math.isfinite(float(self.depth_m)) or self.depth_m <= 0.0:
            raise ValueError(f"grid.depth_m must be finite and > 0, got {self.depth_m!r}")
        layer_sum = sum(float(v) for v in self.vertical_layer_thicknesses)
        if abs(layer_sum - float(self.depth_m)) >= 1e-3:
            raise ValueError(
                f"sum(vertical_layer_thicknesses) {layer_sum} != depth_m {self.depth_m}"
            )
        if self.kind not in {"latlon", "cubed_sphere"}:
            raise ValueError(
                f"GridConfig.kind must be 'latlon' or 'cubed_sphere', got {self.kind!r}"
            )
        # cubed_sphere requires the four face-locating fields; latlon adds no new
        # constraint (any redundant cube_* fields are tolerated, not validated).
        if self.kind == "cubed_sphere":
            missing = [
                name
                for name in (
                    "cube_face_n",
                    "cube_grid_dir",
                    "cube_grid_stem",
                    "cube_bathy_path",
                )
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "GridConfig.kind='cubed_sphere' requires "
                    f"{', '.join(missing)} to be set (got None)"
                )
            if not (self.nx == self.ny == self.cube_face_n):
                raise ValueError(
                    "GridConfig.kind='cubed_sphere' requires nx == ny == cube_face_n; "
                    f"got nx={self.nx}, ny={self.ny}, cube_face_n={self.cube_face_n}"
                )
            # Advection-topology declaration: mandatory, no default.  A missing
            # or misspelled value raises here rather than reaching the grid
            # builder, so a cube case can never run an unstated algorithm.
            if self.cube_advection_topology is None:
                raise ValueError(
                    "GridConfig.kind='cubed_sphere' requires "
                    "cube_advection_topology to be set (one of "
                    f"{list(_CUBE_ADVECTION_TOPOLOGIES)}); got None"
                )
            if self.cube_advection_topology not in _CUBE_ADVECTION_TOPOLOGIES:
                raise ValueError(
                    "GridConfig.cube_advection_topology must be one of "
                    f"{list(_CUBE_ADVECTION_TOPOLOGIES)}, got "
                    f"{self.cube_advection_topology!r}"
                )
            # The tile triplet is one declaration: a partial set would be
            # silently dropped by the grid builder's all-or-nothing assembly,
            # leaving the case half-declared with no signal.
            declared_tile = [
                name
                for name in ("cube_tile_nx", "cube_tile_ny", "cube_tile_halo")
                if getattr(self, name) is not None
            ]
            if declared_tile and len(declared_tile) != 3:
                raise ValueError(
                    "GridConfig cube_tile_nx/cube_tile_ny/cube_tile_halo must "
                    f"be declared together (all or none); got only {declared_tile}"
                )
            if self.cube_advection_topology == "mit_12tile":
                missing_tile = [
                    name
                    for name in ("cube_tile_nx", "cube_tile_ny", "cube_tile_halo")
                    if getattr(self, name) is None
                ]
                if missing_tile:
                    raise ValueError(
                        "GridConfig.cube_advection_topology='mit_12tile' requires "
                        f"{', '.join(missing_tile)} to be set (the MIT SIZE.h "
                        "sNx / sNy / OLx block declaration); got None"
                    )


@dataclass(frozen=True)
class TimeConfig:
    """Time-stepping config.

    Three ways to set the run horizon, in order of decreasing precision:

    1. ``steps: <int>`` — explicit integer step count. Most precise. The
       runtime never has to divide a float to derive a count, so there is
       no rounding question to get wrong.
    2. ``seconds: <int|float|"expr">`` — total simulated seconds. A
       string is parsed as a safe arithmetic expression so ``5 * 86400``
       (5 days), ``240 * 15`` (240 steps at dt=15s), or ``3600`` (1 hour)
       all work. The runtime divides by ``dt`` exactly once; if the
       result is not within ULP of an integer, the SSOT helper raises a
       StepCountTrapError (see defgcm.io.runner._time_steps).
    3. ``days: <float>`` — legacy float-days input. Retained for
       backward compat, but discouraged: ``days: 0.0008680556`` can
       silently become 6 MIT steps instead of 5, which is why ``seconds``
       exists.

    When more than one is set, the SSOT helper picks the most precise
    available, in the order above.
    """

    dt: float
    dt_mom: float = 0.0
    dt_freesurf: float = 0.0
    dt_tracer: float = 0.0
    dt_clock: float = 0.0
    days: float = 1.0
    steps: int | None = None
    # Yaml-friendly total-seconds input. Accepts
    # int, float, or a safe arithmetic expression string (``"5 * 86400"``,
    # ``"240 * 15"``, ``"3600"``). Strings are evaluated at __post_init__
    # time so downstream consumers always see a float. Operators allowed:
    # ``+ - * / // % **`` plus integer/float literals; names, calls, and
    # attribute access are rejected.
    seconds: int | float | str | None = None

    def __post_init__(self):
        if self.steps is not None and self.steps <= 0:
            raise ValueError(f"time.steps must be positive when set, got {self.steps}")
        if not math.isfinite(float(self.dt)) or self.dt <= 0.0:
            raise ValueError(f"time.dt must be finite and > 0, got {self.dt!r}")
        # Resolve seconds from str expression if needed.
        if isinstance(self.seconds, str):
            resolved = _eval_time_seconds_expr(self.seconds)
            object.__setattr__(self, "seconds", resolved)
        elif isinstance(self.seconds, (int, float)):
            object.__setattr__(self, "seconds", float(self.seconds))
        elif self.seconds is not None:
            raise TypeError(
                f"time.seconds must be int|float|str|None, "
                f"got {type(self.seconds).__name__}: {self.seconds!r}"
            )
        if self.seconds is not None and (
            not math.isfinite(float(self.seconds)) or self.seconds <= 0
        ):
            raise ValueError(
                f"time.seconds must be finite and > 0 after resolution, got {self.seconds!r}"
            )
        if self.dt_mom == 0.0:
            object.__setattr__(self, "dt_mom", self.dt)
        if self.dt_freesurf == 0.0:
            object.__setattr__(self, "dt_freesurf", self.dt)
        if self.dt_tracer == 0.0:
            object.__setattr__(self, "dt_tracer", self.dt)
        if self.dt_clock == 0.0:
            object.__setattr__(self, "dt_clock", self.dt)
        for fld in ("dt_mom", "dt_freesurf", "dt_tracer", "dt_clock"):
            v = float(getattr(self, fld))
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"time.{fld} must be finite and > 0, got {v!r}")


_ALLOWED_TIME_SECONDS_NODES = (
    "Expression",
    "BinOp",
    "UnaryOp",
    "Constant",
    "Add",
    "Sub",
    "Mult",
    "Div",
    "FloorDiv",
    "Mod",
    "Pow",
    "USub",
    "UAdd",
    "Load",
)


def _eval_time_seconds_expr(expr: str) -> float:
    """Safely evaluate a yaml ``time.seconds`` arithmetic expression.

    Supports integer/float literals plus ``+ - * / // % **`` and unary
    plus/minus. Refuses anything that could touch the host process —
    names, attribute access, function calls, comprehensions, etc — and
    all constants must be numeric.

    Returns the numeric result as a float.

    Raises
    ------
    ValueError
        Empty string, malformed syntax, disallowed AST node, or
        non-numeric constant.
    """
    import ast

    if not isinstance(expr, str):
        raise TypeError(f"_eval_time_seconds_expr expected str, got {type(expr).__name__}")
    s = expr.strip()
    if not s:
        raise ValueError("time.seconds expression is empty")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"time.seconds is not a valid arithmetic expression: {expr!r} "
            f"({exc.msg})"
        ) from exc
    for node in ast.walk(tree):
        if type(node).__name__ not in _ALLOWED_TIME_SECONDS_NODES:
            raise ValueError(
                f"time.seconds uses disallowed syntax {type(node).__name__!r}: "
                f"{expr!r}"
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(
                f"time.seconds constants must be int/float, got {node.value!r} "
                f"in {expr!r}"
            )
    value = eval(  # noqa: S307 — sandboxed; AST walker above already restricted
        compile(tree, "<time.seconds>", "eval"),
        {"__builtins__": {}},
        {},
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"time.seconds expression {expr!r} did not evaluate to a "
            f"number, got {value!r}"
        )
    return float(value)


@dataclass(frozen=True)
class MitParallelConfig:
    nPx: int = 1
    nPy: int = 1
    sNx: int | None = None
    sNy: int | None = None
    OLx: int = 4
    OLy: int = 4

    @property
    def n_procs(self) -> int:
        return self.nPx * self.nPy


@dataclass(frozen=True)
class NonhydrostaticConfig:
    enabled: bool = False
    cg3d_max_iters: int = 150
    cg3d_target_residual: float = 1.0e-7
    cg3d_target_res_w_unit: float = -1.0
    nh_am2: float = 1.0
    select_3d_cori_scheme: int = 1
    select_cori_map: int = -1
    f_prime: float = 0.0


@dataclass(frozen=True)
class FreeSurfaceConfig:
    """Free-surface options."""
    select_r_star: int = 0
    lin_fs_conserve_tr: bool = False


@dataclass(frozen=True)
class TracerConfig:
    # None preserves "yaml did not say", so manifest assembly can keep the
    # stepper's bitwise-no-op default while legacy stepper kwargs remain valid.
    implicit_vertical_diffusion: bool | None = None
    diffusivity_kr: float | None = None


@dataclass(frozen=True)
class ImplicitVerticalAdvectionConfig:
    temperature: bool = False
    salinity: bool = False
    momentum: bool = False

    def __post_init__(self) -> None:
        for name in ("temperature", "salinity", "momentum"):
            value = getattr(self, name)
            if type(value) is not bool:
                raise ValueError(
                    "physics.implicit_vertical_advection."
                    f"{name} must be bool, got {type(value).__name__}"
                )


@dataclass(frozen=True)
class PhysicsConfig:
    # Boussinesq reference density rhoConst (single source).
    # None = "yaml did not say" -> readers fall back to their legacy channels
    # (case.extras.reference_density on the regional path, cube_cache_knobs
    # _cube_rho_const on the cube path).  When set, it IS the case rhoConst and
    # conflicting legacy entries raise (no dual entry).
    reference_density: float | None = None
    nonlin_free_surf: int = 0
    enable_vertical_shear: bool = False
    split_coriolis: bool = False
    nonhydrostatic: NonhydrostaticConfig = field(default_factory=NonhydrostaticConfig)
    free_surface: FreeSurfaceConfig = field(default_factory=FreeSurfaceConfig)
    viscosity: float = 0.0
    viscosity_scheme: str = "constant"
    visc_c2_leith: float = 0.0
    visc_c2_smag: float = 0.0
    diffusivity: float = 0.0
    # None preserves "yaml did not say", so manifest assembly can keep the
    # stepper's bitwise-no-op default (0.0 = MIT viscAr default) while legacy
    # stepper kwargs stay authoritative.  Mirrors TracerConfig above.
    implicit_viscosity: float | None = None
    # MITgcm implicitViscosity is an algorithm gate independent of viscAr.
    # None preserves legacy manifests whose nonzero coefficient selected the
    # implicit route before the gate was represented explicitly.
    implicit_vertical_viscosity: bool | None = None
    # Lateral no-slip side drag (MITgcm no_slip_sides). Default False = the
    # historical MIT-export baseline; the canonical per-case value lives under
    # model.dynamics.horizontal_viscosity.kwargs (read by LOCOMgcm) and reaches the
    # MIT export via legacy_bridge so both models agree.
    no_slip_sides: bool = False
    tracer: TracerConfig = field(default_factory=TracerConfig)
    implicit_vertical_advection: ImplicitVerticalAdvectionConfig = field(
        default_factory=ImplicitVerticalAdvectionConfig
    )
    ke_scheme: int = 0
    #: MITgcm rotationPeriod [s] (set_defaults.F:115); omega = 2 pi / rotationPeriod
    #: (ini_parms.F:487-490).  The one rotation period of a run: the canonical
    #: assembly hands it to the momentum modules, NonhydrostaticParams, the
    #: stepper (KPP's f) and the $coriolis_* profiles.
    rotation_period: float = ROTATION_PERIOD_EARTH
    # One-to-one with the same-named legacy_bridge.py destinations.
    advective_form: bool = False
    cfl_safety: float = 0.5
    hfac_min_dz: float | None = None
    fast_substeps: int = 8
    transport_correction: Literal["corrected", "raw"] = "corrected"

    def __post_init__(self) -> None:
        gate = self.implicit_vertical_viscosity
        if gate is not None and type(gate) is not bool:
            raise ValueError(
                "physics.implicit_vertical_viscosity must be bool or None, got "
                f"{type(gate).__name__}"
            )


@dataclass(frozen=True)
class MitComparisonConfig:
    """MIT case configuration.

    By default (`sync_from_to=True`) numerical parameters are derived from
    LOCOMgcm `physics` / `advection` / `time` blocks — guarantees apples-to-apples
    comparison. Set `sync_from_to=False` (or override any specific field) to
    decouple, e.g. for ablation studies where MIT uses different viscosity.

    Override semantics:
      - `sync_from_to=True`  → all `override_*` honored if not None,
                               otherwise pulled from LOCOMgcm
      - `sync_from_to=False` → all `override_*` honored if not None,
                               otherwise use class default (numeric 0 etc.)

    Use `effective_*(run_cfg)` getters to resolve the actual value.
    """
    enabled: bool = True
    sync_from_to: bool = True
    # Override knobs (only consulted when sync_from_to=False, OR if explicitly
    # set != None even under sync mode — explicit overrides always win).
    override_viscosity: float | None = None
    override_diffusivity: float | None = None
    override_viscA4: float | None = None
    override_dt: float | None = None
    override_days: float | None = None
    override_tracer_scheme: str | None = None  # mitgcm_adv_scheme alias
    override_multi_dim: bool | None = None
    # Always-explicit (no LOCOMgcm counterpart): MIT-only viscA4 default kept for
    # backwards compat; superseded by override_viscA4 if set.
    viscA4: float = 0.0
    # MIT-only settings.
    cartesian: bool = False
    explicit_viscosity: bool = False
    sponge_relax_tracers: bool = False

    def effective_viscosity(self, model_value: float) -> float:
        """viscAh; ``model_value`` is what LOCOMgcm's horizontal viscosity runs
        (``layer_d/_mit_switches.horizontal_viscosity_value``)."""
        if self.override_viscosity is not None:
            return float(self.override_viscosity)
        if self.sync_from_to:
            return float(model_value)
        return 0.0

    def effective_diffusivity(self, model_value: float) -> float:
        """diffKhT or diffKhS; ``model_value`` is what LOCOMgcm's horizontal tracer
        diffusion runs for that tracer
        (``layer_d/_mit_switches.tracer_harmonic_diffusivity``)."""
        if self.override_diffusivity is not None:
            return float(self.override_diffusivity)
        if self.sync_from_to:
            return float(model_value)
        return 0.0

    def effective_viscA4(self, run_cfg: "RunConfig") -> float:
        if self.override_viscA4 is not None:
            return float(self.override_viscA4)
        return float(self.viscA4)

    def effective_dt(self, run_cfg: "RunConfig") -> float:
        if self.override_dt is not None:
            return float(self.override_dt)
        if self.sync_from_to:
            return float(run_cfg.time.dt)
        return 0.0

    def effective_days(self, run_cfg: "RunConfig") -> float:
        if self.override_days is not None:
            return float(self.override_days)
        if self.sync_from_to:
            return float(run_cfg.time.days)
        return 1.0

    def effective_tracer_scheme(self, model_value: int) -> int:
        """tempAdvScheme or saltAdvScheme; ``model_value`` is the MITgcm code of the
        scheme LOCOMgcm's advecting module runs
        (``layer_d/_mit_switches.tracer_advection_values``)."""
        if self.override_tracer_scheme is not None:
            # accept either int or string alias
            v = self.override_tracer_scheme
            try:
                return int(v)
            except (TypeError, ValueError):
                # fall through to LOCOMgcm
                pass
        if self.sync_from_to:
            return int(model_value)
        return 33

    def effective_multi_dim(self, model_value: bool) -> bool:
        """multiDimAdvection; ``model_value`` is the advecting module's multi_dim."""
        if self.override_multi_dim is not None:
            return bool(self.override_multi_dim)
        if self.sync_from_to:
            return bool(model_value)
        return True


@dataclass(frozen=True)
class AdvectionConfig:
    # Unset (None) resolves to ``tracer`` in __post_init__: MITgcm sets an unset
    # tempVertAdvScheme / saltVertAdvScheme to the horizontal scheme
    # (model/src/ini_parms.F:424-425, :509-510).  Every reader gets the resolved
    # value: the tracer modules' vertical_scheme (_canonical_helpers.py), the
    # MITgcm export's tempVertAdvScheme and the implicit-advection check.
    # The legal names are the tracer modules' HORIZONTAL_TRACER_SCHEMES and
    # VERTICAL_TRACER_SCHEMES (operators/fluxes/tracer_flux.py); the modules
    # built from these two fields refuse any other name.
    vertical: str | None = None
    tracer: str = "dst3_fl"
    multi_dim: bool = True

    def __post_init__(self) -> None:
        if self.vertical is None:
            object.__setattr__(self, "vertical", self.tracer)


@dataclass(frozen=True)
class BoundaryConfig:
    type: Literal["open", "wall", "periodic"] = "open"
    scheme: Literal["clamped", "radiation", "prescribed3d"] = "clamped"
    sides: tuple[str, ...] = ("N", "E", "S", "W")
    sponge_width: int = 3
    sponge_mode: str = "mitgcm"
    sponge_timescale_hours: float = 1.0
    sponge_timescale_inner_hours: float = 72.0
    # Per-orientation OBCS sponge relax timescales in SECONDS (MITgcm
    # obcs_sponge.F: E/W boundaries use Urelaxobcs{bound,inner}, N/S use
    # Vrelaxobcs{bound,inner}; see MITgcm pkg/obcs/obcs_sponge.F).
    # Seconds-native (not hours) so reference values like Gong's 44.714 s survive
    # without an hours round-trip. ``None`` → fall back to the single
    # sponge_timescale_hours/_inner_hours pair (so U==V cases are unchanged).
    sponge_u_timescale_bound_seconds: float | None = None
    sponge_u_timescale_inner_seconds: float | None = None
    sponge_v_timescale_bound_seconds: float | None = None
    sponge_v_timescale_inner_seconds: float | None = None
    use_linear_sponge: bool = False
    sponge_ab2_eligible: bool = False
    # Legacy `--no-sponge-balance-obcs-targets` is `store_false` with default
    # True; emit the resolved bool.
    sponge_balance_obcs_targets: bool = True
    boundary_time_mode: str = "linear"
    # Prescribed3DBoundaryCondition rejects any other value; the former
    # inline_* diagnostics wrote tangential zeros in the mixed order.
    tangential_zero_mode: str = "none"
    stevens_relaxation_hours: float = 6.0
    # OBCS Tides: barotropic tidal velocity overlay at open boundaries.
    # When non-empty, harmonic tidal velocities are added AFTER balance flow.
    # Each entry: {name: "m2", period: 44712.0}
    # Files auto-resolved from tidal_data_dir: OB{N,S,E,W}_{u,v}Tid{Am,Ph}_{name}.bin
    tidal_constituents: tuple[Mapping[str, Any], ...] = ()
    tidal_data_dir: str = ""
    # True when the tidal phase data ALREADY includes the V0 astronomical
    # argument (e.g. Gong native OB _Gph, used as-is by Gong's obcs_add_tides.F
    # cos(w*(t-phase))).  Then both the LOCOMgcm runtime injection and the MIT export
    # SKIP their V0(start_date) correction -- adding it would double-count
    # (~33% M2 period phase error).  Default False = Greenwich phase (TPXO9), so
    # V0 is applied.  Either way LOCOMgcm and MIT do the same thing -> stay aligned.
    tidal_phase_includes_v0: bool = False
    apply_tangential_velocity: bool = False
    # MITgcm ``OBCSfixTopo`` (data.obcs OBCS_PARM01).  When True, open-boundary
    # strip columns are shallowed to min(self, interior neighbour) before hFac /
    # metrics are built.  Default **False** = MITgcm's own default
    # (``pkg/obcs/obcs_readparms.F:291``); a case must opt in explicitly, exactly
    # as it would have to write ``OBCSfixTopo=.TRUE.,`` into ``data.obcs``.
    obcs_fix_topo: bool = False
    # Which points ALONG an open side carry an open boundary — MITgcm's
    # ``OB_Jsouth``/``OB_Jnorth``/``OB_Iwest``/``OB_Ieast`` index table.  This is
    # a CASE policy, not a geometric fact: the same bathymetry admits both forms,
    # and the SCS reference configuration's ``data.obcs`` declares three entire
    # edges open on a domain that has dry boundary cells.
    #   ``full_edge`` — the whole side is open (``OB_Jsouth = Nx*1``); the
    #     default, and byte-for-byte the historical export and runtime behaviour.
    #   ``wet_only``  — a column/row is open where its outermost tracer cell is
    #     ocean, closed where it is land; the form the SCS_CLM reference leg's
    #     ``Input/OB_index_fixed.txt`` generates.
    # Consumed by BOTH the runtime momentum sponge (via enrich_manifest) and the
    # ``data.obcs`` export, from this one field.  See
    # ``defgcm/boundaries/open.py:resolve_ob_along_masks``.
    ob_index_mode: str = "full_edge"

    def __post_init__(self) -> None:
        # Config fail-fast, same reasoning as run.precision above: a typo'd mode
        # must raise at load time.  Without this an unknown value would only
        # surface later — and only for a case that
        # happens to assemble the momentum sponge or run the MIT export, i.e.
        # silently ignored otherwise.  The allowed set is imported rather than
        # re-listed so it cannot drift from the resolver's own table; the import
        # is local because defgcm.boundaries pulls in torch.
        from defgcm.boundaries.open import OB_INDEX_MODES, TANGENTIAL_ZERO_MODES
        from defgcm.dynamics.sponge_weights import SPONGE_MODES
        from defgcm.io.hycom_open_boundary import BoundaryTimeMode

        for name, allowed in (
            ("ob_index_mode", OB_INDEX_MODES),
            ("sponge_mode", SPONGE_MODES),
            ("tangential_zero_mode", TANGENTIAL_ZERO_MODES),
            ("boundary_time_mode", tuple(mode.value for mode in BoundaryTimeMode)),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"boundary.{name} must be one of {list(allowed)}, "
                    f"got {getattr(self, name)!r}"
                )


@dataclass(frozen=True)
class SourceSpec:
    name: str
    type: str
    config: Mapping[str, Any] = field(default_factory=dict)
    on_error: Literal["raise", "skip"] = "raise"
    cache: bool = True
    time_mode: Literal["sequential", "cyclic"] = "sequential"

    def __post_init__(self) -> None:
        from defgcm.external_paths import resolve_input_path

        if "path" in self.config:
            config = dict(self.config)
            config["path"] = str(resolve_input_path(config["path"]))
            object.__setattr__(self, "config", config)


@dataclass(frozen=True)
class OperatorSpec:
    op: str
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    primary_source: str
    operators: tuple[OperatorSpec, ...]
    consumer: str
    consumer_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StationOutputConfig:
    """High-frequency point (mooring) output — like Gong 2023's station dump.

    Records the full water column at each ``stations`` entry's nearest grid
    point every ``interval_seconds`` into a compact ``station_output.nc``
    ``(time, station, depth)``.  Read-only on state; does not touch physics.
    """
    enabled: bool = False
    interval_seconds: float | None = None        # default = dt (every step)
    stations: tuple[Mapping[str, Any], ...] = ()  # {name, lat, lon}
    variables: tuple[str, ...] = ("temp", "salt", "u", "v", "eta")


@dataclass(frozen=True)
class OutputConfig:
    """Output destination + layout policy.

    When `legacy_layout=True` (default), the manifest run dir is
    `<root>/<run.experiment_name>` so artifacts match the legacy
    regional-comparison directory shape:

        outputs/data/<experiment_name>/
        ├── mitgcm/{input,code,build,run}/
        ├── preprocessed.zarr
        ├── nc_snapshots/dayNNN.nc
        ├── comparison_metrics.{details,summary}.json
        ├── config_snapshot/
        ├── metadata.yaml
        └── provenance.yaml

    When `legacy_layout=False`, the manifest run dir is `<root>/<run_id>`
    and each stage writes to its own subdir.
    """
    root: Path = Path("outputs/data/")
    legacy_layout: bool = True
    monitor_interval_steps: int = 10
    nc_output_interval_hours: float = 3.0
    snapshot_interval_hours: float = 3.0
    # Preferred integer-seconds (or arithmetic-expression)
    # output cadence, mirroring time.seconds — avoids float-hours like
    # 0.00416667 for "every step". Strings (e.g. "1 * 3600", "15") are
    # evaluated at __post_init__ via the safe arithmetic evaluator.
    # When set, *_seconds wins over the legacy *_hours field. Use
    # ``snapshot_interval_seconds: 15`` (= dt) to dump every step.
    nc_output_interval_seconds: int | float | str | None = None
    snapshot_interval_seconds: int | float | str | None = None
    station_output: "StationOutputConfig | None" = None

    def __post_init__(self):
        for fld in ("nc_output_interval_seconds", "snapshot_interval_seconds"):
            v = getattr(self, fld)
            if isinstance(v, str):
                object.__setattr__(self, fld, _eval_time_seconds_expr(v))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                object.__setattr__(self, fld, float(v))
            elif v is not None:
                raise TypeError(
                    f"output.{fld} must be int|float|str|None, "
                    f"got {type(v).__name__}: {v!r}"
                )
            resolved = getattr(self, fld)
            if resolved is not None and (
                not math.isfinite(float(resolved)) or resolved <= 0
            ):
                raise ValueError(
                    f"output.{fld} must be finite and > 0, got {resolved!r}"
                )
        for fld in ("nc_output_interval_hours", "snapshot_interval_hours"):
            v = float(getattr(self, fld))
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"output.{fld} must be finite and > 0, got {v!r}")

    def resolved_nc_output_seconds(self) -> float:
        """NC/MIT dump cadence in seconds (seconds field wins over hours)."""
        if self.nc_output_interval_seconds is not None:
            return float(self.nc_output_interval_seconds)
        return float(self.nc_output_interval_hours) * 3600.0

    def resolved_snapshot_seconds(self) -> float:
        """LOCOMgcm snapshot cadence in seconds (seconds field wins over hours)."""
        if self.snapshot_interval_seconds is not None:
            return float(self.snapshot_interval_seconds)
        return float(self.snapshot_interval_hours) * 3600.0


@dataclass(frozen=True)
class CheckpointConfig:
    """Model-state checkpoint (restart / resume) policy.

    Distinct from ``output`` (lossy float32 NetCDF diagnostics): a
    checkpoint is a *lossless* float64 snapshot of the full cross-step
    state, written to ``<run_dir>/checkpoints/`` for bitwise-exact resume.

    Scheduling (``schedule``) is a list of entries compiled by
    ``defgcm.checkpoint.build_schedule`` — any union of:
      - ``{type: interval, every_steps|every_seconds: ...}``
      - ``{type: window, start_seconds, end_seconds, every_seconds|every_steps}``
      - ``{type: explicit, steps: [...]}``
    An empty list means no checkpoint writes; first/final writes require an
    explicit schedule entry.
    The shortcuts ``every_steps`` / ``every_seconds`` desugar (in the
    loader) to a single ``interval`` entry for the common case.

    ``resume``: ``none`` | ``latest`` | ``<int step>`` — restart from a
    previously written checkpoint; output cadence and total run length may
    change across a resume, but physics / grid / dt may not.
    """

    enabled: bool = False
    resume: str = "none"
    resume_dir: str | None = None  # read checkpoints from another run dir
    schedule: tuple[Mapping[str, Any], ...] = ()
    keep_last: int | None = 3
    permanent_steps: tuple[int, ...] = ()
    backend: str = "thread"  # thread (async) | sync
    max_inflight: int = 2

    def __post_init__(self) -> None:
        if self.backend not in {"thread", "sync", "async", "blocking"}:
            raise ValueError(
                f"checkpoint.backend must be thread|sync|async|blocking, got {self.backend!r}"
            )
        if self.max_inflight < 1:
            raise ValueError("checkpoint.max_inflight must be >= 1")
        if self.keep_last is not None and self.keep_last < 0:
            raise ValueError("checkpoint.keep_last must be >= 0 or null")


@dataclass(frozen=True)
class CompareConfig:
    variables: tuple[str, ...] = ("u", "v", "temperature", "salinity", "eta")
    tolerance: Mapping[str, Any] = field(default_factory=dict)
    output_format: tuple[str, ...] = ("json",)


@dataclass(frozen=True)
class DynamicsEntry:
    """One entry in the model.dynamics manifest list.

    `name` is the snake_case registry key (see LIGHTHOUSE catalog).
    `kwargs` is passed to the registered class's constructor.
    """
    name: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepperManifest:
    """model.stepper block - which TimeStepper to instantiate + kwargs.

    `name` may be None when yaml omits the block (+ migrations fill it).
    The builder raises a meaningful error if name is None at build time.
    """
    name: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryConditionManifest:
    """model.boundary_condition block - which BoundaryCondition + kwargs.

    `kwargs.tracer_source` and similar runtime-bound fields are NOT yaml-declared;
    they are bound by the builder at build time from Layer B OBcs source plugins.
    """
    name: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelManifest:
    """yaml-driven model composition: dynamics list + stepper + boundary_condition.

    When all three sub-fields are at defaults (empty/None), the manifest is
    "absent" - yaml didn't declare a model block. The universal builder
    The loader treats an absent manifest as a clear error for framework-path cases.
    """
    dynamics: tuple[DynamicsEntry, ...] = ()
    stepper: StepperManifest = field(default_factory=StepperManifest)
    boundary_condition: BoundaryConditionManifest = field(
        default_factory=BoundaryConditionManifest
    )

    def is_empty(self) -> bool:
        return (
            not self.dynamics
            and self.stepper.name is None
            and self.boundary_condition.name is None
        )


@dataclass(frozen=True)
class CacheConfig:
    root: Path
    hycom_cache_dir: Path | None = None
    enabled: bool = True
    cleanup_policy: Literal["never", "after_stage", "on_exit"] = "after_stage"
    # Maps to legacy --cache-rebuild.
    rebuild: bool = False
    # Content-keyed reuse of expensive external-data regrid pipeline
    # outputs (stage1a state/forcing/boundary/mesh). Default OFF — opt in per
    # case or via `--set cache.reuse=true`. See execution/_pipeline_cache.py.
    reuse: bool = False
    # When reuse hits, recompute anyway and bitwise-assert the cached artifact
    # matches (proves the key is complete). Default OFF; enable in CI /
    # periodically as the second independent source per the alignment protocol.
    verify: bool = False


@dataclass(frozen=True)
class DebugConfig:
    persist_intermediate: bool = False
    log_level: str = "INFO"
    # Maps to legacy --debug-dump-steps.
    dump_steps: int = 0


@dataclass(frozen=True)
class MitPickupConfig:
    """A start from a MITgcm pickup (``ini_fields.F:27-38`` calls READ_PICKUP).

    ``meta`` and ``data`` are the field-list pickup files; ``n_iter0`` is MITgcm's
    nIter0, which decides with the records present whether the first step
    extrapolates each AB2 history (``check_pickup.F:57-65``, ``adams_bashforth2.F:61-65``).
    LOCOMgcm time zero is MITgcm's startTime, which each time-dependent source carries.
    """

    meta: Path
    data: Path
    n_iter0: int
    #: The ``pickup_cd`` file; required with the CD scheme and refused without it.
    cd_data: Path | None = None


@dataclass(frozen=True)
class InitialConditionConfig:
    """Initial T/S source selection and source-model IC semantics.

    Maps to legacy `--initial-temp-salt-source` (`source`) and
    `--woa23-dir` (`woa23_dir`).  The freezing flags are emitted by the
    MITgcm translator and consumed only while constructing the initial state.
    """

    source: Literal["hycom", "woa23"] = "hycom"
    woa23_dir: Path | None = None
    check_ini_temp: bool = False
    allow_freezing: bool = False
    #: The prognostic state and AB2 history start from this pickup instead of the IC files.
    mit_pickup: MitPickupConfig | None = None


@dataclass(frozen=True)
class AdjointConfig:
    """ — first-step adjoint trace knobs.

    Surfaced via the separate ``experiments/pipeline_v2/run_adjoint.py``
    entrypoint, not the staged pipeline. ``first_step_adjoint_trace`` itself
    is target-tail derived (``target='adjoint'`` → emit
    ``first-step-adjoint-trace=True``) and stays NotExposed in the registry.
    """

    target: Literal["mitgcm", "hycom", "npz", "none"] = "hycom"
    fields: tuple[str, ...] = ("eta", "u", "v", "temp", "salt")
    top_k: int = 20
    target_npz: Path | None = None
    detect_anomaly: bool = False


@dataclass(frozen=True)
class CaseConfig:
    kind: str = "regional"
    mit_verification_dir: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SectionTransportUConfig:
    i_face: int
    j0: int
    j1: int

    def __post_init__(self) -> None:
        if not all(type(index) is int for index in (self.i_face, self.j0, self.j1)):
            raise ValueError("section transport indices must be integers")
        if self.i_face < 0 or self.j0 < 0 or self.j1 < 0:
            raise ValueError("section transport indices must be non-negative")
        if self.j0 > self.j1:
            raise ValueError("section transport requires j0 <= j1")


@dataclass(frozen=True)
class ObservationsConfig:
    section_transport_u: SectionTransportUConfig | None = None


@dataclass(frozen=True)
class RunConfig:
    schema_version: str
    case: CaseConfig
    run: RunIdentity
    grid: GridConfig
    time: TimeConfig
    mit_parallel: MitParallelConfig
    physics: PhysicsConfig
    advection: AdvectionConfig
    boundary: BoundaryConfig
    sources: tuple[SourceSpec, ...]
    pipelines: Mapping[str, tuple[PipelineSpec, ...]]
    output: OutputConfig
    compare: CompareConfig
    cache: CacheConfig
    debug: DebugConfig
    mit_comparison: MitComparisonConfig = field(default_factory=MitComparisonConfig)
    initial_condition: InitialConditionConfig = field(default_factory=InitialConditionConfig)
    adjoint: AdjointConfig = field(default_factory=AdjointConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    model: ModelManifest = field(default_factory=ModelManifest)
    observations: ObservationsConfig | None = None

    @cached_property
    def vertical_diffusion(self):
        from .vertical_diffusion import resolve_vertical_diffusion

        return resolve_vertical_diffusion(
            tracer=self.physics.tracer, stepper_name=self.model.stepper.name,
            stepper_kwargs=self.model.stepper.kwargs, dynamics=self.model.dynamics,
            grid_kind=self.grid.kind,
        )

    def __post_init__(self) -> None:
        if self.schema_version not in {"0.1.0", "0.2.0"}:
            raise ValueError(f"unsupported schema_version {self.schema_version}")
        if self.case.kind not in {"regional", "mit_verification", "cube_global"}:
            raise ValueError(
                "RunConfig.case.kind must be 'regional', 'mit_verification', or "
                f"'cube_global'; got '{self.case.kind}'"
            )
        if self.case.kind == "mit_verification" and self.case.mit_verification_dir is None:
            raise ValueError(
                "RunConfig.case.mit_verification_dir is required for "
                "case.kind='mit_verification'"
            )
        if self.case.kind == "regional" and self.case.mit_verification_dir is not None:
            raise ValueError(
                "RunConfig.case.mit_verification_dir must be None for "
                "case.kind='regional'"
            )
        # cube_global <-> cubed_sphere bidirectional lock: the two discriminators
        # must be set together or not at all, so a config cannot drift into a
        # half-cube state (case says global cube but grid is lat-lon, or vice
        # versa). Both directions raise so either field alone is a hard error.
        if self.case.kind == "cube_global" and self.grid.kind != "cubed_sphere":
            raise ValueError(
                "RunConfig.case.kind='cube_global' requires grid.kind='cubed_sphere' "
                f"(got grid.kind='{self.grid.kind}'); case.kind and grid.kind must "
                "be set together"
            )
        if self.grid.kind == "cubed_sphere" and self.case.kind != "cube_global":
            raise ValueError(
                "RunConfig.grid.kind='cubed_sphere' requires case.kind='cube_global' "
                f"(got case.kind='{self.case.kind}'); case.kind and grid.kind must "
                "be set together"
            )
        if self.case.kind == "cube_global" and self.case.mit_verification_dir is not None:
            raise ValueError(
                "RunConfig.case.mit_verification_dir must be None for "
                "case.kind='cube_global'"
            )
        # Cross-field consistency: MPI integer-divides
        if self.mit_parallel.nPx > 1:
            if self.grid.nx % self.mit_parallel.nPx != 0:
                raise ValueError(
                    f"grid.nx {self.grid.nx} not divisible by "
                    f"mit_parallel.nPx {self.mit_parallel.nPx}"
                )
        if self.mit_parallel.nPy > 1:
            if self.grid.ny % self.mit_parallel.nPy != 0:
                raise ValueError(
                    f"grid.ny {self.grid.ny} not divisible by "
                    f"mit_parallel.nPy {self.mit_parallel.nPy}"
                )
        # Any compute_mpi_topology operator in stage1c_export_mit must target
        # the same nProcs as mit_parallel.n_procs.  Caught at load-time;
        # otherwise drift between mit_parallel.{nPx,nPy} and the per-op
        # target_nProcs would let SIZE.h and stage3 mpirun -np disagree.
        expected_nprocs = self.mit_parallel.n_procs
        for stage, plist in self.pipelines.items():
            for p in plist:
                for op in p.operators:
                    if op.op != "compute_mpi_topology":
                        continue
                    op_target = int(op.config.get("target_nProcs", expected_nprocs))
                    if op_target != expected_nprocs:
                        raise ValueError(
                            f"pipeline {stage}.{p.name}: compute_mpi_topology "
                            f"target_nProcs={op_target} != "
                            f"mit_parallel.n_procs={expected_nprocs} "
                            f"(={self.mit_parallel.nPx}x{self.mit_parallel.nPy}); "
                            "these must agree or SIZE.h and stage3 mpirun -np drift"
                        )
        # Sources / pipelines coherence: every pipeline references a known source
        source_names = {s.name for s in self.sources}
        # Implicit stage1a outputs are valid primary_source values too:
        implicit = {
            "stage1a_state",
            "stage1a_forcing",
            "stage1a_boundary",
            "stage1a_mesh",
            "stage1a_initial_state",
            "stage1a_tidal",
        }
        valid_sources = source_names | implicit
        for stage, plist in self.pipelines.items():
            for p in plist:
                if p.primary_source not in valid_sources:
                    raise ValueError(
                        f"pipeline {stage}.{p.name}: unknown primary_source "
                        f"{p.primary_source!r}; known: {sorted(valid_sources)}"
                    )
