"""Explicit inputs and solver state for native C-grid sea-ice dynamics.

Horizontal fields are float64 [bj, bi, j, i] native tiles (SeaiceSpace.shape; offline [2, 2, 27, 46]).
Forces use MITgcm signs (positive accelerates the indicated component).
The component returns candidates; publication belongs to the caller.
"""

from dataclasses import dataclass, fields
from typing import Protocol

import torch
from torch import Tensor

Trim = tuple[int, int, int, int]
Scalar = float | Tensor
FULL: Trim = (0, 0, 0, 0)
INNER: Trim = (3, 3, 3, 3)
D: Trim = (1, 1, 1, 1)
SOLVE: Trim = (2, 2, 2, 2)
CWRITE: Trim = (0, 1, 0, 1)
ZWRITE: Trim = (1, 0, 1, 0)


class DynamicsSpace(Protocol):
    """Spatial operations supplied by seaice.space, with no implicit exchange.

    read returns a rectangle shifted by dj/di within each native tile.
    write returns a new field with that rectangle replaced, retaining its band.
    ordered_sum folds one accumulator over tile, j, i in that order.
    """

    def read(self, field: Tensor, trim: Trim, dj: int = 0, di: int = 0) -> Tensor: ...
    def write(self, field: Tensor, values: Tensor, trim: Trim) -> Tensor: ...
    def exchange_vector(self, u: Tensor, v: Tensor, *, event: str) -> tuple[Tensor, Tensor]: ...
    def ordered_sum(self, field: Tensor, trim: Trim) -> Tensor: ...
    def ordered_sum_pair(self, first: Tensor, second: Tensor, trim: Trim) -> tuple[Tensor, Tensor]: ...
    def max_abs(self, field: Tensor, trim: Trim) -> Tensor: ...
    def tile(self, field: Tensor, d: int) -> Tensor: ...
    def stack_tiles(self, fields: list[Tensor]) -> Tensor: ...


#: Sea-ice dynamics choices this package implements, one entry each: the elliptical
#: rheology solved by LSR (MITgcm's SEAICEuseMCE/TD/PL, SEAICEuseJFNK/Krylov select the
#: others; both SEAICE_no_slip values are implemented, and LSR turns SEAICE_2ndOrderBC
#: off, seaice_readparms.F:907), the SEAICEetaZmethod 0 and 3 averages of
#: e12 to C points (physics.py, seaice_calc_viscosities.F:117-138; seaice_check.F:1160
#: admits no other) and zero tensile strength (program.py, _dynamics_geometry).
RHEOLOGIES = ("ellipse",)
SOLVERS = ("lsr",)
ETA_Z_METHODS = (0, 3)
TENSILE_FACTORS = (0.0,)
#: SEAICEuseDYNAMICS values of a SEAICE_BGRID_DYNAMICS build: only the ocean stress of
#: ostres.F is ported (physics.bgrid_ocean_stress), not the B-grid LSR of lsr.F.
BGRID_USE_DYNAMICS = (False,)
#: (SEAICE_OLx, SEAICE_OLy): the LSR rectangle iMin=1-SEAICE_OLx.. (seaice_lsr.F:176-179)
#: is the native trim SOLVE here and sweep._TRIM, both HALO-1-SEAICE_OLx.
LSR_OVERLAPS = ((1, 1),)
#: SEAICEselectMetricTerms (seaice_readparms.F:909-921): 1 and 2 differ from 0 only through the
#: k-coefficients of DynamicsGeometry (seaice_init_fixed.F:262-345); 2 adds the
#: stress-divergence metric terms (matrix.py, seaice_lsr.F:1740-1760,1896-1915).
METRIC_TERMS = (0, 1, 2)


@dataclass(frozen=True)
class DynamicsParameters:
    """Fixed branch choices and continuous controls for the T/L/B chains.

    Defaults: seaice_readparms.F:268-271,326,370-374,400-438,529-556.
    Required dt values and strength come from the selected case namelist;
    zeta_smooth_regularization is the build's SEAICE_ZETA_SMOOTHREG.
    The compiled rheology is elliptical.
    """

    dt_dyn: Scalar
    dt_therm: float
    strength: Scalar
    use_dynamics: bool
    use_lsr_flex: bool
    nonlinear_iterations: int
    zeta_smooth_regularization: bool
    eta_z_method: int
    no_slip: bool
    lsr_overlap: tuple[int, int]
    metric_terms: int
    linear_iterations: int = 1500
    check_interval: int = 2
    lsr_error: Scalar = 1.0e-5
    nonlinear_tolerance: Scalar = 1.0e-3
    relax_u: Scalar = 0.95
    relax_v: Scalar = 0.95
    mix_initial_guess: int = 1
    rho_ice: Scalar = 910.0
    rho_snow: Scalar = 330.0
    rho_air: Scalar = 1.2
    rho_ocean: Scalar = 1030.0
    air_drag: Scalar = 0.001
    air_drag_south: Scalar = 0.001
    water_drag: Scalar = 0.0055
    water_drag_south: Scalar = 0.0055
    water_drag_min: Scalar = 0.25
    air_turn_angle: Scalar = 0.0
    water_turn_angle: Scalar = 0.0
    stress_factor: Scalar = 1.0
    c_star: Scalar = 20.0
    area_max: Scalar = 1.0
    eccentricity: Scalar = 2.0
    flow_eccentricity: Scalar = 2.0
    delta_min: Scalar = 1.0e-10
    zeta_max_factor: Scalar = 2.5e8
    zeta_min: Scalar = 0.0
    pressure_replacement: Scalar = 1.0
    epsilon: Scalar = 1.0e-10
    add_snow_mass: bool = True
    scale_surface_stress: bool = True
    use_tilt: bool = True
    relative_wind: bool = False
    update_ocean_stress: bool = True

    def __post_init__(self) -> None:
        # mix=0 changes the dynsolver entry velocity via free drift.
        # INTERFACE excludes mix>=2, whose candidate replaces ice velocity.
        if self.mix_initial_guess != 1:
            raise ValueError("The T/L/B dynamics implementation requires LSR_mixIniGuess=1")
        if self.metric_terms not in METRIC_TERMS:
            raise ValueError(f"SEAICEselectMetricTerms must be one of {METRIC_TERMS}, got {self.metric_terms!r}")
        if self.lsr_overlap not in LSR_OVERLAPS:
            raise ValueError(f"(SEAICE_OLx, SEAICE_OLy) must be one of {LSR_OVERLAPS}, got {self.lsr_overlap!r}")
        if self.eta_z_method not in ETA_Z_METHODS:
            raise ValueError(f"SEAICEetaZmethod must be one of {ETA_Z_METHODS}, got {self.eta_z_method!r}")
        if min(self.linear_iterations, self.nonlinear_iterations, self.check_interval) < 1:
            raise ValueError("LSR iteration counts and check interval must be positive")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Tensor) and (value.dtype != torch.float64 or value.ndim != 0):
                raise ValueError(f"{item.name} must be a scalar float64 control")


@dataclass(frozen=True)
class DynamicsGeometry:
    dxF: Tensor
    dyF: Tensor
    dxC: Tensor
    dyC: Tensor
    dxV: Tensor
    dyU: Tensor
    rA: Tensor
    rAw: Tensor
    rAs: Tensor
    rAz: Tensor
    HEFFM: Tensor
    maskInC: Tensor
    SIMaskU: Tensor
    SIMaskV: Tensor
    seaiceMaskU: Tensor
    seaiceMaskV: Tensor
    maskInW: Tensor
    maskInS: Tensor
    YC: Tensor
    fCori: Tensor
    Bo_surf: Tensor
    k1AtC: Tensor
    k2AtC: Tensor
    k1AtZ: Tensor
    k2AtZ: Tensor
    k1AtU: Tensor
    k2AtU: Tensor
    k1AtV: Tensor
    k2AtV: Tensor
    tensileStrFac: Tensor
    globalArea: Tensor


@dataclass(frozen=True)
class IceState:
    HEFF: Tensor
    AREA: Tensor
    HSNOW: Tensor
    UICE: Tensor
    VICE: Tensor


@dataclass(frozen=True)
class DynamicsForcing:
    uwind: Tensor
    vwind: Tensor
    fu: Tensor
    fv: Tensor
    uVel: Tensor
    vVel: Tensor
    etaN: Tensor
    pLoad: Tensor


@dataclass(frozen=True)
class DynamicsCarry:
    """Native storage, including bands untouched by the current call."""

    DWATN: Tensor
    eta: Tensor
    zeta: Tensor
    etaZ: Tensor
    zetaZ: Tensor
    press: Tensor
    deltaC: Tensor
    e11: Tensor
    e22: Tensor
    e12: Tensor


@dataclass(frozen=True)
class WindStress:
    TAUX: Tensor
    TAUY: Tensor
    uwind: Tensor
    vwind: Tensor


@dataclass(frozen=True)
class PreparedDynamics:
    wind: WindStress
    seaiceMassC: Tensor
    seaiceMassU: Tensor
    seaiceMassV: Tensor
    FORCEX0: Tensor
    FORCEY0: Tensor
    PRESS0: Tensor
    SEAICE_zMax: Tensor
    SEAICE_zMin: Tensor


@dataclass(frozen=True)
class Linearization:
    uIceC: Tensor
    vIceC: Tensor
    carry: DynamicsCarry


@dataclass(frozen=True)
class LinearSystem:
    rhsU: Tensor
    rhsV: Tensor
    AU: Tensor
    BU: Tensor
    CU: Tensor
    AV: Tensor
    BV: Tensor
    CV: Tensor
    uRt1: Tensor
    uRt2: Tensor
    vRt1: Tensor
    vRt2: Tensor


@dataclass(frozen=True)
class Record:
    """A paired event with the sequence number first in its integer record."""

    integers: Tensor
    reals: Tensor


@dataclass(frozen=True)
class ScanState:
    """L6(b): old, pre-exchange, and post-exchange native velocities."""

    myIter: int
    ipass: int
    m: int
    old_u: Tensor
    old_v: Tensor
    before_exchange_u: Tensor
    before_exchange_v: Tensor
    after_exchange_u: Tensor
    after_exchange_v: Tensor


@dataclass(frozen=True)
class PassState:
    myIter: int
    ipass: int
    linearization: Linearization
    system: LinearSystem
    control: Record
    flex_control: Tensor | None
    checks: tuple[Record, ...]
    scans: tuple[ScanState, ...]
    end: Record | None
    UICE: Tensor
    VICE: Tensor
    final_residual_u: Tensor | None
    final_residual_v: Tensor | None


@dataclass(frozen=True)
class LsrResult:
    """S1/S2 in checks are N/m2 for flex, m/s for ordinary LSR."""

    UICE: Tensor
    VICE: Tensor
    carry: DynamicsCarry
    passes: tuple[PassState, ...]


@dataclass(frozen=True)
class DynamicsResult:
    wind: WindStress
    gate: Record
    prepared: PreparedDynamics | None
    lsr: LsrResult | None
    UICE: Tensor
    VICE: Tensor
    fu: Tensor
    fv: Tensor
    carry: DynamicsCarry


def scalar(value: Scalar, reference: Tensor) -> Tensor:
    if isinstance(value, Tensor) or torch.compiler.is_compiling():
        return torch.as_tensor(value, dtype=torch.float64, device=reference.device)
    # A fill launch rather than a host copy: an eager stage may run inside a
    # CUDA graph capture, and a pageable copy also waits for the stream.
    return torch.full((), value, dtype=torch.float64, device=reference.device)


@torch.compiler.disable(reason="pack host event integers without specializing each scan counter")
def record(integers: tuple[int, ...], reals: tuple[Scalar, ...], reference: Tensor) -> Record:
    """Integers stay on the host: a device copy would synchronize every scan."""
    return Record(
        torch.tensor(integers, dtype=torch.int32),
        torch.stack([scalar(v, reference) for v in reals]) if reals else reference.new_empty((0,)),
    )
