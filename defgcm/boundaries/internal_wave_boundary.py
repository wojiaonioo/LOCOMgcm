"""Internal-wave open boundary condition.

Translates MITgcm ``obcs_calc.F`` from ``verification/internal_wave``:
- **West boundary**: inject mode-1 internal-wave signal in u and T; salinity
  ``west_salt``.
- **East boundary**: clamped u=0, v=0, T=Tref, salinity ``east_salt``
  (useOrlanskiEast=F branch).
- Boundary placement, tangential handling and masks use the shared prescribed path.

The boundary salinities are configuration, one profile per side and no default:
the upstream file never assigns OBWs, which OBCS_INIT_VARIABLES leaves at 0
(pkg/obcs/obcs_init_variables.F:136), and sets OBEs=sRef(K) (obcs_calc.F:95); a
configuration whose MITgcm build assigns OBWs otherwise passes that value.

Layer-center depths are positive downward in LOCOMgcm and negated at the
analytic formula seam to obtain MITgcm rC.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from defgcm.geometry import StructuredCGrid

from .open import Prescribed3DBoundaryCondition, SideBoundaryTensorSet
from defgcm.boundaries._registry import register_boundary_condition


def build_vertical_structure(rC: torch.Tensor, rF_bottom: torch.Tensor, drF: torch.Tensor) -> torch.Tensor:
    """``vertStructWst`` of the upstream file (obcs_calc.F:59-67), statement by statement.

    ``rC`` [nz] are the MITgcm layer centres (negative), ``rF_bottom`` is
    rF(Nr+1) (negative), ``drF`` [nz] the layer thicknesses (positive).  The
    sum runs k = 1..Nr in order and is divided by rF(Nr+1) before it is
    subtracted, as upstream writes it.
    """
    mz = 1.0
    vert = torch.cos(mz * math.pi * (rC / rF_bottom))
    tmpsum = vert.new_zeros(())
    for k in range(vert.shape[0]):
        tmpsum = tmpsum + vert[k] * drF[k]
    tmpsum = tmpsum / rF_bottom
    return vert - tmpsum


def west_u_profile(
    vert: torch.Tensor, *, u_inflow: float, ob_timescale: float, f0: float, N2: float,
    west_dx: float, time_seconds: float,
) -> torch.Tensor:
    """``OBWu`` [nz] of the upstream file (obcs_calc.F:70-73, 112-120), operands in the
    upstream order: kx = mz*2π/400*sqrt((ω²-f0²)/(N²-ω²)) with ω² written 2π·2π/(T·T),
    OBWu = Uinflow*vert*sin(2π·t/T)*cos(kx*(3-2-0.5)*delX(1)).  ``vert`` is
    ``build_vertical_structure``'s profile, ``west_dx`` delX(1)."""
    obt = ob_timescale
    t = time_seconds
    mz = 1.0
    omega2 = 2.0 * math.pi * 2.0 * math.pi / (obt * obt)
    f02 = f0 * f0
    kx = mz * 2.0 * math.pi / 400.0 * math.sqrt(
        max(omega2 - f02, 0.0) / max(N2 - omega2, 1e-30)
    )
    cos_kx_factor = math.cos(kx * (3.0 - 2.0 - 0.5) * west_dx)
    return u_inflow * vert * math.sin(2.0 * math.pi * t / obt) * cos_kx_factor


def west_temp_profile(
    tref: torch.Tensor, *, u_inflow: float, ob_timescale: float, f0: float, N2: float,
    gravity: float, alpha: float, time_seconds: float,
) -> torch.Tensor:
    """``OBWt`` [nz] of the upstream file (obcs_calc.F:130-135), operands in the upstream
    order: OBWt = tRef(K) + Uinflow*sin(mz*π*(k-0.5)/Nr)*sin(2π·t/T)
    *sqrt(strat/(tAlpha*g))*sqrt(2π/T*2π/T - f0²)/(2π/T), strat = N²/(g*tAlpha)."""
    obt = ob_timescale
    t = time_seconds
    strat = N2 / (gravity * alpha)
    mz = 1.0
    nz = tref.shape[0]
    Nr = float(nz)
    omega2_minus_f02 = 2.0 * math.pi / obt * 2.0 * math.pi / obt - f0 * f0
    if omega2_minus_f02 < 0:
        omega2_minus_f02 = 0.0

    k_indices = torch.arange(1, nz + 1, dtype=tref.dtype, device=tref.device)
    sin_struct = torch.sin(mz * math.pi * (k_indices - 0.5) / Nr)  # [nz]
    t_perturbation = (u_inflow * sin_struct * math.sin(2.0 * math.pi * t / obt)
                      * math.sqrt(strat / (alpha * gravity))
                      * math.sqrt(omega2_minus_f02) / (2.0 * math.pi / obt))
    return tref + t_perturbation


@dataclass
@register_boundary_condition("internal_wave_open")
class InternalWaveOpenBoundaryCondition(Prescribed3DBoundaryCondition):
    """Open boundary condition for the MITgcm ``internal_wave`` test case.

    Attributes
    ----------
    ob_timescale : float
        Wave period T in seconds (default 44567 s).
    u_inflow : float
        Velocity amplitude (default 0.024 m/s).
    f0 : float
        Coriolis parameter (0 for this test case).
    N2 : float
        Buoyancy frequency squared (1e-6 s⁻²).
    gravity : float
    alpha : float
        Thermal expansion coefficient for linear EOS.
    nz : int
    Tref : torch.Tensor or sequence of float
        Reference temperature profile [nz]. Serialized configuration supplies
        this value as a numeric sequence.
    west_salt, east_salt : torch.Tensor or sequence of float
        Salinity profile [nz] the west and the east boundary cells take
        (OBWs and OBEs); required, see the module docstring.
    """

    ob_timescale: float = 44567.0
    u_inflow: float = 0.024
    f0: float = 0.0
    N2: float = 1.0e-6
    gravity: float = 9.81
    alpha: float = 2.0e-4
    nz: int = 20
    Tref: torch.Tensor | list[float] | tuple[float, ...] = field(kw_only=True)
    west_salt: torch.Tensor | list[float] | tuple[float, ...] = field(kw_only=True)
    east_salt: torch.Tensor | list[float] | tuple[float, ...] = field(kw_only=True)
    enforce_obcs_balance: bool = False
    apply_tangential_velocity: bool = True
    name: str = "internal_wave_open_boundary"
    supports_open_boundary: bool = True
    supports_barotropic_open_boundary: bool = True

    _sampling_grid: StructuredCGrid | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.enforce_obcs_balance:
            raise ValueError("internal_wave_open requires enforce_obcs_balance=False")
        if self.tracer_source is not None or self.barotropic_source is not None:
            raise ValueError("internal_wave_open produces its own analytic boundary tensors")
        super().__post_init__()

    def bind_sampling_grid(self, grid: StructuredCGrid) -> None:
        self._sampling_grid = grid

    def _runtime_fields(
        self, device: torch.device | str, dtype: torch.dtype | None,
    ) -> dict[str, object]:
        # Placement of the inherited fields (fallback, tides) stays with
        # Prescribed3DBoundaryCondition.to_runtime. Geometry is bound by
        # assembly after its own runtime conversion.
        return {
            **super()._runtime_fields(device, dtype),
            "Tref": torch.as_tensor(self.Tref, dtype=dtype, device=device),
            "west_salt": torch.as_tensor(self.west_salt, dtype=dtype, device=device),
            "east_salt": torch.as_tensor(self.east_salt, dtype=dtype, device=device),
        }

    def update_model_time(self, model_time_seconds: float) -> None:
        """Produce complete boundary tensors eagerly at the resolved clock."""
        grid = self._sampling_grid
        if grid is None:
            raise ValueError("internal_wave_open: bind_sampling_grid is required before sampling")
        # Runtime conversion fixes the producer dtype before the first sample,
        # preserving the promotion points previously used during enforcement.
        tref = torch.as_tensor(self.Tref)
        dtype, device = tref.dtype, tref.device
        t = float(model_time_seconds)
        self.current_time_seconds = t  # telemetry/cache key only
        west_u = self._west_u(grid, time_seconds=t, dtype=dtype, device=device)
        zeros = torch.zeros_like(west_u)
        self._active_u_values = SideBoundaryTensorSet(west=west_u, east=zeros)
        # Upstream OBWv keeps its ramp even though U/T have it commented out.
        # obcs_calc.F:121-129, operands in the upstream order.
        obt = self.ob_timescale
        vert = self._vertical_structure(grid, dtype=dtype, device=device)
        ramp_time2 = 4.0 * obt
        west_v = (self.u_inflow * self.f0 / (2.0 * math.pi / obt) * vert
                  * math.cos(2.0 * math.pi * t / obt)
                  * (math.exp(t / ramp_time2) - math.exp(-t / ramp_time2))
                  / (math.exp(t / ramp_time2) + math.exp(-t / ramp_time2)))
        self._active_v_values = SideBoundaryTensorSet(
            west=west_v.unsqueeze(-1).expand(grid.nz, grid.ny), east=zeros)
        self._active_temp_values = SideBoundaryTensorSet(
            west=self._west_temp(grid, time_seconds=t, dtype=dtype, device=device),
            east=self._east_temp(grid, dtype=dtype, device=device))
        west_s = torch.as_tensor(self.west_salt, dtype=dtype, device=device)
        east_s = torch.as_tensor(self.east_salt, dtype=dtype, device=device)
        self._active_salt_values = SideBoundaryTensorSet(
            west=west_s.expand(grid.nz).unsqueeze(-1).expand(grid.nz, grid.ny),
            east=east_s.expand(grid.nz).unsqueeze(-1).expand(grid.nz, grid.ny))
        self._balanced_u_values = self._balanced_v_values = None
        self._balance_cache_key = self._prepared_balanced_targets = None

    # -----------------------------------------------------------------
    # West boundary: inject mode-1 internal wave
    # -----------------------------------------------------------------
    def _vertical_structure(self, grid: StructuredCGrid, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """``vertStructWst`` on this grid: rC and rF(Nr+1) are the negated LOCOMgcm depths."""
        interface = grid.interface_depth.to(dtype=dtype, device=device)
        rC = -grid.center_depth.to(dtype=dtype, device=device)
        return build_vertical_structure(rC, -interface[-1], grid.layer_thickness.to(dtype=dtype, device=device))

    def _west_u(self, grid: StructuredCGrid, *, time_seconds: float, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """u at western boundary face: shape [nz, ny]."""
        vert = self._vertical_structure(grid, dtype=dtype, device=device)
        u_boundary = west_u_profile(
            vert, u_inflow=self.u_inflow, ob_timescale=self.ob_timescale, f0=self.f0,
            N2=self.N2, west_dx=grid.metrics.west_dx, time_seconds=time_seconds,
        )  # [nz]
        return u_boundary.unsqueeze(-1).expand(grid.nz, grid.ny)

    def _west_temp(self, grid: StructuredCGrid, *, time_seconds: float, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Temperature at western boundary: Tref + perturbation, shape [nz, ny]."""
        t_boundary = west_temp_profile(
            torch.as_tensor(self.Tref, dtype=dtype, device=device), u_inflow=self.u_inflow,
            ob_timescale=self.ob_timescale, f0=self.f0, N2=self.N2, gravity=self.gravity,
            alpha=self.alpha, time_seconds=time_seconds,
        )  # [nz]
        return t_boundary.unsqueeze(-1).expand(grid.nz, grid.ny)

    # -----------------------------------------------------------------
    # East boundary: clamped u=0, v=0, T=Tref (salinity: east_salt)
    # MITgcm obcs_calc.F:90-95 (useOrlanskiEast=F branch)
    # -----------------------------------------------------------------
    def _east_temp(self, grid: StructuredCGrid, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Clamped to Tref at east boundary, shape [nz, ny]."""
        Tref = self.Tref
        Tref = torch.as_tensor(Tref, dtype=dtype, device=device)
        return Tref.unsqueeze(-1).expand(grid.nz, grid.ny)
