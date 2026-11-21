from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ._rounded_product import rounded_multiply as _mdjwf_multiply

from contracts.valdump import DumpTap, TapSpec
from locomgcm.fields import Field, FieldLocation
from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.vertical_coordinates import reference_layer_centres
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyStepDomain
from locomgcm.dynamics._registry import register_dynamics_module


ReferenceValue = float | Sequence[float] | torch.Tensor


_EOS_DUMP_SPEC = TapSpec(
    variables=("theta", "salt", "pressure_dbar", "rho_anomaly_v1"),
    static_fields=("wet_mask", "z_center"),
    shape_field="theta",
    source_files=(__file__,),
    marker="EOS-DUMP",
)
_EOS_DUMP_TAP: DumpTap | None = None
_EOS_DUMP_RESOLVED = False


def _eos_dump_tap() -> DumpTap:
    global _EOS_DUMP_RESOLVED, _EOS_DUMP_TAP
    if not _EOS_DUMP_RESOLVED:
        _EOS_DUMP_TAP = DumpTap.from_env("LOCOMGCM_EOS_DUMP_DIR", spec=_EOS_DUMP_SPEC)
        _EOS_DUMP_RESOLVED = True
    assert _EOS_DUMP_TAP is not None
    return _EOS_DUMP_TAP


def resolve_eos_dump_tap() -> None:
    """Resolve the dump tap now, in eager code. **Eager (prepare).**

    The tap is process-once and reads its environment on first use.  When that
    first use falls inside the traced step, the tap's lock construction is a
    graph break (``ImplicitFreeSurfaceStepper.prepare_step_host`` calls this so
    it never does).
    """
    _eos_dump_tap()


def _eos_dump_array(tensor: torch.Tensor):
    import numpy as np

    array = tensor.detach().cpu().numpy()
    if array.dtype != np.float64:
        array = array.astype(np.float64)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    return array


def _eos_dump_statics(grid):
    import numpy as np

    hfac = getattr(getattr(grid, "masks", None), "hFacC", None)
    if hfac is not None:
        wet_mask = (hfac > 0).detach().cpu().numpy().astype(np.float64)
    else:
        wet_mask = grid.masks.center.detach().cpu().numpy().astype(np.float64)
    dz = grid.layer_thickness.to(dtype=torch.float64, device="cpu")
    _, z_center = reference_layer_centres(dz)
    return {"wet_mask": wet_mask, "z_center": z_center.numpy()}


def _eos_dump_metadata(module):
    import os

    kind = "mdjwf" if module.eos_type == "MDJWF" else "jmd95"
    coefficients = (
        {"MDJWF_NUM": list(_MDJWF_NUM), "MDJWF_DEN": list(_MDJWF_DEN)}
        if kind == "mdjwf"
        else {
            "JMDCFW": list(_JMDCFW),
            "JMDCSW": list(_JMDCSW),
            "JMDCKFW": list(_JMDCKFW),
            "JMDCKSW": list(_JMDCKSW),
            "JMDCKP": list(_JMDCKP),
        }
    )
    return {
        "eos": {
            "v1_module_name": module.name,
            "kind": kind,
            "params_runtime": {
                "eos_type": module.eos_type,
                "reference_density": float(module.reference_density),
                "gravity": float(module.gravity),
                "dynamic_pressure": bool(module.dynamic_pressure),
                "kernel_pressure_argument_units": "Pa",
                "kernel_internal_pressure_scale": (
                    _SI_TO_DBAR if kind == "mdjwf" else _SI_TO_BAR
                ),
                "kernel_internal_pressure_unit": (
                    "dbar" if kind == "mdjwf" else "bar"
                ),
                "salinity_clamp_min": 0.0,
                "coefficients": coefficients,
                "extraction": (
                    "reference_density/gravity/dynamic_pressure/eos_type read off the "
                    "live dynamics module instance at the kernel call site; the "
                    "coefficient tables and the pressure scale are read off the live "
                    "locomgcm.dynamics.equation_of_state module globals actually "
                    "consumed by the kernel — no defaults assumed"
                ),
            },
        },
        "conventions": {
            "theta": "potential temperature degC",
            "pressure": (
                "dbar = pressure_pa * 1e-4; the array is the kernel's actual "
                "pressure argument (Pa) rescaled to dbar, kept in its unbroadcast "
                "shape (see extras.pressure_shape) — broadcast to [nz,ny,nx] before use"
            ),
            "wet_mask": "1=wet ocean, 0=dry/land, center point",
            "z_center": "LOCOMgcm downward-positive depth (m)",
        },
        "env": {
            name: os.environ.get(name)
            for name in (
                "IVADV",
                "FUSED",
                "LOCOMGCM_IVADV_FUSED_BACKEND",
                "LOCOMGCM_KPP_STATEKPP_CUDA",
                "LOCOMGCM_CG3D_FUSED_PCG",
            )
        },
    }


def _capture_eos_dump(tap, module, temp, salt, pressure, density_anomaly, grid) -> None:
    tap.capture(
        density_anomaly,
        arrays=lambda: {
            "theta": _eos_dump_array(temp),
            "salt": _eos_dump_array(salt),
            "pressure_dbar": _eos_dump_array(pressure) * _SI_TO_DBAR,
            "rho_anomaly_v1": _eos_dump_array(density_anomaly),
        },
        statics=lambda: _eos_dump_statics(grid),
        metadata=lambda: _eos_dump_metadata(module),
    )


try:  # keep observation I/O opaque to torch compilation
    _capture_eos_dump = torch._dynamo.disable(_capture_eos_dump)  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older torch without _dynamo
    pass


def _broadcast_reference_value(
    reference: ReferenceValue,
    field: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    ref = torch.as_tensor(reference, dtype=field.dtype, device=field.device)
    if ref.ndim == 0:
        return ref
    if field.ndim == 0:
        raise ValueError(f"{name} must be scalar when the target field is scalar")

    vertical_axis = 0 if field.ndim < 3 else field.ndim - 3
    if ref.ndim == 1 and ref.shape[0] == field.shape[vertical_axis]:
        shape = [1] * field.ndim
        shape[vertical_axis] = ref.shape[0]
        return ref.view(*shape)

    if ref.ndim <= field.ndim:
        padded_shape = (1,) * (field.ndim - ref.ndim) + tuple(ref.shape)
        if all(size == 1 or size == target for size, target in zip(padded_shape, field.shape)):
            return ref.view(*padded_shape)

    raise ValueError(
        f"{name} with shape {tuple(ref.shape)} is not broadcastable to field shape {tuple(field.shape)}"
    )


def zero_negative_salinity(salt: torch.Tensor) -> torch.Tensor:
    """MITgcm LOOK_FOR_NEG_SALINITY (find_rho.F:1189-1250): negative salt becomes 0.

    Every cell, wet or dry, is tested; -0.0 and NaN are kept.  FIND_RHO_2D calls it on the
    salt it is handed, before any EOS branch, when CHECK_SALINITY_FOR_NEGATIVE_VALUES is
    defined (find_rho.F:85-89).
    """
    return torch.where(salt < 0.0, torch.zeros_like(salt), salt)


def linear_density_anomaly(
    temp: torch.Tensor,
    salt: torch.Tensor,
    *,
    reference_density: float | torch.Tensor,
    rho_const: float | torch.Tensor,
    thermal_expansion: float,
    haline_contraction: float,
    reference_temperature: ReferenceValue,
    reference_salinity: ReferenceValue,
) -> torch.Tensor:
    """MITgcm LINEAR density anomaly w.r.t. ``rho_const`` (find_rho.F:98-108).

    ``reference_density`` is rhoNil: ``rhoNil*(sBeta*(S-sRef) - tAlpha*(T-tRef))
    + (rhoNil - rho_const)``.  A caller without rhoConst passes its rhoNil and
    gets the anomaly w.r.t. rhoNil.
    """
    reference_temperature_tensor = _broadcast_reference_value(
        reference_temperature,
        temp,
        name="reference_temperature",
    )
    reference_salinity_tensor = _broadcast_reference_value(
        reference_salinity,
        salt,
        name="reference_salinity",
    )
    temperature_term = thermal_expansion * (temp - reference_temperature_tensor)
    salinity_term = haline_contraction * (salt - reference_salinity_tensor)
    rho = torch.as_tensor(reference_density, dtype=temp.dtype, device=temp.device)
    d_rho = rho - torch.as_tensor(rho_const, dtype=temp.dtype, device=temp.device)
    return rho * (-temperature_term + salinity_term) + d_rho


def linear_density(
    temp: torch.Tensor,
    salt: torch.Tensor,
    *,
    reference_density: float | torch.Tensor,
    thermal_expansion: float,
    haline_contraction: float,
    reference_temperature: ReferenceValue,
    reference_salinity: ReferenceValue,
) -> torch.Tensor:
    reference_density = torch.as_tensor(reference_density, dtype=temp.dtype, device=temp.device)
    density_anomaly = linear_density_anomaly(
        temp,
        salt,
        reference_density=reference_density,
        rho_const=reference_density,
        thermal_expansion=thermal_expansion,
        haline_contraction=haline_contraction,
        reference_temperature=reference_temperature,
        reference_salinity=reference_salinity,
    )
    return density_anomaly + reference_density


#: MITgcm ``eosType`` of each equation-of-state module (``ini_eos.F:84,105-106,233``),
#: as (eosType, module name, ``dynamic_pressure`` or None when the module has a
#: single type).  The MITgcm names and their modules are written only here.
EOS_TYPES = (
    ("LINEAR", "linear_equation_of_state", None),
    ("JMD95Z", "jmd95_equation_of_state", False),
    ("JMD95P", "jmd95_equation_of_state", True),
    ("MDJWF", "mdjwf_equation_of_state", None),
)


def mit_eos_type(module_name: str, dynamic_pressure: bool = False) -> str:
    """MITgcm ``eosType`` of an equation-of-state module, read from ``EOS_TYPES``."""
    for eos_type, name, dynamic in EOS_TYPES:
        if name == module_name and dynamic in (None, bool(dynamic_pressure)):
            return eos_type
    raise ValueError(f"{module_name!r} is not an equation-of-state module of EOS_TYPES")


def validate_dynamic_pressure_composition(*, modules: Sequence, stepper, face_count: int) -> None:
    """Refuse a dynamic-pressure EOS that no component can feed.

    ``dynamic_pressure`` (MITgcm JMD95P, MDJWF) evaluates the EOS at last
    step's total hydrostatic potential (pressure_for_eos.F:71-86).  Only the
    regional ``ImplicitFreeSurfaceStepper`` carries that history: it seeds it
    like INI_PRESSURE and refreshes it from the ``hydrostatic_pressure_gradient``
    module's ``total_hydrostatic_potential``, which the module publishes with
    the momentum pressure forcing off as well (MITgcm runs CALC_PHI_HYD
    unconditionally, dynamics.F:482).  The cube step has neither the cache nor
    the seed.  One rule for the regional modules and the cube's EOS knobs.
    """
    requests = [f"model.dynamics[{module.name}]" for module in modules
                if getattr(module, "dynamic_pressure", False)]
    if getattr(stepper, "cube_eos_dynamic_pressure", False):
        requests.append("model.stepper.kwargs.cube_eos_dynamic_pressure")
    cube_eos = getattr(getattr(stepper, "cube_static_state", None), "eos", None)
    if getattr(cube_eos, "dynamic_pressure", False):
        requests.append("the cube static state's eos")
    if not requests:
        return
    from locomgcm.dynamics.hydrostatic_pressure import HydrostaticPressureGradient
    from locomgcm.steppers.implicit_free_surface import ImplicitFreeSurfaceStepper

    missing = []
    if not isinstance(stepper, ImplicitFreeSurfaceStepper):
        missing.append(f"a stepper that carries the pressure history (implicit_free_surface, "
                       f"not {type(stepper).__name__})")
    if face_count != 1:
        missing.append("a regional grid (the cube step has no pressure-history cache or "
                       "INI_PRESSURE seed)")
    elif not any(isinstance(module, HydrostaticPressureGradient) for module in modules):
        missing.append("a hydrostatic_pressure_gradient module, the only publisher of the total "
                       "potential (it publishes with momentum_pressure_forcing off too)")
    if missing:
        raise ValueError(
            f"dynamic_pressure ({', '.join(requests)}) needs last step's total hydrostatic "
            f"potential; missing: {'; '.join(missing)}"
        )


@dataclass(frozen=True)
@register_dynamics_module("linear_equation_of_state")
class LinearEquationOfState(TendencyModule):
    reference_density: float | torch.Tensor = 1_025.0
    thermal_expansion: float = 2.0e-4
    haline_contraction: float = 7.4e-4
    reference_temperature: ReferenceValue = 0.0
    reference_salinity: ReferenceValue = 0.0
    name: str = "linear_equation_of_state"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW

    def __post_init__(self) -> None:
        if not isinstance(self.reference_density, torch.Tensor):
            object.__setattr__(self, "reference_density", torch.as_tensor(
                self.reference_density, dtype=torch.float64
            ))
        if self.reference_density.ndim != 0 or not bool(torch.isfinite(self.reference_density)):
            raise ValueError("LinearEquationOfState.reference_density must be a finite scalar")

    def density_anomaly(self, state: PrognosticState) -> torch.Tensor:
        """The anomaly w.r.t. rhoNil (``reference_density``, the field metadata's
        reference); the module holds no rhoConst."""
        return linear_density_anomaly(
            state.temp.data,
            state.salt.data,
            reference_density=self.reference_density,
            rho_const=self.reference_density,
            thermal_expansion=self.thermal_expansion,
            haline_contraction=self.haline_contraction,
            reference_temperature=self.reference_temperature,
            reference_salinity=self.reference_salinity,
        )

    def density(self, state: PrognosticState) -> torch.Tensor:
        return self.density_anomaly(state) + self.reference_density.to(state.temp.data)

    def populate_diagnostic_state(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> None:
        del grid
        density_anomaly = self.density_anomaly(state)
        density = density_anomaly + self.reference_density.to(density_anomaly)
        diagnostics.set_field(
            Field(
                "density_anomaly",
                density_anomaly,
                FieldLocation.CENTER,
                "kg m-3",
                mask_ref="center",
                metadata={"reference_density": self.reference_density},
            )
        )
        diagnostics.set_field(
            Field(
                "density",
                density,
                FieldLocation.CENTER,
                "kg m-3",
                mask_ref="center",
                metadata={"reference_density": self.reference_density},
            )
        )

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        if "density_anomaly" not in diagnostics.fields:
            self.populate_diagnostic_state(state, diagnostics, grid)
        return StateTendency.zeros_like(state)


# ──────────────────────────────────────────────────────────────────────
# JMD95 Nonlinear Equation of State
# ──────────────────────────────────────────────────────────────────────
# Coefficients from MITgcm ``ini_eos.F`` L127-228 (Jackett & McDougall 1995).

# Fresh water density polynomial ρ_fw(T), 6 coefficients
_JMDCFW = [
    999.842594e0,
    6.793952e-02,
    -9.095290e-03,
    1.001685e-04,
    -1.120083e-06,
    6.536332e-09,
]

# Salinity contribution to density at surface pressure, 9 coefficients
_JMDCSW = [
    8.24493e-01,
    -4.0899e-03,
    7.6438e-05,
    -8.2467e-07,
    5.3875e-09,
    -5.72466e-03,
    1.0227e-04,
    -1.6546e-06,
    4.8314e-04,
]

# Secant bulk modulus: fresh water component, 5 coefficients
_JMDCKFW = [
    1.965933e+04,
    1.444304e+02,
    -1.706103e+00,
    9.648704e-03,
    -4.190253e-05,
]

# Secant bulk modulus: salinity component at surface pressure, 7 coefficients
_JMDCKSW = [
    5.284855e+01,
    -3.101089e-01,
    6.283263e-03,
    -5.084188e-05,
    3.886640e-01,
    9.085835e-03,
    -4.619924e-04,
]

# Secant bulk modulus: pressure component, 14 coefficients
_JMDCKP = [
    3.186519e+00,
    2.212276e-02,
    -2.984642e-04,
    1.956415e-06,
    6.704388e-03,
    -1.847318e-04,
    2.059331e-07,
    1.480266e-04,
    2.102898e-04,
    -1.202016e-05,
    1.394680e-07,
    -2.040237e-06,
    6.128773e-08,
    6.207323e-10,
]

# SI to bar conversion factor (MITgcm ``EOS.h``)
_SI_TO_BAR = 1.0e-05

# SI to DECIBAR conversion (MITgcm ``SItodBar``).  MDJWF pressure terms are
# calibrated in decibar (×1e-4), NOT JMD95's bar (×1e-5); using bar here would
# silently bias the pressure dependence by ~10×.
_SI_TO_DBAR = 1.0e-04

# McDougall, Jackett, Wright & Feistel (2003) EOS — rational function
# ρ = Pnum / Pden in (θ, S, p[dbar]).  Coefficients from MITgcm
# ``model/src/ini_eos.F:235-260`` (eosMDJWFnum 0..11, eosMDJWFden 0..12).
_MDJWF_NUM = [
    9.99843699e+02,
    7.35212840e+00,
    -5.45928211e-02,
    3.98476704e-04,
    2.96938239e+00,
    -7.23268813e-03,
    2.12382341e-03,
    1.04004591e-02,
    1.03970529e-07,
    5.18761880e-06,
    -3.24041825e-08,
    -1.23869360e-11,
]
_MDJWF_DEN = [
    1.00000000e+00,
    7.28606739e-03,
    -4.60835542e-05,
    3.68390573e-07,
    1.80809186e-10,
    2.14691708e-03,
    -9.27062484e-06,
    -1.78343643e-10,
    4.76534122e-06,
    1.63410736e-09,
    5.30848875e-06,
    -3.03175128e-16,
    -1.27934137e-17,
]


def _store_reference_column(column: torch.Tensor) -> torch.Tensor:
    return column.clone()


if hasattr(torch.compiler, "nested_compile_region"):
    _store_reference_column = torch.compiler.nested_compile_region(_store_reference_column)


def _published_reference(column: torch.Tensor) -> torch.Tensor:
    # SET_REF_STATE stores this product before PRESSURE_FOR_EOS consumes it.
    # Eager multiplication already publishes a tensor with the required rounding.
    if torch.compiler.is_compiling() and isinstance(column, torch.Tensor):
        return _store_reference_column(column)
    return column


def reference_pressure_from_depth(depth: torch.Tensor, *, rho_const: float, gravity: float) -> torch.Tensor:
    """MITgcm pRef4EOS for z coordinates with uniform gravity (set_ref_state.F)."""
    return _published_reference((rho_const * depth) * gravity)


def reference_pressure_from_layers(
    layer_thickness: torch.Tensor, *, rho_const: float, gravity: float,
) -> torch.Tensor:
    """MITgcm pRef4EOS at the ordered cell centers of a constant-gravity z grid."""
    _, centers = reference_layer_centres(layer_thickness)
    return reference_pressure_from_depth(centers, rho_const=rho_const, gravity=gravity)


def _reference_geopotential(dz: torch.Tensor, gravity: float) -> torch.Tensor:
    """MIT INI_VERTICAL_GRID center distances and SET_REF_STATE stored phiRef(2*k)."""
    _, centers = reference_layer_centres(dz)
    return _published_reference(centers * gravity)


def _jmd95_safe_root(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the JMD95 square roots with a finite zero derivative at closed branches."""
    positive = x > 0.0
    safe = torch.where(positive, x, torch.ones_like(x))
    return torch.where(positive, torch.sqrt(safe), torch.zeros_like(x))


def _jmd95_surface_density_and_bulk_mod(t, s, s_root, p_bar, t2, t3):
    """FIND_RHOP0 and FIND_BULKMOD are shared by FIND_RHO_2D and FIND_ALPHA/BETA."""
    t4 = t3 * t
    s3o2 = s * s_root
    rfresh = (
        _JMDCFW[0]
        + _JMDCFW[1] * t
        + _JMDCFW[2] * t2
        + _JMDCFW[3] * t3
        + _JMDCFW[4] * t4
        + _JMDCFW[5] * t4 * t
    )

    # Salinity contribution
    rsalt = s * (
        _JMDCSW[0]
        + _JMDCSW[1] * t
        + _JMDCSW[2] * t2
        + _JMDCSW[3] * t3
        + _JMDCSW[4] * t4
    ) + s3o2 * (
        _JMDCSW[5]
        + _JMDCSW[6] * t
        + _JMDCSW[7] * t2
    ) + _JMDCSW[8] * s * s

    rhoP0 = rfresh + rsalt

    # ── Secant bulk modulus Kw ──
    # Fresh water component
    bMfresh = (
        _JMDCKFW[0]
        + _JMDCKFW[1] * t
        + _JMDCKFW[2] * t2
        + _JMDCKFW[3] * t3
        + _JMDCKFW[4] * t4
    )

    # Salinity component at surface pressure
    bMsalt = s * (
        _JMDCKSW[0]
        + _JMDCKSW[1] * t
        + _JMDCKSW[2] * t2
        + _JMDCKSW[3] * t3
    ) + s3o2 * (
        _JMDCKSW[4]
        + _JMDCKSW[5] * t
        + _JMDCKSW[6] * t2
    )

    # Pressure component
    bMpres = p_bar * (
        _JMDCKP[0]
        + _JMDCKP[1] * t
        + _JMDCKP[2] * t2
        + _JMDCKP[3] * t3
    ) + p_bar * s * (
        _JMDCKP[4]
        + _JMDCKP[5] * t
        + _JMDCKP[6] * t2
    ) + p_bar * s3o2 * _JMDCKP[7] + p_bar * p_bar * (
        _JMDCKP[8]
        + _JMDCKP[9] * t
        + _JMDCKP[10] * t2
    ) + p_bar * p_bar * s * (
        _JMDCKP[11]
        + _JMDCKP[12] * t
        + _JMDCKP[13] * t2
    )

    bulkMod = bMfresh + bMsalt + bMpres

    return rhoP0, bulkMod


def jmd95_density_anomaly(
    temp: torch.Tensor,
    salt: torch.Tensor,
    pressure_pa: torch.Tensor | float = 0.0,
    *,
    reference_density: float = 1025.0,
) -> torch.Tensor:
    """Compute density anomaly ρ − ρ₀ using JMD95 EOS.

    Uses the JMD95 branch of MITgcm ``FIND_RHO_2D``, including the
    ``FIND_RHOP0`` and ``FIND_BULKMOD`` subtotals in ``model/src/find_rho.F``.

    Parameters
    ----------
    temp : Tensor
        Potential temperature [°C].
    salt : Tensor
        Salinity [PSU].  Negative values clamped to 0.
    pressure_pa : Tensor or float
        Pressure in Pascals (SI).  Converted to bar internally.
        For JMD95Z z-coordinates: ``pressure = rhoConst * g * |z|``.
        For JMD95P z-coordinates: ``pressure = rhoConst *
        (prev_total_phi_hyd + g * |z|)``.
    reference_density : float
        rhoConst [kg/m³].  Subtracted to return anomaly.

    Returns
    -------
    Tensor : ρ(T,S,p) − rhoConst  [kg/m³]

    Reduced-precision guardrail: the JMD95 secant-bulk-modulus polynomials lose
    ~1e-4 relative accuracy when evaluated in float32.  When ``temp`` is not
    float64 the whole kernel is evaluated in float64 and the anomaly cast back
    to the input dtype; float64 inputs run the original path unchanged.
    """
    _out_dtype = temp.dtype
    if _out_dtype is not torch.float64:
        temp = temp.to(torch.float64)
        salt = salt.to(torch.float64)
        if isinstance(pressure_pa, torch.Tensor):
            pressure_pa = pressure_pa.to(torch.float64)
    t = temp
    s = torch.clamp(salt, min=0.0)  # MITgcm clamps negative salinity
    # Safe sqrt for autograd: 1/(2*sqrt(s)) is inf at s=0 (land cells), so
    # feed sqrt a sentinel 1.0 in that branch and outer-where back to 0.
    # Forward stays bitwise: s>0 -> sqrt(s); s==0 -> 0.
    s_root = _jmd95_safe_root(s)

    p_bar = pressure_pa * _SI_TO_BAR

    # ── rhoP0: density at surface pressure ──
    # Fresh water
    t2 = t * t
    t3 = t2 * t
    rhoP0, bulkMod = _jmd95_surface_density_and_bulk_mod(t, s, s_root, p_bar, t2, t3)

    # ── Final density ──
    # rho = rhoP0 / (1 - p_bar / bulkMod)
    # Return anomaly: rho - rhoConst
    rho = rhoP0 / (1.0 - p_bar / bulkMod) if not isinstance(p_bar, (int, float)) or p_bar != 0.0 else rhoP0
    density_anomaly = rho - reference_density
    if _out_dtype is not torch.float64:
        density_anomaly = density_anomaly.to(_out_dtype)
    return density_anomaly


def _jmd95_polynomial(coefficients, temp):
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * temp + coefficient
    return value


def _jmd95_divided_difference(coefficients, temp_a, temp_b):
    # Horner's divided difference also evaluates the derivative at a == b.
    value = coefficients[-1]
    difference = 0.0
    for coefficient in reversed(coefficients[:-1]):
        difference = value + temp_b * difference
        value = value * temp_a + coefficient
    return difference


def _jmd95_difference_terms(temp, salt, root, pressure_bar):
    poly = _jmd95_polynomial
    f, w = _JMDCFW, _JMDCSW
    kf, ks, c = _JMDCKFW, _JMDCKSW, _JMDCKP
    surface_density = (
        poly(f, temp) + salt * poly(w[:5], temp)
        + salt * root * poly(w[5:8], temp) + w[8] * salt * salt
    )
    bulk = (
        poly(kf, temp) + salt * poly(ks[:4], temp)
        + salt * root * poly(ks[4:7], temp)
    )
    bulk = bulk + pressure_bar * (
        poly(c[:4], temp) + salt * poly(c[4:7], temp) + salt * root * c[7]
    ) + pressure_bar * pressure_bar * (
        poly(c[8:11], temp) + salt * poly(c[11:14], temp)
    )
    return surface_density, bulk


def jmd95_density_difference(
    temp_a: torch.Tensor,
    salt_a: torch.Tensor,
    temp_b: torch.Tensor,
    salt_b: torch.Tensor,
    pressure_pa: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Evaluate rho(a, p) - rho(b, p) without subtracting full densities.

    Uses the original JMD95 coefficients, salinity clamp and common pressure
    in Pa. For rho = R*K/(K-p), polynomial divided differences give delta R
    and delta K, then delta rho = delta R*K_a/(K_a-p)
    - R_b*p*delta K/((K_a-p)*(K_b-p)). Equal inputs have an exact zero value
    while retaining their independent temperature and salinity derivatives.
    Reduced-precision inputs use the same float64 island as density anomaly.
    """
    out_dtype = temp_a.dtype
    if out_dtype is not torch.float64:
        temp_a, salt_a, temp_b, salt_b = (
            value.to(torch.float64) for value in (temp_a, salt_a, temp_b, salt_b)
        )
        if isinstance(pressure_pa, torch.Tensor):
            pressure_pa = pressure_pa.to(torch.float64)
    salt_a, salt_b = salt_a.clamp(min=0.0), salt_b.clamp(min=0.0)

    root_a, root_b = _jmd95_safe_root(salt_a), _jmd95_safe_root(salt_b)
    root_sum = root_a + root_b
    positive = root_sum > 0.0
    safe_sum = torch.where(positive, root_sum, torch.ones_like(root_sum))
    # (s_a**1.5 - s_b**1.5)/(s_a - s_b), including equal and zero salts.
    salt_divided = torch.where(
        positive, (salt_a + root_a * root_b + salt_b) / safe_sum,
        torch.zeros_like(root_sum),
    )
    dt, ds = temp_a - temp_b, salt_a - salt_b
    p = pressure_pa * _SI_TO_BAR
    poly, divided = _jmd95_polynomial, _jmd95_divided_difference
    f, w = _JMDCFW, _JMDCSW
    kf, ks, c = _JMDCKFW, _JMDCKSW, _JMDCKP
    delta_r = dt * (
        divided(f, temp_a, temp_b) + salt_a * divided(w[:5], temp_a, temp_b)
        + salt_a * root_a * divided(w[5:8], temp_a, temp_b)
    )
    delta_r = delta_r + ds * (
        poly(w[:5], temp_b) + salt_divided * poly(w[5:8], temp_b)
        + w[8] * (salt_a + salt_b)
    )
    delta_k = dt * (
        divided(kf, temp_a, temp_b) + salt_a * divided(ks[:4], temp_a, temp_b)
        + salt_a * root_a * divided(ks[4:7], temp_a, temp_b)
    )
    delta_k = delta_k + ds * (
        poly(ks[:4], temp_b) + salt_divided * poly(ks[4:7], temp_b)
    )
    delta_k = delta_k + p * (
        dt * (divided(c[:4], temp_a, temp_b)
              + salt_a * divided(c[4:7], temp_a, temp_b))
        + ds * (poly(c[4:7], temp_b) + salt_divided * c[7])
    )
    delta_k = delta_k + p * p * (
        dt * (divided(c[8:11], temp_a, temp_b)
              + salt_a * divided(c[11:14], temp_a, temp_b))
        + ds * poly(c[11:14], temp_b)
    )
    _, bulk_a = _jmd95_difference_terms(temp_a, salt_a, root_a, p)
    surface_b, bulk_b = _jmd95_difference_terms(temp_b, salt_b, root_b, p)
    difference = delta_r * bulk_a / (bulk_a - p) - (
        surface_b * p * delta_k / ((bulk_a - p) * (bulk_b - p))
    )
    return difference.to(out_dtype) if out_dtype is not torch.float64 else difference


def jmd95_alpha_beta(
    temp: torch.Tensor,
    salt: torch.Tensor,
    pressure_pa: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Thermal/haline expansion coefficients α = dρ/dθ, β = dρ/dS (no 1/ρ factor).

    Analytic JMD95 derivatives, reproducing MITgcm ``FIND_ALPHA`` / ``FIND_BETA``
    (``model/src/find_alpha.F`` JMD95 branch L157-218 / L487-535).  Both are the
    analytic partials of  ρ = rhoP0·K/(K − p1)  (K = secant bulk modulus, p1 =
    pressure[bar]) w.r.t. θ and S.  Used by KPP STATEKPP (``ttalpha``/``ssbeta``)
    and surface buoyancy forcing ``bo``.

    Parameters mirror :func:`jmd95_density_anomaly` (same θ, S, pressure[Pa]).
    Returns ``(alpha, beta)``, each broadcast to temp's shape; units kg/m³/°C and
    kg/m³/PSU.  α < 0 (warming lightens water); β > 0.  Grid-independent: pure
    elementwise, no spatial assumptions.

    Reduced-precision guardrail: like :func:`jmd95_density_anomaly`, evaluate the
    (analytic-derivative) polynomials in float64 when the input is not float64
    and cast the coefficients back; float64 inputs run the original path.
    """
    _out_dtype = temp.dtype
    if _out_dtype is not torch.float64:
        temp = temp.to(torch.float64)
        salt = salt.to(torch.float64)
        if isinstance(pressure_pa, torch.Tensor):
            pressure_pa = pressure_pa.to(torch.float64)
    t1 = temp
    s1 = torch.clamp(salt, min=0.0)            # MITgcm clamps negative salinity
    sqrt_s = _jmd95_safe_root(s1)

    t2 = t1 * t1
    t3 = t2 * t1
    p1 = pressure_pa * _SI_TO_BAR
    p2 = p1 * p1
    rhoP0, bulkMod = _jmd95_surface_density_and_bulk_mod(t1, s1, sqrt_s, p1, t2, t3)
    Kmp = bulkMod - p1
    Kmp2 = Kmp * Kmp

    # FIND_ALPHA uses sqrt(s³) for its temperature derivative. Preserve
    # a finite derivative of zero at the inactive or underflowed branch.
    s3o2_a = _jmd95_safe_root(s1 * s1 * s1)
    drho_fresh = (_JMDCFW[1] + 2. * _JMDCFW[2] * t1 + 3. * _JMDCFW[3] * t2
                  + 4. * _JMDCFW[4] * t3 + 5. * _JMDCFW[5] * t3 * t1)
    drho_salt = (s1 * (_JMDCSW[1] + 2. * _JMDCSW[2] * t1 + 3. * _JMDCSW[3] * t2
                       + 4. * _JMDCSW[4] * t3)
                 + s3o2_a * (_JMDCSW[6] + 2. * _JMDCSW[7] * t1))
    dk_fresh = (_JMDCKFW[1] + 2. * _JMDCKFW[2] * t1 + 3. * _JMDCKFW[3] * t2
                + 4. * _JMDCKFW[4] * t3)
    dk_salt = (s1 * (_JMDCKSW[1] + 2. * _JMDCKSW[2] * t1 + 3. * _JMDCKSW[3] * t2)
               + s3o2_a * (_JMDCKSW[5] + 2. * _JMDCKSW[6] * t1))
    dk_pressure = (p1 * (_JMDCKP[1] + 2. * _JMDCKP[2] * t1 + 3. * _JMDCKP[3] * t2)
                   + p1 * s1 * (_JMDCKP[5] + 2. * _JMDCKP[6] * t1)
                   + p2 * (_JMDCKP[9] + 2. * _JMDCKP[10] * t1)
                   + p2 * s1 * (_JMDCKP[12] + 2. * _JMDCKP[13] * t1))
    drhoP0dtheta = drho_fresh + drho_salt
    dKdtheta = dk_fresh + dk_salt + dk_pressure
    alpha = (bulkMod * bulkMod * drhoP0dtheta - bulkMod * p1 * drhoP0dtheta
             - rhoP0 * p1 * dKdtheta) / Kmp2

    # --- beta = dρ/dS  (FIND_BETA; s3o2 = 1.5·sqrt(s)) ---
    s3o2_b = 1.5 * sqrt_s
    drhoP0dS = (
        _JMDCSW[0] + _JMDCSW[1] * t1 + _JMDCSW[2] * t2 + _JMDCSW[3] * t3
        + _JMDCSW[4] * t3 * t1
        + s3o2_b * (_JMDCSW[5] + _JMDCSW[6] * t1 + _JMDCSW[7] * t2)
        + 2. * _JMDCSW[8] * s1
    )
    dk_salt = (
        (_JMDCKSW[0] + _JMDCKSW[1] * t1 + _JMDCKSW[2] * t2 + _JMDCKSW[3] * t3)
        + s3o2_b * (_JMDCKSW[4] + _JMDCKSW[5] * t1 + _JMDCKSW[6] * t2)
    )
    dk_pressure = (
        p1 * (_JMDCKP[4] + _JMDCKP[5] * t1 + _JMDCKP[6] * t2)
        + s3o2_b * p1 * _JMDCKP[7]
        + p2 * (_JMDCKP[11] + _JMDCKP[12] * t1 + _JMDCKP[13] * t2)
    )
    dKdS = dk_salt + dk_pressure
    beta = (bulkMod * bulkMod * drhoP0dS - bulkMod * p1 * drhoP0dS
            - rhoP0 * p1 * dKdS) / Kmp2

    if _out_dtype is not torch.float64:
        alpha = alpha.to(_out_dtype)
        beta = beta.to(_out_dtype)
    return alpha, beta


def _mdjwf_density_and_reciprocal(t, s, sqrt_s, p, t2, pt):
    """FIND_RHONUM/FIND_RHODEN and their absolute density, in source order."""
    n, d = _MDJWF_NUM, _MDJWF_DEN
    mul = _mdjwf_multiply
    numerator = (
        n[0]
        + mul(t, n[1] + mul(t, n[2] + mul(n[3], t)))
        + mul(s, n[4] + mul(n[5], t) + mul(n[6], s))
        + mul(p, n[7] + mul(n[8], t2) + mul(n[9], s)
              + mul(p, n[10] + mul(n[11], t2)))
    )
    denominator = (
        d[0]
        + mul(t, d[1] + mul(t, d[2] + mul(t, d[3] + mul(t, d[4]))))
        + mul(s, d[5] + mul(t, d[6] + mul(d[7], t2))
              + mul(sqrt_s, d[8] + mul(d[9], t2)))
        + mul(p, d[10] + mul(pt, mul(d[11], t2) + mul(d[12], p)))
    )
    reciprocal = 1.0 / denominator
    return mul(numerator, reciprocal), reciprocal


def mdjwf_density_anomaly(
    temp: torch.Tensor,
    salt: torch.Tensor,
    pressure_pa: torch.Tensor | float = 0.0,
    *,
    reference_density: float = 1025.0,
) -> torch.Tensor:
    """Compute density anomaly ρ − ρ₀ using the MDJWF (McDougall et al. 2003) EOS.

    Reproduces MITgcm ``FIND_RHONUM`` / ``FIND_RHODEN`` in ``find_rho.F``:
    ``ρ = rhoNum / rhoDen`` (MITgcm ``epsln = 0`` → exact ``1/den``, no offset).

    Parameters
    ----------
    temp : Tensor
        Potential temperature [°C].
    salt : Tensor
        Salinity [PSU].  Negative values clamped to 0 (MITgcm zeroes both S and
        √S for S ≤ 0).
    pressure_pa : Tensor or float
        Pressure in Pascals (SI).  Converted to DECIBAR internally (``×1e-4``,
        MITgcm ``SItodBar``) — NOT JMD95's bar.  Same pressure field as the
        JMD95P path: ``rhoConst * (prev_total_phi_hyd + g·|z|)``.
    reference_density : float
        rhoConst [kg/m³].  Subtracted to return the anomaly.

    Returns
    -------
    Tensor : ρ(θ,S,p) − rhoConst  [kg/m³]

    Reduced-precision guardrail: mirrors :func:`jmd95_density_anomaly` — the
    rational ρ = Pnum/Pden is evaluated in float64 when the input is not float64
    and cast back; float64 inputs run the original path unchanged.
    """
    _out_dtype = temp.dtype
    if _out_dtype is not torch.float64:
        temp = temp.to(torch.float64)
        salt = salt.to(torch.float64)
        if isinstance(pressure_pa, torch.Tensor):
            pressure_pa = pressure_pa.to(torch.float64)
    t1 = temp
    s = torch.clamp(salt, min=0.0)  # MITgcm zeroes negative salinity
    # Safe sqrt for autograd: 1/(2√s) is inf at s=0 (land cells); feed sqrt a
    # sentinel 1.0 there and outer-where back to 0.  Forward stays bitwise:
    # s>0 → √s ; s==0 → 0  (mirrors the JMD95 kernel above).
    _s_pos = s > 0
    _s_safe = torch.where(_s_pos, s, torch.ones_like(s))
    sp5 = torch.where(_s_pos, torch.sqrt(_s_safe), torch.zeros_like(s))

    p1 = pressure_pa * _SI_TO_DBAR  # decibar
    t2 = t1 * t1
    p1t1 = p1 * t1

    rho, _ = _mdjwf_density_and_reciprocal(t1, s, sp5, p1, t2, p1t1)
    density_anomaly = rho - reference_density
    if _out_dtype is not torch.float64:
        density_anomaly = density_anomaly.to(_out_dtype)
    return density_anomaly


@dataclass(frozen=True)
@register_dynamics_module("jmd95_equation_of_state")
class JMD95EquationOfState(TendencyModule):
    """Jackett & McDougall (1995) nonlinear equation of state.

    Matches MITgcm ``eosType='JMD95Z'`` by default.  When
    ``dynamic_pressure=True``, matches the ``eosType='JMD95P'`` pressure
    path: inputs include ``prev_total_phi_hyd`` from the PREVIOUS step,
    shape ``[nz, ny, nx]`` or broadcastable to ``[batch, nz, ny, nx]``,
    units ``m2 s-2``, with LOCOMgcm's downward-positive z convention.
    """

    gravity: float = 9.81
    reference_density: float = 1_025.0
    dynamic_pressure: bool = False
    name: str = "jmd95_equation_of_state"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW

    @property
    def eos_type(self) -> str:
        return mit_eos_type(self.name, self.dynamic_pressure)

    @staticmethod
    def _broadcast_previous_total_phi_hyd(
        prev_total_phi_hyd: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prev_total_phi_hyd.ndim == target.ndim - 1:
            prev_total_phi_hyd = prev_total_phi_hyd.unsqueeze(0)
        try:
            torch.broadcast_shapes(tuple(prev_total_phi_hyd.shape), tuple(target.shape))
        except RuntimeError as exc:
            raise ValueError(
                "prev_total_phi_hyd must be broadcastable to temperature shape "
                f"{tuple(target.shape)}, got {tuple(prev_total_phi_hyd.shape)}"
            ) from exc
        return prev_total_phi_hyd

    def _pressure_field(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device,
        target: torch.Tensor,
        prev_total_phi_hyd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute hydrostatic pressure at cell centers [Pa].

        JMD95Z: p(k) = rhoConst * g * |z_center(k)|
        JMD95P: p(k) = rhoConst * (prev_total_phi_hyd(k) + g * |z_center(k)|)
        The dynamic branch uses the stored reference geopotential at the cell centers.
        """
        dz = grid.layer_thickness.to(dtype=dtype, device=device)
        if not self.dynamic_pressure:
            return reference_pressure_from_layers(
                dz, rho_const=self.reference_density, gravity=self.gravity,
            ).view(1, grid.nz, 1, 1)
        if prev_total_phi_hyd is None:
            # The INI_PRESSURE seed's first pass passes an explicit zero field.
            raise ValueError(
                f"{self.name}: dynamic_pressure needs last step's total hydrostatic "
                "potential (EvaluationInputs.total_phi_hyd) and got none"
            )
        pressure_potential = _reference_geopotential(dz, self.gravity).view(1, grid.nz, 1, 1)
        prev_phi = torch.as_tensor(prev_total_phi_hyd, dtype=dtype, device=device)
        prev_phi = self._broadcast_previous_total_phi_hyd(prev_phi, target)
        pressure_potential = pressure_potential + prev_phi
        return self.reference_density * pressure_potential  # [Pa], broadcastable

    def density_anomaly_at_pressure(
        self,
        temp: torch.Tensor,
        salt: torch.Tensor,
        pressure: torch.Tensor,
    ) -> torch.Tensor:
        """Pointwise density anomaly ρ(T,S,p) − rhoConst at the given pressure [Pa].

        MITgcm's ``FIND_RHO_2D`` takes the pressure of a reference level
        ``kRef`` that need not be the level of ``T, S`` (``do_oceanic_phys.F:811-823``
        evaluates level ``k-1`` water at level ``k``); callers pass the pressure
        they need, e.g. a level-shifted row of :meth:`_pressure_field`.
        Subclasses (MDJWF) override the formula; the base is JMD95.  Pure
        elementwise — no grid dependence.
        """
        return jmd95_density_anomaly(
            temp, salt, pressure, reference_density=self.reference_density,
        )

    def density_anomaly_from_fields(
        self,
        temp: torch.Tensor,
        salt: torch.Tensor,
        grid: StructuredCGrid,
        *,
        prev_total_phi_hyd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Grid-independent tensor-level density anomaly.

        Shared by the state-based module path (:meth:`density_anomaly`) and the
        cube hardcoded HPG path (``_cube_momentum.cube_phi_hyd``) so both grids
        select the EOS variant + dynamic-pressure handling through ONE
        config-driven implementation, instead of the cube re-hardcoding JMD95Z.
        Pointwise apart from the vertical ``layer_thickness`` used to build the
        depth pressure; no horizontal / grid-topology assumptions.
        """
        # JMD95P reads the lagged potential only from its caller
        # (``EvaluationInputs.total_phi_hyd`` or an explicit argument).
        pressure = self._pressure_field(
            grid,
            dtype=temp.dtype,
            device=temp.device,
            target=temp,
            prev_total_phi_hyd=prev_total_phi_hyd,
        )
        density_anomaly = self.density_anomaly_at_pressure(temp, salt, pressure)
        dump_tap = _eos_dump_tap()
        if dump_tap.enabled:
            _capture_eos_dump(
                dump_tap, self, temp, salt, pressure, density_anomaly, grid
            )
        return density_anomaly

    def density_anomaly(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        prev_total_phi_hyd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.density_anomaly_from_fields(
            state.temp.data,
            state.salt.data,
            grid,
            prev_total_phi_hyd=prev_total_phi_hyd,
        )

    def density(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
    ) -> torch.Tensor:
        return self.density_anomaly(state, grid) + state.temp.data.new_tensor(self.reference_density)

    def density_fields(
        self,
        temp: torch.Tensor,
        salt: torch.Tensor,
        grid: StructuredCGrid,
        *,
        prev_total_phi_hyd: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(density_anomaly, density)`` at the ``eos_cuda`` consumer boundary.

        Both model reads of this EOS come through here, the lat-lon
        diagnostics (:meth:`populate_diagnostic_state`) and the cube's
        hydrostatic pressure (``_cube_momentum.cube_phi_hyd``), so both grids
        take one dispatch.  The reference is :meth:`density_anomaly_from_fields`.
        """
        from locomgcm.accel import (
            consumer_snapshot,
            in_accelerated_dispatch,
            requested,
        )

        if not in_accelerated_dispatch():
            with consumer_snapshot():
                if requested(("eos_cuda",)):
                    from locomgcm.dynamics._eos_cuda import dispatcher

                    return dispatcher.density_fields(
                        self, temp, salt, grid, prev_total_phi_hyd=prev_total_phi_hyd
                    )
        density_anomaly = self.density_anomaly_from_fields(
            temp, salt, grid, prev_total_phi_hyd=prev_total_phi_hyd
        )
        return density_anomaly, density_anomaly + density_anomaly.new_tensor(
            self.reference_density
        )

    def populate_diagnostic_state(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> None:
        inputs = diagnostics.inputs
        density_anomaly, density = self.density_fields(
            state.temp.data, state.salt.data, grid,
            prev_total_phi_hyd=None if inputs is None else inputs.total_phi_hyd,
        )
        diagnostics.set_field(
            Field(
                "density_anomaly",
                density_anomaly,
                FieldLocation.CENTER,
                "kg m-3",
                mask_ref="center",
                metadata={"reference_density": self.reference_density, "eos_type": self.eos_type},
            )
        )
        diagnostics.set_field(
            Field(
                "density",
                density,
                FieldLocation.CENTER,
                "kg m-3",
                mask_ref="center",
                metadata={"reference_density": self.reference_density, "eos_type": self.eos_type},
            )
        )

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        if "density_anomaly" not in diagnostics.fields:
            self.populate_diagnostic_state(state, diagnostics, grid)
        return StateTendency.zeros_like(state)


@dataclass(frozen=True)
@register_dynamics_module("mdjwf_equation_of_state")
class MDJWFEquationOfState(JMD95EquationOfState):
    """McDougall, Jackett, Wright & Feistel (2003) nonlinear equation of state.

    Matches MITgcm ``eosType='MDJWF'``.  Reuses the JMD95 pressure path —
    ``dynamic_pressure=True`` mirrors the ``JMD95P`` hydrostatic-pressure
    handling Gong runs with — and differs only in the density formula: the
    rational ``ρ = Pnum/Pden`` of (θ, S, p[dbar]) via ``mdjwf_density_anomaly``.
    """

    name: str = "mdjwf_equation_of_state"

    def density_anomaly_at_pressure(
        self,
        temp: torch.Tensor,
        salt: torch.Tensor,
        pressure: torch.Tensor,
    ) -> torch.Tensor:
        return mdjwf_density_anomaly(
            temp, salt, pressure, reference_density=self.reference_density,
        )
