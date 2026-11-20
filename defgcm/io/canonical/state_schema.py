"""Canonical model state schema.

This defines the Single Source of Truth (SSOT) for the IO pipeline:
- The canonical state builder is the ONLY producer of CanonicalModelInput
- Format adapters (LOCOMgcm zarr / MIT bin) consume it pure-reformat (0 numerics)
- Runners apply the base-time boundaries once, after the INI_PRESSURE seed
- Diagnostics compare LOCOMgcm/MIT runtime outputs against it

INVARIANT-1: All tensors share batch dim, same device/dtype
INVARIANT-2: convention == TORCH_OCEAN (immutable)
INVARIANT-3: state is the run's initial field, the one the reference model's INI_THETA,
             INI_SALT, INI_VEL and INI_PSURF read (ini_fields.F:29-32); the base-time
             boundary application belongs to defgcm.runner.initialize_fresh_step_state,
             after the INI_PRESSURE seed (ini_fields.F:33, obcs_init_variables.F:401)
INVARIANT-4: hFac mask applied to tracers (tracer * cmask)
INVARIANT-5: u/v are at FACE locations (not center-padded)
INVARIANT-6: bathymetry >= 0, land cells == 0
INVARIANT-7: grid built from same bathymetry baked-in to state
INVARIANT-8: provenance non-empty

validate_canonical defaults to strict=False (warnings only).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from defgcm.geometry import StructuredCGrid


class StagedLocation(Enum):
    """C-grid staggered location."""

    CENTER_3D = "center_3d"
    CENTER_2D = "center_2d"
    U_FACE_3D = "u_face_3d"
    V_FACE_3D = "v_face_3d"
    W_INTERFACE_3D = "w_interface_3d"


class SignConvention(Enum):
    """Sign convention for vertical velocity, bathymetry, surface fluxes."""

    TORCH_OCEAN = "torchocean"


@dataclass(frozen=True)
class StagedField:
    name: str
    data: torch.Tensor
    location: StagedLocation
    units: str
    cf_standard_name: str


@dataclass(frozen=True)
class L3ProvenanceStep:
    builder_id: str
    timestamp: datetime
    bundle_hash: str
    params: dict


@dataclass(frozen=True)
class CanonicalModelInput:
    """SSOT for the IO pipeline. See module docstring for invariants."""

    eta: StagedField
    u: StagedField
    v: StagedField
    temp: StagedField
    salt: StagedField
    grid: "StructuredCGrid"
    obc_source: Any | None
    forcing: Any | None
    run_params: Any
    base_time: datetime
    convention: SignConvention = SignConvention.TORCH_OCEAN
    provenance: tuple[L3ProvenanceStep, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_canonical(self, strict=False)


def validate_canonical(c: CanonicalModelInput, *, strict: bool = False) -> list[str]:
    """Check INVARIANT-1..8. Returns list of violation messages.

    Args:
        strict: If True, raises ValueError on first violation set.
                Default False (warnings only) for gradual integration.

    Returns:
        List of violation messages (empty when valid).
    """
    violations: list[str] = []
    fields = [c.eta, c.u, c.v, c.temp, c.salt]

    # INVARIANT-1: shared batch / device / dtype
    if len({f.data.shape[0] for f in fields}) > 1:
        violations.append("INVARIANT-1: tensors have different batch sizes")
    if len({str(f.data.device) for f in fields}) > 1:
        violations.append("INVARIANT-1: tensors on different devices")
    if len({f.data.dtype for f in fields}) > 1:
        violations.append("INVARIANT-1: tensors have different dtypes")

    # INVARIANT-2: convention immutable
    if c.convention is not SignConvention.TORCH_OCEAN:
        violations.append(f"INVARIANT-2: convention != TORCH_OCEAN ({c.convention})")

    # INVARIANT-5: u/v at FACE locations
    if c.u.location is not StagedLocation.U_FACE_3D:
        violations.append(f"INVARIANT-5: u not U_FACE_3D ({c.u.location})")
    if c.v.location is not StagedLocation.V_FACE_3D:
        violations.append(f"INVARIANT-5: v not V_FACE_3D ({c.v.location})")
    if c.eta.location is not StagedLocation.CENTER_2D:
        violations.append(f"INVARIANT-5: eta not CENTER_2D ({c.eta.location})")
    if c.temp.location is not StagedLocation.CENTER_3D:
        violations.append(f"INVARIANT-5: temp not CENTER_3D ({c.temp.location})")
    if c.salt.location is not StagedLocation.CENTER_3D:
        violations.append(f"INVARIANT-5: salt not CENTER_3D ({c.salt.location})")

    # INVARIANT-5 shape consistency (B, nz, ny, nx)
    b, nz_p1 = c.u.data.shape[0], c.u.data.shape[1]
    ny, nx_plus_1 = c.u.data.shape[2], c.u.data.shape[3]
    nx = nx_plus_1 - 1
    if c.v.data.shape != (b, nz_p1, ny + 1, nx):
        violations.append(
            f"INVARIANT-5: v shape {c.v.data.shape} inconsistent with u {c.u.data.shape}"
        )
    if c.temp.data.shape != (b, nz_p1, ny, nx):
        violations.append(
            f"INVARIANT-5: temp shape {c.temp.data.shape} inconsistent with grid"
        )

    # INVARIANT-8: provenance non-empty
    if len(c.provenance) == 0:
        violations.append("INVARIANT-8: provenance is empty")

    if violations:
        msg = "CanonicalModelInput validation failed:\n  " + "\n  ".join(violations)
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

    return violations
