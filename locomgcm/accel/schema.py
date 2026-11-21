"""Typed declarations for LOCOMgcm accelerator gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GateClass = Literal["capability", "correctness"]
UnmetPolicy = Literal["fallback", "raise"]
# seaice_native: the four native MIT tiles of the offline sea-ice programs.
GridFamily = Literal["latlon", "cube", "seaice_native"]
DistributedMode = Literal["single", "multi"]
HydrostaticMode = Literal["any", "nonhydrostatic_only"]
SettingKind = Literal["boolean", "backend"]
DefaultSource = Literal["library", "driver", "user"]
IgnitionState = Literal[
    "disabled",
    "requested-hit",
    "requested-zero",
    "inapplicable-or-error",
]


@dataclass(frozen=True)
class AcceleratorSpec:
    """Static accelerator declaration.

    ``gate_class`` and ``unmet_policy`` are deliberately independent.  The
    only implication is that a correctness gate may never silently fall back.
    """

    key: str
    env_var: str
    default: bool
    gate_class: GateClass
    unmet_policy: UnmetPolicy
    grid_families: tuple[GridFamily, ...]
    distributed: tuple[DistributedMode, ...]
    hydrostatic: HydrostaticMode
    gate_satisfiable: bool
    production_reachable: bool
    setting_kind: SettingKind = "boolean"

    def __post_init__(self) -> None:
        if not self.key or not self.env_var:
            raise ValueError("accelerator key and environment variable must be non-empty")
        if self.gate_class not in ("capability", "correctness"):
            raise ValueError(f"invalid gate_class {self.gate_class!r}")
        if self.unmet_policy not in ("fallback", "raise"):
            raise ValueError(f"invalid unmet_policy {self.unmet_policy!r}")
        if self.gate_class == "correctness" and self.unmet_policy != "raise":
            raise ValueError("correctness gates must declare unmet_policy='raise'")
        if not self.grid_families or any(
            family not in ("latlon", "cube", "seaice_native") for family in self.grid_families
        ):
            raise ValueError(f"invalid grid_families {self.grid_families!r}")
        if not self.distributed or any(
            mode not in ("single", "multi") for mode in self.distributed
        ):
            raise ValueError(f"invalid distributed modes {self.distributed!r}")
        if self.hydrostatic not in ("any", "nonhydrostatic_only"):
            raise ValueError(f"invalid hydrostatic declaration {self.hydrostatic!r}")
        if self.setting_kind not in ("boolean", "backend"):
            raise ValueError(f"invalid setting_kind {self.setting_kind!r}")


@dataclass(frozen=True)
class GateView:
    """Resolved provenance plus the independently observed ignition state."""

    raw_spelling: str | None
    effective_value: bool
    default_source: DefaultSource
    ignition_state: IgnitionState
    selected: str | None = None
