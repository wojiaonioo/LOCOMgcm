"""Global boundary condition registry.

Plugin classes register themselves via @register_boundary_condition(name).
"""
from __future__ import annotations

from locomgcm.boundaries.base import BoundaryCondition
from locomgcm.registries import PluginRegistry

if not hasattr(BoundaryCondition, "capabilities"):
    BoundaryCondition.capabilities = None  # type: ignore[attr-defined]

BOUNDARY_CONDITIONS: PluginRegistry[BoundaryCondition] = PluginRegistry(kind="boundary_condition")
register_boundary_condition = BOUNDARY_CONDITIONS.register
