"""Global dynamics module registry.

Plugin classes register themselves via @register_dynamics_module(name)
decorator. Centralized eager import in dynamics/__init__.py triggers all
decorators on package load.
"""

from __future__ import annotations

from defgcm.dynamics.base import TendencyModule
from defgcm.registries import PluginRegistry

if not hasattr(TendencyModule, "capabilities"):
    TendencyModule.capabilities = None  # type: ignore[attr-defined]

DYNAMICS_MODULES: PluginRegistry[TendencyModule] = PluginRegistry(kind="dynamics_module")
register_dynamics_module = DYNAMICS_MODULES.register
