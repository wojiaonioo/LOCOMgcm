"""Global stepper registry.

Plugin classes register themselves via @register_stepper(name) decorator.
"""
from __future__ import annotations

from locomgcm.registries import PluginRegistry
from locomgcm.steppers.base import TimeStepper

if not hasattr(TimeStepper, "capabilities"):
    TimeStepper.capabilities = None  # type: ignore[attr-defined]

STEPPERS: PluginRegistry[TimeStepper] = PluginRegistry(kind="stepper")
register_stepper = STEPPERS.register
