"""Global stepper registry.

Plugin classes register themselves via @register_stepper(name) decorator.
"""
from __future__ import annotations

from defgcm.registries import PluginRegistry
from defgcm.steppers.base import TimeStepper

if not hasattr(TimeStepper, "capabilities"):
    TimeStepper.capabilities = None  # type: ignore[attr-defined]

STEPPERS: PluginRegistry[TimeStepper] = PluginRegistry(kind="stepper")
register_stepper = STEPPERS.register
