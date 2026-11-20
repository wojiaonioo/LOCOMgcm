"""Module-level OPERATORS + MERGERS registries.

Plugin classes register via `@register_operator(name)` / `@register_merger(name)`.
"""

from __future__ import annotations

from defgcm.registries import PluginRegistry

from ._types import Merger, Operator

OPERATORS: PluginRegistry[Operator] = PluginRegistry(kind="operator")
MERGERS: PluginRegistry[Merger] = PluginRegistry(kind="merger")

register_operator = OPERATORS.register
register_merger = MERGERS.register
