"""Module-level CONSUMERS registry."""

from __future__ import annotations

from defgcm.registries import PluginRegistry

from ._types import ConsumerAdapter

CONSUMERS: PluginRegistry[ConsumerAdapter] = PluginRegistry(kind="consumer")
register_consumer = CONSUMERS.register
