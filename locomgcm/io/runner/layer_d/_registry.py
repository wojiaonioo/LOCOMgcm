"""Module-level CONSUMERS registry."""

from __future__ import annotations

from locomgcm.registries import PluginRegistry

from ._types import ConsumerAdapter

CONSUMERS: PluginRegistry[ConsumerAdapter] = PluginRegistry(kind="consumer")
register_consumer = CONSUMERS.register
