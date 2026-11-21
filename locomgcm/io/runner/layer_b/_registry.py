"""Module-level DATA_SOURCES registry instance.

Plugin classes register themselves via `@register_source(name)` decorator.
Centralized import in `sources/__init__.py` triggers all decorators on
package load.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from locomgcm.registries import PluginRegistry

from ._types import DataSource, SourceCapabilities

DATA_SOURCES: PluginRegistry[DataSource] = PluginRegistry(kind="data_source")


def _validate_source_plugin_class(cls: type[Any]) -> None:
    capabilities = getattr(cls, "capabilities", None)
    if not isinstance(capabilities, SourceCapabilities):
        raise TypeError(
            f"{cls.__name__}.capabilities must be a SourceCapabilities instance"
        )
    fetch = getattr(cls, "fetch", None)
    if not callable(fetch):
        raise TypeError(f"{cls.__name__} must define callable fetch(self, query)")
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{cls.__name__} must expose an inspectable __init__") from exc
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if not params:
        raise TypeError(f"{cls.__name__}.__init__ must accept SourceContext")
    first = params[0]
    if first.name not in {"ctx", "context"} or first.kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        raise TypeError(
            f"{cls.__name__}.__init__ first argument must be ctx/context"
        )


def register_source(name: str) -> Callable[[type[DataSource]], type[DataSource]]:
    """Decorator for Layer-B source plugins with runtime contract checks."""

    def deco(cls: type[DataSource]) -> type[DataSource]:
        _validate_source_plugin_class(cls)
        return DATA_SOURCES.register(name)(cls)

    return deco
