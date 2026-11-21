"""Dual-purpose registries module.

Provides two distinct registry abstractions:

1. CapabilityLedger[T] — records what a model instance is configured with
   (introspection ledger). Stores arbitrary items (strings, instances).
   Read-only after model build. NOT a plugin dispatcher.

   Used by `locomgcm/model.py` to declare configured operators / steppers /
   boundaries / closures / solvers for capability introspection.

2. PluginRegistry[T] — plugin dispatch with capabilities + lazy instantiation.
   Stores plugin CLASSES with a mandatory `capabilities` class attribute.
   Used by Layer B (DataSource), Layer C (Operator/Merger), Layer D
   (ConsumerAdapter) plugin systems.

Backwards compatibility: legacy class names (NamedRegistry, OperatorRegistry,
StepperRegistry, BoundaryRegistry, ClosureRegistry, SolverRegistry,
RegistryBundle) still exist as deprecated aliases that emit DeprecationWarning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


# ──────────────────────────────────────────────────────────────────────────
# Part 1 — Capability Ledger (was NamedRegistry; ledger semantics)
# ──────────────────────────────────────────────────────────────────────────


class CapabilityLedger(Generic[T]):
    """Records what a model instance is configured with.

    Read-only ledger of (name, item) pairs. Used by `model.py` to declare
    the configured operators/steppers/etc. for introspection. NOT a plugin
    dispatcher — items registered are typically strings or already-built
    instances, never plugin classes.

    For plugin dispatch (Layer B/C/D), use `PluginRegistry` below.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        """Records `item` under `name`. Raises ValueError on duplicate name."""
        if name in self._items:
            raise ValueError(f"{self.kind} ledger already contains {name!r}")
        self._items[name] = item

    def get(self, name: str) -> T:
        """Returns the item registered under `name`. Raises KeyError if absent."""
        return self._items[name]

    def items(self) -> dict[str, T]:
        """Returns a copy of all (name, item) pairs (mutation isolated from internal)."""
        return dict(self._items)


# Per-domain ledger subclasses (preserve existing behavior + names that the
# legacy aliases below need).


class OperatorLedger(CapabilityLedger[object]):
    def __init__(self) -> None:
        super().__init__("operator")


class StepperLedger(CapabilityLedger[object]):
    def __init__(self) -> None:
        super().__init__("stepper")


class BoundaryLedger(CapabilityLedger[object]):
    def __init__(self) -> None:
        super().__init__("boundary")


class ClosureLedger(CapabilityLedger[object]):
    def __init__(self) -> None:
        super().__init__("closure")


class SolverLedger(CapabilityLedger[object]):
    def __init__(self) -> None:
        super().__init__("solver")


@dataclass
class LedgerBundle:
    """Container of all per-domain ledgers for a model instance."""

    operators: OperatorLedger = field(default_factory=OperatorLedger)
    steppers: StepperLedger = field(default_factory=StepperLedger)
    boundaries: BoundaryLedger = field(default_factory=BoundaryLedger)
    closures: ClosureLedger = field(default_factory=ClosureLedger)
    solvers: SolverLedger = field(default_factory=SolverLedger)


# ──────────────────────────────────────────────────────────────────────────
# Part 2 — Plugin Registry (NEW; for Layer B/C/D plugin dispatch)
# ──────────────────────────────────────────────────────────────────────────


class PluginRegistry(Generic[T]):
    """Plugin dispatcher with capabilities + lazy instantiation.

    Distinct from CapabilityLedger: stores plugin CLASSES (not instances),
    supports decorator registration, queries by capabilities, and lazy
    instantiation with dependency-injection context.

    Usage:
        DATA_SOURCES = PluginRegistry[DataSource](kind="data_source")

        @DATA_SOURCES.register("hycom_netcdf")
        class HycomNetcdfSource:
            capabilities = SourceCapabilities(...)
            def __init__(self, ctx, **cfg): ...

        # later:
        cls = DATA_SOURCES.get("hycom_netcdf")
        instance = DATA_SOURCES.instantiate("hycom_netcdf", ctx, path="external:hycom_dir")
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator: register a plugin class under `name`.

        The class MUST have a `capabilities` class attribute, validated at
        registration time. Duplicate names raise ValueError.
        """

        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(
                    f"{self.kind} registry already contains {name!r}"
                )
            if not hasattr(cls, "capabilities"):
                raise TypeError(
                    f"{cls.__name__} missing `capabilities` class attribute; "
                    f"PluginRegistry requires capabilities-bearing classes"
                )
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type[T]:
        """Look up class by name. Raises KeyError with known names listed."""
        if name not in self._items:
            known = sorted(self._items)
            raise KeyError(
                f"unknown {self.kind} plugin {name!r}; known: {known}"
            )
        return self._items[name]

    def instantiate(self, name: str, ctx: Any = None, **config: Any) -> T:
        """Instantiate a registered class with context + config.

        Two calling conventions supported (auto-detected from __init__ signature):
        - Layer B Sources / Layer D Consumers: __init__(self, ctx, **config)
          → ctx passed as first positional argument
        - Layer C Operators / Mergers: __init__(self, **config)
          → ctx ignored; only **config forwarded
        """
        import inspect

        cls = self.get(name)
        try:
            sig = inspect.signature(cls.__init__)
            params = [p for p in sig.parameters.values() if p.name != "self"]
        except (ValueError, TypeError):
            params = []

        if params and params[0].name in {"ctx", "context"} and params[0].kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }:
            return cls(ctx, **config)
        return cls(**config)

    def list_names(self) -> tuple[str, ...]:
        """Sorted tuple of all registered names."""
        return tuple(sorted(self._items))

    def list_compatible(self, query: Callable[[Any], bool]) -> tuple[str, ...]:
        """Sorted tuple of plugin names whose `capabilities` satisfy `query`.

        Example:
            DATA_SOURCES.list_compatible(
                lambda c: LocationKind.GRIDDED_3D_OCEAN in c.produces
            )
        """
        return tuple(
            sorted(
                name
                for name, cls in self._items.items()
                if query(cls.capabilities)
            )
        )

    def items(self) -> dict[str, type[T]]:
        """Copy of (name, class) pairs."""
        return dict(self._items)


# ──────────────────────────────────────────────────────────────────────────
# Part 3 — Deprecated aliases (warn but functional)
# ──────────────────────────────────────────────────────────────────────────


class _DeprecatedAlias:
    """Callable alias that emits DeprecationWarning and forwards to new class."""

    def __init__(self, new_cls: type, old_name: str, new_name: str) -> None:
        self._new_cls = new_cls
        self._old = old_name
        self._new_name = new_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{self._old} is deprecated; use {self._new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._new_cls(*args, **kwargs)


NamedRegistry = _DeprecatedAlias(CapabilityLedger, "NamedRegistry", "CapabilityLedger")
OperatorRegistry = _DeprecatedAlias(OperatorLedger, "OperatorRegistry", "OperatorLedger")
StepperRegistry = _DeprecatedAlias(StepperLedger, "StepperRegistry", "StepperLedger")
BoundaryRegistry = _DeprecatedAlias(BoundaryLedger, "BoundaryRegistry", "BoundaryLedger")
ClosureRegistry = _DeprecatedAlias(ClosureLedger, "ClosureRegistry", "ClosureLedger")
SolverRegistry = _DeprecatedAlias(SolverLedger, "SolverRegistry", "SolverLedger")
RegistryBundle = _DeprecatedAlias(LedgerBundle, "RegistryBundle", "LedgerBundle")
