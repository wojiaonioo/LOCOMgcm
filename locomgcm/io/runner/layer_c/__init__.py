"""Layer C — Operators + Mergers."""

from ._registry import MERGERS, OPERATORS, register_merger, register_operator
from ._types import Merger, MergerCapabilities, Operator, OperatorCapabilities

from . import mergers  # noqa: F401
from . import operators  # noqa: F401

__all__ = [
    "MERGERS",
    "Merger",
    "MergerCapabilities",
    "OPERATORS",
    "Operator",
    "OperatorCapabilities",
    "register_merger",
    "register_operator",
]
