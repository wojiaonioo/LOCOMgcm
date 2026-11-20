"""Layer D — ConsumerAdapter public re-exports + eager registration."""

from ._registry import CONSUMERS, register_consumer
from ._types import (
    ArtifactKind,
    ArtifactRef,
    ConsumerAdapter,
    ConsumerCapabilities,
    ConsumerContext,
)

from . import consumers  # noqa: F401

__all__ = [
    "ArtifactKind",
    "ArtifactRef",
    "CONSUMERS",
    "ConsumerAdapter",
    "ConsumerCapabilities",
    "ConsumerContext",
    "register_consumer",
]
