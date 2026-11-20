"""Eager-imports all consumers."""

from . import mit_binary  # noqa: F401
from . import zarr_consumers  # noqa: F401

__all__: list[str] = []
