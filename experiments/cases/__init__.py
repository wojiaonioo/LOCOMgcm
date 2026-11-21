"""Case runtime for the LOCOMgcm pipeline.

``build_model`` is the public entry for scripts that drive the model directly;
it is imported lazily so that importing this package stays cheap.
"""

from __future__ import annotations

__all__ = ["build_model"]


def __getattr__(name: str):
    if name == "build_model":
        from ._runtime import build_model

        return build_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
