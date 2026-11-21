"""Load requested CUDA extensions before entering compiled model code."""

from __future__ import annotations

import importlib

import torch

from .registry import REGISTRY
from .runtime import requested


_PRELOADED_KEYS: set[str] = set()


def _is_cuda_device(device: str) -> bool:
    return device == "cuda"


def preload_requested_extensions(*, device: str) -> list[str]:
    """Preload enabled CUDA extensions, returning keys loaded by this call."""
    preloadable = tuple(key for key, spec in REGISTRY.items() if spec.preload)
    requested_keys = frozenset(requested(preloadable))
    loaded: list[str] = []

    for key in preloadable:
        spec = REGISTRY[key]
        if key not in requested_keys:
            print(
                f"LOCOMGCM_ACCEL_PRELOAD key={key} status=skipped reason=not_requested",
                flush=True,
            )
            continue
        if not _is_cuda_device(device):
            print(
                f"LOCOMGCM_ACCEL_PRELOAD key={key} status=skipped reason=non_cuda_device",
                flush=True,
            )
            continue
        if not torch.cuda.is_available():
            print(
                f"LOCOMGCM_ACCEL_PRELOAD key={key} status=skipped reason=cuda_unavailable",
                flush=True,
            )
            continue
        if key in _PRELOADED_KEYS:
            print(
                f"LOCOMGCM_ACCEL_PRELOAD key={key} status=skipped reason=already_loaded",
                flush=True,
            )
            continue

        module_name, attribute = spec.preload.rsplit(":", 1)
        loader = getattr(importlib.import_module(module_name), attribute)
        loader()
        _PRELOADED_KEYS.add(key)
        loaded.append(key)
        print(
            f"LOCOMGCM_ACCEL_PRELOAD key={key} status=loaded reason=requested_cuda",
            flush=True,
        )

    return loaded


__all__ = ["preload_requested_extensions"]
