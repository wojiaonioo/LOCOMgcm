"""Eager-owned cache for constants reused by compiled model code."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

import torch

from .tensor_context import ordinary_tensor_mode
from .graph_exit import graph_exit


@torch._dynamo.disable
@ordinary_tensor_mode()
def _build_eager(build: Callable[[], Any]) -> Any:
    return build()


def _constant_build(build: Callable[[], Any]) -> Any:
    from locomgcm.distributed.comm_adjoint import constant_transport

    with constant_transport():
        return build()


class ConstantCache:
    """Own lazy constants outside tracing so CUDAGraph generation rollover cannot poison them.

    The disabled cold builder introduces one graph break only on a miss, while a hit is a plain
    dictionary lookup and remains traceable. Cold construction stays outside
    tracing so constants do not become CUDAGraph outputs.

    Entries are namespaced by ``variant`` (the store key is ``(variant, key)``): several sites
    share the module-level ``constant_cache`` instance; variant names prevent
    collisions between their site-local key tuples.

    The memo lives on the object the constant derives from (``owner``: a grid,
    a module, a KPP static block), never in a process-global table keyed by
    object identity.  Keys hold only values (dtype, device, shape, options), so
    every source a compiled step guards on, ``<owner path>._constant_memo[key]``,
    can be evaluated on any model of the same configuration, and the entries end
    with their owner.  An instance used without ``owner`` (a ``MaskSet`` or a
    module that owns its own cache) keeps its memo on the instance.
    """

    def __init__(self) -> None:
        self._store: dict[Any, Any] = {}
        self._owners: weakref.WeakSet = weakref.WeakSet()

    def _memo(self, owner: Any) -> dict[Any, Any]:
        if owner is None:
            if self is constant_cache:
                raise TypeError("the shared constant cache needs the constant's owner")
            return self._store
        memo = getattr(owner, "_constant_memo", None)
        if memo is None:
            memo = {}
            object.__setattr__(owner, "_constant_memo", memo)
            try:
                self._owners.add(owner)
            except TypeError:
                pass
        return memo

    def get_or_build(
        self,
        key: Any,
        build: Callable[[], Any],
        *,
        variant: str,
        site: str,
        owner: Any = None,
        selector: str = "cache_constants",
        tensor_only: bool = False,
    ) -> Any:
        # Resolved through the package at call time (not bound at import) so the
        # established test seam ``monkeypatch.setattr("locomgcm.accel.record_cache_event", …)``
        # remains available to callers.
        from locomgcm import accel as _accel

        # A constant's exchange is issued once and memoized; it stays outside
        # any bound differentiation window in both the cached and uncached path.
        if not _accel.gate("cache_constants").effective_value:
            if selector != "cache_constants" and _accel.gate(selector).effective_value:
                _accel.record_dispatch(
                    selector,
                    outcome="inapplicable",
                    variant=variant,
                    site=site,
                    backend="torch",
                    rank=_accel.current_rank(),
                    reason="cache_constants_disabled",
                )
            # Conversions and tensor factories need no host lifetime when
            # their result is not retained. Keep the uncached operations in
            # the caller's graph; callers opt in only for builders without
            # host work, communication or persistent outputs.
            if tensor_only:
                return build()
            # The transport override is host state even when memoization is
            # disabled. Keep its entire lifetime outside Dynamo, just as for
            # a cache miss; a builder may itself contain an eager-only call.
            with graph_exit():
                return _build_eager(lambda: _constant_build(build))

        store = self._memo(owner)
        store_key = (variant, key)
        if store_key in store:
            value = store[store_key]
            event = "warm-reuse"
        else:
            value = _build_eager(lambda: _constant_build(build))
            store[store_key] = value
            event = "cold-construction"

        # Ownership is the caller's resource selector, not the storage helper.
        _accel.record_cache_event(
            selector,
            event=event,
            variant=variant,
            site=site,
            backend="torch",
            rank=_accel.current_rank(),
        )
        return value

    def __len__(self) -> int:
        return len(self._store) + sum(len(getattr(owner, "_constant_memo", ()))
                                      for owner in list(self._owners))

    def clear(self) -> None:
        """Drop every entry (tests reset the memo between cases; production never calls this)."""
        self._store.clear()
        for owner in list(self._owners):
            memo = getattr(owner, "_constant_memo", None)
            if memo is not None:
                memo.clear()


constant_cache = ConstantCache()
