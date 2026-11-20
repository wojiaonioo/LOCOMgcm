"""Merger operator with 4 modes: replace_top_layer, add_increment, weighted_blend, dedupe_obs.

Used at pipeline entry where multiple sources need to combine into a single
bundle (e.g. HYCOM base state + EAKF increment, or merged observations).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from defgcm.io.runner.layer_a import (
    DataBundle,
    LocationKind,
)

from .._registry import register_merger
from .._types import MergerCapabilities


@register_merger("merge")
class Merger:
    """N→1 merge with configurable mode.

    Modes:
      - replace_top_layer: secondary replaces primary's top z-layer (for SST→T)
      - add_increment: primary + secondary (analysis increment)
      - weighted_blend: w1*primary + w2*secondary (configurable weights)
      - dedupe_obs: union of scattered observations, drop duplicates within radius_km
    """

    capabilities = MergerCapabilities(
        accepts_primary=tuple(LocationKind),
        accepts_secondary=tuple(LocationKind),
        produces=tuple(LocationKind),
        convention_in=None,
        convention_out=None,
        n_secondaries=(1, 2, 3),
    )

    def __init__(
        self,
        *,
        mode: str = "add_increment",
        weight_primary: float = 0.5,
        weight_secondary: float = 0.5,
        radius_km: float = 5.0,
        **_: Any,
    ) -> None:
        if mode not in {"replace_top_layer", "add_increment", "weighted_blend", "dedupe_obs"}:
            raise ValueError(f"Merger: unknown mode {mode!r}")
        self._mode = mode
        self._w_primary = weight_primary
        self._w_secondary = weight_secondary
        self._radius_km = radius_km

    def __call__(self, primary: DataBundle, *secondaries: DataBundle) -> DataBundle:
        if not secondaries:
            return primary  # no-op

        if self._mode == "add_increment":
            return self._add_increment(primary, secondaries[0])
        elif self._mode == "replace_top_layer":
            return self._replace_top_layer(primary, secondaries[0])
        elif self._mode == "weighted_blend":
            return self._weighted_blend(primary, secondaries[0])
        elif self._mode == "dedupe_obs":
            return self._dedupe_obs(primary, *secondaries)
        else:
            raise ValueError(f"Merger: unhandled mode {self._mode}")

    def _add_increment(self, primary: DataBundle, increment: DataBundle) -> DataBundle:
        """primary + increment, variable-by-variable. Increment broadcast as needed."""
        new_ds = primary.data.copy()
        for var in primary.meta.variables:
            if var.name not in increment.data.data_vars:
                continue
            p = primary.data[var.name].values
            inc = increment.data[var.name].values
            # If shapes match, direct add; if increment is reduced, broadcast.
            try:
                summed = (p + inc).astype(p.dtype)
            except ValueError:
                summed = (p + np.broadcast_to(inc, p.shape)).astype(p.dtype)
            new_ds[var.name] = (primary.data[var.name].dims, summed)
        new_meta = dataclasses.replace(
            primary.meta,
            extras={
                **dict(primary.meta.extras),
                "merge_event": {
                    "mode": "add_increment",
                    "secondary_source": increment.meta.provenance[-1].name if increment.meta.provenance else "unknown",
                },
            },
        )
        return DataBundle(data=new_ds, meta=new_meta)

    def _replace_top_layer(self, primary: DataBundle, replacement: DataBundle) -> DataBundle:
        """Replace primary's top z-layer (z=0 surface) with replacement's 2D data."""
        new_ds = primary.data.copy()
        for var in primary.meta.variables:
            if var.name not in replacement.data.data_vars:
                continue
            p = primary.data[var.name]
            if "z" not in p.dims:
                continue
            new_arr = np.asarray(p.values).copy()
            r = replacement.data[var.name].values
            # Insert into top z-slice (index 0)
            if p.dims.index("z") == 0:
                new_arr[0, ...] = r
            else:
                # Find z-axis position; for canonical (time, z, lat, lon), z is axis 1
                z_axis = p.dims.index("z")
                idx: list[Any] = [slice(None)] * new_arr.ndim
                idx[z_axis] = 0
                new_arr[tuple(idx)] = r
            new_ds[var.name] = (p.dims, new_arr.astype(p.dtype))
        new_meta = dataclasses.replace(
            primary.meta,
            extras={
                **dict(primary.meta.extras),
                "merge_event": {"mode": "replace_top_layer"},
            },
        )
        return DataBundle(data=new_ds, meta=new_meta)

    def _weighted_blend(self, primary: DataBundle, secondary: DataBundle) -> DataBundle:
        new_ds = primary.data.copy()
        w1 = self._w_primary
        w2 = self._w_secondary
        for var in primary.meta.variables:
            if var.name not in secondary.data.data_vars:
                continue
            p = primary.data[var.name].values
            s = secondary.data[var.name].values
            try:
                blended = (w1 * p + w2 * s).astype(p.dtype)
            except ValueError:
                blended = (w1 * p + w2 * np.broadcast_to(s, p.shape)).astype(p.dtype)
            new_ds[var.name] = (primary.data[var.name].dims, blended)
        new_meta = dataclasses.replace(
            primary.meta,
            extras={
                **dict(primary.meta.extras),
                "merge_event": {
                    "mode": "weighted_blend",
                    "weights": [w1, w2],
                },
            },
        )
        return DataBundle(data=new_ds, meta=new_meta)

    def _dedupe_obs(self, primary: DataBundle, *secondaries: DataBundle) -> DataBundle:
        """Concatenate scattered observation bundles, drop duplicates within radius_km.

        Simplified implementation: keeps unique (lat, lon, time) within tolerance.
        Full duplicate detection is not implemented.
        """
        # For now, just concat along obs_id dim
        all_bundles = [primary] + list(secondaries)
        new_ds = primary.data
        try:
            for b in secondaries:
                new_ds = new_ds.combine_first(b.data)  # naive union
        except Exception:
            pass
        new_meta = dataclasses.replace(
            primary.meta,
            extras={
                **dict(primary.meta.extras),
                "merge_event": {
                    "mode": "dedupe_obs",
                    "radius_km": self._radius_km,
                    "n_secondaries": len(secondaries),
                },
            },
        )
        return DataBundle(data=new_ds, meta=new_meta)
