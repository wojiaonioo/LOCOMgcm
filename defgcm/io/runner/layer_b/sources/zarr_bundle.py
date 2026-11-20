"""Zarr-bundle source — round-trip read of a previously-written DataBundle zarr.

Used by stage1b/1c to re-read stage1a outputs uniformly through the Layer B
abstraction. Preserves the stored bundle's location_kind and convention.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from defgcm.external_paths import resolve_input_path
from typing import Any

import xarray as xr

from defgcm.io.runner.layer_a import (
    BundleMeta,
    ConventionTag,
    DataBundle,
    LocationKind,
    ProvenanceStep,
    VariableSpec,
)

from .._registry import register_source
from .._types import (
    DataQuery,
    SourceCapabilities,
    SourceContext,
    TimeCoverage,
)


# zarr stores bundle.meta as an attrs JSON blob keyed 'bundle_meta'.
_META_ATTR_KEY = "bundle_meta"


@register_source("zarr_bundle")
class ZarrBundleSource:
    """Reads a zarr previously written by a Layer D consumer.

    Capabilities declare ALL location_kinds as 'produces' because the actual
    kind depends on the stored bundle's meta. Caller's DataQuery.location_kind
    must match the stored bundle's kind, else raise.
    """

    # Class-level fallback; __init__ overrides with stored kind read from zarr meta.
    capabilities = SourceCapabilities(
        produces=tuple(LocationKind),
        variables=(),  # variable list discovered at fetch time
        time_coverage_strategy=TimeCoverage.INSTANT,
        convention=ConventionTag.OPAQUE,  # preserves stored convention
        requires_target_grid=False,
    )

    def __init__(self, ctx: SourceContext, *, path: Path | str, **_unused: Any) -> None:
        self._ctx = ctx
        self._path = resolve_input_path(path)
        # Peek zarr attrs to discover stored location_kind so framework
        # inference picks correctly. Cheap — only reads attrs, not array data.
        try:
            ds_peek = xr.open_zarr(self._path, consolidated=False)
            raw_meta = ds_peek.attrs.get(_META_ATTR_KEY)
            if raw_meta is not None:
                meta_dict = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
                stored_kind = LocationKind(meta_dict["location_kind"])
                stored_conv = ConventionTag(meta_dict["convention"])
                self.capabilities = SourceCapabilities(
                    produces=(stored_kind,),
                    variables=tuple(v["name"] for v in meta_dict.get("variables", [])),
                    time_coverage_strategy=TimeCoverage.INSTANT,
                    convention=stored_conv,
                    requires_target_grid=False,
                )
            ds_peek.close()
        except Exception:
            # Fall back to class-level (all kinds); fetch() will raise if mismatch
            pass

    def fetch(self, query: DataQuery) -> DataBundle:
        ds = xr.open_zarr(self._path, consolidated=False)

        # Reconstruct meta from zarr attrs (stored as JSON blob)
        raw_meta = ds.attrs.get(_META_ATTR_KEY)
        if raw_meta is None:
            raise ValueError(
                f"ZarrBundleSource: zarr at {self._path} missing '{_META_ATTR_KEY}' attr; "
                f"was it written by a Layer D consumer?"
            )
        if isinstance(raw_meta, str):
            meta_dict = json.loads(raw_meta)
        else:
            meta_dict = raw_meta

        stored_kind = LocationKind(meta_dict["location_kind"])
        if query.location_kind != stored_kind:
            raise ValueError(
                f"ZarrBundleSource: zarr stored location_kind={stored_kind.value}; "
                f"query asked for {query.location_kind.value}"
            )

        stored_convention = ConventionTag(meta_dict["convention"])

        var_specs = tuple(
            VariableSpec(
                name=v["name"],
                cf_standard_name=v["cf_standard_name"],
                units=v["units"],
                dtype=v["dtype"],
                dims=tuple(v["dims"]),
            )
            for v in meta_dict["variables"]
        )

        time_range = None
        if meta_dict.get("time_range") is not None:
            tr = meta_dict["time_range"]
            time_range = (
                datetime.fromisoformat(tr[0]),
                datetime.fromisoformat(tr[1]),
            )

        meta_no_prov = BundleMeta(
            schema_version=meta_dict["schema_version"],
            location_kind=stored_kind,
            convention=stored_convention,
            variables=var_specs,
            time_range=time_range,
            target_grid_id=meta_dict.get("target_grid_id"),
            provenance=(),
            extras=meta_dict.get("extras", {}),
        )
        prov = ProvenanceStep(
            kind="source",
            name="ZarrBundleSource",
            params={"path": str(self._path)},
            timestamp=datetime.utcnow(),
            bundle_hash="00000000",
        )
        meta = BundleMeta(
            schema_version=meta_no_prov.schema_version,
            location_kind=meta_no_prov.location_kind,
            convention=meta_no_prov.convention,
            variables=meta_no_prov.variables,
            time_range=meta_no_prov.time_range,
            target_grid_id=meta_no_prov.target_grid_id,
            provenance=(prov,),
            extras=meta_no_prov.extras,
        )
        return DataBundle(data=ds, meta=meta)
