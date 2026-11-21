"""Zarr-backed ConsumerAdapter family.

Includes 8 consumers:
  - prepared_state_zarr / prepared_forcing_zarr / prepared_boundary_zarr /
    prepared_mesh_zarr  (CF_STANDARD, stage1a outputs)
  - locomgcm_ic_zarr / locomgcm_obc_zarr / locomgcm_forcing_zarr /
    torch_run_zarr  (LOCOMGCM, stage1b/stage2 outputs)
  - mock_zarr  (testing only; permissive)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from locomgcm.io.runner.layer_a import (
    ConventionTag,
    DataBundle,
    LocationKind,
)

from .._registry import register_consumer
from .._types import (
    ArtifactKind,
    ArtifactRef,
    ConsumerCapabilities,
    ConsumerContext,
)
from .._zarr_utils import atomic_write_zarr, serialize_bundle_meta


def _make_zarr_consumer(
    *,
    plugin_name: str,
    accepts: tuple[LocationKind, ...],
    required_convention: ConventionTag,
    required_variables: tuple[str, ...],
    output_filename: str,
    schema_version: str = "0.1.0",
) -> type:
    """Factory for zarr consumer classes.

    Returns a class with the given capabilities; class implements
    ConsumerAdapter Protocol.
    """

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=accepts,
        required_convention=required_convention,
        required_variables=required_variables,
        artifact_kind=ArtifactKind.FILE_ZARR,
        artifact_schema_version=schema_version,
    )

    class _ZarrConsumer:
        # Class attribute (not instance)
        pass

    _ZarrConsumer.capabilities = capabilities

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        # Framework-side pre-validation (location_kind / convention / variables)
        # is the caller's responsibility (the framework call pattern).
        # But we re-check here defensively so direct callers (not via framework)
        # still get a clear error.
        if bundle.meta.location_kind not in self.capabilities.accepts_location_kinds:
            raise ValueError(
                f"{plugin_name}: bundle location_kind {bundle.meta.location_kind} "
                f"not in {self.capabilities.accepts_location_kinds}"
            )
        if bundle.meta.convention != self.capabilities.required_convention:
            raise ValueError(
                f"{plugin_name}: bundle convention {bundle.meta.convention} "
                f"!= required {self.capabilities.required_convention}"
            )
        bundle_var_names = {v.name for v in bundle.meta.variables}
        missing = set(self.capabilities.required_variables) - bundle_var_names
        if missing:
            raise ValueError(
                f"{plugin_name}: bundle missing required variables {missing}"
            )

        # Write atomically; embed meta as zarr attr 'bundle_meta'
        final_path = self._ctx.output_dir / output_filename
        final_path.parent.mkdir(parents=True, exist_ok=True)
        meta_json = serialize_bundle_meta(bundle.meta)
        ds_to_save = bundle.data.assign_attrs(bundle_meta=meta_json)
        final_path = atomic_write_zarr(ds_to_save, final_path)

        # Compute file size
        try:
            size_bytes = sum(f.stat().st_size for f in final_path.rglob("*") if f.is_file())
        except Exception:
            size_bytes = -1

        return ArtifactRef(
            name=plugin_name,
            path=final_path,
            artifact_kind=ArtifactKind.FILE_ZARR,
            bundle_hash=bundle.meta.provenance[-1].bundle_hash if bundle.meta.provenance else "00000000",
            schema_version=self.capabilities.artifact_schema_version,
            extras={"size_bytes": size_bytes},
        )

    _ZarrConsumer.__init__ = __init__  # type: ignore[assignment]
    _ZarrConsumer.consume = consume  # type: ignore[assignment]
    _ZarrConsumer.__name__ = plugin_name + "_Class"
    return _ZarrConsumer


# ============================================================
# Register all 8 zarr consumers
# ============================================================

# CF_STANDARD consumers (stage1a outputs)

PreparedStateZarrAdapter = _make_zarr_consumer(
    plugin_name="prepared_state_zarr",
    accepts=(LocationKind.GRIDDED_3D_OCEAN,),
    required_convention=ConventionTag.CF_STANDARD,
    required_variables=(),  # caller decides what state vars to include
    output_filename="prepared_state.zarr",
)
register_consumer("prepared_state_zarr")(PreparedStateZarrAdapter)

PreparedForcingZarrAdapter = _make_zarr_consumer(
    plugin_name="prepared_forcing_zarr",
    accepts=(LocationKind.GRIDDED_2D_SURFACE,),
    required_convention=ConventionTag.CF_STANDARD,
    required_variables=(),
    output_filename="prepared_forcing.zarr",
)
register_consumer("prepared_forcing_zarr")(PreparedForcingZarrAdapter)

PreparedBoundaryZarrAdapter = _make_zarr_consumer(
    plugin_name="prepared_boundary_zarr",
    accepts=(LocationKind.BOUNDARY_TIMESERIES,),
    required_convention=ConventionTag.CF_STANDARD,
    required_variables=(),
    output_filename="prepared_boundary.zarr",
)
register_consumer("prepared_boundary_zarr")(PreparedBoundaryZarrAdapter)

PreparedMeshZarrAdapter = _make_zarr_consumer(
    plugin_name="prepared_mesh_zarr",
    accepts=(LocationKind.MESH,),
    required_convention=ConventionTag.LOCOMGCM,
    required_variables=(),
    output_filename="prepared_mesh.zarr",
)
register_consumer("prepared_mesh_zarr")(PreparedMeshZarrAdapter)

# Canonical tidal harmonics zarr (Greenwich amplitude/phase per constituent +
# side + along). Start-date independent — V0 is applied at model-load time, not
# here — so this artifact is reusable across simulation start dates.
PreparedTidalZarrAdapter = _make_zarr_consumer(
    plugin_name="prepared_tidal_zarr",
    accepts=(LocationKind.TIDAL_HARMONICS,),
    required_convention=ConventionTag.CF_STANDARD,
    required_variables=(),
    output_filename="prepared_tidal.zarr",
)
register_consumer("prepared_tidal_zarr")(PreparedTidalZarrAdapter)

# LOCOMGCM consumers (stage1b/stage2 outputs)

LOCOMgcmICAdapter = _make_zarr_consumer(
    plugin_name="locomgcm_ic_zarr",
    accepts=(LocationKind.GRIDDED_3D_OCEAN,),
    required_convention=ConventionTag.LOCOMGCM,
    required_variables=(),
    output_filename="locomgcm_ic.zarr",
)
register_consumer("locomgcm_ic_zarr")(LOCOMgcmICAdapter)

LOCOMgcmOBCAdapter = _make_zarr_consumer(
    plugin_name="locomgcm_obc_zarr",
    accepts=(LocationKind.BOUNDARY_TIMESERIES,),
    required_convention=ConventionTag.LOCOMGCM,
    required_variables=(),
    output_filename="locomgcm_obc.zarr",
)
register_consumer("locomgcm_obc_zarr")(LOCOMgcmOBCAdapter)

LOCOMgcmForcingAdapter = _make_zarr_consumer(
    plugin_name="locomgcm_forcing_zarr",
    accepts=(LocationKind.GRIDDED_2D_SURFACE,),
    required_convention=ConventionTag.LOCOMGCM,
    required_variables=(),
    output_filename="locomgcm_forcing.zarr",
)
register_consumer("locomgcm_forcing_zarr")(LOCOMgcmForcingAdapter)

TorchRunOutputAdapter = _make_zarr_consumer(
    plugin_name="torch_run_zarr",
    accepts=(LocationKind.GRIDDED_3D_OCEAN,),
    required_convention=ConventionTag.LOCOMGCM,
    required_variables=(),
    output_filename="torch_run.zarr",
)
register_consumer("torch_run_zarr")(TorchRunOutputAdapter)


# ============================================================
# Mock zarr — accepts anything, records consumed bundles for testing
# ============================================================


@register_consumer("mock_zarr")
class MockZarrAdapter:
    """Test-only consumer; records every bundle it sees in `_recorded`."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,  # permissive
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_ZARR,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx
        self._recorded: list[DataBundle] = []

    @property
    def recorded(self) -> list[DataBundle]:
        return list(self._recorded)

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        # Permissive: accept any convention
        self._recorded.append(bundle)
        return ArtifactRef(
            name="mock_zarr",
            path=None,
            artifact_kind=ArtifactKind.FILE_ZARR,
            bundle_hash=bundle.meta.provenance[-1].bundle_hash if bundle.meta.provenance else "00000000",
            schema_version="0.1.0",
            extras={"mock": True},
        )
