"""Universal yaml-driven model builders (plugin factory).

Consumes RunConfig.model manifest + already-fetched IC bundle + OBcs source
+ already-built grid -> returns assembled TorchRegionalOceanModel + state.
"""

from locomgcm.io.runner.builders.from_manifest import (
    ManifestBuildError,
    build_model_from_manifest,
)
from locomgcm.io.runner.builders.grid_factory import build_grid_from_run_cfg
from locomgcm.io.runner.builders.nc_snapshot import (
    NCVarSpec,
    current_frame_time_seconds,
    default_state_variables,
    make_nc_writer,
    replace_variable,
)
from locomgcm.io.runner.builders.obcs_bridge import (
    BoundaryBundleSample,
    BoundaryBundleSource,
)

__all__ = [
    "build_model_from_manifest",
    "ManifestBuildError",
    "build_grid_from_run_cfg",
    "make_nc_writer",
    "NCVarSpec",
    "current_frame_time_seconds",
    "default_state_variables",
    "replace_variable",
    "BoundaryBundleSample",
    "BoundaryBundleSource",
]
