from .hycom import (
    HYCOMFullDepthPatch,
    HYCOMSurfacePatch,
    HYCOMSurfacePatchRequest,
    choose_stable_timestep,
    extract_full_depth_patch,
    extract_surface_patch,
    full_depth_patch_to_model_state,
    load_full_depth_patch,
    load_surface_patch,
    patch_to_model_state,
)
from .hycom_open_boundary import (
    BarotropicOpenBoundarySource,
    BoundaryTimeMode,
    HYCOMOpenBoundarySource,
    HYCOMTracerOpenBoundary,
    OpenBoundaryField,
    extract_barotropic_open_boundary_source,
    extract_full_depth_tracer_open_boundary_source,
    extract_tracer_open_boundary,
    extract_tracer_open_boundary_source,
    load_tracer_open_boundary,
    load_tracer_open_boundary_source,
    replicate_open_boundary_source_vertical,
)
from .hycom_zarr_cache import HYCOMZarrCacheInfo, build_hycom_zarr_cache, open_hycom_zarr_cache
from .era5 import (
    ERA5BulkInputs,
    ERA5BulkSource,
    ERA5WindSource,
    load_era5_bulk_source_by_time,
    load_era5_wind_source,
)
# NOTE: `load_era5_wind_source_by_time` (and `_extract_wind_source`) are
# intentionally NOT re-exported at the package top; consumers reach them via
# `from defgcm.io.era5 import ...` (submodule-only).
from .topography import ModelBathymetry, load_bathymetry
from .reference_profile import compute_reference_tracer_profile
from .cs32x15_loader import (
    decompose_compact_to_faces,
    load_compact_field,
    load_face_geometry,
    load_proj_2uEvN,
)

__all__ = [
    "BarotropicOpenBoundarySource",
    "BoundaryTimeMode",
    "ERA5BulkInputs",
    "ERA5BulkSource",
    "ERA5WindSource",
    "load_era5_bulk_source_by_time",
    "HYCOMFullDepthPatch",
    "HYCOMSurfacePatch",
    "HYCOMSurfacePatchRequest",
    "HYCOMOpenBoundarySource",
    "HYCOMTracerOpenBoundary",
    "HYCOMZarrCacheInfo",
    "ModelBathymetry",
    "OpenBoundaryField",
    "build_hycom_zarr_cache",
    "choose_stable_timestep",
    "compute_reference_tracer_profile",
    "decompose_compact_to_faces",
    "extract_barotropic_open_boundary_source",
    "extract_full_depth_patch",
    "extract_full_depth_tracer_open_boundary_source",
    "extract_surface_patch",
    "extract_tracer_open_boundary",
    "extract_tracer_open_boundary_source",
    "full_depth_patch_to_model_state",
    "load_bathymetry",
    "load_compact_field",
    "load_era5_wind_source",
    "load_face_geometry",
    "load_full_depth_patch",
    "load_proj_2uEvN",
    "load_surface_patch",
    "load_tracer_open_boundary",
    "load_tracer_open_boundary_source",
    "open_hycom_zarr_cache",
    "patch_to_model_state",
    "replicate_open_boundary_source_vertical",
]
