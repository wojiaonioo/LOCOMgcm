"""Canonical model-ready input builder for runner stages."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

_logger = logging.getLogger(__name__)

import numpy as np
import torch
import xarray as xr
import yaml

from locomgcm.geometry import StructuredCGrid
from locomgcm.io.runner.config import RunConfig
from locomgcm.io.runner.config._schema import DynamicsEntry, ModelManifest
from locomgcm.io.runner.builders import (
    BoundaryBundleSource,
    build_grid_from_run_cfg,
    build_model_from_manifest,
)
from locomgcm.io.runner.layer_a import (
    BundleMeta,
    ConventionTag,
    DataBundle,
    LocationKind,
    ProvenanceStep,
    VariableSpec,
)
from locomgcm.io.runner.layer_b import DATA_SOURCES, DataQuery, Grid, SourceContext
from locomgcm.io.runner.layer_c import OPERATORS
from locomgcm.io.runner.layer_d._types import ArtifactKind, ArtifactRef
from locomgcm.io.runner.layer_d._zarr_utils import atomic_write_zarr, serialize_bundle_meta
from locomgcm.io.runner.legacy_bridge import (
    _assert_no_double_track,
    translate_to_legacy_flat_yaml,
)
from locomgcm.io.runner._canonical_helpers import (
    BarotropicBoundaryBundleSource as _BarotropicBoundaryBundleSource,
    boundary_source_from_bundle as _boundary_source_from_bundle,
    build_bulk_source as _build_bulk_source,
    enrich_manifest as _enrich_manifest,
    reference_profile_from_tensors as _reference_profile_from_tensors,
    runtime_kwargs as _runtime_kwargs,
    seconds_from_time_coord as _seconds_from_time_coord,
    surface_flux_source_from_bundle as _surface_flux_source_from_bundle,
    surface_pressure_load_from_bundle as _surface_pressure_load_from_bundle,
    uses_surface_pressure_load as _uses_surface_pressure_load,
    wind_source_from_bundle as _wind_source_from_bundle,
)


@dataclass(frozen=True)
class ModelInput:
    """Model-ready inputs shared by LOCOMgcm and MITgcm serializers.

    With ``mit_comparison.enabled`` true, ``run_params`` is the resolved
    legacy-args dictionary the MITgcm export reads, not the raw
    ``argparse.Namespace``; the only omitted key is the transient temp
    ``config`` path used to feed the legacy parser.  With it false,
    ``run_params`` holds only a few resolved LOCOMgcm values
    (``_model_input_run_params``).
    """

    grid: StructuredCGrid
    initial_state: Any
    boundary_source: Any
    era5_source: Any
    barotropic_source: Any
    run_params: Mapping[str, Any]


MODEL_INPUT_ARTIFACT_NAME = "model_input_initial_state_zarr"


def build_model_state(
    run_cfg: RunConfig,
    raw_yaml: dict,
    *,
    stage1a_bundles: Mapping[str, DataBundle] | None = None,
) -> ModelInput:
    """Build model-ready initial state from yaml-declared Layer B/C artifacts.

    Canonical state assembly entry point. Five clear phases:

    1. ``_stage1a_bundles`` — pull RawBundles from source pipelines (state /
       boundary / mesh / forcing).
    2. ``_ic_tensors_from_bundles`` — assemble IC tensor map plus the
       positive-depth bathymetry tensor.
    3. ``_apply_canonical_transforms`` — cmask, face reconstruction,
       face mask. The ONE place the canonical pipeline
       does numerical IC transforms.
    4. ``_build_runtime_sources`` — wrap prepared boundary / forcing
       bundles in their streaming-source classes.
    5. ``_assemble_model`` — enrich manifest, instantiate model, build the
       initial state with ``create_state`` (the run's initial field; the
       base-time boundaries are the runner's, canonical/state_schema.py
       INVARIANT-3).

    ``raw_yaml`` is the merged raw YAML dict (from
    ``locomgcm.io.runner.config._loader.merge_yaml_layers``) — the
    canonical translator consumes it for the fail-loud completeness check.
    """

    run_params = _model_input_run_params(run_cfg, raw_yaml)
    device = _resolve_device(run_cfg)
    # Single dtype source for this path, mirroring experiments/cases/_runtime.py:
    # resolve run.precision once here and thread it through the dtype= arguments
    # every builder below already takes.  Default float64 keeps existing runs
    # bit-identical; unknown values are rejected at config load
    # (RunIdentity.__post_init__) and guarded again here.
    _precision = getattr(run_cfg.run, "precision", "float64")
    if _precision == "float64":
        dtype = torch.float64
    elif _precision == "float32":
        dtype = torch.float32
    else:
        raise ValueError(
            f"unsupported run.precision {_precision!r}; expected 'float64' or 'float32'"
        )
    target_grid = _target_grid(run_cfg)
    logger = _NullLogger()

    if stage1a_bundles is None:
        stage1a_bundles = _stage1a_bundles(run_cfg, target_grid=target_grid, logger=logger)
    else:
        # Treat the supplied mapping as an overlay: for any required bundle name
        # that is absent, fall back to the same per-pipeline recomputation that
        # the None path uses.  This handles partial stage1a runs (e.g. a config
        # without a boundary pipeline, or a subset stage1a pass) without
        # raising KeyError where the old None path would silently skip the step.
        stage1a_bundles = dict(stage1a_bundles)
        for _bundle_name in ("state", "boundary", "mesh", "forcing", "tidal"):
            if _bundle_name not in stage1a_bundles:
                try:
                    stage1a_bundles[_bundle_name] = _run_stage1a_pipeline_bundle(
                        run_cfg,
                        _bundle_name,
                        target_grid=target_grid,
                        logger=logger,
                    )
                except KeyError:
                    pass  # pipeline not declared in config; skip as _stage1a_bundles does
    ic_bundle = stage1a_bundles["state"]

    ic_tensors = _ic_tensors_from_bundles(stage1a_bundles, run_cfg, device=device, dtype=dtype)
    grid = _build_grid_from_tensors(
        run_cfg,
        ic_tensors["bathymetry"],
        del_x=ic_tensors.get("del_x"),
        del_y=ic_tensors.get("del_y"),
        device=device,
        dtype=dtype,
    )
    _apply_canonical_transforms(ic_tensors, grid, dtype=dtype, device=device)

    obcs_source, wind_source, surface_flux_source, bulk_source = _build_runtime_sources(
        stage1a_bundles, run_cfg, device=device, dtype=dtype
    )

    surface_pressure_load = (
        _surface_pressure_load_from_bundle(stage1a_bundles["forcing"], device=device, dtype=dtype)
        if _uses_surface_pressure_load(run_cfg) else None
    )

    model, initial_state = _assemble_model(
        run_cfg,
        ic_bundle=ic_bundle,
        ic_tensors=ic_tensors,
        grid=grid,
        obcs_source=obcs_source,
        wind_source=wind_source,
        surface_flux_source=surface_flux_source,
        bulk_source=bulk_source,
        surface_pressure_load=surface_pressure_load,
        device=device,
        tidal_bundle=stage1a_bundles.get("tidal"),
    )

    return ModelInput(
        grid=model.grid,
        initial_state=initial_state,
        boundary_source=obcs_source,
        era5_source=wind_source,
        barotropic_source=_BarotropicBoundaryBundleSource(obcs_source),
        run_params=run_params,
    )


def _ic_tensors_from_bundles(
    bundles: dict[str, "DataBundle"],
    run_cfg: RunConfig,
    *,
    device: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """assemble the IC tensor map and the positive-depth bathymetry.

    Bathymetry preference order:
        1. ``mesh`` bundle ``bathymetry`` variable (regridded GEBCO).
        2. Pre-existing key on the state ``ic_tensors`` (legacy paths).
        3. Constant ``run_cfg.grid.depth_m`` filled to (ny, nx).
    """
    ic_bundle = bundles["state"]
    ic_tensors = _ic_tensor_map(ic_bundle, device=device, dtype=dtype)
    mesh_bundle = bundles.get("mesh")
    # Bathymetry is a geometry input, not a prognostic field: it is consumed
    # only by the grid constructor, which derives the land mask from it by an
    # untoleranced comparison.  Narrowing it to the run dtype here moves the
    # mask itself (grid.py ``_GEOMETRY_DTYPE_NOTE``), so it stays float64 and
    # the grid narrows the *derived* metrics instead.
    if mesh_bundle is not None and "bathymetry" in mesh_bundle.data:
        ic_tensors["bathymetry"] = torch.as_tensor(
            _positive_depth_from_mesh(mesh_bundle),
            dtype=torch.float64,
            device=device,
        )
    elif "bathymetry" not in ic_tensors:
        ic_tensors["bathymetry"] = torch.full(
            (int(run_cfg.grid.ny), int(run_cfg.grid.nx)),
            float(run_cfg.grid.depth_m),
            dtype=torch.float64,
            device=device,
        )
    if mesh_bundle is not None:
        spacing_names = {
            name for name in ("del_x", "del_y") if name in mesh_bundle.data
        }
        if spacing_names and spacing_names != {"del_x", "del_y"}:
            raise ValueError(
                "mesh vector spacing requires both del_x and del_y; "
                f"found {sorted(spacing_names)}"
            )
        for name, count in (
            ("del_x", int(run_cfg.grid.nx)),
            ("del_y", int(run_cfg.grid.ny)),
        ):
            if name not in spacing_names:
                continue
            values = np.asarray(mesh_bundle.data[name].values)
            if values.shape != (count,):
                raise ValueError(
                    f"mesh {name} shape must be ({count},), got {values.shape}"
                )
            ic_tensors[name] = torch.as_tensor(
                np.ascontiguousarray(values, dtype=np.float64),
                dtype=torch.float64,
                device=device,
            )
    return ic_tensors


def _apply_canonical_transforms(
    ic_tensors: dict[str, torch.Tensor],
    grid: StructuredCGrid,
    *,
    dtype: torch.dtype,
    device: str,
) -> None:
    """In-place canonical-IC numerical transforms.

    Sequence (matches MIT's IC processing path):
      1. ``cmask`` zeros T/S on dry cells.
      2. Native MIT u/v receive only LOCOMgcm's redundant outer face;
         cell-center u/v use ``center_to_u_centered`` / ``center_to_v_centered``.
      3. ``u_mask`` / ``v_mask`` zero u/v at wet/dry interface faces
         (mirrors MIT hFacW/hFacS = 0 at those faces).
    """
    if grid.masks.has_3d_masks:
        cmask = grid.masks.center_float(dtype=dtype, device=device)
        wet_center = cmask > 0
        # Match MIT INI_THETA/INI_SALT conditional zeroing: select wet values
        # unchanged instead of multiplying by the mask.
        if "temperature" in ic_tensors:
            temperature = ic_tensors["temperature"]
            ic_tensors["temperature"] = torch.where(
                wet_center, temperature, temperature.new_zeros(())
            )
        if "salinity" in ic_tensors:
            salinity = ic_tensors["salinity"]
            ic_tensors["salinity"] = torch.where(
                wet_center, salinity, salinity.new_zeros(())
            )

    from locomgcm.operators.reconstructions.centered import (
        center_to_u_centered,
        center_to_v_centered,
    )
    if "u" in ic_tensors:
        u_c = ic_tensors["u"]
        if u_c.shape[-2:] == (grid.ny, grid.nx + 1):
            pass
        elif u_c.ndim == 3:
            ic_tensors["u"] = center_to_u_centered(
                u_c.unsqueeze(0), grid, face_mask=None
            ).squeeze(0)
        else:
            ic_tensors["u"] = center_to_u_centered(u_c, grid, face_mask=None)
    if "v" in ic_tensors:
        v_c = ic_tensors["v"]
        if v_c.shape[-2:] == (grid.ny + 1, grid.nx):
            pass
        elif v_c.ndim == 3:
            ic_tensors["v"] = center_to_v_centered(
                v_c.unsqueeze(0), grid, face_mask=None
            ).squeeze(0)
        else:
            ic_tensors["v"] = center_to_v_centered(v_c, grid, face_mask=None)

    if grid.masks.has_3d_masks:
        if "u" in ic_tensors:
            u_mask = grid.masks.u_float(dtype=dtype, device=device)
            u = ic_tensors["u"]
            ic_tensors["u"] = torch.where(u_mask > 0, u, u.new_zeros(()))
        if "v" in ic_tensors:
            v_mask = grid.masks.v_float(dtype=dtype, device=device)
            v = ic_tensors["v"]
            ic_tensors["v"] = torch.where(v_mask > 0, v, v.new_zeros(()))


def _build_runtime_sources(
    bundles: dict[str, "DataBundle"],
    run_cfg: RunConfig,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[Any, Any | None, Any | None, Any | None]:
    """wrap prepared boundary / forcing bundles in streaming sources."""
    obcs_source = _boundary_source_from_bundle(
        bundles["boundary"],
        side_names=tuple(str(v) for v in _extra(run_cfg, "obcs_sides", ("W", "E"))),
        device=device,
        dtype=dtype,
    )
    wind_source = None
    surface_flux_source = None
    forcing_bundle = bundles.get("forcing")
    bulk_source = _build_bulk_source(run_cfg, device=device, dtype=dtype)
    if forcing_bundle is not None:
        data = forcing_bundle.data
        bulk_entries = [e for e in run_cfg.model.dynamics if e.name == "era5_bulk_forcing"]
        velocity_bundle = (bulk_source is not None and not {"tau_x", "tau_y"}.issubset(data)
                           and {"wind_u", "wind_v"}.issubset(data)
                           and dict(bulk_entries[0].kwargs).get("use_atm_wind", True))
        wind_source = bulk_source if velocity_bundle else _wind_source_from_bundle(
            forcing_bundle,
            device=device,
            dtype=dtype,
            time_mode=str(_source_kwargs(run_cfg, "forcing").pop("time_mode", "linear")),
        )
        surface_flux_source = _surface_flux_source_from_bundle(
            forcing_bundle,
            device=device,
            dtype=dtype,
            time_mode=str(_source_kwargs(run_cfg, "forcing").pop("time_mode", "linear")),
        )
    return obcs_source, wind_source, surface_flux_source, bulk_source


def _assemble_model(
    run_cfg: RunConfig,
    *,
    ic_bundle: "DataBundle",
    ic_tensors: dict[str, torch.Tensor],
    grid: StructuredCGrid,
    obcs_source: Any,
    wind_source: Any | None,
    surface_flux_source: Any | None = None,
    bulk_source: Any | None = None,
    surface_pressure_load: torch.Tensor | None = None,
    device: str,
    tidal_bundle: "DataBundle | None" = None,
) -> tuple[Any, Any]:
    """enrich manifest, instantiate model, build the initial field (``create_state``)."""
    enriched_cfg = _enrich_manifest(
        run_cfg,
        ic_bundle,
        ic_tensors,
        obcs_source=obcs_source,
        wind_source=wind_source,
        surface_flux_source=surface_flux_source,
        bulk_source=bulk_source,
        surface_pressure_load=surface_pressure_load,
        device=device,
        tidal_data=(tidal_bundle.data if tidal_bundle is not None else None),
    )
    return build_model_from_manifest(
        enriched_cfg,
        grid=grid,
        ic_bundle=ic_tensors,
        obcs_source=obcs_source,
        forcing_source=None,
    )


def write_model_input(model_input: ModelInput, directory: Path | str) -> dict[str, ArtifactRef]:
    """Serialize the canonical model input artifact for downstream stages.

    consumers only read ``initial_state.zarr``.  ``grid`` and
    ``run_params`` are written alongside it so namelist consumers can
    stop reconstructing model parameters.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    initial_ref = write_initial_state_zarr(
        model_input.initial_state,
        directory / "initial_state.zarr",
        grid=model_input.grid,
    )
    write_grid_artifacts(model_input.grid, directory / "grid")
    write_run_params(model_input.run_params, directory / "run_params.yaml")

    return {"initial_state": initial_ref}


def read_initial_state_dataset(path: Path | str) -> xr.Dataset:
    """Open a serialized ModelInput initial_state zarr dataset."""

    return xr.open_zarr(Path(path), consolidated=False)


def write_initial_state_zarr(
    initial_state: Any,
    path: Path | str,
    *,
    grid: StructuredCGrid,
) -> ArtifactRef:
    """Write ``ModelInput.initial_state`` as a Layer-A compatible zarr bundle."""

    path = Path(path)
    ds = initial_state_to_dataset(initial_state)
    prov = ProvenanceStep(
        kind="source",
        name="ModelInput.initial_state",
        params={},
        timestamp=datetime.utcnow(),
        bundle_hash="00000000",
    )
    meta = _initial_state_meta(ds, grid=grid, provenance=(prov,))
    bundle = DataBundle(data=ds, meta=meta)
    ds_to_save = bundle.data.assign_attrs(bundle_meta=serialize_bundle_meta(bundle.meta))
    final_path = atomic_write_zarr(ds_to_save, path)
    try:
        size_bytes = sum(f.stat().st_size for f in final_path.rglob("*") if f.is_file())
    except Exception:
        size_bytes = -1

    return ArtifactRef(
        name=MODEL_INPUT_ARTIFACT_NAME,
        path=final_path,
        artifact_kind=ArtifactKind.FILE_ZARR,
        bundle_hash=bundle.meta.provenance[-1].bundle_hash,
        schema_version="0.1.0",
        extras={"size_bytes": size_bytes},
    )


def initial_state_to_dataset(initial_state: Any) -> xr.Dataset:
    """Convert a LOCOMgcm PrognosticState into the canonical IC dataset."""

    eta = _state_array(initial_state.eta.data, expected_ndim=2)
    temp = _state_array(initial_state.temp.data, expected_ndim=3)
    salt = _state_array(initial_state.salt.data, expected_ndim=3)
    u = _state_array(initial_state.u.data, expected_ndim=3)
    v = _state_array(initial_state.v.data, expected_ndim=3)

    nz, ny, nx = temp.shape
    if salt.shape != (nz, ny, nx):
        raise ValueError(f"salt shape {salt.shape} != temp shape {(nz, ny, nx)}")
    if eta.shape != (ny, nx):
        raise ValueError(f"eta shape {eta.shape} != {(ny, nx)}")
    if u.shape != (nz, ny, nx + 1):
        raise ValueError(f"u shape {u.shape} != {(nz, ny, nx + 1)}")
    if v.shape != (nz, ny + 1, nx):
        raise ValueError(f"v shape {v.shape} != {(nz, ny + 1, nx)}")

    return xr.Dataset(
        data_vars={
            "eta": (("latitude", "longitude"), eta),
            "temp": (("z", "latitude", "longitude"), temp),
            "salt": (("z", "latitude", "longitude"), salt),
            "u": (("z", "latitude", "longitude_u"), u),
            "v": (("z", "latitude_v", "longitude"), v),
        },
        coords={
            "z": np.arange(nz, dtype=np.int32),
            "latitude": np.arange(ny, dtype=np.int32),
            "longitude": np.arange(nx, dtype=np.int32),
            "latitude_v": np.arange(ny + 1, dtype=np.int32),
            "longitude_u": np.arange(nx + 1, dtype=np.int32),
        },
    )


def write_grid_artifacts(grid: StructuredCGrid, directory: Path | str) -> None:
    """Write enough grid data for inspection and reconstruction."""

    directory = Path(directory)
    tmp_dir = directory.parent / (directory.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    vector_spacing = grid.metrics.del_x is not None or grid.metrics.del_y is not None
    if vector_spacing and (
        grid.metrics.del_x is None or grid.metrics.del_y is None
    ):
        raise ValueError("vector grid artifact requires both del_x and del_y")

    scalars = {
        "nx": int(grid.nx),
        "ny": int(grid.ny),
        "nz": int(grid.nz),
        "dx": float(grid.metrics.dx_scalar if vector_spacing else grid.dx),
        "dy": float(grid.metrics.dy_scalar if vector_spacing else grid.dy),
        "boundary_topology": {
            "west_east": str(grid.boundary_topology.west_east),
            "south_north": str(grid.boundary_topology.south_north),
        },
    }
    if vector_spacing:
        scalars["spacing_mode"] = "vector"
    (tmp_dir / "scalars.yaml").write_text(yaml.safe_dump(scalars, sort_keys=False))

    tensors = {
        "layer_thickness": _to_numpy(grid.layer_thickness),
        "center_depth": _to_numpy(grid.center_depth),
        "interface_depth": _to_numpy(grid.interface_depth),
        "dx_center": _to_numpy(grid.metrics.dx_center),
        "dy_center": _to_numpy(grid.metrics.dy_center),
        "cell_area": _to_numpy(grid.metrics.cell_area),
        "cell_volume": _to_numpy(grid.metrics.cell_volume),
        "depth_center": _to_numpy(grid.metrics.depth_center),
        "face_area_u": _to_numpy(grid.metrics.face_area_u),
        "face_area_v": _to_numpy(grid.metrics.face_area_v),
        "dxG_row": _to_numpy(grid.metrics.dxG_row),
        "dxC_row": _to_numpy(grid.metrics.dxC_row),
        "dxG_row_ext": _to_numpy(grid.metrics.dxG_row_ext),
        "rAz_row": _to_numpy(grid.metrics.rAz_row),
        "mask_center": _to_numpy(grid.masks.center),
        "mask_u": _to_numpy(grid.masks.u),
        "mask_v": _to_numpy(grid.masks.v),
    }
    if vector_spacing:
        vector_names = (
            "del_x",
            "del_y",
            "dxF",
            "dyF",
            "dxG",
            "dyG",
            "dxC",
            "dyC",
            "dxV",
            "dyU",
            "rA",
            "rAw",
            "rAs",
            "rAz",
        )
        tensors.update(
            {name: _to_numpy(getattr(grid.metrics, name)) for name in vector_names}
        )
    optional_masks = {
        "hFacC": grid.masks.hFacC,
        "hFacW": grid.masks.hFacW,
        "hFacS": grid.masks.hFacS,
        "maskInC": grid.masks.maskInC,
        "maskInW": grid.masks.maskInW,
        "maskInS": grid.masks.maskInS,
    }
    tensors.update({k: _to_numpy(v) for k, v in optional_masks.items() if v is not None})
    np.savez(tmp_dir / "tensors.npz", **tensors)

    if directory.exists():
        shutil.rmtree(directory)
    tmp_dir.rename(directory)


def write_run_params(run_params: Mapping[str, Any], path: Path | str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(_plain_yaml(dict(run_params)), sort_keys=False))
    if path.exists():
        path.unlink()
    tmp.rename(path)


def runconfig_to_legacy_args(run_cfg: RunConfig, raw_yaml: dict) -> dict[str, Any]:
    """Translate ``RunConfig`` + raw merged YAML to resolved run params.

    The canonical bridge checks that every configuration field is handled.
    """

    flat = translate_to_legacy_flat_yaml(run_cfg, raw_yaml, target="model_build")
    return {str(key).replace("-", "_"): _yaml_scalar(value) for key, value in flat.items()}


def _resolved_run_params(args: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(args)
    params.pop("config", None)
    return params


def _model_input_run_params(run_cfg: RunConfig, raw_yaml: dict) -> dict[str, Any]:
    """Run params for ``model_input/run_params.yaml``.

    With ``mit_comparison.enabled`` the MITgcm translation runs: its
    completeness check rejects yaml keys the MITgcm parameter table does not
    claim, and the stage 1c export reads the full legacy dictionary.  Without
    it no LOCOMgcm stage reads the file, so only the double-track
    check runs and the file keeps a few resolved values, under the legacy key
    names, for tooling that looks for it.
    """
    if run_cfg.mit_comparison.enabled:
        return _resolved_run_params(runconfig_to_legacy_args(run_cfg, raw_yaml))
    _assert_no_double_track(raw_yaml)
    from locomgcm.io.runner._time_steps import compute_n_steps

    time_cfg = run_cfg.time
    return {
        "experiment_name": str(run_cfg.run.experiment_name),
        "device": _resolve_device(run_cfg),
        "resolution_deg": float(run_cfg.grid.resolution_deg),
        "fixed_dt": float(time_cfg.dt),
        "steps": compute_n_steps(
            days=time_cfg.days,
            dt_seconds=time_cfg.dt,
            steps_override=time_cfg.steps,
            seconds_override=time_cfg.seconds if isinstance(time_cfg.seconds, (int, float)) else None,
        ),
        "snapshot_interval_seconds": run_cfg.output.resolved_snapshot_seconds(),
    }


class _NullLogger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _resolve_device(run_cfg: RunConfig) -> str:
    if run_cfg.run.device != "auto":
        return str(run_cfg.run.device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _extra(run_cfg: RunConfig, key: str, default: Any = None) -> Any:
    extras = dict(run_cfg.case.extras or {})
    for candidate in (key, key.replace("_", "-")):
        if candidate in extras and extras[candidate] is not None:
            return extras[candidate]
    return default


def _target_grid(run_cfg: RunConfig) -> Grid:
    return Grid(
        nx=int(run_cfg.grid.nx),
        ny=int(run_cfg.grid.ny),
        nz=int(run_cfg.grid.nz),
        center_lat=float(run_cfg.grid.center_lat),
        center_lon=float(run_cfg.grid.center_lon),
        resolution_deg=float(run_cfg.grid.resolution_deg),
        depth_m=float(run_cfg.grid.depth_m),
        vertical_layer_thicknesses=tuple(float(v) for v in run_cfg.grid.vertical_layer_thicknesses),
        grid_id=f"{run_cfg.run.experiment_name}_{run_cfg.grid.nx}x{run_cfg.grid.ny}x{run_cfg.grid.nz}",
    )


def _source_config_map(run_cfg: RunConfig, source_name: str) -> dict[str, Any]:
    for source in run_cfg.sources:
        if source.name == source_name or source.type == source_name:
            return dict(source.config)
    return {}


def _source_kwargs(run_cfg: RunConfig, prefix: str) -> dict[str, Any]:
    source_name = _extra(run_cfg, f"{prefix}_source", None)
    specific = dict(_extra(run_cfg, f"{prefix}_source_extras", {}) or {})
    specific.pop("stage1b_consumer", None)
    if source_name is not None:
        merged = _source_config_map(run_cfg, str(source_name))
        merged.update(specific)
        return merged
    return specific


def _source_context(run_cfg: RunConfig, target_grid: Grid, logger: Any) -> SourceContext:
    def grid_resolver(grid_id: str) -> Grid:
        if grid_id in {target_grid.grid_id, "model_grid"}:
            return target_grid
        raise ValueError(f"unknown grid_id {grid_id!r}")

    return SourceContext(
        cache_root=Path(run_cfg.cache.root),
        target_grid_resolver=grid_resolver,
        logger=logger,
    )


def _query_time_range(run_cfg: RunConfig) -> tuple[datetime, datetime]:
    """Return the (start, end) datetime window for source data fetching.

    The window is derived from ``run_cfg.time.days`` verbatim (minimum 1 day).
    When ``time.steps`` or ``time.seconds`` overrides the step count to a run
    longer than ``time.days``, this window will be shorter than the effective
    run duration.  :func:`_warn_fetch_window_short` emits a warning in that
    case when any non-cyclic, non-IC source is present.
    """
    from datetime import timedelta

    start = datetime.combine(run_cfg.run.start_date, datetime.min.time())
    fetch_days = max(1.0, float(run_cfg.time.days))
    _warn_fetch_window_short(run_cfg, fetch_days)
    return start, start + timedelta(days=fetch_days)


def _warn_fetch_window_short(run_cfg: RunConfig, fetch_days: float) -> None:
    """Warn when the effective run duration exceeds the source data fetch window.

    The
    ``_query_time_range`` / ``model_input`` fetch path uses ``time.days``
    verbatim as the data window.  When STEPS= or ``time.steps`` overrides
    the step count to a longer run, the fetch window is silently short.

    Fires only when:
      1. Effective run seconds (steps * dt) > fetch window seconds.
      2. At least one configured source has ``time_mode`` other than
         ``'cyclic'`` (cyclic sources repeat regardless of window length).

    No behaviour change: the window is not modified.
    """
    from locomgcm.io.runner._time_steps import compute_n_steps, exact_total_seconds

    time_cfg = run_cfg.time
    try:
        n_steps = compute_n_steps(
            days=time_cfg.days,
            dt_seconds=time_cfg.dt,
            steps_override=time_cfg.steps,
            seconds_override=time_cfg.seconds if isinstance(time_cfg.seconds, (int, float)) else None,
        )
    except Exception:
        return  # trap errors are raised elsewhere

    effective_seconds = exact_total_seconds(n_steps, time_cfg.dt)
    window_seconds = fetch_days * 86400.0

    if effective_seconds <= window_seconds:
        return

    # Check whether any non-cyclic source is configured.
    non_cyclic = [s for s in run_cfg.sources if s.time_mode != "cyclic"]
    if not non_cyclic:
        return

    source_names = ", ".join(s.name for s in non_cyclic)
    _logger.warning(
        "model_input: source fetch window (%.1f days = %.0f s, from time.days) "
        "is shorter than the effective run duration (%d steps × %.1f s/step = %.0f s "
        "= %.2f days). "
        "Non-cyclic sources [%s] may return fewer time frames than the run needs. "
        "Set time.days (or time.seconds) to at least %.1f days to silence this warning.",
        fetch_days,
        window_seconds,
        n_steps,
        time_cfg.dt,
        effective_seconds,
        effective_seconds / 86400.0,
        source_names,
        effective_seconds / 86400.0,
    )


def _source_for_pipeline(
    run_cfg: RunConfig,
    pipeline_name: str,
    *,
    target_grid: Grid,
    logger: Any,
) -> tuple[Any, Any]:
    pipelines = run_cfg.pipelines.get("stage1a_prepare_common", ())
    for pipeline in pipelines:
        if pipeline.name != pipeline_name:
            continue
        source_spec = next(
            (source for source in run_cfg.sources if source.name == pipeline.primary_source),
            None,
        )
        if source_spec is None:
            raise ValueError(f"stage1a pipeline {pipeline_name!r}: unknown source {pipeline.primary_source!r}")
        source = DATA_SOURCES.instantiate(
            source_spec.type,
            _source_context(run_cfg, target_grid, logger),
            **dict(source_spec.config),
        )
        return pipeline, source
    raise KeyError(pipeline_name)


def _initial_kind_for_pipeline(pipeline: Any, source: Any) -> LocationKind:
    if any(op.op == "subset_boundary" for op in pipeline.operators):
        return LocationKind.GRIDDED_3D_OCEAN
    if pipeline.name == "forcing":
        return LocationKind.GRIDDED_2D_SURFACE
    if pipeline.name == "mesh":
        return LocationKind.MESH
    return source.capabilities.produces[0]


def _run_stage1a_pipeline_bundle(
    run_cfg: RunConfig,
    pipeline_name: str,
    *,
    target_grid: Grid,
    logger: Any,
) -> DataBundle:
    pipeline, source = _source_for_pipeline(
        run_cfg,
        pipeline_name,
        target_grid=target_grid,
        logger=logger,
    )
    bundle = source.fetch(
        DataQuery(
            location_kind=_initial_kind_for_pipeline(pipeline, source),
            variables=source.capabilities.variables,
            time_range=_query_time_range(run_cfg),
            target_grid=target_grid,
            extras={},
        )
    )
    for op_spec in pipeline.operators:
        config = dict(op_spec.config)
        if "target_grid" in config and config["target_grid"] == "model_grid":
            config["target_grid"] = target_grid
        op = OPERATORS.instantiate(op_spec.op, None, **config)
        bundle = op(bundle)
    return bundle


def _stage1a_bundles(
    run_cfg: RunConfig,
    *,
    target_grid: Grid,
    logger: Any,
) -> dict[str, DataBundle]:
    bundles: dict[str, DataBundle] = {}
    for name in ("state", "boundary", "mesh", "forcing"):
        try:
            bundles[name] = _run_stage1a_pipeline_bundle(
                run_cfg,
                name,
                target_grid=target_grid,
                logger=logger,
            )
        except KeyError:
            continue
    return bundles


def _positive_depth_from_mesh(bundle: DataBundle) -> np.ndarray:
    values = _initial_frame(bundle.data["bathymetry"].values, expected_ndim=2)
    if bundle.meta.convention == ConventionTag.CF_STANDARD:
        return np.where(values < 0.0, -values, 0.0).astype(np.float64)
    if bundle.meta.convention == ConventionTag.MITGCM:
        return np.where(values < 0.0, -values, values).astype(np.float64)
    return np.maximum(values, 0.0).astype(np.float64)


def _initial_frame(values: Any, *, expected_ndim: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == expected_ndim + 1:
        arr = arr[0]
    if arr.ndim != expected_ndim:
        raise ValueError(f"expected {expected_ndim}D initial field, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float64)


def append_redundant_face(name: str, values: np.ndarray, count: int) -> np.ndarray:
    """A native MITgcm u (nx west faces) or v (ny south faces) with LOCOMgcm's redundant
    periodic outer face appended; ``count`` is nx for u, ny for v, and an array that
    already has the face passes unchanged."""
    axis = -1 if name == "u" else -2
    if values.shape[axis] == count:
        return np.concatenate((values, np.take(values, [0], axis=axis)), axis=axis)
    if values.shape[axis] != count + 1:
        extent = "nx" if name == "u" else "ny"
        raise ValueError(
            f"native {name} face count {values.shape[axis]} is not {extent}={count} or {extent}+1"
        )
    return values


def _ic_tensor_map(
    ic_bundle: DataBundle,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    aliases = {
        "temperature": "temperature",
        "salinity": "salinity",
        "eta": "eta",
        "u": "u",
        "v": "v",
        "bathymetry": "bathymetry",
    }
    tensors: dict[str, torch.Tensor] = {}
    for target_name, source_name in aliases.items():
        if source_name not in ic_bundle.data:
            continue
        expected_ndim = 2 if target_name in {"eta", "bathymetry"} else 3
        source = ic_bundle.data[source_name]
        values = _initial_frame(source.values, expected_ndim=expected_ndim)
        if target_name == "u" and "longitude_u" in source.dims:
            values = append_redundant_face("u", values, int(ic_bundle.data.sizes["longitude"]))
        elif target_name == "v" and "latitude_v" in source.dims:
            values = append_redundant_face("v", values, int(ic_bundle.data.sizes["latitude"]))
        tensors[target_name] = torch.as_tensor(
            values,
            # Bathymetry feeds the grid constructor, whose land mask comes from
            # an untoleranced comparison — it stays float64 whatever the run
            # precision is (grid.py ``_GEOMETRY_DTYPE_NOTE``).
            dtype=torch.float64 if target_name == "bathymetry" else dtype,
            device=device,
        )
    # Center-located u/v are reconstructed after grid creation. Native MIT
    # west/south-face arrays already have the correct values; only LOCOMgcm's
    # redundant periodic outer face is appended here.
    return tensors


def _build_grid_from_tensors(
    run_cfg: RunConfig,
    bathymetry: torch.Tensor,
    *,
    del_x: torch.Tensor | None = None,
    del_y: torch.Tensor | None = None,
    device: str | torch.device,
    dtype: torch.dtype,
) -> StructuredCGrid:
    extras = dict(run_cfg.case.extras or {})
    grid_mode = str(_extra(run_cfg, "grid_mode", "cartesian"))
    hfac_min = float(_extra(run_cfg, "hfac_min", 0.2))
    hfac_min_dz = float(_extra(run_cfg, "hfac_min_dz", 0.0))
    if grid_mode != "spherical_polar":
        return build_grid_from_run_cfg(
            run_cfg,
            bathymetry=bathymetry,
            del_x=del_x,
            del_y=del_y,
            device=device,
            dtype=dtype,
            hfac_min=hfac_min,
            hfac_min_dz=hfac_min_dz,
        )
    if del_x is not None or del_y is not None:
        raise ValueError(
            "spherical_polar file spacing is not supported; translator must defer it"
        )

    from locomgcm.geometry import BoundaryTopology, BoundaryType

    boundary_cfg = getattr(run_cfg, "boundary", None)
    fallback = str(getattr(boundary_cfg, "type", "periodic"))
    yg_origin = float(
        _extra(
            run_cfg,
            "yg_origin",
            float(run_cfg.grid.center_lat)
            - (int(run_cfg.grid.ny) / 2.0) * float(run_cfg.grid.resolution_deg),
        )
    )
    return StructuredCGrid.spherical_polar(
        nx=int(run_cfg.grid.nx),
        ny=int(run_cfg.grid.ny),
        nz=int(run_cfg.grid.nz),
        ygOrigin_deg=yg_origin,
        delY_deg=float(run_cfg.grid.resolution_deg),
        delX_deg=float(run_cfg.grid.resolution_deg),
        depth=float(run_cfg.grid.depth_m),
        # float64 on purpose — see grid.py ``_GEOMETRY_DTYPE_NOTE``.
        bathymetry=torch.as_tensor(bathymetry, dtype=torch.float64, device=device),
        vertical_layer_thicknesses=tuple(float(v) for v in run_cfg.grid.vertical_layer_thicknesses),
        vertical_center_spacing=tuple(run_cfg.grid.vertical_center_spacing),
        r_sphere=float(extras.get("r_sphere", 6_370_000.0)),
        boundary_topology=BoundaryTopology(
            west_east=BoundaryType(str(_extra(run_cfg, "west_east_boundary", fallback))),
            south_north=BoundaryType(str(_extra(run_cfg, "south_north_boundary", fallback))),
        ),
        obcs_fix_topo=bool(getattr(boundary_cfg, "obcs_fix_topo", False)),
        hfac_min=hfac_min,
        hfac_min_dz=hfac_min_dz,
        device=str(device),
        dtype=dtype,
    )


def _yaml_scalar(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def _initial_state_meta(
    ds: xr.Dataset,
    *,
    grid: StructuredCGrid,
    provenance: tuple[ProvenanceStep, ...],
) -> BundleMeta:
    return BundleMeta(
        schema_version="0.1.0",
        location_kind=LocationKind.GRIDDED_3D_OCEAN,
        convention=ConventionTag.LOCOMGCM,
        variables=tuple(
            VariableSpec(
                name=name,
                cf_standard_name=_INITIAL_STATE_STANDARD_NAMES[name],
                units=_INITIAL_STATE_UNITS[name],
                dtype=str(ds[name].dtype),
                dims=tuple(ds[name].dims),
            )
            for name in ("eta", "temp", "salt", "u", "v")
        ),
        time_range=None,
        target_grid_id=f"model_grid_{grid.nx}x{grid.ny}x{grid.nz}",
        provenance=provenance,
    )


_INITIAL_STATE_STANDARD_NAMES = {
    "eta": "sea_surface_height_above_geoid",
    "temp": "sea_water_potential_temperature",
    "salt": "sea_water_salinity",
    "u": "eastward_sea_water_velocity",
    "v": "northward_sea_water_velocity",
}
_INITIAL_STATE_UNITS = {
    "eta": "m",
    "temp": "degC",
    "salt": "psu",
    "u": "m s-1",
    "v": "m s-1",
}


def _state_array(value: Any, *, expected_ndim: int) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == expected_ndim + 1:
        if arr.shape[0] != 1:
            raise ValueError(f"expected a single batch item, got shape {arr.shape}")
        arr = arr[0]
    if arr.ndim != expected_ndim:
        raise ValueError(f"expected {expected_ndim}D state array, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _plain_yaml(value: Any) -> Any:
    try:
        import numpy as _np
    except Exception:  # pragma: no cover
        _np = None

    if isinstance(value, Mapping):
        return {str(k): _plain_yaml(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_yaml(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if _np is not None and isinstance(value, _np.generic):
        return value.item()
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value
