"""Simple (algorithmic, no sign conversion) Layer C operators.

Operators in this module:
  - subset_vars: select a subset of variables
  - subset_boundary: extract 4-side boundary strips
  - subset_to_domain: clip to target_grid lat/lon extent
  - grid_interp: bilinear regrid via xarray.interp
  - time_interp: resample time dim via xarray.interp
  - bathy_clamp: clamp bathymetry to [depth_min, depth_max]
  - quality_filter: filter scattered obs by qc flag

All Profile M; bugs are loud (shape mismatch / value error).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import dask.array as dka
import numpy as np
import xarray as xr

from defgcm.io.runner.layer_a import (
    BundleMeta,
    ConventionTag,
    DataBundle,
    LocationKind,
    VariableSpec,
)

from .._registry import register_operator
from .._types import OperatorCapabilities


def _replace_data(bundle: DataBundle, new_data: xr.Dataset, new_specs: tuple[VariableSpec, ...]) -> DataBundle:
    """Helper: build a new DataBundle preserving meta (except variables tuple)."""
    new_meta = dataclasses.replace(bundle.meta, variables=new_specs)
    return DataBundle(data=new_data, meta=new_meta)


# ============================================================
# subset_vars
# ============================================================


@register_operator("subset_vars")
class SubsetVarsOperator:
    """Keep only specified variables (drop others)."""

    capabilities = OperatorCapabilities(
        accepts=tuple(LocationKind),
        produces=tuple(LocationKind),
        convention_in=None,
        convention_out=None,
        preserves_variables=False,
    )

    def __init__(self, *, vars: tuple[str, ...] | list[str], **_: Any) -> None:
        self._keep = tuple(vars)

    def __call__(self, bundle: DataBundle) -> DataBundle:
        keep_set = set(self._keep)
        present = [v for v in bundle.meta.variables if v.name in keep_set]
        if not present:
            raise ValueError(
                f"subset_vars: no variables matched {self._keep}; "
                f"bundle has {sorted(v.name for v in bundle.meta.variables)}"
            )
        new_data = bundle.data[[v.name for v in present]]
        return _replace_data(bundle, new_data, tuple(present))


# ============================================================
# subset_boundary (gridded → BOUNDARY_TIMESERIES)
# ============================================================


@register_operator("subset_boundary")
class SubsetBoundaryOperator:
    """Extract N/E/S/W boundary strips from a gridded ocean bundle.

    Note: ``width`` is accepted for config back-compat but is currently
    IGNORED — each side is always the single outermost boundary cell. Multi-cell
    strips are not implemented here.
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE),
        produces=(LocationKind.BOUNDARY_TIMESERIES,),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(
        self,
        *,
        sides: tuple[str, ...] | list[str] = ("N", "E", "S", "W"),
        width: int = 1,
        **_: Any,
    ) -> None:
        self._sides = tuple(sides)
        self._width = int(width)

    def __call__(self, bundle: DataBundle) -> DataBundle:
        sides = self._sides
        # N/S boundaries run along longitude (nx); E/W along latitude (ny). For
        # non-square domains the per-side length differs, so the shared `along`
        # dim is max(nx, ny) and each side fills only its valid prefix
        # [:along_valid]; the tail is zero-padded. The `along_valid` coord +
        # `valid_along` extras let downstream readers (BoundaryBundleSource,
        # mitgcm_obc_binary) slice each side to its true length. For square
        # domains all sides == nx, bit-identical to the old min(nx,ny) form.
        nlat = int(bundle.data.sizes.get("latitude", 1))   # ny
        nlon = int(bundle.data.sizes.get("longitude", 1))  # nx
        side_len = {"N": nlon, "S": nlon, "E": nlat, "W": nlat}
        n_along = max(nlat, nlon)

        def _pad(strip, xp):
            # strip is (..., side_len); zero-pad the last axis to n_along.
            # Bit-identical to the old np.zeros+assign for the numpy path; the
            # dask path (xp=dka) keeps it lazy so the full interior grid is never
            # materialised — only the 4 edge strips flow downstream.
            cur = strip.shape[-1]
            if cur >= n_along:
                return strip
            pad_width = [(0, 0)] * strip.ndim
            pad_width[-1] = (0, n_along - cur)
            return xp.pad(strip, pad_width, mode="constant", constant_values=0.0)

        new_data_vars: dict[str, tuple] = {}
        new_specs: list[VariableSpec] = []
        for var in bundle.meta.variables:
            # `.data` preserves laziness (dask array if the upstream pipeline is
            # dask-backed; plain numpy otherwise). The old `np.asarray(da.values)`
            # force-computed the WHOLE (time, z, lat, lon) interior grid here,
            # which is the stage1a boundary OOM. Edge slicing + zero-pad + stack
            # are value-preserving, so numpy stays bit-identical and dask defers
            # the compute to the streaming to_zarr.
            arr = bundle.data[var.name].data
            xp = dka if hasattr(arr, "chunks") else np
            has_z = "z" in var.dims
            strips = []
            for side in sides:
                if has_z:
                    if side == "N":
                        s = arr[:, :, -1, :]
                    elif side == "S":
                        s = arr[:, :, 0, :]
                    elif side == "E":
                        s = arr[:, :, :, -1]
                    elif side == "W":
                        s = arr[:, :, :, 0]
                    else:
                        continue
                else:
                    if side == "N":
                        s = arr[:, -1, :]
                    elif side == "S":
                        s = arr[:, 0, :]
                    elif side == "E":
                        s = arr[:, :, -1]
                    elif side == "W":
                        s = arr[:, :, 0]
                    else:
                        continue
                strips.append(_pad(s, xp))
            stacked = xp.stack(strips, axis=1)
            dims = ("time", "side", "z", "along") if has_z else ("time", "side", "along")
            new_data_vars[var.name] = (dims, stacked)
            new_specs.append(dataclasses.replace(var, dims=dims))

        coords: dict[str, Any] = {
            "time": bundle.data["time"].values if "time" in bundle.data.coords else np.array([0]),
            "side": np.array(list(sides)),
            "along": np.arange(n_along),
            "along_valid": ("side", np.array([side_len[s] for s in sides])),
            "latitude": bundle.data["latitude"].values,
            "longitude": bundle.data["longitude"].values,
        }
        if "z" in bundle.data.coords:
            coords["z"] = bundle.data["z"].values

        new_ds = xr.Dataset(data_vars=new_data_vars, coords=coords)
        new_extras = dict(getattr(bundle.meta, "extras", {}) or {})
        new_extras["valid_along"] = {s: side_len[s] for s in sides}
        new_meta = dataclasses.replace(
            bundle.meta,
            location_kind=LocationKind.BOUNDARY_TIMESERIES,
            variables=tuple(new_specs),
            extras=new_extras,
        )
        return DataBundle(data=new_ds, meta=new_meta)


# ============================================================
# subset_to_domain
# ============================================================


@register_operator("subset_to_domain")
class SubsetToDomainOperator:
    """Clip lat/lon to target_grid extent. Pass-through if no target_grid."""

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE, LocationKind.MESH),
        produces=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE, LocationKind.MESH),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(
        self,
        *,
        lat_min: float | None = None,
        lat_max: float | None = None,
        lon_min: float | None = None,
        lon_max: float | None = None,
        **_: Any,
    ) -> None:
        self._lat_min = lat_min
        self._lat_max = lat_max
        self._lon_min = lon_min
        self._lon_max = lon_max

    def __call__(self, bundle: DataBundle) -> DataBundle:
        ds = bundle.data
        if self._lat_min is not None or self._lat_max is not None:
            ds = ds.sel(latitude=slice(self._lat_min, self._lat_max))
        if self._lon_min is not None or self._lon_max is not None:
            ds = ds.sel(longitude=slice(self._lon_min, self._lon_max))
        # Variables unchanged (preserves_variables=True)
        return _replace_data(bundle, ds, bundle.meta.variables)


# ============================================================
# grid_interp
# ============================================================


@register_operator("grid_interp")
class GridInterpolator:
    """Re-grid to target_grid lat/lon using xarray.interp (linear bilinear)."""

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE,
                 LocationKind.MESH, LocationKind.BOUNDARY_TIMESERIES),
        produces=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE,
                  LocationKind.MESH, LocationKind.BOUNDARY_TIMESERIES),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(
        self,
        *,
        target_grid: Any = None,
        method: str = "linear",
        **_: Any,
    ) -> None:
        # target_grid may be a Grid frozen-dataclass or a dict; we duck-type.
        self._target = target_grid
        # xarray.interp accepts "linear" / "nearest" / "cubic" etc.
        # Map common aliases (bilinear → linear for 2D N-D interp).
        method_aliases = {"bilinear": "linear", "trilinear": "linear"}
        self._method = method_aliases.get(method, method)

    def __call__(self, bundle: DataBundle) -> DataBundle:
        if self._target is None:
            return bundle  # pass-through
        g = self._target
        if hasattr(g, "center_lat"):
            # Cell-center coordinates — MUST match legacy
            # defgcm/io/hycom.py:_target_coordinate exactly:
            #   offsets = arange(count) - 0.5*(count-1)
            #   coord   = center + offsets * spacing
            # np.linspace(center ± count*res/2, count) is WRONG: spacing
            # becomes count*res/(count-1) — ~2.9% too wide for count=36,
            # shifting every interpolation sample and skewing all regridded
            # fields (state / forcing / boundary / mesh).
            lat_off = np.arange(g.ny, dtype="float64") - 0.5 * (g.ny - 1)
            lon_off = np.arange(g.nx, dtype="float64") - 0.5 * (g.nx - 1)
            target_lat = g.center_lat + lat_off * g.resolution_deg
            target_lon = g.center_lon + lon_off * g.resolution_deg
            grid_id = g.grid_id
        else:
            target_lat = np.asarray(g["latitude"])
            target_lon = np.asarray(g["longitude"])
            grid_id = g.get("grid_id", "custom")

        interp_kwargs = {"latitude": target_lat, "longitude": target_lon}
        new_ds = bundle.data.interp(method=self._method, **interp_kwargs)
        # dtype preservation: xarray.interp may promote; cast back per VariableSpec
        for var in bundle.meta.variables:
            if var.name in new_ds.data_vars:
                target_dtype = np.dtype(var.dtype)
                if new_ds[var.name].dtype != target_dtype:
                    new_ds[var.name] = new_ds[var.name].astype(target_dtype)
        new_meta = dataclasses.replace(bundle.meta, target_grid_id=grid_id)
        return DataBundle(data=new_ds, meta=new_meta)


# ============================================================
# time_interp
# ============================================================


@register_operator("time_interp")
class TimeInterpolator:
    """Resample time dim to a target dt (hours) via xarray.interp."""

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE, LocationKind.BOUNDARY_TIMESERIES),
        produces=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE, LocationKind.BOUNDARY_TIMESERIES),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(
        self,
        *,
        target_dt_hours: float | None = None,
        target_times: Any = None,
        method: str = "linear",
        **_: Any,
    ) -> None:
        self._target_dt_hours = target_dt_hours
        self._target_times = target_times
        self._method = method

    def __call__(self, bundle: DataBundle) -> DataBundle:
        if "time" not in bundle.data.coords:
            return bundle
        if self._target_times is not None:
            new_times = np.asarray(self._target_times)
        elif self._target_dt_hours is not None:
            t_vals = bundle.data["time"].values
            if len(t_vals) < 2:
                return bundle
            t_start, t_end = t_vals[0], t_vals[-1]
            dt_ns = int(self._target_dt_hours * 3600 * 1e9)
            new_times = np.arange(t_start, t_end + np.timedelta64(1, "ns"), np.timedelta64(dt_ns, "ns"))
        else:
            return bundle
        new_ds = bundle.data.interp(time=new_times, method=self._method)
        for var in bundle.meta.variables:
            if var.name in new_ds.data_vars:
                target_dtype = np.dtype(var.dtype)
                if new_ds[var.name].dtype != target_dtype:
                    new_ds[var.name] = new_ds[var.name].astype(target_dtype)
        return _replace_data(bundle, new_ds, bundle.meta.variables)


# ============================================================
# bathy_clamp
# ============================================================


@register_operator("bathy_clamp")
class BathyClampOperator:
    """Clamp bathymetry to [depth_min, depth_max].

    Operates on the 'bathymetry' variable. Records clamp event in extras.
    When ``mask_land=True``, the lower bound is applied only to positive-depth
    ocean cells so land cells encoded as zero stay dry.
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.MESH,),
        produces=(LocationKind.MESH,),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(
        self,
        *,
        depth_max: float = float("inf"),
        depth_min: float = -float("inf"),
        mask_land: bool = False,
        **_: Any,
    ) -> None:
        self._depth_max = depth_max
        self._depth_min = depth_min
        self._mask_land = mask_land

    def __call__(self, bundle: DataBundle) -> DataBundle:
        if "bathymetry" not in bundle.data.data_vars:
            return bundle
        b = bundle.data["bathymetry"]
        values = np.asarray(b.values)
        if self._mask_land:
            ocean = values > 0.0
            clamped = values.copy()
            clamped[ocean] = np.clip(
                values[ocean],
                self._depth_min,
                self._depth_max,
            )
        else:
            clamped = np.clip(values, self._depth_min, self._depth_max)
        clamped = clamped.astype(b.dtype)
        new_ds = bundle.data.copy()
        new_ds["bathymetry"] = (b.dims, clamped)
        new_meta = dataclasses.replace(
            bundle.meta,
            extras={
                **dict(bundle.meta.extras),
                "bathy_clamp_event": {
                    "depth_max": self._depth_max,
                    "depth_min": self._depth_min,
                    "mask_land": self._mask_land,
                },
            },
        )
        return DataBundle(data=new_ds, meta=new_meta)


# ============================================================
# quality_filter (scattered obs)
# ============================================================


@register_operator("quality_filter")
class QualityFilterOperator:
    """Filter scattered observations by qc flag (drop where qc < min_qc)."""

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.SCATTERED_PROFILE, LocationKind.SCATTERED_POINT, LocationKind.TRAJECTORY),
        produces=(LocationKind.SCATTERED_PROFILE, LocationKind.SCATTERED_POINT, LocationKind.TRAJECTORY),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(self, *, min_qc: int = 2, qc_var: str = "qc_flag", **_: Any) -> None:
        self._min_qc = min_qc
        self._qc_var = qc_var

    def __call__(self, bundle: DataBundle) -> DataBundle:
        if self._qc_var not in bundle.data.data_vars:
            return bundle  # no qc var; nothing to filter
        qc = bundle.data[self._qc_var]
        mask = qc >= self._min_qc
        new_ds = bundle.data.where(mask, drop=True)
        return _replace_data(bundle, new_ds, bundle.meta.variables)


# ============================================================
# mask_land_sentinel
# ============================================================


@register_operator("mask_land_sentinel")
class MaskLandSentinelOperator:
    """Mask land / sub-bathymetry cells to NaN before interpolation.

    HYCOM stores 0.0 below the local seafloor and on land. Interpolating
    *through* those zeros contaminates near-coast and near-bottom values
    (a HYCOM cell at 0.0 dragging a real 34 PSU salinity toward 0 in the
    vertical interp). Legacy `extract_interpolated_3d_patch` first builds
    `is_ocean = (water_temp != 0) | (salinity != 0)` and `.where`s the
    subset, so the subsequent ffill/bfill + interp only ever see real
    ocean values.

    The ocean mask is computed from `flag_vars` and applied only to
    variables sharing the flag vars' dims (3D ocean fields). 2D fields
    (e.g. eta) are left untouched — legacy treats them separately.
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.BOUNDARY_TIMESERIES),
        produces=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.BOUNDARY_TIMESERIES),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(
        self,
        *,
        flag_vars: tuple[str, ...] | list[str] = ("temperature", "salinity"),
        **_: Any,
    ) -> None:
        self._flag_vars = tuple(flag_vars)

    def __call__(self, bundle: DataBundle) -> DataBundle:
        ds = bundle.data
        present = [v for v in self._flag_vars if v in ds.data_vars]
        if not present:
            return bundle
        is_ocean = None
        for v in present:
            cond = ds[v] != 0.0
            is_ocean = cond if is_ocean is None else (is_ocean | cond)
        flag_dims = set(ds[present[0]].dims)
        new_ds = ds.copy()
        for var in bundle.meta.variables:
            if var.name not in ds.data_vars:
                continue
            if set(ds[var.name].dims) != flag_dims:
                continue  # 2D field — legacy interpolates it separately
            masked = ds[var.name].where(is_ocean)
            td = np.dtype(var.dtype)
            if masked.dtype != td:
                masked = masked.astype(td)
            new_ds[var.name] = masked
        return _replace_data(bundle, new_ds, bundle.meta.variables)


# ============================================================
# fill_edges
# ============================================================


@register_operator("fill_edges")
class FillEdgesOperator:
    """Forward-fill then back-fill NaN gaps along the given dims.

    Replicates legacy `extract_interpolated_3d_patch` gap-filling:
      - before vertical interp:  `.ffill("depth").bfill("depth")`
      - after horizontal interp: `.ffill("lon").bfill("lon")
                                   .ffill("lat").bfill("lat")`
    `fillna` (optional) sets any residual NaN (whole-slab gaps) to a
    constant — legacy ends with `nan_to_num(nan=0.0)`.
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE,
                 LocationKind.BOUNDARY_TIMESERIES, LocationKind.MESH),
        produces=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE,
                  LocationKind.BOUNDARY_TIMESERIES, LocationKind.MESH),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(
        self,
        *,
        dims: tuple[str, ...] | list[str],
        fillna: float | None = None,
        **_: Any,
    ) -> None:
        self._dims = tuple(dims)
        self._fillna = fillna

    def __call__(self, bundle: DataBundle) -> DataBundle:
        ds = bundle.data
        for d in self._dims:
            if d in ds.dims:
                ds = ds.ffill(dim=d).bfill(dim=d)
        if self._fillna is not None:
            ds = ds.fillna(self._fillna)
        # dtype preservation (ffill/bfill/fillna keep float but be safe)
        for var in bundle.meta.variables:
            if var.name in ds.data_vars:
                td = np.dtype(var.dtype)
                if ds[var.name].dtype != td:
                    ds[var.name] = ds[var.name].astype(td)
        return _replace_data(bundle, ds, bundle.meta.variables)


@register_operator("wind_to_stress")
class WindToStressOperator:
    """Convert 10-m wind (wind_u, wind_v) → surface wind stress (tau_x, tau_y).

    Large & Pond (1981) bulk drag — identical to
    `defgcm/io/era5.py:wind_speed_to_stress` so MIT EXF (direct-stress
    mode) and the LOCOMgcm side apply point-for-point identical momentum
    forcing. Without this, MIT bulk-formula stress would differ from LOCOMgcm's
    Large&Pond stress and break LOCOMgcm↔MIT alignment.

      rho_air = 1.225 kg/m³
      Cd = 1.2e-3                       (|U| < 11 m/s)
         = (0.49 + 0.065·|U|)·1e-3      (|U| ≥ 11 m/s)
      tau = rho_air · Cd · |U| · U
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.GRIDDED_2D_SURFACE,),
        produces=(LocationKind.GRIDDED_2D_SURFACE,),
        convention_in=None,
        convention_out=None,
        preserves_variables=False,  # wind_u/v → tau_x/tau_y
    )

    _RHO_AIR = 1.225

    def __init__(self, *, rho_air: float = 1.225, **_: Any) -> None:
        self._rho_air = float(rho_air)

    def __call__(self, bundle: DataBundle) -> DataBundle:
        if "wind_u" not in bundle.data or "wind_v" not in bundle.data:
            raise ValueError(
                "wind_to_stress: bundle must contain wind_u + wind_v "
                f"(have {list(bundle.data.data_vars)})"
            )
        # Delegate to defgcm.dynamics.bulk_formulae.wind_speed_to_stress
        # rather than re-implementing
        # Large&Pond — a numpy re-implementation differs from the torch legacy
        # path at ~1e-16 (elementwise op ordering), which breaks bitwise parity
        # with legacy ustress/vstress.bin.
        import torch
        from defgcm.dynamics.bulk_formulae import wind_speed_to_stress
        u10 = np.asarray(bundle.data["wind_u"].values, dtype="float64")
        v10 = np.asarray(bundle.data["wind_v"].values, dtype="float64")
        tau_x_t, tau_y_t = wind_speed_to_stress(
            torch.from_numpy(np.ascontiguousarray(u10)),
            torch.from_numpy(np.ascontiguousarray(v10)),
            rho_air=self._rho_air,
        )
        tau_x = tau_x_t.numpy()
        tau_y = tau_y_t.numpy()

        # Preserve dims from wind_u. Emit tau as FLOAT64: legacy era5.py
        # widens the (float32) interpolated wind to float64 before the
        # Large&Pond stress and writes ustress/vstress.bin at float64. The
        # input wind is already widened to float64 above; quantizing tau back
        # to float32 here re-introduced ~3e-8 vs legacy.
        wind_dims = bundle.data["wind_u"].dims
        new_ds = bundle.data.drop_vars(
            [v for v in ("wind_u", "wind_v") if v in bundle.data]
        )
        new_ds = new_ds.assign(
            tau_x=(wind_dims, tau_x.astype("float64")),
            tau_y=(wind_dims, tau_y.astype("float64")),
        )
        new_specs = tuple(
            s for s in bundle.meta.variables if s.name not in ("wind_u", "wind_v")
        ) + (
            VariableSpec(
                name="tau_x", cf_standard_name="surface_downward_eastward_stress",
                units="Pa", dtype="float64", dims=tuple(wind_dims),
            ),
            VariableSpec(
                name="tau_y", cf_standard_name="surface_downward_northward_stress",
                units="Pa", dtype="float64", dims=tuple(wind_dims),
            ),
        )
        return _replace_data(bundle, new_ds, new_specs)


@register_operator("vertical_interp")
class VerticalInterpolator:
    """Interpolate the `z` dimension to the model's layer-center depths.

    grid_interp only regrids lat/lon — source data (HYCOM ~40 native levels)
    keeps its vertical resolution. MITgcm SIZE.h Nr matches the model grid's
    nz, so without this the IC binary has the wrong level count and MITgcm
    silently reads only the top Nr source levels (all warm surface water).

    Target depths = layer-center depths derived from
    `target_grid.vertical_layer_thicknesses`:
        centers[k] = sum(thickness[:k]) + thickness[k]/2
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.BOUNDARY_TIMESERIES),
        produces=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.BOUNDARY_TIMESERIES),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(self, *, target_grid: Any = None, method: str = "linear", **_: Any) -> None:
        self._target = target_grid
        method_aliases = {"bilinear": "linear", "trilinear": "linear"}
        self._method = method_aliases.get(method, method)

    def __call__(self, bundle: DataBundle) -> DataBundle:
        if self._target is None:
            return bundle  # pass-through
        if "z" not in bundle.data.dims:
            return bundle  # no vertical dim — nothing to do
        g = self._target
        vlt = getattr(g, "vertical_layer_thicknesses", None)
        if not vlt:
            return bundle
        vlt = np.asarray(vlt, dtype="float64")
        centers = np.cumsum(vlt) - vlt / 2.0  # layer-center depths

        # xarray.interp along z. Matches legacy extract_interpolated_3d_patch:
        #   subset.interp(depth=target_depths, method="linear",
        #                 kwargs={"fill_value": "extrapolate"})
        # NaN gap-filling along z must happen *before* this op (fill_edges
        # dims:[z]); legacy ffills depth before the interp, not after.
        new_ds = bundle.data.interp(
            z=centers, method=self._method,
            kwargs={"fill_value": "extrapolate"},
        )
        # dtype preservation
        for var in bundle.meta.variables:
            if var.name in new_ds.data_vars:
                td = np.dtype(var.dtype)
                if new_ds[var.name].dtype != td:
                    new_ds[var.name] = new_ds[var.name].astype(td)
        return _replace_data(bundle, new_ds, bundle.meta.variables)
