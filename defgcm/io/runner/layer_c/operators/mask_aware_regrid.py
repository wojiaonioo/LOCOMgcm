"""Mask-aware horizontal regrid operator.

This operator replaces the (grid_interp + fill_edges[lon/lat]) pair for the
state and boundary pipelines.  It eliminates the "poison water" pathway where
fill_edges[z] first propagates shallow-column surface values downward and then
fill_edges[longitude] diffuses those spurious values horizontally into deep
model cells.

Algorithm (per depth level k, per time frame t):
  1. Identify source wet cells: `finite = np.isfinite(source[k])` after
     mask_land_sentinel (NaN = dry/below-bathymetry).
  2. Bilinear-interpolate using ONLY wet source cells.  We achieve this by
     interpolating source arrays that have NaN in dry cells — xarray.interp
     propagates NaN to any target point whose bilinear stencil contains a NaN.
     After interpolation, some target cells remain NaN (coastal, shelf-edge,
     cells where all 4 bilinear neighbours were dry).
  3. Fill any remaining NaN target cells via inverse-distance weighting of the
     k-nearest source WET points at the SAME level k, using a scipy cKDTree.
     Longitude is periodic: source coordinates are tiled with ±360° wraps so
     the tree search works correctly across the dateline.
  4. If level k has ZERO wet source points, copy values from the nearest
     shallower already-filled level (vertical fallback).
  5. 2-D variables (eta, no z dim): same nearest-fill pass over the 2-D plane.

Key guarantees:
  - Land NEVER contaminates deep ocean cells (same-level donors only).
  - No target-bathymetry input required — fill_all target cells; downstream
    hFac masks land at runtime.
  - Longitude-periodic: donors wrap across 0°/360°.
  - fill_edges[z] MUST be removed from the pipeline before this operator, and
    fill_edges[lon/lat] MUST be removed after: this operator is their
    self-contained replacement for the horizontal dimension.  vertical_interp
    may still run after this op if the source vertical grid differs from the
    target (it will interpolate already-filled columns, which is safe).

Registration name: ``mask_aware_regrid``
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from defgcm.io.runner.layer_a import (
    DataBundle,
    LocationKind,
)

from .._registry import register_operator
from .._types import OperatorCapabilities


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lon_periodic_points(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Build (N,2) point array tiled with ±360 wraps for periodic longitude.

    Returns shape (3*N, 2) — original + lon-360 + lon+360 — so that cKDTree
    queries near the dateline find donors on the other side.
    """
    pts = np.column_stack([lat, lon])
    pts_m = np.column_stack([lat, lon - 360.0])
    pts_p = np.column_stack([lat, lon + 360.0])
    return np.vstack([pts, pts_m, pts_p])


def _fill_level_idw(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    src_vals: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
    nan_mask: np.ndarray,
    *,
    k_donors: int = 4,
) -> np.ndarray:
    """Inverse-distance weighted fill of ``nan_mask`` target cells.

    Parameters
    ----------
    src_lat, src_lon : 1-D arrays of all source coordinates (any order)
    src_vals         : 1-D array matching src_lat/src_lon (float; wet values only)
    tgt_lat, tgt_lon : 1-D arrays of target cell centers (ravelled meshgrid)
    nan_mask         : boolean 1-D mask over tgt; True = needs filling
    k_donors         : number of nearest wet source points to use for IDW

    Returns
    -------
    filled : 1-D array same length as nan_mask; nan_mask positions replaced
    """
    if not nan_mask.any():
        return nan_mask.astype("float64")  # nothing to do — return dummy

    # Build periodic tree from wet source points
    tree_pts = _lon_periodic_points(src_lon, src_lat)
    # Values are repeated 3× to match the tiled tree
    tree_vals = np.tile(src_vals, 3)

    tree = cKDTree(tree_pts)

    # Query positions for NaN target cells.  MUST match the tree's axis order:
    # `_lon_periodic_points` builds the tree as (lat, lon) with the ±360 wrap on
    # the lon column, so the query points are (lat, lon) too.  A (lon, lat) query
    # would swap the axes — the KDTree would then treat a target's longitude as a
    # latitude and vice versa, pulling donors from a geographically wrong location
    # (harmless where lat/lon spans are similar, catastrophic for polar-cap
    # extrapolation where lat~85 has donors only at lat<=80 across all lon).
    tgt_pts = np.column_stack([tgt_lat[nan_mask], tgt_lon[nan_mask]])
    k_use = min(k_donors, len(tree_vals))
    dists, idxs = tree.query(tgt_pts, k=k_use, workers=1)

    if k_use == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    # IDW: weight = 1/d; if a donor is at distance 0 use it exactly
    eps = 1e-10
    weights = 1.0 / (dists + eps)
    weights /= weights.sum(axis=1, keepdims=True)
    filled_vals = (weights * tree_vals[idxs]).sum(axis=1)
    return filled_vals


def _regrid_3d(
    src_ds: xr.Dataset,
    var_name: str,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    *,
    k_donors: int,
    method: str,
) -> np.ndarray:
    """Regrid a single 3-D variable (time, z, lat, lon) to target grid.

    Returns ndarray shape (ntime, nz, ny_tgt, nx_tgt) float64.
    """
    da = src_ds[var_name]
    # Compute on concrete arrays (no dask inside the level loops)
    src_arr = np.asarray(da.values, dtype="float64")  # (time, z, ny_src, nx_src)

    src_lat = np.asarray(da.coords["latitude"].values, dtype="float64")
    src_lon = np.asarray(da.coords["longitude"].values, dtype="float64")

    # Bilinear interp: xarray.interp propagates NaN at dry source cells
    interp_da = da.interp(
        latitude=target_lat,
        longitude=target_lon,
        method=method,
        kwargs={"bounds_error": False, "fill_value": np.nan},
    )
    out_arr = np.asarray(interp_da.values, dtype="float64")  # (time, z, ny_tgt, nx_tgt)

    ntime, nz, ny_tgt, nx_tgt = out_arr.shape
    tgt_lat_2d, tgt_lon_2d = np.meshgrid(target_lat, target_lon, indexing="ij")
    tgt_lat_r = tgt_lat_2d.ravel()
    tgt_lon_r = tgt_lon_2d.ravel()

    # --- per-level fill ---
    last_filled: np.ndarray | None = None  # for vertical fallback

    for t in range(ntime):
        for k in range(nz):
            level_src = src_arr[t, k]  # (ny_src, nx_src)
            level_out = out_arr[t, k]  # (ny_tgt, nx_tgt) — may contain NaN

            # Wet source mask at this level
            wet_src = np.isfinite(level_src)
            n_wet_src = wet_src.sum()

            if n_wet_src == 0:
                # Vertical fallback: copy from nearest shallower filled level
                if last_filled is not None:
                    out_arr[t, k] = last_filled
                else:
                    # No shallower level yet — fill with 0 (will be masked by hFac)
                    out_arr[t, k] = 0.0
                continue

            nan_mask = ~np.isfinite(level_out)  # target cells needing IDW fill

            if nan_mask.any():
                wet_lat = src_lat[wet_src.any(axis=1)]
                # Build proper (N,) arrays of wet point coordinates + values
                src_lat_2d, src_lon_2d = np.meshgrid(src_lat, src_lon, indexing="ij")
                flat_lat = src_lat_2d.ravel()
                flat_lon = src_lon_2d.ravel()
                flat_wet = wet_src.ravel()
                flat_val = level_src.ravel()

                wet_lat_pts = flat_lat[flat_wet]
                wet_lon_pts = flat_lon[flat_wet]
                wet_val_pts = flat_val[flat_wet]

                filled = _fill_level_idw(
                    wet_lat_pts,
                    wet_lon_pts,
                    wet_val_pts,
                    tgt_lat_r,
                    tgt_lon_r,
                    nan_mask.ravel(),
                    k_donors=k_donors,
                )
                out_arr[t, k][nan_mask] = filled

            last_filled = out_arr[t, k].copy()

    return out_arr


def _regrid_2d(
    src_ds: xr.Dataset,
    var_name: str,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    *,
    k_donors: int,
    method: str,
) -> np.ndarray:
    """Regrid a 2-D surface variable (time, lat, lon) → (time, ny_tgt, nx_tgt)."""
    da = src_ds[var_name]
    src_arr = np.asarray(da.values, dtype="float64")  # (time, ny_src, nx_src)
    src_lat = np.asarray(da.coords["latitude"].values, dtype="float64")
    src_lon = np.asarray(da.coords["longitude"].values, dtype="float64")

    interp_da = da.interp(
        latitude=target_lat,
        longitude=target_lon,
        method=method,
        kwargs={"bounds_error": False, "fill_value": np.nan},
    )
    out_arr = np.asarray(interp_da.values, dtype="float64")  # (time, ny_tgt, nx_tgt)

    ntime, ny_tgt, nx_tgt = out_arr.shape
    tgt_lat_2d, tgt_lon_2d = np.meshgrid(target_lat, target_lon, indexing="ij")
    tgt_lat_r = tgt_lat_2d.ravel()
    tgt_lon_r = tgt_lon_2d.ravel()

    src_lat_2d, src_lon_2d = np.meshgrid(src_lat, src_lon, indexing="ij")
    flat_lat = src_lat_2d.ravel()
    flat_lon = src_lon_2d.ravel()

    for t in range(ntime):
        level_src = src_arr[t]
        level_out = out_arr[t]
        flat_src = level_src.ravel()
        wet = np.isfinite(flat_src)
        if not wet.any():
            # All dry — fill 0 (will be masked by hFac at runtime)
            out_arr[t] = 0.0
            continue
        nan_mask = ~np.isfinite(level_out.ravel())
        if nan_mask.any():
            filled = _fill_level_idw(
                flat_lat[wet],
                flat_lon[wet],
                flat_src[wet],
                tgt_lat_r,
                tgt_lon_r,
                nan_mask,
                k_donors=k_donors,
            )
            out_arr[t][~np.isfinite(level_out)] = filled

    return out_arr


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


@register_operator("mask_aware_regrid")
class MaskAwareRegridOperator:
    """Land/sea-mask-aware horizontal regrid to target_grid.

    Replaces the (grid_interp + fill_edges[lon/lat]) pair in the state and
    boundary pipelines.  See module docstring for full algorithm description.

    Parameters
    ----------
    target_grid : model_grid object or dict with latitude/longitude keys
        Same duck-typed interface as GridInterpolator.
    method : str, default "linear"
        xarray.interp method for the bilinear pass.  "linear" = bilinear.
    k_donors : int, default 4
        Number of nearest wet source points for the IDW fill pass.
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE,
                 LocationKind.BOUNDARY_TIMESERIES),
        produces=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE,
                  LocationKind.BOUNDARY_TIMESERIES),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(
        self,
        *,
        target_grid: Any = None,
        method: str = "linear",
        k_donors: int = 4,
        **_: Any,
    ) -> None:
        self._target = target_grid
        method_aliases = {"bilinear": "linear", "trilinear": "linear"}
        self._method = method_aliases.get(method, method)
        self._k_donors = int(k_donors)

    def __call__(self, bundle: DataBundle) -> DataBundle:
        if self._target is None:
            return bundle  # pass-through (no target grid configured)

        g = self._target
        if hasattr(g, "center_lat"):
            # Identical coordinate derivation as GridInterpolator to ensure
            # output grids are bitwise-identical to the grid_interp operator.
            lat_off = np.arange(g.ny, dtype="float64") - 0.5 * (g.ny - 1)
            lon_off = np.arange(g.nx, dtype="float64") - 0.5 * (g.nx - 1)
            target_lat = g.center_lat + lat_off * g.resolution_deg
            target_lon = g.center_lon + lon_off * g.resolution_deg
            grid_id = g.grid_id
        else:
            target_lat = np.asarray(g["latitude"])
            target_lon = np.asarray(g["longitude"])
            grid_id = g.get("grid_id", "custom")

        # Materialise any dask arrays: cKDTree / scipy cannot operate on lazy
        # arrays; the bilinear interp pass will rechunk on xarray level.
        ds = bundle.data
        # Force eager evaluation if backed by dask
        try:
            import dask  # noqa: F401
            ds = ds.compute()
        except ImportError:
            pass

        new_data_vars: dict = {}
        for var in bundle.meta.variables:
            vname = var.name
            if vname not in ds.data_vars:
                continue

            arr_dims = ds[vname].dims
            has_z = "z" in arr_dims

            if has_z:
                filled = _regrid_3d(
                    ds, vname, target_lat, target_lon,
                    k_donors=self._k_donors,
                    method=self._method,
                )
                out_dims = ("time", "z", "latitude", "longitude")
            else:
                filled = _regrid_2d(
                    ds, vname, target_lat, target_lon,
                    k_donors=self._k_donors,
                    method=self._method,
                )
                out_dims = ("time", "latitude", "longitude")

            # Cast back to declared dtype
            td = np.dtype(var.dtype)
            if filled.dtype != td:
                filled = filled.astype(td)
            new_data_vars[vname] = (out_dims, filled)

        # Build output coords
        z_vals = ds["z"].values if "z" in ds.coords else None
        time_vals = ds["time"].values if "time" in ds.coords else np.array([0])
        coords: dict = {
            "time": time_vals,
            "latitude": target_lat,
            "longitude": target_lon,
        }
        if z_vals is not None:
            coords["z"] = z_vals

        new_ds = xr.Dataset(data_vars=new_data_vars, coords=coords)
        new_meta = dataclasses.replace(bundle.meta, target_grid_id=grid_id)
        return DataBundle(data=new_ds, meta=new_meta)
