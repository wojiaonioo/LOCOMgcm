"""Spatial strata provider for the verification engine.

This module no longer drives a metrics group of its own — the old
``build_spatial_decomposition`` (O(vars x sampled-nodes x ~33 bins) Python
reductions + per-node quantile sorts, the multi-minute cost on large grids) was
replaced by :mod:`verification_driver`, which reduces the whole field in one
vectorized xskillscore pass and selects strata by zeroing the weight field.

What remains here is purely the *environment* + *bin masks* that the driver
consumes: the wet geometry (``_environment``) and the depth / coast-distance /
OBC-distance / slope / region / hFac-class masks (``_dimension_masks``), plus
their labels (``_bin_definitions``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import MetricsContext


DEPTH_BINS = [
    (0.0, 50.0, "[0,50) m"),
    (50.0, 200.0, "[50,200) m"),
    (200.0, 1000.0, "[200,1000) m"),
    (1000.0, np.inf, "[1000,inf) m"),
]
COAST_BINS = [
    (0.0, 1.5, "0-1 cells"),
    (1.5, 3.5, "1-3 cells"),
    (3.5, 5.5, "3-5 cells"),
    (5.5, 10.5, "5-10 cells"),
    (10.5, np.inf, ">10 cells"),
]
OBC_BINS = [
    (0.0, 3.5, "0-3 cells"),
    (3.5, 7.5, "3-7 cells"),
    (7.5, 15.5, "7-15 cells"),
    (15.5, 30.5, "15-30 cells"),
    (30.5, np.inf, ">30 cells"),
]
SLOPE_LABELS = ["q0-q50", "q50-q75", "q75-q90", "q90-q99", "q99-q100"]
HFAC_CLASSES = [
    ("full", 0.999, np.inf),
    ("partial_near_full", 0.7, 0.999),
    ("partial_mid", 0.3, 0.7),
    ("thin_partial", 0.0, 0.3),
    ("dry_adjacent", -np.inf, 0.0),
]


@dataclass
class Environment:
    wet2d: np.ndarray
    wet3d: np.ndarray
    hfac: np.ndarray
    bathy: np.ndarray
    dist_coast: np.ndarray
    dist_obc: np.ndarray
    slope: np.ndarray
    slope_edges: list[float]
    regions: dict[str, np.ndarray]


def _environment(ctx: MetricsContext) -> Environment:
    hfac = getattr(ctx.grid, "hfac", None)
    if hfac is None:
        hfac_arr = np.ones((ctx.grid.nz, ctx.grid.ny, ctx.grid.nx), dtype=np.float64)
    else:
        hfac_arr = np.asarray(hfac, dtype=np.float64)
    wet3d = hfac_arr > 0.0
    wet2d = np.any(wet3d, axis=0)
    dz = _layer_thickness(ctx)
    bathy = np.sum(hfac_arr * dz[:, None, None], axis=0)
    dist_coast = _distance_to_land(wet2d)
    yy, xx = np.indices(wet2d.shape)
    dist_obc = np.minimum.reduce([yy, xx, wet2d.shape[0] - 1 - yy, wet2d.shape[1] - 1 - xx]).astype(np.float64)
    dist_obc = np.where(wet2d, dist_obc, 0.0)
    slope = _slope(bathy)
    slope_vals = slope[wet2d & np.isfinite(slope)]
    if slope_vals.size:
        q = np.quantile(slope_vals, [0.0, 0.5, 0.75, 0.9, 0.99, 1.0])
        slope_edges = [float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(q[5]) + 1.0e-12]
    else:
        slope_edges = [0.0, 1.0, 2.0, 3.0, 4.0, np.inf]
    regions = _regions(wet2d, bathy, dist_coast, dist_obc, slope, slope_edges)
    return Environment(wet2d, wet3d, hfac_arr, bathy, dist_coast, dist_obc, slope, slope_edges, regions)


def _dimension_masks(env: Environment, sample_shape: tuple[int, ...]) -> dict[str, dict[str, np.ndarray]]:
    is3d = len(sample_shape) == 3
    def lift(mask2d: np.ndarray) -> np.ndarray:
        return np.broadcast_to(mask2d, sample_shape) if is3d else mask2d

    masks: dict[str, dict[str, np.ndarray]] = {
        "depth_bin": {
            label: lift((env.bathy >= lo) & (env.bathy < hi) & env.wet2d)
            for lo, hi, label in DEPTH_BINS
        },
        "coast_distance": {
            label: lift((env.dist_coast >= lo) & (env.dist_coast < hi) & env.wet2d)
            for lo, hi, label in COAST_BINS
        },
        "obc_distance": {
            label: lift((env.dist_obc >= lo) & (env.dist_obc < hi) & env.wet2d)
            for lo, hi, label in OBC_BINS
        },
        "slope_bin": {
            label: lift((env.slope >= lo) & (env.slope < hi) & env.wet2d)
            for label, lo, hi in zip(SLOPE_LABELS, env.slope_edges[:-1], env.slope_edges[1:])
        },
        "region_class": {
            label: lift(mask)
            for label, mask in env.regions.items()
        },
    }
    if is3d:
        masks["hfac_class"] = {
            label: (env.hfac <= 0.0 if label == "dry_adjacent" else (_range_mask(env.hfac, lo, hi) & env.wet3d))
            for label, lo, hi in HFAC_CLASSES
        }
    else:
        h2d = env.hfac[0] if env.hfac.ndim == 3 else np.ones_like(env.wet2d, dtype=float)
        masks["hfac_class"] = {
            label: (h2d <= 0.0 if label == "dry_adjacent" else (_range_mask(h2d, lo, hi) & env.wet2d))
            for label, lo, hi in HFAC_CLASSES
        }
    return masks


def _bin_definitions(env: Environment) -> dict[str, list[str]]:
    return {
        "depth_bin_m": [label for _, _, label in DEPTH_BINS],
        "coast_distance_cells": [label for _, _, label in COAST_BINS],
        "obc_distance_cells": [label for _, _, label in OBC_BINS],
        "hfac_class": [label for label, _, _ in HFAC_CLASSES],
        "slope_bin": SLOPE_LABELS,
        "region_class": list(env.regions),
    }


def _layer_thickness(ctx: MetricsContext) -> np.ndarray:
    vertical = (ctx.metadata.get("grid", {}) or {}).get("vertical")
    if vertical and "layer_thicknesses_m" in vertical:
        dz = np.asarray(vertical["layer_thicknesses_m"], dtype=np.float64)
        if dz.ndim != 1 or dz.size != int(ctx.grid.nz):
            raise ValueError("metadata grid.vertical.layer_thicknesses_m must be 1-D with length nz")
        if not np.isfinite(dz).all() or not (dz > 0.0).all():
            raise ValueError("metadata grid.vertical.layer_thicknesses_m must be finite and positive")
        return dz
    depth = np.asarray(getattr(ctx.grid, "depth", np.arange(ctx.grid.nz)), dtype=np.float64)
    if depth.size <= 1:
        return np.ones(int(ctx.grid.nz), dtype=np.float64)
    edges = np.empty(depth.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    edges[0] = max(0.0, depth[0] - (edges[1] - depth[0]))
    edges[-1] = depth[-1] + (depth[-1] - edges[-2])
    return np.maximum(np.diff(edges), 1.0e-12)


def _distance_to_land(wet2d: np.ndarray) -> np.ndarray:
    if wet2d.all():
        return np.full(wet2d.shape, float(max(wet2d.shape)), dtype=np.float64)
    try:
        from scipy.ndimage import distance_transform_edt

        return np.where(wet2d, distance_transform_edt(wet2d), 0.0)
    except Exception:
        return np.where(wet2d, 1.0, 0.0)


def _slope(bathy: np.ndarray) -> np.ndarray:
    bathy = bathy.astype(np.float64)
    if bathy.shape[0] < 2 and bathy.shape[1] < 2:
        return np.zeros_like(bathy)
    if bathy.shape[0] < 2:
        gx = np.gradient(bathy, axis=1)
        return np.abs(gx)
    if bathy.shape[1] < 2:
        gy = np.gradient(bathy, axis=0)
        return np.abs(gy)
    gy, gx = np.gradient(bathy)
    return np.sqrt(gx * gx + gy * gy)


def _regions(
    wet2d: np.ndarray,
    bathy: np.ndarray,
    dist_coast: np.ndarray,
    dist_obc: np.ndarray,
    slope: np.ndarray,
    slope_edges: list[float],
) -> dict[str, np.ndarray]:
    open_bdy = wet2d & (dist_obc <= 3.0)
    coast = wet2d & ~open_bdy & (dist_coast <= 3.0)
    steep = wet2d & ~open_bdy & ~coast & (slope >= slope_edges[-2])
    shallow = wet2d & ~open_bdy & ~coast & ~steep & (bathy < 200.0)
    shelf_break = wet2d & ~open_bdy & ~coast & ~shallow & steep
    deep = wet2d & ~open_bdy & ~coast & ~shallow & ~shelf_break
    return {
        "open_bdy_band": open_bdy,
        "island_or_obstacle_adjacent": np.zeros_like(wet2d, dtype=bool),
        "coastal_band": coast,
        "shelf_shallow": shallow,
        "shelf_break_or_slope": shelf_break,
        "open_ocean_deep": deep,
    }


def _range_mask(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    lower = values >= lo if np.isfinite(lo) else np.ones_like(values, dtype=bool)
    upper = values < hi if np.isfinite(hi) else np.ones_like(values, dtype=bool)
    return lower & upper
