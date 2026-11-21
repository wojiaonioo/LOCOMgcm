"""Tidal OBCS extraction operator (Layer C).

Interpolates TPXO9 tidal amplitude + Greenwich phase from the staggered
source grids onto the four open-boundary cell-center lines (N/S/E/W) of the
model domain, producing per-constituent boundary amplitude (m/s) and phase
(seconds). Mirrors the boundary OBCS lane's `subset_boundary`, but for the
TIDAL_HARMONICS kind.

The boundary cell-center coordinate construction (np.linspace) and the
bilinear `_interp_along_boundary` kernel are inlined verbatim from the
original TPXO9 extraction script (`extract_for_domain` /
`_interp_along_boundary`) so the lane reproduces it bitwise. CRITICAL: the phase is interpolated in RADIANS and only THEN divided
by omega → seconds, matching the script's exact floating-point order
(interp(rad)/omega, NOT interp(rad/omega)).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import xarray as xr

from locomgcm.io.runner.layer_a import (
    DataBundle,
    LocationKind,
    VariableSpec,
)

from .._registry import register_operator
from .._types import OperatorCapabilities


def _interp_along_boundary(
    field_2d: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    ob_lons: np.ndarray,
    ob_lats: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of a 2D TPXO field to boundary points.

    Verbatim port of extract_tpxo9_tides.py:_interp_along_boundary. TPXO9 data
    layout is [lon, lat] (Fortran order from OTIS).
    """
    from scipy.interpolate import RegularGridInterpolator

    lon_1d = lon_grid
    lat_1d = lat_grid

    # Ensure OB lons match TPXO convention (0-360)
    ob_lons_360 = np.where(ob_lons < 0, ob_lons + 360, ob_lons)

    # field_2d is [lon, lat] → RegularGridInterpolator wants (lon, lat) axes
    interp = RegularGridInterpolator(
        (lon_1d, lat_1d), field_2d,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    points = np.column_stack([ob_lons_360, ob_lats])
    return interp(points)


@register_operator("extract_tidal_obcs")
class ExtractTidalObcsOperator:
    """TIDAL_HARMONICS (staggered grids) → per-side boundary amplitude/phase.

    Output dims: (constituent, side, along) for u_tide_amplitude,
    u_tide_phase, v_tide_amplitude, v_tide_phase. Amplitude in m/s, phase in
    SECONDS (Greenwich, no V0). `side` order is N, S, E, W.
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.TIDAL_HARMONICS,),
        produces=(LocationKind.TIDAL_HARMONICS,),
        convention_in=None,
        convention_out=None,
        # This operator does NOT preserve variables: it reshapes to per-side
        # (constituent, side, along) amplitude/phase and converts phase from
        # radians to seconds, so the output variable set/dims differ from input.
        preserves_variables=False,
    )

    _SIDES = ("N", "S", "E", "W")

    def __init__(self, *, target_grid: Any = None, **_: Any) -> None:
        # target_grid is resolved to a Grid frozen-dataclass by the framework
        # (same injection as GridInterpolator / VerticalInterpolator).
        self._target = target_grid

    def __call__(self, bundle: DataBundle) -> DataBundle:
        g = self._target
        if g is None or not hasattr(g, "center_lat"):
            raise ValueError(
                "extract_tidal_obcs: requires a resolved target_grid "
                "(set `target_grid: model_grid` in the operator config)"
            )
        nx, ny = int(g.nx), int(g.ny)
        res = float(g.resolution_deg)

        # Domain boundary cell-center coordinate lines — verbatim from
        # extract_tpxo9_tides.py:extract_for_domain. np.linspace (NOT the
        # arange offset form used by grid_interp) — must match bitwise.
        half_x = nx * res / 2
        half_y = ny * res / 2
        lat_s = g.center_lat - half_y
        lat_n = g.center_lat + half_y
        lon_w = g.center_lon - half_x
        lon_e = g.center_lon + half_x
        lons_ns = np.linspace(lon_w + res / 2, lon_e - res / 2, nx)
        lats_ew = np.linspace(lat_s + res / 2, lat_n - res / 2, ny)
        boundaries = {
            "N": (lons_ns, np.full(nx, lat_n)),
            "S": (lons_ns, np.full(nx, lat_s)),
            "E": (np.full(ny, lon_e), lats_ew),
            "W": (np.full(ny, lon_w), lats_ew),
        }

        data = bundle.data
        constituents = [str(c) for c in data["constituent"].values]
        periods = np.asarray(data["period"].values, dtype="float64")
        lon_u = np.asarray(data["lon_u"].values)
        lat_u = np.asarray(data["lat_u"].values)
        lon_v = np.asarray(data["lon_v"].values)
        lat_v = np.asarray(data["lat_v"].values)

        # N/S boundaries run along x (nx points), E/W along y (ny points). For
        # non-square domains the `along` dim is max(nx, ny) and each side fills
        # only its valid prefix [:along_valid]; the tail is zero-padded. The
        # `along_valid` coord lets downstream consumers (mitgcm_tidal_binary,
        # model wiring) slice each side to its true length. For square domains
        # there is no padding (all sides == nx), bit-identical to before.
        side_valid = {"N": nx, "S": nx, "E": ny, "W": ny}
        n_along = max(nx, ny)

        u_amp_out = np.zeros((len(constituents), len(self._SIDES), n_along), dtype="float64")
        u_ph_out = np.zeros_like(u_amp_out)
        v_amp_out = np.zeros_like(u_amp_out)
        v_ph_out = np.zeros_like(u_amp_out)

        for c_idx, _cname in enumerate(constituents):
            omega = 2.0 * np.pi / float(periods[c_idx])
            u_amp_grid = np.asarray(data["u_tide_amplitude"].values[c_idx])
            u_ph_grid = np.asarray(data["u_tide_phase"].values[c_idx])  # radians
            v_amp_grid = np.asarray(data["v_tide_amplitude"].values[c_idx])
            v_ph_grid = np.asarray(data["v_tide_phase"].values[c_idx])  # radians

            for s_idx, side in enumerate(self._SIDES):
                ob_lons, ob_lats = boundaries[side]
                vlen = side_valid[side]
                # u: interpolate amplitude + RADIAN phase on the u-grid, then
                # convert rad→seconds AFTER interp (script's FP order).
                u_amp = _interp_along_boundary(u_amp_grid, lon_u, lat_u, ob_lons, ob_lats)
                u_ph_rad = _interp_along_boundary(u_ph_grid, lon_u, lat_u, ob_lons, ob_lats)
                u_ph_sec = u_ph_rad / omega
                # v: same on the v-grid.
                v_amp = _interp_along_boundary(v_amp_grid, lon_v, lat_v, ob_lons, ob_lats)
                v_ph_rad = _interp_along_boundary(v_ph_grid, lon_v, lat_v, ob_lons, ob_lats)
                v_ph_sec = v_ph_rad / omega

                u_amp_out[c_idx, s_idx, :vlen] = u_amp
                u_ph_out[c_idx, s_idx, :vlen] = u_ph_sec
                v_amp_out[c_idx, s_idx, :vlen] = v_amp
                v_ph_out[c_idx, s_idx, :vlen] = v_ph_sec

        dims = ("constituent", "side", "along")
        out_ds = xr.Dataset(
            data_vars={
                "u_tide_amplitude": (dims, u_amp_out),
                "u_tide_phase": (dims, u_ph_out),
                "v_tide_amplitude": (dims, v_amp_out),
                "v_tide_phase": (dims, v_ph_out),
            },
            coords={
                "constituent": np.array(constituents),
                "period": ("constituent", periods),
                "side": np.array(list(self._SIDES)),
                "along": np.arange(n_along),
                "along_valid": ("side", np.array([side_valid[s] for s in self._SIDES])),
            },
        )
        new_specs = (
            VariableSpec("u_tide_amplitude", "sea_water_x_velocity_amplitude_due_to_tides",
                         "m s-1", "float64", dims),
            VariableSpec("u_tide_phase", "sea_water_x_velocity_phase_due_to_tides",
                         "s", "float64", dims),
            VariableSpec("v_tide_amplitude", "sea_water_y_velocity_amplitude_due_to_tides",
                         "m s-1", "float64", dims),
            VariableSpec("v_tide_phase", "sea_water_y_velocity_phase_due_to_tides",
                         "s", "float64", dims),
        )
        new_meta = dataclasses.replace(
            bundle.meta,
            location_kind=LocationKind.TIDAL_HARMONICS,
            variables=new_specs,
            target_grid_id=g.grid_id,
        )
        return DataBundle(data=out_ds, meta=new_meta)
