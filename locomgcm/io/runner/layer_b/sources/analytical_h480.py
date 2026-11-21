"""Deterministic regional shelf/basin inputs, generated on the requested grid.

Depth is positive downward and bathymetry is positive ocean depth, zero land.
Tracers and velocities are cell-centred; the canonical preparation chain owns
wet masking, C-grid face reconstruction and prescribed-boundary enforcement.
Time records repeat daily and are linearly interpolated by the runtime sources.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr

from locomgcm.io.runner.layer_a import (
    BundleMeta, ConventionTag, DataBundle, LocationKind, ProvenanceStep, VariableSpec,
)
from .._registry import register_source
from .._types import DataQuery, SourceCapabilities, SourceContext, TimeCoverage
from .analytical_iw import _grid_coords, _var_spec


PERIOD = 86400.0
CADENCE = 21600.0
_STATE = ("temperature", "salinity", "u", "v", "eta")
_ROLES = {
    "state": (LocationKind.GRIDDED_3D_OCEAN, _STATE),
    "boundary": (LocationKind.BOUNDARY_TIMESERIES, _STATE),
    "forcing": (LocationKind.GRIDDED_2D_SURFACE, ("tau_x", "tau_y")),
    "mesh": (LocationKind.MESH, ("bathymetry",)),
}


def ocean_field(name, x, y, z, i, j, k, seconds=0.0):
    """Fields at broadcastable centre coordinates x,y in (0,1), depth z in m.

    The index perturbation is independent of evaluation order, so direct edge
    evaluation equals slicing the initial volume, including at small grid sizes.
    """
    phase = 2 * np.pi * (seconds % PERIOD) / PERIOD
    noise = np.sin(1.4142135623730951 * (i + 1)
                   + 1.7320508075688772 * (j + 1)
                   + 2.23606797749979 * (k + 1))
    if name == "temperature":
        return (4 + 22 * np.exp(-np.maximum(z - 35, 0) / 600)
                - 0.002 * np.minimum(z, 35) + 0.04 * x + 0.03 * y
                + 1e-5 * noise + 0.002 * np.sin(phase) * np.exp(-z / 600))
    if name == "salinity":
        return (34.3 + 0.7 * (1 - np.exp(-z / 900)) + 0.002 * x + 0.003 * y
                + 1e-6 * noise + 0.0002 * np.sin(phase) * np.exp(-z / 900))
    if name == "u":
        # Two meridional lobes fit inside the wet western shelf (y < 0.55).
        # Both velocity signs survive the uniform OBC transport correction.
        return (0.015 * np.sin(4 * np.pi * y) * np.exp(-z / 900)
                * (1 + 0.1 * np.sin(phase)) + 1e-6 * noise + 0 * x)
    if name == "v":
        return (0.01 * np.sin(2 * np.pi * x) * np.exp(-z / 1100)
                * (1 + 0.1 * np.sin(phase)) + 1e-6 * noise + 0 * y)
    if name == "eta":
        return 0.001 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 0.0001 * np.sin(phase)
    raise ValueError(f"unknown ocean field {name!r}")


@register_source("analytical_h480")
class AnalyticalH480Source:
    """Generate state, mesh, forcing or edge-only boundary bundles without files."""

    capabilities = SourceCapabilities(
        produces=tuple(kind for kind, _ in _ROLES.values()),
        variables=(*_STATE, "tau_x", "tau_y", "bathymetry"),
        convention=ConventionTag.CF_STANDARD,
        time_coverage_strategy=TimeCoverage.ON_DEMAND, requires_target_grid=True,
    )

    def __init__(self, ctx: SourceContext, *, role: str, **_):
        if role not in _ROLES:
            raise ValueError(f"unknown analytical_h480 role {role!r}")
        self.role = role
        kind, variables = _ROLES[role]
        self.capabilities = SourceCapabilities(
            produces=(kind,), variables=variables,
            convention=ConventionTag.LOCOMGCM if role == "mesh" else ConventionTag.CF_STANDARD,
            time_coverage_strategy=TimeCoverage.ON_DEMAND,
            requires_target_grid=True,
        )

    def fetch(self, query: DataQuery) -> DataBundle:
        grid = query.target_grid
        if grid is None or query.location_kind not in self.capabilities.produces:
            raise ValueError("analytical_h480 needs a target grid and its configured location kind")
        coords = _grid_coords(grid, nx=grid.nx, ny=grid.ny, nz=grid.nz, depth=grid.depth_m)
        i, j, k = (np.arange(n, dtype=np.float64) for n in (grid.nx, grid.ny, grid.nz))
        x, y, z = (i + 0.5) / grid.nx, (j + 0.5) / grid.ny, coords["z"]
        names = tuple(v for v in query.variables if v in self.capabilities.variables)
        if not names:
            raise ValueError("analytical_h480 query has no supported variables")
        extras = {}
        if self.role == "mesh":
            q = x[None, :] - 0.10 - 0.12 * y[:, None]
            depth = (80 + 5300 * (1 + np.tanh((q - 0.22) / 0.10)) / 2
                     + 130 * np.cos(2 * np.pi * x[None, :]) * np.sin(np.pi * y[:, None]))
            land = (q <= 0) & (y[:, None] > 0.55)
            depth = np.where(land, 0, np.clip(depth, 40, 5490))
            ds = xr.Dataset({"bathymetry": (("latitude", "longitude"), depth)},
                            coords={a: coords[a] for a in ("latitude", "longitude")})
        elif self.role == "state":
            data = {}
            for name in names:
                if name == "eta":
                    values = ocean_field(name, x[None, :], y[:, None], 0, 0, 0, 0)
                    dims = ("time", "latitude", "longitude")
                else:
                    values = ocean_field(name, x[None, None, :], y[None, :, None], z[:, None, None],
                                         i[None, None, :], j[None, :, None], k[:, None, None])
                    dims = ("time", "z", "latitude", "longitude")
                data[name] = (dims, values[None])
            ds = xr.Dataset(data, coords={**coords, "time": [0.0]})
        else:
            # Include the closing record: ERA5WindSource interpolates through
            # the last interval before taking time modulo the declared period.
            seconds = np.arange(5, dtype=np.float64) * CADENCE
            extras = {"forcing_cycle_seconds": PERIOD, "forcing_period_seconds": CADENCE}
            if self.role == "forcing":
                phase = (2 * np.pi / PERIOD) * (seconds[:, None, None] % PERIOD)
                tx = (0.08 + 0.025 * np.sin(phase)) * (1 + 0.15 * y[None, :, None]) + 0 * x[None, None, :]
                ty = (0.025 + 0.01 * np.sin(phase)) * (1 + 0.1 * x[None, None, :]) + 0 * y[None, :, None]
                data = {name: (("time", "latitude", "longitude"), values)
                        for name, values in (("tau_x", tx), ("tau_y", ty)) if name in names}
                ds = xr.Dataset(data, coords={"time": seconds, "latitude": coords["latitude"],
                                              "longitude": coords["longitude"]})
            else:
                sides = ("N", "E", "S", "W")
                lengths = {"N": grid.nx, "S": grid.nx, "E": grid.ny, "W": grid.ny}
                along = max(grid.nx, grid.ny)
                data = {}
                for name in names:
                    is3d = name != "eta"
                    shape = (len(seconds), 4, grid.nz, along) if is3d else (len(seconds), 4, along)
                    values = np.zeros(shape, dtype=np.float64)
                    for side_idx, side in enumerate(sides):
                        ii = i if side in ("N", "S") else np.full_like(j, grid.nx - 1 if side == "E" else 0)
                        jj = j if side in ("E", "W") else np.full_like(i, grid.ny - 1 if side == "N" else 0)
                        xx, yy = (ii + 0.5) / grid.nx, (jj + 0.5) / grid.ny
                        for it, sec in enumerate(seconds):
                            field = ocean_field(name, xx, yy, z[:, None] if is3d else 0,
                                                ii, jj, k[:, None] if is3d else 0, sec)
                            values[it, side_idx, ..., :lengths[side]] = field
                    dims = ("time", "side", "z", "along") if is3d else ("time", "side", "along")
                    data[name] = (dims, values)
                ds = xr.Dataset(data, coords={"time": seconds, "side": list(sides), "z": z,
                                              "along": np.arange(along),
                                              "along_valid": ("side", [lengths[s] for s in sides])})
                extras["valid_along"] = lengths
        specs = []
        for name in ds.data_vars:
            if name in ("tau_x", "tau_y"):
                specs.append(VariableSpec(name=name, cf_standard_name=(
                    "surface_downward_eastward_stress" if name == "tau_x"
                    else "surface_downward_northward_stress"), units="Pa", dtype="float64",
                    dims=ds[name].dims))
            else:
                specs.append(_var_spec(name, ds[name].dims))
        return DataBundle(data=ds, meta=BundleMeta(
            schema_version="0.1.0", location_kind=query.location_kind,
            convention=self.capabilities.convention, variables=tuple(specs),
            time_range=None, target_grid_id=grid.grid_id, extras=extras,
            provenance=(ProvenanceStep(kind="source", name="AnalyticalH480Source",
                                       params={"role": self.role, "formula_version": 2},
                                       timestamp=datetime.utcnow(), bundle_hash="00000000"),),
        ))
