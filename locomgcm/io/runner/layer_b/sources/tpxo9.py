"""TPXO9-atlas tidal harmonic source.

Reads TPXO9-atlas v5 single-constituent netCDF files (u/v barotropic
transport), converts to tidal velocity amplitude + Greenwich phase on the
two staggered grids (lat_u/lon_u, lat_v/lon_v), and emits a TIDAL_HARMONICS
DataBundle with a `constituent` axis.

The transport→velocity→amplitude/phase load logic is inlined verbatim from
the original TPXO9 extraction script (`_load_tpxo9_transport`) so the lane
reproduces it bitwise — locomgcm must
NOT import the experiments package (keeps the layering clean).

PHASE UNIT: this source emits Greenwich phase in RADIANS (arctan2 of the
velocity imaginary/real parts). The conversion to seconds happens AFTER the
boundary interpolation, inside the `extract_tidal_obcs` operator, to match the
script's exact FP order (interp(rad) then / omega, NOT interp(rad/omega)). V0
astronomical correction is NOT applied here — it is a load-time, start_date
dependent shift (boundaries/tidal._compute_v0_correction), so these artifacts
stay reusable across simulation start dates.

TPXO9-atlas v5 netCDF structure (verified):
    u_*.nc:   lon_u/lat_u/lon_v/lat_v (1D float64),
              uRe/uIm/vRe/vIm (int32 cm^2/s, dims [nx=lon, ny=lat])
    grid_*.nc: hu/hv (float32 m, water depth at u/v points)
Data layout is [lon, lat] (Fortran order from OTIS).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from locomgcm.external_paths import resolve_input_path
from typing import Any

import numpy as np
import xarray as xr

from locomgcm.io.runner.layer_a import (
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


# Constituent catalog: name → period (seconds). Inlined from
# extract_tpxo9_tides.py:CONSTITUENTS (8 main diurnal/semidiurnal).
CONSTITUENTS: dict[str, float] = {
    "m2": 44712.0,
    "s2": 43200.0,
    "k1": 86164.0,
    "o1": 92950.0,
    "n2": 45570.0,
    "k2": 43082.0,
    "p1": 86637.0,
    "q1": 96726.0,
}

# Default extraction set when the source config omits `constituents`.
_DEFAULT_CONSTITUENTS = tuple(CONSTITUENTS.keys())


def _load_tpxo9_transport(
    nc_path: str, grid_path: str, lon_range: tuple, lat_range: tuple
) -> dict[str, np.ndarray]:
    """Load TPXO9-atlas u/v transport from a single-constituent netCDF file.

    Verbatim port of extract_tpxo9_tides.py:_load_tpxo9_transport so the lane
    is bitwise-identical. Returns amplitude (m/s) and Greenwich phase (RADIANS)
    on the subset u/v staggered grids.
    """
    import netCDF4

    ds = netCDF4.Dataset(nc_path, "r")
    lon_u = ds.variables["lon_u"][:].astype(np.float64)
    lat_u = ds.variables["lat_u"][:].astype(np.float64)
    lon_v = ds.variables["lon_v"][:].astype(np.float64)
    lat_v = ds.variables["lat_v"][:].astype(np.float64)

    # Subset to region of interest (±2° margin) to avoid loading full global
    lon_min = (lon_range[0] - 2) % 360
    lon_max = (lon_range[1] + 2) % 360
    lat_min = lat_range[0] - 2
    lat_max = lat_range[1] + 2

    iu = np.where((lon_u >= lon_min) & (lon_u <= lon_max))[0]
    ju = np.where((lat_u >= lat_min) & (lat_u <= lat_max))[0]
    iv = np.where((lon_v >= lon_min) & (lon_v <= lon_max))[0]
    jv = np.where((lat_v >= lat_min) & (lat_v <= lat_max))[0]

    # Transport in cm²/s → m²/s (÷ 10000)
    uRe = ds.variables["uRe"][iu[0]:iu[-1]+1, ju[0]:ju[-1]+1].astype(np.float64) / 1e4
    uIm = ds.variables["uIm"][iu[0]:iu[-1]+1, ju[0]:ju[-1]+1].astype(np.float64) / 1e4
    vRe = ds.variables["vRe"][iv[0]:iv[-1]+1, jv[0]:jv[-1]+1].astype(np.float64) / 1e4
    vIm = ds.variables["vIm"][iv[0]:iv[-1]+1, jv[0]:jv[-1]+1].astype(np.float64) / 1e4
    ds.close()

    # Read depth from grid file
    gs = netCDF4.Dataset(grid_path, "r")
    hu = gs.variables["hu"][iu[0]:iu[-1]+1, ju[0]:ju[-1]+1].astype(np.float64)
    hv = gs.variables["hv"][iv[0]:iv[-1]+1, jv[0]:jv[-1]+1].astype(np.float64)
    gs.close()

    lon_u_sub = lon_u[iu]
    lat_u_sub = lat_u[ju]
    lon_v_sub = lon_v[iv]
    lat_v_sub = lat_v[jv]

    # Transport [m²/s] → velocity [m/s]
    hu_safe = np.where(hu > 0, hu, 1.0)
    hv_safe = np.where(hv > 0, hv, 1.0)
    u_vel_re = np.where(hu > 0, uRe / hu_safe, 0.0)
    u_vel_im = np.where(hu > 0, uIm / hu_safe, 0.0)
    v_vel_re = np.where(hv > 0, vRe / hv_safe, 0.0)
    v_vel_im = np.where(hv > 0, vIm / hv_safe, 0.0)

    # Amplitude and Greenwich phase (radians)
    u_amp = np.sqrt(u_vel_re**2 + u_vel_im**2)
    u_phase_rad = np.arctan2(-u_vel_im, u_vel_re)
    v_amp = np.sqrt(v_vel_re**2 + v_vel_im**2)
    v_phase_rad = np.arctan2(-v_vel_im, v_vel_re)

    return {
        "lon_u": lon_u_sub, "lat_u": lat_u_sub,
        "u_amp": u_amp, "u_phase_rad": u_phase_rad,
        "lon_v": lon_v_sub, "lat_v": lat_v_sub,
        "v_amp": v_amp, "v_phase_rad": v_phase_rad,
    }


def _resolve_nc_file(tpxo_dir: Path, cname: str) -> Path:
    """Find the per-constituent transport netCDF (matches script glob fallback)."""
    nc_file = tpxo_dir / f"u_{cname}_tpxo9_atlas_30_v5.nc"
    if nc_file.exists():
        return nc_file
    candidates = sorted(tpxo_dir.glob(f"u_{cname}*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"tpxo9_netcdf: no transport file for {cname!r} under {tpxo_dir}")


def _resolve_grid_file(tpxo_dir: Path) -> Path:
    grid_file = tpxo_dir / "grid_tpxo9_atlas_30_v5.nc"
    if grid_file.exists():
        return grid_file
    candidates = sorted(tpxo_dir.glob("grid_*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"tpxo9_netcdf: grid file not found under {tpxo_dir}")


@register_source("tpxo9_netcdf")
class Tpxo9NetcdfSource:
    """TPXO9-atlas tidal harmonics on dual staggered grids (per constituent)."""

    capabilities = SourceCapabilities(
        produces=(LocationKind.TIDAL_HARMONICS,),
        variables=(
            "u_tide_amplitude", "u_tide_phase",
            "v_tide_amplitude", "v_tide_phase",
        ),
        # Tidal atlas is a static climatology — no time dimension. INSTANT keeps
        # the pipeline query time_range out of the way (time-invariant artifact,
        # reusable across simulation start dates).
        time_coverage_strategy=TimeCoverage.INSTANT,
        convention=ConventionTag.CF_STANDARD,
        requires_target_grid=True,
    )

    def __init__(
        self,
        ctx: SourceContext,
        *,
        path: Path | str,
        constituents: tuple[str, ...] | list[str] | None = None,
        **_unused: Any,
    ) -> None:
        self._ctx = ctx
        self._path = resolve_input_path(path)
        if not self._path.is_dir():
            raise ValueError(f"tpxo9_netcdf: path must be the atlas directory, got {path}")
        names = tuple(constituents) if constituents else _DEFAULT_CONSTITUENTS
        unknown = [c for c in names if c not in CONSTITUENTS]
        if unknown:
            raise ValueError(
                f"tpxo9_netcdf: unknown constituents {unknown}; "
                f"known: {sorted(CONSTITUENTS)}"
            )
        self._constituents = names
        # Resolve the concrete input files up front: grid + per-constituent
        # transport netCDFs. Exposed as `_files` so the pipeline CAS fingerprint
        # keys on each file's (path, size, mtime) — regenerating an atlas file in
        # place busts the cache.
        self._grid_file = _resolve_grid_file(self._path)
        self._nc_files = {c: _resolve_nc_file(self._path, c) for c in names}
        self._files = [self._grid_file, *self._nc_files.values()]

    def fetch(self, query: DataQuery) -> DataBundle:
        g = query.target_grid
        if g is None or not hasattr(g, "center_lat"):
            raise ValueError("tpxo9_netcdf: requires query.target_grid with grid params")

        # Domain extent (cell-edge box), matching extract_tpxo9_tides.py
        # extract_for_domain: half = count * res / 2 around the center.
        half_x = g.nx * g.resolution_deg / 2
        half_y = g.ny * g.resolution_deg / 2
        lon_w = g.center_lon - half_x
        lon_e = g.center_lon + half_x
        lat_s = g.center_lat - half_y
        lat_n = g.center_lat + half_y

        grid_file = self._grid_file

        u_amp_list, u_ph_list, v_amp_list, v_ph_list = [], [], [], []
        periods: list[float] = []
        ref_grid: dict[str, np.ndarray] | None = None
        for cname in self._constituents:
            nc_file = self._nc_files[cname]
            data = _load_tpxo9_transport(
                str(nc_file), str(grid_file),
                lon_range=(lon_w, lon_e), lat_range=(lat_s, lat_n),
            )
            if ref_grid is None:
                ref_grid = {
                    "lon_u": data["lon_u"], "lat_u": data["lat_u"],
                    "lon_v": data["lon_v"], "lat_v": data["lat_v"],
                }
            else:
                # All constituents share the global TPXO grid → identical subset.
                for k in ("lon_u", "lat_u", "lon_v", "lat_v"):
                    if not np.array_equal(ref_grid[k], data[k]):
                        raise ValueError(
                            f"tpxo9_netcdf: constituent {cname} {k} grid differs "
                            "from the first constituent (cannot stack)"
                        )
            u_amp_list.append(data["u_amp"])
            u_ph_list.append(data["u_phase_rad"])
            v_amp_list.append(data["v_amp"])
            v_ph_list.append(data["v_phase_rad"])
            periods.append(CONSTITUENTS[cname])

        assert ref_grid is not None
        u_amp = np.stack(u_amp_list, axis=0)        # (constituent, lon_u, lat_u)
        u_phase = np.stack(u_ph_list, axis=0)
        v_amp = np.stack(v_amp_list, axis=0)        # (constituent, lon_v, lat_v)
        v_phase = np.stack(v_ph_list, axis=0)

        u_dims = ("constituent", "lon_u", "lat_u")
        v_dims = ("constituent", "lon_v", "lat_v")
        out_ds = xr.Dataset(
            data_vars={
                "u_tide_amplitude": (u_dims, u_amp),
                "u_tide_phase": (u_dims, u_phase),
                "v_tide_amplitude": (v_dims, v_amp),
                "v_tide_phase": (v_dims, v_phase),
            },
            coords={
                "constituent": np.array(list(self._constituents)),
                "period": ("constituent", np.asarray(periods, dtype="float64")),
                "lon_u": ref_grid["lon_u"],
                "lat_u": ref_grid["lat_u"],
                "lon_v": ref_grid["lon_v"],
                "lat_v": ref_grid["lat_v"],
            },
        )
        var_specs = [
            VariableSpec("u_tide_amplitude", "sea_water_x_velocity_amplitude_due_to_tides",
                         "m s-1", "float64", u_dims),
            VariableSpec("u_tide_phase", "sea_water_x_velocity_phase_due_to_tides",
                         "rad", "float64", u_dims),
            VariableSpec("v_tide_amplitude", "sea_water_y_velocity_amplitude_due_to_tides",
                         "m s-1", "float64", v_dims),
            VariableSpec("v_tide_phase", "sea_water_y_velocity_phase_due_to_tides",
                         "rad", "float64", v_dims),
        ]
        return self._wrap(out_ds, var_specs, grid_file)

    def _wrap(
        self, ds: xr.Dataset, var_specs: list[VariableSpec], grid_file: Path
    ) -> DataBundle:
        meta_no_prov = BundleMeta(
            schema_version="0.1.0",
            location_kind=LocationKind.TIDAL_HARMONICS,
            convention=ConventionTag.CF_STANDARD,
            variables=tuple(var_specs),
            time_range=None,           # time-invariant tidal atlas
            target_grid_id=None,       # source-native staggered grids
            provenance=(),
        )
        prov = ProvenanceStep(
            kind="source",
            name="Tpxo9NetcdfSource",
            params={
                "path": str(self._path),
                "grid_file": str(grid_file),
                "constituents": tuple(self._constituents),
            },
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
