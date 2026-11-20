"""MITgcm verification ``input/`` binary sources.

Element size is *not* fixed: MITgcm writes and reads its ``input/*.bin`` files
at the precision named by the ``readBinaryPrec`` entry of ``input/data``
(``model/src/set_defaults.F:355`` leaves it at ``precFloat32`` unless the case
says otherwise), so both sources take ``read_binary_prec`` and resolve it via
:func:`mit_binary_dtype`. The byte order stays big-endian in every branch --
MITgcm is built with ``-convert big_endian`` and a little-endian read silently
yields ~1e308 garbage (docs/VALIDATION.md).

``read_binary_prec`` defaults to 64 so the pre-existing exp4 wiring (which is
``readBinaryPrec=64``) keeps its byte-for-byte behaviour; the translated
verification cases pass their own value from the namelist.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from defgcm.io.runner.layer_a import (
    BundleMeta,
    ConventionTag,
    DataBundle,
    LocationKind,
    ProvenanceStep,
    VariableSpec,
)

from .._registry import register_source
from .._types import DataQuery, Grid, SourceCapabilities, SourceContext, TimeCoverage


_IC_VARIABLES = ("temperature", "salinity", "eta", "u", "v", "bathymetry")
_OBC_VARIABLES = ("temperature", "salinity", "eta", "u", "v", "w")
_FORCING_VARIABLES = ("tau_x", "tau_y", "heat_flux", "freshwater_flux", "pload")

_IC_CF_NAMES = {
    "temperature": "sea_water_potential_temperature",
    "salinity": "sea_water_salinity",
    "eta": "sea_surface_height_above_geoid",
    "u": "eastward_sea_water_velocity",
    "v": "northward_sea_water_velocity",
    "bathymetry": "sea_floor_depth_below_geoid",
}
_OBC_CF_NAMES = {
    **{k: v for k, v in _IC_CF_NAMES.items() if k != "bathymetry"},
    "w": "upward_sea_water_velocity",
}
_FORCING_CF_NAMES = {
    "tau_x": "surface_downward_eastward_stress",
    "tau_y": "surface_downward_northward_stress",
    "heat_flux": "surface_downward_heat_flux_in_sea_water",
    "freshwater_flux": "water_evaporation_flux",
    "pload": "surface_air_pressure",
}
_MESH_CF_NAMES = {
    "bathymetry": _IC_CF_NAMES["bathymetry"],
    "del_x": "grid_cell_width_along_x_axis",
    "del_y": "grid_cell_width_along_y_axis",
}
_UNITS = {
    "temperature": "degC",
    "salinity": "1e-3",
    "eta": "m",
    "u": "m s-1",
    "v": "m s-1",
    "w": "m s-1",
    "bathymetry": "m",
    "del_x": "m",
    "del_y": "m",
    "tau_x": "N m-2",
    "tau_y": "N m-2",
    "heat_flux": "W m-2",
    "freshwater_flux": "m s-1",
    "pload": "Pa",
}

_SIDE_ALIASES = {
    "west": "W",
    "w": "W",
    "east": "E",
    "e": "E",
    "south": "S",
    "s": "S",
    "north": "N",
    "n": "N",
}
_SIDE_NAMES = {"W": "west", "E": "east", "S": "south", "N": "north"}

_OBC_ROLE_ALIASES = {
    "t": "temperature",
    "temperature": "temperature",
    "s": "salinity",
    "salinity": "salinity",
    "u": "u",
    "v": "v",
    "w": "w",
    "eta": "eta",
}

_DEFAULT_OBC_FILES = {
    "west": {"u": "OB_WestU.bin", "salinity": "OBzonalS.bin", "eta": "OB_WestH.bin", "w": "OBzonalW.bin"},
    "east": {"u": "OB_EastU.bin", "eta": "OB_EastH.bin"},
    "south": {"u": "OBmeridU.bin"},
    "north": {"u": "OBmeridU.bin"},
}

#: MITgcm's own default when ``readBinaryPrec`` is absent from ``input/data``
#: (``references/MITgcm/model/src/set_defaults.F:355``).  This module keeps 64
#: as *its* default so the exp4 wiring that predates the parameter is unchanged;
#: callers that translate a namelist always pass the value they read.
_DEFAULT_READ_BINARY_PREC = 64

_MIT_DTYPES = {32: ">f4", 64: ">f8"}


def mit_binary_dtype(read_binary_prec: int | None) -> str:
    """``readBinaryPrec`` bits -> big-endian numpy dtype string.

    Big-endian is not negotiable: MITgcm is compiled with
    ``-convert big_endian`` (docs/VALIDATION.md), so a native-order read passes every
    size check and returns ~1e308 garbage.
    """
    prec = _DEFAULT_READ_BINARY_PREC if read_binary_prec is None else int(read_binary_prec)
    try:
        return _MIT_DTYPES[prec]
    except KeyError:
        raise ValueError(
            f"readBinaryPrec must be 32 or 64, got {read_binary_prec!r}"
        ) from None


_T_REF = np.array(
    [
        0.696834,
        0.497738,
        0.298643,
        0.0995477,
        -0.0995477,
        -0.298643,
        -0.497738,
        -0.696834,
    ],
    dtype=np.float64,
)
_S_REF = 35.0
_BASE_TIME = 10_800.0
_EXTERN_FORCING_PERIOD = 21_600.0
_EXTERN_FORCING_CYCLE = 43_200.0


@dataclass(frozen=True)
class Exp4BoundarySample:
    west_u: np.ndarray
    east_u: np.ndarray
    south_u: np.ndarray
    north_u: np.ndarray
    west_temp: np.ndarray
    east_temp: np.ndarray
    south_temp: np.ndarray
    north_temp: np.ndarray
    west_salt: np.ndarray
    east_salt: np.ndarray
    south_salt: np.ndarray
    north_salt: np.ndarray
    west_eta: np.ndarray
    east_eta: np.ndarray


@dataclass(frozen=True)
class Exp4BoundaryForcing:
    west_u: np.ndarray
    east_u: np.ndarray
    south_u: np.ndarray
    north_u: np.ndarray
    west_salt: np.ndarray
    west_w: np.ndarray
    west_eta: np.ndarray
    east_eta: np.ndarray
    t_ref: np.ndarray
    #: Per-level reference salinity, ``(nz,)`` -- MITgcm's ``sRef`` is a
    #: profile, exactly like ``tRef`` (``model/src/load_ref_files.F:76-84``).
    s_ref: np.ndarray
    time_origin_seconds: float
    forcing_period_seconds: float
    forcing_cycle_seconds: float

    @classmethod
    def load(cls, args: SimpleNamespace) -> "Exp4BoundaryForcing":
        input_dir = Path(args.mitgcm_input_dir)
        nx, ny, nz = int(args.nx), int(args.ny), int(args.nz)
        dtype = mit_binary_dtype(getattr(args, "read_binary_prec", None))
        merid_u = _read_binary(input_dir / "OBmeridU.bin", (2, nz, nx), dtype)
        return cls(
            west_u=_read_binary(input_dir / "OB_WestU.bin", (2, nz, ny), dtype),
            east_u=_read_binary(input_dir / "OB_EastU.bin", (2, nz, ny), dtype),
            south_u=merid_u.copy(),
            north_u=merid_u.copy(),
            west_salt=_read_binary(input_dir / "OBzonalS.bin", (2, nz, ny), dtype),
            west_w=_read_binary(input_dir / "OBzonalW.bin", (2, nz, ny), dtype),
            west_eta=_read_binary(input_dir / "OB_WestH.bin", (2, ny), dtype),
            east_eta=_read_binary(input_dir / "OB_EastH.bin", (2, ny), dtype),
            t_ref=np.asarray(args.t_ref, dtype=np.float64),
            # A scalar sRef is the uniform profile; normalising here keeps the
            # dataclass field a (nz,) array for every caller.
            s_ref=_s_ref(nz, args.s_ref),
            time_origin_seconds=float(args.base_time),
            forcing_period_seconds=float(args.forcing_period),
            forcing_cycle_seconds=float(args.forcing_cycle),
        )

    def _sample_series(self, series: np.ndarray, time_seconds: float) -> np.ndarray:
        if series.shape[0] == 1:
            return series[0].copy()
        phase = (float(time_seconds) - self.time_origin_seconds) % self.forcing_cycle_seconds
        slab = int(math.floor(phase / self.forcing_period_seconds)) % series.shape[0]
        next_slab = (slab + 1) % series.shape[0]
        frac = (phase - slab * self.forcing_period_seconds) / self.forcing_period_seconds
        return ((1.0 - frac) * series[slab] + frac * series[next_slab]).astype(
            np.float64,
            copy=False,
        )

    def sample(self, time_seconds: float) -> Exp4BoundarySample:
        ny = self.west_u.shape[-1]
        nx = self.south_u.shape[-1]
        temp_profile_y = np.repeat(self.t_ref[:, None], ny, axis=1)
        temp_profile_x = np.repeat(self.t_ref[:, None], nx, axis=1)
        salt_profile = np.asarray(self.s_ref, dtype=np.float64)
        default_salt_y = np.repeat(salt_profile[:, None], ny, axis=1)
        default_salt_x = np.repeat(salt_profile[:, None], nx, axis=1)
        return Exp4BoundarySample(
            west_u=self._sample_series(self.west_u, time_seconds),
            east_u=self._sample_series(self.east_u, time_seconds),
            south_u=self._sample_series(self.south_u, time_seconds),
            north_u=self._sample_series(self.north_u, time_seconds),
            west_temp=temp_profile_y,
            east_temp=temp_profile_y,
            south_temp=temp_profile_x,
            north_temp=temp_profile_x,
            west_salt=self._sample_series(self.west_salt, time_seconds),
            east_salt=default_salt_y,
            south_salt=default_salt_x,
            north_salt=default_salt_x,
            west_eta=self._sample_series(self.west_eta, time_seconds),
            east_eta=self._sample_series(self.east_eta, time_seconds),
        )


def _read_binary(path: Path, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    """Read one MITgcm field record, MITgcm's own way.

    ``READ_FLD_XY*``/``READ_FLD_XYZ_*`` hard-code ``iRec=1``
    (``pkg/rw/read_fld_xyz_rl.F:43-49``), so an initial-condition file that
    holds several same-shaped records -- a Levitus monthly climatology reused as
    the IC, say -- contributes only its *first* record and MITgcm ignores the
    rest.  Accepting exactly one record would refuse those files; accepting any
    size would let a genuinely mis-shaped file through, so the file must be a
    whole number of records.
    """
    data = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if data.size != expected:
        if expected <= 0 or data.size % expected != 0:
            raise ValueError(
                f"{path} expected {expected} values (or a whole multiple of "
                f"them), found {data.size}"
            )
        data = data[:expected]
    return data.reshape(shape).astype(np.float64)


def _spacing_file_path(input_dir: Path, filename: str, axis: str) -> Path:
    raw = Path(filename)
    if raw.is_absolute():
        raise ValueError(
            f"del_{axis}_file must be inside the verification input directory, "
            f"got absolute path {filename!r}"
        )
    root = input_dir.resolve()
    path = (input_dir / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(
            f"del_{axis}_file must be inside the verification input directory, "
            f"got {filename!r}"
        ) from None
    if not path.is_file():
        raise FileNotFoundError(f"del_{axis}_file not found: {path}")
    return path


def _read_spacing_file(
    input_dir: Path,
    filename: str,
    *,
    count: int,
    axis: str,
    dtype: str,
) -> np.ndarray:
    """Read one complete MITgcm horizontal-spacing record."""
    path = _spacing_file_path(input_dir, filename, axis)
    values = np.fromfile(path, dtype=dtype)
    if values.size != count:
        raise ValueError(
            f"del_{axis}_file {path} expected exactly {count} values, "
            f"found {values.size}"
        )
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"del_{axis}_file {path} values must all be finite")
    if not (values > 0.0).all():
        raise ValueError(
            f"del_{axis}_file {path} values must all be strictly positive"
        )
    return values


def _inline_spacing(
    values: Sequence[float] | float,
    *,
    count: int,
    axis: str,
) -> np.ndarray:
    spacing = np.asarray(values, dtype=np.float64)
    if spacing.ndim == 0 or spacing.shape == (1,):
        spacing = np.full((count,), float(spacing.reshape(-1)[0]), dtype=np.float64)
    if spacing.shape != (count,):
        raise ValueError(
            f"del_{axis} inline spacing must contain exactly {count} values, "
            f"got shape {spacing.shape}"
        )
    if not np.isfinite(spacing).all():
        raise ValueError(f"del_{axis} inline spacing values must all be finite")
    if not (spacing > 0.0).all():
        raise ValueError(
            f"del_{axis} inline spacing values must all be strictly positive"
        )
    return np.ascontiguousarray(spacing, dtype=np.float64)


def _load_spacing(
    input_dir: Path,
    *,
    filename: str | None,
    inline: Sequence[float] | float | None,
    count: int,
    axis: str,
    dtype: str,
) -> np.ndarray | None:
    if filename:
        return _read_spacing_file(
            input_dir,
            filename,
            count=count,
            axis=axis,
            dtype=dtype,
        )
    if inline is not None:
        return _inline_spacing(inline, count=count, axis=axis)
    return None


def _parse_tref_from_data(path: Path, nz: int) -> np.ndarray:
    text = path.read_text()
    match = re.search(r"tRef=\s*([^\n]+)", text)
    if match is None:
        raise ValueError(f"could not parse tRef from {path}")
    values = [chunk.strip() for chunk in match.group(1).split(",")]
    parsed = [float(value) for value in values if value]
    if len(parsed) != nz:
        raise ValueError(f"expected {nz} tRef values, found {len(parsed)} in {path}")
    return np.asarray(parsed, dtype=np.float64)


def _verification_input_dir(path: str | Path) -> Path:
    base = Path(path)
    return base if base.name == "input" else base / "input"


def _grid_dims(grid: Grid | None, nx: int | None, ny: int | None, nz: int | None) -> tuple[int, int, int]:
    if grid is not None:
        return (
            int(nx if nx is not None else grid.nx),
            int(ny if ny is not None else grid.ny),
            int(nz if nz is not None else grid.nz),
        )
    if nx is None or ny is None or nz is None:
        raise ValueError("mit_input_binary: nx, ny, nz are required without target_grid")
    return int(nx), int(ny), int(nz)


def _coords(grid: Grid | None, nx: int, ny: int, nz: int) -> dict[str, Any]:
    if grid is None:
        return {
            "time": [np.datetime64("2019-01-01", "ns")],
            "z": np.arange(nz, dtype=np.float64),
            "latitude": np.arange(ny, dtype=np.float64),
            "longitude": np.arange(nx, dtype=np.float64),
            "latitude_v": np.arange(ny, dtype=np.float64),
            "longitude_u": np.arange(nx, dtype=np.float64),
        }

    lat_off = np.arange(ny, dtype="float64") - 0.5 * (ny - 1)
    lon_off = np.arange(nx, dtype="float64") - 0.5 * (nx - 1)
    thicknesses = np.asarray(grid.vertical_layer_thicknesses, dtype=np.float64)
    if thicknesses.size == nz:
        interfaces = np.concatenate([[0.0], np.cumsum(thicknesses)])
        z = 0.5 * (interfaces[:-1] + interfaces[1:])
    else:
        z = np.arange(nz, dtype=np.float64)
    return {
        "time": [np.datetime64("2019-01-01", "ns")],
        "z": z,
        "latitude": grid.center_lat + lat_off * grid.resolution_deg,
        "longitude": grid.center_lon + lon_off * grid.resolution_deg,
        "latitude_v": grid.center_lat + (lat_off - 0.5) * grid.resolution_deg,
        "longitude_u": grid.center_lon + (lon_off - 0.5) * grid.resolution_deg,
    }


def _var_spec(name: str, dims: tuple[str, ...], cf_names: Mapping[str, str]) -> VariableSpec:
    return VariableSpec(
        name=name,
        cf_standard_name=cf_names[name],
        units=_UNITS[name],
        dtype="float64",
        dims=dims,
    )


def _wrap_bundle(
    *,
    ds: xr.Dataset,
    location_kind: LocationKind,
    convention: ConventionTag,
    var_specs: Sequence[VariableSpec],
    target_grid_id: str | None,
    source_name: str,
    params: Mapping[str, Any],
    time_range: tuple[datetime, datetime] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> DataBundle:
    meta_no_prov = BundleMeta(
        schema_version="0.1.0",
        location_kind=location_kind,
        convention=convention,
        variables=tuple(var_specs),
        time_range=time_range,
        target_grid_id=target_grid_id,
        provenance=(),
        extras=extras or {},
    )
    prov = ProvenanceStep(
        kind="source",
        name=source_name,
        params=dict(params),
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


def _read_bathymetry(
    input_dir: Path,
    filename: str | None,
    nx: int,
    ny: int,
    dtype: str = ">f8",
) -> tuple[np.ndarray | None, str]:
    """Read ``bathyFile``; ``(None, 'unset')`` when the case declares none.

    MITgcm with a blank ``bathyFile`` sets ``R_low = rF(Nr+1)`` -- a flat bottom
    at the full model depth (``model/src/ini_depths.F:91-102``).  Returning
    ``None`` hands that case to the caller's flat-bottom fill (which uses
    ``grid.depth_m`` = ``sum(delR)``) instead of inventing a topography here.
    """
    if not filename:
        return None, "unset (MIT ini_depths.F:91-102 flat bottom)"
    requested = input_dir / filename
    if not requested.exists():
        raise FileNotFoundError(f"no bathymetry file found at {requested}")
    mit_bathy = _read_binary(requested, (ny, nx), dtype)
    # MIT bathyFile/topog inputs are negative-ocean; LOCOMgcm consumes positive depth.
    return np.where(mit_bathy < 0.0, -mit_bathy, 0.0).astype(np.float64), requested.name


def _t_ref(input_dir: Path, nz: int, override: Sequence[float] | None = None) -> np.ndarray:
    if override is not None:
        values = np.asarray(override, dtype=np.float64)
        if values.shape != (nz,):
            raise ValueError(f"t_ref shape {values.shape} != {(nz,)}")
        return values
    data_file = input_dir / "data"
    if data_file.exists():
        return _parse_tref_from_data(data_file, nz)
    values = np.asarray(_T_REF, dtype=np.float64)
    if values.shape != (nz,):
        raise ValueError(f"default exp4 t_ref shape {values.shape} != {(nz,)}")
    return values


def _s_ref(nz: int, override: Sequence[float] | float | None = None) -> np.ndarray:
    """Resolve ``sRef`` to a ``(nz,)`` profile.

    MITgcm's ``sRef`` is a per-level profile just like ``tRef``
    (``model/src/load_ref_files.F:76-84`` fills an unset level from the one
    above it), so a depth-varying reference salinity has to survive the trip.
    A scalar is still accepted -- it is the uniform profile, and every existing
    caller that passed one keeps its byte-for-byte result.
    """
    if override is None:
        return np.full(nz, _S_REF, dtype=np.float64)
    values = np.asarray(override, dtype=np.float64)
    if values.ndim == 0:
        return np.full(nz, float(values), dtype=np.float64)
    if values.shape != (nz,):
        raise ValueError(f"s_ref shape {values.shape} != {(nz,)}")
    return values


def load_mit_input_binary_ic(
    *,
    mit_verification_dir: str | Path,
    nx: int,
    ny: int,
    nz: int,
    bathy_file: str | None = "bathyFile.bin",
    temp_file: str | None = "hydrogThetaFile.bin",
    salt_file: str | None = "hydrogSaltFile.bin",
    u_file: str | None = "uVelInitFile.bin",
    v_file: str | None = "vVelInitFile.bin",
    eta_file: str | None = "etaInitFile.bin",
    t_ref: Sequence[float] | None = None,
    s_ref: Sequence[float] | float | None = None,
    read_binary_prec: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load MIT ``input/`` IC arrays at the case's own ``readBinaryPrec``.

    File names come from the case's ``PARM05`` entries; a name given as ``None``
    or ``''`` means the namelist declares no such file, which is how MITgcm
    itself decides to fall back to the reference profile / zero / flat bottom.
    A name that *is* given but missing on disk keeps the historical
    exp4 behaviour (reference profile for T/S, zero for u/v/eta).

    The real MITgcm ``verification/exp4`` fixture has no hydrog/u/v IC files;
    MIT initializes those from namelist reference profiles.  For that case this
    adapter uses the same ``tRef``/``sRef``/zero construction as the historical
    exp4 scaffold while still reading bathymetry through the binary reader path.
    """

    input_dir = _verification_input_dir(mit_verification_dir)
    dtype = mit_binary_dtype(read_binary_prec)
    files: dict[str, str] = {"read_binary_prec": dtype}

    bathy, bathy_name = _read_bathymetry(input_dir, bathy_file, nx, ny, dtype)
    files["bathymetry"] = bathy_name

    def read_or_none(filename: str | None, shape: tuple[int, ...]) -> np.ndarray | None:
        if not filename:
            return None
        path = input_dir / filename
        if not path.exists():
            return None
        files[filename] = path.name
        return _read_binary(path, shape, dtype)

    temp = read_or_none(temp_file, (nz, ny, nx))
    if temp is None:
        profile = _t_ref(input_dir, nz, t_ref)
        temp = np.repeat(profile[:, None, None], ny, axis=1)
        temp = np.repeat(temp, nx, axis=2).astype(np.float64)
        files["temperature"] = "data:tRef"
    else:
        files["temperature"] = temp_file

    salt = read_or_none(salt_file, (nz, ny, nx))
    if salt is None:
        salt_profile = _s_ref(nz, s_ref)
        salt = np.repeat(salt_profile[:, None, None], ny, axis=1)
        salt = np.repeat(salt, nx, axis=2).astype(np.float64)
        files["salinity"] = "data:sRef"
    else:
        files["salinity"] = salt_file

    eta = read_or_none(eta_file, (ny, nx))
    if eta is None:
        eta = np.zeros((ny, nx), dtype=np.float64)
        files["eta"] = "zero"
    else:
        files["eta"] = eta_file

    u = read_or_none(u_file, (nz, ny, nx))
    if u is None:
        u = np.zeros((nz, ny, nx), dtype=np.float64)
        files["u"] = "zero"
    else:
        files["u"] = u_file

    v = read_or_none(v_file, (nz, ny, nx))
    if v is None:
        v = np.zeros((nz, ny, nx), dtype=np.float64)
        files["v"] = "zero"
    else:
        files["v"] = v_file

    arrays = {
        "temperature": np.asarray(temp, dtype=np.float64),
        "salinity": np.asarray(salt, dtype=np.float64),
        "eta": np.asarray(eta, dtype=np.float64),
        "u": np.asarray(u, dtype=np.float64),
        "v": np.asarray(v, dtype=np.float64),
    }
    # A case that declares no bathyFile leaves the key out entirely so the
    # caller applies MITgcm's own flat-bottom rule (ini_depths.F:91-102).
    if bathy is not None:
        arrays["bathymetry"] = np.asarray(bathy, dtype=np.float64)
    return arrays, files


@register_source("mit_input_binary_ic")
class MitInputBinaryIcSource:
    """Read MITgcm verification IC binaries into a LOCOMgcm-sign bundle."""

    capabilities = SourceCapabilities(
        produces=(LocationKind.GRIDDED_3D_OCEAN,),
        variables=_IC_VARIABLES,
        time_coverage_strategy=TimeCoverage.INSTANT,
        convention=ConventionTag.TORCH_OCEAN,
        requires_target_grid=True,
    )

    def __init__(
        self,
        ctx: SourceContext,
        *,
        mit_verification_dir: str | Path,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        bathy_file: str | None = "bathyFile.bin",
        temp_file: str | None = "hydrogThetaFile.bin",
        salt_file: str | None = "hydrogSaltFile.bin",
        u_file: str | None = "uVelInitFile.bin",
        v_file: str | None = "vVelInitFile.bin",
        eta_file: str | None = "etaInitFile.bin",
        t_ref: Sequence[float] | None = None,
        s_ref: Sequence[float] | float | None = None,
        read_binary_prec: int | None = None,
        **_unused: Any,
    ) -> None:
        self._ctx = ctx
        self._mit_verification_dir = Path(mit_verification_dir)
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._files = {
            "bathy_file": bathy_file,
            "temp_file": temp_file,
            "salt_file": salt_file,
            "u_file": u_file,
            "v_file": v_file,
            "eta_file": eta_file,
        }
        self._t_ref = t_ref
        self._s_ref = s_ref
        # Validate eagerly: a bad precision must fail at construction, not
        # after the first fetch has already been dispatched.
        self._read_binary_prec = read_binary_prec
        mit_binary_dtype(read_binary_prec)

    def fetch(self, query: DataQuery) -> DataBundle:
        if query.target_grid is None:
            raise ValueError("mit_input_binary_ic: target_grid required")
        nx, ny, nz = _grid_dims(query.target_grid, self._nx, self._ny, self._nz)
        arrays, resolved_files = load_mit_input_binary_ic(
            mit_verification_dir=self._mit_verification_dir,
            nx=nx,
            ny=ny,
            nz=nz,
            bathy_file=self._files["bathy_file"],
            temp_file=self._files["temp_file"],
            salt_file=self._files["salt_file"],
            u_file=self._files["u_file"],
            v_file=self._files["v_file"],
            eta_file=self._files["eta_file"],
            t_ref=self._t_ref,
            s_ref=self._s_ref,
            read_binary_prec=self._read_binary_prec,
        )

        requested = [name for name in query.variables if name in arrays]
        coords = _coords(query.target_grid, nx, ny, nz)
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        specs: list[VariableSpec] = []
        for name in requested:
            if name == "bathymetry":
                dims = ("latitude", "longitude")
                arr = arrays[name]
            elif name == "eta":
                dims = ("time", "latitude", "longitude")
                arr = arrays[name][None, ...]
            elif name == "u":
                # MIT uVel is already on each cell's west face.  The file has
                # nx values; LOCOMgcm adds its redundant east face later.
                dims = ("time", "z", "latitude", "longitude_u")
                arr = arrays[name][None, ...]
            elif name == "v":
                # MIT vVel is already on each cell's south face.  The file has
                # ny values; LOCOMgcm adds its redundant north face later.
                dims = ("time", "z", "latitude_v", "longitude")
                arr = arrays[name][None, ...]
            else:
                dims = ("time", "z", "latitude", "longitude")
                arr = arrays[name][None, ...]
            data_vars[name] = (dims, np.asarray(arr, dtype=np.float64))
            specs.append(_var_spec(name, dims, _IC_CF_NAMES))

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        return _wrap_bundle(
            ds=ds,
            location_kind=LocationKind.GRIDDED_3D_OCEAN,
            convention=ConventionTag.TORCH_OCEAN,
            var_specs=specs,
            target_grid_id=query.target_grid.grid_id,
            source_name="MitInputBinaryIcSource",
            params={
                "mit_verification_dir": str(self._mit_verification_dir),
                "resolved_files": resolved_files,
            },
            time_range=query.time_range,
            extras={"resolved_files": resolved_files},
        )


@register_source("mit_input_binary_mesh")
class MitInputBinaryMeshSource:
    """The same ``bathyFile`` as the IC source, delivered on the ``mesh`` lane.

    stage1a's ``mesh`` lane is the only one whose reader is convention-aware
    (``model_input._positive_depth_from_mesh``), so a bathymetry that travels
    with the ``state`` bundle would be taken verbatim -- a CF elevation read as
    a depth.  Splitting it out keeps one file name, one precision and one sign
    rule; there is no second reader.
    """

    capabilities = SourceCapabilities(
        produces=(LocationKind.MESH,),
        variables=("bathymetry", "del_x", "del_y"),
        time_coverage_strategy=TimeCoverage.INSTANT,
        convention=ConventionTag.TORCH_OCEAN,
        requires_target_grid=True,
    )

    def __init__(
        self,
        ctx: SourceContext,
        *,
        mit_verification_dir: str | Path,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        bathy_file: str | None = "bathyFile.bin",
        del_x_file: str | None = None,
        del_y_file: str | None = None,
        del_x: Sequence[float] | float | None = None,
        del_y: Sequence[float] | float | None = None,
        read_binary_prec: int | None = None,
        **_unused: Any,
    ) -> None:
        self._ctx = ctx
        self._mit_verification_dir = Path(mit_verification_dir)
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._bathy_file = bathy_file
        self._del_x_file = del_x_file
        self._del_y_file = del_y_file
        self._del_x = del_x
        self._del_y = del_y
        self._read_binary_prec = read_binary_prec
        mit_binary_dtype(read_binary_prec)

    def fetch(self, query: DataQuery) -> DataBundle:
        if query.target_grid is None:
            raise ValueError("mit_input_binary_mesh: target_grid required")
        nx, ny, nz = _grid_dims(query.target_grid, self._nx, self._ny, self._nz)
        input_dir = _verification_input_dir(self._mit_verification_dir)
        dtype = mit_binary_dtype(self._read_binary_prec)
        bathy, name = _read_bathymetry(input_dir, self._bathy_file, nx, ny, dtype)
        if bathy is None:
            # MITgcm with a blank bathyFile: flat bottom at the full model
            # depth (ini_depths.F:91-102), which grid.depth_m carries verbatim.
            bathy = np.full((ny, nx), float(query.target_grid.depth_m), dtype=np.float64)
        del_x = _load_spacing(
            input_dir,
            filename=self._del_x_file,
            inline=self._del_x,
            count=nx,
            axis="x",
            dtype=dtype,
        )
        del_y = _load_spacing(
            input_dir,
            filename=self._del_y_file,
            inline=self._del_y,
            count=ny,
            axis="y",
            dtype=dtype,
        )
        coords = _coords(query.target_grid, nx, ny, nz)
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
            "bathymetry": (("latitude", "longitude"), bathy)
        }
        if del_x is not None:
            data_vars["del_x"] = (("longitude",), del_x)
        if del_y is not None:
            data_vars["del_y"] = (("latitude",), del_y)
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={k: v for k, v in coords.items() if k in ("latitude", "longitude")},
        )
        resolved_files = {
            "bathymetry": name,
            "del_x": self._del_x_file or ("inline" if del_x is not None else None),
            "del_y": self._del_y_file or ("inline" if del_y is not None else None),
            "read_binary_prec": dtype,
        }
        return _wrap_bundle(
            ds=ds,
            location_kind=LocationKind.MESH,
            convention=ConventionTag.TORCH_OCEAN,
            var_specs=[
                _var_spec(name, tuple(ds[name].dims), _MESH_CF_NAMES)
                for name in ds.data_vars
            ],
            target_grid_id=query.target_grid.grid_id,
            source_name="MitInputBinaryMeshSource",
            params={
                "mit_verification_dir": str(self._mit_verification_dir),
                "resolved_files": resolved_files,
            },
            time_range=query.time_range,
            extras={"resolved_files": resolved_files},
        )


def _forcing_record_times(
    period_seconds: float | None,
    cycle_seconds: float | None,
) -> tuple[int, np.ndarray, bool]:
    """Record count / record times / wrap flag for MITgcm surface forcing files.

    ``periodicExternalForcing = .FALSE.`` (the default,
    ``model/src/set_defaults.F:331``): ``model/src/ini_forcing.F:70-75`` calls
    ``READ_FLD_XY_RS`` once at initialisation and the field is then held fixed
    for the whole run -- one record, at t = 0.

    ``.TRUE.``: ``model/src/external_fields_load.F:78-82`` hands the clock to
    ``GET_PERIODIC_INTERVAL``, which

    * centres record ``n`` at ``(n - 1/2)*externForcingPeriod``
      (``eesupp/src/get_periodic_interval.F:118`` subtracts ``recSpacing*0.5``
      before indexing), and
    * cycles ``nbRec = NINT(cycle/period)`` records, taking the successor of the
      last record to be the first one (``:123-124`` ``tRec2 = 1 + MOD(tRec1,
      nbRec)``), with linear weights ``:127-128``.

    The wrap is expressed here as two extra records on the time axis -- a copy
    of the last record at ``-period/2`` and a copy of the first at
    ``(nbRec + 1/2)*period`` -- so a plain linear interpolation over
    ``t mod cycle`` reproduces ``GET_PERIODIC_INTERVAL``'s weights everywhere,
    seam included, without a second interpolator.

    The two validity conditions below are MITgcm's own: ``ini_parms.F:1069-1085``
    aborts the run when either ``externForcingCycle`` or ``externForcingPeriod``
    is zero, when the cycle is not a whole multiple of the period, or when the
    cycle is shorter than the period.
    """
    if period_seconds is None:
        return 1, np.zeros(1, dtype=np.float64), False
    period = float(period_seconds)
    cycle = 0.0 if cycle_seconds is None else float(cycle_seconds)
    if period <= 0.0 or cycle <= 0.0:
        raise ValueError(
            "periodicExternalForcing needs externForcingCycle and "
            "externForcingPeriod both non-zero (ini_parms.F:1069-1074); got "
            f"period={period_seconds!r}, cycle={cycle_seconds!r}"
        )
    n_rec = int(round(cycle / period))
    if cycle != n_rec * period:
        raise ValueError(
            "externForcingCycle must be a whole multiple of "
            f"externForcingPeriod (ini_parms.F:1075-1080): cycle={cycle!r}, "
            f"period={period!r}"
        )
    centres = (np.arange(n_rec, dtype=np.float64) + 0.5) * period
    times = np.concatenate(
        [[-0.5 * period], centres, [(n_rec + 0.5) * period]]
    )
    return n_rec, times, True


def _read_forcing_records(
    path: Path,
    filename: str,
    shape: tuple[int, ...],
    dtype: str,
    n_rec: int,
) -> np.ndarray:
    """Read the first ``n_rec`` records without deriving cadence from file size."""
    values = np.fromfile(path, dtype=dtype)
    per_record = int(np.prod(shape))
    if per_record <= 0:
        raise ValueError(f"{filename}: empty record shape {shape}")
    if values.size % per_record != 0:
        raise ValueError(
            f"{filename}: {values.size} values is not a whole number of "
            f"{shape} records at dtype {dtype}"
        )
    available = values.size // per_record
    if available < n_rec:
        raise ValueError(
            f"{filename}: needs {n_rec} records (externForcingCycle / "
            f"externForcingPeriod), file holds {available}"
        )
    return (
        values[: n_rec * per_record]
        .reshape((n_rec, *shape))
        .astype(np.float64)
    )


def _load_mit_input_binary_forcing_fields(
    *,
    mit_verification_dir: str | Path,
    nx: int,
    ny: int,
    zonal_wind_file: str | None = None,
    merid_wind_file: str | None = None,
    surf_qnet_file: str | None = None,
    empmr_file: str | None = None,
    pload_file: str | None = None,
    read_binary_prec: int | None = None,
    forcing_period_seconds: float | None = None,
    forcing_cycle_seconds: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    """Read core MITgcm surface-forcing files on the native tracer-indexed XY grid.

    The arrays keep MITgcm's own staggering: ``fu`` and ``fv`` are declared at
    the "Southwest C-grid U point" and "Southwest C-grid V point"
    (``model/inc/FFIELDS.h:21`` / ``:27``) and ``READ_FLD_XY_RS`` copies the
    file into them verbatim -- there is no centre-to-face reconstruction and no
    land mask on the way in (the dry-face zeroing happens later, through
    ``_recip_hFacW`` in ``model/src/apply_forcing.F:143-147``).  The consumer
    therefore has to place them on faces directly; see the ``c_grid_face``
    stress location in ``defgcm/dynamics/era5_wind_forcing.py``.

    Sign needs no conversion: ``fu > 0`` increases ``uVel`` eastward
    (``FFIELDS.h:17-19``), which is both the LOCOMgcm and the CF
    ``surface_downward_eastward_stress`` sense.

    ``Qnet`` is read at tracer centres in W/m2 and is upward-positive
    (``FFIELDS.h:44,102``).  The bundle is tagged TORCH_OCEAN, so it crosses the
    convention boundary here as ``heat_flux = -Qnet`` (positive warms).  The
    ``EmPmRFile`` bytes are a tracer-centred volume flux in m/s
    (``FFIELDS.h:29-36,104``): ``ini_forcing.F:83-90`` reads them and then
    multiplies by ``rhoConstFresh`` to form the model's kg/m2/s ``EmPmR``.  The
    existing WPFlux producer owns that single multiplication, so this reader
    preserves the file values as ``freshwater_flux`` rather than multiplying
    them early.

    ``pLoadFile`` is the atmospheric pressure anomaly relative to ``surf_pRef``
    at tracer centres in Pa (``FFIELDS.h:68``), read verbatim by
    ``ini_forcing.F:191-195``; it has the same sign in every convention.

    An undeclared component is returned as zeros, matching ``ini_ffields.F:45-55``
    and the blank-name guards in ``ini_forcing.F:70-83,191-195``.
    """
    input_dir = _verification_input_dir(mit_verification_dir)
    dtype = mit_binary_dtype(read_binary_prec)
    file_specs = (
        ("tau_x", zonal_wind_file, 1.0),
        ("tau_y", merid_wind_file, 1.0),
        ("heat_flux", surf_qnet_file, -1.0),
        ("freshwater_flux", empmr_file, 1.0),
        ("pload", pload_file, 1.0),
    )
    if not any(filename for _, filename, _ in file_specs):
        raise ValueError(
            "mit_input_binary_forcing: neither zonalWindFile nor meridWindFile "
            "nor surfQnetFile nor EmPmRFile nor pLoadFile is declared"
        )
    n_rec, times, wrap = _forcing_record_times(
        forcing_period_seconds, forcing_cycle_seconds
    )
    n_time = int(times.size)
    resolved: dict[str, str] = {"read_binary_prec": dtype}
    fields: dict[str, np.ndarray] = {}
    for name, filename, scale in file_specs:
        if not filename:
            fields[name] = np.zeros((n_time, ny, nx), dtype=np.float64)
            resolved[name] = "unset (MIT ini_forcing.F:70-83 leaves it zero)"
            continue
        records = _read_forcing_records(
            input_dir / filename, filename, (ny, nx), dtype, n_rec
        )
        if wrap:
            records = np.concatenate([records[-1:], records, records[:1]], axis=0)
        fields[name] = records if scale == 1.0 else scale * records
        resolved[name] = filename
    return times, fields, resolved


def load_mit_input_binary_forcing(
    *,
    mit_verification_dir: str | Path,
    nx: int,
    ny: int,
    zonal_wind_file: str | None = None,
    merid_wind_file: str | None = None,
    surf_qnet_file: str | None = None,
    empmr_file: str | None = None,
    read_binary_prec: int | None = None,
    forcing_period_seconds: float | None = None,
    forcing_cycle_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Compatibility wrapper returning the historical wind-only tuple."""
    times, fields, resolved = _load_mit_input_binary_forcing_fields(
        mit_verification_dir=mit_verification_dir,
        nx=nx,
        ny=ny,
        zonal_wind_file=zonal_wind_file,
        merid_wind_file=merid_wind_file,
        surf_qnet_file=surf_qnet_file,
        empmr_file=empmr_file,
        read_binary_prec=read_binary_prec,
        forcing_period_seconds=forcing_period_seconds,
        forcing_cycle_seconds=forcing_cycle_seconds,
    )
    return times, fields["tau_x"], fields["tau_y"], resolved


@register_source("mit_input_binary_forcing")
class MitInputBinaryForcingSource:
    """MITgcm core wind/Qnet/EmPmR files as one surface-forcing bundle.

    The bundle carries the file's own record cadence on a ``time_seconds``
    coordinate plus ``forcing_period_seconds`` / ``forcing_cycle_seconds`` in
    ``meta.extras``, which is what tells the runtime wind source that the series
    is cyclic rather than a prefix that may be truncated to the run length.
    """

    capabilities = SourceCapabilities(
        produces=(LocationKind.GRIDDED_2D_SURFACE,),
        variables=_FORCING_VARIABLES,
        time_coverage_strategy=TimeCoverage.INSTANT,
        convention=ConventionTag.TORCH_OCEAN,
        requires_target_grid=True,
    )

    def __init__(
        self,
        ctx: SourceContext,
        *,
        mit_verification_dir: str | Path,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        zonal_wind_file: str | None = None,
        merid_wind_file: str | None = None,
        surf_qnet_file: str | None = None,
        empmr_file: str | None = None,
        pload_file: str | None = None,
        read_binary_prec: int | None = None,
        forcing_period_seconds: float | None = None,
        forcing_cycle_seconds: float | None = None,
        **_unused: Any,
    ) -> None:
        self._ctx = ctx
        self._mit_verification_dir = Path(mit_verification_dir)
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._zonal_wind_file = zonal_wind_file
        self._merid_wind_file = merid_wind_file
        self._surf_qnet_file = surf_qnet_file
        self._empmr_file = empmr_file
        self._pload_file = pload_file
        self._read_binary_prec = read_binary_prec
        self._forcing_period_seconds = forcing_period_seconds
        self._forcing_cycle_seconds = forcing_cycle_seconds
        # Both the precision and the record cadence are contract errors, not
        # data errors: fail at construction rather than after the first fetch.
        mit_binary_dtype(read_binary_prec)
        _forcing_record_times(forcing_period_seconds, forcing_cycle_seconds)

    def fetch(self, query: DataQuery) -> DataBundle:
        if query.target_grid is None:
            raise ValueError("mit_input_binary_forcing: target_grid required")
        nx, ny, nz = _grid_dims(query.target_grid, self._nx, self._ny, self._nz)
        times, arrays, resolved_files = _load_mit_input_binary_forcing_fields(
            mit_verification_dir=self._mit_verification_dir,
            nx=nx,
            ny=ny,
            zonal_wind_file=self._zonal_wind_file,
            merid_wind_file=self._merid_wind_file,
            surf_qnet_file=self._surf_qnet_file,
            empmr_file=self._empmr_file,
            pload_file=self._pload_file,
            read_binary_prec=self._read_binary_prec,
            forcing_period_seconds=self._forcing_period_seconds,
            forcing_cycle_seconds=self._forcing_cycle_seconds,
        )
        base_coords = _coords(query.target_grid, nx, ny, nz)
        dims = ("time", "latitude", "longitude")
        requested = [name for name in query.variables if name in arrays] or list(
            _FORCING_VARIABLES
        )
        ds = xr.Dataset(
            data_vars={name: (dims, arrays[name]) for name in requested},
            coords={
                "time_seconds": ("time", times),
                "latitude": base_coords["latitude"],
                "longitude": base_coords["longitude"],
            },
        )
        return _wrap_bundle(
            ds=ds,
            location_kind=LocationKind.GRIDDED_2D_SURFACE,
            convention=ConventionTag.TORCH_OCEAN,
            var_specs=[
                _var_spec(name, dims, _FORCING_CF_NAMES) for name in requested
            ],
            target_grid_id=query.target_grid.grid_id,
            source_name="MitInputBinaryForcingSource",
            params={
                "mit_verification_dir": str(self._mit_verification_dir),
                "resolved_files": resolved_files,
            },
            time_range=query.time_range,
            extras={
                "resolved_files": resolved_files,
                "forcing_period_seconds": (
                    None
                    if self._forcing_period_seconds is None
                    else float(self._forcing_period_seconds)
                ),
                "forcing_cycle_seconds": (
                    None
                    if self._forcing_cycle_seconds is None
                    else float(self._forcing_cycle_seconds)
                ),
                "stress_location": "c_grid_face",
            },
        )


def _normalize_sides(sides: Sequence[str] | None) -> tuple[str, ...]:
    raw = sides if sides is not None else ("west", "east", "south", "north")
    labels: list[str] = []
    for side in raw:
        label = _SIDE_ALIASES.get(str(side).lower())
        if label is None:
            raise ValueError(f"mit_obcs_binary: unknown side {side!r}")
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def _merge_file_mapping(
    files: Mapping[str, Mapping[str, str]] | None,
    overrides: Mapping[str, str | None],
) -> dict[str, dict[str, str]]:
    merged = (
        {side: dict(mapping) for side, mapping in _DEFAULT_OBC_FILES.items()}
        if files is None
        else {}
    )
    for side, mapping in (files or {}).items():
        side_name = _SIDE_NAMES[_normalize_sides((side,))[0]]
        normalized: dict[str, str] = {}
        for role, filename in mapping.items():
            role_name = _OBC_ROLE_ALIASES.get(str(role).lower())
            if role_name is None:
                raise ValueError(
                    f"mit_obcs_binary: unknown field role {role!r} on side {side!r}"
                )
            normalized[role_name] = str(filename)
        merged.setdefault(side_name, {}).update(normalized)
    for key, value in overrides.items():
        if value is None or not key.endswith("_file"):
            continue
        parts = key[:-5].split("_", 1)
        if len(parts) != 2:
            continue
        side, var = parts
        var = _OBC_ROLE_ALIASES.get(var.lower(), var)
        if side in merged:
            merged[side][var] = str(value)
    return merged


def _is_default_obc_mapping(mapping: Mapping[str, Mapping[str, str]]) -> bool:
    return {k: dict(v) for k, v in mapping.items()} == _DEFAULT_OBC_FILES


def _legacy_obc_args(
    input_dir: Path,
    nx: int,
    ny: int,
    nz: int,
    *,
    t_ref: Sequence[float] | None,
    s_ref: Sequence[float] | float | None,
    base_time: float | None,
    forcing_period: float | None,
    forcing_cycle: float | None,
    read_binary_prec: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        mitgcm_input_dir=input_dir,
        nx=nx,
        ny=ny,
        nz=nz,
        read_binary_prec=read_binary_prec,
        t_ref=_t_ref(input_dir, nz, t_ref),
        s_ref=_s_ref(nz, s_ref),
        base_time=_BASE_TIME if base_time is None else float(base_time),
        forcing_period=_EXTERN_FORCING_PERIOD if forcing_period is None else float(forcing_period),
        forcing_cycle=_EXTERN_FORCING_CYCLE if forcing_cycle is None else float(forcing_cycle),
    )


def load_mit_obcs_binary(
    *,
    mit_verification_dir: str | Path,
    nx: int,
    ny: int,
    nz: int,
    files: Mapping[str, Mapping[str, str]] | None = None,
    t_ref: Sequence[float] | None = None,
    s_ref: Sequence[float] | float | None = None,
    base_time: float | None = None,
    forcing_period: float | None = None,
    forcing_cycle: float | None = None,
    periodic_external_forcing: bool | None = None,
    forcing_period_seconds: float | None = None,
    forcing_cycle_seconds: float | None = None,
    read_binary_prec: int | None = None,
) -> tuple[Any, dict[str, dict[str, str]]]:
    """Load MITgcm OBC records using namelist cadence and per-side file roles.

    The historical no-argument exp4 mapping remains a dedicated compatibility
    branch.  Every explicit mapping must also declare whether MITgcm uses
    periodic external forcing; periodic record counts come exclusively from
    ``externForcingCycle / externForcingPeriod``.
    """

    input_dir = _verification_input_dir(mit_verification_dir)
    dtype = mit_binary_dtype(read_binary_prec)
    mapping = {side: dict(values) for side, values in (files or _DEFAULT_OBC_FILES).items()}
    args = _legacy_obc_args(
        input_dir,
        nx,
        ny,
        nz,
        t_ref=t_ref,
        s_ref=s_ref,
        base_time=base_time,
        forcing_period=forcing_period,
        forcing_cycle=forcing_cycle,
        read_binary_prec=read_binary_prec,
    )
    legacy_cadence = (
        periodic_external_forcing is None
        and forcing_period_seconds is None
        and forcing_cycle_seconds is None
    )
    if _is_default_obc_mapping(mapping) and legacy_cadence:
        return Exp4BoundaryForcing.load(args), mapping

    if periodic_external_forcing is None:
        raise ValueError(
            "mit_obcs_binary: explicit files require periodic_external_forcing; "
            "record cadence must come from the MITgcm namelist"
        )
    period = forcing_period_seconds
    cycle = forcing_cycle_seconds
    n_rec, record_times, _wrap = _forcing_record_times(
        period if periodic_external_forcing else None,
        cycle if periodic_external_forcing else None,
    )
    time_seconds = record_times[1:-1] if periodic_external_forcing else record_times
    if base_time is not None:
        time_seconds = time_seconds + float(base_time)

    fields: dict[str, dict[str, np.ndarray]] = {}
    resolved: dict[str, dict[str, str]] = {}
    for side, file_roles in mapping.items():
        label = _SIDE_NAMES[_normalize_sides((side,))[0]]
        n_along = ny if label in {"west", "east"} else nx
        side_fields: dict[str, np.ndarray] = {}
        side_resolved: dict[str, str] = {}
        for role in _OBC_VARIABLES:
            filename = file_roles.get(role)
            record_shape = (n_along,) if role == "eta" else (nz, n_along)
            if filename:
                side_fields[role] = _read_forcing_records(
                    input_dir / filename,
                    filename,
                    record_shape,
                    dtype,
                    n_rec,
                )
                side_resolved[role] = filename
            elif role == "temperature":
                side_fields[role] = np.broadcast_to(
                    args.t_ref[None, :, None], (n_rec, nz, n_along)
                ).copy()
                side_resolved[role] = "tRef fallback (pkg/obcs/obcs_calc.F:126,176)"
            elif role == "salinity":
                side_fields[role] = np.broadcast_to(
                    args.s_ref[None, :, None], (n_rec, nz, n_along)
                ).copy()
                side_resolved[role] = "sRef fallback (pkg/obcs/obcs_calc.F:127,177)"
            else:
                side_fields[role] = np.zeros((n_rec, *record_shape), dtype=np.float64)
                side_resolved[role] = "zero fallback (pkg/obcs/obcs_calc.F:124-132,174-182)"
        fields[label] = side_fields
        resolved[label] = side_resolved

    return _MitObcsBoundaryForcing(
        time_seconds=np.asarray(time_seconds, dtype=np.float64),
        fields=fields,
        forcing_period_seconds=(
            float(period) if periodic_external_forcing and period is not None else None
        ),
        forcing_cycle_seconds=(
            float(cycle) if periodic_external_forcing and cycle is not None else None
        ),
    ), resolved


@dataclass(frozen=True)
class _MitObcsBoundaryForcing:
    time_seconds: np.ndarray
    fields: Mapping[str, Mapping[str, np.ndarray]]
    forcing_period_seconds: float | None
    forcing_cycle_seconds: float | None


def _empty_4d(n_time: int, n_side: int, nz: int, n_along: int) -> np.ndarray:
    return np.full((n_time, n_side, nz, n_along), np.nan, dtype=np.float64)


def _empty_3d(n_time: int, n_side: int, n_along: int) -> np.ndarray:
    return np.full((n_time, n_side, n_along), np.nan, dtype=np.float64)


def _put_side(arr: np.ndarray, side_index: int, values: np.ndarray) -> None:
    along = values.shape[-1]
    arr[:, side_index, ..., :along] = values


@register_source("mit_obcs_binary")
class MitObcsBinarySource:
    """Read MITgcm OBC binaries into a MIT-convention boundary bundle."""

    capabilities = SourceCapabilities(
        produces=(LocationKind.BOUNDARY_TIMESERIES,),
        variables=_OBC_VARIABLES,
        time_coverage_strategy=TimeCoverage.INTERVAL,
        convention=ConventionTag.MITGCM,
        requires_target_grid=False,
    )

    def __init__(
        self,
        ctx: SourceContext,
        *,
        mit_verification_dir: str | Path,
        nx: int,
        ny: int,
        nz: int,
        sides: Sequence[str] = ("west", "east", "south", "north"),
        files: Mapping[str, Mapping[str, str]] | None = None,
        t_ref: Sequence[float] | None = None,
        s_ref: Sequence[float] | float | None = None,
        base_time: float | None = None,
        forcing_period: float | None = None,
        forcing_cycle: float | None = None,
        periodic_external_forcing: bool | None = None,
        forcing_period_seconds: float | None = None,
        forcing_cycle_seconds: float | None = None,
        read_binary_prec: int | None = None,
        **overrides: Any,
    ) -> None:
        self._ctx = ctx
        self._mit_verification_dir = Path(mit_verification_dir)
        self._nx = int(nx)
        self._ny = int(ny)
        self._nz = int(nz)
        self._sides = _normalize_sides(sides)
        file_overrides = {k: v for k, v in overrides.items() if isinstance(v, str) or v is None}
        self._files = _merge_file_mapping(files, file_overrides)
        self._t_ref = t_ref
        self._s_ref = s_ref
        self._base_time = base_time
        self._forcing_period = forcing_period
        self._forcing_cycle = forcing_cycle
        self._periodic_external_forcing = periodic_external_forcing
        self._forcing_period_seconds = forcing_period_seconds
        self._forcing_cycle_seconds = forcing_cycle_seconds
        self._read_binary_prec = read_binary_prec
        mit_binary_dtype(read_binary_prec)

    def resolved_files(self) -> dict[str, dict[str, str]]:
        return {side: dict(mapping) for side, mapping in self._files.items()}

    def fetch(self, query: DataQuery) -> DataBundle:
        if query.location_kind != LocationKind.BOUNDARY_TIMESERIES:
            raise ValueError(f"mit_obcs_binary: unsupported location_kind {query.location_kind}")

        query_sides = query.extras.get("sides") if query.extras else None
        sides = _normalize_sides(query_sides) if query_sides is not None else self._sides
        forcing, resolved = load_mit_obcs_binary(
            mit_verification_dir=self._mit_verification_dir,
            nx=self._nx,
            ny=self._ny,
            nz=self._nz,
            files=self._files,
            t_ref=self._t_ref,
            s_ref=self._s_ref,
            base_time=self._base_time,
            forcing_period=self._forcing_period,
            forcing_cycle=self._forcing_cycle,
            periodic_external_forcing=self._periodic_external_forcing,
            forcing_period_seconds=self._forcing_period_seconds,
            forcing_cycle_seconds=self._forcing_cycle_seconds,
            read_binary_prec=self._read_binary_prec,
        )

        if isinstance(forcing, _MitObcsBoundaryForcing):
            return self._bundle_from_generic(query, sides, forcing, resolved)

        n_time = int(forcing.west_u.shape[0])
        n_along = max(self._nx, self._ny)
        time_seconds = forcing.time_origin_seconds + np.arange(n_time, dtype=np.float64) * forcing.forcing_period_seconds
        side_index = {label: i for i, label in enumerate(sides)}

        data: dict[str, np.ndarray] = {
            "temperature": _empty_4d(n_time, len(sides), self._nz, n_along),
            "salinity": _empty_4d(n_time, len(sides), self._nz, n_along),
            "u": _empty_4d(n_time, len(sides), self._nz, n_along),
            "v": np.zeros((n_time, len(sides), self._nz, n_along), dtype=np.float64),
            "w": _empty_4d(n_time, len(sides), self._nz, n_along),
            "eta": _empty_3d(n_time, len(sides), n_along),
        }

        for t_idx, time_s in enumerate(time_seconds):
            sample = forcing.sample(float(time_s))
            side_values_4d = {
                "W": {
                    "temperature": sample.west_temp,
                    "salinity": sample.west_salt,
                    "u": sample.west_u,
                    "w": forcing._sample_series(forcing.west_w, float(time_s)),
                },
                "E": {
                    "temperature": sample.east_temp,
                    "salinity": sample.east_salt,
                    "u": sample.east_u,
                },
                "S": {
                    "temperature": sample.south_temp,
                    "salinity": sample.south_salt,
                    "u": sample.south_u,
                },
                "N": {
                    "temperature": sample.north_temp,
                    "salinity": sample.north_salt,
                    "u": sample.north_u,
                },
            }
            side_values_3d = {
                "W": {"eta": sample.west_eta},
                "E": {"eta": sample.east_eta},
            }
            for label in sides:
                idx = side_index[label]
                for name, value in side_values_4d[label].items():
                    data[name][t_idx, idx, :, : value.shape[-1]] = value
                for name, value in side_values_3d.get(label, {}).items():
                    data[name][t_idx, idx, : value.shape[-1]] = value

        requested = [name for name in query.variables if name in data]
        coords: dict[str, Any] = {
            "time": time_seconds,
            "side": np.asarray(sides),
            "z": np.arange(self._nz, dtype=np.float64),
            "along": np.arange(n_along, dtype=np.int32),
        }
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        specs: list[VariableSpec] = []
        for name in requested:
            if name == "eta":
                dims = ("time", "side", "along")
            else:
                dims = ("time", "side", "z", "along")
            data_vars[name] = (dims, data[name])
            specs.append(_var_spec(name, dims, _OBC_CF_NAMES))

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        return _wrap_bundle(
            ds=ds,
            location_kind=LocationKind.BOUNDARY_TIMESERIES,
            convention=ConventionTag.MITGCM,
            var_specs=specs,
            target_grid_id=None,
            source_name="MitObcsBinarySource",
            params={
                "mit_verification_dir": str(self._mit_verification_dir),
                "sides": sides,
                "resolved_files": resolved,
            },
            time_range=query.time_range,
            extras={
                "resolved_files": resolved,
                "side_labels": sides,
                "valid_along": {"W": self._ny, "E": self._ny, "S": self._nx, "N": self._nx},
                "forcing_period_seconds": forcing.forcing_period_seconds,
                "forcing_cycle_seconds": forcing.forcing_cycle_seconds,
            },
        )

    def _bundle_from_generic(
        self,
        query: DataQuery,
        sides: tuple[str, ...],
        forcing: _MitObcsBoundaryForcing,
        resolved: dict[str, dict[str, str]],
    ) -> DataBundle:
        n_time = int(forcing.time_seconds.size)
        n_along = max(self._nx, self._ny)
        data: dict[str, np.ndarray] = {
            name: (
                _empty_3d(n_time, len(sides), n_along)
                if name == "eta"
                else _empty_4d(n_time, len(sides), self._nz, n_along)
            )
            for name in _OBC_VARIABLES
        }
        for side_index, side_label in enumerate(sides):
            side_name = _SIDE_NAMES[side_label]
            if side_name not in forcing.fields:
                raise ValueError(
                    f"mit_obcs_binary: requested side {side_label} has no OB index/file mapping"
                )
            for name, values in forcing.fields[side_name].items():
                _put_side(data[name], side_index, values)

        requested = [name for name in query.variables if name in data]
        coords: dict[str, Any] = {
            "time": forcing.time_seconds,
            "side": np.asarray(sides),
            "z": np.arange(self._nz, dtype=np.float64),
            "along": np.arange(n_along, dtype=np.int32),
        }
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        specs: list[VariableSpec] = []
        for name in requested:
            dims = ("time", "side", "along") if name == "eta" else (
                "time",
                "side",
                "z",
                "along",
            )
            data_vars[name] = (dims, data[name])
            specs.append(_var_spec(name, dims, _OBC_CF_NAMES))
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        valid_along = {
            side: (self._ny if side in {"W", "E"} else self._nx) for side in sides
        }
        return _wrap_bundle(
            ds=ds,
            location_kind=LocationKind.BOUNDARY_TIMESERIES,
            convention=ConventionTag.MITGCM,
            var_specs=specs,
            target_grid_id=None,
            source_name="MitObcsBinarySource",
            params={
                "mit_verification_dir": str(self._mit_verification_dir),
                "sides": sides,
                "resolved_files": resolved,
            },
            time_range=query.time_range,
            extras={
                "resolved_files": resolved,
                "side_labels": sides,
                "valid_along": valid_along,
                "forcing_period_seconds": forcing.forcing_period_seconds,
                "forcing_cycle_seconds": forcing.forcing_cycle_seconds,
            },
        )


__all__ = [
    "MitInputBinaryIcSource",
    "MitInputBinaryMeshSource",
    "MitObcsBinarySource",
    "load_mit_input_binary_ic",
    "load_mit_obcs_binary",
    "mit_binary_dtype",
]
