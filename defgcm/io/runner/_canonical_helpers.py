"""Shared helpers for canonical state assembly.

This module factors out the duplicated logic between the stage1a canonical
builder (``defgcm/io/runner/model_input.py:build_model_state``) and the
runtime loader and integrator (``experiments/cases/_runtime.py:run_simulation``)
so both consumers see exactly the same boundary/wind source wiring,
reference-profile derivation, and manifest enrichment logic.

Subtle per-call-site
differences (e.g. n_steps/dt time-window extension, end_time_seconds
truncation, reference-profile fallback) are exposed as optional kwargs so
callers can plug in their existing semantics without duplicating the
function bodies.
"""

from __future__ import annotations

from defgcm.sampling import SampleClock, SamplingRule, sampling_rule

from dataclasses import replace
from datetime import datetime
from typing import Any, Callable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from defgcm.distributed.layout import ParallelLayout

import numpy as np
import torch

from defgcm.io.runner.builders import BoundaryBundleSource
from defgcm.io.runner.config._schema import DynamicsEntry, ModelManifest
from defgcm.boundaries._registry import BOUNDARY_CONDITIONS
from defgcm.boundaries.open import sponge_ob_policy


__all__ = [
    "seconds_from_time_coord",
    "BarotropicBoundaryBundleSource",
    "boundary_source_from_bundle",
    "wind_source_from_bundle",
    "surface_flux_source_from_bundle",
    "build_bulk_source",
    "ASSEMBLY_SOURCE_TYPES",
    "exf_bulk_source",
    "exf_flux_sources",
    "bulk_kernel_constants",
    "exf_climatology_source",
    "surface_climatology_source",
    "reference_profile_from_tensors",
    "resolve_reference_profiles",
    "runtime_kwargs",
    "implicit_vertical_advection_stepper_kwargs",
    "implicit_vertical_advection_identity",
    "enrich_manifest",
]


def seconds_from_time_coord(values: Any) -> np.ndarray:
    """Convert datetime64 / numeric coord into seconds since the first sample."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return ((arr - arr[0]) / np.timedelta64(1, "s")).astype(np.float64)
    return arr.astype(np.float64)


class BarotropicBoundaryBundleSource:
    """Wraps a BoundaryBundleSource to expose ``eta`` only.

    ``sample()`` returns an object carrying just an ``eta`` attribute (an empty
    ``SideBoundaryTensorSet`` when the bundle has no ``eta`` field). It does NOT
    provide ``normal_velocity`` or any tracer/velocity field, so it cannot drive
    a Flather barotropic condition; it is used by the ``prescribed_3d`` +
    ``use_barotropic_source`` path that only reads ``eta``.
    """

    sampling_rule = SamplingRule(SampleClock.END, providers=("_source",))

    def __init__(self, source: BoundaryBundleSource) -> None:
        self._source = source

    def sample(self, model_time_seconds: float, **_: Any) -> Any:
        from defgcm.boundaries.open import SideBoundaryTensorSet

        try:
            eta = self._source.sample_field("eta", model_time_seconds)
        except (KeyError, ValueError):
            eta = SideBoundaryTensorSet()
        return type("_BarotropicBoundarySample", (), {"eta": eta})()

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "BarotropicBoundaryBundleSource":
        return BarotropicBoundaryBundleSource(self._source.to_runtime(device, dtype))

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "BarotropicBoundaryBundleSource":
        return self.to_runtime(device, dtype)


def boundary_source_from_bundle(
    bundle: Any,
    *,
    side_names: tuple[str, ...],
    device: str | torch.device,
    dtype: torch.dtype,
    n_steps: int | None = None,
    dt: float | None = None,
) -> BoundaryBundleSource:
    """Build a BoundaryBundleSource from a prepared_boundary DataBundle.

    ``n_steps``/``dt`` are only consulted by the runtime path: when the
    bundle has a single time sample, those args let the loader replicate it
    so the integrator's time-interpolator has a valid pair of endpoints.
    The canonical writer always passes both as None — boundary cycles are
    fully populated in stage1a and don't need extension.
    """
    data = bundle.data
    coord = data.coords.get("time_seconds", data.coords.get("time", None))
    if coord is None:
        raise ValueError("boundary bundle must contain a time coordinate")
    times = seconds_from_time_coord(coord.values)
    fields = {
        name: torch.as_tensor(np.asarray(data[name].values), dtype=dtype, device=device)
        for name in ("temperature", "salinity", "u", "v", "w", "eta")
        if name in data
    }
    if n_steps is not None and dt is not None and times.size == 1:
        end_time = max(float(times[0]), float(n_steps) * float(dt))
        times = np.asarray([float(times[0]), end_time], dtype=np.float64)
        fields = {
            name: torch.cat([tensor, tensor], dim=0) for name, tensor in fields.items()
        }
    extras = getattr(bundle.meta, "extras", {})
    valid_along_raw = extras.get("valid_along", None)
    if valid_along_raw is None and "along_valid" in getattr(data, "coords", {}):
        # Fallback to the along_valid coord (survives the zarr round-trip even
        # when meta.extras does not). Indexed by the `side` coord.
        av = np.asarray(data["along_valid"].values)
        side_vals = (
            [str(s) for s in data["side"].values]
            if "side" in data.coords else list(side_names)
        )
        valid_along_raw = {side_vals[i]: int(av[i]) for i in range(len(av))}
    valid_along = (
        {str(k): int(v) for k, v in dict(valid_along_raw).items()}
        if valid_along_raw is not None
        else None
    )
    forcing_period = extras.get("forcing_period_seconds")
    forcing_cycle = extras.get("forcing_cycle_seconds")
    return BoundaryBundleSource(
        times_seconds=tuple(float(v) for v in times),
        side_names=tuple(str(s) for s in side_names),
        fields=fields,
        valid_along=valid_along,
        forcing_period_seconds=None if forcing_period is None else float(forcing_period),
        forcing_cycle_seconds=None if forcing_cycle is None else float(forcing_cycle),
    )


def _forcing_field_values(
    data: Any,
    name: str,
    time_indices: np.ndarray,
) -> np.ndarray:
    field = data[name]
    if time_indices.size != field.shape[0]:
        # Integer indexing owns the selected records even for a single frame;
        # a [:1] view could keep the entire archive alive at matching dtype.
        field = field.isel({field.dims[0]: time_indices})
    return np.asarray(field.values, dtype=np.float64)


def wind_source_from_bundle(
    bundle: Any,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    time_mode: str = "linear",
    end_time_seconds: float | None = None,
) -> Any:
    """Build an ERA5WindSource from a prepared_forcing DataBundle.

    ``end_time_seconds`` truncates the time series to the active integration
    window; passed only at runtime. The canonical writer leaves it None and keeps the full record.

    A bundle that declares ``forcing_cycle_seconds`` in ``meta.extras`` (MITgcm
    ``externForcingCycle``) is *cyclic*: its records are a closed loop, not a
    prefix of a longer archive, so the window truncation is skipped and the
    cycle length is handed to the source as ``period_seconds``.  Truncating a
    loop to the first records inside the run window would silently replace the
    cycle with whichever record happened to come first.
    """
    from defgcm.dynamics.bulk_formulae import wind_speed_to_stress
    from defgcm.io.era5 import ERA5WindSource

    data = bundle.data
    coord = data.coords.get("time_seconds", data.coords.get("time", None))
    if coord is None:
        raise ValueError("forcing source must provide time or time_seconds coordinate")
    times = seconds_from_time_coord(coord.values)
    time_indices = np.arange(times.size)
    raw_cycle = dict(getattr(bundle.meta, "extras", {}) or {}).get(
        "forcing_cycle_seconds"
    )
    cycle_seconds = None if raw_cycle in (None, 0, 0.0) else float(raw_cycle)
    if end_time_seconds is not None and cycle_seconds is None:
        keep = times <= float(end_time_seconds) + 1.0e-9
        if np.any(keep):
            times = times[keep]
            time_indices = time_indices[keep]
    if time_mode == "constant":
        times = times[:1]
        time_indices = time_indices[:1]
    elif time_mode != "linear":
        raise ValueError("forcing time_mode must be 'linear' or 'constant'")
    if "tau_x" in data and "tau_y" in data:
        tau_x = _forcing_field_values(data, "tau_x", time_indices)
        tau_y = _forcing_field_values(data, "tau_y", time_indices)
    elif "wind_u" in data and "wind_v" in data:
        tau_x_t, tau_y_t = wind_speed_to_stress(
            torch.as_tensor(
                _forcing_field_values(data, "wind_u", time_indices),
                dtype=dtype,
                device=device,
            ),
            torch.as_tensor(
                _forcing_field_values(data, "wind_v", time_indices),
                dtype=dtype,
                device=device,
            ),
        )
        tau_x = tau_x_t.detach().cpu().numpy()
        tau_y = tau_y_t.detach().cpu().numpy()
    else:
        raise ValueError("forcing source must provide tau_x/tau_y or wind_u/wind_v fields")
    time_values = tuple(str(v) for v in np.asarray(coord.values)[: len(times)])
    return ERA5WindSource(
        times=torch.as_tensor(times, dtype=torch.float64, device=device),
        time_values=time_values,
        tau_x=torch.as_tensor(tau_x, dtype=dtype, device=device),
        tau_y=torch.as_tensor(tau_y, dtype=dtype, device=device),
        period_seconds=cycle_seconds,
    )


def surface_flux_source_from_bundle(
    bundle: Any,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    time_mode: str = "linear",
    end_time_seconds: float | None = None,
) -> Any | None:
    """Build a prescribed WPFlux source from a prepared forcing bundle."""
    from defgcm.dynamics.surface_flux import PrescribedSurfaceFluxSource

    data = bundle.data
    if "heat_flux" not in data and "freshwater_flux" not in data:
        return None
    coord = data.coords.get("time_seconds", data.coords.get("time", None))
    if coord is None:
        raise ValueError("surface flux source must provide time or time_seconds coordinate")
    times = seconds_from_time_coord(coord.values)
    time_indices = np.arange(times.size)
    raw_cycle = dict(getattr(bundle.meta, "extras", {}) or {}).get(
        "forcing_cycle_seconds"
    )
    cycle_seconds = None if raw_cycle in (None, 0, 0.0) else float(raw_cycle)
    if end_time_seconds is not None and cycle_seconds is None:
        keep = times <= float(end_time_seconds) + 1.0e-9
        if np.any(keep):
            times = times[keep]
            time_indices = time_indices[keep]
    if time_mode == "constant":
        times = times[:1]
        time_indices = time_indices[:1]
    elif time_mode != "linear":
        raise ValueError("forcing time_mode must be 'linear' or 'constant'")
    if "heat_flux" in data:
        heat_flux = _forcing_field_values(data, "heat_flux", time_indices)
        freshwater_flux = (
            _forcing_field_values(data, "freshwater_flux", time_indices)
            if "freshwater_flux" in data
            else np.zeros_like(heat_flux)
        )
    else:
        freshwater_flux = _forcing_field_values(data, "freshwater_flux", time_indices)
        heat_flux = np.zeros_like(freshwater_flux)
    time_values = tuple(str(value) for value in np.asarray(coord.values)[: len(times)])
    return PrescribedSurfaceFluxSource(
        times=torch.as_tensor(times, dtype=torch.float64, device=device),
        time_values=time_values,
        heat_flux=torch.as_tensor(heat_flux, dtype=dtype, device=device),
        freshwater_flux=torch.as_tensor(
            freshwater_flux, dtype=dtype, device=device
        ),
        period_seconds=cycle_seconds,
    )


def uses_surface_pressure_load(run_cfg: Any) -> bool:
    """Whether a hydrostatic_pressure_gradient entry declares MITgcm pLoadFile."""
    return any(
        entry.name == "hydrostatic_pressure_gradient"
        and bool(dict(entry.kwargs).get("surface_pressure_load", False))
        for entry in run_cfg.model.dynamics
    )


def surface_pressure_load_from_bundle(
    bundle: Any,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """MITgcm ``pLoad`` [ny, nx] Pa from a prepared forcing bundle.

    Only the constant field of ``periodicExternalForcing=.FALSE.`` is ported
    (``ini_forcing.F:191-195`` reads it once); a time-varying record set
    (``external_fields_load.F:206-233,327-332``) is refused.
    """
    field = bundle.data["pload"]
    if field.shape[0] != 1:
        raise ValueError(
            f"surface pressure load: {field.shape[0]} records; only a constant pLoad "
            "(periodicExternalForcing=.FALSE.) is ported"
        )
    return torch.as_tensor(
        np.asarray(field.values[0], dtype=np.float64), dtype=dtype, device=device
    )


def uses_era5_bulk_forcing(run_cfg: Any) -> bool:
    """Whether ``build_bulk_source`` reads the ERA5 archive (the case lists ``era5_bulk_forcing``)."""
    return any(getattr(e, "name", "") == "era5_bulk_forcing" for e in run_cfg.model.dynamics)


# Source types read here from ``run_cfg.sources`` at assembly
# (surface_climatology_source, exf_bulk_source, exf_flux_sources,
# exf_climatology_source); Layer B does not instantiate them.
ASSEMBLY_SOURCE_TYPES = ("mit_surface_climatology", "mit_exf", "mit_exf_flux",
                         "mit_exf_climatology")

_EXF_BULK_FIELDS = ("atemp", "aqh", "uwind", "vwind", "swdown", "lwdown", "lwflux", "swflux",
                    "precip", "snowprecip", "runoff")
_EXF_FLUX_FIELDS = ("hflux", "sflux", "ustress", "vstress", "swflux")
_EXF_CLIMATOLOGY_FIELDS = ("climsst", "climsss")


def _exf_field_series(
    cfg: Mapping[str, Any],
    name: str,
    *,
    ny: int,
    nx: int,
    layout: "ParallelLayout | None",
    dt_clock: float,
) -> Any:
    """One ``fields`` entry of an EXF source config as a :class:`FieldSeries`.

    Every record of the file is read big-endian at ``exf_iprec``.  With
    ``use_cal`` a file field of positive period takes its start from
    ``startdate1/2`` (exf_getffield_start.F:80-104), otherwise from
    ``start_time``.  Ocean-only EXF masks these fields with 'c'
    (exf_init_fixed.F:52-83).  A file whose
    period is positive and shorter than ``deltaTClock`` is refused: the source
    reads records ``count0``/``count1`` directly, which equals EXF_SET_FLD's
    record swap (exf_set_fld.F:184-242) only while a step advances at most one
    record.
    """
    from pathlib import Path

    from defgcm.io.exf_source import FieldSeries, exf_field_start_time
    from defgcm.io.runner.layer_b.sources.mit_input_binary import mit_binary_dtype

    item = dict(dict(cfg["fields"])[name])
    period = float(item["period"])
    start_time = float(cfg["start_time"])
    filename = item["file"]
    # EXF_GETFFIELD_START runs only for a field with a file (exf_init_fixed.F:218 for atemp).
    if filename and bool(cfg["use_cal"]) and period > 0.0:
        field_start = exf_field_start_time(
            int(item["startdate1"]), int(item["startdate2"]), calendar=str(cfg["calendar"]),
            base_date1=int(cfg["start_date_1"]), base_date2=int(cfg["start_date_2"]),
            start_time=start_time)
    else:
        field_start = float(item["start_time"])
    if filename and 0.0 < period < dt_clock:
        raise ValueError(f"EXF field {name!r}: period {period!r} is shorter than deltaTClock "
                         f"{dt_clock!r}; a step would skip records of EXF_SET_FLD's swap")
    if filename:
        values = np.fromfile(Path(str(cfg["path"])) / filename,
                             dtype=mit_binary_dtype(int(cfg["exf_iprec"])))
        values = values.reshape(-1, ny, nx).astype(np.float64)
    else:
        values = np.zeros((0, ny, nx), dtype=np.float64)
    if layout is not None:
        values = values[:, layout.y_start:layout.y_stop, layout.x_start:layout.x_stop]
    return FieldSeries(
        torch.as_tensor(np.ascontiguousarray(values)), bool(filename), float(item["const"]),
        period, field_start, float(item["rep_cycle"]), float(item["inscale"]),
        float(item["remove_intercept"]), float(item["remove_slope"]), True)


def _exf_config(run_cfg: Any, spec: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """The source config after the checks both EXF sources share.

    ``exf_options`` carries the build's EXF_OPTIONS.h macros of
    ``EXF_OPTIONS_UNDEFINED``; each must be false.
    """
    from defgcm.io.exf_source import EXF_OPTIONS_UNDEFINED

    if run_cfg.grid.kind != "latlon":
        raise ValueError(f"{spec.type} reads model-grid records of a lat-lon grid only")
    cfg = dict(spec.config)
    options = dict(cfg["exf_options"])
    if set(options) != set(EXF_OPTIONS_UNDEFINED) or any(bool(v) for v in options.values()):
        raise ValueError(f"{spec.type} exf_options must set exactly {EXF_OPTIONS_UNDEFINED} "
                         f"to false (branches not ported), got {options}")
    if set(dict(cfg["fields"])) != set(names):
        raise ValueError(f"{spec.type} fields must be exactly {names}, got {sorted(cfg['fields'])}")
    return cfg


def exf_bulk_source(run_cfg: Any, spec: Any) -> Any:
    """EXF bulk-formula inputs of a ``mit_exf`` source on the model tracer grid.

    The config names the input directory (``path``), ``exf_iprec``, ``use_cal``,
    MITgcm's ``start_time``, ``exf_offset_atemp``, ``exf_options`` (see
    :func:`_exf_config`), the data.exf bulk ``constants`` (read by
    :func:`bulk_kernel_constants`) and, with ``use_cal``, the ``calendar`` and
    ``start_date_1``/``start_date_2`` of data.cal.  ``fields``
    holds exactly the bulk inputs, each with ``file`` (empty for none),
    ``const``, ``period``, ``start_time`` (used without the calendar),
    ``startdate1``/``startdate2`` (used with it), ``rep_cycle``, ``inscale``,
    ``remove_intercept`` and ``remove_slope``: the data.exf values or their
    defaults (exf_readparms.F).
    """
    from defgcm.io.exf_source import ExfBulkSource

    cfg = _exf_config(run_cfg, spec, _EXF_BULK_FIELDS)
    ny, nx = int(run_cfg.grid.ny), int(run_cfg.grid.nx)
    return ExfBulkSource(
        **{name: _exf_field_series(cfg, name, ny=ny, nx=nx, layout=None,
                                   dt_clock=float(run_cfg.time.dt_clock))
           for name in _EXF_BULK_FIELDS},
        atemp_offset=float(cfg["exf_offset_atemp"]), use_cal=bool(cfg["use_cal"]),
        start_time=float(cfg["start_time"]),
    )


def bulk_kernel_constants(run_cfg: Any) -> Any:
    """``era5_bulk_forcing``'s :class:`ExfBulkConstants`.

    The ``constants`` of the case's ``mit_exf`` source hold every field of
    ``ExfBulkConstants`` with its data.exf value; without a ``mit_exf`` source
    (ERA5) the kernel uses the EXF defaults.
    """
    from dataclasses import fields as dataclass_fields

    from defgcm.dynamics.bulk_formulae import EXF_BULK_DEFAULTS, ExfBulkConstants

    specs = [spec for spec in run_cfg.sources if spec.type == "mit_exf"]
    if not specs:
        return EXF_BULK_DEFAULTS
    constants = {name: float(value) for name, value in dict(dict(specs[0].config)["constants"]).items()}
    names = {item.name for item in dataclass_fields(ExfBulkConstants)}
    if set(constants) != names:
        raise ValueError(f"mit_exf constants must be exactly {sorted(names)}, got {sorted(constants)}")
    return ExfBulkConstants(**constants)


def exf_flux_sources(
    run_cfg: Any,
    spec: Any,
    *,
    layout: "ParallelLayout | None" = None,
) -> tuple[Any, Any]:
    """Heat/freshwater and stress sources of a ``mit_exf_flux`` source (no ALLOW_ATM_TEMP).

    The config keys are those of :func:`exf_bulk_source` without
    ``exf_offset_atemp`` and ``constants``, plus ``exf_outscal_hflux``, ``exf_outscal_sflux``,
    ``exf_outscal_swflux``, ``exf_outscal_ustress`` and ``exf_outscal_vstress``; ``fields`` holds
    exactly hflux, sflux, ustress, vstress and swflux.  The stress files are
    cell-centred (stressIsOnCgrid false).
    """
    from defgcm.io.exf_source import ExfFluxSource, ExfStressSource

    cfg = _exf_config(run_cfg, spec, _EXF_FLUX_FIELDS)
    if float(cfg["exf_outscal_ustress"]) != 1.0 or float(cfg["exf_outscal_vstress"]) != 1.0:
        raise ValueError("era5_wind_forcing averages the stress without exf_outscal_ustress/vstress")
    ny, nx = int(run_cfg.grid.ny), int(run_cfg.grid.nx)
    series = {name: _exf_field_series(cfg, name, ny=ny, nx=nx, layout=layout,
                                      dt_clock=float(run_cfg.time.dt_clock))
              for name in _EXF_FLUX_FIELDS}
    use_cal, start_time = bool(cfg["use_cal"]), float(cfg["start_time"])
    return (
        ExfFluxSource(hflux=series["hflux"], sflux=series["sflux"], swflux=series["swflux"],
                      hflux_scale=float(cfg["exf_outscal_hflux"]),
                      sflux_scale=float(cfg["exf_outscal_sflux"]),
                      swflux_scale=float(cfg["exf_outscal_swflux"]),
                      use_cal=use_cal, start_time=start_time),
        ExfStressSource(ustress=series["ustress"], vstress=series["vstress"],
                        use_cal=use_cal, start_time=start_time),
    )


def exf_climatology_source(
    run_cfg: Any,
    spec: Any,
    *,
    layout: "ParallelLayout | None" = None,
) -> Any:
    """SST/SSS targets of ``surface_climate_relaxation`` from a ``mit_exf_climatology`` source.

    The config keys are those of :func:`exf_flux_sources` without the
    ``exf_outscal_*`` flux scales, plus ``climtempfreeze``, ``exf_outscal_sst``
    and ``exf_outscal_sss``; ``fields`` holds exactly climsst and climsss.
    The module's ``tau_theta``/``tau_salt`` are ``climsstTauRelax``/
    ``climsssTauRelax`` (exf_readparms.F:1066-1091).
    """
    from defgcm.io.exf_source import ExfClimatologySource

    cfg = _exf_config(run_cfg, spec, _EXF_CLIMATOLOGY_FIELDS)
    ny, nx = int(run_cfg.grid.ny), int(run_cfg.grid.nx)
    series = {name: _exf_field_series(cfg, name, ny=ny, nx=nx, layout=layout,
                                      dt_clock=float(run_cfg.time.dt_clock))
              for name in _EXF_CLIMATOLOGY_FIELDS}
    return ExfClimatologySource(
        climsst=series["climsst"], climsss=series["climsss"],
        climtempfreeze=float(cfg["climtempfreeze"]),
        sst_scale=float(cfg["exf_outscal_sst"]), sss_scale=float(cfg["exf_outscal_sss"]),
        use_cal=bool(cfg["use_cal"]), start_time=float(cfg["start_time"]),
    )


def build_bulk_source(
    run_cfg: Any,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    window_seconds: int | None = None,
) -> Any | None:
    """Model-grid ERA5 bulk-formula source for the ``era5_bulk_forcing`` module.

    A case with a ``mit_exf`` source gets the EXF binary fields instead
    (:func:`exf_bulk_source`).

    Direct ERA5 nc read (8 bulk vars) + load-time bilinear regrid onto the model
    lat-lon grid, so the runtime module only time-interpolates (mirrors the prep
    wind-stress bake — no per-step regrid).  Shared by ``model_input`` and the runtime
    (``_runtime``).  Returns ``None`` unless the case lists ``era5_bulk_forcing`` (so
    cube / non-lat-lon cases are unaffected).  Assumes a uniform spherical-polar grid.

    ``window_seconds`` bounds the ERA5 read window (default: the full ``time.seconds``);
    callers with the real step count pass ``n_steps*dt + buffer`` to avoid loading the
    whole archive for a short run.
    """
    from defgcm.io.era5 import load_era5_bulk_source_by_time

    if not uses_era5_bulk_forcing(run_cfg):
        return None
    exf_specs = [spec for spec in run_cfg.sources if spec.type == "mit_exf"]
    if exf_specs:
        if len(exf_specs) != 1:
            raise ValueError("era5_bulk_forcing takes at most one `mit_exf` source")
        return exf_bulk_source(run_cfg, exf_specs[0]).to_runtime(device, dtype)
    era5_dir = None
    for src in run_cfg.sources:
        if getattr(src, "name", None) == "era5" or getattr(src, "type", None) == "era5_netcdf":
            era5_dir = dict(src.config).get("path")
            break
    if era5_dir is None:
        raise ValueError("era5_bulk_forcing requires an `era5` source with a `path`")
    g = run_cfg.grid
    res, ny, nx = float(g.resolution_deg), int(g.ny), int(g.nx)
    tlat = (float(g.center_lat) - ny / 2.0 * res) + (np.arange(ny) + 0.5) * res
    tlon = (float(g.center_lon) - nx / 2.0 * res) + (np.arange(nx) + 0.5) * res
    sd = run_cfg.run.start_date
    start = datetime(sd.year, sd.month, sd.day)
    # +1 ERA5 cadence (6 h) so sample() always brackets a pair through the last step.
    window = int(window_seconds if window_seconds is not None else float(run_cfg.time.seconds)) + 6 * 3600
    source = load_era5_bulk_source_by_time(
        era5_dir, start_date=start, window_seconds=window,
        lat_box=(float(tlat.min()) - 1.0, float(tlat.max()) + 1.0),
        target_lat=tlat, target_lon=tlon,
    )
    return source.to_runtime(device, dtype)


def surface_climatology_source(
    run_cfg: Any,
    *,
    layout: "ParallelLayout | None" = None,
) -> Any:
    """SST/SSS records of ``surface_climate_relaxation`` from its ``mit_surface_climatology`` source.

    The source config names the input directory (``path``), ``read_binary_prec``,
    ``theta_clim_file`` / ``salt_clim_file`` (empty for none), the core
    ``periodic_external_forcing`` cadence (``extern_forcing_period`` /
    ``extern_forcing_cycle``) and MITgcm's ``start_time``.  A periodic run reads
    ``NINT(cycle/period)`` records, otherwise the first one
    (``ini_forcing.F:100-105``, ``external_fields_load.F:162-173``).
    """
    from pathlib import Path

    from defgcm.io.forcing_records import SurfaceClimatologySource
    from defgcm.io.runner.layer_b.sources.mit_input_binary import (
        _read_forcing_records,
        mit_binary_dtype,
    )

    specs = [spec for spec in run_cfg.sources if spec.type == "mit_surface_climatology"]
    if len(specs) != 1:
        raise ValueError(
            "surface_climate_relaxation requires exactly one `mit_surface_climatology` source")
    if run_cfg.grid.kind != "latlon":
        raise ValueError("surface_climate_relaxation reads tracer-grid records of a lat-lon grid only")
    cfg = dict(specs[0].config)
    periodic = bool(cfg["periodic_external_forcing"])
    period = float(cfg["extern_forcing_period"])
    cycle = float(cfg["extern_forcing_cycle"])
    count = int(round(cycle / period)) if periodic else 1
    dtype = mit_binary_dtype(int(cfg["read_binary_prec"]))
    ny, nx = int(run_cfg.grid.ny), int(run_cfg.grid.nx)

    def records(key: str) -> torch.Tensor:
        filename = cfg[key]
        if not filename:
            values = np.zeros((0, ny, nx), dtype=np.float64)
        else:
            values = _read_forcing_records(
                Path(str(cfg["path"])) / filename, filename, (ny, nx), dtype, count)
        if layout is not None:
            values = values[:, layout.y_start:layout.y_stop, layout.x_start:layout.x_stop]
        return torch.as_tensor(np.ascontiguousarray(values))

    return SurfaceClimatologySource(
        sst=records("theta_clim_file"), sss=records("salt_clim_file"),
        periodic=periodic, period=period, cycle=cycle, start_time=float(cfg["start_time"]),
    )


def reference_profile_from_tensors(
    ic_tensors: dict[str, torch.Tensor],
    name: str,
) -> torch.Tensor | None:
    """Per-layer T/S reference profile from the IC tensors."""
    tensor_name = "temperature" if name == "reference_temperature" else "salinity"
    values = ic_tensors.get(tensor_name)
    if values is None:
        return None
    if values.ndim == 4:
        values = values[0]
    if values.ndim != 3:
        return None
    profile_values: list[float] = []
    for k in range(values.shape[0]):
        layer = values[k]
        valid = torch.isfinite(layer) & (layer.abs() > 1.0e-12)
        if not bool(valid.any()):
            profile_values.append(0.0)
            continue
        valid_values = layer[valid]
        first = valid_values.flatten()[0]
        if bool(torch.all(valid_values == first)):
            profile_values.append(float(first.item()))
        else:
            profile_values.append(float(valid_values.mean().item()))
    profile = torch.as_tensor(profile_values, dtype=torch.float64, device=values.device)
    if name == "reference_salinity":
        profile = torch.where(profile == 0.0, torch.full_like(profile, 35.0), profile)
    return profile


def _reference_profile_from_ic_source(
    run_cfg: Any,
    name: str,
) -> torch.Tensor | None:
    """Return a profile declared by the state pipeline's primary source."""
    field = "t_ref" if name == "reference_temperature" else "s_ref"
    state_pipeline = next(
        (
            pipeline
            for pipeline in run_cfg.pipelines.get("stage1a_prepare_common", ())
            if pipeline.name == "state"
        ),
        None,
    )
    if state_pipeline is None:
        return None
    source = next(
        (
            source
            for source in run_cfg.sources
            if source.name == state_pipeline.primary_source
        ),
        None,
    )
    if source is None:
        return None
    raw = dict(source.config).get(field)
    if raw is None:
        return None
    profile = torch.as_tensor(raw, dtype=torch.float64)
    nz = int(run_cfg.grid.nz)
    if profile.ndim == 0:
        return torch.full((nz,), float(profile.item()), dtype=torch.float64)
    if profile.shape != (nz,):
        raise ValueError(
            f"sources.{source.name}.{field} shape {tuple(profile.shape)} != {(nz,)}"
        )
    return profile


#: Literal sequence spellings the kwargs boundary (``Mapping[str, Any]``) can
#: carry for a y-profile.  YAML yields lists; a programmatic manifest yields a
#: tuple just as legitimately, and both must reach the slicer.
#: ``str``/``bytes`` are sequences too but never a
#: profile, and they are excluded by naming the two container types.
_LITERAL_SEQUENCE_TYPES = (list, tuple)

_LITERAL_Y_PROFILE_KEYS: dict[str, bool] = {
    # key -> corner (True: yG points, global_ny+1; False: centre, global_ny)
    "coriolis_profile_g": True,
    "tan_phi_v": True,
    "coriolis_profile_c": False,
    "tan_phi_u": False,
    # Stepper override of KPP's cell-centre f (prepare_kpp_static).
    "kpp_coriolis_profile_c": False,
}


def _maybe_slice_literal_profile(
    key: str, values: tuple, layout: "ParallelLayout | None"
) -> tuple:
    """Yaml-LITERAL y-profiles must be sliced too.

    A committed case can pass the four semantic profile kwargs as explicit
    lists (e.g. tutorial_barotropic_gyre's coriolis_profile_c); built at global
    length they fail the consumers' rank-local length checks under nPy>=2.
    For those keys, when a layout is present, validate the GLOBAL length and
    slice with the same helper as the sentinels.  Other list kwargs pass
    through untouched; no layout means single-domain semantics (unchanged).

    The caller admits every ``_LITERAL_SEQUENCE_TYPES``
    spelling, not lists alone.  Keying the branch on ``list`` left a
    programmatic tuple of the SAME semantic key at global length, and the only
    signal was the consumer's own check much later
    (``dynamics/coriolis.py`` ``FCoriolis.evaluate``: "coriolis_profile_c
    length 8 != ny=4").
    """
    if layout is None or key not in _LITERAL_Y_PROFILE_KEYS:
        return values
    corner = _LITERAL_Y_PROFILE_KEYS[key]
    expected = int(layout.global_ny) + (1 if corner else 0)
    if len(values) != expected:
        raise ValueError(
            f"literal {key} length {len(values)} != global "
            f"{'ny+1' if corner else 'ny'} = {expected} required for "
            f"distributed y-localization"
        )
    return _slice_y_profile(values, layout, corner=corner)


def _slice_y_profile(
    values: tuple, layout: "Any | None", *, corner: bool
) -> tuple:
    """Slice a GLOBAL y-profile tuple to this rank's y range.

    Distributed preparation: the $-sentinel profiles are built
    at global ny(+1) but every consumer (KPP forcing cache, VI corner field,
    spherical metric) validates/views them against the rank-LOCAL ny.  Slice —
    never recompute from a local origin — so npy=1 stays bitwise-identical and
    npy>=2 ranks read exactly the same float values the global build produced.
    Corner profiles (yG points) take [y_start, y_stop] inclusive = ny_local+1;
    centre profiles take [y_start, y_stop) = ny_local.
    """
    if layout is None:
        return values
    y0 = int(layout.y_start)
    y1 = int(layout.y_stop)
    return values[y0 : y1 + 1] if corner else values[y0:y1]


#: Dynamics modules whose ``rotation_period`` field the canonical assembly fills
#: from ``physics.rotation_period``.
_ROTATING_MOMENTUM_MODULES = frozenset({"vector_invariant_momentum_advection", "flux_form_momentum"})

#: Flux-form momentum modules whose ``select_r_star`` field the canonical assembly
#: fills from the run's stepper (mom_u_adv_wu.F:93, mom_v_adv_wv.F:94).
_R_STAR_MOMENTUM_MODULES = frozenset({"momentum_advection", "flux_form_momentum"})


def stepper_select_r_star(stepper_name: str, stepper_kwargs: Mapping[str, Any]) -> int:
    """MITgcm ``select_rStar`` a stepper runs: its kwarg, else the stepper
    class's own value; a stepper without the field runs z coordinates (0)."""
    from dataclasses import fields as dataclass_fields

    import defgcm.steppers  # noqa: F401  (registers every stepper)
    from defgcm.steppers._registry import STEPPERS

    if "select_r_star" in stepper_kwargs:
        return int(stepper_kwargs["select_r_star"])
    for f in dataclass_fields(STEPPERS.get(stepper_name)):
        if f.name == "select_r_star":
            return int(f.default)
    return 0


def complete_stepper_owned_kwargs(
    name: str, kwargs: dict[str, Any], *, stepper_name: str, stepper_kwargs: Mapping[str, Any],
) -> None:
    """Write into ``kwargs`` of dynamics entry ``name`` the fields its stepper owns.

    The momentum cores take the stepper's select_r_star, the run's one
    select_rStar; an entry that sets it itself is refused.  ``enrich_manifest``
    and the MITgcm translator's trial construction both call this.
    """
    if name in _R_STAR_MOMENTUM_MODULES:
        if "select_r_star" in kwargs:
            raise ValueError(
                f"model.dynamics[{name}].kwargs.select_r_star: set "
                "model.stepper.kwargs.select_r_star, the run's one select_rStar"
            )
        kwargs["select_r_star"] = stepper_select_r_star(stepper_name, stepper_kwargs)


def runtime_kwargs(
    run_cfg: Any, kwargs: Mapping[str, Any], *, layout: "ParallelLayout | None" = None
) -> dict[str, Any]:
    """Resolve $coriolis_* sentinels and dict-typed structured kwargs.

    ``layout`` (ParallelLayout) y-localizes the four $-profile sentinels for
    distributed ranks via :func:`_slice_y_profile`; yaml-literal profile lists
    are NOT sliced (no production case passes one).
    """
    # Every scalar/profile below is cast to a genuine Python ``float``.  numpy
    # scalars are IEEE doubles too, so the cast is value-preserving -- but a
    # tuple of ``np.float64`` is NOT interchangeable with the declared
    # ``tuple[float, ...]`` under ``torch.compile``: Dynamo wraps each numpy
    # scalar with ``torch.from_numpy`` and installs a dispatch-key guard on it,
    # which then fails with "dispatch key set mismatch" and aborts the step.
    # The y-profiles hit this; the two Coriolis profiles and the scalar did not.
    from defgcm.constants import omega_from_rotation_period

    omega = omega_from_rotation_period(float(run_cfg.physics.rotation_period))
    resolved = dict(kwargs)
    for key, value in list(resolved.items()):
        if value == "$coriolis_parameter":
            resolved[key] = float(
                2.0 * omega * np.sin(np.deg2rad(float(run_cfg.grid.center_lat)))
            )
        elif value == "$coriolis_profile_g":
            yg_origin = float(run_cfg.grid.center_lat) - (int(run_cfg.grid.ny) / 2.0) * float(run_cfg.grid.resolution_deg)
            resolved[key] = _slice_y_profile(
                tuple(
                    float(
                        2.0
                        * omega
                        * np.sin(np.deg2rad(yg_origin + j * float(run_cfg.grid.resolution_deg)))
                    )
                    for j in range(int(run_cfg.grid.ny) + 1)
                ),
                layout,
                corner=True,
            )
        elif value == "$coriolis_profile_c":
            # Flux-form Coriolis: CELL-CENTRE f = 2Ω·sin(yC(j)), yC(j)=yG(0)+(j+½)·dlat.
            # (distinct from the vorticity-corner $coriolis_profile_g used by VI.)
            res = float(run_cfg.grid.resolution_deg)
            yg_origin = float(run_cfg.grid.center_lat) - (int(run_cfg.grid.ny) / 2.0) * res
            resolved[key] = _slice_y_profile(
                tuple(
                    float(2.0 * omega * np.sin(np.deg2rad(yg_origin + (j + 0.5) * res)))
                    for j in range(int(run_cfg.grid.ny))
                ),
                layout,
                corner=False,
            )
        elif value == "$tan_phi_u":
            # Spherical metric: tan(yC(j)) at U-points, length ny.
            res = float(run_cfg.grid.resolution_deg)
            yg_origin = float(run_cfg.grid.center_lat) - (int(run_cfg.grid.ny) / 2.0) * res
            resolved[key] = _slice_y_profile(
                tuple(
                    float(np.tan(np.deg2rad(yg_origin + (j + 0.5) * res)))
                    for j in range(int(run_cfg.grid.ny))
                ),
                layout,
                corner=False,
            )
        elif value == "$tan_phi_v":
            # Spherical metric: tan(yG(j)) at V-points, length ny+1.
            res = float(run_cfg.grid.resolution_deg)
            yg_origin = float(run_cfg.grid.center_lat) - (int(run_cfg.grid.ny) / 2.0) * res
            resolved[key] = _slice_y_profile(
                tuple(
                    float(np.tan(np.deg2rad(yg_origin + j * res)))
                    for j in range(int(run_cfg.grid.ny) + 1)
                ),
                layout,
                corner=True,
            )
        elif isinstance(value, _LITERAL_SEQUENCE_TYPES):
            resolved[key] = _maybe_slice_literal_profile(
                key, tuple(value), layout
            )
        elif key == "nonhydrostatic" and isinstance(value, dict):
            from defgcm.dycore_parameters import NonhydrostaticParams

            resolved[key] = NonhydrostaticParams(**value)
    return resolved


_IMPLICIT_VERTICAL_ADVECTION_FIELDS = (
    ("temperature", "temperature_implicit_vertical_advection"),
    ("salinity", "salinity_implicit_vertical_advection"),
    ("momentum", "momentum_implicit_vertical_advection"),
)


def implicit_vertical_advection_stepper_kwargs(run_cfg: Any) -> dict[str, bool]:
    """Derive the three exact IFS kwargs from the canonical nested block."""
    config = run_cfg.physics.implicit_vertical_advection
    derived: dict[str, bool] = {}
    for leaf, flat_key in _IMPLICIT_VERTICAL_ADVECTION_FIELDS:
        value = getattr(config, leaf)
        if type(value) is not bool:
            raise ValueError(
                f"physics.implicit_vertical_advection.{leaf} must be bool, "
                f"got {type(value).__name__}"
            )
        derived[flat_key] = value

    duplicate_paths: list[str] = []
    stepper_kwargs = dict(run_cfg.model.stepper.kwargs)
    for _, flat_key in _IMPLICIT_VERTICAL_ADVECTION_FIELDS:
        if flat_key in stepper_kwargs:
            duplicate_paths.append(f"model.stepper.kwargs.{flat_key}")
    for index, entry in enumerate(run_cfg.model.dynamics):
        kwargs = dict(entry.kwargs)
        for _, flat_key in _IMPLICIT_VERTICAL_ADVECTION_FIELDS:
            if flat_key in kwargs:
                duplicate_paths.append(
                    f"model.dynamics[{index}:{entry.name}].kwargs.{flat_key}"
                )
    if duplicate_paths:
        raise ValueError(
            "implicit vertical advection has duplicate derived source(s): canonical "
            "physics.implicit_vertical_advection and "
            + ", ".join(duplicate_paths)
        )
    return derived


def implicit_vertical_advection_identity(run_cfg: Any) -> dict[str, bool]:
    """Return the normalized canonical flags for runtime identity writers."""
    flat = implicit_vertical_advection_stepper_kwargs(run_cfg)
    return {
        leaf: flat[flat_key]
        for leaf, flat_key in _IMPLICIT_VERTICAL_ADVECTION_FIELDS
    }


_TIDAL_SIDE_MAP = {"north": "N", "south": "S", "east": "E", "west": "W"}


def _tidal_arrays_from_dataset(tidal_ds: Any, name: str) -> dict[str, torch.Tensor]:
    """Per-side u/v amplitude+phase tensors for one constituent from the lane zarr.

    ``tidal_ds`` is the canonical tidal Dataset (dims constituent, side, along;
    vars u_tide_amplitude/u_tide_phase/v_tide_amplitude/v_tide_phase; phase in
    seconds, Greenwich — V0 is applied later by build_tidal_config_from_constituents).
    Keys match build_tidal_config_from_constituents' `arrays` schema:
    ``{north,south,east,west}_{u,v}_{amplitude,phase}``.
    """
    available = {str(c) for c in tidal_ds["constituent"].values}
    if name not in available:
        raise ValueError(
            f"tidal constituent {name!r} not in lane output {sorted(available)}; "
            "add it to the tpxo9 source `constituents` list"
        )
    has_valid = "along_valid" in tidal_ds.coords
    arrays: dict[str, torch.Tensor] = {}
    for side_full, side_short in _TIDAL_SIDE_MAP.items():
        sel = tidal_ds.sel(constituent=name, side=side_short)
        # Non-square domains zero-pad `along` to max(nx, ny); slice each side to
        # its true length so the per-side amp/phase match the bins / script.
        vlen = int(sel["along_valid"].values) if has_valid else None
        for comp, var in [("u", "u_tide_amplitude"), ("v", "v_tide_amplitude"),
                          ("u", "u_tide_phase"), ("v", "v_tide_phase")]:
            line = np.asarray(sel[var].values)
            if vlen is not None:
                line = line[:vlen]
            kind = "amplitude" if var.endswith("amplitude") else "phase"
            arrays[f"{side_full}_{comp}_{kind}"] = torch.as_tensor(line, dtype=torch.float64)
    return arrays


#: Dynamics modules that take ``resolve_reference_profiles``' tRef/sRef unless
#: their kwargs set them (``enrich_manifest``; the MITgcm data export reads the
#: same rule, ``layer_d/_mit_switches.py``).
REFERENCE_PROFILE_MODULES = frozenset({"linear_equation_of_state", "hydrostatic_pressure_gradient"})


def tracer_vertical_scheme_default(run_cfg: Any) -> Any:
    """``horizontal_tracer_advection``'s vertical scheme when its kwargs omit it:
    advection.vertical (``enrich_manifest``; the MITgcm data export reads the same)."""
    from defgcm.operators import TracerFluxScheme

    return TracerFluxScheme(str(run_cfg.advection.vertical))


def resolve_reference_profiles(
    run_cfg: Any,
    ic_bundle: Any,
    ic_tensors: dict[str, torch.Tensor],
    *,
    device: str,
    reference_profile_fallback: Callable[[Any, Any, str], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve global reference T/S before rank-local IC slicing.

    Source declarations take precedence over IC layer means and the legacy
    fallback. These profiles are shared physical inputs, not tile statistics.
    """
    tref_source = _reference_profile_from_ic_source(run_cfg, "reference_temperature")
    if tref_source is None:
        tref_source = reference_profile_from_tensors(ic_tensors, "reference_temperature")
    if tref_source is None:
        if reference_profile_fallback is not None:
            tref_source = reference_profile_fallback(run_cfg, ic_bundle, "reference_temperature")
        else:
            tref_source = torch.zeros((int(run_cfg.grid.nz),), dtype=torch.float64, device=device)
    tref = tref_source.to(dtype=torch.float64, device=device)
    sref_source = _reference_profile_from_ic_source(run_cfg, "reference_salinity")
    if sref_source is None:
        sref_source = reference_profile_from_tensors(ic_tensors, "reference_salinity")
    if sref_source is None:
        if reference_profile_fallback is not None:
            sref_source = reference_profile_fallback(run_cfg, ic_bundle, "reference_salinity")
        else:
            sref_source = torch.full((int(run_cfg.grid.nz),), 35.0, dtype=torch.float64, device=device)
    sref = sref_source.to(dtype=torch.float64, device=device)
    return tref, sref


def enrich_manifest(
    run_cfg: Any,
    ic_bundle: Any,
    ic_tensors: dict[str, torch.Tensor],
    *,
    obcs_source: BoundaryBundleSource,
    wind_source: Any | None,
    surface_flux_source: Any | None = None,
    bulk_source: Any | None = None,
    surface_pressure_load: torch.Tensor | None = None,
    device: str,
    reference_profile_fallback: Callable[[Any, Any, str], torch.Tensor] | None = None,
    tidal_data: Any | None = None,
    tidal_config_transform: Callable[[Any], Any] | None = None,
    layout: "ParallelLayout | None" = None,
    reference_profiles: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Any:
    """Resolve dynamics/stepper/BC kwargs and inject runtime sources.

    ``reference_profile_fallback`` is the runtime fallback that lets the
    runtime path read ``t_ref``/``s_ref`` from ``case.extras`` when the
    canonical IC tensors are missing T/S (test/exp pipelines). The canonical writer leaves
    it None and defaults to 0/35 like the previous inline implementation.
    """
    for source in (obcs_source, wind_source, surface_flux_source, bulk_source):
        if source is not None:
            sampling_rule(source)
    implicit_vertical_advection_kwargs = implicit_vertical_advection_stepper_kwargs(
        run_cfg
    )
    if reference_profiles is None:
        reference_profiles = resolve_reference_profiles(
            run_cfg, ic_bundle, ic_tensors, device=device,
            reference_profile_fallback=reference_profile_fallback,
        )
    tref, sref = reference_profiles

    # OBCS Tides: build tidal config from boundary.tidal_constituents (before dynamics loop).
    # Preferred source is the canonical tidal lane output (`tidal_data`: the
    # prepared_tidal zarr/bundle from the stage1a tidal lane). Legacy fallback:
    # per-constituent bins under boundary.tidal_data_dir (pre-lane cases).
    _tidal_cfg = None
    if run_cfg.boundary.tidal_constituents:
        from defgcm.boundaries.tidal import build_tidal_config_from_constituents
        tidal_ds = tidal_data.data if hasattr(tidal_data, "meta") else tidal_data  # DataBundle → .data
        tidal_dir = run_cfg.boundary.tidal_data_dir or None
        constituents = []
        for comp in run_cfg.boundary.tidal_constituents:
            comp_dict = dict(comp)
            name = comp_dict.get("name", "")
            if tidal_ds is not None and "arrays" not in comp_dict and "files" not in comp_dict:
                comp_dict["arrays"] = _tidal_arrays_from_dataset(tidal_ds, name)
            elif "files" not in comp_dict and tidal_dir:
                comp_dict["files"] = {
                    f"{side}_{var}": f"{tidal_dir}/OB{s}_{v}_{name}.bin"
                    for side, s in [("north", "N"), ("south", "S"), ("east", "E"), ("west", "W")]
                    for var, v in [("u_amplitude", "uTidAm"), ("u_phase", "uTidPh"),
                                   ("v_amplitude", "vTidAm"), ("v_phase", "vTidPh")]
                }
            constituents.append(comp_dict)
        start_date = getattr(run_cfg.run, 'start_date', None)
        # Skip the V0 correction when the phase data already includes it (Gong
        # native _Gph, used as-is by obcs_add_tides.F) so we don't double-count;
        # the MIT export skips it on the same flag -> LOCOMgcm and MIT stay aligned.
        if getattr(run_cfg.boundary, 'tidal_phase_includes_v0', False):
            start_date = None
        _tidal_cfg = build_tidal_config_from_constituents(
            constituents, data_dir=None, start_date=str(start_date) if start_date else None,
        )
        if _tidal_cfg is not None and tidal_config_transform is not None:
            _tidal_cfg = tidal_config_transform(_tidal_cfg)

    exf_flux_specs = [spec for spec in run_cfg.sources if spec.type == "mit_exf_flux"]
    if exf_flux_specs:
        # EXF without ALLOW_ATM_TEMP reads Qnet, EmPmR and the stress from files.
        if len(exf_flux_specs) != 1 or wind_source is not None or surface_flux_source is not None \
                or bulk_source is not None:
            raise ValueError("a `mit_exf_flux` source is the only surface forcing source of a case")
        surface_flux_source, wind_source = exf_flux_sources(run_cfg, exf_flux_specs[0], layout=layout)

    from defgcm.io.era5 import ERA5BulkSource
    bulk_entries = [e for e in run_cfg.model.dynamics if e.name == "era5_bulk_forcing"]
    wind_entries = [e for e in run_cfg.model.dynamics if e.name == "era5_wind_forcing"]
    bulk_atm_wind = bool(bulk_entries) and dict(bulk_entries[0].kwargs).get(
        "use_atm_wind", isinstance(wind_source, ERA5BulkSource) or wind_source is None,
    )
    if bulk_atm_wind and wind_entries:
        if len(wind_entries) != 1 or dict(wind_entries[0].kwargs).get("stress_location", "center") != "center":
            raise ValueError("wind-derived bulk stress requires one cell-centre wind consumer")
    # GM_AdvForm of the gmredi entry is the one source of the tracer advection's
    # residual velocity (thermodynamics.F:258-276).
    gm_adv_form = any(
        bool(dict(e.kwargs)["adv_form"]) for e in run_cfg.model.dynamics if e.name == "gmredi"
    )
    enriched: list[DynamicsEntry] = []
    for entry in run_cfg.model.dynamics:
        kwargs = runtime_kwargs(run_cfg, dict(entry.kwargs), layout=layout)
        if entry.name in {"horizontal_tracer_advection", "vertical_tracer_transport"}:
            if "gm_residual_flow" in kwargs:
                raise ValueError(
                    f"model.dynamics[{entry.name}].kwargs.gm_residual_flow: it follows the "
                    "gmredi entry's adv_form"
                )
            kwargs["gm_residual_flow"] = gm_adv_form
        if entry.name == "horizontal_tracer_advection":
            # Single source for the module's vertical scheme (multi-dim R pass
            # and combined H+V path): advection.vertical, already resolved to
            # advection.tracer when unset (MITgcm ini_parms.F:509-510), the SAME
            # field driving the MIT export's tempVertAdvScheme, so LOCOMgcm and MIT
            # cannot silently diverge.  setdefault lets an explicit yaml module
            # kwarg still override.
            kwargs.setdefault("vertical_scheme", tracer_vertical_scheme_default(run_cfg))
        if entry.name in _ROTATING_MOMENTUM_MODULES:
            # physics.rotation_period is the run's one rotation period; the
            # momentum module takes it from there, not from its own kwarg.
            if "rotation_period" in kwargs:
                raise ValueError(
                    f"model.dynamics[{entry.name}].kwargs.rotation_period: set "
                    "physics.rotation_period, the one rotation period the momentum, "
                    "NonhydrostaticParams, KPP and the $coriolis_* profiles read"
                )
            kwargs["rotation_period"] = float(run_cfg.physics.rotation_period)
        complete_stepper_owned_kwargs(
            entry.name, kwargs, stepper_name=run_cfg.model.stepper.name,
            stepper_kwargs=dict(run_cfg.model.stepper.kwargs or {}),
        )
        if entry.name == "vertical_tracer_transport":
            # Single source for the tracer time step: time.dt_tracer (the SAME
            # field driving MIT's deltaTLev).  The additive vertical module needs
            # it for the DST3 / DST3-FL Courant number and rejects dt<=0 at
            # construction for those schemes, so without this the only way to
            # reach MITgcm scheme 30/33 on the non-multi-dim path would be to
            # hand-write dt in every yaml.  setdefault lets an explicit module
            # kwarg still override.
            kwargs.setdefault("dt", float(run_cfg.time.dt_tracer))
        if entry.name in REFERENCE_PROFILE_MODULES:
            kwargs.setdefault("reference_temperature", tref)
            kwargs.setdefault("reference_salinity", sref)
        if entry.name == "hydrostatic_pressure_gradient" and kwargs.get("surface_pressure_load", False):
            kwargs.setdefault("pload", surface_pressure_load)
        if entry.name in {"tracer_boundary_sponge_relaxation", "momentum_boundary_sponge_relaxation"}:
            sample = obcs_source.sample(0.0)
            if entry.name == "tracer_boundary_sponge_relaxation":
                kwargs.setdefault("temp_values", sample.temp)
                kwargs.setdefault("salt_values", sample.salt)
            else:
                kwargs.setdefault("u_values", sample.u)
                kwargs.setdefault("v_values", sample.v)
                if _tidal_cfg is not None:
                    kwargs.setdefault("tidal_config", _tidal_cfg)
            # Single source for MITgcm's OB_J*/OB_I* index-table policy, which
            # guards the momentum and the tracer sponge alike (obcs_sponge.F:67,
            # :467): boundary.ob_index_mode, the SAME field that renders the
            # OB_Jsouth/... lines of the exported data.obcs
            # (layer_d/consumers/mit_binary.py), so the LOCOMgcm gate and the
            # reference namelist cannot silently diverge.  setdefault lets an
            # explicit yaml module kwarg still override.
            # The export reads the same policy through case_ob_policy
            # (boundaries/open.py), module overrides included.
            ob_mode, ob_sides = sponge_ob_policy(kwargs, run_cfg.boundary)
            kwargs.setdefault("ob_index_mode", ob_mode)
            # ... and the side set that goes with it, from the SAME
            # boundary.sides the export reads.  Deriving it from the grid
            # topology inside the sponge instead would leave the two ends
            # with different answers whenever an axis is topologically open
            # but boundary.sides omits one of its two edges (SCS_CLM's
            # western wall).  The `or` fallback mirrors the export's legacy
            # `_bcfg.sides or (...)` so the degenerate `sides: []` spelling
            # cannot make the two ends disagree either.
            kwargs.setdefault("ob_sides", ob_sides)
            kwargs.setdefault("source", obcs_source)
        if entry.name == "era5_wind_forcing":
            if wind_source is None and not bulk_atm_wind:
                raise ValueError("era5_wind_forcing requires case.extras.forcing_source")
            kwargs.setdefault("source", None if bulk_atm_wind else wind_source)
            kwargs["use_bulk_stress"] = bool(bulk_atm_wind)
        if entry.name == "era5_bulk_forcing":
            if bulk_source is None:
                raise ValueError("era5_bulk_forcing requires an `era5` source (bulk ingest)")
            kwargs.setdefault("source", bulk_source)
            kwargs.setdefault("constants", bulk_kernel_constants(run_cfg))
            prescribed_center = (len(wind_entries) == 1 and
                dict(wind_entries[0].kwargs).get("stress_location", "center") == "center")
            kwargs.setdefault("use_atm_wind", bulk_atm_wind)
            if not kwargs["use_atm_wind"]:
                if wind_source is None or not prescribed_center:
                    raise ValueError("prescribed-stress ERA5 bulk forcing requires cell-centre era5_wind_forcing")
                kwargs.setdefault("stress_source", wind_source)
        if entry.name == "surface_climate_relaxation":
            stepper_args = dict(run_cfg.model.stepper.kwargs)
            # The nonlinear free-surface thickness factor of
            # forcing_surf_relax.F:108-155 is not ported.
            if int(stepper_args.get("nonlin_free_surf", 0)) > 0:
                raise ValueError("surface_climate_relaxation supports nonlin_free_surf=0 only")
            # One SST/SSS source: the core climatology or EXF's.  MITgcm lets the
            # core EXTERNAL_FIELDS_LOAD overwrite EXF's SST after it
            # (load_fields_driver.F:214,234); that combination is not ported.
            climatology_specs = [spec for spec in run_cfg.sources if spec.type in (
                "mit_surface_climatology", "mit_exf_climatology")]
            if len(climatology_specs) != 1:
                raise ValueError("surface_climate_relaxation requires exactly one "
                                 "`mit_surface_climatology` or `mit_exf_climatology` source")
            if climatology_specs[0].type == "mit_exf_climatology":
                kwargs.setdefault("source", exf_climatology_source(
                    run_cfg, climatology_specs[0], layout=layout))
            else:
                kwargs.setdefault("source", surface_climatology_source(run_cfg, layout=layout))
        if entry.name == "wpflux_surface_forcing":
            if surface_flux_source is None:
                raise ValueError(
                    "wpflux_surface_forcing requires heat_flux/freshwater_flux "
                    "from case.extras.forcing_source"
                )
            kwargs.setdefault("source", surface_flux_source)
        enriched.append(DynamicsEntry(name=entry.name, kwargs=kwargs))

    bc_kwargs = runtime_kwargs(run_cfg, dict(run_cfg.model.boundary_condition.kwargs), layout=layout)
    if run_cfg.model.boundary_condition.name == "internal_wave_open":
        # The boundary salinities (west_salt, east_salt) come only from the
        # configuration; nothing is filled in from the IC source.
        bc_kwargs.setdefault("enforce_obcs_balance", False)
    if run_cfg.model.boundary_condition.name == "prescribed_3d":
        if bool(bc_kwargs.pop("use_barotropic_source", False)):
            bc_kwargs.setdefault("barotropic_source", BarotropicBoundaryBundleSource(obcs_source))
        bc_kwargs.setdefault("apply_tangential_velocity", True)
    boundary_cls = BOUNDARY_CONDITIONS.get(run_cfg.model.boundary_condition.name)
    if _tidal_cfg is not None and boundary_cls.accepts_tidal_config:
        bc_kwargs.setdefault("tidal_config", _tidal_cfg)

    stepper_kwargs = runtime_kwargs(run_cfg, dict(run_cfg.model.stepper.kwargs), layout=layout)
    stepper_kwargs.update(implicit_vertical_advection_kwargs)
    manifest = ModelManifest(
        dynamics=tuple(enriched),
        stepper=replace(
            run_cfg.model.stepper,
            kwargs=stepper_kwargs,
        ),
        boundary_condition=replace(run_cfg.model.boundary_condition, kwargs=bc_kwargs),
    )
    return replace(run_cfg, model=manifest)
