"""Canonical bridge from typed runner config to legacy flat YAML."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
from typing import Any, Callable, Literal

from locomgcm.io.runner.config._schema import RunConfig
from locomgcm.io.runner.config.vertical_diffusion import (
    bind_mit_vertical_diffusion, resolved_vertical_diffusion,
)


class TranslationError(Exception):
    """Raised when new-schema YAML cannot be translated without dropping data."""


@dataclass(frozen=True)
class FromConfig:
    getter: Callable[[RunConfig], Any]
    legacy_key: str
    claims: tuple[str, ...]
    # When True, the bridge omits the legacy key entirely if the getter
    # returns None — preserves legacy defaults (e.g. ``--first-step-target-npz``
    # defaults to ``None`` in argparse).
    omit_if_none: bool = False


@dataclass(frozen=True)
class Derived:
    fn: Callable[[RunConfig], dict[str, Any]]
    claims: tuple[str, ...]


@dataclass(frozen=True)
class FromSource:
    source_type: str
    field: str
    legacy_key: str


@dataclass(frozen=True)
class NotExposed:
    reason: str


LegacyArgRule = FromConfig | Derived | FromSource | NotExposed


LEGACY_TRACER_BOUNDARY_SCHEMES = frozenset({"clamped", "inflow_clamped", "stevens"})
LEGACY_BOUNDARY_TYPES = frozenset({"wall", "open"})


def _run(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.run, path)


def _grid(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.grid, path)


def _time(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.time, path)


def _physics(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.physics, path)


def _implicit_vertical_advection(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.physics.implicit_vertical_advection, path)


# The pair table lives with the loader, which mirrors overrides across the two
# spellings; this module keeps the consistency check.
from .config._double_track import DOUBLE_TRACK_FIELDS as _DOUBLE_TRACK_FIELDS


def _lookup_in_model_kwargs(
    cfg: RunConfig,
    *,
    location: str,
    module_name: str,
    kwarg_key: str,
) -> Any:
    """Look up a kwarg from model.dynamics[<module_name>].kwargs or model.stepper.kwargs.

    Returns the value if found, else a sentinel ``_MISSING`` (NOT None — None is a
    valid stored value). Caller decides fallback.
    """
    model = cfg.model
    if location == "stepper":
        kwargs = dict(model.stepper.kwargs)
        if kwarg_key in kwargs:
            return kwargs[kwarg_key]
        return _MISSING
    for entry in model.dynamics:
        if entry.name == module_name:
            kwargs = dict(entry.kwargs)
            if kwarg_key in kwargs:
                return kwargs[kwarg_key]
    return _MISSING


_MISSING = object()


def _dyn_or_phys(
    *,
    physics_field: str,
    kwarg_key: str,
    location: str,
    module_name: str,
) -> Callable[[RunConfig], Any]:
    """FromConfig getter that reads dynamics/stepper kwargs first, physics second.

    Reading only physics defaults could disagree with the runtime when YAML
    supplies a dynamics or stepper kwarg. Resolve the explicit kwarg first so
    the MIT export and LOCOMgcm runtime use the same parameter.
    """
    def _getter(cfg: RunConfig) -> Any:
        val = _lookup_in_model_kwargs(
            cfg, location=location, module_name=module_name, kwarg_key=kwarg_key
        )
        if val is _MISSING:
            return getattr(cfg.physics, physics_field)
        return val
    return _getter


def _advection(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.advection, path)


def _boundary(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.boundary, path)


def _output(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.output, path)


def _cache(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.cache, path)


def _debug(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.debug, path)


def _mit_cmp(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.mit_comparison, path)


def _initial_condition(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.initial_condition, path)


def _adjoint(path: str) -> Callable[[RunConfig], Any]:
    return lambda cfg: getattr(cfg.adjoint, path)


def _device(cfg: RunConfig) -> str:
    device = cfg.run.device
    if device != "auto":
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _boundary_type(cfg: RunConfig) -> str:
    value = str(cfg.boundary.type)
    if value not in LEGACY_BOUNDARY_TYPES:
        raise TranslationError(
            "boundary.type must be one of "
            f"{sorted(LEGACY_BOUNDARY_TYPES)} for legacy translation, got {value!r}"
        )
    return value


def _tracer_boundary_scheme(cfg: RunConfig) -> str:
    value = str(cfg.boundary.scheme)
    if value not in LEGACY_TRACER_BOUNDARY_SCHEMES:
        raise TranslationError(
            "boundary.scheme maps to legacy tracer-boundary-scheme, whose accepted "
            f"values are {sorted(LEGACY_TRACER_BOUNDARY_SCHEMES)}; got {value!r}"
        )
    return value


def _split_dt(cfg: RunConfig) -> dict[str, Any]:
    return {
        "fixed-dt": cfg.time.dt,
        "dt-mom": cfg.time.dt_mom,
        "dt-tracer": cfg.time.dt_tracer,
        "dt-freesurf": cfg.time.dt_freesurf,
        "dt-clock": cfg.time.dt_clock,
    }


def _nonhydrostatic(cfg: RunConfig) -> dict[str, Any]:
    nh = cfg.physics.nonhydrostatic
    return {
        "nonhydrostatic": bool(nh.enabled),
        "cg3d-max-iters": nh.cg3d_max_iters,
        "cg3d-target-residual": nh.cg3d_target_residual,
        "cg3d-target-res-w-unit": nh.cg3d_target_res_w_unit,
        "nh-am2": nh.nh_am2,
        "select-3d-cori-scheme": nh.select_3d_cori_scheme,
        "select-cori-map": nh.select_cori_map,
        "f-prime": nh.f_prime,
    }


def _mitgcm_numerics(cfg: RunConfig) -> dict[str, Any]:
    from .layer_d._mit_switches import horizontal_viscosity_value, tracer_harmonic_diffusivity

    mc = cfg.mit_comparison
    return {
        "mitgcm-viscosity": mc.effective_viscosity(horizontal_viscosity_value(cfg)),
        # One legacy value; the data export writes diffKhT and diffKhS per tracer.
        "mitgcm-diffusivity": mc.effective_diffusivity(tracer_harmonic_diffusivity(cfg, "temp")),
        "mitgcm-viscA4": mc.effective_viscA4(cfg),
        "mitgcm-adv-scheme": _mitgcm_adv_scheme(cfg),
    }


def _mitgcm_adv_scheme(cfg: RunConfig) -> int | None:
    """One legacy value: temperature's; the data export writes each tracer's."""
    from .layer_d._mit_switches import tracer_advection_values

    code = tracer_advection_values(cfg, "temp")["horizontal"]
    return None if code is None else cfg.mit_comparison.effective_tracer_scheme(code)


def _momentum_advection_form(cfg: RunConfig) -> str:
    """The assembled momentum core (``layer_d/_mit_switches.momentum_advection_form``)."""
    from .layer_d._mit_switches import momentum_advection_form

    return momentum_advection_form(cfg)


def _mitgcm_sponge(cfg: RunConfig) -> dict[str, Any]:
    return {
        "mitgcm-sponge-width": cfg.boundary.sponge_width,
        "mitgcm-sponge-timescale-hours": cfg.boundary.sponge_timescale_hours,
        "mitgcm-sponge-timescale-inner-hours": cfg.boundary.sponge_timescale_inner_hours,
    }


def _implicit_viscosity(cfg: RunConfig) -> float:
    """Resolve viscAr for the MIT export, fail-closed on a dual-entry disagreement.

    Momentum twin of the tracer resolution below.  ``physics.implicit_viscosity``
    is ``None`` when the yaml never declared it, so a case that only carries the
    stepper kwarg (5 of the 34 shipped cases) resolves from there instead of
    tripping the equality check against a dataclass default.
    """
    physics_value = cfg.physics.implicit_viscosity
    kwarg_value = _lookup_in_model_kwargs(
        cfg, location="stepper", module_name="", kwarg_key="implicit_viscosity"
    )
    if kwarg_value is _MISSING or kwarg_value is None:
        return 0.0 if physics_value is None else float(physics_value)
    if physics_value is not None and not math.isclose(
        float(physics_value), float(kwarg_value), rel_tol=1.0e-12, abs_tol=0.0
    ):
        raise TranslationError(
            "implicit viscosity dual-entry mismatch: "
            f"physics.implicit_viscosity={physics_value!r} but "
            f"model.stepper.kwargs gives {kwarg_value!r}"
        )
    return float(kwarg_value)


def _mitgcm_explicit_viscosity(cfg: RunConfig) -> bool:
    """Invert the canonical MIT gate, retaining the legacy override fallback."""
    gate = cfg.physics.implicit_vertical_viscosity
    if gate is None:
        gate = _lookup_in_model_kwargs(
            cfg,
            location="stepper",
            module_name="",
            kwarg_key="implicit_vertical_viscosity",
        )
    if gate is _MISSING or gate is None:
        return bool(cfg.mit_comparison.explicit_viscosity)
    return not bool(gate)


def _tracer(cfg: RunConfig) -> dict[str, Any]:
    # NOTE: legacy gave these two args explicit ``dest=`` overrides — the CLI
    # flags are ``--implicit-tracer-diffusion`` / ``--tracer-diffusivity-kr``
    # but the argparse dests are ``implicit_vertical_diffusion`` /
    # ``diffusivity_kr``. Yaml keys here MUST
    # normalize via ``_load_config_file``'s ``-→_`` rule to those dests;
    # emitting the flag name verbatim normalizes to
    # ``implicit_tracer_diffusion`` / ``tracer_diffusivity_kr``, both filtered
    # out by the parser, silently leaving ``args.implicit_vertical_diffusion``
    # at default ``False`` (per :238) and ``args.diffusivity_kr`` at ``0.0``
    # (per :239).
    values, _ = bind_mit_vertical_diffusion(
        resolved_vertical_diffusion(cfg, error_type=TranslationError)
    )
    return {
        "implicit-vertical-diffusion": values["implicitDiffusion"],
        "diffusivity-kr": values["diffKrT"],
    }


def _free_surface(cfg: RunConfig) -> dict[str, Any]:
    return {
        "select-r-star": cfg.physics.free_surface.select_r_star,
        "lin-fs-conserve-tr": cfg.physics.free_surface.lin_fs_conserve_tr,
    }


_TIME_CLAIMS = (
    "time.dt",
    "time.dt_mom",
    "time.dt_tracer",
    "time.dt_freesurf",
    "time.dt_clock",
    "time_step.dt",
    "time_step.dt_mom",
    "time_step.dt_tracer",
    "time_step.dt_freesurf",
    "time_step.dt_clock",
)

_NH_CLAIMS = (
    "physics.nonhydrostatic.enabled",
    "physics.nonhydrostatic.cg3d_max_iters",
    "physics.nonhydrostatic.cg3d_target_residual",
    "physics.nonhydrostatic.cg3d_target_res_w_unit",
    "physics.nonhydrostatic.nh_am2",
    "physics.nonhydrostatic.select_3d_cori_scheme",
    "physics.nonhydrostatic.select_cori_map",
    "physics.nonhydrostatic.f_prime",
)

_TRACER_CLAIMS = (
    "physics.tracer.implicit_vertical_diffusion",
    "physics.tracer.diffusivity_kr",
)

_MIT_NUMERIC_CLAIMS = (
    "mit_comparison.sync_from_to",
    "mit_comparison.viscA4",
    "mit_comparison.override_viscosity",
    "mit_comparison.override_diffusivity",
    "mit_comparison.override_viscA4",
    "mit_comparison.viscA4_explicit",
    "mit_comparison.override_dt",
    "mit_comparison.override_days",
    "mit_comparison.override_tracer_scheme",
    "mit_comparison.override_mitgcm_adv_scheme",
    "mit_comparison.override_multi_dim",
)

_time_rule = Derived(_split_dt, _TIME_CLAIMS)
_nh_rule = Derived(_nonhydrostatic, _NH_CLAIMS)
_tracer_rule = Derived(_tracer, _TRACER_CLAIMS)
_mit_numeric_rule = Derived(_mitgcm_numerics, _MIT_NUMERIC_CLAIMS)
_mit_sponge_rule = Derived(
    _mitgcm_sponge,
    (
        "boundary.sponge_width",
        "boundary.sponge_timescale_hours",
        "boundary.sponge_timescale_inner_hours",
    ),
)
_free_surface_rule = Derived(
    _free_surface,
    (
        "physics.free_surface.select_r_star",
        "physics.free_surface.lin_fs_conserve_tr",
    ),
)


def _not_exposed(reason: str) -> NotExposed:
    if not reason:
        raise ValueError("NotExposed reason is mandatory")
    return NotExposed(reason)


LEGACY_ARG_RULES: dict[str, LegacyArgRule] = {
    "config": _not_exposed("transient path to the generated legacy YAML"),
    "source": FromSource("hycom_netcdf", "path", "source"),
    "era5": FromSource("era5_file", "path", "era5"),
    "era5_dir": FromSource("era5_netcdf", "path", "era5-dir"),
    "start_date": FromConfig(
        lambda cfg: cfg.run.start_date,
        "start-date",
        ("run.start_date", "sources.*.start_date"),
    ),
    "topo_file": FromSource("gebco_netcdf", "path", "topo-file"),
    "initial_temp_salt_source": FromConfig(
        _initial_condition("source"),
        "initial-temp-salt-source",
        ("initial_condition.source", "initial_condition.woa23_dir"),
    ),
    "woa23_dir": FromConfig(
        _initial_condition("woa23_dir"),
        "woa23-dir",
        ("initial_condition.source", "initial_condition.woa23_dir"),
    ),
    "experiment_name": FromConfig(_run("experiment_name"), "experiment-name", ("run.experiment_name",)),
    "output_dir": _not_exposed(
        "stage2 supplies manifest.run_dir as the optional output_dir kwarg to "
        "translate_to_legacy_flat_yaml; not sourced from run_cfg / raw yaml"
    ),
    "days": FromConfig(_time("days"), "days", ("time.days",)),
    # Expose `time.steps` and `time.seconds` to
    # run_params.yaml so the MIT-binary consumer
    # (locomgcm/io/runner/layer_d/consumers/mit_binary.py) can see an
    # explicit count/horizon and bypass the days*86400/dt ULP trap. Both
    # LOCOMgcm runtime and MIT export route step resolution through
    # locomgcm.io.runner._time_steps.compute_n_steps which honors them
    # in priority order: steps > seconds > days.
    "steps": FromConfig(_time("steps"), "steps", ("time.steps",)),
    "seconds": FromConfig(_time("seconds"), "seconds", ("time.seconds",)),
    "nx": FromConfig(_grid("nx"), "nx", ("grid.nx",)),
    "ny": FromConfig(_grid("ny"), "ny", ("grid.ny",)),
    "resolution_deg": FromConfig(_grid("resolution_deg"), "resolution-deg", ("grid.resolution_deg",)),
    "nz": FromConfig(_grid("nz"), "nz", ("grid.nz",)),
    "depth_m": FromConfig(_grid("depth_m"), "depth-m", ("grid.depth_m",)),
    "vertical_layer_thicknesses": FromConfig(
        lambda cfg: list(cfg.grid.vertical_layer_thicknesses),
        "vertical-layer-thicknesses",
        ("grid.vertical_layer_thicknesses",),
    ),
    "vertical_center_spacing": FromConfig(
        lambda cfg: list(cfg.grid.vertical_center_spacing) or None,
        "vertical-center-spacing",
        ("grid.vertical_center_spacing",),
        omit_if_none=True,
    ),
    "hfac_min_dz": FromConfig(
        # Read from case.extras.hfac_min_dz (yaml convention) instead of
        # legacy physics.hfac_min_dz. scs_hires/pureocean yamls put hfac_min* in
        # case.extras; model_input.py:_extra reads from case.extras too. This
        # makes the LOCOMgcm and MIT consumers see the SAME value via run_params.yaml.
        lambda cfg: (cfg.case.extras or {}).get("hfac_min_dz", 0.0),
        "hfac-min-dz",
        ("case.extras.hfac_min_dz",),
    ),
    "hfac_min": FromConfig(
        # Without this mapping the MIT consumer would use a hard-coded 0.3
        # against the yaml's 0.2.
        lambda cfg: (cfg.case.extras or {}).get("hfac_min", 0.2),
        "hfac-min",
        ("case.extras.hfac_min",),
    ),
    "debug_dump_steps": FromConfig(
        _debug("dump_steps"),
        "debug-dump-steps",
        ("debug.dump_steps",),
    ),
    "center_lat": FromConfig(_grid("center_lat"), "center-lat", ("grid.center_lat",)),
    "center_lon": FromConfig(_grid("center_lon"), "center-lon", ("grid.center_lon",)),
    "cfl_safety": FromConfig(
        _physics("cfl_safety"),
        "cfl-safety",
        ("physics.cfl_safety",),
    ),
    "sponge_width": FromConfig(_boundary("sponge_width"), "sponge-width", ("boundary.sponge_width",)),
    "sponge_timescale_hours": FromConfig(
        _boundary("sponge_timescale_hours"),
        "sponge-timescale-hours",
        ("boundary.sponge_timescale_hours",),
    ),
    "sponge_timescale_inner_hours": FromConfig(
        _boundary("sponge_timescale_inner_hours"),
        "sponge-timescale-inner-hours",
        ("boundary.sponge_timescale_inner_hours",),
    ),
    "sponge_mode": FromConfig(_boundary("sponge_mode"), "sponge-mode", ("boundary.sponge_mode",)),
    "use_linear_sponge": FromConfig(
        _boundary("use_linear_sponge"),
        "use-linear-sponge",
        ("boundary.use_linear_sponge",),
    ),
    "sponge_ab2_eligible": FromConfig(
        _boundary("sponge_ab2_eligible"),
        "sponge-ab2-eligible",
        ("boundary.sponge_ab2_eligible",),
    ),
    "sponge_balance_obcs_targets": FromConfig(
        _boundary("sponge_balance_obcs_targets"),
        "sponge-balance-obcs-targets",
        ("boundary.sponge_balance_obcs_targets",),
    ),
    "tracer_boundary_scheme": FromConfig(_tracer_boundary_scheme, "tracer-boundary-scheme", ("boundary.scheme",)),
    "stevens_relaxation_hours": FromConfig(
        _boundary("stevens_relaxation_hours"),
        "stevens-relaxation-hours",
        ("boundary.stevens_relaxation_hours",),
    ),
    "fast_substeps": FromConfig(
        _physics("fast_substeps"),
        "fast-substeps",
        ("physics.fast_substeps",),
    ),
    "vertical_advection_scheme": FromConfig(
        _advection("vertical"),
        "vertical-advection-scheme",
        ("advection.vertical",),
    ),
    "temperature_implicit_vertical_advection": FromConfig(
        _implicit_vertical_advection("temperature"),
        "temperature_implicit_vertical_advection",
        ("physics.implicit_vertical_advection.temperature",),
    ),
    "salinity_implicit_vertical_advection": FromConfig(
        _implicit_vertical_advection("salinity"),
        "salinity_implicit_vertical_advection",
        ("physics.implicit_vertical_advection.salinity",),
    ),
    "momentum_implicit_vertical_advection": FromConfig(
        _implicit_vertical_advection("momentum"),
        "momentum_implicit_vertical_advection",
        ("physics.implicit_vertical_advection.momentum",),
    ),
    "tracer_advection_scheme": FromConfig(
        _advection("tracer"),
        "tracer-advection-scheme",
        ("advection.tracer",),
    ),
    "multi_dim_advection": FromConfig(
        _advection("multi_dim"),
        "multi-dim-advection",
        ("advection.multi_dim",),
    ),
    "advective_form": FromConfig(
        _physics("advective_form"),
        "advective-form",
        ("physics.advective_form",),
    ),
    "momentum_advection_form": FromConfig(
        _momentum_advection_form,
        "momentum-advection-form",
        ("model.dynamics.*",),
    ),
    "mitgcm_adv_scheme": _mit_numeric_rule,
    "snapshot_interval_hours": FromConfig(
        _output("snapshot_interval_hours"),
        "snapshot-interval-hours",
        ("output.snapshot_interval_hours",),
    ),
    # Integer-seconds cadence (resolved float or None) → run_params.
    "snapshot_interval_seconds": FromConfig(
        _output("snapshot_interval_seconds"),
        "snapshot-interval-seconds",
        ("output.snapshot_interval_seconds",),
    ),
    "boundary_time_mode": FromConfig(
        _boundary("boundary_time_mode"),
        "boundary-time-mode",
        ("boundary.boundary_time_mode",),
    ),
    "tangential_zero_mode": FromConfig(
        _boundary("tangential_zero_mode"),
        "tangential-zero-mode",
        ("boundary.tangential_zero_mode",),
    ),
    "transport_correction": FromConfig(
        _physics("transport_correction"),
        "transport-correction",
        ("physics.transport_correction",),
    ),
    "source_days": _not_exposed("HYCOM source-window override is not modeled in RunConfig"),
    "boundary_type": FromConfig(_boundary_type, "boundary-type", ("boundary.type",)),
    "use_prescribed_obcs": _not_exposed("target='to_run' tail derives this from mit_comparison.enabled"),
    "cache_dir": FromConfig(_cache("root"), "cache-dir", ("cache.root",)),
    # read dynamics/stepper kwargs first, fall back to physics.* —
    # closes the silent-default trap where yaml wrote model.dynamics kwargs
    # but left physics.* at dataclass defaults (e.g. visc_c2_leith=1.0 vs 0).
    "enable_vertical_shear": FromConfig(
        _dyn_or_phys(
            physics_field="enable_vertical_shear",
            kwarg_key="enable_vertical_shear",
            location="dynamics",
            module_name="vector_invariant_momentum_advection",
        ),
        "enable-vertical-shear",
        ("physics.enable_vertical_shear",),
    ),
    "split_coriolis": FromConfig(
        _dyn_or_phys(
            physics_field="split_coriolis",
            kwarg_key="split_coriolis",
            location="dynamics",
            module_name="vector_invariant_momentum_advection",
        ),
        "split-coriolis",
        ("physics.split_coriolis",),
    ),
    "ke_scheme": FromConfig(
        # ke_scheme drives MIT data.PARM01.selectKEscheme which
        # controls mom_vecinv (and calc_r_star) — NOT bot_drag. Bridge reads
        # vec_inv's select_ke_scheme, falling back to physics.ke_scheme.
        _dyn_or_phys(
            physics_field="ke_scheme",
            kwarg_key="select_ke_scheme",
            location="dynamics",
            module_name="vector_invariant_momentum_advection",
        ),
        "ke-scheme",
        ("physics.ke_scheme",),
    ),
    "viscosity": FromConfig(
        _dyn_or_phys(
            physics_field="viscosity",
            kwarg_key="viscosity",
            location="dynamics",
            module_name="horizontal_viscosity",
        ),
        "viscosity",
        ("physics.viscosity",),
    ),
    "diffusivity": FromConfig(
        _dyn_or_phys(
            physics_field="diffusivity",
            kwarg_key="diffusivity",
            location="dynamics",
            module_name="horizontal_tracer_diffusion",
        ),
        "diffusivity",
        ("physics.diffusivity",),
    ),
    "mitgcm_viscosity": _mit_numeric_rule,
    "mitgcm_viscA4": _mit_numeric_rule,
    "mitgcm_diffusivity": _mit_numeric_rule,
    "mitgcm_sponge_width": _mit_sponge_rule,
    "mitgcm_sponge_timescale_hours": _mit_sponge_rule,
    "mitgcm_sponge_timescale_inner_hours": _mit_sponge_rule,
    "sponge_relax_tracers": FromConfig(
        _mit_cmp("sponge_relax_tracers"),
        "sponge-relax-tracers",
        ("mit_comparison.sponge_relax_tracers",),
    ),
    "cache_rebuild": FromConfig(
        _cache("rebuild"),
        "cache-rebuild",
        ("cache.rebuild",),
    ),
    "cache_cleanup": _not_exposed("new pipeline owns cache.cleanup_policy instead of legacy cache-cleanup"),
    "monitor_interval_steps": FromConfig(
        _output("monitor_interval_steps"),
        "monitor-interval-steps",
        ("output.monitor_interval_steps",),
    ),
    "nc_output_interval_hours": FromConfig(
        _output("nc_output_interval_hours"),
        "nc-output-interval-hours",
        ("output.nc_output_interval_hours",),
    ),
    # Integer-seconds cadence (resolved float or None) → run_params.
    # mit_binary consumer prefers this over the float-hours field.
    "nc_output_interval_seconds": FromConfig(
        _output("nc_output_interval_seconds"),
        "nc-output-interval-seconds",
        ("output.nc_output_interval_seconds",),
    ),
    "device": FromConfig(_device, "device", ("run.device",)),
    "mitgcm_comparison": _not_exposed("target='model_build' tail sets this; stage2/to_run leaves it off"),
    "mitgcm_only": _not_exposed(
        "orchestrator selects mitgcm-only via stage filtering; not passed to legacy"
    ),
    "preprocess": FromConfig(_run("preprocess"), "preprocess", ("run.preprocess",)),
    "mitgcm_root": _not_exposed("stage3 owns the MITgcm source root"),
    "viscosity_scheme": FromConfig(
        _dyn_or_phys(
            physics_field="viscosity_scheme",
            kwarg_key="viscosity_scheme",
            location="dynamics",
            module_name="horizontal_viscosity",
        ),
        "viscosity-scheme",
        ("physics.viscosity_scheme",),
    ),
    "visc_c2_leith": FromConfig(
        _dyn_or_phys(
            physics_field="visc_c2_leith",
            kwarg_key="visc_c2_leith",
            location="dynamics",
            module_name="horizontal_viscosity",
        ),
        "visc-c2-leith",
        ("physics.visc_c2_leith",),
    ),
    "visc_c2_smag": FromConfig(
        _dyn_or_phys(
            physics_field="visc_c2_smag",
            kwarg_key="visc_c2_smag",
            location="dynamics",
            module_name="horizontal_viscosity",
        ),
        "visc-c2-smag",
        ("physics.visc_c2_smag",),
    ),
    # no_slip_sides: read from model.dynamics.horizontal_viscosity.kwargs (the
    # same source LOCOMgcm instantiates from) so the MIT export cannot silently
    # disagree with the LOCOMgcm model on lateral side drag. Falls back to
    # physics.no_slip_sides (default False = legacy MIT free-slip baseline).
    "no_slip_sides": FromConfig(
        _dyn_or_phys(
            physics_field="no_slip_sides",
            kwarg_key="no_slip_sides",
            location="dynamics",
            module_name="horizontal_viscosity",
        ),
        "no-slip-sides",
        ("physics.no_slip_sides",),
    ),
    "mitgcm_cartesian": FromConfig(
        _mit_cmp("cartesian"),
        "mitgcm-cartesian",
        ("mit_comparison.cartesian",),
    ),
    "mitgcm_explicit_diffusion": _not_exposed("deprecated legacy alias; use physics.tracer fields"),
    "implicit_vertical_diffusion": _tracer_rule,
    "diffusivity_kr": _tracer_rule,
    "implicit_viscosity": FromConfig(
        _implicit_viscosity,
        "implicit-viscosity",
        ("physics.implicit_viscosity",),
    ),
    "select_r_star": _free_surface_rule,
    "lin_fs_conserve_tr": _free_surface_rule,
    "nonlin_free_surf": FromConfig(
        _dyn_or_phys(
            physics_field="nonlin_free_surf",
            kwarg_key="nonlin_free_surf",
            location="stepper",
            module_name="",
        ),
        "nonlin-free-surf",
        ("physics.nonlin_free_surf",),
    ),
    "mitgcm_explicit_viscosity": FromConfig(
        _mitgcm_explicit_viscosity,
        "mitgcm-explicit-viscosity",
        (
            "physics.implicit_vertical_viscosity",
            "mit_comparison.explicit_viscosity",
        ),
    ),
    "nonhydrostatic": _nh_rule,
    "cg3d_max_iters": _nh_rule,
    "cg3d_target_residual": _nh_rule,
    "cg3d_target_res_w_unit": _nh_rule,
    "nh_am2": _nh_rule,
    "fixed_dt": _time_rule,
    "dt": _not_exposed("legacy dt alias is not emitted; time.dt is emitted as fixed-dt"),
    "dt_mom": _time_rule,
    "dt_tracer": _time_rule,
    "dt_freesurf": _time_rule,
    "dt_clock": _time_rule,
    #  — adjoint trace subsystem surfaced via the separate
    # `experiments/pipeline_v2/run_adjoint.py` orchestrator entrypoint.
    # `first_step_adjoint_trace` itself is target-tail derived (target="adjoint"
    # emits `first-step-adjoint-trace=True`); the other 5 dests are sourced
    # from `run_cfg.adjoint.*` and emitted under all 3 targets.
    "first_step_adjoint_trace": _not_exposed(
        "target='adjoint' tail emits first-step-adjoint-trace=True"
    ),
    "first_step_adjoint_target": FromConfig(
        _adjoint("target"),
        "first-step-adjoint-target",
        ("adjoint.target",),
    ),
    "first_step_target_npz": FromConfig(
        _adjoint("target_npz"),
        "first-step-target-npz",
        ("adjoint.target_npz",),
        omit_if_none=True,
    ),
    "first_step_adjoint_fields": FromConfig(
        lambda cfg: list(cfg.adjoint.fields),
        "first-step-adjoint-fields",
        ("adjoint.fields",),
    ),
    "first_step_adjoint_top_k": FromConfig(
        _adjoint("top_k"),
        "first-step-adjoint-top-k",
        ("adjoint.top_k",),
    ),
    "first_step_detect_anomaly": FromConfig(
        _adjoint("detect_anomaly"),
        "first-step-detect-anomaly",
        ("adjoint.detect_anomaly",),
    ),
}


PIPELINE_OWNED_PATHS: frozenset[str] = frozenset(
    {
        "schema_version",
        "pipelines.*",
        # Compute precision selects the torch dtype every stage-1b builder is
        # handed (``model_input.py`` resolves it once and threads it through the
        # already-parameterised ``dtype=`` arguments).  It is a pipeline-side
        # tensor-construction decision, not a legacy model-build parameter --
        # the legacy builder never saw a dtype knob -- so it is owned here
        # rather than registered as a legacy arg.
        "run.precision",
        "sources.*.on_error",
        "sources.*.cache",
        "sources.*.time_mode",
        # hycom steady mode: single-snapshot (held-constant) input — pipeline
        # source behavior, no legacy model-build counterpart (see
        # layer_b/sources/hycom.py).
        "sources.*.steady",
        # Raw HYCOM source controls. These affect stage1a source/preprocessing
        # only; the legacy model builder consumes the prepared zarr/model_input.
        "sources.*.month",
        "sources.*.pregrid_to_target",
        # tpxo9 tidal source: fully pipeline-owned (feeds the stage1a `tidal`
        # lane; no legacy model-build counterpart).
        "sources.*.constituents",
        # MITgcm-verification binary IC source (layer_b/sources/mit_input_binary.py):
        # the case directory, the namelist's readBinaryPrec, and the PARM05 file
        # names.  All of it selects WHICH bytes stage1a reads into the canonical
        # state bundle; the legacy model builder consumes that bundle, so none of
        # these has (or should have) a legacy model-build arg.  Registered here
        # for the same reason the sponge timescales above were: a source field
        # with a consumer but no registration trips _assert_yaml_complete on a
        # fresh stage1a resolve.
        "sources.*.mit_verification_dir",
        "sources.*.read_binary_prec",
        "sources.*.bathy_file",
        "sources.*.temp_file",
        "sources.*.salt_file",
        "sources.*.u_file",
        "sources.*.v_file",
        "sources.*.eta_file",
        "sources.*.t_ref",
        "sources.*.s_ref",
        # MITgcm prescribed OBCS binary source: native grid dimensions, active
        # sides, per-side field files and explicit namelist cadence are consumed
        # by the stage1a boundary lane.
        "sources.*.nx",
        "sources.*.ny",
        "sources.*.nz",
        "sources.*.sides",
        "sources.*.files.*.temperature",
        "sources.*.files.*.salinity",
        "sources.*.files.*.eta",
        "sources.*.files.*.u",
        "sources.*.files.*.v",
        "sources.*.files.*.w",
        "sources.*.periodic_external_forcing",
        # ... and the same story for the PARM05 surface-forcing file names plus
        # the PARM03 record cadence they are read at (periodicExternalForcing /
        # externForcingPeriod / externForcingCycle): consumed by
        # mit_input_binary_forcing, no legacy model-build arg.
        "sources.*.zonal_wind_file",
        "sources.*.merid_wind_file",
        "sources.*.surf_qnet_file",
        "sources.*.empmr_file",
        "sources.*.forcing_period_seconds",
        "sources.*.forcing_cycle_seconds",
        # MITgcm INI_THETA semantics are consumed by build_model_from_manifest
        # before create_state; they have no legacy model-build argument.
        "initial_condition.check_ini_temp",
        "initial_condition.allow_freezing",
        # A MITgcm pickup start is read by the model build after the IC
        # (experiments/cases/_runtime.py _start_from_mit_pickup).
        "initial_condition.mit_pickup.meta",
        "initial_condition.mit_pickup.data",
        "initial_condition.mit_pickup.n_iter0",
        "initial_condition.mit_pickup.cd_data",
        # Per-orientation OBCS sponge relax timescales (seconds): consumed
        # directly by the stage1c MIT export (layer_d/consumers/mit_binary.py)
        # to write data.obcs Urelax/Vrelax; no legacy model-build arg (the LOCOMgcm
        # side reads its own model.dynamics.*_sponge_* mirror). These fields
        # must be registered so a fresh stage1a resolve can validate them
        # through _assert_yaml_complete.
        "boundary.sponge_u_timescale_bound_seconds",
        "boundary.sponge_u_timescale_inner_seconds",
        "boundary.sponge_v_timescale_bound_seconds",
        "boundary.sponge_v_timescale_inner_seconds",
        "output.root",
        "output.legacy_layout",
        # high-freq station (mooring) output — pipeline/runtime-owned diagnostic,
        # no legacy model-build counterpart (see locomgcm/io/station_output.py).
        "output.station_output.*",
        # checkpoint restart/resume — runtime-owned (locomgcm/checkpoint/),
        # not a legacy model-build arg.
        "checkpoint.*",
        "mit_parallel.*",
        "compare.*",
        "cache.cleanup_policy",
        "cache.enabled",
        "cache.hycom_cache_dir",
        # New-pipeline-only content-keyed prep cache (no legacy MIT-bridge
        # counterpart) — see execution/_pipeline_cache.py.
        "cache.reuse",
        "cache.verify",
        "debug.*",
        "boundary.sides",
        "boundary.tidal_data_dir",
        "boundary.tidal_constituents",
        "boundary.tidal_constituents.*",

        # stage2_torch_run (section transport window accumulator) and by the
        # MIT diagnostics consumer; the legacy model builder never sees them.
        # A fresh stage1a resolve trips _assert_yaml_complete without this.
        "observations.*",
        # consumed by enrich_manifest (LOCOMgcm) + mit_binary (MIT) to skip the V0
        # correction when the phase data already includes it (Gong native _Gph);
        # no legacy model-build arg.
        "boundary.tidal_phase_includes_v0",
        # MITgcm OBCSfixTopo: consumed directly by the grid builders
        # (builders/grid_factory.py, model_input.py, experiments/cases/_runtime.py)
        # when they call StructuredCGrid.*; there is no legacy model-build arg
        # for it, and the legacy path never applied the shallowing conditionally.
        "boundary.obcs_fix_topo",
        # MITgcm OB_J*/OB_I* index-table policy: consumed by the runtime momentum
        # sponge (injected in _canonical_helpers.enrich_manifest) and by the
        # data.obcs export (layer_d/consumers/mit_binary.py) from this one field.
        # There is no legacy model-build arg: the legacy path had no concept of a
        # per-column open boundary and always relaxed the whole edge, which is
        # exactly the 'full_edge' default.
        "boundary.ob_index_mode",
        "mit_comparison.enabled",
        "model.*",
        "case.kind",
        "case.extras.*",
        "time.steps",
    }
)


def translate_to_legacy_flat_yaml(
    run_cfg: RunConfig,
    raw_yaml: dict,
    *,
    target: Literal["to_run", "model_build", "adjoint"],
    output_dir: Path | None = None,
) -> dict:
    """Translate typed RunConfig plus raw YAML into legacy flat-YAML defaults.

    ``output_dir`` is the optional staged-pipeline plumbing: stage2
    passes ``manifest.run_dir`` so legacy's derived
    ``args.output_dir = DEFAULT_OUTPUT_ROOT / experiment_name`` is replaced by
    the manifest's actual run directory, keeping legacy outputs and the new
    pipeline's manifest in sync. Not modeled in ``run_cfg`` / raw yaml on
    purpose — it is a runtime path, not configuration.

    ``target`` selects the tail behavior:

      - ``"to_run"`` (stage2 LOCOMgcm run): emit ``use-prescribed-obcs=True`` iff
        ``mit_comparison.enabled``.
      - ``"model_build"`` (model assembly, MIT export path): emit
        ``mitgcm-comparison=True`` iff ``mit_comparison.enabled``.
      - ``"adjoint"`` (``run_adjoint.py`` entrypoint): emit
        ``first-step-adjoint-trace=True`` so the legacy script runs the
        first-step adjoint trace branch and exits early.
    """

    if target not in {"to_run", "model_build", "adjoint"}:
        raise TranslationError(f"unknown legacy translation target {target!r}")

    _assert_yaml_complete(raw_yaml)
    _assert_no_double_track(raw_yaml)
    flat: dict[str, Any] = {}
    seen_derived: set[int] = set()
    for rule in LEGACY_ARG_RULES.values():
        if isinstance(rule, NotExposed):
            continue
        if isinstance(rule, FromConfig):
            value = rule.getter(run_cfg)
            if rule.omit_if_none and value is None:
                continue
            flat[rule.legacy_key] = _yaml_scalar(value)
        elif isinstance(rule, FromSource):
            value = _source_value(run_cfg, rule)
            if value is not None:
                flat[rule.legacy_key] = _yaml_scalar(value)
        elif isinstance(rule, Derived):
            rule_id = id(rule)
            if rule_id in seen_derived:
                continue
            seen_derived.add(rule_id)
            for key, value in rule.fn(run_cfg).items():
                flat[key] = _yaml_scalar(value)

    if output_dir is not None:
        flat["output-dir"] = _yaml_scalar(Path(output_dir))

    if target == "to_run":
        if run_cfg.mit_comparison.enabled:
            flat["use-prescribed-obcs"] = True
    elif target == "model_build":
        # match old runconfig_to_legacy_args semantics
        if run_cfg.mit_comparison.enabled:
            flat["mitgcm-comparison"] = True
    else:  # target == "adjoint"
        flat["first-step-adjoint-trace"] = True

    return flat


def _source_value(run_cfg: RunConfig, rule: FromSource) -> Any:
    for source in run_cfg.sources:
        if source.type != rule.source_type:
            continue
        if rule.field == "type":
            return source.type
        return source.config.get(rule.field)
    return None


def _yaml_scalar(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_yaml_scalar(v) for v in value]
    if isinstance(value, list):
        return [_yaml_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _yaml_scalar(v) for k, v in value.items()}
    return value


def _assert_yaml_complete(raw_yaml: dict) -> None:
    claimed = _claimed_paths()
    for path in _leaf_paths(raw_yaml):
        if _matches_any(path, claimed) or _matches_any(path, PIPELINE_OWNED_PATHS):
            continue
        raise TranslationError(
            f"yaml key {path!r} maps to no legacy arg and is not pipeline-owned"
        )


def _assert_no_double_track(raw_yaml: dict) -> None:
    """fail-loud when a known double-track field appears in both
    ``physics.*`` and ``model.dynamics[<module>].kwargs.*`` / ``model.stepper.kwargs.*``
    with **inconsistent** values.

    Pure-yaml inspection (does not need RunConfig). If only one side writes
    the field, fall through silently — the runtime ``_dyn_or_phys`` getter
    will pick the dynamics value when present and physics value otherwise,
    so both LOCOMgcm runtime and MIT export see the same number.

    Goal: stop the bug class where yaml writes ``model.dynamics[].kwargs.X``
    (consumed by LOCOMgcm runtime) while physics keeps the dataclass default
    (consumed by MIT export through this bridge) and the two consumers
    silently disagree — exactly the visc_c2_leith case.
    """
    physics_block = dict(raw_yaml.get("physics", {}) or {})
    model_block = raw_yaml.get("model", {}) or {}
    dynamics_list = list(model_block.get("dynamics", []) or [])
    stepper_block = dict(model_block.get("stepper", {}) or {})
    stepper_kwargs = dict(stepper_block.get("kwargs", {}) or {})

    dyn_kwargs_by_name: dict[str, dict[str, Any]] = {}
    for entry in dynamics_list:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name is None:
            continue
        dyn_kwargs_by_name[str(name)] = dict(entry.get("kwargs", {}) or {})

    conflicts: list[str] = []
    implicit_vertical_advection_keys = (
        "temperature_implicit_vertical_advection",
        "salinity_implicit_vertical_advection",
        "momentum_implicit_vertical_advection",
    )
    duplicate_sources: list[str] = []
    for key in implicit_vertical_advection_keys:
        if key in stepper_kwargs:
            duplicate_sources.append(f"model.stepper.kwargs.{key}")
        for module_name, module_kwargs in dyn_kwargs_by_name.items():
            if key in module_kwargs:
                duplicate_sources.append(
                    f"model.dynamics[{module_name}].kwargs.{key}"
                )
    if duplicate_sources:
        raise TranslationError(
            "implicit vertical advection has duplicate derived source(s): canonical "
            "physics.implicit_vertical_advection and "
            + ", ".join(duplicate_sources)
        )

    for physics_field, kwarg_key, location, module_name in _DOUBLE_TRACK_FIELDS:
        if physics_field not in physics_block:
            continue
        physics_val = physics_block[physics_field]
        if location == "stepper":
            if kwarg_key not in stepper_kwargs:
                continue
            model_val = stepper_kwargs[kwarg_key]
            model_path = f"model.stepper.kwargs.{kwarg_key}"
        else:
            module_kwargs = dyn_kwargs_by_name.get(module_name)
            if module_kwargs is None or kwarg_key not in module_kwargs:
                continue
            model_val = module_kwargs[kwarg_key]
            model_path = f"model.dynamics[{module_name}].kwargs.{kwarg_key}"
        if physics_val != model_val:
            conflicts.append(
                f"  physics.{physics_field}={physics_val!r} != {model_path}={model_val!r}"
            )
    if conflicts:
        raise TranslationError(
            "yaml double-tracks the following numerical parameter(s) with "
            "inconsistent values across physics.* and model.*:\n"
            + "\n".join(conflicts)
            + "\nFix: give both spellings the same value, or write only one of them in an "
              "overlay or --set item (the loader copies it to the other). The model.* "
              "side is what the LOCOMgcm runtime instantiates with; physics.* is what "
              "the legacy bridge reads for the MITgcm export. They must not disagree."
        )


def _claimed_paths() -> frozenset[str]:
    claims: set[str] = set()
    for rule in LEGACY_ARG_RULES.values():
        if isinstance(rule, (FromConfig, Derived)):
            claims.update(rule.claims)
        elif isinstance(rule, FromSource):
            claims.add("sources.*.type")
            claims.add(f"sources.*.{rule.field}")
    return frozenset(claims)


def _leaf_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_leaf_paths(child, child_prefix))
        return tuple(paths)
    return (prefix,) if prefix else ()


def _matches_any(path: str, patterns: frozenset[str]) -> bool:
    return any(_path_matches(path, pattern) for pattern in patterns)


def _path_matches(path: str, pattern: str) -> bool:
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return path == prefix or path.startswith(prefix + ".")
    path_parts = path.split(".")
    pattern_parts = pattern.split(".")
    if len(path_parts) != len(pattern_parts):
        return False
    return all(pat == "*" or pat == part for part, pat in zip(path_parts, pattern_parts))
